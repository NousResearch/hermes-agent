"""Regex-based secret redaction for logs and tool output.

Applies pattern matching to mask API keys, tokens, and credentials
before they reach log files, verbose output, or gateway logs.

Short tokens (< 18 chars) are fully masked. Longer tokens preserve
the first 6 and last 4 characters for debuggability.
"""

import logging
import os
import re
import shlex
from urllib.parse import unquote_plus

logger = logging.getLogger(__name__)

# Sensitive query-string parameter names (case-insensitive exact match).
# Ported from nearai/ironclaw#2529 — catches tokens whose values don't match
# any known vendor prefix regex (e.g. opaque tokens, short OAuth codes).
_SENSITIVE_QUERY_PARAMS = frozenset({
    "access_token",
    "refresh_token",
    "id_token",
    "token",
    "api_key",
    "apikey",
    "client_secret",
    "password",
    "auth",
    "jwt",
    "session",
    "secret",
    "key",
    "code",           # OAuth authorization codes
    "signature",      # pre-signed URL signatures
    "x-amz-signature",
})

# Sensitive form-urlencoded / JSON body key names (case-insensitive exact match).
# Exact match, NOT substring — "token_count" and "session_id" must NOT match.
# Ported from nearai/ironclaw#2529.
_SENSITIVE_BODY_KEYS = frozenset({
    "access_token",
    "refresh_token",
    "id_token",
    "token",
    "api_key",
    "apikey",
    "client_secret",
    "password",
    "auth",
    "jwt",
    "secret",
    "private_key",
    "authorization",
    "key",
})

# Snapshot at import time so runtime env mutations (e.g. LLM-generated
# `export HERMES_REDACT_SECRETS=false`) cannot disable redaction
# mid-session.  ON by default — secure default per issue #17691. Users who
# need raw credential values in tool output (e.g. working on the redactor
# itself) can opt out via `security.redact_secrets: false` in config.yaml
# (bridged to this env var in hermes_cli/main.py, gateway/run.py, and
# cli.py) or `HERMES_REDACT_SECRETS=false` in ~/.hermes/.env. An opt-out
# warning is logged at gateway and CLI startup so operators see the
# downgrade — see `_log_redaction_status()` in gateway/run.py and cli.py.
_REDACT_ENABLED = os.getenv("HERMES_REDACT_SECRETS", "true").lower() in {"1", "true", "yes", "on"}

# Known API key prefixes -- match the prefix + contiguous token chars
_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",           # OpenAI / OpenRouter / Anthropic (sk-ant-*)
    r"ghp_[A-Za-z0-9]{10,}",            # GitHub PAT (classic)
    r"github_pat_[A-Za-z0-9_]{10,}",    # GitHub PAT (fine-grained)
    r"gho_[A-Za-z0-9]{10,}",            # GitHub OAuth access token
    r"ghu_[A-Za-z0-9]{10,}",            # GitHub user-to-server token
    r"ghs_[A-Za-z0-9]{10,}",            # GitHub server-to-server token
    r"ghr_[A-Za-z0-9]{10,}",            # GitHub refresh token
    r"xapp-\d+-[A-Za-z0-9-]{10,}",      # Slack app-Level token
    r"xox[baprs]-[A-Za-z0-9-]{10,}",    # Slack bot/app/user tokens
    r"AIza[A-Za-z0-9_-]{30,}",          # Google API keys
    r"pplx-[A-Za-z0-9]{10,}",           # Perplexity
    r"fal_[A-Za-z0-9_-]{10,}",          # Fal.ai
    r"fc-[A-Za-z0-9]{10,}",             # Firecrawl
    r"bb_live_[A-Za-z0-9_-]{10,}",      # BrowserBase
    r"gAAAA[A-Za-z0-9_=-]{20,}",        # Codex encrypted tokens
    r"AKIA[A-Z0-9]{16}",                # AWS Access Key ID
    r"sk_live_[A-Za-z0-9]{10,}",        # Stripe secret key (live)
    r"sk_test_[A-Za-z0-9]{10,}",        # Stripe secret key (test)
    r"rk_live_[A-Za-z0-9]{10,}",        # Stripe restricted key
    r"SG\.[A-Za-z0-9_-]{10,}",          # SendGrid API key
    r"hf_[A-Za-z0-9]{10,}",             # HuggingFace token
    r"r8_[A-Za-z0-9]{10,}",             # Replicate API token
    r"npm_[A-Za-z0-9]{10,}",            # npm access token
    r"pypi-[A-Za-z0-9_-]{10,}",         # PyPI API token
    r"dop_v1_[A-Za-z0-9]{10,}",         # DigitalOcean PAT
    r"doo_v1_[A-Za-z0-9]{10,}",         # DigitalOcean OAuth
    r"am_[A-Za-z0-9_-]{10,}",           # AgentMail API key
    r"sk_[A-Za-z0-9_]{10,}",            # ElevenLabs TTS key (sk_ underscore, not sk- dash)
    r"tvly-[A-Za-z0-9]{10,}",           # Tavily search API key
    r"exa_[A-Za-z0-9]{10,}",            # Exa search API key
    r"gsk_[A-Za-z0-9]{10,}",            # Groq Cloud API key
    r"syt_[A-Za-z0-9]{10,}",            # Matrix access token
    r"retaindb_[A-Za-z0-9]{10,}",       # RetainDB API key
    r"hsk-[A-Za-z0-9]{10,}",            # Hindsight API key
    r"mem0_[A-Za-z0-9]{10,}",           # Mem0 Platform API key
    r"brv_[A-Za-z0-9]{10,}",            # ByteRover API key
    r"xai-[A-Za-z0-9]{30,}",            # xAI (Grok) API key
    r"ntn_[A-Za-z0-9]{10,}",            # Notion internal integration token
    r"fw-[A-Za-z0-9]{30,}",             # Fireworks AI API key
    r"fw_[A-Za-z0-9]{30,}",             # Fireworks AI API key
    r"fpk_[A-Za-z0-9]{30,}",            # Fireworks AI project key
]

# ENV assignment patterns: KEY=value where KEY contains a secret-like name.
# Uppercase keys tolerate spaces around "=" (e.g. ``FOO_SECRET = bar``) because
# an all-caps key is almost never prose/code.
_SECRET_ENV_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)(\S+)\2",
)

# Lowercase / dotted / hyphenated config keys from config files
# (application.properties, .env, YAML-ish dumps): ``spring.datasource.password=secret``,
# ``app.api.key=xyz``, ``password=secret``. The uppercase _ENV_ASSIGN_RE above
# never matched these, so config-file passwords leaked verbatim (issue #16413).
#
# These run only in a config-file context, NOT in prose, code, or URLs — three
# carve-outs preserved from the original design (#4367 + the documented
# web-URL passthrough below):
#   1. The value is bounded by ``[^\s&]`` (stops at whitespace AND ``&``) so
#      form-urlencoded bodies are handled pair-by-pair (by _redact_form_body),
#      not greedily swallowed.
#   2. _CFG_DOTTED_RE only matches when the key is NAMESPACED (contains a dot),
#      which is unambiguously a config key — never a prose word.
#   3. _CFG_ANCHORED_RE matches a bare secret-word key only at line start
#      (optionally after ``export``), so conversational ``I have password=foo``
#      mid-sentence is left alone.
# The colon-form URL guard (skip when ``://`` present) lives at the call site.
_SECRET_CFG_NAMES = r"(?:api[ _.\-]?key|token|secret|passwd|password|credential|auth)"
_CFG_VALUE = r"(['\"]?)([^\s&]+?)\2(?=[\s&]|$)"

# Programmatic env lookups (``os.getenv(...)``, ``os.environ[...]``,
# ``os.environ.get(...)``, ``process.env.X``, ``$ENV{X}``) reference variable
# *names*, not secret values. When one appears as the VALUE of a KEY=... match
# it's a code snippet, not a leaked secret — skip redaction (issue #2852).
_ENV_LOOKUP_VALUE_RE = re.compile(
    r"^(?:os\.(?:getenv|environ)|process\.env|\$ENV\{)"
)
# Namespaced (dotted) key: the secret word may sit anywhere in a dotted path.
_CFG_DOTTED_RE = re.compile(
    rf"((?:[A-Za-z0-9_\-]+\.)+[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*"
    rf"|[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*\.[A-Za-z0-9_.\-]+)"
    rf"={_CFG_VALUE}",
    re.IGNORECASE,
)
# Line-anchored bare key: ``password=…`` / ``export api_key=…`` at start of line.
_CFG_ANCHORED_RE = re.compile(
    rf"(^[ \t]*(?:export[ \t]+)?[A-Za-z0-9_\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_\-]*)={_CFG_VALUE}",
    re.IGNORECASE | re.MULTILINE,
)

# Unquoted YAML / colon config (e.g. ``password: secret``,
# ``spring.datasource.password: hunter2``). The secret keyword must be part of
# the KEY (anchored to the start of the line/indent), and the value is a single
# whitespace-free token — so prose like ``note: secret meeting`` (keyword in the
# value) and ``error: token expired`` are left alone. Bare ``auth`` is excluded
# from the key set so ``Authorization:`` / ``author:`` don't match (the former
# is masked by _AUTH_HEADER_RE); ``auth_token``/``auth-token`` still match via
# the ``token`` keyword. Quoted values defer to _JSON_FIELD_RE via the lookahead.
_YAML_CFG_NAMES = r"(?:api[ _.\-]?key|token|secret|passwd|password|credential)"
_YAML_ASSIGN_RE = re.compile(
    rf"(^[ \t]*[A-Za-z0-9_.\-]*{_YAML_CFG_NAMES}[A-Za-z0-9_.\-]*)(:[ \t]*)(?!['\"])([^\s&]+)",
    re.IGNORECASE | re.MULTILINE,
)

# JSON field patterns: "apiKey": "value", "token": "value", etc.
_JSON_KEY_NAMES = r"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|auth_token|bearer|secret_value|raw_secret|secret_input|key_material)"
_JSON_FIELD_RE = re.compile(
    rf'("{_JSON_KEY_NAMES}")\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)

# Authorization headers — any scheme (Bearer, Basic, Token, Digest, …) plus the
# bare-credential form, and Proxy-Authorization. The credential token is masked
# while the header name and scheme word are preserved for debuggability. The
# previous rule only matched ``Bearer``, so ``Basic <base64 user:pass>`` and
# ``token <pat>`` leaked verbatim into logs/transcripts.
#
# The credential class excludes quote characters (``"`` / ``'``): a token sitting
# flush against a closing quote (``"Authorization: Bearer sk-..."``) must not pull
# that quote into the match, or masking turns value corruption into *syntax*
# corruption — the closing quote vanishes and the command/string no longer parses
# (unterminated quote → shell EOF / Python SyntaxError). Real credentials never
# contain ``"`` or ``'``, so excluding them is safe. See #43083.
_AUTH_HEADER_RE = re.compile(
    r"((?:Proxy-)?Authorization:\s*)([A-Za-z][\w.+-]*\s+)?([^\s\"']+)",
    re.IGNORECASE,
)

# API-key style auth headers carrying a single opaque value (no scheme word).
# Anthropic and many providers authenticate with ``x-api-key``; values without
# a known vendor prefix (custom/local backends) would otherwise leak when a
# request or curl command is logged or echoed into tool output / transcripts.
_SECRET_HEADER_NAMES = (
    r"(?:x-api-key|x-goog-api-key|api-key|apikey|x-api-token|x-auth-token|x-access-token)"
)
_SECRET_HEADER_RE = re.compile(
    rf"({_SECRET_HEADER_NAMES}\s*:\s*)(\S+)",
    re.IGNORECASE,
)

# Telegram bot tokens: bot<digits>:<token> or <digits>:<token>,
# where token part is restricted to [-A-Za-z0-9_] and length >= 30
_TELEGRAM_RE = re.compile(
    r"(bot)?(\d{8,}):([-A-Za-z0-9_]{30,})",
)

# Private key blocks: -----BEGIN RSA PRIVATE KEY----- ... -----END RSA PRIVATE KEY-----
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

# Database connection strings: protocol://user:PASSWORD@host
# Catches postgres, mysql, mongodb, redis, amqp URLs and redacts the password.
# The userinfo and password groups forbid whitespace ([^:\s]+ / [^@\s]+) so the
# match can never span a line break. A real DSN password never contains
# whitespace; without this bound the greedy [^@]+ would scan past the end of a
# code line to the next stray "@" (e.g. a Python decorator), swallowing
# intervening lines and corrupting tool OUTPUT for any source containing a
# postgresql:// f-string template. See issue #33801.
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s]+:)([^@\s]+)(@)",
    re.IGNORECASE,
)

# Bare-token credential in a web/transport URL: ``scheme://TOKEN@host``.
# This is the ``git remote set-url origin https://PASSWORD@github.com/...``
# shape from issue #6396 — a single opaque credential in the userinfo position
# with NO ``user:pass`` colon. It is unambiguously a secret: legitimate
# round-trip URLs (OAuth callbacks, magic links, pre-signed shares — see the
# "Web-URL redaction is intentionally OFF" note in redact_sensitive_text) carry
# their tokens in the QUERY STRING, never in bare userinfo. The colon form
# ``user:pass@`` is deliberately left to pass through (commit "pass web URLs
# through unchanged", #34029) and is NOT matched here — the token class forbids
# ``:``. DB schemes are handled by _DB_CONNSTR_RE above and excluded here.
#
# Guards against false positives:
#   - 8+ char floor skips short usernames (git, admin, root, deploy, ubuntu).
#   - The token class ``[^\s:@/]`` cannot cross ``/``, so an ``@`` sitting in a
#     path or query (e.g. ``?q=user@example.com``) is never treated as userinfo.
_URL_BARE_TOKEN_RE = re.compile(
    r"((?:https?|wss?|git|ssh|ftp|ftps|sftp)://)"  # scheme
    r"([^\s:@/]{8,})"                               # bare token (no colon/slash/@), 8+ chars
    r"(@[^\s]+)",                                   # @host...
    re.IGNORECASE,
)

# JWT tokens: header.payload[.signature] — always start with "eyJ" (base64 for "{")
# Matches 1-part (header only), 2-part (header.payload), and full 3-part JWTs.
_JWT_RE = re.compile(
    r"eyJ[A-Za-z0-9_-]{10,}"           # Header (always starts with eyJ)
    r"(?:\.[A-Za-z0-9_=-]{4,}){0,2}"   # Optional payload and/or signature
)

# E.164 phone numbers: +<country><number>, 7-15 digits
# Negative lookahead prevents matching hex strings or identifiers
_SIGNAL_PHONE_RE = re.compile(r"(\+[1-9]\d{6,14})(?![A-Za-z0-9])")

# URLs containing query strings — matches `scheme://...?...[# or end]`.
# Used to scan text for URLs whose query params may contain secrets.
# Ported from nearai/ironclaw#2529.
_URL_WITH_QUERY_RE = re.compile(
    r"(https?|wss?|ftp)://"          # scheme
    r"([^\s/?#]+)"                    # authority (may include userinfo)
    r"([^\s?#]*)"                     # path
    r"\?([^\s#]+)"                    # query (required)
    r"(#\S*)?",                       # optional fragment
)

# URLs containing userinfo — `scheme://user:password@host` for ANY scheme
# (not just DB protocols already covered by _DB_CONNSTR_RE above).
# Catches things like `https://user:token@api.example.com/v1/foo`.
_URL_USERINFO_RE = re.compile(
    r"(https?|wss?|ftp)://([^/\s:@]+):([^/\s@]+)@",
)

# Strict provider-egress URL redaction accepts more URL-reference forms than
# the display/log helpers above. Parameter delimiters stay in capture groups so
# redaction preserves the original query/fragment layout byte-for-byte, while
# the key is decoded separately for classification. Values stop at query or
# fragment pair separators; both ``&`` and ``;`` are valid in deployed URLs.
_STRICT_URL_PARAM_RE = re.compile(
    r"([?#&;])([A-Za-z0-9_.~+%\-]+)=([^#&;\s\"'<>]*)"
)

# Match userinfo in both absolute (``scheme://user:pass@host``) and
# network-path (``//user:pass@host``) references. The authority boundary stops
# at path/query/fragment delimiters so an ``@`` elsewhere in a URL is ignored.
_STRICT_URL_USERINFO_RE = re.compile(
    r"((?:[A-Za-z][A-Za-z0-9+.-]*:)?//)([^/\s?#@]+)@"
)

# HTTP access logs often use a relative request target rather than a full URL:
# `"POST /webhook?password=... HTTP/1.1"`. The full-URL redactor above only
# sees strings containing `://`, so handle request-target query strings too.
_HTTP_REQUEST_TARGET_QUERY_RE = re.compile(
    r"\b((?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|TRACE|CONNECT)\s+[^ \t\r\n\"']*?)"
    r"\?([^ \t\r\n\"']+)",
    re.IGNORECASE,
)

# Form-urlencoded body detection: conservative — only applies when the entire
# text looks like a query string (k=v&k=v pattern with no newlines).
_FORM_BODY_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*(?:&[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*)+$"
)

# Compile known prefix patterns into one alternation
_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)


def mask_secret(
    value: str,
    *,
    head: int = 4,
    tail: int = 4,
    floor: int = 12,
    placeholder: str = "***",
    empty: str = "",
) -> str:
    """Mask a secret for display, preserving ``head`` and ``tail`` characters.

    Canonical helper for display-time redaction across Hermes — used by
    ``hermes config``, ``hermes status``, ``hermes dump``, and anywhere
    a secret needs to be shown truncated for debuggability while still
    keeping the bulk hidden.

    Args:
        value:       The secret to mask. ``None``/empty returns ``empty``.
        head:        Leading characters to preserve. Default 4.
        tail:        Trailing characters to preserve. Default 4.
        floor:       Values shorter than ``head + tail + floor_margin`` are
                     fully masked (returns ``placeholder``). Default 12 —
                     matches the existing config/status/dump convention.
        placeholder: Value returned for too-short inputs. Default ``"***"``.
        empty:       Value returned when ``value`` is falsy (None, ""). The
                     caller can override this to e.g. ``color("(not set)",
                     Colors.DIM)`` for user-facing display.

    Examples:
        >>> mask_secret("sk-proj-abcdef1234567890")
        'sk-p...7890'
        >>> mask_secret("short")                         # fully masked
        '***'
        >>> mask_secret("")                              # empty default
        ''
        >>> mask_secret("", empty="(not set)")           # empty override
        '(not set)'
        >>> mask_secret("long-token", head=6, tail=4, floor=18)
        '***'
    """
    if not value:
        return empty
    if len(value) < floor:
        return placeholder
    return f"{value[:head]}...{value[-tail:]}"


def _mask_token(token: str) -> str:
    """Mask a log token — conservative 18-char floor, preserves 6 prefix / 4 suffix."""
    # Empty input: historically this returned "***" rather than "". Preserve.
    if not token:
        return "***"
    return mask_secret(token, head=6, tail=4, floor=18)


def _redact_query_string(query: str) -> str:
    """Redact sensitive parameter values in a URL query string.

    Handles `k=v&k=v` format. Sensitive keys (case-insensitive) have values
    replaced with `***`. Non-sensitive keys pass through unchanged.
    Empty or malformed pairs are preserved as-is.
    """
    if not query:
        return query
    parts = []
    for pair in query.split("&"):
        if "=" not in pair:
            parts.append(pair)
            continue
        key, _, value = pair.partition("=")
        if key.lower() in _SENSITIVE_QUERY_PARAMS:
            parts.append(f"{key}=***")
        else:
            parts.append(pair)
    return "&".join(parts)


def _redact_url_query_params(text: str) -> str:
    """Scan text for URLs with query strings and redact sensitive params.

    Catches opaque tokens that don't match vendor prefix regexes, e.g.
    `https://example.com/cb?code=ABC123&state=xyz` → `...?code=***&state=xyz`.
    """
    def _sub(m: re.Match) -> str:
        scheme = m.group(1)
        authority = m.group(2)
        path = m.group(3)
        query = _redact_query_string(m.group(4))
        fragment = m.group(5) or ""
        return f"{scheme}://{authority}{path}?{query}{fragment}"
    return _URL_WITH_QUERY_RE.sub(_sub, text)


def _redact_url_userinfo(text: str) -> str:
    """Strip `user:password@` from HTTP/WS/FTP URLs.

    DB protocols (postgres, mysql, mongodb, redis, amqp) are handled
    separately by `_DB_CONNSTR_RE`.
    """
    return _URL_USERINFO_RE.sub(
        lambda m: f"{m.group(1)}://{m.group(2)}:***@",
        text,
    )


def _canonical_url_param_name(name: str) -> str:
    """Decode a URL parameter name for bounded, case-insensitive matching."""
    decoded = name
    for _ in range(3):
        next_value = unquote_plus(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return decoded.casefold().replace("-", "_")


def _redact_strict_url_credentials(text: str) -> str:
    """Redact credentials from absolute, relative, and network URL references.

    This is intentionally stricter than display/log redaction and is used only
    at explicit secret-egress boundaries. It preserves original keys,
    separators, public parameters, hosts, and paths while masking sensitive
    values and URL userinfo.
    """
    def _redact_param(match: re.Match) -> str:
        if _canonical_url_param_name(match.group(2)) not in _SENSITIVE_QUERY_PARAMS:
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}=***"

    def _redact_userinfo(match: re.Match) -> str:
        userinfo = match.group(2)
        if ":" in userinfo:
            username, _, _password = userinfo.partition(":")
            return f"{match.group(1)}{username}:***@"
        return f"{match.group(1)}***@"

    text = _STRICT_URL_PARAM_RE.sub(_redact_param, text)
    return _STRICT_URL_USERINFO_RE.sub(_redact_userinfo, text)


def redact_cdp_url(value: object) -> str:
    """Mask secrets in a CDP/browser endpoint URL before it is logged.

    The global ``redact_sensitive_text`` deliberately passes web-URL query
    params and ``user:pass@`` userinfo through unmasked (OAuth callbacks,
    magic-link / pre-signed URLs the agent is meant to follow -- see the
    web-URL note above). CDP discovery endpoints are NOT such a workflow:
    their query-string tokens and userinfo passwords are pure credentials
    that must never reach the logs. So for CDP URLs we opt INTO the two URL
    redactors that the global pass leaves off.

    This is the single source of truth for redacting a CDP URL that is passed
    *directly* to a log or error message. Callers that instead need to redact an
    exception whose text embeds the URL (e.g. a ``websockets`` connect error)
    should route that through their own error-text helper, which delegates here
    -- see ``tools.browser_supervisor._redact_cdp_error_text``.
    """
    text = redact_sensitive_text("" if value is None else str(value))
    if not text:
        return text
    text = _redact_url_query_params(text)
    text = _redact_url_userinfo(text)
    return text


def _redact_http_request_target_query_params(text: str) -> str:
    """Redact sensitive query params in HTTP access-log request targets."""
    def _sub(m: re.Match) -> str:
        prefix = m.group(1)
        query = _redact_query_string(m.group(2))
        return f"{prefix}?{query}"
    return _HTTP_REQUEST_TARGET_QUERY_RE.sub(_sub, text)


def _redact_form_body(text: str) -> str:
    """Redact sensitive values in a form-urlencoded body.

    Only applies when the entire input looks like a pure form body
    (k=v&k=v with no newlines, no other text). Single-line non-form
    text passes through unchanged. This is a conservative pass — the
    `_redact_url_query_params` function handles embedded query strings.
    """
    if not text or "\n" in text or "&" not in text:
        return text
    # The body-body form check is strict: only trigger on clean k=v&k=v.
    if not _FORM_BODY_RE.match(text.strip()):
        return text
    return _redact_query_string(text.strip())


def _mask_token_nonreusable(token: str) -> str:
    """Redact a prefix-matched credential to a NON-REUSABLE sentinel.

    Unlike :func:`_mask_token` (which keeps head/tail chars — fine for logs
    that are never fed back into a config), this emits a marker that:

    * cannot be mistaken for a usable-but-truncated key, so an agent that
      reads it from a config file and writes it back does NOT corrupt the
      stored credential into a dead 13-char string (issue #35519); and
    * still does not leak the secret material (no head/tail chars).

    The vendor prefix label is preserved for debuggability so the agent can
    still tell *which* credential is present (e.g. a GitHub PAT vs an OpenAI
    key) without seeing any of its bytes.
    """
    if not token:
        return "«redacted-secret»"
    # Preserve only the recognizable vendor prefix label (e.g. "ghp_", "sk-"),
    # never any of the random secret body.
    label = ""
    for sub in _PREFIX_SUBSTRINGS:
        if token.startswith(sub):
            label = sub
            break
    return f"«redacted:{label}…»" if label else "«redacted-secret»"


def redact_sensitive_text(
    text: str,
    *,
    force: bool = False,
    code_file: bool = False,
    file_read: bool = False,
    redact_url_credentials: bool = False,
) -> str:
    """Apply all redaction patterns to a block of text.

    Safe to call on any string -- non-matching text passes through unchanged.
    Enabled by default. Disable via security.redact_secrets: false in config.yaml.
    Set force=True for safety boundaries that must never return raw secrets
    regardless of the user's global logging redaction preference.

    Set redact_url_credentials=True at non-navigation egress boundaries to
    additionally redact credential-named query parameters and ``user:pass@``
    URL userinfo. The default remains False because actionable OAuth callback,
    magic-link, and pre-signed URLs must survive ordinary tool flows unchanged.

    Set code_file=True to skip the ENV-assignment and JSON-field regex
    patterns when the text is known to be source code (e.g. MAX_TOKENS=***
    constants, "apiKey": "test" fixtures). Prefix patterns, auth headers,
    private keys, DB connstrings, JWTs, and URL secrets are still redacted.

    Set file_read=True for file *content* returned to the agent (read_file /
    search_files / cat). Secrets are STILL redacted — they are never exposed —
    but prefix-matched credentials are replaced with a non-reusable sentinel
    (``«redacted:ghp_…»``) instead of a head/tail-preserving mask
    (``ghp_S1...Pn2T``). The old mask looked like a real-but-truncated key, so
    an agent reading it from config.yaml and writing it back silently corrupted
    the stored credential into a dead 13-char value → 401 (issue #35519). The
    sentinel is syntactically invalid as a token, so it can't be mistaken for a
    usable key or written back as one. Implies code_file=True (config/data
    files shouldn't trigger the source-code ENV/JSON false-positive paths).

    Performance: each regex pattern is gated behind a cheap substring
    pre-check (e.g. ``"=" in text`` for ENV assignments, ``"://" in text``
    for URLs, ``"eyJ" in text`` for JWTs). On a typical hermes log line
    (no secrets) this drops the 13-pattern scan from ~5.6us to ~1.8us per
    record (-68%). The pre-checks are conservative — false positives
    still run the full regex, which then doesn't match. False negatives
    are impossible because every regex requires the gated substring to
    match.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    if not (force or _REDACT_ENABLED):
        return text

    # file_read content shouldn't hit the source-code ENV/JSON false-positive
    # paths either (it's config/data, not log lines).
    if file_read:
        code_file = True

    # Known prefixes (sk-, ghp_, etc.) — gate on substring presence
    if _has_known_prefix_substring(text):
        _prefix_sub = _mask_token_nonreusable if file_read else _mask_token
        text = _PREFIX_RE.sub(lambda m: _prefix_sub(m.group(1)), text)

    # ENV assignments: OPENAI_API_KEY=***  (skip for code files — false positives)
    if not code_file:
        if "=" in text:
            def _redact_env(m):
                name, quote, value = m.group(1), m.group(2), m.group(3)
                # Programmatic env lookups reference variable *names*, not
                # secret values — masking them corrupts code snippets in
                # prose/log contexts (issue #2852): ``KEY=os.getenv('X')``.
                if _ENV_LOOKUP_VALUE_RE.match(value):
                    return m.group(0)
                return f"{name}={quote}{_mask_token(value)}{quote}"
            text = _ENV_ASSIGN_RE.sub(_redact_env, text)
            # Lowercase/dotted config keys (issue #16413). Skip URLs entirely —
            # web-URL query params are intentionally passed through (see note
            # near the bottom of this function); _DB_CONNSTR_RE still guards
            # connection-string passwords.
            if "://" not in text:
                text = _CFG_DOTTED_RE.sub(_redact_env, text)
                text = _CFG_ANCHORED_RE.sub(_redact_env, text)

        # JSON fields: "apiKey": "***"  (skip for code files — false positives)
        if ":" in text and '"' in text:
            def _redact_json(m):
                key, value = m.group(1), m.group(2)
                # Same programmatic-env-lookup exception as _redact_env above
                # (issue #2852): "apiKey": "os.getenv('X')" is a code snippet,
                # not a leaked secret value.
                if _ENV_LOOKUP_VALUE_RE.match(value):
                    return m.group(0)
                return f'{key}: "{_mask_token(value)}"'
            text = _JSON_FIELD_RE.sub(_redact_json, text)

        # Unquoted YAML / colon config: password: ***  (after JSON so quoted
        # values are handled there; the lookahead in _YAML_ASSIGN_RE skips
        # quotes). Skip URLs — web-URL query params pass through by design.
        if ":" in text and "://" not in text:
            def _redact_yaml(m):
                key, sep, value = m.group(1), m.group(2), m.group(3)
                # Same programmatic-env-lookup exception as _redact_env above
                # (issue #2852): api_key: os.getenv('X') is a code snippet,
                # not a leaked secret value.
                if _ENV_LOOKUP_VALUE_RE.match(value):
                    return m.group(0)
                return f"{key}{sep}{_mask_token(value)}"
            text = _YAML_ASSIGN_RE.sub(_redact_yaml, text)

    # Authorization headers — _AUTH_HEADER_RE matches any scheme after
    # "[Proxy-]Authorization:" case-insensitively, so "uthorization" is the
    # cheapest substring gate that covers every casing without a casefold().
    if "uthorization" in text or "UTHORIZATION" in text:
        text = _AUTH_HEADER_RE.sub(
            lambda m: m.group(1) + (m.group(2) or "") + _mask_token(m.group(3)),
            text,
        )

    # API-key style headers (x-api-key, api-key, …). Header values are
    # colon-separated, so gate on ":" — the regex itself is the precise filter.
    if ":" in text:
        text = _SECRET_HEADER_RE.sub(
            lambda m: m.group(1) + _mask_token(m.group(2)),
            text,
        )

    # Telegram bot tokens — pattern requires ":<token>" with digits prefix
    if ":" in text:
        def _redact_telegram(m):
            prefix = m.group(1) or ""
            digits = m.group(2)
            return f"{prefix}{digits}:***"
        text = _TELEGRAM_RE.sub(_redact_telegram, text)

    # Private key blocks
    if "BEGIN" in text and "-----" in text:
        text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)

    # Database connection string passwords. With code_file=True, a password
    # group that is a pure ``{...}`` brace expression is an f-string template
    # reference (e.g. f"postgresql://{user}:{pass}@{host}"), not a literal
    # credential — preserve it. Literal passwords are still redacted. The regex
    # forbids whitespace in the password group, so a single-line template's
    # group(2) is exactly the brace expression. See issue #33801.
    if "://" in text:
        if code_file:
            def _redact_db(m):
                pw = m.group(2)
                if pw.startswith("{") and pw.endswith("}"):
                    return m.group(0)
                return f"{m.group(1)}***{m.group(3)}"
            text = _DB_CONNSTR_RE.sub(_redact_db, text)
        else:
            text = _DB_CONNSTR_RE.sub(lambda m: f"{m.group(1)}***{m.group(3)}", text)

        # Bare-token userinfo in web/transport URLs: ``scheme://TOKEN@host``.
        # The git-remote-with-embedded-password shape from #6396. Only the
        # colon-less bare-token form is redacted — ``user:pass@`` and
        # query-string tokens are left to pass through (see the web-URL note
        # below). See _URL_BARE_TOKEN_RE for the false-positive guards.
        text = _URL_BARE_TOKEN_RE.sub(
            lambda m: f"{m.group(1)}{_mask_token(m.group(2))}{m.group(3)}",
            text,
        )

    # JWT tokens (eyJ... — base64-encoded JSON headers)
    if "eyJ" in text:
        text = _JWT_RE.sub(lambda m: _mask_token(m.group(0)), text)

    # NOTE: Web-URL redaction (query params + userinfo + HTTP access-log
    # request targets) is intentionally OFF. Many legitimate workflows pass
    # opaque tokens through query strings — magic-link checkouts, OAuth
    # callbacks the agent is meant to follow, pre-signed share URLs — and
    # blanket-redacting param values by name breaks those skills mid-flow.
    # Known credential shapes (sk-, ghp_, JWTs, etc.) inside URLs are still
    # caught by _PREFIX_RE and _JWT_RE above. DB connection-string passwords
    # are still caught by _DB_CONNSTR_RE. The ONE userinfo case still redacted
    # is the colon-less bare-token form ``scheme://TOKEN@host`` (#6396, handled
    # by _URL_BARE_TOKEN_RE in the ``://`` block above): a bare credential in
    # userinfo is never a round-trip workflow token (those live in the query
    # string), so masking it can't break a skill. The ``user:pass@`` form is
    # left to pass through per #34029.

    if redact_url_credentials:
        text = _redact_strict_url_credentials(text)

    # Form-urlencoded bodies (only triggers on clean k=v&k=v inputs).
    if "&" in text and "=" in text:
        text = _redact_form_body(text)

    # E.164 phone numbers (Signal, WhatsApp)
    if "+" in text:
        def _redact_phone(m):
            phone = m.group(1)
            if len(phone) <= 8:
                return phone[:2] + "****" + phone[-2:]
            return phone[:4] + "****" + phone[-4:]
        text = _SIGNAL_PHONE_RE.sub(_redact_phone, text)

    return text


# Commands whose stdout is an environment-variable dump (KEY=value lines),
# NOT source code. For these, terminal-output redaction must run the
# ENV-assignment pass (code_file=False) so opaque tokens with no recognized
# vendor prefix (e.g. ``MY_SERVICE_TOKEN=abc123randomstring``) are still
# masked. For all other commands, code_file=True is used to avoid mangling
# legitimate source/config dumps (``MAX_TOKENS=100``, ``"apiKey": "x"``
# fixtures, ``postgresql://{user}`` f-string templates). See issue #43025.
_ENV_DUMP_COMMANDS = frozenset({"env", "printenv", "set", "export", "declare"})


def is_env_dump_command(command: str | None) -> bool:
    """Return True if ``command`` dumps environment variables to stdout.

    Detects ``env`` / ``printenv`` / ``set`` / ``export`` / ``declare`` as the
    first token of any segment in a pipeline or sequence (``;`` / ``&&`` /
    ``||`` / ``|``). Conservative: a parse failure or anything unrecognized
    returns False (callers then fall back to the safer code_file=True path,
    which still masks prefix-shaped keys).
    """
    if not command or not isinstance(command, str):
        return False
    # Split on shell separators, then inspect the first token of each segment.
    segments = re.split(r"[|;&]+", command)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if tokens and tokens[0] in _ENV_DUMP_COMMANDS:
            return True
    return False


def redact_terminal_output(
    output: str, command: str | None = None, *, force: bool = False
) -> str:
    """Redact secrets from terminal/process stdout.

    Single redaction policy for ALL terminal-output surfaces — foreground
    ``terminal`` results AND background ``process(action=poll/log/wait)``
    output — so they can't diverge. Picks ``code_file`` based on whether
    ``command`` is an environment dump:

    - env-dump command (``env``/``printenv``/``set``/``export``/``declare``)
      → ``code_file=False`` so the ENV-assignment pass masks opaque tokens.
    - anything else (or unknown command) → ``code_file=True`` to avoid
      false positives on source/config dumps.

    ``force=True`` bypasses the global ``security.redact_secrets`` preference
    for safety boundaries that must never emit raw credentials.
    """
    if not output:
        return output
    code_file = not is_env_dump_command(command or "")
    return redact_sensitive_text(output, force=force, code_file=code_file)


# Substrings used to gate ``_PREFIX_RE`` execution. If none of these appear in
# the input string, the prefix regex cannot match anything, so we skip it.
# False positives are fine (they just run the regex, which then matches
# nothing) — the bound is "no false negatives" and that holds because every
# pattern in ``_PREFIX_PATTERNS`` has at least one of these as a literal
# substring of its leading characters.
#
# Derived automatically from ``_PREFIX_PATTERNS`` at module load time so a
# future PR that adds a new prefix to the regex list can't silently break
# the screen.

def _extract_literal_prefix(pattern: str) -> str:
    """Return the leading literal characters of a regex pattern.

    Stops at the first regex metacharacter (``[``, ``(``, ``\\``, ``.``,
    ``?``, ``*``, ``+``, ``|``, ``{``, ``^``, ``$``).  Returns the literal
    that any match of the pattern MUST contain as a substring, so the
    pre-screen never produces false negatives.
    """
    meta = "[(\\.?*+|{^$"
    for i, ch in enumerate(pattern):
        if ch in meta:
            return pattern[:i]
    return pattern


_PREFIX_SUBSTRINGS = tuple(
    _extract_literal_prefix(p) for p in _PREFIX_PATTERNS
)


def _has_known_prefix_substring(text: str) -> bool:
    """Return True if ``text`` contains any known credential prefix substring.

    Used as a cheap pre-check before invoking the expensive ``_PREFIX_RE``.
    """
    return any(p in text for p in _PREFIX_SUBSTRINGS)


_HTTP_METHOD_SUBSTRINGS = (
    "GET ",
    "POST ",
    "PUT ",
    "PATCH ",
    "DELETE ",
    "HEAD ",
    "OPTIONS ",
    "TRACE ",
    "CONNECT ",
)


def _has_http_method_substring(text: str) -> bool:
    """Cheap pre-check before scanning for access-log request targets."""
    upper = text.upper()
    return any(method in upper for method in _HTTP_METHOD_SUBSTRINGS)


class RedactingFormatter(logging.Formatter):
    """Log formatter that redacts secrets from all log messages."""

    def __init__(self, fmt=None, datefmt=None, style='%', **kwargs):
        super().__init__(fmt, datefmt, style, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return redact_sensitive_text(original)
