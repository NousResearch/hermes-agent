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
_SECRET_ENV_WORDS = (
    "API_KEY",
    "APIKEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
)
_SECRET_ENV_NAMES = (
    r"(?:" + "|".join(re.escape(word) for word in _SECRET_ENV_WORDS) + r")"
)
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
_AUTH_HEADER_NAMES = ("Authorization", "Proxy-Authorization")
_AUTH_HEADER_RE = re.compile(
    r"((?:" + "|".join(re.escape(name) for name in _AUTH_HEADER_NAMES)
    + r"):\s*)([A-Za-z][\w.+-]*\s+)?([^\s\"']+)",
    re.IGNORECASE,
)

# API-key style auth headers carrying a single opaque value (no scheme word).
# Anthropic and many providers authenticate with ``x-api-key``; values without
# a known vendor prefix (custom/local backends) would otherwise leak when a
# request or curl command is logged or echoed into tool output / transcripts.
_SECRET_HEADER_NAME_VALUES = (
    "x-api-key",
    "x-goog-api-key",
    "api-key",
    "apikey",
    "x-api-token",
    "x-auth-token",
    "x-access-token",
)
_SECRET_HEADER_NAMES = (
    r"(?:" + "|".join(
        re.escape(name) for name in _SECRET_HEADER_NAME_VALUES
    ) + r")"
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
_JWT_HEAD_PATTERN = r"eyJ[A-Za-z0-9_-]{10,}"
_JWT_RE = re.compile(
    _JWT_HEAD_PATTERN                    # Header (always starts with eyJ)
    + r"(?:\.[A-Za-z0-9_=-]{4,}){0,2}"  # Optional payload and/or signature
)
_STREAMING_JWT_CANDIDATE_RE = re.compile(
    _JWT_HEAD_PATTERN + r"[A-Za-z0-9_.=-]*$"
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


def _expand_streaming_prefix(expression: str) -> tuple[str, ...]:
    """Expand the literal portion of a known-prefix regex.

    The current patterns use literal characters, escaped literals, and one
    simple character class (the Slack ``xox[baprs]-`` family). Keeping this
    derivation next to ``_PREFIX_PATTERNS`` prevents the streaming guard from
    becoming a second manually maintained credential list.
    """
    # Slack app-level tokens contain a variable numeric component
    # (``xapp-\d+-...``).  Retaining from the stable ``xapp-`` stem is
    # conservative and lets the static redactor decide whether the completed
    # token is valid once a delimiter arrives.
    variable_digits = expression.find(r"\d+")
    if variable_digits >= 0:
        expression = expression[:variable_digits]

    expanded = [""]
    i = 0
    while i < len(expression):
        char = expression[i]
        if char == "\\" and i + 1 < len(expression):
            expanded = [prefix + expression[i + 1] for prefix in expanded]
            i += 2
            continue
        if char == "[":
            closing = expression.find("]", i + 1)
            if closing < 0:
                return ()
            choices = expression[i + 1:closing]
            if not choices or any(ch in choices for ch in "\\^-"):
                return ()
            expanded = [prefix + choice for prefix in expanded for choice in choices]
            i = closing + 1
            continue
        if char in ".^$*+?{}()|":
            return ()
        expanded = [prefix + char for prefix in expanded]
        i += 1
    return tuple(expanded)


def _build_streaming_prefix_specs():
    """Derive (literal prefixes, token-char regex, minimum) specifications."""
    specs = []
    parts_re = re.compile(r"^(.*)(\[[^\]]+\])\{(\d+)(,?)\}$")
    for pattern in _PREFIX_PATTERNS:
        match = parts_re.fullmatch(pattern)
        if match is None:
            continue
        prefixes = _expand_streaming_prefix(match.group(1))
        if not prefixes:
            continue
        token_chars = re.compile(rf"^{match.group(2)}*$")
        specs.append((prefixes, token_chars, int(match.group(3))))
    return tuple(specs)


_STREAMING_PREFIX_SPECS = _build_streaming_prefix_specs()
_STREAMING_JSON_KEYS = (
    "apikey",
    "api_key",
    "token",
    "secret",
    "password",
    "access_token",
    "refresh_token",
    "auth_token",
    "bearer",
    "secret_value",
    "raw_secret",
    "secret_input",
    "key_material",
)
_STREAMING_JSON_OPENER_RE = re.compile(
    r'"(?:' + "|".join(re.escape(key) for key in _STREAMING_JSON_KEYS)
    + r')"\s*:\s*"',
    re.IGNORECASE,
)
_STREAMING_DB_PROTOCOLS = (
    "postgres://",
    "postgresql://",
    "mysql://",
    "mongodb://",
    "mongodb+srv://",
    "redis://",
    "amqp://",
)
_STREAMING_DB_OPENER_RE = re.compile(
    r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s]+:",
    re.IGNORECASE,
)
_STREAMING_PRIVATE_KEY_OPENER_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----"
)
_STREAMING_PRIVATE_KEY_END_RE = re.compile(
    r"-----END[A-Z ]*PRIVATE KEY-----"
)
_STREAMING_AUTH_HEADER_LITERALS = tuple(
    name.lower() + ":" for name in _AUTH_HEADER_NAMES
)
_STREAMING_SECRET_HEADER_LITERALS = tuple(
    name.lower() + ":" for name in _SECRET_HEADER_NAME_VALUES
)
_STREAMING_TELEGRAM_CANDIDATE_RE = re.compile(
    r"(?:bot)?\d{8,}:[-A-Za-z0-9_]{0,29}$"
)
_STREAMING_JWT_PRETHRESHOLD_RE = re.compile(
    r"e(?:y(?:J[A-Za-z0-9_.=-]{0,9})?)?$"
)
_STREAMING_JWT_BOUNDARY_PRETHRESHOLD_RE = re.compile(
    r"ey(?:J[A-Za-z0-9_.=-]{0,9})?$"
)
_STREAMING_PHONE_CANDIDATE_RE = re.compile(
    r"\+(?:[1-9]\d{0,13})?$"
)
_STREAMING_ENV_ACTIVE_RE = re.compile(
    rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})"
    r"\s*=\s*(['\"]?)",
)
_MAX_STREAMING_ENV_NAME = 100 + max(map(len, _SECRET_ENV_WORDS))


def _known_prefix_candidate_start(
    text: str, *, include_partial: bool = True,
) -> int | None:
    """Return the start of a trailing known-prefix token candidate."""
    held_from = len(text)
    token_boundary_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"

    for prefixes, token_chars, _minimum in _STREAMING_PREFIX_SPECS:
        for prefix in prefixes:
            # A complete literal prefix followed only by token characters stays
            # sensitive until a delimiter arrives. This includes candidates
            # that already reached the static redactor's minimum length.
            start = text.rfind(prefix)
            while start >= 0:
                if (
                    (start == 0 or text[start - 1] not in token_boundary_chars)
                    and token_chars.fullmatch(text[start + len(prefix):])
                ):
                    held_from = min(held_from, start)
                    break
                start = text.rfind(prefix, 0, start)

            # Also retain a literal-prefix fragment split across deltas.
            if not include_partial:
                continue
            for length in range(1, len(prefix)):
                if not text.endswith(prefix[:length]):
                    continue
                start = len(text) - length
                if start == 0 or text[start - 1] not in token_boundary_chars:
                    held_from = min(held_from, start)

    return held_from if held_from < len(text) else None


def _could_be_json_opener(candidate: str) -> bool:
    """Whether a trailing quote-led fragment can become a JSON secret opener."""
    if not candidate.startswith('"'):
        return False
    body = candidate[1:]
    lower = body.lower()
    for key in _STREAMING_JSON_KEYS:
        if len(body) <= len(key):
            if key.startswith(lower):
                return True
            continue
        if not lower.startswith(key):
            continue
        suffix = body[len(key):]
        if not suffix.startswith('"'):
            continue
        suffix = suffix[1:].lstrip()
        if not suffix:
            return True
        if not suffix.startswith(":"):
            continue
        suffix = suffix[1:].lstrip()
        return not suffix or suffix == '"'
    return False


def _could_be_db_opener(candidate: str) -> bool:
    """Whether a trailing fragment can become a DB password opener."""
    lower = candidate.lower()
    for protocol in _STREAMING_DB_PROTOCOLS:
        if protocol.startswith(lower):
            return True
        if not lower.startswith(protocol):
            continue
        username = candidate[len(protocol):]
        if not username:
            return True
        if any(char.isspace() for char in username):
            continue
        if ":" not in username:
            return True
        return username.endswith(":") and username.count(":") == 1
    return False


def _could_be_private_key_opener(candidate: str) -> bool:
    """Whether a trailing fragment can become a private-key BEGIN marker."""
    literal = "-----BEGIN"
    if literal.startswith(candidate):
        return True
    if not candidate.startswith(literal):
        return False
    remainder = candidate[len(literal):]
    marker = "PRIVATE KEY-----"
    for split_at in range(len(remainder) + 1):
        key_type = remainder[:split_at]
        marker_part = remainder[split_at:]
        if all(char == " " or "A" <= char <= "Z" for char in key_type):
            if marker.startswith(marker_part):
                return True
    return False


def _could_be_env_opener(candidate: str) -> bool:
    """Whether a trailing ENV-name fragment can become a secret assignment."""
    match = re.fullmatch(
        rf"([A-Z0-9_]{{1,{_MAX_STREAMING_ENV_NAME}}})(\s*)(?:(=)\s*['\"]?)?",
        candidate,
    )
    if match is None:
        return False
    name = match.group(1)
    def _canonical_word_position(word: str) -> int | None:
        for word_match in re.finditer(re.escape(word), name):
            if (
                word_match.start() <= 50
                and len(name) - word_match.end() <= 50
            ):
                return word_match.start()
        return None

    if match.group(2) or match.group(3):
        return any(_canonical_word_position(word) is not None for word in _SECRET_ENV_WORDS)
    if len(name) <= 50:
        # Any canonical pre-name fragment can still be followed by a full
        # secret word.
        return True
    for word in _SECRET_ENV_WORDS:
        if _canonical_word_position(word) is not None:
            return True
        if any(
            index <= 50
            and word.startswith(name[index:])
            for index in range(len(name))
        ):
            return True
    return False


def _could_be_header_candidate(
    candidate: str,
    literals: tuple[str, ...],
    *,
    authorization: bool,
) -> bool:
    """Whether a trailing fragment can become or extend a secret header."""
    lower = candidate.lower()
    for literal in literals:
        if literal.startswith(lower):
            return True
        if not lower.startswith(literal):
            continue
        remainder = candidate[len(literal):]
        if "\n" in remainder or "\r" in remainder:
            continue
        stripped = remainder.lstrip()
        if not stripped:
            return True
        tokens = stripped.split()
        trailing_space = stripped[-1].isspace()
        if not authorization:
            return len(tokens) == 1 and not trailing_space
        if len(tokens) == 1:
            # A scheme-shaped first token remains ambiguous with the canonical
            # bare-credential form until a following credential arrives.
            return (
                not trailing_space
                or re.fullmatch(r"[A-Za-z][\w.+-]*", tokens[0]) is not None
            )
        if len(tokens) == 2:
            return not trailing_space
        return False
    return False


def _partial_form_opener_start(text: str) -> int | None:
    """Return a pure form body's start when its last key is still incomplete."""
    line_start = text.rfind("\n") + 1
    candidate = text[line_start:]
    if not candidate or any(char.isspace() for char in candidate):
        return None
    parts = candidate.split("&")
    if any(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*", part) is None
        for part in parts[:-1]
    ):
        return None
    last = parts[-1]
    if "=" in last:
        key, value = last.split("=", 1)
    else:
        key = last
    lower = key.lower()
    if not key:
        return line_start
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key or ""):
        return None
    if any(secret.startswith(lower) for secret in _SENSITIVE_QUERY_PARAMS):
        return line_start
    return None


def _active_env_assignment_start(text: str) -> int | None:
    """Return an ENV assignment whose value is still growing at the tail."""
    candidates = []
    for match in _STREAMING_ENV_ACTIVE_RE.finditer(text):
        remainder = text[match.end():]
        quote = match.group(2)
        if quote:
            closing = remainder.find(quote)
            if closing < 0 or not remainder[closing + 1:]:
                candidates.append(match.start())
        elif remainder and not any(char.isspace() for char in remainder):
            candidates.append(match.start())
    return min(candidates) if candidates else None


def _partial_context_opener_start(
    text: str,
    *,
    embedded_prefixes: bool = True,
) -> int | None:
    """Return the earliest trailing fragment that can become an opener."""
    candidates: list[int] = []

    # Canonical JSON whitespace is unbounded. Check every quote that can begin
    # a field so a long run before the colon does not fall out of a tail window.
    for quote_start, char in enumerate(text):
        if char != '"':
            continue
        quote_has_field_boundary = (
            quote_start == 0 or text[quote_start - 1] in "{[, \t\r\n"
        )
        if (
            (embedded_prefixes or quote_has_field_boundary)
            and _could_be_json_opener(text[quote_start:])
        ):
            candidates.append(quote_start)
            break

    # DB usernames are unbounded in the static grammar. Search for each full
    # protocol across the whole buffer, then check short literal fragments at
    # the tail. This retains ``scheme://user`` even when a very long username
    # crosses the platform overflow threshold before its password colon.
    lower = text.lower()
    for protocol in _STREAMING_DB_PROTOCOLS:
        start = lower.rfind(protocol)
        if (
            start >= 0
            and (
                embedded_prefixes
                or start == 0
                or text[start - 1] not in
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
            )
            and _could_be_db_opener(text[start:])
        ):
            candidates.append(start)
        for length in range(1, len(protocol)):
            if not lower.endswith(protocol[:length]):
                continue
            start = len(text) - length
            if (
                embedded_prefixes
                or start == 0
                or text[start - 1] not in
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
            ):
                candidates.append(start)

    # A private-key BEGIN marker is bounded, so only its short trailing window
    # can still be incomplete.
    private_scan_start = 0
    for completed in _PRIVATE_KEY_RE.finditer(text):
        private_scan_start = completed.end()
    for start in range(private_scan_start, len(text)):
        if _could_be_private_key_opener(text[start:]):
            candidates.append(start)
            break

    # ENV names are bounded by the canonical 50/name/50 grammar, while header
    # whitespace is intentionally unbounded. Search the full trailing buffer so
    # the streaming language is exactly prefix-closed over both grammars.
    for start in range(len(text)):
        candidate = text[start:]
        if (
            _could_be_env_opener(candidate)
            or _could_be_header_candidate(
                candidate,
                _STREAMING_AUTH_HEADER_LITERALS,
                authorization=True,
            )
            or _could_be_header_candidate(
                candidate,
                _STREAMING_SECRET_HEADER_LITERALS,
                authorization=False,
            )
        ):
            candidates.append(start)
            break

    form_start = _partial_form_opener_start(text)
    if form_start is not None:
        candidates.append(form_start)
    env_value_start = _active_env_assignment_start(text)
    if env_value_start is not None:
        candidates.append(env_value_start)

    return min(candidates) if candidates else None


def _unterminated_context(text: str):
    """Return the earliest complete sensitive opener lacking its terminator."""
    active = []
    for match in _STREAMING_JSON_OPENER_RE.finditer(text):
        if text.find('"', match.end()) < 0:
            active.append((match.start(), "json", match))
    for match in _STREAMING_DB_OPENER_RE.finditer(text):
        if text.find("@", match.end()) < 0:
            active.append((match.start(), "db", match))
    for match in _STREAMING_PRIVATE_KEY_OPENER_RE.finditer(text):
        if _STREAMING_PRIVATE_KEY_END_RE.search(text, match.end()) is None:
            active.append((match.start(), "private_key", match))
    return min(active, key=lambda item: item[0]) if active else None


def _recognized_token_candidate_start(
    text: str,
    *,
    embedded_prefixes: bool = True,
) -> int | None:
    """Return a trailing non-prefix token already recognized as sensitive.

    These grammars redact an end-of-buffer token before a delimiter arrives.
    Retaining the raw match prevents later token bytes from being appended to
    a destructive mask. The canonical production regexes remain the source of
    truth; JWT only adds an allowed-character continuation around its shared
    canonical header pattern so a partial payload segment stays attached.
    """
    candidates: list[int] = []

    for pattern in (_TELEGRAM_RE, _SIGNAL_PHONE_RE):
        for match in pattern.finditer(text):
            if match.end() == len(text):
                candidates.append(match.start())

    jwt_match = _STREAMING_JWT_CANDIDATE_RE.search(text)
    if jwt_match is not None:
        candidates.append(jwt_match.start())

    # Hold candidates from the earliest point where the grammar is
    # recognizable, not only after the static redactor's minimum. Otherwise a
    # preview exposes almost the entire secret before the final byte masks it.
    telegram_match = _STREAMING_TELEGRAM_CANDIDATE_RE.search(text)
    if telegram_match is not None:
        candidates.append(telegram_match.start())
    jwt_prethreshold_re = (
        _STREAMING_JWT_PRETHRESHOLD_RE
        if embedded_prefixes
        else _STREAMING_JWT_BOUNDARY_PRETHRESHOLD_RE
    )
    jwt_prethreshold_match = jwt_prethreshold_re.search(text)
    if jwt_prethreshold_match is not None:
        candidates.append(jwt_prethreshold_match.start())
    phone_match = _STREAMING_PHONE_CANDIDATE_RE.search(text)
    if phone_match is not None:
        candidates.append(phone_match.start())

    return min(candidates) if candidates else None


def split_incomplete_sensitive_suffix(
    text: str,
    *,
    final: bool = False,
    logical_boundary: bool = False,
    embedded_prefixes: bool = True,
) -> tuple[str, str]:
    """Keep a raw trailing secret candidate out of streaming writes.

    The static redactor needs a whole match. A stream can end a platform write
    before a known-prefix delimiter, JSON closing quote, DB ``@`` delimiter, or
    PEM end marker arrives. Retain that raw suffix so later deltas are matched
    with their original context instead of appending to destructively masked
    display text.

    On the terminal stream tick, conservatively replace an unterminated
    structured value. Prefix-shaped prose below a known pattern's minimum is
    released unchanged because it never became a credential match.
    """
    if not text:
        return text, ""

    context = _unterminated_context(text)
    # ``logical_boundary`` remains in the public call shape for compatibility,
    # but segment/commentary boundaries are attacker-influenced and therefore
    # cannot terminate secret recognition state.
    known_start = _known_prefix_candidate_start(text)
    partial_start = _partial_context_opener_start(
        text,
        embedded_prefixes=embedded_prefixes,
    )
    token_start = _recognized_token_candidate_start(
        text,
        embedded_prefixes=embedded_prefixes,
    )

    if final:
        if context is None:
            return text, ""
        start, kind, match = context
        if kind == "json":
            return text[:start] + match.group(0) + '***"', ""
        if kind == "db":
            return text[:start] + match.group(0) + "***", ""
        return text[:start] + "[REDACTED PRIVATE KEY]", ""

    starts = [
        start
        for start in (
            context[0] if context is not None else None,
            known_start,
            partial_start,
            token_start,
        )
        if start is not None
    ]
    if not starts:
        return text, ""
    held_from = min(starts)
    # A one-character candidate can begin inside otherwise ordinary prose
    # (for example the ``re`` at the end of ``more`` can still become
    # ``redis://`` because the canonical DB grammar has no left boundary).
    # Retain its containing lexical token as well, avoiding permanent
    # mid-word splits when a logical platform boundary lands at that point.
    while (
        held_from > 0
        and text[held_from - 1]
        in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.+-"
    ):
        held_from -= 1
    return text[:held_from], text[held_from:]


def sanitize_terminal_secret_text(text: str) -> str:
    """Force-redact a complete adapter-bound text payload."""
    if not text:
        return text
    terminal_text, _held = split_incomplete_sensitive_suffix(
        str(text),
        final=True,
    )
    return redact_sensitive_text(terminal_text, force=True)


def sanitize_terminal_secret_url(url: str) -> str:
    """Force-redact credentials from an adapter-bound URL.

    General text intentionally preserves opaque web query parameters because
    agents often need to follow signed links. A URL that may fall back to
    plaintext delivery is a different terminal boundary: query credentials
    and userinfo must be neutralized before any native/fallback adapter sees it.
    """
    safe_url = sanitize_terminal_secret_text(url)
    return _redact_strict_url_credentials(safe_url)


class StreamingSecretSanitizer:
    """Stateful forced redaction for sequential adapter-bound text fragments.

    Formatting must be applied after ``feed``: inserting labels, quotes, or
    fences between fragments would otherwise break the canonical grammar that
    the retained bytes are meant to recognize.
    """

    def __init__(self, *, token_candidates_only: bool = False) -> None:
        self._pending = ""
        self._token_candidates_only = token_candidates_only

    @property
    def pending(self) -> str:
        return self._pending

    def feed(self, text: str, *, final: bool = False) -> str:
        combined = self._pending + str(text or "")
        if self._token_candidates_only and not final:
            full_known_start = _known_prefix_candidate_start(
                combined,
                include_partial=False,
            )
            partial_known_start = _known_prefix_candidate_start(combined)
            token_start = _recognized_token_candidate_start(combined)
            context = _unterminated_context(combined)
            starts = []
            if full_known_start is not None:
                starts.append(full_known_start)
            if partial_known_start == 0:
                starts.append(partial_known_start)
            if token_start == 0:
                starts.append(token_start)
            # Progress callbacks are independent display events, so an
            # unterminated DB URL in one event must not absorb the next event.
            # JSON and ENV fragments are different: their opener/value grammar
            # is explicitly structured and may be split by the producer across
            # consecutive callbacks, so retain that complete state.
            if context is not None and context[1] == "json":
                starts.append(context[0])
            for quote_start, char in enumerate(combined):
                quote_has_field_boundary = (
                    quote_start == 0
                    or combined[quote_start - 1] in "{[, \t\r\n"
                )
                if (
                    char == '"'
                    and quote_has_field_boundary
                    and _could_be_json_opener(combined[quote_start:])
                ):
                    starts.append(quote_start)
                    break
            for env_start in range(len(combined)):
                if (
                    (
                        env_start == 0
                        or combined[env_start - 1]
                        not in
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
                    )
                    and _could_be_env_opener(combined[env_start:])
                ):
                    starts.append(env_start)
                    break
            env_value_start = _active_env_assignment_start(combined)
            if env_value_start is not None:
                starts.append(env_value_start)
            if starts:
                held_from = min(starts)
                while (
                    held_from > 0
                    and combined[held_from - 1]
                    in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.+-"
                ):
                    held_from -= 1
                visible, self._pending = (
                    combined[:held_from],
                    combined[held_from:],
                )
            else:
                visible, self._pending = combined, ""
        else:
            visible, self._pending = split_incomplete_sensitive_suffix(
                combined,
                final=final,
            )
        return sanitize_terminal_secret_text(visible)

    def flush(self) -> str:
        return self.feed("", final=True)


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


_CANONICAL_SENSITIVE_QUERY_PARAMS = frozenset(
    key.casefold().replace("-", "_")
    for key in _SENSITIVE_QUERY_PARAMS
)


def _redact_strict_url_credentials(text: str) -> str:
    """Redact credentials from absolute, relative, and network URL references.

    This is intentionally stricter than display/log redaction and is used only
    at explicit secret-egress boundaries. It preserves original keys,
    separators, public parameters, hosts, and paths while masking sensitive
    values and URL userinfo.
    """
    def _redact_param(match: re.Match) -> str:
        if (
            _canonical_url_param_name(match.group(2))
            not in _CANONICAL_SENSITIVE_QUERY_PARAMS
        ):
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
