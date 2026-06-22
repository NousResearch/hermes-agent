"""Regex-based secret redaction for logs and tool output.

Applies pattern matching to mask API keys, tokens, and credentials
before they reach log files, verbose output, or gateway logs.

Short tokens (< 18 chars) are fully masked. Longer tokens preserve
the first 6 and last 4 characters for debuggability.
"""

import logging
import os
import re
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
    "operator_approval",
    "approval_token",
    "live_usb_approval",
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
    "operator_approval",
    "approval_token",
    "live_usb_approval",
})

_SENSITIVE_APPROVAL_KEYS = frozenset({
    "operator_approval",
    "approval_token",
    "live_usb_approval",
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
    r"xox[baprs]-[A-Za-z0-9-]{10,}",    # Slack tokens
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
]

# ENV assignment patterns: KEY=value where KEY contains a secret-like name.
# LIVE_USB_APPROVAL is deliberately included because AgentCyber uses an
# operator-controlled one-time approval as a high-consequence safety gate; it
# must not persist in tool-call history, audit logs, or command output.
_SECRET_ENV_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|LIVE_USB_APPROVAL)"
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)(\S+)\2",
)

# JSON / Python-dict-repr field patterns: "apiKey": "value", 'token': 'value', etc.
_JSON_KEY_NAMES = r"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|auth_token|bearer|secret_value|raw_secret|secret_input|key_material|operator[_-]approval|approval[_-]token|live[_-]usb[_-]approval)"
_JSON_FIELD_RE = re.compile(
    rf'(["\']({_JSON_KEY_NAMES})["\'])\s*:\s*("(?:\\.|[^"])*"|\'(?:\\.|[^\'])*\')',
    re.IGNORECASE,
)

_APPROVAL_KEY_PATTERN = (
    r"(?:(?<=--)|(?<![A-Za-z0-9_.%-]))(?:HERMES_AGENTCYBER_LIVE_USB_APPROVAL|operator[_-]approval|"
    r"approval[_-]token|live[_-]usb[_-]approval)"
)
_APPROVAL_QUOTED_ASSIGN_RE = re.compile(
    rf"((?:--)?{_APPROVAL_KEY_PATTERN}\s*(?:=|:)\s*)(\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')",
    re.IGNORECASE,
)
_APPROVAL_UNQUOTED_ASSIGN_RE = re.compile(
    rf"((?:--)?{_APPROVAL_KEY_PATTERN}\s*(?:=|:)\s*)([^\s&;|,'\"}})]+)",
    re.IGNORECASE,
)
_APPROVAL_LINE_ASSIGN_RE = re.compile(
    rf"((?:--)?{_APPROVAL_KEY_PATTERN}\s*(?:=|:)\s*)([^\r\n;&|,)}}]+)",
    re.IGNORECASE,
)
_APPROVAL_QUOTED_FLAG_RE = re.compile(
    r"(--(?:operator-approval|approval-token|live-usb-approval)\s+)(\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')",
    re.IGNORECASE,
)
_APPROVAL_UNQUOTED_FLAG_RE = re.compile(
    r"(--(?:operator-approval|approval-token|live-usb-approval)\s+)([^\s&;|,'\"})]+)",
    re.IGNORECASE,
)
_APPROVAL_LINE_FLAG_RE = re.compile(
    r"(--(?:operator-approval|approval-token|live-usb-approval)\s+)([^\r\n;&|,)]+)",
    re.IGNORECASE,
)
_FORM_PAIR_ANYWHERE_RE = re.compile(
    r"(?<![A-Za-z0-9_.%+-])([A-Za-z0-9_.%+-]+)=([^\r\n;&|,)]+)"
)
_FORM_QUOTED_PAIR_ANYWHERE_RE = re.compile(
    r"(?<![A-Za-z0-9_.%+-])([A-Za-z0-9_.%+-]+)=(\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')"
)
_APPROVAL_VALUE_SUFFIX_RE = re.compile(
    r"\s+(--[A-Za-z0-9][A-Za-z0-9_-]*\b|-[A-Za-z]\b|(?:https?|wss?|ftp)://)"
)

# Authorization headers — any scheme (Bearer, Basic, Token, Digest, …) plus the
# bare-credential form, and Proxy-Authorization. The credential token is masked
# while the header name and scheme word are preserved for debuggability. The
# previous rule only matched ``Bearer``, so ``Basic <base64 user:pass>`` and
# ``token <pat>`` leaked verbatim into logs/transcripts.
_AUTH_HEADER_RE = re.compile(
    r"((?:Proxy-)?Authorization:\s*)([A-Za-z][\w.+-]*\s+)?(\S+)",
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
# Catches postgres, mysql, mongodb, redis, amqp URLs and redacts the password
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:]+:)([^@]+)(@)",
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

# HTTP access logs often use a relative request target rather than a full URL:
# `"POST /webhook?password=... HTTP/1.1"`. The full-URL redactor above only
# sees strings containing `://`, so handle request-target query strings too.
_HTTP_REQUEST_TARGET_QUERY_RE = re.compile(
    r"\b((?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|TRACE|CONNECT)\s+[^ \t\r\n\"']*?)"
    r"\?([^ \t\r\n\"']+)",
    re.IGNORECASE,
)

# Form-urlencoded body detection: conservative — only applies when the entire
# text looks like a query/form body (k=v or k=v&k=v pattern with no newlines).
_FORM_BODY_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.%+-]*=[^&\s]*(?:&[A-Za-z_][A-Za-z0-9_.%+-]*=[^&\s]*)*$"
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


def _canonical_sensitive_key(key: str) -> str:
    """Normalize query/form keys before comparing with sensitive key sets."""
    decoded = key
    # Decode a bounded number of times so percent-encoded approval keys such as
    # ``operator%5Fapprov%61l`` cannot bypass the approval-token exception.
    for _ in range(3):
        next_decoded = unquote_plus(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    return decoded.lower().replace("-", "_")


def _redact_query_string(
    query: str,
    *,
    sensitive_params: frozenset[str] = _SENSITIVE_QUERY_PARAMS,
) -> str:
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
        decoded_key_raw = unquote_plus(key).lower()
        decoded_key = _canonical_sensitive_key(key)
        normalized_params = {param.replace("-", "_") for param in sensitive_params}
        if decoded_key_raw in sensitive_params or decoded_key in normalized_params:
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


def _redact_live_usb_approval_url_query_params(text: str) -> str:
    """Redact only Live USB approval query params while preserving other URL params.

    Broad URL query redaction is intentionally disabled below because agents
    often need opaque magic-link/OAuth/pre-signed URLs intact. Live USB approval
    tokens are different: they are high-consequence operator approvals and
    should never persist when accidentally embedded in a URL.
    """
    def _sub(m: re.Match) -> str:
        scheme = m.group(1)
        authority = m.group(2)
        path = m.group(3)
        query = _redact_query_string(m.group(4), sensitive_params=_SENSITIVE_APPROVAL_KEYS)
        fragment = m.group(5) or ""
        return f"{scheme}://{authority}{path}?{query}{fragment}"

    return _URL_WITH_QUERY_RE.sub(_sub, text)


def _split_approval_value_suffix(value: str) -> tuple[str, str]:
    """Split a redacted unquoted approval value from following safe context."""
    suffix_match = _APPROVAL_VALUE_SUFFIX_RE.search(value)
    if suffix_match:
        return value[:suffix_match.start()], value[suffix_match.start():]
    return value, ""


def _redact_quoted_approval_value(value: str) -> str | None:
    """Return a redacted quoted approval literal, preserving suffix, if valid."""
    if not value or value[0] not in {'"', "'"}:
        return None
    quote = value[0]
    escaped = False
    for idx in range(1, len(value)):
        ch = value[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == quote:
            return f"{quote}***{quote}{value[idx + 1:]}"
    return None


def _redact_approval_text_patterns(text: str) -> str:
    """Redact Live USB approval tokens in common prose/CLI assignment forms."""
    def _redact_line_value(m: re.Match) -> str:
        value = m.group(2)
        quoted = _redact_quoted_approval_value(value)
        if quoted is not None:
            return f"{m.group(1)}{quoted}"
        _secret, suffix = _split_approval_value_suffix(value)
        return f"{m.group(1)}***{suffix}"

    def _redact_quoted_value(m: re.Match) -> str:
        value_literal = m.group(2)
        return f"{m.group(1)}{value_literal[0]}***{value_literal[-1]}"

    text = _APPROVAL_QUOTED_ASSIGN_RE.sub(_redact_quoted_value, text)
    text = _APPROVAL_QUOTED_FLAG_RE.sub(_redact_quoted_value, text)
    text = _APPROVAL_LINE_ASSIGN_RE.sub(_redact_line_value, text)
    text = _APPROVAL_LINE_FLAG_RE.sub(_redact_line_value, text)
    text = _APPROVAL_UNQUOTED_ASSIGN_RE.sub(lambda m: f"{m.group(1)}***", text)
    return _APPROVAL_UNQUOTED_FLAG_RE.sub(lambda m: f"{m.group(1)}***", text)


def _redact_approval_form_pairs_anywhere(text: str) -> str:
    """Redact approval form pairs embedded in shell/prose text.

    This intentionally only redacts keys that canonicalize to Live USB approval
    names, including percent-encoded spellings such as `operator%5Fapprov%61l`.
    """
    def _sub_quoted(m: re.Match) -> str:
        key = m.group(1)
        if _canonical_sensitive_key(key) not in _SENSITIVE_APPROVAL_KEYS:
            return m.group(0)
        value = m.group(2)
        return f"{key}={value[0]}***{value[-1]}"

    def _sub(m: re.Match) -> str:
        key = m.group(1)
        value = m.group(2)
        if _canonical_sensitive_key(key) not in _SENSITIVE_APPROVAL_KEYS:
            return m.group(0)
        quoted = _redact_quoted_approval_value(value)
        if quoted is not None:
            return f"{key}={quoted}"
        _secret, suffix = _split_approval_value_suffix(value)
        return f"{key}=***{suffix}"

    text = _FORM_QUOTED_PAIR_ANYWHERE_RE.sub(_sub_quoted, text)
    return _FORM_PAIR_ANYWHERE_RE.sub(_sub, text)


def _redact_url_userinfo(text: str) -> str:
    """Strip `user:password@` from HTTP/WS/FTP URLs.

    DB protocols (postgres, mysql, mongodb, redis, amqp) are handled
    separately by `_DB_CONNSTR_RE`.
    """
    return _URL_USERINFO_RE.sub(
        lambda m: f"{m.group(1)}://{m.group(2)}:***@",
        text,
    )


def _redact_http_request_target_query_params(
    text: str,
    *,
    sensitive_params: frozenset[str] = _SENSITIVE_QUERY_PARAMS,
) -> str:
    """Redact sensitive query params in HTTP access-log request targets."""
    def _sub(m: re.Match) -> str:
        prefix = m.group(1)
        query = _redact_query_string(m.group(2), sensitive_params=sensitive_params)
        return f"{prefix}?{query}"
    return _HTTP_REQUEST_TARGET_QUERY_RE.sub(_sub, text)


def _redact_form_body(text: str) -> str:
    """Redact sensitive values in a form-urlencoded body.

    Multi-field bodies keep the historical conservative behavior. Single-field
    bodies are redacted only for Live USB approval keys, avoiding broad false
    positives such as `code=200` or `key=value`.
    """
    if not text or "\n" in text or "=" not in text:
        return text
    stripped = text.strip()
    if not _FORM_BODY_RE.match(stripped):
        return text
    if "&" in stripped:
        return _redact_query_string(stripped)
    key, _, _value = stripped.partition("=")
    if _canonical_sensitive_key(key) in _SENSITIVE_APPROVAL_KEYS:
        return _redact_query_string(stripped, sensitive_params=_SENSITIVE_APPROVAL_KEYS)
    return text


def redact_sensitive_text(text: str, *, force: bool = False, code_file: bool = False) -> str:
    """Apply all redaction patterns to a block of text.

    Safe to call on any string -- non-matching text passes through unchanged.
    Enabled by default. Disable via security.redact_secrets: false in config.yaml.
    Set force=True for safety boundaries that must never return raw secrets
    regardless of the user's global logging redaction preference.

    Set code_file=True to skip the ENV-assignment and JSON-field regex
    patterns when the text is known to be source code (e.g. MAX_TOKENS=***
    constants, "apiKey": "test" fixtures). Prefix patterns, auth headers,
    private keys, DB connstrings, JWTs, and URL secrets are still redacted.

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

    # High-consequence Live USB approvals may contain spaces and are exact-match
    # tokens, so redact their common textual forms before generic env parsing.
    # Preserve code_file=True's historical false-positive guard for source text.
    if not code_file and "approval" in text.lower():
        text = _redact_approval_text_patterns(text)

    # Known prefixes (sk-, ghp_, etc.) — gate on substring presence
    if _has_known_prefix_substring(text):
        text = _PREFIX_RE.sub(lambda m: _mask_token(m.group(1)), text)

    # ENV assignments: OPENAI_API_KEY=***  (skip for code files — false positives)
    if not code_file:
        if "=" in text:
            def _redact_env(m):
                name, quote, value = m.group(1), m.group(2), m.group(3)
                masked = "***" if "LIVE_USB_APPROVAL" in name.upper() else _mask_token(value)
                return f"{name}={quote}{masked}{quote}"
            text = _ENV_ASSIGN_RE.sub(_redact_env, text)

        # JSON fields: "apiKey": "***"  (skip for code files — false positives)
        if ":" in text and ('"' in text or "'" in text):
            def _redact_json(m):
                key, field, value_literal = m.group(1), m.group(2), m.group(3)
                normalized_field = field.lower().replace("-", "_")
                quote = value_literal[0]
                value = value_literal[1:-1]
                masked = "***" if normalized_field in _SENSITIVE_APPROVAL_KEYS else _mask_token(value)
                return f'{key}: {quote}{masked}{quote}'
            text = _JSON_FIELD_RE.sub(_redact_json, text)

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

    # Database connection string passwords
    if "://" in text:
        text = _DB_CONNSTR_RE.sub(lambda m: f"{m.group(1)}***{m.group(3)}", text)

    # JWT tokens (eyJ... — base64-encoded JSON headers)
    if "eyJ" in text:
        text = _JWT_RE.sub(lambda m: _mask_token(m.group(0)), text)

    # Approval-token query params are a narrow exception to the URL-query
    # passthrough below: Live USB operator approvals are high-consequence
    # one-time tokens and must not persist if embedded in URLs or access logs.
    # Run this narrow pass on all URLs/access-log request targets so percent-
    # encoded approval keys cannot bypass a raw substring pre-check.
    if not code_file:
        text = _redact_approval_text_patterns(text)
        text = _redact_approval_form_pairs_anywhere(text)
        if "://" in text:
            text = _redact_live_usb_approval_url_query_params(text)
        if _has_http_method_substring(text):
            text = _redact_http_request_target_query_params(
                text,
                sensitive_params=_SENSITIVE_APPROVAL_KEYS,
            )

    # NOTE: Web-URL redaction (query params + userinfo + HTTP access-log
    # request targets) is intentionally OFF. Many legitimate workflows pass
    # opaque tokens through query strings — magic-link checkouts, OAuth
    # callbacks the agent is meant to follow, pre-signed share URLs — and
    # blanket-redacting param values by name breaks those skills mid-flow.
    # Known credential shapes (sk-, ghp_, JWTs, etc.) inside URLs are still
    # caught by _PREFIX_RE and _JWT_RE above. DB connection-string passwords
    # are still caught by _DB_CONNSTR_RE.

    # Form-urlencoded bodies. Multi-field bodies keep the historical behavior;
    # single-field bodies are limited to approval keys to avoid false positives.
    if not code_file and "=" in text:
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
