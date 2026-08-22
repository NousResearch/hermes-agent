"""Regex-based secret redaction for logs and tool output.

Applies pattern matching to mask API keys, tokens, and credentials
before they reach log files, verbose output, or gateway logs.

Short tokens (< 18 chars) are fully masked. Longer tokens preserve
the first 6 and last 4 characters for debuggability.
"""

import contextvars
import json
import logging
import os
import re
import shlex
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

# Basenames treated as ``.env`` files by _command_reads_env_file. Imported
# from agent/file_safety (the read-block list) so the two defenses can't
# drift: if file_tools blocks a read and the agent falls back to ``cat``,
# the terminal redactor still catches it. file_safety matches
# case-insensitively (``resolved.name.lower()``); the lookup mirrors that.
from agent.file_safety import _BLOCKED_PROJECT_ENV_BASENAMES as _ENV_FILE_BASENAMES

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
    # GitLab token families (each pattern keeps a full literal prefix so the
    # _PREFIX_SUBSTRINGS pre-screen stays false-negative-free). Ported from
    # openclaw/openclaw#112954; follow-up invited in #4541.
    r"glpat-[A-Za-z0-9_\-]{10,}",       # GitLab personal access token
    r"gloas-[A-Za-z0-9_\-]{10,}",       # GitLab OAuth application secret
    r"gldt-[A-Za-z0-9_\-]{10,}",        # GitLab deploy token
    r"glrt-[A-Za-z0-9_.\-]{10,}",       # GitLab runner authentication token (routable tokens are dotted)
    r"glrtr-[A-Za-z0-9_.\-]{10,}",      # GitLab runner registration token (routable)
    r"glcbt-[A-Za-z0-9_\-]{10,}",       # GitLab CI/CD job token
    r"glptt-[A-Za-z0-9_\-]{10,}",       # GitLab pipeline trigger token
    r"glft-[A-Za-z0-9_\-]{10,}",        # GitLab feed token
    r"glimt-[A-Za-z0-9_\-]{10,}",       # GitLab incoming mail token
    r"glagent-[A-Za-z0-9_\-]{10,}",     # GitLab agent (KAS) token
    r"glsoat-[A-Za-z0-9_\-]{10,}",      # GitLab service-account access token
    r"glffct-[A-Za-z0-9_\-]{10,}",      # GitLab feature-flags client token
    r"glwt-[A-Za-z0-9_\-]{10,}",        # GitLab workspace token
    r"GR1348941[A-Za-z0-9_\-]{10,}",    # GitLab legacy runner registration token
    r"pk-lf-[A-Za-z0-9\-]{8,}",         # Langfuse public key (sk-lf- already covered by sk- pattern)
]

# ENV assignment patterns: KEY=value where KEY contains a secret-like name.
# Uppercase keys tolerate spaces around "=" (e.g. ``FOO_SECRET = bar``) because
# an all-caps key is almost never prose/code.
# Bare ``KEY`` / ``PASS`` / ``PW`` suffixes are included (``FAL_KEY=…``,
# ``MYSQL_PASS=…``, ``DB_PW=…``) — issue #77484. The regex is IGNORECASE so
# lowercase env names (``openai_key=…``) are caught here too. The secret name
# must sit at a word boundary (``_``-delimited or whole-word) so generic
# prose words (``password=``, ``token=``, ``KEYBOARD=``, ``PASSAGE=``) do not
# match — those are handled by the config/form/URL paths, and a bare
# ``password=…`` in a form body must not be swallowed greedily by ``\S+``.
_SECRET_ENV_NAMES = r"(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASS|PW|CREDENTIAL|AUTH)"
# Uppercase keys keep the legacy embedded match (``MYTOKEN=…``, ``FOO_SECRET``)
# — an all-caps key is almost never prose.
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)(\S+)\2",
)
# Lowercase env names: only underscore-boundary forms (``openai_key=…``,
# ``FAL_KEY=…``, ``db_pw=…``) — NOT bare ``password=``/``token=``/``secret=``,
# which appear in prose, URLs, and form bodies (issue #77484).
_ENV_ASSIGN_LOWER_RE = re.compile(
    rf"([a-z0-9_]+(?:_|^)(?:key|pass|pw|token|secret|password|passwd|credential|auth)(?=[^a-z0-9_]|$))\s*=\s*(['\"]?)(\S+)\2",
    re.IGNORECASE,
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
# Linear pre-gate for the _CFG_*_RE subs below: a text with no secret keyword
# can never match either pattern, so the (potentially backtrack-heavy) subs
# are skipped entirely for such text. See the call site in
# redact_sensitive_text().
_CFG_SECRET_WORD_RE = re.compile(_SECRET_CFG_NAMES, re.IGNORECASE)

# Programmatic env lookups (``os.getenv(...)``, ``os.environ[...]``,
# ``os.environ.get(...)``, ``process.env.X``, ``$ENV{X}``) reference variable
# *names*, not secret values. When one appears as the VALUE of a KEY=... match
# it's a code snippet, not a leaked secret — skip redaction (issue #2852).
_ENV_LOOKUP_VALUE_RE = re.compile(
    r"^(?:os\.(?:getenv|environ)|process\.env|\$ENV\{)"
)
# Namespaced (dotted) key: the secret word may sit anywhere in a dotted path.
# NOTE(perf): possessive quantifiers (py3.11+) replace the nested quantifier
# ``(?:[A-Za-z0-9_\-]+\.)+`` (exponential backtracking on long dotted runs).
# The ``*`` runs bordering {_SECRET_CFG_NAMES} must stay backtrackable
# (secret words are matchable by the class, e.g. ``app.api.key=…``).
_CFG_DOTTED_RE = re.compile(
    rf"([A-Za-z0-9_\-]++\.[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*+"
    rf"|[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*\.[A-Za-z0-9_.\-]++)"
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
# NOTE(perf): possessive quantifiers wherever the successor is disjoint; the
# leading ``[A-Za-z0-9_.\-]*`` stays backtrackable (see _CFG_DOTTED_RE note).
_YAML_ASSIGN_RE = re.compile(
    rf"(^[ \t]*+[A-Za-z0-9_.\-]*{_YAML_CFG_NAMES}[A-Za-z0-9_.\-]*+)(:[ \t]*+)(?!['\"])([^\s&]++)",
    re.IGNORECASE | re.MULTILINE,
)

# Word-boundary validation for the mixed/lowercase key patterns above
# (_CFG_DOTTED_RE, _CFG_ANCHORED_RE, _YAML_ASSIGN_RE).
#
# Those key classes allow arbitrary alphanumeric affixes around the secret
# keyword so real key names like ``client_secret``, ``clientSecret``, and
# ``s3.secret-key`` match. The side effect: ordinary prose/document words that
# merely CONTAIN a keyword also matched — ``Secretary: J.Smith`` (secret),
# ``tokenizer: cl100k_base`` (token), ``author=Smith`` (auth) — mangling
# legitimate content on the surfaces that run these passes (browser snapshots,
# log lines, kanban summaries, CLI-echoed command output). Ported from
# nearai/ironclaw#6129, where the same substring false positive ("Secretary of
# the Treasury" matching the ``secret`` marker) scrubbed legitimate tool
# results from the replayed transcript and sent the model into a re-fetch
# loop.
#
# A keyword occurrence only counts when it sits at a word boundary within the
# key: at the key's edge, next to a non-letter (``_ - . 3``), or at a
# camelCase transition (``clientSecret``, ``secretKey``, ``APIToken``). A
# trailing plural ``s`` is treated as part of the keyword (``secrets:``,
# ``tokens:``). Common concatenated compounds keep matching via explicit
# alternatives (``authtoken`` ngrok, ``authkey`` tailscale, ``secretkey``
# minio, ``apikey``). Embedded occurrences inside a larger word
# (``secretary``, ``tokenizer``, ``authored``, ``credentialing``) no longer
# match. ALL-CAPS keys keep the legacy embedded matching (``MYTOKEN=…``) — an
# all-caps key is almost never prose, the same rationale as _ENV_ASSIGN_RE.
_KEY_KEYWORD_RE = re.compile(
    r"(?:api|auth|access|refresh|session|secret)[ _.\\-]?(?:key|token)"
    r"|token|secret|passwd|password|pass|pw|credential|auth|key",
    re.IGNORECASE,
)


def _is_word_start(s: str, i: int) -> bool:
    """True if position ``i`` in ``s`` begins a word (not mid-word)."""
    if i == 0:
        return True
    prev, cur = s[i - 1], s[i]
    if not prev.isalpha():
        return True
    if cur.isupper() and prev.islower():
        return True  # camelCase: clientSecret
    # Acronym run ending: APIToken — the 'T' begins a new word when it is
    # followed by lowercase while the preceding run is uppercase.
    if cur.isupper() and prev.isupper() and i + 1 < len(s) and s[i + 1].islower():
        return True
    return False


def _is_word_end(s: str, j: int, *, allow_plural: bool = True) -> bool:
    """True if position ``j`` (exclusive end) in ``s`` ends a word."""
    if j >= len(s):
        return True
    cur = s[j]
    if not cur.isalpha():
        return True
    if cur.isupper() and s[j - 1].islower():
        return True  # camelCase continuation: secretKey
    if allow_plural and cur in "sS":
        return _is_word_end(s, j + 1, allow_plural=False)
    return False


def _key_has_secret_keyword(key: str) -> bool:
    """True if ``key`` contains a secret keyword at a word boundary.

    Post-match validator for _CFG_DOTTED_RE / _CFG_ANCHORED_RE /
    _YAML_ASSIGN_RE hits — rejects prose words that merely embed a keyword
    (``secretary``, ``tokenizer``, ``authored``). Safe to call with the
    _ENV_ASSIGN_RE key too: all-caps keys short-circuit to the legacy
    embedded-match behavior.
    """
    letters = [c for c in key if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        # Legacy all-caps behavior (MYTOKEN=…): an all-caps key is almost
        # never prose. Exception: a bare ``KEY``/``PASS``/``PW`` embedded in
        # a longer all-caps word (``KEYBOARD``, ``PASSAGE``) is prose, not a
        # credential — only a word-bounded compound (``API_KEY``,
        # ``MYSQL_PASSWORD``, ``FAL_KEY``, ``DB_PW``) counts (issue #77484).
        for m in _KEY_KEYWORD_RE.finditer(key):
            if _is_word_start(key, m.start()) and _is_word_end(key, m.end()):
                return True
        return False
    for m in _KEY_KEYWORD_RE.finditer(key):
        if _is_word_start(key, m.start()) and _is_word_end(key, m.end()):
            return True
    return False

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
#
# Anchored on the mandatory ``//`` rather than an optional scheme prefix: the
# scheme sits outside the match either way (replacement callbacks re-emit
# group(1), so ``https:`` stays untouched in the surrounding text), and the
# old optional-scheme prefix ``(?:[A-Za-z][A-Za-z0-9+.-]*:)?`` backtracked
# catastrophically (O(n²)) on long unbroken alphanumeric runs — a 320KB
# synthetic compaction payload spent ~55s inside this pattern per sub() call.
# Output-equivalence to the old pattern was fuzz-verified (20k random strings
# plus targeted URL forms).
_STRICT_URL_USERINFO_RE = re.compile(
    r"(//)([^/\s?#@]+)@"
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

# Control / zero-width characters that can split a token body: a secret
# smuggled as ``sk-abc\x1bdef…`` or ``ghp_abc\n123…`` escapes the contiguous
# prefix regexes (issue #77484). Used by _mask_control_split_tokens.
_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x1f\x7f\u200b-\u200f\u2028-\u202f\u2060\ufeff]"
)

# Union of every _PREFIX_PATTERNS body class — a control-stripped match may
# only span original chars that are token-body or control chars (see
# _mask_control_split_tokens). ``=`` is deliberately excluded: a KEY=value
# assignment separator must never let a match span across unrelated text.
_TOKEN_BODY_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-."
)

# Compile known prefix patterns into one alternation
_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)


def _mask_control_split_tokens(text: str, mask_fn) -> str:
    """Mask tokens whose body is split by control/zero-width characters.

    A credential like ``sk-abc\\x1bdef456…`` or ``ghp_abc\\n123def…`` has its
    token body interrupted, so the contiguous _PREFIX_RE cannot match it and
    the secret leaks verbatim (issue #77484). Strategy: build a copy with all
    control chars removed (the token is contiguous again, matching even when
    each fragment alone is too short), match on that, then mask the
    corresponding span in the *original* — but only when the original span
    contains solely token-body and control chars (a match that crosses into a
    different line's unrelated text, e.g. ``EXA_API_KEY=*** is rejected).
    """
    stripped = _CONTROL_CHARS_RE.sub("", text)
    if stripped == text:
        return text
    orig_idx = [i for i, c in enumerate(text) if not _CONTROL_CHARS_RE.match(c)]
    out = list(text)
    matches = []
    for m in _PREFIX_RE.finditer(stripped):
        body = m.group(1)
        start_orig = orig_idx[m.start(1)]
        end_orig = orig_idx[m.end(1) - 1] + 1
        # If a fragment inside the span already matches _PREFIX_RE on its
        # own AND the span crosses a LINE boundary (\n / \r), do NOT join.
        # A complete token at end-of-line followed by a word line
        # (``ghp_<token>\nbutton [ref=e3]``) joins into one stripped-copy
        # match and the mask eats ``button``. Line structure is legitimate;
        # the self-matching fragment is handled by the ordinary prefix pass
        # (any remainder past the newline is left unmasked — accepted
        # residual to preserve line structure).
        # For NON-newline controls (ESC, ZWSP, ...) the join proceeds even
        # when a fragment self-matches: those bytes never legitimately sit
        # between a token and adjacent prose, and skipping there let the
        # non-matching remainder of a split token leak
        # (``sk-<head>\x1b<tail>`` masked only the head).
        span = text[start_orig:end_orig]
        if ("\n" in span or "\r" in span) and _PREFIX_RE.search(span):
            continue
        # Reject matches whose original span crosses a non-token char
        # (e.g. ``sk_abc…\nTAVILY_API_KEY=…`` — the ``=`` is not part of a
        # token body, so the regex matched across unrelated lines). Also
        # reject when the match runs into a ``KEY=`` name: a real token value
        # is followed by a newline/space/end, not ``=``.
        if (all(c in _TOKEN_BODY_CHARS or _CONTROL_CHARS_RE.match(c)
                for c in span)
                and (end_orig >= len(text) or text[end_orig] != "=")):
            matches.append((start_orig, end_orig, mask_fn(body)))
    for start_orig, end_orig, replacement in reversed(matches):
        out[start_orig:end_orig] = list(replacement)
    return "".join(out)


# Display-mask strip for mask_secret: EVERY control char incl. \n/\t, C1,
# DEL, and zero-width/format chars — a masked secret must never emit
# multiline, tabbed, or invisible bytes into config/status/dump display
# output (#55319, #55321).
_DISPLAY_CONTROL_RE = re.compile(
    r"[\x00-\x1f\x7f\x80-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2064]"
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
    # Visible head/tail must not carry control bytes (newline, NUL, DEL, C1)
    # into config/status/dump output (#55319, #55321). Strip them before
    # slicing — the length check below then sees the displayable length.
    value = _DISPLAY_CONTROL_RE.sub("", value)
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


# Exact-value masking at the provider boundary. External secret sources can
# populate arbitrary variable names, so shape-based redaction cannot identify
# their values when a tool echoes them back (#77487).
_CREDENTIAL_ENV_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_PASSWD",
    "_PASS",
    "_PW",
    # Repository redaction and execution-environment filters recognize these
    # established credential/auth aliases too (for example
    # ``VAULT_CREDENTIAL``). Do not broaden this to generic ``*_CREDS`` or
    # ``*_BEARER`` names: authoritative external-secret snapshots already
    # contribute those values regardless of name, while ambient process/scope
    # values need repository-backed classification to avoid transcript damage.
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_AUTH",
)
_CONFIG_FLAG_ENV_NAMES = frozenset(
    {
        # Honcho documents this as a self-hosted boolean switch, not its JWT.
        # Only a recognized flag value is exempt: an arbitrary value under the
        # same credential-looking name must still be protected.
        "AUTH_USE_AUTH",
    }
)
_CONFIG_FLAG_ENV_PREFIXES = (
    "SKIP_",
    "DEBUG_",
    "REQUIRE_",
    "SHOW_",
    "USE_",
    "DISABLE_",
    "ENABLE_",
    "ALLOW_",
    "VERIFY_",
    "CHECK_",
    "HAS_",
    "IS_",
    "WITH_",
    "AUTO_",
)
_CONFIG_FLAG_ENV_VALUES = frozenset(
    {
        "0",
        "1",
        "true",
        "false",
        "yes",
        "no",
        "on",
        "off",
        "enabled",
        "disabled",
        "auto",
    }
)
_CREDENTIAL_ENV_NAMES = frozenset(
    {
        # Repository-supported credential aliases that deliberately use the
        # otherwise-generic ``*_KEY`` shape. Keep this exact: names such as
        # PARTITION_KEY, SORT_KEY, CACHE_KEY, and IDEMPOTENCY_KEY hold ordinary
        # configuration whose short values would corrupt provider transcripts
        # if collected for literal masking.
        "ALPHA_VANTAGE_KEY",
        "API_KEY",
        "API_SERVER_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_ANTHROPIC_KEY",
        "BUZZ_PRIVATE_KEY",
        "FAL_KEY",
        "FEISHU_ENCRYPT_KEY",
        "GATEWAY_PROXY_KEY",
        "GATEWAY_RELAY_DELIVERY_KEY",
        "HERMES_IRON_PROXY_MGMT_KEY",
        "HERMES_LANGFUSE_SECRET_KEY",
        "HERMES_MEET_REALTIME_KEY",
        "LANGFUSE_SECRET_KEY",
        "MATRIX_RECOVERY_KEY",
        "PORCUPINE_ACCESS_KEY",
        "VOICE_TOOLS_OPENAI_KEY",
        "WECOM_CALLBACK_ENCODING_AES_KEY",
        # Repository-owned bearer-token aliases whose credential word is not
        # terminal. Keep these exact rather than collecting every ambient
        # ``*_BEARER*`` setting.
        "A2A_BEARER_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        # Repository plugin manifests declare these password values even
        # though their established names use otherwise-generic suffixes.
        "A2A_PEER_TOKENS",
        # Common password/auth variables whose established spellings do not
        # have an underscore-delimited credential suffix.
        "PGPASSWORD",
        "MYSQL_PWD",
        "PASSWORD",
        "PASSWD",
        "PASS",
        "PW",
        "REDISCLI_AUTH",
    }
)
_VALUE_SENSITIVE_CREDENTIAL_ENV_NAMES = frozenset(
    {
        # Google Chat accepts either env name as an inline service-account
        # object or a path. Only the inline form is credential material; exact
        # path values must remain visible in provider transcripts.
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON",
    }
)
_INLINE_CREDENTIAL_JSON_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "client_secret",
        "id_token",
        "password",
        "private_key",
        "refresh_token",
    }
)


def _is_credential_env_name(name: str, value: str | None = None) -> bool:
    """Classify credential variables without matching ordinary names."""
    normalized = name.upper()
    if (
        isinstance(value, str)
        and (
            normalized in _CONFIG_FLAG_ENV_NAMES
            or normalized.startswith(_CONFIG_FLAG_ENV_PREFIXES)
        )
        and value.strip().lower() in _CONFIG_FLAG_ENV_VALUES
    ):
        # Suffixes such as ``*_AUTH`` and ``*_TOKEN`` are also used by
        # boolean/configuration switches (for example ``SKIP_AUTH=false``).
        # Exclude only unambiguous flag-shaped names with flag-shaped values.
        # Arbitrary short values under the same names remain exact secrets.
        return False
    if normalized in _VALUE_SENSITIVE_CREDENTIAL_ENV_NAMES:
        return isinstance(value, str) and value.lstrip().startswith("{")
    return normalized in _CREDENTIAL_ENV_NAMES or normalized.endswith(
        _CREDENTIAL_ENV_SUFFIXES
    )


def _resolve_secret_home(home: str | os.PathLike | None) -> Path:
    if home is not None:
        return Path(home)
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def _collect_nested_credential_scalars(value: Any, values: set[str]) -> None:
    """Collect credential-bearing string fields from an inline JSON object."""
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if (
                    isinstance(key, str)
                    and key.lower() in _INLINE_CREDENTIAL_JSON_KEYS
                    and isinstance(item, str)
                    and item
                ):
                    values.add(item)
                if isinstance(item, (dict, list)):
                    pending.append(item)
        elif isinstance(current, list):
            pending.extend(
                item for item in current if isinstance(item, (dict, list))
            )


def _collect_credential_container_values(
    name: str,
    value: str,
    values: set[str],
) -> None:
    """Collect exact component secrets from supported credential containers."""
    normalized = name.upper()
    if normalized == "A2A_PEER_TOKENS":
        # Mirror plugins/platforms/a2a/security.py::get_peer_tokens: only a
        # nonempty name/token pair is a configured credential. Keep the whole
        # container too (collected by the caller), but also protect the token
        # when a peer or tool echoes it independently.
        for pair in value.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            peer_name, token = pair.split(":", 1)
            peer_name, token = peer_name.strip(), token.strip()
            if peer_name and token:
                values.add(token)

    if normalized not in _VALUE_SENSITIVE_CREDENTIAL_ENV_NAMES:
        return
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return
    if isinstance(parsed, dict):
        _collect_nested_credential_scalars(parsed, values)


def _collect_exact_secret_values(home: Path) -> tuple[str, ...]:
    """Return the active profile's exact values eligible for masking."""
    values: set[str] = set()
    try:
        from hermes_cli.env_loader import get_secret_source_values

        snapshot = get_secret_source_values(home)
    except Exception:
        logger.debug(
            "exact-value secret snapshot unavailable for %s", home, exc_info=True
        )
        snapshot = {}
    for name, value in snapshot.items():
        if not isinstance(value, str):
            continue
        values.add(value)
        if isinstance(name, str):
            _collect_credential_container_values(name, value, values)

    # Prefer the context-local profile scope. In a single-profile process it is
    # an overlay on os.environ (matching secret_scope.get_secret), so retain
    # process-only credentials and let the scope win duplicate names. Under
    # multiplexing the scope remains authoritative and process-global values
    # may belong to another profile.
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        scope = current_secret_scope()
        multiplex_active = is_multiplex_active()
        if multiplex_active:
            environment = scope or {}
        elif scope is not None:
            environment = {**os.environ, **scope}
        else:
            environment = os.environ
    except Exception:
        environment = os.environ
    for name, value in environment.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, str)
            or not _is_credential_env_name(name, value)
        ):
            continue
        values.add(value)
        _collect_credential_container_values(name, value, values)

    # Every non-empty value from these authoritative sources is sensitive.
    # Source classification, rather than a length/shape heuristic, is what
    # prevents broad false positives: arbitrary snapshot names are values
    # applied by the active external-secret provider, while the scoped
    # environment contributes only credential-suffixed names.
    return tuple(
        sorted(
            (value for value in values if value),
            key=lambda value: (-len(value), value),
        )
    )


@dataclass(frozen=True)
class _ExactSecretPattern:
    """One operation's exact matcher and collision-safe replacement marker."""

    regex: re.Pattern
    replacement: str
    secret_forms: tuple[str, ...]


def _choose_exact_secret_replacement(secret_forms: set[str]) -> str:
    """Return a marker that cannot expose or turn into an active secret.

    ``***`` stays the stable public marker for ordinary credentials. If it is
    itself sensitive (or contains a short sensitive value), prefer another
    readable marker. The one-character fallback is selected outside every
    character occurring in any active literal/JSON form, which makes it safe
    even for one-character secrets. Deletion is the fail-closed theoretical
    fallback when the configured values collectively contain every Unicode
    scalar value.
    """

    def repeated_candidate_contains(secret_form: str, candidate: str) -> bool:
        # Any match in an arbitrarily repeated marker starts within the first
        # marker period and ends within ``len(secret_form)`` characters. Build
        # only that finite window instead of imposing a repetition count.
        required_length = len(secret_form) + len(candidate) - 1
        repetitions = (required_length + len(candidate) - 1) // len(candidate)
        return secret_form in candidate * repetitions

    def candidate_boundary_composes(secret_form: str, candidate: str) -> bool:
        # Unchanged source immediately before or after a replacement can
        # contribute the rest of a secret. Reject every readable marker whose
        # leading bytes overlap a secret suffix or whose trailing bytes overlap
        # a secret prefix. For example, replacing ``x`` in ``abx`` with ``***``
        # must not synthesize the active secret ``ab*`` at the left boundary.
        overlap_limit = min(len(secret_form), len(candidate))
        return any(
            secret_form.endswith(candidate[:overlap])
            or secret_form.startswith(candidate[-overlap:])
            for overlap in range(1, overlap_limit + 1)
        )

    def safe(candidate: str) -> bool:
        if not candidate:
            return False
        return all(
            candidate not in secret_form
            and not repeated_candidate_contains(secret_form, candidate)
            and not candidate_boundary_composes(secret_form, candidate)
            for secret_form in secret_forms
        )

    for candidate in ("***", "[REDACTED]", "<redacted-secret>"):
        if safe(candidate):
            return candidate

    occupied_characters = {
        character for secret_form in secret_forms for character in secret_form
    }
    # Prefer private-use scalars so collision handling does not turn normal
    # provider prose into a control character. Scan every remaining scalar as
    # a bounded final fallback; a finite secret snapshot normally exits on the
    # first private-use code point.
    codepoint_ranges = (
        range(0xE000, 0xF900),
        range(0xF0000, 0xFFFFE),
        range(0x100000, 0x10FFFE),
        range(0x20, 0xD800),
        range(0xF900, 0xF0000),
    )
    for codepoints in codepoint_ranges:
        for codepoint in codepoints:
            candidate = chr(codepoint)
            if candidate not in occupied_characters:
                return candidate
    return ""


def _compile_exact_secret_pattern(
    values: tuple[str, ...],
) -> _ExactSecretPattern | None:
    """Compile literal and canonical JSON spellings of exact secrets.

    Disposable provider, advisory, and compaction text frequently contains a
    valid JSON serialization rather than the decoded value. Quotes,
    backslashes, and control characters therefore have different wire bytes.
    Matching the canonical JSON string spelling closes that encoding boundary
    without parsing or rewriting caller-owned executable arguments. Replacing
    a complete escaped spelling with one safe scalar also leaves valid JSON
    valid.
    """
    literal_values = {value for value in values if value}
    if not literal_values:
        return None
    literal_forms = set(literal_values)
    literal_forms.update(json.dumps(value)[1:-1] for value in literal_values)
    ordered_forms = sorted(literal_forms, key=lambda value: (-len(value), value))
    return _ExactSecretPattern(
        regex=re.compile("|".join(re.escape(value) for value in ordered_forms)),
        replacement=_choose_exact_secret_replacement(literal_forms),
        secret_forms=tuple(ordered_forms),
    )


# An explicit operation-local scope may reuse one immutable pattern snapshot.
# This is a ContextVar, not a home-keyed/global cache: the value is reset when
# the owning request/compaction returns, and nested profile scopes with a
# different context identity build their own snapshot.
_EXACT_SECRET_PATTERN_SCOPE: contextvars.ContextVar[
    tuple[tuple[str, bool], object | None, _ExactSecretPattern | None] | None
] = contextvars.ContextVar("exact_secret_pattern_scope", default=None)


def _exact_secret_pattern_context(
    home: Path,
) -> tuple[tuple[str, bool], object | None]:
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        scope = current_secret_scope()
        multiplex_active = is_multiplex_active()
    except Exception:
        scope = None
        multiplex_active = False
    return (str(home.resolve()), multiplex_active), scope


def _exact_secret_pattern_for_home(home: Path) -> _ExactSecretPattern | None:
    """Reuse the active operation snapshot, or compile a fresh one."""
    key, scope = _exact_secret_pattern_context(home)
    current = _EXACT_SECRET_PATTERN_SCOPE.get()
    if (
        current is not None
        and current[0] == key
        and current[1] is scope
    ):
        return current[2]
    return _compile_exact_secret_pattern(_collect_exact_secret_values(home))


@contextmanager
def _exact_secret_pattern_scope(home: str | os.PathLike | None = None):
    """Reuse one exact-secret snapshot only for this context-local operation."""
    resolved_home = _resolve_secret_home(home)
    key, scope = _exact_secret_pattern_context(resolved_home)
    current = _EXACT_SECRET_PATTERN_SCOPE.get()
    if (
        current is not None
        and current[0] == key
        and current[1] is scope
    ):
        yield
        return

    pattern = _compile_exact_secret_pattern(_collect_exact_secret_values(resolved_home))
    # Retain the scope object itself for the lifetime of the cached snapshot.
    # A numeric ``id(scope)`` can be reused after the old mapping is collected,
    # which could otherwise bind a new profile to a stale secret pattern.
    token = _EXACT_SECRET_PATTERN_SCOPE.set((key, scope, pattern))
    try:
        yield
    finally:
        _EXACT_SECRET_PATTERN_SCOPE.reset(token)


_JSON_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_JSON_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _scan_json_string_fragment(
    text: str,
    quote_start: int,
    source_spans: list[tuple[int, int]] | None = None,
) -> tuple[int, int | None, str, list[tuple[int, int]]]:
    """Decode one quoted fragment while retaining each character's raw span.

    Unlike ``json.loads``, this deliberately tolerates invalid escapes,
    unescaped control characters, and end-of-input before a closing quote.
    Invalid escapes are retained as their two literal characters so scanning
    can recover and decode later valid escapes in the same fragment.
    """
    decoded: list[str] = []
    raw_spans: list[tuple[int, int]] = []

    def source_span(start: int, end: int) -> tuple[int, int]:
        if source_spans is None:
            return start, end
        return source_spans[start][0], source_spans[end - 1][1]

    # ``quote_start == -1`` represents the virtual quote before byte zero. It
    # lets the same escape-aware decoder scan the prefix (or an entirely
    # unquoted/single-quoted malformed object) without changing cycle-8 quote
    # parity recovery for every later double-quote interval.
    index = quote_start + 1
    while index < len(text):
        char = text[index]
        if char == '"':
            return index + 1, index, "".join(decoded), raw_spans
        if char != "\\":
            decoded.append(char)
            raw_spans.append(source_span(index, index + 1))
            index += 1
            continue

        if index + 1 >= len(text):
            decoded.append("\\")
            raw_spans.append(source_span(index, index + 1))
            index += 1
            continue

        escape = text[index + 1]
        simple = _JSON_SIMPLE_ESCAPES.get(escape)
        if simple is not None:
            decoded.append(simple)
            raw_spans.append(source_span(index, index + 2))
            index += 2
            continue

        unicode_end = index + 6
        if (
            escape == "u"
            and unicode_end <= len(text)
            and all(char in _JSON_HEX_DIGITS for char in text[index + 2 : unicode_end])
        ):
            codepoint = int(text[index + 2 : unicode_end], 16)
            raw_end = unicode_end
            # JSON represents non-BMP characters as a UTF-16 surrogate pair.
            # Decode a valid pair to one Python character and map it to the
            # complete twelve-byte raw escape span.
            if (
                0xD800 <= codepoint <= 0xDBFF
                and unicode_end + 6 <= len(text)
                and text[unicode_end : unicode_end + 2] == "\\u"
                and all(
                    char in _JSON_HEX_DIGITS
                    for char in text[unicode_end + 2 : unicode_end + 6]
                )
            ):
                low = int(text[unicode_end + 2 : unicode_end + 6], 16)
                if 0xDC00 <= low <= 0xDFFF:
                    codepoint = (
                        0x10000
                        + ((codepoint - 0xD800) << 10)
                        + (low - 0xDC00)
                    )
                    raw_end = unicode_end + 6
            decoded.append(chr(codepoint))
            raw_spans.append(source_span(index, raw_end))
            index = raw_end
            continue

        # Preserve malformed escape bytes as separate decoded characters.
        # This prevents one bad escape from terminating recognition of a
        # later alternately escaped credential in the same quoted fragment.
        decoded.extend(("\\", escape))
        raw_spans.extend(
            (
                source_span(index, index + 1),
                source_span(index + 1, index + 2),
            )
        )
        index += 2

    return len(text), None, "".join(decoded), raw_spans


_JSON_ESCAPE_WORK_MULTIPLIER = 4
_JSON_ESCAPE_MIN_EXTRA_WORK = 4_096
_JSON_ESCAPE_MAX_EXTRA_WORK = 8 * 1024 * 1024


def _json_escape_decode_work_budget(text: str) -> int:
    """Bound total scans while always admitting the original text once.

    Proper nested JSON serialization grows geometrically, so all successively
    decoded forms normally total less than a small multiple of the wire text.
    The floor handles small inputs; the ceiling prevents a very large payload
    from authorizing unbounded additional CPU work. If a derived fragment does
    not fit, its mapped original bytes are masked wholesale instead of being
    forwarded at an uninspected nesting depth.
    """
    extra_work = min(
        max(len(text) * _JSON_ESCAPE_WORK_MULTIPLIER, _JSON_ESCAPE_MIN_EXTRA_WORK),
        _JSON_ESCAPE_MAX_EXTRA_WORK,
    )
    return len(text) + extra_work


def _append_decoded_secret_matches(
    decoded: str,
    decoded_source_spans: list[tuple[int, int]],
    pattern: _ExactSecretPattern,
    replacements: list[tuple[int, int]],
) -> None:
    """Map exact matches in decoded text back to original source spans."""
    for match in pattern.regex.finditer(decoded):
        if match.start() < match.end():
            replacements.append(
                (
                    decoded_source_spans[match.start()][0],
                    decoded_source_spans[match.end() - 1][1],
                )
            )


def _collect_json_secret_spans(
    text: str,
    pattern: _ExactSecretPattern,
    replacements: list[tuple[int, int]],
) -> None:
    """Collect matches through iterative, source-mapped escape decoding.

    The worklist removes recursion depth as an input-controlled resource. A
    task maps every decoded character back to the original disposable text;
    child tasks compose those spans again, so a match at any nesting depth can
    still replace the exact outer wire bytes. Budget exhaustion is fail-closed:
    the uninspected mapped fragment is replaced as a whole.
    """
    worklist: deque[tuple[str, list[tuple[int, int]] | None]] = deque(
        [(text, None)]
    )
    remaining_work = _json_escape_decode_work_budget(text)

    while worklist:
        current_text, source_spans = worklist.popleft()
        current_work = len(current_text)
        if current_work > remaining_work:
            if source_spans:
                replacements.append((source_spans[0][0], source_spans[-1][1]))
            elif current_text:
                replacements.append((0, len(current_text)))
            continue
        remaining_work -= current_work

        quote_start = -1
        while quote_start < len(current_text):
            _fragment_end, closing_quote, decoded, decoded_source_spans = (
                _scan_json_string_fragment(
                    current_text,
                    quote_start,
                    source_spans,
                )
            )
            _append_decoded_secret_matches(
                decoded,
                decoded_source_spans,
                pattern,
                replacements,
            )

            fragment_end = (
                closing_quote if closing_quote is not None else len(current_text)
            )
            encoded_fragment = current_text[quote_start + 1 : fragment_end]
            if "\\" in decoded and decoded != encoded_fragment:
                worklist.append((decoded, decoded_source_spans))

            # Treat the closing quote as the next possible opening quote.
            # Invalid JSON can insert or omit an unescaped quote, changing the
            # parity of every later quote. Scanning every interval between
            # consecutive escape-aware quotes resynchronizes immediately.
            if closing_quote is None:
                break
            quote_start = closing_quote


def _replace_exact_secret_spans(
    text: str,
    pattern: _ExactSecretPattern,
    replacements: list[tuple[int, int]],
    replacement_overrides: dict[tuple[int, int], str] | None = None,
) -> str:
    """Apply deduplicated source spans using collision-safe markers."""
    if not replacements:
        return text

    # Recovery windows and nested decoding can discover the same match more
    # than once. Normalize so each source byte is replaced exactly once and a
    # shorter duplicate cannot split or suppress a complete encoded span.
    merged_replacements: list[tuple[int, int]] = []
    for start, end in sorted(set(replacements)):
        if merged_replacements and start < merged_replacements[-1][1]:
            previous_start, previous_end = merged_replacements[-1]
            merged_replacements[-1] = (previous_start, max(previous_end, end))
        else:
            merged_replacements.append((start, end))

    pieces: list[str] = []
    cursor = 0
    for start, end in merged_replacements:
        pieces.extend(
            (
                text[cursor:start],
                (replacement_overrides or {}).get((start, end), pattern.replacement),
            )
        )
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _is_json_structural_key(value: str) -> bool:
    """Keep only low-entropy JSON literal spellings usable as protocol keys."""
    # Quoted numeric keys are strings, not typed JSON numbers. A long numeric
    # credential must therefore be masked just like any other source value.
    return value in {"true", "false", "null"}


def _append_json_secret_matches(
    decoded: str,
    decoded_source_spans: list[tuple[int, int]],
    pattern: _ExactSecretPattern,
    replacements: list[tuple[int, int]],
    *,
    object_key: bool,
    key_matches: list[tuple[tuple[int, int], str]],
) -> None:
    """Collect quoted-token matches, with collision-safe JSON key markers."""
    if object_key and _is_json_structural_key(decoded):
        return
    for match in pattern.regex.finditer(decoded):
        if match.start() >= match.end():
            continue
        span = (
            decoded_source_spans[match.start()][0],
            decoded_source_spans[match.end() - 1][1],
        )
        replacements.append(span)
        if object_key:
            key_matches.append((span, match.group(0)))


def _redact_json_string_fragments(
    text: str, pattern: _ExactSecretPattern
) -> str:
    r"""Decode quoted JSON fragments and mask decoded matches in raw spans.

    JSON permits many byte spellings for the same decoded value (for example
    ``\/`` and ``\uXXXX``). Literal matching cannot enumerate that space, and
    strict token parsing misses invalid or EOF-terminated fragments. The
    tolerant scanner retains a decoded-character-to-raw-span map, recovers
    after invalid escapes, and replaces only matched raw bytes. Caller-owned
    history is never passed to this helper; its inputs are disposable copies.
    """
    if "\\" not in text:
        return text

    replacements: list[tuple[int, int]] = []
    _collect_json_secret_spans(text, pattern, replacements)
    return _replace_exact_secret_spans(text, pattern, replacements)


def _redact_valid_json_string_values(
    text: str,
    pattern: _ExactSecretPattern,
) -> str | None:
    """Mask quoted string-token bytes when *text* is one valid JSON document.

    ``None`` means parsing failed and the caller must use the aggressive
    malformed/prose path. For valid JSON, punctuation plus typed scalars
    (numbers, booleans, and null) are structural and remain byte-exact. Both
    object keys and string values are quoted tokens and can carry an
    authoritative credential, so matching either is source-mapped and masked
    without rewriting formatting or caller-owned input.
    """
    def reject_nonstandard_constant(value: str):
        raise ValueError(f"invalid JSON constant: {value}")

    try:
        json.loads(text, parse_constant=reject_nonstandard_constant)
    except (TypeError, ValueError):
        return None

    replacements: list[tuple[int, int]] = []
    replacement_overrides: dict[tuple[int, int], str] = {}
    key_matches: list[tuple[tuple[int, int], str]] = []
    forbidden_markers: set[str] = set()
    worklist: deque[tuple[str, list[tuple[int, int]]]] = deque()
    remaining_work = _json_escape_decode_work_budget(text) - len(text)

    cursor = 0
    while cursor < len(text):
        quote_start = text.find('"', cursor)
        if quote_start < 0:
            break
        fragment_end, closing_quote, decoded, decoded_source_spans = (
            _scan_json_string_fragment(text, quote_start)
        )
        # ``json.loads`` already proved the document valid, so every opening
        # quote has a closing quote and every token after it is unambiguous.
        if closing_quote is None:  # pragma: no cover - defensive consistency
            return None

        after_token = fragment_end
        while after_token < len(text) and text[after_token].isspace():
            after_token += 1
        object_key = after_token < len(text) and text[after_token] == ":"
        if object_key:
            forbidden_markers.add(decoded)
        _append_json_secret_matches(
            decoded,
            decoded_source_spans,
            pattern,
            replacements,
            object_key=object_key,
            key_matches=key_matches,
        )
        encoded_fragment = text[quote_start + 1 : closing_quote]
        if "\\" in decoded and decoded != encoded_fragment:
            worklist.append((decoded, decoded_source_spans))
        cursor = fragment_end

    # Once a quoted token is decoded, every derived byte belongs to that token,
    # so the tolerant iterative scanner may inspect it globally. This retains
    # deep/double-serialized JSON coverage without matching punctuation or
    # typed scalar tokens from the valid outer document.
    while worklist:
        current_text, source_spans = worklist.popleft()
        current_work = len(current_text)
        if current_work > remaining_work:
            if source_spans:
                replacements.append((source_spans[0][0], source_spans[-1][1]))
            continue
        remaining_work -= current_work

        quote_start = -1
        while quote_start < len(current_text):
            _fragment_end, closing_quote, decoded, decoded_source_spans = (
                _scan_json_string_fragment(
                    current_text,
                    quote_start,
                    source_spans,
                )
            )
            fragment_end = (
                closing_quote if closing_quote is not None else len(current_text)
            )
            after_token = fragment_end
            while after_token < len(current_text) and current_text[after_token].isspace():
                after_token += 1
            object_key = (
                after_token < len(current_text)
                and current_text[after_token] == ":"
            )
            if object_key:
                forbidden_markers.add(decoded)
            _append_json_secret_matches(
                decoded,
                decoded_source_spans,
                pattern,
                replacements,
                object_key=object_key,
                key_matches=key_matches,
            )
            encoded_fragment = current_text[quote_start + 1 : fragment_end]
            if "\\" in decoded and decoded != encoded_fragment:
                worklist.append((decoded, decoded_source_spans))
            if closing_quote is None:
                break
            quote_start = closing_quote

    key_markers: dict[str, str] = {}
    used_markers: set[str] = set()
    for span, token in key_matches:
        marker = key_markers.get(token)
        if marker is None:
            occupied = forbidden_markers | used_markers
            marker = _choose_exact_secret_replacement(
                set(pattern.secret_forms) | occupied
            )
            key_markers[token] = marker
            used_markers.add(marker)
        replacement_overrides[span] = marker

    return _replace_exact_secret_spans(
        text,
        pattern,
        replacements,
        replacement_overrides,
    )


def _redact_exact_with_pattern(
    text: str, pattern: _ExactSecretPattern | None
) -> str:
    if not text or pattern is None:
        return text
    valid_json_redacted = _redact_valid_json_string_values(text, pattern)
    if valid_json_redacted is not None:
        return valid_json_redacted
    # Preserve the existing raw and canonical literal behavior for arbitrary
    # prose and values outside quoted JSON fragments.
    # Run it first so a longer exact value (such as an entire inline service
    # account object) is not partially rewritten by a shorter nested secret
    # before the complete value can match.
    def replace_unshadowed(match: re.Match) -> str:
        # A match can begin at the escaped byte of a JSON spelling: nested
        # ``\\u00e9`` contains canonical ``\u00e9``, while the literal secret
        # ``/`` can appear as the valid simple escape ``\/``. Replacing only
        # that suffix leaves an invalid ``\***`` escape. Defer every match
        # immediately preceded by a backslash to the source-mapped scanner,
        # which decodes and replaces the complete outer span. It also retains
        # malformed escape handling because invalid pairs are source-mapped as
        # their separate literal bytes.
        if (
            match.start() > 0
            and text[match.start() - 1] == "\\"
        ):
            return match.group(0)
        return pattern.replacement

    literal_redacted = pattern.regex.sub(replace_unshadowed, text)
    if "\\" not in literal_redacted:
        return literal_redacted
    fragment_redacted = _redact_json_string_fragments(literal_redacted, pattern)

    def replace_remaining_unshadowed(match: re.Match) -> str:
        if (
            match.start() > 0
            and fragment_redacted[match.start() - 1] == "\\"
        ):
            return match.group(0)
        return pattern.replacement

    return pattern.regex.sub(replace_remaining_unshadowed, fragment_redacted)


def redact_known_secret_values(
    text: str,
    home: str | os.PathLike | None = None,
    *,
    force: bool = False,
) -> str:
    """Mask every occurrence of active-profile authoritative secret values."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if not text or not (force or _REDACT_ENABLED):
        return text
    resolved_home = _resolve_secret_home(home)
    pattern = _exact_secret_pattern_for_home(resolved_home)
    return _redact_exact_with_pattern(text, pattern)


_PROVIDER_STRUCTURAL_STRING_FIELDS = frozenset(
    {
        "type",
        "role",
        "id",
        "name",
        "tool_name",
        "tool_call_id",
        "tool_use_id",
        "call_id",
        "toolUseId",
        "response_item_id",
        "detail",
        "media_type",
        "mime_type",
        "format",
        "ttl",
        "phase",
        "status",
        "finish_reason",
        "file_id",
        "file_name",
        "filename",
        # Issuer-bound replay identifiers/payloads are never model-visible
        # plaintext and must remain byte-exact across provider retries.
        "signature",
        "encrypted_content",
    }
)
_PROVIDER_OPAQUE_METADATA_FIELDS = frozenset(
    {
        # Anthropic/OpenAI cache directives are provider control metadata, not
        # model-visible prose. Rewriting ``ephemeral`` or ``1h`` makes an
        # otherwise valid request fail schema validation.
        "cache_control",
    }
)
_PROVIDER_BINARY_DATA_TYPES = frozenset(
    {
        "base64",
        "image",
        "input_image",
        "audio",
        "input_audio",
    }
)
_PROVIDER_BINARY_CONTAINER_FIELDS = frozenset({"audio", "input_audio"})
_PROVIDER_EXECUTABLE_INPUT_TYPES = frozenset({"tool_use", "function_call"})
_PROVIDER_EXECUTABLE_CONTAINER_FIELDS = frozenset({"toolUse", "function"})
_DATA_URI_BASE64_RE = re.compile(
    r"\A(data:[^,]*;base64,)([A-Za-z0-9+/]*={0,2})\Z",
    re.IGNORECASE,
)


def _redact_provider_url(value: str, pattern: _ExactSecretPattern | None) -> str:
    """Mask ordinary URLs while preserving validated opaque data URIs.

    A data URI's payload is binary media, not model-visible prose. Replacing
    an arbitrary exact value in that payload can make the URI invalid (for
    example, masking ``A`` in ``data:image/png;base64,AAAA``). Keep the
    base64 octets and grammar metadata byte-exact; ordinary non-data URLs
    remain subject to exact-value redaction.
    """
    if pattern is None:
        return value
    match = _DATA_URI_BASE64_RE.match(value)
    if match is None:
        return _redact_exact_with_pattern(value, pattern)
    # Metadata controls such as ``image/png`` and ``base64`` are part of the
    # URI grammar. Preserve the complete validated URI so a short credential
    # collision cannot turn it into an invalid provider payload. Opaque media
    # is intentionally outside display-text redaction; ordinary non-data URLs
    # still take the exact-value path below.
    return value


def _redact_provider_tree(
    value: Any,
    pattern: _ExactSecretPattern | None,
    *,
    field: str | None = None,
) -> Any:
    """Redact model-visible leaves without corrupting provider controls."""
    if isinstance(value, str):
        if field in {"url", "uri", "image_url"}:
            return _redact_provider_url(value, pattern)
        return _redact_exact_with_pattern(value, pattern)
    if isinstance(value, list):
        return [_redact_provider_tree(item, pattern, field=field) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _redact_provider_tree(item, pattern, field=field) for item in value
        )
    if isinstance(value, dict):
        container_type = value.get("type")
        return {
            key: (
                item
                if (
                    key in _PROVIDER_OPAQUE_METADATA_FIELDS
                    or (
                        isinstance(item, str)
                        and key in _PROVIDER_STRUCTURAL_STRING_FIELDS
                    )
                    or (
                        key == "data"
                        and isinstance(item, str)
                        and (
                            container_type in _PROVIDER_BINARY_DATA_TYPES
                            or container_type == "redacted_thinking"
                            or field in _PROVIDER_BINARY_CONTAINER_FIELDS
                        )
                    )
                    or (
                        key in {"input", "arguments"}
                        and (
                            container_type in _PROVIDER_EXECUTABLE_INPUT_TYPES
                            or field in _PROVIDER_EXECUTABLE_CONTAINER_FIELDS
                        )
                    )
                )
                else _redact_provider_tree(item, pattern, field=key)
            )
            for key, item in value.items()
        }
    return value


def _redact_anthropic_ordered_blocks(
    blocks: Any,
    pattern: _ExactSecretPattern | None,
) -> Any:
    """Mask unsigned text unless a signature binds the ordered sequence."""
    if not isinstance(blocks, list):
        return blocks
    has_signed_thinking = any(
        isinstance(block, dict)
        and (
            (block.get("type") == "thinking" and bool(block.get("signature")))
            or (
                block.get("type") == "redacted_thinking"
                and bool(block.get("data"))
            )
        )
        for block in blocks
    )
    if has_signed_thinking:
        # Anthropic signatures cover the ordered assistant content that
        # surrounds them. Rewriting even an intervening text block while
        # retaining a later signature makes the complete replay invalid.
        return blocks
    return [
        {
            **block,
            "text": _redact_exact_with_pattern(block.get("text", ""), pattern),
        }
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        else block
        for block in blocks
    ]


def _redact_codex_message_items(
    items: Any,
    pattern: _ExactSecretPattern | None,
) -> Any:
    """Mask replayed assistant text without modifying item identity fields."""
    if not isinstance(items, list):
        return items
    return [
        {
            **item,
            "content": _redact_provider_tree(item.get("content"), pattern),
        }
        if isinstance(item, dict) and "content" in item
        else item
        for item in items
    ]


def _redact_provider_messages_with_pattern(
    messages: list,
    pattern: _ExactSecretPattern,
) -> list:
    """Redact one provider message/input list using a bound snapshot."""
    import copy

    redacted_messages = copy.deepcopy(messages)
    provider_fields = (
        "content",
        "reasoning",
        "reasoning_content",
    )
    for message in redacted_messages:
        if not isinstance(message, dict):
            continue
        for field in provider_fields:
            if field in message:
                if field == "content" and isinstance(message[field], list) and any(
                    isinstance(block, dict)
                    and (
                        (
                            block.get("type") == "thinking"
                            and bool(block.get("signature"))
                        )
                        or (
                            block.get("type") == "redacted_thinking"
                            and bool(block.get("data"))
                        )
                    )
                    for block in message[field]
                ):
                    # Anthropic signs the ordered content sequence, not merely
                    # the individual thinking block. Preserve the whole replay
                    # unit exactly as the earlier adapter-specific gate does.
                    continue
                message[field] = _redact_provider_tree(message[field], pattern)
        # Codex Responses converts tool-result messages into native
        # ``function_call_output.output`` items before its final preflight.
        # Mask visible output while preserving replayed function-call
        # ``arguments`` and signed/sealed reasoning fields byte-for-byte.
        if message.get("type") == "function_call_output" and "output" in message:
            message["output"] = _redact_provider_tree(message["output"], pattern)
        # Ordered Anthropic blocks bypass ``content`` in the adapter. Text-only
        # unsigned lists are maskable; a list containing signed thinking is an
        # opaque sequence. Executable tool_use inputs always remain byte-exact.
        if "anthropic_content_blocks" in message:
            message["anthropic_content_blocks"] = _redact_anthropic_ordered_blocks(
                message["anthropic_content_blocks"], pattern
            )
        # OpenRouter requires the complete reasoning_details sequence to be
        # replayed unmodified. Treat every current and future detail shape as
        # opaque; selectively rewriting summary/text/unknown fields can
        # invalidate signatures or the sequence contract.
        # Codex reasoning items contain issuer-sealed encrypted replay blobs
        # and must stay exact. Assistant message items are also replayed with
        # stable ids/phases, so sanitize only their visible content payload.
        if "codex_message_items" in message:
            message["codex_message_items"] = _redact_codex_message_items(
                message["codex_message_items"], pattern
            )
    return redacted_messages


def redact_provider_message_values(
    messages: list,
    home: str | os.PathLike | None = None,
    *,
    force: bool = False,
) -> list:
    """Return a provider-bound copy with exact secret values removed.

    The source history is never mutated. This is required by #43083: local
    replay and tool execution retain exact arguments, while the outbound copy
    masks message text and nested non-text metadata such as ``image_url.url``.

    Tool-call arguments deliberately remain exact on the outbound copy. Human's
    review of PR #42846 established that replacing replayed executable values
    makes models copy the placeholder on later turns (#43083). Protecting that
    field requires authenticated vault/token references, not display masking.
    """
    if not messages or not (force or _REDACT_ENABLED):
        return messages
    pattern = _exact_secret_pattern_for_home(_resolve_secret_home(home))
    if pattern is None:
        return messages

    return _redact_provider_messages_with_pattern(messages, pattern)


def redact_provider_api_kwargs(
    api_kwargs: dict,
    home: str | os.PathLike | None = None,
    *,
    force: bool = False,
) -> dict:
    """Rebind provider-native visible payloads after final transformations.

    Provider adapters may normalize text after the generic message gate (for
    example ASCII fallback or Codex Harmony-token neutralization). Inspect the
    final ``messages``/``input``/``instructions`` bytes without rewriting
    executable arguments, identity fields, or caller-owned request objects.
    """
    if not isinstance(api_kwargs, dict) or not (force or _REDACT_ENABLED):
        return api_kwargs
    pattern = _exact_secret_pattern_for_home(_resolve_secret_home(home))
    if pattern is None:
        return api_kwargs

    redacted_kwargs = dict(api_kwargs)
    for field in ("messages", "input"):
        payload = api_kwargs.get(field)
        if isinstance(payload, list):
            redacted_kwargs[field] = _redact_provider_messages_with_pattern(
                payload,
                pattern,
            )
    instructions = api_kwargs.get("instructions")
    if isinstance(instructions, str):
        redacted_kwargs["instructions"] = _redact_exact_with_pattern(
            instructions,
            pattern,
        )
    system = api_kwargs.get("system")
    if isinstance(system, (str, list, tuple, dict)):
        import copy

        redacted_kwargs["system"] = _redact_provider_tree(
            copy.deepcopy(system),
            pattern,
        )
    return redacted_kwargs


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
        # Control/zero-width chars (\\n, \\r, ESC, U+200B, …) split a token
        # body so _PREFIX_RE cannot match across them — a secret smuggled as
        # ``sk-abc\\x1bdef…`` leaks verbatim (issue #77484). Mask such runs by
        # first matching on a control-stripped copy, then re-masking the
        # corresponding span in the original (the stripped copy and the
        # original are aligned 1:1 for non-control chars).
        text = _mask_control_split_tokens(text, _prefix_sub)
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
                # Keyword must sit at a word boundary within the key —
                # ``author=Smith`` / ``press.secretary=…`` are prose, not
                # credentials (ported from nearai/ironclaw#6129). All-caps
                # keys (the _ENV_ASSIGN_RE shape) short-circuit to legacy
                # embedded matching inside the helper.
                if not _key_has_secret_keyword(name):
                    return m.group(0)
                return f"{name}={quote}{_mask_token(value)}{quote}"
            text = _ENV_ASSIGN_RE.sub(_redact_env, text)
            # Lowercase env names (``openai_key=…``). Skip URLs — the query
            # string may contain ``token=``/``key=`` params that are
            # intentionally passed through (see note near the bottom of this
            # function; _redact_strict_url_credentials handles the opt-in
            # case). The uppercase regex above is all-caps-only, so it never
            # matches URL params; the lowercase one would (issue #77484).
            if "://" not in text:
                text = _ENV_ASSIGN_LOWER_RE.sub(_redact_env, text)
            # Lowercase/dotted config keys (issue #16413). Skip URLs entirely —
            # web-URL query params are intentionally passed through (see note
            # near the bottom of this function); _DB_CONNSTR_RE still guards
            # connection-string passwords.
            #
            # Extra gate: every _CFG_*_RE match requires a secret keyword in
            # the key, so a text without any secret keyword cannot match —
            # skipping is exact. This matters because _CFG_DOTTED_RE
            # backtracks quadratically on long unbroken [A-Za-z0-9_.\-] runs
            # (e.g. base64/hex blobs in compaction payloads); the linear
            # keyword scan prevents that pathological path on secret-free
            # text.
            if "://" not in text and _CFG_SECRET_WORD_RE.search(text):
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
                # Keyword must sit at a word boundary within the key —
                # ``Secretary: J.Smith`` / ``tokenizer: cl100k_base`` are
                # document text, not credentials (nearai/ironclaw#6129).
                if not _key_has_secret_keyword(key):
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

# Commands that read file contents to stdout. When the target is a ``.env``
# file, the output is a credential dump — the same as ``printenv`` — so the
# ENV-assignment pass must run (code_file=False). Per AGENTS.md, ``.env`` is
# for secrets only; behavioral settings belong in config.yaml, so running
# the generic ENV redactor on ``.env`` content is the correct behavior.
_FILE_READ_COMMANDS = frozenset({
    "cat", "head", "tail", "type", "bat", "less", "more", "nl",
    "zcat", "tac", "view", "batcat",
})

# Basenames that are treated as ``.env`` files for redaction purposes are
# imported at module top as ``_ENV_FILE_BASENAMES`` (see the
# ``agent.file_safety`` import).


def _command_reads_env_file(command: str | None) -> bool:
    """Return True if ``command`` reads a ``.env`` file to stdout.

    Detects file-read commands (``cat``, ``head``, ``tail``, etc.) where any
    argument's basename is a ``.env``-style file. Template files
    (``.env.example``, ``.env.sample``, ...) are not in the basename list and
    therefore never match. Handles pipelines and command sequences.

    Conservative defense-in-depth, not a boundary — indirect reads
    (``sudo cat .env``, ``/bin/cat .env``, ``$(cat .env)``, redirection,
    ``sed``/``awk``/``xxd`` readers) are not detected, matching the
    precedent of ``is_env_dump_command`` below.
    """
    if not command:
        return False
    segments = re.split(r"[|;&]+", command)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # Use plain split() instead of shlex.split — shlex treats backslashes
        # as escape chars, which mangles Windows paths (``C:\Users\...\.env``).
        # We only need the command name and filename, so shell quoting is not
        # a concern here.
        tokens = seg.split()
        if not tokens or tokens[0] not in _FILE_READ_COMMANDS:
            continue
        # Check all arguments (skip flags like -n, -A, etc.)
        for arg in tokens[1:]:
            if arg.startswith("-"):
                continue
            # Strip shell quotes that plain split() leaves attached
            # (``cat ".env"`` / ``cat '.env'``), then any leading path to
            # get the basename. Handle both / and \.
            arg = arg.strip("\"'")
            basename = arg.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if basename.lower() in _ENV_FILE_BASENAMES:
                return True
    return False


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
    ``command`` is an environment dump or reads a ``.env`` file:

    - env-dump command (``env``/``printenv``/``set``/``export``/``declare``)
      → ``code_file=False`` so the ENV-assignment pass masks opaque tokens.
    - file-read command targeting a ``.env`` file (``cat .env``,
      ``head .env.local``, etc.) → ``code_file=False`` for the same reason.
      Per AGENTS.md, ``.env`` files contain only secrets, so the generic
      ENV pass is the right one (keys whose names carry no secret keyword
      can still slip through it — same limit as the env-dump path).
    - anything else (or unknown command) → ``code_file=True`` to avoid
      false positives on source/config dumps.

    ``force=True`` bypasses the global ``security.redact_secrets`` preference
    for safety boundaries that must never emit raw credentials.
    """
    if not output:
        return output
    cmd = command or ""
    code_file = not (is_env_dump_command(cmd) or _command_reads_env_file(cmd))
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


def _has_top_level_alternation(pattern: str) -> bool:
    """True if ``pattern`` contains a ``|`` outside any group or class.

    A top-level alternation defeats the literal-prefix guarantee:
    ``_extract_literal_prefix`` stops at ``|``, so for ``ab|.*`` it
    returns ``ab`` even though the ``.*`` branch is not bound by that
    prefix and matches anything. Grouped alternation after the prefix
    (``ab(?:x|y)``) keeps the guarantee and stays allowed.
    """
    depth = 0
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "[":
            i += 1
            if i < len(pattern) and pattern[i] == "]":
                i += 1
            while i < len(pattern) and pattern[i] != "]":
                if pattern[i] == "\\":
                    i += 1
                i += 1
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "|" and depth == 0:
            return True
        i += 1
    return False


def _has_nested_unbounded_repeat(pattern: str) -> bool:
    """True if an unbounded quantifier applies to a group containing one.

    ``(a+)+``, ``(?:x*)*``, ``(a{2,})+`` — the canonical catastrophic-
    backtracking (ReDoS) shape. Registered patterns run against every log
    line, tool output, and transcript chunk, so a pathological pattern from
    a buggy plugin would stall the host process, not just the plugin.

    Detects structural nesting only; ambiguity between overlapping
    alternation branches (``(a|aa)+``) is not statically detected and
    remains the plugin author's responsibility.
    """

    def _unbounded_quantifier_follows(j: int) -> bool:
        # Is pattern[j:] an unbounded quantifier (* + {m,}) for the atom
        # that just ended at j?
        if j >= len(pattern):
            return False
        if pattern[j] in "*+":
            return True
        if pattern[j] == "{":
            k = pattern.find("}", j)
            body = pattern[j + 1:k] if k != -1 else ""
            # {m,} is open-ended; {m} and {m,n} are bounded.
            return body[:-1].isdigit() and body.endswith(",")
        return False

    # Stack of flags: does the group at this depth contain an unbounded
    # repeat? Index 0 is the top level.
    contains_unbounded = [False]
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "[":
            i += 1
            if i < len(pattern) and pattern[i] == "]":
                i += 1
            while i < len(pattern) and pattern[i] != "]":
                if pattern[i] == "\\":
                    i += 1
                i += 1
        elif ch == "(":
            contains_unbounded.append(False)
        elif ch == ")":
            inner = contains_unbounded.pop() if len(contains_unbounded) > 1 else False
            if inner and _unbounded_quantifier_follows(i + 1):
                return True
            contains_unbounded[-1] = contains_unbounded[-1] or inner
        elif _unbounded_quantifier_follows(i):
            contains_unbounded[-1] = True
            if ch == "{":
                i = pattern.find("}", i)  # skip the {m,} body
        i += 1
    return False


_PREFIX_SUBSTRINGS = tuple(
    _extract_literal_prefix(p) for p in _PREFIX_PATTERNS
)


def _has_known_prefix_substring(text: str) -> bool:
    """Return True if ``text`` contains any known credential prefix substring.

    Used as a cheap pre-check before invoking the expensive ``_PREFIX_RE``.
    """
    return any(p in text for p in _PREFIX_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Plugin-registered redaction patterns
# ---------------------------------------------------------------------------
#
# Every new vendor token format has historically required a core PR appending
# to ``_PREFIX_PATTERNS`` above (fw_, retaindb_, hsk-, mem0_, brv_, ...).
# This registry lets plugins add their provider's format instead. It is
# ADDITIVE-ONLY by design: a plugin can extend what gets masked but has no
# API to remove or weaken a built-in pattern, so a plugin can only ever
# over-redact, never expose. The operator's global opt-out
# (``security.redact_secrets: false`` / HERMES_REDACT_SECRETS) applies to
# plugin patterns exactly as it does to built-ins.

# Keyed by registration source (e.g. "plugin:my-plugin") so the plugin
# lifecycle/ownership-ledger work (#64229) has a clean seam to drop ONE
# plugin's patterns on unload. There is deliberately no public removal
# API — additive-only stands; unload is a host-owned lifecycle concern.
_PLUGIN_PREFIX_PATTERNS: dict = {}
_registry_lock = threading.Lock()


def _plugin_patterns() -> list:
    """All plugin-registered patterns in registration order."""
    return [p for patterns in _PLUGIN_PREFIX_PATTERNS.values() for p in patterns]


def _rebuild_prefix_matcher() -> None:
    """Recompile the prefix alternation and pre-screen substrings.

    ``redact_sensitive_text`` and ``_mask_token_nonreusable`` look these
    globals up at call time, so swapping the module attributes (atomic
    under the GIL) propagates immediately to every caller.
    """
    global _PREFIX_RE, _PREFIX_SUBSTRINGS
    combined = _PREFIX_PATTERNS + _plugin_patterns()
    _PREFIX_RE = re.compile(
        r"(?<![A-Za-z0-9_-])(" + "|".join(combined) + r")(?![A-Za-z0-9_-])"
    )
    _PREFIX_SUBSTRINGS = tuple(_extract_literal_prefix(p) for p in combined)


def register_redaction_patterns(patterns, source: str = "plugin") -> int:
    """Additively register credential-token regexes with the redaction engine.

    Each accepted pattern joins the vendor-prefix alternation used by
    ``redact_sensitive_text`` (same masking, same head/tail rules, same
    non-reusable sentinel on ``file_read``) — everywhere built-in patterns
    apply: logs, terminal output, transport errors, transcripts.

    Per-pattern validation (invalid entries are warned and skipped, never
    raised — a broken plugin must not break startup):

    * must be a non-empty string that compiles as a regex;
    * must not contain a top-level alternation (``ab|.*`` would escape
      the literal-prefix guarantee below through its unprefixed branch;
      grouped alternation after the prefix, ``ab(?:x|y)``, is allowed);
    * must not nest unbounded quantifiers (``(a+)+``-style patterns can
      backtrack catastrophically, and registered patterns run against
      every log line and tool output — see
      ``_has_nested_unbounded_repeat``);
    * must start with at least 2 literal characters (the pre-screen
      substring gate in ``_has_known_prefix_substring`` needs a literal
      anchor; it also structurally rules out redact-everything patterns
      like ``.*``);
    * duplicates of built-in or already-registered patterns are skipped.

    Args:
        patterns: iterable of regex strings (e.g. ``[r"nvapi-[A-Za-z0-9_-]{20,}"]``).
        source: attribution label for log lines (e.g. ``"plugin:my-plugin"``).

    Returns:
        The number of patterns actually accepted.
    """
    accepted = []
    for pattern in patterns or []:
        if not isinstance(pattern, str) or not pattern.strip():
            logger.warning("%s: skipping empty/non-string redaction pattern", source)
            continue
        pattern = pattern.strip()
        try:
            re.compile(pattern)
        except re.error as exc:
            logger.warning(
                "%s: skipping invalid redaction pattern %r (%s)",
                source, pattern, exc,
            )
            continue
        if _has_top_level_alternation(pattern):
            logger.warning(
                "%s: skipping redaction pattern %r — top-level alternation "
                "escapes the literal-prefix guarantee (in 'ab|.*' the "
                "prefix binds only the first branch); wrap alternation in "
                "a group after the prefix, e.g. 'ab(?:x|y)'",
                source, pattern,
            )
            continue
        if _has_nested_unbounded_repeat(pattern):
            logger.warning(
                "%s: skipping redaction pattern %r — nested unbounded "
                "quantifiers (e.g. '(a+)+') can backtrack catastrophically, "
                "and registered patterns run on every log line and tool "
                "output",
                source, pattern,
            )
            continue
        if len(_extract_literal_prefix(pattern)) < 2:
            logger.warning(
                "%s: skipping redaction pattern %r — must start with at "
                "least 2 literal characters (needed for the pre-screen "
                "substring gate)",
                source, pattern,
            )
            continue
        if pattern in _PREFIX_PATTERNS or pattern in _plugin_patterns() or pattern in accepted:
            logger.debug("%s: redaction pattern %r already registered", source, pattern)
            continue
        accepted.append(pattern)

    if accepted:
        with _registry_lock:
            _PLUGIN_PREFIX_PATTERNS.setdefault(source, []).extend(accepted)
            _rebuild_prefix_matcher()
        logger.info(
            "%s: registered %d redaction pattern(s)", source, len(accepted)
        )
    return len(accepted)


def _reset_plugin_redaction_patterns() -> None:
    """Drop all plugin-registered patterns (tests/teardown only)."""
    with _registry_lock:
        _PLUGIN_PREFIX_PATTERNS.clear()
        _rebuild_prefix_matcher()


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
