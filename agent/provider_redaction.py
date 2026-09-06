"""Exact-secret masking for disposable provider payloads and derived text.

Replay history, executable arguments and signed or sealed provider fields remain
byte-exact; provider-visible copies use the active profile's secret snapshot.
"""

import contextvars
import json
import logging
import os
import re
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent import redact as _redact_settings

logger = logging.getLogger(__name__)

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
    from hermes_constants import get_hermes_home

    return get_hermes_home()


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
    if not text or not (force or _redact_settings._REDACT_ENABLED):
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
        "redactedContent",
        "redactedContentBase64",
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



def _bedrock_signed_prefix_end(messages: list) -> int:
    """Bedrock reasoning signatures bind every preceding conversation message."""
    end = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        for field in ("content", "bedrock_content_blocks"):
            blocks = message.get(field)
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                reasoning = block.get("reasoningContent") if isinstance(block, dict) else None
                text = reasoning.get("reasoningText") if isinstance(reasoning, dict) else None
                if isinstance(text, dict) and text.get("signature"):
                    end = index + 1
    return end


def _reject_signed_bedrock_change() -> None:
    raise ValueError(
        "Secret redaction would alter signed Bedrock history; start a new conversation "
        "without that signed replay before retrying."
    )

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
        if "bedrock_content_blocks" in message:
            message["bedrock_content_blocks"] = _redact_provider_tree(
                message["bedrock_content_blocks"], pattern
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
    signed_end = _bedrock_signed_prefix_end(messages)
    if signed_end and redacted_messages[:signed_end] != messages[:signed_end]:
        # A signature-shaped field is not proof that arbitrary preceding text
        # was already sent. Reject a collision instead of exempting the prefix
        # or forwarding an invalid signature. Unchanged replay stays exact.
        _reject_signed_bedrock_change()
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
    if not messages or not (force or _redact_settings._REDACT_ENABLED):
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
    if not isinstance(api_kwargs, dict) or not (force or _redact_settings._REDACT_ENABLED):
        return api_kwargs
    pattern = _exact_secret_pattern_for_home(_resolve_secret_home(home))
    if pattern is None:
        return api_kwargs

    redacted_kwargs = _redact_api_payload_fields(api_kwargs, pattern)
    extra_body = api_kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        # The SDK merges these fields after its typed request transformation.
        # Inspect the overriding payload as well as the top-level request.
        redacted_kwargs["extra_body"] = _redact_api_payload_fields(extra_body, pattern)
    return redacted_kwargs


def _redact_api_payload_fields(api_kwargs: dict, pattern: _ExactSecretPattern) -> dict:
    redacted_kwargs = dict(api_kwargs)
    for field in ("messages", "input"):
        payload = api_kwargs.get(field)
        if isinstance(payload, list):
            redacted_kwargs[field] = _redact_provider_messages_with_pattern(
                payload,
                pattern,
            )
        elif field == "input" and isinstance(payload, str):
            redacted_kwargs[field] = _redact_exact_with_pattern(payload, pattern)
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
    messages = api_kwargs.get("messages")
    if (
        isinstance(messages, list)
        and _bedrock_signed_prefix_end(messages)
        and redacted_kwargs.get("system") != api_kwargs.get("system")
    ):
        _reject_signed_bedrock_change()
    return redacted_kwargs
