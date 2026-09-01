"""Source-bound policy checks for outbound LLM requests.

The firewall is deliberately transport-agnostic.  Callers hand it the final
logical request and resolved route immediately before invoking a provider.
It returns an immutable allow decision, or raises :class:`EgressBlocked` with
an immutable block decision.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from hmac import compare_digest
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from agent.file_safety import get_read_block_error
from agent.cross_process_file_lock import (
    exclusive_file_lock,
    secure_file_descriptor_permissions,
)
from agent.redact import redact_sensitive_text


class DestinationClass(StrEnum):
    """Trust class for an LLM destination."""

    LOCAL_PROCESS = "local_process"
    LOOPBACK = "loopback"
    REMOTE = "remote"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SourceGrant:
    """Immutable authorization for one exact, already-read file slice."""

    canonical_path: Path
    display_path: str
    line_start: int
    line_end: int
    content_sha256: str
    byte_count: int
    session_id: str
    turn_id: str
    request_id: str
    policy_digest: str


@dataclass(frozen=True, slots=True)
class LiteralSegment:
    """Application-owned literal text for a typed outbound request."""

    text: str


@dataclass(frozen=True, slots=True)
class SanitizedSegment:
    """Non-source text that must still pass final secret and encoding scans."""

    text: str


@dataclass(frozen=True, slots=True)
class GeneratedContextSegment:
    """Hermes-generated remote context after unsafe-text redaction."""

    text: str


@dataclass(frozen=True, slots=True)
class CodexReasoningReplaySegment:
    """Opaque encrypted reasoning state replayed only to the Codex endpoint."""

    text: str


@dataclass(frozen=True, slots=True)
class GeneratedContextKey:
    """Application-owned JSON key for generated provider tool schemas."""

    text: str


@dataclass(frozen=True, slots=True)
class UntrustedProvenanceSegment:
    """Content-free marker for tool bytes with no trusted origin proof."""

    content_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedToolSyntaxSegment:
    """Strictly parsed syntax emitted by a protected local tool result."""

    text: str
    syntax_kind: str


@dataclass(frozen=True, slots=True)
class SourceBoundSegment:
    """Opaque reference whose text is loaded only from a verified grant."""

    source_grant_digest: str


@dataclass(frozen=True, slots=True)
class SourcePresentationSegment:
    """Trusted deterministic presentation of one verified source grant."""

    source_grant_digest: str
    text: str
    presentation_kind: str


@dataclass(frozen=True, slots=True)
class OutboundText:
    """Ordered typed segments that construct one outbound JSON string."""

    segments: tuple[
        LiteralSegment
        | SanitizedSegment
        | GeneratedContextSegment
        | CodexReasoningReplaySegment
        | ValidatedToolSyntaxSegment
        | SourceBoundSegment
        | SourcePresentationSegment
        | UntrustedProvenanceSegment,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class TypedOutboundRequest:
    """Remote request recipe; no independent raw string leaf is permitted."""

    payload: Mapping[str, Any]
    session_id: str
    turn_id: str
    request_id: str
    policy_digest: str


@dataclass(frozen=True, slots=True)
class EgressDecision:
    """Content-free result of one egress preflight."""

    allowed: bool
    destination_class: DestinationClass
    provider: str
    model: str
    payload_sha256: str
    serialized_bytes: int
    estimated_tokens: int
    source_grant_count: int
    source_segment_count: int
    session_id: str
    turn_id: str
    request_id: str
    policy_digest: str
    reason_codes: tuple[str, ...] = ()
    base_url: str = ""
    api_mode: str = ""
    grant_digests: tuple[str, ...] = ()


class EgressBlocked(RuntimeError):
    """Raised when the final request is not authorized for its destination."""

    def __init__(self, decision: EgressDecision):
        self.decision = decision
        reasons = ",".join(decision.reason_codes) or "policy_denied"
        super().__init__(f"LLM egress blocked: {reasons}")


class SanitizedTextRejected(ValueError):
    """Raised when remote text cannot earn the sanitized segment type."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(f"sanitized remote text rejected: {reason_code}")


@dataclass(frozen=True, slots=True)
class AuthorizedEgress:
    """Immutable exact bytes that a provider callback is authorized to send."""

    decision: EgressDecision
    payload_bytes: bytes

    @property
    def allowed(self) -> bool:
        return self.decision.allowed

    @property
    def destination_class(self) -> DestinationClass:
        return self.decision.destination_class

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return self.decision.reason_codes

    def verify_payload(self, candidate: bytes) -> bytes:
        """Return the authorized bytes or reject a post-preflight mutation."""

        if not isinstance(candidate, bytes) or not compare_digest(
            sha256(candidate).hexdigest(),
            self.decision.payload_sha256,
        ):
            raise EgressBlocked(
                replace(
                    self.decision,
                    allowed=False,
                    reason_codes=("payload_digest_mismatch",),
                )
            )
        return self.payload_bytes


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:@/-]{0,256}$")
_LOCAL_PROCESS_MODES = frozenset({"local_process", "in_process"})
_BASE64_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9_+/\-])([A-Za-z0-9_+/\-]{4,}={0,2})(?![A-Za-z0-9_+/=\-])"
)
_CHUNKED_BASE64_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9_+/=-])(?:[A-Za-z0-9_+/=-]{2,4}\s+){2,}"
    r"[A-Za-z0-9_+/=-]{2,4}(?![A-Za-z0-9_+/=-])"
)
_HERMES_TASK_ID = re.compile(r"^t_[0-9a-f]{8}$")
_PROMPT_CACHE_KEY = re.compile(r"^pck_[0-9a-f]{24}$")
_CODEX_ENCRYPTED_REASONING_REPLAY = re.compile(r"^gAAAA[A-Za-z0-9_=-]{20,}$")
_BOUNDED_DURATION = re.compile(r"^(?:0|[1-9][0-9]{0,6})(?:ms|s|m|h)$")
_BOUNDED_CLI_WORD = re.compile(r"^--[a-z]+(?:-[a-z]+)*$")
_SAFE_DIAGNOSTIC_STATUS_WORDS = frozenset({
    "PASS",
    "WARN",
    "SUMMARY",
    "REQUIREMENTS",
    "AVAILABILITY",
    "HANDLING",
    "VERIFICATION",
    "ADVISORY",
    "FAIL",
    "SHA1",
    "CRITICAL",
})
_LINTER_DIAGNOSTIC_CODE = re.compile(r"^[A-Z][0-9]{3,4}$")
_PYTHON_DUNDER_IDENTIFIER = re.compile(r"^__[a-z][a-z0-9_]{0,62}__$")
_PYTHON_PRIVATE_IDENTIFIER = re.compile(r"^_[A-Za-z][A-Za-z0-9_]{1,63}$")
_PYTHON_MIXED_CASE_IDENTIFIER = re.compile(
    r"^[a-z][A-Za-z0-9]*_[A-Za-z0-9_]*[A-Z][A-Za-z0-9_]*$"
)
_BOUNDED_SOURCE_CODE_ATOM = re.compile(
    r"(?:[a-z][a-z0-9]{0,63}(?:_[a-z0-9]{1,64}){1,7}"
    r"|[A-Z][A-Z0-9]{0,63}(?:_[A-Z0-9]{1,64}){1,7}"
    r"|[a-z][a-z0-9]{0,63}(?:-[a-z][a-z0-9]{0,63}){1,7}"
    r"|[A-Z][0-9]{3,4})"
)
# Bounded operational tokens are emitted by ordinary CLI/test tooling. They
# can decode as Base64 by coincidence, but are not opaque encoded payloads.
_BOUNDED_SHORT_CLI_OPTION = re.compile(r"^-[A-Za-z]{1,8}$")
_BOUNDED_LINE_RANGE_OR_UNIT = re.compile(
    r"^(?:L?[0-9]{1,6}(?:-[0-9]{1,6})?|[0-9]{1,6}[A-Za-z])$"
)
_BOUNDED_STATUS_COUNT = re.compile(
    r"^(?:passed|failed|skipped|warnings?)/[0-9]{1,6}$"
)
_BOUNDED_VERSIONED_IDENTIFIER = re.compile(
    r"^(?:[a-z][a-z0-9]*(?:[-_][a-z][a-z0-9]*){1,7}[-_]?[0-9][a-z0-9]*"
    r"|[a-z][a-z0-9]*(?:_[a-z0-9]+){1,7}_[0-9]{4,8})$"
)
_BOUNDED_TEST_ARTIFACT = re.compile(
    r"^(?:tmp|test)_[a-z0-9]+(?:_[a-z0-9]+){2,7}$"
)
_BOUNDED_FUNCTION_IDENTIFIER = re.compile(
    r"^_[a-z][a-z0-9]*(?:_[a-z0-9]+){2,7}_"
    r"(?:task|test|runner|command|path|id|status|result)$"
)
_BOUNDED_COMMAND_PATH = re.compile(r"^/[a-z][a-z0-9_.-]{2,31}$")
_BOUNDED_RENDER_MARKER = re.compile(r"^[nN]---$")
_BOUNDED_SOURCE_CONTROL_FRAGMENT = re.compile(r"^[0-9a-f]{13,39}$")
_SOURCE_CONTROL_CONTEXT = re.compile(
    r"\b(?:commit|sha(?:1|256)?|head|base|revision|digest|hash)\b",
    re.IGNORECASE,
)
# Any-case letters + optional trailing slash: GitHub-style org/repo slugs
# ("NousResearch/hermes") and vault/skill paths ("Memories/Shared/") use
# mixed case; a lone directory reference ("scripts/") has nothing after
# its final slash. Hyphens are still excluded from segments here (a
# repo/org segment containing one, like "hermes-agent", is handled by
# _BOUNDED_KEBAB_WORD below and by the two combining at the call site).
_BOUNDED_SLASH_WORDS = re.compile(
    r"^(?://[A-Za-z0-9]{2,}"
    r"|(?:/{1,2})?[A-Za-z0-9]{2,}(?:/[A-Za-z0-9]{2,})+/?"
    r"|[A-Za-z0-9]{2,}/)$"
)
_MAX_BASE64_CANDIDATE_CHARS = 262_144
_VALIDATED_TOOL_SYNTAX = {
    "tool_protocol_identifier": re.compile(
        r"[A-Za-z0-9][A-Za-z0-9_.|:-]{0,255}"
    ),
    "application_identifier": re.compile(
        r"(?:t_[0-9a-f]{8}|[0-9a-f]{40}|[0-9a-f]{64}|"
        r"[a-z][a-z0-9]{0,31}(?:[_-][a-z][a-z0-9]{0,31}){1,7}"
        r"(?::v[0-9]{1,3})?)"
    ),
    "verified_diagnostic_atom": re.compile(
        r"(?:PASS|WARN|SUMMARY|REQUIREMENTS|AVAILABILITY|HANDLING|VERIFICATION|"
        r"[0-9]{1,10}|0x[0-9a-fA-F]{1,16}|"
        r"_?[A-Za-z][A-Za-z0-9]{0,63}(?:_[A-Za-z0-9]{1,64}){1,7})"
    ),
    "separator": re.compile(r"[ \t\r\n,:=()\[\]{}]+"),
    "github_url": re.compile(
        r"https://github\.com/[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}"
        r"(?:\.git|/(?:pull|issues)/[0-9]{1,10})?"
    ),
    "cli_option": re.compile(r"--[a-z0-9][a-z0-9-]{0,63}"),
    "source_identifier": re.compile(
        r"(?:[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}#[0-9]{1,10}"
        r"|[0-9a-f]{40}|[0-9a-f]{64})"
    ),
    "git_ref": re.compile(
        r"(?:refs/(?:heads|tags)/)?[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
        r"(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,63}){1,8}"
    ),
    "run_counter": re.compile(
        r"(?:run|run_id|attempt|attempt_id)(?:\s+|=)[0-9]{1,10}"
    ),
    "safe_env_counter": re.compile(
        r"HERMES_(?:KANBAN_RUN_ID|TURN_LEASE_TIMEOUT|STREAM_STALE_GIVEUP)="
        r"[0-9]{1,10}"
    ),
}
_PRIVATE_ABSOLUTE_PATH = re.compile(
    r"(?:^|[\s\"'`(])(?:"
    r"/(?:Users|home|private|var/folders|root|Volumes)/[^\s\"'`)]+"
    r"|~(?:/|\\)[^\s\"'`)]+"
    r"|[A-Za-z]:\\+(?:Users|Documents and Settings)\\+[^\s\"'`)]+"
    r")",
    re.IGNORECASE,
)
# These are fixed provider-protocol grammar atoms, not a caller-configurable
# egress allowlist. Several happen to round-trip as unpadded Base64 even though
# they are required JSON schema words. They still go through the final secret
# scan; this set only resolves the mathematical ambiguity in Base64 detection.
_PROTOCOL_GRAMMAR_ATOMS = frozenset(
    {
        "--noEmit",
        "--repository",
        "--result",
        "-removed",
        "100K",
        "2000",
        "2026",
        "4dae",
        "600s",
        "BOTH",
        "COVERAGE",
        "EPUB",
        "FTS5",
        "FULL",
        "GGUF",
        "MMLU",
        "OPTIONAL",
        "RELATIVE",
        "REPLACES",
        "REST",
        "SKIP",
        "THAT",
        "TODO",
        "UNAVAILABLE",
        "WAIT",
        "WHEN",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BRANCH",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_WORKSPACE",
        "HERMES_CONTROL_HOME",
        "HERMES_HOME=",
        "HERMES_SESSION_ID",
        "HERMES_STREAM_STALE_GIVEUP",
        "HERMES_TURN_LEASE_TIMEOUT",
        "HEAD",
        "HTTP",
        "HYGIENE",
        "LAST",
        "MESSAGE",
        "MIME",
        "MODE",
        "MUTUALLY",
        "MUST",
        "NOTE",
        "ONLY",
        "PARALLEL",
        "PATH",
        "PRAGMA",
        "REPL",
        "REPLACE",
        "REQUIRED",
        "SILENTLY",
        "THIS",
        "USER",
        "WSL1",
        "_is_git_worktree",
        "assistant",
        "already-resolved",
        "assignee/profile",
        "computer_call_output",
        "claim/finalize/retry",
        "com/docs",
        "content",
        "developer",
        "doc/",
        "dispatcher_current_directory",
        "acceptance-valid",
        "architecture-diagram",
        "autonomous-ai-agents",
        "available_skills",
        "background-first",
        "document-to-action-items",
        "evaluating-llms-harness",
        "filesystem-writing",
        "find_referencing_symbols",
        "get_symbols_overview",
        "github-pr-workflow",
        "google-workspace",
        "hermes-agent-skill-authoring",
        "match_message_id",
        "meeting-action-items",
        "merge-reconciler",
        "p5js",
        "popular-web-designs",
        "requesting-code-review",
        "software-development",
        "songwriting-and-ai-music",
        "systematic-debugging",
        "weekly-review-planning",
        "50KB",
        "1800",
        "8787",
        "ids/goals/status/transcripts",
        "references/templates/scripts",
        "echo/cat",
        "echo/heredoc",
        "environment-variable",
        "find-and-replace",
        "function_call",
        "function_call_output",
        "+for",
        "repeated_exact_failure_block",
        "_force_close_actionable_pending_routes_for_cycle",
        "github-code-review",
        "grep/rg/find/ls",
        "include_archived",
        "input_image",
        "input_text",
        "JSON",
        "kanban_heartbeat",
        "machine-readable",
        "max_runtime_seconds",
        "messages",
        "n_error_",
        "notification/15s",
        "optional-profile",
        "OPEN/MERGEABLE/CLEAN",
        "output_text",
        "parent/child",
        "parents=",
        "parallel_tool_calls",
        "path/to/file",
        "ppt/",
        "prepare_receipt_worktree",
        "prompt_cache_key",
        "protected-remote",
        "repository-owned",
        "runtime-executed",
        "reasoning",
        "role",
        "sed/awk",
        "servers/daemons",
        "servers/watchers/daemons",
        "session_resolver",
        "skills/plugins/cron/memories",
        "logic-regression",
        "system",
        "tool",
        "user",
        "workspace_access",
    }
)


def classify_destination(
    provider: str,
    base_url: str | None,
    api_mode: str | None,
) -> DestinationClass:
    """Classify without DNS, provider-name, or private-network trust.

    Only an explicit in-process mode or a numeric loopback literal is local.
    LAN, Tailscale, container DNS, ``localhost``, and other hostnames retain
    remote policy.  Missing or malformed endpoint data is unknown.
    """

    del provider  # Provider labels are not a security boundary.
    mode = str(api_mode or "").strip().lower()
    if mode in _LOCAL_PROCESS_MODES:
        return DestinationClass.LOCAL_PROCESS
    if not isinstance(base_url, str) or not base_url.strip():
        return DestinationClass.UNKNOWN
    try:
        parsed = urlsplit(base_url.strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return DestinationClass.UNKNOWN
        # ``hostname`` does not validate the port.  Accessing ``port`` is
        # load-bearing: urllib deliberately raises for non-numeric and
        # out-of-range values that must never inherit loopback trust.
        try:
            parsed.port
        except ValueError:
            return DestinationClass.UNKNOWN
        if parsed.netloc.rsplit("@", 1)[-1].endswith(":"):
            return DestinationClass.UNKNOWN
        host = parsed.hostname
        if not host:
            return DestinationClass.UNKNOWN
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return DestinationClass.REMOTE
        return DestinationClass.LOOPBACK if address.is_loopback else DestinationClass.REMOTE
    except (TypeError, ValueError):
        return DestinationClass.UNKNOWN


def _route_value(route: Any, name: str, default: Any = None) -> Any:
    if isinstance(route, Mapping):
        return route.get(name, default)
    return getattr(route, name, default)


def _request_identity(request: Mapping[str, Any], name: str) -> str:
    value = request.get(name, "")
    return value if isinstance(value, str) else str(value)


def _receipt_identifier(value: str) -> str:
    """Keep ordinary correlation IDs while hashing unsafe or sensitive labels."""

    try:
        safe = (
            _SAFE_ID.fullmatch(value) is not None
            and not value.startswith(("/", "~"))
            and redact_sensitive_text(
                value,
                force=True,
                redact_url_credentials=True,
            )
            == value
        )
    except Exception:
        safe = False
    return value if safe else f"sha256:{sha256(value.encode('utf-8')).hexdigest()}"


def source_grant_digest(grant: SourceGrant) -> str:
    """Return the opaque identity used by a source-segment manifest."""

    bound_fields = {
        "canonical_path": str(grant.canonical_path),
        "display_path": grant.display_path,
        "line_start": grant.line_start,
        "line_end": grant.line_end,
        "content_sha256": grant.content_sha256,
        "byte_count": grant.byte_count,
        "session_id": grant.session_id,
        "turn_id": grant.turn_id,
        "request_id": grant.request_id,
        "policy_digest": grant.policy_digest,
    }
    encoded = json.dumps(bound_fields, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def static_literal_sha256(text: str) -> str:
    """Hash one exact UTF-8 static literal for a policy allowlist."""

    if not isinstance(text, str):
        raise TypeError("static literal must be text")
    return sha256(text.encode("utf-8")).hexdigest()


def validate_tool_syntax(text: str, syntax_kind: str) -> str:
    """Revalidate one complete protected tool-result syntax atom."""

    grammar = _VALIDATED_TOOL_SYNTAX.get(syntax_kind)
    if not isinstance(text, str) or grammar is None or grammar.fullmatch(text) is None:
        raise ValueError("invalid_tool_syntax_segment")
    if _contains_secret(text) or _contains_private_absolute_path(text):
        raise ValueError("invalid_tool_syntax_segment")
    return text


def _canonical_base64_candidate(candidate: str) -> bool:
    """Recognize bounded canonical encodings without flagging ordinary IDs."""

    if candidate in _PROTOCOL_GRAMMAR_ATOMS:
        return False
    if candidate in _SAFE_DIAGNOSTIC_STATUS_WORDS:
        # Exact status labels are not an encoding channel; treating them as
        # Base64 strands workers while replaying ordinary CLI output.
        return False
    if _LINTER_DIAGNOSTIC_CODE.fullmatch(candidate):
        # Ruff/flake8-style findings are ordinary bounded CI metadata. They
        # are not source excerpts or opaque encoded payloads, even when a
        # tool result is carried as a SanitizedSegment rather than generated
        # context where the source-atom mask would already apply.
        return False
    if _PYTHON_DUNDER_IDENTIFIER.fullmatch(candidate):
        # Python's bounded dunder names are source-language structure, not
        # encoded content (for example __file__ and __main__ in CI scripts).
        return False
    if (
        _BOUNDED_VERSIONED_IDENTIFIER.fullmatch(candidate)
        or _BOUNDED_TEST_ARTIFACT.fullmatch(candidate)
        or _BOUNDED_FUNCTION_IDENTIFIER.fullmatch(candidate)
    ):
        # Filenames, test names, model slugs, and config identifiers are
        # normal generated/tool context. They are not encoded content merely
        # because their lexical shape happens to decode canonically.
        return False
    if (
        _BOUNDED_SHORT_CLI_OPTION.fullmatch(candidate)
        or _BOUNDED_LINE_RANGE_OR_UNIT.fullmatch(candidate)
        or _BOUNDED_STATUS_COUNT.fullmatch(candidate)
        or _BOUNDED_COMMAND_PATH.fullmatch(candidate)
        or _BOUNDED_RENDER_MARKER.fullmatch(candidate)
    ):
        return False
    if _BOUNDED_SLASH_WORDS.fullmatch(candidate):
        # Bounded relative paths such as venv/lib/python3 are ordinary local
        # CI metadata, not encoded content. Keep the grammar narrow so an
        # unrecognized underscore atom remains fail-closed below.
        return False
    if _BOUNDED_DURATION.fullmatch(candidate):
        return False
    if re.fullmatch(r"0x[0-9a-fA-F]+", candidate):
        return False
    if not 4 <= len(candidate) <= _MAX_BASE64_CANDIDATE_CHARS:
        return False
    # Short words and word-shaped structural fragments frequently round-trip
    # mathematically as unpadded Base64. Their bounded lexical form is the
    # disambiguating signal; padding, digits, mixed punctuation, and long
    # ambiguous blobs remain eligible for canonical decoding below.
    if len(candidate) < 24:
        if candidate.isalpha() and not candidate.isupper():
            return False
        if _BOUNDED_CLI_WORD.fullmatch(candidate):
            return False
        if _BOUNDED_SLASH_WORDS.fullmatch(candidate):
            return False
    # Short, word-like URL-safe slugs are common model/provider identifiers.
    # Keep genuinely encoding-shaped values such as ``-_8A`` eligible for the
    # canonical decoder below.
    if (
        len(candidate) < 16
        and any(character in "-_" for character in candidate)
        and candidate[0].isalnum()
        and candidate[-1].isalnum()
        and sum(character.isalpha() for character in candidate) >= 2
    ):
        return False
    unpadded = candidate.rstrip("=")
    if "=" in unpadded or len(unpadded) % 4 == 1:
        return False
    padded = unpadded + "=" * (-len(unpadded) % 4)
    encoded = padded.encode("ascii")
    for altchars in (None, b"-_"):
        try:
            decoded = base64.b64decode(encoded, altchars=altchars, validate=True)
        except (binascii.Error, ValueError):
            continue
        canonical_padded = (
            base64.b64encode(decoded)
            if altchars is None
            else base64.urlsafe_b64encode(decoded)
        ).decode("ascii")
        if candidate in {canonical_padded, canonical_padded.rstrip("=")} and decoded:
            return True
    return False


def _canonical_chunked_base64_candidate(candidate: str) -> bool:
    """Recognize fixed-width wrapped encodings without joining ordinary prose."""

    chunks = re.findall(r"[A-Za-z0-9_+/=-]{2,4}", candidate)
    if len(chunks) < 3:
        return False
    if all(_LINTER_DIAGNOSTIC_CODE.fullmatch(chunk) for chunk in chunks):
        # A run of bounded Ruff/flake8 findings is structured CI output, not
        # a wrapped encoding. Without this guard, ``E501 F821 W391`` is joined
        # across spaces and can happen to decode canonically.
        return False
    width = len(chunks[0])
    if not all(len(chunk) == width for chunk in chunks[:-1]):
        return False
    if len(chunks[-1]) > width:
        return False
    joined = "".join(chunks)
    has_encoding_signal = any(
        character.isupper() or character.isdigit() or character in "+/_="
        for character in joined
    )
    if len(chunks) < 8 and not has_encoding_signal:
        return False
    return _canonical_base64_candidate(joined)


# GitHub's legacy GraphQL global node id: base64 of a fixed
# ``<digits>:<TypeName><digits>`` grammar (e.g. "05:Issue160502814" ->
# "MDU6SXNzdWUxNjA1MDI4MTQ0"). `gh api` / `gh issue|pr list` return these in
# every issue/PR/label/comment object's `node_id` field, so any tool-output
# scan that fetches GitHub issue or PR data structurally contains them. They
# are fixed, low-entropy protocol identifiers for public object identity —
# not caller-supplied encoded content — so they get the same treatment as
# the other bounded protocol grammars above (call_/fc_ ids, kanban task ids,
# prompt cache keys). Investigation for t_80e6f80b: this pattern was
# repeatedly and falsely flagged as ``base64_payload`` on every provider
# fallback from a local model to a REMOTE one mid-scan, permanently
# excluding cloud fallback for any GitHub-issue-reading profile.
_GITHUB_LEGACY_NODE_ID_GRAMMAR = re.compile(r"\A\d{1,3}:[A-Za-z]{2,40}\d{1,20}\Z")


def _looks_like_github_legacy_node_id(candidate: str) -> bool:
    unpadded = candidate.rstrip("=")
    if "=" in unpadded or len(unpadded) % 4 == 1:
        return False
    padded = unpadded + "=" * (-len(unpadded) % 4)
    try:
        decoded = base64.b64decode(padded.encode("ascii"), validate=True)
    except (binascii.Error, ValueError):
        return False
    try:
        text = decoded.decode("ascii")
    except UnicodeDecodeError:
        return False
    return bool(_GITHUB_LEGACY_NODE_ID_GRAMMAR.fullmatch(text))


def _contains_canonical_base64(value: Any, *, seen: set[int] | None = None) -> bool:
    if isinstance(value, str):
        # Fixed Hermes/Nous attribution tags are protocol metadata, not an
        # encoded source payload. They remain subject to secret/path scans.
        if value.startswith(("product=hermes-agent", "client=hermes-client-")):
            return False
        for match in _BASE64_CANDIDATE.finditer(value):
            candidate = match.group(1)
            prefix = value[max(0, match.start() - 16) : match.start()].lower()
            source_control_window = value[
                max(0, match.start() - 48) : min(len(value), match.end() + 16)
            ]
            if re.fullmatch(
                r"[0-9a-f]{7,12}|[0-9a-f]{40}|[0-9a-f]{64}",
                candidate.lower(),
            ):
                continue
            if (
                _BOUNDED_SOURCE_CONTROL_FRAGMENT.fullmatch(candidate.lower())
                and _SOURCE_CONTROL_CONTEXT.search(source_control_window)
            ):
                # Shortened git object IDs are ordinary source-control
                # metadata when explicitly labeled as such. Without that
                # context, arbitrary hex remains fail-closed.
                continue
            if candidate.isdigit():
                before = value[: match.start(1)].rstrip()[-1:]
                after = value[match.end(1) :].lstrip()[:1]
                if before in {":", ",", "["} and after in {",", "]", "}"}:
                    # JSON numeric values in tool results are serialized
                    # protocol fields, not encoded text. A quoted numeric
                    # string remains eligible for Base64 detection.
                    continue
            # The fixed Kanban task-id grammar carries only a 32-bit hex
            # database key. It is application protocol metadata, not an
            # encoded source payload.
            if _HERMES_TASK_ID.fullmatch(candidate):
                continue
            # Provider-generated tool-call and response-item identifiers are
            # opaque protocol routing metadata, not caller-supplied encoded
            # content.  Match their complete, fixed grammar only.
            if re.fullmatch(r"(?:call|fc)_[A-Za-z0-9_-]{8,128}", candidate):
                continue
            if candidate in {
                "HERMES_CONTROL_HOME",
                "HERMES_KANBAN_DB",
                "HERMES_KANBAN_WORKSPACES_ROOT",
                "HERMES_PROFILE_HOME",
            }:
                continue
            # Content-addressed cache routing is a fixed application protocol
            # value: the literal ``pck_`` prefix plus exactly 96 bits of hex.
            if _PROMPT_CACHE_KEY.fullmatch(candidate):
                continue
            # GitHub's legacy global node id (see helper docstring above).
            if _looks_like_github_legacy_node_id(candidate):
                continue
            if _canonical_base64_candidate(candidate):
                return True
        # Providers and source-control tools sometimes wrap an otherwise
        # canonical encoding at a fixed column. Normalize only bounded chunks
        # so ordinary prose words are not concatenated into a false candidate.
        for match in _CHUNKED_BASE64_CANDIDATE.finditer(value):
            if _canonical_chunked_base64_candidate(match.group(0)):
                return True
        return False
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
        # Typed request keys are application-owned structure, not payload
        # segments. Only values can carry caller-controlled concealment.
        return any(_contains_canonical_base64(item, seen=seen) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
        return any(_contains_canonical_base64(item, seen=seen) for item in value)
    return False


def _source_text_for_base64_scan(text: str) -> str:
    """Mask bounded code atoms only after exact source-grant validation.

    Snake-case config keys, lowercase kebab-case rule names, and linter codes
    can mathematically round-trip as unpadded Base64.  Their grammar is
    low-entropy and ordinary in source files.  Actual encoded blobs remain
    unchanged and are still rejected by the canonical scanner.
    """

    def is_source_code_atom(candidate: str) -> bool:
        return (
            _BOUNDED_SOURCE_CODE_ATOM.fullmatch(candidate) is not None
            or _PYTHON_DUNDER_IDENTIFIER.fullmatch(candidate) is not None
            or _PYTHON_PRIVATE_IDENTIFIER.fullmatch(candidate) is not None
            or _PYTHON_MIXED_CASE_IDENTIFIER.fullmatch(candidate) is not None
        )

    return _BASE64_CANDIDATE.sub(
        lambda match: (
            "<code>"
            if is_source_code_atom(match.group(1))
            else match.group(0)
        ),
        text,
    )


def _generated_context_text_for_base64_scan(text: str) -> str:
    """Mask bounded application atoms in already-redacted generated context.

    Generated context is produced by Hermes and has already passed the
    secret/path/base64 redaction step.  Its ordinary function names, rule
    names, and schema identifiers still need the source-style lexical mask at
    the final scan; arbitrary encoded values remain untouched and fail closed.
    """

    return _source_text_for_base64_scan(text)


def _contains_secret(value: Any, *, seen: set[int] | None = None) -> bool:
    """Apply forced redaction semantics independently to every request string."""

    if isinstance(value, str):
        return redact_sensitive_text(
            value,
            force=True,
            redact_url_credentials=True,
        ) != value
    if isinstance(value, (bytes, bytearray, memoryview)):
        # Binary request material is not safely inspectable as text.
        return True
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
        return any(
            _contains_secret(key, seen=seen) or _contains_secret(item, seen=seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
        return any(_contains_secret(item, seen=seen) for item in value)
    return False


def _contains_exact_secret(
    value: Any,
    exact_values: Sequence[str],
    *,
    seen: set[int] | None = None,
) -> bool:
    """Match authoritative applied/environment credential bytes exactly."""

    if isinstance(value, str):
        return any(secret in value for secret in exact_values)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
        return any(
            _contains_exact_secret(key, exact_values, seen=seen)
            or _contains_exact_secret(item, exact_values, seen=seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
        return any(
            _contains_exact_secret(item, exact_values, seen=seen) for item in value
        )
    return False


def _contains_private_absolute_path(value: Any, *, seen: set[int] | None = None) -> bool:
    """Reject common host-private absolute paths without blocking API paths."""

    if isinstance(value, str):
        return _PRIVATE_ABSOLUTE_PATH.search(value) is not None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
        return any(
            _contains_private_absolute_path(key, seen=seen)
            or _contains_private_absolute_path(item, seen=seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
        return any(_contains_private_absolute_path(item, seen=seen) for item in value)
    return False


def redact_remote_unsafe_text(text: str) -> str:
    """Redact non-secret unsafe text in Hermes-generated remote context.

    Secrets intentionally remain a hard firewall denial. Private paths and
    canonical base64-shaped protocol text can be replaced while preserving
    the surrounding system/tool instructions needed by remote models.
    """

    if not isinstance(text, str):
        raise TypeError("remote context must be text")

    def replace_path(match: re.Match[str]) -> str:
        value = match.group(0)
        prefix_match = re.match(r"^[\s\"'`(]*", value)
        prefix = prefix_match.group(0) if prefix_match else ""
        return prefix + "<private-path>"

    redacted = _PRIVATE_ABSOLUTE_PATH.sub(replace_path, text)
    def replace_base64(match: re.Match[str]) -> str:
        candidate = match.group(1)
        if re.fullmatch(
            r"[0-9a-f]{7,12}|[0-9a-f]{40}|[0-9a-f]{64}", candidate.lower()
        ):
            return match.group(0)
        if candidate.isdigit():
            before = redacted[: match.start(1)].rstrip()[-1:]
            after = redacted[match.end(1) :].lstrip()[:1]
            if before in {":", ",", "["} and after in {",", "]", "}"}:
                return match.group(0)
        if candidate in _PROTOCOL_GRAMMAR_ATOMS or _HERMES_TASK_ID.fullmatch(candidate):
            return match.group(0)
        if _BOUNDED_SOURCE_CODE_ATOM.fullmatch(candidate):
            return match.group(0)
        if re.fullmatch(r"(?:call|fc)_[A-Za-z0-9_-]{8,128}", candidate):
            return match.group(0)
        if candidate in {
            "HERMES_CONTROL_HOME",
            "HERMES_KANBAN_DB",
            "HERMES_KANBAN_WORKSPACES_ROOT",
            "HERMES_PROFILE_HOME",
        }:
            return match.group(0)
        if _PROMPT_CACHE_KEY.fullmatch(candidate) or _canonical_base64_candidate(candidate):
            return "<redacted-base64>"
        return match.group(0)

    redacted = _BASE64_CANDIDATE.sub(replace_base64, redacted)
    return _CHUNKED_BASE64_CANDIDATE.sub(
        lambda match: "<redacted-base64>"
        if _canonical_chunked_base64_candidate(match.group(0))
        else match.group(0),
        redacted,
    )


def content_free_violation_locations(value: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return structural indexes and reasons without returning request text."""

    locations: list[tuple[str, tuple[str, ...]]] = []
    seen: set[int] = set()

    def visit(item: Any, path: str) -> None:
        if isinstance(item, str):
            reasons: list[str] = []
            if _contains_canonical_base64(item):
                reasons.append("base64_payload")
            if _contains_private_absolute_path(item):
                reasons.append("private_absolute_path")
            if reasons:
                locations.append((path, tuple(reasons)))
            return
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen:
                locations.append((path, ("cyclic_container",)))
                return
            seen.add(identity)
            for index, (key, child) in enumerate(item.items()):
                visit(key, f"{path}.map[{index}].key")
                visit(child, f"{path}.map[{index}].value")
            return
        if isinstance(item, (list, tuple, set, frozenset)):
            identity = id(item)
            if identity in seen:
                locations.append((path, ("cyclic_container",)))
                return
            seen.add(identity)
            for index, child in enumerate(item):
                visit(child, f"{path}.sequence[{index}]")

    visit(value, "$")
    return tuple(locations)


def _contains_grant_substring(grant_content: bytes, candidate: bytes) -> bool:
    """Reject source-derived proper substrings in sanitized text.

    Newline-trimmed line grants commonly appear in JSON without their source
    line ending. Exact containment catches short excerpts; otherwise require
    a 32-byte shared window so ordinary words such as ``checkout`` do not make
    unrelated generated context look source-derived.
    """

    if not grant_content or not candidate:
        return False
    grant_variants = (grant_content, grant_content.rstrip(b"\r\n"))
    for variant in grant_variants:
        if not variant:
            continue
        if variant in candidate:
            return True
        window = min(32, len(candidate), len(variant))
        if window >= 32:
            source_windows = {
                variant[offset : offset + window]
                for offset in range(0, len(variant) - window + 1)
            }
            if any(
                candidate[offset : offset + window] in source_windows
                for offset in range(0, len(candidate) - window + 1)
            ):
                return True
    return False


def validate_sanitized_text(text: str, *, max_bytes: int = 32_768) -> str:
    """Return unchanged bounded remote-safe text or reject it fail-closed.

    This is the only constructor-side admission path for SanitizedSegment.
    The firewall repeats the same scans on the final rendered payload.
    """

    if not isinstance(text, str):
        raise SanitizedTextRejected("invalid_sanitized_text")
    if max_bytes <= 0 or len(text.encode("utf-8")) > max_bytes:
        raise SanitizedTextRejected("sanitized_bytes_exceeded")
    try:
        if _contains_secret(text):
            raise SanitizedTextRejected("secret_detected")
    except SanitizedTextRejected:
        raise
    except Exception as exc:
        raise SanitizedTextRejected("redaction_failed") from exc
    try:
        if _contains_canonical_base64(text):
            raise SanitizedTextRejected("base64_payload")
    except SanitizedTextRejected:
        raise
    except Exception as exc:
        raise SanitizedTextRejected("base64_scan_failed") from exc
    try:
        if _contains_private_absolute_path(text):
            raise SanitizedTextRejected("private_absolute_path")
    except SanitizedTextRejected:
        raise
    except Exception as exc:
        raise SanitizedTextRejected("private_path_scan_failed") from exc
    return text


def _is_strict_sanitized_only_payload(
    value: Any,
    *,
    seen: set[int] | None = None,
) -> tuple[bool, int]:
    """Recognize the one grantless remote shape approved by policy.

    Every text leaf must be an explicit :class:`SanitizedSegment`. Raw text,
    static literals, source references, binary values, cycles, and unsupported
    containers make the entire payload ineligible. The positive count prevents
    an empty structural request from acquiring grantless status.
    """

    if isinstance(value, SanitizedSegment):
        return isinstance(value.text, str), 1 if isinstance(value.text, str) else 0
    if isinstance(value, GeneratedContextSegment):
        return isinstance(value.text, str), 1 if isinstance(value.text, str) else 0
    if isinstance(value, GeneratedContextKey):
        return isinstance(value.text, str), 0
    if isinstance(value, CodexReasoningReplaySegment):
        return (
            isinstance(value.text, str)
            and _CODEX_ENCRYPTED_REASONING_REPLAY.fullmatch(value.text) is not None,
            1 if isinstance(value.text, str) else 0,
        )
    if isinstance(value, UntrustedProvenanceSegment):
        return False, 0
    if isinstance(value, ValidatedToolSyntaxSegment):
        try:
            validate_tool_syntax(value.text, value.syntax_kind)
        except (TypeError, ValueError):
            return False, 0
        return True, 1
    if isinstance(value, LiteralSegment):
        return isinstance(value.text, str), 0
    if isinstance(value, SourceBoundSegment):
        return False, 0
    if isinstance(value, SourcePresentationSegment):
        return False, 0
    if isinstance(value, OutboundText):
        if not value.segments:
            return False, 0
        count = 0
        for segment in value.segments:
            allowed, segment_count = _is_strict_sanitized_only_payload(segment, seen=seen)
            if not allowed:
                return False, 0
            count += segment_count
        return count > 0, count
    if value is None or isinstance(value, (bool, int)):
        return True, 0
    if isinstance(value, float):
        return math.isfinite(value), 0
    if isinstance(value, (str, bytes, bytearray, memoryview, set, frozenset)):
        return False, 0
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return False, 0
        seen.add(identity)
        count = 0
        for key, item in value.items():
            if not isinstance(key, (str, GeneratedContextKey)):
                return False, 0
            allowed, item_count = _is_strict_sanitized_only_payload(item, seen=seen)
            if not allowed:
                return False, 0
            count += item_count
        return True, count
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            return False, 0
        seen.add(identity)
        count = 0
        for item in value:
            allowed, item_count = _is_strict_sanitized_only_payload(item, seen=seen)
            if not allowed:
                return False, 0
            count += item_count
        return True, count
    return False, 0


class LLMEgressFirewall:
    """Validate a final LLM request and record a content-free receipt."""

    def __init__(
        self,
        state_dir: Path | str,
        *,
        max_serialized_bytes: int = 262_144,
        max_sanitized_bytes: int = 32_768,
        max_sanitized_segment_bytes: int = 32_768,
        max_conservative_tokens: int = 87_382,
        max_granted_serialized_bytes: int | None = None,
        max_granted_conservative_tokens: int | None = None,
        conservative_chars_per_token: int = 3,
        policy_digest: str | None = None,
        static_literal_hashes_by_policy: Mapping[str, Sequence[str]] | None = None,
        exact_secret_values: Sequence[str] = (),
    ) -> None:
        if max_serialized_bytes <= 0:
            raise ValueError("max_serialized_bytes must be positive")
        if max_sanitized_bytes <= 0:
            raise ValueError("max_sanitized_bytes must be positive")
        if max_sanitized_segment_bytes <= 0:
            raise ValueError("max_sanitized_segment_bytes must be positive")
        if max_conservative_tokens <= 0:
            raise ValueError("max_conservative_tokens must be positive")
        if max_granted_serialized_bytes is not None and max_granted_serialized_bytes <= 0:
            raise ValueError("max_granted_serialized_bytes must be positive")
        if (
            max_granted_conservative_tokens is not None
            and max_granted_conservative_tokens <= 0
        ):
            raise ValueError("max_granted_conservative_tokens must be positive")
        if conservative_chars_per_token <= 0:
            raise ValueError("conservative_chars_per_token must be positive")
        self._state_dir = Path(state_dir)
        self._receipt_path = self._state_dir / "llm-egress-receipts.jsonl"
        self._max_serialized_bytes = max_serialized_bytes
        self._max_sanitized_bytes = max_sanitized_bytes
        self._max_sanitized_segment_bytes = max_sanitized_segment_bytes
        self._max_conservative_tokens = max_conservative_tokens
        self._max_granted_serialized_bytes = (
            max_serialized_bytes
            if max_granted_serialized_bytes is None
            else max_granted_serialized_bytes
        )
        self._max_granted_conservative_tokens = (
            max_conservative_tokens
            if max_granted_conservative_tokens is None
            else max_granted_conservative_tokens
        )
        self._conservative_chars_per_token = conservative_chars_per_token
        self._policy_digest = str(policy_digest or "")
        self._exact_secret_values = tuple(
            dict.fromkeys(value for value in exact_secret_values if isinstance(value, str) and value)
        )
        self._static_literal_hashes_by_policy = {
            str(policy_digest): frozenset(
                digest
                for digest in digests
                if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
            )
            for policy_digest, digests in (static_literal_hashes_by_policy or {}).items()
        }

    def preflight(
        self,
        request: Mapping[str, Any] | TypedOutboundRequest,
        route: Any,
        *,
        grants: Sequence[SourceGrant] = (),
    ) -> EgressDecision:
        """Preserve the content-free Task 1 decision interface."""

        return self.authorize(request, route, grants=grants).decision

    def authorize(
        self,
        request: Mapping[str, Any] | TypedOutboundRequest,
        route: Any,
        *,
        grants: Sequence[SourceGrant] = (),
    ) -> AuthorizedEgress:
        """Construct and authorize immutable provider bytes or fail closed.

        Remote and unknown destinations accept only ``TypedOutboundRequest``.
        Source text is loaded from verified grants while constructing the
        logical request, so no independent raw source-bearing payload exists
        for a caller to send after preflight.
        """

        provider = str(_route_value(route, "provider", ""))
        model = str(_route_value(route, "model", ""))
        destination = classify_destination(
            provider,
            _route_value(route, "base_url"),
            _route_value(route, "api_mode"),
        )
        base_url = str(_route_value(route, "base_url") or "")
        api_mode = str(_route_value(route, "api_mode") or "")
        typed_request = request if isinstance(request, TypedOutboundRequest) else None
        if typed_request is not None:
            session_id = typed_request.session_id
            turn_id = typed_request.turn_id
            request_id = typed_request.request_id
            policy_digest = typed_request.policy_digest
        else:
            session_id = _request_identity(request, "session_id")
            turn_id = _request_identity(request, "turn_id")
            request_id = _request_identity(request, "request_id")
            policy_digest = _request_identity(request, "policy_digest")

        reasons: list[str] = []
        valid_grants: list[SourceGrant] = []
        grant_contents: dict[str, tuple[SourceGrant, bytes]] = {}
        grant_reasons: list[str] = []
        source_segment_count = 0
        sanitized_only = False
        if typed_request is not None:
            sanitized_shape, sanitized_count = _is_strict_sanitized_only_payload(
                typed_request.payload
            )
            sanitized_only = sanitized_shape and sanitized_count > 0

        if destination == DestinationClass.UNKNOWN:
            reasons.append("unknown_destination")
        if destination in {DestinationClass.REMOTE, DestinationClass.UNKNOWN} and not all(
            (session_id, turn_id, request_id, policy_digest)
        ):
            reasons.append("missing_request_identity")
        if self._policy_digest and policy_digest != self._policy_digest:
            reasons.append("policy_digest_mismatch")
        if destination in {DestinationClass.REMOTE, DestinationClass.UNKNOWN}:
            if typed_request is None:
                reasons.append("typed_request_required")
            # The sole grantless remote lane is a structurally verified request
            # whose every text leaf is an explicit bounded SanitizedSegment.
            # Passing even one purported grant opts back into exact source
            # validation so malformed or unbound authority cannot be ignored.
            if grants or not sanitized_only:
                grant_reasons, valid_grants, grant_contents = self._validate_grants(
                    grants,
                    session_id=session_id,
                    turn_id=turn_id,
                    request_id=request_id,
                    policy_digest=policy_digest,
                )
                reasons.extend(grant_reasons)

        if typed_request is not None:
            (
                logical_request,
                construction_reasons,
                source_segment_count,
                scan_values,
                base64_scan_values,
            ) = (
                self._construct_typed_request(
                    typed_request,
                    grant_contents,
                    allow_sanitized_segments=True,
                )
            )
            reasons.extend(construction_reasons)
        else:
            logical_request = request
            scan_values = request
            base64_scan_values = request

        try:
            serialized = json.dumps(
                logical_request,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            decision = EgressDecision(
                allowed=False,
                destination_class=destination,
                provider=provider,
                model=model,
                payload_sha256="",
                serialized_bytes=0,
                estimated_tokens=0,
                source_grant_count=len(valid_grants),
                source_segment_count=source_segment_count,
                session_id=session_id,
                turn_id=turn_id,
                request_id=request_id,
                policy_digest=policy_digest,
                reason_codes=("serialization_failed",),
                base_url=base_url,
                api_mode=api_mode,
                grant_digests=tuple(source_grant_digest(grant) for grant in valid_grants),
            )
            self._block(decision, valid_grants)

        serialized_bytes = len(serialized)
        estimated_tokens = (
            serialized_bytes + self._conservative_chars_per_token - 1
        ) // self._conservative_chars_per_token
        use_granted_caps = bool(
            typed_request is not None
            and source_segment_count > 0
            and valid_grants
            and not grant_reasons
        )
        serialized_byte_cap = (
            self._max_granted_serialized_bytes
            if use_granted_caps
            else self._max_serialized_bytes
        )
        conservative_token_cap = (
            self._max_granted_conservative_tokens
            if use_granted_caps
            else self._max_conservative_tokens
        )
        if serialized_bytes > serialized_byte_cap:
            reasons.append("serialized_bytes_exceeded")
        if estimated_tokens > conservative_token_cap:
            reasons.append("token_cap_exceeded")

        if destination in {DestinationClass.REMOTE, DestinationClass.UNKNOWN}:
            try:
                if _contains_secret(scan_values):
                    reasons.append("secret_detected")
            except Exception:
                reasons.append("redaction_failed")
            try:
                if _contains_exact_secret(scan_values, self._exact_secret_values):
                    reasons.append("exact_secret_detected")
            except Exception:
                reasons.append("exact_secret_scan_failed")
            try:
                if _contains_canonical_base64(base64_scan_values):
                    reasons.append("base64_payload")
            except Exception:
                reasons.append("base64_scan_failed")
            try:
                if _contains_private_absolute_path(scan_values):
                    reasons.append("private_absolute_path")
            except Exception:
                reasons.append("private_path_scan_failed")

        decision = EgressDecision(
            allowed=not reasons,
            destination_class=destination,
            provider=provider,
            model=model,
            payload_sha256=sha256(serialized).hexdigest(),
            serialized_bytes=serialized_bytes,
            estimated_tokens=estimated_tokens,
            source_grant_count=len(valid_grants),
            source_segment_count=source_segment_count,
            session_id=session_id,
            turn_id=turn_id,
            request_id=request_id,
            policy_digest=policy_digest,
            reason_codes=tuple(dict.fromkeys(reasons)),
            base_url=base_url,
            api_mode=api_mode,
            grant_digests=tuple(source_grant_digest(grant) for grant in valid_grants),
        )
        if not decision.allowed:
            self._block(decision, valid_grants)

        try:
            self._append_receipt(decision, valid_grants)
        except OSError:
            raise EgressBlocked(
                replace(decision, allowed=False, reason_codes=("receipt_unavailable",))
            ) from None
        return AuthorizedEgress(decision=decision, payload_bytes=serialized)

    def _validate_grants(
        self,
        grants: Sequence[SourceGrant],
        *,
        session_id: str,
        turn_id: str,
        request_id: str,
        policy_digest: str,
    ) -> tuple[list[str], list[SourceGrant], dict[str, tuple[SourceGrant, bytes]]]:
        reasons: list[str] = []
        valid: list[SourceGrant] = []
        contents: dict[str, tuple[SourceGrant, bytes]] = {}
        if not grants:
            return ["untrusted_provenance"], valid, contents

        for candidate in grants:
            if not isinstance(candidate, SourceGrant):
                reasons.append("untrusted_provenance")
                continue
            grant = candidate
            if (
                grant.session_id != session_id
                or grant.turn_id != turn_id
                or grant.request_id != request_id
                or grant.policy_digest != policy_digest
            ):
                reasons.append("grant_binding_mismatch")
                continue
            if (
                grant.line_start < 1
                or grant.line_end < grant.line_start
                or grant.byte_count < 0
                or not re.fullmatch(r"[0-9a-f]{64}", grant.content_sha256)
            ):
                reasons.append("invalid_source_grant")
                continue
            display = Path(grant.display_path)
            if display.is_absolute() or ".." in display.parts:
                reasons.append("invalid_display_path")
                continue
            try:
                canonical = Path(grant.canonical_path)
                resolved = canonical.resolve(strict=True)
            except (OSError, RuntimeError, ValueError, TypeError):
                reasons.append("source_unavailable")
                continue
            if not canonical.is_absolute() or canonical != resolved:
                reasons.append("source_path_not_canonical")
                continue
            try:
                blocked = get_read_block_error(str(resolved))
            except Exception:
                reasons.append("source_policy_unavailable")
                continue
            if blocked is not None:
                reasons.append("sensitive_path")
                continue
            try:
                lines = resolved.read_bytes().splitlines(keepends=True)
            except OSError:
                reasons.append("source_unavailable")
                continue
            if grant.line_end > len(lines):
                reasons.append("source_range_mismatch")
                continue
            content = b"".join(lines[grant.line_start - 1 : grant.line_end])
            if len(content) != grant.byte_count or sha256(content).hexdigest() != grant.content_sha256:
                reasons.append("source_hash_mismatch")
                continue
            valid.append(grant)
            contents[source_grant_digest(grant)] = (grant, content)

        if not valid and "untrusted_provenance" not in reasons:
            reasons.append("untrusted_provenance")
        return reasons, valid, contents

    def _construct_typed_request(
        self,
        request: TypedOutboundRequest,
        grant_contents: Mapping[str, tuple[SourceGrant, bytes]],
        *,
        allow_sanitized_segments: bool = False,
    ) -> tuple[Mapping[str, Any], list[str], int, list[str], list[str]]:
        """Build a plain JSON request exclusively from typed segment nodes."""

        reasons: list[str] = []
        referenced_grants: set[str] = set()
        source_segment_count = 0
        sanitized_bytes = 0
        scan_values: list[str] = []
        base64_scan_values: list[str] = []
        allowed_static_hashes = self._static_literal_hashes_by_policy.get(
            request.policy_digest,
            frozenset(),
        )

        def require_static_literal(
            text: str,
            *,
            scan_base64: bool = True,
            scan_secret: bool = True,
        ) -> None:
            # Provenance authorization and content safety are independent.
            # Every rendered text atom is scanned again immediately before
            # authorization, including exact policy-bound static literals and
            # structural keys/scalars.
            if scan_secret:
                scan_values.append(text)
            if scan_base64:
                base64_scan_values.append(text)
            if static_literal_sha256(text) not in allowed_static_hashes:
                reasons.append("static_literal_not_allowed")

        def render_text_segment(
            segment: (
                LiteralSegment
                | SanitizedSegment
                | GeneratedContextSegment
                | CodexReasoningReplaySegment
                | ValidatedToolSyntaxSegment
                | SourceBoundSegment
                | SourcePresentationSegment
                | UntrustedProvenanceSegment
            ),
        ) -> str:
            nonlocal sanitized_bytes, source_segment_count
            if isinstance(segment, LiteralSegment):
                if not isinstance(segment.text, str):
                    reasons.append("invalid_literal_segment")
                    return ""
                require_static_literal(segment.text)
                encoded_literal = segment.text.encode("utf-8")
                if any(content and content in encoded_literal for _, content in grant_contents.values()):
                    reasons.append("source_bytes_in_literal")
                return segment.text
            if isinstance(segment, SanitizedSegment):
                if not allow_sanitized_segments:
                    reasons.append("sanitized_segment_forbidden")
                if isinstance(segment.text, str):
                    encoded = segment.text.encode("utf-8")
                    if len(encoded) > self._max_sanitized_segment_bytes:
                        reasons.append("sanitized_segment_bytes_exceeded")
                    sanitized_bytes += len(encoded)
                    if sanitized_bytes > self._max_sanitized_bytes:
                        reasons.append("sanitized_bytes_exceeded")
                    if any(
                        _contains_grant_substring(content, encoded)
                        for _, content in grant_contents.values()
                    ):
                        reasons.append("source_bytes_in_sanitized_segment")
                    scan_values.append(segment.text)
                    base64_scan_values.append(segment.text)
                    return segment.text
                reasons.append("invalid_literal_segment")
                return ""
            if isinstance(segment, GeneratedContextSegment):
                if not isinstance(segment.text, str):
                    reasons.append("invalid_generated_context_segment")
                    return ""
                # The constructor redacts path/base64-shaped text. Keep the
                # final scans, especially secret detection, as defense in
                # depth, but do not charge generated context to the smaller
                # untrusted-text budget.
                scan_values.append(segment.text)
                base64_scan_values.append(
                    _generated_context_text_for_base64_scan(segment.text)
                )
                return segment.text
            if isinstance(segment, CodexReasoningReplaySegment):
                if not _CODEX_ENCRYPTED_REASONING_REPLAY.fullmatch(segment.text):
                    reasons.append("invalid_codex_reasoning_replay")
                    return ""
                # This opaque token is produced by the Codex Responses API and
                # is typed only for a reasoning item on that route. It must be
                # replayed verbatim for cache and reasoning continuity, but it
                # is not an independently usable credential payload.
                return segment.text
            if isinstance(segment, ValidatedToolSyntaxSegment):
                try:
                    text = validate_tool_syntax(segment.text, segment.syntax_kind)
                except (TypeError, ValueError):
                    reasons.append("invalid_tool_syntax_segment")
                    return ""
                encoded = text.encode("utf-8")
                if len(encoded) > self._max_sanitized_segment_bytes:
                    reasons.append("sanitized_segment_bytes_exceeded")
                sanitized_bytes += len(encoded)
                if sanitized_bytes > self._max_sanitized_bytes:
                    reasons.append("sanitized_bytes_exceeded")
                scan_values.append(text)
                return text
            if isinstance(segment, SourceBoundSegment):
                grant_and_content = grant_contents.get(segment.source_grant_digest)
                if grant_and_content is None:
                    reasons.append("source_segment_grant_mismatch")
                    return ""
                try:
                    text = grant_and_content[1].decode("utf-8")
                except UnicodeDecodeError:
                    reasons.append("source_segment_not_text")
                    return ""
                referenced_grants.add(segment.source_grant_digest)
                source_segment_count += 1
                scan_values.append(text)
                base64_scan_values.append(_source_text_for_base64_scan(text))
                return text
            if isinstance(segment, SourcePresentationSegment):
                grant_and_content = grant_contents.get(segment.source_grant_digest)
                if grant_and_content is None:
                    reasons.append("source_segment_grant_mismatch")
                    return ""
                if segment.presentation_kind != "read_file_json_v1":
                    reasons.append("invalid_source_presentation")
                    return ""
                try:
                    raw_text = grant_and_content[1].decode("utf-8")
                    expected_content = "\n".join(
                        f"{line_number}|{line}"
                        for line_number, line in enumerate(
                            raw_text.split("\n"),
                            start=grant_and_content[0].line_start,
                        )
                    )
                    parsed = json.loads(segment.text)
                except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
                    reasons.append("invalid_source_presentation")
                    return ""
                if not isinstance(parsed, dict) or parsed.get("content") != expected_content:
                    reasons.append("invalid_source_presentation")
                    return ""
                referenced_grants.add(segment.source_grant_digest)
                source_segment_count += 1
                scan_values.append(segment.text)
                base64_scan_values.append(
                    _source_text_for_base64_scan(raw_text)
                )
                return segment.text
            if isinstance(segment, UntrustedProvenanceSegment):
                reasons.append("untrusted_provenance")
                return ""
            reasons.append("invalid_source_segment")
            return ""

        def render(value: Any) -> Any:
            if isinstance(
                value,
                (
                    LiteralSegment,
                    SanitizedSegment,
                    GeneratedContextSegment,
                    CodexReasoningReplaySegment,
                    ValidatedToolSyntaxSegment,
                    SourceBoundSegment,
                    SourcePresentationSegment,
                    UntrustedProvenanceSegment,
                ),
            ):
                return render_text_segment(value)
            if isinstance(value, OutboundText):
                rendered_parts: list[str] = []
                adjacent_sanitized: list[str] = []
                adjacent_non_source: list[str] = []

                def flush_adjacent_sanitized() -> None:
                    if len(adjacent_sanitized) > 1:
                        combined = "".join(adjacent_sanitized)
                        # Segment caps are transport bounds, not scan
                        # boundaries. Re-scan each reconstructed contiguous
                        # sanitized span so splitting cannot conceal a secret,
                        # private path, or encoding across adjacent pieces.
                        scan_values.append(combined)
                        base64_scan_values.append(combined)
                    adjacent_sanitized.clear()

                def flush_adjacent_non_source() -> None:
                    if len(adjacent_non_source) > 1:
                        # Structural typing may exclude an exact validated
                        # atom from Base64 classification, but it must never
                        # split a secret or private path across scan values.
                        scan_values.append("".join(adjacent_non_source))
                    adjacent_non_source.clear()

                for segment in value.segments:
                    rendered = render_text_segment(segment)
                    rendered_parts.append(rendered)
                    if isinstance(segment, SanitizedSegment):
                        adjacent_sanitized.append(rendered)
                    else:
                        flush_adjacent_sanitized()
                    if isinstance(
                        segment,
                        (
                            LiteralSegment,
                            SanitizedSegment,
                            GeneratedContextSegment,
                            CodexReasoningReplaySegment,
                            ValidatedToolSyntaxSegment,
                        ),
                    ):
                        adjacent_non_source.append(rendered)
                    else:
                        flush_adjacent_non_source()
                flush_adjacent_sanitized()
                flush_adjacent_non_source()
                return "".join(rendered_parts)
            if isinstance(value, Mapping):
                rendered: dict[str, Any] = {}
                for key, item in value.items():
                    if isinstance(key, GeneratedContextKey):
                        rendered_key = key.text
                        if not isinstance(rendered_key, str):
                            reasons.append("invalid_generated_context_key")
                            continue
                        require_static_literal(rendered_key, scan_base64=False)
                    elif isinstance(key, str):
                        rendered_key = key
                        # Mapping keys are policy-bound request structure, not
                        # caller-supplied payload.  Scanning them as Base64
                        # turns legitimate protocol fields such as
                        # ``response_item_id`` into false-positive payloads.
                        require_static_literal(rendered_key, scan_base64=False)
                    else:
                        reasons.append("invalid_request_key")
                        continue
                    rendered[rendered_key] = render(item)
                return rendered
            if isinstance(value, (list, tuple)):
                return [render(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                reasons.append("non_finite_number")
                return None
            if value is None or isinstance(value, (bool, int, float)):
                # JSON scalar controls (for example ``max_tokens=4096``) are
                # rendered as unquoted JSON values, never caller-supplied
                # text.  Scanning their string representation as a standalone
                # base64 candidate turns ordinary numeric limits into false
                # egress blocks ("4096" is a valid four-character base64
                # alphabet member).  Keep them policy-bound and in the secret
                # scan, but do not apply a text-payload base64 heuristic.
                require_static_literal(
                    json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
                    scan_base64=False,
                )
                return value
            # In particular, raw strings and bytes are not remote request
            # material. Every outbound string must have a typed owner.
            reasons.append("untyped_request_value")
            return None

        rendered_payload = render(request.payload)
        if not isinstance(rendered_payload, Mapping):
            reasons.append("invalid_typed_request_root")
            rendered_payload = {}
        else:
            bound_identities = {
                "session_id": request.session_id,
                "turn_id": request.turn_id,
                "request_id": request.request_id,
                "policy_digest": request.policy_digest,
            }
            if any(
                field in rendered_payload and rendered_payload[field] != expected
                for field, expected in bound_identities.items()
            ):
                reasons.append("request_identity_mismatch")
        if set(grant_contents) - referenced_grants:
            reasons.append("source_grant_unbound")
        return (
            rendered_payload,
            reasons,
            source_segment_count,
            scan_values,
            base64_scan_values,
        )

    def _block(
        self,
        decision: EgressDecision,
        grants: Sequence[SourceGrant] = (),
    ) -> None:
        try:
            self._append_receipt(decision, grants)
        except OSError:
            decision = replace(
                decision,
                reason_codes=tuple(dict.fromkeys((*decision.reason_codes, "receipt_unavailable"))),
            )
        raise EgressBlocked(decision)

    def _append_receipt(
        self,
        decision: EgressDecision,
        grants: Sequence[SourceGrant],
    ) -> None:
        try:
            if self._state_dir.is_symlink():
                raise OSError("egress state directory must not be a symlink")
        except OSError:
            raise
        self._state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        receipt = asdict(decision)
        receipt["destination_class"] = decision.destination_class.value
        receipt["decision"] = "allow" if decision.allowed else "block"
        for field in (
            "provider",
            "model",
            "base_url",
            "api_mode",
            "session_id",
            "turn_id",
            "request_id",
            "policy_digest",
        ):
            receipt[field] = _receipt_identifier(receipt[field])
        receipt["source_grants"] = [
            {
                "line_start": grant.line_start,
                "line_end": grant.line_end,
                "content_sha256": grant.content_sha256,
                "byte_count": grant.byte_count,
            }
            for grant in grants
        ]
        flags = os.O_APPEND | os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        with exclusive_file_lock(self._receipt_path.with_suffix(".lock")):
            fd = os.open(self._receipt_path, flags, 0o600)
            try:
                secure_file_descriptor_permissions(fd)
                file_size = os.fstat(fd).st_size
                previous_hash = ""
                if file_size:
                    read_start = max(0, file_size - 131_072)
                    os.lseek(fd, read_start, os.SEEK_SET)
                    prior_chunk = os.read(fd, file_size - read_start)
                    prior_lines = prior_chunk.splitlines()
                    if prior_lines:
                        previous_hash = sha256(prior_lines[-1]).hexdigest()
                receipt["receipt_prev_sha256"] = previous_hash
                receipt_material = json.dumps(
                    receipt, ensure_ascii=True, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
                receipt["receipt_sha256"] = sha256(
                    previous_hash.encode("ascii") + receipt_material
                ).hexdigest()
                encoded = (
                    json.dumps(receipt, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
                    + "\n"
                ).encode("utf-8")
                os.lseek(fd, 0, os.SEEK_END)
                if os.write(fd, encoded) != len(encoded):
                    raise OSError("short receipt write")
            finally:
                os.close(fd)


__all__ = [
    "AuthorizedEgress",
    "DestinationClass",
    "EgressBlocked",
    "EgressDecision",
    "GeneratedContextKey",
    "GeneratedContextSegment",
    "LLMEgressFirewall",
    "LiteralSegment",
    "OutboundText",
    "SanitizedSegment",
    "SourceBoundSegment",
    "SourcePresentationSegment",
    "SourceGrant",
    "TypedOutboundRequest",
    "UntrustedProvenanceSegment",
    "ValidatedToolSyntaxSegment",
    "classify_destination",
    "redact_remote_unsafe_text",
    "source_grant_digest",
    "static_literal_sha256",
    "validate_tool_syntax",
]
