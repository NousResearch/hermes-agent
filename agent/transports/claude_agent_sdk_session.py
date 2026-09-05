"""Session adapter for the claude-agent-sdk runtime.

Owns one Claude Agent SDK client per Hermes session — the structural twin of
``codex_app_server_session.py``, with the Codex JSON-RPC subprocess replaced
by Anthropic's official ``claude-agent-sdk`` (which manages the Claude Code
CLI subprocess, its agent loop, and — critically — **subscription OAuth**:
``CLAUDE_CODE_OAUTH_TOKEN`` / the ``~/.claude`` credential store by default;
known metered sources and Extra Usage fail closed unless the operator opts in).
See GitHub issue #25267.

Lifecycle:
    session = ClaudeAgentSdkSession(cwd="/home/x/proj", model="claude-opus-4-8")
    session.ensure_started()                       # loop thread + SDK connect
    result = session.run_turn(user_input="hello")  # blocks until ResultMessage
    session.close()                                # disconnect + stop loop

Threading model: the SDK is async-first, but AIAgent.run_conversation() is
synchronous (the same constraint that made CodexAppServerClient thread-based).
The adapter owns a dedicated background thread running one asyncio event loop
for the whole session lifetime; every SDK coroutine is marshaled onto it with
``asyncio.run_coroutine_threadsafe`` and awaited with a timeout, so the SDK
client keeps stable loop affinity and ``run_turn`` stays blocking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sys
import threading
import time
import traceback
import unicodedata
from typing import Any, Callable, Optional

# TurnResult is the shared contract with the runtime glue — reused verbatim
# from the codex session module (same fields, same semantics) so
# ``run_claude_agent_sdk_turn`` mirrors ``run_codex_app_server_turn`` 1:1.
from agent.redact import redact_sensitive_text
from agent.transports.codex_app_server_session import TurnResult
from agent.transports.claude_sdk_event_projector import ClaudeSdkEventProjector

logger = logging.getLogger(__name__)


# SDK permission payloads are hostile Python objects.  Nothing derived from
# them may reach an auto-allow, callback, log, observer, or card until the
# complete request has crossed this exact-builtin, bounded JSON boundary.
_SDK_CANONICAL_MAX_DEPTH = 64
_SDK_CANONICAL_MAX_NODES = 10_000
_SDK_CANONICAL_MAX_UTF8_BYTES = 64 * 1024
_SDK_CANONICAL_MAX_INPUT_UTF8_BYTES = 64 * 1024
_SDK_TOOL_USE_ID_MAX_UTF8_BYTES = 256
_SDK_PRESENTATION_MAX_UTF8_BYTES = 512
_SDK_CALLBACK_CHOICE_MAX_UTF8_BYTES = 16
_SDK_CALLBACK_REASON_MAX_UTF8_BYTES = 512

_SDK_PATH_TOOL_FIELDS = {
    "Read": ("file_path", "path"),
    "Write": ("file_path", "path"),
    "Edit": ("file_path", "path"),
    "MultiEdit": ("file_path", "path"),
    "Glob": ("path",),
    "Grep": ("path",),
    "NotebookEdit": ("notebook_path", "file_path", "path"),
}
_SDK_FIXED_LOG_TOOL_IDENTITIES = frozenset({"Bash", *_SDK_PATH_TOOL_FIELDS})


def _control_sanitized_text(value: str) -> str:
    safe = "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in value
    )
    return " ".join(safe.split())


def _bounded_control_sanitized_text(value: str, max_utf8_bytes: int) -> str:
    """Return one bounded display line from an already validated string."""
    safe = _control_sanitized_text(value)
    output: list[str] = []
    used = 0
    for char in safe:
        encoded_size = len(char.encode("utf-8"))
        if used + encoded_size > max_utf8_bytes:
            break
        output.append(char)
        used += encoded_size
    return "".join(output)


def _bounded_control_sanitized_head_tail(value: str, max_utf8_bytes: int) -> str:
    """Bound one line while disclosing both ends and explicit truncation."""
    safe = _control_sanitized_text(value)
    if len(safe.encode("utf-8")) <= max_utf8_bytes:
        return safe
    marker = " … [truncated] … "
    marker_bytes = len(marker.encode("utf-8"))
    if marker_bytes > max_utf8_bytes:
        return _bounded_control_sanitized_text(marker, max_utf8_bytes)
    remaining = max_utf8_bytes - marker_bytes
    head_budget = remaining // 2
    tail_budget = remaining - head_budget
    head = _bounded_control_sanitized_text(safe, head_budget)
    tail_chars: list[str] = []
    used = 0
    for char in reversed(safe):
        encoded_size = len(char.encode("utf-8"))
        if used + encoded_size > tail_budget:
            break
        tail_chars.append(char)
        used += encoded_size
    return f"{head}{marker}{''.join(reversed(tail_chars))}"


def _safe_sdk_tool_identity(value: str) -> str:
    """Preserve ordinary identity, but hide embedded credential patterns."""
    bounded = _bounded_control_sanitized_text(value, 96)
    if not bounded:
        return "unknown"
    redacted = redact_sensitive_text(bounded, force=True)
    if redacted != bounded:
        return "unknown"
    # Generic redaction intentionally requires token boundaries. Unknown SDK
    # tool identities may attach a credential prefix directly to punctuation
    # (for example ``Odd-sk-...``), so expose every bounded suffix behind a
    # boundary in one composite probe. Do not attempt to reconstruct a partly
    # secret identity: any match collapses to one fixed, non-sensitive label.
    suffix_probe = "\n".join(bounded[index:] for index in range(len(bounded)))
    if redact_sensitive_text(suffix_probe, force=True) != suffix_probe:
        return "unknown"
    return bounded


_INVALID_SDK_JSON = object()


def _sdk_json_string_sizes(
    value: str,
    input_budget: int = _SDK_CANONICAL_MAX_INPUT_UTF8_BYTES,
) -> tuple[int, int] | None:
    """Count input/token UTF-8 incrementally and stop at the first cap."""
    input_size = 0
    token_size = 2  # surrounding quotes
    try:
        for char in value:
            encoded_size = len(char.encode("utf-8"))
            input_size += encoded_size
            if input_size > input_budget:
                return None
            codepoint = ord(char)
            if char in ('"', "\\") or char in "\b\f\n\r\t":
                token_size += 2
            elif codepoint < 0x20:
                token_size += 6
            else:
                token_size += encoded_size
            if token_size > _SDK_CANONICAL_MAX_UTF8_BYTES:
                return None
    except UnicodeError:
        return None
    return input_size, token_size

def _sdk_json_string_token_utf8_size(value: str) -> int | None:
    """Return the canonical token size without materializing encoded copies."""
    sizes = _sdk_json_string_sizes(value)
    return None if sizes is None else sizes[1]


def _sdk_json_int_token_fits(value: int) -> bool:
    """Conservatively bound decimal conversion without converting the integer."""
    # 30103 / 100000 is a strict upper approximation of log10(2).
    digits_upper_bound = (value.bit_length() * 30_103) // 100_000 + 1
    token_bytes = digits_upper_bound + (1 if value < 0 else 0)
    return token_bytes <= _SDK_CANONICAL_MAX_UTF8_BYTES


def _freeze_bounded_plain_sdk_json(value: object) -> object:
    """Validate and detach one bounded exact-builtin JSON graph in one pass."""
    active: set[int] = set()
    nodes = 0
    input_utf8_bytes = 0

    def freeze(current: object, depth: int) -> object:
        nonlocal nodes, input_utf8_bytes
        nodes += 1
        if nodes > _SDK_CANONICAL_MAX_NODES or depth > _SDK_CANONICAL_MAX_DEPTH:
            return _INVALID_SDK_JSON

        kind = type(current)
        if kind is str:
            remaining = _SDK_CANONICAL_MAX_INPUT_UTF8_BYTES - input_utf8_bytes
            sizes = _sdk_json_string_sizes(current, remaining)
            if sizes is None:
                return _INVALID_SDK_JSON
            input_utf8_bytes += sizes[0]
            return current
        if current is None or kind is bool:
            return current
        if kind is int:
            return current if _sdk_json_int_token_fits(current) else _INVALID_SDK_JSON
        if kind is float:
            return current if math.isfinite(current) else _INVALID_SDK_JSON
        if kind not in (list, dict):
            return _INVALID_SDK_JSON

        identity = id(current)
        if identity in active:
            return _INVALID_SDK_JSON
        active.add(identity)
        try:
            if kind is list:
                frozen_list = []
                for child in current:
                    frozen_child = freeze(child, depth + 1)
                    if frozen_child is _INVALID_SDK_JSON:
                        return _INVALID_SDK_JSON
                    frozen_list.append(frozen_child)
                return frozen_list

            frozen_dict = {}
            for key, child in current.items():
                if type(key) is not str:
                    return _INVALID_SDK_JSON
                frozen_key = freeze(key, depth + 1)
                if frozen_key is _INVALID_SDK_JSON:
                    return _INVALID_SDK_JSON
                frozen_child = freeze(child, depth + 1)
                if frozen_child is _INVALID_SDK_JSON:
                    return _INVALID_SDK_JSON
                frozen_dict[frozen_key] = frozen_child
            return frozen_dict
        except (RuntimeError, ValueError, TypeError):
            return _INVALID_SDK_JSON
        finally:
            active.discard(identity)

    return freeze(value, 0)


def _is_bounded_plain_sdk_json(value: object) -> bool:
    return _freeze_bounded_plain_sdk_json(value) is not _INVALID_SDK_JSON


def _bounded_canonical_sdk_json(value: object) -> str | None:
    """Encode deterministic JSON incrementally and stop at the byte cap."""
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    chunks: list[str] = []
    used = 0
    try:
        for chunk in encoder.iterencode(value):
            encoded_size = len(chunk.encode("utf-8"))
            if used + encoded_size > _SDK_CANONICAL_MAX_UTF8_BYTES:
                return None
            chunks.append(chunk)
            used += encoded_size
    except (
        TypeError, ValueError, RuntimeError, RecursionError, OverflowError,
        UnicodeError,
    ):
        return None
    return "".join(chunks)


def _bounded_utf8_size(value: str, max_bytes: int) -> int | None:
    """Count UTF-8 bytes incrementally, stopping when the cap is crossed."""
    if len(value) > max_bytes:
        return None
    used = 0
    for start in range(0, len(value), 4096):
        used += len(value[start:start + 4096].encode("utf-8"))
        if used > max_bytes:
            return None
    return used

def validate_canonical_sdk_request_serialization(
    value: object,
) -> tuple[str, dict] | None:
    """Validate exact deterministic JSON for one bounded SDK tool request."""
    if type(value) is not str:
        return None
    try:
        if _bounded_utf8_size(value, _SDK_CANONICAL_MAX_UTF8_BYTES) is None:
            return None
        request = json.loads(value)
    except (TypeError, ValueError, RecursionError, OverflowError, UnicodeError):
        return None
    frozen = _freeze_bounded_plain_sdk_json(request)
    if (
        frozen is _INVALID_SDK_JSON
        or type(frozen) is not dict
        or set(frozen) != {"tool_name", "tool_input"}
        or type(frozen["tool_name"]) is not str
        or type(frozen["tool_input"]) is not dict
        or (
            frozen["tool_name"] == "Bash"
            and type(frozen["tool_input"].get("command")) is not str
        )
    ):
        return None
    canonical = _bounded_canonical_sdk_json(frozen)
    if canonical is None or value != canonical:
        return None
    return canonical, frozen


def _canonical_sdk_tool_request(tool_name: object, tool_input: object) -> str | None:
    if type(tool_name) is not str or type(tool_input) is not dict:
        return None
    try:
        frozen = _freeze_bounded_plain_sdk_json({
            "tool_name": tool_name,
            "tool_input": tool_input,
        })
        if (
            frozen is _INVALID_SDK_JSON
            or type(frozen) is not dict
            or type(frozen.get("tool_input")) is not dict
            or (
                tool_name == "Bash"
                and type(frozen["tool_input"].get("command")) is not str
            )
        ):
            return None
        canonical = _bounded_canonical_sdk_json(frozen)
    except (
        TypeError, ValueError, RuntimeError, RecursionError, OverflowError,
        UnicodeError,
    ):
        return None
    if canonical is None:
        return None
    checked = validate_canonical_sdk_request_serialization(canonical)
    return checked[0] if checked is not None else None


def _is_bounded_sdk_callback_string(
    value: object,
    max_utf8_bytes: int,
    *,
    allow_space: bool,
) -> bool:
    """Validate callback text incrementally without encoding a hostile copy."""
    if type(value) is not str:
        return False
    used = 0
    for char in value:
        codepoint = ord(char)
        if unicodedata.category(char).startswith("C"):
            return False
        if char.isspace() and (char != " " or not allow_space):
            return False
        if codepoint <= 0x7F:
            used += 1
        elif codepoint <= 0x7FF:
            used += 2
        elif codepoint <= 0xFFFF:
            used += 3
        else:
            used += 4
        if used > max_utf8_bytes:
            return False
    return True


def _safe_sdk_tool_use_id(context: object) -> str:
    try:
        value = getattr(context, "tool_use_id", "")
    except Exception:
        return ""
    if type(value) is not str or not value:
        return ""
    utf8_bytes = 0
    for char in value:
        code_point = ord(char)
        if code_point <= 0x7F:
            utf8_bytes += 1
        elif code_point <= 0x7FF:
            utf8_bytes += 2
        elif 0xD800 <= code_point <= 0xDFFF:
            return ""
        elif code_point <= 0xFFFF:
            utf8_bytes += 3
        else:
            utf8_bytes += 4
        if utf8_bytes > _SDK_TOOL_USE_ID_MAX_UTF8_BYTES:
            return ""
        if not char.isprintable() or char.isspace():
            return ""
    return value


_UNVALIDATED_SDK_REQUEST = object()


def safe_sdk_tool_presentation_from_canonical(
    canonical_request: object,
    *,
    _validated_request: object = _UNVALIDATED_SDK_REQUEST,
) -> tuple[str, str] | None:
    """Build a bounded actionable card summary from validated request fields."""
    checked = _validated_request
    if checked is _UNVALIDATED_SDK_REQUEST:
        checked = validate_canonical_sdk_request_serialization(canonical_request)
    if checked is None:
        return None
    request = checked[1]
    tool_name = request["tool_name"]
    tool_input = request["tool_input"]
    safe_name = _safe_sdk_tool_identity(tool_name)
    if tool_name == "Bash":
        command = redact_sensitive_text(tool_input["command"], force=True)
        preview = _bounded_control_sanitized_head_tail(command, 480)
        return (
            _bounded_control_sanitized_text(
                f"Bash(command={preview})", _SDK_PRESENTATION_MAX_UTF8_BYTES,
            ),
            "Claude requests SDK tool Bash",
        )
    path_fields = _SDK_PATH_TOOL_FIELDS.get(tool_name)
    if path_fields is not None:
        path = next(
            (tool_input[field] for field in path_fields if type(tool_input.get(field)) is str),
            "(unspecified)",
        )
        safe_path = _bounded_control_sanitized_head_tail(
            redact_sensitive_text(path, force=True), 460,
        )
        return (
            _bounded_control_sanitized_text(
                f"{safe_name}(path={safe_path})", _SDK_PRESENTATION_MAX_UTF8_BYTES,
            ),
            f"Claude requests SDK tool {safe_name}",
        )
    return f"SDK tool {safe_name}", "Claude requests an SDK tool"


_SAFE_SDK_DENY_LOG_REASONS = frozenset({
    "approval callback failed",
    "approval timed out — no operator response",
    "approval expired (turn ended)",
    "no approver available (background context)",
    "approval request could not be delivered to the operator (notify failed)",
})


def _safe_sdk_deny_log_reason(reason: object) -> str:
    if type(reason) is str and reason in _SAFE_SDK_DENY_LOG_REASONS:
        return reason
    return "non-operator denial"


# HERMES_TERMINAL_SECURITY_MODE → SDK permission_mode, read at session
# construction (default "auto"). Precedence: explicit constructor arg, then
# the agent.claude_agent_sdk.permission_mode config key (an SDK mode literal
# — see _configured_permission_mode), then this env mapping.
#
# SDK default posture is intentionally stricter than generic terminal `auto`:
# `default` preserves Hermes' per-tool approval bridge. Mapping `auto` to
# `acceptEdits` skips that bridge entirely, so even the fixed bounded MCP read
# surface is denied/unguarded depending on CLI state. Hermes YOLO requests are
# represented internally as ``bypassPermissions`` but normalized back to
# ``default`` before SDK option construction; the audited callback performs the
# bypass only after immutable floors.
_HERMES_TO_SDK_PERMISSION_MODE = {
    "auto": "default",
    "approval-required": "default",
    "unrestricted": "bypassPermissions",
    "yolo": "bypassPermissions",
}

# The SDK's own permission_mode literals (verified against the installed
# claude-agent-sdk 0.2.120 ClaudeAgentOptions type).
_SDK_PERMISSION_MODES = (
    "default",
    "acceptEdits",
    "plan",
    "bypassPermissions",
    "dontAsk",
    "auto",
)


def _configured_sdk_env() -> dict:
    """agent.claude_agent_sdk.env — extra environment for the CLI subprocess.

    The Claude Code CLI reads operational knobs from its environment that the
    SDK exposes no typed option for. Measured on 0.2.120: only
    ``CLAUDE_CODE_AUTO_COMPACT_WINDOW`` moves the context ceiling and the
    autocompact threshold (300000 -> maxTokens 300000, threshold 267000);
    ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` and ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE``
    are inert. That ratio is exactly why this is a generic config surface and
    not a named option per knob — the knobs are undocumented and shift.

    Values are stringified; a non-mapping or unreadable config yields {} so a
    bad edit cannot strip the scrubbed env that ships alongside it.
    """
    raw = _provider_config().get("env")
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for key, value in raw.items():
        if value is None:
            continue
        try:
            out[str(key)] = str(value)
        except Exception:
            logger.warning(
                "agent.claude_agent_sdk.env[%r] is not stringifiable — ignoring", key
            )
    return out


def _sdk_env_overrides(
    *, metered_allowed: Optional[bool] = None
) -> dict[str, str]:
    """The full env override set handed to the spawned CLI.

    Metered-vector scrub first (see _METERED_ENV_DENYLIST).
    agent.claude_agent_sdk.allow_metered_key: true is the operator's explicit
    "bill me metered" opt-in (the same flag the startup guard honors), so it
    disables the scrub too — otherwise the documented escape hatch would hand
    the CLI an environment with the key blanked.

    Operator-configured env is applied last so deliberate knobs win over
    defaults, but it must NOT win over the scrub: a plain update() would let
    ``env: {ANTHROPIC_API_KEY: ...}`` overwrite the scrub's "" and silently
    re-arm metered billing behind allow_metered_key: false. Denylisted keys
    are therefore dropped (loudly) unless the metered opt-in is set.
    """
    if metered_allowed is None:
        metered_allowed = _provider_flag("allow_metered_key")
    overrides: dict[str, str] = {} if metered_allowed else _scrubbed_sdk_env()
    for key, value in _configured_sdk_env().items():
        if not metered_allowed and _is_metered_sdk_env_value(key, value):
            logger.warning(
                "agent.claude_agent_sdk.env[%s] is a metered billing vector — "
                "ignoring (set allow_metered_key: true to permit it)",
                key,
            )
            continue
        overrides[key] = value
    return overrides


def _configured_permission_mode() -> Optional[str]:
    """agent.claude_agent_sdk.permission_mode from config.yaml, validated.

    Takes a validated SDK permission-mode intent (one of
    _SDK_PERMISSION_MODES — note "auto" here is the SDK's own mode, NOT the
    HERMES_TERMINAL_SECURITY_MODE value of the same name). The literal
    ``bypassPermissions`` intent is emitted as callback-capable ``default`` and
    emulated inside Hermes after immutable approval floors. Empty/absent —
    the default — keeps current behavior: the HERMES_TERMINAL_SECURITY_MODE
    mapping stands, so existing deployments harden without env archaeology
    only when they opt in. Unknown values are ignored with a warning:
    permissions must never silently loosen (or tighten into an unusable
    mode) on a typo."""
    raw = str(_provider_config().get("permission_mode") or "").strip()
    if not raw:
        return None
    if raw not in _SDK_PERMISSION_MODES:
        logger.warning(
            "agent.claude_agent_sdk.permission_mode=%r is not a valid SDK "
            "permission mode (one of %s) — ignoring it; the "
            "HERMES_TERMINAL_SECURITY_MODE mapping stands.",
            raw,
            ", ".join(_SDK_PERMISSION_MODES),
        )
        return None
    return raw


# The SDK's own setting-source literals (verified against the installed
# claude-agent-sdk 0.2.120 SettingSource type).
_SDK_SETTING_SOURCES = ("user", "project", "local")

# Exact names only: these are the SDK profile's two bounded readers. Never
# widen this to an MCP/server wildcard or an unexposed mutation identity.
_SDK_AUTO_ALLOWED_MCP_TOOLS = frozenset({
    "mcp__hermes-tools__read_file",
    "mcp__hermes-tools__search_files",
})


def _configured_setting_sources() -> list:
    """agent.claude_agent_sdk.setting_sources from config.yaml, validated.

    Default (absent/empty) is FULL ISOLATION — the SDK loads no filesystem
    settings, so ambient ``~/.claude`` or project files cannot re-permission
    tools or install hooks underneath the configured posture. Deployments
    whose operating model deliberately stores tool grants in the operator's
    own ``~/.claude/settings.json`` — an unattended box whose cron turns
    must pre-approve WebSearch/MCP tools with no human to answer a prompt —
    opt back in explicitly (``setting_sources: ["user"]``). Unknown entries
    are dropped with a warning: a typo must neither silently widen isolation
    nor quietly load an unintended source."""
    raw = _provider_config().get("setting_sources")
    if not isinstance(raw, (list, tuple)):
        return []
    sources: list = []
    for entry in raw:
        name = str(entry or "").strip()
        if name in _SDK_SETTING_SOURCES:
            if name not in sources:
                sources.append(name)
        elif name:
            logger.warning(
                "agent.claude_agent_sdk.setting_sources entry %r is not a "
                "valid SDK setting source (one of %s) — dropping it.",
                name,
                ", ".join(_SDK_SETTING_SOURCES),
            )
    return sources


# ---------- stdout framing ----------
# The SDK reads the CLI's NDJSON stdout through a line framer and kills the
# whole message reader when one message exceeds max_buffer_size — a FATAL
# CLIJSONDecodeError, not a skipped message, so the turn dies mid-flight and
# the session is retired (see the retire matrix in claude_sdk_runtime). The
# SDK's own default is 1 MiB (subprocess_cli._DEFAULT_MAX_BUFFER_SIZE), which
# a single large tool result clears easily: production forensics on
# 2026-08-17 21:03 EDT caught a turn killed this way right after an Edit,
# with the CLI transcript's largest persisted line at 347 KB — the oversized
# message never reached disk.
#
# Upstream treats the option, not the default, as the fix: PR #416 proposed
# raising the default to 10 MiB and was withdrawn ("the existing
# max_buffer_size parameter already provides the needed functionality"), and
# issue #98 was closed by adding the option. So Hermes sets it explicitly
# rather than carrying an SDK patch. 10 MiB matches the figure that
# discussion converged on.
#
# NOTE: the SDK measures this with len() on a str, so the unit is Unicode
# CODE POINTS despite the "bytes" wording in its error text (upstream issue
# #1165, open). Worst-case real memory for a multibyte-heavy message is
# therefore ~4x this number — still bounded, and the point of the limit is to
# stop an unterminated line growing without end, not to be exact.
_DEFAULT_MAX_BUFFER_SIZE = 10 * 1024 * 1024


def _configured_max_buffer_size() -> int:
    """agent.claude_agent_sdk.max_buffer_size, validated.

    Same warn-and-fall-back contract as the timeout validators: bools are
    rejected before int() (YAML `true` must not become 1), and non-numeric,
    zero or negative values warn and yield the built-in default. There is
    deliberately no `0 = unlimited`: the limit is the only backstop against a
    CLI that never terminates a line, and removing it trades a killed turn
    for an OOM on a memory-constrained host."""
    raw = _provider_config().get("max_buffer_size")
    if raw is None:
        return _DEFAULT_MAX_BUFFER_SIZE
    if isinstance(raw, bool):
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is a boolean, not a "
            "size — ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is not a number — "
            "ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    if isinstance(raw, float) and not raw.is_integer():
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is not a whole number "
            "— ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    if value <= 0:
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is out of range — "
            "ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    return value


# ---------- turn-lifetime defaults ----------
# The soft turn budget. It was a hard wall-clock over the whole turn since the
# provider's birth (transplanted verbatim from the codex twin, where a
# post-tool quiet watchdog compensates); production forensics on six 600s
# kills showed four were actively-working turns (tool loops, human approval
# taps) — so the budget is now evaluated together with activity evidence
# (see _TurnWatch) instead of alone.
_DEFAULT_TURN_TIMEOUT = 600.0
# Post-tool quiet watchdog default WHEN streaming is on (codex parity: its
# post_tool_quiet_timeout=90 mirrors openclaw's #81697 watchdog). With
# streaming OFF there is no liveness signal between a tool result and the
# next complete AssistantMessage — thinking is indistinguishable from wedged
# — so the watchdog defaults to DISABLED there (operator opt-in).
_DEFAULT_POST_TOOL_QUIET_STREAMING = 90.0
# The budget rule only fires when the turn has ALSO been quiet this long
# (capped at the budget itself so tiny test budgets keep tripping at the
# budget): an actively-producing turn is never killed mid-sentence.
_BUDGET_GRACE_IDLE = 30.0
# After a watchdog trip we interrupt the CLI and give _consume_turn this long
# to unwind on the interrupt-ack ResultMessage — a clean unwind preserves the
# partial transcript and the resumable session id; only expiry hard-cancels.
_TURN_ABORT_GRACE = 15.0
# An inter-poll gap this many times the poll interval means the PROCESS was
# stalled (swap/OOM descheduling on a memory-constrained host), not the turn
# — re-baseline instead of tripping on time nobody was actually waiting.
_POLL_STALL_FACTOR = 5.0


def _configured_timeout_seconds(key: str, *, allow_zero: bool) -> Optional[float]:
    """Numeric seconds from `agent.claude_agent_sdk.<key>`, validated.

    Same warn-and-fall-back contract as _configured_max_budget_usd: bools are
    rejected before float() (YAML `true` must not become 1.0), non-numeric and
    negative values warn and yield None (= use the built-in default). `0` is
    key-specific: allowed only where "disabled" is a documented meaning."""
    raw = _provider_config().get(key)
    if raw is None:
        return None
    if isinstance(raw, bool):
        logger.warning(
            "agent.claude_agent_sdk.%s=%r is a boolean, not seconds — "
            "ignoring it (using the built-in default).", key, raw,
        )
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "agent.claude_agent_sdk.%s=%r is not a number of seconds — "
            "ignoring it (using the built-in default).", key, raw,
        )
        return None
    if value < 0 or (value == 0 and not allow_zero):
        logger.warning(
            "agent.claude_agent_sdk.%s=%r is out of range — ignoring it "
            "(using the built-in default).", key, raw,
        )
        return None
    return value


def _configured_turn_timeout() -> Optional[float]:
    """agent.claude_agent_sdk.turn_timeout (seconds). Positive only — there
    is deliberately no `0 = unlimited`: the gateway's inactivity monitor
    (`agent.gateway_timeout`, 1800s) is the outer ceiling for SDK turns and
    an unlimited soft budget under it would leave the quiet watchdog as the
    only transport-level bound."""
    return _configured_timeout_seconds("turn_timeout", allow_zero=False)


def _configured_post_tool_quiet_timeout() -> Optional[float]:
    """agent.claude_agent_sdk.post_tool_quiet_timeout (seconds).
    `0` = explicitly disabled. Absent/None = streaming-dependent default
    (90s with streaming on, disabled with streaming off)."""
    return _configured_timeout_seconds("post_tool_quiet_timeout", allow_zero=True)


# Upper bound on how long a compaction may suspend the watchdog. compact_boundary
# is NOT guaranteed -- measured 2026-08-16, a compaction that started at 03:57:17
# never produced one -- so an unbounded gate would trade a killed turn for a hung
# one. 600s is an order of magnitude above the ~90-125s compactions observed in
# production while still landing well inside the gateway's 1800s ceiling.
_COMPACTION_MAX_SUSPEND = 600.0


class _TurnWatch:
    """Activity evidence for one in-flight turn.

    Threading contract: mutating calls happen on the session's loop thread
    (message drain, projections, approval bridge) with ONE sanctioned
    exception — rebaseline(), called from the run_turn poll thread, also
    writes last_activity. That dual-writer race is benign by construction:
    float stores are GIL-atomic (never torn), and a lost update leaves
    last_activity merely STALE, which the stall detector re-baselines and
    the two-poll debounce absorbs before any verdict — trips can only be
    DELAYED by it, never wrongly fired. Everything else is single-writer;
    the poll thread otherwise only READS. No lock, deliberately: a lock
    shared with the loop thread would risk stalling the SDK stream drain.

    Evidence gate: a turn with tool calls outstanding (issued ToolUseBlocks
    minus resolved ToolResultBlocks — server tools never enter the count,
    they resolve inside their own assistant message) or an approval prompt
    awaiting a human tap is PROVABLY working/waiting and is never tripped.
    If the CLI never resolves an issued tool id (interrupted mid-tool), the
    suspension persists and the gateway's 1800s inactivity ceiling remains
    the backstop — documented, deliberate."""

    def __init__(self) -> None:
        now = time.monotonic()
        self.started = now
        self.last_activity = now
        self.post_tool_armed = False
        self.outstanding_tools = 0
        self.approvals_pending = 0
        self.compaction_active = 0
        self.compaction_started = 0.0

    # -- loop-thread writers --

    def tick(self) -> None:
        self.last_activity = time.monotonic()

    def note_tools_issued(self, count: int) -> None:
        if count > 0:
            self.outstanding_tools += count

    def note_tools_resolved(self, count: int) -> None:
        if count > 0:
            self.outstanding_tools = max(0, self.outstanding_tools - count)

    def arm_post_tool(self) -> None:
        self.post_tool_armed = True

    def disarm_post_tool(self) -> None:
        self.post_tool_armed = False

    def approval_begin(self) -> None:
        self.approvals_pending += 1
        self.tick()

    def approval_end(self) -> None:
        self.approvals_pending = max(0, self.approvals_pending - 1)
        self.tick()

    def compaction_begin(self) -> None:
        """PreCompact fired: the CLI is about to go silent, legitimately.

        Earliest start wins -- a re-entrant PreCompact must not restart the
        bounding clock, or a pathological loop could extend the suspension
        indefinitely, which is exactly what the bound exists to prevent.
        """
        if self.compaction_active == 0:
            self.compaction_started = time.monotonic()
        self.compaction_active += 1
        self.tick()

    def compaction_end(self) -> None:
        """compact_boundary arrived: resume normal watchdog rules.

        tick() is load-bearing, not hygiene. Without it a turn that compacted
        for 91s would resume already 91s idle and trip on the very next poll --
        the same kill, one poll later.
        """
        self.compaction_active = max(0, self.compaction_active - 1)
        self.tick()

    # -- caller-thread readers --

    def rebaseline(self) -> None:
        """After a detected process stall: the elapsed gap was spent
        descheduled, not waiting — restamp so neither rule fires on it."""
        self.last_activity = time.monotonic()

    def check(self, *, budget: float, quiet: float) -> Optional[str]:
        """Returns None (keep waiting), "post_tool_quiet", or "budget"."""
        if self.outstanding_tools > 0 or self.approvals_pending > 0:
            return None
        now = time.monotonic()
        # A compacting CLI is indistinguishable from a wedged one: between
        # PreCompact and compact_boundary it emits nothing at all. Without this
        # gate the post_tool_quiet rule reads that silence as a wedge and
        # interrupts the CLI mid-compaction, so the terminal ResultMessage never
        # arrives -- surfacing next turn as "discarding N stale unsolicited
        # text(s)" and, to the user, as a turn that simply died. Bounded, so a
        # boundary that never arrives cannot hang the turn instead.
        if (
            self.compaction_active > 0
            and (now - self.compaction_started) < _COMPACTION_MAX_SUSPEND
        ):
            return None
        idle = now - self.last_activity
        if quiet > 0 and self.post_tool_armed and idle >= quiet:
            return "post_tool_quiet"
        elapsed = now - self.started
        if elapsed >= budget and idle >= min(_BUDGET_GRACE_IDLE, budget):
            return "budget"
        return None


def _swallow_interrupt_result(future: Any) -> None:
    """Done-callback for the fire-and-forget client.interrupt() future: the
    SDK's control request times out after 60s on a wedged CLI and an
    unretrieved exception would log 'Future exception was never retrieved'
    at teardown — retrieve and demote it."""
    try:
        future.result()
    except Exception as exc:
        logger.debug(
            "SDK interrupt control request failed: %s",
            _safe_sdk_error_text(exc),
        )


def _http_mcp_entries_from_config() -> dict[str, dict]:
    """Return explicitly opted-in, credential-free HTTP MCPs for the SDK.

    Direct registration shares the ``hybrid_mcp_bridge`` opt-in and exclusion
    list. Header-bearing, templated, userinfo-bearing, and credential-like
    query or fragment URLs are refused because the SDK serializes this config
    into the Claude CLI's ``--mcp-config`` process argument. Returns ``{}`` on
    any read/parse failure and never logs a URL or credential value.
    """
    import os as _os
    import re as _re
    from urllib.parse import unquote as _unquote, urlsplit as _urlsplit

    try:
        import yaml as _yaml  # type: ignore
    except Exception:
        return {}

    home = _os.environ.get("HERMES_HOME") or _os.path.expanduser("~/.hermes")
    profile = _os.environ.get("HERMES_PROFILE") or "default"
    paths = [
        _os.path.join(home, "config.yaml"),
        _os.path.join(home, "profiles", profile, "config.yaml"),
    ]

    merged: dict[str, Any] = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = _yaml.safe_load(fh) or {}
            block = (data.get("mcp_servers") or {}) if isinstance(data, dict) else {}
            if isinstance(block, dict):
                merged.update(block)
        except FileNotFoundError:
            continue
        except Exception:
            logger.debug("failed to read %s for HTTP MCP discovery", p, exc_info=True)
            continue

    if not _provider_flag("hybrid_mcp_bridge", default=False):
        return {}

    _env_re = _re.compile(r"\$\{[^}]*\}")
    _secret_field_re = _re.compile(
        r"(?:api[_-]?key|token|secret|password|credential|authorization|auth)",
        _re.I,
    )
    excluded_names = set(_configured_hybrid_exclude())

    out: dict[str, dict] = {}
    for raw_name, cfg in merged.items():
        if not isinstance(cfg, dict):
            continue
        url = cfg.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        name = str(raw_name)
        url = url.strip()
        if name in excluded_names:
            logger.info("claude-agent-sdk: HTTP MCP %r excluded by config", name)
            continue
        if cfg.get("headers"):
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because it has headers",
                name,
            )
            continue
        if _env_re.search(url):
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL is templated",
                name,
            )
            continue
        try:
            parsed = _urlsplit(url)
        except ValueError:
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL is malformed",
                name,
            )
            continue
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL is not HTTP(S)",
                name,
            )
            continue
        # Direct HTTP MCP configuration is passed to the SDK/CLI process.
        # Treat every query, fragment, or userinfo component as secret-bearing:
        # a denylist cannot safely distinguish signed/public query parameters
        # from encoded credentials (including nested percent encoding).
        if parsed.username is not None or parsed.query or parsed.fragment:
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL has userinfo, query, or fragment",
                name,
            )
            continue
        out[name] = {"type": "http", "url": url}
    return out


def _swallow_steer_result(future: Any) -> None:
    """Same contract as _swallow_interrupt_result, for the fire-and-forget
    steer query(). The caller has already returned True by the time this
    resolves, so a failure here can only be logged, not surfaced."""
    try:
        future.result()
    except Exception:
        logger.debug("SDK steer query failed after scheduling", exc_info=True)


# Substrings in SDK/CLI errors that signal broken subscription credentials.
# Conservative on purpose — mirrors codex's _OAUTH_REFRESH_FAILURE_HINTS
# contract: every needle is a phrase, never a bare token. Bare "401" matched
# tool ids and byte offsets; bare "credentials" matched an MCP server
# complaining about its OWN files — and a hit retires the session, so a
# false positive is a wrong shutdown.
_AUTH_FAILURE_HINTS = (
    "not logged in",
    "please run /login",
    "invalid api key",
    "authentication_error",
    "oauth token",
    "token has expired",
    "expired token",
    "invalid bearer token",
    "setup-token",
)

_AUTH_401_UNAUTHORIZED_RE = re.compile(r"\b401\W{0,3}unauthorized\b")
_AUTH_UNAUTHORIZED_HTTP_401_RE = re.compile(
    r"\bunauthorized\W{0,15}\(?http\s*401\b"
)


def _safe_sdk_error_text(value: Any) -> str:
    """Redact provider/SDK diagnostics before they cross a transport boundary."""
    return redact_sensitive_text(str(value or ""), force=True)


def _safe_sdk_traceback(exc: BaseException) -> str:
    """Format traceback frames without re-emitting the exception payload.

    ``logger(..., exc_info=True)`` is useful for startup diagnosis, but it also
    appends ``str(exc)`` verbatim after the frames. SDK/import errors can carry
    credentials, so retain the actionable call stack while routing the error
    text itself through the normal forced-redaction boundary.
    """
    try:
        frames = traceback.extract_tb(exc.__traceback__)
        return "".join(traceback.format_list(frames)).rstrip()
    except Exception:
        return "<traceback unavailable>"


def classify_auth_failure(
    *parts: str, mcp_attributed: bool = False
) -> Optional[str]:
    """Return a user-friendly re-auth hint if the strings look like a Claude
    subscription auth failure; otherwise None. The hint keeps the underlying
    error text: a hit retires the session, so the evidence must survive the
    redirect (codex surfaces the original error the same way)."""
    haystack = " ".join(p for p in parts if p).lower()
    if not haystack:
        return None
    if mcp_attributed or "mcp__" in haystack or "mcp server" in haystack:
        return None
    matched_401 = (
        _AUTH_401_UNAUTHORIZED_RE.search(haystack)
        or _AUTH_UNAUTHORIZED_HTTP_401_RE.search(haystack)
    )
    for needle in (None, *_AUTH_FAILURE_HINTS):
        if (needle is None and matched_401) or (
            needle is not None and needle in haystack
        ):
            original = _safe_sdk_error_text(
                next((p.strip() for p in parts if p and p.strip()), "")
            )
            if len(original) > 400:
                original = original[:400] + "…"
            return (
                "Claude authentication failed — the subscription OAuth token "
                "looks expired or invalid. Refresh it with `claude setup-token` "
                "(or `claude login` on this machine) and update "
                "CLAUDE_CODE_OAUTH_TOKEN, then retry. "
                f"(underlying error: {original})"
            )
    return None


def check_claude_sdk_available() -> tuple[bool, str]:
    """Preflight: the optional SDK extra must be importable, and it bundles /
    locates the Claude Code CLI itself. Mirrors check_codex_binary()."""
    # Fast path FIRST: when the SDK already imports, never enter the lazy
    # installer. ensure() can shell out to `uv pip install` and calls
    # importlib.invalidate_caches(); doing either immediately before importing
    # claude_agent_sdk -> mcp -> anyio rewrites site-packages and drops import
    # caches under a live interpreter, which intermittently surfaces as
    #     KeyError: 'anyio'
    # from importlib._bootstrap._find_and_load — a hard, flaky session-start
    # failure on installs where the extra is ALREADY present.
    try:
        import claude_agent_sdk  # noqa: F401

        return True, "ok"
    except ImportError:
        pass

    # Lazy-install lane, mirroring agent/anthropic_adapter._get_anthropic_sdk:
    # the extra is opt-in (excluded from [all]), so first use on a lean
    # install goes through tools.lazy_deps.ensure. FeatureUnavailable falls
    # through to the ImportError message below — same fail shape either way.
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("provider.claude_agent_sdk", prompt=False)
    except ImportError:
        pass
    except Exception:
        # FeatureUnavailable — fall through to ImportError handling below
        pass
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return (
            False,
            "claude-agent-sdk is not installed. "
            "Install with: pip install 'hermes-agent[claude-agent-sdk]'",
        )
    return True, "ok"


def _hermes_repo_root() -> str:
    """Repo root for the hermes-tools MCP subprocess (PYTHONPATH)."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


# The SDK spawns the Claude Code CLI with the FULL inherited parent env and
# merges ``ClaudeAgentOptions.env`` ON TOP of it (subprocess_cli.py builds
# ``{**os.environ, ..., **options.env, ...}``), so a key can never be REMOVED
# from the options side — the only lever is an explicit override. Every
# metered billing vector below is overridden to "" (the CLI and the AWS/GCP
# SDKs treat an empty value as unset), so a credentialed gateway environment
# cannot silently re-route this provider's billing off the Claude
# subscription. Why each class of key is stripped:
#
#   ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN — the CLI prefers these over
#     subscription OAuth: the exact silent-rebilling the fail-closed startup
#     guard exists to stop. The guard covers Hermes' own process at startup;
#     this scrub covers the spawned CLI, which otherwise inherits them.
#   CLAUDE_CODE_USE_BEDROCK + AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
#     AWS_SESSION_TOKEN — flips the CLI onto AWS Bedrock, billing the AWS
#     account (metered) instead of the subscription; the AWS static
#     credentials are the vector that makes the flip actually authenticate.
#   CLAUDE_CODE_USE_VERTEX + GOOGLE_APPLICATION_CREDENTIALS — the same
#     takeover via Google Vertex, billing the GCP project.
#
# Deliberately NOT stripped: HOME/PATH (the CLI needs them to run and to find
# its credential store) and the subscription token flow itself —
# CLAUDE_CODE_OAUTH_TOKEN / ~/.claude — which is what this provider runs on.
_METERED_ENV_DENYLIST = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def _is_subscription_oauth_token(value: str) -> bool:
    """True when an ANTHROPIC_TOKEN value is OAuth/setup-token shaped — the
    subscription lane itself, not a metered vector. Unknown shapes count as
    metered (fail closed), including when the classifier cannot be imported."""
    try:
        from agent.anthropic_adapter import _is_oauth_token
    except Exception:
        return False
    try:
        return bool(_is_oauth_token(value))
    except Exception:
        return False


def _is_metered_sdk_env_value(key: str, value: str) -> bool:
    """Whether an SDK child env value can switch billing off subscription.

    ``ANTHROPIC_TOKEN`` is ambiguous in Hermes: a setup/OAuth token is the
    desired subscription credential, while every unrecognised shape is
    treated as metered.  The other denylisted variables are always metered
    routing vectors when non-empty.
    """
    if key not in _METERED_ENV_DENYLIST or not value:
        return False
    if key == "ANTHROPIC_TOKEN":
        return not _is_subscription_oauth_token(value)
    return True


def _scrubbed_sdk_env() -> dict[str, str]:
    """Empty-string overrides for every metered billing vector currently set
    in the parent environment. Only PRESENT keys are overridden — writing
    ``""`` for absent ones would introduce empty vars the child never had
    (an empty AWS_ACCESS_KEY_ID can itself confuse AWS credential chains)."""
    return {
        key: ""
        for key in _METERED_ENV_DENYLIST
        if _is_metered_sdk_env_value(key, os.environ.get(key, ""))
    }


# The SDK serializes the stdio MCP config — env INCLUDED — into the claude
# CLI's --mcp-config argument, i.e. onto the subprocess argv, which any local
# user can read via ps. Nothing secret may ever ride this dict: the env is a
# minimal ALLOWLIST, never a copy of the credentialed environment. Keyed
# Hermes tools inside the server degrade via their own check_fns — the
# subscription lane's fail-closed posture.
_MCP_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "PYTHONUTF8",
    "HERMES_HOME",
    "HERMES_KANBAN_TASK",
    "HERMES_MCP_STATE_DB",  # the shims' documented state-DB override — a path, not a secret
    "HERMES_QUIET",
    "HERMES_REDACT_SECRETS",
)


def _provider_config() -> dict:
    """The `agent.claude_agent_sdk` config block ({} when absent/unreadable)."""
    try:
        from hermes_cli.config import load_config_readonly

        block = ((load_config_readonly() or {}).get("agent", {}) or {}).get(
            "claude_agent_sdk", {}
        )
        return block if isinstance(block, dict) else {}
    except Exception:
        return {}


def _provider_flag(config_key: str, default: bool = False) -> bool:
    """Behavioural flag read from `agent.claude_agent_sdk.<key>` in config.yaml.

    config.yaml is the ONLY interface. AGENTS.md keeps non-secret behavioural
    settings out of `HERMES_*` environment variables, so there is deliberately
    no env override here — a deployment sets the key in config.yaml.
    Canonical defaults live in `hermes_cli/config.py::DEFAULT_CONFIG`.
    """
    value = _provider_config().get(config_key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def _configured_hybrid_exclude() -> list:
    """agent.claude_agent_sdk.hybrid_mcp_bridge_exclude from config.yaml.

    Names to drop from the hybrid bridge (both buckets). Match on the raw
    Hermes registry name, no ``mcp__`` prefix. Non-string entries are
    dropped silently — a typo is a config error the operator will notice
    when the tool doesn't disappear, not a reason to widen exposure.
    """
    raw = _provider_config().get("hybrid_mcp_bridge_exclude")
    if not isinstance(raw, (list, tuple)):
        return []
    out: list = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        name = entry.strip()
        if name and name not in out:
            out.append(name)
    return out


def _build_hermes_tools_mcp_config(
    hermes_session_id: Optional[str] = None,
) -> dict[str, Any]:
    """The stdio MCP server exposing Hermes tools into the SDK agent loop —
    the exact server the codex runtime uses (backend-agnostic), launched with
    this venv's interpreter. McpStdioServerConfig has no cwd field, so the
    repo root rides PYTHONPATH."""
    env = {
        key: os.environ[key]
        for key in _MCP_ENV_ALLOWLIST
        if os.environ.get(key)
    }
    env["PYTHONPATH"] = _hermes_repo_root() + os.pathsep + os.environ.get("PYTHONPATH", "")
    if hermes_session_id:
        # Lets the stateless session_search shim exclude the calling
        # session's own lineage from recall results (#26567). The shim reads
        # the canonical HERMES_SESSION_ID — set explicitly with THIS
        # session's id rather than allowlisting the ambient variable, so a
        # multi-session host can never leak a sibling session's id into the
        # subprocess.
        env["HERMES_SESSION_ID"] = str(hermes_session_id)
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": [
            "-m",
            "agent.transports.hermes_tools_mcp_server",
            "--profile",
            "claude-agent-sdk",
        ],
        "env": env,
    }


class _StreamEnd:
    """Reader-loop sentinel: the SDK message stream ended (CLI exited or the
    transport tore down). Routed to the in-flight turn so it fails fast and
    retires cleanly instead of waiting out its full turn_timeout on a dead
    stream."""

    def __init__(self, error: Optional[str] = None) -> None:
        self.error = error


class ClaudeAgentSdkSession:
    """One SDK client per Hermes session, lifetime owned by AIAgent.

    Not thread-safe from the caller's side — one caller drives it at a time,
    matching AIAgent.run_conversation(). Internally owns a loop thread."""

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        permission_mode: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        approval_bypass_provider: Optional[Callable[[], bool]] = None,
        on_tool_started: Optional[Callable[[str, str, dict], None]] = None,
        max_budget_usd: Optional[float] = None,
        client_factory: Optional[Callable[..., Any]] = None,
        include_hermes_tools: bool = True,
        hermes_session_id: Optional[str] = None,
        resume_session_id: Optional[str] = None,
        on_stream_delta: Optional[Callable[[str], None]] = None,
        on_interim_assistant: Optional[Callable[[str], None]] = None,
        on_tool_iteration: Optional[Callable[[], None]] = None,
        on_unsolicited_result: Optional[Callable[[list[str]], None]] = None,
        on_compaction: Optional[Callable[[str], None]] = None,
        on_compact_boundary: Optional[Callable[[str], None]] = None,
        # Hybrid MCP bridge (ported from PR #56413): the explicit config
        # opt-in plus both `agent` and `tools` activate in-process servers
        # exposing the full Hermes registry (including proxified third-party
        # MCPs). Missing either input or the opt-in preserves fcava's original
        # stdio-only behavior.
        agent: Optional[Any] = None,
        tools: Optional[list[dict]] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._model = model
        requested_permission_mode = (
            permission_mode
            or _configured_permission_mode()
            or _HERMES_TO_SDK_PERMISSION_MODE.get(
                os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"),
                "default",
            )
        )
        self._sdk_approval_bypass_requested = (
            requested_permission_mode == "bypassPermissions"
        )
        # The SDK auto-approves before can_use_tool in bypassPermissions mode.
        # Keep Hermes in callback-capable mode and carry the bypass intent into
        # this mandatory session wrapper, where canonical validation and
        # immutable hardline/sudo/user-deny floors run before every selected
        # gateway, CLI, ACP, plugin, or custom callback.
        self._permission_mode = (
            "default"
            if self._sdk_approval_bypass_requested
            else requested_permission_mode
        )
        self._system_prompt_append = system_prompt_append
        self._approval_callback = approval_callback
        self._approval_bypass_provider = approval_bypass_provider
        self._on_tool_started = on_tool_started
        self._on_compaction = on_compaction
        self._on_compact_boundary = on_compact_boundary
        self._max_budget_usd = max_budget_usd
        self._client_factory = client_factory  # test seam
        self._include_hermes_tools = include_hermes_tools
        # Hermes-side session id, exported to the hermes-tools MCP subprocess
        # so the stateless session_search shim can exclude its own lineage.
        self._hermes_session_id = hermes_session_id
        # SDK-side session id to resume (#25267 continuity). Verified live:
        # resume restores the model context and keeps the SAME session id; a
        # stale id fails the session start (the caller retires + retries
        # fresh).
        self._resume_session_id = resume_session_id
        # Display-only partial-text consumer (W4 streaming). Deltas never
        # enter the projected transcript; the gateway's stream consumer
        # handles rate limiting and the already_sent final-send dedup.
        self._turn_callback_lock = threading.RLock()
        self._on_stream_delta = on_stream_delta
        # Hybrid MCP bridge inputs (see param docstring above).
        self._agent = agent
        self._tools = tools
        # Completed assistant prose that accompanies a tool call is a true
        # interim status, not final-answer text. It follows the gateway's
        # existing commentary callback so platforms can render it separately.
        # These callbacks are refreshed before EVERY run_turn: a session may
        # safely span several Hermes turns, while visibility must stay scoped
        # to the current one.
        self._on_interim_assistant = on_interim_assistant
        self._on_tool_iteration = on_tool_iteration

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._session_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        self._closed = False
        # Activity evidence for the in-flight turn (None between turns).
        self._turn_watch: Optional[_TurnWatch] = None
        # Snapshot the streaming posture once: the quiet-watchdog default is
        # derived from it, and a mid-session config edit must not flip the
        # watchdog's semantics while the live client still has the old
        # include_partial_messages option.
        self._streaming = _provider_flag("streaming")
        # Stream ownership (see _reader_loop). The reader task is the ONLY
        # consumer of the SDK stream; `_turn_inbox` is non-None exactly while a
        # turn is in flight, which is what makes "unsolicited" decidable.
        self._reader_task: Any = None
        self._turn_inbox: Any = None
        self._turn_claims: Any = None
        self._turn_claim_requested = False
        self._unsolicited_results = 0
        self._stream_ended: Optional[_StreamEnd] = None
        # Delivery half of the stream-ownership fix (dasbrow-hermes-coder#2):
        # a finished background Agent task's answer is captured here and
        # handed to the callback — it must never enter a turn's result, but
        # dropping it entirely left completed work silently undelivered
        # (observed live 2026-07-29: answers sat in the CLI session until the
        # operator poked). No callback wired = the historical drop semantics.
        self._on_unsolicited_result = on_unsolicited_result
        self._unsolicited_text: list[str] = []
        self._unsolicited_delivered: set[str] = set()
        # One immutable billing posture per child. The startup environment,
        # post-start evidence guard, and accounting label must never disagree
        # because config.yaml changed while a long-lived SDK session was live.
        self._allow_metered = _provider_flag("allow_metered_key")
        self._billing_evidence: dict[str, Any] = {}
        self._billing_guard_error: Optional[str] = None

    def set_turn_visibility_callbacks(
        self,
        *,
        on_interim_assistant: Optional[Callable[[str], None]],
        on_tool_iteration: Optional[Callable[[], None]],
    ) -> None:
        """Atomically install current-turn, runtime-owned visibility hooks."""
        with self._turn_callback_lock:
            self._on_interim_assistant = on_interim_assistant
            self._on_tool_iteration = on_tool_iteration

    # ---------- lifecycle ----------

    def ensure_started(self) -> str:
        """Start the loop thread, build the SDK client, connect. Idempotent —
        returns the session marker (SDK session ids arrive on first result)."""
        if self._client is not None:
            return self._session_id or "pending"
        # Hard default, enforced fail-closed: this provider targets the Claude
        # SUBSCRIPTION. If a metered ANTHROPIC_API_KEY is present the
        # underlying CLI would silently prefer it — refuse to start instead.
        # ANTHROPIC_TOKEN is in the set because it alone authenticates
        # Hermes' NATIVE Anthropic lane (x-api-key, metered) — but Hermes
        # also persists subscription setup tokens there, so only an
        # API-key-shaped value counts as a metered vector.
        for metered_var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN"):
            value = os.environ.get(metered_var)
            if not value or self._allow_metered:
                continue
            if metered_var == "ANTHROPIC_TOKEN" and _is_subscription_oauth_token(value):
                continue
            raise RuntimeError(
                f"claude-agent-sdk runtime refuses to start: {metered_var} "
                "is set, which would silently switch billing from the "
                "Claude subscription to metered API usage. Unset it, or "
                "set agent.claude_agent_sdk.allow_metered_key: true in "
                "config.yaml to explicitly allow it."
            )
        if self._client_factory is None:
            ok, msg = check_claude_sdk_available()
            if not ok:
                raise RuntimeError(msg)

        self._start_loop_thread()
        client = self._build_client()
        # Assign BEFORE connect: a connect timeout/cancel leaves a
        # half-connected client whose CLI subprocess close() must still reap
        # — a None _client would skip disconnect and orphan it.
        self._client = client
        self._run_coro(client.connect(), timeout=60.0)
        # From here on exactly ONE consumer owns the SDK stream (_reader_loop).
        # Started after connect so the client is live, before any turn so no
        # message can arrive unowned.
        self._start_reader()
        logger.info(
            "claude-agent-sdk session started: model=%s mode=%s cwd=%s",
            self._model or "cli-default",
            self._permission_mode,
            self._cwd,
        )
        return self._session_id or "pending"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Cancel the reader BEFORE disconnect so it unwinds on a live stream
        # instead of raising against a torn-down one.
        self._stop_reader()
        if self._client is not None and self._loop is not None:
            # Budget must exceed the SDK transport's own escalation ladder
            # (stdin lock + graceful wait + SIGTERM + SIGKILL); on timeout the
            # loop thread dies below and its shielded reap never finishes, so
            # fall back to killing the child directly.
            pid = _sdk_child_pid(self._client)
            # Capture the psutil identity before waiting on disconnect. If the
            # original child exits and its PID is reused during the timeout,
            # this object will fail psutil's identity check instead of
            # resolving the reused PID as a fresh kill target.
            child_process = _own_sdk_child_process(pid) if pid else None
            try:
                self._run_coro(
                    self._client.disconnect(), timeout=_SDK_DISCONNECT_TIMEOUT_S
                )
            except Exception:  # pragma: no cover - best-effort cleanup
                if child_process is not None:
                    _force_kill_sdk_child(pid, process=child_process)
            self._client = None
        self._stop_loop_thread()

    def __enter__(self) -> "ClaudeAgentSdkSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- compaction ----------

    def _build_compaction_hooks(self) -> Optional[dict]:
        """PreCompact -> watchdog suspension, and on_compaction(trigger).

        ``trigger`` is the SDK's own literal: "auto" for the CLI's automatic
        compaction (the one that silently stalls a turn) or "manual" for an
        explicit /compact. The callback is best-effort: it must never fail the
        hook, because refusing a hook can block the compaction itself.

        Wired even when ``_on_compaction`` is None. The hook's primary job is
        no longer the status notice but telling _TurnWatch that the coming
        silence is legitimate; skipping it when only the status callback is
        unset would leave the turn killable mid-compaction for no benefit.
        """
        try:
            from claude_agent_sdk import HookMatcher
        except Exception:  # pragma: no cover - SDK predates hooks
            logger.debug("claude-agent-sdk: HookMatcher unavailable", exc_info=True)
            return None

        async def _on_pre_compact(input_data, tool_use_id, context):
            # FIRST and unconditionally, outside the try: if a status callback
            # raises, the suspension must still be in place. This ordering is
            # the fix -- everything below is the pre-existing status notice.
            #
            # getattr, not attribute access: a hook that raises AttributeError
            # would break the very turn it exists to protect, and the attribute
            # is genuinely absent on sessions built without __init__.
            watch = getattr(self, "_turn_watch", None)
            if watch is not None:
                watch.compaction_begin()
            try:
                trigger = ""
                if isinstance(input_data, dict):
                    trigger = str(input_data.get("trigger") or "")
                if self._on_compaction is not None:
                    self._on_compaction(trigger or "auto")
            except Exception:
                logger.debug("compaction status callback failed", exc_info=True)
            return {}

        return {"PreCompact": [HookMatcher(hooks=[_on_pre_compact])]}

    def _handle_compact_boundary(self, message: Any) -> None:
        """compact_boundary -> on_compact_boundary(trigger): compaction FINISHED.

        The SDK exposes ``PreCompact`` as a hook but has no post-side
        counterpart, which is why the completion edge was originally deferred to
        the end of the turn. That was wrong: the CLI *does* announce completion,
        as a plain ``system`` message with ``subtype="compact_boundary"``, and
        the SDK's parser passes unknown subtypes through its generic fallback,
        so it arrives on the normal message stream mid-turn.

        The distinction is not cosmetic. Emitting at turn end put the notice in
        the one place it could never be seen: non-durable statuses are deleted
        by end-of-turn progress cleanup, so it was created and destroyed in the
        same instant (measured 2026-08-16 -- boundary at 07:41:16, deferred emit
        at 07:42:17, 61s late and invisible). Firing here restores the native
        contract every other provider already follows -- notice right after
        compaction, cleaned up with the rest of the turn's progress -- and makes
        COMPACTION_DONE_STATUS's "continuing turn" literally true again.

        Best-effort by construction: a raising callback must not break the turn
        that is still streaming.
        """
        if (
            type(message).__name__ != "SystemMessage"
            or getattr(message, "subtype", "") != "compact_boundary"
        ):
            return
        # Lift the suspension before anything else -- including the unwired-
        # callback early return further down. Leaving it armed would hold the
        # gate open until the bounded ceiling on every compacting turn.
        # getattr for the same reason as the PreCompact side: this runs on the
        # message drain, where an AttributeError would break a streaming turn.
        watch = getattr(self, "_turn_watch", None)
        if watch is not None:
            watch.compaction_end()
        data = getattr(message, "data", None)
        metadata = None
        if isinstance(data, dict):
            # The SDK hands this over snake_cased ("compact_metadata", observed
            # in production 2026-08-16); the CLI's own transcript writes the
            # camelCase original. Accept both so the trigger stays accurate if
            # the normalization changes -- an unreadable trigger is not fatal
            # (it falls back to "auto"), just less honest.
            metadata = data.get("compact_metadata") or data.get("compactMetadata")
        trigger = ""
        if isinstance(metadata, dict):
            trigger = str(metadata.get("trigger") or "")
        logger.info(
            "claude-agent-sdk compact_boundary: session=%s trigger=%s",
            getattr(message, "session_id", None) or self._session_id or "none",
            trigger or "unknown",
        )
        if self._on_compact_boundary is None:
            return
        try:
            self._on_compact_boundary(trigger or "auto")
        except Exception:
            logger.debug("compact_boundary callback failed", exc_info=True)

    def _observe_billing_evidence(self, message: Any) -> None:
        """Record the CLI's machine-readable billing lane and fail closed.

        Environment scrubbing protects against billing vectors Hermes knows
        about. The child remains the authority for what it actually selected:
        ``system/init.apiKeySource`` reports API-key use, while the pinned SDK's
        typed ``RateLimitEvent`` exposes subscription Extra Usage. Unless the
        operator explicitly set ``allow_metered_key``, either signal is a
        fatal configuration/account-state mismatch, not an "included" turn.
        """
        name = type(message).__name__
        if name == "SystemMessage" and getattr(message, "subtype", "") == "init":
            data = getattr(message, "data", None)
            if isinstance(data, dict) and (
                "apiKeySource" in data or "api_key_source" in data
            ):
                source = data.get("apiKeySource", data.get("api_key_source"))
                source_text = str(source or "none").strip() or "none"
                self._billing_evidence["api_key_source"] = source_text
                if (
                    not self._allow_metered
                    and source_text.lower() != "none"
                    and self._billing_guard_error is None
                ):
                    self._billing_guard_error = (
                        "claude-agent-sdk billing guard: the CLI reported "
                        f"API-key source {source_text!r}. Remove the metered "
                        "credential, or set agent.claude_agent_sdk."
                        "allow_metered_key: true to opt in explicitly."
                    )
            return

        if name != "RateLimitEvent":
            return
        info = getattr(message, "rate_limit_info", None)
        if info is None:
            return
        raw = getattr(info, "raw", None)
        raw = raw if isinstance(raw, dict) else {}
        is_using_overage = raw.get("isUsingOverage")
        if isinstance(is_using_overage, bool):
            self._billing_evidence["is_using_overage"] = is_using_overage
        overage_status = getattr(info, "overage_status", None)
        if overage_status is None:
            overage_status = raw.get("overageStatus")
        if overage_status is not None:
            self._billing_evidence["overage_status"] = str(overage_status)
        rate_limit_type = getattr(info, "rate_limit_type", None)
        if rate_limit_type is None:
            rate_limit_type = raw.get("rateLimitType")
        if rate_limit_type is not None:
            self._billing_evidence["rate_limit_type"] = str(rate_limit_type)

        if self._allow_metered or self._billing_guard_error is not None:
            return
        if is_using_overage is True:
            self._billing_guard_error = (
                "claude-agent-sdk billing guard: metered subscription Extra "
                "Usage is active. Disable Extra Usage in the Claude account, "
                "or set agent.claude_agent_sdk.allow_metered_key: true to "
                "opt in explicitly."
            )
        elif str(overage_status or "").lower() in {"allowed", "allowed_warning"}:
            self._billing_guard_error = (
                "claude-agent-sdk billing guard: subscription extra usage is "
                "enabled and could silently become metered when the included "
                "limit is exhausted. Disable Extra Usage in the Claude "
                "account, or set agent.claude_agent_sdk.allow_metered_key: "
                "true to opt in explicitly."
            )

    def _reported_billing_mode(self) -> str:
        source = str(self._billing_evidence.get("api_key_source") or "").lower()
        using_overage = self._billing_evidence.get("is_using_overage")
        rate_limit_type = str(
            self._billing_evidence.get("rate_limit_type") or ""
        ).lower()
        if (source and source != "none") or using_overage is True:
            return "sdk_reported_metered"
        if rate_limit_type == "overage" and using_overage is not False:
            return "sdk_reported_metered"
        if not self._allow_metered:
            # Any contrary evidence trips the guard before accounting.
            return "subscription_included"
        if source == "none" and using_overage is False:
            return "subscription_included"
        return "unknown"

    # ---------- context usage ----------

    def context_usage(self) -> Optional[dict]:
        """Live context usage reported by the CLI, or None if unavailable.

        Ground truth, unlike Hermes' own estimate on this lane: api_messages
        holds FULL tool payloads that are never sent to the CLI, so the local
        estimate over-reports by an order of magnitude (1.5-2.4M tokens for a
        transcript whose real size was ~111k). Callers that need a real number
        -- status lines, compaction heuristics -- must use this instead.

        Returns the SDK's ContextUsageResponse mapping: totalTokens, maxTokens
        (already reduced by the autocompact buffer), contextWindow, the
        percentage used, model, and isAutoCompactEnabled.

        Best-effort by design: a disconnected session, an older SDK without the
        method, or a query failure all yield None rather than raising into a
        status path.
        """
        client = self._client
        if client is None or self._loop is None:
            return None
        getter = getattr(client, "get_context_usage", None)
        if not callable(getter):
            return None  # SDK predates get_context_usage()
        try:
            usage = self._run_coro(getter(), timeout=10.0)
        except Exception:
            logger.debug(
                "claude-agent-sdk context-usage query failed", exc_info=True
            )
            return None
        return usage if isinstance(usage, dict) else None

    # ---------- interrupt ----------

    def consume_interrupt(self) -> None:
        """Clear a pending interrupt signal — the caller honored it through
        another path (e.g. the runtime's cold-agent short-circuit)."""
        self._interrupt_event.clear()

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to interrupt and unwind."""
        self._interrupt_event.set()
        if self._client is not None and self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._client.interrupt(), self._loop
                )
                future.add_done_callback(_swallow_interrupt_result)
            except Exception as exc:  # pragma: no cover
                logger.debug(
                    "SDK interrupt scheduling failed: %s",
                    _safe_sdk_error_text(exc),
                )

    def steer(self, text: str) -> bool:
        """Inject a mid-turn user message using the SDK's own streaming input.

        Why this exists: on this lane the SDK owns tool execution, so Hermes'
        tool-batch drain points (agent/tool_executor.py) are never reached and
        a /steer would strand in ``_pending_steer`` until the turn finalizer
        handed it back — where the gateway only redelivers it if no other
        message is queued. Steers were therefore silently lost.

        ``ClaudeSDKClient.query()`` is documented as "send a new request in
        streaming mode", and we already connect in streaming mode (``connect()``
        with no prompt), so a second query() on the LIVE session is the SDK's
        own steering contract — not an interrupt and not a session restart,
        which is what the claude-cli-live transport has to do.

        Verified 2026-08-15 against SDK 0.2.120: a query() issued while a tool
        call was in flight did not raise and was honored at the next turn
        boundary ~3s later. Note it also produces a SECOND ResultMessage once
        the superseded work unwinds; that arrives unclaimed and is handled by
        the existing unsolicited-result path (``_on_unsolicited_result``), so
        ``deliver_background_results`` is load-bearing here.

        Returns True only when the steer was actually scheduled onto a live
        turn. False means "not applicable" — the caller must fall back to the
        ordinary pending-steer stash. Crucially this is never both: returning
        True suppresses the stash, so a steer cannot be delivered twice.
        """
        if not text or not text.strip():
            return False
        # No claimed turn means there is nothing to steer INTO. Sending anyway
        # would open a fresh unclaimed turn on this session whose output the
        # reader routes to the unsolicited path — a reply appearing from
        # nowhere. Decline instead and let the caller queue it normally.
        if self._turn_inbox is None:
            return False
        client = self._client
        loop = self._loop
        if client is None or loop is None:
            return False
        cleaned = text.strip()
        try:
            future = asyncio.run_coroutine_threadsafe(client.query(cleaned), loop)
            future.add_done_callback(_swallow_steer_result)
        except Exception:
            logger.debug("SDK steer scheduling failed", exc_info=True)
            return False
        logger.info(
            "claude-agent-sdk: steered live turn via streaming input (%d chars)",
            len(cleaned),
        )
        return True

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: Optional[float] = None,
        post_tool_quiet_timeout: Optional[float] = None,
        watch_poll_interval: float = 1.0,
        abort_grace: float = _TURN_ABORT_GRACE,
    ) -> TurnResult:
        """Send a user message and block until the SDK's ResultMessage,
        projecting the typed stream into Hermes' messages shape.

        Turn lifetime is activity-aware, not a bare wall clock (production
        forensics: four of six 600s kills were actively-working turns).
        `turn_timeout` (explicit arg > agent.claude_agent_sdk.turn_timeout >
        600s) is a SOFT budget: it only fires when nothing is outstanding —
        no tool running, no approval awaiting a human — and the stream has
        ALSO been quiet ≥ min(30s, budget). A post-tool quiet watchdog
        (`post_tool_quiet_timeout`; default 90s with streaming on, disabled
        with streaming off) catches wedges early: armed when a tool result
        arrives, cleared by any later activity. On a trip the CLI is
        interrupted and given `abort_grace` to unwind — a clean unwind keeps
        the partial transcript and the resumable session id (no retire);
        only a grace expiry hard-cancels and retires."""
        result = TurnResult()
        prompt = _coerce_turn_input(user_input)
        if isinstance(prompt, str) and not prompt.strip():
            self._interrupt_event.clear()
            result.final_text = (
                "This Claude Agent SDK route can't process an empty message. "
                "Please send text or a supported image."
            )
            result.error = result.final_text
            result.api_call_made = False
            return result
        try:
            self.ensure_started()
        except Exception as exc:
            safe_exc = _safe_sdk_error_text(exc)
            hint = classify_auth_failure(safe_exc)
            result.error = hint or f"claude-agent-sdk startup failed: {safe_exc}"
            result.should_retire = True
            # Keep the traceback: `str(exc)` alone collapses a KeyError to a
            # bare quoted key (e.g. "'anyio'"), which is undiagnosable from the
            # user-facing error string. WARNING so it reaches errors.log.
            logger.warning(
                "claude-agent-sdk startup failed (%s: %s)\n"
                "Traceback (most recent call last):\n%s",
                type(exc).__name__,
                safe_exc,
                _safe_sdk_traceback(exc),
            )
            # A refusal to start is fatal to the run, not turn-scoped: the
            # metered-key guard and an uninstallable SDK are config errors
            # ("startup" — NOT "billing": kanban maps failure_reason
            # "billing" to the transient EX_TEMPFAIL requeue sentinel, and
            # retrying cannot fix a present metered key); an auth-classified
            # failure is "auth".
            result.fatal_reason = "auth" if hint else "startup"
            return result

        # Text buffered before this turn whose terminal ResultMessage never
        # arrived is partial mid-burst content — a later unrelated result
        # must never pick it up (misattribution is the proven worse failure;
        # texts parked DURING the turn re-buffer via the residue drain and
        # are unaffected). Never a silent drop: WARN what is discarded.
        stale = list(self._unsolicited_text)
        if stale:
            self._unsolicited_text.clear()
            logger.warning(
                "claude-agent-sdk: discarding %d stale unsolicited text(s) "
                "(%d chars) buffered before this turn — their terminal "
                "ResultMessage never arrived; attaching them to a later "
                "unrelated result is the proven worse failure",
                len(stale), sum(len(t) for t in stale),
            )

        # An interrupt that arrived between turns or during connect (up to
        # 60s) targets THIS turn — honor it instead of erasing it. (The old
        # unconditional clear() silently swallowed that window.)
        if self._interrupt_event.is_set():
            self._interrupt_event.clear()
            result.interrupted = True
            return result

        import concurrent.futures

        budget = (
            float(turn_timeout)
            if turn_timeout is not None
            else (_configured_turn_timeout() or _DEFAULT_TURN_TIMEOUT)
        )
        if post_tool_quiet_timeout is not None:
            quiet = float(post_tool_quiet_timeout)
        else:
            configured_quiet = _configured_post_tool_quiet_timeout()
            if configured_quiet is not None:
                quiet = configured_quiet  # 0 = explicitly disabled
            else:
                quiet = (
                    _DEFAULT_POST_TOOL_QUIET_STREAMING if self._streaming else 0.0
                )
        poll = max(0.01, float(watch_poll_interval))

        assert self._loop is not None, "loop thread not started"
        watch = _TurnWatch()
        self._turn_watch = watch
        trip: Optional[str] = None
        trip_elapsed = trip_idle = 0.0
        hard_trip = False
        turn_data: Optional[dict[str, Any]] = None
        future = asyncio.run_coroutine_threadsafe(
            self._consume_turn(prompt), self._loop
        )
        try:
            pending_verdict: Optional[str] = None
            prev_poll = time.monotonic()
            while turn_data is None and trip is None:
                try:
                    turn_data = future.result(timeout=poll)
                except (TimeoutError, concurrent.futures.TimeoutError):
                    # py3.11 unifies TimeoutError/asyncio.TimeoutError/
                    # concurrent.futures.TimeoutError — a DONE future here
                    # means the COROUTINE settled in the race window: with an
                    # exception (typically its own TimeoutError — a socket/
                    # pipe timeout under the CLI), re-raise into the
                    # classification path instead of misreading it as a poll
                    # expiry (which would busy-spin until the budget); with a
                    # RESULT, harvest it — the turn completed.
                    if future.done():
                        if future.exception() is not None:
                            raise
                        turn_data = future.result()
                        continue
                    now = time.monotonic()
                    if now - prev_poll > _POLL_STALL_FACTOR * poll:
                        # The PROCESS was descheduled (swap/OOM), not the
                        # turn — that gap proves nothing about the stream.
                        watch.rebaseline()
                        prev_poll = now
                        pending_verdict = None
                        continue
                    prev_poll = now
                    verdict = watch.check(budget=budget, quiet=quiet)
                    if verdict is None:
                        pending_verdict = None
                        continue
                    if pending_verdict != verdict:
                        # Debounce: two consecutive polls must agree before
                        # a trip (closes the tick-vs-check race window).
                        pending_verdict = verdict
                        continue
                    now = time.monotonic()
                    trip = verdict
                    trip_elapsed = now - watch.started
                    trip_idle = now - watch.last_activity
            if trip is not None and turn_data is None:
                if future.done():
                    # Completed between check() and the trip decision —
                    # completion wins, the trip is void.
                    turn_data = future.result()
                    trip = None
                else:
                    # Interrupt through request_interrupt(): setting
                    # _interrupt_event is REQUIRED — _consume_turn's W22 EDE
                    # mask keys on the turn-local flag, so the CLI's
                    # interrupt-ack is masked instead of delivered as a
                    # fresh error.
                    self.request_interrupt()
                    try:
                        turn_data = future.result(timeout=abort_grace)
                    except (TimeoutError, concurrent.futures.TimeoutError):
                        if future.done():
                            if future.exception() is not None:
                                raise
                            turn_data = future.result()
                        elif not future.cancel():
                            # Settled in the cancel race window — a real
                            # result beats a fabricated hard trip.
                            if future.exception() is not None:
                                raise
                            turn_data = future.result()
                        else:
                            hard_trip = True
            if trip is not None and turn_data is not None and not hard_trip:
                if (
                    not turn_data["error"]
                    and turn_data.get("result_uuid")
                    and turn_data["final_text"]
                ):
                    # The turn FINISHED inside the grace window with a real
                    # answer — deliver it in full; the trip never happened.
                    # final_text is REQUIRED: the W22-masked interrupt ack
                    # also has error=None + a result uuid, but its final_text
                    # is empty (projection stopped at the interrupt) — that
                    # shape must stay a trip, or it degrades into a silent
                    # dropped turn. Consume our own interrupt signal so the
                    # mapping below cannot misread it. (A real user /stop
                    # racing this exact window loses its signal too — the
                    # completed answer it targeted is delivered, same as any
                    # near-boundary stop today.)
                    self._interrupt_event.clear()
                    logger.info(
                        "claude-agent-sdk: turn completed during watchdog "
                        "grace (%.0fs elapsed) — delivered in full",
                        time.monotonic() - watch.started,
                    )
                    trip = None
        except Exception as exc:
            self._interrupt_event.clear()
            safe_exc = _safe_sdk_error_text(exc)
            hint = classify_auth_failure(safe_exc)
            result.error = hint or f"claude-agent-sdk turn failed: {safe_exc}"
            result.should_retire = True
            if hint is not None:
                # Auth failures are fatal; other mid-turn exceptions stay
                # transient (retire + retry semantics unchanged).
                result.fatal_reason = "auth"
            return result
        finally:
            self._turn_watch = None

        if turn_data is None:
            # Hard trip: the CLI ignored the interrupt for the whole grace —
            # nothing was harvested. Retire (today's shape, now the rare
            # fallback for a genuinely unresponsive CLI).
            self._interrupt_event.clear()
            result.interrupted = True
            result.error = self._format_trip_error(
                trip, budget, quiet, trip_elapsed, trip_idle
            )
            result.should_retire = True
            return result

        result.final_text = turn_data["final_text"]
        result.projected_messages = turn_data["messages"]
        result.tool_iterations = turn_data["tool_iterations"]
        result.token_usage_last = turn_data["usage"]
        result.token_usage_total = turn_data["usage"]
        result.model_last = turn_data.get("model")
        result.billing_mode = turn_data.get("billing_mode", "unknown")
        result.billing_evidence = turn_data.get("billing_evidence", {})
        result.total_cost_usd = turn_data.get("total_cost_usd")
        result.api_call_made = turn_data.get("api_call_made", True)
        result.thread_id = self._session_id
        result.turn_id = turn_data.get("result_uuid")
        result.interrupted = self._interrupt_event.is_set()
        if result.interrupted:
            # Consume the honored interrupt so it cannot bleed into the
            # next turn on this session object.
            self._interrupt_event.clear()
        if turn_data["error"]:
            # A prior MCP tool use is not evidence that this terminal SDK
            # error belongs to MCP; preserve fail-closed Claude auth handling
            # unless the error text itself identifies an MCP origin.
            hint = classify_auth_failure(turn_data["error"])
            result.error = hint or turn_data["error"]
            if hint is not None:
                result.should_retire = True
                result.fatal_reason = "auth"
        if turn_data.get("billing_guard_violation"):
            # This is durable account/config state, not a transient provider
            # billing error: the generic "billing" fatal_reason is a retry
            # sentinel, which would repeatedly spend calls against the same
            # unsafe lane. "startup" stops the run until the operator fixes
            # Extra Usage/credentials or explicitly opts into metering.
            result.should_retire = True
            result.fatal_reason = "startup"
        if turn_data.get("stream_ended"):
            # A dead stream is permanent on this session object: the reader
            # exited, _stream_ended stays set, and ensure_started() keeps
            # returning early while _client is non-None — so WITHOUT retire,
            # every later turn short-circuits to this same error with zero
            # model calls, forever (the CLI died once and poisoned the
            # session; observed as the unrecoverable half of the 2026-08-09
            # desync incident). Retire lets the runtime rebuild a fresh CLI
            # on the next attempt/turn. Model-level result subtypes
            # (error_max_turns, error_max_budget_usd) stay non-retiring.
            result.should_retire = True
        if trip is not None:
            # Clean-ack trip: the CLI honored our interrupt inside the grace,
            # the partial transcript above is preserved, and the session id
            # is resumable — mirror the user-interrupt lane (close-but-keep-
            # resume-id) instead of retiring. The trip text WINS over
            # whatever error the drain surfaced (typically the W22-masked
            # interrupt ack) — EXCEPT auth and stream-death, which keep
            # their retire verdicts from the mapping above.
            if result.fatal_reason != "auth":
                result.error = self._format_trip_error(
                    trip, budget, quiet, trip_elapsed, trip_idle
                )
            result.interrupted = True
        return result

    def _format_trip_error(
        self,
        kind: Optional[str],
        budget: float,
        quiet: float,
        elapsed: float,
        idle: float,
    ) -> str:
        """≤200 chars (the gateway truncates at str(err)[:200]); keeps the
        literal "turn timed out" needle; names the cause."""
        if kind == "post_tool_quiet":
            return (
                f"turn timed out: no SDK activity for {idle:.0f}s after a "
                f"tool result (turn ran {elapsed:.0f}s, quiet limit "
                f"{quiet:.0f}s)"
            )
        return (
            f"turn timed out after {elapsed:.0f}s "
            f"(budget {budget:.0f}s, idle {idle:.0f}s)"
        )

    # ---------- internals ----------

    async def _consume_turn(self, prompt: Any) -> dict[str, Any]:
        """The async side of one turn: query, then read THIS turn's messages
        off the reader loop's inbox until the ResultMessage.

        Deliberately NOT `receive_response()` — that helper serves the oldest
        buffered ResultMessage, which is not necessarily ours. See
        `_reader_loop` for why that is a permanent-corruption bug."""
        projector = ClaudeSdkEventProjector()
        out: dict[str, Any] = {
            "final_text": "",
            "messages": [],
            "tool_iterations": 0,
            "usage": None,
            "error": None,
            "result_uuid": None,
            "model": None,
            "stream_ended": False,
            "mcp_tool_seen": False,
            "billing_guard_violation": False,
            "billing_mode": self._reported_billing_mode(),
            "billing_evidence": dict(self._billing_evidence),
            "total_cost_usd": None,
            "api_call_made": True,
        }
        ended = self._stream_ended
        if ended is not None:
            out["error"] = "SDK message stream ended before this turn" + (
                f": {_safe_sdk_error_text(ended.error)}" if ended.error else ""
            )
            out["stream_ended"] = True
            return out
        if self._billing_guard_error is not None:
            out["error"] = self._billing_guard_error
            out["billing_guard_violation"] = True
            out["billing_mode"] = self._reported_billing_mode()
            out["billing_evidence"] = dict(self._billing_evidence)
            out["api_call_made"] = False
            return out
        inbox: Any = asyncio.Queue()
        # Ask the sole reader to drain everything already ahead of this turn
        # in the resumed client's FIFO, then atomically install our inbox.  A
        # direct assignment here races an already-buffered background result:
        # the reader can observe the new inbox before it observes that older
        # result and mis-serve it as this query's answer.
        claims = self._turn_claims
        if claims is None:
            out["error"] = "SDK message reader is not ready to claim this turn"
            out["stream_ended"] = True
            return out
        claim_ack = asyncio.get_running_loop().create_future()
        self._turn_claim_requested = True
        claims.put_nowait(("claim", inbox, claim_ack))
        interrupted = False
        billing_guarded = False
        try:
            try:
                await claim_ack
            finally:
                self._turn_claim_requested = False
            ended = self._stream_ended
            if ended is not None:
                out["error"] = "SDK message stream ended before this turn" + (
                    f": {_safe_sdk_error_text(ended.error)}" if ended.error else ""
                )
                out["stream_ended"] = True
                return out
            query_input = (
                _sdk_user_message_stream(prompt)
                if isinstance(prompt, list)
                else prompt
            )
            await self._client.query(query_input)
            while True:
                message = await inbox.get()
                watch = self._turn_watch
                if watch is not None:
                    # Any stream message is liveness — stamp before anything
                    # else so the caller-thread watchdog sees it.
                    watch.tick()
                if isinstance(message, _StreamEnd):
                    out["error"] = (
                        "SDK message stream ended before this turn's result"
                        + (f": {_safe_sdk_error_text(message.error)}" if message.error else "")
                    )
                    out["stream_ended"] = True
                    break
                # Capture the SDK session id from ANY message that carries it —
                # the init SystemMessage announces it first, so even a turn
                # interrupted before its ResultMessage keeps a resumable id.
                early_sid = getattr(message, "session_id", None)
                if early_sid:
                    self._session_id = early_sid
                self._handle_compact_boundary(message)
                if self._billing_guard_error is not None and not billing_guarded:
                    billing_guarded = True
                    out["error"] = self._billing_guard_error
                    out["billing_guard_violation"] = True
                    try:
                        await self._client.interrupt()
                    except Exception:
                        logger.debug(
                            "claude-agent-sdk billing-guard interrupt failed",
                            exc_info=True,
                        )
                if self._interrupt_event.is_set():
                    # Previously `break`. Bailing out before this turn's
                    # ResultMessage orphaned it in the shared stream, and the
                    # NEXT turn was then served that stale result — the same
                    # off-by-one corruption `_reader_loop` exists to prevent,
                    # reachable through the interrupt path with no Agent tool
                    # involved. Keep draining to the result; stop projecting.
                    interrupted = True
                if type(message).__name__ == "StreamEvent":
                    if watch is not None:
                        # A partial delta proves the post-tool model call is
                        # alive — the quiet watchdog stands down.
                        watch.disarm_post_tool()
                    if not interrupted and not billing_guarded:
                        self._forward_stream_delta(message)
                    continue
                for block in getattr(message, "content", None) or []:
                    if (
                        type(block).__name__ == "ToolUseBlock"
                        and str(getattr(block, "name", "")).startswith("mcp__")
                    ):
                        out["mcp_tool_seen"] = True
                if not interrupted and not billing_guarded:
                    self._notify_tool_started(message)
                    self._notify_interim_assistant(message)
                projection = projector.project(message)
                if watch is not None:
                    # Outstanding-tool evidence: ToolUseBlocks issue, tool
                    # results resolve (server tools never enter — the
                    # projector resolves them inside their own assistant
                    # message). A turn with a tool in flight is suspended
                    # from BOTH watchdog rules.
                    issued = sum(
                        len(m.get("tool_calls") or [])
                        for m in projection.messages
                        if m.get("role") == "assistant"
                    )
                    if issued:
                        watch.note_tools_issued(issued)
                    if projection.is_tool_iteration:
                        watch.note_tools_resolved(
                            sum(
                                1
                                for m in projection.messages
                                if m.get("role") == "tool"
                            )
                        )
                        # Codex-parity arm point: a tool result just landed;
                        # silence from here on is the wedge signature.
                        watch.arm_post_tool()
                    elif projection.messages or projection.final_text is not None:
                        watch.disarm_post_tool()
                if projection.model:
                    # Last reported id wins; captured even on interrupted
                    # turns — the tokens were still spent on that model.
                    out["model"] = projection.model
                # A genuine terminal SDK error carries a result string, but it
                # is transport diagnostics, not an assistant answer. The outer
                # runtime needs `out["error"]` to activate provider fallback;
                # emitting this text first persists it as an assistant message
                # and poisons the fallback's history with a false “I am
                # blocked” claim. Contradictory success envelopes remain a
                # success exactly as handled below.
                _result_subtype = getattr(message, "subtype", "") or ""
                _result_is_error = bool(
                    projection.is_result
                    and (
                        getattr(message, "is_error", False)
                        or _result_subtype not in ("", "success")
                    )
                )
                _result_is_contradictory_success = bool(
                    _result_is_error
                    and _result_subtype == "success"
                    and not (getattr(message, "errors", None) or [])
                    and not getattr(message, "api_error_status", None)
                )
                if not interrupted and not billing_guarded:
                    if projection.messages:
                        out["messages"].extend(projection.messages)
                    if projection.is_tool_iteration:
                        out["tool_iterations"] += 1
                        self._notify_tool_iteration()
                    if projection.final_text is not None and (
                        not _result_is_error or _result_is_contradictory_success
                    ):
                        out["final_text"] = projection.final_text
                if projection.is_result:
                    usage = getattr(message, "usage", None)
                    if isinstance(usage, dict):
                        out["usage"] = dict(usage)
                    sid = getattr(message, "session_id", None)
                    if sid:
                        self._session_id = sid
                    out["result_uuid"] = getattr(message, "uuid", None)
                    out["total_cost_usd"] = getattr(
                        message, "total_cost_usd", None
                    )
                    subtype = getattr(message, "subtype", "") or ""
                    if getattr(message, "is_error", False):
                        errors = getattr(message, "errors", None) or []
                        api_error_status = getattr(message, "api_error_status", None)
                        if subtype == "success" and not errors and not api_error_status:
                            # Contradictory envelope: is_error=True yet
                            # subtype="success" with nothing in errors. The
                            # CLI emits this shape rarely (2026-08-11: it
                            # killed a cron run as "RuntimeError: SDK result
                            # error (subtype=success): success" — a dead job
                            # over a turn that had actually produced its
                            # answer). There is no error to report, so the
                            # error flag loses to the subtype; kept loud for
                            # diagnosis. A genuine failure carries a non-empty
                            # errors list or a non-success subtype and still
                            # takes the honest path below.
                            logger.warning(
                                "claude-agent-sdk: contradictory result "
                                "envelope (is_error=True, subtype=success, "
                                "no errors) — treated as success (diagnostic omitted)",
                            )
                            break
                        detail = _safe_sdk_error_text(
                            "; ".join(str(e) for e in errors)
                            or getattr(message, "result", None)
                            or subtype
                        )
                        err_text = f"SDK result error (subtype={subtype}): {detail}"
                        if api_error_status:
                            err_text += f" (HTTP {api_error_status})"
                        # A turn WE interrupted before any assistant content
                        # ends as is_error/error_during_execution in the CLI
                        # ("[ede_diagnostic] result_type=user…") — that is the
                        # interrupt being honored, not a failure; surfacing it
                        # paged the operator with a false "Processing stopped"
                        # (2026-08-09 barge-in incident). Mask ONLY that exact
                        # shape, only when THIS turn's interrupt flag is set
                        # (never a fresh event read — a late /stop must not
                        # reclassify a genuine error), and only when no auth
                        # needle is present (an auth failure must keep its
                        # error + retire path regardless of interrupts).
                        # A real transient failure colliding with an interrupt
                        # re-fires next turn with no interrupt pending and
                        # takes the honest path; the masked text is INFO-kept.
                        if (
                            interrupted
                            and subtype == "error_during_execution"
                            and classify_auth_failure(err_text) is None
                        ):
                            logger.info(
                                "claude-agent-sdk: masked %s on requested "
                                "interrupt (interrupt honored, not a "
                                "failure): %s",
                                subtype,
                                _safe_sdk_error_text(err_text),
                            )
                        else:
                            if not billing_guarded:
                                out["error"] = err_text
                    elif subtype not in ("", "success") and not billing_guarded:
                        # e.g. error_max_turns / error_max_budget_usd — surface
                        # honestly; the partial transcript is still projected.
                        out["error"] = f"SDK turn ended: {subtype}"
                    break
        finally:
            if self._turn_inbox is inbox:
                # Relinquish ownership through the same reader arbiter.  It
                # drains immediately buffered post-result residue into this
                # inbox before acknowledging release, preserving the existing
                # overlap/own-answer dedup handling below.
                if self._stream_ended is None and self._turn_claims is not None:
                    release_ack = asyncio.get_running_loop().create_future()
                    self._turn_claims.put_nowait(("release", inbox, release_ack))
                    await release_ack
                else:
                    self._turn_inbox = None
                # Stream death can win the release handshake after the
                # terminal ResultMessage was already delivered.  The reader's
                # death path acknowledges queued operations so waiters wake,
                # but it deliberately does not install/apply their ownership.
                # Re-check after the acknowledgement: preserve the valid turn
                # result while retiring the now-dead session, and never leave
                # a between-turn inbox falsely claimed.
                if self._stream_ended is not None:
                    if self._turn_inbox is inbox:
                        self._turn_inbox = None
                    out["stream_ended"] = True
            # Anything the reader parked after our ResultMessage belongs to a
            # CLI-initiated turn that overlapped ours. Route it now — left in
            # a discarded queue it would be lost, and left in the stream it
            # would become the next turn's answer. EXCEPT a residue result
            # that repeats THIS turn's own answer: delivering it through the
            # background lane re-presents the agent's own text as a fake
            # completion (the 2026-08-06 echo class) — suppress it, dedup-mark
            # it, and consume its burst buffer. Genuinely different residue is
            # a real background completion and still delivers.
            final_norm = " ".join(str(out["final_text"]).split())
            while not inbox.empty():
                residue = inbox.get_nowait()
                if final_norm and type(residue).__name__ == "ResultMessage":
                    res_text = getattr(residue, "result", None)
                    if (
                        isinstance(res_text, str)
                        and " ".join(res_text.split()) == final_norm
                    ):
                        uuid = getattr(residue, "uuid", None)
                        if uuid:
                            self._unsolicited_delivered.add(uuid)
                        self._unsolicited_results += 1
                        buffered = len(self._unsolicited_text)
                        self._unsolicited_text.clear()
                        logger.warning(
                            "claude-agent-sdk: residue ResultMessage %s "
                            "matches this turn's own answer — suppressed, "
                            "never delivered as a background result "
                            "(%d buffered text(s) consumed)",
                            uuid, buffered,
                        )
                        continue
                self._handle_unsolicited(residue)
        out["billing_mode"] = self._reported_billing_mode()
        out["billing_evidence"] = dict(self._billing_evidence)
        return out

    # ---------- stream ownership ----------

    async def _reader_loop(self) -> None:
        """Sole owner of the SDK message stream for the client's lifetime.

        Hermes used to call `receive_response()` once per turn. That helper
        returns the OLDEST buffered ResultMessage — not necessarily this
        turn's — and the SDK offers no per-turn correlator (`query()` takes
        only a `session_id`, which is constant across turns; `ResultMessage`
        has a `uuid` with no request-side counterpart). Ordering is therefore
        the ONLY thing keeping answers matched to questions.

        The Claude Code CLI breaks that ordering by itself: when a background
        Agent task completes it injects a `<task-notification>` and runs a
        FULL assistant turn nobody asked for, leaving an unconsumed
        ResultMessage in the shared FIFO. Every later turn then pops a stale
        result. Because a desynced turn still looks like success — no error,
        no timeout, no interrupt — no retire path fires, so the skew is
        permanent and silent. Observed live 2026-07-25: four such turns left a
        constant off-by-4 for hours (dasbrow-hermes-coder#2).

        One reader for the whole client lifetime makes the rule enforceable:
        a message arriving while no turn is in flight is unsolicited BY
        DEFINITION, and gets routed away instead of poisoning the next turn."""
        end: _StreamEnd
        iterator = self._client.receive_messages().__aiter__()
        message_task = asyncio.ensure_future(iterator.__anext__())
        claim_task = asyncio.ensure_future(self._turn_claims.get())
        try:
            while True:
                done, _pending = await asyncio.wait(
                    (message_task, claim_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if message_task in done:
                    try:
                        message = message_task.result()
                    except StopAsyncIteration:
                        end = _StreamEnd(error=None)
                        break
                    self._observe_billing_evidence(message)
                    inbox = self._turn_inbox
                    if inbox is not None:
                        inbox.put_nowait(message)
                    else:
                        self._handle_unsolicited(message)
                    # Message wins ties with a claim. Re-arm first, then loop:
                    # every immediately available pre-claim FIFO entry is
                    # classified unsolicited before the claim is acknowledged.
                    message_task = asyncio.ensure_future(iterator.__anext__())
                    continue

                operation, inbox, claim_ack = claim_task.result()
                claim_task = asyncio.ensure_future(self._turn_claims.get())
                if claim_ack.cancelled():
                    continue
                if operation == "claim":
                    if self._turn_inbox is not None:
                        claim_ack.set_exception(
                            RuntimeError("SDK message stream already has a turn owner")
                        )
                        continue
                    self._turn_inbox = inbox
                elif operation == "release":
                    if self._turn_inbox is not inbox:
                        claim_ack.set_exception(
                            RuntimeError("SDK message stream release owner mismatch")
                        )
                        continue
                    self._turn_inbox = None
                else:  # pragma: no cover - internal invariant
                    claim_ack.set_exception(
                        RuntimeError(f"unknown SDK stream ownership operation: {operation}")
                    )
                    continue
                claim_ack.set_result(None)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            message_task.cancel()
            claim_task.cancel()
            raise
        except Exception as exc:  # pragma: no cover - stream torn down
            logger.debug(
                "claude-agent-sdk reader loop ended: %s",
                _safe_sdk_error_text(exc),
            )
            end = _StreamEnd(error=_safe_sdk_error_text(exc))
        finally:
            if not message_task.done():
                message_task.cancel()
            if not claim_task.done():
                claim_task.cancel()
        # The stream is gone (CLI exited or transport died). Mark it before
        # acknowledging every queued ownership operation.  A foreground turn
        # can pass its initial stream check, enqueue a claim behind buffered
        # messages, and then lose the reader to EOF; resolving the claim makes
        # it re-check this terminal state instead of waiting out turn_timeout.
        self._stream_ended = end
        pending_claims = []
        if claim_task.done() and not claim_task.cancelled():
            try:
                pending_claims.append(claim_task.result())
            except Exception:  # pragma: no cover - queue task failed
                pass
        claims = self._turn_claims
        if claims is not None:
            while True:
                try:
                    pending_claims.append(claims.get_nowait())
                except asyncio.QueueEmpty:
                    break
        for _operation, _owner, claim_ack in pending_claims:
            if not claim_ack.done():
                claim_ack.set_result(None)
        inbox = self._turn_inbox
        if inbox is not None:
            inbox.put_nowait(end)

    def _handle_unsolicited(self, message: Any) -> None:
        """Route a message that arrived with no turn in flight.

        These are real CLI output (typically a finished background Agent task
        reporting in). They answer nothing Hermes asked, so they must never
        enter a turn's result — but their CONTENT is completed work the user
        is waiting on: with a delivery callback wired, capture each top-level
        assistant message's text and hand the FULL burst over as an ordered
        list on the terminal ResultMessage (uuid-deduped). Without a
        callback, the historical WARN-drop stands."""
        if isinstance(message, _StreamEnd):
            return
        sid = getattr(message, "session_id", None)
        if sid:
            self._session_id = sid
        name = type(message).__name__
        if name == "ResultMessage":
            self._unsolicited_results += 1
            if self._on_unsolicited_result is None:
                self._unsolicited_text.clear()
                logger.warning(
                    "claude-agent-sdk: dropped unsolicited ResultMessage (no "
                    "turn in flight, total=%d) — CLI-initiated turn; see "
                    "dasbrow-hermes-coder#2",
                    self._unsolicited_results,
                )
                return
            uuid = getattr(message, "uuid", None)
            if uuid and uuid in self._unsolicited_delivered:
                self._unsolicited_text.clear()
                logger.debug(
                    "claude-agent-sdk: duplicate unsolicited ResultMessage "
                    "%s ignored", uuid,
                )
                return
            result_text = getattr(message, "result", None)
            texts = list(self._unsolicited_text)
            self._unsolicited_text.clear()
            if isinstance(result_text, str) and result_text.strip():
                # The CLI's result text repeats the turn's final assistant
                # message — never hand the same text over twice.
                if not texts or texts[-1] != result_text:
                    texts.append(result_text)
            if uuid:
                self._unsolicited_delivered.add(uuid)
            if not texts:
                logger.warning(
                    "claude-agent-sdk: unsolicited ResultMessage carried no "
                    "text (total=%d) — nothing to deliver",
                    self._unsolicited_results,
                )
                return
            logger.info(
                "claude-agent-sdk: delivering unsolicited result burst "
                "(background task finished, total=%d, %d message(s), "
                "%d chars)",
                self._unsolicited_results, len(texts),
                sum(len(t) for t in texts),
            )
            try:
                self._on_unsolicited_result(texts)
            except Exception:
                logger.warning(
                    "claude-agent-sdk: unsolicited-result delivery callback "
                    "raised — answer may be lost", exc_info=True,
                )
        elif name == "AssistantMessage":
            # Buffer top-level text, one entry per message so the burst
            # delivers in message granularity; subagent streams
            # (parent_tool_use_id set) are noise — the same gate
            # _forward_stream_delta uses.
            if (
                self._on_unsolicited_result is not None
                and not getattr(message, "parent_tool_use_id", None)
            ):
                parts = [
                    getattr(block, "text", "") or ""
                    for block in getattr(message, "content", None) or []
                    if type(block).__name__ == "TextBlock"
                ]
                message_text = "\n".join(p for p in parts if p)
                if message_text:
                    self._unsolicited_text.append(message_text)
            logger.info(
                "claude-agent-sdk: unsolicited %s outside a turn", name,
            )
        else:
            logger.debug(
                "claude-agent-sdk: unsolicited %s outside a turn", name,
            )

    def _start_reader(self) -> None:
        """Spawn the reader on the session loop. Idempotent."""
        if self._reader_task is not None:
            return

        async def _spawn() -> Any:
            self._turn_claims = asyncio.Queue()
            return asyncio.ensure_future(self._reader_loop())

        self._reader_task = self._run_coro(_spawn(), timeout=10.0)

    def _stop_reader(self) -> None:
        task = self._reader_task
        self._reader_task = None
        if task is None or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(task.cancel)
        except Exception:  # pragma: no cover - loop already gone
            pass

    def _forward_stream_delta(self, message: Any) -> None:
        """Relay a top-level text delta to the display callback (never the
        transcript). Subagent streams (parent_tool_use_id set) stay quiet."""
        if self._on_stream_delta is None:
            return
        if getattr(message, "parent_tool_use_id", None):
            return
        event = getattr(message, "event", None) or {}
        if event.get("type") != "content_block_delta":
            return
        delta = event.get("delta") or {}
        if delta.get("type") != "text_delta":
            return
        text = delta.get("text")
        if not text:
            return
        try:
            self._on_stream_delta(text)
        except Exception:  # pragma: no cover - display callback
            logger.debug("stream delta callback raised", exc_info=True)

    def _notify_interim_assistant(self, message: Any) -> None:
        """Relay completed tool-adjacent assistant prose as commentary."""
        if getattr(message, "parent_tool_use_id", None):
            return
        with self._turn_callback_lock:
            callback = self._on_interim_assistant
        if callback is None:
            return
        if type(message).__name__ != "AssistantMessage":
            return
        blocks = list(getattr(message, "content", None) or [])
        if not any(type(block).__name__ == "ToolUseBlock" for block in blocks):
            return
        text = "\n".join(
            str(getattr(block, "text", "") or "")
            for block in blocks
            if type(block).__name__ == "TextBlock" and getattr(block, "text", "")
        ).strip()
        if not text:
            return
        try:
            callback(text)
        except Exception:  # pragma: no cover - display callback
            logger.debug("interim assistant callback raised", exc_info=True)

    def _notify_tool_iteration(self) -> None:
        with self._turn_callback_lock:
            callback = self._on_tool_iteration
        if callback is None:
            return
        try:
            callback()
        except Exception:  # pragma: no cover - display callback
            logger.debug("tool-iteration callback raised", exc_info=True)

    def _notify_tool_started(self, message: Any) -> None:
        """Bridge ToolUseBlocks to Hermes tool-progress (gateway breadcrumbs),
        mirroring codex_runtime._codex_note_to_tool_progress (#38835)."""
        if self._on_tool_started is None:
            return
        if type(message).__name__ != "AssistantMessage":
            return
        for block in getattr(message, "content", None) or []:
            if type(block).__name__ != "ToolUseBlock":
                continue
            name = getattr(block, "name", "") or "unknown"
            args = getattr(block, "input", None) or {}
            if not isinstance(args, dict):
                args = {"input": args}
            preview = _tool_preview(name, args)
            try:
                self._on_tool_started(name, preview, args)
            except Exception:  # pragma: no cover - display callback
                logger.debug("tool-progress callback raised", exc_info=True)

    def build_option_fields(self) -> dict[str, Any]:
        """The ClaudeAgentOptions field dict — plain data so tests can assert
        on it without importing the SDK."""
        mcp_servers: dict[str, Any] = {}
        # Hybrid in-process MCP bridge (ported from PR #56413) — exposes the
        # full Hermes tool registry, including proxified third-party MCP
        # servers, which the stdio `hermes-tools` wrapper cannot reach
        # because credentials live in the gateway process env and don't
        # propagate to the subprocess. Activation requires ALL of:
        #   1. the operator opted in via
        #      ``agent.claude_agent_sdk.hybrid_mcp_bridge: true``
        #      (checked here as well as at the runtime call site);
        #   2. the caller supplied both ``agent`` and ``tools``.
        # The bridge splits into two in-process servers:
        #   - ``hermes-tools`` — stdio-legacy names (see
        #     ``HERMES_TOOLS_LEGACY_NAMES``). Preserves operator grants
        #     stored in ``~/.claude/settings.json`` that key on
        #     ``mcp__hermes-tools__<tool>``: a box on
        #     ``permission_mode: default`` would otherwise face an
        #     approval storm for tools it already granted.
        #   - ``hermes-hybrid`` — everything else (proxified MCPs +
        #     agent-level tools).
        # The exclude list from config is applied to BOTH buckets so an
        # operator can keep the wide bridge for proxified MCPs without
        # inheriting a specific tool (delegate_task, cron_*, terminal, ...).
        hybrid_active = False
        hybrid_opted_in = _provider_flag("hybrid_mcp_bridge", default=False)
        if hybrid_opted_in and self._agent is not None and self._tools:
            try:
                from agent.transports.hermes_hybrid_mcp import (
                    HERMES_TOOLS_SERVER,
                    HYBRID_SERVER,
                    build_hybrid_mcp_server,
                )
                from agent.transports.hermes_tool_exposure import (
                    HERMES_TOOLS_LEGACY_NAMES,
                )

                exclude_names = _configured_hybrid_exclude()
                mcp_servers[HERMES_TOOLS_SERVER] = build_hybrid_mcp_server(
                    self._agent,
                    self._tools,
                    server_name=HERMES_TOOLS_SERVER,
                    only_names=HERMES_TOOLS_LEGACY_NAMES,
                    exclude_names=exclude_names,
                )
                mcp_servers[HYBRID_SERVER] = build_hybrid_mcp_server(
                    self._agent,
                    self._tools,
                    server_name=HYBRID_SERVER,
                    exclude_names=(
                        list(exclude_names) + list(HERMES_TOOLS_LEGACY_NAMES)
                    ),
                )
                hybrid_active = True
            except Exception as exc:  # noqa: BLE001
                # Never break session start on hybrid bridge failure — the
                # stdio wrapper still provides the curated tool set.
                logger.warning(
                    "hybrid MCP bridge failed to build (%s) — falling back "
                    "to stdio hermes-tools only",
                    exc,
                )
        # The stdio ``hermes-tools`` wrapper exposes ~25 curated tools that
        # the hybrid bridge already re-exposes under the same server name
        # (``hermes-tools``) when active — registering both concurrently
        # would send Claude two copies of every curated tool under the same
        # server. Skip stdio when hybrid is active. (Hybrid failure above
        # falls back to stdio via ``hybrid_active`` staying False.)
        if self._include_hermes_tools and not hybrid_active:
            mcp_servers["hermes-tools"] = _build_hermes_tools_mcp_config(
                hermes_session_id=self._hermes_session_id
            )

        # Headerless third-party HTTP MCPs configured in Hermes (config.yaml
        # mcp_servers.<name>.url) can be exposed directly to the SDK because
        # the registry snapshot may not contain their late/proxified tools.
        # This is part of the SAME wide-surface security choice as the hybrid
        # bridge, never an independent back door: discovery runs only after
        # the explicit opt-in passed and both in-process bridge buckets built
        # successfully. Header-bearing entries are refused by the loader
        # because the SDK puts its MCP config in the Claude CLI argv.
        if hybrid_active:
            for entry_name, entry_cfg in _http_mcp_entries_from_config().items():
                if entry_name in mcp_servers:
                    continue
                mcp_servers[entry_name] = entry_cfg

        system_prompt: Any = {"type": "preset", "preset": "claude_code"}
        if self._system_prompt_append:
            system_prompt = {
                "type": "preset",
                "preset": "claude_code",
                "append": self._system_prompt_append,
            }

        # This wrapper is Hermes' mandatory permission-policy owner, including
        # when runtime callback selection legitimately produced no downstream
        # callback.  Callbackless non-bypass sessions intentionally fail closed
        # here instead of falling through to SDK-native prompting/permission
        # behavior; bounded readers and trusted bypass still resolve only after
        # canonical validation, correlation validation, and immutable floors.
        can_use_tool = self._make_can_use_tool()

        # Metered-vector scrub for the spawned CLI (see _METERED_ENV_DENYLIST).
        # agent.claude_agent_sdk.allow_metered_key: true is the operator's
        # explicit "bill me metered" opt-in (the same flag the startup guard
        # honors), so it disables the scrub too — otherwise the documented
        # escape hatch would hand the CLI an environment with the key blanked.
        env_overrides = _sdk_env_overrides(
            metered_allowed=self._allow_metered
        )

        fields = {
            "model": self._model,
            "cwd": self._cwd,
            "permission_mode": self._permission_mode,
            "system_prompt": system_prompt,
            "mcp_servers": mcp_servers,
            "max_budget_usd": self._max_budget_usd,
            "can_use_tool": can_use_tool,
            "env": env_overrides,
            # Explicit SDK isolation. None (the SDK default, verified against
            # claude-agent-sdk 0.2.120) means the CLI loads ALL filesystem
            # settings — ~/.claude/settings.json, .claude/settings.json,
            # .claude/settings.local.json — so ambient operator/project
            # settings could re-permission tools or install hooks UNDERNEATH
            # the permission_mode/can_use_tool posture configured above. The
            # empty list is the SDK's documented isolation mode: settings
            # come from Hermes config only. Accepted side effect: "project"
            # is also what loads CLAUDE.md, which this runtime doesn't want —
            # it composes its own system-prompt append. Operators whose
            # deployment stores tool grants in ~/.claude/settings.json
            # (unattended cron turns with nobody to answer a prompt) opt
            # back in via agent.claude_agent_sdk.setting_sources — see
            # _configured_setting_sources.
            "setting_sources": _configured_setting_sources(),
            # Explicit, because the SDK's 1 MiB default is below what this
            # lane's own tool results routinely produce and overflowing it
            # kills the turn outright — see _configured_max_buffer_size.
            "max_buffer_size": _configured_max_buffer_size(),
            # AskUserQuestion has no native answer bridge. Native Read stays
            # behind the protected-path-aware, bounded Hermes MCP surface in
            # every supported SDK permission mode.
            "disallowed_tools": ["AskUserQuestion", "Read"],
        }
        # The CLI owns compaction on this lane, so its PreCompact hook is the
        # only honest signal that a turn stalled to compact. Registered only
        # when a consumer asked for it, so the default option set is unchanged.
        _compaction_hooks = self._build_compaction_hooks()
        if _compaction_hooks is not None:
            fields["hooks"] = _compaction_hooks
        if self._resume_session_id:
            fields["resume"] = self._resume_session_id
        # Default OFF (upstream-conservative): partial messages only when the
        # operator opts in via agent.claude_agent_sdk.streaming in config.yaml.
        # Reads the __init__ snapshot so option and quiet-watchdog semantics
        # can never diverge across a mid-session config edit.
        if self._streaming:
            fields["include_partial_messages"] = True
        return fields

    def _build_client(self) -> Any:
        fields = self.build_option_fields()
        if self._client_factory is not None:
            return self._client_factory(options=fields)
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        return ClaudeSDKClient(options=ClaudeAgentOptions(**fields))

    def _sdk_approval_bypass_active(self) -> bool:
        """Resolve only trusted session/process/config bypass intent."""
        if self._sdk_approval_bypass_requested:
            return True
        provider = self._approval_bypass_provider
        if provider is not None:
            try:
                return provider() is True
            except Exception:
                return False
        try:
            from tools.approval import is_approval_bypass_active_for_session

            return is_approval_bypass_active_for_session(
                self._hermes_session_id or "",
            )
        except Exception:
            return False

    def _make_can_use_tool(self) -> Any:
        """Bridge SDK permission requests onto Hermes' approval callback.
        Fail-closed: any callback failure denies.

        Silent-deny observability (P2.d): every deny that was NOT an
        operator's choice is logged at INFO here — this is the choke point
        every SDK-lane deny transits with tool name and honest reason in
        hand (the 2026-08-06 incident's silent denies had no log line at
        all). "denied by user" (with or without ": <text>") appears IFF a
        human chose deny — W8/W11 reserved that wording — so the prefix is
        the discriminator; operator denies are not silent and not logged
        here. Honesty boundary: settings deny-rule hits on the SDK side
        (the CLI consulting ~/.claude/settings.json deny rules, e.g. an
        installer-only skills-dir rule) never invoke can_use_tool and never
        transit hermes code — they are UNLOGGABLE here by construction;
        operator-facing relief for that class is a separate deny-notice
        feature decision."""
        approval_callback = self._approval_callback
        hermes_session_id = self._hermes_session_id

        async def _can_use_tool(tool_name: str, tool_input: dict, context: Any):
            from claude_agent_sdk import (
                PermissionResultAllow,
                PermissionResultDeny,
            )

            # Capture the watch OBJECT at entry: an approval that outlives
            # its turn (orphaned wait, self-expiring at approvals.timeout)
            # must decrement the watch it suspended, never a later turn's.
            # None = no Hermes turn in flight (unsolicited CLI-side turn) —
            # nothing to suspend.
            watch = self._turn_watch
            if watch is not None:
                # A human is being asked — their think time is not the
                # turn's silence. Both watchdog rules stand down until the
                # tap (or the approval machinery's own timeout) resolves.
                watch.approval_begin()
            try:
                return await self._resolve_can_use_tool(
                    tool_name, tool_input, context,
                    approval_callback, hermes_session_id,
                    PermissionResultAllow, PermissionResultDeny,
                )
            finally:
                if watch is not None:
                    watch.approval_end()

        return _can_use_tool

    async def _resolve_can_use_tool(
        self,
        tool_name: str,
        tool_input: dict,
        context: Any,
        approval_callback: Any,
        hermes_session_id: Any,
        PermissionResultAllow: Any,
        PermissionResultDeny: Any,
    ) -> Any:
        try:
            canonical_tool_input = _canonical_sdk_tool_request(tool_name, tool_input)
            checked_request = validate_canonical_sdk_request_serialization(
                canonical_tool_input,
            )
            presentation = safe_sdk_tool_presentation_from_canonical(
                canonical_tool_input,
            )
        except Exception:
            checked_request = None
            presentation = None
        if checked_request is None or presentation is None:
            return PermissionResultDeny(message="canonical request is unassessable")
        frozen_tool_input = checked_request[1]["tool_input"]
        # Validate correlation metadata before every policy decision. Legacy
        # callbacks retain their exact ABI, but no floor or bypass decision can
        # be made from callback markers or callback-controlled text.
        safe_tool_use_id = _safe_sdk_tool_use_id(context)
        callback_command, callback_description = presentation
        log_tool_identity = (
            tool_name if tool_name in _SDK_FIXED_LOG_TOOL_IDENTITIES else "sdk-tool"
        )
        # Native Read remains unavailable even if this callback is invoked
        # directly despite the SDK disallowed_tools option.
        if tool_name == "Read":
            return PermissionResultDeny(message="native SDK Read is disallowed")
        if tool_name == "Bash":
            try:
                from tools.approval_sdk_gateway import sdk_bash_immutable_floor_reason

                floor_reason = sdk_bash_immutable_floor_reason(
                    frozen_tool_input.get("command"),
                )
            except Exception:
                floor_reason = "canonical request is unassessable"
            if floor_reason is not None:
                return PermissionResultDeny(message="approval denied by callback")
        if tool_name in _SDK_AUTO_ALLOWED_MCP_TOOLS:
            return PermissionResultAllow(updated_input=frozen_tool_input)
        if self._sdk_approval_bypass_active():
            return PermissionResultAllow(updated_input=frozen_tool_input)
        if approval_callback is None:
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, "approval callback unavailable",
            )
            return PermissionResultDeny(message="approval callback unavailable")
        try:
            kwargs: dict = {"allow_permanent": False}
            try:
                from tools.approval_sdk_gateway import (
                    is_trusted_sdk_gateway_approval_callback,
                )

                trusted_gateway_callback = (
                    is_trusted_sdk_gateway_approval_callback(approval_callback)
                )
            except Exception:
                trusted_gateway_callback = False
            # Correlation and canonical JSON are additive only for callbacks
            # that explicitly advertise the corresponding SDK ABI extension.
            if getattr(approval_callback, "_accepts_tool_use_id", False):
                kwargs["tool_use_id"] = safe_tool_use_id
            if getattr(approval_callback, "_accepts_canonical_tool_input", False):
                kwargs["canonical_tool_input"] = canonical_tool_input
            result = await asyncio.to_thread(
                approval_callback,
                callback_command,
                callback_description,
                **kwargs,
            )
        except Exception:
            logger.warning("SDK approval callback failed at protected boundary")
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, "approval callback failed",
            )
            return PermissionResultDeny(message="approval callback failed")
        # Widened callback contract: a plain choice string, or a dict
        # {"choice": str, "reason": str} carrying an honest deny reason
        # (no-approver / timeout / notify-failure / teardown-expiry).
        # "denied by user" is reserved for a real human deny — a
        # reason-bearing deny must never be attributed to the user.
        try:
            reason = None
            operator_denied = False
            reason_shape = False
            operator_denial_shape = False
            choice = result
            if type(result) is dict:
                # The callback result is untrusted. Protocol dicts have exactly
                # one of three tiny widths; reject all others before collecting,
                # copying, hashing, or otherwise traversing callback keys:
                # {choice}, {choice, reason}, or the trusted gateway-only
                # {choice, operator_denial, reason} structural denial.
                width = len(result)
                if width == 1 and "choice" in result:
                    choice = result["choice"]
                elif width == 2 and "choice" in result and "reason" in result:
                    choice = result["choice"]
                    reason = result["reason"]
                    reason_shape = True
                elif (
                    width == 3
                    and "choice" in result
                    and "operator_denial" in result
                    and "reason" in result
                ):
                    choice = result["choice"]
                    reason = result["reason"]
                    reason_shape = True
                    operator_denial_shape = True
                else:
                    raise ValueError("malformed callback result shape")
            elif type(result) is not str:
                raise TypeError("unsupported callback result")
            if not _is_bounded_sdk_callback_string(
                choice,
                _SDK_CALLBACK_CHOICE_MAX_UTF8_BYTES,
                allow_space=False,
            ) or (
                reason is not None
                and not _is_bounded_sdk_callback_string(
                    reason,
                    _SDK_CALLBACK_REASON_MAX_UTF8_BYTES,
                    allow_space=True,
                )
            ):
                raise TypeError("malformed callback result")
            if reason_shape and choice != "deny":
                raise ValueError("reason is valid only for deny")
            if operator_denial_shape:
                operator_denied = (
                    choice == "deny"
                    and result["operator_denial"] is True
                    and trusted_gateway_callback
                )
                if not operator_denied:
                    raise ValueError("untrusted operator-denial result")
            if choice not in {"once", "session", "always", "deny", "timeout"}:
                raise ValueError("unknown callback choice")
        except Exception:
            logger.warning("SDK approval callback failed at protected boundary")
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, "approval callback failed",
            )
            return PermissionResultDeny(message="approval callback failed")
        if choice in ("once", "session", "always"):
            return PermissionResultAllow(updated_input=frozen_tool_input)
        if operator_denied:
            message = "denied by user"
            if reason:
                message = f"{message}: {reason}"
        elif choice == "timeout":
            message = "approval timed out — no operator response"
        else:
            safe_reason = _safe_sdk_deny_log_reason(reason)
            message = (
                safe_reason
                if safe_reason != "non-operator denial"
                else "approval denied by callback"
            )
        if not operator_denied:
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, _safe_sdk_deny_log_reason(message),
            )
        return PermissionResultDeny(message=message)

    # ---------- loop-thread plumbing ----------

    def _start_loop_thread(self) -> None:
        if self._loop_thread is not None:
            return
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=_run, name="claude-sdk-loop", daemon=True
        )
        thread.start()
        ready.wait(timeout=10)
        self._loop = loop
        self._loop_thread = thread

    def _stop_loop_thread(self) -> None:
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:  # pragma: no cover
                pass
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
        self._loop = None
        self._loop_thread = None

    def _run_coro(self, coro: Any, *, timeout: float) -> Any:
        import concurrent.futures

        assert self._loop is not None, "loop thread not started"
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except (TimeoutError, concurrent.futures.TimeoutError):
            future.cancel()
            raise asyncio.TimeoutError(f"coroutine exceeded {timeout}s")


# Ceiling on the SDK transport's close ladder (5s stdin lock + 5s graceful +
# 5s SIGTERM + 5s SIGKILL), plus slack.
_SDK_DISCONNECT_TIMEOUT_S = 25.0


def _sdk_child_pid(client: Any) -> Optional[int]:
    """OS pid of the CLI subprocess behind an SDK client, if reachable."""
    try:
        proc = getattr(getattr(client, "_transport", None), "_process", None)
        pid = getattr(proc, "pid", None)
        return int(pid) if pid else None
    except Exception:
        return None


def _own_sdk_child_process(pid: int) -> Any:
    """Return the live psutil Process when ``pid`` is our direct child.

    psutil is Hermes' canonical cross-platform PID layer.  Keeping the
    ``Process`` object also protects the TERM→KILL ladder against PID reuse:
    psutil checks the process identity before destructive operations.
    """
    import psutil

    try:
        process = psutil.Process(int(pid))
        if process.ppid() != os.getpid():
            return None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except (psutil.Error, OSError, TypeError, ValueError):
        return None


def _is_own_sdk_child(pid: int) -> bool:
    """Guard against PID reuse: only reap a live child of this process."""
    return _own_sdk_child_process(pid) is not None


def _force_kill_sdk_child(pid: Optional[int], *, process: Any = None) -> None:
    """Last-resort reap when disconnect() times out and strands the CLI child."""
    if not pid:
        return
    if process is None:
        process = _own_sdk_child_process(pid)
    if process is None:
        return
    import psutil

    try:
        if (
            int(process.pid) != int(pid)
            or process.ppid() != os.getpid()
            or not process.is_running()
            or process.status() == psutil.STATUS_ZOMBIE
        ):
            return
        process.terminate()
    except (psutil.Error, OSError, TypeError, ValueError):
        return
    try:
        process.wait(timeout=5.0)
        logger.info("claude-agent-sdk stranded child %s reaped (terminate)", pid)
        return
    except psutil.NoSuchProcess:
        return
    except psutil.TimeoutExpired:
        pass
    except (psutil.Error, OSError):
        return
    try:
        # is_running() performs psutil's identity check, so a reused PID is
        # never killed as though it were the original CLI child.
        if process.is_running():
            process.kill()
            logger.warning(
                "claude-agent-sdk stranded child %s required forced kill", pid
            )
    except (psutil.NoSuchProcess, psutil.Error, OSError):
        pass


def _tool_preview(name: str, args: dict) -> str:
    """Short human preview of a tool call for progress breadcrumbs."""
    for key in ("command", "file_path", "path", "url", "query", "prompt"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value[:120]
    return name


def _sdk_image_content_block(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Translate one Hermes/OpenAI image part to an SDK-native image block.

    The Agent SDK accepts structured user messages in streaming-input mode.
    Preserve base64 data URIs and http(s) image URLs instead of pretending a
    text-only query still carries the attachment. Return ``None`` for an
    unsupported/malformed source; callers add an explicit user-visible marker.
    """
    import base64 as _base64
    import re as _re
    from urllib.parse import urlsplit as _urlsplit

    source = item.get("source")
    if isinstance(source, dict):
        source_type = source.get("type")
        if source_type == "base64":
            media_type = source.get("media_type")
            data = source.get("data")
            if (
                isinstance(media_type, str)
                and media_type.startswith("image/")
                and isinstance(data, str)
                and data
            ):
                try:
                    _base64.b64decode(data, validate=True)
                except Exception:
                    return None
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
        elif source_type == "url":
            raw_url = source.get("url")
            if isinstance(raw_url, str):
                parsed = _urlsplit(raw_url)
                if parsed.scheme in {"http", "https"} and parsed.netloc:
                    return {
                        "type": "image",
                        "source": {"type": "url", "url": raw_url},
                    }

    raw_url: Any = item.get("image_url")
    if isinstance(raw_url, dict):
        raw_url = raw_url.get("url")
    if not isinstance(raw_url, str):
        raw_url = item.get("url")
    if not isinstance(raw_url, str) or not raw_url:
        return None

    data_match = _re.fullmatch(
        r"data:(image/[A-Za-z0-9.+-]+);base64,(.+)",
        raw_url,
        flags=_re.DOTALL,
    )
    if data_match:
        media_type, data = data_match.groups()
        try:
            _base64.b64decode(data, validate=True)
        except Exception:
            return None
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    parsed = _urlsplit(raw_url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return {
            "type": "image",
            "source": {"type": "url", "url": raw_url},
        }
    return None


def _coerce_turn_input(user_input: Any) -> Any:
    """Preserve Hermes/OpenAI rich images while keeping text-only turns plain.

    ClaudeSDKClient.query accepts either a string or an async stream of SDK
    message dictionaries. A content list with at least one valid image becomes
    SDK-native blocks; a text-only list keeps the historical joined-string
    behavior. Invalid image sources become truthful text, never a fabricated
    claim that an image is attached.
    """
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        blocks: list[dict[str, Any]] = []
        has_valid_image = False
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    blocks.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                if item is not None:
                    blocks.append({"type": "text", "text": str(item)})
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    blocks.append({"type": "text", "text": str(text)})
            elif item_type in {"image", "image_url", "input_image"}:
                image = _sdk_image_content_block(item)
                if image is not None:
                    blocks.append(image)
                    has_valid_image = True
                else:
                    logger.warning(
                        "claude-agent-sdk: image attachment has an unsupported "
                        "or malformed source; sending an explicit unavailable marker"
                    )
                    blocks.append({
                        "type": "text",
                        "text": (
                            "[image attachment unavailable: unsupported or "
                            "malformed source]"
                        ),
                    })
        if has_valid_image:
            return blocks
        return "\n\n".join(
            str(block.get("text") or "")
            for block in blocks
            if block.get("type") == "text"
        ).strip()
    return "" if user_input is None else str(user_input)


async def _sdk_user_message_stream(content: list[dict[str, Any]]):
    """One-message async stream accepted by ``ClaudeSDKClient.query``."""
    yield {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }
