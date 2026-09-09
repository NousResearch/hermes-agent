"""Fail-closed contract for operator-required plugin lifecycle hooks.

The plugin manager remains the delivery seam.  This module owns only the
closed policy/result types and the bounded per-turn failure latch so optional
plugin hooks can retain their compatibility behavior.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping


REQUIRED_LIFECYCLE_CAPABILITY = "required-lifecycle-hooks/v1"
REQUIRED_LIFECYCLE_HOOKS = frozenset(
    {
        "pre_llm_call",
        "pre_tool_call",
        "post_tool_call",
        "transform_llm_output",
    }
)
REQUIRED_LIFECYCLE_FAILURE_TEXT = (
    "Hermes blocked this turn because a required lifecycle guard was unavailable."
)
_PROVIDER_CONTINUITY_FIELDS = frozenset(
    {
        "reasoning_content",
        "reasoning_details",
        "anthropic_content_blocks",
        "codex_reasoning_items",
        "codex_message_items",
    }
)


def _provider_continuity_key(message: Mapping[str, Any]) -> tuple[Any, ...] | None:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    call_ids: list[str] = []
    for call in tool_calls:
        if not isinstance(call, Mapping):
            return None
        call_id = call.get("call_id") or call.get("id")
        if not isinstance(call_id, str) or not call_id:
            return None
        call_ids.append(call_id)
    return (
        message.get("timestamp"),
        message.get("finish_reason"),
        tuple(call_ids),
    )
_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?\Z")
_MAX_PLUGINS = 16
_MAX_REGISTRATIONS_PER_HOOK = 8
_MAX_LATCHES = 1024


class RequiredLifecycleError(RuntimeError):
    """Stable, redacted failure raised by a required lifecycle guard."""

    def __init__(self, reason_code: str, hook_name: str = "") -> None:
        self.reason_code = reason_code
        self.hook_name = hook_name
        super().__init__(f"{reason_code}:{hook_name or 'lifecycle'}")


@dataclass(frozen=True)
class RequiredHookResult:
    """Host-verifiable acknowledgement from one exact required callback."""

    registration_id: str
    result: Any = None


def required_hook_result(registration_id: str, result: Any = None) -> RequiredHookResult:
    """Return the acknowledgement envelope required callbacks must emit."""
    if not isinstance(registration_id, str) or not _IDENTIFIER.fullmatch(
        registration_id
    ):
        raise ValueError("required lifecycle registration id is invalid")
    return RequiredHookResult(registration_id=registration_id, result=result)


def quarantine_required_provider_fields(agent: Any, message: dict[str, Any]) -> None:
    """Keep opaque continuity data in agent-owned, turn-local memory only."""
    private = {
        key: message.pop(key)
        for key in _PROVIDER_CONTINUITY_FIELDS
        if key in message
    }
    tool_extras: list[tuple[int, Any]] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for index, call in enumerate(tool_calls):
            if isinstance(call, dict) and "extra_content" in call:
                tool_extras.append((index, call.pop("extra_content")))
    key = _provider_continuity_key(message)
    if (private or tool_extras) and key is not None:
        continuity = getattr(agent, "_required_provider_continuity", None)
        if not isinstance(continuity, dict):
            continuity = {}
            setattr(agent, "_required_provider_continuity", continuity)
        continuity[key] = (private, tuple(tool_extras))


def restore_required_provider_fields(
    agent: Any,
    source: dict[str, Any],
    message: dict[str, Any],
) -> None:
    """Restore quarantined fields only on a request-local provider clone."""
    continuity = getattr(agent, "_required_provider_continuity", None)
    key = _provider_continuity_key(source)
    record = continuity.get(key) if isinstance(continuity, dict) else None
    if not isinstance(record, tuple) or len(record) != 2:
        return
    private, tool_extras = record
    if not isinstance(private, dict) or not set(private) <= _PROVIDER_CONTINUITY_FIELDS:
        return
    if not isinstance(tool_extras, tuple):
        return
    message.update(private)
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for item in tool_extras:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], int)
            or isinstance(item[0], bool)
            or item[0] < 0
            or item[0] >= len(tool_calls)
            or not isinstance(tool_calls[item[0]], dict)
        ):
            return
        tool_calls[item[0]]["extra_content"] = item[1]


def scrub_required_provider_fields(agent: Any) -> None:
    """Destroy all ephemeral provider continuity at terminalization."""
    setattr(agent, "_required_provider_continuity", {})


def parse_required_lifecycle_policy(
    config: Mapping[str, Any],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Parse the closed ``plugins.required_lifecycle_hooks`` config subtree.

    Absence means compatibility mode.  Presence is strict: malformed policy
    raises so it can never silently degrade into "no requirements".
    """
    plugins = config.get("plugins")
    if not isinstance(plugins, Mapping) or "required_lifecycle_hooks" not in plugins:
        return {}
    raw = plugins.get("required_lifecycle_hooks")
    if not isinstance(raw, Mapping) or not raw or len(raw) > _MAX_PLUGINS:
        raise RequiredLifecycleError("required_lifecycle_policy_invalid")
    parsed: dict[str, dict[str, tuple[str, ...]]] = {}
    for plugin_key, hooks in raw.items():
        if not isinstance(plugin_key, str) or not _IDENTIFIER.fullmatch(plugin_key):
            raise RequiredLifecycleError("required_lifecycle_policy_invalid")
        if not isinstance(hooks, Mapping) or set(hooks) != REQUIRED_LIFECYCLE_HOOKS:
            raise RequiredLifecycleError("required_lifecycle_policy_invalid")
        parsed_hooks: dict[str, tuple[str, ...]] = {}
        for hook_name in sorted(REQUIRED_LIFECYCLE_HOOKS):
            ids = hooks.get(hook_name)
            if (
                not isinstance(ids, list)
                or not ids
                or len(ids) > _MAX_REGISTRATIONS_PER_HOOK
                or any(
                    not isinstance(item, str) or not _IDENTIFIER.fullmatch(item)
                    for item in ids
                )
                or len(set(ids)) != len(ids)
                or (
                    hook_name in {"pre_tool_call", "transform_llm_output"}
                    and len(ids) != 1
                )
            ):
                raise RequiredLifecycleError("required_lifecycle_policy_invalid")
            parsed_hooks[hook_name] = tuple(ids)
        parsed[plugin_key] = parsed_hooks
    for authority_hook in ("pre_tool_call", "transform_llm_output"):
        if (
            sum(
                len(hooks[authority_hook])
                for hooks in parsed.values()
            )
            != 1
        ):
            raise RequiredLifecycleError("required_lifecycle_policy_invalid")
    return parsed


def validate_required_semantic_result(hook_name: str, result: Any) -> bool:
    """Return whether an acknowledged result fits the existing hook contract."""
    if hook_name == "pre_llm_call":
        return result is None or (
            isinstance(result, str) and bool(result.strip())
        ) or (
            isinstance(result, Mapping)
            and set(result) == {"context"}
            and isinstance(result.get("context"), str)
            and bool(result["context"].strip())
        )
    if hook_name == "post_tool_call":
        return result is None
    if hook_name == "transform_llm_output":
        return result is None or (isinstance(result, str) and bool(result))
    if hook_name == "pre_tool_call":
        # The downstream directive consumer intentionally accepts only a
        # concrete JSON-style object.  A custom Mapping must not be
        # acknowledged here and then silently ignored there.
        if not isinstance(result, dict):
            return False
        keys = set(result)
        if not keys:
            return True
        action = result.get("action")
        if action == "block":
            return (
                keys <= {"action", "message"}
                and isinstance(result.get("message"), str)
                and bool(result["message"])
            )
        if action == "approve":
            return keys <= {"action", "message", "rule_key"} and all(
                value is None or isinstance(value, str)
                for key, value in result.items()
                if key != "action"
            )
        if action == "modify":
            return keys == {"action", "args"} and isinstance(
                result.get("args"), dict
            )
        return False
    return False


class RequiredLifecycleLatch:
    """Thread-safe bounded failure memory keyed by profile/session/turn."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._failures: OrderedDict[tuple[str, str, str], RequiredLifecycleError] = (
            OrderedDict()
        )
        self._scope_failures: dict[str, RequiredLifecycleError] = {}
        self._active_generations: OrderedDict[tuple[str, str, str], int] = (
            OrderedDict()
        )

    @staticmethod
    def key(scope_key: str, payload: Mapping[str, Any]) -> tuple[str, str, str]:
        session_id = payload.get("session_id")
        turn_id = payload.get("turn_id")
        if not isinstance(session_id, str) or not session_id:
            raise RequiredLifecycleError("required_lifecycle_identity_missing")
        if not isinstance(turn_id, str) or not turn_id:
            raise RequiredLifecycleError("required_lifecycle_identity_missing")
        return (scope_key, session_id, turn_id)

    def check(self, key: tuple[str, str, str]) -> None:
        with self._lock:
            scope_failure = self._scope_failures.get(key[0])
            if scope_failure is not None:
                raise RequiredLifecycleError(
                    scope_failure.reason_code, scope_failure.hook_name
                )
            failure = self._failures.get(key)
        if failure is not None:
            raise RequiredLifecycleError(failure.reason_code, failure.hook_name)

    def begin(self, key: tuple[str, str, str], generation: int) -> None:
        """Bind a turn to the registration generation that admitted it."""
        self.check(key)
        with self._lock:
            previous = self._active_generations.get(key)
            if previous is None:
                self._active_generations[key] = generation
                self._active_generations.move_to_end(key)
                if len(self._active_generations) > _MAX_LATCHES:
                    self._active_generations.pop(key, None)
                    failure = RequiredLifecycleError(
                        "required_lifecycle_latch_capacity"
                    )
                    self._scope_failures[key[0]] = failure
                    raise RequiredLifecycleError(failure.reason_code)
                return
        if previous != generation:
            raise self.fail(
                key,
                "required_lifecycle_registration_changed",
                "lifecycle",
            )

    def is_active(self, key: tuple[str, str, str]) -> bool:
        with self._lock:
            return key in self._active_generations

    def has_active(self) -> bool:
        """Return whether any turn currently owns the registration lease."""
        with self._lock:
            return bool(self._active_generations)

    def check_generation(
        self,
        key: tuple[str, str, str],
        generation: int,
        hook_name: str,
    ) -> None:
        self.check(key)
        with self._lock:
            active_generation = self._active_generations.get(key)
        if active_generation is not None and active_generation != generation:
            raise self.fail(
                key,
                "required_lifecycle_registration_changed",
                hook_name,
            )

    def fail(
        self,
        key: tuple[str, str, str],
        reason_code: str,
        hook_name: str,
    ) -> RequiredLifecycleError:
        failure = RequiredLifecycleError(reason_code, hook_name)
        with self._lock:
            scope_failure = self._scope_failures.get(key[0])
            if scope_failure is not None:
                return RequiredLifecycleError(
                    scope_failure.reason_code, scope_failure.hook_name
                )
            self._failures[key] = failure
            self._failures.move_to_end(key)
            if len(self._failures) > _MAX_LATCHES:
                # Evicting an active failed turn would reopen enforcement.
                # Capacity exhaustion therefore closes this profile until its
                # next bounded lifecycle/reload instead.
                failure = RequiredLifecycleError(
                    "required_lifecycle_latch_capacity", hook_name
                )
                self._scope_failures[key[0]] = failure
        return failure

    def fail_uncorrelated(
        self,
        scope_key: str,
        reason_code: str,
        hook_name: str,
    ) -> RequiredLifecycleError:
        """Close a profile when a required event lacks turn correlation."""
        failure = RequiredLifecycleError(reason_code, hook_name)
        with self._lock:
            self._scope_failures[scope_key] = failure
        return failure

    def clear(self, key: tuple[str, str, str]) -> None:
        with self._lock:
            self._failures.pop(key, None)
            self._active_generations.pop(key, None)
