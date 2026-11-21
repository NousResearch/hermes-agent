"""One hook run: the callback ``invoke_hook()`` calls per firing, and the failure semantics that turn
a run of ``spawn.spawn_hook`` into the hook's contribution."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from agent.shell_hooks.spawn import spawn_hook
from agent.shell_hooks.spec import _BLOCKING_EVENTS, _TOOL_EVENTS, ShellHookSpec
from agent.shell_hooks.wire import _DEFAULT_BLOCK_MESSAGE, _is_json_object, _parse_response, _serialize_payload

logger = logging.getLogger(__name__)

# Exit code that signals "block this action" independent of stdout (Claude Code / Cursor).
BLOCK_EXIT_CODE = 2

_STDERR_MESSAGE_LIMIT = 400


def _make_callback(spec: ShellHookSpec) -> Callable[..., Optional[dict[str, Any]]]:
    """Build the closure that ``invoke_hook()`` will call per firing."""

    def _callback(**kwargs: Any) -> Optional[dict[str, Any]]:
        from agent.safe_worker_policy import safe_worker_enabled
        from tools.hosted_tool_policy import host_feature_enabled

        # A hosted scope's events carry the tenant's payload; a shell hook is a process on this host.
        if safe_worker_enabled() or not host_feature_enabled("shell_hooks"):
            return None
        if spec.event in _TOOL_EVENTS and not spec.matches_tool(kwargs.get("tool_name")):
            return None
        return _evaluate_result(spec, spawn_hook(spec, _serialize_payload(spec.event, kwargs)))

    _callback.__name__ = _callback.__qualname__ = f"shell_hook[{spec.event}:{spec.command}]"
    return _callback


def _fail_closed_block(spec: ShellHookSpec, reason: str) -> dict[str, Any]:
    return {"action": "block", "message": f"hook {spec.command} failed closed: {reason}"}


def _evaluate_result(
    spec: ShellHookSpec, r: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Turn a :func:`spawn.spawn_hook` diagnostic dict into the hook's contribution.

    Single place that encodes the failure semantics:

    * spawn error / timeout — fail open (log + ``None``) unless the spec
      is ``fail_closed`` on a blocking-capable event, in which case a
      canonical block shape is returned;
    * exit code 2 on a blocking-capable event — block, with the message
      taken from stdout block JSON, then stderr, then a default
      (Claude-Code / Cursor compatible);
    * other non-zero exits — warn, then parse stdout normally; a
      ``fail_closed`` hook blocks if no directive was produced;
    * non-JSON / unparseable stdout on a ``fail_closed`` blocking hook —
      block instead of silently contributing nothing.

    Shared by the live callback path (:func:`_make_callback`) and the CLI
    test helper (:func:`run_once`) so ``hermes hooks test`` reflects
    production behaviour exactly.
    """
    blocking_event = spec.event in _BLOCKING_EVENTS
    fail_closed = spec.fail_closed and blocking_event
    if r["error"]:
        logger.warning("shell hook failed (event=%s command=%s): %s", spec.event, spec.command, r["error"])
    elif r["timed_out"]:
        logger.warning("shell hook timed out after %.2fs (event=%s command=%s)", r["elapsed_seconds"], spec.event, spec.command)
    if r["error"] or r["timed_out"]:
        return _fail_closed_block(spec, r["error"] or f"timed out after {spec.timeout}s") if fail_closed else None
    stderr = r["stderr"].strip()
    if stderr:
        logger.debug("shell hook stderr (event=%s command=%s): %s", spec.event, spec.command, stderr[:_STDERR_MESSAGE_LIMIT])
    if r["returncode"] == BLOCK_EXIT_CODE and blocking_event:
        parsed = _parse_response(spec.event, r["stdout"])
        if isinstance(parsed, dict) and parsed.get("action") == "block":
            return parsed
        message = stderr[:_STDERR_MESSAGE_LIMIT] or _DEFAULT_BLOCK_MESSAGE
        logger.info("shell hook exited %d — blocking (event=%s command=%s): %s", BLOCK_EXIT_CODE, spec.event, spec.command, message)
        return {"action": "block", "message": message}
    # Other non-zero exits: still parse stdout so exit-code failures can carry a block directive.
    if r["returncode"] != 0:
        logger.warning("shell hook exited %d (event=%s command=%s); stderr=%s",
                       r["returncode"], spec.event, spec.command, stderr[:_STDERR_MESSAGE_LIMIT])
    stdout = (r["stdout"] or "").strip()
    parsed = _parse_response(spec.event, stdout)
    if parsed is None and fail_closed and r["returncode"] != 0:
        return _fail_closed_block(
            spec, f"hook exited {r['returncode']} with no directive",
        )

    if parsed is None and fail_closed and stdout and not _is_json_object(stdout):
        # The hook produced output we could not turn into a directive.
        # A fail-closed gate must not silently allow the action on
        # garbage output (e.g. a stack trace on stdout).
        return _fail_closed_block(spec, "unparseable stdout (expected a JSON object)")
    return parsed


def run_once(spec: ShellHookSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fire one hook with a synthetic payload (``hermes hooks test`` / doctor) through the production path."""
    result = spawn_hook(spec, _serialize_payload(spec.event, kwargs))
    result["parsed"] = _evaluate_result(spec, result)
    return result
