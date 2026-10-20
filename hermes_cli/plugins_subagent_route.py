"""``pre_subagent_route`` plugin hook: per-child routing for ``delegate_task``.

The hook fires once per delegate_task child, on the main thread, BEFORE the child's runtime is resolved.
Kwargs: parent_session_id, parent_subagent_id, task_index, task_count, child_goal, child_context,
child_role, default_provider, default_model, default_reasoning_effort. A callback returns ``None`` to keep
the batch route, or a dict with any of ``provider`` (requires ``model``), ``model`` and
``reasoning_effort``. Privacy: goal and context may contain user and project content.

Credentials stay in core: ``tools.delegate_tool`` resolves a returned provider through the same runtime
provider path as ``delegation.provider``; an unknown or uncredentialed provider fails the spawn instead of
falling back to the batch model. A caller-pinned route (``credentials_cfg``, e.g. /review) skips the hook.
Shell hooks cannot return routes (SHELL_UNSUPPORTED_HOOKS).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

HOOK_NAME = "pre_subagent_route"
_MAX_CHARS = 200


def _route_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and len(value) <= _MAX_CHARS else None


def sanitize_subagent_route(result: Any) -> Optional[dict[str, Any]]:
    """One hook return value as ``{provider?, model?, reasoning_effort?}``, or ``None``."""
    if not isinstance(result, dict):
        return None
    route: dict[str, Any] = {}
    provider, model = _route_str(result.get("provider")), _route_str(result.get("model"))
    if provider and not model:
        # A provider switch without a model would inherit the parent's model on a foreign provider.
        logger.warning("pre_subagent_route: ignored provider %r without a model", provider)
        provider = None
    if provider:
        route["provider"] = provider
    if model:
        route["model"] = model
    effort = result.get("reasoning_effort")
    if effort is False or isinstance(effort, str):
        from hermes_constants import parse_reasoning_effort
        if parse_reasoning_effort(effort) is None:
            logger.warning("pre_subagent_route: ignored unknown reasoning_effort %r", effort)
        else:
            route["reasoning_effort"] = effort.strip().lower() if isinstance(effort, str) else effort
    return route or None


def get_plugin_subagent_route(
    *, parent_session_id: Any = None, parent_subagent_id: Any = None, task_index: int = 0, task_count: int = 1,
    child_goal: str = "", child_context: Optional[str] = None, child_role: str = "leaf",
    default_provider: Optional[str] = None, default_model: Optional[str] = None,
    default_reasoning_effort: Any = None,
) -> Optional[dict[str, Any]]:
    """Consult ``pre_subagent_route`` hooks for one delegate_task child.

    Run-all-then-pick-first, like ``get_plugin_error_classification``: every callback runs with failures
    isolated, the first valid route in registration order wins and losing valid routes are logged. Invalid
    returns degrade to a no-op; callback errors are isolated by ``invoke_hook``.
    """
    from hermes_cli import plugins

    if not plugins.has_hook(HOOK_NAME):
        return None
    hook_results = plugins.invoke_hook(
        HOOK_NAME, parent_session_id=parent_session_id, parent_subagent_id=parent_subagent_id,
        task_index=task_index, task_count=task_count, child_goal=child_goal, child_context=child_context,
        child_role=child_role, default_provider=default_provider, default_model=default_model,
        default_reasoning_effort=default_reasoning_effort,
    )
    valid = [route for result in hook_results if (route := sanitize_subagent_route(result)) is not None]
    if not valid:
        return None
    if len(valid) > 1:
        logger.warning("pre_subagent_route: skipped %d valid route(s) after the first result in registration "
                       "order won (run-all-then-pick-first)", len(valid) - 1)
    return valid[0]
