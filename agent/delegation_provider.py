"""Delegation Provider ABC.

A delegation provider decides HOW one ``delegate_task`` task becomes a child run: which model,
provider, endpoint and reasoning effort the child uses, or which Hermes profile it runs as. It may
also add per-task parameters to the ``delegate_task`` schema (an enum of profile names, of titled
model options, ...). Providers register via :meth:`PluginContext.register_delegation_provider`;
``delegation.backend`` in config.yaml selects the active one (unset = ``default``, today's
behaviour, provided by the bundled ``plugins/delegation/default`` plugin). Core keeps the whole lifecycle (spawn, isolation, roles, depth, concurrency,
background units, stop fan-out, results, ``output_schema``, progress); a provider only answers
"which route does this task take".

Everything crossing this interface is plain JSON data (no live agent objects), so a provider runs
unchanged in-process and under ``plugins.isolation: host``.

Route vocabulary (:meth:`DelegationProvider.route_task` returns a dict with any of):

* ``model``, ``provider``, ``base_url``, ``api_key``, ``api_mode``, ``request_overrides``,
  ``reasoning_effort``, ``fallback_providers`` - shaped exactly like the ``delegation`` config
  section and resolved by the same credential resolver ``delegation.*`` uses (empty route = the
  child inherits the parent's route, like an empty ``delegation`` section). An unknown
  ``reasoning_effort`` refuses the task.
* ``service_tier`` - ``normal``, ``fast`` or ``ultrafast``, checked like ``/fast`` against the
  resolved route (and, for ultrafast, the Codex account); a tier the route does not offer refuses
  the task, and the tier is re-applied after the child is built.
* ``profile`` - the name of a Hermes profile. Core runs the child as that profile: its config,
  primary model and credentials, skills, ``SOUL.md`` and terminal policy, bound through the same
  scope helpers a multiplexed gateway uses for a routed turn. A profile route ignores the other
  route keys; the profile's own config decides them.
* ``label`` - short text shown with the child's progress and result (e.g. the option title).
* ``max_iterations`` - a whole number of 1 or more: the turn limit for this child. It replaces
  ``delegation.max_iterations`` for this task only. Any other value refuses the task. A profile
  route ignores it: that child uses the profile's own ``delegation.max_iterations`` (default 250).
* ``child_timeout_seconds`` - a number of 0 or more: the idle timeout for this child, with the same
  meaning as ``delegation.child_timeout_seconds`` (0 disables, floor 30 s). It replaces the global
  value for this task only. Any other value refuses the task. A profile route ignores it: that
  child uses the profile's own ``delegation.child_timeout_seconds`` (default: no idle timeout).

* ``escalate_reasoning`` - a boolean (default false). Reasoning escalation runs for this child only
  when the route sets both ``escalate_reasoning: true`` and ``escalate_ceiling``; either one alone
  leaves it off. Route-only: not a ``delegation`` section global. A profile route ignores it.
* ``escalate_ceiling`` - a reasoning ladder level: none, minimal, low, medium, high, xhigh, max,
  or ultra (case-insensitive, whitespace trimmed). Required for escalation, never enough on its own.
  Route-only; a profile route ignores it. Invalid values refuse only this task.
* A provider can decide the retry of a budget stop with :meth:`DelegationProvider.next_attempt`:
  keep core's one-level climb (``None``), stop (``{"retry": False}``) or pick the level
  (``{"effort": level}``, same route, not past the ceiling).

A backend's own settings live under ``delegation.backends.<backend name>`` (:meth:`DelegationProvider.settings`),
never beside the core ``delegation`` keys.
"""

from __future__ import annotations

import abc
import math
from typing import Any, Mapping, Optional

from agent.provider_base import ProviderBase
from agent.reasoning_effort import EFFORT_LADDER

#: Per-task property names core owns. A provider-declared parameter with one of these names is
#: refused at schema-build time: plugins add parameters, they never redefine core ones, and the
#: raw route fields (``provider``/``model``/``base_url``/``api_key``) never become model-facing.
RESERVED_TASK_PARAMETERS = frozenset({
    "goal", "context", "index", "output_schema", "images", "group", "role", "toolsets",
    "acp_command", "acp_args", "provider", "model", "base_url", "api_key", "api_mode",
})

#: Keys a route may carry; anything else in a provider's answer is ignored.
ROUTE_KEYS = frozenset({
    "model", "provider", "base_url", "api_key", "api_mode", "request_overrides",
    "reasoning_effort", "service_tier", "fallback_providers", "profile", "label",
    "max_iterations", "child_timeout_seconds", "escalate_reasoning", "escalate_ceiling",
})


def valid_route_max_iterations(value: Any) -> bool:
    """A route's ``max_iterations``: a whole number of 1 or more (never a bool)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def valid_route_child_timeout(value: Any) -> bool:
    """A route's ``child_timeout_seconds``: a finite int/float of 0 or more (never a bool). An int too large for a
    float is not representable as seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def valid_route_escalate_reasoning(value: Any) -> bool:
    """Escalation is opt-in with a real boolean, never a truthy string or integer."""
    return isinstance(value, bool)


def valid_route_escalate_ceiling(value: Any) -> bool:
    """A route ceiling must name a level on the reasoning effort ladder."""
    return isinstance(value, str) and value.strip().lower() in EFFORT_LADDER


class DelegationRouteError(ValueError):
    """Raised by :meth:`DelegationProvider.route_task` to refuse a task; the message becomes the
    ``delegate_task`` tool error the model reads, so phrase it as an instruction."""


class DelegationProvider(ProviderBase):
    """Abstract base class for a pluggable delegation strategy.

    :attr:`name` is the ``delegation.backend`` value. Every method receives ``config``, the
    ``delegation`` section of the active profile's config.yaml (read-only): ``dict(config)`` is the
    default route, and :meth:`settings` is this backend's own block, ``delegation.backends.<name>``.
    """

    def settings(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """This backend's settings: ``delegation.backends.<name>`` (empty when unset or not a mapping)."""
        backends = config.get("backends") if isinstance(config, Mapping) else None
        own = backends.get(self.name) if isinstance(backends, Mapping) else None
        return own if isinstance(own, Mapping) else {}

    @property
    def summary(self) -> str:
        """One user-facing line for backend pickers (``/delegation``); never shown to the model."""
        return ""

    def is_available(self) -> bool:
        """Whether this provider can run here. MUST NOT make network calls. False fails delegation closed: a
        ``delegate_task`` call under this backend is a tool error naming it, never a run on another route."""
        return True

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Extra per-task JSON-schema properties, ``{name: schema}``, merged into
        ``delegate_task``'s ``tasks.items.properties``.

        Called when the tool definitions are built for a session, never mid-conversation, and the
        result must be deterministic for a given ``config`` (same input, same bytes): the schema
        is part of the cached prompt prefix. Names in :data:`RESERVED_TASK_PARAMETERS` are
        refused. An ``enum`` here is enforced by core before :meth:`route_task` runs, and a task
        carrying a parameter this method did not declare is refused before it reaches you.
        """
        return {}

    def description(self, config: Mapping[str, Any]) -> str:
        """Optional guidance appended to the ``delegate_task`` description (same caching rule
        as :meth:`task_parameters`)."""
        return ""

    @abc.abstractmethod
    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        """Return the route for one task (vocabulary in the module docstring).

        ``task`` carries ``goal``, ``context``, ``index`` and the values the model supplied for
        this provider's :meth:`task_parameters` (absent when omitted). Raise
        :class:`DelegationRouteError` to refuse the task.
        """

    def next_attempt(self, stop: Mapping[str, Any], config: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """Decide the retry after a budget stop of a child whose route has reasoning escalation on.

        ``stop`` carries ``effort`` (the level the attempt ran at), ``exit_reason`` (``max_iterations`` or
        ``timeout``), ``api_calls``, ``attempt`` (1 for the first attempt), ``ceiling`` (the route's
        ``escalate_ceiling``), ``route`` (the route report, or None) and ``summary`` (the attempt's final text,
        or None). Return ``None`` for core's rule: one level up, never past the ceiling. Return
        ``{"retry": False}`` to end the chain. Return ``{"effort": level}`` to retry on the same route at that
        level: it must be a ladder level, not past the ceiling, and stronger than ``effort`` after the
        provider's clamp, else core logs a warning and uses its rule. Never called for a finished, failed or
        interrupted attempt, or for a route without escalation. An exception here is logged and treated as
        ``None``.
        """
        return None
