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

A backend's own settings live under ``delegation.backends.<backend name>`` (:meth:`DelegationProvider.settings`),
never beside the core ``delegation`` keys.
"""

from __future__ import annotations

import abc
from typing import Any, Mapping

from agent.provider_base import ProviderBase

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
})


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
