"""Capability-gated ``PluginContext`` surface: ``has_capability`` probing, ``side_question`` (ask
the session's own agent after a turn, ``llm.side_question``), ``fork_session`` (run a bounded
background copy of it, ``llm.session_fork``) and ``quota_fetch`` (documented provider quota
counters, ``llm.quota_observation``)."""

from __future__ import annotations

import logging
import re
from typing import Optional

from hermes_cli.plugin_capabilities import plugin_capability_granted
from hermes_cli.plugins_loader import _plugin_home_scope

logger = logging.getLogger(__name__)


class PluginCapabilityCtxMixin:
    """Mixed into :class:`hermes_cli.plugins.PluginContext` (``manifest``, ``_manager``, ``plugin_id``)."""

    # -- capability probing (#64228) -----------------------------------------
    def has_capability(self, capability: str) -> bool:
        """True when *capability* is live for this plugin (probe, then degrade gracefully). Bundled
        plugins are trusted for ``tools.override``; otherwise granted_capabilities or the legacy
        ``allow_*`` key decides. Unknown ids / unreadable consent -> False (fail closed)."""
        if self.manifest.source == "bundled" and capability == "tools.override":
            return True
        return plugin_capability_granted(self.plugin_id, capability)

    def side_question(self, token: str, prompt: str, *, timeout: float = 15.0) -> Optional[str]:
        """Ask the session's own agent ``prompt`` about the turn that just ended; text or None.

        ``token`` is ``post_llm_call``'s ``side_question_token``: one attempt per turn, shared by
        every plugin of the session's profile, valid for 60 s and only until the next turn starts
        (cron runs and subagent turns get no token). Same model, the whole
        session (memory included) and its warm prompt cache; tools never run and nothing is
        written to the transcript. Needs the ``llm.side_question`` capability; without it this
        returns None and leaves the attempt to a consented plugin."""
        from agent.side_question_lease import SIDE_QUESTION_CAPABILITY, redeem_side_question

        if not self._granted_in_own_home(SIDE_QUESTION_CAPABILITY):
            if not getattr(self, "_side_question_denial_logged", False):
                self._side_question_denial_logged = True
                logger.info("Plugin %s called ctx.side_question without the %s capability; returning None",
                            self.plugin_id, SIDE_QUESTION_CAPABILITY)
            return None
        return redeem_side_question(
            token, prompt, plugin_id=self.plugin_id, home=self._manager.home_path, timeout=timeout)

    def fork_session(
        self, token: str, prompt: str, *, tools=(), max_iterations: int = 8, input_budget: Optional[int] = None,
        task: Optional[str] = None, timeout: Optional[float] = None,
    ):
        """Fork the session's own agent after the turn that just ended and run ``prompt`` in the
        background; a :class:`agent.plugin_session_fork.PluginForkHandle`, or None.

        ``token`` is ``post_llm_call``'s ``side_question_token`` (same lease rules as
        :meth:`side_question`). The fork sees the whole session (memory included) on its warm
        prompt cache, may call only ``tools`` (this plugin's own registered tools the session
        already advertises; naming any other tool returns None and leaves the token unspent), runs
        at most ``max_iterations`` (<= 16) loop iterations within the review's input-token budget
        (``input_budget`` can only lower it), auto-denies dangerous commands, writes nothing to
        the transcript, and is cancelled when the session starts its next turn.
        ``task`` routes it through one of this plugin's ``register_auxiliary_task`` lanes. Needs
        ``llm.session_fork``; without it this returns None and leaves the token unspent."""
        from agent.plugin_session_fork import start_session_fork
        from agent.side_question_lease import SESSION_FORK_CAPABILITY

        if not self._granted_in_own_home(SESSION_FORK_CAPABILITY):
            if not getattr(self, "_session_fork_denial_logged", False):
                self._session_fork_denial_logged = True
                logger.info("Plugin %s called ctx.fork_session without the %s capability; returning None",
                            self.plugin_id, SESSION_FORK_CAPABILITY)
            return None
        owned = [key for key, entry in self._manager._aux_tasks.items() if entry.get("plugin") == self.plugin_id]
        return start_session_fork(
            token, prompt, plugin_id=self.plugin_id, home=self._manager.home_path, tools=tools,
            max_iterations=max_iterations, input_budget=input_budget, task=task, timeout=timeout, owned_tasks=owned,
            own_tools=self._own_tool_names)

    def quota_fetch(
        self, *, provider_id: str, credential_scope_id: str, quota_source: str,
        timeout: float = 2.0,
    ) -> dict:
        """Fetch documented quota counters for one opaque credential scope, or unavailable.

        Host-held credentials only: the credential for ``credential_scope_id`` is resolved
        from the persisted pool (an entry the host owns), used for exactly one documented
        provider request, and never returned. Plugins pass opaque scope ids, never tokens.

        Requires the ``llm.quota_observation`` capability. Only OpenRouter's documented
        key-info counters (``quota_source="official_key_limits"``) are implemented;
        every other provider/source reports unavailable rather than guessing an endpoint.
        """
        from agent.quota_fetch import (
            DEFAULT_TIMEOUT, MAX_TIMEOUT, OFFICIAL_KEY_LIMITS, fetch_openrouter_key_limits,
        )
        from agent.quota_observation import QUOTA_OBSERVATION_CAPABILITY

        def _unavailable(reason):
            return {"status": "unavailable", "provider_id": provider_id if isinstance(provider_id, str) else "",
                    "quota_source": quota_source if isinstance(quota_source, str) else "", "reason": reason}

        if not self._granted_in_own_home(QUOTA_OBSERVATION_CAPABILITY):
            if not getattr(self, "_quota_fetch_denial_logged", False):
                self._quota_fetch_denial_logged = True
                logger.info("Plugin %s called ctx.quota_fetch without the %s capability; refusing",
                            self.plugin_id, QUOTA_OBSERVATION_CAPABILITY)
            return _unavailable("capability_not_granted")
        if provider_id != "openrouter":
            return _unavailable("unsupported_provider")
        if quota_source != OFFICIAL_KEY_LIMITS:
            return _unavailable("unsupported_quota_source")
        if not isinstance(credential_scope_id, str) or re.fullmatch(r"[0-9a-f]{32}", credential_scope_id) is None:
            return _unavailable("invalid_scope")
        effective = timeout if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) else DEFAULT_TIMEOUT
        effective = min(max(float(effective), 0.5), MAX_TIMEOUT)
        try:
            from hermes_cli.auth import read_credential_pool
            # Bind the pool read to THIS manager's immutable profile home. The
            # ambient home may belong to another profile in a multiplexed
            # process; an opaque scope id alone must never authorize the use of
            # a token from a different profile. The documented global-root
            # fallback inside read_credential_pool remains intact.
            with _plugin_home_scope(self._manager.home_path):
                rows = read_credential_pool("openrouter")
        except Exception:  # health: allow BLE001 -- unreadable pool state is 'scope unavailable', fail closed
            logger.debug("quota_fetch: credential pool unreadable")
            return _unavailable("pool_unavailable")
        entry = next(
            (row for row in rows if isinstance(row, dict) and row.get("scope_id") == credential_scope_id),
            None,
        )
        if entry is None:
            return _unavailable("scope_not_found")
        token = entry.get("access_token")
        if not isinstance(token, str) or not token:
            return _unavailable("scope_token_unavailable")
        result = fetch_openrouter_key_limits(token, credential_scope_id=credential_scope_id, timeout=effective)
        return result

    def _own_tool_names(self) -> frozenset:
        """Tool names whose live registration is this plugin's: per name, the newest active ``tool``
        entry in the manager's ledger decides (a later re-registration by another plugin owns it).
        Core and MCP tools never appear here: they are not registered through a plugin context."""
        owners: dict[str, str] = {}
        for registration in reversed(list(self._manager._registration_order)):
            if registration.kind == "tool" and registration.active:
                owners.setdefault(registration.key, registration.plugin_key)
        return frozenset(name for name, owner in owners.items() if owner == self.plugin_id)

    def _granted_in_own_home(self, capability: str) -> bool:
        """Consent read from THIS manager's profile home: plugins call from their own threads,
        where the active profile may be the launch profile's."""
        try:
            from hermes_cli.config import load_config
            with _plugin_home_scope(self._manager.home_path):
                cfg = load_config() or {}
        except Exception:  # health: allow BLE001 -- fail closed: unreadable consent state never grants
            return False
        return plugin_capability_granted(self.plugin_id, capability, config=cfg)
