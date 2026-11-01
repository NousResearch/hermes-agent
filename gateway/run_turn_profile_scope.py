"""Profile runtime scopes for a turn's source (moved out of ``gateway/run_turn.py``)."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # string annotations only
    from gateway.session import SessionSource


class GatewayTurnProfileScopeMixin:
    """Which profile home and secret scope a turn, slash command or media delivery binds for its source."""

    def _profile_scope_for_source(self, source: SessionSource):
        """``_profile_runtime_scope`` for ``source``'s profile when a secret scope is required.

        Under multiplexing config/skills/memory resolve to the source profile's home AND credentials
        come from its secret scope (never process-global ``os.environ``). A standalone gateway
        (``multiplex_profiles`` off) still binds once a hosted room has flipped the process-wide
        credential guard — see ``_standalone_launch_scope``."""
        from gateway.run import _profile_runtime_scope
        from gateway.session_policy import local_desktop_turn
        home = self._profile_scope_key_for_source(source)
        return local_desktop_turn(self, source, _profile_runtime_scope(home) if home is not None
                                  else self._standalone_launch_scope())

    def _profile_scope_key_for_source(self, source: SessionSource) -> Optional[Path]:
        """Profile home ``_profile_scope_for_source`` binds for ``source``, or ``None`` when it falls
        back to the standalone launch scope. The single owner of that branch condition: callers that
        group work per scope (heartbeat restore) key on this so they cannot drift from the scope
        actually entered."""
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return self._resolve_profile_home_for_source(source)
        return None

    def _async_profile_scope_for_source(self, source: SessionSource):
        """``async with`` twin of :meth:`_profile_scope_for_source` (secret hydration off-loop).

        Slash dispatch runs under the RECEIVING bot's scope (auth needs its ``.env``), which is not
        the routed runtime when a bot serves another profile's chat; every handler reading
        home-relative state (pending writes, memory store, config) binds the runtime here (#119915)."""
        from gateway.run import _async_profile_runtime_scope
        home = self._profile_scope_key_for_source(source)
        if home is not None:
            return _async_profile_runtime_scope(home)
        from tui_gateway.launch_profile_policy import async_launch_profile_scope_if_multiplexed
        return async_launch_profile_scope_if_multiplexed()

    @staticmethod
    def _standalone_launch_scope():
        """Scope for a standalone gateway's own (launch-profile) work: a no-op until the process hosts
        another profile home, then the launch profile's OWN runtime scope.

        A native hosted room running a second profile calls
        ``tui_gateway.launch_profile_policy.activate_multi_profile_hosting`` inside the gateway process,
        so ``get_secret`` fails closed for every unscoped read afterwards — including the standalone
        gateway's ordinary turns, which never bound a scope because ``multiplex_profiles`` is off
        (#112878). The launch profile is a profile too: bind its ``.env`` over the env frozen at
        activation (a key injected by systemd / ``op run`` has no file to rebuild it from), never a
        secondary's scope and never live ``os.environ``."""
        from tui_gateway.launch_profile_policy import launch_profile_scope_if_multiplexed
        return launch_profile_scope_if_multiplexed()

    def _media_delivery_scope_for_source(self, source: SessionSource):
        """Home + terminal-policy scope for validating a turn's MEDIA / local-file paths on the
        adapter's delivery side, which runs after the routed turn scope was reset.

        Docker path translation (``platforms/base.py::_translate_docker_container_media_path``)
        infers the producing container from the ACTIVE profile (``get_active_profile_name``) and the
        scope-aware ``TERMINAL_DOCKER_VOLUMES``; without this a secondary's ``MEDIA:/output/x.png``
        resolves against the default profile's sandbox and mounts (#109024). No secret hydration:
        path validation reads no credentials and this runs on the event loop."""
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return nullcontext()
        from gateway.run import _profile_runtime_scope
        return _profile_runtime_scope(self._resolve_profile_home_for_source(source), {})
