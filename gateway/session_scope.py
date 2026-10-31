"""Effective group/thread isolation shared by adapter guards and gateway sessions."""

from gateway.config import _coerce_bool
from gateway.session_identity import identity_of


def config_session_scope(config, platform=None) -> dict[str, bool]:
    """Resolve a platform's configured flags over its owning gateway's defaults."""
    platforms = getattr(config, "platforms", None) or {}
    extra = getattr(platforms.get(platform), "extra", None) or {}
    return {
        key: _coerce_bool(extra.get(key), getattr(config, key, default))
        for key, default in (("group_sessions_per_user", True), ("thread_sessions_per_user", False))
    }


class SessionScopeMixin:
    def register_platform_session_scope(
        self, platform, *, profile, group_sessions_per_user: bool, thread_sessions_per_user: bool,
    ) -> None:
        """Register an accepted adapter's resolved policy before it can receive traffic."""
        if not hasattr(self, "_platform_session_scopes"):
            self._platform_session_scopes = {}
        self._platform_session_scopes[(profile, platform)] = {
            "group_sessions_per_user": group_sessions_per_user,
            "thread_sessions_per_user": thread_sessions_per_user,
        }

    def resolve_session_scope(self, source) -> dict[str, bool]:
        """Receiving-bot policy, including restored/routed sources; never another bot's flags."""
        identity = identity_of(source)
        profile = (identity.transport_profile if identity is not None else
                   getattr(source, "profile", None) or self._resolve_profile_for_key(source)
                   or self._active_profile_name())
        scope = getattr(self, "_platform_session_scopes", {}).get((profile, source.platform.value))
        return scope.copy() if scope is not None else config_session_scope(self.config, source.platform)


def register_adapter_session_scope(runner, adapter) -> None:
    """Normalize an adapter's declaration using its owner's defaults, then register it."""
    store = getattr(runner, "session_store", None)
    if not isinstance(store, SessionScopeMixin):
        return
    profile = (getattr(adapter, "_owner_profile", None)
               or getattr(runner, "_primary_profile_name", None) or "default")
    configs = getattr(runner, "_profile_configs", None) or {}
    defaults = config_session_scope(configs.get(profile, runner.config), adapter.platform)
    extra = adapter.config.extra
    scope = {key: _coerce_bool(extra.get(key), default) for key, default in defaults.items()}
    extra.update(scope)
    store.register_platform_session_scope(adapter.platform.value, profile=profile, **scope)


def scope_for_context(config, source, session_store=None) -> dict[str, bool]:
    if isinstance(session_store, SessionScopeMixin):
        return session_store.resolve_session_scope(source)
    scope = config_session_scope(config, source.platform)
    identity = identity_of(source)
    adapter = identity.adapter() if identity is not None else None
    if adapter is not None:
        extra = adapter.config.extra
        scope = {key: _coerce_bool(extra.get(key), default) for key, default in scope.items()}
    return scope


def scope_for_runner(runner, source) -> dict[str, bool]:
    return scope_for_context(getattr(runner, "config", None), source, getattr(runner, "session_store", None))
