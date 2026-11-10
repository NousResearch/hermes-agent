"""Read-only model presentation for platform adapters; never resolves credentials."""

from contextlib import nullcontext
from copy import copy
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from gateway.session import SessionSource


class SessionModelSettings(TypedDict):
    """Detached display snapshot, not runtime credentials or a provider request route."""

    session_key: str
    model: str
    provider: str
    base_url: str


def _display_base_url(value: str) -> str:
    """Keep the endpoint path, excluding URL userinfo, query and fragment credentials."""
    if not value:
        return ""
    try:
        parts = urlsplit(value)
        if not parts.scheme or not parts.hostname:
            return ""
        parts.port  # Reject malformed ports rather than echoing possible credentials.
    except ValueError:
        return ""
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))


class _SessionModelSettingsMixin:
    def get_session_model_settings(self, source: "SessionSource") -> SessionModelSettings | None:
        """Read configured + in-memory session model settings for a platform settings UI.

        Returns None without a runner or for a rejected profile route. Reads never allocate a
        session, rehydrate an override, resolve credentials, or change the supplied source.
        See the platform adapter guide for precedence and display-only limitations.
        """
        runner = self.gateway_runner
        if runner is None:
            return None
        from gateway.run import _hermes_home, _load_gateway_config, _profile_runtime_scope
        from gateway.session_identity import identity_of

        source = copy(source)
        self._canonicalize(source)
        if getattr(source, "profile_route_rejected", False):
            return None
        home = _hermes_home
        scoped = identity_of(source) is not None or getattr(runner.config, "multiplex_profiles", False)
        if scoped:
            home = runner._resolve_profile_home_for_source(source)
        with _profile_runtime_scope(home) if scoped else nullcontext():
            cfg = _load_gateway_config(config_path=home / "config.yaml") or {}
        model_cfg = cfg.get("model") or {}
        if not isinstance(model_cfg, dict):
            model_cfg = {"default": model_cfg}
        key = runner._session_key_for_source(source)
        # A restart leaves the persisted /model pick unloaded until the next turn; read it, never rehydrate.
        store = getattr(runner, "session_store", None)
        override = runner._session_model_override(key) or (store and store.get_model_override(key)) or {}
        channel = runner._channel_override_for(source)
        defaults = {
            "model": model_cfg.get("default") or model_cfg.get("model") or "",
            "provider": model_cfg.get("provider") or "openrouter",
            "base_url": model_cfg.get("base_url") or "",
        }
        if channel is not None:
            defaults.update(
                model=channel.model or defaults["model"],
                provider=channel.provider or defaults["provider"],
            )
        result = {field: str(override.get(field) or default).strip() for field, default in defaults.items()}
        return SessionModelSettings(
            session_key=key, model=result["model"], provider=result["provider"],
            base_url=_display_base_url(result["base_url"]),
        )
