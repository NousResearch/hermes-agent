"""Credential-free adapter results; raw picker inputs remain inside core."""

from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
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


class SessionModelPickerProvider(TypedDict):
    """Detached picker presentation row; operational endpoints and credentials are excluded."""

    slug: str
    name: str
    models: list[str]
    total_models: int
    is_current: bool


def _resolve_session_model_settings(runner, source, cfg):
    """Resolve raw presentation inputs only; no session allocation or runtime preparation."""
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
    return {"session_key": key, **result}


class _SessionModelSettingsMixin:
    def get_session_model_settings(self, source: "SessionSource") -> SessionModelSettings | None:
        """Read configured + session model settings for a platform settings UI.

        Returns None without a runner or for a rejected profile route. Reads never allocate a
        session, rehydrate an override, resolve credentials, or change the supplied source.
        See the platform adapter guide for precedence and display-only limitations.
        """
        with self._session_model_settings_context(source) as resolved:
            if resolved is None:
                return None
            settings, _ = resolved
            return SessionModelSettings(**{**settings, "base_url": _display_base_url(settings["base_url"])})

    def list_session_model_picker_providers(
        self, source: "SessionSource", *, max_models: int | None = 100, include_moa: bool = True,
    ) -> list[SessionModelPickerProvider] | None:
        """List detached picker rows using the conversation's raw endpoint internally.

        Uses /model's non-blocking catalog path (disk-cached remote catalogs, refreshed in the
        background); files and a short local LM Studio probe remain, so use asyncio.to_thread.
        Returns None under the same conditions as get_session_model_settings. Never allocates
        or rehydrates a session, or changes its settings. Custom endpoint live probes are
        disabled because they can persist discovered models into config; cached/curated
        catalogs remain available. Catalog infrastructure may maintain its own caches.
        Only slug, name, models, total_models and is_current leave core. On older Hermes,
        feature-detect this method with getattr (see the platform adapter guide).
        """
        from hermes_cli.config import get_compatible_custom_providers
        from hermes_cli.model_switch_providers import list_picker_providers

        with self._session_model_settings_context(source) as resolved:
            if resolved is None:
                return None
            settings, cfg = resolved
            catalog_cfg = cfg.get("model_catalog")
            rows = list_picker_providers(
                current_provider=settings["provider"], current_base_url=settings["base_url"],
                current_model=settings["model"], user_providers=deepcopy(cfg.get("providers")),
                custom_providers=get_compatible_custom_providers(cfg), max_models=max_models,
                include_moa=include_moa,
                excluded_providers=deepcopy(catalog_cfg.get("excluded_providers"))
                if isinstance(catalog_cfg, dict) else None,
                probe_custom_providers=False, probe_current_custom_provider=False,
                non_blocking_catalogs=True,  # as /model: a prefetch pool would drop the profile scope
            )
            return [SessionModelPickerProvider(
                slug=row["slug"], name=row["name"], models=list(row["models"]),
                total_models=row["total_models"], is_current=row["is_current"],
            ) for row in rows]

    @contextmanager
    def _session_model_settings_context(self, source):
        runner = self.gateway_runner
        if runner is None:
            yield None
            return
        from gateway.run import _hermes_home, _load_gateway_config, _profile_runtime_scope
        from gateway.session_identity import identity_of

        source = copy(source)
        self._canonicalize(source)
        if getattr(source, "profile_route_rejected", False):
            yield None
            return
        home = _hermes_home
        scoped = identity_of(source) is not None or getattr(runner.config, "multiplex_profiles", False)
        if scoped:
            home = runner._resolve_profile_home_for_source(source)
        with _profile_runtime_scope(home) if scoped else nullcontext():
            cfg = _load_gateway_config(config_path=home / "config.yaml") or {}
            yield _resolve_session_model_settings(runner, source, cfg), cfg
