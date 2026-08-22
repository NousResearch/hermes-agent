from __future__ import annotations

from typing import Any


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


_URL_KEYS = ("base_url", "url", "api", "baseUrl")


def _entry_url(entry: dict[str, Any]) -> str:
    """Normalized route identity of a providers entry, honoring the same URL key
    aliases the config normalizer accepts (``base_url`` / ``url`` / ``api`` /
    ``baseUrl``)."""
    from hermes_cli.route_identity import normalize_route_base_url

    for key in _URL_KEYS:
        raw = entry.get(key)
        if isinstance(raw, str) and raw.strip():
            return normalize_route_base_url(raw.strip())
    return ""


def _is_custom_like(provider_id: str) -> bool:
    """True when ``provider_id`` is what a *named* custom provider runs as at
    runtime: ``custom``, ``custom:<x>``, or a registry alias of the custom
    profile (``ollama``, ``vllm``, ``llamacpp`` …).  Built-in providers
    (``openrouter``, ``anthropic`` …) are never custom-like, so they can never
    inherit timeouts from an unrelated custom entry that shares their URL."""
    pid = (provider_id or "").strip().lower()
    if not pid:
        return False
    if pid == "custom" or pid.startswith("custom:"):
        return True
    try:
        from providers import get_provider_profile

        profile = get_provider_profile(pid)
    except Exception:
        return False
    return bool(profile is not None and getattr(profile, "name", "") == "custom")


def _resolve_provider_config(
    config: dict[str, Any], provider_id: str, base_url: str | None
) -> dict[str, Any] | None:
    """Find the ``providers`` entry that governs this request.

    * Built-in provider ids resolve by key only: ``providers.<provider_id>``.
    * Custom-like ids (see :func:`_is_custom_like`) resolve by the ACTIVE
      route first — the first *enabled* ``providers.*`` entry (then legacy
      ``custom_providers``) whose URL matches ``base_url`` — and only then by
      key.  The active URL must beat the key because a user may literally name
      an entry ``providers.custom`` while running a different named provider.

    Why: a *named* custom provider (``providers.mlx-lm``) runs with
    ``AIAgent.provider == "custom"`` — the config key never reaches the agent,
    so keying on ``provider_id`` alone silently ignores every timeout set on
    that entry.  Matching by URL mirrors
    :func:`hermes_cli.config.get_custom_provider_context_length` (#15779).
    Duplicate URLs resolve to the first match in config order.
    """
    providers = config.get("providers", {})
    providers = providers if isinstance(providers, dict) else {}

    # Hard dependency, deliberately NOT guarded: a missing/renamed
    # is_provider_enabled must fail loudly, never fall open to "enabled".
    from hermes_cli.config import is_provider_enabled

    def _explicit() -> dict[str, Any] | None:
        entry = providers.get(provider_id)
        if not isinstance(entry, dict) or not entry or not is_provider_enabled(entry):
            return None
        return entry

    if not _is_custom_like(provider_id):
        return _explicit()

    if base_url:
        from hermes_cli.route_identity import normalize_route_base_url

        target = normalize_route_base_url(base_url)
        if target:
            candidates: list[Any] = list(providers.values())
            legacy = config.get("custom_providers")
            if isinstance(legacy, list):
                candidates.extend(legacy)
            for entry in candidates:
                if not isinstance(entry, dict) or not is_provider_enabled(entry):
                    continue
                if _entry_url(entry) == target:
                    return entry
    return _explicit()


def _provider_config_for(
    provider_id: str, base_url: str | None
) -> dict[str, Any] | None:
    if not provider_id and not base_url:
        return None
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None
    if not isinstance(config, dict):
        return None
    return _resolve_provider_config(config, provider_id or "", base_url)


def get_provider_request_timeout(
    provider_id: str, model: str | None = None, base_url: str | None = None
) -> float | None:
    """Per-call request timeout from config: per-model ``timeout_seconds``
    beats provider-wide ``request_timeout_seconds``.  ``base_url`` lets named
    custom providers (runtime ``provider_id == "custom"``) resolve their entry."""
    provider_config = _provider_config_for(provider_id, base_url)
    if provider_config is None:
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("request_timeout_seconds"))


def get_provider_stale_timeout(
    provider_id: str, model: str | None = None, base_url: str | None = None
) -> float | None:
    """Stale-watchdog timeout from config: per-model beats provider-wide
    ``stale_timeout_seconds``.  ``base_url`` as in :func:`get_provider_request_timeout`."""
    provider_config = _provider_config_for(provider_id, base_url)
    if provider_config is None:
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("stale_timeout_seconds"))


def _get_model_config(
    provider_config: dict[str, object], model: str | None
) -> dict[str, object] | None:
    if not model:
        return None

    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    if isinstance(model_config, dict):
        return model_config
    return None
