from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def _provider_timeout_candidates(
    provider_id: str,
    requested_provider_id: str | None,
) -> tuple[str, ...]:
    """Return config provider IDs from most to least route-specific."""
    runtime_id = (provider_id or "").strip().lower()
    requested_id = (requested_provider_id or "").strip().lower()
    candidates: list[str] = []

    # Named custom routes intentionally canonicalize to ``custom`` at runtime.
    # Preserve the requested identity for config lookup, while retaining bare
    # ``providers.custom`` as a backward-compatible fallback.
    if runtime_id == "custom" and requested_id and requested_id != "custom":
        candidates.append(requested_id)
        if requested_id.startswith("custom:"):
            candidates.append(requested_id.removeprefix("custom:"))
        else:
            candidates.append(f"custom:{requested_id}")

    candidates.append(runtime_id)
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _provider_config_for_candidate(
    providers: dict[object, object], candidate: str
) -> object:
    """Resolve a provider block using runtime custom-provider aliases."""
    provider_config = providers.get(candidate)
    if provider_config is not None:
        return provider_config

    for key, value in providers.items():
        route_ids: set[str] = set()
        for raw_name in (key, value.get("name") if isinstance(value, dict) else None):
            if raw_name is None:
                continue
            raw_id = str(raw_name).strip().lower()
            normalized_id = raw_id.replace(" ", "-")
            route_ids.update((raw_id, normalized_id, f"custom:{normalized_id}"))
        if candidate in route_ids:
            return value
    return None


def _get_provider_timeout(
    provider_id: str,
    model: str | None,
    *,
    requested_provider_id: str | None,
    model_field: str,
    provider_field: str,
) -> float | None:
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    if not isinstance(providers, dict):
        return None

    for candidate in _provider_timeout_candidates(provider_id, requested_provider_id):
        provider_config = _provider_config_for_candidate(providers, candidate)
        if not isinstance(provider_config, dict):
            continue

        model_config = _get_model_config(provider_config, model)
        if model_config is not None:
            timeout = _coerce_timeout(model_config.get(model_field))
            if timeout is not None:
                return timeout

        timeout = _coerce_timeout(provider_config.get(provider_field))
        if timeout is not None:
            return timeout

    return None


def get_provider_request_timeout(
    provider_id: str,
    model: str | None = None,
    *,
    requested_provider_id: str | None = None,
) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    return _get_provider_timeout(
        provider_id,
        model,
        requested_provider_id=requested_provider_id,
        model_field="timeout_seconds",
        provider_field="request_timeout_seconds",
    )


def get_provider_stale_timeout(
    provider_id: str,
    model: str | None = None,
    *,
    requested_provider_id: str | None = None,
) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    return _get_provider_timeout(
        provider_id,
        model,
        requested_provider_id=requested_provider_id,
        model_field="stale_timeout_seconds",
        provider_field="stale_timeout_seconds",
    )


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
