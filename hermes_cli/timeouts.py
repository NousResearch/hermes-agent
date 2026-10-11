from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def timeout_provider_id(provider: str | None, requested_provider: str | None = None) -> str | None:
    """The ``providers.<id>`` key whose timeouts apply to a runtime ``provider``.

    A named ``providers.<id>`` entry resolves to runtime provider ``custom`` and keeps its id only
    as ``requested_provider`` (a fallback/switch target may also carry it as ``custom:<id>``), so a
    lookup by the runtime id never finds the entry's ``request_timeout_seconds`` /
    ``stale_timeout_seconds``. Only those named custom providers remap; every other provider, and
    bare ``custom``, keeps its own id.
    """
    runtime_id = (provider or "").strip()
    if runtime_id.lower() == "custom":
        named = (requested_provider or "").strip()
    elif runtime_id.lower().startswith("custom:"):
        named = runtime_id
    else:
        return provider
    if named.lower().startswith("custom:"):
        named = named[len("custom:"):].strip()
    return named if named and named.lower() != "custom" else provider


def _named_custom_entry(providers: dict, named_id: str) -> object:
    """First enabled ``providers:`` entry matching ``named_id`` the way runtime resolution matches
    it (case-insensitive key, display name, or ``custom:<key>`` slug), so a lookup never misses an
    entry that resolution selected."""
    from hermes_cli.config import is_provider_enabled
    from hermes_cli.providers import custom_provider_aliases

    requested = named_id.strip().lower().replace(" ", "-")
    for key, entry in providers.items():
        if not isinstance(entry, dict) or not is_provider_enabled(entry):
            continue
        if requested in custom_provider_aliases(str(entry.get("name", "") or key), str(key)):
            return entry
    return None


def _entry_timeout(provider_config: object, model: str | None, model_key: str, provider_key: str) -> float | None:
    """Per-model ``models.<model>.<model_key>`` wins over the entry's ``<provider_key>``."""
    if not isinstance(provider_config, dict):
        return None
    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get(model_key))
        if timeout is not None:
            return timeout
    return _coerce_timeout(provider_config.get(provider_key))


def _configured_timeout(provider_id: str, model: str | None, model_key: str, provider_key: str,
                        requested_provider: str | None = None) -> float | None:
    """Per-model ``providers.<id>.models.<model>.<model_key>`` wins over ``providers.<id>.<provider_key>``.

    For a named custom provider the named entry is consulted first; the runtime id's own entry
    (e.g. a ``providers.custom`` block) still applies when the named entry sets no value.
    """
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
    named_id = timeout_provider_id(provider_id, requested_provider)
    if named_id and named_id != provider_id:
        timeout = _entry_timeout(_named_custom_entry(providers, named_id), model, model_key, provider_key)
        if timeout is not None:
            return timeout
    return _entry_timeout(providers.get(provider_id), model, model_key, provider_key)


def get_provider_request_timeout(provider_id: str, model: str | None = None, *,
                                 requested_provider: str | None = None) -> float | None:
    """Return a configured provider request timeout in seconds, if any.

    Pass the agent's ``requested_provider`` so a named custom provider's own entry is honoured.
    """
    return _configured_timeout(provider_id, model, "timeout_seconds", "request_timeout_seconds", requested_provider)


def get_provider_stale_timeout(provider_id: str, model: str | None = None, *,
                               requested_provider: str | None = None) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any (see ``get_provider_request_timeout``)."""
    return _configured_timeout(provider_id, model, "stale_timeout_seconds", "stale_timeout_seconds", requested_provider)


def _get_model_config(provider_config: dict[str, object], model: str | None) -> dict[str, object] | None:
    if not model:
        return None
    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    return model_config if isinstance(model_config, dict) else None
