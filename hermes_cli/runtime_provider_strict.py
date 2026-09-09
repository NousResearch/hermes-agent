"""Explicit primary-route validation; auxiliary routing remains independently configured."""


def validate_explicit_runtime(requested, model, runtime):
    actual = runtime.get("provider")
    matches = actual == requested
    if actual == "custom" and requested != "custom":
        from hermes_cli.runtime_provider_custom import _get_named_custom_provider
        entry = _get_named_custom_provider(requested)
        matches = bool(entry and entry.get("base_url", "").rstrip("/") == runtime.get("base_url", "").rstrip("/"))
    if not matches or runtime.get("model", model) != model:
        raise ValueError("Explicit provider/model could not be resolved without changing route")
    overrides = runtime.get("request_overrides") or {}
    for payload in (overrides, overrides.get("extra_body") or {}):
        if any(key in payload for key in ("model", "provider", "fallback", "fallbacks", "models")):
            raise ValueError("Provider request overrides cannot change a strict primary route")
    return runtime
