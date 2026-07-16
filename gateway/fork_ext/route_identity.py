"""Pure gateway route-identity helpers."""

from __future__ import annotations

from typing import Any, Callable


def configured_route_identity(
    user_config: dict | None,
    infer_api_mode_from_provider: Callable[[str], str],
) -> tuple[str, str, str]:
    """Read model/provider/API mode from config without resolving credentials."""
    cfg = user_config if isinstance(user_config, dict) else {}
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg, "", ""
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model = str(model_cfg.get("default") or model_cfg.get("model") or "")
    provider = str(model_cfg.get("provider") or "").strip().lower()
    api_mode = str(model_cfg.get("api_mode") or "").strip().lower()
    if not api_mode:
        api_mode = infer_api_mode_from_provider(provider) if provider else ""
    return model, provider, api_mode


def persisted_session_route_identity(
    session_store: Any,
    session_key: str,
    persisted_lookup_type: type,
) -> Any:
    """Return authoritative ABSENT/VALID/UNAVAILABLE persisted route state."""
    if session_store is None or not session_key:
        return persisted_lookup_type("absent")

    # Inspect the class, not the instance: Mock/MagicMock and lightweight
    # fixture stores synthesize arbitrary attributes on demand and must not
    # accidentally claim authority over persisted routing state.
    lookup = getattr(type(session_store), "lookup_persisted_route_identity", None)
    if not callable(lookup):
        return persisted_lookup_type("absent")
    try:
        result = lookup(session_store, session_key)
    except Exception:
        return persisted_lookup_type("unavailable")
    if not isinstance(result, persisted_lookup_type):
        return persisted_lookup_type("unavailable")
    return result
