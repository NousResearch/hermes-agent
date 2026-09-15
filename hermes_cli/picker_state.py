"""Read-only presentation of persisted pool state, never runtime eligibility.

Use the existing store reader's precedence/suppression contract. Do not load a
CredentialPool here: loading may seed, heal, prune, refresh or persist state.
"""
from datetime import datetime, timezone
import math
from typing import Any
import time


def recorded_catalog_models(provider: str, fallback: list[str]) -> list[str]:
    """Cache/static-only catalog for newly visible rejected/cooling credentials.

    A normal cache miss can refresh OAuth; rendering a warning must not cause it.
    """
    from hermes_cli.models import (
        _load_provider_models_cache, _credential_fingerprint, _cache_entry_valid,
        _model_requires_account_discovery,
    )
    entry = _load_provider_models_cache().get(provider)
    if _cache_entry_valid(entry, _credential_fingerprint(provider)):
        return [m for m in entry["models"] if not _model_requires_account_discovery(provider, m)]
    if provider == "openai-codex":
        from hermes_cli.codex_models import DEFAULT_CODEX_MODELS
        fallback = DEFAULT_CODEX_MODELS
    return list(fallback)


def recorded_pool_state(provider: str) -> dict:
    from hermes_cli.auth import read_credential_pool
    from agent.credential_pool import _parse_absolute_timestamp

    try:
        entries: Any = read_credential_pool(provider)
    except Exception:
        return {}
    entries = [e for e in entries if isinstance(e, dict)
               and isinstance(e.get("access_token"), str) and e["access_token"].strip()]
    if not entries:
        return {}
    states = [e.get("last_status") for e in entries]
    result = {"credential_present": True, "auth_state": "present", "available": None,
              "availability_source": "recorded_pool"}
    if all(s == "dead" for s in states):
        result.update(auth_state="invalid", available=False,
                      warning="Stored credentials are marked dead; authorization needs attention (not checked remotely).")
    elif any(s == "exhausted" for s in states):
        resets = []
        for entry in entries:
            if entry.get("last_status") != "exhausted":
                continue
            reset = _parse_absolute_timestamp(entry.get("last_error_reset_at"))
            if reset is not None and math.isfinite(reset):
                resets.append(reset)
        # A recorded reset is not a fresh quota check or a per-model limit.
        if (all(s in ("dead", "exhausted") for s in states) and resets
                and len(resets) == states.count("exhausted") and min(resets) > time.time()):
            result["available"] = False
        result["warning"] = "Credential pool has a recorded limit; remote availability and model scope not checked."
        if resets:
            try:
                reset_text = datetime.fromtimestamp(min(resets), timezone.utc).isoformat()
            except (ValueError, OverflowError, OSError):
                pass
            else:
                result["reset_at"] = reset_text
                result["warning"] += f" Recorded reset: {reset_text}."
    return result
