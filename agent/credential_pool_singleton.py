"""Singleton admission into credential pools, preserving known token exclusions."""

from dataclasses import replace


def _exclusion_rank(entry):
    from agent.credential_pool import STATUS_DEAD, _exhausted_until
    return entry.last_status == STATUS_DEAD, _exhausted_until(entry) or 0


def _inherit_codex_alias_exclusions(seed):
    """A new source name is not a new credential; preserve same-token, same-route exclusions."""
    from agent.credential_pool import STATUS_DEAD, STATUS_EXHAUSTED, _CLEAR_STATUS
    from agent.credential_pool_endpoints import entry_endpoint
    from agent.credential_pool_model_cooldowns import merge_model_cooldowns

    for index, alias in enumerate(seed.entries):
        if alias.source != "device_code":
            continue
        matching = [entry for entry in seed.entries if entry.access_token == alias.access_token
                    and entry_endpoint(entry) == entry_endpoint(alias)]
        updates = {"model_cooldowns": merge_model_cooldowns(*(entry.model_cooldowns for entry in matching)) or None}
        donor = max(matching, key=_exclusion_rank)
        if donor.last_status in {STATUS_DEAD, STATUS_EXHAUSTED}:
            updates.update({key: getattr(donor, key) for key in _CLEAR_STATUS})
            extra = dict(alias.extra)
            extra.pop("failure_reason", None)
            if donor.failure_reason:
                extra["failure_reason"] = donor.failure_reason
            updates["extra"] = extra
        updated = replace(alias, **updates)
        if updated != alias:
            seed.entries[index] = updated
            seed.changed = True


def seed_tokens_singleton(seed, auth_store: dict) -> None:
    """Codex / xAI: surface the auth.json ``providers.<id>.tokens`` singleton as ``device_code``.

    Hermes owns its own Codex auth state and does NOT auto-import
    ~/.codex/auth.json: refresh tokens are single-use, so sharing them with
    Codex CLI / VS Code causes refresh_token_reused races. Adoption is an
    explicit one-time prompt via `hermes auth openai-codex`.
    """
    from agent.credential_pool import (
        AUTH_TYPE_OAUTH, _load_provider_state_with_source, auth_mod, label_from_token,
        owned_profile_reads_root_state,
    )
    state, source_path = _load_provider_state_with_source(auth_store, seed.provider)
    if owned_profile_reads_root_state(auth_store, seed.provider, source_path):
        return
    tokens = state.get("tokens") if isinstance(state, dict) else None
    if not (isinstance(tokens, dict) and tokens.get("access_token")):
        return
    if seed.provider == "openai-codex":
        base_url = auth_mod.DEFAULT_CODEX_BASE_URL
        custom_label = str(state.get("label") or "").strip()
    else:
        base_url = auth_mod.DEFAULT_XAI_OAUTH_BASE_URL
        custom_label = ""
    seed.upsert("device_code", {
        "auth_type": AUTH_TYPE_OAUTH,
        "access_token": tokens.get("access_token", ""),
        "refresh_token": tokens.get("refresh_token"),
        "base_url": base_url,
        "last_refresh": state.get("last_refresh"),
        "label": custom_label or label_from_token(tokens.get("access_token", ""), "device_code"),
    })
    if seed.provider == "openai-codex":
        _inherit_codex_alias_exclusions(seed)
