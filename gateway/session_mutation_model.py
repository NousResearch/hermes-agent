"""Prepare gateway-owned session model mutations over canonical domains."""
import asyncio
import json
from dataclasses import asdict, replace

from hermes_state_runtime import RuntimeStoreError


async def prepare_model(authority, live, payload, prepared):
    from gateway.session_model_resolution import resolve_session_model
    from gateway.session_policy import restore_policy, launch_key, bind_launch_key
    from gateway.session_policy_credentials import recover_config_secrets

    old = restore_policy(prepared["snapshot"]["receipt"]["policy"])
    config = old.config(authority)
    try:
        result = await asyncio.to_thread(
            resolve_session_model,
            config=config,
            raw_model=payload["model"],
            explicit_provider=payload.get("provider", ""),
            current_provider=old.provider or "",
            current_base_url=old.base_url or "",
        )
    except Exception as exc:
        raise RuntimeStoreError("model_resolution_failed") from exc

    frozen = old.config()
    frozen.setdefault("model", {}).update(
        default=result.model,
        provider=result.provider,
        base_url=result.base_url,
    )
    policy = replace(old, model=result.model, config_json=json.dumps(frozen))
    secrets = recover_config_secrets(authority, old) if old.config_secret_ref else {}

    key = launch_key(authority, old)
    if result.provider_changed:
        policy = replace(policy, credential_ref=None)
        key = None
    policy = bind_launch_key(
        authority,
        prepared["snapshot"]["receipt"]["session_id"],
        policy,
        key,
        config_secrets=secrets,
    )
    return dict(prepared, policy=asdict(policy))
