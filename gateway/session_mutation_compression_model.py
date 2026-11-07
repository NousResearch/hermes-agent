"""Explicitly replace one local conversation's frozen compression route."""

import asyncio
import copy
import json
from dataclasses import asdict, replace

from hermes_state_runtime import RuntimeStoreError


def _compression_section(config):
    auxiliary = config.get("auxiliary")
    if auxiliary is None:
        return {}
    if not isinstance(auxiliary, dict):
        raise RuntimeStoreError(
            "invalid_params", field="auxiliary", detail="must be a mapping"
        )
    block = auxiliary.get("compression")
    if block is not None and not isinstance(block, dict):
        raise RuntimeStoreError(
            "invalid_params", field="auxiliary.compression", detail="must be a mapping"
        )
    return copy.deepcopy(block or {})


async def prepare_compression_model(authority, live, payload, prepared):
    return await asyncio.to_thread(_prepare, authority, prepared)


def _prepare(authority, prepared):
    from gateway.session_policy import (
        _extract_config_secrets,
        bind_launch_key,
        launch_key,
        profile_config,
        restore_policy,
    )
    from gateway.session_policy_credentials import (
        disable_dropped_secrets,
        recover_config_secrets,
    )

    old = restore_policy(prepared["snapshot"]["receipt"]["policy"])
    # A bypass session must never start reading profile settings or credentials.
    current = {} if old.ignore_user_config else profile_config(authority)
    block = _compression_section(current)
    frozen = json.loads(old.config_json)
    secrets = recover_config_secrets(authority, old) if old.config_secret_ref else {}
    disable_dropped_secrets(
        frozen, getattr(secrets, "dropped", ()), getattr(secrets, "session", None)
    )
    secrets = {
        path: value
        for path, value in secrets.items()
        if path[:2] != ("auxiliary", "compression")
    }
    _extract_config_secrets(block, secrets, ("auxiliary", "compression"))
    # Store an explicit empty task too: this is an intentional reset to automatic selection.
    # It must not accidentally restore the retired route from the old frozen block.
    frozen["auxiliary"] = {**(frozen.get("auxiliary") or {}), "compression": block}
    policy = replace(old, config_json=json.dumps(frozen))
    policy = bind_launch_key(
        authority,
        prepared["snapshot"]["receipt"]["session_id"],
        policy,
        launch_key(authority, old),
        config_secrets=secrets,
    )
    return dict(prepared, policy=asdict(policy))


def publish_compression_model(authority, ref):
    """Keep the cached prompt, but stop pinning a warm compressor to its earlier fallback."""
    from gateway.session_policy import policy_for_source, restore_policy
    from hermes_state_local import local_receipt

    live = authority.sessions[ref.session_id]
    previous = policy_for_source(authority.runner, live.source)
    committed = restore_policy(local_receipt(authority.db, ref.session_id)["policy"])
    if previous == committed:
        return  # Exact retry or cold recovery must not reset a new turn's fallback.
    agent = authority.agent(ref)
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        compressor.reset_summary_route()
