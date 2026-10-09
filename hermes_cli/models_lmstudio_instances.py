"""Track only the LM Studio instances that this process loaded for each profile."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from hermes_constants import hermes_home_key

_owned_instances: dict[tuple[str, str, str], str] = {}


def _claim_key(server_root: str, model: str) -> tuple[str, str, str]:
    return hermes_home_key(), server_root, model


def remember_instance(server_root: str, model: str, instance_id: str) -> None:
    """Keep an exact load-response ID within the current profile."""
    _owned_instances[_claim_key(server_root, model)] = instance_id


def owned_instance(server_root: str, model: str) -> str | None:
    return _owned_instances.get(_claim_key(server_root, model))


def forget_instance(server_root: str, model: str) -> None:
    _owned_instances.pop(_claim_key(server_root, model), None)


def normalize_lmstudio_unload_policy(value) -> str:
    """Keep resident models when the setting disables unload."""
    disabled = {"never", "off", "no", "none", "false", "disabled", "disable", "0", "keep"}
    return "never" if str(value).strip().lower() in disabled else "always"


def unload_previous_instance(server_root, previous_model, raw_models, headers, timeout) -> dict | None:
    """Unload the exact owned ID, after the catalog confirms that it is still present."""
    from hermes_cli.models import _urlopen_model_catalog_request
    from hermes_cli.models_local import _lmstudio_entry_for

    instance_id = owned_instance(server_root, previous_model)
    if not instance_id:
        return
    entry = _lmstudio_entry_for(raw_models, previous_model)
    instances = entry.get("loaded_instances") if entry else None
    resident = next((item for item in instances if isinstance(item, dict) and item.get("id") == instance_id), None) if isinstance(instances, list) else None
    if resident is None:
        forget_instance(server_root, previous_model)
        return
    request = urllib.request.Request(
        server_root + "/api/v1/models/unload",
        data=json.dumps({"instance_id": instance_id}).encode(),
        headers={**headers, "Content-Type": "application/json"}, method="POST",
    )
    try:
        with _urlopen_model_catalog_request(request, timeout=timeout):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        forget_instance(server_root, previous_model)
        return None
    forget_instance(server_root, previous_model)
    return {"model": previous_model, "config": resident.get("config")}


def restore_previous_if_failed(previous, context_length, server_root, headers, timeout) -> None:
    """Restore the previous model if the destination load has no verified context."""
    if previous is None or context_length is not None:
        return
    from hermes_cli.models import _urlopen_model_catalog_request
    from hermes_cli.models_local import _positive_int

    payload = {"model": previous["model"], "echo_load_config": True}
    config = previous.get("config")
    context = _positive_int(config.get("context_length")) if isinstance(config, dict) else None
    if context is not None:
        payload["context_length"] = context
    request = urllib.request.Request(
        server_root + "/api/v1/models/load", data=json.dumps(payload).encode(),
        headers={**headers, "Content-Type": "application/json"}, method="POST",
    )
    try:
        with _urlopen_model_catalog_request(request, timeout=timeout) as response:
            result = json.loads(response.read().decode())
        instance_id = result.get("instance_id") if isinstance(result, dict) else None
        load_config = result.get("load_config") if isinstance(result, dict) else None
        loaded_context = _positive_int(load_config.get("context_length")) if isinstance(load_config, dict) else None
        if isinstance(instance_id, str) and instance_id and loaded_context is not None:
            remember_instance(server_root, previous["model"], instance_id)
    except Exception as exc:
        logging.getLogger(__name__).warning("LM Studio could not restore %s: %s", previous["model"], exc, exc_info=True)
