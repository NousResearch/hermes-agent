"""Digest current configuration for future auto-route cache lookups."""
import hashlib
import json


def routing_configuration_key(task_config: dict) -> bytes:
    from hermes_cli.config import load_config_readonly
    # Named providers, inherited main fallbacks and plugin task defaults all affect
    # auto resolution. Retain only a digest; existing bounded cache owns old clients.
    snapshot = {"profile_config": load_config_readonly(), "effective_task": task_config}
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, default=str).encode()).digest()
