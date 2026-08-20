import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from types import ModuleType

import pytest

from plugins.builder_adapter.attestation import (
    ArtifactBinding,
    GovernanceAttestor,
    HermesProfileResolver,
)
from plugins.builder_adapter.errors import AdapterError
from tests.plugins.test_builder_adapter_schema import POLICY


def _governance_repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    policy_path = path / "providers/policies/deepseek-builder.v1.json"
    interface_path = path / "providers/interfaces/hermes.builder_dispatch.v1.json"
    policy_path.parent.mkdir(parents=True)
    interface_path.parent.mkdir(parents=True)
    policy_raw = (json.dumps(POLICY, indent=2) + "\n").encode()
    policy_path.write_bytes(policy_raw)
    interface_raw = (
        json.dumps(
            {
                "capability_id": "hermes.builder_dispatch.v1",
                "routing_policy": {
                    "policy_artifact": {
                        "path": "providers/policies/deepseek-builder.v1.json",
                        "sha256": hashlib.sha256(policy_raw).hexdigest(),
                    }
                },
            },
            indent=2,
        )
        + "\n"
    ).encode()
    interface_path.write_bytes(interface_raw)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "governance"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, policy_raw, interface_raw


def test_authoritative_policy_and_interface_bytes_are_hash_bound(tmp_path):
    commit, policy_raw, interface_raw = _governance_repo(tmp_path)
    attestor = GovernanceAttestor(
        tmp_path,
        policy=ArtifactBinding(
            "providers/policies/deepseek-builder.v1.json",
            commit,
            hashlib.sha256(policy_raw).hexdigest(),
        ),
        interface=ArtifactBinding(
            "providers/interfaces/hermes.builder_dispatch.v1.json",
            commit,
            hashlib.sha256(interface_raw).hexdigest(),
        ),
    )
    policy, interface = attestor.load()
    assert policy["provider"] == "deepseek"
    assert interface["capability_id"] == "hermes.builder_dispatch.v1"
    bad = GovernanceAttestor(
        tmp_path,
        policy=ArtifactBinding(
            "providers/policies/deepseek-builder.v1.json", commit, "0" * 64
        ),
        interface=attestor.interface_binding,
    )
    with pytest.raises(AdapterError) as raised:
        bad.load()
    assert raised.value.code == "PROFILE_POLICY_MISMATCH"


def test_effective_profile_is_observed_through_public_interfaces(monkeypatch, tmp_path):
    policy = deepcopy(POLICY)
    policy["allowed_tools"] = [
        "builder_patch",
        "builder_read_execution_packet",
        "builder_read_file",
        "builder_run_validation_profile",
        "builder_search_files",
        "builder_write_file",
        "kanban_block",
        "kanban_complete",
        "kanban_heartbeat",
    ]
    constants = ModuleType("hermes_constants")
    constants.set_hermes_home_override = lambda value: value
    constants.reset_hermes_home_override = lambda token: None
    cli = ModuleType("hermes_cli")
    config = ModuleType("hermes_cli.config")
    profiles = ModuleType("hermes_cli.profiles")
    plugins = ModuleType("hermes_cli.plugins")
    plugins.discover_plugins = lambda force=False: None
    profiles.profile_exists = lambda name: name == "deepseek-builder"
    profiles.get_profile_dir = lambda name: tmp_path / name
    observed = {
        "model": {
            "provider": "deepseek",
            "default": "deepseek-v4-pro",
        },
        "fallback_providers": [],
        "platform_toolsets": {"cli": ["builder_adapter", "no_mcp"]},
        "plugins": {"enabled": ["builder_adapter"]},
        "builder_dispatch": {
            "confinement": {
                "kind": "application_tool_mediated",
                "os_sandbox": False,
                "terminal_tools": False,
                "process_tools": False,
            },
        },
    }
    config.load_config_readonly = lambda: observed
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)
    monkeypatch.setitem(sys.modules, "hermes_cli", cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config)
    monkeypatch.setitem(sys.modules, "hermes_cli.profiles", profiles)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", plugins)
    monkeypatch.setattr(
        "model_tools._compute_tool_definitions",
        lambda *_args, **_kwargs: [
            {"function": {"name": name}} for name in policy["allowed_tools"]
        ],
    )
    effective = HermesProfileResolver().resolve(policy)
    assert effective.profile == "deepseek-builder"
    assert effective.provider == "deepseek"
    assert effective.model == "deepseek-v4-pro"
    assert effective.fallback_chain == ()
    assert effective.evidence()["profile_configuration_sha256"]

    observed["fallback_providers"] = [
        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}
    ]
    with pytest.raises(AdapterError) as raised:
        HermesProfileResolver().resolve(policy)
    assert raised.value.code == "PROFILE_POLICY_MISMATCH"
