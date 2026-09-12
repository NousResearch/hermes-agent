"""TDD contract tests for operator-principal policy config mutations."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from hermes_cli import config
from hermes_cli.policy_mutation import (
    POLICY_CONFIG_KEYS,
    PolicyMutationBroker,
    PolicyMutationDenied,
    is_policy_config_key,
)


def test_shared_policy_classification_covers_policy_key_families():
    assert is_policy_config_key("approvals.mode")
    assert is_policy_config_key("security.tirith_enabled")
    assert is_policy_config_key("command_allowlist")
    assert is_policy_config_key("command_allowlist.commands")
    assert is_policy_config_key("yolo")
    assert is_policy_config_key("persistent.yolo")
    assert "approvals." in POLICY_CONFIG_KEYS
    assert not is_policy_config_key("display.skin")
    assert not is_policy_config_key("agent.max_turns")


def test_proof_is_bound_one_shot_and_expires(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    broker = PolicyMutationBroker(ttl_seconds=60)
    request = broker.request("session-1", "approvals.mode", "set")
    proof = broker.operator_confirm(request.request_id)
    assert proof.policy_digest == broker.policy_digest
    assert proof.target_key == "approvals.mode"
    assert proof.operation == "set"
    assert broker.consume(proof, "approvals.mode", "set", "session-1")
    with pytest.raises(PolicyMutationDenied, match="replay"):
        broker.consume(proof, "approvals.mode", "set", "session-1")

    request = broker.request("session-2", "security.tirith_enabled", "unset")
    proof = broker.operator_confirm(request.request_id)
    monkeypatch.setattr(time, "time", lambda: proof.expires_at + 1)
    with pytest.raises(PolicyMutationDenied, match="expired"):
        broker.consume(proof, "security.tirith_enabled", "unset", "session-2")


def test_policy_writer_refuses_without_proof_and_ordinary_key_stays_writable(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text("approvals:\n  mode: manual\ndisplay:\n  skin: default\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(PolicyMutationDenied):
        config.set_config_value("approvals.mode", "off")
    assert path.read_bytes() == before
    config.set_config_value("display.skin", "dark")
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["display"]["skin"] == "dark"


def test_operator_proof_allows_writer_then_cache_is_fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    assert config.load_config_readonly()["approvals"]["mode"] == "manual"
    broker = PolicyMutationBroker()
    request = broker.request("session-3", "approvals.mode", "set")
    proof = broker.operator_confirm(request.request_id)
    config.set_config_value("approvals.mode", "off", proof=proof, session_id="session-3")
    assert config.load_config_readonly()["approvals"]["mode"] == "off"


def test_subprocess_agent_attempt_cannot_mutate_policy_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text("approvals:\n  mode: manual\nsecurity:\n  tirith_enabled: true\n", encoding="utf-8")
    before = path.read_bytes()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[2])
    env["HERMES_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "config", "set", "approvals.mode", "off"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert path.read_bytes() == before
    print(f"SUBPROCESS_RC={result.returncode}")
    print(f"SUBPROCESS_STDERR={result.stderr.strip()!r}")
    print(f"POLICY_SHA256={hashlib.sha256(path.read_bytes()).hexdigest()}")
