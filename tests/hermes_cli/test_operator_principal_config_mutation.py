"""TDD contract tests for operator-principal policy config mutations."""

from __future__ import annotations

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


def test_direct_operator_confirm_without_surface_settlement_is_denied(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    broker = PolicyMutationBroker(ttl_seconds=60)
    request = broker.request("session-1", "approvals.mode", "set")
    with pytest.raises(PolicyMutationDenied, match="settlement"):
        broker.operator_confirm(request.request_id)


def test_proof_is_bound_one_shot_and_expires(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    broker = PolicyMutationBroker(ttl_seconds=60)
    request = broker.request("session-1", "approvals.mode", "set")
    from hermes_cli.policy_mutation import _operator_settlement_scope
    broker.record_settlement(request.request_id, "test-confirm")
    with _operator_settlement_scope(request.request_id, "test-confirm"):
        proof = broker.operator_confirm(request.request_id)
    assert proof.policy_digest == broker.policy_digest
    assert proof.target_key == "approvals.mode"
    assert proof.operation == "set"
    assert broker.consume(proof, "approvals.mode", "set", "session-1")
    with pytest.raises(PolicyMutationDenied, match="replay"):
        broker.consume(proof, "approvals.mode", "set", "session-1")

    request = broker.request("session-2", "security.tirith_enabled", "unset")
    broker.record_settlement(request.request_id, "test-confirm-2")
    with _operator_settlement_scope(request.request_id, "test-confirm-2"):
        proof = broker.operator_confirm(request.request_id)
    monkeypatch.setattr(time, "time", lambda: proof.expires_at + 1)
    with pytest.raises(PolicyMutationDenied, match="expired"):
        broker.consume(proof, "security.tirith_enabled", "unset", "session-2")


@pytest.mark.parametrize("operation", ["set", "unset"])
def test_policy_writer_refuses_without_proof_byte_identical(monkeypatch, tmp_path, operation):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_bytes(b"approvals:\n  mode: manual\ndisplay:\n  skin: default\n")
    before = path.read_bytes()
    writer = config.set_config_value if operation == "set" else config.unset_config_value
    args = ("approvals.mode", "off") if operation == "set" else ("approvals.mode",)
    with pytest.raises(PolicyMutationDenied) as exc_info:
        writer(*args)
    assert str(exc_info.value) == "operator confirmation proof required"
    assert path.read_bytes() == before


def test_tui_raw_writer_refuses_policy_mutation_without_proof_byte_identical(monkeypatch, tmp_path):
    monkeypatch.setattr("tui_gateway.server._hermes_home", tmp_path)
    path = tmp_path / "config.yaml"
    path.write_bytes(b"approvals:\n  mode: manual\n")
    before = path.read_bytes()
    from tui_gateway.server import _write_config_key
    with pytest.raises(PolicyMutationDenied) as exc_info:
        _write_config_key("approvals.mode", "off")
    assert str(exc_info.value) == "operator confirmation proof required"
    assert path.read_bytes() == before


def test_ordinary_key_stays_writable(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text("display:\n  skin: default\n", encoding="utf-8")
    config.set_config_value("display.skin", "dark")
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["display"]["skin"] == "dark"


def test_operator_proof_allows_writer_then_cache_is_fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    assert config.load_config_readonly()["approvals"]["mode"] == "manual"
    broker = PolicyMutationBroker()
    request = broker.request("session-3", "approvals.mode", "set")
    from hermes_cli.policy_mutation import _operator_settlement_scope
    broker.record_settlement(request.request_id, "test-confirm")
    with _operator_settlement_scope(request.request_id, "test-confirm"):
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


def _mint(broker, request, confirmation_id="test-confirm"):
    from hermes_cli.policy_mutation import _operator_settlement_scope
    broker.record_settlement(request.request_id, confirmation_id)
    with _operator_settlement_scope(request.request_id, confirmation_id):
        return broker.operator_confirm(request.request_id)


def test_cross_key_and_session_replay_is_refused_without_consuming(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    broker = PolicyMutationBroker()
    request = broker.request("s1", "approvals.mode", "set")
    proof = _mint(broker, request)
    with pytest.raises(PolicyMutationDenied, match="binding") as exc_info:
        broker.consume(proof, "security.tirith_enabled", "set", "s2")
    assert str(exc_info.value) == "proof binding mismatch"
    assert broker.consume(proof, "approvals.mode", "set", "s1")


def test_save_config_refuses_policy_payload_without_proof_byte_identical(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_bytes(b"approvals:\n  mode: manual\ndisplay:\n  skin: default\n")
    before = path.read_bytes()
    with pytest.raises(PolicyMutationDenied) as exc_info:
        config.save_config({"approvals": {"mode": "off"}, "display": {"skin": "default"}})
    assert str(exc_info.value) == "operator confirmation proof required"
    assert path.read_bytes() == before


def test_forged_proof_refusal_output(monkeypatch, tmp_path):
    """A proof-shaped payload cannot bypass the ledger-backed save_config sink."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_bytes(b"approvals:\n  mode: manual\n")
    before = path.read_bytes()
    from hermes_cli.policy_mutation import PolicyMutationProof, policy_config_digest
    forged = PolicyMutationProof(
        token="forged-token", policy_digest=policy_config_digest(path),
        target_key="approvals.mode", operation="set", session_id="session-1",
        nonce="nonexistent-nonce", expires_at=time.time() + 60,
    )
    with pytest.raises(PolicyMutationDenied) as exc_info:
        config.save_config({"approvals": {"mode": "off"}}, proof=forged, session_id="session-1")
    assert str(exc_info.value) == "proof replay or unknown nonce"
    assert path.read_bytes() == before


def test_replay_refusal_output(monkeypatch, tmp_path):
    """The bulk save sink consumes a real proof and refuses reusing it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_bytes(b"approvals:\n  mode: manual\n")
    broker = PolicyMutationBroker()
    proof = _mint(broker, broker.request("session-1", "approvals.mode", "set"))
    config.save_config({"approvals": {"mode": "off"}}, proof=proof, session_id="session-1")
    before = path.read_bytes()
    with pytest.raises(PolicyMutationDenied) as exc_info:
        config.save_config({"approvals": {"mode": "manual"}}, proof=proof, session_id="session-1")
    assert str(exc_info.value) == "proof replay or unknown nonce"
    assert path.read_bytes() == before


def test_policy_consume_is_exactly_once_under_concurrency(monkeypatch, tmp_path):
    import concurrent.futures
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    broker = PolicyMutationBroker()
    request = broker.request("s1", "approvals.mode", "set")
    proof = _mint(broker, request)

    def consume():
        try:
            return broker.consume(proof, "approvals.mode", "set", "s1")
        except PolicyMutationDenied as exc:
            return str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _n: consume(), range(16)))
    assert results.count(True) == 1
    assert sum(isinstance(result, str) and "replay" in result for result in results) == 15


def test_policy_write_fails_closed_when_state_db_is_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_bytes(b"approvals:\n  mode: manual\n")
    before = path.read_bytes()
    (tmp_path / "state.db").write_bytes(b"not sqlite")
    with pytest.raises(PolicyMutationDenied, match="ledger") as exc_info:
        config.save_config({"approvals": {"mode": "off"}})
    assert str(exc_info.value) == "policy ledger unavailable"
    assert path.read_bytes() == before


def test_save_config_fails_closed_when_policy_ledger_is_unavailable(monkeypatch, tmp_path):
    """A state.db path that cannot be opened is ledger unavailability, not absence."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_bytes(b"approvals:\n  mode: manual\n")
    before = path.read_bytes()
    (tmp_path / "state.db").mkdir()
    with pytest.raises(PolicyMutationDenied, match="ledger") as exc_info:
        config.save_config({"approvals": {"mode": "off"}})
    assert str(exc_info.value) == "policy ledger unavailable"
    assert path.read_bytes() == before


def test_config_env_route_refuses_policy_payload_without_proof_byte_identical(monkeypatch, tmp_path):
    """PUT /api/config uses the shared save_config sink; approvals.mode without proof is refused."""
    import asyncio
    from hermes_cli.web_models import ConfigUpdate
    from hermes_cli.web_routers.config_env import update_config
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_bytes(b"approvals:\n  mode: manual\n")
    before = path.read_bytes()
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(update_config(ConfigUpdate(config={"approvals": {"mode": "off"}})))
    assert exc_info.value.status_code == 500
    assert path.read_bytes() == before


def test_analytics_raw_route_refuses_policy_payload_without_proof_byte_identical(monkeypatch, tmp_path):
    """PUT /api/config/raw uses the shared save_config sink; policy YAML without proof is refused."""
    import asyncio
    from hermes_cli.web_models import RawConfigUpdate
    from hermes_cli.web_routers.analytics import update_config_raw
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_bytes(b"approvals:\n  mode: manual\n")
    before = path.read_bytes()
    with pytest.raises(PolicyMutationDenied):
        asyncio.run(update_config_raw(RawConfigUpdate(yaml_text="approvals:\n  mode: off\n")))
    assert path.read_bytes() == before


def test_save_permanent_allowlist_refuses_policy_mutation_without_proof_byte_identical(monkeypatch, tmp_path):
    """save_permanent_allowlist reaches save_config with command_allowlist and no proof."""
    import tools.approval as approval
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_bytes(b"command_allowlist: []\n")
    before = path.read_bytes()
    approval.save_permanent_allowlist({"git status"})
    assert path.read_bytes() == before


def test_save_config_value_refuses_policy_mutation_without_proof_byte_identical(monkeypatch, tmp_path):
    """cli.save_config_value reaches its policy sink and refuses approvals.mode without proof."""
    from cli import save_config_value
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_bytes(b"approvals:\n  mode: manual\n")
    before = path.read_bytes()
    with pytest.raises(PolicyMutationDenied) as exc_info:
        save_config_value("approvals.mode", "off")
    assert str(exc_info.value) == "operator confirmation proof required"
    assert path.read_bytes() == before


def test_operator_confirm_requires_recorded_settlement_receipt(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    broker = PolicyMutationBroker()
    request = broker.request("s1", "approvals.mode", "set")
    from hermes_cli.policy_mutation import _operator_settlement_scope
    with _operator_settlement_scope():
        with pytest.raises(PolicyMutationDenied, match="receipt"):
            broker.operator_confirm(request.request_id)
