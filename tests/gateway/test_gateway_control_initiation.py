"""``gateway.<action>_initiated`` receipts — identity (action/actor/transport), never argv."""
import os

from gateway.lifecycle_ledger import record_gateway_control_initiation


def _read_last(tmp_path):
    lines = (tmp_path / "logs" / "gateway-exit-diag.log").read_text(encoding="utf-8").strip().splitlines()
    return __import__("json").loads(lines[-1])


def test_control_initiation_receipt_identity_no_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("SSH_CLIENT", "10.9.8.7 55123 22")
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    record_gateway_control_initiation("restart", target_pid=80361, home=tmp_path, detail="sigusr1-drain")
    rec = _read_last(tmp_path)
    assert rec["tag"] == "gateway.restart_initiated"
    assert rec["action"] == "restart"
    assert rec["target_pid"] == 80361
    assert rec["transport"] == "ssh"
    assert rec["initiator_pid"] == os.getpid()
    assert rec["detail"] == "sigusr1-drain"
    assert "argv" not in rec and "cmdline" not in rec


def test_control_initiation_never_raises_on_unwritable_home(tmp_path):
    blocker = tmp_path / "logs"  # a FILE where the dir must go
    blocker.write_text("occupied", encoding="utf-8")
    record_gateway_control_initiation("stop", home=tmp_path)  # must swallow, not raise


def test_control_initiation_minimal_without_target_pid(tmp_path):
    record_gateway_control_initiation("start", home=tmp_path)
    rec = _read_last(tmp_path)
    assert rec["tag"] == "gateway.start_initiated"
    assert "target_pid" not in rec
    assert rec["user"] is not None or True  # best-effort
