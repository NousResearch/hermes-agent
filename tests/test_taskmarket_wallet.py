import pytest

from hermes_cli import taskmarket_wallet as tmw


def test_get_taskmarket_init_command_prefers_taskmarket(monkeypatch):
    monkeypatch.setattr(tmw.shutil, "which", lambda name: "/usr/local/bin/taskmarket" if name == "taskmarket" else None)
    assert tmw.get_taskmarket_init_command() == ["taskmarket", "init"]


def test_get_taskmarket_init_command_falls_back_to_npx(monkeypatch):
    monkeypatch.setattr(tmw.shutil, "which", lambda name: "/usr/local/bin/npx" if name == "npx" else None)
    assert tmw.get_taskmarket_init_command() == ["npx", "-y", "@lucid-agents/taskmarket@latest", "init"]


def test_ensure_taskmarket_wallet_initialized_runs_init(monkeypatch, tmp_path):
    keystore_path = tmp_path / "keystore.json"
    monkeypatch.setenv("X402_TASKMARKET_KEYSTORE_PATH", str(keystore_path))
    monkeypatch.setattr(tmw, "get_taskmarket_init_command", lambda: ["taskmarket", "init"])

    calls = {}

    def _fake_run(cmd, check=False):
        calls["cmd"] = cmd
        keystore_path.write_text('{"walletAddress":"0xabc","deviceId":"dev","apiToken":"tok","encryptedKey":"deadbeef"}')
        class _Result:
            returncode = 0
        return _Result()

    monkeypatch.setattr(tmw.subprocess, "run", _fake_run)

    result = tmw.ensure_taskmarket_wallet_initialized()

    assert calls["cmd"] == ["taskmarket", "init"]
    assert result["walletAddress"] == "0xabc"


def test_ensure_taskmarket_wallet_initialized_raises_when_no_launcher(monkeypatch):
    monkeypatch.setattr(tmw.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError):
        tmw.get_taskmarket_init_command()
