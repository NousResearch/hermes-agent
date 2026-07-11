from __future__ import annotations

from types import SimpleNamespace

from agent.local_backend_recovery import maybe_recover_local_backend


def test_recovery_hook_skips_remote_endpoint(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_BACKEND_RECOVERY_COMMAND", "echo recover")

    called = {"run": False}

    def fake_run(*_args, **_kwargs):
        called["run"] = True

    monkeypatch.setattr("subprocess.run", fake_run)

    assert not maybe_recover_local_backend(
        SimpleNamespace(provider="openai"),
        reason="stale",
        model="gpt-5.4",
        base_url="https://api.openai.com/v1",
    )
    assert called["run"] is False


def test_recovery_hook_runs_command_with_sanitized_metadata(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_BACKEND_RECOVERY_COMMAND", "recover-local --flag")
    monkeypatch.setenv("HERMES_LOCAL_BACKEND_RECOVERY_COOLDOWN", "0")

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    agent = SimpleNamespace(provider="local", session_id="s1")
    assert maybe_recover_local_backend(
        agent,
        reason="local_first_chunk_timeout",
        model="local-q4",
        base_url="http://127.0.0.1:9090/v1",
        context_tokens=1234,
        elapsed=75.5,
        threshold=75.0,
    )

    assert captured["argv"] == ["recover-local", "--flag"]
    env = captured["env"]
    assert env["HERMES_RECOVERY_REASON"] == "local_first_chunk_timeout"
    assert env["HERMES_RECOVERY_MODEL"] == "local-q4"
    assert env["HERMES_RECOVERY_PROVIDER"] == "local"
    assert env["HERMES_RECOVERY_BASE_URL"] == "http://127.0.0.1:9090/v1"
    assert env["HERMES_RECOVERY_CONTEXT_TOKENS"] == "1234"
    assert env["HERMES_RECOVERY_SESSION_ID"] == "s1"


def test_recovery_hook_enforces_process_cooldown(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_BACKEND_RECOVERY_COMMAND", "recover-local")
    monkeypatch.setenv("HERMES_LOCAL_BACKEND_RECOVERY_COOLDOWN", "60")

    calls = {"count": 0}

    def fake_run(*_args, **_kwargs):
        calls["count"] += 1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    agent = SimpleNamespace(provider="local")
    assert maybe_recover_local_backend(
        agent,
        reason="first",
        model="local-q4",
        base_url="http://127.0.0.1:9090/v1",
    )
    assert not maybe_recover_local_backend(
        agent,
        reason="second",
        model="local-q4",
        base_url="http://127.0.0.1:9090/v1",
    )
    assert calls["count"] == 1
