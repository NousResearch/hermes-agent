from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.khaw_usage_recorder import record_openrouter_generation


def _agent(**overrides):
    data = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3.7-max",
        "session_id": "sess-123",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_records_openrouter_generation_without_secrets(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"recorded_count": 1}', stderr="")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/khaw")
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    monkeypatch.setenv("HERMES_KHAW_IDENTITY", "leonardo.pavani")

    response = SimpleNamespace(id="gen-abc123", model="qwen/qwen3.7-max")

    assert record_openrouter_generation(_agent(), response) is True
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    joined = " ".join(cmd)
    assert cmd[:4] == ["/usr/local/bin/khaw", "fde", "usage", "record"]
    assert "--generation-id" in cmd
    assert "gen-abc123" in cmd
    assert "--source" in cmd
    assert "hermes" in cmd
    assert "--model" in cmd
    assert "qwen/qwen3.7-max" in cmd
    assert "--session-id" in cmd
    assert "sess-123" in cmd
    assert "--identity" in cmd
    assert "leonardo.pavani" in cmd
    assert "sk-or-secret" not in joined
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False


def test_records_without_identity_and_lets_khaw_default_to_user(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout='{"recorded_count": 1}', stderr="")

    monkeypatch.delenv("KHAW_FDE_IDENTITY", raising=False)
    monkeypatch.delenv("HERMES_FDE_IDENTITY", raising=False)
    monkeypatch.delenv("HERMES_KHAW_IDENTITY", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/khaw")
    monkeypatch.setattr("subprocess.run", fake_run)

    response = SimpleNamespace(id="gen-abc123", model="qwen/qwen3.7-max")

    assert record_openrouter_generation(_agent(), response) is True
    assert "--identity" not in calls[0]


def test_skips_non_openrouter(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/khaw")
    monkeypatch.setattr("subprocess.run", run)

    response = SimpleNamespace(id="gen-abc123", model="x")

    assert record_openrouter_generation(
        _agent(provider="openai", base_url="https://api.openai.com/v1"),
        response,
    ) is False
    run.assert_not_called()


def test_skips_synthetic_stream_ids(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/khaw")
    monkeypatch.setattr("subprocess.run", run)

    response = SimpleNamespace(id="stream-00000000-0000-0000-0000-000000000000", model="x")

    assert record_openrouter_generation(_agent(), response) is False
    run.assert_not_called()


def test_khaw_failure_is_best_effort(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/khaw")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout="", stderr="boom"),
    )

    response = SimpleNamespace(id="gen-abc123", model="x")

    assert record_openrouter_generation(_agent(), response) is False
