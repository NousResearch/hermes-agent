"""Cron output logging to Honcho."""

from __future__ import annotations


class _FakeHonchoConfig:
    enabled = True
    save_messages = True
    message_max_chars = 25000
    ai_peer = "hermes"
    write_frequency = "session"
    context_tokens = None
    dialectic_reasoning_level = "low"
    dialectic_dynamic = True
    dialectic_max_chars = 600
    observation_mode = "directional"
    user_observe_me = True
    user_observe_others = True
    ai_observe_me = True
    ai_observe_others = True
    dialectic_max_input_chars = 10000
    peer_name = None
    pin_peer_name = False
    user_peer_aliases = {}
    runtime_peer_prefix = ""


class _FakeSession:
    def __init__(self, key: str):
        self.key = key
        self.messages = []

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})


class _FakeManager:
    instances = []

    def __init__(self, config=None, runtime_user_peer_name=None, **kwargs):
        self.config = config
        self.runtime_user_peer_name = runtime_user_peer_name
        self.session = None
        self.saved = []
        self.flushed = False
        self.shutdown_called = False
        self.__class__.instances.append(self)

    def get_or_create(self, key: str):
        self.session = _FakeSession(key)
        return self.session

    def save(self, session):
        self.saved.append(session)

    def flush_all(self):
        self.flushed = True

    def shutdown(self):
        self.shutdown_called = True


def test_log_cron_output_to_honcho_requires_config_opt_in(monkeypatch):
    import cron.scheduler as scheduler

    _FakeManager.instances.clear()
    monkeypatch.setattr(
        scheduler,
        "load_config",
        lambda: {"memory": {"memory_enabled": True, "provider": "honcho"}, "cron": {}},
    )

    assert scheduler._log_cron_output_to_honcho(
        {"id": "job1", "name": "Example"},
        "hello",
        success=True,
        output_file="/tmp/out.md",
    ) is False
    assert _FakeManager.instances == []


def test_log_cron_output_to_honcho_writes_redacted_final_output(monkeypatch):
    import cron.scheduler as scheduler
    import plugins.memory.honcho.client as honcho_client
    import plugins.memory.honcho.session as honcho_session

    _FakeManager.instances.clear()
    monkeypatch.setattr(
        scheduler,
        "load_config",
        lambda: {
            "memory": {"memory_enabled": True, "provider": "honcho"},
            "cron": {"honcho_output_logging": True},
        },
    )
    monkeypatch.setattr(honcho_client.HonchoClientConfig, "from_global_config", classmethod(lambda cls: _FakeHonchoConfig()))
    monkeypatch.setattr(honcho_session, "HonchoSessionManager", _FakeManager)

    ok = scheduler._log_cron_output_to_honcho(
        {"id": "job1", "name": "Example"},
        "finished with api_key=sk-testsecret123456789012345 and token=abc123",
        success=True,
        output_file="/tmp/out.md",
    )

    assert ok is True
    manager = _FakeManager.instances[0]
    assert manager.runtime_user_peer_name == "cron"
    assert manager.session.key == "cron:job1"
    assert manager.flushed is True
    assert manager.shutdown_called is True
    message = manager.session.messages[0]
    assert message["role"] == "assistant"
    assert "<cron_output>" in message["content"]
    assert "Job: Example" in message["content"]
    assert "sk-testsecret" not in message["content"]
    assert "token=abc123" not in message["content"]
    assert "[REDACTED]" in message["content"]


def test_run_one_job_logs_deliver_content_to_honcho(monkeypatch, tmp_path):
    import cron.scheduler as scheduler

    calls = []
    job = {"id": "job1", "name": "Example"}

    monkeypatch.setattr(scheduler, "claim_dispatch", lambda job_id: True)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda j, **kwargs: (True, "full output", "final response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda job_id, output: tmp_path / "out.md")
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        scheduler,
        "_log_cron_output_to_honcho",
        lambda job, content, **kwargs: calls.append((job, content, kwargs)) or True,
    )

    assert scheduler.run_one_job(job) is True
    assert calls == [
        (
            job,
            "final response",
            {"success": True, "output_file": tmp_path / "out.md", "error": None},
        )
    ]
