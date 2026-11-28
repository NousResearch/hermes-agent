"""``run_job`` must hand the HERMES_CRON_AUTO_DELIVER_* ContextVars back exactly as it found them.

``_CronRunScope`` pins them to ``""`` for the run and ``_reload_dotenv_and_publish_delivery_target``
publishes the job's target; teardown used to ``set("")`` instead of resetting, so a caller that
runs ``run_job`` in its own context (an embedding app, an eval harness, a test) kept an
"explicitly cleared" value afterwards. ``get_session_env`` treats ``""`` as authoritative, so the
caller's later ``send_message`` / ``cronjob`` defaults stopped falling back to ``os.environ``.
"""

import contextvars
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.session_context import _UNSET, _VAR_MAP, get_session_env

_NAMES = (
    "HERMES_CRON_AUTO_DELIVER_PLATFORM",
    "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
    "HERMES_CRON_AUTO_DELIVER_THREAD_ID",
)
_RUNTIME = {
    "api_key": "***",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}


class _FakeAgent:
    seen: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def run_conversation(self, *args, **kwargs):
        _FakeAgent.seen = {name: get_session_env(name) for name in _NAMES}
        return {"final_response": "ok"}


def _run_job_in_this_context(tmp_path):
    """Drive the real ``run_job`` body synchronously, in the caller's context (no ``to_thread``
    copy), the way a direct in-process caller would."""
    from cron.scheduler import run_job

    job = {"id": "scope-job", "name": "scope", "prompt": "hello", "deliver": "telegram"}
    execution_id = "exec-scope"
    fake_db = MagicMock()
    fake_db.db_path = str(tmp_path / "state.db")
    owner = (SimpleNamespace(db=fake_db), "canonical-cron-session", job["id"], execution_id)
    with patch("gateway.session_cron.current_execution", return_value=owner), \
         patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._preflight_job_config", return_value=None), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_RUNTIME), \
         patch("run_agent.AIAgent", _FakeAgent):
        return run_job(job, execution_id=execution_id)


@pytest.fixture
def home_channel(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TELEGRAM_HOME_CHANNEL=-2002\n")
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    _FakeAgent.seen = {}
    return tmp_path


def test_run_job_returns_delivery_vars_to_never_set(home_channel, monkeypatch):
    def body():
        success, _output, final_response, error = _run_job_in_this_context(home_channel)
        assert (success, final_response, error) == (True, "ok", None)
        # The job itself saw its published target...
        assert _FakeAgent.seen["HERMES_CRON_AUTO_DELIVER_PLATFORM"] == "telegram"
        assert _FakeAgent.seen["HERMES_CRON_AUTO_DELIVER_CHAT_ID"] == "-2002"
        # ...and the caller gets the never-set state back, so the env fallback works again.
        assert [_VAR_MAP[name].get() for name in _NAMES] == [_UNSET, _UNSET, _UNSET]
        monkeypatch.setenv("HERMES_CRON_AUTO_DELIVER_PLATFORM", "discord")
        assert get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM") == "discord"

    contextvars.Context().run(body)


def test_run_job_restores_an_enclosing_scopes_delivery_vars(home_channel):
    def body():
        for name, value in zip(_NAMES, ("slack", "C123", "1700000000.1")):
            _VAR_MAP[name].set(value)
        success, _output, _final, _error = _run_job_in_this_context(home_channel)
        assert success is True
        assert _FakeAgent.seen["HERMES_CRON_AUTO_DELIVER_PLATFORM"] == "telegram"
        assert [_VAR_MAP[name].get() for name in _NAMES] == ["slack", "C123", "1700000000.1"]

    contextvars.Context().run(body)
