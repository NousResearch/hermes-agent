"""Cron agents carry the job creator's identity from ``job["origin"]`` as a hook-only principal
(``agent._principal_identity``) so tool-call hooks can attribute a scheduled fire to the user who
scheduled it. It is NOT the agent's own identity (``agent._user_id`` etc. stay unset, so memory
providers and the cron session row are untouched) and the HERMES_SESSION_* ContextVars stay blank
on purpose (``_CronRunScope``): the origin user is the principal, not a live sender; ``platform``
stays ``"cron"``."""

from unittest.mock import MagicMock, patch

from agent.inline_tool_executors import principal_hook_fields
from cron.scheduler import _CronAgentSetup, _construct_cron_agent, run_job
from gateway.session_context import get_session_env

_IDENTITY_KEYS = ("user_id", "user_name", "chat_id", "chat_type", "thread_id")


class _RecordingAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.platform = kwargs.get("platform")


def _construct(job):
    setup = _CronAgentSetup(
        model="test-model",
        runtime={"api_key": "k", "base_url": "https://example.invalid/v1", "provider": "custom"},
    )
    return _construct_cron_agent(
        _RecordingAgent, job, {}, setup, workdir=None, session_id="cron-sess", session_db=None)


def test_origin_identity_becomes_hook_principal_not_agent_identity():
    job = {
        "id": "j-origin", "name": "origin", "prompt": "hi",
        "origin": {
            "platform": "discord", "chat_id": "c1", "chat_type": "channel",
            "user_id": "222", "user_name": "wife", "thread_id": "t9",
        },
    }
    agent = _construct(job)
    assert agent._principal_identity == {
        "user_id": "222", "user_name": "wife", "chat_id": "c1", "chat_type": "channel", "thread_id": "t9",
    }
    # Never the AIAgent identity kwargs: those re-scope memory providers and the session row.
    for key in _IDENTITY_KEYS:
        assert key not in agent.kwargs, key
    # The origin's platform is where the job came FROM; the agent still runs as cron.
    assert agent.kwargs["platform"] == "cron"
    fields = principal_hook_fields(agent)
    assert fields["user_id"] == "222" and fields["user_name"] == "wife"
    assert fields["chat_id"] == "c1" and fields["chat_type"] == "channel" and fields["thread_id"] == "t9"
    assert fields["platform"] == "cron"


def test_origin_less_job_has_no_principal():
    agent = _construct({"id": "j-local", "name": "local", "prompt": "hi"})
    assert agent.kwargs["platform"] == "cron"
    assert not hasattr(agent, "_principal_identity")
    for key in _IDENTITY_KEYS:
        assert key not in agent.kwargs, key


def test_run_job_carries_origin_user_as_hook_principal_not_session_var(tmp_path):
    """E2E through run_job: identity reaches the hook principal while the per-run scope keeps
    HERMES_SESSION_USER_ID blank — a cron origin must never look like a live sender to the
    terminal/messaging tools."""
    job = {
        "id": "j-e2e", "name": "e2e", "prompt": "hello",
        "origin": {"platform": "discord", "chat_id": "c1", "chat_type": "channel", "user_id": "222"},
    }
    seen = {}

    def _run_conversation(*_a, **_kw):
        seen["session_user_id"] = get_session_env("HERMES_SESSION_USER_ID")
        return {"final_response": "ok"}

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("dotenv.load_dotenv"), \
         patch("hermes_state_registry.acquire", return_value=MagicMock()), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key", "base_url": "https://example.invalid/v1",
                 "provider": "custom", "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.side_effect = _run_conversation
        mock_agent_cls.return_value = mock_agent

        success, _output, final_response, error = run_job(job)

    assert (success, final_response, error) == (True, "ok", None)
    assert mock_agent._principal_identity == {"user_id": "222", "chat_id": "c1", "chat_type": "channel"}
    assert "user_id" not in mock_agent_cls.call_args.kwargs
    assert mock_agent_cls.call_args.kwargs["platform"] == "cron"
    assert seen["session_user_id"] == ""
