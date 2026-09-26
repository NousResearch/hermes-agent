"""A cron worker turn is told its authoritative delivery target.

The target is published into session ContextVars, which the model cannot see; without a hint, a
worker that launches a helper emitting status/ACK/heartbeat messages infers the destination from
task text and can post to a chat the job does not own.
"""
from unittest.mock import MagicMock, patch

from cron.scheduler import _bind_cron_delivery_target_hint, run_job


def test_hint_is_a_prefix_with_the_exact_target_json():
    out = _bind_cron_delivery_target_hint("do work", {"platform": "discord", "chat_id": 42, "thread_id": None})
    assert out.endswith("\n\ndo work")
    assert '{"platform":"discord","chat_id":"42","thread_id":""}' in out
    assert "CRON DELIVERY TARGET (authoritative)" in out


def test_no_target_leaves_the_prompt_untouched():
    assert _bind_cron_delivery_target_hint("do work", None) == "do work"
    assert _bind_cron_delivery_target_hint("do work", {}) == "do work"


def test_run_job_tells_worker_to_route_nested_status_to_the_job_target(tmp_path):
    job = {
        "id": "target-bound-job",
        "name": "target bound",
        "prompt": "Check work referenced in discord:FOREIGN_CHAT, then launch a helper that emits an ACK.",
        "deliver": "origin",
        "origin": {"platform": "discord", "chat_id": "MINE_CHAT", "thread_id": "MINE_THREAD"},
    }
    fake_db = MagicMock()
    fake_db.get_compression_tip.side_effect = lambda session_id: session_id
    seen = {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run_conversation(self, prompt, *args, **kwargs):
            from gateway.session_context import get_session_env

            seen["prompt"] = prompt
            seen["target"] = {
                "platform": get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM"),
                "chat_id": get_session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID"),
                "thread_id": get_session_env("HERMES_CRON_AUTO_DELIVER_THREAD_ID"),
            }
            return {"final_response": "ok"}

        def close(self):
            pass

        def __getattr__(self, name):
            return MagicMock()

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": "openrouter",
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent", FakeAgent):
        success, output, final_response, error = run_job(job)

    assert (success, error, final_response) == (True, None, "ok")
    assert seen["target"] == {"platform": "discord", "chat_id": "MINE_CHAT", "thread_id": "MINE_THREAD"}
    assert "CRON DELIVERY TARGET (authoritative)" in seen["prompt"]
    assert '{"platform":"discord","chat_id":"MINE_CHAT","thread_id":"MINE_THREAD"}' in seen["prompt"]
    for name in ("HERMES_CRON_AUTO_DELIVER_PLATFORM", "HERMES_CRON_AUTO_DELIVER_CHAT_ID", "HERMES_CRON_AUTO_DELIVER_THREAD_ID"):
        assert name in seen["prompt"]
    assert "Never infer or hardcode a delivery target from task content" in seen["prompt"]
    assert seen["prompt"].endswith(job["prompt"]) or job["prompt"] in seen["prompt"]
