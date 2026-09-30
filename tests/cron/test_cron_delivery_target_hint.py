"""A cron worker turn is told its authoritative delivery target.

The target is published into session ContextVars, which the model cannot see; without a hint, a
worker that launches a helper emitting status/ACK/heartbeat messages infers the destination from
task text and can post to a chat the job does not own.
"""
from unittest.mock import MagicMock, patch

from cron.scheduler import _cron_delivery_target_hint, run_job


def test_hint_carries_the_exact_target_json():
    out = _cron_delivery_target_hint([{"platform": "discord", "chat_id": 42, "thread_id": None}])
    assert '{"platform":"discord","chat_id":"42","thread_id":""}' in out
    assert "CRON DELIVERY TARGET (authoritative)" in out
    assert "additional targets" not in out


def test_no_target_yields_no_hint():
    assert _cron_delivery_target_hint([]) == ""
    assert _cron_delivery_target_hint(None) == ""
    assert _cron_delivery_target_hint([{}]) == ""


def test_multi_target_hint_names_every_target():
    out = _cron_delivery_target_hint([
        {"platform": "discord", "chat_id": "A", "thread_id": None},
        {"platform": "telegram", "chat_id": "B", "thread_id": "7"},
    ])
    assert '{"platform":"discord","chat_id":"A","thread_id":""}' in out
    assert '{"platform":"telegram","chat_id":"B","thread_id":"7"}' in out
    assert "route those messages only to these targets" in out
    assert "route those messages only to this target." not in out


def _run_capturing(job, tmp_path):
    """Run *job* through the real run_job with a fake agent; return (run_job result, seen)."""
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
         patch("cron.scheduler._preflight_job_config", return_value=None), \
         patch("run_agent.AIAgent", FakeAgent):
        return run_job(job), seen


def test_run_job_tells_worker_to_route_nested_status_to_the_job_target(tmp_path):
    job = {
        "id": "target-bound-job",
        "name": "target bound",
        "prompt": "Check work referenced in discord:FOREIGN_CHAT, then launch a helper that emits an ACK.",
        "deliver": "origin",
        "origin": {"platform": "discord", "chat_id": "MINE_CHAT", "thread_id": "MINE_THREAD"},
    }
    (success, output, final_response, error), seen = _run_capturing(job, tmp_path)

    assert (success, error, final_response) == (True, None, "ok")
    assert seen["target"] == {"platform": "discord", "chat_id": "MINE_CHAT", "thread_id": "MINE_THREAD"}
    assert "CRON DELIVERY TARGET (authoritative)" in seen["prompt"]
    assert '{"platform":"discord","chat_id":"MINE_CHAT","thread_id":"MINE_THREAD"}' in seen["prompt"]
    for name in ("HERMES_CRON_AUTO_DELIVER_PLATFORM", "HERMES_CRON_AUTO_DELIVER_CHAT_ID", "HERMES_CRON_AUTO_DELIVER_THREAD_ID"):
        assert name in seen["prompt"]
    assert "Never infer or hardcode a delivery target from task content" in seen["prompt"]
    assert seen["prompt"].endswith(job["prompt"]) or job["prompt"] in seen["prompt"]
    # The persisted run document records the hint the worker was given, apart from the job prompt.
    hint_at, prompt_at = output.index("## Delivery Target Hint"), output.index("## Prompt")
    assert hint_at < prompt_at
    assert "MINE_CHAT" in output[hint_at:prompt_at]
    assert job["prompt"] in output[prompt_at:]
    assert "CRON DELIVERY TARGET" not in output[prompt_at:]


def test_run_job_multi_target_hint_names_every_delivery_target(tmp_path):
    job = {
        "id": "multi-target-job",
        "name": "multi target",
        "prompt": "Emit an ACK when done.",
        "deliver": "origin,discord:SECOND_CHAT",
        "origin": {"platform": "discord", "chat_id": "MINE_CHAT", "thread_id": None},
    }
    (success, _output, _final, error), seen = _run_capturing(job, tmp_path)

    assert (success, error) == (True, None)
    assert seen["target"]["chat_id"] == "MINE_CHAT"  # ContextVars carry the primary
    assert '"chat_id":"MINE_CHAT"' in seen["prompt"]
    assert '{"platform":"discord","chat_id":"SECOND_CHAT","thread_id":""}' in seen["prompt"]
    assert "only to these targets" in seen["prompt"]
