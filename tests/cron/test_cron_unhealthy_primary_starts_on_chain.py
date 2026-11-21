"""An unpinned cron job whose primary provider this process already holds unhealthy (the auxiliary
ladder's payment/credit mark: an exhausted balance) starts on its fallback chain, the way a side job
skips that provider, instead of paying a refused first call every run. A pinned job keeps its pin, and
a healthy primary is untouched (Nov 13 2026: a profile's hourly jobs on an empty Anthropic balance paid
95 refused first calls in 7 days before each fallback answered). Drives the real ``run_job`` path with
AIAgent and ``resolve_runtime_provider`` mocked."""
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client
from tests.cron.test_scheduler import _run_owned_job

_JOB = {"id": "learn", "name": "memory learn", "prompt": "hello", "model": None, "provider": None,
        "provider_snapshot": None, "base_url": None}


def _mark(tmp_path, ttl=3600):
    """The mark as the job's own profile holds it (health is scoped to the profile home)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(tmp_path)
    try:
        auxiliary_client._mark_provider_unhealthy("anthropic", ttl, reason="credits empty")
    finally:
        reset_hermes_home_override(token)


@pytest.fixture(autouse=True)
def _clean_health():
    auxiliary_client._reset_aux_unhealthy_cache()
    yield
    auxiliary_client._reset_aux_unhealthy_cache()


def _run(tmp_path, job=None, *, codex_fails=False):
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: claude-opus-5-5\n  provider: anthropic\n"
        "fallback_providers:\n  - provider: openai-codex\n    model: gpt-6-astra\n"
        "  - provider: xai-oauth\n    model: grok-4.7\n")
    asked = []
    fake_db = MagicMock()

    def _resolve(**kwargs):
        provider = kwargs.get("requested") or "anthropic"
        asked.append(provider)
        if codex_fails and provider == "openai-codex":
            raise RuntimeError("no codex login")
        return {"api_key": "k", "base_url": "https://example.invalid/v1", "provider": provider,
                "api_mode": "chat_completions"}

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve), \
         patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value.run_conversation.return_value = {"final_response": "[SILENT]"}
        success, _output, _final, error = _run_owned_job(dict(job or _JOB), tmp_path, fake_db)
    return success, error, agent_cls.call_args.kwargs, asked


def test_a_healthy_primary_runs_first(tmp_path):
    success, error, agent, asked = _run(tmp_path)
    assert (success, error) == (True, None)
    assert (agent["provider"], agent["model"]) == ("anthropic", "claude-opus-5-5") and asked == ["anthropic"]


def test_a_primary_marked_out_of_credits_is_not_tried_first(tmp_path):
    _mark(tmp_path)
    success, error, agent, asked = _run(tmp_path)
    assert (success, error) == (True, None)
    assert (agent["provider"], agent["model"]) == ("openai-codex", "gpt-6-astra")
    assert asked == ["openai-codex"]  # the marked primary is never resolved, so never called


def test_the_next_entry_carries_it_when_the_first_one_fails_to_resolve(tmp_path):
    _mark(tmp_path)
    success, _error, agent, asked = _run(tmp_path, codex_fails=True)
    assert success and (agent["provider"], agent["model"]) == ("xai-oauth", "grok-4.7")
    assert asked == ["openai-codex", "xai-oauth"]


def test_another_profiles_mark_does_not_move_this_one(tmp_path):
    auxiliary_client._mark_provider_unhealthy("anthropic", 3600, reason="credits empty")  # the launch home's
    _success, _error, agent, _asked = _run(tmp_path)
    assert agent["provider"] == "anthropic"


def test_a_pinned_job_keeps_its_pin(tmp_path):
    _mark(tmp_path)
    pinned = {**_JOB, "provider": "anthropic", "model": "claude-opus-5-5"}
    _success, _error, agent, asked = _run(tmp_path, pinned)
    assert agent["provider"] == "anthropic" and set(asked) == {"anthropic"}


def test_an_expired_mark_puts_the_primary_first_again(tmp_path, monkeypatch):
    _mark(tmp_path, 60)
    real = auxiliary_client.time.time()
    monkeypatch.setattr(auxiliary_client.time, "time", lambda: real + 120)
    _success, _error, agent, asked = _run(tmp_path)
    assert agent["provider"] == "anthropic" and set(asked) == {"anthropic"}
