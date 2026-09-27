"""A cron job whose primary provider failed credential resolution and ran on a fallback entry must say
so in the delivered report (#74349): the cron agent has no status rail, so the pre-agent switch would
otherwise live only in the scheduler log. Drives the real ``run_job`` path with AIAgent and
``resolve_runtime_provider`` mocked."""
from unittest.mock import MagicMock, patch

from cron.scheduler import run_job
from hermes_cli.auth import AuthError

_JOB = {"id": "fb-test", "name": "fb test", "prompt": "hello", "model": None, "provider": None,
        "provider_snapshot": None, "base_url": None}


def _run(tmp_path, *, response: str, primary_fails: bool):
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: gpt-5.6-sol\n  provider: openai-codex\n"
        "fallback_providers:\n  - provider: anthropic\n    model: claude-sonnet-5\n    api_key: fb-key\n")

    def _resolve(**kwargs):
        # An unpinned job resolves the primary with requested=None (persisted config); only the
        # fallback entry names its provider explicitly.
        if primary_fails and kwargs.get("requested") != "anthropic":
            raise AuthError("expired")
        return {"api_key": "k", "base_url": "https://example.invalid/v1",
                "provider": kwargs.get("requested") or "openai-codex", "api_mode": "chat_completions"}

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=MagicMock()), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve), \
         patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value.run_conversation.return_value = {"final_response": response}
        success, _output, final, error = run_job(dict(_JOB))
    return success, final, error, agent_cls.call_args.kwargs


def test_fallback_run_prepends_the_switch_notice_to_the_delivered_report(tmp_path):
    success, final, error, agent_kwargs = _run(tmp_path, response="Morning brief: all green.", primary_fails=True)
    assert (success, error) == (True, None)
    assert agent_kwargs["provider"] == "anthropic" and agent_kwargs["model"] == "claude-sonnet-5"
    assert "_fallback_notice" not in agent_kwargs
    first, _, rest = final.partition("\n\n")
    assert "openai-codex/gpt-5.6-sol" in first and "anthropic/claude-sonnet-5" in first
    assert rest == "Morning brief: all green."


def test_primary_run_and_silent_fallback_run_are_untouched(tmp_path):
    _s, final, _e, _k = _run(tmp_path, response="Morning brief: all green.", primary_fails=False)
    assert final == "Morning brief: all green."
    # [SILENT] keeps its whole-response contract so the delivery stays suppressed.
    _s, final, _e, _k = _run(tmp_path, response="[SILENT]", primary_fails=True)
    assert final == "[SILENT]"
