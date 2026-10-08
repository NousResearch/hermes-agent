"""Unattended cron may use a backup only when automatic activation is enabled."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

import hermes_yaml as yaml
from cron import scheduler
from cron.scheduler_preflight import _preflight_check_provider_key
from hermes_cli.auth import AuthError

_CHAIN = [{"provider": "deepseek", "model": "backup-model", "api_key": "test-only"}]
_JOB = {"id": "manual-cron", "name": "manual cron", "prompt": "hello"}
_ERRORS = [
    pytest.param(AuthError("primary credential unavailable"), id="auth"),
    pytest.param(httpx.ConnectError("temporary failure in name resolution"), id="dns"),
]


@pytest.fixture
def profile(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    token = set_hermes_home_override(tmp_path)
    try:
        with patch("httpx.Client.send", side_effect=RuntimeError("offline test")):
            yield tmp_path
    finally:
        reset_hermes_home_override(token)


def _config(profile, policy, *, job=None):
    cfg = {
        "model": {"default": "primary-model", "provider": "custom"},
        "cron": {"preflight": False},
        "fallback_providers": list(_CHAIN),
    }
    if policy != "absent":
        cfg["fallback"] = {"auto_activate": policy}
    (profile / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return scheduler._load_cron_job_config(job or _JOB, _JOB["id"], _JOB["name"])


def _runtime(provider="custom"):
    return {"provider": provider, "api_key": "test-only", "base_url": "https://primary.invalid/v1",
            "api_mode": "chat_completions"}


@pytest.mark.parametrize("policy", [False, "invalid", None])
@pytest.mark.parametrize("primary_error", _ERRORS)
def test_manual_bootstrap_never_resolves_a_backup(profile, policy, primary_error):
    jc = _config(profile, policy)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=primary_error) as resolve:
        with pytest.raises(RuntimeError) as raised:
            scheduler._resolve_job_runtime(_JOB, _JOB["id"], jc)
    assert raised.value.__cause__ is primary_error
    assert str(primary_error) in str(raised.value)
    resolve.assert_called_once_with(requested=None, target_model="primary-model")


@pytest.mark.parametrize("policy", [True, "absent"])
@pytest.mark.parametrize("primary_error", _ERRORS)
def test_automatic_bootstrap_still_resolves_the_chain(profile, policy, primary_error):
    jc = _config(profile, policy)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               side_effect=[primary_error, _runtime("deepseek")]) as resolve:
        runtime, model = scheduler._resolve_job_runtime(_JOB, _JOB["id"], jc)
    assert (runtime["provider"], model) == ("deepseek", "backup-model")
    assert [call.kwargs["requested"] for call in resolve.call_args_list] == [None, "deepseek"]


@pytest.mark.parametrize("policy", [False, "invalid"])
def test_manual_chain_does_not_hide_a_missing_primary_key(profile, policy):
    jc = _config(profile, policy)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               side_effect=AuthError("primary credential unavailable")) as resolve:
        reason = _preflight_check_provider_key(_JOB, jc.cfg)
    assert reason and "primary credential unavailable" in reason
    resolve.assert_called_once()


def test_manual_failure_notice_does_not_claim_backups_were_tried(profile):
    jc = _config(profile, False)
    with patch.object(scheduler, "load_config", return_value=jc.cfg):
        phrase = scheduler._fallback_chain_phrase(_JOB)
    assert "manual" in phrase.lower()
    assert "cannot" in phrase.lower() and "cron" in phrase.lower()
    assert "succeeded" not in phrase


@pytest.mark.parametrize("policy", [False, True, "absent"])
@pytest.mark.parametrize("pin", [{}, {"provider": "custom"}, {"model": "pinned-model"},
                                  {"provider": "custom", "base_url": "https://pinned.invalid/v1"}])
def test_real_cron_agent_keeps_policy_and_pin_at_runtime(profile, policy, pin):
    from agent.error_classifier import FailoverReason
    from run_agent import AIAgent

    job = {**_JOB, **pin}
    jc = _config(profile, policy, job=job)
    with (
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_runtime()),
        patch.object(scheduler, "_init_cron_mcp_tools"),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        setup = scheduler._resolve_cron_agent_setup(job, job["id"], job["name"], jc)
        assert setup.blocked is None
        agent = scheduler._construct_cron_agent(
            AIAgent, job, jc.cfg, setup, workdir=None, session_id="manual-cron-test", session_db=None)
    callback = MagicMock()
    agent.clarify_callback = callback
    fallback_client = MagicMock(base_url="https://backup.invalid/v1", api_key="test-only")
    try:
        assert agent._fallback_auto_activate is (policy is not False)
        assert agent._fallback_selection_interactive is False
        assert agent._fallback_chain == ([] if pin else _CHAIN)
        with (
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(fallback_client, "backup-model")) as resolve_backup,
            patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, _p: m),
        ):
            activated = agent._try_activate_fallback(FailoverReason.rate_limit)
        expected = not pin and policy is not False
        assert activated is expected
        assert resolve_backup.call_count == int(expected)
        callback.assert_not_called()
        if not expected:
            assert agent.provider == "custom"
            assert agent.model == job.get("model", "primary-model")
    finally:
        agent.close()
