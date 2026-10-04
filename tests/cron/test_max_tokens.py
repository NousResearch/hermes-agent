"""Regression coverage for per-job cron output-token caps."""

import inspect
from types import SimpleNamespace

import pytest

from cron.jobs import _normalize_max_tokens, create_job, update_job
from cron.scheduler import _resolve_cron_max_tokens


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def test_max_tokens_round_trips_through_create_and_update(tmp_cron_dir):
    job = create_job(prompt="bounded task", schedule="every 1h", max_tokens=4096)
    assert job["max_tokens"] == 4096

    updated = update_job(job["id"], {"max_tokens": 2048})
    assert updated["max_tokens"] == 2048


def test_max_tokens_rejects_non_positive_values(tmp_cron_dir):
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        create_job(prompt="invalid", schedule="every 1h", max_tokens=0)

    job = create_job(prompt="valid", schedule="every 1h")
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        update_job(job["id"], {"max_tokens": -1})


def test_max_tokens_normalization_matches_cli_integer_input(tmp_cron_dir):
    assert _normalize_max_tokens(32768) == 32768
    assert _normalize_max_tokens(" 32768 ") == 32768
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        _normalize_max_tokens(32768.0)
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        _normalize_max_tokens("32768.0")


def test_cron_cap_preserves_provider_route_default(monkeypatch):
    profile = SimpleNamespace(default_max_tokens=65536, get_max_tokens=lambda model: 65536)
    monkeypatch.setattr("providers.get_provider_profile", lambda provider: profile)

    assert _resolve_cron_max_tokens(None, provider="qwen-oauth", model="qwen3-max") == 65536
    assert _resolve_cron_max_tokens(32768, provider="qwen-oauth", model="qwen3-max") == 32768
    assert _resolve_cron_max_tokens(16384, provider="qwen-oauth", model="qwen3-max") == 16384
    assert _resolve_cron_max_tokens(131072, provider="qwen-oauth", model="qwen3-max") == 65536


def test_cron_cap_leaves_unset_transport_without_provider_default(monkeypatch):
    profile = SimpleNamespace(get_max_tokens=lambda model: None)
    monkeypatch.setattr("providers.get_provider_profile", lambda provider: profile)

    assert _resolve_cron_max_tokens(None, provider="openrouter", model="qwen3-max") is None
    assert _resolve_cron_max_tokens(131072, provider="openrouter", model="qwen3-max") == 131072


def test_max_tokens_is_not_model_settable():
    from tools.cronjob_tools import CRONJOB_SCHEMA, cronjob

    assert "max_tokens" not in CRONJOB_SCHEMA["parameters"]["properties"]
    assert "max_tokens" not in inspect.signature(cronjob).parameters


@pytest.mark.parametrize("job_cap", [None, ""])
def test_cleared_job_cap_inherits_global_default(tmp_cron_dir, monkeypatch, job_cap):
    from cron import scheduler

    job = create_job(prompt="inherit global budget", schedule="every 1h", max_tokens=4096)
    job = update_job(job["id"], {"max_tokens": job_cap})
    assert job is not None
    assert job["max_tokens"] is None
    jc = SimpleNamespace(model="qwen3-max", cfg={"cron": {"max_tokens_default": 8000}})
    monkeypatch.setattr(scheduler, "_load_prefill_messages", lambda *args: None)
    monkeypatch.setattr(scheduler, "_guard_job_credential_exfil", lambda *args: None)
    monkeypatch.setattr(scheduler, "_preflight_or_block", lambda *args: None)
    monkeypatch.setattr(scheduler, "_resolve_job_runtime", lambda *args: ({"provider": "qwen-oauth"}, jc.model))
    monkeypatch.setattr(scheduler, "_resolve_job_reasoning_config", lambda *args: None)
    monkeypatch.setattr(scheduler, "_job_fallback_chain", lambda *args: None)
    monkeypatch.setattr(scheduler, "_load_credential_pool", lambda *args: None)
    monkeypatch.setattr(scheduler, "_init_cron_mcp_tools", lambda *args: None)
    monkeypatch.setattr(scheduler, "_cron_preflight_enabled", lambda *args: False)
    monkeypatch.setattr("providers.get_provider_profile", lambda provider: SimpleNamespace(get_max_tokens=lambda model: 65536))

    setup = scheduler._resolve_cron_agent_setup(job, job["id"], job["name"], jc)
    assert setup.max_tokens == 8000


@pytest.mark.parametrize("requested", [0, -5, True, 1.5, "1.5", "garbage"])
def test_fire_time_rejects_invalid_hand_written_caps(monkeypatch, requested):
    monkeypatch.setattr("providers.get_provider_profile", lambda provider: SimpleNamespace(get_max_tokens=lambda model: None))
    assert _resolve_cron_max_tokens(requested, provider="openrouter", model="model") is None


def test_model_handler_does_not_forward_hidden_cap(monkeypatch):
    from tools import cronjob_tools

    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return "{}"
    monkeypatch.setattr(cronjob_tools, "cronjob", capture)
    cronjob_tools._cronjob_handler({"action": "list", "max_tokens": 131072})
    assert "max_tokens" not in captured
