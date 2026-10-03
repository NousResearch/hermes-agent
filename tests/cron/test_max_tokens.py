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
