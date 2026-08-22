"""Regression coverage for cron custom routing and long-running commands."""

import pytest

from cron import scheduler
from cron.jobs import create_job, get_job, update_job
from hermes_cli.model_switch import DirectAlias
from tools import terminal_tool


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def test_cron_route_dereferences_direct_model_alias(monkeypatch):
    from hermes_cli import model_switch
    from hermes_cli import runtime_provider

    direct_aliases = {
        "ornith": DirectAlias(
            model="Ornith-1.5-35B-Q6_K.gguf",
            provider="custom:ornith",
            base_url="http://127.0.0.1:8085/v1",
        )
    }
    monkeypatch.setattr(model_switch, "_load_direct_aliases", lambda: direct_aliases)
    monkeypatch.setattr(runtime_provider, "has_named_custom_provider", lambda _: False)

    assert scheduler._resolve_cron_inference_route(
        model="ornith", provider=None, base_url=None
    ) == (
        "Ornith-1.5-35B-Q6_K.gguf",
        "custom",
        "http://127.0.0.1:8085/v1",
    )


def test_cron_route_keeps_configured_named_custom_provider(monkeypatch):
    from hermes_cli import model_switch
    from hermes_cli import runtime_provider

    direct_aliases = {
        "ornith": DirectAlias(
            model="aliased-model",
            provider="custom:ornith",
            base_url="https://ornith.example/v1",
        )
    }
    monkeypatch.setattr(model_switch, "_load_direct_aliases", lambda: direct_aliases)
    monkeypatch.setattr(runtime_provider, "has_named_custom_provider", lambda _: True)

    assert scheduler._resolve_cron_inference_route(
        model="ornith", provider=None, base_url=None
    ) == (
        "aliased-model",
        "custom:ornith",
        "https://ornith.example/v1",
    )


def test_cron_route_normalizes_unconfigured_custom_provider_without_model_alias(
    monkeypatch,
):
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, "has_named_custom_provider", lambda _: False)

    assert scheduler._resolve_cron_inference_route(
        model="canonical-model",
        provider="custom:ornith",
        base_url="http://127.0.0.1:8085/v1",
        config={},
    ) == (
        "canonical-model",
        "custom",
        "http://127.0.0.1:8085/v1",
    )


def test_cron_route_keeps_explicit_base_url_for_alias_only_custom_provider(monkeypatch):
    from hermes_cli import model_switch
    from hermes_cli import runtime_provider

    direct_aliases = {
        "ornith": DirectAlias(
            model="aliased-model",
            provider="custom:alias",
            base_url="http://alias.invalid/v1",
        )
    }
    monkeypatch.setattr(model_switch, "_load_direct_aliases", lambda: direct_aliases)
    monkeypatch.setattr(runtime_provider, "has_named_custom_provider", lambda _: False)

    assert scheduler._resolve_cron_inference_route(
        model="ornith",
        provider="custom:explicit",
        base_url="http://127.0.0.1:9999/v1/",
    ) == (
        "aliased-model",
        "custom",
        "http://127.0.0.1:9999/v1",
    )


def test_cron_route_reloads_aliases_for_each_fire(monkeypatch):
    from hermes_cli import model_switch

    aliases = iter(
        [
            {"ornith": DirectAlias("model-v1", "custom", "http://one/v1")},
            {"ornith": DirectAlias("model-v2", "custom", "http://two/v1")},
        ]
    )
    monkeypatch.setattr(model_switch, "_load_direct_aliases", lambda: next(aliases))

    first = scheduler._resolve_cron_inference_route(
        model="ornith", provider=None, base_url=None
    )
    second = scheduler._resolve_cron_inference_route(
        model="ornith", provider=None, base_url=None
    )

    assert first == ("model-v1", "custom", "http://one/v1")
    assert second == ("model-v2", "custom", "http://two/v1")


def test_alias_owned_provider_does_not_trigger_drift_guard():
    job = {
        "model": "ornith",
        "provider": None,
        "provider_snapshot": "openrouter",
        "model_snapshot": None,
    }

    assert scheduler._cron_model_drift_axes_for_route(
        job,
        current_provider="custom",
        current_model="Ornith-1.5-35B-Q6_K.gguf",
        config={},
        alias_supplied_provider=True,
    ) == []


def test_job_terminal_timeout_round_trips_and_validates(tmp_cron_dir):
    job = create_job(
        prompt="Build project",
        schedule="every 1h",
        terminal_timeout=900,
    )
    assert get_job(job["id"])["terminal_timeout"] == 900

    updated = update_job(job["id"], {"terminal_timeout": 1200})
    assert updated["terminal_timeout"] == 1200

    with pytest.raises(ValueError, match="positive"):
        update_job(job["id"], {"terminal_timeout": 0})


def test_cron_terminal_timeout_prefers_job_then_fleet_then_terminal_default():
    assert scheduler._resolve_cron_terminal_timeout(
        {"terminal_timeout": 900},
        {"cron": {"terminal_timeout": 600}, "terminal": {"timeout": 180}},
    ) == 900
    assert scheduler._resolve_cron_terminal_timeout(
        {}, {"cron": {"terminal_timeout": 600}, "terminal": {"timeout": 180}}
    ) == 600
    assert scheduler._resolve_cron_terminal_timeout(
        {}, {"terminal": {"timeout": 180}}
    ) == 180


def test_terminal_task_override_controls_default_timeout():
    assert terminal_tool._resolve_command_timeout(
        {"timeout": 30}, {"timeout": 900}
    ) == 900
    assert terminal_tool._resolve_command_timeout(
        {"timeout": 30}, {}
    ) == 30
