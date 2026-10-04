import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_cron_store(tmp_path, monkeypatch):
    from cron.jobs import use_cron_store

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with use_cron_store(tmp_path):
        yield


def future_job(**extra):
    return {"id": "health-test", "name": "Daily", "deliver": "local",
            "next_run_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), **extra}


def test_success_timestamp_survives_later_failure():
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="check", schedule="every 1h")
    mark_job_run(job["id"], True, delivery_error="not_in_channel")
    successful = get_job(job["id"])
    mark_job_run(job["id"], False, error="expired")
    failed = get_job(job["id"])
    assert failed is not None and successful is not None
    assert failed["last_success_at"] == successful["last_run_at"]


@pytest.mark.parametrize("config, overrides, expected", [
    ({"model": "global"}, {}, "global"),
    ({"model": {"model": "alias"}}, {}, "alias"),
    ({"model": "global", "cron": {"model": "fleet"}}, {}, "fleet"),
    ({"cron": {"model": "fleet"}}, {"model": "pinned"}, "pinned"),
])
def test_health_model_matches_scheduler(monkeypatch, tmp_path, config, overrides, expected):
    from cron.health import inspect_job
    from cron.scheduler import _load_cron_job_config
    import yaml

    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    job = future_job(**overrides)
    report = inspect_job(job, config=config)
    resolved = _load_cron_job_config(job, job["id"], job["name"])
    assert report["model"] == resolved.model == expected


def test_health_reports_destination_and_separate_failures(monkeypatch):
    from cron.health import inspect_job

    monkeypatch.setenv("SLACK_HOME_CHANNEL", "C-test")
    report = inspect_job(future_job(deliver="slack", last_status="error", last_error="provider expired",
                                   last_delivery_error="not_in_channel", last_success_at="yesterday"),
                         config={"model": {"provider": "openai-codex", "model": "chosen"}})
    assert report["delivery"][0]["chat_id"] == "C-test"
    assert report["last_success_at"] == "yesterday"
    assert report["provider_check"] == "not_checked"
    messages = [item["message"] for item in report["issues"]]
    assert any("last run failed" in message for message in messages)
    assert any("last delivery failed" in message for message in messages)


def test_health_is_offline_by_default_and_redacts(monkeypatch):
    from cron.health import inspect_job

    def forbidden(**kwargs):
        pytest.fail("Offline health must not resolve credentials")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", forbidden)
    report = inspect_job(future_job(last_status="error", last_error="https://user:password@example.test?token=private"), config={})
    rendered = json.dumps(report)
    assert "password" not in rendered and "private" not in rendered


def test_health_checks_selected_credentials_only_on_request(monkeypatch):
    from cron.health import inspect_job

    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return {"provider": "custom"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    report = inspect_job(future_job(), config={"cron": {"model": "fleet", "model_provider": "custom"}}, check_provider=True)
    assert calls == [{"requested": "custom", "target_model": "fleet"}]
    assert report["provider_check"] == "credentials_resolved"


def test_health_reports_credential_failure(monkeypatch):
    from cron.health import inspect_job
    from hermes_cli.auth import AuthError

    def expired(**kwargs):
        raise AuthError("Expired login", code="refresh_token_reused", provider="openai-codex", relogin_required=True)

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", expired)
    report = inspect_job(future_job(), config={}, check_provider=True)
    assert report["provider_check"] == "failed"
    assert any(item["code"] == "provider_unavailable" for item in report["issues"])


def test_health_preserves_upstream_unverified_delivery_and_grace():
    from cron.health import inspect_job

    job = future_job(last_status="delivery_queued", last_delivery_unverified=[{"platform": "slack", "chat_id": "C-test"}],
                     next_run_at=(datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat())
    report = inspect_job(job, config={})
    messages = [item["message"] for item in report["issues"]]
    assert any("unverified" in message for message in messages)
    assert not any("overdue" in message or "last run failed" in message for message in messages)


def test_health_detects_missing_destination(monkeypatch):
    from cron.health import inspect_job

    monkeypatch.delenv("SLACK_HOME_CHANNEL", raising=False)
    report = inspect_job(future_job(deliver="slack"), config={})
    assert any(item["code"] == "missing_delivery_target" for item in report["issues"])


def test_health_uses_active_profile_config(tmp_path):
    from cron.health import inspect_jobs
    from cron.jobs import create_job

    (tmp_path / "config.yaml").write_text("model:\n  provider: custom\n  default: profile-model\n", encoding="utf-8")
    create_job(prompt="check", schedule="every 1h", deliver="local")
    reports = inspect_jobs()
    assert reports[0]["model"] == "profile-model"
    assert reports[0]["provider"] == "custom"
