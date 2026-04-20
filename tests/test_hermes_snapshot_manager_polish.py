from __future__ import annotations

from fastapi.testclient import TestClient

from hermes_snapshot_manager.core.config import load_settings
from hermes_snapshot_manager.core.paths import build_paths
from hermes_snapshot_manager.main import app
from hermes_snapshot_manager.services.scheduler_service import describe_cron


def test_settings_page_has_schedule_builder_and_saved_banner(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    snapshot_home = tmp_path / ".hermes-snapshot-manager"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SNAPSHOT_HOME", str(snapshot_home))

    client = TestClient(app)
    response = client.get("/settings?saved=1")

    assert response.status_code == 200
    assert "Make cron easier to configure" in response.text
    assert "Weekly routine" in response.text
    assert "Settings saved." in response.text
    assert "Add pattern" in response.text
    assert "Reset excludes" in response.text


def test_settings_post_updates_retention_and_excludes(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    snapshot_home = tmp_path / ".hermes-snapshot-manager"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SNAPSHOT_HOME", str(snapshot_home))

    client = TestClient(app)
    response = client.post(
        "/settings",
        data={
            "schedule_enabled": "on",
            "schedule_cron": "0 8 * * 1",
            "retention_hourly": "12",
            "retention_daily": "14",
            "retention_weekly": "8",
            "exclude_patterns": "cache/**\nsessions/**\n",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/settings?saved=1"

    settings = load_settings(build_paths())
    assert settings.schedule_enabled is True
    assert settings.retention_hourly == 12
    assert settings.retention_daily == 14
    assert settings.retention_weekly == 8
    assert "cache/**" in settings.exclude_patterns
    assert "sessions/**" in settings.exclude_patterns


def test_describe_cron_humanizes_common_patterns():
    assert describe_cron("0 */6 * * *") == "Every 6 hours"
    assert describe_cron("0 3 * * *") == "Daily at 03:00 UTC"
    assert describe_cron("0 8 * * 1") == "Monday at 08:00 UTC"


def test_invalid_cron_rejects_save(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    snapshot_home = tmp_path / ".hermes-snapshot-manager"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SNAPSHOT_HOME", str(snapshot_home))

    client = TestClient(app)
    response = client.post(
        "/settings",
        data={
            "schedule_enabled": "on",
            "schedule_cron": "not a cron",
            "retention_hourly": "12",
            "retention_daily": "14",
            "retention_weekly": "8",
            "exclude_patterns": "cache/**\n",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")


def test_snapshots_page_has_filters_and_pagination(tmp_path, monkeypatch):
    from hermes_snapshot_manager.core.config import SnapshotManagerSettings
    from hermes_snapshot_manager.services.snapshot_service import SnapshotService
    from hermes_snapshot_manager.main import snapshot_service as main_snapshot_service

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    snapshot_home = tmp_path / ".hermes-snapshot-manager"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SNAPSHOT_HOME", str(snapshot_home))

    settings = SnapshotManagerSettings(source_root=str(hermes_home), snapshot_root=str(snapshot_home / "snapshots"))
    service = SnapshotService(build_paths(), settings=settings)
    (hermes_home / "config.yaml").write_text("model: one\n", encoding="utf-8")
    service.create_snapshot(label="ui")
    monkeypatch.setattr("hermes_snapshot_manager.main.snapshot_service", service)

    client = TestClient(app)
    response = client.get("/snapshots")

    assert response.status_code == 200
    assert "All statuses" in response.text
    assert "Page 1" in response.text
    assert "No snapshots match the current filters." in response.text
