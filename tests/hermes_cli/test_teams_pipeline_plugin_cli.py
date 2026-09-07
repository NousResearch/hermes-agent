"""Tests for the teams_pipeline plugin CLI."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest

from plugins.teams_pipeline.cli import register_cli, teams_pipeline_command
from plugins.teams_pipeline.store import TeamsPipelineStore


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _make_args(**kwargs):
    defaults = {
        "teams_pipeline_action": None,
        "store_path": "",
        "status": "",
        "limit": 20,
        "job_id": "",
        "meeting_id": "",
        "join_web_url": "",
        "tenant_id": "",
        "call_record_id": "",
        "resource": "",
        "notification_url": "",
        "change_type": "updated",
        "expiration": "",
        "client_state": "",
        "lifecycle_notification_url": "",
        "latest_supported_tls_version": "v1_2",
        "subscription_id": "",
        "force_refresh": False,
        "renew_within_hours": 24,
        "extend_hours": 24,
        "dry_run": False,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)






def test_show_prints_job_json(capsys, tmp_path):
    store = TeamsPipelineStore(tmp_path / "teams_pipeline_store.json")
    store.upsert_job(
        "job-1",
        {
            "event_id": "evt-1",
            "source_event_type": "updated",
            "dedupe_key": "evt-1",
            "status": "completed",
            "meeting_ref": {"meeting_id": "meeting-1"},
        },
    )

    teams_pipeline_command(
        _make_args(
            teams_pipeline_action="show",
            job_id="job-1",
            store_path=str(tmp_path / "teams_pipeline_store.json"),
        )
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["job_id"] == "job-1"
    assert payload["meeting_ref"]["meeting_id"] == "meeting-1"










def test_validate_accepts_msgraph_credentials_for_graph_delivery(monkeypatch, capsys, tmp_path):
    from gateway.config import Platform, PlatformConfig

    monkeypatch.setenv("MSGRAPH_TENANT_ID", "tenant")
    monkeypatch.setenv("MSGRAPH_CLIENT_ID", "client")
    monkeypatch.setenv("MSGRAPH_CLIENT_SECRET", "secret")

    gateway_config = SimpleNamespace(
        platforms={
            Platform.MSGRAPH_WEBHOOK: PlatformConfig(enabled=True, extra={}),
            Platform("teams"): PlatformConfig(
                enabled=True,
                extra={
                    "delivery_mode": "graph",
                    "team_id": "team-1",
                    "channel_id": "channel-1",
                },
            ),
        }
    )
    monkeypatch.setattr(
        "plugins.teams_pipeline.cli.load_gateway_config",
        lambda: gateway_config,
    )

    teams_pipeline_command(
        _make_args(
            teams_pipeline_action="validate",
            store_path=str(tmp_path / "teams_pipeline_store.json"),
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["issues"] == []
