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


def test_register_cli_builds_tree():
    parser = ArgumentParser()
    register_cli(parser)
    args = parser.parse_args(["list"])
    assert args.teams_pipeline_action == "list"
    args = parser.parse_args(["release", "job-1"])
    assert args.teams_pipeline_action == "release"
    assert args.job_id == "job-1"


def test_list_prints_recent_jobs(capsys, tmp_path):
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
            teams_pipeline_action="list",
            store_path=str(tmp_path / "teams_pipeline_store.json"),
        )
    )
    out = capsys.readouterr().out
    assert "job-1" in out
    assert "meeting-1" in out


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


def test_fetch_requires_meeting_identifier(capsys):
    teams_pipeline_command(_make_args(teams_pipeline_action="fetch"))
    out = capsys.readouterr().out
    assert "meeting_id or join_web_url is required" in out


def test_release_prints_release_result(monkeypatch, capsys, tmp_path):
    class FakePipeline:
        async def release_teams_report(self, job_id):
            assert job_id == "job-1"
            return {"released": True, "sink_key": "teams:meeting-1"}

    monkeypatch.setattr(
        "plugins.teams_pipeline.cli._build_release_pipeline",
        lambda store: FakePipeline(),
    )

    teams_pipeline_command(
        _make_args(
            teams_pipeline_action="release",
            job_id="job-1",
            store_path=str(tmp_path / "teams_pipeline_store.json"),
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"released": True, "sink_key": "teams:meeting-1"}


def test_subscriptions_lists_graph_subscriptions(monkeypatch, capsys):
    class FakeClient:
        async def collect_paginated(self, path):
            assert path == "/subscriptions"
            return [
                {
                    "id": "sub-1",
                    "resource": "communications/onlineMeetings/getAllTranscripts",
                    "changeType": "updated",
                    "expirationDateTime": "2026-05-05T00:00:00Z",
                }
            ]

    monkeypatch.setattr("plugins.teams_pipeline.cli.build_graph_client", lambda: FakeClient())
    teams_pipeline_command(_make_args(teams_pipeline_action="subscriptions"))
    out = capsys.readouterr().out
    assert "sub-1" in out
    assert "getAllTranscripts" in out


def test_subscribe_defaults_to_created_for_transcript_resources(monkeypatch, capsys):
    captured = {}

    class FakeClient:
        async def post_json(self, path, json_body=None, headers=None):
            captured["path"] = path
            captured["json_body"] = json_body
            return {
                "id": "sub-transcript",
                "resource": json_body["resource"],
                "changeType": json_body["changeType"],
                "notificationUrl": json_body["notificationUrl"],
                "expirationDateTime": json_body["expirationDateTime"],
            }

    monkeypatch.setattr("plugins.teams_pipeline.cli.build_graph_client", lambda: FakeClient())
    teams_pipeline_command(
        _make_args(
            teams_pipeline_action="subscribe",
            resource="communications/onlineMeetings/getAllTranscripts",
            notification_url="https://example.com/webhooks/msgraph",
            change_type="",
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert captured["path"] == "/subscriptions"
    assert captured["json_body"]["changeType"] == "created"
    assert payload["changeType"] == "created"


def test_token_health_force_refresh(monkeypatch, capsys):
    class FakeProvider:
        def inspect_token_health(self):
            return {"configured": True, "cache_state": "warm"}

        async def get_access_token(self, force_refresh=False):
            assert force_refresh is True
            return "token-123"

    monkeypatch.setattr(
        "plugins.teams_pipeline.cli.MicrosoftGraphTokenProvider",
        SimpleNamespace(from_env=lambda: FakeProvider()),
    )
    teams_pipeline_command(_make_args(teams_pipeline_action="token-health", force_refresh=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["configured"] is True
    assert payload["last_refresh_succeeded"] is True
    assert payload["access_token_length"] == len("token-123")


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
    assert payload["video_storage_policy"] == {
        "durable_storage": "onedrive_or_google_drive",
        "local_storage": "temporary_processing_only",
        "tmp_dir": None,
        "retention": "deleted_after_transcription",
    }


def test_validate_rejects_local_durable_video_storage(monkeypatch, capsys, tmp_path):
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
                    "meeting_pipeline": {"video_storage": {"provider": "laptop"}},
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
    assert payload["ok"] is False
    assert "video_storage.provider must be onedrive, google_drive, or sharepoint." in payload["issues"]
