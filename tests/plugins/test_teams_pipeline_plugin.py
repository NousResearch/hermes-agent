"""Tests for the Teams pipeline plugin package."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from gateway.config import GatewayConfig, Platform, PlatformConfig
from plugins.teams_pipeline import register
from plugins.teams_pipeline.pipeline import TeamsMeetingPipeline
from plugins.teams_pipeline.store import TeamsPipelineStore
from plugins.teams_pipeline.models import MeetingArtifact


class FakeGraphClient:
    def __init__(self) -> None:
        self.downloaded = False


async def _transcript_meeting_resolver(client, *, meeting_id=None, join_web_url=None, tenant_id=None):
    from plugins.teams_pipeline.models import TeamsMeetingRef

    return TeamsMeetingRef(
        meeting_id=str(meeting_id),
        tenant_id=tenant_id,
        metadata={"subject": "Weekly Sync", "participants": [{"displayName": "Ada"}]},
    )


async def _no_call_record(*args, **kwargs):
    return None


def test_register_adds_cli_only():
    mgr = PluginManager()
    manifest = PluginManifest(name="teams_pipeline")
    ctx = PluginContext(manifest, mgr)

    register(ctx)

    assert "teams-pipeline" in mgr._cli_commands
    entry = mgr._cli_commands["teams-pipeline"]
    assert entry["plugin"] == "teams_pipeline"
    assert callable(entry["setup_fn"])
    assert callable(entry["handler_fn"])


def test_runtime_config_uses_existing_teams_platform_settings():
    from plugins.teams_pipeline.runtime import build_pipeline_runtime_config

    gateway_config = GatewayConfig(
        platforms={
            Platform("teams"): PlatformConfig(
                enabled=True,
                extra={
                    "delivery_mode": "graph",
                    "team_id": "team-1",
                    "channel_id": "channel-1",
                    "meeting_pipeline": {
                        "transcript_min_chars": 120,
                        "notion": {"enabled": True, "database_id": "db-1"},
                    },
                },
            )
        }
    )

    runtime_config = build_pipeline_runtime_config(gateway_config)

    assert runtime_config["transcript_min_chars"] == 120
    assert runtime_config["notion"]["database_id"] == "db-1"
    assert runtime_config["teams_delivery"] == {
        "enabled": True,
        "mode": "graph",
        "team_id": "team-1",
        "channel_id": "channel-1",
    }


def test_build_pipeline_runtime_reuses_existing_teams_adapter_surface(monkeypatch, tmp_path):
    from plugins.teams_pipeline import runtime as runtime_module

    class FakeWriter:
        def __init__(self, platform_config=None, **kwargs) -> None:
            self.platform_config = platform_config

    monkeypatch.setattr(runtime_module, "build_graph_client", lambda: object())
    monkeypatch.setattr(runtime_module, "resolve_teams_pipeline_store_path", lambda: tmp_path / "teams-store.json")
    monkeypatch.setattr("plugins.platforms.teams.adapter.TeamsSummaryWriter", FakeWriter)

    gateway = SimpleNamespace(
        config=GatewayConfig(
            platforms={
                Platform("teams"): PlatformConfig(
                    enabled=True,
                    extra={
                        "delivery_mode": "incoming_webhook",
                        "incoming_webhook_url": "https://example.com/hook",
                    },
                )
            }
        )
    )

    runtime = runtime_module.build_pipeline_runtime(gateway)

    assert isinstance(runtime.teams_sender, FakeWriter)
    assert runtime.teams_sender.platform_config is gateway.config.platforms[Platform("teams")]


@pytest.mark.anyio
async def test_bind_gateway_runtime_attaches_scheduler(monkeypatch, tmp_path):
    from plugins.teams_pipeline import runtime as runtime_module

    class FakeAdapter:
        def __init__(self) -> None:
            self.scheduler = None

        def set_notification_scheduler(self, scheduler) -> None:
            self.scheduler = scheduler

    class FakePipeline:
        def __init__(self) -> None:
            self.notifications = []

        async def run_notification(self, notification):
            self.notifications.append(notification)

    adapter = FakeAdapter()
    pipeline = FakePipeline()
    gateway = SimpleNamespace(
        adapters={Platform.MSGRAPH_WEBHOOK: adapter},
        config=GatewayConfig(platforms={}),
        _teams_pipeline_runtime=None,
        _teams_pipeline_runtime_error=None,
    )

    monkeypatch.setattr(runtime_module, "build_pipeline_runtime", lambda gateway_runner: pipeline)

    bound = runtime_module.bind_gateway_runtime(gateway)

    assert bound is True
    assert gateway._teams_pipeline_runtime is pipeline
    assert callable(adapter.scheduler)

    notification = {"id": "notif-1"}
    await adapter.scheduler(notification, object())
    assert pipeline.notifications == [notification]


@pytest.mark.anyio
async def test_bind_gateway_runtime_drops_notifications_when_unavailable(monkeypatch):
    from plugins.teams_pipeline import runtime as runtime_module
    from tools.microsoft_graph_auth import MicrosoftGraphConfigError

    class FakeAdapter:
        def __init__(self) -> None:
            self.scheduler = None

        def set_notification_scheduler(self, scheduler) -> None:
            self.scheduler = scheduler

    adapter = FakeAdapter()
    gateway = SimpleNamespace(
        adapters={Platform.MSGRAPH_WEBHOOK: adapter},
        config=GatewayConfig(platforms={}),
        _teams_pipeline_runtime=None,
        _teams_pipeline_runtime_error=None,
    )

    def _raise(_gateway_runner):
        raise MicrosoftGraphConfigError("missing graph env")

    monkeypatch.setattr(runtime_module, "build_pipeline_runtime", _raise)

    bound = runtime_module.bind_gateway_runtime(gateway)

    assert bound is False
    assert "missing graph env" in gateway._teams_pipeline_runtime_error
    assert callable(adapter.scheduler)
    await adapter.scheduler({"id": "notif-2"}, object())


def test_store_persists_subscription_event_and_job_state(tmp_path):
    store_path = tmp_path / "teams-store.json"
    store = TeamsPipelineStore(store_path)
    store.upsert_subscription(
        "sub-1",
        {"client_state": "abc", "resource": "communications/onlineMeetings"},
    )
    store.record_event_timestamp("evt-1", "2026-05-03T19:30:00Z")
    store.upsert_job("job-1", {"status": "received", "event_id": "evt-1"})
    store.upsert_sink_record("notion:meeting-1", {"page_id": "page-1"})

    reloaded = TeamsPipelineStore(store_path)
    subscription = reloaded.get_subscription("sub-1")
    job = reloaded.get_job("job-1")
    sink = reloaded.get_sink_record("notion:meeting-1")

    assert subscription is not None
    assert subscription["subscription_id"] == "sub-1"
    assert subscription["client_state"] == "abc"
    assert reloaded.get_event_timestamp("evt-1") == "2026-05-03T19:30:00Z"
    assert job is not None
    assert job["status"] == "received"
    assert sink is not None
    assert sink["page_id"] == "page-1"


def test_store_notification_receipts_are_idempotent(tmp_path):
    store = TeamsPipelineStore(tmp_path / "teams-store.json")
    notification = {
        "subscriptionId": "sub-1",
        "resource": "communications/onlineMeetings/meeting-1",
        "changeType": "updated",
    }
    receipt_key = TeamsPipelineStore.build_notification_receipt_key(notification)

    assert store.record_notification_receipt(receipt_key, notification) is True
    assert store.record_notification_receipt(receipt_key, notification) is False
    assert store.has_notification_receipt(receipt_key) is True

    reloaded = TeamsPipelineStore(tmp_path / "teams-store.json")
    assert reloaded.has_notification_receipt(receipt_key) is True


@pytest.mark.anyio
class TestTeamsMeetingPipeline:
    async def test_transcript_first_path_persists_state_and_skips_recording(self, tmp_path, monkeypatch):
        from plugins.teams_pipeline import pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "resolve_meeting_reference", _transcript_meeting_resolver)

        async def _fetch_transcript(client, meeting_ref):
            return (
                MeetingArtifact(artifact_type="transcript", artifact_id="tx-1", display_name="meeting.vtt"),
                "Action: Send draft by Friday.\nDecision: Ship the transcript-first path.\nDetailed transcript content.",
            )

        async def _call_record(client, meeting_ref, *, call_record_id=None, allow_permission_errors=True):
            return MeetingArtifact(
                artifact_type="call_record",
                artifact_id="call-1",
                metadata={"metrics": {"participant_count": 4}},
            )

        async def _summarize(**kwargs):
            return pipeline_module.TeamsMeetingSummaryPayload(
                meeting_ref=kwargs["resolved_meeting"],
                title="Weekly Sync",
                transcript_text=kwargs["transcript_text"],
                summary="Short summary",
                key_decisions=["Ship the transcript-first path."],
                action_items=["Send draft by Friday."],
                risks=["Timeline risk."],
                confidence="high",
                confidence_notes="Transcript available.",
                source_artifacts=kwargs["artifacts"],
            )

        monkeypatch.setattr(pipeline_module, "fetch_preferred_transcript_text", _fetch_transcript)
        monkeypatch.setattr(pipeline_module, "enrich_meeting_with_call_record", _call_record)

        store = TeamsPipelineStore(tmp_path / "teams-store.json")
        pipeline = TeamsMeetingPipeline(
            graph_client=FakeGraphClient(),
            store=store,
            config={"transcript_min_chars": 20},
            summarize_fn=_summarize,
        )

        job = await pipeline.run_notification(
            {
                "id": "notif-1",
                "changeType": "updated",
                "resource": "communications/onlineMeetings/meeting-123",
                "resourceData": {"id": "meeting-123"},
            }
        )

        assert job.status == "completed"
        assert job.selected_artifact_strategy == "transcript_first"
        assert job.summary_payload is not None
        assert job.summary_payload.summary == "Short summary"
        stored = store.get_job(job.job_id)
        assert stored is not None
        assert stored["status"] == "completed"

    async def test_recording_fallback_uses_stt_and_updates_sink_records(self, tmp_path, monkeypatch):
        from plugins.teams_pipeline import pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "resolve_meeting_reference", _transcript_meeting_resolver)

        async def _no_transcript(client, meeting_ref):
            return None, None

        async def _recordings(client, meeting_ref):
            return [
                MeetingArtifact(
                    artifact_type="recording",
                    artifact_id="rec-1",
                    display_name="recording.mp4",
                    download_url="https://files.example/recording.mp4",
                )
            ]

        async def _download(client, meeting_ref, recording, destination):
            target = Path(destination)
            target.write_bytes(b"video-bytes")
            return {"path": str(target), "size_bytes": 11, "content_type": "video/mp4"}

        async def _prepare_audio(self, recording_path):
            audio_path = recording_path.with_suffix(".wav")
            audio_path.write_bytes(b"audio-bytes")
            return audio_path

        def _transcribe(file_path, model):
            return {"success": True, "transcript": "Action: Follow up with Legal.\nRisk: Budget approval pending.", "provider": "local"}

        async def _summarize(**kwargs):
            return pipeline_module.TeamsMeetingSummaryPayload(
                meeting_ref=kwargs["resolved_meeting"],
                title="Weekly Sync",
                transcript_text=kwargs["transcript_text"],
                summary="Fallback summary",
                key_decisions=[],
                action_items=["Follow up with Legal."],
                risks=["Budget approval pending."],
                confidence="medium",
                confidence_notes="Generated from STT fallback.",
                source_artifacts=kwargs["artifacts"],
            )

        class FakeNotionWriter:
            async def write_summary(self, payload, config, existing_record=None):
                return {"page_id": existing_record.get("page_id") if existing_record else "page-1", "url": "https://notion.so/page-1"}

        async def _teams_sender(payload, config, existing_record=None):
            return {"message_id": existing_record.get("message_id") if existing_record else "msg-1"}

        monkeypatch.setattr(pipeline_module, "fetch_preferred_transcript_text", _no_transcript)
        monkeypatch.setattr(pipeline_module, "list_recording_artifacts", _recordings)
        monkeypatch.setattr(pipeline_module, "download_recording_artifact", _download)
        monkeypatch.setattr(pipeline_module.TeamsMeetingPipeline, "_prepare_audio_path", _prepare_audio)
        monkeypatch.setattr(pipeline_module, "enrich_meeting_with_call_record", _no_call_record)

        store = TeamsPipelineStore(tmp_path / "teams-store.json")
        pipeline = TeamsMeetingPipeline(
            graph_client=FakeGraphClient(),
            store=store,
            config={
                "notion": {"enabled": True, "database_id": "db-1"},
                "teams_delivery": {"enabled": True, "channel_id": "channel-1"},
            },
            transcribe_fn=_transcribe,
            summarize_fn=_summarize,
            notion_writer=FakeNotionWriter(),
            teams_sender=_teams_sender,
        )

        job = await pipeline.run_notification(
            {
                "id": "notif-2",
                "changeType": "updated",
                "resource": "communications/onlineMeetings/meeting-456",
                "resourceData": {"id": "meeting-456"},
            }
        )

        assert job.status == "completed"
        assert job.selected_artifact_strategy == "recording_stt_fallback"
        assert job.summary_payload is not None
        assert job.summary_payload.summary == "Fallback summary"
        notion_record = store.get_sink_record("notion:meeting-456")
        teams_record = store.get_sink_record("teams:meeting-456")
        assert notion_record is not None
        assert notion_record["page_id"] == "page-1"
        assert teams_record is not None
        assert teams_record["message_id"] == "msg-1"

    async def test_join_web_url_returns_pending_reference_when_graph_cannot_resolve(self, monkeypatch):
        from plugins.teams_pipeline import meetings as meetings_module

        class FakeClient:
            async def get_json(self, path, params=None, headers=None):
                raise meetings_module.MicrosoftGraphAPIError(
                    400,
                    "GET",
                    f"https://graph.microsoft.com/v1.0{path}",
                    "BadRequest",
                )

        ref = await meetings_module.resolve_meeting_reference(
            FakeClient(),
            join_web_url="https://teams.microsoft.com/meet/278515858493828?p=8FExxBj8D8N6Zxd1vu",
        )

        assert ref.join_web_url == "https://teams.microsoft.com/meet/278515858493828?p=8FExxBj8D8N6Zxd1vu"
        assert ref.metadata["pending_resolution"] is True
        assert ref.meeting_id == "278515858493828"

    async def test_chat_url_bridges_through_call_record_lookup(self, monkeypatch):
        from plugins.teams_pipeline import meetings as meetings_module

        chat_url = "https://teams.microsoft.com/l/chat/19:meeting_MGQ3MGMxNDQtOGU0ZC00Yzc0LTkxY2MtZjRhNDU3NTcxY2Q4@thread.v2/conversations?context=%7B%22contextType%22%3A%22chat%22%7D"
        call_join_url = "https://teams.microsoft.com/l/meetup-join/19%3ameeting_MGQ3MGMxNDQtOGU0ZC00Yzc0LTkxY2MtZjRhNDU3NTcxY2Q4%40thread.v2/0?context=%7B%22Tid%22%3A%2219cd252c-b188-4d97-96dd-c74217c18f6d%22%2C%22Oid%22%3A%2229427ce2-dce0-42ce-b212-92a227f61979%22%7D"

        class FakeClient:
            async def collect_paginated(self, path):
                if path == "/communications/callRecords":
                    return [
                        {
                            "id": "call-record-1",
                            "joinWebUrl": call_join_url,
                            "organizer": {"user": {"id": "user-123"}},
                        }
                    ]
                raise AssertionError(f"Unexpected Graph call: {path}")

            async def get_json(self, path, params=None, headers=None):
                if path == "/communications/onlineMeetings":
                    raise meetings_module.MicrosoftGraphAPIError(
                        404,
                        "GET",
                        f"https://graph.microsoft.com/v1.0{path}",
                        "NotFound",
                    )
                expected_filter = "JoinWebUrl eq '{}'".format(call_join_url.replace("'", "''"))
                if path == "/users/user-123/onlineMeetings" and params == {"$filter": expected_filter}:
                    return {
                        "value": [
                            {
                                "id": "meeting-123",
                                "joinWebUrl": call_join_url,
                                "subject": "Weekly Sync",
                                "organizer": {"user": {"id": "user-123"}},
                                "participants": [{"displayName": "Ada"}],
                                "chatInfo": {"threadId": "19:meeting_MGQ3MGMxNDQtOGU0ZC00Yzc0LTkxY2MtZjRhNDU3NTcxY2Q4@thread.v2"},
                            }
                        ]
                    }
                raise AssertionError(f"Unexpected Graph call: {path} {params}")

        ref = await meetings_module.resolve_meeting_reference(FakeClient(), join_web_url=chat_url)

        assert ref.meeting_id == "meeting-123"
        assert ref.organizer_user_id == "user-123"
        assert ref.join_web_url == call_join_url
        assert ref.thread_id == "19:meeting_MGQ3MGMxNDQtOGU0ZC00Yzc0LTkxY2MtZjRhNDU3NTcxY2Q4@thread.v2"

    async def test_call_record_notification_bridges_past_online_meeting_lookup_error(self, monkeypatch):
        from plugins.teams_pipeline import meetings as meetings_module

        call_record_id = "call-record-1"
        call_join_url = "https://teams.microsoft.com/l/meetup-join/19%3ameeting_callrecord@thread.v2/0?context=%7B%22Tid%22%3A%2219cd252c-b188-4d97-96dd-c74217c18f6d%22%2C%22Oid%22%3A%22user-123%22%7D"

        class FakeClient:
            async def get_json(self, path, params=None, headers=None):
                if path == f"/communications/onlineMeetings/{call_record_id}":
                    raise meetings_module.MicrosoftGraphAPIError(
                        400,
                        "GET",
                        f"https://graph.microsoft.com/v1.0{path}",
                        "BadRequest",
                    )
                if path == f"/communications/callRecords/{call_record_id}":
                    return {
                        "id": call_record_id,
                        "joinWebUrl": call_join_url,
                        "organizer": {"user": {"id": "user-123"}},
                    }
                # VTC filter may be tried first (Strategy 1) — expect it to fail
                if path == "/communications/onlineMeetings" and params and "$filter" in params:
                    if "VideoTeleconferenceId" in params["$filter"]:
                        raise meetings_module.MicrosoftGraphAPIError(
                            404, "GET", f"https://graph.microsoft.com/v1.0{path}", "NotFound"
                        )
                    # JoinWebUrl filter on /communications (Strategy 2) — also fails
                    if "JoinWebUrl eq" in params["$filter"]:
                        raise meetings_module.MicrosoftGraphAPIError(
                            400, "GET", f"https://graph.microsoft.com/v1.0{path}", "BadRequest"
                        )
                expected_filter = "JoinWebUrl eq '{}'".format(call_join_url.replace("'", "''"))
                if path == "/users/user-123/onlineMeetings" and params == {"$filter": expected_filter}:
                    return {
                        "value": [
                            {
                                "id": "meeting-123",
                                "joinWebUrl": call_join_url,
                                "organizer": {"user": {"id": "user-123"}},
                            }
                        ]
                    }
                raise AssertionError(f"Unexpected Graph call: {path} {params}")

        ref = await meetings_module.resolve_meeting_reference(FakeClient(), meeting_id=call_record_id)

        assert ref.meeting_id == "meeting-123"
        assert ref.organizer_user_id == "user-123"
        assert ref.join_web_url == call_join_url

    async def test_join_web_url_permission_errors_surface(self, monkeypatch):
        from plugins.teams_pipeline import meetings as meetings_module

        class FakeClient:
            async def get_json(self, path, params=None, headers=None):
                raise meetings_module.MicrosoftGraphAPIError(
                    403,
                    "GET",
                    f"https://graph.microsoft.com/v1.0{path}",
                    "Forbidden",
                )

        with pytest.raises(meetings_module.TeamsMeetingPermissionError):
            await meetings_module.resolve_meeting_reference(
                FakeClient(),
                join_web_url="https://teams.microsoft.com/meet/278515858493828?p=8FExxBj8D8N6Zxd1vu",
            )

    async def test_fetch_preferred_transcript_text_uses_resolved_meeting_for_download(self, monkeypatch):
        from plugins.teams_pipeline import meetings as meetings_module

        pending_ref = meetings_module.TeamsMeetingRef(
            meeting_id="278515858493828",
            join_web_url="https://teams.microsoft.com/meet/278515858493828?p=8FExxBj8D8N6Zxd1vu",
            tenant_id="tenant-1",
            metadata={"pending_resolution": True, "join_web_url": "https://teams.microsoft.com/meet/278515858493828?p=8FExxBj8D8N6Zxd1vu"},
        )
        resolved_ref = meetings_module.TeamsMeetingRef(
            meeting_id="meeting-123",
            organizer_user_id="user-123",
            join_web_url="https://teams.microsoft.com/l/meetup-join/real",
            tenant_id="tenant-1",
            metadata={"subject": "Weekly Sync"},
        )
        captured = {}

        async def _resolve_artifact_meeting_ref(client, meeting_ref):
            return resolved_ref

        async def _list_transcript_artifacts(client, meeting_ref):
            captured["listed_ref"] = meeting_ref.meeting_id
            return [
                MeetingArtifact(
                    artifact_type="transcript",
                    artifact_id="tx-1",
                    display_name="meeting.vtt",
                    download_url=None,
                )
            ]

        async def _download_transcript_text(
            client,
            meeting_ref,
            transcript,
            *,
            resolved_meeting_ref=None,
            encoding="utf-8",
        ):
            captured["download_ref"] = meeting_ref.meeting_id
            captured["resolved_download_ref"] = resolved_meeting_ref.meeting_id if resolved_meeting_ref else None
            return "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world\n"

        monkeypatch.setattr(meetings_module, "_resolve_artifact_meeting_ref", _resolve_artifact_meeting_ref)
        monkeypatch.setattr(meetings_module, "list_transcript_artifacts", _list_transcript_artifacts)
        monkeypatch.setattr(meetings_module, "download_transcript_text", _download_transcript_text)

        transcript, text = await meetings_module.fetch_preferred_transcript_text(object(), pending_ref)

        assert transcript is not None
        assert text is not None
        assert captured["listed_ref"] == "meeting-123"
        assert captured["download_ref"] == "meeting-123"
        assert captured["resolved_download_ref"] == "meeting-123"

    async def test_download_recording_artifact_uses_resolved_meeting_for_download(self, tmp_path):
        from plugins.teams_pipeline import meetings as meetings_module

        captured = {}

        class FakeClient:
            async def download_to_file(self, path, destination):
                captured["path"] = path
                destination.write_text("binary", encoding="utf-8")
                return {"path": str(destination), "size_bytes": destination.stat().st_size, "content_type": "video/mp4"}

        pending_ref = meetings_module.TeamsMeetingRef(
            meeting_id="278515858493828",
            join_web_url="https://teams.microsoft.com/meet/278515858493828?p=8FExxBj8D8N6Zxd1vu",
            tenant_id="tenant-1",
            metadata={"pending_resolution": True, "join_web_url": "https://teams.microsoft.com/meet/278515858493828?p=8FExxBj8D8N6Zxd1vu"},
        )
        resolved_ref = meetings_module.TeamsMeetingRef(
            meeting_id="meeting-123",
            organizer_user_id="user-123",
            join_web_url="https://teams.microsoft.com/l/meetup-join/real",
            tenant_id="tenant-1",
            metadata={"subject": "Weekly Sync"},
        )
        recording = MeetingArtifact(
            artifact_type="recording",
            artifact_id="rec-1",
            display_name="meeting.mp4",
            download_url=None,
        )

        result = await meetings_module.download_recording_artifact(
            FakeClient(),
            pending_ref,
            recording,
            tmp_path / "out.mp4",
            resolved_meeting_ref=resolved_ref,
        )

        assert captured["path"] == "/users/user-123/onlineMeetings/meeting-123/recordings/rec-1/content"
        assert result["size_bytes"] > 0

    async def test_transcript_artifacts_use_organizer_specific_endpoint(self, tmp_path, monkeypatch):
        from plugins.teams_pipeline import meetings as meetings_module

        captured = {"paths": []}

        class FakeClient:
            async def collect_paginated(self, path):
                captured["paths"].append(path)
                return [
                    {
                        "id": "tx-1",
                        "displayName": "meeting.vtt",
                        "contentType": "text/vtt",
                        "downloadUrl": None,
                    }
                ]

            async def download_to_file(self, path, destination):
                captured["paths"].append(path)
                destination.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world\n", encoding="utf-8")
                return {"path": str(destination), "size_bytes": destination.stat().st_size, "content_type": "text/vtt"}

        meeting_ref = meetings_module.TeamsMeetingRef(
            meeting_id="meeting-123",
            organizer_user_id="user-123",
            join_web_url="https://teams.microsoft.com/l/meetup-join/abc",
            tenant_id="tenant-1",
            metadata={"subject": "Weekly Sync"},
        )

        transcript, text = await meetings_module.fetch_preferred_transcript_text(FakeClient(), meeting_ref)

        assert transcript is not None
        assert text is not None
        assert captured["paths"][0] == "/users/user-123/onlineMeetings/meeting-123/transcripts"
        assert captured["paths"][1] == "/users/user-123/onlineMeetings/meeting-123/transcripts/tx-1/content"
        assert "Hello world" in text


    async def test_missing_transcript_endpoint_falls_back_to_recordings(self, tmp_path, monkeypatch):
        from plugins.teams_pipeline import meetings as meetings_module

        async def _missing(*args, **kwargs):
            raise meetings_module.TeamsMeetingNotFoundError("No transcripts found for Teams meeting meeting-789")

        monkeypatch.setattr(meetings_module, "list_transcript_artifacts", _missing)

        meeting_ref = await _transcript_meeting_resolver(FakeGraphClient())
        transcript_artifact, transcript_text = await meetings_module.fetch_preferred_transcript_text(
            FakeGraphClient(),
            meeting_ref,
        )

        assert transcript_artifact is None
        assert transcript_text is None

    async def test_missing_transcript_and_recording_schedules_retry(self, tmp_path, monkeypatch):
        from plugins.teams_pipeline import pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "resolve_meeting_reference", _transcript_meeting_resolver)
        monkeypatch.setattr(pipeline_module, "fetch_preferred_transcript_text", lambda *a, **kw: asyncio.sleep(0, result=(None, None)))
        monkeypatch.setattr(pipeline_module, "list_recording_artifacts", lambda *a, **kw: asyncio.sleep(0, result=[]))

        store = TeamsPipelineStore(tmp_path / "teams-store.json")
        pipeline = TeamsMeetingPipeline(
            graph_client=FakeGraphClient(),
            config={},
            summarize_fn=lambda **kwargs: asyncio.sleep(0, result=None),
            store=store,
        )

        job = await pipeline.run_notification(
            {
                "id": "notif-3",
                "changeType": "updated",
                "resource": "communications/onlineMeetings/meeting-789",
                "resourceData": {"id": "meeting-789"},
            }
        )

        assert job.status == "retry_scheduled"
        assert job.error_info["retryable"] is True
        assert "Recording unavailable" in job.error_info["message"]

    async def test_duplicate_notification_reuses_completed_job(self, tmp_path, monkeypatch):
        from plugins.teams_pipeline import pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "resolve_meeting_reference", _transcript_meeting_resolver)

        async def _fetch_transcript(client, meeting_ref):
            return (
                MeetingArtifact(artifact_type="transcript", artifact_id="tx-dup", display_name="meeting.vtt"),
                "Decision: Keep duplicate notifications idempotent.\nAction: Verify the cached job is reused.",
            )

        summarize_calls = 0

        async def _summarize(**kwargs):
            nonlocal summarize_calls
            summarize_calls += 1
            return pipeline_module.TeamsMeetingSummaryPayload(
                meeting_ref=kwargs["resolved_meeting"],
                title="Weekly Sync",
                transcript_text=kwargs["transcript_text"],
                summary="Duplicate-safe summary",
                key_decisions=["Keep duplicate notifications idempotent."],
                action_items=["Verify the cached job is reused."],
                confidence="high",
                confidence_notes="Transcript available.",
                source_artifacts=kwargs["artifacts"],
            )

        monkeypatch.setattr(pipeline_module, "fetch_preferred_transcript_text", _fetch_transcript)
        monkeypatch.setattr(pipeline_module, "enrich_meeting_with_call_record", _no_call_record)

        store = TeamsPipelineStore(tmp_path / "teams-store.json")
        pipeline = TeamsMeetingPipeline(
            graph_client=FakeGraphClient(),
            store=store,
            config={"transcript_min_chars": 20},
            summarize_fn=_summarize,
        )
        notification = {
            "id": "notif-dup",
            "changeType": "updated",
            "resource": "communications/onlineMeetings/meeting-dup",
            "resourceData": {"id": "meeting-dup"},
        }

        first_job = await pipeline.run_notification(notification)
        second_job = await pipeline.run_notification(notification)

        assert first_job.status == "completed"
        assert second_job.status == "completed"
        assert second_job.job_id == first_job.job_id
        assert summarize_calls == 1
        assert len(store.list_jobs()) == 1
        receipt_key = TeamsPipelineStore.build_notification_receipt_key(notification)
        assert store.has_notification_receipt(receipt_key) is True
