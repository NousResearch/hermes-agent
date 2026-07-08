from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.teams_context.models import TeamsChatMessage, parse_chat_resource, strip_html
from plugins.teams_context.meeting_download import (
    DownloadedMeetingFiles,
    MeetingDownloadError,
    classify_sharepoint_stream_url,
    download_meeting,
    download_meetings,
    existing_meeting_files,
    pair_downloaded_files,
    parse_url_file,
    safe_filename,
    sanitize_url_for_metadata,
    write_download_report,
)
from plugins.teams_context.recording import (
    RecordingIngestError,
    download_recording_url,
    ingest_recording,
    parse_vtt_text,
)
from plugins.teams_context.store import TeamsContextStore
from plugins.teams_context.ui_scrape import parse_teams_channel_html, scrape_and_store


def test_strip_html_preserves_readable_text():
    assert strip_html("<p>Hello <b>Teams</b><br>second line</p>") == "Hello Teams\nsecond line"


def test_parse_chat_resource_supports_path_and_quoted_shapes():
    assert parse_chat_resource("chats/19:abc/messages/123") == ("19:abc", "123")
    assert parse_chat_resource("chats('19:abc')/messages('123')") == ("19:abc", "123")


def test_store_upsert_search_thread_and_delete(tmp_path: Path):
    store = TeamsContextStore(tmp_path / "teams.sqlite")
    message = TeamsChatMessage.from_graph(
        "19:chat",
        {
            "id": "msg-1",
            "createdDateTime": "2026-06-25T01:00:00Z",
            "lastModifiedDateTime": "2026-06-25T01:00:00Z",
            "from": {"user": {"id": "u1", "displayName": "Alice"}},
            "body": {"content": "<p>The queue is the scaling boundary.</p>"},
            "webUrl": "https://teams.example/msg-1",
        },
        tenant_id="tenant",
    )
    store.upsert_message(message)

    results = store.search("queue", chat_id="19:chat")
    assert len(results) == 1
    assert results[0]["message_id"] == "msg-1"

    thread = store.thread("19:chat", "msg-1")
    assert [row["message_id"] for row in thread] == ["msg-1"]

    store.mark_deleted("19:chat", "msg-1")
    assert store.search("queue", chat_id="19:chat") == []


@pytest.mark.anyio
async def test_graph_ingest_notification_fetches_and_stores(tmp_path: Path):
    from plugins.teams_context.graph import TeamsContextGraph

    class FakeClient:
        async def get_json(self, path):
            assert path.endswith("/messages/msg-1")
            return {
                "id": "msg-1",
                "createdDateTime": "2026-06-25T01:00:00Z",
                "from": {"user": {"displayName": "Alice"}},
                "body": {"content": "hello graph"},
            }

    store = TeamsContextStore(tmp_path / "teams.sqlite")
    graph = TeamsContextGraph(client=FakeClient(), store=store, tenant_id="tenant")
    result = await graph.ingest_notification(
        {
            "subscriptionId": "sub",
            "changeType": "created",
            "resource": "chats/19:chat/messages/msg-1",
            "clientState": "state",
        }
    )

    assert result["stored"] is True
    assert store.search("hello", chat_id="19:chat")[0]["message_id"] == "msg-1"


def test_teams_ui_fixture_parser_extracts_messages():
    fixture = Path("tests/fixtures/teams_channel_saved.html").read_text(encoding="utf-8")
    messages = parse_teams_channel_html(fixture, label="Example Planning")

    assert len(messages) == 2
    assert messages[0].message_id == "msg-1"
    assert messages[0].sender_name == "Alice Example"
    assert messages[0].channel_title == "Example Team / Planning"
    assert "scaling boundary" in messages[0].text
    assert messages[1].message_id.startswith("teams_ui:")


def test_scrape_and_store_uses_runtime_parser_result(monkeypatch, tmp_path: Path):
    from plugins.teams_context import ui_scrape

    fixture = Path("tests/fixtures/teams_channel_saved.html").read_text(encoding="utf-8")
    parsed = parse_teams_channel_html(fixture, label="Example Planning")
    monkeypatch.setattr(ui_scrape, "scrape_open_teams_tab", lambda **_kwargs: parsed)

    store = TeamsContextStore(tmp_path / "teams.sqlite")
    result = scrape_and_store(
        label="Example Planning",
        max_scrolls=1,
        since_days=7,
        store=store,
    )

    assert result["stored"] == 2
    assert store.search("boundary", chat_id="Example Planning")[0]["source_type"] == "channel"


def test_vtt_parser_removes_headers_and_timings():
    text = parse_vtt_text(
        """WEBVTT

1
00:00:01.000 --> 00:00:03.000
Alice: We should keep this searchable.

2
00:00:04.000 --> 00:00:06.000
<v Bob>Recording context matters.</v>
"""
    )

    assert "WEBVTT" not in text
    assert "-->" not in text
    assert "keep this searchable" in text
    assert "Recording context matters." in text


def test_recording_ingest_prefers_vtt_transcript(tmp_path: Path):
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"not a real mp4 because transcript is supplied")
    transcript = tmp_path / "meeting.vtt"
    transcript.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nTeamContext recording transcript.\n",
        encoding="utf-8",
    )
    store = TeamsContextStore(tmp_path / "teams.sqlite")

    result = ingest_recording(
        str(recording),
        meeting_label="Example meeting",
        transcript_path=str(transcript),
        store=store,
        artifact_cache=tmp_path / "cache",
    )

    assert result["source_type"] == "transcript"
    rows = store.unified_search("recording")
    assert len(rows) == 1
    assert rows[0]["source_type"] == "transcript"
    assert rows[0]["source_label"] == "Example meeting"


def test_recording_ingest_uses_ffmpeg_and_stt_when_transcript_absent(monkeypatch, tmp_path: Path):
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake")
    store = TeamsContextStore(tmp_path / "teams.sqlite")
    monkeypatch.setattr("plugins.teams_context.recording.shutil.which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(cmd, check, stdout, stderr):
        Path(cmd[-1]).write_bytes(b"wav")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("plugins.teams_context.recording.subprocess.run", fake_run)
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {"success": True, "transcript": "STT generated meeting knowledge."},
    )

    result = ingest_recording(
        str(recording),
        meeting_label="STT meeting",
        store=store,
        artifact_cache=tmp_path / "cache",
    )

    assert result["source_type"] == "recording"
    assert store.unified_search("generated")[0]["source_label"] == "STT meeting"


def test_authenticated_recording_urls_are_rejected(tmp_path: Path):
    with pytest.raises(RecordingIngestError, match="Download the recording locally"):
        download_recording_url(
            "https://tenant.sharepoint.com/sites/example/recording.mp4",
            cache_dir=tmp_path,
        )


def test_ordinary_recording_url_download_is_allowed(monkeypatch, tmp_path: Path):
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size=-1):
            if getattr(self, "_done", False):
                return b""
            self._done = True
            return b"recording-bytes"

    monkeypatch.setattr("plugins.teams_context.recording.urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())

    path = download_recording_url("https://example.com/recording.mp4", cache_dir=tmp_path)

    assert path.read_bytes() == b"recording-bytes"


def test_store_migration_is_idempotent_and_unified_searches_messages_and_kb(tmp_path: Path):
    db = tmp_path / "teams.sqlite"
    store = TeamsContextStore(db)
    store = TeamsContextStore(db)
    message = TeamsChatMessage.from_relay(
        {
            "chat_id": "Example channel",
            "message_id": "msg-1",
            "source_label": "Example channel",
            "text": "channel alpha",
        }
    )
    store.upsert_message(message)
    store.upsert_kb_chunk(
        source_id="meeting:example",
        item_id="meeting:example:chunk:0",
        source_type="recording",
        source_label="Example recording",
        text="recording alpha",
    )

    rows = store.unified_search("alpha", limit=10)

    assert {row["source_type"] for row in rows} == {"channel", "recording"}


def test_mcp_format_includes_source_type_and_metadata():
    from plugins.teams_context.mcp_server import _format_rows

    rows = _format_rows(
        [
            {
                "source_id": "meeting:example",
                "source_type": "recording",
                "source_label": "Example recording",
                "item_id": "chunk-1",
                "text": "hello",
                "metadata": {"meeting_label": "Example recording"},
            }
        ]
    )

    assert rows[0]["source_type"] == "recording"
    assert rows[0]["metadata"]["meeting_label"] == "Example recording"


def test_dashboard_payload_has_placeholder_labels_only(tmp_path: Path):
    store = TeamsContextStore(tmp_path / "teams.sqlite")
    store.upsert_kb_chunk(
        source_id="meeting:example",
        item_id="meeting:example:chunk:0",
        source_type="transcript",
        source_label="Example transcript",
        text="dashboard visible transcript",
    )

    payload = store.dashboard_items(source_type="transcript")

    assert payload["sources"][0]["label"] == "Example transcript"
    assert payload["items"][0]["source_type"] == "transcript"


def test_teams_context_cli_exposes_download_commands(tmp_path: Path):
    import argparse

    from plugins.teams_context.cli import register_cli

    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    teams = subs.add_parser("teams-context")
    register_cli(teams)

    one = parser.parse_args([
        "teams-context",
        "download-meeting",
        "https://tenant.sharepoint.com/sites/team/recording.mp4",
        "--output-dir",
        str(tmp_path),
    ])
    batch = parser.parse_args([
        "teams-context",
        "download-meetings",
        "--url-file",
        str(tmp_path / "urls.txt"),
        "--output-dir",
        str(tmp_path),
        "--no-ingest",
    ])

    assert one.teams_context_action == "download-meeting"
    assert batch.teams_context_action == "download-meetings"
    assert batch.no_ingest is True


def test_meeting_download_safe_filename_and_url_sanitization():
    assert safe_filename("  Exec: JCorp / Teams? Meeting  ") == "Exec-JCorp-Teams-Meeting"
    sanitized = sanitize_url_for_metadata(
        "https://user:pass@tenant.sharepoint.com/sites/x/Recording.mp4?e=secret&download=1#token"
    )

    assert sanitized == "https://tenant.sharepoint.com/sites/x/Recording.mp4"
    assert "secret" not in sanitized
    assert "user:pass" not in sanitized
    assert classify_sharepoint_stream_url(sanitized) == "sharepoint"


def test_meeting_download_existing_file_skip_and_force(tmp_path: Path):
    existing = tmp_path / "Example-Meeting.mp4"
    existing.write_bytes(b"recording")
    url = "https://tenant.sharepoint.com/sites/team/Example%20Meeting.mp4?access_token=secret"
    calls = {"count": 0}

    def fake_downloader(**_kwargs):
        calls["count"] += 1
        return DownloadedMeetingFiles(recording_path=existing)

    skipped = download_meeting(
        url,
        output_dir=tmp_path,
        ingest=False,
        browser_downloader=fake_downloader,
    )
    forced = download_meeting(
        url,
        output_dir=tmp_path,
        force=True,
        ingest=False,
        browser_downloader=fake_downloader,
    )

    assert existing_meeting_files(tmp_path, "Example-Meeting").skipped is True
    assert skipped.status == "skipped"
    assert calls["count"] == 1
    assert forced.status == "downloaded"


def test_meeting_download_url_file_batch_parsing(tmp_path: Path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n# comment\nhttps://tenant.sharepoint.com/sites/team/a.mp4\n  https://stream.microsoft.com/video/b  \n",
        encoding="utf-8",
    )

    assert parse_url_file(url_file) == [
        "https://tenant.sharepoint.com/sites/team/a.mp4",
        "https://stream.microsoft.com/video/b",
    ]


def test_meeting_download_report_excludes_sensitive_query_strings(tmp_path: Path):
    result = download_meeting(
        "https://tenant.sharepoint.com/sites/team/meeting.mp4?access_token=secret-token",
        output_dir=tmp_path,
        ingest=False,
        browser_downloader=lambda **_kwargs: DownloadedMeetingFiles(recording_path=tmp_path / "meeting.mp4"),
    )
    Path(result.recording_path).write_bytes(b"recording")

    report = write_download_report(tmp_path, [result])
    payload = report.read_text(encoding="utf-8")

    assert "secret-token" not in payload
    assert "access_token" not in payload
    assert "https://tenant.sharepoint.com/sites/team/meeting.mp4" in payload


def test_meeting_download_pairs_recording_with_transcript(tmp_path: Path):
    recording = tmp_path / "JCorp-Weekly.mp4"
    transcript = tmp_path / "JCorp-Weekly.vtt"
    other = tmp_path / "Other.vtt"
    for path in (recording, transcript, other):
        path.write_bytes(b"x")

    paired = pair_downloaded_files([other, recording, transcript])

    assert paired.recording_path == recording
    assert paired.transcript_path == transcript


def test_meeting_download_ingests_with_transcript_when_present(monkeypatch, tmp_path: Path):
    recording = tmp_path / "Meeting.mp4"
    transcript = tmp_path / "Meeting.vtt"
    recording.write_bytes(b"recording")
    transcript.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n", encoding="utf-8")
    captured = {}

    def fake_ingest(path_or_url, **kwargs):
        captured["path_or_url"] = path_or_url
        captured.update(kwargs)
        return {"chunks": 1, "transcript_source": "transcript"}

    monkeypatch.setattr("plugins.teams_context.meeting_download.ingest_recording", fake_ingest)

    result = download_meeting(
        "https://tenant.sharepoint.com/sites/team/Meeting.mp4?e=secret",
        output_dir=tmp_path,
        store=TeamsContextStore(tmp_path / "teams.sqlite"),
        browser_downloader=lambda **_kwargs: DownloadedMeetingFiles(recording_path=recording, transcript_path=transcript),
    )

    assert result.status == "ingested"
    assert captured["transcript_path"] == str(transcript)
    assert captured["metadata"]["source_url"] == "https://tenant.sharepoint.com/sites/team/Meeting.mp4"
    assert "e=secret" not in captured["metadata"]["source_url"]


def test_meeting_download_ingests_without_transcript_for_stt_fallback(monkeypatch, tmp_path: Path):
    recording = tmp_path / "Meeting.mp4"
    recording.write_bytes(b"recording")
    captured = {}

    def fake_ingest(path_or_url, **kwargs):
        captured["path_or_url"] = path_or_url
        captured.update(kwargs)
        return {"chunks": 1, "transcript_source": "stt"}

    monkeypatch.setattr("plugins.teams_context.meeting_download.ingest_recording", fake_ingest)

    result = download_meeting(
        "https://tenant.sharepoint.com/sites/team/Meeting.mp4",
        output_dir=tmp_path,
        store=TeamsContextStore(tmp_path / "teams.sqlite"),
        browser_downloader=lambda **_kwargs: DownloadedMeetingFiles(recording_path=recording),
    )

    assert result.status == "ingested"
    assert captured["transcript_path"] is None
    assert result.ingested["transcript_source"] == "stt"


def test_meeting_download_missing_browser_action_is_retryable(tmp_path: Path):
    def fake_downloader(**_kwargs):
        raise MeetingDownloadError("No recording or transcript download action was visible.", retryable=True)

    payload = download_meetings(
        ["https://tenant.sharepoint.com/sites/team/Meeting.mp4"],
        output_dir=tmp_path,
        ingest=False,
        browser_downloader=fake_downloader,
    )

    assert payload["failed"] == 1
    assert payload["results"][0]["retryable"] is True
    assert "download action" in payload["results"][0]["error"]
