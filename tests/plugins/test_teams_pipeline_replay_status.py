"""A Teams meeting job whose artifacts are not ready yet waits for an operator replay, and says so."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.msgraph_webhook import MSGraphWebhookAdapter
from plugins.teams_pipeline import cli as pipeline_cli, runtime as pipeline_runtime

CLIENT_STATE = "client-state-secret"
TRANSCRIPT = "WEBVTT\n\n00:00.000 --> 00:05.000\nAda: Decision: ship the replay runbook this week.\nAction: Bob updates the guide.\n"


class _Graph:
    """Microsoft Graph boundary: the meeting exists; its transcript appears once ``ready`` is set."""

    ready = False

    async def get_json(self, path, *, params=None, headers=None):
        return {"id": "meeting-1", "subject": "Weekly Sync"} if "onlineMeetings" in path else {}

    async def collect_paginated(self, path, *, params=None, headers=None):
        if type(self).ready and path.endswith("/transcripts"):
            return [{"id": "tx-1", "displayName": "meeting.vtt", "status": "completed"}]
        return []

    async def download_to_file(self, path, destination, *, headers=None, **_kw):
        Path(destination).write_text(TRANSCRIPT, encoding="utf-8")
        return {"size_bytes": len(TRANSCRIPT)}


class _Request:
    def __init__(self, body):
        self.query, self.remote, self.content_length = {}, "127.0.0.1", None
        self._raw = json.dumps(body).encode()

    async def read(self):
        return self._raw


def _cli(capsys, action, **kwargs):
    capsys.readouterr()
    pipeline_cli.teams_pipeline_command(Namespace(teams_pipeline_action=action, store_path="", **kwargs))
    return capsys.readouterr().out


@pytest.mark.anyio
async def test_job_without_artifacts_awaits_replay_and_replay_completes_it(monkeypatch, capsys):
    monkeypatch.setattr(_Graph, "ready", False)
    monkeypatch.setattr(pipeline_runtime, "build_graph_client", _Graph)
    monkeypatch.setattr(pipeline_cli, "build_graph_client", _Graph)
    adapter = MSGraphWebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "client_state": CLIENT_STATE}))
    gateway = SimpleNamespace(adapters={Platform.MSGRAPH_WEBHOOK: adapter}, config=GatewayConfig(platforms={}))
    assert pipeline_runtime.bind_gateway_runtime(gateway) is True

    response = await adapter._handle_notification(_Request({"value": [{
        "id": "notif-1", "subscriptionId": "sub-1", "clientState": CLIENT_STATE, "changeType": "updated",
        "resource": "communications/onlineMeetings/meeting-1", "resourceData": {"id": "meeting-1"}}]}))
    assert response.status == 202
    await asyncio.gather(*list(adapter._background_tasks))

    listing = _cli(capsys, "list", status="awaiting_replay", limit=20)
    assert "1 Teams pipeline job(s)" in listing, _cli(capsys, "list", status="", limit=20)
    job_id = listing.split("◆", 1)[1].split()[0]

    monkeypatch.setattr(_Graph, "ready", True)
    replayed = json.loads(await asyncio.to_thread(_cli, capsys, "run", job_id=job_id))
    assert replayed["status"] == "completed"
    assert replayed["selected_artifact_strategy"] == "transcript_first"
