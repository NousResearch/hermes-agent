"""Writes from the `hermes teams-pipeline` CLI survive a running gateway's later writes to the same store."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.msgraph_webhook import MSGraphWebhookAdapter
from hermes_constants import get_hermes_home
from plugins.teams_pipeline import cli as pipeline_cli, runtime as pipeline_runtime
from plugins.teams_pipeline.store import TeamsPipelineStore

CLIENT_STATE = "client-state-secret"


class _Graph:
    """Microsoft Graph boundary: accepts subscription creation; meeting lookups find nothing."""

    async def post_json(self, path, *, json_body=None, headers=None):
        return {"id": "sub-cli", "resource": json_body["resource"], "changeType": json_body["changeType"],
                "notificationUrl": json_body["notificationUrl"], "expirationDateTime": json_body["expirationDateTime"]}

    async def get_json(self, path, *, params=None, headers=None):
        return {}

    async def collect_paginated(self, path, *, params=None, headers=None):
        return []


class _Request:
    def __init__(self, body):
        self.query, self.remote, self.content_length = {}, "127.0.0.1", None
        self._raw = json.dumps(body).encode()

    async def read(self):
        return self._raw


@pytest.mark.anyio
async def test_cli_subscription_survives_gateway_notification(monkeypatch):
    monkeypatch.setattr(pipeline_runtime, "build_graph_client", _Graph)
    monkeypatch.setattr(pipeline_cli, "build_graph_client", _Graph)
    adapter = MSGraphWebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "client_state": CLIENT_STATE}))
    gateway = SimpleNamespace(adapters={Platform.MSGRAPH_WEBHOOK: adapter}, config=GatewayConfig(platforms={}))
    assert pipeline_runtime.bind_gateway_runtime(gateway) is True

    await asyncio.to_thread(pipeline_cli.teams_pipeline_command, Namespace(
        teams_pipeline_action="subscribe", store_path="", resource="communications/onlineMeetings/getAllTranscripts",
        notification_url="https://hooks.example.test/msgraph/webhook", change_type="", expiration="2030-01-01T00:00:00Z",
        client_state=CLIENT_STATE, lifecycle_notification_url="", latest_supported_tls_version="v1_2"))

    response = await adapter._handle_notification(_Request({"value": [{
        "id": "notif-1", "subscriptionId": "sub-cli", "clientState": CLIENT_STATE, "changeType": "updated",
        "resource": "communications/onlineMeetings/meeting-1", "resourceData": {"id": "meeting-1"}}]}))
    assert response.status == 202
    await asyncio.gather(*list(adapter._background_tasks))

    store = TeamsPipelineStore(get_hermes_home() / "teams_pipeline_store.json")
    assert len(store.list_jobs()) == 1
    assert list(store.list_subscriptions()) == ["sub-cli"]
