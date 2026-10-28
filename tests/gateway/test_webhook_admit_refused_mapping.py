"""_admit_agent_run maps admission refusals to outcomes the webhook answers:

- runtime_draining  -> "retry"   -> 503 + Retry-After, the provider redelivers
- profile_mismatch  -> "refused" -> 404,  the provider stops redelivering a delivery this
                                     gateway can never admit
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.platforms.webhook import WebhookAdapter
from hermes_state_runtime import RuntimeStoreError


class _StubAdapter:
    def __init__(self):
        self._delivery_info = {}
        self._delivery_info_created = {}
        self._delivery_info_order = []

    def _prune_delivery_info(self, now):
        pass

    def _render_delivery_extra(self, extra, payload):
        return {}

    def _admission_source(self, route_name, profile, delivery_id):
        return SimpleNamespace()


async def _admit(stub, reason):
    return await WebhookAdapter._admit_agent_run(
        stub, {"text": "hi"}, "hi", "d1", 1.0,
        route_config={"deliver": "log"}, route_name="hook", profile=None, event_type="message",
    )


@pytest.mark.asyncio
async def test_profile_mismatch_is_refused_not_retried(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeStoreError("profile_mismatch")

    with patch("gateway.platforms.webhook_ingress.admit_producer", side_effect=boom):
        outcome = await _admit(_StubAdapter(), "profile_mismatch")
    assert outcome == "refused"


@pytest.mark.asyncio
async def test_runtime_draining_is_retry(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeStoreError("runtime_draining")

    with patch("gateway.platforms.webhook_ingress.admit_producer", side_effect=boom):
        outcome = await _admit(_StubAdapter(), "runtime_draining")
    assert outcome == "retry"
