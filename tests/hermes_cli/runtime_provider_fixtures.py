"""Synthetic credentials and offline catalog transport for runtime-provider tests."""

import base64
import json
import time

import pytest


@pytest.fixture(autouse=True)
def offline_provider_catalog(monkeypatch):
    monkeypatch.setattr("agent.models_dev._fetch_models_dev_from_network", lambda **kw: ({}, None))


def _fake_invoke_jwt(ttl_seconds=3600):
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "scope": "inference:invoke",
                "exp": int(time.time() + ttl_seconds),
            }
        ).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.sig"
