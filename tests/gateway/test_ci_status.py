import json
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.dev_control.ci_status import ci_ship_gate, fetch_ci_status
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_ci_status_combines_statuses_and_check_runs():
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/status"):
            return FakeResponse({
                "statuses": [{"context": "legacy-status", "state": "success", "target_url": "https://ci/status"}],
                "repository": {"html_url": "https://github.com/Felippen/Oryn"},
            })
        return FakeResponse({
            "check_runs": [
                {"name": "oryn-workspace-ci", "status": "completed", "conclusion": "success", "html_url": "https://ci/ok"},
                {"name": "shell-ci", "status": "completed", "conclusion": "failure", "html_url": "https://ci/fail"},
            ]
        })

    with patch("gateway.dev_control.ci_status._github_token", return_value="token"):
        result = fetch_ci_status(repo="Felippen/Oryn", ref="main", opener=fake_urlopen)

    assert result["state"] == "failure"
    assert result["total"] == 3
    assert result["failing"][0]["name"] == "shell-ci"
    assert result["ship_gate"]["status"] == "blocked_by_ci"
    assert any(url.endswith("/check-runs") for url in calls)


def test_ci_status_fail_opens_to_unknown_on_api_error():
    def failing_urlopen(request, timeout):
        raise RuntimeError("network down")

    result = fetch_ci_status(repo="Felippen/Oryn", ref="main", opener=failing_urlopen)

    assert result["state"] == "unknown"
    assert result["warnings"]
    assert result["ship_gate"]["blocks_ship"] is False


def test_ci_ship_gate_blocks_failure_and_pending_but_not_unknown():
    assert ci_ship_gate({"state": "failure"})["blocks_ship"] is True
    assert ci_ship_gate({"state": "pending"})["blocks_ship"] is True
    assert ci_ship_gate({"state": "success"})["blocks_ship"] is False
    assert ci_ship_gate({"state": "unknown"})["blocks_ship"] is False


@pytest.mark.asyncio
async def test_ci_status_api_returns_fail_open_payload(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app.router.add_get("/v1/dev/ci-status", adapter._handle_dev_ci_status)
    monkeypatch.setattr(
        "gateway.platforms.api_server.fetch_ci_status",
        lambda repo, ref: {
            "ok": True,
            "object": "hermes.dev_ci_status",
            "repo": repo,
            "ref": ref,
            "state": "unknown",
            "total": 0,
            "failing": [],
            "warnings": ["fake"],
            "ship_gate": {"status": "ci_unknown", "blocks_ship": False},
        },
    )

    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(
            "/v1/dev/ci-status?repo=Felippen/Oryn&ref=main",
            headers={"Authorization": "Bearer sk-secret"},
        )
        data = await response.json()

    assert response.status == 200
    assert data["repo"] == "Felippen/Oryn"
    assert data["state"] == "unknown"
