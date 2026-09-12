"""Exercise the real xAI pool and proxy route using temporary auth state."""
import asyncio
import json
import time
from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")
from aiohttp.test_utils import make_mocked_request

from hermes_cli.proxy import cli, server
from hermes_cli.proxy.adapters.xai import XAIGrokAdapter


@pytest.fixture
def auth_store(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def write(*, code=429, reason=None, status="exhausted", delay=900, empty=False):
        now = time.time()
        entry = {
            "id": "test-key", "label": "test-key", "auth_type": "oauth",
            "priority": 0, "source": "manual:xai_pkce",
            "access_token": "synthetic-test-bearer", "expires_at": now + 3600,
            "last_status": status, "last_status_at": now,
            "last_error_code": code, "last_error_reset_at": now + delay,
        }
        if reason:
            entry["failure_reason"] = reason
        (home / "auth.json").write_text(json.dumps({
            "version": 1, "credential_pool": {"xai-oauth": [] if empty else [entry]},
        }), encoding="utf-8")
        return home
    return write


def request(adapter):
    async def run():
        app = server.create_app(adapter)
        req = make_mocked_request("POST", "/v1/chat/completions", app=app)
        match = await app.router.resolve(req)
        req._match_info = match
        response = await match.handler(req)
        return response.status, dict(response.headers), json.loads(response.body)
    return asyncio.run(run())


@pytest.mark.parametrize("code,reason", [(429, None), (503, None), (403, "rate_limit")])
def test_cooling_credentials_return_retry_after(auth_store, code, reason):
    auth_store(code=code, reason=reason)
    status, headers, body = request(XAIGrokAdapter())
    assert status == 429
    # Respect the actual pool deadline, including waits longer than five minutes.
    assert 895 <= int(headers["Retry-After"]) <= 900
    assert body["error"]["code"] == "upstream_rate_limited"


@pytest.mark.parametrize("entry", [
    {"empty": True}, {"code": 401}, {"code": 401, "status": "dead"},
    {"code": 403, "reason": "billing"}, {"code": 403},
])
def test_auth_failures_are_not_reported_as_retryable(auth_store, entry):
    auth_store(**entry)
    status, headers, body = request(XAIGrokAdapter())
    assert status == 401
    assert "Retry-After" not in headers
    assert body["error"]["code"] == "upstream_auth_failed"


def test_proxy_start_accepts_cooling_credentials(auth_store, monkeypatch):
    auth_store()
    started = []

    async def run_server(adapter, *, host, port):
        started.append(adapter.name)

    monkeypatch.setattr(cli, "run_server", run_server)
    assert cli.cmd_proxy_start(Namespace(provider="xai", host="127.0.0.1", port=8645)) == 0
    assert started == ["xai"]


def test_cooldown_expiry_restores_normal_credentials(auth_store):
    auth_store(delay=-1)
    adapter = XAIGrokAdapter()
    assert adapter.get_credential().bearer == "synthetic-test-bearer"


def test_proxy_status_describes_cooldown(auth_store, capsys, monkeypatch):
    auth_store()
    monkeypatch.setattr(cli, "ADAPTERS", {"xai": XAIGrokAdapter})
    assert cli.cmd_proxy_status(Namespace()) == 0
    output = capsys.readouterr().out
    assert "cooling down" in output
    assert "not logged in" not in output


@pytest.mark.parametrize("peer_code,peer_status,reason,expected_delay", [
    (401, "exhausted", None, 900),
    (403, "exhausted", "billing", 900),
    (401, "dead", None, 900),
    (503, "exhausted", None, 60),
])
def test_mixed_pool_retries_at_the_first_transient_recovery(
    auth_store, peer_code, peer_status, reason, expected_delay,
):
    home = auth_store(delay=900)
    path = home / "auth.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    entries = state["credential_pool"]["xai-oauth"]
    peer = dict(entries[0], id="peer-key", label="peer-key", last_error_code=peer_code,
                last_status=peer_status, last_error_reset_at=time.time() + 60)
    if reason:
        peer["failure_reason"] = reason
    entries.append(peer)
    path.write_text(json.dumps(state), encoding="utf-8")

    status, headers, body = request(XAIGrokAdapter())
    assert status == 429
    # An earlier auth/billing deadline cannot promise the retryable credential is ready.
    assert expected_delay - 5 <= int(headers["Retry-After"]) <= expected_delay
    assert body["error"]["code"] == "upstream_rate_limited"
