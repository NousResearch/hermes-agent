"""Real expiry consumers must count suspend, without moving execution clocks (#126516)."""
import asyncio
import json
import shlex
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest
import hermes_time


def _advance_suspend(monkeypatch):
    # This host's native expiry clock advances while Python's execution clock
    # does not. Patch the clock input, never the cache or restoration decision.
    clock = hermes_time.time.clock_gettime
    offset = [0.0]
    monkeypatch.setattr(hermes_time.time, "clock_gettime", lambda clock_id: clock(clock_id) + offset[0])
    return offset


@pytest.mark.platforms("linux", "macos")
def test_key_command_and_search_cache_expire_after_suspend(tmp_path, monkeypatch):
    from agent.command_token_source import CommandTokenSource
    from tools.web_result_cache import SearchMemo

    offset = _advance_suspend(monkeypatch)
    counter = tmp_path / "mints"
    script = tmp_path / "mint.py"
    script.write_text(
        "import json\nfrom pathlib import Path\n"
        f"p=Path({str(counter)!r})\n"
        "n=int(p.read_text())+1 if p.exists() else 1\np.write_text(str(n))\n"
        "print(json.dumps({'access_token':str(n), 'expires_in':3600}))\n"
    )
    source = CommandTokenSource(f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}")
    memo = SearchMemo()
    response = {"success": True, "data": {"web": []}}
    assert source() == source() == "1"
    memo.store("fixture", "query", 5, response)
    assert memo.lookup("fixture", "query", 5) == response
    offset[0] += 8 * 3600
    assert source() == "2"
    assert counter.read_text() == "2"
    assert memo.lookup("fixture", "query", 5) is None


@pytest.mark.platforms("linux", "macos")
def test_provider_reset_and_teams_tokens_expire_after_suspend(monkeypatch):
    from agent.error_classifier import FailoverReason
    from tests.agent.test_provider_reset_cooldown import _agent_with_one_fallback, _fallback_client
    from tests.gateway.test_teams import TestTeamsBotFrameworkAttachments, _teams_mod

    offset = _advance_suspend(monkeypatch)
    agent = _agent_with_one_fallback()
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "gpt-5.5")):
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit,
                                          reset_at=hermes_time.time.time() + 3600)
    assert agent._restore_primary_runtime() is False
    calls = []

    class TokenHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({"access_token": f"token-{len(calls)}", "expires_in": 3600}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), TokenHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(_teams_mod, "_bf_token_request", lambda *args:
                        (f"http://127.0.0.1:{server.server_port}/token", {"grant_type": "client_credentials"}))
    adapter = TestTeamsBotFrameworkAttachments()._make_adapter()

    async def check():
        assert await adapter._get_botframework_token() == "token-1"
        assert await adapter._get_botframework_token() == "token-1"
        offset[0] += 8 * 3600
        assert await adapter._get_botframework_token() == "token-2"
        assert len(calls) == 2
        assert agent._restore_primary_runtime() is True

    try:
        asyncio.run(check())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        agent.close()
