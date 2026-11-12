"""``hermes browser export-signins``: the pipe Desktop reads the real-profile browser's cookies from."""

from __future__ import annotations

import io
import json
import threading

import pytest

from hermes_agent.hermes_cli import browser_preview_signins as signins

COOKIES = [
    {"name": "sid", "value": "s3cret", "domain": "app.test", "path": "/", "httpOnly": True, "secure": True,
     "session": False, "expires": 2e9, "sameSite": "Lax", "size": 9, "priority": "Medium"},
    {"name": "chips", "value": "x", "domain": "embed.test", "path": "/",
     "partitionKey": {"topLevelSite": "https://top.test", "hasCrossSiteAncestor": True}},
]


@pytest.fixture
def cdp_server():
    """A browser-level CDP endpoint that sends an event first, then answers Storage.getCookies."""
    from websockets.sync.server import serve

    def handler(ws):
        req = json.loads(ws.recv())
        ws.send(json.dumps({"method": "Target.targetCreated", "params": {}}))
        assert req["method"] == "Storage.getCookies"
        ws.send(json.dumps({"id": req["id"], "result": {"cookies": COOKIES}}))

    server = serve(handler, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}/devtools/browser/x"
    server.shutdown()


@pytest.fixture
def released(monkeypatch):
    calls = []
    monkeypatch.setattr("hermes_agent.tools.browser_session_real_profile.terminate_real_profile_browser",
                        lambda: calls.append(True))
    return calls


def _run(monkeypatch, *, tty=False):
    out = io.StringIO()
    out.isatty = lambda: tty  # type: ignore[method-assign]
    monkeypatch.setattr(signins.sys, "stdout", out)
    code = signins.run_export_signins(type("A", (), {"json": True})())
    return code, out.getvalue()


def test_exports_the_real_profile_browsers_cookies_to_stdout(monkeypatch, cdp_server, released):
    monkeypatch.setattr("hermes_agent.tools.browser_session_real_profile.real_profile_cdp", lambda: (cdp_server, None))

    code, out = _run(monkeypatch)

    verdict = json.loads(out)
    assert code == 0 and verdict["status"] == "ok"
    # Partitioned (CHIPS) cookies are skipped: set unpartitioned they would widen the site's scope.
    assert verdict["skipped"] == 1
    assert verdict["cookies"] == [{k: COOKIES[0][k] for k in signins._COOKIE_FIELDS}]
    assert released == [True]


def test_consent_off_reports_disabled_and_reads_nothing(monkeypatch, released):
    monkeypatch.setattr("hermes_agent.tools.browser_session_real_profile.real_profile_cdp", lambda: (None, None))

    code, out = _run(monkeypatch)

    assert (code, json.loads(out)) == (0, {"status": "disabled"})


def test_a_browser_that_cannot_start_is_an_error_not_an_empty_import(monkeypatch, released):
    monkeypatch.setattr("hermes_agent.tools.browser_session_real_profile.real_profile_cdp", lambda: (None, "profile locked"))

    code, out = _run(monkeypatch)

    assert (code, json.loads(out)) == (1, {"status": "error", "error": "profile locked"})


def test_refuses_to_print_cookie_values_to_a_terminal(monkeypatch, capsys):
    monkeypatch.setattr("hermes_agent.tools.browser_session_real_profile.real_profile_cdp",
                        lambda: pytest.fail("must not read cookies for a terminal"))

    code, out = _run(monkeypatch, tty=True)

    assert code == 2 and out == ""
    assert "Hermes Desktop" in capsys.readouterr().err
