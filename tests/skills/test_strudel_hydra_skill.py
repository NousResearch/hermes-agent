"""Regression tests for the `strudel-hydra` optional skill.

Standard library + pytest only; no browser, no sockets, no network. Covers the
security gate added for review: the server's write endpoints (`/push`,
`/telemetry`) require a same-origin JSON request bearing the per-run capability
token, the bind host is restricted to loopback unless opted out, and the client
discovers the token automatically. The HTTP handler is driven directly with a
fake request (no socket); `urllib` in the client is mocked.
"""
from __future__ import annotations

import email.message
import io
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SKILL = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "strudel-hydra"
)
SCRIPTS = SKILL / "scripts"
TEMPLATE = SKILL / "templates" / "page.html"


@pytest.fixture(scope="module", autouse=True)
def _add_scripts_to_path():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield
    finally:
        sys.path.remove(str(SCRIPTS))


# --- sh_server: pure security helpers -------------------------------------

def test_loopback_classification():
    import sh_server as m

    assert m.is_loopback_host("127.0.0.1")
    assert m.is_loopback_host("localhost")
    assert m.is_loopback_host("::1")
    assert not m.is_loopback_host("0.0.0.0")
    assert not m.is_loopback_host("192.168.1.5")


def test_validate_bind_host_rejects_non_loopback_without_optin():
    import sh_server as m

    assert m.validate_bind_host("127.0.0.1", allow_remote=False) == "127.0.0.1"
    with pytest.raises(SystemExit):
        m.validate_bind_host("0.0.0.0", allow_remote=False)
    # explicit opt-in permits it
    assert m.validate_bind_host("0.0.0.0", allow_remote=True) == "0.0.0.0"


def test_origin_allowed_rejects_cross_site_only():
    import sh_server as m

    # no Origin (curl / urllib) is allowed; the token still gates the write
    assert m.origin_allowed(None, "127.0.0.1:8765")
    assert m.origin_allowed("", "127.0.0.1:8765")
    # same-origin browser request matches
    assert m.origin_allowed("http://127.0.0.1:8765", "127.0.0.1:8765")
    assert m.origin_allowed("https://127.0.0.1:8765", "127.0.0.1:8765")
    # a different site is refused
    assert not m.origin_allowed("http://evil.example", "127.0.0.1:8765")
    assert not m.origin_allowed("http://127.0.0.1:9999", "127.0.0.1:8765")


def test_content_type_must_be_json():
    import sh_server as m

    assert m.content_type_is_json("application/json")
    assert m.content_type_is_json("application/json; charset=utf-8")
    assert not m.content_type_is_json("text/plain")
    assert not m.content_type_is_json(None)


def test_token_file_path_is_per_port():
    import sh_server as m

    assert "8765" in m.token_file_path(8765).name
    assert m.token_file_path(9000) != m.token_file_path(8765)


def test_template_carries_token_placeholder_and_inject_replaces_it():
    import sh_server as m

    html = TEMPLATE.read_text(encoding="utf-8")
    assert m.TOKEN_PLACEHOLDER in html                 # the page ships a slot
    injected = m.inject_token(html, "s3cret-token")
    assert m.TOKEN_PLACEHOLDER not in injected
    assert "s3cret-token" in injected


# --- sh_server: the handler's write gate, driven without a socket ----------

def _make_handler(server_mod, *, path, headers, body=b"", token="the-real-token"):
    """Build a Handler bound to a fake request (no socket) and capture _send."""
    h = server_mod.Handler.__new__(server_mod.Handler)
    h.token = token
    h.path = path
    msg = email.message.Message()
    for k, v in headers.items():
        msg[k] = v
    h.headers = msg
    h.rfile = io.BytesIO(body)
    h.wfile = io.BytesIO()
    sent = {}

    def _send(code, ctype, resp=b""):
        sent["code"] = code
        sent["ctype"] = ctype
        sent["body"] = resp

    h._send = _send
    return h, sent


def _json_headers(token="the-real-token", origin=None, ctype="application/json", body=b""):
    hdrs = {"Content-Length": str(len(body))}
    if ctype is not None:
        hdrs["Content-Type"] = ctype
    if token is not None:
        hdrs["X-SH-Token"] = token
    if origin is not None:
        hdrs["Origin"] = origin
    return hdrs


def test_push_without_token_is_rejected():
    import sh_server as m

    body = b'{"audio":"note(\\"c3\\")"}'
    h, sent = _make_handler(m, path="/push", headers=_json_headers(token=None, body=body), body=body)
    h.do_POST()
    assert sent["code"] == 401


def test_push_with_wrong_token_is_rejected():
    import sh_server as m

    body = b'{"audio":"x"}'
    h, sent = _make_handler(m, path="/push", headers=_json_headers(token="nope", body=body), body=body)
    h.do_POST()
    assert sent["code"] == 401


def test_push_from_cross_origin_is_rejected_even_with_token():
    import sh_server as m

    body = b'{"audio":"x"}'
    hdrs = _json_headers(origin="http://evil.example", body=body)
    h, sent = _make_handler(m, path="/push", headers=hdrs, body=body)
    h.do_POST()
    assert sent["code"] == 403


def test_push_with_non_json_content_type_is_rejected():
    import sh_server as m

    body = b'{"audio":"x"}'
    hdrs = _json_headers(ctype="text/plain", body=body)
    h, sent = _make_handler(m, path="/push", headers=hdrs, body=body)
    h.do_POST()
    assert sent["code"] == 415


def test_push_non_object_body_is_rejected_after_auth():
    import sh_server as m

    body = b'["not","an","object"]'
    h, sent = _make_handler(m, path="/push", headers=_json_headers(body=body), body=body)
    h.do_POST()
    assert sent["code"] == 400


def test_valid_push_is_accepted_and_fanned_out():
    import sh_server as m

    q = m.broker.subscribe()  # stand in for a connected browser
    try:
        payload = {"audio": "note(\"c3\")", "label": "riff"}
        body = json.dumps(payload).encode()
        h, sent = _make_handler(m, path="/push", headers=_json_headers(body=body), body=body)
        h.do_POST()
        assert sent["code"] == 200
        assert json.loads(sent["body"])["ok"] is True
        delivered = json.loads(q.get_nowait())
        assert delivered["label"] == "riff"
    finally:
        m.broker.unsubscribe(q)


def test_valid_telemetry_post_is_recorded():
    import sh_server as m

    body = json.dumps({"level": 0.4, "bands": [0.1, 0.2], "centroid": 0.5}).encode()
    h, sent = _make_handler(m, path="/telemetry", headers=_json_headers(body=body), body=body)
    h.do_POST()
    assert sent["code"] == 200
    assert m.telemetry.get()["data"]["level"] == 0.4


def test_served_page_has_token_injected_and_placeholder_gone():
    import sh_server as m

    h, sent = _make_handler(m, path="/", headers={}, token="page-token-xyz")
    h.do_GET()
    assert sent["code"] == 200
    served = sent["body"].decode()
    assert "page-token-xyz" in served
    assert m.TOKEN_PLACEHOLDER not in served


# --- sh_client: token discovery + transport --------------------------------

def test_resolve_token_prefers_env(monkeypatch):
    import sh_client as c

    monkeypatch.setenv("SH_TOKEN", "from-env")
    assert c.resolve_token("http://127.0.0.1:8765") == "from-env"


def test_resolve_token_reads_per_port_file(monkeypatch, tmp_path):
    import sh_client as c

    monkeypatch.delenv("SH_TOKEN", raising=False)
    monkeypatch.setattr(c, "token_file_path", lambda port: tmp_path / f"tok-{port}")
    (tmp_path / "tok-8765").write_text("file-token", encoding="utf-8")
    assert c.resolve_token("http://127.0.0.1:8765") == "file-token"


def test_resolve_token_none_when_absent(monkeypatch, tmp_path):
    import sh_client as c

    monkeypatch.delenv("SH_TOKEN", raising=False)
    monkeypatch.setattr(c, "token_file_path", lambda port: tmp_path / "missing")
    assert c.resolve_token("http://127.0.0.1:8765") is None


def test_post_attaches_token_header(monkeypatch):
    import sh_client as c

    captured = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout=5):
        captured["headers"] = dict(req.header_items())
        captured["data"] = req.data
        return FakeResp()

    monkeypatch.setattr(c.urllib.request, "urlopen", fake_urlopen)
    out = c.post("http://127.0.0.1:8765", "/push", {"audio": "x"}, token="explicit-token")
    assert out == {"ok": True}
    # header names are title-cased by urllib
    assert captured["headers"].get("X-sh-token") == "explicit-token"
    assert captured["headers"].get("Content-type") == "application/json"
    assert json.loads(captured["data"]) == {"audio": "x"}


def test_push_set_builds_payload_and_passes_token(monkeypatch):
    import sh_client as c

    calls = {}

    def fake_post(base, path, data, timeout=5, token=None):
        calls.update(base=base, path=path, data=data, token=token)
        return {"ok": True}

    monkeypatch.setattr(c, "post", fake_post)
    c.push_set("http://h", audio="A", visual="V", label="riff", token="tk")
    assert calls["path"] == "/push"
    assert calls["token"] == "tk"
    assert calls["data"] == {"label": "riff", "audio": "A", "visual": "V"}


# --- sh_examples: standalone export ---------------------------------------

def test_export_writes_standalone_page(tmp_path):
    import sh_examples as ex

    out = tmp_path / "acid.html"
    ex.export("acid", str(out))
    html = out.read_text(encoding="utf-8")
    assert "window.__SET__" in html
    assert ex.SETS["acid"]["label"] in html
    # exported page is self-contained: it carries a baked set, not an SSE hookup
    assert "__SET__ = {" in html.replace("window.", "")
