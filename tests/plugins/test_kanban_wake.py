"""Regression coverage for the kanban-wake contrib plugin.

The plugin is loaded BY PATH from contrib/den-plugins/kanban-wake/__init__.py
(it is not an importable package), so these tests exercise the real shipped
source rather than a copy.
"""
from __future__ import annotations

import hashlib
import hmac
import http.server
import importlib.util
import json
import os
import threading
import time
import urllib.error
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_PATH = REPO_ROOT / "contrib" / "den-plugins" / "kanban-wake" / "__init__.py"

SECRET = "s3cr3t-kanban-wake"


def _load_plugin():
    spec = importlib.util.spec_from_file_location("kanban_wake_plugin_under_test", PLUGIN_PATH)
    assert spec and spec.loader, f"cannot load plugin from {PLUGIN_PATH}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin(monkeypatch):
    mod = _load_plugin()
    mod._last.clear()
    monkeypatch.setattr(mod, "_secret", lambda: SECRET)
    monkeypatch.setattr(mod, "_cfg", lambda: {"debounce_seconds": 60})
    return mod


def _expected_v2(ts: str, body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()


class _FakeHeaders:
    """Case-insensitive header mapping, like aiohttp's CIMultiDict."""

    def __init__(self, mapping):
        self._d = {k.lower(): v for k, v in mapping.items()}

    def get(self, name, default=""):
        return self._d.get(name.lower(), default)


class _FakeRequest:
    def __init__(self, headers):
        self.headers = _FakeHeaders(headers)
        self.match_info = {"route_name": "kanban-wake"}


# ------------------------------------------------------------------ helpers --

def _capture_post(plugin, monkeypatch):
    """Redirect urllib so _post's exact request can be inspected without a socket."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["headers"] = dict(req.headers)

        class _R:
            def read(self):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _R()

    monkeypatch.setattr(plugin.urllib.request, "urlopen", fake_urlopen)
    return captured


def _hdr(headers: dict, name: str) -> str:
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    raise KeyError(name)


# --------------------------------------------------------------------- D1 ----

def test_signature_header_is_bare_hex_digest(plugin, monkeypatch):
    captured = _capture_post(plugin, monkeypatch)
    # Rebind the plugin's own `time` reference (never the shared module) so the
    # timestamp is known.
    monkeypatch.setattr(
        plugin, "time", SimpleNamespace(time=lambda: 1_700_000_000.0), raising=True
    )

    plugin._post("hello board", "T-1", "blocked")

    body = captured["body"]
    ts = _hdr(captured["headers"], "X-Webhook-Timestamp")
    sig = _hdr(captured["headers"], "X-Webhook-Signature-V2")

    assert ts == "1700000000"
    assert sig == _expected_v2(ts, body)
    assert not sig.startswith("sha256=")
    assert sig == sig.lower()
    assert len(sig) == 64


# --------------------------------------------------------------------- D4 ----

def test_request_id_header_is_a_uuid4(plugin, monkeypatch):
    captured = _capture_post(plugin, monkeypatch)
    plugin._post("hello", "T-2", "blocked")
    rid = _hdr(captured["headers"], "X-Request-Id")
    assert uuid.UUID(rid).version == 4

    first = rid
    plugin._post("hello", "T-2", "blocked")
    assert _hdr(captured["headers"], "X-Request-Id") != first


# ------------------------------------------- D1 against the REAL verifier ----

def test_real_gateway_verifier_accepts_emitted_request(plugin, monkeypatch):
    from gateway.platforms.webhook import WebhookAdapter

    captured = _capture_post(plugin, monkeypatch)
    plugin._post("wake up", "T-3", "blocked")

    body = captured["body"]
    req = _FakeRequest(captured["headers"])

    class _DummySelf:
        _v1_signature_warned = set()

    assert (
        WebhookAdapter._validate_signature(_DummySelf(), req, body, SECRET) is True
    )


def test_real_gateway_verifier_rejects_tampered_body(plugin, monkeypatch):
    from gateway.platforms.webhook import WebhookAdapter

    captured = _capture_post(plugin, monkeypatch)
    plugin._post("wake up", "T-4", "blocked")

    req = _FakeRequest(captured["headers"])
    tampered = captured["body"] + b" "

    class _DummySelf:
        _v1_signature_warned = set()

    assert (
        WebhookAdapter._validate_signature(_DummySelf(), req, tampered, SECRET) is False
    )


# ------------------------------------------------- real socket round-trip ----

class _VerifyingHandler(http.server.BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        ts = self.headers.get("X-Webhook-Timestamp", "")
        sig = self.headers.get("X-Webhook-Signature-V2", "")
        expected = hmac.new(
            SECRET.encode(), ts.encode() + b"." + body, hashlib.sha256
        ).hexdigest()
        ok = bool(ts) and hmac.compare_digest(sig, expected)
        type(self).received.append(
            {
                "path": self.path,
                "body": body,
                "sig": sig,
                "request_id": self.headers.get("X-Request-ID", ""),
                "ok": ok,
            }
        )
        self.send_response(200 if ok else 401)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):  # silence
        pass


@pytest.fixture
def live_server():
    _VerifyingHandler.received = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _VerifyingHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv, _VerifyingHandler
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def test_live_endpoint_accepts_good_signature(plugin, monkeypatch, live_server):
    srv, handler = live_server
    host, port = srv.server_address[0], srv.server_address[1]
    url = f"http://{host}:{port}/webhooks/kanban-wake"
    monkeypatch.setattr(plugin, "_cfg", lambda: {"url": url, "debounce_seconds": 60})

    plugin._post("live wake", "T-5", "blocked")  # raises on non-2xx

    assert len(handler.received) == 1
    rec = handler.received[0]
    assert rec["ok"] is True
    assert rec["path"] == "/webhooks/kanban-wake"
    assert json.loads(rec["body"])["task_id"] == "T-5"
    assert uuid.UUID(rec["request_id"]).version == 4


def test_live_endpoint_rejects_corrupted_signature(plugin, monkeypatch, live_server):
    srv, handler = live_server
    host, port = srv.server_address[0], srv.server_address[1]
    url = f"http://{host}:{port}/webhooks/kanban-wake"
    monkeypatch.setattr(plugin, "_cfg", lambda: {"url": url, "debounce_seconds": 60})
    # Corrupt the signature exactly the way D1 did: prefix it with "sha256=".
    # Rebind only the PLUGIN's `hmac` name — mutating the shared hmac module
    # would corrupt the server thread's verification too.
    class _Prefixed:
        def __init__(self, inner):
            self._inner = inner

        def hexdigest(self):
            return "sha256=" + self._inner.hexdigest()

    shim = SimpleNamespace(new=lambda *a, **kw: _Prefixed(hmac.new(*a, **kw)))
    monkeypatch.setattr(plugin, "hmac", shim, raising=True)

    with pytest.raises(urllib.error.HTTPError) as ei:
        plugin._post("live wake", "T-6", "blocked")

    assert ei.value.code == 401
    assert handler.received[-1]["ok"] is False


# --------------------------------------------------------------------- D3 ----

def test_on_blocked_sends_once_per_task(plugin, monkeypatch):
    calls = []
    monkeypatch.setattr(plugin, "_post", lambda *a, **kw: calls.append(a))

    plugin._on_blocked(task_id="T-7", title="t", reason="r")
    assert len(calls) == 1

    plugin._on_blocked(task_id="T-7", title="t", reason="r")
    assert len(calls) == 1, "a delivered wake must stay debounced"


def test_failed_send_is_fail_open_and_immediately_retryable(plugin, monkeypatch):
    calls = []

    def boom(*a, **kw):
        calls.append(a)
        raise OSError("connection refused")

    monkeypatch.setattr(plugin, "_post", boom)

    assert plugin._on_blocked(task_id="T-8", title="t", reason="r") is None
    assert len(calls) == 1
    assert "blocked:T-8" not in plugin._last, "a failed send must not be debounced"

    # Immediate retry must NOT be suppressed.
    assert plugin._on_blocked(task_id="T-8", title="t", reason="r") is None
    assert len(calls) == 2

    # And once it finally succeeds, the debounce window engages.
    ok_calls = []
    monkeypatch.setattr(plugin, "_post", lambda *a, **kw: ok_calls.append(a))
    plugin._on_blocked(task_id="T-8", title="t", reason="r")
    plugin._on_blocked(task_id="T-8", title="t", reason="r")
    assert len(ok_calls) == 1
    assert plugin._last["blocked:T-8"] <= time.time()


# ---------------------------------------------------------------------------
# Packaging guards for the shipped plugin directory.
#
# These do not exercise plugin behaviour; they assert that the directory we
# actually ship stays loadable and that its manifest is not lying about what
# register() wires up.
# ---------------------------------------------------------------------------

PKG_DIR = PLUGIN_PATH.parent
MANIFEST_PATH = PKG_DIR / "plugin.yaml"


def test_plugin_package_has_no_self_referential_symlink():
    """No entry in the package may point at, or above, its own package root.

    A link back to the package dir makes any follow-links walk (packaging,
    copy, discovery, pip sdist) recurse forever. Derived from the real tree —
    any future entry that reintroduces the shape fails here.
    """
    pkg_root = PKG_DIR.resolve()
    offenders = []

    for dirpath, dirnames, filenames in os.walk(PKG_DIR, followlinks=False):
        for entry in list(dirnames) + list(filenames):
            candidate = Path(dirpath) / entry
            if not candidate.is_symlink():
                continue
            target = candidate.resolve()
            # Self-reference, or a link to an ancestor of the package root:
            # both make the package contain itself.
            if target == pkg_root or target in pkg_root.parents:
                offenders.append(f"{candidate} -> {target}")

    assert not offenders, (
        "self-referential / recursive symlink(s) inside the plugin package: "
        + ", ".join(offenders)
    )


def test_manifest_hook_declarations_match_registered_hooks():
    """provides_hooks must equal exactly what register() wires up.

    Both sides are derived: the manifest from plugin.yaml, the truth from a
    real register() call against a recording ctx. Fails in both directions —
    an undeclared hook and a declared-but-never-registered hook.
    """
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert "hooks" not in manifest, (
        "plugin.yaml uses the stale key 'hooks'; the loader reads "
        "'provides_hooks' (hermes_cli/plugins.py:4547), so a 'hooks:' block "
        "is silently ignored"
    )

    declared = manifest.get("provides_hooks") or []
    assert declared, "plugin.yaml declares no provides_hooks"

    registered: list[str] = []

    class _RecordingCtx:
        def register_hook(self, name, fn):
            assert callable(fn), f"hook {name!r} registered with a non-callable"
            registered.append(name)

        def __getattr__(self, _name):  # tolerate any other ctx surface
            return lambda *a, **kw: None

    _load_plugin().register(_RecordingCtx())

    assert set(declared) == set(registered), (
        "plugin.yaml provides_hooks and register() disagree: "
        f"declared-but-not-registered={sorted(set(declared) - set(registered))}, "
        f"registered-but-not-declared={sorted(set(registered) - set(declared))}"
    )
