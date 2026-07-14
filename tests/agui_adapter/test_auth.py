import pytest
from agui_adapter import auth


def test_loopback_without_token_is_allowed():
    # 127.0.0.1 + no token → no raise (OS boundary is the auth)
    auth.require_token_or_refuse("127.0.0.1", None)


def test_non_loopback_without_token_refuses_to_start():
    with pytest.raises(SystemExit):
        auth.require_token_or_refuse("0.0.0.0", None)


def test_non_loopback_with_weak_token_refuses_to_start():
    with pytest.raises(SystemExit):
        auth.require_token_or_refuse("0.0.0.0", "short")


def test_non_loopback_with_strong_token_is_allowed():
    auth.require_token_or_refuse("0.0.0.0", "x" * 32)


from fastapi.testclient import TestClient
from agui_adapter.server import create_app


def _run_body():
    return {"threadId": "t1", "runId": "r1", "state": {}, "messages": [],
            "tools": [], "context": [], "forwardedProps": {}}


def test_health_is_open_even_with_token():
    client = TestClient(create_app(session_token="x" * 32), base_url="http://127.0.0.1")
    assert client.get("/health").status_code == 200


def test_post_without_token_is_401_when_token_configured():
    client = TestClient(create_app(session_token="x" * 32), base_url="http://127.0.0.1")
    r = client.post("/", json=_run_body())
    assert r.status_code == 401


def _stub_agent(monkeypatch):
    """Stub the agent so an authorized POST completes instantly and offline,
    instead of driving a full (slow, network-dependent) agent turn."""
    import agui_adapter.server as server

    class _Fake:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def run_conversation(self, *a, **k):
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _Fake())


def test_post_with_header_token_is_authorized(monkeypatch):
    _stub_agent(monkeypatch)
    client = TestClient(create_app(session_token="x" * 32), base_url="http://127.0.0.1")
    r = client.post("/", json=_run_body(), headers={"X-Hermes-Session-Token": "x" * 32})
    # 200 proves the request passed the auth/Host/content-type gating and began
    # streaming (the endpoint returns a StreamingResponse, whose status is fixed
    # at 200 once streaming starts — a worker-level failure would surface as a
    # RUN_ERROR inside the SSE body, not as a non-200 code). That gating pass is
    # exactly what this auth test asserts.
    assert r.status_code == 200


def test_post_with_query_token_is_authorized(monkeypatch):
    _stub_agent(monkeypatch)
    client = TestClient(create_app(session_token="x" * 32), base_url="http://127.0.0.1")
    r = client.post("/?token=" + "x" * 32, json=_run_body())
    assert r.status_code == 200


def test_no_token_configured_allows_loopback_post(monkeypatch):
    _stub_agent(monkeypatch)
    client = TestClient(create_app(session_token=None), base_url="http://127.0.0.1")
    r = client.post("/", json=_run_body())
    assert r.status_code == 200


def test_post_with_wrong_token_is_401():
    # Loopback bind WITH a configured token: a mismatched token must still be
    # rejected via the hmac.compare_digest mismatch path (not the "no token
    # supplied" path), for both the header and query-param carriers.
    client = TestClient(create_app(session_token="x" * 32), base_url="http://127.0.0.1")
    r = client.post("/", json=_run_body(), headers={"X-Hermes-Session-Token": "y" * 32})
    assert r.status_code == 401
    r2 = client.post("/?token=" + "y" * 32, json=_run_body())
    assert r2.status_code == 401


def test_bad_host_header_rejected_on_loopback_bind():
    client = TestClient(create_app(), base_url="http://evil.example.com")
    r = client.get("/health")
    assert r.status_code == 400


def test_host_accepted_loopback_bind():
    assert auth.host_accepted("127.0.0.1:8000", "127.0.0.1") is True
    assert auth.host_accepted("localhost", "127.0.0.1") is True
    assert auth.host_accepted("evil.example.com", "127.0.0.1") is False
    assert auth.host_accepted("", "127.0.0.1") is False


def test_host_accepted_wildcard_bind_accepts_any():
    assert auth.host_accepted("anything.example.com", "0.0.0.0") is True


def test_host_accepted_explicit_host_requires_exact_match():
    assert auth.host_accepted("myhost:9119", "myhost") is True
    assert auth.host_accepted("other", "myhost") is False


def test_host_accepted_ipv6_loopback():
    assert auth.host_accepted("[::1]:8000", "::1") is True


def test_host_accepted_bare_ipv6_loopback():
    # Bare (unbracketed) IPv6 literal — the deliberate divergence from the
    # mirrored sibling (hermes_cli/web_server.py::_is_accepted_host), which
    # mangles a bare "::1" via its plain rsplit(":", 1) port-stripping (it
    # only special-cases the bracketed form), reducing it to ":" and always
    # rejecting it. This module adds an explicit branch for host headers with
    # more than one colon so a bare "::1" is recognized as the loopback
    # literal it is.
    assert auth.host_accepted("::1", "::1") is True
    assert auth.host_accepted("::1", "127.0.0.1") is True


# --- create_app() fail-closed wiring (not just the standalone helper) -------

def test_create_app_refuses_non_loopback_without_token():
    # The embedder "refuse to start" guard: create_app must couple bound_host
    # into require_token_or_refuse, so a network-accessible bind with no token
    # raises SystemExit at construction (before any request is served).
    with pytest.raises(SystemExit):
        create_app(session_token=None, bound_host="0.0.0.0")


def test_create_app_builds_non_loopback_with_strong_token():
    # Same wiring, satisfied: a strong token on a network bind builds cleanly.
    app = create_app(session_token="x" * 32, bound_host="0.0.0.0")
    assert app is not None


def test_bad_host_header_rejected_on_post_route():
    # The DNS-rebind Host guard must also fire on the real agent route (POST /),
    # not only GET /health, and must precede token/CSRF checks: with no token
    # configured, a bad Host still yields 400 (not a 200 dispatch).
    client = TestClient(create_app(), base_url="http://evil.example.com")
    r = client.post("/", json=_run_body())
    assert r.status_code == 400


# --- entry.main(): the authoritative fail-closed guard ----------------------

def test_entry_main_refuses_non_loopback_without_token(monkeypatch):
    import agui_adapter.entry as entry

    called = {}
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: called.setdefault("run", True))
    monkeypatch.setenv("HERMES_AGUI_HOST", "0.0.0.0")
    monkeypatch.delenv("HERMES_AGUI_SESSION_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        entry.main()
    assert "run" not in called  # refused before ever reaching uvicorn.run


def test_entry_main_passes_same_host_to_guard_and_uvicorn(monkeypatch):
    # main() feeds the host it validated to uvicorn.run — asserts uvicorn is
    # bound to the network host the guard accepted (with a token). The
    # complementary fail-closed direction (guard sees loopback while binding
    # wide, no token) is covered by test_entry_main_refuses_non_loopback_...
    # above; main() uses a single `host` local for all call sites, so the two
    # cannot decouple without a rewrite.
    import agui_adapter.entry as entry

    captured = {}

    def _fake_run(app, **k):
        captured["app"] = app
        captured.update(k)

    monkeypatch.setattr("uvicorn.run", _fake_run)
    monkeypatch.setenv("HERMES_AGUI_HOST", "0.0.0.0")
    monkeypatch.setenv("HERMES_AGUI_SESSION_TOKEN", "x" * 32)
    entry.main()
    assert captured["host"] == "0.0.0.0"
    assert captured["app"] is not None  # create_app(bound_host="0.0.0.0", token) built


# --- _usable(): placeholder/entropy floor, including the ImportError fallback

def test_usable_rejects_placeholders_and_accepts_strong():
    assert auth._usable(None) is False
    assert auth._usable("short") is False
    # A known placeholder that is >=16 chars is rejected for BEING a
    # placeholder (not merely for length) on the primary has_usable_secret
    # path — "your_api_key_here" (17 chars) is in hermes_cli's placeholder set.
    assert auth._usable("your_api_key_here") is False
    assert auth._usable("x" * 32) is True


def test_create_app_neutralizes_interrupt_preempting_env(monkeypatch):
    # Inherited HERMES_GATEWAY_SESSION / HERMES_EXEC_ASK route dangerous-command
    # approval into tools.approval's gateway/ask branch (read from os.environ,
    # not the interactive contextvar), which would silently disable the native
    # interrupt-approval flow — the same failure class as setting a session
    # platform. create_app() must clear them for the AG-UI process.
    import os

    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    create_app()  # loopback default; no token needed
    assert "HERMES_GATEWAY_SESSION" not in os.environ
    assert "HERMES_EXEC_ASK" not in os.environ


def test_usable_import_error_fallback_still_rejects_placeholders(monkeypatch):
    # Force the `from hermes_cli.auth import has_usable_secret` ImportError
    # branch by shadowing the module with one lacking that symbol, and confirm
    # the reduced inline check still rejects placeholders / short tokens and
    # accepts a strong one (it must not silently fail open).
    import sys
    import types

    monkeypatch.setitem(sys.modules, "hermes_cli.auth", types.ModuleType("hermes_cli.auth"))
    assert auth._usable("changeme00000000") is False
    assert auth._usable("your_api_key_here") is False
    assert auth._usable("short") is False
    assert auth._usable("x" * 32) is True
