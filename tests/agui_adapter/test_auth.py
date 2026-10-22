import pytest
from agui_adapter import auth

TEST_TOKEN = "u5D9mW4xR7qT2nK8vH6cB3sF0aL1pZ4e"


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
    auth.require_token_or_refuse("0.0.0.0", TEST_TOKEN)


from fastapi.testclient import TestClient
from agui_adapter.server import create_app


def _run_body():
    return {
        "threadId": "t1",
        "runId": "r1",
        "state": {},
        "messages": [],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }


def test_health_is_open_even_with_token():
    client = TestClient(
        create_app(session_token=TEST_TOKEN), base_url="http://127.0.0.1"
    )
    assert client.get("/health").status_code == 200


def test_post_without_token_is_401_when_token_configured():
    client = TestClient(
        create_app(session_token=TEST_TOKEN), base_url="http://127.0.0.1"
    )
    r = client.post("/", json=_run_body())
    assert r.status_code == 401


def _stub_agent(monkeypatch):
    """Stub the agent so an authorized POST completes instantly and offline,
    instead of driving a full (slow, network-dependent) agent turn."""
    from agui_adapter import server

    class _Fake:
        def close(self):
            pass

        def __init__(self):
            for a in (
                "stream_delta_callback",
                "reasoning_callback",
                "tool_progress_callback",
                "step_callback",
                "thinking_callback",
            ):
                setattr(self, a, None)

        def run_conversation(self, *a, **k):
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _Fake())


def test_post_with_header_token_is_authorized(monkeypatch):
    _stub_agent(monkeypatch)
    client = TestClient(
        create_app(session_token=TEST_TOKEN), base_url="http://127.0.0.1"
    )
    r = client.post(
        "/", json=_run_body(), headers={"X-Hermes-Session-Token": TEST_TOKEN}
    )
    # 200 proves the request passed the auth/Host/content-type gating and began
    # streaming (the endpoint returns a StreamingResponse, whose status is fixed
    # at 200 once streaming starts — a worker-level failure would surface as a
    # RUN_ERROR inside the SSE body, not as a non-200 code). That gating pass is
    # exactly what this auth test asserts.
    assert r.status_code == 200


def test_post_with_query_token_is_authorized(monkeypatch):
    _stub_agent(monkeypatch)
    client = TestClient(
        create_app(session_token=TEST_TOKEN), base_url="http://127.0.0.1"
    )
    r = client.post("/?token=" + TEST_TOKEN, json=_run_body())
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
    client = TestClient(
        create_app(session_token=TEST_TOKEN), base_url="http://127.0.0.1"
    )
    r = client.post("/", json=_run_body(), headers={"X-Hermes-Session-Token": "y" * 32})
    assert r.status_code == 401
    r2 = client.post("/?token=" + "y" * 32, json=_run_body())
    assert r2.status_code == 401


def test_bad_host_header_rejected_on_loopback_bind():
    client = TestClient(create_app(), base_url="http://evil.example.com")
    r = client.get("/health")
    assert r.status_code == 400


@pytest.mark.parametrize(
    "bound_host,authority,status",
    [
        ("::1", "[0:0:0:0:0:0:0:1]:0", 200),
        ("0:0:0:0:0:0:0:1", "::1", 200),
        ("127.0.0.1", "[::ffff:127.0.0.1]:65535", 200),
        ("::ffff:127.0.0.1", "localhost", 200),
        ("127.0.0.1", "localhost:00065535", 200),
        ("127.0.0.1", "127.0.0.2", 200),
        ("2001:db8::1", "[2001:0db8:0:0:0:0:0:1]:8000", 200),
        ("2001:db8::1", "[2001:db8::2]", 400),
        ("0:0:0:0:0:0:0:0", "example.com:8000", 200),
        ("127.0.0.1", "[::1]evil.example", 400),
        ("127.0.0.1", "[::1]:garbage", 400),
        ("127.0.0.1", "localhost:bad", 400),
        ("0.0.0.0", "[::1]evil.example", 400),
        ("0.0.0.0", "localhost:", 400),
        ("0.0.0.0", "localhost:65536", 400),
        ("0.0.0.0", "localhost:-1", 400),
        ("0.0.0.0", "[::1", 400),
        ("0.0.0.0", "[127.0.0.1]", 400),
        ("0.0.0.0", ":8000", 400),
        ("0.0.0.0", "localhost/path?x#y", 400),
        ("0.0.0.0", "user@localhost", 400),
        ("0.0.0.0", " localhost ", 400),
        ("0.0.0.0", "local\x00host", 400),
        ("::", "bad::ip", 400),
    ],
)
def test_host_authority_middleware(bound_host, authority, status):
    token = TEST_TOKEN if auth.is_network_accessible(bound_host) else None
    client = TestClient(create_app(bound_host=bound_host, session_token=token))
    assert client.get("/health", headers={"Host": authority}).status_code == status
    if token:
        assert client.post(
            "/", json=_run_body(), headers={"Host": authority}
        ).status_code == (401 if status == 200 else 400)


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
    app = create_app(session_token=TEST_TOKEN, bound_host="0.0.0.0")
    assert app is not None


def test_bad_host_header_rejected_on_post_route():
    # The DNS-rebind Host guard must also fire on the real agent route (POST /),
    # not only GET /health, and must precede token/CSRF checks: with no token
    # configured, a bad Host still yields 400 (not a 200 dispatch).
    client = TestClient(create_app(), base_url="http://evil.example.com")
    r = client.post("/", json=_run_body())
    assert r.status_code == 400


# --- entry.main(): the authoritative fail-closed guard ----------------------


def test_setup_logging_redacts_credentials_in_messages_and_tracebacks(monkeypatch):
    import io
    import logging
    from agui_adapter import entry

    stream = io.StringIO()
    monkeypatch.setattr(entry.sys, "stderr", stream)
    entry._setup_logging()
    logger = logging.getLogger("agui_adapter.server")
    secret = "sk-demo12345678901234567890"
    logger.warning("request credential: %s", secret)
    try:
        raise RuntimeError(f"upstream credential: {secret}")
    except RuntimeError:
        logger.exception("run failed")
    output = stream.getvalue()
    assert secret not in output
    assert "request credential:" in output
    assert "run failed" in output
    assert "Traceback (most recent call last):" in output
    assert "RuntimeError: upstream credential:" in output


def test_entry_main_refuses_non_loopback_without_token(monkeypatch):
    from agui_adapter import entry

    called = {}
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: called.setdefault("run", True))
    monkeypatch.delenv("HERMES_AGUI_SESSION_TOKEN", raising=False)
    # host comes from the --host flag (config.yaml is the other source); the
    # token is the only environment input and is absent here.
    with pytest.raises(SystemExit):
        entry.main(host="0.0.0.0")
    assert "run" not in called  # refused before ever reaching uvicorn.run


def test_entry_main_passes_same_host_to_guard_and_uvicorn(monkeypatch):
    # main() feeds the host it validated to uvicorn.run — asserts uvicorn is
    # bound to the network host the guard accepted (with a token). The
    # complementary fail-closed direction (guard sees loopback while binding
    # wide, no token) is covered by test_entry_main_refuses_non_loopback_...
    # above; main() uses a single `host` local for all call sites, so the two
    # cannot decouple without a rewrite.
    from agui_adapter import entry

    captured = {}

    def _fake_run(app, **k):
        captured["app"] = app
        captured.update(k)

    monkeypatch.setattr("uvicorn.run", _fake_run)
    monkeypatch.setenv("HERMES_AGUI_SESSION_TOKEN", TEST_TOKEN)
    entry.main(host="0.0.0.0")
    assert captured["host"] == "0.0.0.0"
    assert captured["app"] is not None  # create_app(bound_host="0.0.0.0", token) built


# --- _usable(): placeholder/entropy floor, including the ImportError fallback


def test_entry_main_uses_owning_secret_scope_not_launch_environment(monkeypatch):
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from agui_adapter import entry

    monkeypatch.setenv(
        "HERMES_AGUI_SESSION_TOKEN", "launch-profile-token-should-not-leak"
    )
    captured = {}
    monkeypatch.setattr(
        "agui_adapter.server.create_app", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    token = set_secret_scope({"HERMES_AGUI_SESSION_TOKEN": TEST_TOKEN})
    try:
        entry.main(host="0.0.0.0")
    finally:
        reset_secret_scope(token)
    assert captured["session_token"] == TEST_TOKEN


def test_standalone_startup_loads_profile_dotenv_credentials(monkeypatch):
    from agui_adapter import entry
    from agui_adapter.session import AgentConfig
    from hermes_cli.config import save_env_value

    save_env_value("HERMES_AGUI_SESSION_TOKEN", TEST_TOKEN)
    save_env_value("OPENAI_API_KEY", "fake-model-credential-from-profile-file")
    # Saving publishes to the current process; discard that mirror to model
    # a fresh standalone launch that has only the on-disk profile credentials.
    monkeypatch.delenv("HERMES_AGUI_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        captured["api_key"] = AgentConfig().api_key

    monkeypatch.setattr("agui_adapter.server.create_app", create)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    entry.main(host="0.0.0.0")
    assert captured["session_token"] == TEST_TOKEN
    assert captured["api_key"] == "fake-model-credential-from-profile-file"


def test_usable_rejects_placeholders_and_accepts_strong():
    assert auth._usable(None) is False
    assert auth._usable("short") is False
    # A known placeholder that is >=16 chars is rejected for BEING a
    # placeholder (not merely for length) on the primary has_usable_secret
    # path — "your_api_key_here" (17 chars) is in hermes_cli's placeholder set.
    assert auth._usable("your_api_key_here") is False
    assert auth._usable(TEST_TOKEN) is True


@pytest.mark.parametrize(
    "token", ["changeme00000000", "placeholder-value", "  PLACEHOLDER-VALUE  "]
)
def test_normal_helper_rejects_adapter_placeholders_before_network_bind(token):
    assert auth._usable(token) is False
    with pytest.raises(SystemExit):
        auth.require_token_or_refuse("0.0.0.0", token)


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

    monkeypatch.setitem(
        sys.modules, "hermes_cli.auth", types.ModuleType("hermes_cli.auth")
    )
    assert auth._usable("changeme00000000") is False
    assert auth._usable("your_api_key_here") is False
    assert auth._usable("short") is False
    assert auth._usable(TEST_TOKEN) is True
