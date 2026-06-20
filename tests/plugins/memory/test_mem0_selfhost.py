"""Tests for the self-hosted Mem0 direct-REST backend."""

import io
import json
import sys
import types
import urllib.error
import urllib.request
from email.message import Message
from urllib.parse import parse_qs, urlparse

import pytest

from plugins.memory.mem0 import Mem0MemoryProvider


class _HTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        if self._payload is None:
            return b""
        return json.dumps(self._payload).encode("utf-8")


def _header(request, name):
    name = name.lower()
    for key, value in request.header_items():
        if key.lower() == name:
            return value
    return None


def _json_body(request):
    if not request.data:
        return None
    return json.loads(request.data.decode("utf-8"))


def _install_exploding_memory_client(monkeypatch):
    # Patch MemoryClient ON the real mem0 module rather than replacing sys.modules["mem0"]
    # with a fake ModuleType. Replacing the whole module deletes the real one (and its
    # submodules) on monkeypatch revert when absent, polluting other tests that import mem0
    # (caused order-dependent hindsight failures under pytest-randomly). setattr on the real
    # module is reverted cleanly and leaves the module graph intact.
    # mem0 is an OPTIONAL runtime dependency (the provider imports it lazily;
    # it's not in pyproject/requirements). Skip — don't fail — when it's absent,
    # e.g. on CI runners without the optional package installed.
    _real_mem0 = pytest.importorskip("mem0")

    class ExplodingMemoryClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("MemoryClient must not be constructed for MEM0_HOST")

    monkeypatch.setattr(_real_mem0, "MemoryClient", ExplodingMemoryClient)


def _selfhost_provider(monkeypatch, tmp_path, *, destructive=False):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_HOST", "http://mem0.test")
    monkeypatch.setenv("MEM0_ADMIN_API_KEY", "admin-key")
    monkeypatch.setenv("MEM0_USER_ID", "ace")
    monkeypatch.setenv("MEM0_AGENT_ID", "daedalus")
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    monkeypatch.setenv("MEM0_DESTRUCTIVE_TOOLS", "true" if destructive else "false")
    provider = Mem0MemoryProvider()
    provider.initialize("test-session")
    return provider


def test_selfhost_tools_use_direct_rest_with_api_key_and_response_mapping(monkeypatch, tmp_path):
    _install_exploding_memory_client(monkeypatch)
    calls = []

    def fake_urlopen(request, timeout=0, context=None):
        parsed = urlparse(request.full_url)
        call = {
            "method": request.get_method(),
            "path": parsed.path,
            "query": parse_qs(parsed.query),
            "headers": {k.lower(): v for k, v in request.header_items()},
            "api_key": _header(request, "X-API-Key"),
            "body": _json_body(request),
            "timeout": timeout,
        }
        calls.append(call)
        if call["method"] == "POST" and call["path"] == "/memories":
            return _HTTPResponse({"results": [{"id": "m-add", "memory": "stored fact"}]})
        if call["method"] == "POST" and call["path"] == "/search":
            return _HTTPResponse({"results": [{"id": "m-search", "memory": "matched fact", "score": 0.87}]})
        if call["method"] == "GET" and call["path"] == "/memories":
            return _HTTPResponse({"results": [{"id": "m-profile", "memory": "profile fact"}]})
        raise AssertionError(f"unexpected HTTP call: {call}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = _selfhost_provider(monkeypatch, tmp_path)

    conclude = json.loads(provider.handle_tool_call("mem0_conclude", {"conclusion": "store this"}))
    search = json.loads(provider.handle_tool_call("mem0_search", {"query": "needle", "top_k": 3}))
    profile = json.loads(provider.handle_tool_call("mem0_profile", {}))

    assert conclude == {"result": "Fact stored."}
    assert search == {"results": [{"memory": "matched fact", "score": 0.87}], "count": 1}
    assert profile == {"result": "profile fact", "count": 1}

    assert [(c["method"], c["path"]) for c in calls] == [
        ("POST", "/memories"),
        ("POST", "/search"),
        ("GET", "/memories"),
    ]
    assert all(c["api_key"] == "admin-key" for c in calls)

    add_call, search_call, profile_call = calls
    assert add_call["body"]["messages"] == [{"role": "user", "content": "store this"}]
    assert add_call["body"]["user_id"] == "ace"
    assert add_call["body"]["agent_id"] == "daedalus"
    assert search_call["body"]["query"] == "needle"
    assert search_call["body"]["user_id"] == "ace"
    # search is a READ -> user-scoped only, no agent_id
    assert "agent_id" not in search_call["body"]
    # mem0_profile -> get_all (READ) -> user-scoped only, no agent_id
    assert profile_call["query"] == {"user_id": ["ace"]}


def test_selfhost_delete_uses_rest_delete_after_read_before_destroy(monkeypatch, tmp_path):
    _install_exploding_memory_client(monkeypatch)
    calls = []

    def fake_urlopen(request, timeout=0, context=None):
        parsed = urlparse(request.full_url)
        call = {
            "method": request.get_method(),
            "path": parsed.path,
            "api_key": _header(request, "X-API-Key"),
            "body": _json_body(request),
        }
        calls.append(call)
        if call["method"] == "GET" and call["path"] == "/memories/m-delete":
            return _HTTPResponse({"id": "m-delete", "memory": "doomed", "metadata": {}})
        if call["method"] == "DELETE" and call["path"] == "/memories/m-delete":
            return _HTTPResponse({"id": "m-delete"})
        raise AssertionError(f"unexpected HTTP call: {call}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = _selfhost_provider(monkeypatch, tmp_path, destructive=True)

    result = json.loads(provider.handle_tool_call("mem0_delete", {"memory_id": "m-delete"}))

    assert result["deleted"] == 1
    assert result["results"] == [{"id": "m-delete", "outcome": "deleted", "was": "doomed"}]
    assert [(c["method"], c["path"]) for c in calls] == [
        ("GET", "/memories/m-delete"),
        ("DELETE", "/memories/m-delete"),
    ]
    assert all(c["api_key"] == "admin-key" for c in calls)


@pytest.mark.parametrize("host_value", [None, "", "   "])
def test_unset_or_blank_host_uses_existing_memoryclient_path(monkeypatch, tmp_path, host_value):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "cloud-key")
    monkeypatch.setenv("MEM0_ADMIN_API_KEY", "admin-key")
    if host_value is None:
        monkeypatch.delenv("MEM0_HOST", raising=False)
    else:
        monkeypatch.setenv("MEM0_HOST", host_value)

    constructed = []
    # mem0 is an OPTIONAL runtime dependency (imported lazily by the provider);
    # skip rather than fail when it isn't installed (e.g. clean CI runners).
    _real_mem0 = pytest.importorskip("mem0")

    class FakeMemoryClient:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    # setattr on the real module (reverted cleanly) instead of replacing sys.modules["mem0"]
    # — the whole-module swap pollutes other tests under random ordering (see helper above).
    monkeypatch.setattr(_real_mem0, "MemoryClient", FakeMemoryClient)

    # Capture the bounded-client limits the fallback passes, without poking httpx internals.
    captured_limits = {}
    try:
        import httpx
        has_httpx = True
        _RealClient = httpx.Client

        class _CapturingClient(_RealClient):
            def __init__(self, *args, **kw):
                limits = kw.get("limits")
                if limits is not None:
                    captured_limits["max_connections"] = limits.max_connections
                    captured_limits["max_keepalive_connections"] = limits.max_keepalive_connections
                    captured_limits["keepalive_expiry"] = limits.keepalive_expiry
                super().__init__(*args, **kw)

        monkeypatch.setattr(httpx, "Client", _CapturingClient)
    except ImportError:
        has_httpx = False

    provider = Mem0MemoryProvider()
    provider.initialize("test-session")
    provider._get_client()

    # Exactly one MemoryClient constructed, always with the api_key.
    assert len(constructed) == 1
    assert constructed[0].get("api_key") == "cloud-key"
    # When httpx is importable the cloud fallback must hand the SDK a *bounded*
    # client (limits + keepalive_expiry) so idle keepalive sockets don't rot into
    # CLOSE_WAIT and leak fds in a long-lived gateway (HANDOFF-fd-leak-client-pool.md).
    if has_httpx:
        assert "client" in constructed[0], "cloud fallback must pass a bounded httpx.Client"
        assert captured_limits == {
            "max_connections": 10,
            "max_keepalive_connections": 5,
            "keepalive_expiry": 30.0,
        }
    else:
        # No httpx -> graceful degradation to the default (unbounded) client.
        assert "client" not in constructed[0]


def test_mem0_json_overrides_selfhost_env_config(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_HOST", "http://env-host")
    monkeypatch.setenv("MEM0_ADMIN_API_KEY", "env-key")
    monkeypatch.setenv("MEM0_USER_ID", "env-user")
    monkeypatch.setenv("MEM0_AGENT_ID", "env-agent")
    (tmp_path / "mem0.json").write_text(json.dumps({
        "host": "http://file-host/",
        "admin_api_key": "file-key",
        "user_id": "file-user",
        "agent_id": "file-agent",
    }))

    def fake_urlopen(request, timeout=0, context=None):
        parsed = urlparse(request.full_url)
        calls.append({
            "url": request.full_url,
            "netloc": parsed.netloc,
            "api_key": _header(request, "X-API-Key"),
            "body": _json_body(request),
        })
        return _HTTPResponse({"results": [{"id": "m1", "memory": "file scoped", "score": 0.9}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = Mem0MemoryProvider()
    provider.initialize("test-session")

    result = json.loads(provider.handle_tool_call("mem0_search", {"query": "scope"}))

    assert result["count"] == 1
    assert calls[0]["netloc"] == "file-host"
    assert calls[0]["api_key"] == "file-key"
    assert calls[0]["body"]["user_id"] == "file-user"
    # search is a READ -> user-scoped only, no agent_id injected
    assert "agent_id" not in calls[0]["body"]


def test_selfhost_401_surfaces_error_and_records_failure_without_fabricated_memory(monkeypatch, tmp_path):
    def fake_urlopen(request, timeout=0, context=None):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=Message(),
            fp=io.BytesIO(b'{"detail":"bad key"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = _selfhost_provider(monkeypatch, tmp_path)

    result = json.loads(provider.handle_tool_call("mem0_search", {"query": "needle"}))

    assert "error" in result
    assert "401" in result["error"]
    assert "results" not in result
    assert "No relevant memories found" not in result["error"]
    assert provider._consecutive_failures == 1


def test_direct_rest_client_scopes_writes_both_reads_user_only():
    """B3/B4: on a shared multi-agent store, add/search/get_all with NO explicit scope
    must still be constrained to the client's configured user_id AND agent_id —
    never querying or writing globally."""
    from importlib import import_module
    mod = import_module("plugins.memory.mem0")
    client = mod._DirectRestMem0Client(
        host="http://mem0.test", admin_api_key="k", agent_id="daedalus", user_id="ace"
    )
    sent = []

    def _fake_request(method, path, *, body=None, params=None):
        sent.append({"method": method, "path": path, "body": body, "params": params})
        return {"results": []}

    client._request = _fake_request

    # add() (WRITE) with no user_id/agent_id kwargs must inject BOTH (attribution + B4)
    client.add([{"role": "user", "content": "x"}])
    assert sent[-1]["body"]["user_id"] == "ace"
    assert sent[-1]["body"]["agent_id"] == "daedalus"

    # search() (READ) injects user_id ONLY — NOT agent_id. Reads are user-scoped for
    # cross-session recall, and historical memories stored agent-scoped-without-user
    # would be silently dropped by an agent AND-filter (the live-cutover 0-results bug).
    client.search(query="q")
    assert sent[-1]["body"]["user_id"] == "ace"
    assert "agent_id" not in sent[-1]["body"]

    # get_all() (READ) likewise injects user_id ONLY, never agent_id.
    client.get_all()
    assert sent[-1]["params"]["user_id"] == "ace"
    assert "agent_id" not in sent[-1]["params"]

    # an explicit caller READ filter is respected (user override, still no agent injected)
    client.search(query="q", filters={"user_id": "other"})
    assert sent[-1]["body"]["user_id"] == "other"
    assert "agent_id" not in sent[-1]["body"]

    # an explicit agent_id in a READ filter IS honored (caller opted in)
    client.search(query="q", filters={"agent_id": "explicit"})
    assert sent[-1]["body"]["agent_id"] == "explicit"
    assert sent[-1]["body"]["user_id"] == "ace"


def _count_open_fds() -> int:
    """Best-effort open-fd count for THIS process, cross-platform.

    Linux: count /proc/self/fd entries. macOS/BSD: fall back to psutil if present,
    else resource-based proc fd listing via /dev/fd. The soak test asserts a
    *plateau*, so an approximate-but-consistent counter is sufficient.
    """
    import os
    for path in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(path):
            try:
                return len(os.listdir(path))
            except OSError:
                continue
    try:
        import psutil  # type: ignore
        return psutil.Process().num_fds()
    except Exception:
        return -1


def test_direct_rest_client_fd_count_plateaus_under_soak(tmp_path):
    """REAL regression test for HANDOFF-fd-leak-client-pool.md.

    Drive thousands of real add/search calls through the real _DirectRestMem0Client
    (urllib, real sockets) against a real loopback HTTP server in ONE process, and
    assert the open-fd count PLATEAUS rather than growing monotonically. This is the
    end-to-end proof the client doesn't strand sockets — a unit test on pool config
    can't catch a path that leaks fds; only exercising the real socket lifecycle can.

    The direct-REST client uses urllib with a per-call `with urlopen(...)` that closes
    each connection, so it must not accumulate CLOSE_WAIT/idle fds. If a future change
    swaps in a pooled client without bounding it, this test fails.
    """
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    fd_probe = _count_open_fds()
    if fd_probe < 0:
        pytest.skip("no way to count fds on this platform (no /proc, /dev/fd, or psutil)")

    class _Handler(BaseHTTPRequestHandler):
        def _reply(self):
            body = json.dumps({"results": []}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length:
                self.rfile.read(length)
            self._reply()

        def do_GET(self):
            self._reply()

        def log_message(self, format, *args):
            pass  # silence

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        from importlib import import_module
        mod = import_module("plugins.memory.mem0")
        client = mod._DirectRestMem0Client(
            host=f"http://127.0.0.1:{port}",
            admin_api_key="k",
            agent_id="daedalus",
            user_id="ace",
        )

        # Warm up so transient import/buffer fds aren't counted as growth.
        for _ in range(50):
            client.add([{"role": "user", "content": "warmup"}])
            client.search(query="warmup")

        baseline = _count_open_fds()

        # Soak: thousands of real round-trips in one long-lived process.
        n = 2000
        for i in range(n):
            client.add([{"role": "user", "content": f"memory {i}"}])
            client.search(query=f"query {i}")

        peak = _count_open_fds()
    finally:
        server.shutdown()
        server.server_close()

    # The fd count must PLATEAU. A leaking client would grow ~1 fd per call
    # (thousands); a non-leaking one stays flat give-or-take a small jitter from
    # GC timing and the server's own worker threads. Allow generous slack but far
    # below the per-call-leak signal.
    growth = peak - baseline
    assert growth < 50, (
        f"fd count grew by {growth} over {n} calls "
        f"(baseline={baseline}, peak={peak}) — client is leaking sockets/fds"
    )


def test_ca_bundle_builds_ssl_context_for_https_private_ca(tmp_path):
    """A private-CA HTTPS endpoint (e.g. mem0.ace, signed by the LAN root CA) must be
    verifiable via an explicit CA bundle, since fleet hosts don't carry the private CA
    in their system trust store. With a bundle + https host, the client builds an SSL
    context; without one (or over http) it stays None (urllib system-trust default).
    Regression for the CERTIFICATE_VERIFY_FAILED that the live cutover surfaced.
    """
    from importlib import import_module
    mod = import_module("plugins.memory.mem0")

    # a real (self-signed) PEM so ssl.create_default_context(cafile=...) accepts it
    import ssl
    import datetime
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        pytest.skip("cryptography not available to mint a test CA PEM")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Local Root CA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    ca_pem = tmp_path / "test-ca.crt"
    ca_pem.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    # https + bundle -> context built
    c1 = mod._DirectRestMem0Client(
        host="https://mem0.ace", admin_api_key="k", agent_id="a", user_id="u",
        ca_bundle=str(ca_pem),
    )
    assert isinstance(c1._ssl_context, ssl.SSLContext)

    # https + NO bundle -> None (system trust store default)
    c2 = mod._DirectRestMem0Client(
        host="https://mem0.ace", admin_api_key="k", agent_id="a", user_id="u",
    )
    assert c2._ssl_context is None

    # http + bundle -> None (no TLS to verify)
    c3 = mod._DirectRestMem0Client(
        host="http://mem0.ace", admin_api_key="k", agent_id="a", user_id="u",
        ca_bundle=str(ca_pem),
    )
    assert c3._ssl_context is None


def _clear_mem0_env(monkeypatch):
    for var in (
        "MEM0_API_KEY", "MEM0_HOST", "MEM0_ADMIN_API_KEY",
        "MEM0_CA_BUNDLE", "MEM0_USER_ID", "MEM0_AGENT_ID",
    ):
        monkeypatch.delenv(var, raising=False)


def test_api_key_not_required_in_schema_so_selfhost_setup_is_not_blocked():
    """The config schema must NOT mark ``api_key`` as required.

    ``is_available()`` bypasses ``api_key`` entirely when ``host`` is set
    (self-hosted mode is gated on ``admin_api_key``). A schema that marks
    ``api_key`` required would force any validation/setup UI to demand a cloud
    key even for a self-hosted server that never uses one. Behavior contract,
    not a value snapshot: api_key is optional; the host/admin pair carries
    self-hosted availability.
    """
    provider = Mem0MemoryProvider()
    schema = {field["key"]: field for field in provider.get_config_schema()}
    assert schema["api_key"].get("required", False) is False, (
        "api_key must be optional — is_available() bypasses it in self-hosted mode"
    )


def test_is_available_selfhost_without_api_key(monkeypatch, tmp_path):
    """Self-hosted (host + admin_api_key, NO api_key) must report available."""
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    _clear_mem0_env(monkeypatch)
    monkeypatch.setenv("MEM0_HOST", "https://mem0.ace")
    monkeypatch.setenv("MEM0_ADMIN_API_KEY", "admin-secret")
    assert Mem0MemoryProvider().is_available() is True


def test_is_available_cloud_requires_api_key(monkeypatch, tmp_path):
    """Cloud mode (no host) still requires api_key — fix doesn't weaken that path."""
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    _clear_mem0_env(monkeypatch)
    assert Mem0MemoryProvider().is_available() is False
    monkeypatch.setenv("MEM0_API_KEY", "cloud-key")
    assert Mem0MemoryProvider().is_available() is True
