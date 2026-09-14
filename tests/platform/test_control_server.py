"""The HTTP transport's safety properties, and the dashboard's boundary."""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from nova.control import ControlAPI
from nova.control.server import STATIC_DIR, build_server
from nova.errors import NovaError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def api(bundle, runtime):
    return ControlAPI(bundle, runtime)


@pytest.fixture
def live(api):
    """A running server on an ephemeral port, torn down after the test."""
    server = build_server(api, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(url: str, *, method: str = "GET", token: str = "") -> tuple[int, bytes, dict]:
    req = urllib.request.Request(url, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


# -- bind safety -------------------------------------------------------------


def test_refuses_non_loopback_bind_without_a_token(api):
    """The control plane lists a customer's agents and work. It must not leak by default."""
    with pytest.raises(NovaError, match="refusing to bind"):
        build_server(api, host="0.0.0.0", port=0)


def test_a_loopback_bind_needs_no_credentials(api):
    """Superseded the old single-token contract: a loopback caller already has shell
    access to everything the control plane reports. See test_control_auth.py."""
    build_server(api, host="127.0.0.1", port=0).server_close()


# -- the method surface ------------------------------------------------------
#
# Writes arrived in Phase 8, so "no method but GET" is no longer the invariant. What
# replaces it is narrower and worth more: POST reaches only the routes declared in
# WRITE_ROUTES, and the other write verbs are still refused before a handler runs.
# The write path itself is tested in test_control_write.py.


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_unused_write_methods_are_refused(live, method):
    status, body, _ = request(f"{live}/platform/v1/agents", method=method)
    assert status == 405
    assert "only GET, HEAD and POST" in json.loads(body)["error"]["message"]


def test_a_read_route_cannot_be_posted_to(live):
    """`/agents` is a read. POSTing to it must not find a handler by accident — an
    undeclared write path is unroutable, not merely admin-only."""
    status, _, _ = request(f"{live}/platform/v1/agents", method="POST")
    assert status in (404, 415)


# -- auth --------------------------------------------------------------------


def test_a_token_is_required_for_a_remote_caller(api, tmp_path):
    """The single shared ``--token`` is gone; callers are named principals with roles.
    ``behind_tls_proxy`` is what makes this loopback socket behave as a remote one —
    see test_control_auth.py for why that is the right switch."""
    from nova.control.auth import PrincipalStore, hash_token, new_token

    secret = new_token()
    path = tmp_path / "p.yaml"
    path.write_text(
        "principals:\n  - name: ops\n    role: admin\n"
        f"    token_sha256: {hash_token(secret)}\n",
        encoding="utf-8",
    )
    server = build_server(
        api, host="127.0.0.1", port=0,
        principals=PrincipalStore.load(path), behind_tls_proxy=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        assert request(f"{base}/platform/v1/health")[0] == 401
        assert request(f"{base}/platform/v1/health", token="wrong")[0] == 401
        assert request(f"{base}/platform/v1/health", token=secret)[0] == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# -- serving -----------------------------------------------------------------


def test_serves_the_api(live):
    status, body, headers = request(f"{live}/platform/v1/health")
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body)["platform"]["tenant_id"] == "acme"


def test_serves_the_dashboard(live):
    status, body, headers = request(f"{live}/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"<title>" in body


def test_security_headers_are_present(live):
    _, _, headers = request(f"{live}/")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_favicon_is_answered_quietly(live):
    """Browsers request it unprompted; a 404 on every page load is noise, not signal."""
    assert request(f"{live}/favicon.ico")[0] == 204


@pytest.mark.parametrize("attempt", ["/../nova/cli.py", "/%2e%2e/%2e%2e/etc/passwd", "/../../errors.py"])
def test_static_serving_cannot_escape_its_directory(live, attempt):
    status, _, _ = request(f"{live}{attempt}")
    assert status in (400, 404)


def test_unknown_static_path_is_404(live):
    assert request(f"{live}/nope.js")[0] == 404


# -- the control-plane boundary ----------------------------------------------


#: The dashboard's TypeScript source. The built bundle is what ships, but the *source* is
#: what a reviewer reads and what these boundary tests must hold against — a minified bundle
#: carrying React's own internals cannot be asserted on usefully.
UI_SRC = ROOT / "nova" / "control" / "ui" / "src"


def ui_sources() -> list[Path]:
    return sorted(p for p in UI_SRC.rglob("*.ts*") if p.is_file())


def _ts_without_comments(source: str) -> str:
    """Strip /* */ and // comments. Prose may mention a footgun; code may not use it."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"(?m)^\s*//.*$", "", source)


def test_dashboard_only_ever_calls_the_control_api():
    """The dashboard must never reach a runtime endpoint. This is the boundary in §5.

    Asserted two ways because they fail differently: the source must route every fetch
    through the one API constant, and the *built* bundle must contain no absolute origin —
    which catches a dependency that phones home as well as a hand-written mistake.
    """
    api_module = (UI_SRC / "lib" / "api.ts").read_text(encoding="utf-8")
    assert 'export const API = "/platform/v1"' in api_module

    for path in ui_sources():
        code = _ts_without_comments(path.read_text(encoding="utf-8"))
        for call in re.findall(r"fetch\(([^)]*)\)", code):
            assert "${API}" in call, f"{path.name}: fetch does not go through the API prefix: {call}"

    # Origins that appear in the bundle as *text* rather than as a request: XML namespace
    # identifiers React writes into SVG elements, and the documentation link React puts in
    # its own error messages. Neither is ever fetched. Anything else — a CDN, a font host,
    # a telemetry endpoint — must fail, because the control plane has to work on an
    # isolated network.
    NOT_REQUESTS = ("http://www.w3.org", "https://www.w3.org", "https://react.dev")
    for bundle in (STATIC_DIR / "assets").glob("*.js"):
        text = bundle.read_text(encoding="utf-8", errors="ignore")
        for origin in set(re.findall(r"https?://[a-zA-Z0-9.-]+", text)):
            assert origin.startswith(NOT_REQUESTS), (
                f"built dashboard references an external origin: {origin}"
            )


def test_dashboard_never_injects_api_data_as_markup():
    """Task titles and channel names are customer-controlled strings, never markup.

    React escapes interpolated text by default, so the whole risk collapses to one escape
    hatch. Keeping the raw-DOM names in the assertion too: a future component that reaches
    for `innerHTML` directly would bypass React's escaping just as effectively.
    """
    for path in ui_sources():
        code = _ts_without_comments(path.read_text(encoding="utf-8"))
        for hatch in ("dangerouslySetInnerHTML", "innerHTML", "insertAdjacentHTML", "document.write"):
            assert hatch not in code, f"{path.name} uses {hatch}"


def test_the_dashboard_needs_no_csp_relaxation():
    """Radix positions a tooltip by writing through the CSSOM, which CSP does not restrict —
    so the strict policy that shipped before the rewrite still ships after it.

    Pinned as a test because the tempting fix for a mispositioned popover is
    ``style-src 'unsafe-inline'``, and that would be a real regression made for a cosmetic
    reason. Verified in a browser: the page produces zero CSP violations while a deliberate
    inline-style probe on the same page is refused.
    """
    from nova.control.server import SECURITY_HEADERS

    policy = SECURITY_HEADERS["Content-Security-Policy"]
    assert "script-src 'self'" in policy
    assert "style-src 'self'" in policy
    assert "unsafe-inline" not in policy
    assert "unsafe-eval" not in policy


def test_the_built_dashboard_is_committed():
    """The control server is a stdlib file server: it cannot build anything. Whatever is in
    ``static/`` is what an operator gets, so the build output is committed deliberately."""
    index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assets = list((STATIC_DIR / "assets").glob("*"))
    assert assets, "no built dashboard bundle is present"

    # Both directions, because each catches a different mistake and I made the second one:
    # an asset on disk that nothing references is a stale build, and an asset referenced
    # but absent is a page that loads without its stylesheet.
    for asset in assets:
        assert asset.name in index, f"{asset.name} is not referenced by index.html (stale build?)"
    for referenced in re.findall(r'assets/[A-Za-z0-9._-]+', index):
        assert (STATIC_DIR / referenced).is_file(), (
            f"index.html references {referenced}, which is not present — the page would "
            f"load without it"
        )


def test_dashboard_has_no_external_dependencies():
    """No build step, no CDN: the control plane must work in an isolated network."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', html), "external resource referenced"
    assert "cdn" not in html.lower()


# --- graceful shutdown -------------------------------------------------------------
#
# A production container's main process is PID 1, and the kernel applies no default signal
# dispositions to PID 1. A server that does not install a SIGTERM handler therefore ignores
# SIGTERM outright: every `docker stop` waits out its grace period and ends in SIGKILL, with
# in-flight requests dropped. That was the observed behaviour before
# `_install_shutdown_handlers` existed, so these tests pin it.


def test_sigterm_stops_the_serving_loop(api):
    """SIGTERM ends `serve_forever` rather than being ignored."""
    import os
    import signal
    import time

    from nova.control.server import _install_shutdown_handlers

    server = build_server(api, host="127.0.0.1", port=0)
    before = signal.getsignal(signal.SIGTERM)
    restore = _install_shutdown_handlers(server)
    try:
        assert signal.getsignal(signal.SIGTERM) is not before, "no handler was installed"
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        # Let the loop enter its poll before signalling it.
        time.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)
        serving.join(timeout=5)
        assert not serving.is_alive(), "SIGTERM did not stop serve_forever"
    finally:
        restore()
        server.server_close()

    assert signal.getsignal(signal.SIGTERM) is before, "the previous handler was not restored"


def test_shutdown_handlers_are_not_installed_off_the_main_thread(api):
    """An embedding application that serves on a worker keeps its own signal handling."""
    import signal

    from nova.control.server import _install_shutdown_handlers

    server = build_server(api, host="127.0.0.1", port=0)
    before = signal.getsignal(signal.SIGTERM)
    result: dict[str, object] = {}

    thread = threading.Thread(target=lambda: result.update(restore=_install_shutdown_handlers(server)))
    thread.start()
    thread.join()
    server.server_close()

    assert signal.getsignal(signal.SIGTERM) is before
    # Still returns a callable, so a caller's `finally: restore()` needs no special case.
    assert callable(result["restore"])
    result["restore"]()
