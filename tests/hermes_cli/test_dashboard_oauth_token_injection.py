"""Regression test for #59274: dashboard model-provider OAuth fails under
basic_auth because __HERMES_SESSION_TOKEN__ is never injected when the auth
gate is active.

Background: when the SPA's OAuth-start helper tries to begin a model-provider
OAuth flow, it reads ``window.__HERMES_SESSION_TOKEN__`` and throws if absent.
Under the basic_auth provider, ``app.state.auth_required`` is True (the gate is
active) and the server intentionally omits the session token from the served
``index.html`` — the SPA is supposed to use cookie auth for /api/* instead.
But the model-OAuth flow doesn't have that cookie-based fallback: it still
reads the global, finds nothing, and aborts with
"Session token not available — page must be served by the Hermes dashboard
server".

Fix: in the gated path, also inject the session token. The token is a
per-process secret that already lives in the HTML for loopback mode; the
basic_auth cookie session is bound to the same client that received the
token, so exposing it doesn't broaden the trust model.

Failing-test-first:
  test_gated_index_injects_session_token — bug detection. Asserts that
    __HERMES_SESSION_TOKEN__ is present in index.html when auth_required
    is True. Fails on unfixed code (the gated branch only injects
    __HERMES_AUTH_REQUIRED__, not the token). Passes after the fix.

The test calls ``_serve_index`` directly (the inner function that
builds the SPA HTML). This avoids the heavy "Sign-in unavailable" gate
that blocks the public index route when no auth provider is installed in
the test env — we only want to assert what the SPA HTML contains, not
exercise the full auth gate plumbing.
"""

import re

import pytest

from hermes_cli import web_server


def _get_serve_index():
    """Pull the inner _serve_index closure out of mount_spa.

    mount_spa() returns early when WEB_DIST is missing (with a 404
    handler). When WEB_DIST exists, it defines _serve_index as a
    closure. The first SPA route handler's source contains the
    function reference we need.
    """
    from pathlib import Path
    web_dist = Path(web_server.__file__).parent / "web_dist"
    if not web_dist.exists():
        pytest.skip("WEB_DIST not built in this env")
    # mount_spa is idempotent; re-running is fine. Capture the closure
    # by reaching into the registered route handler's source.
    for route in web_server.app.routes:
        # The SPA mount is the catch-all GET /{full_path:path} handler.
        if getattr(route, "path", "") == "/{full_path:path}":
            # The closure is the route.endpoint itself (a regular
            # function in the catch-all).
            closure = route.endpoint
            if hasattr(closure, "__closure__") and closure.__closure__:
                for cell in closure.__closure__:
                    val = cell.cell_contents
                    if callable(val) and val.__name__ == "_serve_index":
                        return val
    pytest.skip("Could not locate _serve_index closure")


@pytest.fixture
def serve_index():
    return _get_serve_index()


def test_gated_index_injects_session_token(serve_index):
    """Under basic_auth (auth_required=True), the served index.html must
    still expose the session token so the model-provider OAuth flow can
    authenticate.

    Before the fix: only __HERMES_AUTH_REQUIRED__ was injected in the
    gated branch, so the SPA's hf() helper threw "Session token not
    available" and the model-OAuth login flow failed.
    """
    # Simulate gated state by manipulating app.state directly.
    prev = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = True
    try:
        response = serve_index(prefix="")
        # _serve_index returns an HTMLResponse; read the body.
        body = response.body.decode("utf-8")
    finally:
        web_server.app.state.auth_required = prev

    assert "__HERMES_SESSION_TOKEN__" in body, (
        "Gated index.html is missing __HERMES_SESSION_TOKEN__ — model-provider "
        "OAuth flow will fail with 'Session token not available' (#59274)"
    )
    # The auth-required flag should ALSO be present (the gated branch's
    # original contract is preserved alongside the new token injection).
    assert "__HERMES_AUTH_REQUIRED__" in body


def test_gated_session_token_matches_server_secret(serve_index):
    """The injected token must match web_server._SESSION_TOKEN, not be a
    stale or empty value. The OAuth helper compares the injected token
    against the X-Hermes-Session-Token header it sends — if they don't
    match, every model-OAuth call 401s.
    """
    prev = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = True
    try:
        body = serve_index(prefix="").body.decode("utf-8")
    finally:
        web_server.app.state.auth_required = prev

    match = re.search(
        r"window\.__HERMES_SESSION_TOKEN__\s*=\s*['\"]([^'\"]+)['\"]",
        body,
    )
    assert match, (
        "Gated index.html has __HERMES_SESSION_TOKEN__ but it's not in the "
        "expected assignment form (window.__HERMES_SESSION_TOKEN__='***'). "
        "Cannot verify token value."
    )
    assert match.group(1) == web_server._SESSION_TOKEN, (
        f"Injected session token does not match web_server._SESSION_TOKEN. "
        f"Got: {match.group(1)[:8]}..., "
        f"expected: {web_server._SESSION_TOKEN[:8]}..."
    )


def test_gated_index_still_injects_base_path_and_chat_flag(serve_index):
    """Regression guard: the fix must not drop the other injected globals.

    The gated branch's original contract is to inject:
      - __HERMES_DASHBOARD_EMBEDDED_CHAT__
      - __HERMES_BASE_PATH__
      - __HERMES_AUTH_REQUIRED__
    These must all still be present after adding the session token.
    """
    prev = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = True
    try:
        body = serve_index(prefix="").body.decode("utf-8")
    finally:
        web_server.app.state.auth_required = prev

    assert "__HERMES_DASHBOARD_EMBEDDED_CHAT__" in body
    assert "__HERMES_BASE_PATH__" in body
    assert "__HERMES_AUTH_REQUIRED__" in body


def test_loopback_index_still_injects_session_token(serve_index):
    """Sanity: the existing loopback behaviour is preserved.

    The loopback (non-gated) path already injects the token. The fix
    must not break this — only extend it to the gated path.
    """
    prev = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    try:
        body = serve_index(prefix="").body.decode("utf-8")
    finally:
        web_server.app.state.auth_required = prev

    assert "__HERMES_SESSION_TOKEN__" in body
    assert web_server._SESSION_TOKEN in body