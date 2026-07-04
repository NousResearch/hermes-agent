# Fix MCP OAuth Port Global — PR #34260 (PR 1/2)

> **For Hermes:** Use code-with-review-hook skill to implement this plan task-by-task.

**Goal:** Eliminate module-level `_oauth_port` global by introducing `OAuthCallbackServer` that binds the HTTP server before the OAuth flow starts, eliminating TOCTOU and concurrent-flow collision.

**Architecture:** New `OAuthCallbackServer` class holds the bound `HTTPServer` + port. Created in `_configure_callback_port()`, started immediately, polled by `_wait_for_callback()`. Port flows through `functools.partial` closures, not globals.

**Tech Stack:** Python 3.12+, `http.server.HTTPServer`, `asyncio`, `threading`, `functools.partial`

**Files:**
- Modify: `tools/mcp_oauth.py` (core changes)
- Modify: `tools/mcp_oauth_manager.py` (adapter changes)
- Modify: `tests/tools/test_mcp_oauth.py` (new + updated tests)

---

## Task 1: Create `OAuthCallbackServer` class (no tests yet — class is testable via its own tests later)

**Objective:** New class that binds an `HTTPServer`, starts it in a daemon thread, and provides async polling for results.

**Files:**
- Modify: `tools/mcp_oauth.py` — add class after `_make_callback_handler` (line ~456)

**Step 1: Add `OAuthCallbackServer` class**

```python
class OAuthCallbackServer:
    """Persistent localhost HTTP server for OAuth callback capture.

    Binds the server at construction time, eliminating the TOCTOU gap
    between port discovery and server startup. Runs ``handle_request()``
    in a loop until the OAuth callback arrives or timeout expires.

    Attributes:
        port: The actual port the server is bound to.
        _result: Shared dict written by the HTTP handler.
    """

    def __init__(self, port: int = 0, timeout: float = 300.0):
        self._result: dict[str, Any] = {"auth_code": None, "state": None, "error": None}
        self._timeout = timeout
        handler_cls = self._make_handler()
        # bind server at construction time — port consumed immediately
        self._server = HTTPServer(("127.0.0.1", port), handler_cls)
        self._server.timeout = 1.0  # handle_request() polls
        self.port: int = self._server.server_address[1]
        self._thread: threading.Thread | None = None

    def _make_handler(self) -> type:
        """Build a per-instance HTTP handler with path filtering."""
        result = self._result

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                # Only process /callback; ignore favicon, preflight, etc.
                if parsed.path != "/callback":
                    self.send_response(404)
                    self.end_headers()
                    return
                params = parse_qs(parsed.query)
                result["auth_code"] = params.get("code", [None])[0]
                result["state"] = params.get("state", [None])[0]
                result["error"] = params.get("error", [None])[0]
                body = (
                    "<html><body><h2>Authorization Successful</h2>"
                    "<p>You can close this tab and return to Hermes.</p></body></html>"
                ) if result["auth_code"] else (
                    "<html><body><h2>Authorization Failed</h2>"
                    f"<p>Error: {result['error'] or 'unknown'}</p></body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("OAuth callback: %s", fmt % args)

        return _Handler

    def start(self) -> None:
        """Start the server in a daemon thread."""
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def _serve_loop(self) -> None:
        """Process requests until callback arrives or timeout."""
        deadline = time.time() + self._timeout
        while time.time() < deadline:
            try:
                self._server.handle_request()
            except Exception:
                # Connection errors (client disconnect, etc.) are non-fatal
                pass
            if self._result["auth_code"] is not None or self._result["error"] is not None:
                break

    async def wait(self, timeout: float = 300.0) -> tuple[str, str | None]:
        """Async-poll the result until callback arrives or timeout.

        Returns (auth_code, state).  Raises RuntimeError on timeout/error.
        """
        poll_interval = 0.5
        elapsed = 0.0
        while elapsed < timeout:
            if self._result["auth_code"] is not None or self._result["error"] is not None:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        if self._result["error"]:
            raise RuntimeError(f"OAuth authorization failed: {self._result['error']}")
        if self._result["auth_code"] is None:
            raise OAuthNonInteractiveError(
                "OAuth callback timed out — no authorization code received."
            )
        return self._result["auth_code"], self._result["state"]

    def close(self) -> None:
        """Shut down the server and wait for the thread."""
        self._server.server_close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
```

**Step 2: Verify syntax**
Run: `python3 -c "import ast; ast.parse(open('tools/mcp_oauth.py').read()); print('Syntax OK')"`

**Acceptance Criteria:**
1. `OAuthCallbackServer(port=0)` binds to an actual port
2. `server.port` returns the bound port
3. `server.start()` launches daemon thread
4. `server.close()` shuts down cleanly
5. No syntax errors
6. Existing tests still import: `from tools.mcp_oauth import ...` works

---

## Task 2: Rewrite `_configure_callback_port` — remove `_oauth_port` global, return `OAuthCallbackServer`

**Objective:** Port discovery + server bind happen atomically. The callback server is started immediately.

**Files:**
- Modify: `tools/mcp_oauth.py:_configure_callback_port` (line ~707)

**Step 1: Rewrite function**

Replace the current `_configure_callback_port` (lines 707-725):

```python
def _configure_callback_port(cfg: dict) -> "OAuthCallbackServer":
    """Resolve the OAuth callback port and bind the callback server.

    Creates and starts an :class:`OAuthCallbackServer` so the port is
    consumed immediately — no TOCTOU gap between port discovery and
    actual server startup.  Stores the server in ``cfg['_callback_server']``
    and the resolved port in ``cfg['_resolved_port']``.

    Also sets the legacy module-level ``_oauth_port`` so existing callers
    of ``_wait_for_callback`` that haven't been migrated yet keep working.
    This will be removed in a follow-up PR once all consumers use the
    server instance.
    """
    global _oauth_port
    requested = int(cfg.get("redirect_port", 0))
    server = OAuthCallbackServer(port=requested)
    server.start()
    cfg["_resolved_port"] = server.port
    cfg["_callback_server"] = server
    _oauth_port = server.port  # legacy: for existing _wait_for_callback callers
    return server
```

**Step 2: Delete `_find_free_port()` usage — `OAuthCallbackServer(port=0)` handles this internally**
Note: `_find_free_port()` is still used by tests directly, so keep the function definition but `_configure_callback_port` no longer calls it.

**Step 3: Verify existing tests still pass**
Run: `python -m pytest tests/tools/test_mcp_oauth.py -x -q 2>&1 | tail -20`

Expected: Tests that use `_configure_callback_port` should still pass. Some `_wait_for_callback` tests may fail because they set `mod._oauth_port` directly — we'll fix those in the next task.

**Acceptance Criteria:**
1. `_configure_callback_port` no longer calls `_find_free_port()`
2. Server is started and bound before function returns
3. `cfg['_resolved_port']` equals `server.port`
4. `cfg['_callback_server']` is the `OAuthCallbackServer` instance
5. Legacy `_oauth_port` still set for backward compat

---

### Task 3a: Rewrite `_wait_for_callback` — accept server parameter, race paste fallback

**Objective:** `_wait_for_callback` gets called by the SDK with no args. When server is provided, poll the pre-bound server; also start paste reader thread that writes to the **same** `server._result` dict so either path wins.

**Files:**
- Modify: `tools/mcp_oauth.py:_wait_for_callback` (line ~515)

**Step 1: Replace `_wait_for_callback`**

```python
async def _wait_for_callback(server: "OAuthCallbackServer | None" = None) -> tuple[str, str | None]:
    """Wait for the OAuth callback.

    If *server* is provided (from ``_configure_callback_port`` via
    ``functools.partial``), polls the already-bound server for the result.
    The paste fallback writes directly to ``server._result`` so it races
    naturally with the HTTP listener.

    Falls back to the legacy module-level ``_oauth_port`` path when
    *server* is None (backward compat for callers that haven't been
    migrated yet).
    """
    # New path: server bound ahead of time, just poll
    if server is not None:
        # Paste fallback: race a stdin reader against the HTTP listener.
        # Both write to `server._result`, so whichever finishes first wins.
        if _is_interactive():
            print(
                "\n  Or paste the redirect URL here (or the ``?code=...&state=...`` "
                "portion) and press Enter. Type ``skip`` + Enter to continue "
                "without this server:",
                file=sys.stderr, flush=True,
            )
            threading.Thread(
                target=_paste_callback_reader, args=(server._result,), daemon=True
            ).start()
        result = await server.wait()
        if result[1] is None and server._result.get("error") == _USER_SKIPPED_SENTINEL:
            raise OAuthNonInteractiveError("user_skipped")
        return result

    # Legacy path — remove after all callers migrated to functools.partial
    if _oauth_port is None:
        raise RuntimeError(
            "OAuth callback port not set — build_oauth_auth must be called "
            "before _wait_for_oauth_callback"
        )
    # ... (existing legacy implementation unchanged)
```

**Step 2: Verify syntax**
Run: `python3 -c "import ast; ast.parse(open('tools/mcp_oauth.py').read()); print('Syntax OK')"`

**Acceptance Criteria:**
1. `_wait_for_callback(server=server_instance)` polls pre-bound server
2. Paste prompt shown when `_is_interactive()` — paste writes to `server._result`
3. HTTP callback and paste race correctly (either wins)
4. Skip token raises `OAuthNonInteractiveError("user_skipped")`
5. Legacy path (server=None) still works via `_oauth_port`

---

### Task 3b: Update `build_oauth_auth` and `_redirect_handler` to use functools.partial

**Objective:** Wire the new `OAuthCallbackServer` into both callbacks via `functools.partial`.

**Files:**
- Modify: `tools/mcp_oauth.py:build_oauth_auth` (line ~789)
- Modify: `tools/mcp_oauth.py:_redirect_handler` (line ~464)

**Step 1: Update `_redirect_handler` to accept optional port parameter**

Change signature from:
```python
async def _redirect_handler(authorization_url: str) -> None:
```
to:
```python
async def _redirect_handler(authorization_url: str, port: int | None = None) -> None:
```

And replace `_oauth_port` usage with:
```python
    actual_port = port or _oauth_port
    if actual_port and (os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY")):
        print(f"  Remote session detected... http://127.0.0.1:{actual_port}/callback", ...)
```

**Step 2: Update `build_oauth_auth`**

Replace:
```python
    return OAuthClientProvider(
        ...
        redirect_handler=_redirect_handler,
        callback_handler=_wait_for_callback,
    )
```
with:
```python
    import functools
    callback_server = cfg.get("_callback_server")
    return OAuthClientProvider(
        ...
        redirect_handler=functools.partial(_redirect_handler, port=callback_server.port),
        callback_handler=functools.partial(_wait_for_callback, server=callback_server),
    )
```

**Step 3: Run test to check for breakage**
Run: `python -m pytest tests/tools/test_mcp_oauth.py -x -q 2>&1 | tail -20`

Expected: Some `_wait_for_callback` tests fail (they call without server). Marked for Task 4.

**Acceptance Criteria:**
1. `_redirect_handler(auth_url, port=8787)` uses 8787 for SSH hint
2. `_redirect_handler(auth_url)` falls back to `_oauth_port`
3. `build_oauth_auth` passes both handlers via `functools.partial`
4. Existing `TestRedirectHandlerSshHint` tests still pass

---

### Task 3c: Update `_build_provider` in manager to use functools.partial for BOTH handlers

**Objective:** Complete the manager migration — `redirect_handler` and `callback_handler` both use `functools.partial`.

**Files:**
- Modify: `tools/mcp_oauth_manager.py:_build_provider` (line ~492)

**Step 1: Update `_build_provider`**

Replace:
```python
        return _HERMES_PROVIDER_CLS(
            ...
            redirect_handler=_redirect_handler,
            callback_handler=_wait_for_callback,
        )
```
with (inline import, matching `build_oauth_auth` pattern):
```python
        import functools
        callback_server = cfg.get("_callback_server")
        return _HERMES_PROVIDER_CLS(
            ...
            redirect_handler=functools.partial(_redirect_handler, port=callback_server.port),
            callback_handler=functools.partial(_wait_for_callback, server=callback_server),
        )
```

**Step 2: Verify syntax**
Run: `python3 -c "import ast; ast.parse(open('tools/mcp_oauth_manager.py').read()); print('Syntax OK')"`

**Step 3: Run existing tests**
Run: `python -m pytest tests/tools/test_mcp_oauth_manager.py -x -q 2>&1 | tail -10`

**Acceptance Criteria:**
1. `_build_provider` passes `functools.partial(_redirect_handler, port=...)` 
2. `_build_provider` passes `functools.partial(_wait_for_callback, server=...)`
3. Manager tests still pass
4. No `import functools` at module top (inline only — avoids unused import when OAuth unavailable)

---

### Task 4: Update test file — adapt to new `_wait_for_callback` signature and add new tests

**Objective:** Tests that call `_wait_for_callback()` directly need a server instance. Add tests for bind-first and path filtering.

**Files:**
- Modify: `tests/tools/test_mcp_oauth.py` — TestWaitForCallbackPasteIntegration, TestWaitForCallbackSkipIntegration, new test classes

**Step 1: Add test helper (top of test class section, after existing helpers at ~line 33)**

```python
def _mock_callback_server(port: int = 0) -> "OAuthCallbackServer":
    """Create and start an OAuthCallbackServer for testing."""
    from tools.mcp_oauth import OAuthCallbackServer
    server = OAuthCallbackServer(port=port)
    server.start()
    return server
```

**Step 2: Update `TestWaitForCallbackPasteIntegration` tests**

Replace patterns like:
```python
mod._oauth_port = _find_free_port()
asyncio.run(_wait_for_callback())
```
with:
```python
server = _mock_callback_server()
asyncio.run(_wait_for_callback(server=server))
```

Update `test_paste_prompt_mentions_skip` to use `server._result` race pattern — when paste wins, `server.wait()` returns paste's result.

**Step 3: Add new test classes**

```python
class TestOAuthCallbackServerBindFirst:
    """Bind-first: server is alive before _configure_callback_port returns."""

    def test_server_bound_on_configure_return(self):
        cfg = {}
        from tools.mcp_oauth import _configure_callback_port
        server = _configure_callback_port(cfg)
        assert server.port > 0
        assert cfg["_callback_server"] is server
        assert cfg["_resolved_port"] == server.port
        # Server must be running — try connecting
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(("127.0.0.1", server.port))
            s.close()
        finally:
            server.close()


class TestOAuthCallbackServerPathFiltering:
    """Only /callback is processed; other paths get 404."""

    def test_callback_path_accepted(self):
        server = _mock_callback_server()
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"http://127.0.0.1:{server.port}/callback?code=abc&state=xyz")
            assert resp.status == 200
            # Wait a moment for the handler to write result
            import time
            time.sleep(0.3)
            assert server._result["auth_code"] == "abc"
            assert server._result["state"] == "xyz"
        finally:
            server.close()

    def test_non_callback_path_rejected(self):
        server = _mock_callback_server()
        try:
            import urllib.request
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"http://127.0.0.1:{server.port}/favicon.ico")
            assert exc_info.value.code == 404
            # Server must still be alive — /favicon didn't consume the slot
            resp = urllib.request.urlopen(f"http://127.0.0.1:{server.port}/callback?code=test")
            assert resp.status == 200
            import time
            time.sleep(0.3)
            assert server._result["auth_code"] == "test"
        finally:
            server.close()
```

**Step 4: Run full test suite**
Run: `python -m pytest tests/tools/test_mcp_oauth.py tests/tools/test_mcp_oauth_manager.py -x -q 2>&1 | tail -10`

Expected: All tests pass

**Acceptance Criteria:**
1. All existing tests pass with adapted signatures
2. `test_server_bound_on_configure_return` — server is connectable when `_configure_callback_port` returns
3. `test_callback_path_accepted` — `/callback?code=...` returns 200
4. `test_non_callback_path_rejected` — `/favicon.ico` returns 404, server stays alive
5. Paste fallback tests work with `server._result` race pattern

---

## Task 5: Commit and verify final state

**Step 1: Run full MCP OAuth test suite**
Run: `python -m pytest tests/tools/test_mcp_oauth.py tests/tools/test_mcp_oauth_manager.py tests/tools/test_mcp_oauth_integration.py tests/tools/test_mcp_oauth_bidirectional.py -v 2>&1 | tail -20`

**Step 2: Check for any remaining `_oauth_port` reads in production code**
Run: `grep -n '_oauth_port' tools/mcp_oauth.py tools/mcp_oauth_manager.py`

Expected: Only in `_configure_callback_port` (setting legacy value) and `_wait_for_callback` (legacy fallback path).

**Step 3: Commit**
```bash
git add tools/mcp_oauth.py tools/mcp_oauth_manager.py tests/tools/test_mcp_oauth.py
git commit -m "fix(mcp-oauth): eliminate _oauth_port global, bind OAuth callback server first

Introduce OAuthCallbackServer that binds HTTServer at construction,
then polls for the callback result. Eliminates the TOCTOU gap between
_find_free_port() and HTTPServer bind, and isolates per-flow port
state so concurrent OAuth flows don't collide.

Changes:
- New OAuthCallbackServer class with bind-first semantics
- _configure_callback_port returns the server, starts it immediately
- _wait_for_callback accepts server param via functools.partial
- Path filtering: only /callback is processed
- Legacy _oauth_port kept for backward compat

Fixes: #34260 (PR 1/2)"
```

**Acceptance Criteria:**
1. All MCP OAuth tests pass (0 failures)
2. `_oauth_port` only referenced in backward-compat paths
3. No new imports break module loading
4. `python3 -c "from tools.mcp_oauth import OAuthCallbackServer"` succeeds
```

