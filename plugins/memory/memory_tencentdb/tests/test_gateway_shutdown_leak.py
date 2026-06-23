"""End-to-end tests for the "A mode" Gateway shutdown contract.

Background
----------
When the ``memory_tencentdb`` provider runs under hermes and the Gateway
is launched **by** the hermes process (Mode A — supervisor as parent),
``provider.shutdown()`` used to leave the Gateway subprocess running.
Because the supervisor spawns the Gateway with ``start_new_session=True``,
an un-shutdown Gateway is reparented to PID 1 and survives as an orphan.

Two concrete bugs fell out of that:

1. Orphan Gateway processes accumulate across hermes restarts.
2. The next hermes process's ``is_running()`` health-check sees the stale
   Gateway as healthy and *reuses it*, silently ignoring any config the
   user rotated between restarts (e.g. a new LLM API key installed via
   ``memory-tencentdb-ctl --hermes config llm``).

The fix: ``provider.shutdown()`` now calls ``supervisor.shutdown()``.
This test module locks that contract in.

Test suite layout
-----------------
* :class:`GatewayShutdownLeakTest`
  Core contract tests against a fake Python HTTP Gateway. Fast (≤ a few
  seconds), no Node/pnpm/tsx dependency, safe for CI. Covers:
    - ``test_provider_shutdown_should_stop_supervisor_gateway``
      Supervisor-owned Gateway **must** die on provider.shutdown().
    - ``test_external_gateway_is_not_killed``
      If the Gateway was already running when the provider attached
      (``ensure_running`` returns early without spawning), shutdown must
      **not** terminate it — we only own what we started.
    - ``test_second_provider_does_not_reuse_stale_gateway``
      End-to-end reproduction of the "stale LLM config" user report:
      provider-A starts a Gateway, shuts down, provider-B starts up;
      provider-B must not silently reuse the old Gateway.
* :class:`RealGatewayShutdownTest`
  Integration test against the actual Node Gateway under
  ``src/gateway/server.ts``. Validates graceful shutdown (SIGTERM-driven
  ``gateway.stop()`` runs, SQLite WAL is checkpointed so ``*-wal``/
  ``*-shm`` sidecars don't leak). Skipped by default because it requires
  a working ``pnpm``/``tsx`` toolchain and ~30s to start; opt in via
  ``TDAI_E2E_REAL_GATEWAY=1``.

Run directly::

    python3 hermes-plugin/memory/memory_tencentdb/tests/test_gateway_shutdown_leak.py

Or scope to one case::

    python3 hermes-plugin/memory/memory_tencentdb/tests/test_gateway_shutdown_leak.py \\
        GatewayShutdownLeakTest.test_external_gateway_is_not_killed
"""

from __future__ import annotations

import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

# tdai-memory-openclaw-plugin / hermes-plugin / memory / memory_tencentdb / tests / THIS FILE
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]
_HERMES_PLUGIN_ROOT = _PROJECT_ROOT / "hermes-plugin"


def _ensure_importable() -> Optional[str]:
    """Inject plugin + hermes-agent roots into ``sys.path``.

    Returns an informational skip reason if hermes-agent can't be located,
    otherwise None. Each test method checks the return value and skips if
    set, so the whole file still imports cleanly in environments without
    a hermes checkout.
    """
    if str(_HERMES_PLUGIN_ROOT) not in sys.path:
        sys.path.insert(0, str(_HERMES_PLUGIN_ROOT))

    hermes_agent_root = os.environ.get("HERMES_AGENT_ROOT")
    if not hermes_agent_root:
        candidate = _PROJECT_ROOT.parent / "hermes-agent"
        if candidate.is_dir():
            hermes_agent_root = str(candidate)
    if not hermes_agent_root or not pathlib.Path(hermes_agent_root, "agent").is_dir():
        return (
            "hermes-agent checkout not found — set HERMES_AGENT_ROOT to "
            "point at a sibling hermes-agent repo to run this test."
        )
    if hermes_agent_root not in sys.path:
        sys.path.insert(0, hermes_agent_root)
    return None


# ---------------------------------------------------------------------------
# Fake Gateway (Python HTTP server) helpers
# ---------------------------------------------------------------------------

def _pick_free_port() -> int:
    """Ask the kernel for an ephemeral port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_fake_gateway_script(tmpdir: pathlib.Path, pid_file: pathlib.Path) -> pathlib.Path:
    """Write a small Python HTTP server that impersonates the Gateway.

    Behaviour:
      * On startup, writes its own PID into ``pid_file`` and also a
        line-per-request log into ``<tmpdir>/gateway.trace`` so tests can
        assert which instance answered which request.
      * Serves ``GET /health`` with the Gateway's canonical JSON shape.
        Echoes the ``MEMORY_TENCENTDB_LLM_API_KEY`` env var back in a
        ``fingerprint`` field so "stale config reuse" tests can see which
        instance answered.
      * SIGTERM handler: remove the pid file and exit cleanly — lets us
        distinguish "supervisor sent SIGTERM" from "orphaned, still up".
    """
    script = tmpdir / "fake_gateway.py"
    trace = tmpdir / "gateway.trace"
    script.write_text(textwrap.dedent(
        f"""\
        import hashlib, json, os, signal, sys
        from http.server import BaseHTTPRequestHandler, HTTPServer

        PID_FILE = {str(pid_file)!r}
        TRACE = {str(trace)!r}
        PORT = int(os.environ["MEMORY_TENCENTDB_GATEWAY_PORT"])

        # Stamp startup so tests know this is the correct instance.
        FINGERPRINT = hashlib.sha1(
            os.environ.get("MEMORY_TENCENTDB_LLM_API_KEY", "").encode()
        ).hexdigest()[:12]

        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        with open(TRACE, "a", encoding="utf-8") as f:
            f.write(f"start pid={{os.getpid()}} fp={{FINGERPRINT}}\\n")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    body = json.dumps({{
                        "status": "ok",
                        "version": "fake-v1",
                        "uptime": 1,
                        "fingerprint": FINGERPRINT,
                        "stores": {{
                            "vectorStore": True,
                            "embeddingService": True,
                        }},
                    }}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    with open(TRACE, "a", encoding="utf-8") as f:
                        f.write(f"GET /health pid={{os.getpid()}} fp={{FINGERPRINT}}\\n")
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, fmt, *args):
                pass

        def _term(_signum, _frame):
            try:
                os.unlink(PID_FILE)
            except OSError:
                pass
            with open(TRACE, "a", encoding="utf-8") as f:
                f.write(f"stop pid={{os.getpid()}}\\n")
            sys.exit(0)

        signal.signal(signal.SIGTERM, _term)
        signal.signal(signal.SIGINT, _term)

        HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
        """
    ))
    return script


def _pid_alive(pid: int) -> bool:
    """Return True if the OS says this pid is still a live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pid_file(pid_file: pathlib.Path, timeout: float = 5.0) -> int:
    """Poll until the fake gateway writes its pid file; return the pid."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pid_file.exists():
            raw = pid_file.read_text().strip()
            if raw:
                return int(raw)
        time.sleep(0.05)
    raise TimeoutError(f"fake gateway did not write {pid_file} within {timeout}s")


def _wait_until_dead(pid: int, timeout: float = 5.0) -> bool:
    """Poll up to ``timeout`` seconds for the pid to disappear."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return False


def _kill_if_alive(pid: int) -> None:
    """Best-effort SIGTERM→SIGKILL for cleanup paths."""
    if not _pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.2)
        if _pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _set_env(overrides: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Apply env overrides, returning a restore dict."""
    prior: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return prior


def _restore_env(prior: Dict[str, Optional[str]]) -> None:
    for k, v in prior.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Core contract tests — against fake Python HTTP Gateway
# ---------------------------------------------------------------------------

class GatewayShutdownLeakTest(unittest.TestCase):
    """Supervisor lifecycle contract (fast; no Node dependency)."""

    def setUp(self) -> None:
        skip = _ensure_importable()
        if skip:
            self.skipTest(skip)
        self._tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="tdai-shutdown-leak-"))
        self._pid_file = self._tmpdir / "gateway.pid"
        self._fake_script = _make_fake_gateway_script(self._tmpdir, self._pid_file)
        self._rogue_pids: List[int] = []

    def tearDown(self) -> None:
        if self._pid_file.exists():
            try:
                pid = int(self._pid_file.read_text().strip())
            except Exception:
                pid = 0
            if pid:
                _kill_if_alive(pid)
        for pid in self._rogue_pids:
            _kill_if_alive(pid)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # -- utilities ----------------------------------------------------------

    def _fake_gateway_cmd(self) -> str:
        return f"{sys.executable} {self._fake_script}"

    def _spawn_external_gateway(self, port: int, api_key: str = "") -> int:
        """Start a fake Gateway *outside* the supervisor's control.

        Simulates "Gateway already running when provider attaches" —
        e.g. started manually by the user or by a previous process that
        legitimately left it behind.
        """
        env = os.environ.copy()
        env["MEMORY_TENCENTDB_GATEWAY_PORT"] = str(port)
        if api_key:
            env["MEMORY_TENCENTDB_LLM_API_KEY"] = api_key
        proc = subprocess.Popen(
            [sys.executable, str(self._fake_script)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # wait for it to come up
        pid = _wait_for_pid_file(self._pid_file, timeout=8.0)
        self.assertEqual(pid, proc.pid)
        self._rogue_pids.append(pid)
        return pid

    # -- tests --------------------------------------------------------------

    def test_provider_shutdown_should_stop_supervisor_gateway(self) -> None:
        """A-mode contract: Gateway we started MUST die on shutdown()."""
        from memory.memory_tencentdb import MemoryTencentdbProvider

        port = _pick_free_port()
        prior = _set_env({
            "MEMORY_TENCENTDB_GATEWAY_HOST": "127.0.0.1",
            "MEMORY_TENCENTDB_GATEWAY_PORT": str(port),
            "MEMORY_TENCENTDB_GATEWAY_CMD": self._fake_gateway_cmd(),
        })
        try:
            provider = MemoryTencentdbProvider()
            provider.initialize(session_id="leak-test-session", user_id="tester")

            pid = _wait_for_pid_file(self._pid_file, timeout=8.0)
            self.assertTrue(_pid_alive(pid))

            provider.shutdown()

            died = _wait_until_dead(pid, timeout=3.0)
            self.assertTrue(
                died,
                f"Gateway pid={pid} still alive 3s after provider.shutdown(); "
                "supervisor teardown did not propagate.",
            )
        finally:
            _restore_env(prior)

    def test_external_gateway_is_not_killed(self) -> None:
        """Symmetry contract: don't kill what we didn't start.

        If the Gateway was already serving on the configured port when
        the provider attached, ``supervisor.ensure_running()`` returns
        without spawning and leaves ``_process = None``. In that case
        ``shutdown()`` must be a no-op for the Gateway — killing it would
        break anyone else already using it.
        """
        from memory.memory_tencentdb import MemoryTencentdbProvider

        port = _pick_free_port()
        external_pid = self._spawn_external_gateway(port)

        prior = _set_env({
            "MEMORY_TENCENTDB_GATEWAY_HOST": "127.0.0.1",
            "MEMORY_TENCENTDB_GATEWAY_PORT": str(port),
            # Supply a CMD too — we want to prove the supervisor takes the
            # is_running() fast path and *doesn't* spawn a second copy.
            "MEMORY_TENCENTDB_GATEWAY_CMD": self._fake_gateway_cmd(),
        })
        try:
            provider = MemoryTencentdbProvider()
            provider.initialize(session_id="external-gw-session", user_id="tester")

            # Sanity: the external Gateway is still the pid-file holder.
            pid = int(self._pid_file.read_text().strip())
            self.assertEqual(
                pid, external_pid,
                "Supervisor unexpectedly started a second Gateway; "
                "is_running() fast path must be taken when a healthy "
                "Gateway is already serving the port.",
            )

            provider.shutdown()

            # External gateway must survive.
            time.sleep(0.5)
            self.assertTrue(
                _pid_alive(external_pid),
                f"External Gateway pid={external_pid} was killed by "
                "provider.shutdown(); supervisor must only terminate "
                "processes it started itself.",
            )
        finally:
            _restore_env(prior)

    def test_second_provider_does_not_reuse_stale_gateway(self) -> None:
        """Stale-config reproduction.

        Mirrors the user report: rotate ``MEMORY_TENCENTDB_LLM_API_KEY``
        between two hermes runs. The second provider must end up with a
        Gateway whose env has the *new* key — i.e. a brand-new process,
        not the first provider's leftover. The fake Gateway publishes
        ``fingerprint = sha1(api_key)[:12]`` over ``/health`` so we can
        tell the two apart by a single HTTP call.
        """
        from memory.memory_tencentdb import MemoryTencentdbProvider
        from memory.memory_tencentdb.client import MemoryTencentdbSdkClient

        port = _pick_free_port()

        def _health_fingerprint() -> str:
            client = MemoryTencentdbSdkClient(
                base_url=f"http://127.0.0.1:{port}", timeout=2,
            )
            return client.health(timeout=2).get("fingerprint", "")

        prior = _set_env({
            "MEMORY_TENCENTDB_GATEWAY_HOST": "127.0.0.1",
            "MEMORY_TENCENTDB_GATEWAY_PORT": str(port),
            "MEMORY_TENCENTDB_GATEWAY_CMD": self._fake_gateway_cmd(),
            "MEMORY_TENCENTDB_LLM_API_KEY": "old-key-AAA",
        })
        try:
            # --- first provider run (the "before rotation" hermes) ---
            provider_a = MemoryTencentdbProvider()
            provider_a.initialize(session_id="sess-a", user_id="tester")
            pid_a = _wait_for_pid_file(self._pid_file, timeout=8.0)
            fp_a = _health_fingerprint()
            self.assertTrue(fp_a, "first Gateway did not publish a fingerprint")

            provider_a.shutdown()
            self.assertTrue(
                _wait_until_dead(pid_a, timeout=3.0),
                "first Gateway still alive after provider_a.shutdown() — "
                "stale-config bug would reappear.",
            )

            # --- user rotates the LLM key between hermes restarts ---
            os.environ["MEMORY_TENCENTDB_LLM_API_KEY"] = "new-key-ZZZ"

            # --- second provider run (the "after rotation" hermes) ---
            provider_b = MemoryTencentdbProvider()
            provider_b.initialize(session_id="sess-b", user_id="tester")
            pid_b = _wait_for_pid_file(self._pid_file, timeout=8.0)
            fp_b = _health_fingerprint()

            self.assertNotEqual(
                pid_a, pid_b,
                "provider_b reused provider_a's Gateway pid — the "
                "orphan survived shutdown and was picked up by "
                "is_running() (classic stale-config bug).",
            )
            self.assertNotEqual(
                fp_a, fp_b,
                "provider_b's Gateway still reports the old key "
                f"fingerprint ({fp_a}); the new env never reached a "
                "fresh process.",
            )

            provider_b.shutdown()
            self.assertTrue(_wait_until_dead(pid_b, timeout=3.0))
        finally:
            _restore_env(prior)


# ---------------------------------------------------------------------------
# Integration test — against the real Node Gateway (opt-in)
# ---------------------------------------------------------------------------

class RealGatewayShutdownTest(unittest.TestCase):
    """Opt-in integration test for graceful shutdown of the real Gateway.

    Enabled only when ``TDAI_E2E_REAL_GATEWAY=1`` is set, because it:
      * depends on ``pnpm`` / ``tsx`` being available on PATH,
      * costs ~10-30s (Node cold start + first /health),
      * writes to a temp SQLite data dir.

    Verifies two properties that matter beyond "pid dies":
      1. ``gateway.stop()`` actually ran — SIGTERM was delivered and the
         in-process shutdown handler finished before ``process.exit(0)``.
         Proxy signal: SQLite files are in a clean state (no leftover
         ``*-wal`` with unflushed bytes).
      2. The process exits within a reasonable grace window.
    """

    def setUp(self) -> None:
        if os.environ.get("TDAI_E2E_REAL_GATEWAY") != "1":
            self.skipTest(
                "Real-Gateway test skipped; set TDAI_E2E_REAL_GATEWAY=1 "
                "to enable (requires pnpm + tsx on PATH)."
            )
        skip = _ensure_importable()
        if skip:
            self.skipTest(skip)

        server_ts = _PROJECT_ROOT / "src" / "gateway" / "server.ts"
        if not server_ts.is_file():
            self.skipTest(f"src/gateway/server.ts not found at {server_ts}")
        if shutil.which("pnpm") is None:
            self.skipTest("pnpm not on PATH")

        self._tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="tdai-real-gw-"))
        self._data_dir = self._tmpdir / "data"
        self._data_dir.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_real_gateway_graceful_shutdown(self) -> None:
        from memory.memory_tencentdb import MemoryTencentdbProvider

        port = _pick_free_port()
        gateway_cmd = (
            f"sh -c 'cd {_PROJECT_ROOT} && exec pnpm exec tsx src/gateway/server.ts'"
        )

        prior = _set_env({
            "MEMORY_TENCENTDB_GATEWAY_HOST": "127.0.0.1",
            "MEMORY_TENCENTDB_GATEWAY_PORT": str(port),
            "MEMORY_TENCENTDB_GATEWAY_CMD": gateway_cmd,
            # The supervisor exports MEMORY_TENCENTDB_GATEWAY_{HOST,PORT}
            # into the child env, but ``src/gateway/config.ts`` currently
            # reads ``TDAI_GATEWAY_{HOST,PORT}``. Export both so this test
            # is agnostic to that mismatch (which is tracked separately).
            "TDAI_GATEWAY_HOST": "127.0.0.1",
            "TDAI_GATEWAY_PORT": str(port),
            "TDAI_DATA_DIR": str(self._data_dir),
            # Supply a placeholder LLM key: /health doesn't need it, but
            # unset keys make the L1 extractor log loud errors. A fake
            # key keeps the log clean and has no effect on the shutdown
            # path we're actually testing.
            "TDAI_LLM_API_KEY": "sk-test-placeholder-not-used",
            "MEMORY_TENCENTDB_LLM_API_KEY": "sk-test-placeholder-not-used",
        })
        try:
            provider = MemoryTencentdbProvider()
            provider.initialize(session_id="real-gw-sess", user_id="tester")

            # Fail loudly if the Gateway didn't actually come up — otherwise
            # a failed startup would mask the shutdown assertions below and
            # let a regression slip through. Surface the stderr log tail
            # (same location the supervisor uses) to make diagnosis easy.
            if not provider._gateway_available:  # noqa: SLF001 (test access)
                log_path = pathlib.Path(
                    os.environ.get("HOME", "") or "/",
                    ".hermes", "logs", "memory_tencentdb", "gateway.stderr.log",
                )
                tail = ""
                if log_path.is_file():
                    data = log_path.read_bytes()
                    tail = data[-2048:].decode("utf-8", errors="replace")
                self.fail(
                    "real Node Gateway failed to become healthy; cannot "
                    f"test shutdown. Recent stderr:\n{tail}"
                )

            # The supervisor stores the Popen object; reach in (test-only)
            # to grab the pid so we can watch it across shutdown.
            supervisor = provider._supervisor  # noqa: SLF001 (test access)
            self.assertIsNotNone(supervisor, "supervisor must be set after initialize()")
            proc = supervisor._process  # noqa: SLF001
            self.assertIsNotNone(
                proc,
                "real Node Gateway was expected to be spawned by the supervisor; "
                "got None — did the health check fail?",
            )
            pid = proc.pid

            t0 = time.monotonic()
            provider.shutdown()
            elapsed = time.monotonic() - t0

            self.assertTrue(
                _wait_until_dead(pid, timeout=12.0),
                f"real Node Gateway pid={pid} did not exit within 12s of "
                "SIGTERM — graceful shutdown path hung.",
            )
            # Graceful stop should typically finish well under the 10s
            # supervisor timeout; flag long waits so regressions are loud.
            self.assertLess(
                elapsed, 10.0,
                f"provider.shutdown() took {elapsed:.1f}s — suspiciously "
                "close to the SIGKILL fallback. Check gateway.stop() for "
                "blocking work.",
            )

            # Graceful-exit witness: no stray SQLite WAL/SHM should remain
            # under the data dir. If the Gateway was SIGKILL'd mid-write,
            # these sidecars would be left behind with uncommitted bytes.
            leftovers = sorted(
                p for p in self._data_dir.rglob("*")
                if p.suffix in (".db-wal", ".db-shm")
            )
            self.assertEqual(
                leftovers, [],
                f"found leftover SQLite sidecars after graceful shutdown: "
                f"{[str(p) for p in leftovers]}",
            )
        finally:
            _restore_env(prior)


    def test_wal_checkpoint_after_capture_and_sigterm(self) -> None:
        """Write data via capture(), then SIGTERM — graceful close verified.

        End-to-end proof that ``gateway.stop()`` actually runs (not just
        "pid disappears") when the supervisor sends SIGTERM:

          1. Start a fresh real Node Gateway pointed at a temp data dir.
          2. Send several ``/capture`` calls to produce L0 data.
          3. Confirm data was actually written to disk (JSONL and/or .db).
          4. ``provider.shutdown()`` → SIGTERM → ``gateway.stop()`` →
             ``core.destroy()`` → ``vectorStore.close()`` (which runs
             an implicit ``PRAGMA wal_checkpoint``).
          5. Assert the process exited **cleanly** (exit code 0 via
             SIGTERM handler, not 137 from SIGKILL).
          6. Assert shutdown finished well under the 10s SIGKILL fallback.
          7. If any ``.db`` files exist, assert no dirty WAL / SHM remain.
          8. Confirm JSONL data files are intact (non-empty, valid JSON
             lines) — proves L0 writes were fully flushed.

        Note: when ``sqlite-vec`` is not available, VectorStore enters
        degraded mode and L0 goes through JSONL only. The test adapts:
        it always checks JSONL; WAL assertions only fire when ``.db``
        files actually exist.
        """
        from memory.memory_tencentdb import MemoryTencentdbProvider
        from memory.memory_tencentdb.client import MemoryTencentdbSdkClient
        import json as _json

        port = _pick_free_port()
        gateway_cmd = (
            f"sh -c 'cd {_PROJECT_ROOT} && exec pnpm exec tsx src/gateway/server.ts'"
        )

        prior = _set_env({
            "MEMORY_TENCENTDB_GATEWAY_HOST": "127.0.0.1",
            "MEMORY_TENCENTDB_GATEWAY_PORT": str(port),
            "MEMORY_TENCENTDB_GATEWAY_CMD": gateway_cmd,
            "TDAI_GATEWAY_HOST": "127.0.0.1",
            "TDAI_GATEWAY_PORT": str(port),
            "TDAI_DATA_DIR": str(self._data_dir),
            "TDAI_LLM_API_KEY": "sk-test-placeholder-not-used",
            "MEMORY_TENCENTDB_LLM_API_KEY": "sk-test-placeholder-not-used",
        })
        try:
            provider = MemoryTencentdbProvider()
            provider.initialize(session_id="wal-ckpt-sess", user_id="wal-tester")

            if not provider._gateway_available:  # noqa: SLF001
                log_path = pathlib.Path(
                    os.environ.get("HOME", "") or "/",
                    ".hermes", "logs", "memory_tencentdb", "gateway.stderr.log",
                )
                tail = ""
                if log_path.is_file():
                    data = log_path.read_bytes()
                    tail = data[-2048:].decode("utf-8", errors="replace")
                self.fail(
                    "real Node Gateway failed to become healthy; cannot "
                    f"test WAL checkpoint. Recent stderr:\n{tail}"
                )

            supervisor = provider._supervisor  # noqa: SLF001
            proc = supervisor._process  # noqa: SLF001
            self.assertIsNotNone(proc, "Gateway process must be spawned")
            pid = proc.pid

            # ---- Step 2: write data via /capture ----
            client = MemoryTencentdbSdkClient(
                base_url=f"http://127.0.0.1:{port}", timeout=10,
            )
            n_captures = 5
            for i in range(n_captures):
                try:
                    client.capture(
                        user_content=f"Test message {i}: the quick brown fox",
                        assistant_content=f"Acknowledged message {i}.",
                        session_key="wal-ckpt-sess",
                        user_id="wal-tester",
                    )
                except Exception:
                    # capture() may partially fail (e.g. LLM extraction) but
                    # L0 write still happens before extraction kicks in.
                    pass

            # Give the Gateway a moment to flush writes.
            time.sleep(0.5)

            # ---- Step 3: confirm data was written to disk ----
            # JSONL (always present, even when VectorStore is degraded):
            jsonl_files = sorted(self._data_dir.rglob("*.jsonl"))
            self.assertTrue(
                len(jsonl_files) > 0,
                f"expected at least one .jsonl file under {self._data_dir} "
                f"after {n_captures} capture() calls.",
            )
            total_lines_before = 0
            for jf in jsonl_files:
                lines = [l for l in jf.read_text().splitlines() if l.strip()]
                total_lines_before += len(lines)
                # Validate each line is parseable JSON.
                for idx, line in enumerate(lines):
                    try:
                        _json.loads(line)
                    except _json.JSONDecodeError:
                        self.fail(
                            f"invalid JSON on line {idx+1} of {jf}: {line[:120]}"
                        )
            self.assertGreater(
                total_lines_before, 0,
                "JSONL files exist but contain no data lines.",
            )

            # .db files (only present when sqlite-vec loaded successfully):
            db_files = sorted(self._data_dir.rglob("*.db"))
            has_sqlite = len(db_files) > 0

            # ---- Step 4: SIGTERM via provider.shutdown() ----
            # Grab a reference to the Popen *before* supervisor.shutdown()
            # sets it to None, so we can check returncode afterwards.
            popen_ref = proc

            t0 = time.monotonic()
            provider.shutdown()
            elapsed = time.monotonic() - t0

            # ---- Step 5: verify clean exit (SIGTERM handler ran) ----
            self.assertTrue(
                _wait_until_dead(pid, timeout=12.0),
                f"real Node Gateway pid={pid} did not exit within 12s.",
            )

            # returncode semantics:
            #   0        → Node SIGTERM handler ran and called process.exit(0)
            #   -15      → SIGTERM killed the process directly (normal for
            #               multi-layer launchers like ``pnpm exec tsx``: the
            #               supervisor's terminate() hits pnpm, which exits on
            #               signal; tsx/node children then exit as a cascade)
            #   -9       → SIGKILL (supervisor had to force-kill after 10s
            #               timeout — that's a regression)
            #   positive → unexpected crash exit code
            rc = popen_ref.returncode
            self.assertIsNotNone(rc, "process should have exited")
            self.assertNotEqual(
                rc, -9,
                "Gateway was SIGKILL'd (exit code -9) — the SIGTERM path "
                "failed to terminate within the supervisor's 10s timeout. "
                "Graceful shutdown is broken.",
            )
            # For direct-node launches rc==0 means the handler ran. For
            # pnpm-wrapped launches rc==-15 is expected (pnpm doesn't trap
            # SIGTERM). Both are acceptable; anything else is suspicious.
            self.assertIn(
                rc, (0, -15, -2),  # 0=handler, -15=SIGTERM, -2=SIGINT
                f"Gateway exited with unexpected code {rc}. Expected 0 "
                "(graceful handler) or -15 (signal). Investigate.",
            )

            # ---- Step 6: timing ----
            self.assertLess(
                elapsed, 10.0,
                f"provider.shutdown() took {elapsed:.1f}s — close to the "
                "SIGKILL fallback; gateway.stop() may be blocked.",
            )

            # ---- Step 7: WAL/SHM cleanliness (only when .db exists) ----
            if has_sqlite:
                shm_leftovers = sorted(self._data_dir.rglob("*.db-shm"))
                self.assertEqual(
                    shm_leftovers, [],
                    f"SHM files should not survive graceful shutdown: "
                    f"{[str(p) for p in shm_leftovers]}",
                )

                dirty_wals = sorted(
                    f for f in self._data_dir.rglob("*.db-wal")
                    if f.stat().st_size > 0
                )
                self.assertEqual(
                    dirty_wals, [],
                    f"non-empty WAL files found after graceful shutdown — "
                    f"wal_checkpoint was NOT completed: "
                    f"{[(str(f), f.stat().st_size) for f in dirty_wals]}",
                )

            # ---- Step 8: JSONL integrity post-shutdown ----
            # The same JSONL files should still be intact and no smaller
            # (gateway.stop → core.destroy should not truncate them).
            total_lines_after = 0
            for jf in jsonl_files:
                if jf.exists():
                    lines = [l for l in jf.read_text().splitlines() if l.strip()]
                    total_lines_after += len(lines)
            self.assertGreaterEqual(
                total_lines_after, total_lines_before,
                f"JSONL data shrank after shutdown "
                f"(before={total_lines_before}, after={total_lines_after}); "
                f"graceful shutdown may have corrupted L0 data.",
            )
        finally:
            _restore_env(prior)


if __name__ == "__main__":
    unittest.main(verbosity=2)
