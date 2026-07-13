#!/usr/bin/env python3
"""
Regression test for teknium1's review comments on PR #32331.

These tests guard against the two bugs found in `captcha_relay.py` and
`captcha_test.py` before the fix:

  Comment 2 — `captcha_relay.py:115`:
    "A token file left by any earlier run makes this condition false, so
     the advertised two-minute timeout never shuts down. The later result
     path also reads that stale file."

  Symptoms we test here:
    1. STALE_FILE_LEAKS_INTO_NEW_RUN: if /tmp/captcha_token.txt exists from a
       prior run, a fresh run must NOT return that stale token.
    2. TIMEOUT_FIRES_ON_STALE_FILE: the timeout must still fire when a stale
       file is present, otherwise a 2-minute wait becomes an indefinite wait.
    3. SHUTDOWN_IS_REQUEST_LOCAL: the timeout decision must be driven by a
       request-handler-set flag, not by the file's existence on disk.

Run:
    python3 scripts/test_no_stale_token.py

Expected: all tests pass.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _write_script_with_token_path(token_path: Path, timeout_seconds: int) -> Path:
    """Build a tiny standalone script that calls into _relay_common.run_relay
    but with a custom token path and a custom timeout. This isolates the test
    from the production entry-points while still exercising the same shared
    logic — which is exactly the surface teknium1's review targeted."""
    body = textwrap.dedent(
        f"""\
        import sys
        sys.path.insert(0, r"{HERE}")
        from _relay_common import run_relay

        def html_factory():
            return "<html><body>ok</body></html>"

        token = run_relay(
            html_factory=html_factory,
            banner="test banner",
            port=0,  # bind-any; we'll discover via the returned port below
            token_file=r"{token_path}",
            timeout_seconds={timeout_seconds},
        )
        sys.exit(0 if token else 1)
        """
    )
    # NOTE: port=0 isn't supported by run_relay's TCPServer() bind in this
    # version; we want a stable port for the test, so we patch by editing
    # the body — set port to a high random port instead.
    raise NotImplementedError("Use _start_relay_subprocess instead.")


def _start_relay_subprocess(token_path: Path, timeout_seconds: int, port: int) -> subprocess.Popen:
    """Launch run_relay in a subprocess with the given token_path and timeout."""
    # We use runpy so the same import path that the production scripts use is
    # exercised. This catches any drift if someone changes _relay_common.
    body = textwrap.dedent(
        f"""\
        import sys
        sys.path.insert(0, r"{HERE}")
        from _relay_common import run_relay

        def html_factory():
            return "<html><body>ok</body></html>"

        token = run_relay(
            html_factory=html_factory,
            banner="test banner",
            port={port},
            token_file=r"{token_path}",
            timeout_seconds={timeout_seconds},
        )
        # Exit 0 on token received, 2 on timeout. Distinct codes so the
        # test can tell them apart.
        if token is None:
            sys.exit(2)
        sys.exit(0)
        """
    )
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    f.write(body)
    f.close()
    return subprocess.Popen(
        [sys.executable, f.name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_server(port: int, deadline_s: float = 5.0) -> None:
    """Poll the local relay until it accepts a TCP connection, or fail."""
    import socket

    deadline = time.monotonic() + deadline_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError as e:
            last_err = e
            time.sleep(0.05)
    raise RuntimeError(
        f"relay server on :{port} did not come up within {deadline_s}s ({last_err})"
    )


def _get_free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #

class StaleTokenRegressionTests(unittest.TestCase):
    """Regression tests for teknium1's review on PR #32331.

    Each test is shaped to FAIL on the original (pre-fix) code and PASS on
    the fix. See the bug summary in this file's docstring.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="captcha-regress-")
        self.token_path = Path(self.tmpdir) / "captcha_token.txt"
        self.port = _get_free_port()

    def tearDown(self) -> None:
        # Clean up any leftover token file (in case the test wrote one).
        try:
            self.token_path.unlink()
        except FileNotFoundError:
            pass

    # ---- Test 1: stale file does not leak into a new run ----------------- #

    def test_stale_file_does_not_leak_into_new_run(self) -> None:
        """If a token file exists from a previous run, a fresh run that
        does NOT receive a solve must NOT return that stale token.

        On the broken code, run_relay reads `if os.path.exists(token_file)`
        AFTER serve_forever returns, so a leftover file makes it look like
        a solve happened. On the fix, the request-local flag
        `server.token_received` is what gates the read, not the file.
        """
        # 1. Plant a stale token file before the relay even starts.
        self.token_path.write_text(
            "03AFcWeA_STALE_TOKEN_FROM_PRIOR_RUN" + "_" * 2050
        )

        # 2. Start the relay with a SHORT timeout (we'll patch sleep, see below).
        # We use a 1-second timeout for test speed.
        proc = _start_relay_subprocess(
            token_path=self.token_path, timeout_seconds=1, port=self.port
        )

        try:
            _wait_for_server(self.port, deadline_s=5.0)
            # 3. Do NOT call /token. Just wait for timeout.
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            self.fail("relay subprocess did not exit after timeout")

        # 4. On the BROKEN code: subprocess exits 0 because stale file exists,
        #    and the printed "Result: {...}" contains the stale token.
        #    On the FIXED code: subprocess exits 2 (no token received) and no
        #    "Result:" line is printed.
        combined = (stdout or b"") + (stderr or b"")
        text = combined.decode("utf-8", errors="replace")
        self.assertIn("Timeout", text,
                      "expected the timeout path to fire; got:\n" + text)

        # The critical assertion: no "Result:" line containing the stale
        # token. (On the broken code this is exactly what happens.)
        self.assertNotIn("STALE_TOKEN_FROM_PRIOR_RUN", text,
                         "stale token leaked into new run output; this is the "
                         "exact bug teknium1 flagged at captcha_relay.py:115")

        # And the file on disk should be gone — clear_token_file runs on
        # startup, and the timeout path never wrote a new value.
        self.assertFalse(
            self.token_path.exists(),
            "stale file should have been cleared on startup and not "
            "recreated by the timeout path"
        )

    # ---- Test 2: timeout still fires when a stale file exists ------------ #

    def test_timeout_fires_even_when_stale_file_exists(self) -> None:
        """The 2-minute timeout must still shut the server down, regardless
        of whether a stale token file is present. On the broken code the
        `if not os.path.exists(TOKEN_FILE)` check is False when the stale
        file is present, so the timeout thread silently exits and the
        server blocks for the full TCP keep-alive timeout (or forever).
        """
        # Plant a stale file.
        self.token_path.write_text("stale-but-present")

        # Start with a 1-second timeout.
        started = time.monotonic()
        proc = _start_relay_subprocess(
            token_path=self.token_path, timeout_seconds=1, port=self.port
        )

        try:
            _wait_for_server(self.port, deadline_s=5.0)
            # No /token call. Wait for the subprocess to exit on its own.
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            self.fail(
                "relay did NOT exit within 15s after a 1-second timeout "
                "with a stale token file present. This is the exact bug "
                "teknium1 flagged: `if not os.path.exists(TOKEN_FILE)` is "
                "False when a stale file exists, so the timeout never shuts "
                "the server down."
            )

        elapsed = time.monotonic() - started
        # Generous upper bound: 1s timeout + 5s server-up + slack.
        self.assertLess(elapsed, 10.0,
                        f"server took {elapsed:.1f}s to exit; timeout should "
                        f"have fired within ~1s")
        self.assertEqual(proc.returncode, 2,
                         f"expected exit 2 (timeout/no-token), got {proc.returncode}")

    # ---- Test 3: /token POST does write the file (happy path) ----------- #

    def test_solve_writes_file_and_returns_token(self) -> None:
        """Sanity check: when /token is actually hit, the file is written
        and the subprocess exits 0 with the token in its output. This guards
        against an over-aggressive fix that breaks the happy path.
        """
        # No stale file this time — start clean.
        proc = _start_relay_subprocess(
            token_path=self.token_path, timeout_seconds=10, port=self.port
        )
        try:
            _wait_for_server(self.port, deadline_s=5.0)
            # Simulate the JS callback firing.
            token = "03AFcWeA" + "X" * 2090
            url = f"http://127.0.0.1:{self.port}/token?t={token}"
            with urllib.request.urlopen(url, timeout=5) as r:
                self.assertEqual(r.status, 200)
            # The handler spawns a shutdown thread; give it a moment.
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                self.fail("relay did not exit after /token")
        finally:
            if proc.poll() is None:
                proc.kill()

        text = (stdout or b"").decode("utf-8", errors="replace")
        self.assertIn(token, text,
                      "the token returned to /token must appear in the "
                      "subprocess output")
        self.assertIn("Result:", text,
                      "expected 'Result: {...}' line after /token; got:\n" + text)
        self.assertEqual(proc.returncode, 0,
                         f"expected exit 0 on successful solve, got {proc.returncode}")

    # ---- Test 4: token file is cleared on startup ----------------------- #

    def test_token_file_cleared_on_startup(self) -> None:
        """When the relay starts, it must remove any pre-existing token file.
        This is the upstream half of the fix: even if the timeout were also
        gated on file existence (it isn't, on the fix), clearing on startup
        means there is no stale data on disk to leak.
        """
        # Plant a stale file.
        self.token_path.write_text("garbage-from-prior-run")

        proc = _start_relay_subprocess(
            token_path=self.token_path, timeout_seconds=1, port=self.port
        )
        try:
            _wait_for_server(self.port, deadline_s=5.0)
            # Give the startup-clear a moment to fire before timeout.
            time.sleep(0.2)
            # By now the file should be gone (startup cleared it).
            self.assertFalse(
                self.token_path.exists(),
                "token file must be removed on relay startup; "
                "this is the first half of the fix."
            )
            # Let the timeout fire and exit.
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            self.fail("relay did not exit after timeout")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(StaleTokenRegressionTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())