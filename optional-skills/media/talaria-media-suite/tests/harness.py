"""Shared E2E harness for the talaria test suite.

Offline by construction: a dead-proxy egress trap points every proxy var
at closed loopback port 9, with no_proxy covering 127.0.0.1/localhost so
loopback traffic reaches the mocks while ANY accidental real-egress
request dies instantly. HOME is redirected into a throwaway sandbox so no
test ever touches the real user config. Dummy keys are shared with the
mock server by construction (it reads the same env at start).
"""

import json
import os
import subprocess
import sys
import os
import tempfile
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
PY = TESTS.parent
# Per-run request log OUTSIDE the repo tree — running the suite must
# never dirty tracked files. Override with TALARIA_MOCK_LOG.
MOCK_LOG = Path(os.environ.get(
    "TALARIA_MOCK_LOG",
    Path(tempfile.gettempdir()) / "talaria-mock-request-log.jsonl"))

DUMMY_CREDS = {
    "SONARR_KEY": "test-sonarr-key",
    "RADARR_KEY": "test-radarr-key",
    "LIDARR_KEY": "test-lidarr-key",
    "PROWLARR_KEY": "test-prowlarr-key",
    "BAZARR_KEY": "test-bazarr-key",
    "OVERSEERR_KEY": "test-overseerr-key",
    "JELLYFIN_KEY": "test-jellyfin-key",
}

DEAD_PROXY = {
    "http_proxy": "http://127.0.0.1:9",
    "https_proxy": "http://127.0.0.1:9",
    "all_proxy": "http://127.0.0.1:9",
    "HTTP_PROXY": "http://127.0.0.1:9",
    "HTTPS_PROXY": "http://127.0.0.1:9",
    "ALL_PROXY": "http://127.0.0.1:9",
    "no_proxy": "127.0.0.1,localhost",
    "NO_PROXY": "127.0.0.1,localhost",
}

SANDBOX = None
MOCK = None


def make_env(**overrides):
    env = os.environ.copy()
    env.update(DEAD_PROXY)
    env.update(DUMMY_CREDS)
    env["TALARIA_HOST"] = "127.0.0.1"
    env["HOME"] = str(SANDBOX)
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def run_cli(argv, timeout=60, env=None, cwd=None):
    """Run `python3 -m talaria <argv>` under the trap env."""
    return subprocess.run(
        [sys.executable, "-m", "talaria"] + list(argv),
        capture_output=True, text=True, timeout=timeout,
        cwd=str(cwd or (PY)), env=env or make_env())


def start_mock():
    global MOCK
    if MOCK is not None and MOCK.poll() is None:
        return
    MOCK_LOG.parent.mkdir(parents=True, exist_ok=True)
    MOCK_LOG.write_text("")
    MOCK = subprocess.Popen(
        [sys.executable, str(TESTS / "mock_server.py"), "--log",
         str(MOCK_LOG)],
        env=make_env())
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            for line in MOCK_LOG.read_text().splitlines():
                if json.loads(line).get("event") == "ready":
                    return
        except (json.JSONDecodeError, ValueError, OSError):
            pass
        time.sleep(0.2)
    raise RuntimeError("mock server did not become ready")


def stop_mock():
    global MOCK
    if MOCK is None:
        return
    MOCK.terminate()
    try:
        MOCK.wait(timeout=10)
    except subprocess.TimeoutExpired:
        MOCK.kill()
    MOCK = None


def mock_records():
    """Parsed JSONL request log entries (ready event excluded)."""
    if not MOCK_LOG.is_file():
        return []
    out = []
    for line in MOCK_LOG.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("event") == "ready":
            continue
        out.append(rec)
    return out


def requests_to(port=None, path=None, method=None):
    """Filtered request records from the mock log."""
    return [r for r in mock_records()
            if (port is None or r.get("port") == port)
            and (path is None or (r.get("path") or "").startswith(path))
            and (method is None or r.get("method") == method)]


def post_bodies(port=None, path=None):
    """Decoded JSON bodies of POST/PUT/DELETE requests."""
    bodies = []
    for rec in requests_to(port=port, path=path, method=None):
        if rec.get("method") in ("POST", "PUT", "DELETE") and rec.get("body"):
            try:
                bodies.append(json.loads(rec["body"]))
            except ValueError:
                bodies.append(rec["body"])
    return bodies


def setup_sandbox():
    global SANDBOX
    SANDBOX = Path(tempfile.mkdtemp(prefix="talaria-test-"))


def teardown_sandbox():
    global SANDBOX
    if SANDBOX is None:
        return
    import shutil
    shutil.rmtree(SANDBOX, ignore_errors=True)
    SANDBOX = None
