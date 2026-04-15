"""Shared fixtures for the hermes-agent test suite."""

import asyncio
import logging
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ─── Early HERMES_HOME isolation (runs at conftest import, before any test
# module is collected — so setup_logging() calls triggered by test-module
# imports will use the tmp dir, not ~/.hermes/logs/). ────────────────────
#
# Why this matters: setup_logging() caches its handlers on the root logger
# keyed by baseFilename. Once those handlers attach (e.g. because importing
# a test module pulled in run_agent which called setup_logging), later
# per-test monkeypatches to HERMES_HOME can no longer redirect the files.
# Without this guard, pytest pollutes ~/.hermes/logs/errors.log and triggers
# false alerts in the hermes-monitor cron.
_TEST_HERMES_HOME = Path(
    tempfile.mkdtemp(
        prefix=f"hermes-pytest-{os.getpid()}-"
               f"{os.environ.get('PYTEST_XDIST_WORKER', 'main')}-"
    )
)
(_TEST_HERMES_HOME / "logs").mkdir(exist_ok=True)
(_TEST_HERMES_HOME / "sessions").mkdir(exist_ok=True)
(_TEST_HERMES_HOME / "cron").mkdir(exist_ok=True)
os.environ["HERMES_HOME"] = str(_TEST_HERMES_HOME)

# Also purge any HERMES_* env vars leaking from the parent process (e.g. when
# pytest is invoked from inside a running Hermes gateway shell). These leak
# session identity into tests that explicitly assert "clean slate" behavior.
# Allowlist the isolation var we just set; drop everything else.
_HERMES_ENV_ALLOWLIST = {"HERMES_HOME"}
for _k in [k for k in os.environ if k.startswith("HERMES_")]:
    if _k not in _HERMES_ENV_ALLOWLIST:
        os.environ.pop(_k, None)

# Defensive: if any pre-existing root handler was somehow attached to the
# real ~/.hermes/logs/ before conftest loaded (shouldn't happen with proper
# pytest invocation, but belt-and-suspenders), detach it. We only remove
# handlers pointing at the REAL logs dir — never handlers pointing at our
# test tmp, and never non-file handlers (stdout etc).
_REAL_LOGS_DIR = str((Path.home() / ".hermes" / "logs").resolve())
for _h in list(logging.getLogger().handlers):
    _base = getattr(_h, "baseFilename", "") or ""
    if _base and _base.startswith(_REAL_LOGS_DIR):
        logging.getLogger().removeHandler(_h)
        try:
            _h.close()
        except Exception:
            pass


def pytest_sessionfinish(session, exitstatus):
    """Clean up the session-level test HERMES_HOME at pytest exit."""
    # Close any RotatingFileHandler pointing at our tmp first, so Windows
    # doesn't complain about locked files during rmtree.
    for h in list(logging.getLogger().handlers):
        base = getattr(h, "baseFilename", "") or ""
        if base.startswith(str(_TEST_HERMES_HOME)):
            logging.getLogger().removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    shutil.rmtree(_TEST_HERMES_HOME, ignore_errors=True)


# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME to a per-test temp dir.

    Note: this only isolates code paths that read HERMES_HOME at runtime
    (sessions/, cron/, memories/, etc.). Logging handlers were locked to
    the session-level tmp at conftest import — see the module-level block
    above. Tests that need fresh logging must explicitly re-init with
    setup_logging(hermes_home=..., force=True).
    """
    fake_home = tmp_path / "hermes_test"
    fake_home.mkdir()
    (fake_home / "sessions").mkdir()
    (fake_home / "cron").mkdir()
    (fake_home / "memories").mkdir()
    (fake_home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    # Reset plugin singleton so tests don't leak plugins from ~/.hermes/plugins/
    try:
        import hermes_cli.plugins as _plugins_mod
        monkeypatch.setattr(_plugins_mod, "_plugin_manager", None)
    except Exception:
        pass
    # Tests should not inherit the agent's current gateway/messaging surface.
    # Individual tests that need gateway behavior set these explicitly.
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    # Avoid making real calls during tests if this key is set in the env files
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a temporary directory that is cleaned up automatically."""
    return tmp_path


@pytest.fixture()
def mock_config():
    """Return a minimal hermes config dict suitable for unit tests."""
    return {
        "model": "test/mock-model",
        "toolsets": ["terminal", "file"],
        "max_turns": 10,
        "terminal": {
            "backend": "local",
            "cwd": "/tmp",
            "timeout": 30,
        },
        "compression": {"enabled": False},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "command_allowlist": [],
    }


# ── Global test timeout ─────────────────────────────────────────────────────
# Kill any individual test that takes longer than 30 seconds.
# Prevents hanging tests (subprocess spawns, blocking I/O) from stalling the
# entire test suite.

def _timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded 30 second timeout")

@pytest.fixture(autouse=True)
def _ensure_current_event_loop(request):
    """Provide a default event loop for sync tests that call get_event_loop().

    Python 3.11+ no longer guarantees a current loop for plain synchronous tests.
    A number of gateway tests still use asyncio.get_event_loop().run_until_complete(...).
    Ensure they always have a usable loop without interfering with pytest-asyncio's
    own loop management for @pytest.mark.asyncio tests.
    """
    if request.node.get_closest_marker("asyncio") is not None:
        yield
        return

    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        loop = None

    created = loop is None or loop.is_closed()
    if created:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        yield
    finally:
        if created and loop is not None:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)


@pytest.fixture(autouse=True)
def _enforce_test_timeout():
    """Kill any individual test that takes longer than 30 seconds.
    SIGALRM is Unix-only; skip on Windows."""
    if sys.platform == "win32":
        yield
        return
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(30)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)
