"""Shared fixtures for the hermes-agent test suite."""

import asyncio
import importlib
import importlib.util
import os
import signal
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _dependency_available(module_name: str) -> bool:
    """Return True when the optional dependency can be imported for real."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _reload_optional_gateway_adapter(module_name: str, dependency: str, broken_predicate) -> None:
    """Reload a gateway adapter if a previous test imported it in fallback mode."""
    module = sys.modules.get(module_name)
    if module is None or not _dependency_available(dependency):
        return
    try:
        if broken_predicate(module):
            importlib.reload(module)
    except Exception:
        # Tests that intentionally simulate missing optional deps may leave
        # the module in a transient state. Ignore cleanup failures here and let
        # the next test control its own import path.
        pass


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME to a temp dir so tests never write to ~/.hermes/."""
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


@pytest.fixture(autouse=True)
def _reset_process_state(monkeypatch):
    """Clear process-global state that otherwise leaks across xdist workers."""
    for env_var in (
        "VIRTUAL_ENV",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "COPILOT_GITHUB_TOKEN",
        "COPILOT_API_BASE_URL",
        "CAMOFOX_URL",
    ):
        monkeypatch.delenv(env_var, raising=False)

    _reload_optional_gateway_adapter(
        "gateway.platforms.discord",
        "discord",
        lambda mod: getattr(mod, "DISCORD_AVAILABLE", True) is False or getattr(mod, "discord", object()) is None,
    )
    _reload_optional_gateway_adapter(
        "gateway.platforms.telegram",
        "telegram",
        lambda mod: getattr(mod, "TELEGRAM_AVAILABLE", True) is False or getattr(mod, "ChatType", object()) is None,
    )

    camofox = sys.modules.get("tools.browser_camofox")
    if camofox is not None:
        try:
            with camofox._sessions_lock:
                camofox._sessions.clear()
        except Exception:
            pass
        monkeypatch.setattr(camofox, "_vnc_url", None, raising=False)
        monkeypatch.setattr(camofox, "_vnc_url_checked", False, raising=False)

    checkpoint_manager = sys.modules.get("tools.checkpoint_manager")
    if checkpoint_manager is not None:
        try:
            from hermes_constants import get_hermes_home
            monkeypatch.setattr(
                checkpoint_manager,
                "CHECKPOINT_BASE",
                get_hermes_home() / "checkpoints",
                raising=False,
            )
        except Exception:
            pass


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
