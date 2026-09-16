"""The suite must ignore the Hermes runtime env of the session that launched it.

Regression cover for the Desktop/gateway env leak. When pytest is run from
inside a Hermes session — the Desktop terminal, a gateway-spawned shell, an
agent's own tool call — the parent exports its runtime state to the child:
``HERMES_DESKTOP=1``, ``HERMES_SERVE_HEADLESS=1``, ``HERMES_WEB_DIST=<packaged
dist>``, ``HERMES_RPC_*``, ``HERMES_SESSION_*``, and more. Product code branches
on that state (``mount_spa``/``_serve_index`` check ``HERMES_SERVE_HEADLESS``
per call; ``hermes_cli.web_server`` freezes ``HERMES_WEB_DIST`` into ``WEB_DIST``
at import), so an inherited value silently reconfigures the code under test:
the web-server tests answered from the headless no-frontend catch-all and six
tests in ``tests/gateway/test_hosted_room_peer.py`` failed on an otherwise
untouched ``main``.

``hermes_cli/main.py::_dashboard_sanitize_desktop_env`` already strips the same
inherited state from a standalone ``hermes dashboard`` launch (#52945).
``tests/conftest.py`` now does the test-side equivalent, once, at import time —
the only moment that can beat a module-level ``WEB_DIST`` freeze.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import (
    _HERMES_BEHAVIORAL_VARS,
    _HERMES_ENV_ALLOWLIST,
    _scrub_inherited_hermes_env,
)

# State the Desktop app / gateway hands to every shell it spawns. Each name
# here changes product behavior, so none of them may survive into a test
# process.
_DESKTOP_RUNTIME_STATE = (
    "HERMES_DESKTOP",
    "HERMES_SERVE_HEADLESS",
    "HERMES_WEB_DIST",
)
_RUNTIME_STATE_PREFIXES = (
    "HERMES_DESKTOP",
    "HERMES_SERVE_",
    "HERMES_WEB_DIST",
    "HERMES_RPC_",
    "HERMES_KERNEL_",
    "HERMES_SESSION_ID",
    "HERMES_SESSION_KEY",
)

# Instructions TO the run: removing them would disable a lane the operator
# asked for, or break a Windows developer's shell resolution.
_RUN_CONFIGURATION = (
    "HERMES_HOME",
    "HERMES_TEST_ISOLATION",
    "HERMES_GIT_BASH_PATH",
    "HERMES_TEST_WORKERS",
)

# The one-node-of-this-file command the subprocess test runs on itself.
_SELF_NODE = (
    "tests/ci/test_conftest_inherited_env.py"
    "::test_no_inherited_runtime_state_survived_import"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _runtime_state_in(environ) -> list[str]:
    return sorted(
        name
        for name in environ
        if name not in _HERMES_ENV_ALLOWLIST
        and any(name.startswith(prefix) for prefix in _RUNTIME_STATE_PREFIXES)
    )


def test_no_inherited_runtime_state_survived_import():
    """Live invariant: the launching session's runtime state is gone.

    On unmodified ``main`` this fails whenever pytest is started from a Hermes
    session — it is the regression itself, not a formality. (Run with an empty
    environment it is trivially satisfied; the parameterized and subprocess
    tests below keep the guarantee honest everywhere.)
    """
    leaked = _runtime_state_in(os.environ)
    assert leaked == [], (
        "HERMES_* runtime state leaked from the session that launched pytest: "
        f"{leaked}. Product code reads these — some at import time — so they "
        "reconfigure the code under test."
    )


def test_desktop_runtime_state_is_absent_from_this_process():
    present = [name for name in _DESKTOP_RUNTIME_STATE if name in os.environ]
    assert present == []


def test_a_poisoned_parent_env_cannot_reach_the_test_process():
    """End-to-end: launch pytest from a shell that looks like a Desktop one.

    The child runs one node of this file, which asserts the same invariant
    in-process; before the conftest scrub it fails, after it passes.
    """
    env = dict(os.environ)
    env.update({
        "HERMES_DESKTOP": "1",
        "HERMES_SERVE_HEADLESS": "1",
        "HERMES_WEB_DIST": str(_REPO_ROOT / "app.asar.unpacked" / "dist"),
        "HERMES_RPC_SOCKET": str(_REPO_ROOT / "fake.sock"),
    })
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", _SELF_NODE,
            "-o", "addopts=", "-p", "no:cacheprovider", "-q", "--no-header",
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


def test_scrub_removes_runtime_state_and_keeps_run_configuration():
    poisoned = {
        "HERMES_DESKTOP": "1",
        "HERMES_SERVE_HEADLESS": "1",
        "HERMES_WEB_DIST": "/tmp/desktop/app.asar.unpacked/dist",
        "HERMES_RPC_SOCKET": "/tmp/hermes.sock",
        "HERMES_RPC_TOKEN": "[REDACTED]",
        "HERMES_SESSION_ID": "sess-123",
        "HERMES_HOME": "/tmp/sandbox-home",
        "HERMES_TEST_WORKERS": "8",
        "PATH": "/usr/bin",
    }

    removed = _scrub_inherited_hermes_env(poisoned)

    assert removed == [
        "HERMES_DESKTOP",
        "HERMES_RPC_SOCKET",
        "HERMES_RPC_TOKEN",
        "HERMES_SERVE_HEADLESS",
        "HERMES_SESSION_ID",
        "HERMES_WEB_DIST",
    ]
    assert set(poisoned) == {"HERMES_HOME", "HERMES_TEST_WORKERS", "PATH"}
    assert _scrub_inherited_hermes_env(poisoned) == []  # idempotent


def test_scrub_targets_the_process_environment_by_default(monkeypatch):
    """Scrubbing must work without an explicit mapping, and keep run config.

    The live environment also holds vars the suite sets for itself (e.g. the
    lazy-install kill-switch in ``_hermetic_environment``), so only membership
    of the injected names is asserted here.
    """
    monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")
    monkeypatch.setenv("HERMES_WEB_DIST", "/tmp/desktop/dist")
    home_before = os.environ.get("HERMES_HOME")

    removed = _scrub_inherited_hermes_env()

    assert "HERMES_SERVE_HEADLESS" in removed
    assert "HERMES_WEB_DIST" in removed
    assert "HERMES_SERVE_HEADLESS" not in os.environ
    assert "HERMES_WEB_DIST" not in os.environ
    assert os.environ.get("HERMES_HOME") == home_before


@pytest.mark.parametrize("name", _DESKTOP_RUNTIME_STATE)
def test_desktop_runtime_vars_stay_registered(name):
    """Dropping one from the registry would silently reopen the leak.

    Either home is acceptable: the behavioral deny-list (blanked per test, and
    what a developer exporting the var has to hit) or the allowlist
    (deliberately honored, i.e. an instruction to the run).
    """
    assert name in _HERMES_BEHAVIORAL_VARS or name in _HERMES_ENV_ALLOWLIST


def test_runtime_state_prefixes_are_not_allowlisted():
    allowlisted = sorted(
        name
        for name in _HERMES_ENV_ALLOWLIST
        if any(name.startswith(prefix) for prefix in _RUNTIME_STATE_PREFIXES)
    )
    assert allowlisted == []


def test_run_configuration_survives_a_scrub():
    poisoned = {name: "set" for name in _RUN_CONFIGURATION}
    assert _scrub_inherited_hermes_env(poisoned) == []
    assert set(poisoned) == set(_RUN_CONFIGURATION)
