"""Regression: the 2026-09-17 live-board leak via the dispatcher env pin.

The dispatcher injects ``HERMES_KANBAN_DB=<live ~/.hermes/kanban.db>`` into
every dispatched worker env (``hermes_cli/kanban_db_dispatch.py``). Workers
(and Pi/conductor sessions under them) run pytest with that env inherited, so
``kanban_db_path()`` (``hermes_cli/kanban_db.py``) resolved the pin FIRST —
overriding any ``HERMES_HOME`` sandbox a test had set — and fixture cards
landed in the LIVE board via the real tool layer (``kt._handle_create``) or
``kb.create_task`` (leak waves 12:15–12:42 UTC; worker ``/tmp/kt-base-check``
recurrence at 14:54).

The connect-time choke ``kanban_db_connect._ensure_test_isolation`` must make
that write impossible from a test context regardless of how the path was
resolved — env pin, board, or explicit db_path — and must fire BEFORE any
mkdir/sqlite touch. These tests replay the exact topology against the REAL
platform root with the pin pointing at the REAL live board path and assert
the guard raises without creating or mutating anything.

``@pytest.mark.live_system_guard_bypass`` (the established escape hatch, the
same marker hermes_state's live-DB guard honors) must keep working for tests
that genuinely need the live board, so the opt-out is pinned here too.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_state_guard import _real_platform_state_root

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def pinned_live_env(_hermetic_environment, monkeypatch, tmp_path):
    """Replay the dispatcher env: ``HERMES_KANBAN_DB`` pinned at the REAL live
    board, ``HERMES_HOME`` redirected to a per-test sandbox tmp_path."""
    root = _real_platform_state_root()
    assert root is not None, "platform state root must resolve for this regression"
    sandbox = tmp_path / "worker-home"
    sandbox.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(sandbox))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(root / "kanban.db"))
    for var in (
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_GUARD_BYPASS",
    ):
        monkeypatch.delenv(var, raising=False)
    return root


def _live_stat(root: Path) -> tuple[int, int]:
    st = (root / "kanban.db").stat()
    return st.st_mtime_ns, st.st_size


# ---------------------------------------------------------------------------
# The incident topology: pin at the live board + sandboxed HERMES_HOME.
# The pin still wins path resolution; the choke must refuse the write.
# ---------------------------------------------------------------------------


def test_connect_refuses_live_db_with_env_pin_and_sandboxed_home(pinned_live_env):
    """kanban_db_path() honors the pin (the incident's first half); connect()
    and init_db() must still refuse the live board from a test context —
    before any mkdir, lock sidecar, or sqlite touch."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    root = pinned_live_env
    pinned = Path(os.environ["HERMES_KANBAN_DB"])
    assert pinned == root / "kanban.db"
    resolved = kb.kanban_db_path()
    assert resolved == pinned, (
        f"kanban_db_path() resolved {resolved}, expected the pin {pinned}"
    )
    before = _live_stat(root)
    # Either guard may fire (the autouse conftest wrapper patches connect);
    # the contract under test is: raise, never touch the live file.
    with pytest.raises(RuntimeError, match="guard"):
        kbc.connect()  # no db_path arg: resolves via the pin
    with pytest.raises(RuntimeError, match="guard"):
        kbc.connect(str(pinned))  # explicit live path refused too
    with pytest.raises(RuntimeError, match="guard"):
        kbc.init_db()  # init path guarded as well
    assert _live_stat(root) == before, "refused calls must not touch the live DB"


def test_choke_direct_refuses_pinned_live_path(pinned_live_env):
    """The product choke itself (what an unhardened checkout without the
    conftest guard relies on) refuses the pinned live path."""
    from hermes_cli import kanban_db_connect as kbc

    root = pinned_live_env
    with pytest.raises(RuntimeError, match="test-isolation guard"):
        kbc._ensure_test_isolation(root / "kanban.db")
    with pytest.raises(RuntimeError, match="test-isolation guard"):
        kbc._ensure_test_isolation(root / "kanban" / "boards" / "second" / "kanban.db")


def test_tool_layer_create_refuses_live_board(pinned_live_env):
    """kt._handle_create (the real worker tool) must not be able to write the
    live board when the env pin leaks into pytest. The handler converts
    exceptions to structured tool errors, so the contract is: error payload,
    never a successful create, and zero mutation of the live DB."""
    from tools import kanban_tools as kt

    root = pinned_live_env
    before = _live_stat(root)
    out = kt._handle_create(
        {
            "title": "leak-probe-t-cd6b5114",
            "assignee": "default",
            "body": "regression t_cd6b5114: must never reach the live board",
        }
    )
    assert "guard" in out, f"tool layer accepted the live-board create: {out}"
    assert not out.startswith('{"ok"'), f"create reported success: {out}"
    assert _live_stat(root) == before
    live = sqlite3.connect(f"file:{root / 'kanban.db'}?mode=ro", uri=True)
    try:
        n = live.execute(
            "SELECT COUNT(*) FROM tasks WHERE title = 'leak-probe-t-cd6b5114'"
        ).fetchone()[0]
    finally:
        live.close()
    assert n == 0, "fixture row reached the live board"


# ---------------------------------------------------------------------------
# Marker escape hatch: live_system_guard_bypass keeps the sanctioned opt-out.
# ---------------------------------------------------------------------------


@pytest.mark.live_system_guard_bypass
def test_live_guard_bypass_marker_disables_kanban_choke(pinned_live_env):
    """The established escape hatch must keep working: with the marker applied
    the kanban choke guard must NOT raise on the live path. Verified without
    writing — only the guard predicate itself is invoked."""
    from hermes_cli import kanban_db_connect as kbc

    root = pinned_live_env
    kbc._ensure_test_isolation(root / "kanban.db")  # must not raise


# ---------------------------------------------------------------------------
# Out-of-pytest subprocess wave (14:54 worker /tmp/kt-base-check): a plain
# worker shell runs pytest as a child with the pin inherited — the child's
# own test-context signal is what must arm the choke.
# ---------------------------------------------------------------------------


def test_subprocess_pytest_with_inherited_pin_is_refused(pinned_live_env, tmp_path):
    """A worker shell (no PYTEST_* vars in the parent) runs pytest as a child;
    the child inherits pin + sandboxed HERMES_HOME. The child's choke must
    refuse the live path. The probe calls the predicate only — it can never
    touch the live DB even if the guard were missing (RED-safe)."""
    code = (
        "import os, sys\n"
        "sys.path.insert(0, os.environ['REPO'])\n"
        "from hermes_cli import kanban_db as kb\n"
        "from hermes_cli import kanban_db_connect as kbc\n"
        "path = kb.kanban_db_path()\n"
        "try:\n"
        "    kbc._ensure_test_isolation(path)\n"
        "except RuntimeError as e:\n"
        "    assert 'test-isolation guard' in str(e), e\n"
        "    sys.exit(0)\n"
        "sys.exit(3)\n"
    )
    probe = tmp_path / "probe.py"
    probe.write_text(code)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTEST_CURRENT_TEST", "PYTEST_VERSION", "HERMES_TEST_ISOLATION")
    }
    # The subprocess IS the pytest run: arm it exactly the way pytest does.
    env["PYTEST_CURRENT_TEST"] = "tests/hermes_cli/test_kanban_env_pin_leak.py::child"
    env["REPO"] = str(_REPO)
    result = subprocess.run(
        [sys.executable, str(probe)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"child accepted the pinned live path (rc={result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
