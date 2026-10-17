"""An interrupted non-quiet one-shot ``hermes -q`` turn exits 130 instead of leaking a traceback.

The quiet (``-Q``) path already caught ``KeyboardInterrupt`` around its turn and reported
``Interrupted``, but ``_run_single_query_mode`` (the ``-q`` path the Paperclip ``hermes_local``
adapter spawns) let it escape ``cli_single_query.py:548``. The interpreter then printed
``Traceback (most recent call last): ... KeyboardInterrupt`` to stderr, and the adapter booked
the first stderr line as the run's ``error`` — the fleet's recurring
``errorReason: "Traceback (most recent call last):"`` (run ``a8af5966``, 2026-10-17T12:09:25Z).

Both tests drive the real ``_run_single_query_mode`` tail with a ``chat`` that raises
``KeyboardInterrupt`` (what ``_install_single_query_signal_handlers`` raises on SIGINT). The
subprocess arm is the one that can see the interpreter's own traceback print.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli

DRIVER = textwrap.dedent(
    """
    import sys
    from types import SimpleNamespace

    import cli

    cli._should_seed_interactive = lambda *a, **k: False
    cli._collect_query_images = lambda q, i: (q, [])
    cli._collect_kanban_task_images = lambda imgs: []
    cli._finalize_single_query = lambda c: None
    cli._emit_interrupted_session_end = lambda c, **k: print(
        "SESSION_END:" + str(k.get("reason")), file=sys.stderr)


    class Stub:
        _single_query_mode = False
        session_id = "s-oneshot"
        model = "stub-model"
        conversation_history = []
        _last_turn_result = None

        def _claim_active_session(self, *a, **k):
            return True

        def _show_security_advisories(self):
            pass

        def _print_exit_summary(self, **k):
            pass

        def chat(self, *a, **k):
            raise KeyboardInterrupt()


    stub = Stub()
    stub.console = SimpleNamespace(print=lambda *a, **k: None)
    cli._run_single_query_mode(stub, "query", None, False, True)
    """
)


@pytest.fixture(autouse=True)
def _no_inherited_kanban_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)


def _stub_cli():
    return SimpleNamespace(
        _single_query_mode=False,
        session_id="s-oneshot",
        model="stub-model",
        conversation_history=[],
        _last_turn_result=None,
        _claim_active_session=lambda *a, **k: True,
        _show_security_advisories=lambda: None,
        _print_exit_summary=lambda **k: None,
        chat=lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
        console=SimpleNamespace(print=lambda *a, **k: None),
    )


def _isolate_one_shot_path(monkeypatch, on_interrupt):
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)
    monkeypatch.setattr(cli, "_emit_interrupted_session_end", on_interrupt)


def test_interrupted_oneshot_turn_exits_130_and_ends_the_session(monkeypatch, capsys):
    """The turn-end record is the difference between an interrupted run and a crash: the session
    must be ended with ``keyboard_interrupt`` and the process must exit 130, not unwind."""
    reasons = []
    _isolate_one_shot_path(monkeypatch, lambda _cli, **k: reasons.append(k.get("reason")))

    code = None
    try:
        with pytest.raises(SystemExit) as exc:
            cli._run_single_query_mode(_stub_cli(), "query", None, False, True)
        code = exc.value.code
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped the one-shot path instead of exiting 130")
    assert code == 130
    assert reasons == ["keyboard_interrupt"]
    assert "session_id: s-oneshot" in capsys.readouterr().err


def test_interrupted_oneshot_turn_prints_no_traceback(tmp_path):
    """End to end through a real interpreter: the SIGINT path must leave no ``Traceback`` (the
    string the Paperclip adapter stores as ``error``) and must exit 130."""
    repo_root = Path(cli.__file__).resolve().parent
    driver = tmp_path / "drive_oneshot.py"
    driver.write_text(DRIVER, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(repo_root), "PYTHONIOENCODING": "utf-8"}
    env.pop("HERMES_KANBAN_TASK", None)
    env.pop("HERMES_KANBAN_GOAL_MODE", None)

    proc = subprocess.run(
        [sys.executable, str(driver)], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=300, check=False,
    )

    assert "Traceback" not in proc.stderr, proc.stderr
    assert "KeyboardInterrupt" not in proc.stderr, proc.stderr
    assert "SESSION_END:keyboard_interrupt" in proc.stderr, proc.stderr
    assert proc.returncode == 130, proc.stderr
