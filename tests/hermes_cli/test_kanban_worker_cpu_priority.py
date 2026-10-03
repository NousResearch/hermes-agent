"""Kanban worker gateways must run at BACKGROUND CPU priority (2026-09-20 incident).

A daedalus-opus kanban worker launched 384 busy-loops in the background on the
production Mac Studio and never reaped them. Load average reached 538 for 55
minutes. The RESIDENT gateway on the same host starved: its loop-liveness
watchdog fired twice ("missed 3 consecutive liveness probes; exiting with code
75"), a subsequent boot took 12 minutes, and all four platform adapters timed
out at boot — including the localhost webhook listener. The agent was deaf on
every surface until the burners were killed by hand.

The class fix is priority, not policing: a kanban worker is batch work and must
never be able to outbid the interactive gateway for CPU. ``_default_spawn``
applies ``nice 19`` (and SCHED_IDLE where available) in a ``preexec_fn``, so
every descendant the worker spawns — its terminal-tool children included —
inherits the deprioritisation. ``kanban.worker_cpu_priority: normal`` opts out.

The detector half is in ``tests/gateway/test_loop_liveness_load_context.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure the worktree (not a stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants

        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    from hermes_cli.kanban_db_connect import _INITIALIZED_PATHS
    _INITIALIZED_PATHS.clear()
    return home


def _task(task_id: str = "t_prio") -> "kb.Task":
    return kb.Task(
        id=task_id,
        title="cpu priority",
        body=None,
        assignee="teknium",
        status="ready",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )


def _spawn_and_capture_kwargs(fresh_home, monkeypatch, task_id="t_prio") -> dict:
    """Drive ``_default_spawn`` with ``Popen`` stubbed; return its kwargs."""
    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured.update(kwargs)
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kbd._default_spawn(_task(task_id), str(fresh_home / "ws"), board=None)
    return captured


# ---------------------------------------------------------------------------
# (a) the spawn helper applies nice 19 — asserted in a REAL child process
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(os, "setpriority"), reason="POSIX priority API unavailable"
)
def test_worker_preexec_applies_nice_19_in_a_real_child(fresh_home, monkeypatch):
    """The preexec hook _default_spawn installs really deprioritises a child.

    This runs the hook against a trivial ``python -c`` child (never a gateway)
    and reads the priority back from inside that child via ``os.getpriority``.
    A hook that is wired but inert would return 0 here.
    """
    captured = _spawn_and_capture_kwargs(fresh_home, monkeypatch)
    preexec = captured.get("preexec_fn")
    assert preexec is not None, (
        "_default_spawn did not install a preexec_fn; worker CPU priority "
        "is not being lowered and worker load can starve the gateway"
    )

    # Restore the real Popen for the live child probe.
    monkeypatch.undo()
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; print(os.getpriority(os.PRIO_PROCESS, 0))",
        ],
        capture_output=True,
        text=True,
        preexec_fn=preexec,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == 19, (
        f"child niceness was {out.stdout.strip()!r}, expected 19"
    )


@pytest.mark.skipif(
    not hasattr(os, "setpriority") or not hasattr(os, "fork"),
    reason="POSIX priority/fork API unavailable",
)
def test_nice_is_inherited_by_forked_grandchildren(fresh_home, monkeypatch):
    """A fork()'d grandchild of the worker must also be nice 19.

    Second incident, 2026-09-20 11:31 (card t_2b81d847): a worker launched a
    Python script that ``os.fork()``ed 2*ncpu spinners for 2400s — five minutes
    after the first burn was killed. The burners are never the worker process
    itself, so deprioritising only the direct child would be theatre. Niceness
    survives both fork and exec, which is exactly why the preexec hook is the
    right seam; this test pins that property rather than assuming it.
    """
    captured = _spawn_and_capture_kwargs(fresh_home, monkeypatch, "t_fork")
    preexec = captured.get("preexec_fn")
    assert preexec is not None
    monkeypatch.undo()

    # A grandchild via fork(), and a great-grandchild via fork()+exec, both
    # report their own priority. Never a gateway — trivial python children.
    program = (
        "import os,sys,subprocess\n"
        "r,w=os.pipe()\n"
        "if os.fork()==0:\n"
        "    os.close(r)\n"
        "    forked=os.getpriority(os.PRIO_PROCESS,0)\n"
        "    execed=subprocess.run([sys.executable,'-c',\n"
        "        \"import os;print(os.getpriority(os.PRIO_PROCESS,0))\"],\n"
        "        capture_output=True,text=True).stdout.strip()\n"
        "    os.write(w,f'{forked} {execed}'.encode())\n"
        "    os._exit(0)\n"
        "os.close(w)\n"
        "print(os.read(r,64).decode())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        preexec_fn=preexec,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    forked_nice, execed_nice = out.stdout.strip().split()
    assert int(forked_nice) == 19, f"fork()'d grandchild at nice {forked_nice}"
    assert int(execed_nice) == 19, f"exec'd great-grandchild at nice {execed_nice}"


def test_worker_cpu_priority_resolves_background_by_default():
    mode, nice = kbd.worker_cpu_priority_config({})
    assert mode == "background"
    assert nice == 19


def test_worker_cpu_priority_invalid_value_falls_back_to_background():
    mode, nice = kbd.worker_cpu_priority_config({"worker_cpu_priority": "turbo"})
    assert mode == "background"
    assert nice == 19


# ---------------------------------------------------------------------------
# (b) the knob set to 'normal' leaves priority untouched
# ---------------------------------------------------------------------------


def test_worker_cpu_priority_normal_opts_out():
    mode, nice = kbd.worker_cpu_priority_config({"worker_cpu_priority": "normal"})
    assert mode == "normal"
    assert nice == 0


def test_normal_mode_installs_no_preexec_fn(fresh_home, monkeypatch):
    """With the knob at 'normal', the spawn must not touch child priority."""
    monkeypatch.setattr(
        kbd, "worker_cpu_priority_config", lambda cfg=None: ("normal", 0)
    )
    captured = _spawn_and_capture_kwargs(fresh_home, monkeypatch, "t_normal")
    assert captured.get("preexec_fn") is None, (
        "'normal' must leave the child at inherited priority"
    )


def test_config_yaml_knob_reaches_the_resolver(fresh_home, monkeypatch):
    """Round-trip through the real loader, not a hand-built dict.

    Guards the ``from_dict``/loader class of bug where a declared knob is read
    back at its default because nothing extracts it from the YAML section.
    """
    (fresh_home / "config.yaml").write_text(
        "kanban:\n  worker_cpu_priority: normal\n", encoding="utf-8"
    )
    from hermes_cli.config import load_config

    cfg = load_config()
    assert (cfg.get("kanban") or {}).get("worker_cpu_priority") == "normal"
    mode, nice = kbd.worker_cpu_priority_config(cfg.get("kanban") or {})
    assert (mode, nice) == ("normal", 0)


def test_knob_is_declared_in_config_defaults():
    """`hermes config set kanban.worker_cpu_priority` must not cry wolf."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert "worker_cpu_priority" in DEFAULT_CONFIG["kanban"]
    assert DEFAULT_CONFIG["kanban"]["worker_cpu_priority"] == "background"


# ---------------------------------------------------------------------------
# Spawn observability: one auditable line per worker
# ---------------------------------------------------------------------------


def test_spawn_logs_cpu_priority_line(fresh_home, monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger=kb._log.name)
    _spawn_and_capture_kwargs(fresh_home, monkeypatch, "t_logline")
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "PHASE=worker_spawn" in line
    assert "task=t_logline" in line
    assert "cpu_priority=" in line
    assert "nice=" in line
