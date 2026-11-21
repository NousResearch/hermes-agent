"""Regression tests for the external-oneshot dispatch guard.

A gateway session that hands a card to a one-shot CLI worker spawns
``hermes -p <profile> -z "接单:部署卡 t_x…" --yolo`` and never calls ``claim_task``, so the row
stays ``ready`` with no claim for the whole run. Both existing guards (``check_respawn_guard``,
the ``claim_task`` CAS) are DB-side and cannot see a process, so the next dispatcher tick claimed
the same card and spawned a twin worker on the same workspace — same ports, one ``.env``
(t_c18fc591, then t_bb66a4a2). The dispatcher now probes the live process table first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_kanban_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with a clean kanban DB."""
    test_home = tmp_path / ".hermes"
    test_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(test_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db
    yield kanban_db, test_home


def _fake_spawn_factory(spawns: list):
    """Stand-in for the real worker spawn — records the task and returns a fake PID."""
    def _fake_spawn(*args, **kwargs):
        spawns.append(args[0] if args else kwargs.get("task"))
        return 12345
    return _fake_spawn


def _events(conn, task_id: str, kind: str) -> list:
    return [json.loads(row[1]) for row in conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind = ?",
        (task_id, kind),
    )]


def test_live_external_oneshot_defers_the_card(
    isolated_kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Core contract: a ready card owned by a live external oneshot is NOT claimed.

    Zero spawns this tick, the ``(task_id, pid)`` pair recorded on the result, a
    ``respawn_guarded`` event naming the reason + PID for ``hermes kanban tail``, and the row
    left ``ready`` — a later tick picks it up once that worker exits."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_db_dispatch_oneshot as oneshot

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="deploy-me", assignee="default")

    # The dispatcher reaches the probe through kanban_db_dispatch_oneshot, so that is the module
    # whose attribute the patch must replace.
    monkeypatch.setattr(oneshot, "find_external_oneshot_worker", lambda tid: 4242)
    spawns: list = []
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), dry_run=False)

    assert spawns == []
    assert not res.spawned
    assert res.external_oneshot_skipped == [ (task_id, 4242) ]
    with kbc.connect_closing() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["status"] == "ready"
        guarded = _events(conn, task_id, "respawn_guarded")
    assert any(e.get("reason") == "external_oneshot_worker" and e.get("pid") == 4242
               for e in guarded)


def test_no_external_oneshot_spawns_normally(
    isolated_kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The probe finding nothing must not change the happy path: claim + spawn as before."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    from hermes_cli import kanban_db_dispatch_oneshot as oneshot

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="deploy-me", assignee="default")

    monkeypatch.setattr(oneshot, "find_external_oneshot_worker", lambda tid: None)
    spawns: list = []
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), dry_run=False)

    assert [t.id for t in spawns] == [task_id]
    assert [t for t, *_ in res.spawned] == [task_id]
    assert res.external_oneshot_skipped == []
    with kbc.connect_closing() as conn:
        assert _events(conn, task_id, "respawn_guarded") == []


def test_toggle_off_skips_the_probe_entirely(
    isolated_kanban_home, all_assignees_spawnable, monkeypatch,
):
    """``kanban.detect_external_oneshot: false`` must not consult the process table at all — the
    probe costs one sweep per spawnable card per tick, so an operator who turns it off is paid
    back in tick time. The patched probe raises; a clean spawn proves it was never called."""
    import hermes_cli.config as cfgmod
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_db_dispatch_oneshot as oneshot

    assert oneshot.external_oneshot_detection_enabled() is True  # ships on
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: dict(kanban=dict(detect_external_oneshot=False)),
    )
    assert oneshot.external_oneshot_detection_enabled() is False

    kb, _home = isolated_kanban_home
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="deploy-me", assignee="default")

    def _boom(_task_id):
        raise AssertionError("probe consulted with kanban.detect_external_oneshot=false")

    monkeypatch.setattr(oneshot, "find_external_oneshot_worker", _boom)
    spawns: list = []
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), dry_run=False)

    assert [t.id for t in spawns] == [task_id]
    assert res.external_oneshot_skipped == []


# A real store/Docker launcher spawn: the interpreter's ``-c`` source is a Hermes bootstrap and
# the oneshot argv is what that bootstrap runs in-process, so the probe peels the wrapper
# (gateway.status.inline_bootstrap_argv) before reading argv.
_BOOTSTRAP_SRC = (
    "import os, site, sys; sys.argv[0]='hermes'; "
    "site.addsitedir(os.environ['HERMES_SITE']); "
    "from hermes_cli.main import main; sys.exit(main())"
)
_BOOTSTRAP_SPAWN = "python -I -c " + _BOOTSTRAP_SRC + " -p forge -z 接单:部署卡 t_deadbeef… --yolo"


@pytest.mark.parametrize("command,task_id,wanted", [
    # the real spawn shape, bootstrap-wrapped
    (_BOOTSTRAP_SPAWN, "t_deadbeef", True),
    # the plain spelling: prompt quoted, id mid-sentence
    ('hermes -p forge --oneshot "分析 t_deadbeef 的失败" --yolo', "t_deadbeef", True),
    # a gateway merely carrying the id is not the worker: no oneshot flag token
    ("python -m hermes_cli.main gateway run --board t_deadbeef", "t_deadbeef", False),
    # an oneshot working a different card
    ('hermes -z "整理归档" --yolo', "t_deadbeef", False),
    # a longer id sharing the prefix is NOT this card (findall equality: the greedy hex run
    # yields only t_deadbeef1) — and it is its own card
    ('hermes -z "接单 t_deadbeef1"', "t_deadbeef", False),
    ('hermes -z "接单 t_deadbeef1"', "t_deadbeef1", True),
    # degenerate inputs
    ("hermes -z", "t_deadbeef", False),
    ("", "t_deadbeef", False),
])
def test_oneshot_argv_names_task(command, task_id, wanted):
    from hermes_cli import kanban_db_dispatch_oneshot as oneshot
    assert oneshot._oneshot_argv_names_task(command, task_id) is wanted


def test_looks_like_hermes_cli_is_token_exact():
    """Identity by basename/token equality: a path that merely CONTAINS ``hermes`` or
    ``hermes_cli.main`` names some other program."""
    from hermes_cli import kanban_db_dispatch_oneshot as oneshot
    assert oneshot._looks_like_hermes_cli("hermes", "hermes -p forge -z hi") is True
    assert oneshot._looks_like_hermes_cli("/usr/local/bin/hermes.exe", "x") is True
    assert oneshot._looks_like_hermes_cli("python", "python -m hermes_cli.main gateway run") is True
    assert oneshot._looks_like_hermes_cli(
        "python", "python /opt/hermes/hermes_cli/main.py -z hi") is True
    assert oneshot._looks_like_hermes_cli(
        "/opt/hermes-notify/bin/notify", "--watch hermes") is False
    assert oneshot._looks_like_hermes_cli("python", "python -m hermes_cli.mainlib run") is False
    # and an empty task id never matches anything
    assert oneshot.find_external_oneshot_worker("") is None


class _FakeProc:
    """One ``psutil.process_iter`` row; ``raise_on_info`` mimics a process that vanished or
    belongs to another user between the listing and the read."""

    def __init__(self, info=None, raise_on_info=False):
        self._info = info if info is not None else dict()
        self._raise = raise_on_info

    @property
    def info(self):
        if self._raise:
            raise RuntimeError("process vanished mid-scan")
        return self._info


class _FakePsutil:
    def __init__(self, procs, raise_from_iter=False):
        self._procs = procs
        self._raise = raise_from_iter
        self.attrs = None

    def process_iter(self, attrs=None):
        self.attrs = attrs
        if self._raise:
            raise OSError("procfs unavailable")
        yield from self._procs


def test_iter_process_cmdlines_skips_own_pid_and_swallows_errors(monkeypatch):
    from gateway import status_proc_table as spt

    fake = _FakePsutil([
        _FakeProc(dict(pid=4242, cmdline=["hermes", "-z", "hi"])),
        _FakeProc(dict(pid=7, cmdline=["sh", "-c", "sleep 1"])),   # this process: skipped
        _FakeProc(dict(pid=8, cmdline=[])),                        # unreadable → absent
        _FakeProc(dict(pid=9, cmdline=None)),                      # denied → absent
        _FakeProc(dict(pid=0, cmdline=["hermes"])),                # not a real pid → absent
        _FakeProc(raise_on_info=True),                             # vanished mid-scan
    ])
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr(spt.os, "getpid", lambda: 7)

    entries = spt.iter_process_cmdlines()

    assert fake.attrs == ["pid", "cmdline"]
    assert [pid for pid, _ in entries] == [4242]
    assert entries == [ (4242, "hermes -z hi") ]


def test_iter_process_cmdlines_never_raises_when_psutil_fails(monkeypatch):
    """psutil missing or raising → the POSIX ``ps`` fallback; Windows has neither and degrades
    to ``[]``. Either way the caller gets a list, never an exception. ``_read_process_cmdline``
    and ``_IS_WINDOWS`` are late-imported from ``gateway.status`` INSIDE the function, so the
    patch targets that defining module, not the reader's own module."""
    from gateway import status
    from gateway import status_proc_table as spt

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil([], raise_from_iter=True))
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)
    assert isinstance(spt.iter_process_cmdlines(), list)

    monkeypatch.setattr(status, "_IS_WINDOWS", True)
    assert spt.iter_process_cmdlines() == []


def test_ps_fallback_prefilters_by_comm_basename(monkeypatch):
    """The ``ps`` fallback must not fork a ``ps`` per process: only Hermes-looking ``comm``
    basenames become candidates (exact names plus the versioned ``python*`` family)."""
    from gateway import status
    from gateway import status_proc_table as spt
    from gateway.status_proc_table import _comm_looks_like_hermes

    assert _comm_looks_like_hermes("/usr/local/bin/hermes") is True
    assert _comm_looks_like_hermes("hermes.exe") is True
    assert _comm_looks_like_hermes("/usr/bin/python3.12") is True
    assert _comm_looks_like_hermes("/bin/sh") is True
    assert _comm_looks_like_hermes("sherlock-monitor") is False
    assert _comm_looks_like_hermes("shipmentd") is False
    assert _comm_looks_like_hermes("") is False

    monkeypatch.setitem(sys.modules, "psutil", None)  # the ImportError path
    own_line = str(spt.os.getpid()) + " /bin/sh\n"
    asked: list = []

    class _PsResult:
        returncode = 0
        stdout = own_line + (
            "4242 /usr/local/bin/hermes\n"
            "4243 /usr/bin/python3.12\n"
            "4244 /usr/libexec/sherlock-monitor\n"
            "notapid junk\n"
        )

    def fake_run(cmd, *a, **k):
        asked.append(cmd)
        return _PsResult()

    monkeypatch.setattr(spt.subprocess, "run", fake_run)
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "cmdline-of-" + str(pid))
    entries = spt.iter_process_cmdlines()

    assert asked == [ ["ps", "-ax", "-o", "pid=,comm="] ]
    # own PID and the non-Hermes basename never reached a per-PID read
    assert sorted(entries) == [ (4242, "cmdline-of-4242"), (4243, "cmdline-of-4243") ]
