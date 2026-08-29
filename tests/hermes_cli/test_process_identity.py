"""Tests for hermes_cli.process_identity — spawn tags, the machine spawn
ledger, and the updater's ledger-identified reap rung.

Layer context (Aug 2026, after the 12-minute Windows update hang): reapers
previously inferred process lineage from PPIDs and cmdline shape. These
primitives make identity positive instead: spawners stamp children
(HERMES_SPAWN), long-lived processes self-register (pid, create_time,
purpose, spawner) in spawn-ledger.json, and `hermes update` reaps holders the
ledger PROVES are orphaned backends — in any update context, no hand-off
contract needed.

Runs on any host: psutil interactions go through a fake module.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import process_identity as pi


class _FakeNoSuchProcess(Exception):
    pass


def _fake_psutil(procs: dict[int, float]):
    """pid -> create_time; missing pid raises NoSuchProcess."""

    def _process(pid: int):
        if pid not in procs:
            raise _FakeNoSuchProcess(pid)
        proc = MagicMock()
        proc.pid = pid
        proc.create_time.return_value = procs[pid]
        return proc

    return types.SimpleNamespace(Process=_process, NoSuchProcess=_FakeNoSuchProcess)


# ---------------------------------------------------------------------------
# Spawn tags
# ---------------------------------------------------------------------------

def test_spawn_tag_roundtrip():
    with patch.dict(sys.modules, {"psutil": _fake_psutil({999: 1234.5})}):
        with patch.object(pi.os, "getpid", return_value=999):
            raw = pi.build_spawn_tag("serve", project_root=Path("/x/install"))
    tag = pi.parse_spawn_tag(raw)
    assert tag is not None
    assert tag.purpose == "serve"
    assert tag.spawner_pid == 999
    assert tag.spawner_create == pytest.approx(1234.5, abs=0.01)
    assert tag.install == pi.install_id(Path("/x/install"))


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "v0:abc:serve:1:2.0",          # wrong version
        "v1:abc:serve:notanint:2.0",   # bad pid
        "v1:abc:serve:-4:2.0",         # non-positive pid
        "v1::serve:10:2.0",            # empty install
        "v1:abc::10:2.0",              # empty purpose
        "v1:abc:serve:10",             # wrong arity
        "v1:abc:serve:10:zzz",         # bad create
    ],
)
def test_parse_spawn_tag_rejects_malformed(raw):
    assert pi.parse_spawn_tag(raw) is None


def test_parse_spawn_tag_dash_create_is_none():
    tag = pi.parse_spawn_tag("v1:abcdef:serve:42:-")
    assert tag is not None
    assert tag.spawner_create is None


def test_desktop_style_tag_parses():
    # The Electron side emits install `-` with a winms-derived create time.
    tag = pi.parse_spawn_tag("v1:-:serve:5100:1755689000.123")
    assert tag is not None
    assert tag.spawner_pid == 5100


def test_install_id_stable_and_path_scoped():
    a = pi.install_id(Path("/opt/hermes"))
    assert a == pi.install_id(Path("/opt/hermes"))
    assert a != pi.install_id(Path("/opt/other"))
    assert len(a) == 12


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def _entry(pid, create, purpose="serve", install=None, spawner_pid=None, spawner_create=None):
    return {
        "pid": pid,
        "create_time": create,
        "purpose": purpose,
        "install": install or pi.install_id(Path("/x/install")),
        "spawner_pid": spawner_pid,
        "spawner_create": spawner_create,
        "registered_at": 0.0,
        "argv": "",
    }


def test_register_self_writes_and_prunes_dead(tmp_path):
    ledger = tmp_path / "spawn-ledger.json"
    # Pre-existing: pid 300 dead, pid 400 alive.
    ledger.write_text(json.dumps([_entry(300, 1.0), _entry(400, 2.0)]), encoding="utf-8")
    fake = _fake_psutil({400: 2.0, 999: 50.0})
    with patch.dict(sys.modules, {"psutil": fake}), \
         patch.object(pi, "_ledger_path", return_value=ledger), \
         patch.object(pi.os, "getpid", return_value=999):
        assert pi.register_self("serve", project_root=Path("/x/install")) is True
    entries = json.loads(ledger.read_text(encoding="utf-8"))
    pids = {e["pid"] for e in entries}
    assert pids == {400, 999}  # 300 pruned as provably dead
    me = next(e for e in entries if e["pid"] == 999)
    assert me["purpose"] == "serve"
    assert me["create_time"] == pytest.approx(50.0, abs=0.01)


def test_register_self_inherits_spawn_tag_lineage(tmp_path):
    ledger = tmp_path / "spawn-ledger.json"
    fake = _fake_psutil({999: 50.0})
    env = {pi.SPAWN_ENV_VAR: "v1:-:serve:777:123.000"}
    with patch.dict(sys.modules, {"psutil": fake}), \
         patch.dict(pi.os.environ, env, clear=False), \
         patch.object(pi, "_ledger_path", return_value=ledger), \
         patch.object(pi.os, "getpid", return_value=999):
        pi.register_self("serve", project_root=Path("/x/install"))
    me = json.loads(ledger.read_text(encoding="utf-8"))[0]
    assert me["spawner_pid"] == 777
    assert me["spawner_create"] == pytest.approx(123.0, abs=0.01)


def test_register_self_falls_back_to_desktop_parent_env(tmp_path):
    # Legacy Desktop: no HERMES_SPAWN, but HERMES_PARENT_PID + winms marker.
    ledger = tmp_path / "spawn-ledger.json"
    fake = _fake_psutil({999: 50.0})
    env = {
        "HERMES_PARENT_PID": "888",
        "HERMES_PARENT_START_MARKER": "winms:1755689000123",
    }
    with patch.dict(sys.modules, {"psutil": fake}), \
         patch.dict(pi.os.environ, env, clear=False), \
         patch.object(pi.os.environ, "get", pi.os.environ.get), \
         patch.object(pi, "_ledger_path", return_value=ledger), \
         patch.object(pi.os, "getpid", return_value=999):
        pi.os.environ.pop(pi.SPAWN_ENV_VAR, None)
        pi.register_self("serve", project_root=Path("/x/install"))
    me = json.loads(ledger.read_text(encoding="utf-8"))[0]
    assert me["spawner_pid"] == 888
    assert me["spawner_create"] == pytest.approx(1755689000.123, abs=0.01)


def test_corrupt_ledger_quarantined_not_rewritten_blind(tmp_path):
    # #89298 contract: corruption is quarantined aside, never treated as [].
    ledger = tmp_path / "spawn-ledger.json"
    ledger.write_text("{ not json", encoding="utf-8")
    fake = _fake_psutil({999: 50.0})
    with patch.dict(sys.modules, {"psutil": fake}), \
         patch.object(pi, "_ledger_path", return_value=ledger), \
         patch.object(pi.os, "getpid", return_value=999):
        assert pi.register_self("serve", project_root=Path("/x/install")) is True
    parked = tmp_path / "spawn-ledger.json.corrupt"
    assert parked.exists()
    assert parked.read_text(encoding="utf-8") == "{ not json"
    entries = json.loads(ledger.read_text(encoding="utf-8"))
    assert [e["pid"] for e in entries] == [999]


def test_ledger_entries_filters_dead_reused_and_foreign(tmp_path):
    ledger = tmp_path / "spawn-ledger.json"
    entries = [
        _entry(100, 10.0),                                  # alive, matches
        _entry(200, 20.0),                                  # pid reused (create mismatch)
        _entry(300, 30.0),                                  # dead
        _entry(400, 40.0, install="ffffffffffff"),          # other install
    ]
    ledger.write_text(json.dumps(entries), encoding="utf-8")
    fake = _fake_psutil({100: 10.0, 200: 9999.0, 400: 40.0})
    with patch.dict(sys.modules, {"psutil": fake}), \
         patch.object(pi, "_ledger_path", return_value=ledger):
        live = pi.ledger_entries(project_root=Path("/x/install"))
    assert [e["pid"] for e in live] == [100]


def test_spawner_is_dead_tristate():
    fake = _fake_psutil({500: 5.0})
    with patch.dict(sys.modules, {"psutil": fake}):
        assert pi.spawner_is_dead(_entry(1, 1.0, spawner_pid=500, spawner_create=5.0)) is False
        assert pi.spawner_is_dead(_entry(1, 1.0, spawner_pid=600, spawner_create=6.0)) is True
        # PID reuse: recorded spawner create differs from live process → dead.
        assert pi.spawner_is_dead(_entry(1, 1.0, spawner_pid=500, spawner_create=999.0)) is True
        assert pi.spawner_is_dead(_entry(1, 1.0)) is None


# ---------------------------------------------------------------------------
# Updater rung: _ledger_reapable_backend_pids
# ---------------------------------------------------------------------------

def _holders(*pids):
    return [(p, "python.exe", f"python.exe -m hermes_cli.main --profile p{p} serve") for p in pids]


def test_updater_reaps_ledger_proven_orphans():
    from hermes_cli import main as cli_main

    entries = [
        _entry(200, 2.0, spawner_pid=700, spawner_create=7.0),   # spawner dead → reap
        _entry(201, 2.1, spawner_pid=500, spawner_create=5.0),   # spawner alive → keep
        _entry(202, 2.2, purpose="chat", spawner_pid=700, spawner_create=7.0),  # not reapable purpose
    ]
    fake = _fake_psutil({500: 5.0})
    with patch.dict(sys.modules, {"psutil": fake}), \
         patch.object(pi, "ledger_entries", return_value=entries), \
         patch.object(pi, "spawner_is_dead", wraps=pi.spawner_is_dead):
        assert cli_main._ledger_reapable_backend_pids(_holders(200, 201, 202, 203)) == [200]


def test_updater_ledger_rung_empty_without_ledger():
    from hermes_cli import main as cli_main

    with patch.object(pi, "ledger_entries", return_value=[]):
        assert cli_main._ledger_reapable_backend_pids(_holders(200)) == []


def test_updater_ledger_rung_never_raises():
    from hermes_cli import main as cli_main

    with patch.object(pi, "ledger_entries", side_effect=RuntimeError("boom")):
        assert cli_main._ledger_reapable_backend_pids(_holders(200)) == []
