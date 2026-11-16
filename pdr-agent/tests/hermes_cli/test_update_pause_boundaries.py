"""Real-process pause/consume/recovery contracts, not native Windows SCM evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

REPO = Path(__file__).resolve().parents[2]


def _wait(path, process):
    deadline = time.monotonic() + 30
    while not path.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, str(path)
        time.sleep(0.01)


@pytest.fixture
def cell(tmp_path):
    # An unchanged copy owns its private checkout lock/root; never the live install's.
    root = tmp_path / "checkout"
    (root / "hermes_cli").mkdir(parents=True)
    shutil.copyfile(REPO / "hermes_cli/update_pause_record.py", root / "hermes_cli/update_pause_record.py")
    home = tmp_path / "home"
    (home / ".hermes").mkdir(parents=True)
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home),
           "HERMES_HOME": str(home / ".hermes"), "PYTHONPATH": str(REPO),
           "PYTHONDONTWRITEBYTECODE": "1"}
    children = []

    def launch(code, *args):
        script = tmp_path / f"child-{len(children)}.py"
        script.write_text(textwrap.dedent('''
            import importlib.util, json, os, sys, time
            from pathlib import Path
            root = Path(sys.argv[1])
            spec = importlib.util.spec_from_file_location("hermes_cli.update_pause_record", root / "hermes_cli/update_pause_record.py")
            r = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = r
            spec.loader.exec_module(r)
        ''') + textwrap.dedent(code), encoding="utf-8")
        proc = subprocess.Popen([sys.executable, str(script), str(root), *map(str, args)],
                                cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        children.append(proc)
        return proc

    yield tmp_path, home / ".hermes", launch
    for proc in children:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=10)


def _result(proc):
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 0, (out, err)
    return json.loads(out.splitlines()[-1])


@pytest.mark.parametrize("claim_first", [False, True])
def test_consumption_checkpoints_debt_before_unlink_and_producer_death(cell, claim_first):
    base, home, launch = cell
    consumer = launch('''
        from gateway.status import consume_planned_stop_marker_for_self
        Path(sys.argv[2]).write_text(str(os.getpid()), encoding="utf-8")
        while not Path(sys.argv[3]).exists(): time.sleep(.01)
        accepted = consume_planned_stop_marker_for_self()
        Path(sys.argv[4]).write_text(json.dumps(accepted), encoding="utf-8")
        while True: time.sleep(.05)
    ''', base / "ready", base / "consume", base / "consumed")
    _wait(base / "ready", consumer)
    owner = launch('''
        from hermes_cli import update_cmd_windows as w
        from types import SimpleNamespace
        pid = int(sys.argv[2])
        home = Path(os.environ["HERMES_HOME"])
        token = r.record_pause({"resume_needed": True, "profiles": {"default": pid},
                               "identities": {str(pid): r.identity(pid)["ct"]}}, None, [])
        # Discovery and the gateway can spell the same profile path differently.
        r.mark_stop_requested(token, [pid], {pid: home / ".." / ".hermes" / ".gateway-planned-stop.json"})
        # The real callback is the first operation after marker publication. Hold it
        # before its checkpoint while another interpreter runs the actual consumer.
        def on_request(pid):
            Path(sys.argv[3]).touch()
            while True: time.sleep(.05)
        w._request_socket_pauses([pid], {pid: SimpleNamespace(profile="default", path=home)},
                                 set(), on_request=on_request)
    ''', consumer.pid, base / "published")
    _wait(base / "published", owner)
    recovery = None
    if claim_first:
        owner.kill()
        owner.communicate(timeout=10)
        recovery = launch('''
            won = r.claim(r.record_path())
            assert won is not None
            Path(sys.argv[2]).touch()
            while not Path(sys.argv[3]).exists(): time.sleep(.01)
            r._resume_claimed(*won)
            print(json.dumps([body["token"] for _, body in r.orphans()]))
        ''', base / "claimed", base / "recover")
        _wait(base / "claimed", recovery)
    (base / "consume").touch()
    _wait(base / "consumed", consumer)
    assert json.loads((base / "consumed").read_text(encoding="utf-8-sig")) is True
    assert not (home / ".gateway-planned-stop.json").exists()
    if owner.poll() is None:
        owner.kill()
        owner.communicate(timeout=10)
    assert consumer.poll() is None
    if recovery is not None:
        (base / "recover").touch()
    else:
        recovery = launch('''
            r.recover(["status"])
            print(json.dumps([body["token"] for _, body in r.orphans()]))
        ''')
    after = _result(recovery)
    assert len(after) == 1, "consumed request lost all durable restart debt"
    assert after[0]["profiles"] == {"default": consumer.pid}
    assert after[0]["stop_sent"] == [str(consumer.pid)]


def test_failed_consumer_checkpoint_keeps_request_and_user_stop_is_not_update_debt(cell):
    _, _, launch = cell
    result = _result(launch('''
        from gateway import status
        from hermes_cli.update_cmd_windows import _write_update_planned_stop_marker
        pid = os.getpid()
        home = Path(os.environ["HERMES_HOME"])
        path = home / ".gateway-planned-stop.json"
        token = r.record_pause({"resume_needed": True, "profiles": {"default": pid},
                               "identities": {str(pid): r.identity(pid)["ct"]}}, None, [])
        r.mark_stop_requested(token, [pid], {pid: path})
        assert _write_update_planned_stop_marker(home, pid)
        # An actual directory at the atomic writer's temp path rejects publication.
        obstruction = r.record_path().with_name(f"{r.record_path().name}.{pid}.tmp")
        obstruction.mkdir()
        # The checkpoint cannot land: the stop is still planned, and the request stays on disk.
        try:
            planned = status.consume_planned_stop_marker_for_self()
        except OSError as exc:
            planned = type(exc).__name__
        retained = path.exists() and r.read()["token"]["stop_sent"] == []
        obstruction.rmdir()
        marker = json.loads(path.read_text(encoding="utf-8-sig"))
        path.write_text(json.dumps({**marker, "stopper_pid": pid + 1}), encoding="utf-8")
        user_consumed = status.consume_planned_stop_marker_for_self()
        user_debt = r.read()["token"]["stop_sent"]
        assert _write_update_planned_stop_marker(home, pid)
        accepted = status.consume_planned_stop_marker_for_self()
        print(json.dumps({"planned": planned, "retained": retained, "user_consumed": user_consumed,
                          "user_debt": user_debt, "accepted": accepted, "sent": r.read()["token"]["stop_sent"],
                          "pid": pid}))
    '''))
    assert result == {"planned": True, "retained": True, "user_consumed": True, "user_debt": [],
                      "accepted": True, "sent": [str(result["pid"])], "pid": result["pid"]}


def test_a_busy_pause_mutex_never_turns_a_planned_stop_into_an_unplanned_one(cell):
    base, _, launch = cell
    holder = launch('''
        r.write({"pause_id": "busy", "resume_needed": True})  # a pause on disk: the consume needs the mutex
        with r._mutex():
            Path(sys.argv[2]).touch()
            while not Path(sys.argv[3]).exists(): time.sleep(.05)
    ''', base / "held", base / "release")
    _wait(base / "held", holder)
    try:
        result = _result(launch('''
            from gateway import status
            from hermes_cli.update_cmd_windows import _write_update_planned_stop_marker
            pid = os.getpid()
            home = Path(os.environ["HERMES_HOME"])
            assert _write_update_planned_stop_marker(home, pid)
            started = time.monotonic()
            try:
                planned = status.consume_planned_stop_marker_for_self()
            except OSError as exc:
                planned = type(exc).__name__
            print(json.dumps({"planned": planned, "seconds": time.monotonic() - started,
                              "retained": (home / ".gateway-planned-stop.json").exists()}))
        '''))
    finally:
        (base / "release").touch()
    assert result["planned"] is True and result["retained"], result
    assert result["seconds"] < 5, f"the gateway's shutdown blocked {result['seconds']:.1f}s on the pause mutex"


def test_recovery_only_accepts_markers_the_live_consumer_accepts(cell):
    _, _, launch = cell
    result = _result(launch('''
        from datetime import datetime, timedelta, timezone
        from gateway import status
        pid = os.getpid()
        home = Path(os.environ["HERMES_HOME"])
        marker_path = home / ".gateway-planned-stop.json"
        token = {"profiles": {"default": pid}, "identities": {str(pid): r.identity(pid)["ct"]},
                 "stopper_pid": pid, "stop_requested": [str(pid)], "stop_markers": {str(pid): str(marker_path)}}
        current = status.get_process_start_time(pid)
        assert current is not None
        fresh = {"target_pid": pid, "stopper_pid": pid, "target_start_time": current,
                 "written_at": datetime.now(timezone.utc).isoformat()}
        cases = {"valid": fresh,
                 "expired": {**fresh, "written_at": (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()},
                 "wrong_incarnation": {**fresh, "target_start_time": current + 1},
                 "malformed": {"target_pid": pid, "stopper_pid": pid}}
        out = {}
        for name, marker in cases.items():
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            kept = r.drop_never_stopped(dict(token))
            consumed = status.consume_planned_stop_marker_for_self()
            out[name] = {"kept": bool(kept["profiles"]), "consumed": consumed}
        print(json.dumps(out))
    '''))
    assert result["valid"] == {"kept": True, "consumed": True}
    for name in ("expired", "wrong_incarnation", "malformed"):
        assert result[name] == {"kept": False, "consumed": False}, (name, result)


def test_orphan_adoption_preserves_cold_start_generation_and_current_precedence(cell):
    _, _, launch = cell
    _result(launch('''
        token = {"resume_needed": True, "profiles": {}, "cold_start_if_installed": True,
                 "attested_generation": "old-active", "cold_start_profiles": {"beta": "old-beta", "gamma": "old-gamma"}}
        r.write(r.stamp_tree(token), owner=r.UNOWNED)
        print(json.dumps(r.read()["token"]))
    '''))
    adopted = _result(launch('''
        adopted, claims = r.adopt_orphans()
        token = {"resume_needed": True, "profiles": {}, "cold_start_profiles": {"beta": "current-beta"}}
        r.record_pause(token, adopted, claims)
        print(json.dumps(r.read()["token"]))
    '''))
    assert adopted.get("cold_start_if_installed") is True
    assert adopted.get("attested_generation") == "old-active"
    assert adopted.get("cold_start_profiles") == {"beta": "current-beta", "gamma": "old-gamma"}
    again = _result(launch('''
        adopted, claims = r.adopt_orphans()
        r.record_pause({"resume_needed": True, "profiles": {}, "cold_start_if_installed": True,
                        "attested_generation": "current-active"}, adopted, claims)
        print(json.dumps(r.read()["token"]))
    '''))
    assert again["attested_generation"] == "current-active"
    assert again["cold_start_profiles"] == adopted["cold_start_profiles"]
