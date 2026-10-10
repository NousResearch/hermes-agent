"""CP43: durable Task ownership and startup completion govern update recovery."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import status
from hermes_cli import gateway_windows, main, update_cmd_windows as update
from hermes_cli import update_pause_record as record


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("has_child", [False, True])
@pytest.mark.parametrize("write_fails", [False, True])
def test_supervisor_marker_requires_durable_profile_debt(monkeypatch, tmp_path, has_child, write_fails):
    homes = {"default": str(tmp_path / "default"), "other": str(tmp_path / "other")}
    monkeypatch.setattr(record, "record_path", lambda: tmp_path / "pause.json")
    monkeypatch.setattr(record, "stamp_tree", lambda token: token.update(pause_id="test-pause"))
    adopted = {"pause_id": "orphan", "supervisor_paused_profiles": {"adopted": str(tmp_path / "adopted")}}
    monkeypatch.setattr(record, "adopt_orphans", lambda: (adopted, []))
    processes = {os.getpid(): SimpleNamespace(profile="default", path=homes["default"])} if has_child else {}
    monkeypatch.setattr(update, "_discover_windows_gateways", lambda: (processes, [], set(), list(processes)))
    monkeypatch.setattr(update, "_windows_supervisor_profile_homes", lambda _: homes)
    monkeypatch.setattr(update, "_cold_start_pause_token", lambda *_: None)
    monkeypatch.setattr(update, "_record_attested_cold_start_profiles", lambda *_: None)
    monkeypatch.setattr(update, "_stop_windows_gateways", lambda *_a, **_k: {"default": os.getpid()})
    monkeypatch.setattr(gateway_windows, "_gateway_supervisor_pids", lambda: [123])
    monkeypatch.setattr(gateway_windows, "wait_for_supervisor_pause", lambda _: None)
    original_write = record._atomic_write
    armed = []

    def persist(path, body):
        if write_fails and body["token"].get("supervisor_paused_profiles", {}).get("other"):
            raise OSError("disk full")
        original_write(path, body)

    def arm():
        home = str(gateway_windows._hermes_home())
        saved = record.read(record.record_path())["token"]
        assert home in saved.get("supervisor_paused_profiles", {}).values()
        armed.append(home)
        marker = tmp_path / "stop"
        marker.write_text("nonce", encoding="utf-8")
        return marker

    monkeypatch.setattr(record, "_atomic_write", persist)
    monkeypatch.setattr(gateway_windows, "_arm_supervisor_stop_marker", arm)
    if write_fails:
        with pytest.raises(OSError, match="disk full"):
            update._pause_windows_gateways_for_update()
        assert armed == [homes["default"]]
        saved = record.read(record.record_path())["token"]
        assert saved["supervisor_paused_profiles"] == {
            **adopted["supervisor_paused_profiles"], "default": homes["default"],
        }
    else:
        token = update._pause_windows_gateways_for_update()
        assert armed == list(homes.values())
        assert token["supervisor_paused_profiles"] == {**adopted["supervisor_paused_profiles"], **homes}


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("state,birth,ready", [
    ("starting", "current", False),
    ("startup_failed", "current", False),  # fatal config exit 78
    ("stopped", "current", False),
    ("running", "stale", False),
    ("running", "current", True),
    ("degraded", "current", True),
    ("degraded", "stale", False),
])
def test_resume_requires_runtime_ready_from_the_live_incarnation(monkeypatch, tmp_path, state, birth, ready):
    pid = os.getpid()
    current = status.get_process_start_time(pid)
    assert current is not None
    (tmp_path / "gateway_state.json").write_text(json.dumps({
        "pid": pid, "start_time": current if birth == "current" else current - 100,
        "gateway_state": state, "hermes_home": str(tmp_path),
    }), encoding="utf-8")
    monkeypatch.setattr(main, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(gateway_windows, "start", lambda: None)
    monkeypatch.setattr(status, "_record_matches_live_gateway_pid", lambda *_a, **_k: True)

    def wait(**kwargs):
        return kwargs.get("pid_filter", lambda pids: pids)([pid])

    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", wait)
    token = {"resume_needed": True, "supervisor_paused_profiles": {"default": str(tmp_path)},
             "profiles": {"default": 10}, "unmapped": []}
    if ready:
        update._resume_paused_set(token)
        assert not token["supervisor_paused_profiles"]
    else:
        with pytest.raises(RuntimeError, match="did not become ready"):
            update._resume_paused_set(token)
        assert token["supervisor_paused_profiles"] == {"default": str(tmp_path)}
        assert token["profiles"] == {"default": 10}
        assert token["resume_needed"]


@pytest.mark.parametrize("prefix,options,expected", [
    ([], ["--home", "C:/home"], True),
    (["-u", "-X", "utf8"], ["--home", "C:/home"], True),
    (["-uB", "-BE"], ["--home=C:/home"], True),
    (["-Wignore", "-Xutf8"], ["--home=C:/home", "--max-failures=3"], True),
    (["-W", "ignore", "-X", "utf8"], ["--home", "C:/home"], True),
    (["-uBWignore", "-BX", "utf8"], ["--home=C:/home"], True),
    ([], ["--home=C:/home", "--", "child", "--home=C:/other"], True),
    ([], ["--home=C:/home", "--home=C:/home"], False),
    ([], ["--home=C:/home", "--max-failures=bad"], False),
    ([], ["--home=C:/home", "unrelated"], False),
    ([], ["--unknown", "--home=C:/home"], False),
    (["--"], ["--home=C:/home"], False),
    (["-Bc", "pass"], ["--home=C:/home"], False),
    (["other.py"], ["--home", "C:/home"], False),
    (["-m", "other"], ["--home", "C:/home"], False),
    (["-c", "pass"], ["--home", "C:/home"], False),
    ([], ["--", "child", "--home", "C:/home"], False),
    ([], ["--home", "C:/home2", "--", "child", "--home", "C:/home"], False),
    ([], ["--home", "C:/home", "--home", "C:/home2"], False),
    ([], ["--home", "C:/home", "--home=C:/home2"], False),
])
def test_supervisor_python_entrypoint_and_option_boundary(prefix, options, expected):
    argv = ["python.exe", *prefix, "-m", "hermes_cli.gateway_windows_supervisor", *options]
    assert gateway_windows._process_matches_gateway_supervisor(
        "python.exe", argv, home=Path("C:/home"), launcher=Path("C:/home/task.vbs")
    ) is expected
