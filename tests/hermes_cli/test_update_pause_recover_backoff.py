"""A launch's paused-gateway recovery never makes every command wait on the same unready relaunch.

``recover()`` runs before every command (chat, doctor, a Desktop-spawned serve). The Windows resume
it calls waits out a readiness budget per relaunched gateway; that resume cannot run off Windows, so
a stand-in reports what it was handed and fails the way an unready relaunch does. The record, the
claim, the hand-back and the backoff are the real ones, on real files.
"""

from __future__ import annotations

import json
import sys

import pytest

from hermes_cli import update_pause_record as pause_record

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in replaces the native resume")


def _claims() -> list:
    return pause_record._claims(pause_record.record_path())


def test_an_unready_relaunch_stalls_one_launch_per_backoff_window_and_stays_owed(tmp_path, monkeypatch):
    import hermes_cli.update_cmd_windows as w
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    pause_record.write(pause_record.stamp_tree({"resume_needed": True, "profiles": {"alpha": 4194000}}),
                       owner=pause_record.UNOWNED)  # a killed update's set; pid 4194000 is long gone
    handed = []

    def resume(token):
        handed.append(sorted(token.get("profiles") or {}))
        print("relaunched; waiting for readiness")
        raise RuntimeError("Windows gateway relaunch after update was not verified alive: alpha")

    monkeypatch.setattr(w, "_resume_windows_gateways_after_update", resume)
    pause_record.recover(["status"])
    pause_record.recover(["chat"])
    assert handed == [["alpha"]], "a launch inside the backoff waited on the unready relaunch again"
    [claim] = _claims()
    token = pause_record.read(claim)["token"]
    assert token["profiles"] == {"alpha": 4194000}, "the unready gateway's restart debt was dropped"
    assert token["relaunch_retry"]["profile:alpha"]["attempts"] == 1

    body = json.loads(claim.read_text(encoding="utf-8"))
    body["token"]["relaunch_retry"]["profile:alpha"]["next_at"] = 0  # the window elapsed
    claim.write_text(json.dumps(body), encoding="utf-8")
    pause_record.recover(["status"])
    assert handed == [["alpha"], ["alpha"]], "the debt was not retried once its backoff elapsed"
    [claim] = _claims()
    assert pause_record.read(claim)["token"]["relaunch_retry"]["profile:alpha"]["attempts"] == 2
