"""Tests for optional-skills/productivity/windows-tray.

The tray's UI layer is Windows-only (pystray/ctypes); the interesting logic is
the state machine that decides the dot color and the needs-input marker
contract. Both are pure stdlib and run on any host — this file exercises those
behavioral contracts against a synthetic Hermes home, never the real one.
"""
import importlib.util
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "productivity" / "windows-tray"


def _load(module_name, rel_path):
    spec = importlib.util.spec_from_file_location(module_name, SKILL_DIR / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# SKILL.md authoring contract (duplicates of the global CI checks are fine —
# they document what this skill commits to)
# ---------------------------------------------------------------------------

def _frontmatter():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


class TestSkillMetadata:
    def test_name_matches_directory(self):
        assert _frontmatter()["name"] == "windows-tray"

    def test_description_short_and_period(self):
        desc = _frontmatter()["description"]
        assert len(desc) <= 60
        assert desc.endswith(".")

    def test_windows_gated(self):
        assert set(_frontmatter()["platforms"]) == {"windows"}

    def test_required_files_ship(self):
        for rel in ("scripts/hermes_tray.py",
                    "scripts/windows_tray_state.py",
                    "scripts/install_tray.ps1",
                    "scripts/tray-needs-input/plugin.yaml",
                    "scripts/tray-needs-input/__init__.py"):
            assert (SKILL_DIR / rel).exists(), rel

    def test_no_stale_watchdog_artifacts(self):
        # v2 merged the watchdog into the tray; leftovers would contradict SKILL.md
        for gone in ("scripts/tray_watchdog.py", "scripts/start_watchdog.js"):
            assert not (SKILL_DIR / gone).exists(), gone

    def test_no_machine_local_paths_anywhere(self):
        bad = re.compile(r"[A-Za-z]:\\+Users\\+|/home/[a-z0-9_-]+/")
        for p in SKILL_DIR.rglob("*"):
            if p.is_file() and p.suffix in {".py", ".md", ".js", ".ps1", ".yaml"}:
                text = p.read_text(encoding="utf-8")
                m = bad.search(text)
                assert not m, f"{p.name}: machine-local path {m.group(0)!r}"


# ---------------------------------------------------------------------------
# State machine: the four dot states as a truth table over synthetic homes
# ---------------------------------------------------------------------------

state = _load("windows_tray_state", "scripts/windows_tray_state.py")


@pytest.fixture()
def home(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "gui.log").write_text("", encoding="utf-8")
    (tmp_path / "logs" / "agent.log").write_text("", encoding="utf-8")
    return str(tmp_path)


def _db(home, expires_at):
    con = sqlite3.connect(os.path.join(home, "state.db"))
    con.execute("CREATE TABLE session_turn_leases (conversation_id text, expires_at real)")
    con.execute("INSERT INTO session_turn_leases VALUES ('s1', ?)", (expires_at,))
    con.commit()
    con.close()


def _gui(home, status):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    (Path(home) / "logs" / "gui.log").write_text(
        f"{ts},000 INFO tui_gateway.server: tui turn finished: ui_session=a status={status} "
        f"error_retained=False duration=1.0s\n", encoding="utf-8")


def _marker(home, pending, kind="clarify", ts=None):
    json.dump({"pending": pending, "kind": kind, "ts": ts or time.time()},
              open(os.path.join(home, "tray-needs-input.json"), "w", encoding="utf-8"))


class TestComputeState:
    def test_active_lease_no_marker_is_blue(self, home):
        _db(home, time.time() + 60)
        assert state.compute_state(home) == "active"

    def test_active_lease_with_fresh_marker_is_amber(self, home):
        _db(home, time.time() + 60)
        _marker(home, True)
        assert state.compute_state(home) == "needs_input"

    def test_stale_marker_never_lights_amber(self, home):
        _db(home, time.time() + 60)
        _marker(home, True, ts=time.time() - state.NI_STALE_SECS - 10)
        assert state.compute_state(home) == "active"

    def test_expired_lease_is_idle(self, home):
        _db(home, time.time() - 1)
        _gui(home, "complete")
        assert state.compute_state(home) == "idle"

    def test_last_turn_error_sticky(self, home):
        _db(home, time.time() - 1)
        _gui(home, "error")
        assert state.compute_state(home) == "error"

    def test_error_clears_after_next_complete_turn(self, home):
        _db(home, time.time() - 1)
        _gui(home, "error")
        assert state.compute_state(home) == "error"
        _gui(home, "complete")
        assert state.compute_state(home) == "idle"

    def test_missing_db_falls_back_to_log_freshness(self, home):
        # no state.db at all; fresh agent.log => active
        os.utime(os.path.join(home, "logs", "agent.log"), None)
        assert state.compute_state(home) == "active"

    def test_missing_db_and_old_logs_is_idle(self, home):
        old = time.time() - state.ACTIVE_SECS - 10
        for n in ("agent.log", "gui.log"):
            p = os.path.join(home, "logs", n)
            os.utime(p, (old, old))
        assert state.compute_state(home) == "idle"

    def test_marker_only_honored_while_turn_runs(self, home):
        # no lease: a pending marker with no running turn must not light amber
        _db(home, time.time() - 1)
        _gui(home, "complete")
        _marker(home, True)
        assert state.compute_state(home) == "idle"

    def test_interrupted_turn_is_not_error(self, home):
        _db(home, time.time() - 1)
        _gui(home, "interrupted")
        assert state.compute_state(home) == "idle"


# ---------------------------------------------------------------------------
# Icon visibility policy: the resident tray shows/hides its icon as the
# desktop process appears/changes/disappears. Pure function, no win32.
# ---------------------------------------------------------------------------

class TestIconVisibility:
    def test_desktop_appears_shows_icon(self):
        assert state.icon_visibility(100, None, False) == (True, True)

    def test_new_session_clears_stale_hide_flag(self):
        # desktop PID swapped: last session's quit flag must not leak over
        visible, clear = state.icon_visibility(200, 100, True)
        assert visible and clear

    def test_desktop_gone_hides_and_clears_flag(self):
        assert state.icon_visibility(None, 100, False) == (False, True)

    def test_up_and_stable_follows_flag(self):
        assert state.icon_visibility(100, 100, False) == (True, False)
        assert state.icon_visibility(100, 100, True) == (False, False)

    def test_down_and_stable_stays_hidden(self):
        assert state.icon_visibility(None, None, False) == (False, False)

    def test_first_poll_with_desktop_running_shows(self):
        # sentinel prev_pid means "never observed": a desktop already up wins
        assert state.icon_visibility(100, "sentinel", False)[0] is True

    def test_flag_never_forces_visible(self):
        # safety property: visible implies a live desktop process
        for pid in (None, 5):
            for prev in (None, 5, 9, "sentinel"):
                for flag in (False, True):
                    vis, _ = state.icon_visibility(pid, prev, flag)
                    assert not vis or pid is not None


# ---------------------------------------------------------------------------
# Marker contract: the plugin writer and the tray reader must agree on the
# file (name, keys, atomic visibility) without importing Hermes internals.
# ---------------------------------------------------------------------------

class TestMarkerContract:
    @pytest.fixture()
    def plugin(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mod = _load("tni_under_test", "scripts/tray-needs-input/__init__.py")
        return mod, tmp_path

    def test_clarify_open_then_close(self, plugin):
        mod, tmp = plugin
        mod._on_pre_tool_call(tool_name="clarify", session_id="s1")
        d = json.loads((tmp / "tray-needs-input.json").read_text(encoding="utf-8"))
        assert d["pending"] and d["kind"] == "clarify"
        mod._on_post_tool_call(tool_name="clarify", result="{}")
        d = json.loads((tmp / "tray-needs-input.json").read_text(encoding="utf-8"))
        assert not d["pending"]

    def test_other_tools_do_not_touch_marker(self, plugin):
        mod, tmp = plugin
        mod._on_pre_tool_call(tool_name="terminal", args={}, session_id="s1")
        mod._on_post_tool_call(tool_name="terminal", result="ok", session_id="s1")
        assert not (tmp / "tray-needs-input.json").exists()

    def test_approval_open_then_close(self, plugin):
        mod, tmp = plugin
        mod._on_pre_approval(command="rm -rf x", session_key="k")
        assert state.needs_input(str(tmp / "tray-needs-input.json"))
        mod._on_post_approval(command="rm -rf x", session_key="k", choice="once")
        assert not state.needs_input(str(tmp / "tray-needs-input.json"))

    def test_marker_is_readable_by_tray_reader(self, plugin):
        mod, tmp = plugin
        _db(str(tmp), time.time() + 60)  # a live turn: the marker is honored
        mod._write(True, "clarify", "s1")
        # same file, keys, and freshness rule the tray reads each poll
        assert state.compute_state(str(tmp)) == "needs_input"

    def test_session_end_clears_open_marker(self, plugin):
        mod, tmp = plugin
        mod._write(True, "approval", "s1")
        mod._on_session_end(session_id="s1")
        assert not state.needs_input(str(tmp / "tray-needs-input.json"))

    def test_every_hook_returns_none(self, plugin):
        mod, _tmp = plugin
        assert mod._on_pre_tool_call(tool_name="terminal") is None
        assert mod._on_post_tool_call(tool_name="clarify", result="x") is None
        assert mod._on_pre_approval(command="c") is None
        assert mod._on_post_approval(command="c", choice="once") is None
        assert mod._on_session_end() is None
