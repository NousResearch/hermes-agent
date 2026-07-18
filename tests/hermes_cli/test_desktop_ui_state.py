"""Durable profile-backed Desktop sidebar state."""

import json
import stat
import sys


def test_missing_state_returns_absent_empty_list(tmp_path):
    from hermes_cli.desktop_ui_state import read_pinned_sessions

    assert read_pinned_sessions(tmp_path) == (False, [])


def test_write_normalizes_deduplicates_and_reads_back(tmp_path):
    from hermes_cli.desktop_ui_state import PINNED_SESSIONS_RELATIVE_PATH, read_pinned_sessions, write_pinned_sessions

    saved = write_pinned_sessions(tmp_path, ["session-a", "", "session-a", 9, "session-b"])

    assert saved == ["session-a", "session-b"]
    assert read_pinned_sessions(tmp_path) == (True, ["session-a", "session-b"])

    path = tmp_path / PINNED_SESSIONS_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["pinned_session_ids"] == ["session-a", "session-b"]
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_corrupt_or_wrong_shaped_state_falls_back_to_legacy_local_cache(tmp_path):
    from hermes_cli.desktop_ui_state import PINNED_SESSIONS_RELATIVE_PATH, read_pinned_sessions

    path = tmp_path / PINNED_SESSIONS_RELATIVE_PATH
    path.parent.mkdir()
    path.write_text('{"pinned_session_ids":"not-an-array"}', encoding="utf-8")

    assert read_pinned_sessions(tmp_path) == (False, [])


def test_write_works_on_platforms_without_fchmod(tmp_path, monkeypatch):
    import hermes_cli.desktop_ui_state as desktop_ui_state

    monkeypatch.delattr(desktop_ui_state.os, "fchmod")

    assert desktop_ui_state.write_pinned_sessions(tmp_path, ["session-a"]) == ["session-a"]
    assert desktop_ui_state.read_pinned_sessions(tmp_path) == (True, ["session-a"])
