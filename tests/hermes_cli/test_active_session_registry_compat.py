"""Registry entries stay readable across the authority-lease marker (HERMES 37.12).

A tracked claim with no stored session id writes nothing: one empty-id entry made every strict
reader raise (guards, claims, its own owner's release) until removed by hand. The authority marker
lives inside ``metadata``, so a pre-marker reader accepts marked entries and the current reader
accepts pre-marker ones; mixed-build fleets share one registry."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import active_sessions

# The last revision before the marker: its reader is the one older builds still run.
_PRE_MARKER_REVISION = "8edc5ea255aaf17b1f1f944bc2dcce52fce06fa8"
_REPO = Path(__file__).resolve().parents[2]


def _registry(home):
    return home / "runtime" / "active_sessions.json"


def test_a_tracked_claim_without_a_session_id_writes_nothing(tmp_path):
    lease, refusal = active_sessions.try_acquire_active_session(
        session_id="", surface="desktop", config={}, registry_home=tmp_path, track_liveness=True)
    assert refusal is None and lease is not None and not lease.enabled
    lease.release()
    assert not _registry(tmp_path).exists()
    # Every strict reader still answers for real sessions.
    with active_sessions.active_session_liveness_guard("other", registry_home=tmp_path) as held:
        assert held is False
    real, refusal = active_sessions.try_acquire_active_session(
        session_id="real", surface="desktop", config={}, registry_home=tmp_path, track_liveness=True)
    assert real is not None, refusal
    assert [e["session_id"] for e in active_sessions.active_session_registry_snapshot(tmp_path, strict=True)] == ["real"]
    real.release()


def _pre_marker_reader(monkeypatch):
    shown = subprocess.run(["git", "show", f"{_PRE_MARKER_REVISION}:hermes_cli/active_sessions.py"],
                           cwd=_REPO, capture_output=True, text=True, check=False)
    if shown.returncode != 0:
        pytest.skip(f"pre-marker revision not in this checkout: {shown.stderr.strip()}")
    spec = importlib.util.spec_from_loader("pre_marker_active_sessions", loader=None)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    exec(compile(shown.stdout, "pre_marker_active_sessions.py", "exec"), module.__dict__)
    return module


def test_marked_and_unmarked_entries_read_on_both_sides_of_the_marker(tmp_path, monkeypatch):
    from tui_gateway import server
    old = _pre_marker_reader(monkeypatch)
    pane, refusal = server._claim_active_session_slot(
        "pane-chat", live_session_id="pane-1", surface="desktop", profile_home=tmp_path)
    assert pane is not None, refusal
    unmarked, refusal = old.try_acquire_active_session(
        session_id="old-chat", surface="desktop", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "old-pane"}, track_liveness=True)
    assert unmarked is not None, refusal

    entries = {e["session_id"]: e for e in active_sessions._read_entries(_registry(tmp_path), strict=True)}
    assert set(entries) == {"pane-chat", "old-chat"}
    assert active_sessions.is_authority_lease(entries["pane-chat"])
    assert not active_sessions.is_authority_lease(entries["old-chat"])
    assert {e["session_id"] for e in old._read_entries(_registry(tmp_path), strict=True)} == {"pane-chat", "old-chat"}
    # The old reader's guard still fences the marked lease, and its release drops only its own.
    with old.active_session_liveness_guard("pane-chat", registry_home=tmp_path) as held:
        assert held is True
    old.release_active_session(unmarked)
    pane.release()
    assert active_sessions._read_entries(_registry(tmp_path), strict=True) == []
