"""WIP caps are board-level, not a property of the dispatcher lock winner.

The dispatcher lock is machine-global. The gateway that wins it used to read
only its own profile config. A winner with ``kanban.max_in_progress`` /
``max_in_progress_per_profile`` unset dispatched uncapped (memory-derived
default is None where memory cannot be read) even when another profile had
set a real cap. Live re-read of that winner's file does not fix it.

This is the topology verified live: lock holder unset, another profile set
to a small cap, a 4th ready card for one assignee must stay ``ready``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_constants
from hermes_cli import kanban_dispatch_caps as caps


@pytest.fixture(autouse=True)
def _clear_cap_cache():
    caps._file_caps.clear()
    caps._known_paths.clear()
    yield
    caps._file_caps.clear()
    caps._known_paths.clear()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _bind_lock_holder(monkeypatch, holder: Path) -> None:
    """Point this process at the profile that won the dispatcher lock."""
    monkeypatch.setenv("HERMES_HOME", str(holder))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    hermes_constants._default_hermes_root_memo = None


def _layout(tmp_path: Path, monkeypatch, *, other_yaml: str, holder_yaml: str = "kanban: {}\n"):
    root = tmp_path / "hermes"
    holder = root / "profiles" / "marketing-manager"
    other = root / "profiles" / "reviewer"
    # Default home has no opinion. The bug was treating "unset" as authoritative.
    _write(root / "config.yaml", "kanban: {}\n")
    _write(holder / "config.yaml", holder_yaml)
    _write(other / "config.yaml", other_yaml)
    _bind_lock_holder(monkeypatch, holder)
    return root, holder, other


def test_lock_holder_unset_still_honors_other_profile_cap(tmp_path, monkeypatch):
    """4th ready card for one assignee stays ready when the lock winner has no cap.

    reviewer sets max_in_progress_per_profile=3 (and a host cap of 6, so the
    4th is held by the per-profile cap, not the host cap). marketing-manager,
    the profile whose gateway holds the lock, sets neither.
    """
    _layout(
        tmp_path,
        monkeypatch,
        other_yaml=(
            "kanban:\n"
            "  max_in_progress: 6\n"
            "  max_in_progress_per_profile: 3\n"
        ),
    )
    from hermes_cli.config import load_config, read_raw_config
    from gateway.kanban_watchers_dispatcher import (
        _resolve_dispatcher_settings,
        _reread_dispatcher_settings,
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    # The lock winner's own file really is unset. A passing test that only
    # re-reads this file would still be uncapped — that is the gap in the
    # live-reread suite.
    own = read_raw_config().get("kanban") or {}
    assert own.get("max_in_progress") is None
    assert own.get("max_in_progress_per_profile") is None

    kb.init_db()
    boot = _resolve_dispatcher_settings({}, kb, announce=False)
    live = _reread_dispatcher_settings(load_config, kb, boot)
    assert live.max_in_progress == 6
    assert live.max_in_progress_per_profile == 3
    # Live re-read must not widen back to the lock winner's unset values.
    assert boot.max_in_progress_per_profile == 3

    def _fake_spawn(task, workspace, board=None):
        return 4242

    with kbc.connect_closing() as conn:
        ids = [
            kb.create_task(conn, title=f"card-{i}", assignee="reviewer")
            for i in range(4)
        ]
        kbd.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            max_in_progress=live.max_in_progress,
            max_in_progress_per_profile=live.max_in_progress_per_profile,
            failure_limit=kb.DEFAULT_FAILURE_LIMIT,
        )
        statuses = {
            row["id"]: row["status"]
            for row in conn.execute("SELECT id, status FROM tasks")
        }

    assert len(ids) == 4
    running = [tid for tid in ids if statuses[tid] == "running"]
    ready = [tid for tid in ids if statuses[tid] == "ready"]
    assert len(running) == 3
    assert len(ready) == 1


def test_unreadable_profile_config_does_not_uncap_or_crash(tmp_path, monkeypatch):
    root, _holder, other = _layout(
        tmp_path,
        monkeypatch,
        other_yaml="kanban:\n  max_in_progress_per_profile: 3\n",
    )
    broken = root / "profiles" / "broken"
    _write(broken / "config.yaml", "kanban: [\n")

    mip, per_profile = caps.explicit_dispatch_caps(None, None)
    assert mip is None
    assert per_profile == 3

    # Mid-write / corrupt re-read of the profile that owns the cap must keep
    # the last good value. Skipping it as "unset" would uncap the board.
    _write(other / "config.yaml", "kanban: [\n")
    _mip, kept = caps.explicit_dispatch_caps(None, None)
    assert kept == 3

    # A successful read that omits the key is intentional and clears it.
    _write(other / "config.yaml", "kanban: {}\n")
    _mip, cleared = caps.explicit_dispatch_caps(None, None)
    assert cleared is None


def test_tighter_explicit_value_wins_regardless_of_lock_holder(tmp_path, monkeypatch):
    _layout(
        tmp_path,
        monkeypatch,
        holder_yaml="kanban:\n  max_in_progress: 10\n  max_in_progress_per_profile: 8\n",
        other_yaml="kanban:\n  max_in_progress: 6\n  max_in_progress_per_profile: 3\n",
    )
    assert caps.explicit_dispatch_caps(10, 8) == (6, 3)
    # Lock holder tighter than every other profile still wins.
    assert caps.explicit_dispatch_caps(2, 1) == (2, 1)
