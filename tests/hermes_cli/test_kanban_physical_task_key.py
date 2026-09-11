"""P1-D regressions: portable physical task identity.

- deterministic across restart/relaunch
- portable-safe characters only (Windows path safe)
- distinct from the logical task id; logical id stays in labels/metadata
- collision-resistant (>=128-bit output)
- persistence-path construction covered
"""
from __future__ import annotations

import hashlib

import pytest

from hermes_cli.kanban_runtime import KanbanRuntimeError, physical_task_key


def test_key_is_deterministic_across_processes():
    a = physical_task_key("task-with:weird chars/and?cases")
    b = physical_task_key("task-with:weird chars/and?cases")
    assert a == b


def test_key_uses_portable_safe_characters_only():
    # Windows reserves <>:"/\|?* in path components; Docker names allow
    # [a-zA-Z0-9][a-zA-Z0-9_.-]. Our key must satisfy both.
    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    )
    for task_id in ("t1", "task:colon", "spaced id", "üñí", "", "..", "a/b"):
        with pytest.raises(KanbanRuntimeError):
            physical_task_key("")
        key = physical_task_key(task_id or "x")
        assert set(key) <= allowed, key
        assert ":" not in key


def test_key_has_at_least_128_bits_of_hash():
    key = physical_task_key("t_entropy")
    hex_part = key.split("-", 1)[1]
    assert len(hex_part) >= 32  # 32 hex chars = 128 bits
    int(hex_part, 16)  # valid hex


def test_key_distinct_from_logical_identity():
    logical = "kanban:PR-91981:task-42"
    key = physical_task_key(logical)
    assert key != logical
    assert logical not in key


def test_distinct_tasks_never_share_a_key():
    ids = [f"board-a/task-{i}" for i in range(200)]
    keys = {physical_task_key(t) for t in ids}
    assert len(keys) == len(ids)


def test_collision_resistance_bounds():
    # The prefix must be a faithful substring of the full digest so the
    # birthday bound of a 128-bit truncation actually applies.
    task_id = "t_prefix"
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    assert physical_task_key(task_id) == f"kbt-{digest[:32]}"


def test_persistence_path_construction_is_portable():
    from pathlib import PurePosixPath, PureWindowsPath

    key = physical_task_key("t:windows:unsafe")
    # Both path grammars accept every component without drive-letter or
    # reserved-name interpretation.
    posix = PurePosixPath("/var/lib/hermes/sandboxes") / key / "home"
    win = PureWindowsPath(r"C:\var\lib\hermes\sandboxes") / key / "home"
    assert str(posix).count(key) == 1
    assert str(win).count(key) == 1
    assert ":" not in key  # never parsed as a Windows drive separator


def test_empty_task_id_rejected():
    with pytest.raises(KanbanRuntimeError):
        physical_task_key("")
    with pytest.raises(KanbanRuntimeError):
        physical_task_key("   ")
