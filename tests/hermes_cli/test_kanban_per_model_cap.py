"""Per-provider/model concurrency limits for explicit Kanban overrides."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def kanban_with_profiles(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_per_model_cap_test_")
    for profile in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", profile), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db

    yield kanban_db


def _create_overridden(kb, conn, title, *, assignee="alpha", provider, model):
    return kb.create_task(
        conn,
        title=title,
        assignee=assignee,
        provider_override=provider,
        model_override=model,
    )


def test_same_explicit_provider_model_is_capped(kanban_with_profiles):
    kb = kanban_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        first = _create_overridden(
            kb, conn, "first", provider="ollama-launch", model="qwen3.6:27b"
        )
        second = _create_overridden(
            kb, conn, "second", provider="ollama-launch", model="qwen3.6:27b"
        )

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: os.getpid(),
            dry_run=True,
            max_in_progress_per_model=1,
        )

    assert [task_id for task_id, _who, _workspace in result.spawned] == [first]
    assert result.skipped_per_model_capped == [
        (second, "ollama-launch", "qwen3.6:27b", 1)
    ]


def test_different_provider_or_model_and_unoverridden_tasks_remain_independent(
    kanban_with_profiles,
):
    kb = kanban_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_ids = [
            _create_overridden(
                kb, conn, "ollama qwen", provider="ollama-launch", model="qwen3.6:27b"
            ),
            _create_overridden(
                kb, conn, "ollama coder", provider="ollama-launch", model="qwen3-coder:30b"
            ),
            _create_overridden(
                kb, conn, "remote qwen", provider="openrouter", model="qwen3.6:27b"
            ),
            kb.create_task(conn, title="profile default", assignee="alpha"),
        ]

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: os.getpid(),
            dry_run=True,
            max_in_progress_per_model=1,
        )

    assert [task_id for task_id, _who, _workspace in result.spawned] == task_ids
    assert result.skipped_per_model_capped == []


@pytest.mark.parametrize("release", ["completed", "reclaimed"])
def test_terminal_or_reclaimed_task_releases_model_capacity(
    kanban_with_profiles, release
):
    kb = kanban_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        first = _create_overridden(
            kb, conn, "first", provider="ollama-launch", model="qwen3.6:27b"
        )
        second = _create_overridden(
            kb, conn, "second", provider="ollama-launch", model="qwen3.6:27b"
        )

        initial = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: os.getpid(),
            max_in_progress_per_model=1,
        )
        assert [task_id for task_id, _who, _workspace in initial.spawned] == [first]
        assert initial.skipped_per_model_capped == [
            (second, "ollama-launch", "qwen3.6:27b", 1)
        ]

        if release == "completed":
            assert kb.complete_task(conn, first, result="done")
        else:
            assert kb.reclaim_task(
                conn,
                first,
                reason="test release",
                signal_fn=lambda _pid, _claim_lock: True,
            )
            conn.execute("UPDATE tasks SET priority = 1 WHERE id = ?", (second,))

        following = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: os.getpid(),
            max_in_progress_per_model=1,
        )

    assert [task_id for task_id, _who, _workspace in following.spawned] == [second]
    if release == "completed":
        assert following.skipped_per_model_capped == []
    else:
        assert following.skipped_per_model_capped == [
            (first, "ollama-launch", "qwen3.6:27b", 1)
        ]
