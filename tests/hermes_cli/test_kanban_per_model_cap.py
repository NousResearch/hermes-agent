"""Per-provider/model concurrency limits for explicit Kanban overrides."""
from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import patch

import pytest


@pytest.fixture()
def kanban_with_profiles(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_per_model_cap_test_")
    for profile in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", profile), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    def is_hermes_module(name):
        return (
            name.startswith("hermes_cli")
            or name.startswith("hermes_state")
            or name == "hermes_constants"
        )

    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if is_hermes_module(name)
    }
    for name in saved_modules:
        del sys.modules[name]
    try:
        from hermes_cli import kanban_db

        yield kanban_db
    finally:
        for name in list(sys.modules):
            if is_hermes_module(name):
                del sys.modules[name]
        sys.modules.update(saved_modules)


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


def test_per_model_override_tightens_below_the_global_cap(kanban_with_profiles):
    """live incident, 2026-08-28: a global max_in_progress_per_model that
    suits remote providers (no real concurrency limit) is unsafe for a
    locally-hosted model served by a single-concurrency inference server
    (llama-server -np 1) -- up to 4 tasks were dispatching concurrently
    against the same local model and crashing each other with connection
    timeouts. kanban.max_in_progress_by_model must be able to cap one
    (provider, model) pair tighter than the global default without
    affecting any other model."""
    kb = kanban_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        first = _create_overridden(
            kb, conn, "first", provider="ollama-launch", model="devstral-small-2:24b"
        )
        second = _create_overridden(
            kb, conn, "second", provider="ollama-launch", model="devstral-small-2:24b"
        )
        # A different model stays capped at the global default (4), not
        # dragged down by the override on devstral.
        third = _create_overridden(
            kb, conn, "third", provider="ollama-launch", model="qwen3.5:4b"
        )

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: os.getpid(),
            dry_run=True,
            max_in_progress_per_model=4,
            max_in_progress_by_model={"ollama-launch/devstral-small-2:24b": 1},
        )

    assert set(
        task_id for task_id, _who, _workspace in result.spawned
    ) == {first, third}
    assert result.skipped_per_model_capped == [
        (second, "ollama-launch", "devstral-small-2:24b", 1)
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


def test_local_first_profile_routes_share_model_capacity(kanban_with_profiles):
    """Profile-selected local routes must count before the worker is spawned."""
    kb = kanban_with_profiles
    from pathlib import Path

    root = Path(__import__("os").environ["HERMES_HOME"])
    root.joinpath("config.yaml").write_text(
        "kanban:\n  local_first: true\n", encoding="utf-8"
    )
    for profile in ("alpha", "beta"):
        root.joinpath("profiles", profile, "config.yaml").write_text(
            """
model:
  provider: nous
  default: poolside/laguna-xs-2.1:free
fallback_model:
  - provider: ollama-launch
    model: hermes-cron-fast:latest
""".lstrip(),
            encoding="utf-8",
        )

    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        first = kb.create_task(conn, title="first", assignee="alpha")
        second = kb.create_task(conn, title="second", assignee="beta")

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: os.getpid(),
            dry_run=True,
            max_in_progress_per_model=1,
        )

    assert [task_id for task_id, _who, _workspace in result.spawned] == [first]
    assert result.skipped_per_model_capped == [
        (second, "ollama-launch", "hermes-cron-fast:latest", 1)
    ]


def test_profile_default_routes_share_model_capacity_without_local_first(
    kanban_with_profiles,
):
    """live incident, 2026-08-28: several profiles with no persisted override
    and no kanban.local_first opt-in still all resolve to the same
    single-concurrency local model at spawn time. The per-model cap must
    count that real capacity regardless of the local_first flag -- that
    flag governs a different concern (whether local routes are preferred
    at spawn), not whether concurrent usage is counted accurately."""
    kb = kanban_with_profiles
    from pathlib import Path

    root = Path(__import__("os").environ["HERMES_HOME"])
    # Deliberately no kanban.local_first write here -- this is the default,
    # unset state most installs run with.
    for profile in ("alpha", "beta"):
        root.joinpath("profiles", profile, "config.yaml").write_text(
            """
model:
  provider: ollama-launch
  default: qwen3.5:4b
""".lstrip(),
            encoding="utf-8",
        )

    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        first = kb.create_task(conn, title="first", assignee="alpha")
        second = kb.create_task(conn, title="second", assignee="beta")

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: os.getpid(),
            dry_run=True,
            max_in_progress_per_model=1,
        )

    assert [task_id for task_id, _who, _workspace in result.spawned] == [first]
    assert result.skipped_per_model_capped == [
        (second, "ollama-launch", "qwen3.5:4b", 1)
    ]


def test_codex_primary_profile_is_not_miscounted_against_local_fallback(
    kanban_with_profiles,
):
    """live incident, 2026-08-28 (second occurrence, same day): a profile
    whose PRIMARY model was moved to a remote provider (openai-codex) but
    which still lists a local model as its fallback_providers entry was
    being resolved, with kanban.local_first left at its default (off), to
    that local *fallback* instead of its actual remote primary -- because
    the accounting path reused _resolve_local_first_route unconditionally,
    and that resolver's job is to find the first local route ANYWHERE in
    the chain (for the separate local_first substitution feature), not to
    report what a task genuinely runs. With local_first off, no
    substitution happens, so the task truly runs on its remote primary --
    but the miscount made it appear to share single-concurrency Ollama
    capacity with real local-only profiles, falsely capping unrelated
    ready work that was never going to touch Ollama at all."""
    kb = kanban_with_profiles
    from pathlib import Path

    root = Path(__import__("os").environ["HERMES_HOME"])
    # Deliberately no kanban.local_first write here -- default, unset state.
    root.joinpath("profiles", "alpha", "config.yaml").write_text(
        """
model:
  provider: openai-codex
  default: gpt-5.3-codex-spark
fallback_providers:
  - provider: ollama-launch
    model: qwen3.5:4b
""".lstrip(),
        encoding="utf-8",
    )
    root.joinpath("profiles", "default", "config.yaml").write_text(
        """
model:
  provider: ollama-launch
  default: qwen3.5:4b
""".lstrip(),
        encoding="utf-8",
    )

    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        codex_task = kb.create_task(conn, title="codex primary", assignee="alpha")
        local_task = kb.create_task(conn, title="genuinely local", assignee="default")

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: os.getpid(),
            dry_run=True,
            max_in_progress_per_model=1,
        )

    assert set(
        task_id for task_id, _who, _workspace in result.spawned
    ) == {codex_task, local_task}
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
            alive = {os.getpid(): True}

            def signal_worker(pid, _signal):
                alive[int(pid)] = False

            with patch.object(
                kb,
                "_pid_alive",
                side_effect=lambda pid: alive.get(int(pid), False),
            ):
                assert kb.reclaim_task(
                    conn,
                    first,
                    reason="test release",
                    signal_fn=signal_worker,
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
