"""Regression coverage for Kanban CLI process exit status propagation and the author contract.

Every case drives the real CLI in a subprocess against a throwaway HERMES_HOME, then reads the
attribution back out of that home's ``task_events`` — never a mock, so a silent "author resolved
somewhere" regression cannot pass.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]

AUTHOR_REQUIRED = "kanban: cannot determine author; pass --author (or set HERMES_PROFILE)"


def _run_hermes(
    home: Path,
    *args: str,
    marker: bool = False,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_HOME"] = str(home)
    for name in (
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        # Unpinned is the default under test: the invoking process's own profile must not leak in.
        "HERMES_PROFILE",
        "HERMES_PROFILE_NAME",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if marker:
        env["HERMES_DELEGATED_CHILD_CONTEXT"] = "1"
    else:
        env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _home(tmp_path: Path, name: str = "hermes") -> Path:
    home = tmp_path / name
    home.mkdir()
    return home


def _live_profile(home: Path, name: str) -> Path:
    """A named profile the CLI will resolve: the directory plus an identity marker.

    ``named_profile_is_live`` (hermes_constants) requires one of the identity
    markers, so a bare directory is refused as a ghost shell before the
    author-signal path under test is ever reached.
    """
    profile = home / "profiles" / name
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    return profile


def _kanban_db(home: Path) -> Path:
    """The default-board DB the CLI writes for ``HERMES_KANBAN_HOME=<home>``."""
    dbs = sorted(home.glob("**/kanban.db"))
    assert dbs, f"no kanban.db under {home}"
    return dbs[0]


def _commented_author(home: Path, task_id: str) -> str | None:
    """Author recorded on the newest ``commented`` event for *task_id*, straight from the store."""
    with sqlite3.connect(_kanban_db(home)) as conn:
        row = conn.execute(
            "SELECT json_extract(payload, '$.author') FROM task_events "
            "WHERE task_id = ? AND kind = 'commented' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    return row[0] if row else None


def _create_task(home: Path, *extra: str) -> str:
    """Create a task with an explicit ``--created-by`` (creating is itself author-attributing)."""
    created = _run_hermes(home, "kanban", "create", "author probe", "--json", "--created-by", "probe", *extra)
    assert created.returncode == 0, created.stderr
    return json.loads(created.stdout)["id"]


def test_delegated_child_kanban_cli_refusal_returns_nonzero_exit_status(tmp_path):
    """A printed Kanban mutation refusal must not look like CLI success."""
    home = _home(tmp_path)
    task_id = _create_task(home)

    refused = _run_hermes(
        home,
        "kanban",
        "comment",
        task_id,
        "must be refused",
        marker=True,
    )

    assert refused.returncode == 1
    assert "delegate_task child contexts cannot mutate Kanban tasks via the CLI" in refused.stderr


def test_unpinned_author_with_no_sticky_profile_fails_loudly(tmp_path):
    """No explicit signal and no active_profile: refuse instead of inventing an author."""
    home = _home(tmp_path)
    task_id = _create_task(home)

    done = _run_hermes(home, "kanban", "comment", task_id, "unpinned")

    assert done.returncode != 0
    assert AUTHOR_REQUIRED in done.stderr
    assert _commented_author(home, task_id) is None


def test_unpinned_author_does_not_borrow_the_sticky_active_profile(tmp_path):
    """The flappable ``active_profile`` must not silently name the lane that last ran `profile use`."""
    home = _home(tmp_path)
    _live_profile(home, "research-stl")
    task_id = _create_task(home)
    # Sticky profile: the pre-fix fallback attributed the comment to it.
    (home / "active_profile").write_text("research-stl\n", encoding="utf-8")

    done = _run_hermes(home, "kanban", "comment", task_id, "must not be research-stl")

    assert done.returncode != 0
    assert AUTHOR_REQUIRED in done.stderr
    assert _commented_author(home, task_id) is None


def test_explicit_author_flag_attributes_to_that_name(tmp_path):
    home = _home(tmp_path)
    task_id = _create_task(home)

    done = _run_hermes(home, "kanban", "comment", task_id, "from the flag", "--author", "foo")

    assert done.returncode == 0, done.stderr
    assert _commented_author(home, task_id) == "foo"


def test_hermes_profile_env_attributes_to_that_name(tmp_path):
    home = _home(tmp_path)
    task_id = _create_task(home)

    done = _run_hermes(
        home,
        "kanban",
        "comment",
        task_id,
        "from the env",
        env_extra={"HERMES_PROFILE": "bar"},
    )

    assert done.returncode == 0, done.stderr
    assert _commented_author(home, task_id) == "bar"


def test_explicit_profile_flag_still_attributes_to_that_profile(tmp_path):
    """`hermes -p X kanban …` is an explicit signal and must keep working."""
    home = _home(tmp_path)
    _live_profile(home, "research-stl")
    task_id = _create_task(home)

    done = _run_hermes(home, "-p", "research-stl", "kanban", "comment", task_id, "from -p")

    assert done.returncode == 0, done.stderr
    assert _commented_author(home, task_id) == "research-stl"


# --- The calling surface's explicit identity: the gateway /kanban path ---
#
# The gateway process carries no HERMES_PROFILE and serves many chats from one process, so it
# hands run_slash the routed profile instead of exporting an env var (which concurrent chats
# would share). These two exercise that binding directly: it must attribute, and it must not
# outlive the call.


def test_run_slash_author_binding_attributes_and_does_not_outlive_the_call(tmp_path, monkeypatch):
    home = _home(tmp_path)
    task_id = _create_task(home)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)

    from hermes_cli import kanban as kc
    from hermes_cli.kanban_author import KanbanAuthorRequired

    output = kc.run_slash(f"comment {task_id} routed from the gateway", author="research-stl")

    assert "Comment added" in output
    assert _commented_author(home, task_id) == "research-stl"
    # Bound for the call only: the next shell-out from this process is unpinned again.
    with pytest.raises(KanbanAuthorRequired):
        kc.resolve_author()


def test_run_slash_author_binding_beats_the_ambient_profile_env(tmp_path, monkeypatch):
    """A stale HERMES_PROFILE in a long-lived gateway must not outvote the routed chat profile."""
    home = _home(tmp_path)
    task_id = _create_task(home)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "stale-process-profile")
    monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)

    from hermes_cli import kanban as kc

    output = kc.run_slash(f"comment {task_id} routed", author="research-stl")

    assert "Comment added" in output
    assert _commented_author(home, task_id) == "research-stl"


def test_run_slash_without_an_author_still_refuses_when_unpinned(tmp_path, monkeypatch):
    """The interactive CLI path passes no author: unpinned verbs must refuse in-band, not 500."""
    home = _home(tmp_path)
    task_id = _create_task(home)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)

    from hermes_cli import kanban as kc

    output = kc.run_slash(f"comment {task_id} unpinned")

    assert AUTHOR_REQUIRED in output
    assert _commented_author(home, task_id) is None


if __name__ == "__main__":  # pragma: no cover - convenience for a manual run
    raise SystemExit(pytest.main([__file__, "-v"]))
