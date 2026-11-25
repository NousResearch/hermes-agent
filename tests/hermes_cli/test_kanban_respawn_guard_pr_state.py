"""``active_pr`` is a duplicate-work guard, not a tombstone.

t1634 stranded a parent card for 60+ minutes (61 guarded ticks) because a
child's comment still carried its PR URL hours after that PR merged. Only an
OPEN PR can be duplicated; MERGED/CLOSED must release the card, while a lookup
that cannot answer must keep the old fail-closed behaviour.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

PR_URL = "https://github.com/NousResearch/hermes-agent/pull/282"
OWNER, REPO, NUMBER = "NousResearch", "hermes-agent", 282


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def _clean_pr_state_cache():
    """The state cache is process-local; never leak an answer across tests."""
    kbd._respawn_guard_pr_state_cache.clear()
    yield
    kbd._respawn_guard_pr_state_cache.clear()


def _card_with_comment(conn, body: str, title: str = "pr card") -> str:
    tid = kb.create_task(conn, title=title, assignee="dev")
    kb.add_comment(conn, tid, author="dev", body=body)
    return tid


@pytest.mark.parametrize("state", ["MERGED", "CLOSED"])
def test_merged_or_closed_pr_comment_does_not_guard(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """A finished PR is not duplicate work: the card may spawn again."""
    seen: list[tuple[str, str, int]] = []

    def _lookup(owner: str, repo: str, number: int) -> str:
        seen.append((owner, repo, number))
        return state

    monkeypatch.setattr(kbd, "_lookup_pr_state", _lookup)
    with kbc.connect() as conn:
        assert (
            kbd.check_respawn_guard(
                conn, _card_with_comment(conn, f"Opened {PR_URL} for review.")
            )
            is None
        )
    assert seen == [(OWNER, REPO, NUMBER)], (
        "owner/repo/number must come from the comment URL"
    )


def test_open_pr_comment_still_guards(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OPEN PR remains the duplicate-PR signal ``active_pr`` was built for."""
    monkeypatch.setattr(kbd, "_lookup_pr_state", lambda owner, repo, number: "OPEN")
    with kbc.connect() as conn:
        tid = _card_with_comment(conn, f"Opened {PR_URL} for review.")
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


@pytest.mark.parametrize("failure", ["missing-gh", "gh-error", "bad-json"])
def test_lookup_failure_still_guards(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """No answer from ``gh`` must fail closed, exactly like the pre-lookup guard."""

    def _run(*_a, **_k):
        if failure == "missing-gh":
            raise FileNotFoundError("gh")
        if failure == "gh-error":
            return subprocess.CompletedProcess(["gh"], 1, "", "not logged in")
        return subprocess.CompletedProcess(["gh"], 0, "not json", "")

    monkeypatch.setattr(kbd.subprocess, "run", _run)
    with kbc.connect() as conn:
        tid = _card_with_comment(conn, f"Opened {PR_URL} for review.")
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
    # The failure is cached too: a broken gh is not re-probed on every tick.
    assert (
        kbd._respawn_guard_pr_state_cache[(OWNER.lower(), REPO.lower(), NUMBER)][1]
        is None
    )


def test_non_pr_comment_is_unaffected(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A comment with no PR URL never consults GitHub and never guards."""
    calls: list[tuple[str, str, int]] = []
    monkeypatch.setattr(kbd, "_lookup_pr_state", lambda *a: calls.append(a) or "OPEN")
    with kbc.connect() as conn:
        tid = _card_with_comment(conn, "All tests pass; see the summary above.")
        assert kbd.check_respawn_guard(conn, tid) is None
    assert calls == []


def test_pr_state_is_cached_per_repo_and_number(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two cards citing the same PR cost one lookup inside the TTL."""
    calls: list[tuple[str, str, int]] = []

    def _lookup(owner: str, repo: str, number: int) -> str:
        calls.append((owner, repo, number))
        return "MERGED"

    monkeypatch.setattr(kbd, "_lookup_pr_state", _lookup)
    with kbc.connect() as conn:
        first = _card_with_comment(conn, f"Opened {PR_URL}", title="first")
        second = _card_with_comment(conn, f"Also {PR_URL}", title="second")
        assert kbd.check_respawn_guard(conn, first) is None
        assert kbd.check_respawn_guard(conn, second) is None
    assert calls == [(OWNER, REPO, NUMBER)]


def test_one_open_pr_in_a_mixed_comment_still_guards(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closed PR beside an open one must not lift the guard for the open one."""
    states = {"1": "CLOSED", "2": "OPEN"}
    monkeypatch.setattr(
        kbd,
        "_lookup_pr_state",
        lambda owner, repo, number: states[str(number)],
    )
    with kbc.connect() as conn:
        tid = _card_with_comment(
            conn,
            "Landed https://github.com/o/r/pull/1, follow-up https://github.com/o/r/pull/2.",
        )
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"state": "MERGED", "mergedAt": "2026-11-24T00:00:00Z"}, "MERGED"),
        ({"state": "CLOSED", "mergedAt": None}, "CLOSED"),
        ({"state": "OPEN", "mergedAt": None}, "OPEN"),
        ({"state": "open", "mergedAt": None}, "OPEN"),
        ({"state": "SOMETHING_NEW", "mergedAt": None}, None),
    ],
)
def test_lookup_pr_state_parses_gh_json(
    monkeypatch: pytest.MonkeyPatch, payload: dict, expected: str | None
) -> None:
    """Pin the ``gh`` invocation and the state mapping it feeds the guard."""
    calls: list[list[str]] = []

    def _run(cmd, **_kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(kbd.subprocess, "run", _run)
    assert kbd._lookup_pr_state(OWNER, REPO, NUMBER) == expected
    assert calls == [
        [
            "gh",
            "pr",
            "view",
            str(NUMBER),
            "--repo",
            f"{OWNER}/{REPO}",
            "--json",
            "state,mergedAt",
        ]
    ]
