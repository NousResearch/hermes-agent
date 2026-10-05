"""Lane-wide skill injection: ``kanban.injected_skills``.

Every card's worker -- not only a review run -- can carry a harness-injected set
of skills resolved per LANE at claim time. These tests pin the card contract:

* the resolver's shape and specificity order: exact lane -> prefix glob ->
  ``"*"`` floor, UNIONED (a lane bucket ADDS to the floor, it does not replace
  it), most-specific bucket first, deduped;
* the two convenience forms: a flat list (lane-independent floor) and an absent
  or empty key (no injection at all -- no behaviour change for a host that
  configures nothing);
* the claim path merging card + lane + review name lists with dedupe, the
  card's own names keeping their slot on conflict, and the review lane staying a
  SUPERSET of the lane bucket;
* the ADVISORY contract: an injected name that does not resolve for the lane is
  RECORDED ON THE CARD and the run still starts -- a missing skill must never
  kill a run at INIT.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _patch_config(monkeypatch: pytest.MonkeyPatch, kanban: dict) -> None:
    """Point the dispatcher's ``_kanban_config()`` at *kanban*.

    ``_kanban_config`` imports ``load_config`` lazily from ``hermes_cli.config``,
    so patching it on that module is what the real code path reads.
    """
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {"kanban": kanban})


def _capture_spawn():
    captured: list[dict] = []

    def spawn(task, workspace):
        captured.append({
            "skills": list(task.skills or []),
            "advisory": list(getattr(task, "advisory_skills", ()) or ()),
        })
        return None

    return captured, spawn


def _wire_dispatch(monkeypatch: pytest.MonkeyPatch, resolvable) -> None:
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(kbd, "_profile_skill_resolvable", resolvable)
    monkeypatch.setattr(kbd, "check_respawn_guard", lambda _c, _t, **_k: None)


# ---------------------------------------------------------------------------
# Resolver: shape, specificity order, union semantics
# ---------------------------------------------------------------------------


def test_resolver_unions_exact_glob_and_floor_most_specific_first(monkeypatch) -> None:
    _patch_config(monkeypatch, {"injected_skills": {
        "*": ["floor-a", "shared"],
        "platform-*": ["sdlc", "shared"],
        "platform-coder": ["coder-only", "sdlc"],
    }})
    # exact bucket first, then the prefix glob, then the floor; each bucket ADDS
    # to the ones before it and duplicates are dropped on first sight.
    assert kbd.injected_skills_for("platform-coder") == (
        "coder-only", "sdlc", "shared", "floor-a",
    )


def test_resolver_unknown_lane_gets_floor_only(monkeypatch) -> None:
    _patch_config(monkeypatch, {"injected_skills": {
        "*": ["floor"],
        "platform-*": ["only-for-platform"],
    }})
    assert kbd.injected_skills_for("research-coder") == ("floor",)


def test_resolver_narrower_prefix_glob_wins_ordering(monkeypatch) -> None:
    _patch_config(monkeypatch, {"injected_skills": {
        "platform-*": ["wide"],
        "platform-coder-*": ["narrow"],
    }})
    assert kbd.injected_skills_for("platform-coder-x") == ("narrow", "wide")


def test_resolver_flat_list_is_the_floor(monkeypatch) -> None:
    _patch_config(monkeypatch, {"injected_skills": ["alpha", "beta"]})
    assert kbd.injected_skills_for("any-lane") == ("alpha", "beta")
    assert kbd.injected_skills_for(None) == ("alpha", "beta")


def test_resolver_string_bucket_is_a_single_name(monkeypatch) -> None:
    _patch_config(monkeypatch, {"injected_skills": {"*": "solo"}})
    assert kbd.injected_skills_for("x") == ("solo",)


@pytest.mark.parametrize("value", [None, {}, [], (), "   "])
def test_resolver_absent_or_empty_yields_nothing(monkeypatch, value) -> None:
    _patch_config(monkeypatch, {"injected_skills": value})
    assert kbd.injected_skills_for("platform-coder") == ()


def test_resolver_missing_key_yields_nothing(monkeypatch) -> None:
    _patch_config(monkeypatch, {})
    assert kbd.injected_skills_for("platform-coder") == ()


# ---------------------------------------------------------------------------
# Claim path: merge, dedupe, and the advisory (never fatal) contract
# ---------------------------------------------------------------------------


def test_ready_card_is_injected_and_keeps_its_own_skills_first(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NON-review card carries the lane's injected skills, too.

    The card's own skill list keeps its slot; injected names are appended and
    deduped across both sources.
    """
    _patch_config(monkeypatch, {"injected_skills": {
        "builder": ["lane-skill"],
        "*": ["floor-skill"],
    }})
    _wire_dispatch(monkeypatch, lambda _home, _name: True)
    captured, spawn = _capture_spawn()

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="plain work", assignee="builder",
            skills=["card-skill", "floor-skill"],
        )
        result = kbd.dispatch_once(conn, spawn_fn=spawn)

    assert task_id in [t[0] for t in result.spawned]
    assert captured == [{
        "skills": ["card-skill", "floor-skill", "lane-skill"],
        "advisory": ["lane-skill", "floor-skill"],
    }]


def test_bogus_injected_name_is_recorded_on_the_card_and_run_still_starts(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An injected name the lane cannot resolve must never be fatal.

    It is skipped (never handed to the worker's preload loader, which raises
    ``Unknown skill(s)`` and kills the run at INIT), flagged ADVISORY so the
    worker's own loader only warns, and RECORDED ON THE CARD so the operator
    can see the review/execution started without it.
    """
    _patch_config(monkeypatch, {
        "injected_skills": {"*": ["real-skill", "bogus-skill"]},
    })
    _wire_dispatch(monkeypatch, lambda _home, name: name != "bogus-skill")
    captured, spawn = _capture_spawn()

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="plain work", assignee="builder")
        result = kbd.dispatch_once(conn, spawn_fn=spawn)
        events = [row[0] for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
        ).fetchall()]
        comments = [row[0] for row in conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ? ORDER BY id", (task_id,),
        ).fetchall()]

    assert task_id in [t[0] for t in result.spawned]
    assert captured == [{
        "skills": ["real-skill"],
        "advisory": ["real-skill", "bogus-skill"],
    }]
    assert "injected_skill_skipped" in events
    assert any("do not resolve for profile" in body for body in comments)


def test_review_lane_remains_a_superset_of_the_lane_bucket(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review run gets the lane bucket AND the review list."""
    _patch_config(monkeypatch, {
        "review_dispatch": True,
        "review_skills": ["sdlc-review"],
        "injected_skills": {"*": ["floor-skill"]},
    })
    _wire_dispatch(monkeypatch, lambda _home, _name: True)
    captured, spawn = _capture_spawn()

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="review me", assignee="builder")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.request_review(
            conn, task_id, summary="ready",
            reviewer="reviewer", expected_run_id=implementation.current_run_id,
        )
        result = kbd.dispatch_once(conn, spawn_fn=spawn)

    assert task_id in [t[0] for t in result.spawned]
    assert captured == [{"skills": ["floor-skill", "sdlc-review"],
                         "advisory": ["floor-skill", "sdlc-review"]}]
