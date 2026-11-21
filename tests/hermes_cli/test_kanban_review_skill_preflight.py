"""The review lane refuses a claim whose forced skill does not resolve.

``review_dispatch`` defaults true on the premise that the install carries ``sdlc-review``, but the
skill resolves from the ASSIGNEE PROFILE's skills dir. When it is absent there (or operator-disabled)
the worker raises ``kanban_skills_unavailable`` **before its first model call**, so claiming buys
nothing: two spawn deaths of the card's retry budget and a human-facing "managed worker failed" naming
no cause.

The dispatcher now asks the SAME resolver the worker calls (``build_preloaded_skills_prompt``),
under the SAME profile home ``hermes -p <assignee>`` will use, as the last gate before a worker
exists (after the claim, which is the only authority on the row). An unresolvable forced skill
refuses the spawn and blocks the card once with the real reason.

Two contracts this pins down beyond the refusal itself:

* the refusal must NOT read as a host spawn failure — ``check_respawn_guard`` turns a
  ``spawn_failed`` run stamped ``infrastructure`` into a cooldown, and one profile's missing skill
  must never suspend dispatch (one authoring mistake is not an outage);
* a preflight that cannot run at all fails OPEN — a resolver bug must not be able to stop the lane.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

# Keep the test importable for source-reverted RED runs. The contract still
# exercises the same forced skill on both heads.
FORCED = getattr(kbd, "FORCED_REVIEW_SKILL", "sdlc-review")


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB and NO ``sdlc-review`` skill."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _seed_review_skill(home: Path) -> Path:
    """Install the forced skill where a real install ships it: ``<home>/skills/devops/sdlc-review``."""
    skill = home / "skills" / "devops" / FORCED
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {FORCED}\ndescription: Review Kanban handoffs (test stand-in).\n---\n\n"
        f"# {FORCED}\n\nApprove or request changes.\n",
        encoding="utf-8",
    )
    return skill


def _review_dispatch_on(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}})


def _all_profiles_spawnable(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.profiles as profmod
    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)


def _fake_spawn(spawns: list):
    def spawn(task, workspace, board=None):
        spawns.append(task)
        return 4242
    return spawn


def _park_in_review(conn, title: str, assignee: str = "reviewer") -> str:
    tid = kb.create_task(conn, title=title, assignee=assignee)
    assert kb.claim_task(conn, tid) is not None
    assert kb.request_review(
        conn, tid, summary="done",
        expected_run_id=kb.get_task(conn, tid).current_run_id,
    )
    assert kb.get_task(conn, tid).status == "review"
    return tid


def _events(conn, tid: str, kind: str) -> list[dict]:
    return [
        json.loads(row["payload"]) if row["payload"] else {}
        for row in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
            (tid, kind),
        )
    ]


def _runs(conn, tid: str) -> list:
    return conn.execute(
        "SELECT outcome, status, metadata FROM task_runs WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()


# ---------------------------------------------------------------------------
# 1. The refusal itself
# ---------------------------------------------------------------------------


def test_unresolvable_forced_skill_refuses_the_claim_and_blocks_once(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No worker is spawned, and the card blocks ONCE naming the skill it could not load."""
    _review_dispatch_on(monkeypatch)
    _all_profiles_spawnable(monkeypatch)
    spawns: list = []

    with kbc.connect() as conn:
        tid = _park_in_review(conn, "review me")
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(spawns))
        task = kb.get_task(conn, tid)
        gave_up = _events(conn, tid, "gave_up")
        runs = _runs(conn, tid)

    assert spawns == [], "the claim must be refused BEFORE a worker exists"
    assert res.spawned == []
    assert res.skills_unavailable == [(tid, FORCED)]
    assert task.status == "blocked"
    assert task.consecutive_failures == 1, "one block, not one charge per spawn death"
    assert f"kanban_skills_unavailable: {FORCED}" in (task.last_failure_error or "")

    # Exactly one trip, sticky (so ``recompute_ready`` holds it for an operator instead of
    # re-promoting the card into the same refusal), and it says what actually happened.
    assert len(gave_up) == 1
    assert gave_up[0]["trigger_outcome"] == "skills_unavailable"
    assert gave_up[0]["sticky"] is True
    assert gave_up[0]["retry_status"] == "review"

    # The claim is taken and immediately released as a sticky trip — the run is closed as ``gave_up``
    # (never ``spawn_failed``), and the card carries the reason an operator needs.
    assert [r["outcome"] for r in runs] == ["review_requested", "gave_up"]
    assert task.claim_lock is None


def test_dry_run_reports_the_refusal_instead_of_a_spawn(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--dry-run`` must not promise a spawn the gate would refuse, and must write nothing."""
    _review_dispatch_on(monkeypatch)
    _all_profiles_spawnable(monkeypatch)

    with kbc.connect() as conn:
        tid = _park_in_review(conn, "review me")
        res = kbd.dispatch_once(conn, dry_run=True)
        task = kb.get_task(conn, tid)
        gave_up = _events(conn, tid, "gave_up")

    assert res.spawned == []
    assert res.skills_unavailable == [(tid, FORCED)]
    assert (task.status, task.consecutive_failures) == ("review", 0)
    assert gave_up == []


def test_refusal_is_sticky_and_a_later_tick_does_not_retry_it(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The card stays blocked across ticks: the same refusal must not be re-charged every 30s."""
    _review_dispatch_on(monkeypatch)
    _all_profiles_spawnable(monkeypatch)
    spawns: list = []

    with kbc.connect() as conn:
        tid = _park_in_review(conn, "review me")
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn(spawns))
        res2 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(spawns))
        task = kb.get_task(conn, tid)
        gave_up = _events(conn, tid, "gave_up")

    assert spawns == []
    assert res2.skills_unavailable == []
    assert task.status == "blocked"
    assert task.consecutive_failures == 1
    assert len(gave_up) == 1


# ---------------------------------------------------------------------------
# 2. The infrastructure-cooldown interaction
# ---------------------------------------------------------------------------


def test_refusal_does_not_arm_the_infrastructure_cooldown(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing skill is a per-profile authoring fault, not a host spawn failure.

    ``check_respawn_guard`` reports ``infrastructure_cooldown`` for a card whose latest run is a
    ``spawn_failed`` stamped ``infrastructure``. Stamping the skill refusal that way would let one
    profile's skill set suspend the lane, so both directions are pinned: the guard genuinely fires for
    a real infrastructure run (positive control — otherwise this asserts nothing) and not for the
    refusal.
    """
    assert kb._resolve_rate_limit_cooldown_seconds() > 0, "cooldown disabled: control would be vacuous"
    _review_dispatch_on(monkeypatch)
    _all_profiles_spawnable(monkeypatch)
    spawns: list = []

    now = int(time.time())
    with kbc.connect() as conn:
        # Positive control: the same guard, on a genuine host spawn refusal.
        control = kb.create_task(conn, title="control", assignee="reviewer")
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at, metadata) "
                "VALUES (?, 'reviewer', 'spawn_failed', 'spawn_failed', ?, ?, ?)",
                (control, now, now, json.dumps({"infrastructure": True})),
            )
        assert kbd.check_respawn_guard(conn, control, lane="review") == "infrastructure_cooldown"

        tid = _park_in_review(conn, "review me")
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn(spawns))

        assert kbd.check_respawn_guard(conn, tid, lane="review") is None
        runs = _runs(conn, tid)
        task = kb.get_task(conn, tid)

    assert task.status == "blocked"
    assert "spawn_failed" not in [r["outcome"] for r in runs]
    assert all(not (json.loads(r["metadata"] or "{}") or {}).get("infrastructure") for r in runs)


def test_refused_review_does_not_stop_the_rest_of_the_board(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One refused review must not take the board down with it: ready work still dispatches."""
    _review_dispatch_on(monkeypatch)
    _all_profiles_spawnable(monkeypatch)
    spawns: list = []

    with kbc.connect() as conn:
        tid = _park_in_review(conn, "review me")
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn(spawns))
        ready_id = kb.create_task(conn, title="ready work", assignee="reviewer")
        res2 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(spawns))

    assert [t.id for t in spawns] == [ready_id]
    assert [s[0] for s in res2.spawned] == [ready_id]
    assert kb.get_task(conn, tid).status == "blocked"


# ---------------------------------------------------------------------------
# 3. The resolving paths (no regression)
# ---------------------------------------------------------------------------


def test_resolvable_forced_skill_spawns_and_is_appended(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the skill installed the lane behaves exactly as before: spawn + forced skill."""
    _seed_review_skill(kanban_home)
    _review_dispatch_on(monkeypatch)
    _all_profiles_spawnable(monkeypatch)
    spawns: list = []

    with kbc.connect() as conn:
        tid = _park_in_review(conn, "review me")
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(spawns))

    assert [s[0] for s in res.spawned] == [tid]
    assert res.skills_unavailable == []
    assert [t.id for t in spawns] == [tid]
    assert FORCED in (spawns[0].skills or [])
    assert kb.get_task(conn, tid).status == "running"


def test_operator_disabled_skill_is_refused_before_spawning(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled counts as missing — for the worker (``kanban_skills_unavailable``) and so for us."""
    _seed_review_skill(kanban_home)
    _review_dispatch_on(monkeypatch)
    _all_profiles_spawnable(monkeypatch)
    # Same gate the resolver applies: ``build_preloaded_skills_prompt`` treats a disabled skill as
    # missing, which is exactly what makes the reviewer raise before its first model call.
    monkeypatch.setattr(
        "agent.skill_commands._disabled_skill_names", lambda platform=None: {FORCED},
    )
    spawns: list = []

    with kbc.connect() as conn:
        tid = _park_in_review(conn, "review me")
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(spawns))
        task = kb.get_task(conn, tid)

    assert spawns == []
    assert res.skills_unavailable == [(tid, FORCED)]
    assert task.status == "blocked"


@pytest.mark.parametrize("skill_home", ["dispatcher", "profile"])
def test_preflight_reads_the_assignee_profile_home(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, skill_home: str,
) -> None:
    """The check must read the home the WORKER will resolve from, not the dispatcher's.

    ``hermes -p <assignee>`` resolves skills from the assignee's own profile home. A preflight on the
    dispatcher's home would spawn straight into the reported failure when only the dispatcher has the
    skill, and refuse a card that would have worked when only the profile has it.
    """
    profile_home = kanban_home / "profiles" / "reviewer"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")  # profile identity marker
    _seed_review_skill(kanban_home if skill_home == "dispatcher" else profile_home)
    _review_dispatch_on(monkeypatch)
    _all_profiles_spawnable(monkeypatch)
    spawns: list = []

    with kbc.connect() as conn:
        tid = _park_in_review(conn, "review me")
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(spawns))

    if skill_home == "profile":
        assert [s[0] for s in res.spawned] == [tid]
        assert res.skills_unavailable == []
    else:
        assert spawns == []
        assert res.skills_unavailable == [(tid, FORCED)]
        assert kb.get_task(conn, tid).status == "blocked"


def test_unrunnable_preflight_fails_open(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolver bug must never be able to stall the review lane."""
    _review_dispatch_on(monkeypatch)
    _all_profiles_spawnable(monkeypatch)

    def boom(*_a, **_k):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr("agent.skill_commands.build_preloaded_skills_prompt", boom)
    spawns: list = []

    with kbc.connect() as conn:
        tid = _park_in_review(conn, "review me")
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(spawns))

    assert [s[0] for s in res.spawned] == [tid], "fail open: dispatch as before"
    assert res.skills_unavailable == []
