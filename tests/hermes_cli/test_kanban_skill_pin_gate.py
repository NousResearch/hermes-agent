"""Skill-pin gate: a card must not carry a pin the assignee seat cannot load.

A pinned skill the assignee's profile lacks used to kill the worker before its
first model call, and the submitter wrapped that death as
``Kanban owner refused: execution unknown; retained by owner`` — a permissions
sounding message for an authoring bug (fleet incident, cards t_0fc1ff32 /
t_08119859 for a seat without ``verification-before-ship``). Every layer that
refuses now says ``skill not found: <name> (assignee=<profile>)``, and refusal
happens at creation, at dispatch, and at owner admission — never as a crashed
worker.

Breakthrough contract: pin a name the seat provably lacks → red, at all three
gates; a card can no longer reach a worker with a dead pin.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_client as wc
from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
from gateway.session_contract import Principal
from gateway.session_kanban import build_kanban_policy
from hermes_state_runtime import RuntimeStoreError

DEAD_PIN = "verification-before-ship"  # exists on some seats, not on the test seat


@pytest.fixture
def seat_home(tmp_path, monkeypatch):
    """Temp HERMES_HOME with one live profile ``franky`` holding exactly one skill.

    ``only-default`` lives in the default home to prove island semantics: what the
    seat cannot load, the seat cannot be pinned to.
    """
    home = tmp_path / ".hermes"
    (home / "skills" / "only-default").mkdir(parents=True)
    (home / "skills" / "only-default" / "SKILL.md").write_text(
        "---\nname: only-default\ndescription: default-home skill\n---\nbody\n")
    franky = home / "profiles" / "franky"
    (franky / "skills" / "devops" / "sdlc-review").mkdir(parents=True)
    (franky / "skills" / "devops" / "sdlc-review" / "SKILL.md").write_text(
        "---\nname: sdlc-review\ndescription: review lane skill\n---\nbody\n")
    (franky / "SOUL.md").write_text("# franky\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _exact(assignee: str, name: str) -> str:
    return f"skill not found: {name} (assignee={assignee})"


def _rpc_error(detail: str | None):
    data = {"reason": "kanban_skill_pin_unavailable"}
    if detail:
        data["detail"] = detail
    return GatewayRPCError("kanban_skill_pin_unavailable", 4001, data)


# ---------------------------------------------------------------------------
# Resolution semantics of the shared checker
# ---------------------------------------------------------------------------

def test_dead_pin_is_missing_and_island_semantics_hold(seat_home):
    assert kb.missing_pinned_skills("franky", [DEAD_PIN]) == [DEAD_PIN]
    # Profiles are islands: a skill the DEFAULT home has is still dead for franky.
    assert kb.missing_pinned_skills("franky", ["only-default"]) == ["only-default"]
    assert kb.missing_pinned_skills("default", ["only-default"]) == []


def test_loadable_and_unverifiable_pins_pass_the_checker(seat_home):
    assert kb.missing_pinned_skills("franky", ["sdlc-review", " devops/sdlc-review "]) == []
    # A ghost assignee is not this gate's business (the dispatcher refuses it elsewhere).
    assert kb.missing_pinned_skills("nosuchprofile", [DEAD_PIN]) == []
    # Plugin-qualified names cannot be enumerated cross-profile: unverifiable, pass.
    assert kb.missing_pinned_skills("franky", ["someplugin:some-skill"]) == []
    assert kb.missing_pinned_skills(None, [DEAD_PIN]) == []
    assert kb.missing_pinned_skills("franky", None) == []


# ---------------------------------------------------------------------------
# Gate 1 — refuse to CREATE (every surface funnels through create_task)
# ---------------------------------------------------------------------------

def test_create_task_refuses_a_pin_the_assignee_lacks(seat_home):
    with kbc.connect() as conn:
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(conn, title="dead pin", assignee="franky", skills=[DEAD_PIN])
        assert str(excinfo.value) == _exact("franky", DEAD_PIN)
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_create_task_names_every_dead_pin(seat_home):
    with kbc.connect() as conn:
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(conn, title="two dead", assignee="franky",
                           skills=["sdlc-review", DEAD_PIN, "gone-second"])
        assert str(excinfo.value) == f"{_exact('franky', DEAD_PIN)}; {_exact('franky', 'gone-second')}"


def test_create_task_accepts_a_pin_the_assignee_has(seat_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="good pin", assignee="franky", skills=["sdlc-review"])
        task = kb.get_task(conn, task_id)
        assert task is not None and task.skills == ["sdlc-review"]


# ---------------------------------------------------------------------------
# Gate 2 — refuse to DISPATCH: legacy/updated cards never reach a worker
# ---------------------------------------------------------------------------

def test_dispatcher_refuses_spawn_and_books_the_exact_error(seat_home):
    spawned: list[str] = []
    with kbc.connect() as conn:
        # Simulate a card that predates the gate: the dead pin lands via raw SQL,
        # exactly the shape tonight's t_0fc1ff32 had when it was picked up.
        task_id = kb.create_task(conn, title="legacy", assignee="franky")
        conn.execute("UPDATE tasks SET skills = ? WHERE id = ?", (f'["{DEAD_PIN}"]', task_id))
        conn.commit()
        kb.recompute_ready(conn)
        result = kbd.dispatch_once(conn, spawn_fn=lambda task, ws, **kw: spawned.append(task.id) or 1)
        assert spawned == [], "a worker must never start with a dead pin"
        assert all(t != task_id for t, *_ in result.spawned)
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.last_failure_error == _exact("franky", DEAD_PIN)
        # The claim was released and the run closed: the card waits, it did not burn.
        assert task.claim_lock is None
        # Installing the missing skill clears the card on the next tick — the gate
        # must not sticky-block a fixable authoring error.
        seat = Path(seat_home) / "profiles" / "franky" / "skills" / DEAD_PIN
        seat.mkdir(parents=True)
        (seat / "SKILL.md").write_text(f"---\nname: {DEAD_PIN}\ndescription: installed late\n---\nbody\n")
        kb.recompute_ready(conn)
        result2 = kbd.dispatch_once(conn, spawn_fn=lambda task, ws, **kw: spawned.append(task.id) or 1)
        assert spawned == [task_id]
        assert [t for t, *_ in result2.spawned] == [task_id]


def test_review_lane_judges_the_injected_skill_too(seat_home, monkeypatch):
    """The dispatcher's review lane force-loads sdlc-review; the gate judges the
    EFFECTIVE list, so a reviewer seat missing that skill is refused as honestly —
    and a seat that HAS it (franky) must pass the injected name untouched."""
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(cfgmod, "load_config",
                        lambda *a, **k: {"kanban": {"review_dispatch": True}})
    # A second live seat WITHOUT sdlc-review: its review cards must be refused.
    robo = Path(seat_home) / "profiles" / "robo"
    (robo / "skills" / "domain-specific-review").mkdir(parents=True)
    (robo / "skills" / "domain-specific-review" / "SKILL.md").write_text(
        "---\nname: domain-specific-review\ndescription: d\n---\nbody\n")
    (robo / "SOUL.md").write_text("# robo\n")
    spawned: list[str] = []
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="needs review", assignee="robo",
                                 skills=["domain-specific-review"])
        implementation = kb.claim_task(conn, task_id)
        kb.request_review(conn, task_id, summary="ready", expected_run_id=implementation.current_run_id)
        kbd.dispatch_once(conn, spawn_fn=lambda task, ws, **kw: spawned.append(task.id) or 1)
        assert spawned == []
        after = kb.get_task(conn, task_id)
        assert after is not None and after.last_failure_error == _exact("robo", "sdlc-review")
    # Same effective list against a seat that carries sdlc-review: nothing is missing.
    assert kb.missing_pinned_skills("franky", ["sdlc-review"]) == []
    assert kb.missing_pinned_skills("franky", ["domain-specific-review", "sdlc-review"]) == ["domain-specific-review"]


# ---------------------------------------------------------------------------
# Gate 3 — owner admission refuses BEFORE the session is created
# ---------------------------------------------------------------------------

def test_admission_refuses_a_dead_pin_before_owner_admitted(seat_home, monkeypatch):
    import hermes_cli.profiles as profmod
    franky_home = str(Path(seat_home) / "profiles" / "franky")
    monkeypatch.setattr(profmod, "resolve_profile_env", lambda name: franky_home)
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="admission", assignee="franky")
        conn.execute("UPDATE tasks SET skills = ? WHERE id = ?", (f'["{DEAD_PIN}"]', task_id))
        conn.execute("UPDATE tasks SET workspace_kind='dir', workspace_path=? WHERE id = ?",
                     (str(seat_home), task_id))
        conn.commit()
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    assert task is not None
    actor = Principal("owner", franky_home, frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=franky_home), actor=actor,
                                 native_owner=True)
    params = dict(board="default", task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    config = {"model": {"default": "m"}, "platform_toolsets": {"cli": ["terminal"]}}
    with pytest.raises(RuntimeStoreError) as excinfo:
        build_kanban_policy(connection, params, config)
    assert excinfo.value.reason == "kanban_skill_pin_unavailable"
    assert excinfo.value.detail == _exact("franky", DEAD_PIN)


# ---------------------------------------------------------------------------
# Submitter-side honesty (the mislabeled owner-refusal itself)
# ---------------------------------------------------------------------------

def test_submitter_prints_the_refusal_verbatim_never_owner_refused(monkeypatch, capsys):
    async def _raise(params, board_db):
        raise _rpc_error(_exact("franky", DEAD_PIN))

    monkeypatch.setattr(wc, "run", _raise)
    _set_worker_env(monkeypatch)
    assert wc._submit() == 1
    err = capsys.readouterr().err
    assert _exact("franky", DEAD_PIN) in err
    assert "owner refused" not in err


@pytest.mark.asyncio
async def test_submitter_unknown_receipt_quotes_the_booked_skill_reason(monkeypatch, capsys):
    monkeypatch.setattr("gateway.session_kanban.worker_result",
                        lambda db, params: {"exit_code": 1, "last_output": _exact("franky", DEAD_PIN)})
    code = wc._exit_code({"status": "unknown", "admission_id": "a1"},
                         {"board": "b", "task_id": "t", "run_id": 1, "claim_lock": "l"}, "/unused/kanban.db")
    captured = capsys.readouterr()
    assert code == 1
    assert "owner refused" not in captured.err
    assert _exact("franky", DEAD_PIN) in captured.err
    assert "execution unknown" not in captured.err


@pytest.mark.asyncio
async def test_submitter_without_booked_reason_still_says_retained(monkeypatch):
    monkeypatch.setattr("gateway.session_kanban.worker_result", lambda db, params: {})
    with pytest.raises(GatewayClientError, match="execution unknown"):
        wc._exit_code({"status": "unknown", "admission_id": "a1"},
                      {"board": "b", "task_id": "t", "run_id": 1, "claim_lock": "l"}, "/unused/kanban.db")


def _set_worker_env(monkeypatch):
    for key, value in {"HERMES_KANBAN_BOARD": "b", "HERMES_KANBAN_TASK": "t",
                       "HERMES_KANBAN_RUN_ID": "1", "HERMES_KANBAN_CLAIM_LOCK": "l",
                       "HERMES_KANBAN_DB": "/tmp/none.db"}.items():
        monkeypatch.setenv(key, value)


def test_submit_labels_transport_outcomes_not_refusals(monkeypatch, capsys):
    async def _raise(params, board_db):
        raise GatewayClientError("Kanban execution unknown; retained by owner")

    monkeypatch.setattr(wc, "run", _raise)
    _set_worker_env(monkeypatch)
    assert wc._submit() == 1
    captured = capsys.readouterr()
    assert captured.err == "Kanban submit failed: Kanban execution unknown; retained by owner\n"
    assert "owner refused" not in captured.err
