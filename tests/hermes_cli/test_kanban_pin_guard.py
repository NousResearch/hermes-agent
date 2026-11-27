"""Pin guard: a review pin must never land on a coder hop.

A card's ``model_override`` belongs to the card, not to the hop. Two leaks are
covered here:

* ``request_changes()`` handed the card back to the implementer with the
  reviewer's pin still attached, so the next coder run used the review model.
* A timed-out coder hop that was re-queued to ``ready`` kept the pin and was
  re-spawned on it.

The dispatcher now refuses a pinned coder card (``kanban.coder_profiles``)
unless ``force_pin`` is set, blocks it with ``needs_input`` and records a
``pin_refused`` event; ``request_changes()`` clears the pin in the same
transaction it reassigns the implementer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_ops


PINNED_MODEL = "example/review-model"
PINNED_PROVIDER = "nous"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB (real schema from init_db)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home: Path):
    connection = kbc.connect()
    try:
        yield connection
    finally:
        connection.close()


def _events(conn, task_id: str) -> list[tuple[str, dict | None]]:
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [(row["kind"], json.loads(row["payload"]) if row["payload"] else None) for row in rows]


def _kinds(conn, task_id: str) -> list[str]:
    return [kind for kind, _ in _events(conn, task_id)]


def _allow_profiles(monkeypatch: pytest.MonkeyPatch, names=("coder", "reviewer", "other")) -> None:
    """Every assignee in *names* is a spawnable profile this home may claim."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: name in names)
    monkeypatch.setattr(
        cfgmod, "load_config", lambda *args, **kwargs: {"kanban": {"review_dispatch": True}},
    )


def _pinned_review(conn, *, title: str = "pinned review") -> tuple[str, int]:
    """Claim a ``coder`` card, pin it, hand it to ``reviewer`` and claim the review run."""
    task_id = kb.create_task(conn, title=title, assignee="coder")
    assert kb.set_model_override(conn, task_id, PINNED_MODEL, provider=PINNED_PROVIDER)
    implementation = kb.claim_task(conn, task_id, claimer="coder:1")
    assert implementation is not None
    assert kb.request_review(
        conn, task_id, summary="ready", reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
    assert review is not None
    return task_id, review.current_run_id


# ---------------------------------------------------------------------------
# Change 1: request_changes() clears the pin in the same transaction
# ---------------------------------------------------------------------------


def test_request_changes_clears_pin_and_records_old_model(conn) -> None:
    """Send-back drops the review pin, keeps the handoff, and names the old model."""
    task_id, review_run_id = _pinned_review(conn)
    # Parent gating: an unfinished parent added while the review run is live
    # must land the card in ``todo``, exactly as before this change.
    parent_id = kb.create_task(conn, title="upstream", assignee="coder")
    # The reviewer owns the live review run, so it may link its own card.
    assert kb.link_tasks(
        conn, parent_id, task_id, expected_child_run_id=review_run_id,
    ) is False  # child was not ready

    ok, implementer = kb.request_changes(
        conn, task_id, reason="fix tests", expected_run_id=review_run_id,
    )
    assert (ok, implementer) == (True, "coder")

    row = conn.execute(
        "SELECT status, assignee, model_override, provider_override, current_run_id "
        "FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    assert row["model_override"] is None
    assert row["provider_override"] is None
    assert row["assignee"] == "coder"
    assert row["status"] == "todo"  # parent gating preserved
    assert row["current_run_id"] is None

    cleared = [payload for kind, payload in _events(conn, task_id) if kind == "model_override_cleared"]
    assert len(cleared) == 1
    assert cleared[0]["model"] == PINNED_MODEL
    assert cleared[0]["provider"] == PINNED_PROVIDER
    assert cleared[0]["reviewer"] == "reviewer"
    assert cleared[0]["reason"] == "changes_requested"

    # The ordinary handoff event is untouched.
    changes = [payload for kind, payload in _events(conn, task_id) if kind == "changes_requested"]
    assert len(changes) == 1
    assert changes[0]["implementer"] == "coder"
    assert changes[0]["reviewer"] == "reviewer"


def test_request_changes_without_pin_records_no_clear_event(conn) -> None:
    """No pin on the card: nothing is cleared and no event is appended."""
    task_id = kb.create_task(conn, title="unpinned", assignee="coder")
    implementation = kb.claim_task(conn, task_id, claimer="coder:1")
    assert kb.request_review(
        conn, task_id, summary="ready", reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id, claimer="reviewer:1")

    ok, implementer = kb.request_changes(
        conn, task_id, reason="fix", expected_run_id=review.current_run_id,
    )
    assert (ok, implementer) == (True, "coder")

    row = conn.execute(
        "SELECT status, assignee, model_override, provider_override FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row["status"] == "ready"
    assert row["assignee"] == "coder"
    assert row["model_override"] is None
    assert row["provider_override"] is None
    assert "model_override_cleared" not in _kinds(conn, task_id)
    assert "changes_requested" in _kinds(conn, task_id)


def test_request_changes_rolls_back_pin_and_status_on_failure(conn, monkeypatch) -> None:
    """A failure after the UPDATE leaves the pin and the status exactly as they were."""
    task_id, review_run_id = _pinned_review(conn, title="atomic")

    def boom(*args, **kwargs):
        raise RuntimeError("injected failure after the UPDATE")

    monkeypatch.setattr(kb, "_end_run", boom)
    with pytest.raises(RuntimeError, match="injected failure"):
        kb.request_changes(conn, task_id, reason="fix", expected_run_id=review_run_id)

    row = conn.execute(
        "SELECT status, assignee, model_override, provider_override, current_run_id "
        "FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["assignee"] == "reviewer"
    assert row["model_override"] == PINNED_MODEL
    assert row["provider_override"] == PINNED_PROVIDER
    assert row["current_run_id"] == review_run_id
    assert "changes_requested" not in _kinds(conn, task_id)
    assert "model_override_cleared" not in _kinds(conn, task_id)


# ---------------------------------------------------------------------------
# Change 2: dispatch refuses a pinned coder hop
# ---------------------------------------------------------------------------


def test_dispatch_refuses_pinned_coder_card(conn, monkeypatch) -> None:
    """A pinned ``coder`` card is blocked, never claimed or spawned."""
    _allow_profiles(monkeypatch)
    spawns: list[str] = []

    def spawn(task, workspace, **kwargs):
        spawns.append(task.id)
        return 4242

    task_id = kb.create_task(conn, title="pinned coder", assignee="coder")
    assert kb.set_model_override(conn, task_id, PINNED_MODEL, provider=PINNED_PROVIDER)

    res = kbd.dispatch_once(conn, spawn_fn=spawn, coder_profiles=("coder",))

    row = conn.execute(
        "SELECT status, block_kind, block_recurrences, model_override FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "needs_input"
    # The pin is left for the operator to clear; it is just not spent.
    assert row["model_override"] == PINNED_MODEL
    assert spawns == []

    kinds = _kinds(conn, task_id)
    assert "claimed" not in kinds
    assert "worker_bound" not in kinds
    refusals = [payload for kind, payload in _events(conn, task_id) if kind == "pin_refused"]
    assert len(refusals) == 1
    command = f"hermes kanban --board default set-model {task_id} none"
    assert refusals[0]["assignee"] == "coder"
    assert refusals[0]["model"] == PINNED_MODEL
    assert refusals[0]["provider"] == PINNED_PROVIDER
    assert refusals[0]["command"] == command

    assert [entry["task_id"] for entry in res.pin_refused] == [task_id]
    assert res.pin_refused[0]["command"] == command
    assert not res.spawned


def test_cli_dispatch_exits_nonzero_and_prints_pin_refusal(monkeypatch, capsys) -> None:
    """The CLI names the assignee, model and command, and exits non-zero."""
    refusal = {
        "task_id": "t_pinned",
        "assignee": "coder",
        "model": PINNED_MODEL,
        "provider": PINNED_PROVIDER,
        "command": "hermes kanban --board default set-model t_pinned none",
    }
    monkeypatch.setattr(
        kanban_ops.kbd, "dispatch_once",
        lambda *args, **kwargs: kbd.DispatchResult(pin_refused=[refusal]),
    )
    rc = kanban_ops._cmd_dispatch(
        SimpleNamespace(dry_run=False, max=None, failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
                        json=False, force_pin=False)
    )
    output = capsys.readouterr().out
    assert rc != 0
    assert "coder" in output
    assert PINNED_MODEL in output
    assert refusal["command"] in output


def test_cli_dispatch_json_reports_pin_refusal(monkeypatch, capsys) -> None:
    """``--json`` carries the refusal bucket and still exits non-zero."""
    refusal = {
        "task_id": "t_pinned",
        "assignee": "coder",
        "model": PINNED_MODEL,
        "provider": PINNED_PROVIDER,
        "command": "hermes kanban --board default set-model t_pinned none",
    }
    monkeypatch.setattr(
        kanban_ops.kbd, "dispatch_once",
        lambda *args, **kwargs: kbd.DispatchResult(pin_refused=[refusal]),
    )
    rc = kanban_ops._cmd_dispatch(
        SimpleNamespace(dry_run=True, max=None, failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
                        json=True, force_pin=False)
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc != 0
    assert payload["pin_refused"] == [refusal]


def test_dispatch_spawns_unpinned_coder_and_pinned_reviewer(conn, monkeypatch) -> None:
    """No pin on the coder spawns; a reviewer card keeps its pin and spawns."""
    _allow_profiles(monkeypatch)
    spawns: list[str] = []

    def spawn(task, workspace, **kwargs):
        spawns.append(task.assignee or "")
        return 4343

    coder_id = kb.create_task(conn, title="unpinned coder", assignee="coder")
    reviewer_id = kb.create_task(conn, title="pinned reviewer", assignee="reviewer")
    assert kb.set_model_override(conn, reviewer_id, PINNED_MODEL, provider=PINNED_PROVIDER)

    res = kbd.dispatch_once(conn, spawn_fn=spawn)  # default: guard off

    assert {entry[0] for entry in res.spawned} == {coder_id, reviewer_id}
    assert spawns == ["coder", "reviewer"]
    assert not res.pin_refused
    assert "pin_refused" not in _kinds(conn, coder_id)
    assert kb.get_task(conn, reviewer_id).model_override == PINNED_MODEL


def test_force_pin_spawns_pinned_coder_and_records_pin_forced(conn, monkeypatch) -> None:
    """``force_pin`` is the explicit operator override, recorded before the claim."""
    _allow_profiles(monkeypatch)
    spawns: list[str] = []

    def spawn(task, workspace, **kwargs):
        spawns.append(task.id)
        return 4444

    task_id = kb.create_task(conn, title="forced coder", assignee="coder")
    assert kb.set_model_override(conn, task_id, PINNED_MODEL, provider=PINNED_PROVIDER)

    res = kbd.dispatch_once(conn, spawn_fn=spawn, force_pin=True, coder_profiles=("coder",))

    assert [entry[0] for entry in res.spawned] == [task_id]
    assert spawns == [task_id]
    assert not res.pin_refused
    kinds = _kinds(conn, task_id)
    forced = [payload for kind, payload in _events(conn, task_id) if kind == "pin_forced"]
    assert len(forced) == 1
    assert forced[0] == {"assignee": "coder", "model": PINNED_MODEL, "provider": PINNED_PROVIDER}
    # Recorded before the claim, so the audit trail reads cause -> effect.
    assert kinds.index("pin_forced") < kinds.index("claimed")


def test_requeued_timed_out_coder_card_with_pin_is_refused(conn, monkeypatch) -> None:
    """The real timeout path (claim -> stale reclaim) must not re-spawn on the pin."""
    _allow_profiles(monkeypatch)
    spawns: list[str] = []

    def spawn(task, workspace, **kwargs):
        spawns.append(task.id)
        return 4545

    task_id = kb.create_task(conn, title="timed out", assignee="coder")
    assert kb.set_model_override(conn, task_id, PINNED_MODEL, provider=PINNED_PROVIDER)
    claimed = kb.claim_task(conn, task_id, ttl_seconds=1, claimer="coder:1")
    assert claimed is not None

    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 60, task_id),
        )
    assert kb.release_stale_claims(conn, failure_limit=5) == 1
    assert kb.get_task(conn, task_id).status == "ready"

    res = kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=5, coder_profiles=("coder",))

    row = conn.execute(
        "SELECT status, block_kind FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "needs_input"
    assert spawns == []
    assert [entry["task_id"] for entry in res.pin_refused] == [task_id]


def test_coder_profiles_custom_list_refuses_named_profile(conn, monkeypatch) -> None:
    """``kanban.coder_profiles`` naming ``other`` guards that profile too."""
    _allow_profiles(monkeypatch)
    spawns: list[str] = []

    def spawn(task, workspace, **kwargs):
        spawns.append(task.id)
        return 4646

    task_id = kb.create_task(conn, title="other coder", assignee="other")
    assert kb.set_model_override(conn, task_id, PINNED_MODEL, provider=PINNED_PROVIDER)

    res = kbd.dispatch_once(conn, spawn_fn=spawn, coder_profiles=("other",))

    assert spawns == []
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()["status"] == "blocked"
    assert [entry["assignee"] for entry in res.pin_refused] == ["other"]


def test_empty_coder_profiles_refuses_nobody(conn, monkeypatch) -> None:
    """An explicit empty list is the operator opt-out: the pin is not guarded."""
    _allow_profiles(monkeypatch)
    spawns: list[str] = []

    def spawn(task, workspace, **kwargs):
        spawns.append(task.id)
        return 4747

    task_id = kb.create_task(conn, title="opted out", assignee="coder")
    assert kb.set_model_override(conn, task_id, PINNED_MODEL, provider=PINNED_PROVIDER)

    res = kbd.dispatch_once(conn, spawn_fn=spawn, coder_profiles=())

    assert spawns == [task_id]
    assert [entry[0] for entry in res.spawned] == [task_id]
    assert not res.pin_refused


def test_cli_dispatch_reads_coder_profiles_from_config(monkeypatch) -> None:
    """``hermes kanban dispatch`` threads ``kanban.coder_profiles`` from config."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(
        cfgmod, "load_config", lambda *args, **kwargs: {"kanban": {"coder_profiles": ["other"]}},
    )
    captured: dict = {}

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kbd.DispatchResult()

    monkeypatch.setattr(kanban_ops.kbd, "dispatch_once", fake_dispatch_once)
    rc = kanban_ops._cmd_dispatch(
        SimpleNamespace(dry_run=True, max=None, failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
                        json=False, force_pin=False)
    )
    assert rc == 0
    assert captured.get("coder_profiles") == ("other",)
    assert captured.get("force_pin") is False


def test_gateway_settings_resolve_coder_profiles() -> None:
    """The gateway tick reads the same key into its dispatch settings."""
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings

    configured = _resolve_dispatcher_settings({"coder_profiles": ["other", "coder"]}, kb)
    assert configured.coder_profiles == ("other", "coder")
    # An explicit empty list is an opt-out, not "unset".
    assert _resolve_dispatcher_settings({"coder_profiles": []}, kb).coder_profiles == ()
    # Unset keeps the fail-closed default.
    assert _resolve_dispatcher_settings({}, kb).coder_profiles == ()  # opt-in


def test_dry_run_reports_pin_refusal_without_writing(conn, monkeypatch) -> None:
    """A dry run names the refusal but blocks nothing and appends no event."""
    _allow_profiles(monkeypatch)
    spawns: list[str] = []

    def spawn(task, workspace, **kwargs):
        spawns.append(task.id)
        return 4848

    task_id = kb.create_task(conn, title="dry pinned", assignee="coder")
    assert kb.set_model_override(conn, task_id, PINNED_MODEL, provider=PINNED_PROVIDER)
    before = _kinds(conn, task_id)

    res = kbd.dispatch_once(conn, spawn_fn=spawn, dry_run=True, coder_profiles=("coder",))

    assert [entry["task_id"] for entry in res.pin_refused] == [task_id]
    assert not res.spawned
    assert spawns == []
    row = conn.execute(
        "SELECT status, block_kind FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    assert row["status"] == "ready"
    assert row["block_kind"] is None
    assert _kinds(conn, task_id) == before


def test_second_tick_does_not_duplicate_pin_refused_event(conn, monkeypatch) -> None:
    """The blocked card leaves ``ready``, so a second tick adds no new refusal."""
    _allow_profiles(monkeypatch)
    spawns: list[str] = []

    def spawn(task, workspace, **kwargs):
        spawns.append(task.id)
        return 4949

    task_id = kb.create_task(conn, title="tick tick", assignee="coder")
    assert kb.set_model_override(conn, task_id, PINNED_MODEL, provider=PINNED_PROVIDER)

    kbd.dispatch_once(conn, spawn_fn=spawn, coder_profiles=("coder",))
    kbd.dispatch_once(conn, spawn_fn=spawn, coder_profiles=("coder",))

    kinds = _kinds(conn, task_id)
    assert kinds.count("pin_refused") == 1
    assert spawns == []


@pytest.mark.parametrize("value", ["", "   ", " , ", ","])
def test_blank_coder_profiles_string_falls_back_to_default(value):
    """A blank string names no profile: the default applies."""
    from hermes_cli import kanban_db_dispatch as kdd

    assert kdd.normalize_coder_profiles(value) == kdd.DEFAULT_CODER_PROFILES


def test_coder_profiles_list_and_string_forms():
    from hermes_cli import kanban_db_dispatch as kdd

    assert kdd.normalize_coder_profiles([]) == ()
    assert kdd.normalize_coder_profiles("coder, reviewer") == ("coder", "reviewer")
