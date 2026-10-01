"""Filing-time rejection of reserved, undispatchable assignees.

Defect class: reserved-assignee dead letter. A card filed to `hermes`/`root`/
`sudo`/`test`/`tmp` is structurally undispatchable — `hermes -p <name>` refuses
to start, so the card dies `spawn_failed` -> `gave_up` and no worker ever runs
it (three cards carry that failure; two P0 assignments stranded this way had to
be re-routed by hand, decisions/2026-09-29-reserved-assignee-dead-letter-reroute.md).

The gate must therefore:
  1. REFUSE `hermes` (and the rest of the reserved-unspawnable set) at create.
  2. ACCEPT `default`, which is in `_RESERVED_NAMES` but genuinely spawns
     (t_e17c4c53 / t_49eddcdf ran to done under profile `default`).
  3. live on the WRITE path, not in a scheduled audit — proven by asserting the
     refusal comes out of `create_task`/`assign_task`/`specify_triage_task` and
     not from any cron.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_assignee_gate as gate
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_graph as kbg


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# --- the predicate ---------------------------------------------------------

@pytest.mark.parametrize("name", ["hermes", "root", "sudo", "test", "tmp"])
def test_reserved_unspawnable_names_are_refused(name):
    """Every reserved name whose `resolve_profile_env` raises is refused."""
    assert gate.reserved_and_unspawnable(name) is True


def test_default_is_reserved_but_ALLOWED():
    """`default` is in _RESERVED_NAMES yet spawns (`resolve_profile_env`
    returns ~/.hermes). Refusing it would break a working lane."""
    from hermes_cli.profiles import _RESERVED_NAMES
    assert "default" in _RESERVED_NAMES          # it IS reserved
    assert gate.is_reserved_name("default") is True
    assert gate.reserved_and_unspawnable("default") is False   # but spawnable


def test_ordinary_names_are_allowed():
    for name in ("turing", "morgan", "worker", "some-unknown-profile"):
        assert gate.reserved_and_unspawnable(name) is False


def test_none_and_empty_are_allowed():
    """None/'' mean 'unassigned' or 'leave unchanged' — not our business."""
    assert gate.reserved_and_unspawnable(None) is False
    assert gate.reserved_and_unspawnable("") is False
    assert gate.reserved_and_unspawnable("   ") is False


# --- the write path (the whole point: refused at FILING, not by a cron) -----

def test_create_task_refuses_reserved_assignee(kanban_home):
    with kbc.connect() as conn:
        with pytest.raises(ValueError) as ei:
            kb.create_task(conn, title="should never land", assignee="hermes")
        assert "RESERVED" in str(ei.value)
        # and nothing was written
        assert kb.list_tasks(conn) == []


def test_create_task_accepts_default_and_ordinary(kanban_home):
    with kbc.connect() as conn:
        tid_default = kb.create_task(conn, title="default lane", assignee="default")
        tid_worker = kb.create_task(conn, title="worker lane", assignee="worker")
        assert kb.get_task(conn, tid_default).assignee == "default"
        assert kb.get_task(conn, tid_worker).assignee == "worker"


def test_assign_task_refuses_reserved_target(kanban_home):
    """The 'reassign to a real profile' cure must not itself point at a
    name that cannot spawn."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="card", assignee="worker")
        with pytest.raises(ValueError):
            kb.assign_task(conn, tid, "root")
        assert kb.get_task(conn, tid).assignee == "worker"   # unchanged


def test_specify_triage_task_refuses_reserved_assignee(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="triage card", triage=True, assignee="worker")
        with pytest.raises(ValueError):
            kb.specify_triage_task(conn, tid, title="x", assignee="sudo")
        assert kb.get_task(conn, tid).assignee == "worker"


def test_decompose_refuses_reserved_child_before_building_graph(kanban_home):
    """A reserved child assignee must abort the fan-out, not half-build it."""
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="root", triage=True, assignee="worker")
        children = [
            {"title": "ok child", "assignee": "worker"},
            {"title": "bad child", "assignee": "hermes"},
        ]
        with pytest.raises(ValueError):
            kbg.decompose_triage_task(conn, root, root_assignee="worker", children=children)
        # atomicity: no children survived the refusal
        assert kb.get_task(conn, root).status == "triage"


def test_request_review_refuses_reserved_reviewer(kanban_home):
    """`--reviewer` is stamped onto the row as its assignee, so it is a fifth
    write path: `kanban create --assignee worker` then
    `request-review --reviewer hermes` re-creates the dead letter."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="card", assignee="worker")
        kb.claim_task(conn, tid, claimer=None)
        with pytest.raises(ValueError):
            kb.request_review(conn, tid, summary="done", reviewer="hermes")
        assert kb.get_task(conn, tid).assignee == "worker"     # unchanged
        # the spawnable control still works
        assert kb.request_review(conn, tid, summary="done", reviewer="turing")
        assert kb.get_task(conn, tid).assignee == "turing"


# --- the dispatcher's kanban.default_assignee is a 5th write path ----------

def test_default_assignee_config_refuses_reserved_unspawnable(monkeypatch):
    """`kanban.default_assignee: hermes` must not stamp the dead lane onto
    unassigned ready rows. `profile_exists('hermes')` is True (regex-only
    resolver), so the exists-check alone would have let it through."""
    from hermes_cli import kanban_db_dispatch as kbd

    # The trap: the exists-check says yes for the reserved name.
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda n: True))
    assert kbd._resolve_default_assignee("hermes") is None
    assert kbd._resolve_default_assignee("root") is None
    # Spawnable names still pass, including reserved-but-spawnable `default`.
    assert kbd._resolve_default_assignee("turing") == "turing"
    assert kbd._resolve_default_assignee("default") == "default"
    assert kbd._resolve_default_assignee("") is None
    assert kbd._resolve_default_assignee(None) is None


# --- the decomposer roster never offers an unspawnable name ----------------

def test_roster_drops_unspawnable_reserved_names(monkeypatch):
    from hermes_cli import kanban_decompose as decomp

    class _P:
        def __init__(self, name):
            self.name, self.description = name, ""

    monkeypatch.setattr(
        decomp.profiles_mod, "list_profiles",
        lambda **kw: [_P("hermes"), _P("default"), _P("turing")],
    )
    roster, valid = decomp._build_roster()
    offered = {e["name"] for e in roster}
    assert "hermes" not in offered   # the lane the LLM must never be offered
    assert "default" in offered      # reserved but spawnable — keep
    assert "turing" in offered
    # valid_names stays "every profile that exists": both its callers
    # (_normalize_assignee_choice, the unknown-assignee info log) mean exactly
    # that by it. The write-path gate is what refuses hermes, not this set.
    assert valid == {"hermes", "default", "turing"}
