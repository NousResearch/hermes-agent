"""Card work class -> default reasoning effort (``tasks.work_class``).

The class is an explicit card field (``hermes kanban create --work-class`` and
``kanban_create(work_class=...)``), never inferred from a card's title or body.
``effective_reasoning_effort`` is the one resolver shared by ``hermes kanban
show`` and the gateway's ``kanban.run`` worker policy, so these assertions cover
both the display and the dispatched effort.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_output as kout


@pytest.fixture
def board(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_mechanical_classes_get_a_lowered_default():
    for work_class in ("deploy", "approval", "qa-handoff", "dependency-bump", "formatting"):
        assert kb.resolve_work_class_effort(work_class) == "low"
    # A merge can hit conflicts and ordering calls, so it keeps a working budget.
    assert kb.resolve_work_class_effort("merge") == "medium"


def test_substantive_and_unknown_classes_inherit():
    for work_class in ("implementation", "design", "diagnosis", "research", None, "", "no-such-class"):
        assert kb.resolve_work_class_effort(work_class) is None


def test_every_valid_class_either_lowers_or_inherits():
    mechanical = set(kb.WORK_CLASS_REASONING_EFFORT)
    assert mechanical <= set(kb.VALID_WORK_CLASSES)
    assert all(kb.resolve_work_class_effort(c) for c in mechanical)
    # Classes outside the mapping table inherit instead of raising at dispatch.
    assert all(kb.resolve_work_class_effort(c) is None for c in set(kb.VALID_WORK_CLASSES) - mechanical)


def test_normalize_work_class_is_case_and_separator_insensitive():
    assert kb.normalize_work_class("  QA_Handoff ") == "qa-handoff"
    assert kb.normalize_work_class("MERGE") == "merge"
    assert kb.normalize_work_class("") is None
    assert kb.normalize_work_class(None) is None


def test_unknown_class_is_rejected_rather_than_silently_inherited():
    with pytest.raises(ValueError) as excinfo:
        kb.normalize_work_class("merg")
    message = str(excinfo.value)
    assert "work_class must be one of" in message
    assert "'merg'" in message


def test_create_task_persists_the_class_without_materialising_an_effort(board):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="merge wt/x into main", assignee="devops",
                                 work_class="Merge")
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.work_class == "merge"
    # The class is a default, not a pin: the stored effort stays NULL so a later
    # mapping change (or an explicit pin) is still authoritative.
    assert task.reasoning_effort is None
    assert kb.effective_reasoning_effort(task) == "medium"


def test_explicit_pin_beats_the_class_default(board):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="merge with a known gnarly conflict",
                                 assignee="devops", work_class="merge", reasoning_effort="xhigh")
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert kb.effective_reasoning_effort(task) == "xhigh"


def test_inheriting_card_reports_no_effective_effort(board):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="design the dispatcher split", assignee="senior-dev",
                                 work_class="design")
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert kb.effective_reasoning_effort(task) is None


def test_work_class_reaches_the_list_and_show_json_surface(board):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="bump ruff", assignee="devops",
                                 work_class="dependency-bump")
        task = kb.get_task(conn, task_id)
    assert task is not None
    payload = kout._task_to_dict(task)
    assert payload["work_class"] == "dependency-bump"
    assert payload["reasoning_effort"] is None


def test_show_label_names_the_source_of_the_effort(board):
    with kbc.connect() as conn:
        class_default = kb.get_task(conn, kb.create_task(
            conn, title="deploy", assignee="devops", work_class="deploy"))
        pinned = kb.get_task(conn, kb.create_task(
            conn, title="merge", assignee="devops", work_class="merge", reasoning_effort="max"))
        substantive = kb.get_task(conn, kb.create_task(
            conn, title="diagnose", assignee="senior-dev", work_class="diagnosis"))
        unclassified = kb.get_task(conn, kb.create_task(conn, title="ad hoc", assignee="devops"))
    assert class_default is not None and pinned is not None
    assert substantive is not None and unclassified is not None
    assert kout.task_reasoning_label(class_default) == "low (class: deploy)"
    assert kout.task_reasoning_label(pinned) == "max (task)"
    assert kout.task_reasoning_label(substantive) == "inherit (profile; class: diagnosis)"
    assert kout.task_reasoning_label(unclassified) == "inherit (profile)"


def test_create_task_refuses_an_unknown_class(board):
    with kbc.connect() as conn, pytest.raises(ValueError):
        kb.create_task(conn, title="typo'd class", assignee="devops", work_class="merg")
