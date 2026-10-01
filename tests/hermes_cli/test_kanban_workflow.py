"""Kanban workflow definition: shape, validation, and the copies that derive from it.

``DEFAULT_WORKFLOW`` is data the kernel does not consult yet (Phase 0 of #54818).
The manual move matrix is pinned to the live dashboard API in
``tests/plugins/test_kanban_workflow_matrix.py``.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_output
from hermes_cli import kanban_workflow as kw
from tools import kanban_tools_schemas


W = kw.DEFAULT_WORKFLOW


def test_default_workflow_is_valid_and_excludes_archived():
    W.validate()
    assert kw.ARCHIVED not in W.keys()


def test_status_copies_derive_from_the_workflow():
    assert kb.VALID_STATUSES == set(W.keys()) | {kw.ARCHIVED}
    # CLI: every listable status has a glyph (was missing triage + review).
    assert set(kanban_output._STATUS_ICONS) == kb.VALID_STATUSES


def test_agent_tool_status_enum_covers_every_status():
    # Was missing scheduled + review, so an agent could not filter on them.
    schema = next(s for s in _all_tool_schemas() if s["name"] == "kanban_list")
    enum = schema["parameters"]["properties"]["status"]["enum"]
    assert set(enum) == kb.VALID_STATUSES


def _all_tool_schemas():
    for value in vars(kanban_tools_schemas).values():
        if isinstance(value, dict) and "name" in value and "parameters" in value:
            yield value


def test_every_trait_is_used_by_the_default():
    # A trait no default column carries describes no current behavior — dead data.
    for trait in kw.TRAITS:
        assert W.keys_with(trait), trait
    with pytest.raises(ValueError):
        W.keys_with("nope")


def test_archive_is_always_a_manual_move():
    assert all(W.can_move(k, kw.ARCHIVED) for k in W.keys())
    assert not any(W.can_move(k, "running") for k in W.keys())


@pytest.mark.parametrize("call", [
    lambda: W.can_move("bogus", kw.ARCHIVED),
    lambda: W.can_move("ready", "bogus"),
])
def test_unknown_column_keys_are_rejected_not_defaulted(call):
    with pytest.raises(ValueError, match="unknown kanban workflow column"):
        call()


@pytest.mark.parametrize("mutate, message", [
    (lambda w: replace(w, columns=w.columns + (w.columns[0],)), "duplicate"),
    (lambda w: replace(w, columns=w.columns + (kw.Column(key=kw.ARCHIVED, label="A"),)), "reserved"),
    (lambda w: replace(w, manual=MappingProxyType({"ready": frozenset({"gone"})})), "manual"),
    (lambda w: replace(w, manual=MappingProxyType({"gone": frozenset({"ready"})})), "manual"),
    (lambda w: replace(w, columns=tuple(replace(c, traits=frozenset()) if c.key == "done" else c
                                        for c in w.columns)), "terminal"),
    (lambda w: replace(w, columns=tuple(replace(c, traits=frozenset({"bogus"})) if c.key == "done" else c
                                        for c in w.columns)), "unknown traits"),
])
def test_validate_rejects_inconsistent_workflows(mutate, message):
    with pytest.raises(ValueError, match=message):
        mutate(W).validate()
