"""Invariant tests for :mod:`hermes_cli.kanban_gate_precheck`.

Pure-function unit tests — no DB, no filesystem, no skill-file inspection
(per AGENTS.md: "never read source code in tests" — this module IS the
extracted, testable logic, not a regex over a skill's prose).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_gate_precheck as gp


@dataclass
class _FakeTask:
    """Minimal stand-in for ``kanban_db.Task`` — only the attributes
    ``gate_precheck`` reads."""

    title: str = "Untitled"
    body: Optional[str] = None
    ears_sentence: Optional[str] = None
    scope_paths: Optional[list] = field(default_factory=lambda: None)


# ---------------------------------------------------------------------------
# (1) fast_lane
# ---------------------------------------------------------------------------

def test_fast_lane_single_doc_file_no_behavior_keywords():
    task = _FakeTask(
        title="Correct a typo in README",
        body="Correct 'recieve' to 'receive'.",
        scope_paths=["README.md"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_FAST_LANE
    assert "README.md" in result["reason"]


def test_fast_lane_json_config_file():
    task = _FakeTask(
        title="Update package version string",
        body=None,
        scope_paths=["package.json"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_FAST_LANE


def test_fast_lane_rejected_when_behavior_keyword_present():
    # Same glob-matching file, but the title names a behavior change —
    # must NOT fast-lane; falls through to the EARS/scope checks instead.
    task = _FakeTask(
        title="Fix race condition in config.yaml loader",
        body=None,
        scope_paths=["config.yaml"],
        ears_sentence=None,
    )
    result = gp.gate_precheck(task)
    assert result["status"] != gp.STATUS_FAST_LANE
    assert result["status"] == gp.STATUS_NOT_GATED


def test_fast_lane_rejected_when_multiple_scope_paths():
    task = _FakeTask(
        title="Update docs",
        body=None,
        scope_paths=["README.md", "CHANGELOG.md"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] != gp.STATUS_FAST_LANE


def test_fast_lane_rejected_when_extension_not_doc_config():
    task = _FakeTask(
        title="One line change",
        body=None,
        scope_paths=["hermes_cli/kanban_db.py"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] != gp.STATUS_FAST_LANE


# ---------------------------------------------------------------------------
# (2) not_gated
# ---------------------------------------------------------------------------

def test_not_gated_when_ears_sentence_is_none():
    task = _FakeTask(
        title="Some task never routed through the EARS gate",
        body="No ears_sentence field populated at all.",
        ears_sentence=None,
        scope_paths=["hermes_cli/kanban_db.py"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_NOT_GATED


def test_not_gated_takes_priority_over_scope_size_when_no_ears_sentence():
    task = _FakeTask(
        title="Big task, not yet gated",
        ears_sentence=None,
        scope_paths=[f"file_{i}.py" for i in range(10)],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_NOT_GATED


# ---------------------------------------------------------------------------
# (3) blocked — EARS-shape heuristic
# ---------------------------------------------------------------------------

def test_blocked_when_ears_sentence_not_ears_shaped():
    task = _FakeTask(
        title="Add a feature",
        ears_sentence="Make the button work better.",
        scope_paths=["a.py"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_BLOCKED
    assert "EARS" in result["reason"]


def test_blocked_when_ears_sentence_missing_shall():
    task = _FakeTask(
        title="Add a feature",
        ears_sentence="When the user clicks submit, the system responds.",
        scope_paths=["a.py"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_BLOCKED


def test_blocked_when_ears_sentence_bundles_two_requirements():
    task = _FakeTask(
        title="Add a feature",
        ears_sentence="When the user clicks submit, the system shall save the form and then also email a receipt.",
        scope_paths=["a.py"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_BLOCKED
    assert "sequencing" in result["reason"]


def test_pass_when_ears_sentence_is_when_shaped():
    task = _FakeTask(
        title="Persist form on submit",
        ears_sentence="When the user clicks submit, the system shall persist the form data.",
        scope_paths=["a.py"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_PASS


def test_pass_when_ears_sentence_is_ubiquitous_shaped():
    task = _FakeTask(
        title="Log all writes",
        ears_sentence="The system shall log every write to the audit table.",
        scope_paths=["a.py"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_PASS


def test_pass_when_ears_sentence_is_while_shaped():
    task = _FakeTask(
        title="Disable input during save",
        ears_sentence="While a save is in progress, the system shall disable the submit button.",
        scope_paths=["a.py"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_PASS


def test_pass_when_ears_sentence_is_if_then_shaped():
    task = _FakeTask(
        title="Reject invalid input",
        ears_sentence="If the input is empty, then the system shall reject the submission.",
        scope_paths=["a.py"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_PASS


def test_pass_when_ears_sentence_is_where_shaped():
    task = _FakeTask(
        title="Enable dark mode toggle",
        ears_sentence="Where dark mode is enabled, the system shall render the dark palette.",
        scope_paths=["a.py"],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_PASS


# ---------------------------------------------------------------------------
# (4) blocked — scope-size threshold
# ---------------------------------------------------------------------------

def test_blocked_when_scope_paths_exceeds_threshold():
    task = _FakeTask(
        title="Large refactor",
        ears_sentence="The system shall migrate every caller to the new API.",
        scope_paths=[f"file_{i}.py" for i in range(gp.SCOPE_SIZE_THRESHOLD + 1)],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_BLOCKED
    assert "decomposable" in result["reason"]


def test_pass_when_scope_paths_at_threshold():
    task = _FakeTask(
        title="Medium refactor",
        ears_sentence="The system shall migrate every caller to the new API.",
        scope_paths=[f"file_{i}.py" for i in range(gp.SCOPE_SIZE_THRESHOLD)],
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_PASS


def test_pass_when_scope_paths_is_none():
    task = _FakeTask(
        title="Task with no declared scope yet",
        ears_sentence="The system shall persist the audit log.",
        scope_paths=None,
    )
    result = gp.gate_precheck(task)
    assert result["status"] == gp.STATUS_PASS


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------

def test_result_status_always_in_valid_statuses():
    for task in (
        _FakeTask(scope_paths=["README.md"]),
        _FakeTask(ears_sentence=None),
        _FakeTask(ears_sentence="not ears shaped"),
        _FakeTask(ears_sentence="The system shall work.", scope_paths=list(range(50))),
    ):
        result = gp.gate_precheck(task)
        assert result["status"] in gp.VALID_STATUSES
        assert isinstance(result["reason"], str) and result["reason"]
