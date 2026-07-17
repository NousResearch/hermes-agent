"""P0-C (2026-07-18) integration regression: vao_engine.workflow_dsl's
``--board``/``compile_workflow(board=...)`` must fail closed, not silently
redirect, when invoked from a pinned worker targeting a different board.

vao_engine lives in the feelcraft-org-infra repo (products/virtual-ai-office),
a separate, already-merged, read-only dependency this suite imports via
sys.path -- the same cross-repo relationship vao_engine's own CLI already
assumes in reverse (its main() hardcodes ~/.hermes/hermes-agent onto
sys.path to import hermes_cli). compile_workflow() takes kdb as an injected
parameter specifically so it can be tested against this clean worktree's
patched hermes_cli.kanban_db without touching the live installation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_VAO_ENGINE_ROOT = (
    "/home/curioctylab/projects/_system/feelcraft-org-infra/.claude/worktrees/"
    "w1-w6-three-entry-workflow/products/virtual-ai-office"
)
if _VAO_ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _VAO_ENGINE_ROOT)

from vao_engine import workflow_dsl as wfd  # noqa: E402

from hermes_cli import kanban_db as kb  # noqa: E402


FIXTURE_YAML = """
workflow:
  name: p0c-fixture
  nodes:
    - id: only-node
      title: "Only node"
      body: |
        fixture body
      assignee: claude-coder
"""


def _pin_worker(tmp_path, monkeypatch, board="pinned-board"):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    pinned_db = tmp_path / board / "kanban.db"
    pinned_db.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", board)
    return pinned_db


def test_pinned_worker_conflicting_board_fails_closed(tmp_path, monkeypatch):
    """5. Workflow DSL統合回帰: pinned worker相当のenvで
    workflow_dsl fixture --board Y (Y != pinned board) を実行すると、
    exit相当の例外・created=0・reused=0・pinned board task数不変・
    target board task数不変・明確なconflict messageとなること。"""
    pinned_db = _pin_worker(tmp_path, monkeypatch, board="pinned-board")
    other_board_dir = kb.board_dir("different-board")

    with pytest.raises(kb.BoardPinConflictError) as exc_info:
        wfd.compile_workflow(kb, FIXTURE_YAML, board="different-board")

    message = str(exc_info.value)
    assert "pinned-board" in message
    assert "different-board" in message
    assert not other_board_dir.exists()
    assert not pinned_db.exists()


def test_pinned_worker_same_board_succeeds(tmp_path, monkeypatch):
    """6. 同一board Workflow DSL: pinned board Xでworkflow_dsl fixture
    --board Xを実行すると、正常にcompile/submitされ、既存のworker
    containmentは維持されること。"""
    pinned_db = _pin_worker(tmp_path, monkeypatch, board="pinned-board")

    res = wfd.compile_workflow(kb, FIXTURE_YAML, board="pinned-board")

    assert res.created == ["only-node"]
    assert res.reused == []
    assert pinned_db.exists()


def test_dry_run_does_not_silently_accept_board_conflict(tmp_path, monkeypatch):
    """7. dry-run: board conflictを黙って受理しないこと。実submitと
    異なる誤解を生む結果(dry_run=True で「成功したように見える」出力)を
    返さないこと。"""
    pinned_db = _pin_worker(tmp_path, monkeypatch, board="pinned-board")
    other_board_dir = kb.board_dir("different-board")

    with pytest.raises(kb.BoardPinConflictError):
        wfd.compile_workflow(
            kb, FIXTURE_YAML, board="different-board", dry_run=True,
        )

    assert not other_board_dir.exists()
    assert not pinned_db.exists()


def test_dry_run_same_board_still_creates_nothing(tmp_path, monkeypatch):
    """dry-run + same boardは従来どおり何も作らずcompileのみ行う
    (conflict-guard追加がdry-runの既存no-op契約を壊していないこと)。"""
    pinned_db = _pin_worker(tmp_path, monkeypatch, board="pinned-board")

    res = wfd.compile_workflow(
        kb, FIXTURE_YAML, board="pinned-board", dry_run=True,
    )

    assert res.dry_run is True
    assert res.created == []
    assert res.reused == []
    assert not pinned_db.exists()
