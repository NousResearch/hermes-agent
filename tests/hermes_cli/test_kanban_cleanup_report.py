from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="hermes")
    sub = root.add_subparsers(dest="command")
    kc.build_parser(sub)
    return root


def _run_kanban(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    args = _parser().parse_args(["kanban", *argv])
    rc = kc.kanban_command(args)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def test_cleanup_report_requires_dependency_graph_unless_inventory_only(
    kanban_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cleanup-report.json"

    rc, out, err = _run_kanban(["cleanup", "report", "--output", str(output)], capsys)

    assert rc == 2
    assert out == ""
    assert "requires --task or --inventory-only" in err
    assert not output.exists()


def test_cleanup_report_writes_compact_dependency_graph_state(
    kanban_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(conn, title="child", assignee="worker", parents=[parent])

    output = tmp_path / "cleanup-report.json"

    rc, out, err = _run_kanban(
        ["cleanup", "report", "--task", parent, "--output", str(output)],
        capsys,
    )

    assert rc == 0
    assert err == ""
    assert "cleanup report wrote" in out
    assert output.exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["board"] == "default"
    assert report["scope"] == {"type": "dependency_graph", "root_task_id": parent}
    assert [card["id"] for card in report["cards_inspected"]] == [parent, child]
    assert report["state_mutations"] == []
    assert report["comments_added"] == []
    assert report["verification_after_mutation"]["ok"] is True
    assert report["remaining_gated_items"] == [
        {"id": child, "status": "todo", "waiting_on": [parent]}
    ]


def test_cleanup_report_inventory_only_is_the_explicit_all_board_escape_hatch(
    kanban_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="standalone", assignee="worker")
    output = tmp_path / "inventory.json"

    rc, out, err = _run_kanban(
        ["cleanup", "report", "--inventory-only", "--output", str(output)],
        capsys,
    )

    assert rc == 0
    assert err == ""
    assert "usage:" not in out.lower()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scope"] == {"type": "inventory_only", "root_task_id": None}
    assert [card["id"] for card in report["cards_inspected"]] == [task_id]
    assert report["state_mutations"] == []
    assert report["comments_added"] == []
