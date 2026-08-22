"""Parser-level regression tests for `hermes kanban` CLI options."""

from __future__ import annotations

import argparse

from hermes_cli import kanban


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    kanban.build_parser(subparsers)
    return parser


def test_kanban_block_parses_kind_with_reason_tokens():
    parser = _build_parser()
    args = parser.parse_args(["kanban", "block", "task123", "--kind", "needs_input", "multi", "word"])

    assert args.command == "kanban"
    assert args.kanban_action == "block"
    assert args.task_id == "task123"
    assert args.kind == "needs_input"
    assert args.reason == ["multi", "word"]
    assert args.ids is None


def test_kanban_block_without_kind_keeps_reason_optional():
    parser = _build_parser()
    args = parser.parse_args(["kanban", "block", "task123", "fix", "this", "task"])

    assert args.kanban_action == "block"
    assert args.task_id == "task123"
    assert args.reason == ["fix", "this", "task"]
    assert args.kind is None


def test_kanban_block_with_kind_and_bulk_ids_still_parses():
    parser = _build_parser()
    args = parser.parse_args(
        ["kanban", "block", "task123", "--kind", "dependency", "--ids", "task124", "task125"]
    )

    assert args.kanban_action == "block"
    assert args.task_id == "task123"
    assert args.ids == ["task124", "task125"]
    assert args.kind == "dependency"
    assert args.reason == []
