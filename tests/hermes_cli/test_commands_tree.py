"""``hermes commands`` renders the whole subcommand tree in one call (port of block/buzz#7584)."""

import argparse
import json

from hermes_cli.commands_tree import render_command_tree


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hermes")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("chat", help="Interactive chat")
    kb = sub.add_parser("kanban", help="Collaboration board")
    kb_sub = kb.add_subparsers(dest="kanban_command")
    boards = kb_sub.add_parser("boards", aliases=["b"], help="Manage boards")
    boards.add_subparsers(dest="boards_command").add_parser("list", help="List boards")
    sub.add_parser("login")  # registered without help = hidden, like the deprecated real one
    return p


def test_text_tree_nests_every_level_and_omits_hidden_nodes():
    out = render_command_tree(_make_parser())

    assert "3 groups" not in out and "2 groups, 4 commands." in out
    assert "\nkanban" in out and "\n  boards" in out and "\n    list" in out
    assert "login" not in out and "\n  b " not in out  # hidden node; alias not duplicated


def test_json_tree_matches_text_tree_and_is_machine_readable():
    parser = _make_parser()
    data = json.loads(render_command_tree(parser, as_json=True))

    assert data["prog"] == "hermes"
    assert [n["name"] for n in data["subcommands"]] == ["chat", "kanban"]
    boards = data["subcommands"][1]["subcommands"][0]
    assert boards["help"] == "Manage boards"
    assert boards["subcommands"][0]["name"] == "list"
