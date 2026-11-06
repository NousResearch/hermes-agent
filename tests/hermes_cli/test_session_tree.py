"""``/tree``'s words and its classic-CLI picker (`hermes_cli/session_tree.py`).

The rows are shared with the messaging gateway and `slash.exec`, so the tests pin the wording,
the indentation and the two defaults the lane asks for: compression continuations fold into
their parent and subagent runs are hidden unless the view asks for them.
"""

from __future__ import annotations

import io

import pytest

from hermes_cli import session_tree, session_tree_browser
from hermes_cli.cli_commands_tree import CLICommandsTreeMixin
from hermes_state import SessionDB
from hermes_state_family import build_session_family


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session("root", source="cli")
    session_db.append_message("root", "user", "one")
    session_db.append_message("root", "assistant", "two")
    session_db.end_session("root", "compression")
    session_db.create_session("cont", source="cli", parent_session_id="root")
    session_db.append_message("cont", "user", "after compaction")
    session_db.create_session("branch", source="cli", parent_session_id="cont",
                              model_config={"_branched_from": "cont"})
    session_db.create_session("sub", source="tool", parent_session_id="branch",
                              model_config={"_delegate_from": "branch"})
    try:
        yield session_db
    finally:
        session_db.close()


def test_a_row_reads_mark_title_kind_size_and_age(db) -> None:
    db.set_session_title("branch", "side-main #2")  # a title follows the live tip of a lineage
    rows = session_tree.load_family(db, "branch").nodes
    line = session_tree.node_line(next(row for row in rows if row["session_id"] == "branch"))

    assert line.startswith("└─ ☤ side-main #2 · branch · 0 msgs · just now")
    assert line.endswith(" · " + session_tree.t("cli.tree.here"))


def test_the_builder_draws_tree_guides_for_every_level(db) -> None:
    rows = session_tree.load_family(db, "sub", show_all=True).nodes

    assert [row["guide"] for row in rows] == ["", "└─ ", "   └─ ", "      └─ "]
    assert session_tree.node_line(rows[3]).startswith("      └─ ☤ untitled sub · subagent")
    assert session_tree.node_line(rows[2]).startswith("   └─ ◇ untitled branch · branch")


def test_an_untitled_row_names_its_id_instead_of_inventing_a_title(db) -> None:
    node = next(node for node in build_session_family(db, "branch").nodes
                if node["session_id"] == "cont")

    assert session_tree.node_line(node).startswith("≋ untitled cont · compressed")


def test_a_continuation_says_which_message_it_continues(db) -> None:
    node = next(node for node in build_session_family(db, "branch").nodes
                if node["session_id"] == "cont")

    assert "after msg 2" in session_tree.node_line(node)


def test_the_default_view_folds_continuations_and_hides_subagent_runs(db) -> None:
    lines = session_tree.tree_lines(session_tree.load_family(db, "branch"))

    assert len(lines) == 2
    assert lines[0].startswith("◆ untitled root · original")
    assert "+1 compressed" in lines[0]
    assert lines[1].startswith("└─ ☤ untitled branch · branch")


def test_the_all_view_lists_the_folded_rows_too(db) -> None:
    lines = session_tree.tree_lines(session_tree.load_family(db, "branch", show_all=True))

    assert [line.split(" · ")[1] for line in lines] == ["original", "compressed", "branch", "subagent"]


def test_a_row_carries_a_message_weight_bar_scaled_to_the_family(db) -> None:
    assert session_tree.weight_bar(0, 10) == " "
    assert session_tree.weight_bar(10, 10) == "█"
    assert session_tree.weight_bar(1, 10) < session_tree.weight_bar(5, 10)
    lines = session_tree.tree_lines(session_tree.load_family(db, "branch"))
    assert "█ 2 msgs" in lines[0]


def test_the_preview_says_where_a_row_came_from_and_its_last_exchange(db) -> None:
    db.set_session_title("root", "side-main")
    rows = session_tree.load_family(db, "branch", show_all=True).nodes
    cont = next(row for row in rows if row["session_id"] == "cont")
    root = next(row for row in rows if row["session_id"] == "root")

    assert session_tree.preview_lines(cont)[0][1].startswith("compressed of “side-main” · 1 msg")
    assert session_tree.preview_lines(root)[1:] == [("you", "one"), ("hermes", "two")]
    branch = next(row for row in rows if row["session_id"] == "branch")
    assert session_tree.preview_lines(branch)[1] == ("", "no messages yet")


def test_the_preview_never_reads_rewound_messages(db) -> None:
    db.append_message("cont", "user", "kept")
    db._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET active = 0 WHERE session_id = 'cont' AND content = 'kept'"))
    rows = session_tree.load_family(db, "branch", show_all=True).nodes
    cont = next(row for row in rows if row["session_id"] == "cont")

    assert cont["last_prompt"] == "after compaction"


def test_the_text_answer_carries_a_header_and_the_rows(db) -> None:
    text = session_tree.tree_text(session_tree.load_family(db, "branch"))

    header, *rows = text.splitlines()
    assert header == "session tree · untitled root · 2 in view"
    assert len(rows) == 2


def test_an_empty_family_says_so_rather_than_printing_a_blank_tree(tmp_path) -> None:
    empty = SessionDB(db_path=tmp_path / "empty.db")
    try:
        text = session_tree.tree_text(session_tree.load_family(empty, "nope"))
    finally:
        empty.close()

    assert text == session_tree.empty_line()


def test_a_folded_row_says_so_and_drops_its_subtree(db) -> None:
    family = session_tree.load_family(db, "branch", folded=("root",))

    assert [row["session_id"] for row in family.nodes] == ["root"]
    assert family.nodes[0]["folded"] is True
    assert session_tree.t("cli.tree.folded") in session_tree.node_line(family.nodes[0])


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class _CLI(CLICommandsTreeMixin):
    """The smallest object `_handle_tree_command` can run against: a store, a session, a resume seam."""

    def __init__(self, db: SessionDB, session_id: str) -> None:
        self._session_db = db
        self.session_id = session_id
        self.resumed: list[str] = []

    def _handle_resume_command(self, cmd_original: str) -> None:
        self.resumed.append(cmd_original)


def test_the_command_opens_the_picker_and_switches_on_a_pick(db, monkeypatch, capsys) -> None:
    """A terminal gets the picker only: the rows are drawn there, and printing them first would
    race the prompt's queued repaints when curses takes the screen."""
    monkeypatch.setattr("sys.stdin", _TTY())
    monkeypatch.setattr(session_tree_browser, "browse_tree", lambda *_a, **_k: "branch")
    cli = _CLI(db, "cont")

    cli._handle_tree_command("/tree")

    assert capsys.readouterr().out == ""
    assert cli.resumed == ["/resume branch"]


def test_the_command_prints_only_the_family_without_a_terminal(db, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO())
    monkeypatch.setattr(session_tree_browser, "browse_tree",
                        lambda *_a, **_k: pytest.fail("no browser without a terminal"))
    cli = _CLI(db, "branch")

    cli._handle_tree_command("/tree")

    assert "◆ untitled root · original" in capsys.readouterr().out
    assert cli.resumed == []


def test_the_command_refuses_a_midway_turn_instead_of_drawing_over_it(db, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", _TTY())
    cli = _CLI(db, "branch")
    cli._agent_running = True

    cli._handle_tree_command("/tree")

    assert "busy" in capsys.readouterr().out.lower()
    assert cli.resumed == []


@pytest.mark.parametrize("arg, expected", [("", False), ("--all", True), ("ALL", True),
                                           ("--all all", True), ("nonsense", None), ("--all x", None)])
def test_one_parser_reads_the_argument_for_every_surface(arg, expected) -> None:
    assert session_tree.parse_show_all(arg) is expected


def test_the_command_rejects_an_unknown_argument(db, capsys) -> None:
    cli = _CLI(db, "branch")

    cli._handle_tree_command("/tree nonsense")

    assert session_tree.usage_line() in capsys.readouterr().out


def test_the_command_says_so_when_the_store_is_unavailable(capsys) -> None:
    cli = _CLI(None, "branch")  # type: ignore[arg-type]

    cli._handle_tree_command("/tree")

    assert "session history" in capsys.readouterr().out.lower()


def test_the_classic_cli_store_can_answer_the_family_read() -> None:
    """The classic CLI's session store is a THIN CLIENT: only names in
    ``gateway_session_store._READS`` reach its local read-only state.db reader, and anything else
    raises ``GatewayClientError`` ("... is not on the gateway session yet"). The family read is one
    of those names: without it /tree prints nothing at all in the classic CLI, which is exactly
    what the live run caught. A free function poking ``_read_one`` cannot work there.
    """
    from hermes_cli.gateway_session_store import _READS
    from hermes_state import SessionDB

    assert "session_family" in _READS
    assert callable(getattr(SessionDB, "session_family", None))
