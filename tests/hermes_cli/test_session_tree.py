"""``/tree``'s words and its classic-CLI picker (`hermes_cli/session_tree.py`).

The rows are shared with the messaging gateway and `slash.exec`, so the tests pin the wording,
the indentation and the two defaults the lane asks for: compression continuations fold into
their parent and subagent runs are hidden unless the view asks for them.
"""

from __future__ import annotations

import io

import pytest

from hermes_cli import session_tree
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


def test_a_row_reads_title_kind_size_and_when(db) -> None:
    db.set_session_title("branch", "side-main #2")  # a title follows the live tip of a lineage
    node = next(node for node in build_session_family(db, "branch").nodes
                if node["session_id"] == "branch")

    line = session_tree.node_line(node).strip()

    assert line.startswith("side-main #2 · branch · 0 msgs · ")
    assert line.endswith(" · " + session_tree.t("cli.tree.current"))


def test_indentation_is_two_spaces_per_level(db) -> None:
    nodes = build_session_family(db, "sub").nodes
    sub = next(node for node in nodes if node["session_id"] == "sub")

    assert sub["depth"] == 3
    assert session_tree.node_line(sub).startswith("      ")


def test_an_untitled_row_names_its_id_instead_of_inventing_a_title(db) -> None:
    node = next(node for node in build_session_family(db, "branch").nodes
                if node["session_id"] == "cont")

    assert session_tree.node_line(node).strip().startswith("untitled cont · compression")


def test_a_continuation_says_which_message_it_continues(db) -> None:
    node = next(node for node in build_session_family(db, "branch").nodes
                if node["session_id"] == "cont")

    assert "after msg 2" in session_tree.node_line(node)


def test_the_default_view_folds_continuations_and_hides_subagent_runs(db) -> None:
    lines = session_tree.tree_lines(session_tree.load_family(db, "branch"))

    assert len(lines) == 2
    assert lines[0].startswith("untitled root · root")
    assert "+1 compressed" in lines[0]
    assert lines[1].startswith("  untitled branch · branch")


def test_the_all_view_lists_the_folded_rows_too(db) -> None:
    lines = session_tree.tree_lines(session_tree.load_family(db, "branch", show_all=True))

    assert [line.strip().split(" · ")[1] for line in lines] == [
        "root", "compression", "branch", "subagent"]


def test_the_text_answer_carries_a_header_and_the_rows(db) -> None:
    text = session_tree.tree_text(session_tree.load_family(db, "branch"))

    header, *rows = text.splitlines()
    assert header.startswith("Session tree · untitled root · 2 session(s)")
    assert len(rows) == 2


def test_an_empty_family_says_so_rather_than_printing_a_blank_tree(tmp_path) -> None:
    empty = SessionDB(db_path=tmp_path / "empty.db")
    try:
        text = session_tree.tree_text(session_tree.load_family(empty, "nope"))
    finally:
        empty.close()

    assert text == session_tree.empty_line()


def test_the_picker_returns_the_picked_session_id(db, monkeypatch) -> None:
    seen: dict = {}

    def _choose(title, items, default_index=0, *, cancel_label="Cancel", searchable=False):
        seen.update(items=items, default_index=default_index, searchable=searchable)
        return 1  # the second row is the branch

    monkeypatch.setattr(session_tree, "curses_single_select", _choose)

    assert session_tree.pick_session(session_tree.load_family(db, "branch")) == "branch"
    assert seen["searchable"] is True
    assert seen["default_index"] == 1


def test_the_picker_is_skipped_when_there_is_nothing_to_switch_to(db, monkeypatch) -> None:
    def _never(*_args, **_kwargs):
        raise AssertionError("a one-session family must not open a picker")

    monkeypatch.setattr(session_tree, "curses_single_select", _never)
    db.create_session("lonely", source="cli")

    assert session_tree.pick_session(session_tree.load_family(db, "lonely")) is None


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
    monkeypatch.setattr(session_tree, "pick_session", lambda *_a, **_k: "branch")
    cli = _CLI(db, "cont")

    cli._handle_tree_command("/tree")

    assert capsys.readouterr().out == ""
    assert cli.resumed == ["/resume branch"]


def test_the_command_prints_only_the_family_without_a_terminal(db, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO())
    monkeypatch.setattr(session_tree, "pick_session",
                        lambda *_a, **_k: pytest.fail("no picker without a terminal"))
    cli = _CLI(db, "branch")

    cli._handle_tree_command("/tree")

    assert "untitled root · root" in capsys.readouterr().out
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
