"""File-backed comments preserve multiline text through the CLI (#137599)."""
import argparse
import io
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def task_id(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title="file comment", body="initial")


def comment(task_id, *argv):
    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers())
    args = parser.parse_args(["kanban", "comment", task_id, *argv])
    return kc.kanban_command(args)


def comments(task_id):
    with kbc.connect_closing() as conn:
        return kb.list_comments(conn, task_id)


def test_comment_file_preserves_text(task_id, tmp_path, monkeypatch):
    body = "  `literal` $variable\n--json\nUnicode: 中文\n" + "long comment " * 10
    source = tmp_path / "comment.md"
    source.write_text(body, encoding="utf-8-sig")
    assert comment(task_id, "--body-file", str(source), "--author", "reviewer") == 0
    # The shared DB writer strips surrounding whitespace for every comment source.
    assert (comments(task_id)[-1].body, comments(task_id)[-1].author) == (body.strip(), "reviewer")
    monkeypatch.setattr("sys.stdin", io.StringIO(body))
    assert comment(task_id, "--body-file", "-") == 0
    assert comments(task_id)[-1].body == body.strip()
    assert comment(task_id, "--author", "legacy", "old", "text") == 0
    assert comments(task_id)[-1].body == "old text"
    assert comment(task_id, "--body-file", str(source), "--max-len", "45") == 0
    assert comments(task_id)[-1].body.endswith("[trimmed to 45 chars by --max-len]")


def test_comment_file_refuses_invalid_input_without_writes(task_id, tmp_path, monkeypatch):
    source = tmp_path / "body.md"
    source.write_text("content", encoding="utf-8")
    assert comment(task_id, "inline", "--body-file", str(source)) == 2
    assert comment(task_id, "--body-file", str(tmp_path / "missing")) == 2
    assert comment(task_id) == 2
    assert comment(task_id, "--body-file", str(source), "--max-len", "0") == 2
    source.write_bytes(b"\xff")
    assert comment(task_id, "--body-file", str(source)) == 2
    source.write_text(" \n", encoding="utf-8")
    assert comment(task_id, "--body-file", str(source)) == 2
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert comment(task_id, "--body-file", "-") == 2
    class BrokenStdin(io.StringIO):
        def read(self, *args):
            raise OSError("unreadable stream")
    monkeypatch.setattr("sys.stdin", BrokenStdin())
    assert comment(task_id, "--body-file", "-") == 2
    assert "local CLI only" in kc.run_slash(f'comment {task_id} --body-file "{source}"')
    assert "local CLI only" in kc.run_slash(f"comment {task_id} --body-file -")
    assert comments(task_id) == []
    assert "Comment added" in kc.run_slash(f'comment {task_id} "inline still works"')
    assert comments(task_id)[-1].body == "inline still works"


def test_comment_delimiter_keeps_flags_literal(task_id):
    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers())
    for prefix in (["--", task_id], [task_id, "--"]):
        args = parser.parse_args(["kanban", "comment", *prefix, "--body-file", "-", "--author", "literal"])
        assert args.body_file is None
        assert args.author is None
        assert kc.kanban_command(args) == 0
        assert comments(task_id)[-1].body == "--body-file - --author literal"
    args = parser.parse_args(["kanban", "comment", task_id, "--author", "reviewer", "--", "--body-file", "-"])
    assert args.body_file is None
    assert args.author == "reviewer"
    assert kc.kanban_command(args) == 0
    assert comments(task_id)[-1].body == "--body-file -"
    assert comment(task_id, "--author", "reviewer", "hello", "--", "--body-file", "-") == 0
    assert comments(task_id)[-1].body == "hello --body-file -"
    assert comment(task_id, "hello", "--author", "reviewer", "world", "--", "--", "--body-file") == 0
    assert comments(task_id)[-1].body == "hello world -- --body-file"


def test_file_input_boundaries_match_inline(task_id, tmp_path, monkeypatch):
    source = tmp_path / "padded.md"
    source.write_text(" " * 100 + "short comment", encoding="utf-8")
    assert comment(task_id, "--body-file", str(source), "--max-len", "45") == 0
    assert comments(task_id)[-1].body == "short comment"
    monkeypatch.setattr("sys.stdin", io.StringIO(" " * 100 + "short comment"))
    assert comment(task_id, "--body-file", "-", "--max-len", "45") == 0
    assert comments(task_id)[-1].body == "short comment"
    assert comment(task_id, "--body-file", str(source), "--") == 0
    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers())
    args = parser.parse_args(["kanban", "comment", "--body-file", str(source), "--", task_id])
    assert kc.kanban_command(args) == 0
    assert comments(task_id)[-1].body == "short comment"
    before = len(comments(task_id))
    for abbreviation in ("--body", "--b"):
        with pytest.raises(SystemExit):
            comment(task_id, abbreviation, str(source))
    assert len(comments(task_id)) == before
