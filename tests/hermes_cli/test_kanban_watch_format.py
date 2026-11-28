"""Readable ``hermes kanban watch`` output (``--format pretty|raw|auto``)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest
from wcwidth import wcswidth

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_ops
from hermes_cli.kanban import kanban_command
from hermes_cli.kanban_output import format_watch_event
from hermes_cli.kanban_parser import build_parser

ANSI = re.compile(r"\033\[[0-9;]*m")


# ---------------------------------------------------------------------------
# format_watch_event (pure)
# ---------------------------------------------------------------------------

def test_completed_event_shows_title_and_summary():
    lines = format_watch_event(
        kind="completed", task_id="t_abcd1234", assignee="alice", title="Write the report",
        payload={"result_len": 0, "summary": "Report written and filed"}, ts=0, width=100,
    )
    assert len(lines) == 2
    assert "✓ DONE" in lines[0]
    assert "alice" in lines[0] and "t_abcd1234" in lines[0] and "Write the report" in lines[0]
    assert lines[1].strip() == "└ Report written and filed"
    assert not ANSI.search("".join(lines))


@pytest.mark.parametrize("kind,payload,expected", [
    ("blocked", {"reason": "need API key", "kind": "needs_input"}, "need API key"),
    ("gave_up", {"failures": 2, "error": "pid 42 not alive"}, "pid 42 not alive"),
    ("crashed", {"pid": 42, "worker_output": "Traceback: boom"}, "Traceback: boom"),
    ("timed_out", {"pid": 42, "elapsed_seconds": 610, "limit_seconds": 600}, "ran 610s, limit 600s"),
])
def test_detail_line_uses_most_meaningful_payload_field(kind, payload, expected):
    lines = format_watch_event(kind=kind, task_id="t_1", assignee="bob", title="x",
                               payload=payload, ts=0)
    assert lines[-1].strip() == f"└ {expected}"


def test_no_detail_line_when_payload_has_nothing_readable():
    lines = format_watch_event(kind="spawned", task_id="t_1", assignee="bob", title="x",
                               payload={"pid": 42}, ts=0)
    assert len(lines) == 1 and "▶ START" in lines[0]


def test_long_title_and_detail_fit_width():
    width = 70
    lines = format_watch_event(
        kind="blocked", task_id="t_abcd1234", assignee="a-very-long-profile-name", title="T" * 200,
        payload={"reason": "word " * 200}, ts=0, width=width,
    )
    assert len(lines) == 3  # headline + detail capped at two lines
    assert all(len(line) <= width for line in lines)
    assert lines[0].endswith("…") and lines[-1].endswith("…")


@pytest.mark.parametrize("width", [48, 70, 100])
@pytest.mark.parametrize("kind,assignee,title,payload", [
    ("completed", "研究员", "調査レポートを書く" * 10, {"summary": "完了しました。" * 40}),
    ("timed_out", "alice", "x" * 300, {"elapsed_seconds": 610, "limit_seconds": 600}),
    ("blocked", "a-very-long-profile-name", "🚀 launch " * 30, {"reason": "🙂 " * 200}),
    ("crashed", "bob", "❤️ " * 200, {"worker_output": "❤️ " * 200}),  # VS16 emoji presentation
    ("gave_up", "👩‍💻", "👨‍👩‍👧 family " * 40, {"error": "👨‍👩‍👧" * 300}),  # ZWJ sequences
])
def test_every_line_fits_terminal_cells(width, kind, assignee, title, payload):
    lines = format_watch_event(kind=kind, task_id="t_abcd1234", assignee=assignee, title=title,
                               payload=payload, ts=0, width=width, use_color=True)
    for line in lines:
        # Independent oracle: wcswidth over the whole rendered line.
        assert wcswidth(ANSI.sub("", line)) <= width, (width, line)


def test_huge_unbroken_detail_is_bounded(monkeypatch):
    from hermes_cli import kanban_output

    calls = {"n": 0}
    real = kanban_output._grapheme_cells

    def counting(g):
        calls["n"] += 1
        return real(g)

    monkeypatch.setattr(kanban_output, "_grapheme_cells", counting)
    lines = format_watch_event(kind="blocked", task_id="t_1", assignee="a", title="調" * 30_000,
                               payload={"reason": "調" * 30_000}, ts=0, width=70)
    assert len(lines) == 3 and lines[-1].endswith("…")
    # Work scales with the visible rows, not the 60k characters of input.
    assert calls["n"] < 1_000


def test_narrow_pane_drops_assignee_column():
    (line,) = format_watch_event(kind="spawned", task_id="t_1", assignee="alice", title="x",
                                 payload=None, ts=0, width=48)
    assert "alice" not in line and "t_1" in line


def test_multiline_title_stays_on_one_headline():
    lines = format_watch_event(kind="completed", task_id="t_1", assignee="a",
                               title="first\nsecond\tthird", payload=None, ts=0)
    assert lines == [lines[0]] and "first second third" in lines[0]


def test_unknown_kind_falls_back_to_label():
    (line,) = format_watch_event(kind="some_new_kind", task_id="t_1", assignee=None, title="x",
                                 payload=None, ts=0)
    assert "SOME_NEW" in line and " - " in line


def test_color_only_when_requested():
    plain = format_watch_event(kind="completed", task_id="t_1", assignee="a", title="x",
                               payload={"summary": "ok"}, ts=0)
    colored = format_watch_event(kind="completed", task_id="t_1", assignee="a", title="x",
                                 payload={"summary": "ok"}, ts=0, use_color=True)
    assert ANSI.search(colored[0])
    assert [ANSI.sub("", line) for line in colored] == plain


# ---------------------------------------------------------------------------
# _cmd_watch end to end
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _run_watch(monkeypatch, capsys, *extra: str) -> str:
    """Run ``hermes kanban watch`` for one tick after a task heartbeats and completes."""
    calls = {"n": 0}

    def fake_sleep(_interval):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt
        with kbc.connect_closing() as conn:
            tid = kb.create_task(conn, title="Ship the widget", assignee="alice")
            with conn:
                kb._append_event(conn, tid, "heartbeat", {"note": "still going"})
            assert kb.complete_task(conn, tid, summary="Widget shipped")

    monkeypatch.setattr(kanban_ops.time, "sleep", fake_sleep)
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers())
    assert kanban_command(parser.parse_args(["kanban", "watch", *extra])) == 0
    return capsys.readouterr().out


def test_watch_pretty_renders_title_and_summary(kanban_home, monkeypatch, capsys):
    out = _run_watch(monkeypatch, capsys, "--format", "pretty")
    assert "✓ DONE" in out and "Ship the widget" in out and "└ Widget shipped" in out
    assert "+ NEW" in out
    assert "heartbeat" not in out.lower()  # quiet kind hidden by default
    assert "{'" not in out  # no raw payload dicts


def test_watch_pretty_shows_quiet_kinds_when_requested(kanban_home, monkeypatch, capsys):
    out = _run_watch(monkeypatch, capsys, "--format", "pretty", "--kinds", "heartbeat,completed")
    assert "HEARTBEA" in out and "✓ DONE" in out and "+ NEW" not in out


@pytest.mark.parametrize("extra", [("--format", "raw"), ()])
def test_watch_raw_and_piped_auto_keep_legacy_lines(kanban_home, monkeypatch, capsys, extra):
    # capsys stdout is not a TTY, so the default ``auto`` format must stay raw.
    out = _run_watch(monkeypatch, capsys, *extra)
    assert re.search(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] t_\w+\s+completed\s+\(@alice\) \{.*'summary': "
                     r"'Widget shipped'", out, re.MULTILINE)
    assert re.search(r"\s+heartbeat\s+\(@alice\)", out)
    assert "✓ DONE" not in out
