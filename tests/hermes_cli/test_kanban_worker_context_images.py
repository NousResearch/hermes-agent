"""Comment and attachment images in the worker context: a worker can't see an inline
markdown image, so each one is surfaced as a path plus the tool that can view it."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    c = kbc.connect()
    yield c
    c.close()


def test_truncated_comment_keeps_image_line(conn):
    tid = kb.create_task(conn, title="look at the screenshot")
    # Image ref sits past the per-comment cap, so the capped body loses it.
    body = "x" * 3000 + " ![shot.png](/abs/att/shot.png)"
    kb.add_comment(conn, tid, author="desktop", body=body)

    ctx = kb.build_worker_context(conn, tid)

    assert "](/abs/att/shot.png)" not in ctx
    assert "Image attached: `/abs/att/shot.png` — view it with vision_analyze (Claude Code: Read the path)" in ctx


def test_angle_bracket_image_path_is_unescaped(conn):
    tid = kb.create_task(conn, title="t")
    kb.add_comment(conn, tid, author="desktop", body=r"![a \[b\]](<C:\\att\\shot (1).png>)")

    ctx = kb.build_worker_context(conn, tid)

    assert r"Image attached: `C:\att\shot (1).png`" in ctx


def test_image_attachment_points_at_vision_analyze(conn, tmp_path):
    tid = kb.create_task(conn, title="t")
    for name, ctype in (("shot.png", None), ("notes.txt", "text/plain")):
        kb.add_attachment(conn, tid, filename=name, stored_path=str(tmp_path / name), content_type=ctype, size=4)

    ctx = kb.build_worker_context(conn, tid)

    png_line = next(line for line in ctx.splitlines() if "`shot.png`" in line)
    txt_line = next(line for line in ctx.splitlines() if "`notes.txt`" in line)
    assert "vision_analyze" in png_line
    assert "vision_analyze" not in txt_line
