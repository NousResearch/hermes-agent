"""Worker-side image enrichment for kanban tasks.

When a kanban task body contains a local image path or an ``http(s)://``
image URL, the worker must surface that image to the model on its first
user turn — matching the CLI/gateway behaviour for inbound images.

The dispatcher spawns the worker as
``hermes -p <profile> chat -q "work kanban task <id>"``. The task body
itself never appears in argv; the worker has to read it from the kanban
DB during startup. These tests cover the round-trip:

  task body  →  kanban_db.get_task  →  extract_image_refs  →
  build_native_content_parts  →  multimodal user turn
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from agent.image_routing import (
    build_native_content_parts,
    extract_image_refs,
)


# Tiny 1×1 transparent PNG used to back any path the tests stick into a
# task body. extract_image_refs validates the path exists on disk, so the
# byte content has to be a real readable file (any image bytes will do).
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="
)


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch):
    """Isolated HERMES_HOME with a fresh kanban DB for each test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _add_task_with_body(body: str, *, title: str = "Look at this") -> str:
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title=title,
            body=body,
            assignee="worker-a",
            tenant=None,
        )
    finally:
        conn.close()
    return task_id


def _read_body(task_id: str) -> str:
    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        return (task.body if task is not None else "") or ""
    finally:
        conn.close()


class TestExtractFromTaskBody:
    """Read a real kanban task body and run it through extract_image_refs."""

    def test_local_path_in_body_round_trips(self, kanban_home, tmp_path):
        img = tmp_path / "screenshot.png"
        img.write_bytes(_PNG)
        tid = _add_task_with_body(
            f"Please review the screenshot at {img} and confirm "
            "the alignment is right."
        )

        body = _read_body(tid)
        paths, urls = extract_image_refs(body)
        assert paths == [str(img)]
        assert urls == []


class TestBuildPartsFromTaskBody:
    """Verify the full pipeline produces a multimodal user turn."""

    def test_local_path_becomes_native_image_part(self, kanban_home, tmp_path):
        img = tmp_path / "design.png"
        img.write_bytes(_PNG)
        tid = _add_task_with_body(f"Check out {img} — what's broken?")
        body = _read_body(tid)
        paths, urls = extract_image_refs(body)

        # Mirrors the cli.py wiring: pass the worker's literal -q argument
        # (the dispatcher uses ``"work kanban task <id>"``) plus the
        # extracted refs through build_native_content_parts.
        parts, skipped = build_native_content_parts(
            f"work kanban task {tid}",
            paths,
            image_urls=urls or None,
        )

        assert skipped == []
        # text part + one image_url part
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[0]["text"].startswith(f"work kanban task {tid}")
        assert f"[Image attached at: {img}]" in parts[0]["text"]
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")




    def test_code_block_example_is_not_attached(self, kanban_home, tmp_path):
        # Only the real image outside the fenced code block should attach.
        real = tmp_path / "real.png"
        real.write_bytes(_PNG)
        tid = _add_task_with_body(
            f"Real screenshot:\n{real}\n\n"
            "Example we DON'T want attached:\n"
            "```\n"
            "image: /tmp/example_only.png\n"
            "url: https://example.com/example.png\n"
            "```\n"
        )
        body = _read_body(tid)
        paths, urls = extract_image_refs(body)

        assert paths == [str(real)]
        assert urls == []
