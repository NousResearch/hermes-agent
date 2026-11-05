"""Tests for the external-editor last-reply block (``display.editor_include_last_reply``).

Runs the shared vectors in ``tests/fixtures/editor_context_vectors.json`` — the same file the Ink
composer's suite reads — so the two interfaces cannot drift on which lines count as the block.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.editor_context import (
    MARKER_END,
    MARKER_START,
    build_editor_seed,
    build_last_reply_block,
    config_enabled,
    editor_seed,
    strip_last_reply_block,
)

VECTORS = json.loads(
    (Path(__file__).resolve().parent.parent / "fixtures" / "editor_context_vectors.json").read_text("utf-8")
)


@pytest.mark.parametrize("case", VECTORS["seed"], ids=lambda c: (c["draft"] or "<empty>")[:24])
def test_seed_matches_shared_vectors(case: dict) -> None:
    assert build_editor_seed(case["draft"], case["last_reply"]) == case["expected"]


@pytest.mark.parametrize("case", VECTORS["strip"], ids=lambda c: c["expected"][:24])
def test_strip_matches_shared_vectors(case: dict) -> None:
    assert strip_last_reply_block(case["text"]) == case["expected"]


def test_round_trip_through_an_editor_that_changes_nothing() -> None:
    seed = build_editor_seed("keep this", "a reply")

    assert strip_last_reply_block(seed) == "keep this"


def test_only_the_marked_block_is_removed() -> None:
    text = "\n".join(
        [
            "## My heading",
            "# my own comment",
            "",
            MARKER_START,
            "# agent text",
            "",
            MARKER_END,
            "# another of my comments",
        ]
    )

    assert strip_last_reply_block(text) == "## My heading\n# my own comment\n\n# another of my comments"


def test_reply_text_cannot_close_the_block_early() -> None:
    block = build_last_reply_block(f"before\n{MARKER_END}\nafter")

    # The fake marker is neutralised, so the block still has exactly one real end marker.
    assert block.count(MARKER_END) == 1
    assert strip_last_reply_block("draft\n\n" + block) == "draft"


def test_blank_reply_produces_no_block() -> None:
    assert build_last_reply_block("") == ""
    assert build_last_reply_block("   \n\n") == ""
    assert build_editor_seed("draft", "   ").endswith("draft")


def test_disabled_setting_seeds_the_draft_alone() -> None:
    assert editor_seed("draft", "a reply", enabled=False) == "draft"
    assert editor_seed("draft", "a reply", enabled=True).startswith("draft\n\n" + MARKER_START)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, True), (False, False), ("true", True), ("YES", True), ("0", False), (None, False), (1, True)],
)
def test_config_enabled_reads_yaml_booleans(raw: object, expected: bool) -> None:
    assert config_enabled(raw) is expected
