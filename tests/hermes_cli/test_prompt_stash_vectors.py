"""Cross-surface parity for the Ctrl+S stash: the classic CLI against the Ink composer.

`tests/fixtures/prompt_stash_vectors.json` is shared with the TUI's vitest suite
(`ui-tui/src/__tests__/promptStash.test.ts`). Both implementations must agree on the
decision table, the preview shaping, the placeholder hints and the stack cap, so a change
on one side fails the other side's suite instead of drifting silently.

Nothing here changes behaviour: these are the vectors the ported TypeScript state machine
is held to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.prompt_stash import (
    MAX_STASH_ITEMS,
    PromptStash,
    build_preview,
    resolve_ctrl_s,
)

VECTORS = json.loads((Path(__file__).resolve().parent.parent / "fixtures" / "prompt_stash_vectors.json").read_text("utf-8"))


def _seed(entries: list[str], *, panel_open: bool = False) -> PromptStash:
    """A stash holding `entries`, pushed oldest-first so `entries[-1]` is newest."""
    stash = PromptStash()
    for text in entries:
        assert stash.stash(text) is True
    if panel_open:
        assert stash.open_panel() is True
    return stash


def test_vectors_file_covers_every_family() -> None:
    assert VECTORS["build_preview"] and VECTORS["resolve"] and VECTORS["placeholder_hint"]


@pytest.mark.parametrize("case", VECTORS["build_preview"], ids=lambda c: repr(c["text"])[:40])
def test_build_preview_matches_shared_vectors(case: dict) -> None:
    width = case.get("width", 60)
    assert build_preview(case["text"], width) == case["expected"]


@pytest.mark.parametrize("case", VECTORS["resolve"], ids=lambda c: c["action"])
def test_resolve_ctrl_s_matches_shared_vectors(case: dict) -> None:
    stash = _seed(case["entries"], panel_open=case["panel_open"])
    images = [f"img-{i}.png" for i in range(case["tokens"])]

    action, payload = resolve_ctrl_s(stash, case["buffer"], images)

    assert action == case["action"]
    if action == "restored":
        assert payload == (case["entries"][0], [])
    elif action in {"stashed", "noop"}:
        assert payload is None


@pytest.mark.parametrize("case", VECTORS["placeholder_hint"], ids=lambda c: c["expected"][:24])
def test_placeholder_hint_matches_shared_vectors(case: dict) -> None:
    assert _seed(case["entries"]).placeholder_hint() == case["expected"]


def test_stack_cap_matches_shared_vectors() -> None:
    case = VECTORS["cap"]
    stash = PromptStash()

    for i in range(1, case["push"] + 1):
        assert stash.stash(f"draft {i}") is True

    assert len(stash) == case["expected_len"] == MAX_STASH_ITEMS
    newest = stash.peek(0)
    assert newest is not None
    assert newest.text == case["expected_newest"]
