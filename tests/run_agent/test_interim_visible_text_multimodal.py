"""Regression: _interim_assistant_visible_text must not crash on list content.

On the first turn of a conversation the "previous message" used for interim
dedup can be a *user* message whose content is a list of multimodal parts
(text + image_url).  Passing that list into the think-block regex raised
``TypeError: expected string or bytes-like object, got 'list'`` inside the
outer conversation loop, which then retried the same API call indefinitely,
streaming duplicated progress lines until the user cancelled.
"""
from __future__ import annotations

import run_agent


class _Host:
    """Minimal host exposing only what _interim_assistant_visible_text needs."""

    show_commentary = True

    def _extract_codex_interim_visible_text(self, assistant_msg):
        return run_agent.AIAgent._extract_codex_interim_visible_text(
            self, assistant_msg
        )

    def _extract_codex_interim_visible_parts(self, assistant_msg):
        return run_agent.AIAgent._extract_codex_interim_visible_parts(
            self, assistant_msg
        )

    def _strip_think_blocks(self, content):
        return run_agent.AIAgent._strip_think_blocks(self, content)

    def _interim_assistant_visible_text(self, assistant_msg):
        return run_agent.AIAgent._interim_assistant_visible_text(
            self, assistant_msg
        )


def test_multimodal_list_content_does_not_raise():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "change the openrouter app url"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
        ],
    }
    assert _Host()._interim_assistant_visible_text(msg) == (
        "change the openrouter app url"
    )


def test_plain_string_content_unchanged():
    msg = {"role": "assistant", "content": "<think>hidden</think>visible"}
    assert _Host()._interim_assistant_visible_text(msg) == "visible"


def test_none_content_returns_empty():
    assert _Host()._interim_assistant_visible_text({"content": None}) == ""
