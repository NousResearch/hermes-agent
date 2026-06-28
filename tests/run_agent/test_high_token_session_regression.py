"""Sanitized regressions for high-token, cache-friendly Hermes sessions.

The fixture intentionally stores aggregate counts only. It models the expensive
pattern from a production session without retaining transcript text, tool
outputs, file paths, prompts, or private user data.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from agent.context_compressor import ContextCompressor
from agent.prompt_caching import apply_anthropic_cache_control


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "observability"
    / "high_token_sessions.json"
)


def _fixture() -> dict:
    data = json.loads(FIXTURE_PATH.read_text())
    return data["sessions"][0]


def _has_cache_marker(message: dict) -> bool:
    if message.get("cache_control"):
        return True
    content = message.get("content")
    if isinstance(content, list):
        return any(
            isinstance(part, dict) and bool(part.get("cache_control"))
            for part in content
        )
    return False


def _synthetic_long_messages(turns: int = 28) -> list[dict]:
    """Build a transcript-shaped fixture with synthetic, non-private content."""
    messages: list[dict] = [
        {
            "role": "system",
            "content": "stable cached system/context prefix " * 512,
        }
    ]
    for idx in range(turns):
        messages.append({"role": "user", "content": f"synthetic request {idx}"})
        messages.append(
            {
                "role": "assistant",
                "content": "synthetic progress reply " + ("detail " * 64),
            }
        )
    return messages


class TestHighTokenSessionFixture:
    def test_fixture_is_aggregate_only_and_cache_friendly(self):
        session = _fixture()

        assert "source_session_id_redacted" in session
        assert "transcript" not in session
        assert "messages" not in session
        assert "tool_outputs" not in session

        cache_read_ratio = session["api_cache_read_tokens"] / session["api_input_tokens"]
        uncached_input_per_call = (
            session["api_input_tokens"] - session["api_cache_read_tokens"]
        ) / session["api_calls"]

        assert cache_read_ratio > 0.95
        assert uncached_input_per_call < 6_000

    def test_cumulative_session_tokens_do_not_drive_compression_threshold(self):
        session = _fixture()
        compressor = ContextCompressor(
            model=session["model"],
            threshold_percent=session["compression_threshold_ratio"],
            quiet_mode=True,
            config_context_length=session["assumed_context_length_tokens"],
        )

        assert compressor.threshold_tokens == 256_000
        assert session["max_api_prompt_tokens"] < compressor.threshold_tokens

        # The production pattern was expensive because many requests reused a
        # large mostly-cached prefix. Compression should key off the current
        # request size, not the cumulative token sum for the whole session.
        assert not compressor.should_compress(session["max_api_prompt_tokens"])
        assert compressor.should_compress(session["api_input_tokens"])

    def test_prompt_cache_markers_preserve_stable_prefix_on_long_history(self):
        session = _fixture()
        messages = _synthetic_long_messages(turns=session["agent_turns"])
        original = copy.deepcopy(messages)

        cached = apply_anthropic_cache_control(messages, cache_ttl="5m")

        assert messages == original, "cache-control insertion must not mutate history"
        assert _has_cache_marker(cached[0]), "stable system prefix must stay cacheable"

        non_system_indexes = [
            idx for idx, message in enumerate(cached) if message.get("role") != "system"
        ]
        marked_non_system = [
            idx for idx in non_system_indexes if _has_cache_marker(cached[idx])
        ]

        assert marked_non_system == non_system_indexes[-3:]
        assert sum(1 for message in cached if _has_cache_marker(message)) == 4
