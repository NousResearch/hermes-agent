"""Claude Haiku 5.5 has a 1M window, unlike every earlier Haiku (200K).

The live lookups already report 1M, but the static fallbacks did not list it: the
native route fell to the 200K ``claude`` catch-all and Bedrock to its 128K
unknown-model default, so compaction fired at a fifth (or an eighth) of the window.
https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.bedrock_adapter import get_bedrock_context_length
from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, _longest_key_match


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-haiku-5.5"])
def test_static_context_is_one_million_not_the_claude_catch_all(model):
    assert _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model)[1] == 1_000_000


def test_older_haiku_keeps_the_200k_catch_all():
    assert _longest_key_match(DEFAULT_CONTEXT_LENGTHS, "claude-haiku-4-5")[1] == 200_000


@pytest.mark.parametrize("model_id", [
    "anthropic.claude-haiku-5-5",
    "global.anthropic.claude-haiku-5-5",
    "us.anthropic.claude-haiku-5-5",
])
def test_bedrock_static_context_is_one_million_not_the_128k_default(model_id):
    with patch("agent.bedrock_adapter.probe_bedrock_context_length", return_value=None):
        assert get_bedrock_context_length(model_id, probe=False) == 1_000_000


def test_bedrock_older_haiku_stays_at_200k():
    assert get_bedrock_context_length("anthropic.claude-haiku-4-5-20251001-v1:0", probe=False) == 200_000
