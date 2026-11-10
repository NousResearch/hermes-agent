"""Claude Mythos models have the same specs as their Fable twins
(https://platform.claude.com/docs/en/models/mythos-5-2/overview). Without a live lookup they get
the same 1M window, on the Claude API and on Bedrock."""

import pytest

from agent.bedrock_adapter import get_bedrock_context_length
from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, _longest_key_match


@pytest.mark.parametrize("mythos", ["claude-mythos-5-2", "claude-mythos-5-1", "claude-mythos-5", "anthropic/claude-mythos-5.2"])
def test_mythos_offline_window_matches_fable(mythos):
    hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, mythos)
    fable = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, mythos.replace("mythos", "fable"))
    assert hit is not None and fable is not None
    assert hit[1] == fable[1] == 1_000_000


@pytest.mark.parametrize("scope", ["", "us.", "global."])
def test_mythos_bedrock_window_is_1m(scope):
    assert get_bedrock_context_length(f"{scope}anthropic.claude-mythos-5-2", probe=False) == 1_000_000
