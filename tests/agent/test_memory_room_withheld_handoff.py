"""Room filtering and a pre-LLM policy veto both hold at memory-provider hand-offs."""

from copy import deepcopy

import pytest

from agent.message_metadata import PRE_LLM_BLOCK_WITHHELD_PROMPT, WITHHELD_PROMPT
from tests.agent.test_memory_group_orchestration import _HUMAN, _TRANSCRIPT_FANOUTS

pytest_plugins = ["tests.agent.test_memory_group_orchestration"]


@pytest.mark.parametrize("fanout", sorted(_TRANSCRIPT_FANOUTS))
def test_room_and_withheld_input_share_a_safe_snapshot_without_mutating_history(manager, fanout, room_prompt):
    manager, provider = manager
    provider.pre_compress_checkpoint_api_version = 2 if fanout == "pre_compress_v2" else 1
    typed = "Keep the embargoed launch date private until next Thursday."
    before = [{"role": "user", "content": "Use the release branch."},
              {"role": "assistant", "content": "I have checked it out."}]
    blocked = [{"role": "user", "content": PRE_LLM_BLOCK_WITHHELD_PROMPT,
                "display_metadata": {WITHHELD_PROMPT: typed, "policy": "launch-embargo"}},
               {"role": "assistant", "content": "Blocked by the launch policy."}]
    after = [{"role": "user", "content": _HUMAN},
             {"role": "assistant", "content": "I will include them."}]
    rows = [*before,
            {"role": "user", "content": room_prompt},
            {"role": "assistant", "content": "room-only reply"},
            *blocked, *after]
    original = deepcopy(rows)

    _TRANSCRIPT_FANOUTS[fanout](manager, rows)
    assert manager.flush_pending(timeout=5)

    assert len(provider.seen) == 1
    observed = provider.seen[0][2] if fanout == "sync" else provider.seen[0]
    safe_blocked = [{"role": "user", "content": PRE_LLM_BLOCK_WITHHELD_PROMPT,
                     "display_metadata": {"policy": "launch-embargo"}}, blocked[1]]
    assert observed == before + safe_blocked + after
    assert typed not in repr(provider.seen)
    assert all(WITHHELD_PROMPT not in row.get("display_metadata", {}) for row in observed)
    assert rows == original
