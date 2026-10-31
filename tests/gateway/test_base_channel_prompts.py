"""Behavior contracts for additive platform-wide channel prompts."""

import pytest

from gateway.platforms.base_channel_prompts import resolve_channel_prompt


@pytest.mark.parametrize(
    ("prompts", "channel_id", "parent_id", "expected"),
    [
        pytest.param({"*": " Global policy "}, "channel-a", None,
                     "Global policy", id="wildcard-only"),
        pytest.param({"*": "Global policy"}, "channel-b", "parent",
                     "Global policy", id="wildcard-unmatched-thread"),
        pytest.param({"*": "Global policy", "thread": "Exact", "parent": "Parent"},
                     "thread", "parent", "Global policy\n\nExact", id="exact-before-parent"),
        pytest.param({"*": "Global policy", "parent": "Parent"}, "thread", "parent",
                     "Global policy\n\nParent", id="inherited-parent"),
        pytest.param({"*": "Global policy", "thread": "  ", "parent": "Parent"},
                     "thread", "parent", "Global policy\n\nParent", id="blank-exact-falls-back"),
        pytest.param({"*": "Policy", "thread": " Policy ", "parent": "Parent"},
                     "thread", "parent", "Policy", id="duplicate-exact-still-selected"),
        pytest.param({"*": " Policy ", "parent": "Policy"}, "thread", "parent",
                     "Policy", id="duplicate-parent"),
        pytest.param({"thread": "Exact", "parent": "Parent"}, "thread", "parent",
                     "Exact", id="legacy-exact"),
        pytest.param({"parent": "Parent"}, "thread", "parent", "Parent", id="legacy-parent"),
        pytest.param({"thread": 0, "parent": "Parent"}, "thread", "parent",
                     "0", id="legacy-non-string-exact"),
        pytest.param({"parent": "Parent"}, "other", None, None, id="legacy-unmatched"),
        pytest.param({"*": "Policy", "parent": "Parent"}, "*", "parent",
                     "Policy\n\nParent", id="wildcard-is-not-an-exact-tier"),
        *[
            pytest.param({"*": wildcard, "thread": "Exact"}, "thread", None,
                         "Exact", id=f"invalid-wildcard-exact-{index}")
            for index, wildcard in enumerate([None, "", "   ", [], {}, 0, False])
        ],
        *[
            pytest.param({"*": wildcard}, "other", None, None,
                         id=f"invalid-wildcard-unmatched-{index}")
            for index, wildcard in enumerate([None, "", "   ", [], {}, 0, False])
        ],
        *[
            pytest.param(prompts, "thread", "parent", None, id=f"invalid-mapping-{index}")
            for index, prompts in enumerate([None, [], "not a mapping"])
        ],
    ],
)
def test_channel_prompt_composes_wildcard_with_selected_tier(prompts, channel_id, parent_id, expected):
    assert resolve_channel_prompt({"channel_prompts": prompts}, channel_id, parent_id) == expected
