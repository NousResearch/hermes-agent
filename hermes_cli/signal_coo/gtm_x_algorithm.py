"""Public X algorithm lens for Magnus GTM content packages."""

from __future__ import annotations

import copy
from typing import Any


X_ALGORITHM_SOURCE_URL = "https://github.com/xai-org/x-algorithm"

X_ALGORITHM_SIGNAL_LENS: dict[str, Any] = {
    "schema_version": 1,
    "source": {
        "name": "xai-org/x-algorithm",
        "url": X_ALGORITHM_SOURCE_URL,
        "type": "public_repository",
        "caveat": "Use as public ranking-system evidence and a content pressure-test, not as a guarantee of private production weights.",
    },
    "ranking_frame": {
        "retrieval": [
            "in-network candidates from followed accounts",
            "out-of-network candidates from ML retrieval over the global corpus",
        ],
        "ranking": "Phoenix/Grok predicts engagement probabilities, then a weighted scorer combines predicted actions into a final score.",
        "filtering": "Pre- and post-scoring filters remove duplicates, unsafe content, muted or blocked contexts, already-seen items, and other ineligible candidates.",
    },
    "positive_signals": [
        "reply",
        "repost",
        "quote",
        "share",
        "click",
        "profile_click",
        "dwell",
        "follow_author",
        "video_view",
        "photo_expand",
    ],
    "negative_signals": [
        "not_interested",
        "block_author",
        "mute_author",
        "report",
    ],
    "pressure_tests": [
        "Does the opener create a concrete reason to reply, not just nod?",
        "Would someone repost or quote this because it gives them a useful stance?",
        "Does the post create enough curiosity for profile clicks without becoming clickbait?",
        "Does the body reward dwell with a real idea, source, or operator move?",
        "Does the framing avoid rage bait, vague AI hype, and low-trust claims that invite negative feedback?",
    ],
    "content_rules": [
        "Open with the operator problem, not the source title.",
        "Make the practical move explicit enough that a reader can use it today.",
        "Use sources to support the argument, not as the argument itself.",
        "Prefer specific disagreement or decision framing over generic advice.",
        "Do not manufacture engagement bait; optimize for useful responses and trust.",
    ],
}


def x_algorithm_signal_lens() -> dict[str, Any]:
    return copy.deepcopy(X_ALGORITHM_SIGNAL_LENS)


def x_algorithm_brief_line() -> str:
    return (
        "X algorithm lens: pressure-test hooks for reply, repost/quote, profile-click, "
        "dwell, and follow intent; avoid not-interested, block, mute, or report signals."
    )
