"""Recall candidate ranking: server score + L2 leaf boost, no query-token overlap boost.

The OpenViking server's similarity scores for one query typically sit in a narrow band
(about +/-0.01 in practice). A literal query-token overlap bonus of 0.05/token was larger
than that whole band, so it decided the order: memories that merely repeated the query's
words (often raw chat-log events) outranked the semantically closer fact, and memories in a
different language from the query (no shared tokens at all) were pushed out of the top N.
"""

from plugins.memory.openviking import OpenVikingMemoryProvider


def _item(uri, score, *, abstract="", level=2, category="preferences"):
    return {
        "uri": uri,
        "score": score,
        "abstract": abstract,
        "level": level,
        "category": category,
    }


def _select(items, query, limit=5, score_threshold=0.0):
    selected = OpenVikingMemoryProvider._select_recall_candidates(
        items, query, limit=limit, score_threshold=score_threshold
    )
    return [item["uri"] for item in selected]


def test_literal_overlap_does_not_outrank_a_higher_server_score():
    query = "what hookah tobacco does the user like best"
    items = [
        # English summary of a fact the user stated in another language: no shared tokens.
        _item(
            "viking://user/memories/preferences/tobacco.md",
            0.512,
            abstract="Prefers Darkside Bananapapa, mild blends",
        ),
        # Chat-log event that repeats the query's words but answers nothing.
        _item(
            "viking://user/memories/events/chat_1.md",
            0.505,
            abstract="user asked what hookah tobacco does the user like best",
            category="events",
        ),
    ]

    assert _select(items, query) == [
        "viking://user/memories/preferences/tobacco.md",
        "viking://user/memories/events/chat_1.md",
    ]


def test_ranking_matches_server_order_among_leaves_regardless_of_query_text():
    items = [
        _item("viking://user/memories/a.md", 0.53, abstract="alpha"),
        _item("viking://user/memories/b.md", 0.52, abstract="beta beta beta"),
        _item("viking://user/memories/c.md", 0.51, abstract="gamma"),
    ]

    assert (
        _select(items, "beta beta beta beta")
        == _select(items, "")
        == [
            "viking://user/memories/a.md",
            "viking://user/memories/b.md",
            "viking://user/memories/c.md",
        ]
    )


def test_leaf_boost_threshold_dedupe_and_limit_are_kept():
    items = [
        _item(
            "viking://user/memories/dir", 0.60, abstract="directory overview", level=1
        ),
        _item("viking://user/memories/leaf.md", 0.55, abstract="leaf fact"),
        _item("viking://user/memories/leaf.md", 0.55, abstract="leaf fact"),  # same uri
        _item(
            "viking://user/memories/copy.md", 0.54, abstract="Leaf  fact"
        ),  # same abstract+category
        _item("viking://user/memories/low.md", 0.10, abstract="below threshold"),
        _item("viking://user/memories/other.md", 0.50, abstract="other fact"),
    ]

    # L2 leaf (+0.12) beats the higher-scored L1 directory; duplicates and sub-threshold drop out.
    assert _select(items, "leaf", limit=5, score_threshold=0.2) == [
        "viking://user/memories/leaf.md",
        "viking://user/memories/other.md",
        "viking://user/memories/dir",
    ]
    assert _select(items, "leaf", limit=1, score_threshold=0.2) == [
        "viking://user/memories/leaf.md"
    ]
