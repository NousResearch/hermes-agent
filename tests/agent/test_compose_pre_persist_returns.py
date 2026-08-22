"""Unit tests for ``_compose_pre_persist_returns`` — the pure composer behind the
``pre_persist_user_message`` hook. No host state needed."""

from agent.turn_context import _compose_pre_persist_returns as compose


def test_context_and_bare_str_both_append_at_tail():
    out = compose("hi", [{"context": "a"}, "b"])
    assert out == "hi\n\na\n\nb"


def test_append_order_follows_result_order():
    out = compose("base", ["first", {"context": "second"}, "third"])
    assert out == "base\n\nfirst\n\nsecond\n\nthird"


def test_user_message_replaces_body():
    out = compose("original", [{"user_message": "brand new"}])
    assert out == "brand new"


def test_highest_priority_replace_wins():
    out = compose(
        "x",
        [
            {"user_message": "low", "data": {"priority": 1}},
            {"user_message": "high", "data": {"priority": 9}},
        ],
    )
    assert out == "high"


def test_replace_ties_keep_first():
    out = compose(
        "x",
        [
            {"user_message": "first", "data": {"priority": 5}},
            {"user_message": "second", "data": {"priority": 5}},
        ],
    )
    assert out == "first"


def test_multiple_replaces_logs_warning(caplog):
    with caplog.at_level("WARNING"):
        compose(
            "x",
            [
                {"user_message": "a", "data": {"priority": 1}},
                {"user_message": "b", "data": {"priority": 2}},
            ],
        )
    assert any("user_message" in r.message for r in caplog.records)


def test_replace_sets_body_then_appends_still_apply():
    # A replace picks the body (among replaces); context/str appends still stack
    # at the tail on top of the winning body — they are not discarded.
    out = compose(
        "orig",
        [{"context": "tail note"}, {"user_message": "owned body"}],
    )
    assert out == "owned body\n\ntail note"


def test_none_and_empty_and_non_dict_ignored():
    assert compose("keep", [None]) == "keep"
    assert compose("keep", [{}]) == "keep"
    assert compose("keep", [{"context": ""}]) == "keep"
    assert compose("keep", [123]) == "keep"


def test_no_returns_is_identity():
    assert compose("unchanged", []) == "unchanged"
