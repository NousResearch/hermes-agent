"""Tests for the pre_gateway_dispatch result MERGE contract (Task 3).

The merge replaces the old short-circuit (which stopped at the first
rewrite/allow). It must let independent plugins each contribute: the gh-review
session binding AND the topic-rename seed rewrite both apply on one turn.
"""
from gateway.run import _merge_pre_dispatch_results
from gateway.session import SessionBinding


def test_empty_and_none_results():
    for hr in ([], None, [None, "str", 123]):
        m = _merge_pre_dispatch_results(hr)
        assert m["skip"] is False
        assert m["rewrite_text"] is None
        assert m["binding"] is None
        assert m["binding_conflict"] is None


def test_binding_and_rewrite_both_apply():
    """gh-review binding (one hook) + topic-rename seed rewrite (other hook)."""
    results = [
        {"action": "allow", "session_binding": {"namespace": "gh-review", "key": "gh-pr-7735"}},
        {"action": "rewrite", "text": "seed-prefixed user text"},
    ]
    m = _merge_pre_dispatch_results(results)
    assert m["skip"] is False
    assert m["binding"] == SessionBinding(namespace="gh-review", key="gh-pr-7735")
    assert m["rewrite_text"] == "seed-prefixed user text"
    assert m["extra_rewrites"] == 0
    assert m["binding_conflict"] is None


def test_binding_accepts_sessionbinding_instance():
    b = SessionBinding(namespace="gh-review", key="k")
    m = _merge_pre_dispatch_results([{"action": "allow", "session_binding": b}])
    assert m["binding"] == b


def test_skip_is_terminal():
    m = _merge_pre_dispatch_results([
        {"action": "rewrite", "text": "x"},
        {"action": "skip", "reason": "blocked"},
    ])
    assert m["skip"] is True
    assert m["skip_reason"] == "blocked"


def test_first_rewrite_wins_extras_counted():
    m = _merge_pre_dispatch_results([
        {"action": "rewrite", "text": "first"},
        {"action": "rewrite", "text": "second"},
        {"action": "rewrite", "text": "third"},
    ])
    assert m["rewrite_text"] == "first"
    assert m["extra_rewrites"] == 2


def test_conflicting_distinct_bindings_refused():
    m = _merge_pre_dispatch_results([
        {"action": "allow", "session_binding": {"namespace": "gh-review", "key": "k1"}},
        {"action": "allow", "session_binding": {"namespace": "gh-review", "key": "k2"}},
    ])
    assert m["binding"] is None
    assert m["binding_conflict"] == [("gh-review", "k1"), ("gh-review", "k2")]


def test_same_binding_twice_is_not_a_conflict():
    m = _merge_pre_dispatch_results([
        {"action": "allow", "session_binding": {"namespace": "gh-review", "key": "k"}},
        {"action": "rewrite", "text": "t", "session_binding": {"namespace": "gh-review", "key": "k"}},
    ])
    assert m["binding"] == SessionBinding(namespace="gh-review", key="k")
    assert m["binding_conflict"] is None


def test_invalid_binding_dropped_not_fatal():
    # 'telegram' is a reserved namespace -> SessionBinding.from_dict returns None
    m = _merge_pre_dispatch_results([
        {"action": "allow", "session_binding": {"namespace": "telegram", "key": "x"}},
    ])
    assert m["binding"] is None
    assert m["binding_conflict"] is None


def test_allow_is_noop_does_not_suppress():
    m = _merge_pre_dispatch_results([
        {"action": "allow"},
        {"action": "allow", "session_binding": {"namespace": "gh-review", "key": "k"}},
    ])
    assert m["binding"] == SessionBinding(namespace="gh-review", key="k")
