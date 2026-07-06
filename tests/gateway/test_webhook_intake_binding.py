"""Tests for WebhookAdapter._resolve_intake_binding (Task 3b / Task 6 ambiguity).

The intake resolver must pick exactly one binding fail-closed so two
deliveries that should share a session do, and an ambiguous resolution never
silently binds to the wrong session.
"""
from gateway.platforms.webhook import WebhookAdapter
from gateway.session import SessionBinding


def test_no_results_returns_none():
    assert WebhookAdapter._resolve_intake_binding([]) is None
    assert WebhookAdapter._resolve_intake_binding(None) is None
    assert WebhookAdapter._resolve_intake_binding([None, None]) is None


def test_single_binding_dict():
    out = WebhookAdapter._resolve_intake_binding(
        [{"namespace": "gh-review", "key": "gh-pr-7735"}]
    )
    assert out == SessionBinding(namespace="gh-review", key="gh-pr-7735")


def test_single_binding_instance():
    b = SessionBinding(namespace="gh-review", key="k")
    assert WebhookAdapter._resolve_intake_binding([b]) == b


def test_result_dict_carrying_session_binding():
    out = WebhookAdapter._resolve_intake_binding(
        [{"action": "allow", "session_binding": {"namespace": "gh-review", "key": "k"}}]
    )
    assert out == SessionBinding(namespace="gh-review", key="k")


def test_same_binding_from_two_hooks_is_not_conflict():
    out = WebhookAdapter._resolve_intake_binding([
        {"namespace": "gh-review", "key": "k"},
        {"namespace": "gh-review", "key": "k"},
    ])
    assert out == SessionBinding(namespace="gh-review", key="k")


def test_distinct_bindings_refused_fail_closed():
    out = WebhookAdapter._resolve_intake_binding([
        {"namespace": "gh-review", "key": "k1"},
        {"namespace": "gh-review", "key": "k2"},
    ])
    assert out is None


def test_invalid_binding_ignored():
    # reserved namespace -> from_dict returns None -> ignored
    out = WebhookAdapter._resolve_intake_binding([
        {"namespace": "telegram", "key": "x"},
    ])
    assert out is None
