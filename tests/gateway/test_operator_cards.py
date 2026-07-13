"""Behavior contract for platform-agnostic operator cards."""

import pytest

from gateway.operator_cards import (
    CARD_TYPES,
    SEVERITIES,
    OperatorCard,
    OperatorCardValidationError,
    render_operator_card_text,
)


def _payload(**overrides):
    payload = {
        "kind": "operator_card",
        "version": 1,
        "card_type": "approval",
        "title": "Contract redlines",
        "severity": "needs_review",
        "summary": "Legal language changed in the indemnity section.",
        "fields": [
            {"label": "Impact", "value": "The liability boundary changed."},
            {"label": "Next", "value": "Approve or ask for revision."},
        ],
        "actions": [
            {"id": "approve", "label": "Approve", "style": "success"},
            {"id": "revise", "label": "Revise", "style": "secondary"},
            {"id": "escalate", "label": "Escalate", "style": "danger"},
        ],
        "links": [{"label": "Linear", "url": "https://linear.app/example"}],
        "state_ref": "opaque-state-ref",
    }
    payload.update(overrides)
    return payload


def test_round_trip_preserves_the_contract_without_platform_fields():
    card = OperatorCard.from_mapping(_payload())

    assert card.to_mapping() == _payload()
    assert "discord" not in repr(card).lower()


@pytest.mark.parametrize("card_type", sorted(CARD_TYPES))
def test_every_card_type_uses_the_same_plaintext_contract(card_type):
    rendered = render_operator_card_text(OperatorCard.from_mapping(_payload(card_type=card_type)))

    assert "Contract redlines" in rendered
    assert "Legal language changed" in rendered
    assert "Impact: The liability boundary changed." in rendered
    assert "Actions: Approve · Revise · Escalate" in rendered
    assert "[Linear](https://linear.app/example)" in rendered


@pytest.mark.parametrize(
    ("severity", "label"),
    [
        ("done", "Done"),
        ("info", "Info"),
        ("needs_review", "Needs review"),
        ("blocked", "Blocked"),
        ("critical", "Critical"),
    ],
)
def test_every_severity_has_a_mobile_readable_label(severity, label):
    rendered = render_operator_card_text(OperatorCard.from_mapping(_payload(severity=severity)))

    assert label in rendered.splitlines()[0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"kind": "message"}, "kind"),
        ({"version": 2}, "version"),
        ({"card_type": "unknown"}, "card_type"),
        ({"severity": "warning"}, "severity"),
        ({"title": ""}, "title"),
        ({"summary": ""}, "summary"),
        ({"state_ref": ""}, "state_ref"),
        ({"actions": [{"id": "approve", "label": "Approve", "style": "neon"}]}, "style"),
        ({"links": [{"label": "Bad", "url": "javascript:alert(1)"}]}, "url"),
    ],
)
def test_invalid_values_fail_closed(change, message):
    with pytest.raises(OperatorCardValidationError, match=message):
        OperatorCard.from_mapping(_payload(**change))


@pytest.mark.parametrize("missing", ["kind", "version", "card_type", "title", "severity", "summary", "state_ref"])
def test_missing_required_fields_fail_closed(missing):
    payload = _payload()
    payload.pop(missing)

    with pytest.raises(OperatorCardValidationError, match=missing):
        OperatorCard.from_mapping(payload)


def test_unknown_top_level_fields_cannot_leak_to_downstream_renderers():
    payload = _payload(provider_payload={"private": "must-not-pass"})

    with pytest.raises(OperatorCardValidationError, match="unknown fields"):
        OperatorCard.from_mapping(payload)


def test_plaintext_renderer_is_bounded_without_splitting_the_status_header():
    card = OperatorCard.from_mapping(
        _payload(summary="x" * 500, fields=[{"label": "Impact", "value": "y" * 1000}])
    )

    rendered = render_operator_card_text(card, max_length=240)

    assert len(rendered) <= 240
    assert rendered.startswith("🟡 **Needs review — Contract redlines**")
    assert rendered.endswith("…")


def test_constant_sets_match_the_values_accepted_by_the_parser():
    assert CARD_TYPES == {
        "approval",
        "task_run",
        "digest",
        "deal",
        "capture",
        "ops_alert",
        "thread_header",
    }
    assert SEVERITIES == {"done", "info", "needs_review", "blocked", "critical"}
