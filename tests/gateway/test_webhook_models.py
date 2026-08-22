"""Canonical webhook route model and legacy compatibility tests."""

import pytest
from pydantic import ValidationError

from gateway.platforms.webhook_models import (
    WebhookRouteConfig,
    from_legacy_route,
    to_legacy_route,
)


def test_legacy_translation_preserves_secret_and_delivery_fields():
    legacy = {
        "description": "GitHub lane",
        "events": ["push"],
        "secret": "secret-value",
        "prompt": "Handle {payload}",
        "skills": ["github"],
        "script": "normalize.py",
        "deliver": "telegram",
        "deliver_extra": {"chat_id": "123", "thread_id": "9"},
        "deliver_only": True,
    }
    route = from_legacy_route("github-pr", legacy)
    assert route.name == "github-pr"
    assert route.secret_ref == "secret-value"
    assert route.deliveries == [
        {"target": "telegram", "chat_id": "123", "thread_id": "9"}
    ]
    assert route.model_dump()["deliveries"][0]["chat_id"] == "123"
    round_trip = to_legacy_route(route)
    assert round_trip["secret"] == legacy["secret"]
    assert round_trip["deliver"] == legacy["deliver"]
    assert round_trip["deliver_extra"] == legacy["deliver_extra"]
    assert round_trip["deliver_only"] is True


def test_canonical_serialization_has_typed_route_fields():
    route = WebhookRouteConfig(name="builds", secret_ref="s", events=["push"])
    data = route.model_dump()
    assert data["name"] == "builds"
    assert data["signature_mode"] == "generic_v2"
    assert data["session_mode"] == "event"
    assert data["approval_mode"] == "deny"
    assert data["response_mode"] == "accepted"
    assert "secret_ref" in data and "secret" not in data


@pytest.mark.parametrize("name", ["Bad Name", "../escape", "-leading", "", "UPPER"])
def test_invalid_route_names_are_rejected(name):
    with pytest.raises(ValidationError):
        WebhookRouteConfig(name=name, secret_ref="s")


@pytest.mark.parametrize(
    "values",
    [
        {"deliver_only": True},
        {"approval_mode": "delivery_target"},
        {"deliver_only": True, "response_mode": "callback", "callback": {"url": "https://x"}},
    ],
)
def test_invalid_delivery_combinations_are_rejected(values):
    with pytest.raises(ValidationError):
        WebhookRouteConfig(name="route", secret_ref="s", **values)


@pytest.mark.parametrize("mode", ["github", "gitlab", "svix", "generic_v2", "generic_v1"])
def test_explicit_signature_modes_are_supported(mode):
    route = WebhookRouteConfig(name="route", secret_ref="s", signature_mode=mode)
    assert route.signature_mode == mode


def test_callback_response_requires_valid_callback_and_rejects_stray_callback():
    with pytest.raises(ValidationError):
        WebhookRouteConfig(name="route", secret_ref="s", response_mode="callback")
    with pytest.raises(ValidationError):
        WebhookRouteConfig(
            name="route", secret_ref="s", callback={"url": "https://example.test"}
        )
    route = WebhookRouteConfig(
        name="route",
        secret_ref="s",
        response_mode="callback",
        callback={"url": "https://example.test/callback"},
    )
    assert route.callback["url"].startswith("https://")


def test_session_policies_require_key_template_only_for_keyed_sessions():
    with pytest.raises(ValidationError):
        WebhookRouteConfig(name="route", secret_ref="s", session_mode="keyed")
    with pytest.raises(ValidationError):
        WebhookRouteConfig(
            name="route", secret_ref="s", session_key_template="{id}"
        )
    route = WebhookRouteConfig(
        name="route",
        secret_ref="s",
        session_mode="keyed",
        session_key_template="event:{id}",
    )
    assert route.session_key_template == "event:{id}"


def test_unknown_fields_are_ignored_like_other_pydantic_config_models():
    route = WebhookRouteConfig(name="route", secret_ref="s", future_option=True)
    assert route.model_dump()["future_option"] is True
