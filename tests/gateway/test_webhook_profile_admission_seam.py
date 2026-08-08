"""Seam tests for the webhook profile-admission mixin extraction."""

import typing

from aiohttp import web

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.webhook import WebhookAdapter, _PROFILE_REJECTED as legacy_rejected
from gateway.platforms.webhook_profile_admission import (
    WebhookProfileAdmissionMixin,
    _PROFILE_REJECTED as admission_rejected,
)


def test_webhook_composes_profile_admission_mixin_without_wrappers():
    assert WebhookAdapter.__mro__[:3] == (
        WebhookAdapter,
        WebhookProfileAdmissionMixin,
        BasePlatformAdapter,
    )
    assert (
        WebhookAdapter._resolve_request_profile
        is WebhookProfileAdmissionMixin._resolve_request_profile
    )
    assert (
        WebhookAdapter._route_allows_profile
        is WebhookProfileAdmissionMixin._route_allows_profile
    )


def test_profile_admission_request_annotation_resolves_at_runtime():
    assert typing.get_type_hints(WebhookAdapter._resolve_request_profile)["request"] is web.Request


def test_legacy_profile_rejection_sentinel_is_the_canonical_object():
    assert legacy_rejected is admission_rejected
