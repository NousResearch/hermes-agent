"""Auto-routed Meta API-key and subscription recovery must not cross accounts."""
from types import SimpleNamespace

import pytest

from agent.auxiliary_client import _auth_refresh_provider_for_route, _recoverable_pool_provider

BASE = "https://api.meta.ai/v1"


@pytest.mark.parametrize("effective,expected", [
    ("meta-ai", "auto"),
    ("meta-oauth", "meta-oauth"),
    ("", "meta-oauth"),
])
def test_auto_refresh_respects_meta_sibling(effective, expected):
    assert _auth_refresh_provider_for_route("auto", BASE, effective) == expected


@pytest.mark.parametrize("effective,expected", [
    ("meta-ai", "meta-ai"),
    ("meta-oauth", "meta-oauth"),
])
def test_auto_pool_recovery_respects_meta_sibling(effective, expected):
    client = SimpleNamespace(base_url=BASE, api_key="fake", _hermes_aux_effective_provider=effective)
    assert _recoverable_pool_provider("auto", client) == expected
