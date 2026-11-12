"""``/api/model/info`` resolves a ``custom:<name>`` route through the endpoint's profile.

Regression: the route handed the raw ``model.provider`` (``custom:<relay>``) to both the
context-length resolver and the models.dev capability lookup. The label named a
``providers.<name>`` config row, not a provider profile, so the relay's live catalog was
never consulted: a relay publishing 131072 reported the hardcoded family default and
empty capabilities, while the gateway turn path (which normalizes to ``custom``) had the
right window.
"""

from __future__ import annotations

import yaml

import providers
from providers.base import ProviderProfile

RELAY_HOST = "model-info-relay.l4-test.test"
RELAY_NAME = "l4-model-info-relay"
RELAY_MODEL = "some-vendor-model"
RELAY_WINDOW = 131_072
BASE_URL = f"http://{RELAY_HOST}:4000/v1"


class _LiveContextProfile(ProviderProfile):
    """A relay whose window only its own ``/models`` knows."""

    def live_context_length(self, model, *, api_key="", base_url=""):
        return RELAY_WINDOW


def _register_profile(request) -> None:
    profile = _LiveContextProfile(name=RELAY_NAME, base_url=BASE_URL, hostnames=(RELAY_HOST,))
    providers.list_providers()
    providers.register_provider(profile)

    def _cleanup():
        providers._REGISTRY.pop(RELAY_NAME, None)
        providers._PROVIDER_LIST_CACHE = None
        providers._HOST_PROFILE_MEMO.clear()

    request.addfinalizer(_cleanup)


def _write_model_config() -> None:
    """The dashboard's own config shape: a named custom route plus catalog data for the
    provider the host claims (never for the raw ``custom:<name>`` id)."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "model": {"default": RELAY_MODEL, "provider": f"custom:{RELAY_NAME}", "base_url": BASE_URL},
            "model_overrides": {RELAY_NAME: {RELAY_MODEL: {"context_window": RELAY_WINDOW, "tool_call": True}}},
        }),
        encoding="utf-8",
    )


def test_model_info_reports_the_relays_window_and_capabilities_for_a_named_route(request):
    from hermes_cli.web_routers import models as router

    _register_profile(request)
    _write_model_config()

    info = router.get_model_info()

    assert info["provider"] == f"custom:{RELAY_NAME}"  # the configured label is reported verbatim
    assert info["auto_context_length"] == RELAY_WINDOW  # ...but resolved through the host's profile
    assert info["capabilities"]["supports_tools"] is True
    assert info["capabilities"]["context_window"] == RELAY_WINDOW
