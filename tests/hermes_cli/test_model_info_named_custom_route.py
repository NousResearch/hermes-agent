"""``/api/model/info`` resolves a named custom route through its endpoint.

Regression: the route handed the raw ``model.provider`` (``custom:<relay>``) to both the
context-length resolver and the models.dev capability lookup. The label named a
``providers.<name>`` config row, not a provider profile, so the relay's live catalog was
never consulted: a relay publishing 131072 reported the hardcoded family default and
empty capabilities, while the gateway turn path (which normalizes to ``custom``) had the
right window. The route may also carry its endpoint only in that entry — ``model`` with no
``base_url`` — which left the lookups with no URL at all.
"""

from __future__ import annotations

import hermes_yaml as yaml

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


def _write_entry_only_config() -> None:
    """The route declares its endpoint ONLY under ``providers.<name>`` — the runtime resolves
    that entry, so the dashboard's own metadata lookups must read the same URL."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "model": {"default": RELAY_MODEL, "provider": f"custom:{RELAY_NAME}"},
            "providers": {RELAY_NAME: {"base_url": BASE_URL, "key_env": "RELAY_PROBE_KEY"}},
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


def test_model_info_reads_a_named_entrys_url_when_the_model_section_omits_it(request):
    """``model`` holding only the ``custom:<name>`` id must still probe the entry's endpoint.

    The runtime resolves the named entry, so the dashboard has to as well: with no URL the
    resolver cannot reach the endpoint's own catalog and the UI reports the hardcoded family
    default (and no capabilities) for a model whose real window it publishes.
    """
    from hermes_cli.web_routers import models as router

    _register_profile(request)
    _write_entry_only_config()

    info = router.get_model_info()

    assert info["provider"] == f"custom:{RELAY_NAME}"
    assert info["auto_context_length"] == RELAY_WINDOW
    assert info["capabilities"]["context_window"] == RELAY_WINDOW
