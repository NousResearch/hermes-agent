"""Regression tests for issue #55112.

Before the fix, when ``auxiliary.vision.provider`` resolved to ``zai`` with no
explicit vision ``base_url``, ``resolve_vision_provider_client`` tried only
metered pay-as-you-go endpoints:

  - https://open.bigmodel.cn/api/paas/v4
  - https://api.z.ai/api/paas/v4

Z.AI coding-plan subscribers were therefore silently billed PAYG cash for every
vision request, because the coding-plan endpoint that consumes subscription
quota (``https://api.z.ai/api/coding/paas/v4``) was absent from the fallback
list — even though main + delegation traffic correctly ran on the subscription
endpoint.

The fix prepends the coding-plan endpoint so subscription quota is tried first,
falling back to the metered endpoints only for non-subscription keys. The
three tests below cover the three layers enumerated in LAYERS.md:

  1. coding-plan endpoint is the FIRST url attempted (the billing fix itself),
  2. the metered endpoints remain as an ordered fallback for non-subscription
     keys,
  3. an explicit ``auxiliary.vision.base_url`` still takes precedence and
     bypasses the hardcoded list entirely.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Test infrastructure (mirrors test_vision_routing_31179.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(monkeypatch):
    """Temp HERMES_HOME with config + clean credential env vars."""
    test_home = tempfile.mkdtemp(prefix="hermes_test_55112_")
    hermes_home = os.path.join(test_home, ".hermes")
    os.makedirs(hermes_home)
    monkeypatch.setenv("HERMES_HOME", hermes_home)

    # Strip all credential-shaped env vars so each scenario starts hermetic.
    for k in list(os.environ.keys()):
        if k.endswith("_API_KEY") or k.endswith("_TOKEN"):
            monkeypatch.delenv(k, raising=False)

    yield hermes_home
    shutil.rmtree(test_home, ignore_errors=True)


def _write_config(home: str, text: str) -> None:
    with open(os.path.join(home, "config.yaml"), "w") as fp:
        fp.write(text)


def _fresh_modules():
    """Drop cached hermes modules so each test reloads against current env."""
    for mod in list(sys.modules.keys()):
        if mod.startswith(("agent.auxiliary_client", "agent.image_routing",
                           "tools.vision_tools", "hermes_cli.config")):
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# The three zai vision endpoints under test
# ---------------------------------------------------------------------------

CODING_PLAN_URL = "https://api.z.ai/api/coding/paas/v4"
METERED_BIGMODEL_URL = "https://open.bigmodel.cn/api/paas/v4"
METERED_ZAI_URL = "https://api.z.ai/api/paas/v4"
HARDCODED_URLS = (CODING_PLAN_URL, METERED_BIGMODEL_URL, METERED_ZAI_URL)


# ---------------------------------------------------------------------------
# Layer 1: coding-plan endpoint tried first (the billing fix)
# ---------------------------------------------------------------------------


class TestZaiVisionCodingPlanFirst:
    """Issue #55112: zai vision must try the coding-plan endpoint first."""

    def test_coding_plan_endpoint_is_first_tried(self, isolated_home, monkeypatch):
        """For a subscription key, the very first base_url attempted must be the
        coding-plan endpoint so quota — not PAYG cash — is consumed."""
        _write_config(isolated_home, """
auxiliary:
  vision:
    provider: zai
    model: glm-4.5v
""")
        monkeypatch.setenv("ZAI_API_KEY", "sk-zai-test")
        _fresh_modules()

        import agent.auxiliary_client as ac

        tried_urls = []

        def fake_cached_client(*args, **kwargs):
            url = kwargs.get("base_url")
            tried_urls.append(url)
            # Simulate the coding-plan endpoint succeeding for a subscription
            # key, so the loop returns after the first attempt.
            if url == CODING_PLAN_URL:
                return MagicMock(), "glm-4.5v"
            return None, None

        with patch.object(ac, "_get_cached_client", side_effect=fake_cached_client):
            provider, client, model = ac.resolve_vision_provider_client()

        assert provider == "zai"
        assert client is not None, "zai vision should resolve to a client"
        # The very first base_url attempted must be the coding-plan endpoint.
        assert tried_urls[0] == CODING_PLAN_URL, (
            f"coding-plan endpoint must be tried first to avoid metered "
            f"billing; got order {tried_urls}"
        )

    # -- Layer 2: metered endpoints remain as ordered fallback ---------------

    def test_metered_endpoints_remain_as_fallback(self, isolated_home, monkeypatch):
        """When the coding-plan endpoint is unavailable (non-subscription key),
        the metered endpoints must still be attempted as fallbacks — in order,
        coding-plan first then metered."""
        _write_config(isolated_home, """
auxiliary:
  vision:
    provider: zai
    model: glm-4.5v
""")
        monkeypatch.setenv("ZAI_API_KEY", "sk-zai-test")
        _fresh_modules()

        import agent.auxiliary_client as ac

        tried_urls = []

        def fake_cached_client(*args, **kwargs):
            url = kwargs.get("base_url")
            tried_urls.append(url)
            # No endpoint succeeds — exhausts the whole list.
            return None, None

        with patch.object(ac, "_get_cached_client", side_effect=fake_cached_client):
            provider, client, model = ac.resolve_vision_provider_client()

        assert provider == "zai"
        assert client is None, "no endpoint available -> no client"
        # Full ordered fallback chain: coding-plan first, then the two metered.
        assert tried_urls[:3] == [
            CODING_PLAN_URL,
            METERED_BIGMODEL_URL,
            METERED_ZAI_URL,
        ], f"expected coding-plan-first fallback order, got {tried_urls}"

    # -- Layer 3: explicit base_url takes precedence ------------------------

    def test_explicit_base_url_skips_hardcoded_list(self, isolated_home, monkeypatch):
        """An explicit ``auxiliary.vision.base_url`` must take precedence and
        skip the hardcoded zai fallback list entirely."""
        _write_config(isolated_home, """
auxiliary:
  vision:
    provider: zai
    model: glm-4.5v
    base_url: https://my-private-zai-proxy.example.com/v1
""")
        monkeypatch.setenv("ZAI_API_KEY", "sk-zai-test")
        _fresh_modules()

        import agent.auxiliary_client as ac

        hardcoded_used = []

        def fake_cached_client(*args, **kwargs):
            url = kwargs.get("base_url")
            if url in HARDCODED_URLS:
                hardcoded_used.append(url)
            return MagicMock(), "glm-4.5v"

        with patch.object(ac, "_get_cached_client", side_effect=fake_cached_client):
            provider, client, model = ac.resolve_vision_provider_client()

        assert provider == "zai"
        assert client is not None, "explicit base_url should still resolve a client"
        # The hardcoded metered/coding list must NOT be used when an explicit
        # base_url is configured — the user's endpoint wins.
        assert hardcoded_used == [], (
            f"explicit base_url must bypass the hardcoded zai list; hardcoded "
            f"urls were used: {hardcoded_used}"
        )
