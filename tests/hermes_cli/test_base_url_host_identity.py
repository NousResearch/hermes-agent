"""Regression tests: provider-identity checks must compare URL *hostnames*,
not raw substrings.

Port of earendil-works/pi#7933's bug class (DeepSeek base-URL detection used a
substring check, missing case variants and matching lookalike URLs). Hermes
had the same class at several sites: keyless-endpoint detection, /model
catalog routing, local-endpoint detection, and Nous Portal cache-layout
detection all used ``"host" in base_url``. A proxy URL that merely *contains*
a provider host in its path (``https://proxy.internal/openrouter.ai/v1``) or
a lookalike domain (``https://openrouter.ai.evil.com``) must not be treated
as that provider, and casing must not matter.
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _Host(CLIAgentSetupMixin):
    def __init__(self):
        self.requested_provider = "auto"
        self._explicit_api_key = None
        self._explicit_base_url = None


def _ready_with(runtime: dict) -> bool:
    host = _Host()
    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=runtime,
    ):
        return host._runtime_credentials_ready()


def test_keyless_real_openrouter_not_ready():
    # OpenRouter itself requires a key: keyless => not ready.
    assert _ready_with({"api_key": None, "base_url": "https://openrouter.ai/api/v1"}) is False


def test_keyless_uppercase_openrouter_not_ready():
    # Case variants of the real host are still the real host (pi#7933 class).
    assert _ready_with({"api_key": None, "base_url": "https://OpenRouter.AI/api/v1"}) is False


def test_keyless_proxy_with_openrouter_in_path_is_ready():
    # A custom proxy whose *path* contains the substring is NOT OpenRouter —
    # it's a keyless custom endpoint and must count as ready.
    assert _ready_with({"api_key": None, "base_url": "https://proxy.internal/openrouter.ai/v1"}) is True


def test_keyless_lookalike_domain_is_ready():
    assert _ready_with({"api_key": None, "base_url": "https://openrouter.ai.evil.com/v1"}) is True


def test_keyless_local_endpoint_is_ready():
    assert _ready_with({"api_key": None, "base_url": "http://localhost:11434/v1"}) is True


def test_validate_requested_model_proxy_url_routes_to_custom():
    """/model validation: an 'openrouter' provider pointed at a non-OpenRouter
    host is a custom endpoint, even when the URL contains the substring."""
    from utils import base_url_host_matches

    assert base_url_host_matches("https://openrouter.ai/api/v1", "openrouter.ai")
    assert base_url_host_matches("https://OPENROUTER.AI/api/v1", "openrouter.ai")
    assert not base_url_host_matches("https://proxy.internal/openrouter.ai/v1", "openrouter.ai")
    assert not base_url_host_matches("https://openrouter.ai.evil.com/v1", "openrouter.ai")


def test_local_endpoint_hostname_detection():
    from utils import base_url_hostname

    assert base_url_hostname("http://localhost:11434/v1") == "localhost"
    assert base_url_hostname("http://127.0.0.1:1234") == "127.0.0.1"
    # A remote host with "localhost" embedded in its name is not local.
    assert base_url_hostname("https://my-localhost-mirror.com/v1") not in (
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
    )


def test_nous_portal_host_detection():
    from utils import base_url_host_matches

    assert base_url_host_matches("https://inference-api.nousresearch.com/v1", "nousresearch.com")
    assert base_url_host_matches("https://portal.nousresearch.com", "nousresearch.com")
    assert not base_url_host_matches("https://nousresearch.com.evil.io/v1", "nousresearch.com")
    assert not base_url_host_matches("https://proxy.example/nousresearch.com/v1", "nousresearch.com")
