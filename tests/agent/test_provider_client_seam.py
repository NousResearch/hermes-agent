"""A provider profile can supply its own client — the registration seam.

Most providers speak OpenAI-over-HTTP and want the client the core builds. A
provider whose wire protocol is something else (the ACP subprocess shims) must
be able to supply its own — and must be able to do so from *outside* this tree,
otherwise every such integration has to edit ``create_openai_client`` by hand.

These tests pin the hook, the two resolution paths into it (provider name and
``base_url``), its failure isolation, and the capability flags that let such a
client opt out of the transport/async wrappers without this code importing it.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import providers as _providers  # noqa: E402
from providers.base import ProviderProfile  # noqa: E402


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _SeamProfile(ProviderProfile):
    def create_client(self, **kwargs):
        return _FakeClient(**kwargs)


class _ExplodingProfile(ProviderProfile):
    def create_client(self, **kwargs):
        raise RuntimeError("plugin is broken")


@pytest.fixture
def registered():
    """Register profiles for one test and restore the registry afterwards."""
    _providers._discover_providers()
    snapshot = (
        dict(_providers._REGISTRY),
        dict(_providers._ALIASES),
        _providers._PROVIDER_LIST_CACHE,
    )

    def _register(profile):
        _providers.register_provider(profile)
        return profile

    yield _register

    _providers._REGISTRY.clear()
    _providers._REGISTRY.update(snapshot[0])
    _providers._ALIASES.clear()
    _providers._ALIASES.update(snapshot[1])
    _providers._PROVIDER_LIST_CACHE = snapshot[2]


def _agent(provider: str = ""):
    """The minimal agent surface create_openai_client touches."""
    return SimpleNamespace(
        provider=provider,
        _client_log_context=lambda: "",
        _build_keepalive_http_client=lambda *a, **k: None,
    )


# ── the hook itself ──────────────────────────────────────────────────────────


def test_the_default_profile_supplies_no_client():
    """Every ordinary provider inherits this and keeps the standard client."""
    assert ProviderProfile(name="plain").create_client(api_key="k") is None


def test_a_profile_can_supply_its_own_client(registered):
    from agent.agent_runtime_helpers import _provider_supplied_client

    registered(_SeamProfile(name="seam-test", base_url="acp://seam-test"))
    client = _provider_supplied_client(_agent("seam-test"), {"api_key": "k"})
    assert isinstance(client, _FakeClient)
    assert client.kwargs == {"api_key": "k"}


def test_an_alias_reaches_the_same_profile(registered):
    from agent.agent_runtime_helpers import _provider_supplied_client

    registered(_SeamProfile(name="seam-test", aliases=("seam",), base_url="acp://seam-test"))
    assert isinstance(_provider_supplied_client(_agent("seam"), {}), _FakeClient)


def test_the_base_url_resolves_the_profile_when_the_name_does_not(registered):
    """A runtime configured only by URL still reaches its provider's client."""
    from agent.agent_runtime_helpers import _provider_supplied_client

    registered(_SeamProfile(name="seam-test", base_url="acp://seam-test"))
    client = _provider_supplied_client(_agent(""), {"base_url": "acp://seam-test"})
    assert isinstance(client, _FakeClient)
    # Prefix, not equality — the branch this replaced keyed on startswith().
    assert isinstance(
        _provider_supplied_client(_agent(""), {"base_url": "acp://seam-test/x"}),
        _FakeClient,
    )
    # A different scheme must not be captured.
    assert _provider_supplied_client(_agent(""), {"base_url": "https://api.example/v1"}) is None


def test_an_unregistered_provider_gets_no_client(registered):
    from agent.agent_runtime_helpers import _provider_supplied_client

    assert _provider_supplied_client(_agent("nothing-registered-here"), {}) is None


def test_a_broken_plugin_cannot_take_the_turn_down(registered):
    """A third-party profile that raises falls through to the standard path."""
    from agent.agent_runtime_helpers import _provider_supplied_client

    registered(_ExplodingProfile(name="seam-boom", base_url="acp://seam-boom"))
    assert _provider_supplied_client(_agent("seam-boom"), {"api_key": "k"}) is None


# ── the in-tree consumer ─────────────────────────────────────────────────────


def test_copilot_acp_still_gets_its_acp_client():
    """copilot-acp lost its hardcoded branch in create_openai_client; it must
    still be constructed, now via its profile."""
    from agent.copilot_acp_client import CopilotACPClient
    from providers import get_provider_profile

    profile = get_provider_profile("copilot-acp")
    assert profile is not None
    client = profile.create_client(api_key="copilot-acp", base_url="acp://copilot")
    assert isinstance(client, CopilotACPClient)


def test_copilot_acp_reaches_the_seam_by_name_and_by_url():
    from agent.agent_runtime_helpers import _provider_supplied_client
    from agent.copilot_acp_client import CopilotACPClient

    by_name = _provider_supplied_client(
        _agent("copilot-acp"), {"api_key": "copilot-acp", "base_url": "acp://copilot"}
    )
    assert isinstance(by_name, CopilotACPClient)
    by_url = _provider_supplied_client(
        _agent(""), {"api_key": "copilot-acp", "base_url": "acp://copilot"}
    )
    assert isinstance(by_url, CopilotACPClient)


# ── the wiring, not just the helper ──────────────────────────────────────────


def test_create_openai_client_actually_consults_the_profile(registered):
    """Exercise the real construction entry point, not the helper it calls.

    The helper being correct is worth nothing if ``create_openai_client`` stops
    calling it — that is the regression this file exists to prevent.
    """
    from agent.agent_runtime_helpers import create_openai_client

    registered(_SeamProfile(name="seam-test", base_url="acp://seam-test"))
    client = create_openai_client(
        _agent("seam-test"),
        {"api_key": "k", "base_url": "acp://seam-test"},
        reason="test",
        shared=False,
    )
    assert isinstance(client, _FakeClient)


def test_create_openai_client_still_builds_the_standard_client(registered):
    """A provider with no profile hook is untouched by the seam."""
    from openai import OpenAI

    from agent.agent_runtime_helpers import create_openai_client

    client = create_openai_client(
        _agent("openai-api"),
        {"api_key": "k", "base_url": "https://api.example/v1"},
        reason="test",
        shared=False,
    )
    assert isinstance(client, OpenAI)


def test_create_openai_client_builds_the_copilot_acp_client():
    """The in-tree consumer, through the same entry point its hardcoded branch
    used to serve."""
    from agent.agent_runtime_helpers import create_openai_client
    from agent.copilot_acp_client import CopilotACPClient

    client = create_openai_client(
        _agent("copilot-acp"),
        {"api_key": "copilot-acp", "base_url": "acp://copilot"},
        reason="test",
        shared=False,
    )
    assert isinstance(client, CopilotACPClient)


# ── capability flags replacing the isinstance checks ─────────────────────────


def test_clients_declaring_skip_flags_are_not_wrapped():
    from agent.auxiliary_client import _client_declares

    class _Final:
        HERMES_SKIP_TRANSPORT_WRAP = True
        HERMES_SKIP_ASYNC_WRAP = True

    class _Plain:
        pass

    final = _Final()
    assert _client_declares(final, "HERMES_SKIP_TRANSPORT_WRAP")
    assert _client_declares(final, "HERMES_SKIP_ASYNC_WRAP")
    # Absent attribute → False, so every ordinary client keeps its behaviour.
    assert not _client_declares(_Plain(), "HERMES_SKIP_TRANSPORT_WRAP")
    assert not _client_declares(None, "HERMES_SKIP_TRANSPORT_WRAP")


def test_the_in_tree_clients_declare_what_their_isinstance_checks_used_to_say():
    """copilot-acp skipped both wrappers; the Gemini native client skipped only
    the transport one (its async path is a real conversion)."""
    from agent.copilot_acp_client import CopilotACPClient
    from agent.gemini_native_adapter import GeminiNativeClient

    assert CopilotACPClient.HERMES_SKIP_TRANSPORT_WRAP is True
    assert CopilotACPClient.HERMES_SKIP_ASYNC_WRAP is True
    assert GeminiNativeClient.HERMES_SKIP_TRANSPORT_WRAP is True
    assert getattr(GeminiNativeClient, "HERMES_SKIP_ASYNC_WRAP", False) is False


def test_an_out_of_tree_client_is_covered_by_the_same_flags():
    """The point of the flags: auxiliary_client never imports this class."""
    from agent.auxiliary_client import _maybe_wrap_anthropic, _to_async_client

    class _ThirdPartyClient:
        HERMES_SKIP_TRANSPORT_WRAP = True
        HERMES_SKIP_ASYNC_WRAP = True
        api_key = "k"
        base_url = "acp://third-party"

    client = _ThirdPartyClient()
    assert _maybe_wrap_anthropic(client, "m", "k", "acp://third-party") is client
    assert _to_async_client(client, "m")[0] is client
