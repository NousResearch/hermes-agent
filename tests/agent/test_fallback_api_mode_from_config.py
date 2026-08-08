"""Fallback wire-protocol selection must honour operator config.

``try_activate_fallback`` derives the api_mode (wire protocol) for the
fallback target.  It used to do so purely from heuristics that recognise
Anthropic only by hostname (``api.anthropic.com``) or a ``/anthropic``
path suffix.  An Anthropic-Messages-compatible endpoint on any other host
— a self-hosted shim, a reverse proxy, a Cloudflare Worker — matched
neither, silently fell through to ``chat_completions``, and POSTed
``/chat/completions`` to a server that only serves ``/v1/messages``.
Every fallback attempt then failed with HTTP 404, so a transient empty
response on the primary model became a hard turn failure with no
recovery.

The primary path already honours both an explicit ``api_mode`` and the
provider's declared ``transport`` (via ``determine_api_mode()``); only
the fallback path ignored them.  These tests pin that contract.
"""

import pytest

from agent.chat_completion_helpers import configured_fallback_api_mode
from hermes_cli.providers import TRANSPORT_TO_API_MODE


@pytest.fixture
def config(monkeypatch):
    """Patch the config loader the resolver reads, returning a setter."""

    def _set(cfg):
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: cfg, raising=False
        )

    return _set


class TestExplicitApiMode:
    def test_explicit_api_mode_on_entry_wins(self, config):
        # An operator who spells out api_mode must be obeyed even when the
        # provider config disagrees — the entry is the more specific signal.
        config({"providers": {"shim": {"transport": "openai_chat"}}})
        entry = {"provider": "shim", "api_mode": "anthropic_messages"}
        assert configured_fallback_api_mode(entry, "shim") == "anthropic_messages"

    def test_whitespace_only_api_mode_is_not_a_declaration(self, config):
        config({})
        assert configured_fallback_api_mode({"api_mode": "   "}, "p") == ""

    def test_none_api_mode_does_not_crash(self, config):
        config({})
        assert configured_fallback_api_mode({"api_mode": None}, "p") == ""


class TestTransportFallback:
    def test_anthropic_shim_on_arbitrary_host_resolves_to_messages(self, config):
        # The regression: an Anthropic-Messages worker on a non-Anthropic
        # hostname. Without this, api_mode fell through to chat_completions
        # and the fallback POSTed /chat/completions -> 404.
        config(
            {
                "providers": {
                    "myworker": {
                        "base_url": "https://worker.example.workers.dev",
                        "transport": "anthropic_messages",
                    }
                }
            }
        )
        entry = {"provider": "myworker", "model": "claude-opus-4.8"}
        assert configured_fallback_api_mode(entry, "myworker") == "anthropic_messages"

    def test_openai_chat_transport_maps_to_chat_completions(self, config):
        config({"providers": {"vllm": {"transport": "openai_chat"}}})
        assert configured_fallback_api_mode({}, "vllm") == "chat_completions"

    @pytest.mark.parametrize("transport,expected", sorted(TRANSPORT_TO_API_MODE.items()))
    def test_every_known_transport_maps(self, config, transport, expected):
        # Guards the mapping as a contract rather than freezing a literal
        # list: a newly supported transport is covered automatically.
        config({"providers": {"p": {"transport": transport}}})
        assert configured_fallback_api_mode({}, "p") == expected

    def test_unknown_transport_falls_through_to_heuristics(self, config):
        config({"providers": {"p": {"transport": "carrier-pigeon"}}})
        assert configured_fallback_api_mode({}, "p") == ""


class TestNoDeclarationLeavesHeuristicsIntact:
    """Returning "" is load-bearing: the caller then runs its heuristics.

    Any of these accidentally returning a mode would override the
    provider/base-URL/model detection that existing providers rely on.
    """

    def test_provider_absent_from_config(self, config):
        config({"providers": {"other": {"transport": "openai_chat"}}})
        assert configured_fallback_api_mode({}, "openrouter") == ""

    def test_provider_entry_without_transport(self, config):
        config({"providers": {"p": {"base_url": "https://x.example"}}})
        assert configured_fallback_api_mode({}, "p") == ""

    def test_no_providers_section(self, config):
        config({})
        assert configured_fallback_api_mode({}, "p") == ""

    def test_null_providers_section(self, config):
        config({"providers": None})
        assert configured_fallback_api_mode({}, "p") == ""

    def test_config_loader_returning_none(self, config):
        config(None)
        assert configured_fallback_api_mode({}, "p") == ""

    def test_non_dict_provider_entry_is_ignored(self, config):
        # A malformed config (string where a mapping belongs) must not raise
        # mid-failover — failover is the recovery path, it cannot crash.
        config({"providers": {"p": "https://x.example"}})
        assert configured_fallback_api_mode({}, "p") == ""


def test_config_load_failure_does_not_break_failover(monkeypatch):
    """A broken/unreadable config must degrade, never raise.

    This runs while the agent is already failing over; an exception here
    would turn a recoverable provider blip into a dead turn.
    """

    def boom():
        raise OSError("config.yaml unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config", boom, raising=False)
    assert configured_fallback_api_mode({}, "p") == ""
