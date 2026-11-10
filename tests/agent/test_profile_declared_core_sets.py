"""Core sets that used to name one provider now come from the registered provider profiles.

Each contract runs through a fake, non-built-in profile: whatever a profile declares (an ambient
credential, a message-level reasoning carrier, its name) is what core honours, with no provider
named in core.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from providers.base import ProviderProfile

ACME = ProviderProfile(name="acme-cloud", aliases=("acme",), base_url="https://api.acme.test/v1",
                       ambient_credentials=True, message_carrier_opaque="acme_sig",
                       message_carrier_text="acme_thoughts")
PLAIN = ProviderProfile(name="plain-key", base_url="https://api.plain.test/v1")
_PROFILES = {p.name: p for p in (ACME, PLAIN)}


def _catalog_down(_name):
    raise RuntimeError("catalog unavailable")


def test_ambient_credential_profile_keeps_identity_when_catalog_fails():
    import agent.auxiliary_client as ac

    with patch("hermes_cli.providers.get_provider", side_effect=_catalog_down), \
            patch("hermes_cli.runtime_provider._get_named_custom_provider", return_value=None), \
            patch("providers.get_provider_profile", side_effect=_PROFILES.get):
        assert ac._preserve_provider_with_base_url("acme-cloud") is True
        assert ac._preserve_provider_with_base_url("plain-key") is False


def test_ack_detector_transition_cue_names_only_acp_agents():
    """"Launching it on <X>" is a pending launch only for an ACP agent (an external_process profile: its
    name, alias or binary); "Launching it on Nous." names an inference provider and is a final answer."""
    from agent.agent_runtime_helpers import looks_like_codex_intermediate_ack

    agent = SimpleNamespace(_strip_think_blocks=lambda s: s)
    acp = ProviderProfile(name="acme-acp", aliases=("acme",), auth_type="external_process", process_command="acmecli")
    nous = ProviderProfile(name="nous", aliases=("nous-portal",), base_url="https://api.nous.test/v1")

    def ack(text):
        return looks_like_codex_intermediate_ack(agent, "go", text, [], require_workspace=False)

    with patch("providers.list_providers", return_value=[acp, nous, PLAIN]):
        assert ack("Launching it on acme-acp via acpx.") is True
        assert ack("Relaunching the session on acme.") is True
        assert ack("Launching it on acmecli via acpx.") is True
        assert ack("Starting the job on monday.") is False
        assert ack("Launching it on Nous.") is False
        assert ack("Relaunching the session on plain-key.") is False


def test_declared_message_carriers_are_unpriced():
    from agent.message_metadata import unpriced_replay_fields
    from agent.model_metadata import estimate_messages_tokens_rough

    plain = {"role": "assistant", "content": "ok"}
    carried = {**plain, "acme_sig": "S" * 4000, "acme_thoughts": "T" * 4000}
    with patch("providers.list_providers", return_value=[ACME, PLAIN]):
        assert {"acme_sig", "acme_thoughts"} <= unpriced_replay_fields()
        assert estimate_messages_tokens_rough([carried]) == estimate_messages_tokens_rough([plain])
    with patch("providers.list_providers", return_value=[PLAIN]):
        fresh = {**plain, "acme_sig": "s" * 4000, "acme_thoughts": "t" * 4000}  # new strings: no memo hit
        assert estimate_messages_tokens_rough([fresh]) > estimate_messages_tokens_rough([plain])
