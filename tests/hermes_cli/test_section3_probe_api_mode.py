"""Section 3 (``providers:``) must probe ``/models`` with the wire protocol the entry declares.

``api_mode`` and ``transport`` are two accepted spellings of one setting: ``_entry_api_mode``
resolves both and already keys the group identity. The probe argument must resolve the same way,
or a ``transport:``-only entry is discovered over the wrong protocol — an Anthropic-compatible
endpoint probed with a bearer answers 400, discovery yields nothing, and the picker silently
falls back to the config-declared model list.
"""

import pytest

import hermes_cli.models as models_mod
import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.model_switch_providers import _entry_api_mode


@pytest.fixture
def probed_api_mode(monkeypatch):
    """Run the picker over one ``providers:`` entry; return the api_mode the probe received."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setenv("RELAY_TOKEN", "relay-key")

    def run(entry, *, serves_only="anthropic_messages"):
        seen = {}

        def fake_cached_fetch(api_key, base_url, **kwargs):
            # A real Anthropic-compatible host answers a bearer probe with 400
            # ("anthropic-version: header is required"), which reaches the picker as no catalog.
            seen["api_mode"] = kwargs.get("api_mode")
            return ["discovered-model"] if kwargs.get("api_mode") == serves_only else None

        monkeypatch.setattr(models_mod, "cached_fetch_api_models", fake_cached_fetch)
        rows = list_authenticated_providers(
            user_providers={"relay": {"name": "Relay", "base_url": "https://relay.example/v1",
                                      "key_env": "RELAY_TOKEN", "model": "declared-model", **entry}},
            custom_providers=[], max_models=50)
        user_rows = [r for r in rows if r.get("source") == "user-config"]
        assert len(user_rows) == 1
        return seen.get("api_mode"), user_rows[0]

    return run


@pytest.mark.parametrize("spelling", ["api_mode", "transport"])
def test_probe_uses_the_declared_wire_protocol(probed_api_mode, spelling):
    """Either spelling reaches the probe. ``transport``-only is the shape config migration
    writes (``_custom_provider_entry_to_provider_config``), so it is the common one."""
    entry = {spelling: "anthropic_messages"}
    probed, _row = probed_api_mode(entry)
    assert probed == _entry_api_mode(entry) == "anthropic_messages"


def test_probe_api_mode_matches_the_group_identity(probed_api_mode):
    """The group is keyed by ``_entry_api_mode``; probing under a different value would
    discover one row's catalog over another row's protocol."""
    entry = {"transport": "anthropic_messages"}
    probed, _row = probed_api_mode(entry)
    assert probed == _entry_api_mode(entry)


def test_declared_catalog_is_replaced_by_the_discovered_one(probed_api_mode):
    """The symptom users see: probed over the protocol the host serves, the row carries live
    models instead of the ids hand-written in config.yaml. Probed over the wrong one the host
    refuses, and a model absent from config.yaml can never appear in the picker."""
    _probed, row = probed_api_mode({"transport": "anthropic_messages"})
    assert row["models"] == ["discovered-model"]
