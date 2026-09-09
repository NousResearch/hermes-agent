"""Stale-aux reporting must ignore pins aimed at private/local endpoints.

Aux slots whose ``base_url`` is a private endpoint (localhost, LAN, mDNS — the
per-task base_url feature) can never bill a provider, so switching the main
model does not orphan them: they must not surface in the stale-aux report that
drives the desktop "still run on <provider> — Reset all to main" nudge.
"""

from hermes_cli.web_server_config import _AUX_TASK_SLOTS, _stale_aux_pins


def _cfg_with_slot(provider: str, model: str, base_url: str = "") -> dict:
    slot_cfg: dict = {"provider": provider, "model": model}
    if base_url:
        slot_cfg["base_url"] = base_url
    return {"auxiliary": {_AUX_TASK_SLOTS[0]: slot_cfg}}


class TestStaleAuxPinsPrivateEndpoints:
    def test_private_endpoint_pin_is_not_stale(self):
        """Ollama-style LAN/mDNS pin never bills a provider -> excluded."""
        cfg = _cfg_with_slot("openai", "gpt-4o", base_url="http://byron.local:11434/v1")
        assert _stale_aux_pins(cfg, "nous") == []

    def test_rfc1918_pin_is_not_stale(self):
        cfg = _cfg_with_slot("openai", "gpt-4o", base_url="http://192.168.1.10:11434/v1")
        assert _stale_aux_pins(cfg, "nous") == []

    def test_loopback_pin_is_not_stale(self):
        cfg = _cfg_with_slot("openai", "gpt-4o", base_url="http://127.0.0.1:11434/v1")
        assert _stale_aux_pins(cfg, "nous") == []

    def test_provider_pin_without_base_url_is_stale(self):
        """No base_url -> ordinary provider pin, still reported."""
        stale = _stale_aux_pins(_cfg_with_slot("openai", "gpt-4o"), "nous")
        assert len(stale) == 1
        assert stale[0]["provider"] == "openai"
        assert stale[0]["model"] == "gpt-4o"

    def test_public_custom_endpoint_pin_is_stale(self):
        """A public custom gateway can still bill -> keep reporting it."""
        cfg = _cfg_with_slot("openai", "gpt-4o", base_url="https://api.example.com/v1")
        assert len(_stale_aux_pins(cfg, "nous")) == 1

    def test_same_provider_pin_never_stale(self):
        cfg = _cfg_with_slot("nous", "hermes-4", base_url="https://api.example.com/v1")
        assert _stale_aux_pins(cfg, "nous") == []
