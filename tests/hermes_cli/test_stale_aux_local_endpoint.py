"""Aux pins on local/LAN endpoints are not "stale" — they never bill a provider (#106228)."""

import pytest

from agent.model_metadata import is_local_endpoint
from hermes_cli.web_server_config import _stale_aux_pins


def test_local_endpoint_pins_are_excluded_from_stale_aux_report():
    cfg = {"auxiliary": {
        "title_generation": {"provider": "openai", "model": "llama3.2:3b", "base_url": "http://byron.local:11434/v1"},
        "vision": {"provider": "openai", "model": "llama3.2-vision:11b", "base_url": "http://192.168.1.10:11434/v1"},
        "compression": {"provider": "openai", "model": "gpt-4o-mini", "base_url": "https://api.example.com/v1"},
        "curator": {"provider": "openai", "model": "gpt-4o-mini"},
    }}
    stale = _stale_aux_pins(cfg, "ollama-cloud")
    # Only the pins that can still bill a provider survive: public custom URL, no base_url.
    assert sorted(entry["task"] for entry in stale) == ["compression", "curator"]


@pytest.mark.parametrize(("url", "local"), [
    ("http://byron.local:11434/v1", True),  # RFC 6762 mDNS, LAN-only
    ("http://[fd00::1]:11434", True),  # IPv6 ULA
    ("http://[2607:f8b0::1]:11434", False),  # global IPv6 must not ride the "no dots" rule
])
def test_is_local_endpoint_mdns_and_ipv6_scope(url, local):
    assert is_local_endpoint(url) is local
