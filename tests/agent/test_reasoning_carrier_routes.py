"""Route-level reasoning carrier selection: remote custom relays drop `reasoning`.

NVIDIA NIM (and similar OpenAI-compatible relays) treat `reasoning` as an alias
of `reasoning_content` and reject chat-completions payloads that carry both
keys. The carrier router must therefore send `reasoning_content` only to
remote custom providers, while local servers keep the full carrier set.
"""

from agent.message_sanitization import RC, route_reasoning_carriers


def test_remote_custom_provider_drops_reasoning_carrier():
    remote = route_reasoning_carriers("custom", "https://integrate.api.nvidia.com/v1")
    assert remote == frozenset({RC}), f"remote custom must send reasoning_content only, got {remote}"


def test_local_custom_provider_keeps_full_carrier_set():
    local = route_reasoning_carriers("custom", "http://127.0.0.1:8080")
    assert RC in local
    assert len(local) >= 2, f"local server should keep reasoning + reasoning_content, got {local}"


def test_known_remote_custom_base_urls_drop_reasoning_carrier():
    for base_url in (
        "https://integrate.api.nvidia.com/v1",
        "https://custom-relay.example.com/v1",
    ):
        carriers = route_reasoning_carriers("custom", base_url)
        assert "reasoning" not in carriers, f"{base_url}: reasoning leaked to remote custom relay"
