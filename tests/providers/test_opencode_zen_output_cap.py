"""OpenCode Zen/Go relay profiles declare an output cap (#110126 Layer 1)."""

from providers import get_provider_profile


class TestOpenCodeGoDeclaredOutputCap:
    """The Zen relays apply their own very large default when we put no
    ``max_tokens`` on the wire, so a reasoning model can spend the whole
    budget on hidden thinking and come back ``finish_reason="length"`` with a
    truncated/empty body. The compressor treats that as a hard summary
    failure and aborts without rotating the session, which strands the loop.
    Declaring a ceiling keeps a real cap on the wire."""

    def test_go_relay_declares_output_cap(self):
        p = get_provider_profile("opencode-go")
        assert p is not None
        assert p.default_max_tokens == 65536

    def test_go_relay_cap_reaches_the_wire_for_plain_models(self):
        p = get_provider_profile("opencode-go")
        assert p is not None
        assert p.get_max_tokens("deepseek-v4.1-flash") == 65536

    def test_go_relay_model_specific_cap_still_wins(self):
        p = get_provider_profile("opencode-go")
        assert p is not None
        assert p.get_max_tokens("mimo-v2.5-pro") == 131072

    def test_zen_relay_declares_output_cap(self):
        p = get_provider_profile("opencode-zen")
        assert p is not None
        assert p.default_max_tokens == 65536
