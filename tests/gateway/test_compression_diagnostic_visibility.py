"""Compression route diagnostics must pass messaging status filters (#72636)."""

from unittest.mock import MagicMock

from agent.context_compressor_attribution import SummaryRoute


def test_diagnostic_survives_gateway_noise_filter() -> None:
    """The diagnostic wording must pass ``_TELEGRAM_NOISY_STATUS_RE`` so it
    reaches Telegram/Discord/Slack, not just local/CLI. The generic
    "Compression aborted" warning already passes; this pins that the
    companion identity block does too — across the failure classes.
    """
    from gateway.platforms.base import Platform
    from gateway.run import _prepare_gateway_status_message

    from agent.conversation_compression import _emit_compression_auth_hint

    # Build a minimal stand-in agent whose _emit_warning captures the
    # message, then route that message through the real gateway filter.
    for failure_class in ("auth", "network", "other"):
        comp = MagicMock()
        comp._last_attempt_failure_class = failure_class
        comp._last_aux_route = SummaryRoute("aux-provider", "aux-model", "https://aux.example", dispatched=True)
        comp.provider = "main-provider"
        comp.model = "main-model"
        comp.base_url = "https://main.example"
        agent = MagicMock()
        agent.context_compressor = comp
        captured: list[str] = []
        agent._emit_warning = lambda message, _c=captured: _c.append(message)

        _emit_compression_auth_hint(agent)
        assert captured, f"no diagnostic emitted for class={failure_class}"

        for platform in (Platform.TELEGRAM, "discord", "slack"):
            result = _prepare_gateway_status_message(platform, "warn", captured[0])
            assert result is not None, (
                f"diagnostic swallowed by gateway filter "
                f"(class={failure_class}, platform={platform}): {captured[0]}"
            )
