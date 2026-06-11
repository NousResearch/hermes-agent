from gateway.run import _prepare_gateway_status_message
from gateway.platforms.base import Platform


def test_codex_gpt55_autoraise_notice_is_not_sent_to_discord_status():
    notice = (
        "ℹ Codex gpt-5.5 caps context at 272K, so auto-compaction was raised "
        "to 85% (from 50%) to use more of the window before summarizing.\n"
        "  Opt back out: hermes config set compression.codex_gpt55_autoraise false"
    )

    assert _prepare_gateway_status_message(Platform.DISCORD, "lifecycle", notice) is None


def test_non_autoraise_compression_warning_still_sent_to_discord_status():
    warning = (
        "⚠ Compression model small-model (openrouter) context is 80,000 tokens, "
        "but the main model gpt-5.5 (openai-codex)'s compression threshold was "
        "136,000 tokens. Auto-lowered this session's threshold to 80,000 tokens "
        "so compression can run."
    )

    assert _prepare_gateway_status_message(Platform.DISCORD, "lifecycle", warning) == warning
