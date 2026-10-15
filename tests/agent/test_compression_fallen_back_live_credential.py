"""Latched compression fallback must send the live main credential.

Regression for #138469: after an aux summary failure, ``_apply_summary_route`` pinned the
compressor's stored ``api_key``. That snapshot can already have diverged from the session's
working client when the main loop never 401'd, so #121756 never moved holders. Later
compactions then 401'd with the revoked key for the rest of the session.

When a turn has published a same-provider runtime, that live key wins (so the pool's
priority-0 row cannot replace a working session token). When no live runtime is present,
an OAuth-shaped stored snapshot is omitted from both the pin and ``main_runtime`` instead
of being sent as an explicit key.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.auxiliary_client import scoped_runtime_main
from agent.context_compressor import ContextCompressor


# Shape prefixes only, assembled at runtime — not a real credential.
OAUTH_STALE = "sk-ant-" + "oat-stale-snapshot"
OAUTH_LIVE = "sk-ant-" + "oat-live-working"
STATIC_STALE = "stale-compressor-snapshot-key"


def _compressor(api_key):
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="claude-opus-5-5",
            provider="anthropic",
            quiet_mode=True,
            base_url="https://api.anthropic.com",
            api_key=api_key,
            summary_model_override="quota-exhausted-aux",
        )


def _ok_response():
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="summary via live credential " * 4),
        finish_reason="stop",
    )])


def _capture():
    seen = {}

    def _fake(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return _ok_response()

    return seen, _fake


def _sent_keys(seen):
    return seen.get("api_key"), (seen.get("main_runtime") or {}).get("api_key")


@pytest.mark.parametrize("entry", ["summary", "micro"])
@pytest.mark.parametrize(
    "live_runtime, stored_key, expected_key",
    [
        ({"provider": "anthropic", "model": "claude-opus-5-5", "api_key": OAUTH_LIVE}, OAUTH_STALE, OAUTH_LIVE),
        ({"provider": "openai-codex", "model": "gpt-5.6-luna", "api_key": OAUTH_LIVE}, STATIC_STALE, STATIC_STALE),
        (None, STATIC_STALE, STATIC_STALE),
        (None, OAUTH_STALE, None),
    ],
)
def test_fallen_back_route_prefers_live_credential_and_omits_stale_oauth(
    entry, live_runtime, stored_key, expected_key,
):
    compressor = _compressor(stored_key)
    compressor._fallback_to_main_for_compression(Exception("OAuth access token has been revoked."), "failed")
    seen, fake = _capture()
    patch_target = (
        "agent.context_compressor.call_llm" if entry == "summary" else "agent.auxiliary_client.call_llm"
    )
    prompt_patch = (
        patch.object(
            compressor, "_build_micro_summary_prompt", return_value=[{"role": "user", "content": "x"}],
        )
        if entry == "micro"
        else nullcontext()
    )
    runtime_cm = scoped_runtime_main(live_runtime) if live_runtime is not None else nullcontext()
    with runtime_cm, patch(patch_target, side_effect=fake), prompt_patch:
        if entry == "summary":
            compressor._call_summary_llm("prompt", 0.0)
        else:
            compressor._micro_summarize_one("an exchange")

    pinned_key, runtime_key = _sent_keys(seen)
    if expected_key is None:
        assert pinned_key in (None, "")
        assert runtime_key in (None, "")
        assert compressor.api_key == stored_key
        return
    assert pinned_key == expected_key
    assert runtime_key == expected_key
    assert compressor.api_key == expected_key


def test_pre_compression_refresh_adopts_agent_oauth_without_runtime_context():
    """compress_context refreshes then summarizes on a worker. Equality-gated holder
    moves miss a diverged snapshot; the live agent key must still be pinned (#138469)."""
    from agent.conversation_compression import _refresh_main_credential

    compressor = _compressor(OAUTH_STALE)
    compressor._fallback_to_main_for_compression(Exception("OAuth access token has been revoked."), "failed")
    agent = SimpleNamespace(
        api_mode="anthropic_messages",
        api_key=OAUTH_LIVE,
        _anthropic_api_key=OAUTH_LIVE,
        provider="anthropic",
        context_compressor=compressor,
        _try_refresh_anthropic_client_credentials=lambda: False,
    )
    _refresh_main_credential(agent)
    seen, fake = _capture()
    with patch("agent.context_compressor.call_llm", side_effect=fake):
        compressor._call_summary_llm("prompt", 0.0)
    pinned_key, runtime_key = _sent_keys(seen)
    assert pinned_key == OAUTH_LIVE
    assert runtime_key == OAUTH_LIVE
    assert compressor.api_key == OAUTH_LIVE
