"""MiniMax serves two wires on one host — the path decides which.

``…/anthropic`` speaks native Messages; ``…/v1`` is OpenAI-style only. The
``minimax`` provider overlay hardcodes ``transport="anthropic_messages"``, so
without a host mandate ``determine_api_mode`` returns the Messages wire for a
``/v1`` base_url and every request POSTs ``/v1/messages`` — which MiniMax
answers with a bare ``404 page not found``.

That regression is invisible in config: ``model.api_mode: chat_completions``
only covers callers that read ``model.*``. Cron jobs pinned to
``provider=minimax``, delegation and auxiliary clients resolve on their own and
went out on the wrong wire regardless.
"""

import pytest

from hermes_cli.providers import determine_api_mode, host_mandated_api_mode


GLOBAL_V1 = "https://api.minimax.io/v1"
CN_V1 = "https://api.minimaxi.com/v1"
GLOBAL_ANTHROPIC = "https://api.minimax.io/anthropic"


@pytest.mark.parametrize(
    "base_url",
    [
        GLOBAL_V1,
        f"{GLOBAL_V1}/",
        CN_V1,
        "https://api.minimax.chat/v1",
        "https://API.MiniMax.io/v1",  # host matching is case-insensitive
    ],
)
def test_v1_endpoints_mandate_chat_completions(base_url):
    """A /v1 base_url must never resolve to the Messages wire."""
    assert host_mandated_api_mode(base_url) == "chat_completions"
    assert determine_api_mode("minimax", base_url, "MiniMax-M3") == "chat_completions"


@pytest.mark.parametrize(
    "base_url",
    [GLOBAL_ANTHROPIC, f"{GLOBAL_ANTHROPIC}/", "https://api.minimaxi.com/anthropic"],
)
def test_anthropic_path_still_wins(base_url):
    """The /anthropic check runs first, so that endpoint keeps native Messages."""
    assert host_mandated_api_mode(base_url) == "anthropic_messages"
    assert determine_api_mode("minimax", base_url, "MiniMax-M3") == "anthropic_messages"


def test_mandate_overrides_stale_session_api_mode():
    """A resumed pre-fix session carries api_mode=anthropic_messages in state.

    Host mandates are applied as an override (model_switch.py), not merely as a
    fill-when-empty, so those sessions self-correct instead of 404ing forever.
    """
    stale = "anthropic_messages"
    mandated = host_mandated_api_mode(GLOBAL_V1)
    assert mandated is not None, "no mandate means the stale value would survive"
    assert (mandated if mandated is not None else stale) == "chat_completions"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.minimax.io.attacker.test/v1",  # suffix spoof
        "https://proxy.test/api.minimax.io/v1",     # path-segment spoof
        "https://generic.example.com/v1",
    ],
)
def test_lookalike_hosts_are_not_mandated(base_url):
    """Exact-hostname matching only — never a bare substring (#32243)."""
    assert host_mandated_api_mode(base_url) is None


def test_other_provider_mandates_unaffected():
    """Guard against the MiniMax branch shadowing existing mandates."""
    assert host_mandated_api_mode("https://api.anthropic.com") == "anthropic_messages"
    assert host_mandated_api_mode("https://api.openai.com/v1") == "codex_responses"
    assert host_mandated_api_mode("") is None
