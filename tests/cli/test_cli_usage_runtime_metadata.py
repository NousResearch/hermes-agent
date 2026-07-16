from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from cli import HermesCLI


def _cli(*, reasoning_config, amount, status):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "openai/gpt-5.6-sol"
    cli_obj.provider = None
    cli_obj.base_url = ""
    cli_obj.api_key = ""
    cli_obj.session_start = datetime.now() - timedelta(minutes=5)
    cli_obj.conversation_history = [{"role": "user", "content": "hello"}]
    cli_obj.verbose = False
    cli_obj._print_nous_credits_block = lambda: False
    cli_obj.agent = SimpleNamespace(
        model=cli_obj.model,
        provider=None,
        base_url="",
        reasoning_config=reasoning_config,
        session_estimated_cost_usd=amount,
        session_cost_status=status,
        session_cost_source="official_docs_snapshot",
        session_input_tokens=1000,
        session_output_tokens=200,
        session_reasoning_tokens=50,
        session_prompt_tokens=1000,
        session_completion_tokens=200,
        session_total_tokens=1200,
        session_api_calls=2,
        get_rate_limit_state=lambda: None,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=1200,
            context_length=200_000,
            compression_count=0,
        ),
    )
    return cli_obj


@pytest.mark.parametrize(
    ("reasoning_config", "amount", "status", "effort", "cost"),
    [
        (
            {"enabled": True, "effort": "high"},
            0.1234,
            "estimated",
            "high",
            "~$0.12",
        ),
        ({"enabled": False}, 0.0, "included", "none", "included"),
        (None, None, "unknown", "provider-default", "unavailable"),
    ],
)
def test_usage_reports_live_reasoning_and_honest_session_cost(
    capsys,
    reasoning_config,
    amount,
    status,
    effort,
    cost,
):
    _cli(
        reasoning_config=reasoning_config,
        amount=amount,
        status=status,
    )._show_usage()

    output = capsys.readouterr().out
    assert f"Reasoning effort:            {effort}" in output
    assert f"Session cost:                {cost}" in output
