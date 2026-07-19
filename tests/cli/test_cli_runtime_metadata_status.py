from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from cli import HermesCLI


def _cli(*, reasoning_config, amount=0.1234, status="estimated"):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "openai/gpt-5.6-sol"
    cli_obj.session_start = datetime.now() - timedelta(minutes=14, seconds=32)
    cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
    cli_obj.config = {
        "display": {
            "show_reasoning_effort": True,
            "show_cost": True,
        }
    }
    cli_obj.agent = SimpleNamespace(
        model=cli_obj.model,
        provider="openai",
        base_url="",
        reasoning_config=reasoning_config,
        session_estimated_cost_usd=amount,
        session_cost_status=status,
        session_cost_source="official_docs_snapshot",
        session_input_tokens=10000,
        session_output_tokens=5000,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_prompt_tokens=10000,
        session_completion_tokens=5000,
        session_total_tokens=15000,
        session_api_calls=7,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=50000,
            context_length=200_000,
            compression_count=0,
        ),
    )
    return cli_obj


def test_status_bar_shows_live_reasoning_and_estimated_cost_when_enabled():
    text = _cli(
        reasoning_config={"enabled": True, "effort": "medium"},
    )._build_status_bar_text(width=120)

    assert "r:medium" in text
    assert "~$0.12" in text


def test_status_bar_shows_subscription_cost_as_included():
    text = _cli(
        reasoning_config={"enabled": False},
        amount=0.0,
        status="included",
    )._build_status_bar_text(width=120)

    assert "r:none" in text
    assert "included" in text
    assert "$0.00" not in text


def test_narrow_status_bar_keeps_reasoning_and_drops_cost():
    text = _cli(
        reasoning_config={"enabled": True, "effort": "high"},
        status="actual",
    )._build_status_bar_text(width=60)

    assert "r:high" in text
    assert "$0.12" not in text


def test_status_bar_hides_runtime_metadata_when_flags_are_disabled():
    cli_obj = _cli(reasoning_config={"enabled": True, "effort": "high"})
    cli_obj.config["display"].update(
        show_reasoning_effort=False,
        show_cost=False,
    )

    text = cli_obj._build_status_bar_text(width=120)

    assert "r:high" not in text
    assert "$" not in text


def test_rich_status_bar_fragments_match_plain_runtime_metadata(monkeypatch):
    cli_obj = _cli(reasoning_config={"enabled": True, "effort": "high"})
    cli_obj._status_bar_visible = True
    cli_obj._model_picker_state = None
    monkeypatch.setattr(cli_obj, "_get_tui_terminal_width", lambda: 120)

    rendered = "".join(text for _, text in cli_obj._get_status_bar_fragments())

    assert "r:high" in rendered
    assert "~$0.12" in rendered
