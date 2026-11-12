"""Cost meter (display.show_cost): status-bar $ segment, per-turn delta, /usage row.

The meter is display-only and deliberately silent whenever pricing is unknown — a fake
$0.00 on a model with no pricing entry would read as "free" when it is actually unmetered.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import cli as cli_mod
from cli import HermesCLI


def _make_cli(model: str = "anthropic/claude-sonnet-4-20250514"):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = model
    cli_obj.session_start = datetime.now() - timedelta(minutes=14, seconds=32)
    cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
    cli_obj.agent = None
    return cli_obj


def _attach_cost(cli_obj, *, usd: float, status: str):
    cli_obj.agent = SimpleNamespace(
        model=cli_obj.model,
        provider="anthropic",
        base_url="",
        session_input_tokens=1000,
        session_output_tokens=500,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_prompt_tokens=1500,
        session_completion_tokens=500,
        session_total_tokens=2000,
        session_api_calls=3,
        session_estimated_cost_usd=usd,
        session_cost_status=status,
        session_cost_source="official_docs_snapshot",
        get_rate_limit_state=lambda: None,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=2000, context_length=200_000, compression_count=0,
        ),
    )
    return cli_obj


def _bar(cli_obj, display: dict, width: int = 120) -> str:
    with patch.object(cli_mod, "CLI_CONFIG", {"display": display}):
        return cli_obj._build_status_bar_text(width=width)


class TestStatusBarCostMeter:
    def test_renders_sub_cent_estimate_when_enabled(self):
        cli_obj = _attach_cost(_make_cli(), usd=0.0043, status="estimated")

        text = _bar(cli_obj, {"show_cost": True})

        assert "~$0.0043" in text

    def test_hidden_when_show_cost_is_off(self):
        cli_obj = _attach_cost(_make_cli(), usd=0.0043, status="estimated")

        text = _bar(cli_obj, {"show_cost": False})

        assert "$" not in text

    def test_no_segment_when_route_has_no_rate_card(self):
        # Unknown status + a model with no price row: nothing is invented, no $0.00.
        cli_obj = _attach_cost(_make_cli(model="acme/unpriced-model-xyz"), usd=0.0, status="unknown")

        text = _bar(cli_obj, {"show_cost": True})

        assert "$" not in text

    def test_included_route_renders_label_not_a_dollar_amount(self):
        cli_obj = _attach_cost(_make_cli(), usd=0.0, status="included")

        text = _bar(cli_obj, {"show_cost": True})

        assert "included" in text
        assert "$" not in text

    def test_explicit_fields_list_can_request_cost(self):
        cli_obj = _attach_cost(_make_cli(), usd=1.2345, status="estimated")

        text = _bar(cli_obj, {"status_bar": {"fields": ["model", "cost"]}})

        assert "~$1.23" in text

    def test_explicit_fields_list_without_cost_hides_it(self):
        cli_obj = _attach_cost(_make_cli(), usd=1.2345, status="estimated")

        text = _bar(cli_obj, {"show_cost": True, "status_bar": {"fields": ["model", "duration"]}})

        assert "$" not in text


class TestPerTurnCostDelta:
    def test_turn_delta_is_appended_to_the_summary_line(self):
        cli_obj = _attach_cost(_make_cli(), usd=0.0107, status="estimated")
        cli_obj._turn_cost_baseline = 0.0064  # spend before this turn

        with patch.object(cli_mod, "CLI_CONFIG", {"display": {"show_cost": True}}):
            line = cli_obj._append_turn_cost("⋯ 12.4s · ran 2 commands")

        assert line.endswith(" · ~$0.0043")

    def test_no_delta_leaves_the_line_untouched(self):
        cli_obj = _attach_cost(_make_cli(), usd=0.0107, status="estimated")
        cli_obj._turn_cost_baseline = 0.0107

        with patch.object(cli_mod, "CLI_CONFIG", {"display": {"show_cost": True}}):
            line = cli_obj._append_turn_cost("⋯ 12.4s · ran 2 commands")

        assert line == "⋯ 12.4s · ran 2 commands"

    def test_off_switch_leaves_the_line_untouched(self):
        cli_obj = _attach_cost(_make_cli(), usd=0.0107, status="estimated")
        cli_obj._turn_cost_baseline = 0.0

        with patch.object(cli_mod, "CLI_CONFIG", {"display": {"show_cost": False}}):
            line = cli_obj._append_turn_cost("⋯ 12.4s")

        assert line == "⋯ 12.4s"


class TestUsageCostRow:
    def test_estimated(self):
        cli_obj = _attach_cost(_make_cli(), usd=0.0043, status="estimated")

        assert cli_obj._session_cost_row(cli_obj.agent) == "~$0.0043"

    def test_unknown_pricing_says_so_instead_of_zero(self):
        cli_obj = _attach_cost(_make_cli(), usd=0.0, status="unknown")

        assert cli_obj._session_cost_row(cli_obj.agent) == "no pricing data"
