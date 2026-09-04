"""Tests for hermes_cli/tips.py — random tip display at session start."""

from hermes_cli.tips import TIPS, _tip_command_refs, _tips_for_surface, get_random_tip
from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command


class TestTipsCorpus:
    """Validate the tip corpus itself."""

    def test_has_at_least_200_tips(self):
        assert len(TIPS) >= 200, f"Expected 200+ tips, got {len(TIPS)}"


    def test_all_tips_are_strings(self):
        for i, tip in enumerate(TIPS):
            assert isinstance(tip, str), f"Tip {i} is not a string: {type(tip)}"


class TestGetRandomTip:
    """Validate the get_random_tip() function."""

    def test_returns_string(self):
        tip = get_random_tip()
        assert isinstance(tip, str)
        assert len(tip) > 0

    def test_returns_tip_from_corpus(self):
        tip = get_random_tip()
        assert tip in TIPS

    def test_randomness(self):
        """Multiple calls should eventually return different tips."""
        seen = set()
        for _ in range(50):
            seen.add(get_random_tip())
        # With 200+ tips and 50 draws, we should see at least 10 unique
        assert len(seen) >= 10, f"Only got {len(seen)} unique tips in 50 draws"


class TestTipIntegrationInCLI:
    """Test that the tip display code in cli.py works correctly."""


    def test_tip_display_format(self):
        """Verify the Rich markup format doesn't break."""
        tip = get_random_tip()
        color = "#B8860B"
        markup = f"[dim {color}]✦ Tip: {tip}[/]"
        # Should not contain nested/broken Rich tags
        assert markup.count("[/]") == 1
        assert "[dim #B8860B]" in markup


def test_gateway_tips_only_reference_gateway_commands() -> None:
    gateway_tips = _tips_for_surface("gateway")

    assert "busy" in GATEWAY_KNOWN_COMMANDS
    assert len(gateway_tips) < len(TIPS)
    for tip in gateway_tips:
        for command in _tip_command_refs(tip):
            if command == "command":
                continue
            assert command in GATEWAY_KNOWN_COMMANDS, tip


def test_cli_tips_do_not_reference_gateway_only_commands() -> None:
    cli_tips = _tips_for_surface("cli")
    cli_tip_text = "\n".join(cli_tips)

    assert "/topic in Telegram DMs" not in cli_tip_text
    assert len(cli_tips) < len(TIPS)
    for tip in cli_tips:
        for command in _tip_command_refs(tip):
            command_def = resolve_command(command)
            assert command_def is None or not command_def.gateway_only, tip
