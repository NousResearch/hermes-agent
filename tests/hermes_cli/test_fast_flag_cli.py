"""``--fast`` picks a fast-mode tier per invocation, the way ``--reasoning`` picks an effort.

The gap: ``--reasoning`` reaches every surface, ``agent.service_tier`` had no argv equivalent, so a
headless ``hermes chat -q`` run could only use the profile-wide tier. These pin the flag's parser
contract and the resolution that lets it win over ``config.yaml`` for this launch only.
"""

import logging

import pytest

from hermes_cli._parser import build_top_level_parser, top_level_value_flag_sets
from hermes_cli.cli_config_load import _UNKNOWN_TIER, _parse_cli_service_tier
from hermes_cli.cli_init_mixin import CLIInitMixin


def _chat_parse(argv):
    """Parse ``hermes chat ...`` the way ``main`` wires the parsers, so SUPPRESS defaults apply."""
    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=lambda args: args)
    return parser.parse_args(["chat", *argv])


def _parse(argv):
    """Parse a top-level invocation (flags before any subcommand)."""
    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=lambda args: args)
    return parser.parse_args(argv)


class TestParserSurface:
    def test_bare_fast_means_fast(self):
        assert _chat_parse(["--fast"]).fast == "fast"

    @pytest.mark.parametrize("word", ["fast", "auto", "cold", "ultrafast", "normal", "FAST", "auto "])
    def test_every_tier_word_is_taken_verbatim(self, word):
        assert _chat_parse(["--fast", word]).fast == word

    def test_absent_flag_is_suppressed_not_none(self):
        # SUPPRESS: an unpassed flag leaves no attribute at all, which is why main.py reads
        # getattr(args, "fast", None) rather than args.fast. main.py does exactly that.
        args = _chat_parse([])
        assert getattr(args, "fast", None) is None
        assert not hasattr(args, "fast")

    def test_equals_form(self):
        assert _chat_parse(["--fast=cold"]).fast == "cold"

    def test_flag_is_chat_only_so_it_cannot_eat_the_subcommand(self):
        # nargs="?" on the TOP-LEVEL parser would make `hermes --fast chat` parse `chat` as the
        # tier word, the misparse #93530 fixed for --reasoning. -c/--continue is chat-only too.
        parser = build_top_level_parser()[0]
        top_level = {opt for action in parser._actions for opt in action.option_strings}
        assert "--fast" not in top_level
        required, optional = top_level_value_flag_sets()
        assert "--fast" not in required | optional

    def test_flag_never_leaks_into_the_oneshot_path(self):
        # `hermes -z` builds its own agent and reads the tier from config alone, so the flag is
        # declared only where it is honored.
        args = _parse(["-z", "prompt here"])
        assert not hasattr(args, "fast") or args.fast is None

    def test_optional_value_flag_is_classified_for_the_argv_scanners(self):
        # The chat subparser's own scanners must know --fast takes an optional value.
        assert _chat_parse(["--fast"]).fast == "fast"


class TestTierResolution:
    @pytest.mark.parametrize("word,expected", [
        ("fast", "priority"),
        ("priority", "priority"),
        ("ultrafast", "ultrafast"),
        ("auto", "auto"),
        ("cold", "cold"),
        ("FAST", "priority"),
        ("  auto  ", "auto"),
    ])
    def test_known_words_map_to_a_tier(self, word, expected):
        assert _parse_cli_service_tier(word) == expected

    @pytest.mark.parametrize("word", ["normal", "off", "none", "default", "standard", ""])
    def test_normal_words_mean_no_fast(self, word):
        assert _parse_cli_service_tier(word) is None

    @pytest.mark.parametrize("word", ["turbo", "fastt", "yes", "1", "priority!"])
    def test_unknown_word_is_distinguishable_from_normal(self, word):
        # The config parser returns None for both; only an explicit --fast can warn about one.
        assert _parse_cli_service_tier(word) is _UNKNOWN_TIER

    def test_every_word_agrees_with_the_config_parser(self):
        from agent.fast_mode import NORMAL_TIER_WORDS, parse_service_tier

        from hermes_cli.cli_config_load import _parse_service_tier_config

        for word in ("fast", "auto", "cold", "ultrafast", "on", *NORMAL_TIER_WORDS):
            assert _parse_cli_service_tier(word) == _parse_service_tier_config(word), word
            assert _parse_cli_service_tier(word) == parse_service_tier(word), word


class _StubCli(CLIInitMixin):
    """The slice of HermesCLI that ``_init_prompt_and_reasoning`` needs: no agent, no config load."""

    model: str = "gpt-6-astra"
    service_tier: "str | None"
    _explicit_service_tier: "str | None"


@pytest.fixture
def cli_config(monkeypatch):
    """CLI_CONFIG with a configured tier, so the config branch is exercised rather than stubbed."""
    import cli as cli_mod

    config = {"agent": {"service_tier": "cold"}}
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", config, raising=False)
    return config


def _resolve(flag):
    obj = _StubCli()
    obj._init_prompt_and_reasoning(None, flag)  # returns None; the state lands on obj
    return obj


class TestFlagBeatsConfig:
    def test_bare_flag_overrides_the_config_tier(self, cli_config):
        cli = _resolve("fast")
        assert cli.service_tier == "priority"
        assert cli._explicit_service_tier == "priority"

    def test_flag_is_not_persisted(self, cli_config):
        _resolve("ultrafast")
        assert cli_config["agent"]["service_tier"] == "cold"

    def test_normal_flag_turns_a_configured_tier_off(self, cli_config):
        assert _resolve("normal").service_tier is None

    @pytest.mark.parametrize("flag", [None, "", "   "])
    def test_absent_or_blank_flag_leaves_the_config_tier_alone(self, cli_config, flag):
        cli = _resolve(flag)
        assert cli.service_tier == "cold"
        assert cli._explicit_service_tier is None

    def test_unknown_word_warns_and_keeps_the_config_tier(self, cli_config, caplog):
        with caplog.at_level(logging.WARNING):
            cli = _resolve("turbo")
        assert cli.service_tier == "cold"
        assert cli._explicit_service_tier is None
        assert any("turbo" in record.getMessage() for record in caplog.records)
