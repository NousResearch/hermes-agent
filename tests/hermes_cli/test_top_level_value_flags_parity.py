"""Top-level value-flag classification must match the real parser (#93530).

``_first_positional_argv`` and ``_apply_profile_override`` in ``main.py``
historically kept hand-maintained copies of "which top-level options consume
a value". Those copies drift — ``--reasoning`` was a value-taking top-level
option absent from both sets, so every ``hermes --reasoning high <cmd> …``
invocation mis-classified ``high`` as the first positional and forced eager
plugin CLI discovery (~500-650ms startup tax).

Both sites now share ``hermes_cli._parser.top_level_value_flag_sets()``,
derived from ``build_top_level_parser()`` (mirroring the
``update_cmd._holder_value_flags`` precedent). These tests pin:

  1. parity — the helper covers every value-taking option the live parser
     declares, so future drift fails CI instead of silently degrading
     startup;
  2. the ``--reasoning`` misparse regression itself.
"""

import sys

import pytest

from hermes_cli._parser import build_top_level_parser, top_level_value_flag_sets
from hermes_cli.main import _first_positional_argv, _plugin_cli_discovery_needed


def _parser_value_flags() -> tuple[set, set]:
    """(required, optional) option strings of value-taking top-level actions.

    An action consumes a value when its nargs is not zero-valued: Store /
    Append with nargs=None take exactly one; '?' takes an optional value.
    Boolean flags in this parser expose nargs == 0.
    """
    parser = build_top_level_parser()[0]
    required: set = set()
    optional: set = set()
    for action in parser._actions:
        if not action.option_strings or action.nargs == 0:
            continue
        target = optional if action.nargs == "?" else required
        target.update(action.option_strings)
    return required, optional


def test_helper_sets_match_top_level_parser():
    required, optional = top_level_value_flag_sets()
    parser_required, parser_optional = _parser_value_flags()
    missing_required = parser_required - required
    missing_optional = parser_optional - optional
    assert not missing_required, (
        f"value flags missing from required set: {sorted(missing_required)} "
        "(they will be misread as positionals and force eager plugin discovery)"
    )
    assert not missing_optional, (
        f"optional-value flags missing: {sorted(missing_optional)}"
    )


def test_reasoning_is_classified_as_value_flag():
    required, optional = top_level_value_flag_sets()
    assert "--reasoning" in (required | optional)


def test_first_positional_skips_reasoning_value(monkeypatch):
    monkeypatch.setattr("sys.argv", ["hermes", "--reasoning", "high", "chat", "hello"])
    assert _first_positional_argv() == "chat"
    assert _plugin_cli_discovery_needed() is False


@pytest.mark.parametrize("argv", [
    ["hermes", "--reasoning", "ultra", "--provider", "openai", "-z", "prompt here"],
    ["hermes", "-m", "x", "--reasoning=low", "chat"],
])
def test_reasoning_forms_do_not_shadow_the_subcommand(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", argv)
    first = _first_positional_argv()
    assert first not in {"high", "ultra", "low"}, (
        f"a reasoning level leaked through as the positional: {first!r}"
    )
