"""Invariant tests for the plain-language startup messages of the ``hermes`` CLI.

Contracts (not snapshots): a mistyped subcommand names the typo, offers the closest
match and ``hermes --help`` without dumping the 70-entry choice list; a rejected
profile name explains the rule; missing optional dependencies point at ``hermes update``.
"""

import io
import os
import sys
from contextlib import redirect_stderr

import pytest

from hermes_cli._parser import build_top_level_parser


def _parse_error(argv: list[str]) -> str:
    parser, subparsers, _chat = build_top_level_parser()
    for name in ("sessions", "model", "profile"):
        subparsers.add_parser(name)
    gw = subparsers.add_parser("gateway").add_subparsers(dest="gateway_action", metavar="<action>")
    for name in ("start", "stop", "status"):
        gw.add_parser(name)
    err = io.StringIO()
    with redirect_stderr(err), pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 2
    return err.getvalue()

def test_unknown_subcommand_names_typo_suggests_closest_and_hides_choice_list():
    text = _parse_error(["sesions"])
    assert "'sesions' is not a `hermes` command" in text
    assert "Did you mean: sessions" in text
    assert "hermes --help" in text
    assert "choose from" not in text
    assert "invalid choice" not in text

def test_nested_group_typo_names_the_group_and_suggests():
    text = _parse_error(["gateway", "stat"])
    assert "'stat' is not a `hermes gateway` command" in text
    assert "Did you mean:" in text and "status" in text

def test_invalid_profile_flag_value_explains_rule_and_exits(monkeypatch):
    from hermes_cli import main as _main

    monkeypatch.setattr(sys, "argv", ["hermes", "-p", "Work Bot", "status"])
    err = io.StringIO()
    with redirect_stderr(err), pytest.raises(SystemExit) as exc:
        _main._apply_profile_override()
    assert exc.value.code == 2
    assert "'Work Bot' is not a valid profile name" in err.getvalue()
    assert "Run `hermes profile list`" in err.getvalue()

def test_pytest_style_dash_p_is_still_ignored(monkeypatch):
    from hermes_cli import main as _main

    monkeypatch.setattr(sys, "argv", ["pytest", "-p", "no:xdist", "tests/"])
    assert _main._scan_profile_flag(sys.argv[1:]) == (None, 0, None)

def test_option_looking_dash_p_value_is_a_silent_skip_even_under_hermes(monkeypatch):
    from hermes_cli import main as _main

    # `-p no:xdist` reaching us through a differently named runner (tox, nox, python -m) must not exit.
    monkeypatch.setattr(sys, "argv", ["hermes", "-p", "no:xdist", "tests/"])
    assert _main._scan_profile_flag(sys.argv[1:]) == (None, 0, None)
    monkeypatch.setattr(sys, "argv", ["hermes", "-p", "--flag"])
    assert _main._scan_profile_flag(sys.argv[1:]) == (None, 0, None)

@pytest.mark.parametrize(
    ("flag", "attached", "consume"),
    (
        pytest.param("-p", False, 2, id="short-separated"),
        pytest.param("--profile", False, 2, id="long-separated"),
        pytest.param("--profile", True, 1, id="long-attached"),
    ),
)
def test_ascii_mixed_case_profile_selector_is_normalised(flag, attached, consume):
    from hermes_cli import main as _main

    profile_args = [f"{flag}= Work "] if attached else [flag, " Work "]
    assert _main._scan_profile_flag([*profile_args, "status"]) == ("work", consume, 0)


@pytest.mark.parametrize(
    ("flag", "attached"),
    (
        pytest.param("-p", False, id="short-separated"),
        pytest.param("--profile", False, id="long-separated"),
        pytest.param("--profile", True, id="long-attached"),
    ),
)
@pytest.mark.parametrize(
    ("source", "ascii_alias"),
    (
        pytest.param("ẞ", "ss", id="casefold-expansion"),
        pytest.param("K", "k", id="simple-lower-alias"),
    ),
)
def test_unicode_profile_selector_cannot_alias_an_ascii_profile(
    tmp_path, monkeypatch, capsys, flag, attached, source, ascii_alias
):
    from hermes_cli import main as _main

    root = tmp_path / ".hermes"
    aliased_profile = root / "profiles" / ascii_alias
    aliased_profile.mkdir(parents=True)
    (aliased_profile / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    profile_args = [f"{flag}={source}"] if attached else [flag, source]
    monkeypatch.setattr(sys, "argv", ["hermes", *profile_args, "status"])

    with pytest.raises(SystemExit) as exc:
        _main._apply_profile_override()

    stderr = capsys.readouterr().err
    assert exc.value.code == 2
    assert f"{source!r} is not a valid profile name" in stderr
    assert "Use lowercase letters, numbers, '-' or '_'" in stderr
    assert "Run `hermes profile list`" in stderr
    assert os.environ["HERMES_HOME"] == str(root)


def test_invalid_dash_p_after_a_subcommand_is_left_to_that_subcommand(monkeypatch):
    from hermes_cli import main as _main

    # A plugin/subcommand flag such as `hermes kanban serve -p "Work Bot"` is not our profile selector.
    monkeypatch.setattr(sys, "argv", ["hermes", "kanban", "serve", "-p", "Work Bot"])
    assert _main._scan_profile_flag(sys.argv[1:]) == (None, 0, None)
