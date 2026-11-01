"""`hermes profile alias <name> --remove` must clean up an ORPHAN alias (#90983).

An orphan alias is a wrapper in ``~/.local/bin`` whose profile no longer exists;
``hermes doctor`` lists it together with the remove command. Removal used to die on
the profile-exists check before it looked at ``--remove``.
"""

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants
from hermes_cli import doctor_state, gateway_migrate, profile_cmd
from hermes_cli.profiles import create_wrapper_script
from hermes_cli.subcommands.profile import build_profile_parser


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolate the profile root (HERMES_HOME) and the wrapper dir (Path.home())."""
    (tmp_path / ".hermes").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr(gateway_migrate, "duplicate_credential_findings", list)
    return tmp_path


def _alias(profile, *, remove, name=None):
    profile_cmd._profile_alias(SimpleNamespace(profile_name=profile, remove=remove, alias_name=name))


def _list_aliases():
    profile_cmd._profile_alias(SimpleNamespace(profile_name=None, remove=False, alias_name=None, list_aliases=True))


def test_alias_list_parser_accepts_no_profile_name():
    parser = argparse.ArgumentParser()
    build_profile_parser(parser.add_subparsers(), cmd_profile=profile_cmd.cmd_profile)

    args = parser.parse_args(["profile", "alias", "--list"])

    assert args.profile_name is None
    assert args.list_aliases is True


def test_alias_without_name_keeps_a_clear_error(capsys):
    with pytest.raises(SystemExit):
        profile_cmd._profile_alias(SimpleNamespace(profile_name=None, remove=False, alias_name=None, list_aliases=False))

    assert "profile name is required" in capsys.readouterr().out


@pytest.mark.platforms("any")
def test_list_reports_live_and_orphan_aliases(isolated_home, capsys):
    live = isolated_home / ".hermes" / "profiles" / "work"
    live.mkdir(parents=True)
    (live / "config.yaml").write_text("model: m\n", encoding="utf-8")
    assert create_wrapper_script("work") is not None
    assert create_wrapper_script("old-bot", target="gone") is not None

    _list_aliases()

    output = capsys.readouterr().out
    assert "Profile aliases:" in output
    assert "work → work (ok)" in output
    assert "old-bot → gone (orphan)" in output


@pytest.mark.platforms("any")  # the wrapper name differs per host (.bat on Windows)
def test_doctor_names_the_command_that_removes_an_orphan_alias(isolated_home, capsys):
    keep = isolated_home / ".hermes" / "profiles" / "keep"  # doctor's section needs one live profile
    keep.mkdir(parents=True)
    (keep / "config.yaml").write_text("model: m\n", encoding="utf-8")
    wrapper = create_wrapper_script("petcare")
    assert wrapper is not None
    report = io.StringIO()
    with redirect_stdout(report):
        doctor_state._check_profiles(False)
    assert "(remove with: hermes profile alias petcare --remove)" in report.getvalue()

    _alias("petcare", remove=True)

    assert not wrapper.exists()
    assert "Removed alias 'petcare'" in capsys.readouterr().out


def test_remove_refuses_unknown_profile_and_never_unlinks_a_foreign_file(isolated_home):
    foreign = isolated_home / ".local" / "bin" / "notes"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("just my notes\n", encoding="utf-8")
    bot = create_wrapper_script("bot", target="work")

    with pytest.raises(SystemExit) as exc:
        _alias("notes", remove=True)
    assert exc.value.code == 1
    assert foreign.read_text(encoding="utf-8") == "just my notes\n"
    with pytest.raises(SystemExit):
        _alias("ghost", remove=True, name="bot")  # --name still requires a real profile
    assert bot is not None and bot.exists()
    with pytest.raises(SystemExit):
        _alias("ghost", remove=False)  # creating an alias still requires the profile
