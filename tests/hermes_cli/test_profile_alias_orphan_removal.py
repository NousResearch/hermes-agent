"""`hermes profile alias <name> --remove` must clean up an ORPHAN alias (#90983).

An orphan alias is a wrapper in ``~/.local/bin`` whose profile no longer exists;
``hermes doctor`` lists it together with the remove command. Removal used to die on
the profile-exists check before it looked at ``--remove``.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants
from hermes_cli import doctor_state, gateway_migrate, profile_cmd
from hermes_cli.profiles import create_wrapper_script


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
