"""`hermes profile alias <name> --remove` must clean up an ORPHAN alias.

An orphan alias is a wrapper in ``~/.local/bin`` pointing at a profile that no
longer exists — the state ``hermes doctor`` reports as
``Orphan alias: <name> -> profile '<p>' no longer exists``.

Removal is the one path that must work in that state, and it is the only cleanup
the doctor can point users at. It used to be impossible: the handler checked
``profile_exists()`` before looking at ``--remove``, so the command died with
"Profile '<p>' does not exist" and the wrapper was only removable by hand.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import profile_cmd


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolate the profile root (HERMES_HOME) and the wrapper dir (Path.home())."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return tmp_path


def _write_wrapper(home: Path, name: str) -> Path:
    wrapper = home / ".local" / "bin" / name
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(f'#!/bin/sh\nexec hermes -p {name} "$@"\n', encoding="utf-8")
    return wrapper


def test_remove_alias_whose_profile_is_gone(isolated_home, capsys):
    wrapper = _write_wrapper(isolated_home, "petcare")
    assert not (isolated_home / ".hermes" / "profiles" / "petcare").exists()

    profile_cmd._profile_alias(
        SimpleNamespace(profile_name="petcare", remove=True, alias_name=None)
    )

    assert not wrapper.exists(), "the orphan wrapper must be gone"
    assert "Removed alias 'petcare'" in capsys.readouterr().out


def test_remove_never_unlinks_a_file_that_is_not_a_wrapper(isolated_home, capsys):
    """Dropping the profile check is safe only because removal stays name- and
    content-guarded: it refuses a traversal-shaped name and only unlinks a file
    that reads as a Hermes wrapper."""
    not_a_wrapper = isolated_home / ".local" / "bin" / "notes"
    not_a_wrapper.parent.mkdir(parents=True, exist_ok=True)
    not_a_wrapper.write_text("just my notes\n", encoding="utf-8")

    profile_cmd._profile_alias(
        SimpleNamespace(profile_name="notes", remove=True, alias_name=None)
    )

    assert not_a_wrapper.exists()
    # A dead end must point at the tool that surfaces these mismatches (#90983).
    out = capsys.readouterr().out
    assert "hermes doctor" in out and "Orphan alias" in out


def test_create_still_requires_an_existing_profile(isolated_home):
    """The reorder must not weaken the create path."""
    with pytest.raises(SystemExit):
        profile_cmd._profile_alias(
            SimpleNamespace(profile_name="ghost", remove=False, alias_name=None)
        )
