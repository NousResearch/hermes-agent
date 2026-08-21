"""`hermes profile alias <name> --remove` must work for orphaned wrappers (#90983).

The action validated `profile_exists(name)` before the `--remove` branch, so
the remove path was unreachable exactly when a wrapper is orphaned (its target
profile no longer exists) — the state `hermes doctor` reports as "Orphan
alias". Removal now skips the profile check; `remove_wrapper_script()` itself
only unlinks files inside the wrapper dir whose content carries the
`hermes -p` marker, so the fail-closed properties hold without it.
"""

import types
from pathlib import Path

import pytest

import hermes_cli.main as main_mod
from hermes_cli.profiles import remove_wrapper_script


def _args(action: str, name: str, remove: bool = False, alias_name=None):
    return types.SimpleNamespace(
        profile_name=name, profile_action=action, remove=remove, alias_name=alias_name
    )


@pytest.fixture
def wrapper_dir(tmp_path, monkeypatch):
    import hermes_cli.profiles as profiles_mod

    d = tmp_path / "bin"
    d.mkdir()
    monkeypatch.setattr(profiles_mod, "_get_wrapper_dir", lambda: d)
    return d


def test_remove_clears_orphaned_wrapper_without_profile(wrapper_dir, monkeypatch, capsys):
    """The #90983 shape: wrapper exists (marker intact), profile is gone —
    removal must succeed instead of exiting with 'Profile does not exist'."""
    import hermes_cli.profiles as profiles_mod

    (wrapper_dir / "demo").write_text("#!/bin/sh\nhermes -p demo \"$@\"\n")
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda _n: False)
    monkeypatch.setattr(
        main_mod, "profile_exists", lambda _n: False, raising=False
    )

    main_mod.cmd_profile(_args("alias", "demo", remove=True))

    assert not (wrapper_dir / "demo").exists()
    out = capsys.readouterr().out
    assert "Removed alias 'demo'" in out


def test_remove_refuses_non_wrapper_file_in_wrapper_dir(wrapper_dir, monkeypatch, capsys):
    """Fail-closed: a same-named file without the `hermes -p` marker is NOT
    unlinked — skipping the profile check must not turn --remove into an
    arbitrary-file-delete primitive."""
    import hermes_cli.profiles as profiles_mod

    innocent = wrapper_dir / "demo"
    innocent.write_text("#!/bin/sh\necho not-a-hermes-wrapper\n")
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda _n: False)
    monkeypatch.setattr(
        main_mod, "profile_exists", lambda _n: False, raising=False
    )

    main_mod.cmd_profile(_args("alias", "demo", remove=True))

    assert innocent.exists()
    out = capsys.readouterr().out
    assert "No alias 'demo' found to remove." in out
    # The dead-end message points at the tool that surfaces these states.
    assert "hermes doctor" in out


def test_add_still_requires_existing_profile(wrapper_dir, monkeypatch, capsys):
    """The create path keeps its precondition — the orphan carve-out is
    removal-only."""
    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "profile_exists", lambda _n: False)
    monkeypatch.setattr(
        main_mod, "profile_exists", lambda _n: False, raising=False
    )

    with pytest.raises(SystemExit):
        main_mod.cmd_profile(_args("alias", "ghost", remove=False))
    out = capsys.readouterr().out
    assert "does not exist" in out
