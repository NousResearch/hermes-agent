"""The per-home author key: ``kanban.review_profile`` names this home's board writes.

A home declares, in its own ``config.yaml``, the profile its own board writes are attributed
to. That declaration travels with the home, so a child process whose env carries another
profile's ``HERMES_PROFILE`` — or none at all — still writes as the profile that owns the home,
instead of landing as ``default``/``user``/a stale sticky profile.

Three things stay true, and are asserted here because each is a way the key could go wrong:
unset keeps the historical derivation exactly (``HERMES_PROFILE_NAME`` -> ``HERMES_PROFILE``,
else a loud refusal — never a guessed profile); an author bound for the call outranks the key, so
a multiplexed gateway keeps attributing to the routed chat's profile; and a blank value is unset,
not an author named ``""``.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def home(monkeypatch):
    """A fresh HERMES_HOME with no ``config.yaml`` until a test writes one."""
    test_home = Path(tempfile.mkdtemp(prefix="kanban_author_home_"))
    monkeypatch.setenv("HERMES_HOME", str(test_home))
    # The config loader caches per home, so drop the modules holding that cache and the
    # HERMES_HOME-derived constants before importing the contract fresh.
    for mod in list(sys.modules):
        if mod.startswith("hermes_cli") or mod == "hermes_constants":
            del sys.modules[mod]
    return test_home


def _set_home_author(home: Path, value: str) -> None:
    home.joinpath("config.yaml").write_text(
        f'kanban:\n  review_profile: "{value}"\n', encoding="utf-8",
    )


def test_home_key_names_the_author_over_the_ambient_env(home, monkeypatch):
    """The defect this closes: the env said another profile, so the write landed mis-attributed."""
    from hermes_cli.kanban_author import resolve_author

    _set_home_author(home, "home-profile")
    monkeypatch.setenv("HERMES_PROFILE", "env-profile")
    monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)

    assert resolve_author() == "home-profile"


def test_unset_key_keeps_the_existing_env_derivation(home, monkeypatch):
    from hermes_cli.kanban_author import resolve_author

    monkeypatch.setenv("HERMES_PROFILE", "env-profile")
    monkeypatch.setenv("HERMES_PROFILE_NAME", "env-profile-name")
    assert resolve_author() == "env-profile-name"

    monkeypatch.delenv("HERMES_PROFILE_NAME")
    assert resolve_author() == "env-profile"


def test_unset_key_with_no_env_still_refuses_rather_than_guessing(home, monkeypatch):
    from hermes_cli.kanban_author import KanbanAuthorRequired, resolve_author

    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)
    with pytest.raises(KanbanAuthorRequired):
        resolve_author()


def test_bound_author_outranks_the_home_key(home):
    """A routed chat keeps its own author: one home-wide name must not swallow it."""
    from hermes_cli.kanban_author import bind_author, resolve_author

    _set_home_author(home, "home-profile")
    with bind_author("routed-profile"):
        assert resolve_author() == "routed-profile"


def test_blank_key_is_unset_not_an_author(home, monkeypatch):
    from hermes_cli.kanban_author import resolve_author

    _set_home_author(home, "   ")
    monkeypatch.setenv("HERMES_PROFILE", "env-profile")

    assert resolve_author() == "env-profile"
