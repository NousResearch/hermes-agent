"""Forced-skill validation binds the ASSIGNEE's profile scope, and gives it back.

One gateway process serves many profiles, so a validator that reads the launch
profile's home (module globals, ``os.environ``) silently approves whatever the
LAUNCH profile has installed. Proven live against two real homes in A -> B -> A
order under multiplex — a single temp ``HERMES_HOME`` cannot show the leak.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.hermes_cli.test_kanban_forced_skills import install_skill


@pytest.fixture()
def two_homes(tmp_path, monkeypatch):
    """Two profile homes with disjoint skills, multiplex on, launch home = neither."""
    from agent.secret_scope import set_multiplex_active

    root = tmp_path / ".hermes"
    home_a = root / "profiles" / "alpha"
    home_b = root / "profiles" / "beta"
    for home in (home_a, home_b):
        home.mkdir(parents=True)
        (home / ".env").write_text("HERMES_TEST_MARKER=x\n", encoding="utf-8")
    install_skill(home_a, "writing", "only-in-alpha")
    install_skill(home_b, "research", "only-in-beta")
    # The launch profile has BOTH names — if scope leaks, every check passes.
    install_skill(root, "writing", "only-in-alpha")
    install_skill(root, "research", "only-in-beta")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    set_multiplex_active(True)
    try:
        yield root
    finally:
        set_multiplex_active(False)


def _bad(assignee: str, name: str) -> bool:
    from hermes_cli import kanban_db_skills as kbs
    return bool(kbs.check_forced_skills([name], assignee=assignee))


def test_each_profile_sees_only_its_own_skills(two_homes):
    """A -> B -> A: the third call must not be answered from B's (or the launch
    profile's) skills."""
    assert not _bad("alpha", "only-in-alpha")
    assert _bad("alpha", "only-in-beta")

    assert not _bad("beta", "only-in-beta")
    assert _bad("beta", "only-in-alpha")

    # Back to A — a cache keyed on anything but the home would answer from B here.
    assert not _bad("alpha", "only-in-alpha")
    assert _bad("alpha", "only-in-beta")


def test_scope_is_released_after_validation(two_homes):
    """The bound home / secret scope must not survive the call: the next thing the
    gateway does in this context is another profile's turn."""
    from agent.secret_scope import current_secret_scope
    from hermes_constants import get_hermes_home, get_hermes_home_override

    before_home = get_hermes_home()
    assert get_hermes_home_override() is None
    assert current_secret_scope() is None

    _bad("alpha", "only-in-beta")

    assert get_hermes_home_override() is None
    assert current_secret_scope() is None
    assert get_hermes_home() == before_home


@pytest.mark.parametrize("blast_radius", ["scan", "secret_scope"])
def test_a_failing_validation_still_releases_scope(two_homes, monkeypatch, blast_radius):
    """Scope teardown is in a ``finally``: nothing that raises between binding the
    home and returning may strand the gateway context on the assignee's home.

    ``secret_scope`` covers the window BEFORE the scan — reading the assignee's
    ``.env`` is itself failure-prone, and it happens after the home is already bound.
    """
    import agent.secret_scope as secret_scope
    from hermes_cli import kanban_db_skills as kbs
    from hermes_constants import get_hermes_home_override

    def _boom(*_args, **_kwargs):
        raise RuntimeError(f"{blast_radius} exploded")

    if blast_radius == "scan":
        monkeypatch.setattr(kbs, "_installed_skill_index", _boom)
    else:
        monkeypatch.setattr(secret_scope, "build_profile_secret_scope", _boom)

    # Fail open rather than refusing a card because the check itself broke.
    assert kbs.check_forced_skills(["only-in-beta"], assignee="alpha") == []
    assert get_hermes_home_override() is None
    assert secret_scope.current_secret_scope() is None
