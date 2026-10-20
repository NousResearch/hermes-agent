"""The runtime scope (``hermes_scope`` + ``agent/runtime_scope.py``) is one value.

Inside a bound scope, home, secrets and terminal policy always describe the same profile. The
legacy setters cannot split them: on a local install a call naming another profile moves the
whole scope and its reset moves it back; in a hosted scope the call refuses. Driven against real
temp profile homes, with no mocks of the scope machinery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state, bind_turn, scope_for_profile
from agent.secret_scope import (
    build_profile_secret_scope,
    current_secret_scope,
    current_secret_scope_home,
    reset_secret_scope,
    set_secret_scope,
)
from hermes_constants import get_hermes_home, hermes_home_key, reset_hermes_home_override, set_hermes_home_override
from hermes_scope import (
    LOCAL_OWNER,
    Principal,
    Scope,
    ScopeSwitchRefused,
    System,
    TurnScope,
    current_scope,
)
from tools.terminal_scope import get_terminal_scope, install_profile_terminal_scope, reset_terminal_scope


def _profile(root: Path, name: str, backend: str) -> Path:
    home = root / "profiles" / name
    home.mkdir(parents=True)
    (home / ".env").write_text(f"PROFILE_KEY=key-{name}\n", encoding="utf-8")
    (home / "config.yaml").write_text(f"terminal:\n  backend: {backend}\n", encoding="utf-8")
    return home


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    return _profile(tmp_path, "a", "local"), _profile(tmp_path, "b", "docker")


def _state() -> tuple[str, str | None, str | None, str]:
    """(home, secret-scope home, PROFILE_KEY, terminal backend) as readers see them now."""
    scope = get_terminal_scope() or {}
    return (hermes_home_key(get_hermes_home()), current_secret_scope_home(),
            (current_secret_scope() or {}).get("PROFILE_KEY"), scope.get("TERMINAL_ENV", ""))


def _expect(home: Path, name: str, backend: str) -> tuple[str, str, str, str]:
    return hermes_home_key(home), str(home), f"key-{name}", backend


# --- value types ---

def test_scope_requires_all_three_parts():
    Scope("local", "/h", LOCAL_OWNER)
    with pytest.raises(ValueError):
        Scope("", "/h", LOCAL_OWNER)
    with pytest.raises(ValueError):
        Scope("local", "", LOCAL_OWNER)
    with pytest.raises(TypeError):
        Scope("local", "/h", "owner")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TurnScope("local", "/h", System("boot"), "", "t1")
    assert Scope("local", "/h", LOCAL_OWNER).home() == "/h"
    assert Scope("acme", "agent-7", Principal("u1")).home() is None


# --- binding ---

def test_bind_profile_scope_binds_home_secrets_and_terminal_together(two_profiles):
    a, _b = two_profiles
    before = _state()
    with bind_profile_scope(a) as scope:
        assert current_scope() == scope == Scope("local", str(a), LOCAL_OWNER)
        assert _state() == _expect(a, "a", "local")
    assert _state() == before
    assert current_scope() is None


def test_bind_turn_keeps_the_scope_and_adds_session_and_turn(two_profiles):
    a, _b = two_profiles
    with bind_profile_scope(a):
        with bind_turn("s1", "adm-1") as turn:
            assert turn == TurnScope("local", str(a), LOCAL_OWNER, "s1", "adm-1")
            # A nested bind of the same profile (the gateway re-enters it inside a turn) keeps the turn.
            assert scope_for_profile(a) == turn
        assert type(current_scope()) is Scope


def test_bind_turn_without_a_scope_binds_nothing():
    with bind_turn("s1", "adm-1") as turn:
        assert turn is None and current_scope() is None


# --- legacy setters cannot split a local scope ---

@pytest.mark.parametrize("setter", ["home", "secret", "terminal"])
def test_each_legacy_setter_for_another_profile_moves_the_whole_scope(two_profiles, setter):
    a, b = two_profiles
    with bind_profile_scope(a):
        assert _state() == _expect(a, "a", "local")
        if setter == "home":
            token, reset = set_hermes_home_override(str(b)), reset_hermes_home_override
        elif setter == "secret":
            token, reset = set_secret_scope(build_profile_secret_scope(b), profile_home=str(b)), reset_secret_scope
        else:
            token, reset = install_profile_terminal_scope(b), reset_terminal_scope
        assert _state() == _expect(b, "b", "docker"), "the scope split"
        assert current_scope() == Scope("local", str(b), LOCAL_OWNER)
        reset(token)
        assert _state() == _expect(a, "a", "local")
        assert current_scope() == Scope("local", str(a), LOCAL_OWNER)


def test_legacy_setter_for_the_bound_profile_passes_through(two_profiles):
    a, _b = two_profiles
    with bind_profile_scope(a):
        bound = current_scope()
        token = set_hermes_home_override(str(a))
        assert current_scope() is bound
        reset_hermes_home_override(token)
        assert current_scope() is bound


def test_legacy_setter_keeps_the_actor(two_profiles):
    a, b = two_profiles
    with bind_state(Scope("local", str(a), Principal("alice")), hydrate_secrets=False):
        token = set_hermes_home_override(str(b))
        assert current_scope() == Scope("local", str(b), Principal("alice"))
        reset_hermes_home_override(token)


def test_legacy_setters_without_a_scope_behave_as_before(two_profiles):
    a, _b = two_profiles
    token = set_hermes_home_override(str(a))
    assert hermes_home_key(get_hermes_home()) == hermes_home_key(a)
    assert current_scope() is None and current_secret_scope() is None
    reset_hermes_home_override(token)


# --- hosted scopes refuse ---

def test_legacy_setters_refuse_in_a_hosted_scope(two_profiles):
    _a, b = two_profiles
    with bind_state(Scope("acme", "agent-7", Principal("u1"))):
        for call in (lambda: set_hermes_home_override(str(b)),
                     lambda: set_hermes_home_override(None),
                     lambda: set_secret_scope({"K": "v"}, profile_home=str(b)),
                     lambda: install_profile_terminal_scope(b)):
            with pytest.raises(ScopeSwitchRefused):
                call()
        assert current_scope() == Scope("acme", "agent-7", Principal("u1"))
        # Clearing the secret scope is not a switch.
        reset_secret_scope(set_secret_scope(None))


def test_hosted_scope_binds_no_directory_state():
    with bind_state(Scope("acme", "agent-7", Principal("u1"))):
        assert current_secret_scope() is None and get_terminal_scope() is None
