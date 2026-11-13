"""The runtime scope (``hermes_scope`` + ``agent/runtime_scope.py``) is one value.

Inside a bound scope, home, secrets and terminal policy always describe the same profile. The
legacy setters cannot split them: on a local install a call naming another profile moves the
whole scope and its reset moves it back; in a hosted scope the call refuses. Driven against real
temp profile homes, with no mocks of the scope machinery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state, bind_turn, claim_fence, scope_for_profile
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
    Fence,
    Group,
    Principal,
    Scope,
    ScopeSwitchRefused,
    System,
    TurnScope,
    current_scope,
)
from tools.terminal_scope import (
    build_profile_terminal_scope,
    get_terminal_scope,
    install_profile_terminal_scope,
    reset_terminal_scope,
    set_terminal_scope,
)


F1 = Fence(1, 1)


def _hosted(tenant: str = "acme", agent: str = "agent-7", who: str = "u1") -> Scope:
    """A hosted scope on a personal agent: the person acting is its owner."""
    return Scope(tenant, agent, Principal(who), owner=Principal(who), can_administer=True)


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
        TurnScope("local", "/h", System("boot"), "", "t1", F1)
    assert Scope("local", "/h", LOCAL_OWNER).home() == "/h"
    assert _hosted().home() is None


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
        with bind_turn("s1", "adm-1", F1) as turn:
            assert turn == TurnScope("local", str(a), LOCAL_OWNER, "s1", "adm-1", F1)
            # A nested bind of the same profile (the gateway re-enters it inside a turn) keeps the turn.
            assert scope_for_profile(a) == turn
        assert type(current_scope()) is Scope


def test_bind_turn_without_a_scope_binds_nothing():
    with bind_turn("s1", "adm-1", F1) as turn:
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
    with bind_state(_hosted()):
        for call in (lambda: set_hermes_home_override(str(b)),
                     lambda: set_hermes_home_override(None),
                     lambda: set_secret_scope({"K": "v"}, profile_home=str(b)),
                     lambda: install_profile_terminal_scope(b)):
            with pytest.raises(ScopeSwitchRefused):
                call()
        assert current_scope() == _hosted()


def test_hosted_scope_binds_no_directory_state():
    with bind_state(_hosted()):
        assert current_secret_scope() is None and get_terminal_scope() is None


def test_a_local_bind_refuses_inside_a_hosted_scope(two_profiles):
    _a, b = two_profiles
    hosted = _hosted()
    with bind_state(hosted):
        with pytest.raises(ScopeSwitchRefused):
            with bind_profile_scope(b):
                pass
        with pytest.raises(ScopeSwitchRefused):
            bind_state(Scope("local", str(b), LOCAL_OWNER))
        assert current_scope() == hosted and get_terminal_scope() is None


# --- no public path splits a bound scope ---

def test_the_bound_profiles_own_frozen_policy_replaces_its_terminal_policy(two_profiles):
    """A session's frozen policy, declared as the bound profile's, replaces the scope's policy for the
    turn; the scope keeps its home and secrets."""
    a, _b = two_profiles
    with bind_profile_scope(a):
        token = set_terminal_scope({"TERMINAL_ENV": "ssh"}, profile_home=a)
        assert _state() == (*_expect(a, "a", "local")[:3], "ssh")
        assert current_scope() == Scope("local", str(a), LOCAL_OWNER)
        reset_terminal_scope(token)
        assert _state() == _expect(a, "a", "local")


@pytest.mark.parametrize("owner", ["other", "undeclared"])
def test_a_terminal_policy_not_owned_by_the_bound_profile_refuses(two_profiles, owner):
    """The reviewer's raw-policy case: B's policy inside A's scope would run A's turn with B's backend."""
    a, b = two_profiles
    with bind_profile_scope(a):
        with pytest.raises(ScopeSwitchRefused):
            set_terminal_scope(build_profile_terminal_scope(b), **({"profile_home": b} if owner == "other" else {}))
        assert _state() == _expect(a, "a", "local")


@pytest.mark.parametrize("clear", ["secrets", "terminal"])
def test_clearing_one_piece_of_a_bound_scope_refuses(two_profiles, clear):
    a, _b = two_profiles
    with bind_profile_scope(a):
        with pytest.raises(ScopeSwitchRefused):
            set_secret_scope(None) if clear == "secrets" else set_terminal_scope(None, profile_home=a)
        assert _state() == _expect(a, "a", "local")


def test_a_raw_terminal_mapping_refuses_inside_a_hosted_scope():
    with bind_state(_hosted()):
        with pytest.raises(ScopeSwitchRefused):
            set_terminal_scope({"TERMINAL_ENV": "local"}, profile_home="agent-7")
        assert get_terminal_scope() is None


def test_a_hosted_scope_cannot_bind_another_hosted_agent_or_tenant():
    hosted = _hosted(agent="one")
    with bind_state(hosted):
        for other in (_hosted(agent="two"), _hosted(tenant="other", agent="one")):
            with pytest.raises(ScopeSwitchRefused):
                bind_state(other)
        with bind_turn("s1", "t1", F1):  # a turn of the same agent is not a switch
            assert current_scope() == _hosted(agent="one").for_turn("s1", "t1", F1)
        assert current_scope() == hosted


def test_a_partial_bind_never_inherits_a_piece_from_an_earlier_change(two_profiles):
    """The reviewer's replay case: a bind that keeps the current home override copies it, so removing
    the earlier override it came from cannot take a piece out from under it."""
    a, _b = two_profiles
    outer = set_hermes_home_override(str(a))
    with bind_state(scope_for_profile(a), hydrate_secrets=False, home_override=False):
        before = _state()
        reset_hermes_home_override(outer)
        assert _state() == before
        assert current_scope() == Scope("local", str(a), LOCAL_OWNER)


def test_a_cross_profile_terminal_switch_keeps_the_callers_overlay(two_profiles):
    a, b = two_profiles
    with bind_profile_scope(a):
        with install_profile_terminal_scope(b, env_overlay={"TERMINAL_SSH_HOST": "custom-host"}):
            assert get_terminal_scope()["TERMINAL_SSH_HOST"] == "custom-host"
            assert current_scope() == Scope("local", str(b), LOCAL_OWNER)


def test_setters_whose_tokens_are_discarded_do_not_grow_the_state(two_profiles):
    import hermes_scope
    a, _b = two_profiles

    def depth() -> int:
        node, n = hermes_scope._STATE.get(), 0
        while node is not None:
            node, n = node.parent, n + 1
        return n

    def run() -> int:
        for _ in range(500):
            set_hermes_home_override(str(a))
        return depth()

    import contextvars
    assert contextvars.copy_context().run(run) <= 2  # the newest change folds on the next push


# --- resets always land on a whole state ---

def test_an_out_of_order_reset_keeps_the_later_whole_switch(two_profiles):
    """Resetting an earlier same-profile token after a switch to B removes only that token's own
    change: B's home, secrets and terminal policy stay together."""
    a, b = two_profiles
    with bind_profile_scope(a):
        same = set_hermes_home_override(str(a))
        other = set_hermes_home_override(str(b))
        reset_hermes_home_override(same)
        assert _state() == _expect(b, "b", "docker")
        assert current_scope() == Scope("local", str(b), LOCAL_OWNER)
        reset_hermes_home_override(other)
        assert _state() == _expect(a, "a", "local")


def test_resetting_an_outer_override_from_inside_a_scope_keeps_the_scope_whole(two_profiles):
    """Removing an override set before the scope was bound drops only that override: the scope
    bound on top of it stays bound, whole."""
    a, b = two_profiles
    before = _state()
    outer = set_hermes_home_override(str(b))
    with bind_profile_scope(a):
        reset_hermes_home_override(outer)
        assert _state() == _expect(a, "a", "local")
    assert _state() == before


def test_resetting_an_outer_bind_while_an_inner_one_is_bound_unwinds_both(two_profiles):
    """The reviewer's resurrection case: resetting outer A while inner B is bound, then leaving B,
    must not bring A back."""
    a, b = two_profiles
    before = _state()
    outer = bind_state(Scope("local", str(a), LOCAL_OWNER), hydrate_secrets=False)
    with bind_profile_scope(b):
        outer.reset()
        assert current_scope() is None and _state() == before
    assert current_scope() is None and _state() == before


def test_a_reset_in_a_copied_context_leaves_the_original_context_bound_and_releasable(two_profiles):
    import contextvars
    a, _b = two_profiles
    before = _state()
    token = bind_state(Scope("local", str(a), LOCAL_OWNER), hydrate_secrets=False)
    contextvars.copy_context().run(token.reset)
    assert _state() == _expect(a, "a", "local")  # the copy's reset changed only the copy
    token.reset()
    assert _state() == before and current_scope() is None


def test_resets_are_idempotent(two_profiles):
    a, _b = two_profiles
    before = _state()
    token = bind_state(Scope("local", str(a), LOCAL_OWNER), hydrate_secrets=False)
    token.reset()
    token.reset()
    assert _state() == before


def test_a_reset_cannot_leave_a_hosted_agent_from_inside_it(two_profiles):
    """Inside a hosted activity, resetting the bind it was entered from would drop the hosted scope
    (state made inside a scope goes with it); that refuses. Only the bind that entered it can."""
    a, _b = two_profiles
    local = bind_state(Scope("local", str(a), LOCAL_OWNER), hydrate_secrets=False)
    hosted = _hosted()
    with bind_state(hosted):
        with pytest.raises(ScopeSwitchRefused):
            local.reset()
        assert current_scope() == hosted
    assert current_scope() == Scope("local", str(a), LOCAL_OWNER)
    local.reset()
    assert current_scope() is None


def test_a_mirrored_hermes_home_is_not_mistaken_for_the_launch_profile(tmp_path, monkeypatch):
    """An embedding host pins the launch home and mirrors the served profile into ``HERMES_HOME``:
    binding that profile must give it its own secrets, never the launch profile's env-only keys."""
    from hermes_constants import pin_process_hermes_home
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting
    launch, served = tmp_path / "launch", _profile(tmp_path, "b", "local")
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("OPENAI_API_KEY", "launch-only-key")
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", False)
    pin_process_hermes_home(launch)
    try:
        activate_multi_profile_hosting()
        monkeypatch.setenv("HERMES_HOME", str(served))
        with bind_profile_scope(served, hydrate_secrets=False):
            assert (current_secret_scope() or {}).get("OPENAI_API_KEY") is None
            assert current_secret_scope_home() == str(served)
    finally:
        pin_process_hermes_home(None)


def test_home_reads_during_another_threads_first_scope_import_see_no_override(monkeypatch):
    """``hermes_scope`` is in ``sys.modules`` before its body has run; a home read in that window is
    an unbound read, not an AttributeError."""
    import sys
    import types
    from hermes_constants import get_hermes_home_override
    monkeypatch.setitem(sys.modules, "hermes_scope", types.ModuleType("hermes_scope"))
    assert get_hermes_home_override() is None


def test_a_reassigned_token_variable_does_not_grow_the_state(two_profiles):
    import contextvars
    import hermes_scope
    a, _b = two_profiles

    def run() -> int:
        token = None
        for _ in range(500):
            token = set_hermes_home_override(str(a))
        node, depth = hermes_scope._STATE.get(), 0
        while node is not None:
            node, depth = node.parent, depth + 1
        del token
        return depth

    assert contextvars.copy_context().run(run) <= 3


def test_legacy_switch_builds_the_target_policy_on_first_read_only(two_profiles, monkeypatch):
    """A legacy setter naming another profile moves the whole scope there, but builds that profile's
    terminal policy on its first read: a caller that only redirects file paths (a credential reader
    inside the profile's first config load) never reads the profile's config. The policy, once
    read, is the same one an eager bind builds."""
    from tools import terminal_scope

    a, b = two_profiles
    built = []
    real = terminal_scope.prepare_profile_terminal_scope
    monkeypatch.setattr(terminal_scope, "prepare_profile_terminal_scope",
                        lambda home, **kw: built.append(Path(home)) or real(home, **kw))
    with bind_profile_scope(a):
        built.clear()
        token = set_hermes_home_override(str(b))
        try:
            assert current_scope() == Scope("local", str(b), LOCAL_OWNER)
            assert built == [], "the switch built the target's terminal policy before anything read it"
            assert _state() == _expect(b, "b", "docker")
            get_terminal_scope()
            assert built == [b], "the policy is built once, on its first read"
        finally:
            reset_hermes_home_override(token)
        assert _state() == _expect(a, "a", "local")


def test_a_bind_hydrates_the_target_profile_under_its_own_home(two_profiles, monkeypatch):
    """Secret sources are hydrated from the TARGET profile's ``secrets:`` config: hydration reads it
    through ``get_hermes_home()``, so the bind moves the home first, as main's per-turn loader did.
    Entering profile B from inside A must never hydrate B from A's configuration."""
    from hermes_cli import env_loader

    a, b = two_profiles
    (a / "config.yaml").write_text("secrets:\n  command:\n    enabled: false\nterminal:\n  backend: local\n")
    (b / "config.yaml").write_text("secrets:\n  command:\n    enabled: true\n    command: b-vault\n"
                                   "terminal:\n  backend: docker\n")
    seen = []
    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources",
                        lambda home: seen.append((Path(home), hermes_home_key(get_hermes_home()),
                                                  env_loader._load_secrets_config(Path(home))["command"]["enabled"])))
    with bind_profile_scope(a, hydrate_secrets=False):
        with bind_profile_scope(b):
            pass
        assert hermes_home_key(get_hermes_home()) == hermes_home_key(a)
    assert seen == [(b, hermes_home_key(b), True)]


def test_a_deferred_policy_keeps_the_overlay_it_was_bound_with(two_profiles):
    """The terminal policy a legacy switch defers is built from the overlay passed at the switch; a
    caller changing its own mapping afterwards changes nothing."""
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope

    a, b = two_profiles
    (b / "config.yaml").write_text("terminal: {}\n")
    overlay = {"TERMINAL_ENV": "docker"}
    with bind_profile_scope(a, hydrate_secrets=False):
        token = install_profile_terminal_scope(b, env_overlay=overlay)
        try:
            overlay["TERMINAL_ENV"] = "local"
            assert get_terminal_scope()["TERMINAL_ENV"] == "docker"
        finally:
            reset_terminal_scope(token)


def test_a_managed_reference_in_the_terminal_policy_resolves_against_the_target_profile(
        two_profiles, tmp_path, monkeypatch):
    """A managed ``${VAR}`` in ``terminal:`` reads the profile being bound, as a standalone load of
    that profile reads its own environment: entering B from inside A's bind must not freeze A's
    value into B's policy, and an unbound entry must not take the launch environment's."""
    from hermes_cli import managed_scope

    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("terminal:\n  ssh_host: ${SSH_TARGET}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("SSH_TARGET", "launch-host")
    managed_scope.invalidate_managed_cache()
    a, b = two_profiles
    for home, name in ((a, "a"), (b, "b")):
        with (home / ".env").open("a", encoding="utf-8") as f:
            f.write(f"SSH_TARGET=target-{name}\n")
    try:
        with bind_profile_scope(b, hydrate_secrets=False):
            assert get_terminal_scope()["TERMINAL_SSH_HOST"] == "target-b"
        with bind_profile_scope(a, hydrate_secrets=False):
            assert get_terminal_scope()["TERMINAL_SSH_HOST"] == "target-a"
            with bind_profile_scope(b, hydrate_secrets=False):
                assert get_terminal_scope()["TERMINAL_SSH_HOST"] == "target-b"
    finally:
        managed_scope.invalidate_managed_cache()


# --- owner, administration and fence ---

def test_local_scope_defaults_owner_and_administration_to_the_local_person():
    assert Scope("local", "/h", LOCAL_OWNER).owner == LOCAL_OWNER
    assert Scope("local", "/h", Principal("alice")).owner == Principal("alice")
    boot = Scope("local", "/h", System("boot"))
    assert (boot.owner, boot.can_administer) == (LOCAL_OWNER, True)


def test_hosted_scope_must_name_owner_and_administration():
    with pytest.raises(ValueError, match="owner and whether it may administer"):
        Scope("acme", "agent-7", Principal("u1"))
    with pytest.raises(ValueError, match="owner and whether it may administer"):
        Scope("acme", "agent-7", Principal("u1"), owner=Principal("u1"))
    with pytest.raises(ValueError, match="owner and whether it may administer"):
        Scope("acme", "agent-7", Principal("u1"), can_administer=True)


def test_scope_values_are_checked():
    with pytest.raises(TypeError, match="can_administer must be a bool"):
        Scope("acme", "bot", Principal("m"), owner=Group("g"), can_administer=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="owner must be"):
        Scope("acme", "bot", Principal("m"), owner="support", can_administer=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Group("")


def test_a_shared_personal_agent_takes_a_teammate_who_cannot_administer_it():
    """A personal agent its owner shared: the teammate's turn runs on it, as the teammate, and
    may not change the agent; the owner's own turn may."""
    shared = Scope("acme", "x", Principal("bob"), owner=Principal("alice"), can_administer=False)
    assert (shared.actor, shared.owner, shared.can_administer) == (Principal("bob"), Principal("alice"), False)
    assert Scope("acme", "x", Principal("alice"), owner=Principal("alice"), can_administer=True).can_administer
    # Work no person started (a cron tick) still runs on the owner's agent.
    assert Scope("acme", "x", System("cron"), owner=Principal("alice"), can_administer=True).can_administer


def test_fence_is_two_positive_ints():
    for bad in ((0, 1), (1, 0), (True, 1), ("1", 1), (1, 1.0)):
        with pytest.raises(ValueError):
            Fence(*bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fence must be a Fence"):
        TurnScope("local", "/h", LOCAL_OWNER, "s1", "t1", (1, 1))  # type: ignore[arg-type]


def test_a_turn_keeps_owner_and_administration_and_carries_its_fence(two_profiles):
    a, _b = two_profiles
    member = Scope("acme", "bot", Principal("u"), owner=Group("support"), can_administer=False)
    with bind_state(member):
        with bind_turn("s1", "t1", Fence(4, 9)) as turn:
            assert (turn.owner, turn.can_administer, turn.fence) == (Group("support"), False, Fence(4, 9))
    with bind_profile_scope(a), bind_turn("s1", "t1", Fence(2, 3)) as turn:
        assert (turn.owner, turn.can_administer, turn.fence) == (LOCAL_OWNER, True, Fence(2, 3))
        # A turn re-marked inside a turn (a nested drain) carries the newer claim's fence.
        with bind_turn("s1", "t2", Fence(2, 4)) as inner:
            assert (inner.turn, inner.fence) == ("t2", Fence(2, 4))
        assert current_scope().fence == Fence(2, 3)


@pytest.mark.parametrize("change", [
    dict(actor=Principal("other")),
    dict(can_administer=True),
    dict(owner=Group("other-team")),
])
def test_hosted_scope_refuses_a_bind_that_changes_who_it_acts_for(change):
    from dataclasses import replace
    member = Scope("acme", "bot", Principal("u"), owner=Group("support"), can_administer=False)
    with bind_state(member):
        with pytest.raises(ScopeSwitchRefused):
            bind_state(replace(member, **change))
        assert current_scope() == member


def _hosted_turn() -> TurnScope:
    member = Scope("acme", "bot", Principal("u"), owner=Group("support"), can_administer=False)
    return member.for_turn("s1", "t1", Fence(2, 3))


def test_a_same_identity_bind_inside_a_hosted_turn_keeps_the_turn_and_its_fence():
    """Rebinding the same hosted scope inside a turn must not drop the turn's session, turn id or
    fence: writes in the body are checked against that fence."""
    turn = _hosted_turn()
    plain = Scope("acme", "bot", Principal("u"), owner=Group("support"), can_administer=False)
    with bind_state(turn):
        with bind_state(plain):
            inner = current_scope()
            assert isinstance(inner, TurnScope)
            assert (inner.session, inner.turn, inner.fence) == ("s1", "t1", Fence(2, 3))
        assert current_scope() == turn


def test_a_nested_turn_inside_a_hosted_turn_carries_the_newer_fence():
    with bind_state(_hosted_turn()):
        with bind_turn("s1", "t2", Fence(2, 4)) as inner:
            assert (inner.turn, inner.fence) == ("t2", Fence(2, 4))
            assert current_scope() == inner
        assert current_scope().fence == Fence(2, 3)


@pytest.mark.parametrize("change", [dict(actor=Principal("other")), dict(can_administer=True)])
def test_a_hosted_turn_still_refuses_a_bind_that_changes_who_it_acts_for(change):
    from dataclasses import replace
    turn = _hosted_turn()
    plain = Scope("acme", "bot", Principal("u"), owner=Group("support"), can_administer=False)
    with bind_state(turn):
        with pytest.raises(ScopeSwitchRefused):
            bind_state(replace(plain, **change))
        assert current_scope() == turn


def test_a_newer_turn_bound_inside_a_hosted_turn_is_bound_as_given():
    with bind_state(_hosted_turn()):
        newer = _hosted_turn().for_turn("s1", "t2", Fence(2, 4))
        with bind_state(newer):
            assert current_scope() == newer
        assert current_scope().fence == Fence(2, 3)


def test_a_local_turn_still_binds_another_profile(two_profiles):
    a, b = two_profiles
    with bind_profile_scope(a), bind_turn("s1", "t1", F1):
        with bind_profile_scope(b) as other:
            assert current_scope() == other == Scope("local", str(b), LOCAL_OWNER)
            assert _state() == _expect(b, "b", "docker")
        assert current_scope().fence == F1


def test_claim_fence_reads_the_claimed_admission(tmp_path):
    """The fence a turn binds is the one the claim stamped on its admission row."""
    from contextlib import closing

    import hermes_agent.hermes_state_runtime as rt
    from hermes_agent.hermes_state import SessionDB
    from tests.hermes_state.test_target_advance_fence import _local_session

    with closing(SessionDB(tmp_path / "state.db")) as db:
        rt.begin_runtime_epoch(db, instance_id="owner")
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")  # a restarted owner: epoch 2
        sid = _local_session(db, epoch, tmp_path)
        for n in range(3):
            rt.admit_session_input(db, epoch=epoch, principal_id="human", session_id=sid,
                                   request_id=f"r{n}", payload={"text": "hi"})
            row = rt.claim_session_input(db, epoch=epoch, session_id=sid)
            rt.settle_session_input(db, epoch=epoch, admission_id=row["admission_id"],
                                    generation=row["generation"], outcome="completed", result={})
        assert (epoch, row["generation"]) == (2, 3)
        assert claim_fence(row) == Fence(2, 3) == Fence(epoch, db.get_session(sid)["runtime_generation"])
