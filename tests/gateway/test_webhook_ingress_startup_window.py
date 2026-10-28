"""Webhook producer admission in the startup/restart window.

The webhook port starts serving before the runner wires a profile's transport home, so a
delivery that lands in that window targets a profile this gateway serves but has not
registered yet. It must be RETRYABLE (runtime_draining -> 503 + Retry-After) so the
provider redelivers, never a permanent profile_mismatch that drops the event. The
provider does not redeliver a 503 (the delivery id never comes back), so a permanent
refusal here loses events for good.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hermes_state_runtime import RuntimeStoreError
from gateway.platforms.webhook_ingress import producer_scope


def _runner(
    *,
    registered: bool = True,
    profile=None,
    home=None,
    authority=None,
    parked: bool = False,
    resolver=None,
):
    """Stub of the GatewayRunner surface producer_scope touches."""
    runner = SimpleNamespace(
        session_authority=SimpleNamespace(db=SimpleNamespace(db_path="/other/db")),
        session_authorities=SimpleNamespace(for_home=lambda h: authority),
        _native_transport_homes={} if home is None else {profile: home},
        config=SimpleNamespace(multiplex_profiles=False),
        _routed_profile_home=resolver,
    )

    def _owning_profile(adapter, platform):
        return (registered, profile)

    runner._owning_profile = _owning_profile

    def _admit_primary_source(source, home_value):
        # A primary source is served at its launch home.
        return home_value

    runner._admit_primary_source = _admit_primary_source

    def _canonicalize(source, transport_profile=None):
        pass

    runner._canonicalize = _canonicalize

    def _resolve_profile_home_for_source(source):
        return "/served/home"

    runner._resolve_profile_home_for_source = _resolve_profile_home_for_source

    if parked:
        runner.session_runtime_descriptor = {"parked_profiles": {profile or "p": {}}}
    return runner


def _adapter(runner):
    return SimpleNamespace(_message_handler=SimpleNamespace(__self__=runner),
                           platform="webhook")


def _event(profile=None):
    return SimpleNamespace(source=SimpleNamespace(profile=profile))


@pytest.mark.asyncio
async def test_startup_window_unwired_primary_is_retryable(monkeypatch, tmp_path):
    """Primary adapter connected but its transport home not yet published: the delivery is
    ours and must be deferred (runtime_draining), not refused for good."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = _runner(registered=True, profile=None, home=None, authority=None)
    exc = None
    try:
        async with producer_scope(_adapter(runner), _event()):
            pass
    except RuntimeStoreError as exc_info:
        exc = exc_info
    assert isinstance(exc, RuntimeStoreError)
    assert exc.reason == "runtime_draining"


@pytest.mark.asyncio
async def test_unregistered_adapter_is_permanent_refusal():
    """An adapter this gateway does not own at all: profile_mismatch is final."""
    runner = _runner(registered=False, profile=None, home=None, authority=None)
    with pytest.raises(RuntimeStoreError) as exc:
        async with producer_scope(_adapter(runner), _event()):
            pass
    assert exc.value.reason == "profile_mismatch"


@pytest.mark.asyncio
async def test_unresolvable_named_profile_is_permanent_refusal():
    """A named profile whose home cannot be resolved (deleted mid-run): permanent refusal."""
    runner = _runner(registered=True, profile="ghost", home=None, authority=None,
                     resolver=None)
    with pytest.raises(RuntimeStoreError) as exc:
        async with producer_scope(_adapter(runner), _event(profile="ghost")):
            pass
    assert exc.value.reason == "profile_mismatch"


@pytest.mark.asyncio
async def test_parked_profile_is_retryable():
    """A parked profile comes back on a rescan: ParkedProfileRefusal (runtime_draining)."""
    runner = _runner(registered=True, profile="p", home=None, authority=None, parked=True)
    with pytest.raises(RuntimeStoreError) as exc:
        async with producer_scope(_adapter(runner), _event(profile="p")):
            pass
    assert exc.value.reason == "runtime_draining"


@pytest.mark.asyncio
async def test_wired_profile_is_admitted(monkeypatch):
    """Fully wired profile (home + authority published): producer_scope yields the authority,
    no refusal at all."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def null_scope(*_args, **_kwargs):
        yield

    import gateway.run as run_mod

    monkeypatch.setattr(run_mod, "_async_profile_runtime_scope", null_scope)

    authority = SimpleNamespace()
    runner = _runner(registered=True, profile="p", home="/served/home", authority=authority)
    async with producer_scope(_adapter(runner), _event(profile="p")) as got:
        assert got is authority
