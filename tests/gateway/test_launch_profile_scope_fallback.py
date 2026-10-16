"""Who owns a body with no routed profile home — and who must NOT.

``GatewayAdapterLifecycleMixin._scope_or_null`` used to return ``contextlib.nullcontext()`` whenever
the profile home was ``None`` — the launch profile's own handoff reclaims, reconnect attention
flags and platform events therefore ran completely unscoped on a multiplexing host: a legitimate
launch-profile ``get_secret`` failed closed, and anything reading process env picked up whatever a
secondary context had left there.

``None`` must mean exactly one thing. A NAMED profile whose home no longer resolves answers the
:data:`UNRESOLVED_PROFILE_HOME` sentinel instead, and binds nothing: handing it the launch
profile's scope would serve a secondary's inbound message with the LAUNCH profile's credentials.
"""
from types import SimpleNamespace

import pytest

from agent import secret_scope
from agent.secret_scope import UnscopedSecretError, get_secret
from gateway.run_adapters import UNRESOLVED_PROFILE_HOME, GatewayAdapterLifecycleMixin
from tui_gateway import launch_profile_policy

POISON = "LAUNCHSCOPE_TEST_KEY"


@pytest.fixture
def multiplexing_host(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / ".env").write_text(f"{POISON}=launch-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    launch_profile_policy.activate_multi_profile_hosting()
    # A secondary's context poisons the process env AFTER activation; the frozen snapshot must win.
    monkeypatch.setenv(POISON, "secondary-poison")
    return launch


def test_scope_or_null_binds_launch_profile_when_no_routed_home(multiplexing_host):
    with GatewayAdapterLifecycleMixin._scope_or_null(lambda home: None, None):
        # Base: nullcontext -> UnscopedSecretError on a legitimate launch-profile read.
        assert get_secret(POISON) == "launch-dotenv"


@pytest.mark.asyncio
async def test_async_scope_or_null_binds_launch_profile_when_no_routed_home(multiplexing_host):
    async with GatewayAdapterLifecycleMixin._async_scope_or_null(lambda home: None, None):
        assert get_secret(POISON) == "launch-dotenv"


def test_single_profile_host_keeps_ambient_precedence(monkeypatch):
    """Never activated -> no binding at all, so ``os.environ`` precedence is byte-identical."""
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setenv(POISON, "ambient")
    with GatewayAdapterLifecycleMixin._scope_or_null(lambda home: None, None):
        assert get_secret(POISON) == "ambient"


class _Runner(GatewayAdapterLifecycleMixin):
    """Just enough runner for the production handler factory."""

    def __init__(self, seen):
        self.seen = seen

    async def _handle_message(self, _event):
        try:
            self.seen.append(get_secret(POISON))
        except UnscopedSecretError:
            self.seen.append("<fail-closed>")


@pytest.mark.asyncio
async def test_named_secondary_with_unresolvable_home_cannot_read_the_launch_secret(
        multiplexing_host, monkeypatch):
    """The invariant: a profile never borrows another profile's credential value.

    Base resolved the missing home to ``None`` and the handler ran under the LAUNCH profile's
    scope, so an inbound message on profile B's own bot read ``launch-dotenv``.
    """
    from hermes_cli import profiles as profiles_mod

    def _gone(_name):
        raise FileNotFoundError("profile deleted mid-run")

    monkeypatch.setattr(profiles_mod, "get_profile_dir", _gone)

    seen: list[str] = []
    handler = _Runner(seen)._make_profile_message_handler("b")
    await handler(SimpleNamespace(source=None))

    assert seen == ["<fail-closed>"], f"secondary's handler resolved a credential to {seen}"


def test_unresolvable_home_is_a_sentinel_not_none(multiplexing_host, monkeypatch):
    """Overloading ``None`` is what made the fallback silent; keep the two answers distinct."""
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda _n: (_ for _ in ()).throw(OSError()))
    assert GatewayAdapterLifecycleMixin._routed_profile_home("b") is UNRESOLVED_PROFILE_HOME
