"""The launch profile is a tenant: a body with no routed profile binds ITS scope, not ambient env.

``GatewayAdapterLifecycleMixin._scope_or_null`` used to return ``contextlib.nullcontext()`` whenever
the profile home was ``None`` — the launch profile's own handoff reclaims, reconnect attention
flags and platform events therefore ran completely unscoped on a multiplexing host: a legitimate
launch-profile ``get_secret`` failed closed, and anything reading process env picked up whatever a
secondary context had left there.
"""
import pytest

from agent import secret_scope
from agent.secret_scope import get_secret
from gateway.run_adapters import GatewayAdapterLifecycleMixin
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
