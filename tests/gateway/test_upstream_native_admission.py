"""A fresh native input from an adapter that declares ``authorization_is_upstream`` is
admitted without a principal allowlist (#139199).

The unified gateway's durable admission re-checks authorization inside
``session_envelope._validate_native`` with ``allow_adapter_delegation`` derived from the
relay seal / role reauthorization only. A direct adapter whose entire authorization model
IS the upstream (A2A peers: bearer-token authenticated in ``do_POST``) has no relay seal
and no roles, so the re-check fell to the principal allowlist and — with no
``A2A_ALLOWED_USERS`` — default-denied every fresh native message that the same gateway's
inbound filter had just accepted.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_envelope import snapshot_native
from hermes_state_runtime import RuntimeStoreError


def _clear_authz_env(monkeypatch) -> None:
    for key in (
        "A2A_ALLOWED_USERS",
        "A2A_ALLOW_ALL_USERS",
        "DISCORD_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _fresh_native_event(
    source, text="hello from a peer", message_id="m1"
) -> MessageEvent:
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, message_id=message_id, source=source
    )


@pytest.mark.asyncio
async def test_upstream_authorized_adapter_admits_fresh_native_without_allowlist(
    tmp_path, monkeypatch
):
    from plugins.platforms.a2a.adapter import A2AAdapter

    _clear_authz_env(monkeypatch)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform("a2a"): adapter}
    source = adapter.build_source(
        chat_id="ctx-1",
        chat_name="a2a:peer",
        chat_type="dm",
        user_id="peer",
        user_name="peer",
    )
    snapshot = snapshot_native(runner, _fresh_native_event(source))
    assert snapshot["text"] == "hello from a peer"
    assert snapshot["native_text_v1"]["source"]["user_id"] == "peer"


@pytest.mark.asyncio
async def test_non_upstream_adapter_still_default_denies_without_allowlist(
    tmp_path, monkeypatch
):
    from gateway.session import SessionSource

    _clear_authz_env(monkeypatch)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner.adapters = {
        Platform.DISCORD: SimpleNamespace(
            send=AsyncMock(),
            authorization_is_upstream=False,
            enforces_own_access_policy=False,
        )
    }
    discord_source = SessionSource(
        platform=Platform.DISCORD,
        user_id="123",
        chat_id="456",
        user_name="someone",
        chat_type="dm",
    )
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        snapshot_native(runner, _fresh_native_event(discord_source))


@pytest.mark.asyncio
async def test_upstream_authorized_a2a_input_admits_durally_with_provenance(
    tmp_path, monkeypatch
):
    """The live path: ``admit_native`` inside the adapter's ``native_callback`` context, i.e. the
    same provenance-bearing re-check the unified gateway runs for every A2A task (#139199)."""
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.a2a.adapter import A2AAdapter

    _clear_authz_env(monkeypatch)
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform("a2a"): adapter}
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="a2a-upstream"
    )
    register_transport_home(runner, None, get_hermes_home())
    source = adapter.build_source(
        chat_id="ctx-1",
        chat_name="a2a:peer",
        chat_type="dm",
        user_id="peer",
        user_name="peer",
    )
    event = _fresh_native_event(source, text="a2a task from a bearer-authenticated peer")
    with native_callback(runner, event, get_hermes_home()):
        receipt = await authority.admit_native(event)
    payload = list_session_admissions(
        authority.db, session_id=receipt.ref.session_id, pending_only=False
    )[-1]["payload"]
    assert payload["text"] == "a2a task from a bearer-authenticated peer"
    assert payload["native_text_v1"]["provenance"]["platform"] == "a2a"


@pytest.mark.asyncio
@pytest.mark.parametrize("forge", [False, True])
async def test_same_principal_without_adapter_declaration_is_refused_by_native_admission(
    tmp_path, monkeypatch, forge
):
    """The A2A peer admitted above, through real native admission on an adapter that does not
    declare ``authorization_is_upstream``, is still refused. A declaration stamped on the event's
    source (``forge``) changes nothing: only the adapter object is read."""
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_constants import get_hermes_home
    from plugins.platforms.a2a.adapter import A2AAdapter

    class UndeclaredAdapter(A2AAdapter):
        @property
        def authorization_is_upstream(self) -> bool:
            return False

    _clear_authz_env(monkeypatch)
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = UndeclaredAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform("a2a"): adapter}
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="a2a-undeclared"
    )
    register_transport_home(runner, None, get_hermes_home())
    source = adapter.build_source(
        chat_id="ctx-1",
        chat_name="a2a:peer",
        chat_type="dm",
        user_id="peer",
        user_name="peer",
    )
    if forge:
        vars(source)["authorization_is_upstream"] = True
    event = _fresh_native_event(source, text="a2a task from a bearer-authenticated peer")
    with native_callback(runner, event, get_hermes_home()):
        with pytest.raises(RuntimeStoreError, match="permission_denied") as refused:
            await authority.admit_native(event)
    assert refused.value.field == "user_not_authorized"
