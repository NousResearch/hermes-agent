"""P5(b): connector egress DECLINES surface as real errors, not swallowed successes.

The connector's egress-authorization floor answers an unauthorized destination
with a DEFINITE failure whose text is deliberately uniform (finding F-005 — the
caller must not learn *why*). The gateway's job is to report faithfully THAT it
happened. The failure mode these tests pin is specific: several relay lanes
degrade a *transport drop* by design, and that same degradation used to swallow
an *authorization refusal* — either into a wrong reason ("prompt op
unavailable") or, worse, into a DIFFERENT op re-addressed at the very chat the
connector just refused (media falling back to a plain text notice).

Every lane below drives the REAL `RelayAdapter` built from the REAL descriptor;
the only substitution is the transport, which is what the connector is.
"""

from __future__ import annotations

import os

import asyncio
import logging
from typing import Any, Dict, List, Optional

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.egress import (
    EGRESS_DECLINE_CODE,
    decline_error,
    is_egress_decline,
)

DECLINE_TEXT = (
    "discord egress declined: target is not an approved destination for this connection"
)
DECLINE: Dict[str, Any] = {"success": False, "error": DECLINE_TEXT}

ALL_OPS = (
    "send",
    "edit",
    "typing",
    "delete",
    "react",
    "send_media",
    "prompt",
    "draft",
    "task_card",
    "task_card_stop",
    "thread_create",
    "thread_rename",
)


class DecliningConnector:
    """A connector that refuses EVERY destination, like the real egress floor."""

    def __init__(self, descriptor: CapabilityDescriptor) -> None:
        self._descriptor = descriptor
        self.ops: List[str] = []
        self._identities = [(descriptor.platform, "b1")]

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def handshake(self) -> CapabilityDescriptor:
        return self._descriptor

    def set_inbound_handler(self, handler) -> None:
        return None

    def set_passthrough_handler(self, handler) -> None:
        return None

    async def send_outbound(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        self.ops.append(str(action.get("op")))
        return dict(DECLINE)

    async def send_follow_up(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        return dict(DECLINE)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

    async def send_interrupt(self, session_key, reason=None) -> None:
        return None

    async def go_idle(self, timeout_s: float = 10.0) -> bool:
        return True


@pytest.fixture
def relay():
    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Relay",
        max_message_length=4096,
        supports_draft_streaming=True,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="plain",
        len_unit="chars",
        supported_ops=ALL_OPS,
    )
    connector = DecliningConnector(descriptor)
    adapter = RelayAdapter(
        PlatformConfig(enabled=True, extra={}), descriptor, transport=connector
    )
    return adapter, connector


# ── the classifier itself ────────────────────────────────────────────────

def test_decline_is_recognised_by_code_and_by_uniform_text():
    # The wire contract, pinned as a LITERAL. Asserting against the imported
    # constant is a tautology — it cannot fail when the constant changes, and
    # review mutation M05 survived exactly there. The connector stamps this
    # exact string (gateway-gateway routedEgressGuard); changing either side
    # alone is a silent cross-repo break, so the literal is the point.
    assert EGRESS_DECLINE_CODE == "egress_declined"
    assert is_egress_decline({"success": False, "code": "egress_declined"}) is True
    assert is_egress_decline({"success": False, "code": EGRESS_DECLINE_CODE}) is True

    assert is_egress_decline(DECLINE) is True

    # M10: the `.lower()` in is_egress_decline was untested, so making the
    # marker match case-SENSITIVE survived — a connector emitting "Egress
    # declined:" would silently stop being classified as a decline and start
    # falling back into the refused chat. Every fixture happened to be
    # lowercase, which is what hid it.
    for variant in (
        "Discord Egress Declined: target is not approved",
        "EGRESS DECLINED: target is not approved",
        "discord EGRESS declined: target is not approved",
    ):
        assert is_egress_decline({"success": False, "error": variant}) is True, variant



def test_an_ambiguous_failure_is_not_a_decline():
    """A lost ack may well have been APPLIED — it is a transport outcome.

    Classifying it as a decline would convert the relay's deliberate
    optimistic-retry behaviour into a hard error on a message that landed.
    """
    assert (
        is_egress_decline(
            {"success": False, "error": DECLINE_TEXT, "ambiguous": True}
        )
        is False
    )


def test_an_ordinary_failure_is_not_a_decline():
    assert is_egress_decline({"success": False, "error": "file too large"}) is False
    assert is_egress_decline({"success": True}) is False
    assert is_egress_decline(None) is False


def test_decline_error_is_the_connector_text_verbatim():
    """No re-wording, no reason-parsing: the uniform sentence, unchanged."""
    assert decline_error(DECLINE) == DECLINE_TEXT


# ── lanes that must report the decline to their caller ───────────────────

def test_send_reports_the_decline(relay):
    adapter, connector = relay
    result = asyncio.run(adapter.send("C1", "hi"))

    assert result.success is False
    assert result.error == DECLINE_TEXT
    assert connector.ops == ["send"]


def test_media_decline_does_not_fall_back_to_a_text_send(relay):
    """The worst swallow: a refused destination re-addressed by a different op.

    `_send_media` returning None hands the caller back to
    `BasePlatformAdapter.send_image`, which sends the URL as TEXT — into the
    very chat the connector just refused. The lane must report the refusal
    instead, and emit exactly ONE op.
    """
    adapter, connector = relay
    result = asyncio.run(adapter.send_image("C1", "https://x/y.png", caption="cap"))

    assert result.success is False
    assert result.error == DECLINE_TEXT
    assert connector.ops == ["send_media"]


def test_exec_approval_decline_is_not_reported_as_op_unavailable(relay):
    """"prompt op unavailable" is a WRONG reason that triggers a text fallback.

    The whole observable: the refusal reaches the caller verbatim, exactly one
    op is emitted, and the minted prompt is UNREGISTERED — a prompt left
    pending for a card that was never delivered would silently capture the
    user's next reply in that chat as an approval press.
    """
    adapter, connector = relay
    result = asyncio.run(adapter.send_exec_approval("C1", "rm -rf /", "sk1"))

    assert result.success is False
    assert result.error == DECLINE_TEXT
    assert connector.ops == ["prompt"]
    assert adapter._pending_prompts == {}


def test_slash_confirm_decline_is_not_reported_as_op_unavailable(relay):
    adapter, connector = relay
    result = asyncio.run(
        adapter.send_slash_confirm("C1", "Title", "Body", "sk1", "cf1")
    )

    assert result.success is False
    assert result.error == DECLINE_TEXT
    assert connector.ops == ["prompt"]
    assert adapter._pending_prompts == {}


def test_clarify_decline_does_not_fall_back_to_a_numbered_text_send(relay):
    """The base class's numbered-text clarify would `send()` into the refused chat."""
    adapter, connector = relay
    result = asyncio.run(
        adapter.send_clarify("C1", "Which?", ["a", "b"], "cl1", "sk1")
    )

    assert result.success is False
    assert result.error == DECLINE_TEXT
    assert connector.ops == ["prompt"]
    assert adapter._pending_prompts == {}


def test_a_delivered_prompt_stays_registered(relay, monkeypatch):
    """Guard the converse: the cleanup must not unregister a LIVE prompt."""
    adapter, connector = relay

    async def _ok(action, *, platform=None):
        connector.ops.append(str(action.get("op")))
        return {"success": True, "message_id": "pm1"}

    monkeypatch.setattr(connector, "send_outbound", _ok)
    result = asyncio.run(adapter.send_exec_approval("C1", "ls", "sk1"))

    assert result.success is True
    assert connector.ops == ["prompt"]
    assert len(adapter._pending_prompts) == 1


def test_task_card_stop_decline_carries_the_error(relay):
    """The stop lane discarded the error entirely (`success=` only)."""
    adapter, connector = relay
    result = asyncio.run(adapter.stop_native_task_card_progress("C1"))

    assert result.success is False
    assert result.error == DECLINE_TEXT
    assert connector.ops == ["task_card_stop"]


# ── lanes that legitimately degrade, but must still SAY so ───────────────

@pytest.mark.parametrize(
    "lane,call,expected",
    [
        ("typing", lambda a: a.send_typing("C1"), None),
        ("delete", lambda a: a.delete_message("C1", "m1"), False),
        ("thread_create", lambda a: a.create_handoff_thread("C1", "n"), None),
        ("thread_rename", lambda a: a.rename_thread("T1", "n"), False),
    ],
)
def test_cosmetic_lane_degrades_but_logs_the_decline_at_warning(
    relay, caplog, lane, call, expected
):
    """These return bool/None by contract; a refusal must not vanish silently."""
    adapter, _connector = relay
    with caplog.at_level(logging.WARNING, logger="gateway.relay.egress"):
        assert asyncio.run(call(adapter)) == expected

    declines = [
        r
        for r in caplog.records
        if r.name == "gateway.relay.egress" and "DECLINED" in r.getMessage()
    ]
    assert len(declines) == 1
    assert DECLINE_TEXT in declines[0].getMessage()


# ── round-2 review survivors: lines the docstrings call load-bearing ────────
#
# M25 and M21 both survived because nothing exercised them, while the code
# around them explains at length why they matter. A comment is not a guard.


def test_thread_qualified_session_id_attests_the_bare_chat():
    """M25: session ids may be "chat:thread"; the connector authorizes the CHAT.

    `_session_ids` deliberately adds BOTH forms. Without the split, a gateway
    whose session origin is `-100999:77` cannot send to `-100999` — the chat it
    is demonstrably already talking in.
    """
    import gateway.channel_directory as cd
    import gateway.relay.egress as eg

    original = cd._build_from_sessions
    try:
        cd._build_from_sessions = lambda _p: [{"id": "-100999:77"}]
        ids = eg._session_ids("telegram")
    finally:
        cd._build_from_sessions = original

    assert "-100999:77" in ids, "the qualified form must stay attested"
    assert "-100999" in ids, "the bare chat must be attested (M25)"


def test_relay_plane_attests_the_union_of_fronted_platforms():
    """M21: a relay session is filed under its LOGICAL platform.

    `attested_relay_targets("relay")` must span every fronted platform, or the
    generic plane refuses chats the agent is already in — the exact failure the
    docstring warns about.
    """
    import gateway.channel_directory as cd
    import gateway.relay.egress as eg

    orig_dir = cd.load_directory
    orig_env = os.environ.get("GATEWAY_RELAY_PLATFORMS")
    try:
        # Drive the REAL `relay_fronted_platforms()` through its env source —
        # the `GATEWAY_RELAY_PLATFORMS` deploy stamp. My first version patched
        # `_relay_fronted` itself, so the mutation that emptied it survived:
        # the test was asserting against its own stub instead of production.
        os.environ["GATEWAY_RELAY_PLATFORMS"] = "discord,slack"
        cd.load_directory = lambda: {
            "platforms": {
                "discord": [{"id": "C-DISCORD"}],
                "slack": [{"id": "C-SLACK"}],
                "relay": [],
            }
        }
        targets = eg.attested_relay_targets("relay")
    finally:
        cd.load_directory = orig_dir
        if orig_env is None:
            os.environ.pop("GATEWAY_RELAY_PLATFORMS", None)
        else:
            os.environ["GATEWAY_RELAY_PLATFORMS"] = orig_env

    assert "C-DISCORD" in targets and "C-SLACK" in targets, (
        "the relay plane must union the fronted platforms (M21)"
    )


# ── B-2: Telegram `@username` is authorized by the CONNECTOR ────────────────
#
# Provenance stores RESOLVED numeric chat ids; a public `@channel` is not a
# destination until the Bot API resolves it at send time. Comparing the two
# could only ever refuse, which regressed the username support added in
# #53573. The guard fires only on relay-fronted deployments, where the
# CONNECTOR holds the bot token — so the gateway has no way to resolve it, and
# the connector's own egress floor (gg#238) is the layer that authorizes it.
#
# These tests pin the carve-out's EDGES. It must not widen.


def _relay_env(monkeypatch, platform="telegram", directory=None):
    import gateway.channel_directory as cd

    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", platform)
    monkeypatch.setattr(
        cd, "load_directory", lambda: {"platforms": {platform: directory or []}}
    )
    monkeypatch.setattr(cd, "_build_from_sessions", lambda _p: [])
    import gateway.relay.egress as eg

    monkeypatch.setattr(eg, "_home_channel_id", lambda _p: None)
    monkeypatch.setattr(eg, "_has_live_native_adapter", lambda _p: False)
    return eg


def test_telegram_username_defers_to_the_connector(monkeypatch):
    """The regression case: a public handle must not be refused here."""
    eg = _relay_env(monkeypatch)
    assert eg.authorize_relay_target("telegram", "@some_public_channel") is None


def test_numeric_telegram_target_is_still_guarded(monkeypatch):
    """The carve-out must not leak to resolved ids — the guard's whole point."""
    eg = _relay_env(monkeypatch)
    denial = eg.authorize_relay_target("telegram", "-1009999999999")
    assert denial is not None and "-1009999999999" in denial


def test_carve_out_is_telegram_only(monkeypatch):
    """Another platform's `@` form is NOT a Telegram handle.

    Matrix targets `@user:server.org`; Slack has `@handle` pseudo-ids. Neither
    is resolved by the Telegram Bot API, so neither may ride this exemption.
    """
    eg = _relay_env(monkeypatch, platform="matrix")
    denial = eg.authorize_relay_target("matrix", "@someone:server.org")
    assert denial is not None, "the carve-out widened beyond telegram"


def test_attested_handle_takes_the_normal_path(monkeypatch):
    """Order check: attestation is consulted BEFORE the carve-out.

    A handle that IS attested must pass as attested, not as an exemption —
    otherwise the carve-out would be masking whether attestation still works.

    Both paths return None, so asserting the verdict cannot tell them apart:
    my first version of this test passed happily with the carve-out moved
    ABOVE the attestation lookup. Observe the MECHANISM instead — attestation
    must actually be consulted — which is the difference between the two
    orderings.
    """
    eg = _relay_env(monkeypatch, directory=[{"id": "@known_channel"}])

    consulted: list[str] = []
    real = eg.attested_relay_targets
    monkeypatch.setattr(
        eg,
        "attested_relay_targets",
        lambda p: (consulted.append(p), real(p))[1],
    )

    assert eg.authorize_relay_target("telegram", "@known_channel") is None
    assert consulted == ["telegram"], (
        "attestation was skipped — the carve-out is short-circuiting it"
    )
    assert "@known_channel" in real("telegram")


def test_username_like_but_not_prefixed_is_guarded(monkeypatch):
    """No `@`, no exemption — a bare name is still an unattested target."""
    eg = _relay_env(monkeypatch)
    assert eg.authorize_relay_target("telegram", "some_public_channel") is not None
