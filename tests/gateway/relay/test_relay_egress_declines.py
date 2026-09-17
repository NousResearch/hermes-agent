"""P5(b): connector egress DECLINES surface as real errors, not swallowed successes.

The connector's egress-authorization floor answers an unauthorized destination
with a DEFINITE failure whose text is deliberately uniform (finding F-005 — the
caller must not learn *why*). The gateway's job is to report faithfully THAT it
happened. The failure mode these tests pin is specific: several relay lanes
degrade a *transport drop* by design, and that same degradation used to swallow
an *authorization refusal* — either into a wrong reason ("prompt op
unavailable") or, worse, into a DIFFERENT op re-addressed at the very chat the
connector just refused (media falling back to a plain text notice).

Most lanes below drive the REAL `RelayAdapter` built from the REAL descriptor;
the only substitution is the transport, which is what the connector is. The
exceptions are explicit: the stream-consumer tests construct
`StreamTransportMixin` / `StreamFallbackMixin` doubles and inject a `SendResult`
directly, because the behaviour under test belongs to the CALLER of the adapter,
not to the adapter itself.
"""

from __future__ import annotations

import asyncio
import logging
import os
from types import SimpleNamespace
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
        # Real entries carry `thread_id` alongside the composed id
        # (`_session_entry_id` builds f"{chat_id}:{thread_id}"), and the parent
        # is now recovered from that field rather than by splitting on ":" —
        # splitting invented attestations for ids whose colon is part of the
        # address (Matrix). The PROPERTY below is unchanged.
        cd._build_from_sessions = lambda _p: [{"id": "-100999:77", "thread_id": "77"}]
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


# ── code-only declines: the wire shape that has NO marker colon ─────────────
#
# Review's finding: every case above declines with marker TEXT, so deleting
# `raw_response=result` from the production return path left all 34 tests
# green. The structured `code` is documented as the PREFERRED signal precisely
# because a connector may send no prose at all, and a caller that rebuilds
# `{"success": False, "error": ...}` from `error` alone cannot see it.

CODE_ONLY_DECLINE: Dict[str, Any] = {"success": False, "code": EGRESS_DECLINE_CODE}


class CodeOnlyDecliningConnector(DecliningConnector):
    """Refuses with a structured code and NO error prose."""

    async def send_outbound(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        self.ops.append(str(action.get("op")))
        return dict(CODE_ONLY_DECLINE)


@pytest.fixture
def code_only_relay():
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
    connector = CodeOnlyDecliningConnector(descriptor)
    adapter = RelayAdapter(
        PlatformConfig(enabled=True, extra={}), descriptor, transport=connector
    )
    return adapter, connector


def test_code_only_prompt_decline_reaches_the_caller_as_a_decline(code_only_relay):
    """The REAL adapter's SendResult must carry the structured decline through.

    Drives the production `send_exec_approval` -> `_send_prompt` path and feeds
    its real SendResult to the real `_approval_send_outcome`, rather than
    hand-building a SimpleNamespace. With `raw_response` dropped, the verdict
    degrades to "failed" — the cue that triggers the text fallback into the
    chat the connector just refused.
    """
    from gateway.run import _approval_send_outcome

    adapter, connector = code_only_relay
    result = asyncio.run(
        adapter.send_exec_approval("C1", "rm -rf /", "sk1", description="danger")
    )

    assert result is not None
    assert result.success is False
    assert connector.ops == ["prompt"]
    # The structured body must survive to the caller.
    assert isinstance(result.raw_response, dict)
    assert is_egress_decline(result.raw_response)

    class _Fut:
        def result(self, timeout=None):
            return result

    assert _approval_send_outcome(_Fut(), timeout=1) == "declined"
    # A prompt that never rendered must not stay pending.
    assert adapter._pending_prompts == {}


def test_code_only_media_decline_does_not_fall_back(code_only_relay):
    """Same wire shape on the media lane: a failed lane, never a silent None."""
    adapter, connector = code_only_relay
    result = asyncio.run(
        adapter.send_image("C1", "https://example.invalid/a.png", caption="hi")
    )
    assert result is not None
    assert result.success is False
    assert connector.ops == ["send_media"]
    assert isinstance(result.raw_response, dict)
    assert is_egress_decline(result.raw_response)

def _code_only_adapter():
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
    connector = CodeOnlyDecliningConnector(descriptor)
    adapter = RelayAdapter(
        PlatformConfig(enabled=True, extra={}), descriptor, transport=connector
    )
    return adapter, connector


# ── sibling content lanes that also re-address the same chat ────────────────


def test_declined_draft_seal_does_not_replay_as_a_plain_send():
    """Review round 3, finding 1 — a real content leak, reproduced end to end.

    On a stream-is-the-message platform (Slack) the turn-final is converted
    into draft(final=True). When the connector REFUSES that seal, the old code
    read it as a lane failure and delivered the same content through `send` —
    ops were [draft(partial), draft(final,SECRET), send(SECRET)].
    """
    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="slack",  # stream-is-the-message; discord would not arm a seal
        label="Relay",
        max_message_length=4096,
        supports_draft_streaming=True,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="plain",
        len_unit="chars",
        supported_ops=ALL_OPS,
    )

    class SealDecliner(DecliningConnector):
        async def send_outbound(self, action, *, platform=None):
            op = str(action.get("op"))
            self.ops.append(op)
            if op == "draft" and not action.get("final"):
                return {"success": True, "message_id": "m1"}  # stream opens
            return dict(CODE_ONLY_DECLINE)  # the SEAL is refused

    connector = SealDecliner(descriptor)
    adapter = RelayAdapter(
        PlatformConfig(enabled=True, extra={}), descriptor, transport=connector
    )

    asyncio.run(adapter.send_draft("C1", 1, "partial"))
    asyncio.run(adapter.send("C1", "SECRET", metadata={}))

    # The seal was attempted and refused; the content must NOT be replayed.
    assert connector.ops == ["draft", "draft"]
    assert "send" not in connector.ops


def test_declined_task_card_progress_carries_the_decline_to_its_caller():
    """Review round 3, finding 6 — the same laundering, in the card lane.

    The TurnRunner reads a bare failure as "card lane unavailable" and sends
    fallback TEXT to the same chat, so the decline must reach it structured.
    """
    adapter, connector = _code_only_adapter()
    result = asyncio.run(
        adapter.send_native_task_card_progress(
            "C1", [{"text": "step one"}], title="Working"
        )
    )
    assert result.success is False
    assert connector.ops == ["task_card"]
    assert is_egress_decline(result.raw_response)


def test_declined_INITIAL_draft_is_not_retried_as_a_plain_send():
    """R5-3: the round-3 fix covered a declined SEAL, not a declined OPEN.

    `send_draft` logged the decline but returned a bare failure, so the stream
    consumer read it as "draft transport unusable", disabled drafts, and fell
    through to `_first_send`. Measured through the real adapter + real
    StreamTransportMixin: ops were ['draft', 'send'].
    """
    from gateway.platforms.base import SendResult
    from gateway.stream_consumer_transport import StreamTransportMixin

    adapter, connector = _code_only_adapter()

    class _Consumer(StreamTransportMixin):
        def __init__(self):
            self.adapter = adapter
            self.chat_id = "C1"
            self._draft_id = 1
            self._use_draft_streaming = True
            self._draft_failures = 0
            self._initial_reply_to_id = None
            self._already_sent = False
            self._last_sent_text = ""
            self._edit_supported = True

        def _draft_metadata(self):
            return {}

        def _metadata_for_send(self, **k):
            return {}

        def _visible_prefix(self):
            return ""

        def _enter_fallback_mode(self, *a):
            return None

        def _adopt_message_id(self, mid):
            return None

        def _track_preview_ids_from_result(self, r):
            return None

    consumer = _Consumer()
    assert asyncio.run(consumer._send_draft_frame("partial")) is False
    # The turn-final must NOT be replayed into the refused chat.
    assert asyncio.run(consumer._first_send("SECRET", finalize=True)) is False
    assert connector.ops == ["draft"]


def test_edit_message_carries_the_structured_decline():
    """R6-2 root cause: `edit_message` dropped the connector response.

    THREE callers read a bare edit failure as "editing is unavailable" and
    re-send the content as a NEW message to the same chat (stream edit
    fallback, queued reconciliation, task-card fallback). One dropped field,
    three leaks — so the fix belongs here, at the source.
    """
    adapter, connector = _code_only_adapter()
    result = asyncio.run(adapter.edit_message("C1", "m1", "SECRET"))

    assert result.success is False
    assert connector.ops == ["edit"]
    assert is_egress_decline(result.raw_response)


def test_declined_stream_edit_does_not_send_the_unseen_tail():
    """R6-2: the stream consumer's edit-failure funnel.

    A refused edit put the consumer into ordinary fallback mode, which delivers
    the unseen tail as a plain send. Measured: ops were ['edit', 'edit', 'send'].

    THE DOUBLE IS DELIBERATELY COMPLETE. An earlier version implemented only the
    guarded path, so removing either guard raised AttributeError inside the fake
    (`_is_flood_error`, `_clean_for_display`) and the test went red for the wrong
    reason, proving nothing. Each guard now dies on its OWN observable, and the
    two are different — state for the first, the wire for the second:

        remove the _on_edit_failure decline check
            -> assert consumer._egress_declined is True
               (execution stops here; this mutant never reaches the fallback)
        remove the _send_fallback_final early return
            -> AssertionError: the unseen tail reached the wire: ['send']
    """
    from gateway.platforms.base import SendResult
    from gateway.stream_consumer_fallback import StreamFallbackMixin
    from gateway.stream_consumer_transport import StreamTransportMixin

    adapter, connector = _code_only_adapter()

    class _Consumer(StreamFallbackMixin, StreamTransportMixin):
        def __init__(self):
            self.adapter = adapter
            self.chat_id = "C1"
            self._egress_declined = False
            self._edit_supported = True
            self._message_id = "m1"
            self._last_sent_text = ""
            self._final_content_delivered = False
            self._draft_id = None
            self._use_draft_streaming = False
            self._draft_failures = 0
            self._fallback_prefix = ""
            self._flood_strikes = 0
            self._current_edit_interval = 1.0
            self._last_edit_time = 0.0
            self.thread_id = None
            self.cfg = SimpleNamespace(cursor=None)

        # ── reached only when a guard is REMOVED; present so the mutant
        #    reaches the wire instead of dying inside the double ──
        def _is_flood_error(self, *a, **kw):
            return False

        def _clean_for_display(self, text):
            return text

        def _notify_new_message(self, *a, **kw):
            return None

        async def _try_strip_cursor(self, *a, **kw):
            return None

        def _visible_prefix(self):
            return ""

        def _record_turn_final_payload(self, text):
            return None

        def _enter_fallback_mode(self, *a):
            return None

        def _draft_metadata(self):
            return {}

        def _send_metadata(self, *a, **kw):
            return {}

        def _metadata_for_send(self, *a, **kw):
            return {}

    consumer = _Consumer()
    declined = SendResult(
        success=False, error="declined", raw_response=dict(CODE_ONLY_DECLINE)
    )

    asyncio.run(
        consumer._on_edit_failure(declined, "tail", finalize=True, is_turn_final=True)
    )
    assert consumer._egress_declined is True

    # Drive the real fallback: this is the lane that re-sent the tail.
    asyncio.run(consumer._send_fallback_final("SECRET-TAIL"))

    # The decline arrived as an injected SendResult, so no frame was sent for
    # it; the invariant is that the fallback put NOTHING on the wire.
    assert connector.ops == [], f"the unseen tail reached the wire: {connector.ops}"


# ── holes found by attacking the latch itself ──────────────────────────────


def test_follow_up_carries_the_structured_decline():
    """`send_follow_up` is addressed by session_key, so it has no latch identity
    — but it must still not DISCARD the connector's verdict, which is exactly
    how the edit lane laundered declines into a plain send."""
    adapter, connector = _code_only_adapter()

    async def _declining_follow_up(action, *, platform=None):
        connector.ops.append("follow_up")
        return dict(CODE_ONLY_DECLINE)

    connector.send_follow_up = _declining_follow_up

    result = asyncio.run(adapter.send_follow_up("sess-1", "discord.interaction_token", "SECRET"))

    assert not result.success
    assert is_egress_decline(result.raw_response)


# ── one classifier for SendResult declines ─────────────────────────────────


def test_declined_send_accepts_both_wire_shapes():
    """Eight gateway lanes hand-rolled this unwrapping with TWO answers.

    Six checked only `raw_response`; two also checked the error text. A
    connector answering with the uniform decline SENTENCE and no structured
    code — the documented contract for older connectors — was therefore
    classified as an ordinary failure by those six, so the lane treated a
    refusal as "editing unavailable" and retried through another op.

    Measured before the fix: text-only decline -> six-site False, two-site True.
    """
    from gateway.platforms.base import SendResult
    from gateway.relay.egress import declined_send

    # Structured body.
    assert declined_send(
        SendResult(success=False, error="x", raw_response=dict(CODE_ONLY_DECLINE))
    )
    # Uniform decline sentence only — the shape six lanes used to miss.
    assert declined_send(
        SendResult(success=False, error="egress declined: destination not approved")
    )
    # AMBIGUOUS is a transport outcome, never an authorization one: the frame
    # may have been applied, so it must not be classified as a refusal.
    assert not declined_send(
        SendResult(
            success=False,
            error="lost ack",
            raw_response={"success": False, "ambiguous": True, "code": EGRESS_DECLINE_CODE},
        )
    )
    # Ordinary failures stay ordinary — over-refusal is the larger risk now.
    assert not declined_send(SendResult(success=False, error="rate limited"))
    assert not declined_send(SendResult(success=True, message_id="m1"))


def test_declined_draft_frame_is_terminal_for_the_run():
    """Covers `_send_draft_frame`, whose decline check SURVIVED mutation.

    A declined draft frame must set `_egress_declined`, not merely disable
    drafts: disabling drafts alone routes the turn-final to `_first_send`,
    which is a plain send into the chat the connector just refused.

    Also asserts the text-only wire shape here specifically — this lane is one
    of the six that used to read `raw_response` only, so before the shared
    classifier it missed exactly this reply.
    """
    from gateway.platforms.base import SendResult
    from gateway.stream_consumer_transport import StreamTransportMixin

    class _Adapter:
        def __init__(self, result):
            self._result = result
            self.draft_calls = 0

        async def send_draft(self, **kwargs):
            self.draft_calls += 1
            return self._result

    class _Consumer(StreamTransportMixin):
        def __init__(self, adapter):
            self.adapter = adapter
            self.chat_id = "C1"
            self._draft_id = "d1"
            self._use_draft_streaming = True
            self._draft_failures = 0
            self._last_sent_text = None
            self._egress_declined = False

        def _draft_metadata(self):
            return {}

    # Text-only decline: no structured body at all.
    adapter = _Adapter(SendResult(success=False, error="egress declined: not approved"))
    consumer = _Consumer(adapter)
    assert asyncio.run(consumer._send_draft_frame("partial")) is False
    assert consumer._egress_declined is True

    # CONTROL: an ordinary draft failure disables drafts WITHOUT going terminal,
    # otherwise this fix silently becomes "one flaky frame mutes the chat".
    adapter2 = _Adapter(SendResult(success=False, error="rate limited"))
    consumer2 = _Consumer(adapter2)
    assert asyncio.run(consumer2._send_draft_frame("partial")) is False
    assert consumer2._egress_declined is False
    assert consumer2._use_draft_streaming is False


# ── round 10 blockers ──────────────────────────────────────────────────────


def test_ambiguous_draft_result_never_reads_as_a_decline():
    """AMBIGUOUS is a transport outcome: the frame may well have been
    delivered, so it must never terminate the run.

    The ambiguous projection discarded `raw_response`, so `declined_send` fell
    through to the error-text branch — and an ambiguous result whose text
    happens to carry the decline marker ("... egress declined: ack lost") read
    as a DEFINITE refusal.
    """
    from gateway.relay.egress import declined_send

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

    class _Ambiguous(DecliningConnector):
        async def send_outbound(self, action, *, platform=None):
            self.ops.append(str(action.get("op")))
            return {
                "success": False,
                "ambiguous": True,
                "error": "discord egress declined: ack lost",
            }

    connector = _Ambiguous(descriptor)
    adapter = RelayAdapter(
        PlatformConfig(enabled=True, extra={}), descriptor, transport=connector
    )

    result = asyncio.run(adapter.send_draft("C1", "d1", "partial"))

    assert result.raw_response is not None, "the ambiguous body must be carried"
    assert result.raw_response.get("ambiguous") is True
    assert not declined_send(result), "ambiguous must not classify as a decline"


def test_declined_send_ignores_ack_lost_text_without_a_structured_body():
    """DEFENCE IN DEPTH for a projection that loses the `ambiguous` flag.

    The text-only branch cannot see `ambiguous`, so any error text saying the
    ack was lost is a transport outcome — never authorization.
    """
    from gateway.platforms.base import SendResult
    from gateway.relay.egress import declined_send

    assert not declined_send(
        SendResult(success=False, error="discord egress declined: ack lost")
    )
    # CONTROL: a genuine text-only decline is still detected.
    assert declined_send(
        SendResult(success=False, error="egress declined: destination not approved")
    )
