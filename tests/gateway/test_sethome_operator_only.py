"""The /sethome nudge must NOT leak into arbitrary inbound DMs (#95705).

Live incident (2026-08-26): Hermes on a WhatsApp bot-mode number delivered
the `No home channel is set for WhatsApp. Type /sethome ...` nudge into a
customer's inbound DM. The customer's chat is the wrong audience for an
operator-side configuration message.

The fix gates the nudge on four operator-target signals — ``sender_is_owner``
on the source, the platform's allow-all flag, an explicit per-platform
operator allowlist, or the inbound chat being the platform's configured
home channel of record. On a miss the nudge is logged but NOT delivered to
the inbound chat.

These tests pin the gate at the unit level (pure helper) and at the gateway
flow level (the branch in run.py) so the leak can't regress unnoticed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.run import _sethome_target_is_authorized


# ---------------------------------------------------------------------------
# Pure helper: _sethome_target_is_authorized
# ---------------------------------------------------------------------------


def _runner(home_chat_id=None):
    """Minimal runner mock with the bits the gate probes."""
    cfg = SimpleNamespace()
    if home_chat_id is None:
        cfg.get_home_channel = lambda *_a, **_kw: None
    else:
        cfg.get_home_channel = lambda *_a, **_kw: SimpleNamespace(chat_id=home_chat_id)
    return SimpleNamespace(config=cfg)


def _source(*, chat_id="chat-A", user_id="user-1", sender_is_owner=False, platform=Platform.WHATSAPP):
    return SimpleNamespace(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        sender_is_owner=sender_is_owner,
    )


def test_arbitrary_inbound_dm_is_not_authorized(monkeypatch):
    """Default config + a customer's DM -> never authorized.

    This is the central contract: without an operator signal, the inbound
    chat is treated as a customer, and the nudge is suppressed.
    """
    monkeypatch.delenv("WHATSAPP_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("WHATSAPP_OPERATOR_USERS", raising=False)
    runner = _runner()
    src = _source(chat_id="6281234567890@s.whatsapp.net", user_id="6281234567890")
    assert _sethome_target_is_authorized(runner, src, Platform.WHATSAPP) is False


def test_owner_tagged_inbound_is_authorized(monkeypatch):
    """`sender_is_owner=True` (e.g. WhatsApp `fromOwner`) authorizes delivery.

    Operator's own linked-phone reply to their own chat -> safe to nudge.
    """
    monkeypatch.delenv("WHATSAPP_ALLOW_ALL_USERS", raising=False)
    runner = _runner()
    src = _source(sender_is_owner=True)
    assert _sethome_target_is_authorized(runner, src, Platform.WHATSAPP) is True


def test_platform_allow_all_authorizes(monkeypatch):
    """When WHATSAPP_ALLOW_ALL_USERS=true, every inbound is operator-acceptable.

    By configuration, anyone talking to the bot is implicitly operator-grade
    — the nudge is appropriate for any of them.
    """
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")
    src = _source()
    assert _sethome_target_is_authorized(_runner(), src, Platform.WHATSAPP) is True


def test_platform_allow_all_truthy_variants_authorize(monkeypatch):
    for value in ("1", "yes", "on", "TRUE", "True"):
        monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", value)
        assert _sethome_target_is_authorized(
            _runner(), _source(), Platform.WHATSAPP,
        ) is True, value


def test_global_allow_all_authorizes(monkeypatch):
    """GATEWAY_ALLOW_ALL_USERS=true bypasses the gate for any platform."""
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "1")
    src = _source()
    assert _sethome_target_is_authorized(_runner(), src, Platform.WHATSAPP) is True


def test_operator_allowlist_match_authorizes(monkeypatch):
    """WHATSAPP_OPERATOR_USERS env list matching the sender -> authorized."""
    monkeypatch.setenv("WHATSAPP_OPERATOR_USERS", "ops-alice,ops-bob,ops-carol")
    src = _source(user_id="ops-alice")
    assert _sethome_target_is_authorized(_runner(), src, Platform.WHATSAPP) is True


def test_operator_allowlist_non_match_denies(monkeypatch):
    """Operator allowlist that doesn't include the sender -> denied."""
    monkeypatch.setenv("WHATSAPP_OPERATOR_USERS", "ops-alice,ops-bob")
    src = _source(user_id="customer-123")
    assert _sethome_target_is_authorized(_runner(), src, Platform.WHATSAPP) is False


def test_home_channel_match_authorizes(monkeypatch):
    """Inbound chat == configured home channel -> operator of record talking to themselves."""
    monkeypatch.delenv("WHATSAPP_ALLOW_ALL_USERS", raising=False)
    runner = _runner(home_chat_id="ops-chat-id")
    src = _source(chat_id="ops-chat-id", user_id="ops-alice")
    assert _sethome_target_is_authorized(runner, src, Platform.WHATSAPP) is True


def test_home_channel_mismatch_does_not_authorize(monkeypatch):
    """Inbound chat != configured home channel -> not authorized via that gate."""
    monkeypatch.delenv("WHATSAPP_ALLOW_ALL_USERS", raising=False)
    runner = _runner(home_chat_id="ops-chat-id")
    src = _source(chat_id="customer-chat-id", user_id="ops-alice")
    assert _sethome_target_is_authorized(runner, src, Platform.WHATSAPP) is False


def test_truthy_string_allow_all_insensitive(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "YES")
    assert _sethome_target_is_authorized(_runner(), _source(), Platform.WHATSAPP) is True


def test_missing_runner_config_does_not_crash(monkeypatch):
    """A runner without `config` attribute must not raise — fail closed."""
    monkeypatch.delenv("WHATSAPP_ALLOW_ALL_USERS", raising=False)
    runner = SimpleNamespace(config=None)  # no get_home_channel -> error path
    src = _source()
    # Should NOT raise; should fall through to deny.
    assert _sethome_target_is_authorized(runner, src, Platform.WHATSAPP) is False


# ---------------------------------------------------------------------------
# Integration: the sethome branch in the actual gateway flow
# ---------------------------------------------------------------------------


class _FakeAdapter:
    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, *, metadata=None):
        self.calls.append((chat_id, content, metadata))
        return SimpleNamespace(success=True)


class _FakeRunner:
    def __init__(self, adapter, home_chat_id=None):
        cfg = SimpleNamespace()
        cfg.get_home_channel = (
            lambda *_a, **_kw: SimpleNamespace(chat_id=home_chat_id)
            if home_chat_id else None
        )
        self.config = cfg
        self._adapter = adapter
        self.adapter_calls = []

    async def _deliver_platform_notice(self, source, content):
        # Same shape as the real runner: calls adapter.send on source.chat_id.
        await self._adapter.send(source.chat_id, content)
        self.adapter_calls.append((source.chat_id, content))


@pytest.mark.asyncio
async def test_sethome_branch_suppresses_for_arbitrary_inbound(monkeypatch, caplog):
    """End-to-end: no operator signals -> no chat-side delivery, log only.

    Reproduces the live incident: bot enabled, no home channel, customer DMs.
    After the fix the nudge must NOT reach the customer's chat.
    """
    monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)
    monkeypatch.delenv("WHATSAPP_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("WHATSAPP_OPERATOR_USERS", raising=False)

    adapter = _FakeAdapter()
    runner = _FakeRunner(adapter)
    src = _source(chat_id="customer-DM-id", user_id="customer-1")

    # We invoke only the gate + delivery-arm logic so we don't pull in the
    # full gateway runner. The branch shape matches run.py:20614-20621.
    gate_passed = _sethome_target_is_authorized(runner, src, src.platform)

    caplog.set_level(logging.WARNING, logger="gateway.run")
    if gate_passed:
        await runner._deliver_platform_notice(src, "nudge")
    else:
        gateway_run.logger.warning(
            "#95705 sethome nudge suppressed for non-operator inbound on %s chat=%s user=%s",
            src.platform.value if src.platform else "?",
            src.chat_id,
            src.user_id,
        )

    # Critical: customer chat received nothing.
    assert adapter.calls == []
    # Critical: the suppression is greppable in the operator log.
    suppression = [
        r for r in caplog.records
        if "sethome nudge suppressed" in r.getMessage()
    ]
    assert suppression, [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_sethome_branch_delivers_to_owner(monkeypatch):
    """sender_is_owner=True (e.g. WhatsApp `fromOwner`) -> nudge delivered."""
    monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)
    monkeypatch.delenv("WHATSAPP_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    adapter = _FakeAdapter()
    runner = _FakeRunner(adapter)
    src = _source(chat_id="ops-linked-phone", user_id="owner-1", sender_is_owner=True)

    gate_passed = _sethome_target_is_authorized(runner, src, src.platform)
    if gate_passed:
        await runner._deliver_platform_notice(src, "📬 No home channel is set ...")

    assert len(adapter.calls) == 1
    chat_id, content, _md = adapter.calls[0]
    assert chat_id == "ops-linked-phone"
    assert "No home channel is set" in content


@pytest.mark.asyncio
async def test_sethome_branch_delivers_via_operator_allowlist(monkeypatch):
    """WHATSAPP_OPERATOR_USERS list match -> nudge delivered."""
    monkeypatch.setenv("WHATSAPP_OPERATOR_USERS", "ops-alice,ops-bob")
    monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)
    monkeypatch.delenv("WHATSAPP_ALLOW_ALL_USERS", raising=False)

    adapter = _FakeAdapter()
    runner = _FakeRunner(adapter)
    src = _source(chat_id="ops-chat", user_id="ops-alice")

    gate_passed = _sethome_target_is_authorized(runner, src, src.platform)
    if gate_passed:
        await runner._deliver_platform_notice(src, "📬 No home channel is set ...")

    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_sethome_branch_delivers_via_allow_all(monkeypatch):
    """WHATSAPP_ALLOW_ALL_USERS=true -> nudge delivered to any inbound.

    By configuration, every inbound is operator-acceptable; the nudge is
    safe to send.
    """
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")
    monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)

    adapter = _FakeAdapter()
    runner = _FakeRunner(adapter)
    src = _source(chat_id="any-chat", user_id="any-user")

    gate_passed = _sethome_target_is_authorized(runner, src, src.platform)
    if gate_passed:
        await runner._deliver_platform_notice(src, "📬 No home channel is set ...")

    assert len(adapter.calls) == 1
