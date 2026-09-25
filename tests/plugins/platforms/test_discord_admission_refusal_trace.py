"""The trace a dropped Discord message leaves (INS1-676 / REQ-INS1-SAAS-082 Beh 3).

The channel-admission gate used to refuse in silence: ``_handle_message`` returned False
before anything recorded the message, and the drop's only line was a ``logger.debug`` —
below gateway.log's INFO handler level, on the very logger that routes there. A dropped
message was therefore indistinguishable from a dead bot, from the operator's side.

What is pinned here is the contract the fix owes, never the wording of the line:

* every refusing branch writes at least one operator-visible line naming the channel id and
  a typed refusal type, for the channel that was actually refused;
* the branches carry DISTINCT, readable tokens drawn from one closed vocabulary;
* a dropped message cannot pass silently — asserting the line IS the assertion, so a missing
  line fails the test rather than quietly passing;
* a message the gate admits writes no refusal line, so the trace marks drops and not traffic.

Behaviour only: the line is read off the logger the adapter really uses.
"""

import asyncio
import logging
import os
import re
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


# ── discord.py is an optional dependency: stub it when it is absent ──────────
# `_handle_message` narrows on `isinstance(channel, discord.Thread/DMChannel)`, so the
# adapter module needs a non-None `discord` to run the gate at all.
def _ensure_discord_mock() -> None:
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", MagicMock())


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from hermes_logging import COMPONENT_PREFIXES, _ComponentFilter  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


ADAPTER_LOGGER = "plugins.platforms.discord.adapter"

# The two facts the class needs. Both are read from the line itself, so the tests do not
# care about field order or about the rest of the message.
_REFUSAL_TYPE_RE = re.compile(r"\btype=([a-z_][a-z0-9_]*)\b")
_CHANNEL_ID_RE = re.compile(r"\bchannel_id=([0-9]+)\b")

# Every gate env var these tests touch, so one test's env cannot leak into the next.
_GATE_VARS = (
    "DISCORD_ALLOWED_CHANNELS",
    "DISCORD_IGNORED_CHANNELS",
    "DISCORD_FREE_RESPONSE_CHANNELS",
    "DISCORD_REQUIRE_MENTION",
    "DISCORD_AUTO_THREAD",
    "DISCORD_THREAD_REQUIRE_MENTION",
    "DISCORD_NO_THREAD_CHANNELS",
    "DISCORD_ALLOW_BOTS",
)


class _StubChannel:
    """Non-DM, non-thread channel carrying the two keys the gate's channel lookups read."""

    def __init__(self, channel_id: int, name: str = "general"):
        self.id = channel_id
        self.name = name
        self.guild = None
        self.parent = None
        self.parent_id = None
        self.topic = None


def _message(channel, *, content: str = "hello"):
    return SimpleNamespace(
        id=1234,
        content=content,
        mentions=[],
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=SimpleNamespace(id=42, display_name="member", name="member", bot=False),
        type=discord_platform.discord.MessageType.default,
    )


@pytest.fixture
def adapter(monkeypatch):
    for var in _GATE_VARS:
        monkeypatch.delenv(var, raising=False)
        # delenv on an ABSENT var records nothing, so a value written during a test would
        # leak past monkeypatch — scrub explicitly as well.
        os.environ.pop(var, None)

    monkeypatch.setattr(discord_platform, "discord", sys.modules["discord"], raising=False)
    monkeypatch.setattr(discord_platform, "DISCORD_AVAILABLE", True, raising=False)

    instance = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    instance._client = SimpleNamespace(user=SimpleNamespace(id=999))
    instance._gate_env_snapshot = None
    instance._text_batch_delay_seconds = 0
    instance.handle_message = AsyncMock()
    return instance


def _refusals(caplog, channel_id: int):
    """Every refusal line in the capture, as ``(record, refusal_type)`` for that channel.

    A line counts only when it carries BOTH facts: the channel id this message came from and
    a token from the adapter's own vocabulary.
    """
    found = []
    for record in caplog.records:
        if record.name != ADAPTER_LOGGER:
            continue
        text = record.getMessage()
        if "admission refused" not in text:
            continue
        type_match = _REFUSAL_TYPE_RE.search(text)
        id_match = _CHANNEL_ID_RE.search(text)
        if not type_match or not id_match:
            continue
        if id_match.group(1) != str(channel_id):
            continue
        found.append((record, type_match.group(1)))
    return found


def _assert_readable_type(token: str) -> None:
    """A token an operator can read and grep for — it must name the refusal, not the class."""
    assert re.fullmatch(r"[a-z][a-z0-9_]*", token), f"refusal type {token!r} is not a readable token"
    assert token not in {"refused", "denied", "ignored", "dropped", "blocked"}, (
        f"refusal type {token!r} names the class of the drop, not the reason for it"
    )


def _drop_the_message(adapter, caplog, message):
    """Drive the real refusal path with DEBUG captured, so a silent drop cannot hide.

    caplog is opened at DEBUG on purpose: if the line ever regressed to ``logger.debug`` the
    record is still captured and the level assertion below fails, instead of the test passing
    because nothing was logged at all.
    """
    caplog.set_level(logging.DEBUG, logger=ADAPTER_LOGGER)
    return asyncio.run(adapter._handle_message(message))


class TestTheTwoRefusingBranchesWriteATrace:
    def test_non_allowed_channel_drop_names_the_channel_and_its_type(self, adapter, caplog, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111")

        dropped = _drop_the_message(adapter, caplog, _message(_StubChannel(999)))

        assert dropped is False, "the gate must still refuse an unlisted channel"
        found = _refusals(caplog, 999)
        assert found, "a dropped message left no line naming its channel and refusal type"
        record, refusal_type = found[0]
        _assert_readable_type(refusal_type)
        assert record.levelno >= logging.INFO, (
            "the line is below gateway.log's handler level, so the operator never sees it"
        )

    def test_ignored_channel_drop_names_the_channel_and_its_type(self, adapter, caplog, monkeypatch):
        monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "999")

        dropped = _drop_the_message(adapter, caplog, _message(_StubChannel(999)))

        assert dropped is False, "the gate must still refuse an ignored channel"
        found = _refusals(caplog, 999)
        assert found, "a dropped message left no line naming its channel and refusal type"
        record, refusal_type = found[0]
        _assert_readable_type(refusal_type)
        assert record.levelno >= logging.INFO, (
            "the line is below gateway.log's handler level, so the operator never sees it"
        )

    def test_the_two_branches_are_told_apart_by_distinct_readable_types(self, adapter, caplog, monkeypatch):
        types = []

        # Branch 1: the channel is outside the allowlist.
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111")
        _drop_the_message(adapter, caplog, _message(_StubChannel(222)))
        outside_allowlist = _refusals(caplog, 222)
        assert outside_allowlist, "no refusal line for the non-allowed-channel branch"
        types.append(outside_allowlist[0][1])

        # Branch 2: the channel is admitted by the allowlist and ignored all the same.
        caplog.clear()
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "*")
        monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "999")
        _drop_the_message(adapter, caplog, _message(_StubChannel(999)))
        ignored = _refusals(caplog, 999)
        assert ignored, "no refusal line for the ignored-channel branch"
        types.append(ignored[0][1])

        assert types[0] != types[1], (
            "two different refusals share one token, so the line cannot name the reason"
        )
        for token in types:
            _assert_readable_type(token)

    def test_a_refusal_type_does_not_drift_between_identical_drops(self, adapter, caplog, monkeypatch):
        """An operator greps the token: the same refusal must not rename itself run to run."""
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111")
        seen = set()

        for _ in range(3):
            caplog.clear()
            _drop_the_message(adapter, caplog, _message(_StubChannel(222)))
            found = _refusals(caplog, 222)
            assert found, "no refusal line for the non-allowed-channel branch"
            seen.add(found[0][1])

        assert len(seen) == 1, f"the same refusal reported {sorted(seen)}"


class TestADroppedMessageCannotPassSilently:
    @pytest.mark.parametrize(
        "gate_env, channel_id",
        [
            ({"DISCORD_ALLOWED_CHANNELS": "111"}, 999),
            ({"DISCORD_IGNORED_CHANNELS": "999"}, 999),
        ],
    )
    def test_the_assertion_itself_is_the_guard(
        self, adapter, caplog, monkeypatch, gate_env, channel_id,
    ):
        for key, value in gate_env.items():
            monkeypatch.setenv(key, value)

        dropped = _drop_the_message(adapter, caplog, _message(_StubChannel(channel_id)))

        # Two readings, not one: the refusal happened AND it is readable. Drop either half
        # and this fails — a refusal that leaves no line is a failure, not a silent pass.
        assert dropped is False
        assert _refusals(caplog, channel_id), (
            f"message in channel {channel_id} was refused ({gate_env}) without leaving "
            "one line carrying the channel id and the refusal type"
        )

    def test_the_trace_marks_the_channel_that_was_actually_refused(self, adapter, caplog, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111")

        _drop_the_message(adapter, caplog, _message(_StubChannel(4242)))

        assert _refusals(caplog, 4242), "no line for the refused channel"
        assert not _refusals(caplog, 111), "a line names a channel the gate never refused"

    def test_the_line_lands_on_the_surface_the_platform_already_has(self, adapter, caplog, monkeypatch):
        """The trace must land on the operator surface that already exists.

        gateway.log is written by a handler at INFO, gated on ``COMPONENT_PREFIXES["gateway"]``
        — the very surface this adapter's own logger routes to. A line below that level, or on a
        logger outside that component, is a line no operator ever reads, which is exactly how
        the drop stayed invisible.
        """
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111")

        _drop_the_message(adapter, caplog, _message(_StubChannel(999)))

        gateway_surface = _ComponentFilter(COMPONENT_PREFIXES["gateway"])
        records = [record for record, _ in _refusals(caplog, 999)]
        assert records, "a dropped message left no line on the operator surface"
        for record in records:
            assert gateway_surface.filter(record), (
                f"{record.name} is outside the gateway component, so gateway.log drops the line"
            )
            assert record.levelno >= logging.INFO, (
                "gateway.log's handler level is INFO: a lower line never reaches the file"
            )


class TestTheAdmittedMessageCarriesNoRefusalTrace:
    def test_an_admitted_message_writes_no_refusal_line(self, adapter, caplog, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "*")
        monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "*")
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
        caplog.set_level(logging.DEBUG, logger=ADAPTER_LOGGER)

        admitted = asyncio.run(adapter._handle_message(_message(_StubChannel(999))))

        assert admitted is True, "the admitted arm must reach dispatch — the gate can still fall"
        assert not _refusals(caplog, 999), "an admitted message left a refusal line"
