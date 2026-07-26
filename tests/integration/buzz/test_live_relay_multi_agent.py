"""Live-relay proof: N agents, N Buzz identities, one gateway process.

Drives TWO real ``BuzzAdapter`` connections (one per agent, each with its
own freshly minted Nostr key) against a real Buzz relay, plus a third
"poster" identity playing the human:

1. The poster creates a fresh channel; both agents join and watch it.
2. The poster @mentions agent A in the channel → agent A's connection
   dispatches the event stamped ``agent_id="agent_a"``; agent B's
   connection stays silent (its own mention gate, its own identity).
3. The poster DMs agent B → only agent B dispatches (chat_type="dm",
   ``agent_id="agent_b"``) via the p-tag DM classification (block/buzz#2897
   workaround), and B's REPLY is verified on the relay as authored by B's
   own pubkey — each agent is its own workspace member end to end.

Skip-unless-env (CI-safe): the test runs only when ALL of these are set —

    BUZZ_RELAY_URL             relay URL (e.g. ws://myhost:3000)
    BUZZ_TEST_POSTER_NSEC      throwaway key for the human-role poster
    BUZZ_TEST_AGENT_A_NSEC     throwaway key for agent A
    BUZZ_TEST_AGENT_B_NSEC     throwaway key for agent B

— and the ``buzz`` CLI binary AND ``nak`` (https://github.com/fiatjaf/nak)
are resolvable. All three keys must already be members of the relay (closed
relays: ``buzz-admin add-member --pubkey …``). Use FRESH throwaway keys,
never production identities. The test is also marked ``integration`` so
default CI runs (-m 'not integration') never collect it.

Why nak for the DM leg: the Buzz desktop client stamps every DM message
with a structural ``["p", <recipient>]`` tag — that tag is what the
adapter's #2897 DM classification keys on. ``buzz messages send`` does NOT
add it (see the "CLI-sent DMs carry no p-tag" note in the PR), so the test
publishes the desktop-shaped event with nak to exercise the real workaround
path end to end.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from unittest.mock import AsyncMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")
BuzzAdapter = _buzz_mod.BuzzAdapter

from gateway.config import PlatformConfig

_REQUIRED_ENV = (
    "BUZZ_RELAY_URL",
    "BUZZ_TEST_POSTER_NSEC",
    "BUZZ_TEST_AGENT_A_NSEC",
    "BUZZ_TEST_AGENT_B_NSEC",
)

_missing = [v for v in _REQUIRED_ENV if not os.getenv(v, "").strip()]
_cli = shutil.which("buzz") or (
    str(Path.home() / ".local" / "bin" / "buzz")
    if (Path.home() / ".local" / "bin" / "buzz").is_file()
    else None
)
_nak = shutil.which("nak")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        bool(_missing) or _cli is None or _nak is None,
        reason=(
            "live Buzz relay test needs env "
            + ", ".join(_REQUIRED_ENV)
            + " plus the buzz and nak CLI binaries"
        ),
    ),
]

_POLL_INTERVAL = 1.0
_WAIT_TIMEOUT = 30.0


def _run_buzz(key_env: str, *args: str, timeout: float = 30.0) -> dict | list:
    """Run the buzz CLI as the identity held in *key_env* (name, not value).

    The key travels via the subprocess environment only — never argv, never
    logs. Raises on nonzero exit with the CLI's (secret-free) stderr.
    """
    env = os.environ.copy()
    env["BUZZ_PRIVATE_KEY"] = os.environ[key_env]
    proc = subprocess.run(
        [_cli, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"buzz {' '.join(args[:2])} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()}"
        )
    try:
        return json.loads(proc.stdout or "null")
    except ValueError:
        return {}


def _make_agent_adapter(agent_id: str, key_env: str, channel: str) -> BuzzAdapter:
    """Build a per-agent adapter exactly as gateway/agent_platforms.py does:
    credential by NAME (private_key_env), identity-scoped settings, and
    connection identity as the routing default."""
    cfg = PlatformConfig(
        enabled=True,
        extra={
            "agent_id": agent_id,
            "private_key_env": key_env,
            "channels": [channel],
            "poll_interval": _POLL_INTERVAL,
            "require_mention": True,
        },
    )
    adapter = BuzzAdapter(cfg)
    adapter.set_routing_context(routes=[], default_agent=agent_id)
    adapter._message_handler = AsyncMock()
    adapter._events = []

    async def _capture_handle_message(event):
        adapter._attach_agent_id(event)  # the real routing stamp
        adapter._events.append(event)

    adapter.handle_message = _capture_handle_message
    return adapter


async def _wait_for(predicate, timeout: float = _WAIT_TIMEOUT, step: float = 0.5):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = predicate()
        if result:
            return result
        await asyncio.sleep(step)
    return None


@pytest.mark.asyncio
async def test_two_agents_two_identities_one_process():
    nonce = uuid.uuid4().hex[:8]
    poster_env = "BUZZ_TEST_POSTER_NSEC"
    a_env = "BUZZ_TEST_AGENT_A_NSEC"
    b_env = "BUZZ_TEST_AGENT_B_NSEC"

    # ── Arrange: profiles, fresh channel, memberships ─────────────────────
    name_a = f"WireA{nonce}"
    name_b = f"WireB{nonce}"
    _run_buzz(poster_env, "users", "set-profile", "--name", f"Poster{nonce}")
    _run_buzz(a_env, "users", "set-profile", "--name", name_a)
    _run_buzz(b_env, "users", "set-profile", "--name", name_b)

    created = _run_buzz(
        poster_env,
        "channels", "create",
        "--name", f"hermes-ma-{nonce}",
        "--type", "stream",
        "--visibility", "open",
    )
    channel = created["channel_id"]
    _run_buzz(a_env, "channels", "join", "--channel", channel)
    _run_buzz(b_env, "channels", "join", "--channel", channel)

    a = _make_agent_adapter("agent_a", a_env, channel)
    b = _make_agent_adapter("agent_b", b_env, channel)
    try:
        assert await a.connect(), "agent A failed to connect to the live relay"
        assert await b.connect(), "agent B failed to connect to the live relay"
        # Each connection learned its OWN identity from the relay.
        assert a._display_name == name_a
        assert b._display_name == name_b
        assert a._self_pubkey != b._self_pubkey

        # ── Act 1: channel message mentioning agent A only ────────────────
        _run_buzz(
            poster_env,
            "messages", "send",
            "--channel", channel,
            "--content", f"@{name_a} ping-{nonce}",
        )
        got_a = await _wait_for(
            lambda: [e for e in a._events if nonce in e.text]
        )
        assert got_a, "agent A never received its channel mention"
        event_a = got_a[0]
        assert event_a.source.agent_id == "agent_a"
        assert event_a.source.chat_id == channel
        assert event_a.text == f"ping-{nonce}"  # leading @mention stripped

        # ── Act 2: DM addressed to agent B only ───────────────────────────
        opened = _run_buzz(poster_env, "dms", "open", "--pubkey", b._self_pubkey)
        dm_id = opened.get("dm_id") or opened.get("channel_id")
        assert dm_id, f"dms open returned no conversation id: {opened!r}"
        # Publish the desktop-client-shaped DM message: kind 9, h-tag for the
        # conversation, structural p-tag for the recipient, NO visible
        # mention — the exact shape the adapter's #2897 DM classifier keys
        # on. Secret travels via env (NOSTR_SECRET_KEY), never argv.
        nak_env = os.environ.copy()
        nak_env["NOSTR_SECRET_KEY"] = os.environ[poster_env]
        relay_url = os.environ["BUZZ_RELAY_URL"]
        proc = subprocess.run(
            [
                _nak, "event",
                "-k", "9",
                "-t", f"h={dm_id}",
                "-t", f"p={b._self_pubkey}",
                "-c", f"dm-{nonce} for B's eyes only",
                "--auth", relay_url,
            ],
            capture_output=True,
            text=True,
            env=nak_env,
            timeout=30,
        )
        assert proc.returncode == 0, f"nak publish failed: {proc.stderr.strip()}"
        got_b = await _wait_for(
            lambda: [e for e in b._events if f"dm-{nonce}" in e.text]
        )
        assert got_b, "agent B never received its DM"
        event_b = got_b[0]
        assert event_b.source.agent_id == "agent_b"
        assert event_b.source.chat_type == "dm"

        # ── Assert isolation both ways ────────────────────────────────────
        assert not [e for e in b._events if f"ping-{nonce}" in e.text], (
            "agent B dispatched a channel message addressed to agent A"
        )
        assert not [e for e in a._events if f"dm-{nonce}" in e.text], (
            "agent A dispatched a DM addressed to agent B"
        )

        # ── Act 3: B replies in the DM as ITSELF ──────────────────────────
        sent = await b.send(dm_id, f"ack-{nonce}")
        assert sent.success, f"agent B reply failed: {sent.error}"
        dm_events = _run_buzz(poster_env, "messages", "get", "--channel", dm_id,
                              "--limit", "20")
        acks = [
            e for e in dm_events
            if isinstance(e, dict) and f"ack-{nonce}" in str(e.get("content", ""))
        ]
        assert acks, "agent B's reply not visible on the relay"
        assert acks[0]["pubkey"].lower() == b._self_pubkey, (
            "reply was not authored by agent B's own member identity"
        )
    finally:
        await a.disconnect()
        await b.disconnect()
        try:
            _run_buzz(poster_env, "channels", "archive", "--channel", channel)
        except Exception:
            pass
