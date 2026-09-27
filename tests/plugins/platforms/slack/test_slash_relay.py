"""Tests for the cross-profile Slack slash relay.

Covers the channel→profile resolver (scored claims from config.yaml +
channel_directory.json), the SQLite relay queue (exclusive claims, expiry,
purge), the adapter seam (forwarding decision, consumer execution,
unclaimed-row warning), the adapter lifecycle and slash delivery through a
mocked AsyncApp, and the ``slack.slash_relay`` kill switch loaded from a real
config.yaml.
"""

import asyncio
import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from types import SimpleNamespace

import pytest

from plugins.platforms.slack import slash_relay


# ---------------------------------------------------------------------------
# Fixtures: a fake hermes root with profile homes
# ---------------------------------------------------------------------------

def _write_profile(
    home,
    *,
    free_response: str = "",
    allowed: str = "",
    directory_channels=(),
):
    home.mkdir(parents=True, exist_ok=True)
    slack_cfg = {
        "require_mention": True,
        "free_response_channels": free_response,
        "allowed_channels": allowed,
    }
    cfg = home / "config.yaml"
    lines = ["slack:"]
    for k, v in slack_cfg.items():
        lines.append(f"  {k}: {json.dumps(v)}")
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    directory = {
        "platforms": {
            "slack": [{"id": cid, "name": cid, "type": "private"} for cid in directory_channels]
        }
    }
    (home / "channel_directory.json").write_text(
        json.dumps(directory), encoding="utf-8"
    )


@pytest.fixture()
def profiles_root(tmp_path):
    """A multi-profile install: default owns C_MAIN (explicit + observed),
    two profiles cloned from a template inherited C_MAIN in config only
    (explicit, no observed traffic), and alpha/beta own their channels."""
    root = tmp_path / "hermes"
    _write_profile(
        root,
        free_response="C_MAIN",
        directory_channels=("C_MAIN", "C_HEALTH", "D_DEF:171.1"),
    )
    _write_profile(
        root / "profiles" / "alpha",
        free_response="C_ALPHA",
        directory_channels=("C_ALPHA", "D_ALPHA"),
    )
    _write_profile(
        root / "profiles" / "beta",
        free_response="C_BETA",
        directory_channels=("C_BETA", "C_BETA:1781.2", "D_BETA"),
    )
    _write_profile(root / "profiles" / "tmpl_a", free_response="C_MAIN")
    _write_profile(root / "profiles" / "tmpl_b", free_response="C_MAIN")
    slash_relay.clear_owner_cache()
    yield root
    slash_relay.clear_owner_cache()


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class TestResolveChannelOwner:
    def test_unique_explicit_plus_observed_claim(self, profiles_root):
        assert slash_relay.resolve_channel_owner("C_ALPHA", profiles_root) == "alpha"
        assert slash_relay.resolve_channel_owner("C_BETA", profiles_root) == "beta"

    def test_explicit_plus_observed_beats_template_explicit_only(self, profiles_root):
        # Shared-channel case: default (score 3) vs the template clones (2 each).
        assert slash_relay.resolve_channel_owner("C_MAIN", profiles_root) == "default"

    def test_observed_only_claim_resolves(self, profiles_root):
        assert slash_relay.resolve_channel_owner("C_HEALTH", profiles_root) == "default"

    def test_dm_channels_resolve_to_their_bot_profile(self, profiles_root):
        assert slash_relay.resolve_channel_owner("D_ALPHA", profiles_root) == "alpha"
        assert slash_relay.resolve_channel_owner("D_BETA", profiles_root) == "beta"

    def test_thread_suffixed_directory_entries_collapse(self, profiles_root):
        # C_BETA appears as "C_BETA:1781.2" too; base id must still resolve
        # (and the query side also strips a :thread suffix).
        assert (
            slash_relay.resolve_channel_owner("C_BETA:999.9", profiles_root)
            == "beta"
        )

    def test_unknown_channel_is_unrouted(self, profiles_root):
        assert slash_relay.resolve_channel_owner("C_NOPE", profiles_root) is None

    def test_equal_claims_are_unrouted(self, tmp_path):
        root = tmp_path / "hermes"
        _write_profile(root, free_response="C_SHARED")
        _write_profile(root / "profiles" / "alpha", free_response="C_SHARED")
        slash_relay.clear_owner_cache()
        assert slash_relay.resolve_channel_owner("C_SHARED", root) is None

    def test_empty_channel_id_is_unrouted(self, profiles_root):
        assert slash_relay.resolve_channel_owner("", profiles_root) is None
        assert slash_relay.resolve_channel_owner(None, profiles_root) is None

    def test_malformed_config_and_directory_fail_open(self, tmp_path, caplog):
        root = tmp_path / "hermes"
        _write_profile(root / "profiles" / "alpha", free_response="C_ALPHA")
        (root / "config.yaml").write_text(": not yaml [", encoding="utf-8")
        (root / "channel_directory.json").write_text("{broken", encoding="utf-8")
        slash_relay.clear_owner_cache()
        with caplog.at_level("WARNING", logger=slash_relay.logger.name):
            # Broken default home doesn't poison the scan; alpha still resolves.
            assert slash_relay.resolve_channel_owner("C_ALPHA", root) == "alpha"
        # ...but the failure is not silent: each unreadable file is named.
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any(str(root / "config.yaml") in m for m in warnings), warnings
        assert any(str(root / "channel_directory.json") in m for m in warnings), warnings

    def test_missing_channel_directory_is_quiet(self, tmp_path, caplog):
        root = tmp_path / "hermes"
        home = root / "profiles" / "alpha"
        _write_profile(home, free_response="C_ALPHA")
        (home / "channel_directory.json").unlink()
        slash_relay.clear_owner_cache()
        with caplog.at_level("WARNING", logger=slash_relay.logger.name):
            assert slash_relay.resolve_channel_owner("C_ALPHA", root) == "alpha"
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_list_valued_channels_config(self, tmp_path):
        root = tmp_path / "hermes"
        home = root / "profiles" / "alpha"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "slack:\n  free_response_channels:\n    - C_A\n    - C_B\n",
            encoding="utf-8",
        )
        slash_relay.clear_owner_cache()
        assert slash_relay.resolve_channel_owner("C_B", root) == "alpha"

    def test_cache_ttl_respected(self, profiles_root):
        assert slash_relay.resolve_channel_owner("C_ALPHA", profiles_root) == "alpha"
        # Rewriting ownership is invisible until the TTL lapses…
        _write_profile(
            profiles_root / "profiles" / "alpha", directory_channels=()
        )
        assert slash_relay.resolve_channel_owner("C_ALPHA", profiles_root) == "alpha"
        # …and visible with a zero TTL.
        assert slash_relay.resolve_channel_owner("C_ALPHA", profiles_root, ttl_s=0.0) is None


# ---------------------------------------------------------------------------
# Relay queue
# ---------------------------------------------------------------------------

PAYLOAD = {
    "command": "/cron",
    "text": "list",
    "user_id": "U1",
    "channel_id": "C_BETA",
    "team_id": "T1",
    "response_url": "https://hooks.slack.com/commands/T1/1/xyz",
}


class TestRelayQueue:
    def test_enqueue_claim_done_roundtrip(self, tmp_path):
        row_id = slash_relay.enqueue("beta", "alpha", PAYLOAD, root=tmp_path)
        assert not slash_relay.is_claimed(row_id, root=tmp_path)
        rows = slash_relay.claim_pending("beta", root=tmp_path)
        assert [r["id"] for r in rows] == [row_id]
        assert rows[0]["payload"] == PAYLOAD
        assert slash_relay.is_claimed(row_id, root=tmp_path)
        slash_relay.mark_done(row_id, root=tmp_path)
        assert slash_relay.claim_pending("beta", root=tmp_path) == []

    def test_claims_are_exclusive(self, tmp_path):
        slash_relay.enqueue("beta", "alpha", PAYLOAD, root=tmp_path)
        first = slash_relay.claim_pending("beta", root=tmp_path)
        second = slash_relay.claim_pending("beta", root=tmp_path)
        assert len(first) == 1
        assert second == []

    def test_claim_filters_by_target_profile(self, tmp_path):
        slash_relay.enqueue("beta", "alpha", PAYLOAD, root=tmp_path)
        assert slash_relay.claim_pending("gamma", root=tmp_path) == []
        assert len(slash_relay.claim_pending("beta", root=tmp_path)) == 1

    def test_stale_rows_never_claimed(self, tmp_path):
        row_id = slash_relay.enqueue("beta", "alpha", PAYLOAD, root=tmp_path)
        assert (
            slash_relay.claim_pending("beta", root=tmp_path, max_age_s=0.0)
            == []
        )
        assert not slash_relay.is_claimed(row_id, root=tmp_path)

    def test_purge_removes_old_rows(self, tmp_path):
        slash_relay.enqueue("beta", "alpha", PAYLOAD, root=tmp_path)
        assert slash_relay.purge(root=tmp_path, keep_s=0.0) == 1
        assert slash_relay.claim_pending("beta", root=tmp_path) == []

    def test_is_claimed_missing_row(self, tmp_path):
        assert not slash_relay.is_claimed(12345, root=tmp_path)

    def test_db_is_owner_only(self, tmp_path):
        import os
        import stat

        old_umask = os.umask(0o022)
        try:
            slash_relay.enqueue("beta", "alpha", PAYLOAD, root=tmp_path)
        finally:
            os.umask(old_umask)
        for name in (slash_relay.DB_FILENAME, slash_relay.DB_FILENAME + "-wal"):
            path = tmp_path / name
            if path.exists():
                assert stat.S_IMODE(path.stat().st_mode) == 0o600, name

    def test_poll_purges_rows_older_than_a_day(self, tmp_path):
        import sqlite3
        from contextlib import closing

        old_id = slash_relay.enqueue("beta", "alpha", PAYLOAD, root=tmp_path)
        other_id = slash_relay.enqueue("gamma", "alpha", PAYLOAD, root=tmp_path)
        fresh_id = slash_relay.enqueue("gamma", "alpha", PAYLOAD, root=tmp_path)
        db = tmp_path / slash_relay.DB_FILENAME
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute(
                "UPDATE slash_relay SET created_at = created_at - ? WHERE id IN (?, ?)",
                (25 * 3600, old_id, other_id),
            )

        # A poll by ANY profile purges every expired row, whoever it was for.
        assert slash_relay.claim_pending("beta", root=tmp_path) == []

        with closing(sqlite3.connect(db)) as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM slash_relay")]
        assert ids == [fresh_id]


# ---------------------------------------------------------------------------
# Adapter seam
# ---------------------------------------------------------------------------

def _adapter_stub(profile: str = "alpha", extra: Optional[Dict[str, Any]] = None):
    """A bare object carrying just what the relay methods need."""
    from plugins.platforms.slack.adapter import SlackAdapter

    stub = MagicMock()
    stub.config.extra = extra or {}
    stub._slash_relay_profile = MagicMock(return_value=profile)
    stub._slash_relay_enabled = lambda: SlackAdapter._slash_relay_enabled(stub)
    stub._handle_slash_command = AsyncMock()
    stub._send_slash_ephemeral = AsyncMock()
    stub._warn_if_slash_unclaimed = AsyncMock()
    return stub


class _LoopStub:
    """Minimal non-mock stand-in for the consumer-loop tests: real
    ``_running`` property (True for exactly one iteration) without touching
    MagicMock class attributes."""

    def __init__(self):
        self._slash_relay_poll_s = 0.0
        self._runs = iter([True, False])
        self._handle_slash_command = AsyncMock()

    @property
    def _running(self):
        return next(self._runs)

    def _slash_relay_profile(self):
        return "alpha"


class TestForwardingDecision:
    def test_foreign_channel_resolves_to_owner(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="alpha")
        with patch.object(
            slash_relay, "resolve_channel_owner", return_value="beta"
        ):
            owner = SlackAdapter._resolve_foreign_slash_owner(stub, PAYLOAD)
        assert owner == "beta"

    def test_own_channel_is_not_foreign(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="beta")
        with patch.object(
            slash_relay, "resolve_channel_owner", return_value="beta"
        ):
            assert SlackAdapter._resolve_foreign_slash_owner(stub, PAYLOAD) is None

    def test_unknown_channel_is_not_foreign(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="alpha")
        with patch.object(slash_relay, "resolve_channel_owner", return_value=None):
            assert SlackAdapter._resolve_foreign_slash_owner(stub, PAYLOAD) is None

    def test_kill_switch_disables_forwarding(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="alpha", extra={"slash_relay": "false"})
        with patch.object(
            slash_relay, "resolve_channel_owner", return_value="beta"
        ):
            assert SlackAdapter._resolve_foreign_slash_owner(stub, PAYLOAD) is None

    def test_resolver_error_fails_open_to_local(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="alpha")
        with patch.object(
            slash_relay, "resolve_channel_owner", side_effect=RuntimeError("boom")
        ):
            assert SlackAdapter._resolve_foreign_slash_owner(stub, PAYLOAD) is None


class TestForwardAndWarn:
    def test_forward_enqueues_for_owner(self, tmp_path):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="alpha")
        with patch.object(slash_relay, "enqueue", MagicMock(return_value=7)) as enq:
            asyncio.run(
                SlackAdapter._forward_slash_to_profile(stub, "beta", PAYLOAD)
            )
        enq.assert_called_once_with("beta", "alpha", PAYLOAD)
        stub._handle_slash_command.assert_not_awaited()
        stub._warn_if_slash_unclaimed.assert_called_once()

    def test_forward_failure_falls_back_to_local(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="alpha")
        with patch.object(
            slash_relay, "enqueue", MagicMock(side_effect=RuntimeError("disk"))
        ):
            asyncio.run(
                SlackAdapter._forward_slash_to_profile(stub, "beta", PAYLOAD)
            )
        stub._handle_slash_command.assert_awaited_once_with(PAYLOAD)

    def test_unclaimed_row_warns_via_response_url(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="alpha")
        with patch.object(slash_relay, "is_claimed", MagicMock(return_value=False)):
            asyncio.run(
                SlackAdapter._warn_if_slash_unclaimed(
                    stub, "beta", PAYLOAD, 7, delay_s=0.0
                )
            )
        stub._send_slash_ephemeral.assert_awaited_once()
        ctx, text = stub._send_slash_ephemeral.await_args.args
        assert ctx["response_url"] == PAYLOAD["response_url"]
        assert "beta" in text and "/cron" in text

    def test_claimed_row_stays_silent(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="alpha")
        with patch.object(slash_relay, "is_claimed", MagicMock(return_value=True)):
            asyncio.run(
                SlackAdapter._warn_if_slash_unclaimed(
                    stub, "beta", PAYLOAD, 7, delay_s=0.0
                )
            )
        stub._send_slash_ephemeral.assert_not_awaited()


class TestConsumerLoop:
    def test_consumer_executes_payload_and_marks_done(self, tmp_path):
        """One full pass: enqueue → claim → _handle_slash_command → done."""
        from plugins.platforms.slack.adapter import SlackAdapter

        row_id = slash_relay.enqueue("alpha", "default", PAYLOAD, root=tmp_path)

        stub = _LoopStub()

        real_claim = slash_relay.claim_pending
        real_done = slash_relay.mark_done
        with patch.object(
            slash_relay,
            "claim_pending",
            lambda profile, **kw: real_claim(profile, root=tmp_path),
        ), patch.object(
            slash_relay,
            "mark_done",
            lambda rid, **kw: real_done(rid, root=tmp_path),
        ), patch.object(slash_relay, "purge", MagicMock(return_value=0)):
            asyncio.run(SlackAdapter._slash_relay_loop(stub))

        stub._handle_slash_command.assert_awaited_once_with(PAYLOAD, relayed=True)
        assert slash_relay.is_claimed(row_id, root=tmp_path)
        assert slash_relay.claim_pending("alpha", root=tmp_path) == []

    def test_handler_error_still_marks_done(self, tmp_path):
        from plugins.platforms.slack.adapter import SlackAdapter

        slash_relay.enqueue("alpha", "default", PAYLOAD, root=tmp_path)
        stub = _LoopStub()
        stub._handle_slash_command = AsyncMock(side_effect=RuntimeError("boom"))

        real_claim = slash_relay.claim_pending
        real_done = slash_relay.mark_done
        with patch.object(
            slash_relay,
            "claim_pending",
            lambda profile, **kw: real_claim(profile, root=tmp_path),
        ), patch.object(
            slash_relay,
            "mark_done",
            lambda rid, **kw: real_done(rid, root=tmp_path),
        ), patch.object(slash_relay, "purge", MagicMock(return_value=0)):
            asyncio.run(SlackAdapter._slash_relay_loop(stub))

        # Errors are contained and the row is not re-delivered forever.
        assert slash_relay.claim_pending("alpha", root=tmp_path) == []


class TestRelayedReplyAttribution:
    """Relayed slash replies must be attributed to the OWNING profile's app:
    posted with its own bot token (chat.postEphemeral), with response_url
    kept only as a delivery fallback."""

    def _ctx(self, **extra):
        return {
            "response_url": PAYLOAD["response_url"],
            "ts": 0.0,
            "via_bot": True,
            "user_id": "U1",
            "replace_original": False,
            **extra,
        }

    def _stub(self):
        stub = MagicMock()
        stub.format_message = lambda c: c
        stub.truncate_message = lambda c, _n: [c]
        stub.MAX_MESSAGE_LENGTH = 40000
        stub._send_slash_ephemeral = AsyncMock(return_value="fallback-result")
        return stub

    def test_reply_posts_via_own_bot_token(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = self._stub()
        client = MagicMock()
        client.chat_postEphemeral = AsyncMock()
        stub._get_client = MagicMock(return_value=client)

        result = asyncio.run(
            SlackAdapter._send_slash_ephemeral_via_bot(
                stub, "C_BETA", self._ctx(), "the jobs"
            )
        )
        client.chat_postEphemeral.assert_awaited_once_with(
            channel="C_BETA", user="U1", text="the jobs"
        )
        assert result.success
        stub._send_slash_ephemeral.assert_not_awaited()

    def test_post_failure_falls_back_to_response_url(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = self._stub()
        client = MagicMock()
        client.chat_postEphemeral = AsyncMock(side_effect=RuntimeError("nope"))
        stub._get_client = MagicMock(return_value=client)

        result = asyncio.run(
            SlackAdapter._send_slash_ephemeral_via_bot(
                stub, "C_BETA", self._ctx(), "the jobs"
            )
        )
        stub._send_slash_ephemeral.assert_awaited_once()
        assert result == "fallback-result"

    def test_missing_user_id_goes_straight_to_fallback(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = self._stub()
        result = asyncio.run(
            SlackAdapter._send_slash_ephemeral_via_bot(
                stub, "C_BETA", self._ctx(user_id=""), "the jobs"
            )
        )
        stub._send_slash_ephemeral.assert_awaited_once()
        assert result == "fallback-result"

    def test_no_route_at_all_reports_failure(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = self._stub()
        result = asyncio.run(
            SlackAdapter._send_slash_ephemeral_via_bot(
                stub, "C_BETA", {"via_bot": True, "user_id": ""}, "x"
            )
        )
        assert not result.success


class TestRelayedContextStash:
    def test_relayed_context_is_flagged_for_bot_token_reply(self):
        """A relayed context is flagged ``via_bot`` and opts out of
        ``replace_original``, so send() routes it through
        ``_send_slash_ephemeral_via_bot``."""
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = SimpleNamespace(
            _slash_command_contexts={}, _SLASH_CTX_MAX=64,
            _purge_stale_slash_contexts=lambda: None)
        SlackAdapter._stash_slash_context(
            stub, "T1", "C1", "U1", "https://hooks.example/x", relayed=True)
        ctx = stub._slash_command_contexts[("T1", "C1", "U1")]
        assert ctx["via_bot"] is True
        # The receiving app acked silently, so a fallback POST must create a new
        # message rather than replace a placeholder that does not exist.
        assert ctx["replace_original"] is False


# ---------------------------------------------------------------------------
# Adapter lifecycle and slash delivery through a mocked AsyncApp
# ---------------------------------------------------------------------------

class _ConnectedAdapter:
    """Run ``SlackAdapter.connect()`` / ``disconnect()`` against a mocked
    ``AsyncApp`` that records every ``@app.command`` registration.

    The Socket Mode handler and its watchdog are stubbed out; the slash relay
    consumer is left real so the tests observe the task the adapter starts.
    The relay queue itself is patched so nothing touches disk.
    """

    def __init__(self, extra: Optional[Dict[str, Any]] = None, profile: Optional[str] = "alpha"):
        import os

        import plugins.platforms.slack.adapter as slack_mod
        from gateway.config import PlatformConfig

        self.slack_mod = slack_mod
        # slack-bolt is optional; the adapter only needs the names the
        # patches below replace.
        self.adapter = slack_mod.SlackAdapter(
            PlatformConfig(enabled=True, token="xoxb-fake", extra=dict(extra or {}))
        )
        self.adapter._slash_relay_poll_s = 0.0
        if profile is not None:
            self.adapter._slash_relay_profile = lambda: profile
        self.adapter._start_socket_mode_handler = MagicMock()
        self.adapter._ensure_socket_watchdog = MagicMock()
        self.commands: list = []

        def _record_command(matcher):
            def decorator(fn):
                self.commands.append((matcher, fn))
                return fn
            return decorator

        def _passthrough(*_a, **_kw):
            def decorator(fn):
                return fn
            return decorator

        app = MagicMock()
        app.command = _record_command
        app.event = _passthrough
        app.action = _passthrough
        app.view = _passthrough
        app.options = _passthrough
        app.shortcut = _passthrough
        app.client = AsyncMock()

        web_client = AsyncMock()
        web_client.auth_test = AsyncMock(return_value={
            "user_id": "U_BOT", "user": "testbot", "team_id": "T1", "team": "Team",
        })
        plugin_mgr = MagicMock()
        plugin_mgr.get_slack_action_handlers.return_value = []

        self.claim_pending = MagicMock(return_value=[])
        self._patches = [
            patch.object(slack_mod, "SLACK_AVAILABLE", True),
            patch.object(slack_mod, "AsyncApp", MagicMock(return_value=app), create=True),
            patch.object(slack_mod, "AsyncWebClient", MagicMock(return_value=web_client), create=True),
            patch.object(slack_mod, "AsyncSocketModeHandler", MagicMock(), create=True),
            patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}),
            patch("gateway.status.acquire_scoped_lock", return_value=(True, None)),
            patch("gateway.status.release_scoped_lock"),
            patch("hermes_cli.plugins.get_plugin_manager", return_value=plugin_mgr),
            patch.object(slash_relay, "claim_pending", self.claim_pending),
            patch.object(slash_relay, "purge", MagicMock(return_value=0)),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False

    def slash_handler(self):
        """The single native-slash handler the adapter registered."""
        assert len(self.commands) == 1, self.commands
        return self.commands[0][1]


def _native_slash() -> str:
    from hermes_cli.commands_platforms import slack_native_slashes

    return "/" + slack_native_slashes()[0][0]


class TestAdapterLifecycle:
    def test_connect_registers_one_native_slash_matcher(self):
        from hermes_cli.commands_platforms import slack_native_slashes

        async def scenario(h):
            assert await h.adapter.connect() is True
            try:
                assert len(h.commands) == 1
                matcher = h.commands[0][0]
                for name, _desc, _hint in slack_native_slashes():
                    assert matcher.match("/" + name), name
                assert not matcher.match("/not-a-hermes-command")
            finally:
                await h.adapter.disconnect()

        with _ConnectedAdapter() as h:
            asyncio.run(scenario(h))

    def test_connect_starts_relay_consumer_and_disconnect_cancels_it(self):
        async def scenario(h):
            assert await h.adapter.connect() is True
            task = h.adapter._slash_relay_task
            assert task is not None and not task.done()
            # Let the consumer run a poll: it claims rows for THIS profile.
            for _ in range(5):
                await asyncio.sleep(0)
            h.claim_pending.assert_called_with("alpha")

            await h.adapter.disconnect()
            assert h.adapter._slash_relay_task is None
            assert task.cancelled()

        with _ConnectedAdapter() as h:
            asyncio.run(scenario(h))

    def test_kill_switch_starts_no_relay_consumer(self):
        async def scenario(h):
            assert await h.adapter.connect() is True
            try:
                assert h.adapter._slash_relay_task is None
                h.claim_pending.assert_not_called()
            finally:
                await h.adapter.disconnect()

        with _ConnectedAdapter(extra={"slash_relay": False}) as h:
            asyncio.run(scenario(h))


class TestSlashDelivery:
    """Drive the registered ``@app.command`` handler with a Slack payload."""

    def _payload(self, channel: str) -> Dict[str, Any]:
        return {**PAYLOAD, "command": _native_slash(), "channel_id": channel}

    def _run(self, h, payload, owner):
        enqueue = MagicMock(return_value=7)
        ack = AsyncMock()

        async def scenario():
            assert await h.adapter.connect() is True
            try:
                h.adapter._handle_slash_command = AsyncMock()
                h.adapter._warn_if_slash_unclaimed = AsyncMock()
                with patch.object(
                    slash_relay, "resolve_channel_owner", return_value=owner
                ), patch.object(slash_relay, "enqueue", enqueue):
                    await h.slash_handler()(ack=ack, command=payload)
                    # Let the fire-and-forget unclaimed-warning task start.
                    await asyncio.sleep(0)
            finally:
                await h.adapter.disconnect()

        asyncio.run(scenario())
        return ack, enqueue

    def test_foreign_channel_payload_is_enqueued_not_answered_locally(self):
        payload = self._payload("C_BETA")
        with _ConnectedAdapter(profile="alpha") as h:
            ack, enqueue = self._run(h, payload, owner="beta")
            # Silent ack: any text would be attributed to the receiving app.
            ack.assert_awaited_once_with()
            enqueue.assert_called_once_with("beta", "alpha", payload)
            h.adapter._handle_slash_command.assert_not_awaited()
            h.adapter._warn_if_slash_unclaimed.assert_awaited_once_with("beta", payload, 7)

    def test_owned_channel_payload_is_answered_locally(self):
        payload = self._payload("C_ALPHA")
        with _ConnectedAdapter(profile="alpha") as h:
            ack, enqueue = self._run(h, payload, owner="alpha")
            ack.assert_awaited_once()
            assert ack.await_args.kwargs.get("response_type") == "ephemeral"
            enqueue.assert_not_called()
            h.adapter._handle_slash_command.assert_awaited_once_with(payload)

    def test_kill_switch_answers_foreign_channel_locally(self):
        payload = self._payload("C_BETA")
        with _ConnectedAdapter(extra={"slash_relay": False}, profile="alpha") as h:
            ack, enqueue = self._run(h, payload, owner="beta")
            enqueue.assert_not_called()
            h.adapter._handle_slash_command.assert_awaited_once_with(payload)


# ---------------------------------------------------------------------------
# Kill switch through a real config.yaml load
# ---------------------------------------------------------------------------

class TestKillSwitchConfig:
    @pytest.fixture()
    def hermes_home(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        # The YAML hook bridges to SLACK_SLASH_RELAY; record it so the
        # monkeypatch undo removes whatever the load writes.
        monkeypatch.setenv("SLACK_SLASH_RELAY", "")
        monkeypatch.delenv("SLACK_SLASH_RELAY")
        return home

    def _load_adapter(self):
        from gateway.config import Platform, load_gateway_config
        from plugins.platforms.slack.adapter import SlackAdapter

        config = load_gateway_config()
        return SlackAdapter(config.platforms[Platform.SLACK])

    @pytest.mark.parametrize("yaml_text", [
        "slack:\n  slash_relay: false\n",
        "platforms:\n  slack:\n    slash_relay: false\n",
    ], ids=["top-level", "nested"])
    def test_slash_relay_false_in_config_yaml_disables_relay(self, hermes_home, yaml_text):
        (hermes_home / "config.yaml").write_text(yaml_text, encoding="utf-8")

        adapter = self._load_adapter()

        assert adapter.config.extra["slash_relay"] is False
        assert adapter._slash_relay_enabled() is False

    def test_relay_is_on_when_config_yaml_is_silent(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "slack:\n  require_mention: true\n", encoding="utf-8")

        adapter = self._load_adapter()

        assert "slash_relay" not in adapter.config.extra
        assert adapter._slash_relay_enabled() is True

    def test_env_var_disables_relay(self, hermes_home, monkeypatch):
        (hermes_home / "config.yaml").write_text("slack: {}\n", encoding="utf-8")
        monkeypatch.setenv("SLACK_SLASH_RELAY", "false")

        adapter = self._load_adapter()

        assert adapter._slash_relay_enabled() is False

    def test_slack_yaml_hook_bridges_slash_relay(self, hermes_home):
        """The Slack ``apply_yaml_config_fn`` declares the key: it returns it for
        ``PlatformConfig.extra`` and bridges it to ``SLACK_SLASH_RELAY``."""
        import os

        from plugins.platforms.slack.adapter import _apply_yaml_config

        seeded = _apply_yaml_config({}, {"slash_relay": False})

        assert seeded == {"slash_relay": False}
        assert os.environ.get("SLACK_SLASH_RELAY") == "false"


class TestWarningTaskLifecycle:
    def test_pending_unclaimed_warning_is_cancelled_on_disconnect(self):
        payload = {**PAYLOAD, "command": _native_slash(), "channel_id": "C_BETA"}

        async def scenario(h):
            assert await h.adapter.connect() is True
            with patch.object(
                slash_relay, "resolve_channel_owner", return_value="beta"
            ), patch.object(slash_relay, "enqueue", MagicMock(return_value=7)):
                await h.slash_handler()(ack=AsyncMock(), command=payload)
            # The real warning coroutine sleeps 20 s before checking the row.
            tasks = set(h.adapter._slash_relay_warn_tasks)
            assert len(tasks) == 1
            await h.adapter.disconnect()
            (task,) = tasks
            assert task.cancelled()
            assert h.adapter._slash_relay_warn_tasks == set()

        with _ConnectedAdapter() as h:
            asyncio.run(scenario(h))


class TestMultiplexedProfile:
    """Under a multiplexed gateway, a secondary profile's adapter connects
    inside that profile's ``HERMES_HOME`` override (a ContextVar), not the
    process's ``HERMES_HOME``. The adapter must take its identity from that
    scope and keep it after the scope is gone (Bolt dispatches later)."""

    def test_profile_comes_from_the_home_override_contextvar(self, tmp_path, monkeypatch):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        root = tmp_path / "hermes"
        (root / "profiles" / "beta").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(root))
        for name in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
            monkeypatch.delenv(name, raising=False)
        payload = {**PAYLOAD, "command": _native_slash(), "channel_id": "C_GAMMA"}
        enqueue = MagicMock(return_value=7)

        async def connect_in_scope(h):
            token = set_hermes_home_override(str(root / "profiles" / "beta"))
            try:
                assert await h.adapter.connect() is True
                for _ in range(5):
                    await asyncio.sleep(0)
            finally:
                reset_hermes_home_override(token)

        async def scenario(h):
            # Outside any override the process home is the root: "default".
            assert h.adapter._resolve_slash_relay_profile() == "default"
            await connect_in_scope(h)
            try:
                h.claim_pending.assert_called_with("beta")
                h.adapter._warn_if_slash_unclaimed = AsyncMock()
                # Dispatch OUTSIDE the scope: the captured identity still holds.
                with patch.object(
                    slash_relay, "resolve_channel_owner", return_value="gamma"
                ), patch.object(slash_relay, "enqueue", enqueue):
                    await h.slash_handler()(ack=AsyncMock(), command=payload)
            finally:
                await h.adapter.disconnect()

        with _ConnectedAdapter(profile=None) as h:
            asyncio.run(scenario(h))
        enqueue.assert_called_once_with("gamma", "beta", payload)
