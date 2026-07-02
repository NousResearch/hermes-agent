"""Tests for the cross-profile Slack slash relay.

Covers the channel→profile resolver (scored claims from config.yaml +
channel_directory.json), the SQLite relay queue (exclusive claims, expiry,
purge), and the adapter seam (forwarding decision, consumer execution,
unclaimed-row warning).
"""

import asyncio
import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

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
def fleet_root(tmp_path):
    """Mimic Justin's fleet: default owns #olympus (explicit + observed),
    media/orchestrator inherited #olympus in template configs (explicit
    only, no observed traffic), crypto/strategist own their channels."""
    root = tmp_path / "hermes"
    _write_profile(
        root,
        free_response="C_OLY",
        directory_channels=("C_OLY", "C_HEALTH", "D_DEF:171.1"),
    )
    _write_profile(
        root / "profiles" / "crypto",
        free_response="C_CRY",
        directory_channels=("C_CRY", "D_CRY"),
    )
    _write_profile(
        root / "profiles" / "strategist",
        free_response="C_STR",
        directory_channels=("C_STR", "C_STR:1781.2", "D_STR"),
    )
    _write_profile(root / "profiles" / "media", free_response="C_OLY")
    _write_profile(root / "profiles" / "orchestrator", free_response="C_OLY")
    slash_relay.clear_owner_cache()
    yield root
    slash_relay.clear_owner_cache()


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class TestResolveChannelOwner:
    def test_unique_explicit_plus_observed_claim(self, fleet_root):
        assert slash_relay.resolve_channel_owner("C_CRY", fleet_root) == "crypto"
        assert slash_relay.resolve_channel_owner("C_STR", fleet_root) == "strategist"

    def test_explicit_plus_observed_beats_template_explicit_only(self, fleet_root):
        # The #olympus case: default (score 3) vs media/orchestrator (2 each).
        assert slash_relay.resolve_channel_owner("C_OLY", fleet_root) == "default"

    def test_observed_only_claim_resolves(self, fleet_root):
        assert slash_relay.resolve_channel_owner("C_HEALTH", fleet_root) == "default"

    def test_dm_channels_resolve_to_their_bot_profile(self, fleet_root):
        assert slash_relay.resolve_channel_owner("D_CRY", fleet_root) == "crypto"
        assert slash_relay.resolve_channel_owner("D_STR", fleet_root) == "strategist"

    def test_thread_suffixed_directory_entries_collapse(self, fleet_root):
        # C_STR appears as "C_STR:1781.2" too; base id must still resolve
        # (and the query side also strips a :thread suffix).
        assert (
            slash_relay.resolve_channel_owner("C_STR:999.9", fleet_root)
            == "strategist"
        )

    def test_unknown_channel_is_unrouted(self, fleet_root):
        assert slash_relay.resolve_channel_owner("C_NOPE", fleet_root) is None

    def test_equal_claims_are_unrouted(self, tmp_path):
        root = tmp_path / "hermes"
        _write_profile(root, free_response="C_SHARED")
        _write_profile(root / "profiles" / "crypto", free_response="C_SHARED")
        slash_relay.clear_owner_cache()
        assert slash_relay.resolve_channel_owner("C_SHARED", root) is None

    def test_empty_channel_id_is_unrouted(self, fleet_root):
        assert slash_relay.resolve_channel_owner("", fleet_root) is None
        assert slash_relay.resolve_channel_owner(None, fleet_root) is None

    def test_malformed_config_and_directory_fail_open(self, tmp_path):
        root = tmp_path / "hermes"
        _write_profile(root / "profiles" / "crypto", free_response="C_CRY")
        (root / "config.yaml").write_text(": not yaml [", encoding="utf-8")
        (root / "channel_directory.json").write_text("{broken", encoding="utf-8")
        slash_relay.clear_owner_cache()
        # Broken default home doesn't poison the scan; crypto still resolves.
        assert slash_relay.resolve_channel_owner("C_CRY", root) == "crypto"

    def test_list_valued_channels_config(self, tmp_path):
        root = tmp_path / "hermes"
        home = root / "profiles" / "crypto"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "slack:\n  free_response_channels:\n    - C_A\n    - C_B\n",
            encoding="utf-8",
        )
        slash_relay.clear_owner_cache()
        assert slash_relay.resolve_channel_owner("C_B", root) == "crypto"

    def test_cache_ttl_respected(self, fleet_root):
        assert slash_relay.resolve_channel_owner("C_CRY", fleet_root) == "crypto"
        # Rewriting ownership is invisible until the TTL lapses…
        _write_profile(
            fleet_root / "profiles" / "crypto", directory_channels=()
        )
        assert slash_relay.resolve_channel_owner("C_CRY", fleet_root) == "crypto"
        # …and visible with a zero TTL.
        assert slash_relay.resolve_channel_owner("C_CRY", fleet_root, ttl_s=0.0) is None


# ---------------------------------------------------------------------------
# Relay queue
# ---------------------------------------------------------------------------

PAYLOAD = {
    "command": "/cron",
    "text": "list",
    "user_id": "U1",
    "channel_id": "C_STR",
    "team_id": "T1",
    "response_url": "https://hooks.slack.com/commands/T1/1/xyz",
}


class TestRelayQueue:
    def test_enqueue_claim_done_roundtrip(self, tmp_path):
        row_id = slash_relay.enqueue("strategist", "crypto", PAYLOAD, root=tmp_path)
        assert not slash_relay.is_claimed(row_id, root=tmp_path)
        rows = slash_relay.claim_pending("strategist", root=tmp_path)
        assert [r["id"] for r in rows] == [row_id]
        assert rows[0]["payload"] == PAYLOAD
        assert slash_relay.is_claimed(row_id, root=tmp_path)
        slash_relay.mark_done(row_id, root=tmp_path)
        assert slash_relay.claim_pending("strategist", root=tmp_path) == []

    def test_claims_are_exclusive(self, tmp_path):
        slash_relay.enqueue("strategist", "crypto", PAYLOAD, root=tmp_path)
        first = slash_relay.claim_pending("strategist", root=tmp_path)
        second = slash_relay.claim_pending("strategist", root=tmp_path)
        assert len(first) == 1
        assert second == []

    def test_claim_filters_by_target_profile(self, tmp_path):
        slash_relay.enqueue("strategist", "crypto", PAYLOAD, root=tmp_path)
        assert slash_relay.claim_pending("x-expert", root=tmp_path) == []
        assert len(slash_relay.claim_pending("strategist", root=tmp_path)) == 1

    def test_stale_rows_never_claimed(self, tmp_path):
        row_id = slash_relay.enqueue("strategist", "crypto", PAYLOAD, root=tmp_path)
        assert (
            slash_relay.claim_pending("strategist", root=tmp_path, max_age_s=0.0)
            == []
        )
        assert not slash_relay.is_claimed(row_id, root=tmp_path)

    def test_purge_removes_old_rows(self, tmp_path):
        slash_relay.enqueue("strategist", "crypto", PAYLOAD, root=tmp_path)
        assert slash_relay.purge(root=tmp_path, keep_s=0.0) == 1
        assert slash_relay.claim_pending("strategist", root=tmp_path) == []

    def test_is_claimed_missing_row(self, tmp_path):
        assert not slash_relay.is_claimed(12345, root=tmp_path)


# ---------------------------------------------------------------------------
# Adapter seam
# ---------------------------------------------------------------------------

def _adapter_stub(profile: str = "crypto", extra: Optional[Dict[str, Any]] = None):
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
        self._last_slash_relay_purge = 0.0
        self._runs = iter([True, False])
        self._handle_slash_command = AsyncMock()

    @property
    def _running(self):
        return next(self._runs)

    def _slash_relay_profile(self):
        return "crypto"


class TestForwardingDecision:
    def test_foreign_channel_resolves_to_owner(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="crypto")
        with patch.object(
            slash_relay, "resolve_channel_owner", return_value="strategist"
        ):
            owner = SlackAdapter._resolve_foreign_slash_owner(stub, PAYLOAD)
        assert owner == "strategist"

    def test_own_channel_is_not_foreign(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="strategist")
        with patch.object(
            slash_relay, "resolve_channel_owner", return_value="strategist"
        ):
            assert SlackAdapter._resolve_foreign_slash_owner(stub, PAYLOAD) is None

    def test_unknown_channel_is_not_foreign(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="crypto")
        with patch.object(slash_relay, "resolve_channel_owner", return_value=None):
            assert SlackAdapter._resolve_foreign_slash_owner(stub, PAYLOAD) is None

    def test_kill_switch_disables_forwarding(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="crypto", extra={"slash_relay": "false"})
        with patch.object(
            slash_relay, "resolve_channel_owner", return_value="strategist"
        ):
            assert SlackAdapter._resolve_foreign_slash_owner(stub, PAYLOAD) is None

    def test_resolver_error_fails_open_to_local(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="crypto")
        with patch.object(
            slash_relay, "resolve_channel_owner", side_effect=RuntimeError("boom")
        ):
            assert SlackAdapter._resolve_foreign_slash_owner(stub, PAYLOAD) is None


class TestForwardAndWarn:
    def test_forward_enqueues_for_owner(self, tmp_path):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="crypto")
        with patch.object(slash_relay, "enqueue", MagicMock(return_value=7)) as enq:
            asyncio.run(
                SlackAdapter._forward_slash_to_profile(stub, "strategist", PAYLOAD)
            )
        enq.assert_called_once_with("strategist", "crypto", PAYLOAD)
        stub._handle_slash_command.assert_not_awaited()
        stub._warn_if_slash_unclaimed.assert_called_once()

    def test_forward_failure_falls_back_to_local(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="crypto")
        with patch.object(
            slash_relay, "enqueue", MagicMock(side_effect=RuntimeError("disk"))
        ):
            asyncio.run(
                SlackAdapter._forward_slash_to_profile(stub, "strategist", PAYLOAD)
            )
        stub._handle_slash_command.assert_awaited_once_with(PAYLOAD)

    def test_unclaimed_row_warns_via_response_url(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="crypto")
        with patch.object(slash_relay, "is_claimed", MagicMock(return_value=False)):
            asyncio.run(
                SlackAdapter._warn_if_slash_unclaimed(
                    stub, "strategist", PAYLOAD, 7, delay_s=0.0
                )
            )
        stub._send_slash_ephemeral.assert_awaited_once()
        ctx, text = stub._send_slash_ephemeral.await_args.args
        assert ctx["response_url"] == PAYLOAD["response_url"]
        assert "strategist" in text and "/cron" in text

    def test_claimed_row_stays_silent(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = _adapter_stub(profile="crypto")
        with patch.object(slash_relay, "is_claimed", MagicMock(return_value=True)):
            asyncio.run(
                SlackAdapter._warn_if_slash_unclaimed(
                    stub, "strategist", PAYLOAD, 7, delay_s=0.0
                )
            )
        stub._send_slash_ephemeral.assert_not_awaited()


class TestConsumerLoop:
    def test_consumer_executes_payload_and_marks_done(self, tmp_path):
        """One full pass: enqueue → claim → _handle_slash_command → done."""
        from plugins.platforms.slack.adapter import SlackAdapter

        row_id = slash_relay.enqueue("crypto", "default", PAYLOAD, root=tmp_path)

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
        assert slash_relay.claim_pending("crypto", root=tmp_path) == []

    def test_handler_error_still_marks_done(self, tmp_path):
        from plugins.platforms.slack.adapter import SlackAdapter

        slash_relay.enqueue("crypto", "default", PAYLOAD, root=tmp_path)
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
        assert slash_relay.claim_pending("crypto", root=tmp_path) == []


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
                stub, "C_STR", self._ctx(), "the jobs"
            )
        )
        client.chat_postEphemeral.assert_awaited_once_with(
            channel="C_STR", user="U1", text="the jobs"
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
                stub, "C_STR", self._ctx(), "the jobs"
            )
        )
        stub._send_slash_ephemeral.assert_awaited_once()
        assert result == "fallback-result"

    def test_missing_user_id_goes_straight_to_fallback(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = self._stub()
        result = asyncio.run(
            SlackAdapter._send_slash_ephemeral_via_bot(
                stub, "C_STR", self._ctx(user_id=""), "the jobs"
            )
        )
        stub._send_slash_ephemeral.assert_awaited_once()
        assert result == "fallback-result"

    def test_no_route_at_all_reports_failure(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        stub = self._stub()
        result = asyncio.run(
            SlackAdapter._send_slash_ephemeral_via_bot(
                stub, "C_STR", {"via_bot": True, "user_id": ""}, "x"
            )
        )
        assert not result.success


class TestAdapterWiring:
    """Guard the adapter integration points against accidental removal
    (same pattern as test_run_py_dispatches_cron)."""

    def _adapter_source(self) -> str:
        import pathlib

        import plugins.platforms.slack.adapter as slack_adapter

        return pathlib.Path(slack_adapter.__file__).read_text(encoding="utf-8")

    def test_command_closure_forwards_foreign_channels(self):
        src = self._adapter_source()
        assert "_resolve_foreign_slash_owner(command)" in src
        assert "_forward_slash_to_profile(owner, command)" in src

    def test_connect_starts_relay_consumer(self):
        src = self._adapter_source()
        assert "_ensure_slash_relay_task()" in src

    def test_disconnect_cancels_relay_consumer(self):
        src = self._adapter_source()
        assert "relay_task.cancel()" in src

    def test_relayed_replies_attributed_to_owning_bot(self):
        # Guard the attribution chain: relayed contexts are flagged via_bot,
        # and send() routes them through _send_slash_ephemeral_via_bot.
        src = self._adapter_source()
        assert 'ctx["via_bot"] = True' in src
        assert "_send_slash_ephemeral_via_bot" in src
        assert 'slash_ctx.get("via_bot")' in src
