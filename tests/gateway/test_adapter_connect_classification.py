"""Connect-failure classification + reconnect-queue escalation (OOF-156).

Four platforms (telegram, discord, photon, email) used to funnel every
startup failure into an indefinitely-retried state — including permanent
failures like revoked tokens, missing privileged intents, and sidecar deps
that can never install on an immutable image. Fleet triage found agents that
had been silently "retrying" for weeks (OOF-151/152/153).

Two-part fix, both covered here:

1. Per-adapter classification: auth/permission/deterministic failures are
   classified by exception TYPE (never message text) as ``retryable=False``
   so they exit via the existing non-retryable fatal path.
2. Gateway escalation: platforms continuously in the reconnect queue past a
   threshold get ``needs_attention`` flagged in runtime status. Retries never
   stop — the deliberate removal of auto-pause stands (a transient outage
   must self-heal without operator action).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import (
    GatewayRunner,
    _reconnect_needs_attention,
)


# ── Telegram: type-based auth classification ───────────────────────────


class InvalidToken(Exception):  # noqa: N818 — name-matched stand-in
    pass


class Forbidden(Exception):  # noqa: N818
    pass


class NetworkError(Exception):  # noqa: N818
    pass


class TimedOut(Exception):  # noqa: N818
    pass


def _telegram_classifier():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    return TelegramAdapter._looks_like_auth_error


class TestTelegramAuthClassification:
    """InvalidToken/Forbidden are terminal; transient transports are not."""

    def test_invalid_token_is_auth_error(self):
        # Classifier matches on class name so tests do not need real
        # telegram.error types — mirrors _looks_like_network_error's design.
        assert _telegram_classifier()(InvalidToken("401 Unauthorized")) is True

    def test_forbidden_is_auth_error(self):
        assert _telegram_classifier()(Forbidden("bot was deleted")) is True

    def test_network_error_is_not_auth_error(self):
        assert _telegram_classifier()(NetworkError("dns failure")) is False

    def test_timeout_is_not_auth_error(self):
        assert _telegram_classifier()(TimedOut("read timeout")) is False

    def test_generic_exception_is_not_auth_error(self):
        # Unknown types must stay retryable — a false terminal recreates the
        # "silently dead bot" problem the auto-pause removal fixed.
        assert _telegram_classifier()(RuntimeError("weird")) is False

    def test_auth_message_text_does_not_classify(self):
        # Guard against text matching creeping back in: an exception whose
        # MESSAGE mentions auth but whose type is generic must stay retryable.
        assert _telegram_classifier()(RuntimeError("InvalidToken Forbidden")) is False


# ── Discord: connect exception classification ──────────────────────────


class LoginFailure(Exception):
    """Name-matched stand-in for discord.LoginFailure."""


class PrivilegedIntentsRequired(Exception):
    """Name-matched stand-in for discord.PrivilegedIntentsRequired."""


def _discord_classifier():
    from plugins.platforms.discord.adapter import DiscordAdapter

    # Instance-bound (the intents branch reads the adapter's allowlists to
    # tailor its guidance); a bare instance with empty allowlists suffices.
    adapter = object.__new__(DiscordAdapter)
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    return adapter._classify_connect_exception


class TestDiscordConnectClassification:
    def test_login_failure_is_terminal(self):
        code, message, retryable = _discord_classifier()(LoginFailure("Improper token"))
        assert code == "discord_auth_error"
        assert retryable is False
        assert "Developer Portal" in message

    def test_privileged_intents_is_terminal(self):
        code, message, retryable = _discord_classifier()(
            PrivilegedIntentsRequired("shard 0 requested privileged intents")
        )
        assert code == "discord_intents_required"
        assert retryable is False
        assert "Message Content Intent" in message

    def test_unknown_exception_is_retryable_with_explicit_code(self):
        # The old behavior set NO fatal code at all, which the gateway read
        # as "probably transient". Every failure must now carry a code.
        code, message, retryable = _discord_classifier()(OSError("connection reset"))
        assert code == "discord_connect_error"
        assert retryable is True

    def test_auth_message_text_does_not_classify(self):
        code, _message, retryable = _discord_classifier()(
            RuntimeError("LoginFailure: PrivilegedIntentsRequired")
        )
        assert code == "discord_connect_error"
        assert retryable is True


# ── Photon: typed sidecar startup errors ───────────────────────────────


class TestPhotonSidecarStartupClassification:
    def _make_adapter(self, monkeypatch):
        monkeypatch.setenv("PHOTON_PROJECT_ID", "pid")
        monkeypatch.setenv("PHOTON_PROJECT_SECRET", "psecret")
        from plugins.platforms.photon.adapter import PhotonAdapter

        return PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))

    @pytest.mark.asyncio
    async def test_typed_startup_error_sets_nonretryable_fatal(self, monkeypatch):
        from plugins.platforms.photon import adapter as photon_adapter

        adapter = self._make_adapter(monkeypatch)

        async def _boom():
            raise photon_adapter.PhotonSidecarStartupError(
                "deps could not be installed",
                code="SIDECAR_DEPS_MISSING",
                retryable=False,
            )

        monkeypatch.setattr(adapter, "_start_sidecar", _boom)
        ok = await adapter.connect()

        assert ok is False
        assert adapter.fatal_error_code == "SIDECAR_DEPS_MISSING"
        assert adapter.fatal_error_retryable is False

    @pytest.mark.asyncio
    async def test_untyped_startup_error_stays_retryable(self, monkeypatch):
        # Ambiguous failures (crash before ready, health timeout) must keep
        # retrying; the gateway's needs_attention escalation is the backstop.
        adapter = self._make_adapter(monkeypatch)

        async def _boom():
            raise RuntimeError("sidecar exited with code 1 before becoming ready")

        monkeypatch.setattr(adapter, "_start_sidecar", _boom)
        ok = await adapter.connect()

        assert ok is False
        assert adapter.fatal_error_code == "SIDECAR_FAILED"
        assert adapter.fatal_error_retryable is True

    def test_deps_install_failure_raises_typed_nonretryable(self, monkeypatch):
        from plugins.platforms.photon import adapter as photon_adapter

        adapter = self._make_adapter(monkeypatch)
        monkeypatch.setattr(photon_adapter, "sidecar_deps_installed", lambda: False)
        monkeypatch.setattr(photon_adapter, "_reinstall_sidecar_deps", lambda: None)

        with pytest.raises(photon_adapter.PhotonSidecarStartupError) as exc_info:
            asyncio.get_event_loop().run_until_complete(adapter._start_sidecar())

        assert exc_info.value.code == "SIDECAR_DEPS_MISSING"
        assert exc_info.value.retryable is False


# ── Email: explicit fatal codes on IMAP/SMTP failure ───────────────────


class TestEmailConnectClassification:
    def _make_adapter(self, monkeypatch):
        for key, value in {
            "EMAIL_ADDRESS": "bot@example.com",
            "EMAIL_PASSWORD": "app-password",
            "EMAIL_IMAP_HOST": "imap.example.com",
            "EMAIL_SMTP_HOST": "smtp.example.com",
        }.items():
            monkeypatch.setenv(key, value)
        from plugins.platforms.email.adapter import EmailAdapter

        return EmailAdapter(PlatformConfig(enabled=True, token=""))

    @pytest.mark.asyncio
    async def test_imap_failure_sets_explicit_retryable_fatal(self, monkeypatch):
        # The old code returned False with NO fatal info — the gateway's
        # "no info = transient" branch then retried forever with zero owner
        # signal ("stuck retrying 22h").
        adapter = self._make_adapter(monkeypatch)
        from plugins.platforms.email import adapter as email_adapter

        def _raise(*a, **k):
            raise email_adapter.imaplib.IMAP4.error(b"[AUTHENTICATIONFAILED]")

        monkeypatch.setattr(email_adapter.imaplib, "IMAP4_SSL", _raise)
        ok = await adapter.connect()

        assert ok is False
        assert adapter.fatal_error_code == "email_imap_connect_error"
        # IMAP4.error is the same type for bad creds and transient server
        # NOs, so a type-based terminal classification is not safe here.
        assert adapter.fatal_error_retryable is True

    @pytest.mark.asyncio
    async def test_smtp_auth_failure_is_terminal(self, monkeypatch):
        import smtplib

        adapter = self._make_adapter(monkeypatch)
        from plugins.platforms.email import adapter as email_adapter

        imap = MagicMock()
        imap.uid.return_value = ("OK", [b""])
        monkeypatch.setattr(email_adapter.imaplib, "IMAP4_SSL", lambda *a, **k: imap)

        def _smtp_fail():
            raise smtplib.SMTPAuthenticationError(535, b"authentication failed")

        monkeypatch.setattr(adapter, "_connect_smtp", _smtp_fail)
        ok = await adapter.connect()

        assert ok is False
        assert adapter.fatal_error_code == "email_auth_error"
        assert adapter.fatal_error_retryable is False

    @pytest.mark.asyncio
    async def test_smtp_transient_failure_stays_retryable(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        from plugins.platforms.email import adapter as email_adapter

        imap = MagicMock()
        imap.uid.return_value = ("OK", [b""])
        monkeypatch.setattr(email_adapter.imaplib, "IMAP4_SSL", lambda *a, **k: imap)

        def _smtp_fail():
            raise OSError("connection refused")

        monkeypatch.setattr(adapter, "_connect_smtp", _smtp_fail)
        ok = await adapter.connect()

        assert ok is False
        assert adapter.fatal_error_code == "email_smtp_connect_error"
        assert adapter.fatal_error_retryable is True


# ── Gateway: needs_attention escalation ────────────────────────────────


class TestReconnectNeedsAttention:
    def test_fresh_entry_is_not_flagged_and_gets_stamped(self):
        # In-flight upgrade path: entries queued before queued_at existed are
        # treated as newly queued, not instantly escalated.
        info = {"attempts": 3}
        now = time.monotonic()
        assert _reconnect_needs_attention(info, now) is False
        assert info["queued_at"] == now

    def test_below_threshold_is_not_flagged(self):
        now = time.monotonic()
        info = {"queued_at": now - 60}
        assert _reconnect_needs_attention(info, now) is False

    def test_past_threshold_is_flagged(self):
        import gateway.run as run_module

        now = time.monotonic()
        info = {"queued_at": now - (run_module._RECONNECT_ATTENTION_AFTER_SECONDS + 1)}
        assert _reconnect_needs_attention(info, now) is True

    def test_zero_threshold_disables_escalation(self, monkeypatch):
        import gateway.run as run_module

        monkeypatch.setattr(run_module, "_RECONNECT_ATTENTION_AFTER_SECONDS", 0)
        info = {"queued_at": time.monotonic() - 999999}
        assert _reconnect_needs_attention(info, time.monotonic()) is False


def _make_runner():
    """Minimal GatewayRunner via object.__new__ (same pattern as
    test_platform_reconnect.py)."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test")}
    )
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_with_failure = False
    runner._exit_cleanly = False
    runner._failed_platforms = {}
    runner.adapters = {}
    runner.delivery_router = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._honcho_managers = {}
    runner._honcho_configs = {}
    runner._shutdown_all_gateway_honcho = lambda: None
    runner.session_store = MagicMock()
    return runner


class TestWatcherAttentionEscalation:
    @pytest.mark.asyncio
    async def test_watcher_flags_long_queued_platform_and_keeps_retrying(self, monkeypatch):
        import gateway.run as run_module

        runner = _make_runner()
        status_writes = []
        monkeypatch.setattr(
            runner,
            "_update_platform_runtime_status",
            lambda platform, **kw: status_writes.append((platform, kw)),
        )

        threshold = run_module._RECONNECT_ATTENTION_AFTER_SECONDS
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": PlatformConfig(enabled=True, token="test"),
            "attempts": 40,
            # Not yet due for a retry — escalation must not depend on the
            # backoff schedule lining up.
            "next_retry": time.monotonic() + 300,
            "queued_at": time.monotonic() - threshold - 10,
        }

        real_sleep = asyncio.sleep
        call_count = 0

        async def fake_sleep(n):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                runner._running = False
            await real_sleep(0)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await runner._platform_reconnect_watcher()

        attention = [kw for _p, kw in status_writes if kw.get("needs_attention")]
        assert attention, f"expected a needs_attention status write, got {status_writes!r}"
        assert attention[0]["platform_state"] == "retrying"
        assert attention[0].get("retrying_since")
        # Platform must STILL be queued — escalation is a signal, never a
        # circuit breaker.
        assert Platform.TELEGRAM in runner._failed_platforms
        assert runner._failed_platforms[Platform.TELEGRAM].get("attention_flagged") is True

    @pytest.mark.asyncio
    async def test_watcher_flags_only_once(self, monkeypatch):
        import gateway.run as run_module

        runner = _make_runner()
        status_writes = []
        monkeypatch.setattr(
            runner,
            "_update_platform_runtime_status",
            lambda platform, **kw: status_writes.append((platform, kw)),
        )

        threshold = run_module._RECONNECT_ATTENTION_AFTER_SECONDS
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": PlatformConfig(enabled=True, token="test"),
            "attempts": 40,
            "next_retry": time.monotonic() + 300,
            "queued_at": time.monotonic() - threshold - 10,
        }

        real_sleep = asyncio.sleep
        call_count = 0

        async def fake_sleep(n):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                runner._running = False
            await real_sleep(0)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await runner._platform_reconnect_watcher()

        attention = [kw for _p, kw in status_writes if kw.get("needs_attention")]
        assert len(attention) == 1, (
            f"needs_attention must be written once per episode, got {len(attention)}"
        )


# ── Status file: new platform fields round-trip ────────────────────────


class TestRuntimeStatusAttentionFields:
    def test_needs_attention_and_retrying_since_persisted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from gateway import status as status_module

        status_module.write_runtime_status(
            platform="telegram",
            platform_state="retrying",
            error_code="telegram_connect_error",
            error_message="boom",
            needs_attention=True,
            retrying_since="2026-08-11T00:00:00+00:00",
        )
        payload = status_module.read_runtime_status()
        platform = payload["platforms"]["telegram"]
        assert platform["needs_attention"] is True
        assert platform["retrying_since"] == "2026-08-11T00:00:00+00:00"

        # Reconnect clears both.
        status_module.write_runtime_status(
            platform="telegram",
            platform_state="connected",
            error_code=None,
            error_message=None,
            needs_attention=False,
            retrying_since=None,
        )
        payload = status_module.read_runtime_status()
        platform = payload["platforms"]["telegram"]
        assert platform["needs_attention"] is False
        assert platform["retrying_since"] is None


# -- Gateway: the flap that defeats the escalation (#92178) -------------


class TestReconnectClockCarriesAcrossFlaps:
    """A brief reconnect is instability, not recovery.

    The escalation above is per queue-spell: a successful reconnect deletes the
    queue entry, and the next failure re-enters with a fresh ``queued_at``. Two
    systemd units racing the same bot token is exactly the workload that keeps
    winning briefly, so the 2-hour clock never accumulates and a four-day
    outage presents as ordinary ``retrying`` the whole time (#92178).

    ``_reconnect_clock_start`` hands the old clock back when the "recovery" was
    shorter than the stability window, so cumulative instability escalates
    while a genuine outage-then-recovery still starts over.
    """

    def test_no_history_starts_the_clock_now(self):
        runner = _make_runner()
        runner._recent_platform_recoveries = {}

        now = time.monotonic()
        assert runner._reconnect_clock_start(Platform.TELEGRAM, now) == now

    def test_a_brief_recovery_carries_the_old_clock_forward(self):
        """The reporter's shape: down for hours, up for seconds, down again."""
        runner = _make_runner()
        runner._recent_platform_recoveries = {}

        unstable_since = time.monotonic() - 9000  # 2.5 hours of trouble
        runner._note_reconnect_recovery(
            Platform.TELEGRAM, {"queued_at": unstable_since}
        )

        # It loses the token again five seconds later.
        carried = runner._reconnect_clock_start(Platform.TELEGRAM)

        assert carried == unstable_since, (
            "a reconnect that lasted seconds must not reset the instability "
            "clock; resetting it is what kept #92178 silent for four days"
        )

    def test_a_sustained_recovery_starts_a_fresh_clock(self):
        """The behaviour the escalation was designed around is preserved.

        A platform that fails, comes back, and *stays* back for longer than the
        stability window has genuinely recovered. Its next outage is a new
        episode and must not inherit an ancient start time, or a single blip a
        month ago would flag the next one instantly.
        """
        import gateway.run as run_module

        runner = _make_runner()
        runner._recent_platform_recoveries = {}

        now = time.monotonic()
        window = run_module._RECONNECT_STABLE_AFTER_SECONDS
        runner._recent_platform_recoveries[Platform.TELEGRAM] = (
            now - 9000,            # unstable_since, long ago
            now - window - 1,      # recovered_at, just past the window
        )

        assert runner._reconnect_clock_start(Platform.TELEGRAM, now) == now
        assert Platform.TELEGRAM not in runner._recent_platform_recoveries, (
            "a mark past its window is dead weight and must be pruned"
        )

    def test_zero_window_disables_the_carry_over(self, monkeypatch):
        import gateway.run as run_module

        monkeypatch.setattr(run_module, "_RECONNECT_STABLE_AFTER_SECONDS", 0)
        runner = _make_runner()
        runner._recent_platform_recoveries = {}

        runner._note_reconnect_recovery(
            Platform.TELEGRAM, {"queued_at": time.monotonic() - 9000}
        )
        now = time.monotonic()

        assert runner._recent_platform_recoveries == {}
        assert runner._reconnect_clock_start(Platform.TELEGRAM, now) == now

    def test_a_requeue_after_a_flap_is_immediately_escalatable(self, monkeypatch):
        """End to end over the two calls the flap actually goes through.

        Recover, drop the entry, fail again, re-enter via
        ``_queue_retryable_fatal_platform`` -- and the new entry must already be
        past the attention threshold, because the platform has been in trouble
        for hours even though this particular spell is seconds old.
        """
        import gateway.run as run_module

        runner = _make_runner()
        runner._recent_platform_recoveries = {}
        monkeypatch.setattr(runner, "_adapter_credential_claim", lambda p, a: None)
        monkeypatch.setattr(runner, "_adapter_listener_claim", lambda p, a: None)
        monkeypatch.setattr(runner, "_ensure_reconnect_watcher_running", lambda: None)

        threshold = run_module._RECONNECT_ATTENTION_AFTER_SECONDS
        info = {
            "config": PlatformConfig(enabled=True, token="test"),
            "attempts": 3,
            "next_retry": time.monotonic(),
            "queued_at": time.monotonic() - threshold - 10,
        }
        runner._failed_platforms[Platform.TELEGRAM] = info

        # One of the brief successful binds: the watcher notes it, then drops
        # the entry (and with it, before this fix, the whole clock).
        runner._note_reconnect_recovery(Platform.TELEGRAM, info)
        del runner._failed_platforms[Platform.TELEGRAM]

        adapter = MagicMock()
        adapter.platform = Platform.TELEGRAM
        adapter.fatal_error_retryable = True
        assert runner._queue_retryable_fatal_platform(adapter) is True

        requeued = runner._failed_platforms[Platform.TELEGRAM]
        assert _reconnect_needs_attention(requeued, time.monotonic()) is True, (
            "the platform has been failing for over the threshold; a five "
            "second reconnect in the middle does not make it healthy"
        )

    @pytest.mark.asyncio
    async def test_watcher_flags_a_flapping_platform(self, monkeypatch):
        """The signal the reporter never got, through the real watcher."""
        import gateway.run as run_module

        runner = _make_runner()
        runner._recent_platform_recoveries = {}
        monkeypatch.setattr(runner, "_adapter_credential_claim", lambda p, a: None)
        monkeypatch.setattr(runner, "_adapter_listener_claim", lambda p, a: None)
        monkeypatch.setattr(runner, "_ensure_reconnect_watcher_running", lambda: None)
        status_writes = []
        monkeypatch.setattr(
            runner,
            "_update_platform_runtime_status",
            lambda platform, **kw: status_writes.append((platform, kw)),
        )

        threshold = run_module._RECONNECT_ATTENTION_AFTER_SECONDS
        info = {
            "config": PlatformConfig(enabled=True, token="test"),
            "attempts": 3,
            "next_retry": time.monotonic(),
            "queued_at": time.monotonic() - threshold - 10,
        }
        runner._failed_platforms[Platform.TELEGRAM] = info
        runner._note_reconnect_recovery(Platform.TELEGRAM, info)
        del runner._failed_platforms[Platform.TELEGRAM]

        adapter = MagicMock()
        adapter.platform = Platform.TELEGRAM
        adapter.fatal_error_retryable = True
        runner._queue_retryable_fatal_platform(adapter)
        # Hold off the retry itself so the pass exercises escalation only.
        runner._failed_platforms[Platform.TELEGRAM]["next_retry"] = (
            time.monotonic() + 300
        )

        real_sleep = asyncio.sleep
        call_count = 0

        async def fake_sleep(n):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                runner._running = False
            await real_sleep(0)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await runner._platform_reconnect_watcher()

        attention = [kw for _p, kw in status_writes if kw.get("needs_attention")]
        assert attention, (
            "a platform flapping past the attention threshold must be flagged; "
            f"got {status_writes!r}"
        )
        assert attention[0]["platform_state"] == "retrying"
        # Still queued: escalation is a signal, never a circuit breaker.
        assert Platform.TELEGRAM in runner._failed_platforms


class TestOneEpisodeEscalatesOnce:
    """A flapping platform must warn once per episode, not once per flap.

    ``_reconnect_clock_start`` carries WHEN the trouble started across a brief
    bind. On its own that has a side effect nobody asked for: the queue entry
    is still destroyed on every bind, and with it the per-entry
    ``attention_flagged`` marker that makes the watcher's escalation fire once.
    The requeued entry is therefore un-flagged with a ``queued_at`` already
    past the threshold, so the very next watcher pass warns again. At the
    reporter's 30-120s bind cadence that is order 10^3 warnings and status
    writes a day, where before the carry there were none.

    The mirror problem is the status write on the way out: the successful
    reconnect branch cleared ``needs_attention`` unconditionally, so `hermes
    status` and fleet monitoring watched the platform blink healthy on every
    bind in the middle of an episode the gateway had already declared
    unhealthy.

    Both halves are the same missing idea -- a bind is not a recovery until it
    lasts -- so the flag now survives the bind and is retired by
    ``_expire_stable_recoveries`` once the stability window proves it.
    """

    # -- _carried_attention_flag -------------------------------------------

    def test_a_flap_inside_the_window_carries_the_escalation(self):
        runner = _make_runner()
        runner._recent_platform_recoveries = {}

        now = time.monotonic()
        runner._note_reconnect_recovery(
            Platform.TELEGRAM,
            {"queued_at": now - 9000, "attention_flagged": True},
            now,
        )

        assert runner._carried_attention_flag(Platform.TELEGRAM, now + 5) is True, (
            "the platform was already declared NEEDS_ATTENTION and has been "
            "back for five seconds; re-announcing it on every bind is noise"
        )

    def test_a_sustained_recovery_drops_the_escalation(self):
        """The next outage is a new episode and gets its own first warning."""
        import gateway.run as run_module

        runner = _make_runner()
        runner._recent_platform_recoveries = {}
        window = run_module._RECONNECT_STABLE_AFTER_SECONDS

        now = time.monotonic()
        runner._note_reconnect_recovery(
            Platform.TELEGRAM,
            {"queued_at": now - 9000, "attention_flagged": True},
            now,
        )

        assert runner._carried_attention_flag(Platform.TELEGRAM, now + window + 1) is False

    def test_no_history_carries_no_escalation(self):
        runner = _make_runner()
        runner._recent_platform_recoveries = {}

        assert runner._carried_attention_flag(Platform.TELEGRAM) is False

    def test_an_unflagged_flap_carries_no_escalation(self):
        """The both-directions half: carrying is conditional, not automatic."""
        runner = _make_runner()
        runner._recent_platform_recoveries = {}

        now = time.monotonic()
        runner._note_reconnect_recovery(Platform.TELEGRAM, {"queued_at": now - 60}, now)

        assert runner._carried_attention_flag(Platform.TELEGRAM, now + 5) is False, (
            "a platform that never crossed the threshold must not inherit an "
            "escalation it never earned"
        )

    def test_a_zero_window_carries_no_escalation(self, monkeypatch):
        import gateway.run as run_module

        monkeypatch.setattr(run_module, "_RECONNECT_STABLE_AFTER_SECONDS", 0)
        runner = _make_runner()
        runner._recent_platform_recoveries = {}

        runner._note_reconnect_recovery(
            Platform.TELEGRAM,
            {"queued_at": time.monotonic() - 9000, "attention_flagged": True},
        )

        assert runner._carried_attention_flag(Platform.TELEGRAM) is False

    def test_a_two_tuple_mark_still_reads(self):
        """Marks written before this change are a 2-tuple; do not crash on one."""
        runner = _make_runner()
        now = time.monotonic()
        runner._recent_platform_recoveries = {Platform.TELEGRAM: (now - 9000, now)}

        assert runner._carried_attention_flag(Platform.TELEGRAM, now + 5) is False
        assert runner._reconnect_clock_start(Platform.TELEGRAM, now + 5) == now - 9000

    # -- the requeue --------------------------------------------------------

    def _flap_once(self, runner, monkeypatch, flagged):
        """Recover, drop the entry, fail again -- the two real calls."""
        import gateway.run as run_module

        threshold = run_module._RECONNECT_ATTENTION_AFTER_SECONDS
        info = {
            "config": PlatformConfig(enabled=True, token="test"),
            "attempts": 3,
            "next_retry": time.monotonic(),
            "queued_at": time.monotonic() - threshold - 10,
            "attention_flagged": flagged,
        }
        runner._failed_platforms[Platform.TELEGRAM] = info
        runner._note_reconnect_recovery(Platform.TELEGRAM, info)
        del runner._failed_platforms[Platform.TELEGRAM]

        adapter = MagicMock()
        adapter.platform = Platform.TELEGRAM
        adapter.fatal_error_retryable = True
        assert runner._queue_retryable_fatal_platform(adapter) is True
        return runner._failed_platforms[Platform.TELEGRAM]

    def _requeue_runner(self, monkeypatch):
        runner = _make_runner()
        runner._recent_platform_recoveries = {}
        monkeypatch.setattr(runner, "_adapter_credential_claim", lambda p, a: None)
        monkeypatch.setattr(runner, "_adapter_listener_claim", lambda p, a: None)
        monkeypatch.setattr(runner, "_ensure_reconnect_watcher_running", lambda: None)
        return runner

    def test_the_requeued_entry_arrives_already_flagged(self, monkeypatch):
        runner = self._requeue_runner(monkeypatch)
        requeued = self._flap_once(runner, monkeypatch, flagged=True)

        assert requeued["attention_flagged"] is True, (
            "the escalation belongs to the episode, not to the queue entry; "
            "a five second bind must not buy a second warning"
        )

    def test_an_unflagged_episode_requeues_unflagged(self, monkeypatch):
        """The guard is not over-tight: a first spell still gets its warning."""
        runner = self._requeue_runner(monkeypatch)
        requeued = self._flap_once(runner, monkeypatch, flagged=False)

        assert requeued["attention_flagged"] is False
        assert _reconnect_needs_attention(requeued, time.monotonic()) is True, (
            "carrying the clock is what #92247 fixed and must still hold"
        )

    def test_a_run_of_flaps_warns_once_not_once_each(self, monkeypatch):
        """The headline count, over the exact predicate the watcher uses.

        Five binds in a row against a platform that is hours into trouble. The
        watcher escalates when ``not info["attention_flagged"] and
        _reconnect_needs_attention(info, now)``, so counting the passes where
        that is true counts the warnings the operator would actually receive.
        """
        runner = self._requeue_runner(monkeypatch)

        warnings = 0
        flagged = False
        for _ in range(5):
            entry = self._flap_once(runner, monkeypatch, flagged=flagged)
            now = time.monotonic()
            if not entry.get("attention_flagged") and _reconnect_needs_attention(entry, now):
                warnings += 1
                entry["attention_flagged"] = True
            flagged = bool(entry.get("attention_flagged"))

        assert warnings == 1, (
            f"one episode, one warning; got {warnings} -- one per flap is the "
            "alert churn the carried clock would otherwise introduce"
        )

    # -- _expire_stable_recoveries -----------------------------------------

    def _capture_status(self, monkeypatch):
        """Record what actually reaches the runtime-status file.

        Deliberately NOT a stub on ``_update_platform_runtime_status``: the
        whole point of passing ``None`` / ``_UNSET`` is that that method then
        omits the field, so stubbing it out would test the sentinel rather than
        the behaviour an operator sees.
        """
        writes = []
        monkeypatch.setattr(
            "gateway.status.write_runtime_status",
            lambda **kw: writes.append(kw),
        )
        return writes


    def test_the_sweep_clears_the_flag_once_the_recovery_holds(self, monkeypatch):
        import gateway.run as run_module

        runner = _make_runner()
        runner._recent_platform_recoveries = {}
        writes = self._capture_status(monkeypatch)

        now = time.monotonic()
        window = run_module._RECONNECT_STABLE_AFTER_SECONDS
        runner._recent_platform_recoveries[Platform.TELEGRAM] = (
            now - 9000,
            now - window - 1,
            True,
        )

        runner._expire_stable_recoveries(now)

        assert Platform.TELEGRAM not in runner._recent_platform_recoveries
        assert len(writes) == 1, (
            "a bind that outlived the stability window is the recovery; "
            f"something has to say so, and nothing else visits it. got {writes!r}"
        )
        assert writes[0]["platform"] == "telegram"
        assert writes[0]["needs_attention"] is False
        assert writes[0]["retrying_since"] is None

    def test_the_sweep_leaves_a_platform_that_flapped_again_alone(self, monkeypatch):
        import gateway.run as run_module

        runner = _make_runner()
        runner._recent_platform_recoveries = {}
        writes = self._capture_status(monkeypatch)

        now = time.monotonic()
        window = run_module._RECONNECT_STABLE_AFTER_SECONDS
        runner._recent_platform_recoveries[Platform.TELEGRAM] = (
            now - 9000,
            now - window - 1,
            True,
        )
        # It is back in the queue: the episode is not over, whatever the
        # timestamps say about the last bind.
        runner._failed_platforms[Platform.TELEGRAM] = {"queued_at": now - 9000}

        runner._expire_stable_recoveries(now)

        assert writes == [], (
            "declaring a currently-failing platform healthy is the exact blink "
            "this change exists to stop"
        )
        assert Platform.TELEGRAM in runner._recent_platform_recoveries

    def test_the_sweep_is_silent_for_an_unflagged_recovery(self, monkeypatch):
        """No flag was ever set, so there is nothing to clear and no write."""
        import gateway.run as run_module

        runner = _make_runner()
        runner._recent_platform_recoveries = {}
        writes = self._capture_status(monkeypatch)

        now = time.monotonic()
        window = run_module._RECONNECT_STABLE_AFTER_SECONDS
        runner._recent_platform_recoveries[Platform.TELEGRAM] = (
            now - 200,
            now - window - 1,
            False,
        )

        runner._expire_stable_recoveries(now)

        assert Platform.TELEGRAM not in runner._recent_platform_recoveries
        assert writes == []

    # -- the watcher --------------------------------------------------------

    def _reconnect_runner(self, monkeypatch):
        runner = _make_runner()
        runner._recent_platform_recoveries = {}
        runner._busy_text_mode = "off"
        runner._sync_voice_mode_state_to_adapter = MagicMock()
        runner._primary_message_handler = MagicMock(return_value=MagicMock())
        runner._primary_platform_event_handler = MagicMock(return_value=MagicMock())
        runner._handle_adapter_fatal_error = MagicMock()
        runner._handle_active_session_busy_message = MagicMock()
        runner._handle_reaction_event = MagicMock()
        runner._recover_telegram_topic_thread_id = MagicMock()
        runner._make_adapter_auth_check = MagicMock(return_value=MagicMock())
        runner._handle_voice_channel_input = MagicMock()
        runner._schedule_resume_pending_sessions = MagicMock()
        runner._create_adapter = MagicMock(return_value=MagicMock())
        runner._connect_adapter_with_timeout = AsyncMock(return_value=True)
        return runner

    async def _run_one_watcher_pass(self, runner):
        real_sleep = asyncio.sleep
        call_count = 0

        async def fake_sleep(n):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                runner._running = False
            await real_sleep(0)

        with patch("gateway.run.build_channel_directory", create=True):
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await runner._platform_reconnect_watcher()

    @pytest.mark.asyncio
    async def test_a_bind_mid_episode_does_not_declare_the_platform_healthy(
        self, monkeypatch
    ):
        writes = self._capture_status(monkeypatch)
        runner = self._reconnect_runner(monkeypatch)
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": PlatformConfig(enabled=True, token="test"),
            "attempts": 40,
            "next_retry": time.monotonic() - 1,
            "queued_at": time.monotonic() - 9000,
            "attention_flagged": True,
        }

        await self._run_one_watcher_pass(runner)

        connected = [kw for kw in writes if kw.get("platform_state") == "connected"]
        assert connected, f"the reconnect must still be reported; got {writes!r}"
        assert "needs_attention" not in connected[0], (
            "at bind time this is indistinguishable from the next flap; "
            "clearing here is what made the platform blink healthy every "
            f"30-120s mid-episode. got {connected[0]!r}"
        )
        assert "retrying_since" not in connected[0], (
            "the episode's start time is still the truth until the recovery "
            f"proves itself. got {connected[0]!r}"
        )

    @pytest.mark.asyncio
    async def test_an_unescalated_reconnect_still_clears_immediately(self, monkeypatch):
        """The other direction: an ordinary blip keeps today's exact behaviour.

        Nothing was ever flagged, so there is nothing to protect and no reason
        to make an operator wait out the stability window for a clean status.
        """
        writes = self._capture_status(monkeypatch)
        runner = self._reconnect_runner(monkeypatch)
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": PlatformConfig(enabled=True, token="test"),
            "attempts": 1,
            "next_retry": time.monotonic() - 1,
            "queued_at": time.monotonic() - 30,
        }

        await self._run_one_watcher_pass(runner)

        connected = [kw for kw in writes if kw.get("platform_state") == "connected"]
        assert connected, f"the reconnect must still be reported; got {writes!r}"
        assert connected[0]["needs_attention"] is False
        assert connected[0]["retrying_since"] is None

    @pytest.mark.asyncio
    async def test_the_watcher_sweeps_even_with_an_empty_queue(self, monkeypatch):
        """Where the sweep sits in the loop is the whole point of it.

        A platform that recovered has left ``_failed_platforms``. If the sweep
        ran after the empty-queue early-out, the one case it exists for -- the
        last platform recovering alone -- is the one case it would never see,
        and the flag would stay set until the next unrelated outage.
        """
        import gateway.run as run_module

        writes = self._capture_status(monkeypatch)
        runner = self._reconnect_runner(monkeypatch)
        now = time.monotonic()
        window = run_module._RECONNECT_STABLE_AFTER_SECONDS
        runner._recent_platform_recoveries[Platform.TELEGRAM] = (
            now - 9000,
            now - window - 1,
            True,
        )
        assert runner._failed_platforms == {}

        await self._run_one_watcher_pass(runner)

        assert len(writes) == 1, (
            "with nothing queued the watcher still owes the recovered platform "
            f"its all-clear. got {writes!r}"
        )
        assert writes[0]["platform"] == "telegram"
        assert writes[0]["needs_attention"] is False
        assert writes[0]["retrying_since"] is None
