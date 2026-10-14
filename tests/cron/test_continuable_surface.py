"""Flat continuable cron delivery must seed the same lane as an inbound reply."""

from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import GatewayConfig
from cron.scheduler import _deliver_result


class TestCronContinuableSurfaceInChannel:
    """cron_continuable_surface: in_channel — deliver a continuable cron FLAT
    into a channel (no dedicated thread), so a plain channel reply continues the
    job via the shared-channel session (platform, chat_id, None).

    Design: decisions.md D1/D2/D6 + F5. The scheduler reads the per-platform key
    generically from pconfig.extra; the in_channel branch is gated on the
    adapter capability flag ``supports_inchannel_continuable`` (Slack=True,
    others fail SAFE to thread). In in_channel mode the thread-open branch is
    SKIPPED (thread_id stays None), then ``_seed_cron_channel_session`` CREATES
    the flat shared-channel session and mirrors the brief into it (the shipped
    mirror only APPENDS to an existing session, and the flat channel row is
    otherwise absent for a chat_postMessage delivery).
    """

    def _slack_cfg(self, extra):
        """A mock GatewayConfig with a Slack pconfig carrying ``extra``."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        pconfig.extra = extra
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.SLACK: pconfig}
        return mock_cfg

    def _run_inchannel_delivery(self, extra, adapter, *, mirror_ok=True, origin=None,
                                attach_to_session=True):
        """Drive _deliver_result down the live-adapter path for a Slack
        channel-origin job with the given ``extra`` config. Returns the
        _open_continuable_cron_thread mock and the mirror_to_session mock."""
        from gateway.config import Platform
        from concurrent.futures import Future

        mock_cfg = self._slack_cfg(extra)

        loop = MagicMock()
        loop.is_running.return_value = True

        def fake_run_coro(coro, _loop):
            future = Future()
            try:
                import asyncio as _asyncio
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:
                future.set_exception(_e)
            return future

        job = {
            "id": "brief-job",
            "name": "Daily Brief",
            "deliver": "origin",
            # Channel origin: no thread_id (flat channel message scheduled it).
            # Carries the scheduling user's id — the in_channel seed must key
            # the flat channel session to THIS user (see build_session_key).
            "origin": origin or {"platform": "slack", "chat_id": "C123", "user_id": "U_HUMAN"},
            # Opt into the continuable mirror (parameterized: the seed must
            # NOT depend on this — see the no-attach regression test).
            "attach_to_session": attach_to_session,
        }

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("cron.scheduler_delivery._open_continuable_cron_thread") as open_thread_mock, \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro), \
             patch("gateway.mirror.mirror_to_session", return_value=mirror_ok) as mirror_mock:
            _deliver_result(
                job, "Here is today's brief.",
                adapters={Platform.SLACK: adapter}, loop=loop,
            )
        return open_thread_mock, mirror_mock

    def _slack_adapter(self, supports_inchannel=True, with_store=True):
        adapter = AsyncMock()
        adapter._canonicalize = MagicMock()
        adapter.get_chat_info.return_value = {"type": "group", "scope_id": "workspace"}
        adapter.send.return_value = MagicMock(
            success=True, message_id="msg_1", raw_response=None,
        )
        # Capability flag read via getattr in the scheduler.
        adapter.supports_inchannel_continuable = supports_inchannel
        # Pin the per-platform accessor OFF: an unspecced AsyncMock would
        # auto-create it as a truthy callable, silently routing every test
        # through the relay accessor branch instead of the native scalar
        # fallback these tests describe (and making supports_inchannel=False
        # unenforceable). None -> not callable -> scalar path, like a real
        # native adapter that never defines the method.
        adapter.supports_inchannel_continuable_for_platform = None
        # A live session store so the in_channel seed can CREATE the flat row
        # (the real bug: without a create step the mirror no-ops on a missing
        # session and the brief is lost). Use a plain MagicMock store.
        if with_store:
            adapter._session_store = MagicMock(config=GatewayConfig())
        return adapter

    def test_in_channel_skips_thread_open(self):
        """G2: in_channel mode must NOT open a handoff thread."""
        adapter = self._slack_adapter(supports_inchannel=True)
        open_thread_mock, _ = self._run_inchannel_delivery(
            {"cron_continuable_surface": "in_channel"}, adapter,
        )
        open_thread_mock.assert_not_called()

    # --- _seed_cron_channel_session: the create-then-mirror unit + the
    #     KEY-MATCH invariant (seed key must equal the inbound reply's key) ---

    def test_seed_channel_session_key_matches_inbound_channel_reply(self):
        """The whole point: the flat session the seed CREATES must be keyed
        identically to what a plain inbound channel reply resolves to. Assert
        the invariant directly via build_session_key, not just call args."""
        from cron.scheduler_sessions import _seed_cron_channel_session
        from gateway.session import build_session_key, SessionSource
        from gateway.config import Platform

        store = MagicMock(config=GatewayConfig())
        adapter = MagicMock()
        adapter._session_store = store

        with patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            ok = _seed_cron_channel_session(
                {"id": "j1", "name": "Brief"}, adapter, "slack", "C123",
                "Daily brief", is_dm=False, user_id="U_HUMAN", chat_name="ops",
            )
        assert ok is True
        seeded_source = store.get_or_create_session.call_args[0][0]
        seed_key = build_session_key(seeded_source)

        # What a plain top-level channel reply (reply_in_thread:false → thread
        # None) from the same user resolves to:
        inbound = SessionSource(
            platform=Platform.SLACK, chat_id="C123", chat_type="group",
            user_id="U_HUMAN", thread_id=None,
        )
        assert seed_key == build_session_key(inbound), (
            f"seed key {seed_key} != inbound reply key {build_session_key(inbound)} "
            "— the reply would NOT continue the seeded session"
        )
        mirror_mock.assert_called_once()
        assert mirror_mock.call_args.kwargs.get("thread_id") is None
        assert mirror_mock.call_args.kwargs.get("user_id") == "U_HUMAN"

    def test_in_channel_seed_fires_without_attach_to_session(self):
        """REGRESSION (live, Alice 2026-08-19): the in_channel seed was gated on
        mirror_this_target (mirror_enabled AND origin match), so a continuable
        in_channel cron created WITHOUT attach_to_session (and with the
        cron.mirror_delivery global at its default False) delivered the brief
        flat but never seeded the flat session — the next plain reply hit a
        blank session and the agent had no idea about its own delivery message.

        in_channel IS the continuation surface: the seed must fire on origin
        match alone. attach_to_session stays the opt-in for the SEPARATE
        default-surface mirror behavior; it must not be required here."""
        from cron.scheduler import _deliver_result

        adapter = self._slack_adapter(supports_inchannel=True)
        with patch("cron.scheduler_sessions._seed_cron_channel_session", return_value=True) as seed_mock:
            self._run_inchannel_delivery(
                {"slack": {"cron_continuable_surface": "in_channel"}}, adapter,
                attach_to_session=False,
            )
        seed_mock.assert_called_once()
        # user_id must ride along even without the mirror opt-in — the flat
        # session key includes it on per-user-isolated chats.
        assert seed_mock.call_args.kwargs.get("user_id") == "U_HUMAN"

    def test_in_channel_flattens_thread_without_attach_to_session(self):
        """REGRESSION: the thread-id-clearing gate was `mirror_this_target`
        while the seed below had been decoupled to origin-match alone. With
        the default knobs off (attach_to_session=False, cron.mirror_delivery
        unset) and an origin carrying a REAL thread_id, the brief still
        delivered INTO the origin thread while the flat (thread_id=None)
        session got seeded — brief and continuation surface in different
        places. The flatten must use the same gate as the seed: origin_target.
        Asserts on the routed DeliveryTarget, not just that the seed ran."""
        captured = {}

        class _SpyRouter:
            def __init__(self, *a, **k):
                pass

            async def _deliver_to_platform(self, target, text, metadata, transport=None):
                captured["target"] = target
                return {"success": True, "message_id": "msg_1"}

        adapter = self._slack_adapter(supports_inchannel=True)
        origin_with_thread = {
            "platform": "slack", "chat_id": "C123", "user_id": "U_HUMAN",
            # Genuine origin thread (job created from inside a thread).
            "thread_id": "1787188000.000100",
        }
        with patch("gateway.delivery.DeliveryRouter", _SpyRouter), \
             patch("cron.scheduler_sessions._seed_cron_channel_session", return_value=True) as seed_mock:
            self._run_inchannel_delivery(
                {"slack": {"cron_continuable_surface": "in_channel"}}, adapter,
                attach_to_session=False, origin=origin_with_thread,
            )
        seed_mock.assert_called_once()
        assert captured["target"].thread_id is None, (
            "in_channel delivery routed into the origin thread "
            f"({captured['target'].thread_id}) — the flat seeded session "
            "does not match where the brief actually landed"
        )

    def test_origin_scope_id_rides_delivery_metadata(self):
        """REGRESSION (restart-shaped): the connector's fail-closed tenant
        guard resolves the workspace from metadata.scope_id. After a gateway
        restart the RelayAdapter's per-chat scope cache is cold, and
        DeliveryRouter stamps scope only for the configured HOME channel —
        so a scoped Slack origin that is NOT the home chat egressed with no
        scope_id and could be rejected before delivery. The scheduler must
        stamp the persisted origin scope onto origin-matching routing
        metadata (and never onto fan-out targets, which the origin-match
        gate already excludes)."""
        captured = {}

        class _SpyRouter:
            def __init__(self, *a, **k):
                pass

            async def _deliver_to_platform(self, target, text, metadata, transport=None):
                captured["metadata"] = metadata
                return {"success": True, "message_id": "msg_1"}

        adapter = self._slack_adapter(supports_inchannel=True)
        scoped_origin = {
            "platform": "slack", "chat_id": "C123", "user_id": "U_HUMAN",
            # Persisted workspace scope (captured at job creation). C123 is
            # not any configured home channel in this harness.
            "scope_id": "T0AAAA111",
        }
        with patch("gateway.delivery.DeliveryRouter", _SpyRouter), \
             patch("cron.scheduler_sessions._seed_cron_channel_session", return_value=True):
            self._run_inchannel_delivery(
                {"slack": {"cron_continuable_surface": "in_channel"}}, adapter,
                attach_to_session=False, origin=scoped_origin,
            )
        assert captured["metadata"].get("scope_id") == "T0AAAA111", (
            "persisted origin scope_id did not reach the delivery metadata — "
            "a cold-cache relay egress has no tenant discriminator"
        )

    def test_legacy_origin_uses_adapter_resolved_scope(self):
        """A scope absent from the job is resolved by the adapter and pinned through egress."""
        captured = {}

        class _SpyRouter:
            def __init__(self, *a, **k):
                pass

            async def _deliver_to_platform(self, target, text, metadata, transport=None):
                captured["metadata"] = metadata
                return {"success": True, "message_id": "msg_1"}

        adapter = self._slack_adapter(supports_inchannel=True)
        with patch("gateway.delivery.DeliveryRouter", _SpyRouter), \
             patch("cron.scheduler_sessions._seed_cron_channel_session", return_value=True):
            self._run_inchannel_delivery(
                {"slack": {"cron_continuable_surface": "in_channel"}}, adapter,
                attach_to_session=False,
            )
        assert captured["metadata"]["scope_id"] == "workspace"

    def test_native_adapter_scalar_false_fails_safe_to_thread(self):
        """D6 fallback boundary with a REAL (non-mock) adapter shape: a native
        adapter defines only the scalar supports_inchannel_continuable and no
        per-platform accessor — MagicMock-based fixtures can't prove this
        branch because they fabricate a truthy accessor. Scalar False must
        fail safe to thread mode: the in_channel seed never fires."""

        class _NativeShapedAdapter:
            supports_inchannel_continuable = False
            _session_store = None

            def __init__(self):
                self.sent = []

            async def send(self, chat_id, content, metadata=None, **kwargs):
                self.sent.append((chat_id, content))
                return MagicMock(success=True, message_id="msg_1",
                                 raw_response=None)

        adapter = _NativeShapedAdapter()
        assert not callable(
            getattr(adapter, "supports_inchannel_continuable_for_platform", None)
        )
        with patch("cron.scheduler_sessions._seed_cron_channel_session") as seed_mock:
            self._run_inchannel_delivery(
                {"slack": {"cron_continuable_surface": "in_channel"}}, adapter,
                attach_to_session=False,
            )
        # Delivery must have gone through the LIVE adapter — otherwise a
        # broken harness that never delivers would also leave the seed
        # uncalled and this test would pass for the wrong reason.
        assert len(adapter.sent) == 1
        assert adapter.sent[0][0] == "C123"
        # Capability absent -> surface fails safe to thread -> flat seed
        # must NOT run (D6).
        seed_mock.assert_not_called()

    def test_seed_mirrors_into_exact_created_session_on_populated_chat(self):
        """REGRESSION (live, Alice 2026-08-19 19:17): 'in_channel seed did NOT
        land'. The seed created the flat session row, then mirror_to_session
        re-discovered the target via origin heuristics — and on a populated
        chat (flat session + N per-message thread sessions sharing chat_id,
        mixed user_ids) find_session_by_origin's multi-candidate bail-out
        returned None, silently dropping the brief. The seed must mirror into
        the EXACT session row it just created, no rediscovery."""
        from cron.scheduler_sessions import _seed_cron_channel_session

        store = MagicMock(config=GatewayConfig())
        created = MagicMock()
        created.session_id = "sess-flat-exact"
        store.get_or_create_session.return_value = created
        adapter = MagicMock()
        adapter._session_store = store

        with patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            ok = _seed_cron_channel_session(
                {"id": "j1", "name": "Brief"}, adapter, "slack", "D0BJTDCSR7C",
                "Daily brief", is_dm=True, user_id="U_HUMAN", chat_name=None,
            )
        assert ok is True
        # The mirror received the exact created session id — origin-scan
        # heuristics (and their populated-chat bail-out) are out of the path.
        assert mirror_mock.call_args.kwargs.get("session_id") == "sess-flat-exact"

    def test_in_channel_also_seeds_thread_surface_of_delivered_brief(self):
        """REGRESSION (live, Alice 2026-08-19 19:19): the user replied IN THE
        BRIEF'S THREAD (Slack's natural affordance on a flat message). That
        reply keys to (chat, thread=<brief ts>) — a session in_channel mode
        never seeded, so the agent had no idea about its own brief. The flat
        delivery's message_id must anchor a companion thread-surface seed."""
        adapter = self._slack_adapter(supports_inchannel=True)
        with patch("cron.scheduler_sessions._seed_cron_channel_session", return_value=True), \
             patch("cron.scheduler_sessions._seed_cron_thread_session") as thread_seed_mock:
            self._run_inchannel_delivery(
                {"slack": {"cron_continuable_surface": "in_channel"}}, adapter,
                attach_to_session=False,
            )
        thread_seed_mock.assert_called_once()
        # Anchored on the delivered message id (the router's SendResult).
        assert thread_seed_mock.call_args.args[4] == "msg_1"
