"""Unit tests for the Kimi platform adapter.

Focus is on pure-function correctness: envelope codec, chat-id routing,
dedup, MessageEvent synthesis, slash-command detection. Live-network tests
(GetMe, Subscribe against real Kimi) are gated behind a
``KIMI_INTEGRATION_TOKEN`` env var and skipped by default.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import sys
import tempfile
import unittest
import uuid
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import MessageType, SendResult
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/kimi/adapter.py under plugin_adapter_kimi so it
# cannot collide with sibling platform-plugin tests in the same xdist worker.
_kimi = load_plugin_adapter("kimiclaw")
# The source test still patches/imports kimi_adapter by name and asserts that
# logger name, so provide a test-local alias to the isolated plugin module.
sys.modules["kimi_adapter"] = _kimi
_kimi.logger = logging.getLogger("kimi_adapter")

_ARRIVAL_TIME_CACHE_DEFAULT_MAX = _kimi._ARRIVAL_TIME_CACHE_DEFAULT_MAX
_CONNECT_FLAG_COMPRESSED = _kimi._CONNECT_FLAG_COMPRESSED
_CONNECT_FLAG_END_STREAM = _kimi._CONNECT_FLAG_END_STREAM
_DMInflight = _kimi._DMInflight
_ROOM_CACHE_DEFAULT_MAX = _kimi._ROOM_CACHE_DEFAULT_MAX
_WS_MAX_FRAME_SIZE = _kimi._WS_MAX_FRAME_SIZE
KimiAdapter = _kimi.KimiAdapter
KimiAuthError = _kimi.KimiAuthError
KimiProtocolError = _kimi.KimiProtocolError
KimiRpcError = _kimi.KimiRpcError
KimiTransientError = _kimi.KimiTransientError
_BoundedLRU = _kimi._BoundedLRU
_extract_blocks_payload = _kimi._extract_blocks_payload
_extract_short_id_from_text = _kimi._extract_short_id_from_text
_resolve_env_template = _kimi._resolve_env_template
_extract_user_identity = _kimi._extract_user_identity
_is_standalone_slash_command = _kimi._is_standalone_slash_command
_parse_iso8601 = _kimi._parse_iso8601
_split_for_streaming = _kimi._split_for_streaming
_ulid_time_ms = _kimi._ulid_time_ms
_standalone_send = _kimi._standalone_send
check_kimi_requirements = _kimi.check_kimi_requirements
send_kimi_message = _kimi.send_kimi_message

from gateway.session import SessionSource


def setUpModule():
    """Make ``Platform("kimiclaw")`` resolvable in tests by populating the registry.

    The plugin's :func:`kimi_adapter.register` does this via
    ``ctx.register_platform()`` in production, but tests don't go through
    the plugin discovery flow.  Without this fixture, every test that
    constructs ``KimiAdapter`` or compares platforms fails because
    ``Platform._missing_`` returns ``None`` for unregistered names and
    ``Platform("kimiclaw")`` raises ``ValueError``.

    Skips silently in pre-v0.13.0 envs where ``gateway.platform_registry``
    doesn't exist — tests that depend on the new API will then fail with
    the same ``ImportError``/``ValueError`` they would at runtime, which
    is the right diagnostic signal.
    """
    try:
        from gateway.platform_registry import platform_registry, PlatformEntry
    except ImportError:
        return

    if not platform_registry.is_registered("kimiclaw"):
        platform_registry.register(PlatformEntry(
            name="kimiclaw",
            label="KimiClaw",
            adapter_factory=lambda cfg: None,  # not invoked in unit tests
            check_fn=lambda: True,
        ))


class _FakeWSStatusError(Exception):
    """Mimic websockets 12+ upgrade-rejection exception shape."""
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status_code = status


def _cfg(**extra) -> PlatformConfig:
    """Test config factory."""
    defaults = {"enable_dms": True, "enable_groups": True}
    defaults.update(extra)
    return PlatformConfig(
        enabled=True,
        token="km_b_prod_TEST_TOKEN",
        extra=defaults,
    )


class HelpersTests(unittest.TestCase):
    def test_is_standalone_slash_command(self):
        self.assertTrue(_is_standalone_slash_command("/status"))
        self.assertTrue(_is_standalone_slash_command("  /new  "))
        self.assertTrue(_is_standalone_slash_command("/compact"))
        self.assertFalse(_is_standalone_slash_command("/status please"))
        self.assertFalse(_is_standalone_slash_command("hello /status"))
        self.assertFalse(_is_standalone_slash_command("hi"))

    def test_split_for_streaming_short(self):
        self.assertEqual(_split_for_streaming("hi", 100), ["hi"])

    def test_split_for_streaming_long(self):
        text = "a" * 8000
        chunks = _split_for_streaming(text, 3500)
        # Rejoining should preserve length (minus stripped whitespace between).
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 3500 for c in chunks))

    def test_split_for_streaming_respects_size_and_preserves_content(self):
        text = "para one.\n\npara two.\n\npara three.\n\n" + ("x" * 5000)
        chunks = _split_for_streaming(text, 3500)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 3500 for c in chunks))
        # Length preservation (minus leading whitespace stripped between chunks).
        rejoined_len = sum(len(c) for c in chunks)
        self.assertGreaterEqual(rejoined_len, len(text) - 4 * len(chunks))

    def test_parse_iso8601(self):
        self.assertIsNotNone(_parse_iso8601("2026-04-23T16:53:48Z"))
        self.assertIsNotNone(_parse_iso8601("2026-04-23T16:53:48+00:00"))
        self.assertIsNone(_parse_iso8601("not-a-date"))
        self.assertIsNone(_parse_iso8601(""))


class RequirementsTests(unittest.TestCase):
    def test_dependencies_available(self):
        # websockets + aiohttp ship via the hermes-agent[messaging] extra —
        # the test env installs them, so this should always pass.
        self.assertTrue(check_kimi_requirements())


class CheckForRegistryTests(unittest.TestCase):
    """Auto-enable gating: deps alone must NOT enable KimiClaw —
    ``KIMI_BOT_TOKEN`` must also be set. Prevents the platform from
    lighting up on installs that have the ``[messaging]`` extra now
    that we declare ``websockets`` there. Mirrors LINE / SimpleX /
    Google Chat precedent.
    """

    def test_registry_check_requires_bot_token(self):
        from kimi_adapter import _check_for_registry
        env_no_kimi = {k: v for k, v in os.environ.items()
                       if not k.startswith("KIMI_")}
        with patch.dict(os.environ, env_no_kimi, clear=True):
            self.assertFalse(_check_for_registry())

    def test_registry_check_passes_with_bot_token(self):
        from kimi_adapter import _check_for_registry
        with patch.dict(os.environ, {"KIMI_BOT_TOKEN": "km_b_prod_TEST"}, clear=False):
            self.assertTrue(_check_for_registry())


class EnvTemplateResolverTests(unittest.TestCase):
    """Regression test for the 2026-05-16 token-substitution bug.

    Hermes does not invoke ``${VAR}`` substitution for external-plugin
    PlatformConfig fields, so ``token: ${KIMI_BOT_TOKEN}`` in config.yaml
    used to arrive as a truthy literal and short-circuit the
    ``or os.getenv(...)`` fallback chain in ``__init__``.  The adapter
    then sent the template string to kimi.com as the bot token (HTTP 401).
    """

    def test_resolves_dollar_brace_to_env(self):
        with patch.dict(os.environ, {"PROBE_VAR_A": "actual-secret-value"}, clear=False):
            self.assertEqual(
                _resolve_env_template("${PROBE_VAR_A}"), "actual-secret-value"
            )

    def test_unset_env_var_returns_empty(self):
        # Ensure the variable is unset.
        os.environ.pop("PROBE_VAR_UNSET_XYZ", None)
        self.assertEqual(_resolve_env_template("${PROBE_VAR_UNSET_XYZ}"), "")

    def test_plain_string_passthrough(self):
        self.assertEqual(_resolve_env_template("km_b_prod_real_token"), "km_b_prod_real_token")

    def test_none_returns_empty(self):
        self.assertEqual(_resolve_env_template(None), "")

    def test_partial_template_passthrough(self):
        # Strings that *look* like templates but don't match the full
        # ``${...}`` shape are passed through unchanged.
        self.assertEqual(_resolve_env_template("${incomplete"), "${incomplete")
        self.assertEqual(_resolve_env_template("trailing${VAR}"), "trailing${VAR}")
        self.assertEqual(_resolve_env_template("${}"), "${}")

    def test_whitespace_inside_template(self):
        with patch.dict(os.environ, {"PROBE_VAR_B": "spaced-value"}, clear=False):
            self.assertEqual(
                _resolve_env_template("${ PROBE_VAR_B }"), "spaced-value"
            )

    def test_non_string_coerces(self):
        self.assertEqual(_resolve_env_template(42), "42")
        self.assertEqual(_resolve_env_template(True), "True")


class AdapterTokenResolutionTests(unittest.TestCase):
    """End-to-end: the adapter ``__init__`` resolves ``${KIMI_BOT_TOKEN}``
    in ``config.token`` from the env, not literally."""

    # The unique sentinel below is what the test SHOULD see if the resolver
    # actually ran.  If we just asserted on a generic value like
    # ``"km_b_prod_RESOLVED_TOKEN"`` and the dev's shell happened to export
    # ``KIMI_BOT_TOKEN`` with that exact value, the assertion would pass
    # even if ``_resolve_env_template`` were broken.  A UUID-derived
    # sentinel makes that vanishingly unlikely.
    _SENTINEL = "km_b_prod_TEST_SENTINEL_" + uuid.uuid4().hex

    def test_template_in_config_token_resolves_from_env(self):
        # Pop any pre-existing KIMI_BOT_TOKEN from the shell so the
        # ``patch.dict(..., clear=False)`` block is what's actually
        # being asserted against, not a leaky shell var.
        prior = os.environ.pop("KIMI_BOT_TOKEN", None)
        try:
            cfg = PlatformConfig(
                enabled=True,
                token="${KIMI_BOT_TOKEN}",  # the bug pattern
                extra={"enable_dms": True, "enable_groups": False},
            )
            with patch.dict(
                os.environ, {"KIMI_BOT_TOKEN": self._SENTINEL}, clear=False
            ):
                adapter = KimiAdapter(cfg)
            self.assertEqual(adapter._bot_token, self._SENTINEL)
            self.assertNotIn("${", adapter._bot_token)
        finally:
            if prior is not None:
                os.environ["KIMI_BOT_TOKEN"] = prior


class StandaloneSendTokenResolutionTests(unittest.TestCase):
    """Regression test for the 2026-05-16 v2.0.1 fix.

    The standalone ``send_kimi_message`` helper (used by cron delivery and
    ``send_message_tool`` when no live adapter is available) shares the
    same bot-token resolution surface as ``KimiAdapter.__init__``.  The
    v2.0.0 fix wrapped ``__init__`` only, so a ``token: ${KIMI_BOT_TOKEN}``
    config.yaml line would 401 silently on every cron-driven kimi
    delivery while the live bot path worked fine.  These tests verify the
    helper now resolves the template before reaching ``_runtime_headers``.
    """

    _SENTINEL = "km_b_prod_STANDALONE_SENTINEL_" + uuid.uuid4().hex

    def test_template_in_config_token_resolves_for_standalone_send(self):
        prior = os.environ.pop("KIMI_BOT_TOKEN", None)
        try:
            cfg = PlatformConfig(
                enabled=True,
                token="${KIMI_BOT_TOKEN}",  # the bug pattern
                extra={"enable_groups": True},
            )
            with patch.dict(
                os.environ, {"KIMI_BOT_TOKEN": self._SENTINEL}, clear=False
            ):
                # Mock _runtime_headers to capture the bot_token it
                # receives, then short-circuit the rest of the send by
                # making the aiohttp POST raise — we don't care what
                # happens after token resolution.
                with patch("kimi_adapter._runtime_headers") as mock_headers:
                    mock_headers.return_value = {}
                    with patch("kimi_adapter.aiohttp.ClientSession") as mock_session:
                        mock_session.side_effect = RuntimeError("short-circuit")
                        try:
                            asyncio.run(
                                send_kimi_message(
                                    cfg,
                                    chat_id="room:test-room-id",
                                    text="hello",
                                )
                            )
                        except RuntimeError:
                            pass  # expected
            mock_headers.assert_called_once()
            kwargs = mock_headers.call_args.kwargs
            self.assertEqual(kwargs["bot_token"], self._SENTINEL)
            self.assertNotIn("${", kwargs["bot_token"])
        finally:
            if prior is not None:
                os.environ["KIMI_BOT_TOKEN"] = prior

    def test_unresolved_template_yields_no_token_error(self):
        # When KIMI_BOT_TOKEN is unset, the template resolves to empty
        # string, and the empty-token guard should fire BEFORE any HTTP
        # call. We must not send a literal "${KIMI_BOT_TOKEN}" to kimi.
        prior = os.environ.pop("KIMI_BOT_TOKEN", None)
        try:
            cfg = PlatformConfig(
                enabled=True,
                token="${KIMI_BOT_TOKEN}",
                extra={"enable_groups": True},
            )
            result = asyncio.run(
                send_kimi_message(
                    cfg,
                    chat_id="room:test-room-id",
                    text="hello",
                )
            )
            self.assertFalse(result.success)
            self.assertIn("no bot_token configured", result.error)
            self.assertNotIn("${", result.error)  # nothing template-shaped leaked
        finally:
            if prior is not None:
                os.environ["KIMI_BOT_TOKEN"] = prior


class StandaloneSendRegistryWrapperTests(unittest.TestCase):
    """The registered standalone_sender_fn wrapper matches upstream's contract.

    ``send_message_tool`` applies ``filter_media_delivery_paths()`` before
    invoking the wrapper, so ``media_files`` arrives as ``(path, is_voice)``
    tuples; these tests assert that shape is normalized to bare upload paths.
    """

    def test_success_result_converts_to_send_message_tool_dict(self):
        cfg = PlatformConfig(enabled=True, token="tok", extra={})
        with patch(
            "kimi_adapter.send_kimi_message",
            new=AsyncMock(return_value=SendResult(success=True, message_id="msg-123")),
        ) as mock_send:
            result = asyncio.run(
                _standalone_send(
                    cfg,
                    "room:abc",
                    "hello",
                    thread_id="thread-1",
                    # Real registry contract: filter_media_delivery_paths()
                    # yields (path, is_voice) tuples before standalone_sender_fn.
                    media_files=[("/tmp/a.png", False)],
                )
            )

        self.assertEqual(result, {"success": True, "message_id": "msg-123"})
        mock_send.assert_awaited_once_with(
            cfg,
            "room:abc",
            "hello",
            thread_id="thread-1",
            media_paths=["/tmp/a.png"],
        )

    def test_failure_result_converts_to_error_dict(self):
        cfg = PlatformConfig(enabled=True, token="tok", extra={})
        with patch(
            "kimi_adapter.send_kimi_message",
            new=AsyncMock(
                return_value=SendResult(
                    success=False,
                    error="network error",
                    retryable=True,
                )
            ),
        ):
            result = asyncio.run(_standalone_send(cfg, "room:abc", "hello"))

        self.assertEqual(result, {"error": "network error", "retryable": True})

    def test_force_document_is_accepted_but_ignored(self):
        cfg = PlatformConfig(enabled=True, token="tok", extra={})
        with patch(
            "kimi_adapter.send_kimi_message",
            new=AsyncMock(return_value=SendResult(success=True)),
        ) as mock_send:
            result = asyncio.run(
                _standalone_send(
                    cfg,
                    "room:abc",
                    "hello",
                    media_files=[("/tmp/a.pdf", False)],
                    force_document=True,
                )
            )

        self.assertEqual(result, {"success": True})
        mock_send.assert_awaited_once_with(
            cfg,
            "room:abc",
            "hello",
            thread_id=None,
            media_paths=["/tmp/a.pdf"],
        )

    def test_media_files_empty_list_forwards_empty(self):
        cfg = PlatformConfig(enabled=True, token="tok", extra={})
        with patch(
            "kimi_adapter.send_kimi_message",
            new=AsyncMock(return_value=SendResult(success=True)),
        ) as mock_send:
            asyncio.run(_standalone_send(cfg, "room:abc", "hi", media_files=[]))
        self.assertEqual(mock_send.await_args.kwargs["media_paths"], [])

    def test_media_files_none_default_forwards_empty(self):
        cfg = PlatformConfig(enabled=True, token="tok", extra={})
        with patch(
            "kimi_adapter.send_kimi_message",
            new=AsyncMock(return_value=SendResult(success=True)),
        ) as mock_send:
            asyncio.run(_standalone_send(cfg, "room:abc", "hi"))
        self.assertEqual(mock_send.await_args.kwargs["media_paths"], [])

    def test_media_files_voice_tuple_keeps_path_drops_flag(self):
        # A (path, True) voice tuple: KimiClaw uploads the file like any other
        # attachment and discards the voice flag (no voice-note wire concept).
        cfg = PlatformConfig(enabled=True, token="tok", extra={})
        with patch(
            "kimi_adapter.send_kimi_message",
            new=AsyncMock(return_value=SendResult(success=True)),
        ) as mock_send:
            asyncio.run(
                _standalone_send(
                    cfg, "room:abc", "voice", media_files=[("/tmp/v.ogg", True)]
                )
            )
        self.assertEqual(mock_send.await_args.kwargs["media_paths"], ["/tmp/v.ogg"])

    def test_media_files_mixed_tuples_normalize_to_paths(self):
        # Multiple attachments, mixed voice flags -> ordered list of bare paths.
        cfg = PlatformConfig(enabled=True, token="tok", extra={})
        with patch(
            "kimi_adapter.send_kimi_message",
            new=AsyncMock(return_value=SendResult(success=True)),
        ) as mock_send:
            asyncio.run(
                _standalone_send(
                    cfg,
                    "room:abc",
                    "both",
                    media_files=[("/tmp/a.png", False), ("/tmp/v.ogg", True)],
                )
            )
        self.assertEqual(
            mock_send.await_args.kwargs["media_paths"],
            ["/tmp/a.png", "/tmp/v.ogg"],
        )


class SendKimiMessageStandalonePolicyTests(unittest.IsolatedAsyncioTestCase):
    """v2.1.5 regression coverage for the standalone send path's timeout policy.

    v2.1.4 marked ``asyncio.TimeoutError`` non-retryable in the live
    ``send()`` arm, but the standalone ``send_kimi_message`` helper (used by
    cron delivery and by ``send_message_tool`` when no live adapter is
    available) only caught ``aiohttp.ClientError``.  Bare TimeoutError
    propagated uncaught, so cron-driven Kimi sends could still duplicate
    via any retry layer that interpreted the absent ``retryable`` field as
    "retry".  v2.1.5 adds ``except asyncio.TimeoutError`` ahead of the
    existing ``except aiohttp.ClientError`` clause.
    """

    async def test_standalone_timeout_returns_non_retryable(self):
        cfg = PlatformConfig(
            enabled=True,
            token="km_b_prod_STANDALONE_TIMEOUT_TEST",
            extra={"enable_groups": True},
        )

        # Patch aiohttp.ClientSession so session.post(...) raises
        # asyncio.TimeoutError — the exact failure mode the v2.1.5 except
        # clause catches.
        class _TimeoutPostCtx:
            async def __aenter__(self_inner):
                raise asyncio.TimeoutError()

            async def __aexit__(self_inner, *exc):
                return False

        fake_session = MagicMock()
        fake_session.post = MagicMock(return_value=_TimeoutPostCtx())

        class _FakeSessionFactory:
            async def __aenter__(self_inner):
                return fake_session

            async def __aexit__(self_inner, *exc):
                return False

        with patch(
            "kimi_adapter.aiohttp.ClientSession",
            return_value=_FakeSessionFactory(),
        ):
            with self.assertLogs("kimi_adapter", level="WARNING") as cm:
                result = await send_kimi_message(
                    cfg,
                    chat_id="room:test-timeout-uuid",
                    text="hello",
                )

        self.assertFalse(result.success)
        self.assertFalse(result.retryable)
        self.assertIn("timed out", result.error)
        warning_text = "\n".join(cm.output)
        self.assertIn("room:test-timeout-uuid", warning_text)
        self.assertIn("Marking non-retryable", warning_text)
        # Mirror the live-arm message phrasing for cross-arm consistency.
        self.assertIn("standalone SendMessage", warning_text)

    async def test_standalone_network_error_remains_retryable(self):
        # ClientError (genuine network failure) should still be retryable —
        # only TimeoutError gets the non-retryable treatment.
        import aiohttp as _aiohttp  # local import; not in top-level test imports

        cfg = PlatformConfig(
            enabled=True,
            token="km_b_prod_STANDALONE_NETERR_TEST",
            extra={"enable_groups": True},
        )

        class _NetErrPostCtx:
            async def __aenter__(self_inner):
                raise _aiohttp.ClientError("simulated network drop")

            async def __aexit__(self_inner, *exc):
                return False

        fake_session = MagicMock()
        fake_session.post = MagicMock(return_value=_NetErrPostCtx())

        class _FakeSessionFactory:
            async def __aenter__(self_inner):
                return fake_session

            async def __aexit__(self_inner, *exc):
                return False

        with patch(
            "kimi_adapter.aiohttp.ClientSession",
            return_value=_FakeSessionFactory(),
        ):
            result = await send_kimi_message(
                cfg,
                chat_id="room:test-neterr-uuid",
                text="hello",
            )

        self.assertFalse(result.success)
        self.assertTrue(result.retryable)
        self.assertIn("network error", result.error)


class CrossLoopSessionTests(unittest.TestCase):
    """Regression coverage for the cross-loop aiohttp ``ClientSession`` bug.

    Hermes's ``send_message_tool`` dispatches ``adapter.send()`` from a
    worker-thread event loop via ``_run_async`` →
    ``worker_loop.run_until_complete``.  But ``adapter.connect()`` runs
    on the gateway's main loop and creates ``self._http_session`` there;
    aiohttp binds ``ClientSession`` to the running loop at ``__init__``.
    Using the gateway-bound session from a worker loop later raises
    ``RuntimeError("Timeout context manager should be used inside a
    task")`` because ``asyncio.current_task(loop=session._loop)`` returns
    ``None`` from the worker loop's perspective.

    ``_session_for_current_loop`` is the plugin-side workaround.  An
    upstream issue tracks the broader fix in
    ``send_message_tool._send_via_adapter``.
    """

    def _make_adapter(self) -> KimiAdapter:
        return KimiAdapter(_cfg())

    def test_first_call_caches_session_on_current_loop(self):
        adapter = self._make_adapter()
        self.assertIsNone(adapter._http_session)
        self.assertIsNone(adapter._http_session_loop)

        async def scenario():
            async with adapter._session_for_current_loop() as session:
                self.assertIs(adapter._http_session, session)
                self.assertIs(
                    adapter._http_session_loop,
                    asyncio.get_running_loop(),
                )
                return session

        # Use new_event_loop + run_until_complete (not asyncio.run) so the
        # loop stays alive long enough to close the cached session cleanly
        # — otherwise the session is bound to a closed loop on teardown
        # and aiohttp emits an "Unclosed client session" ResourceWarning.
        loop = asyncio.new_event_loop()
        try:
            cached = loop.run_until_complete(scenario())
            self.assertIs(adapter._http_session, cached)
            self.assertFalse(cached.closed)
        finally:
            if adapter._http_session is not None and not adapter._http_session.closed:
                loop.run_until_complete(adapter._http_session.close())
            loop.close()

    def test_same_loop_reuses_cached_session(self):
        adapter = self._make_adapter()

        async def scenario():
            async with adapter._session_for_current_loop() as s1:
                first = s1
            async with adapter._session_for_current_loop() as s2:
                second = s2
            return first, second

        loop = asyncio.new_event_loop()
        try:
            first, second = loop.run_until_complete(scenario())
            self.assertIs(first, second, "Same loop must reuse the cached session")
        finally:
            if adapter._http_session is not None and not adapter._http_session.closed:
                loop.run_until_complete(adapter._http_session.close())
            loop.close()

    def test_cross_loop_yields_ephemeral_session(self):
        """The exact scenario send_message_tool exercises in production."""
        adapter = self._make_adapter()

        # Loop A: simulate connect() — populate _http_session and
        # _http_session_loop. Keep the loop alive afterwards so the
        # cached session stays bound to a live loop.
        loop_a = asyncio.new_event_loop()
        try:
            async def seed_on_loop_a():
                async with adapter._session_for_current_loop() as s:
                    return s
            cached = loop_a.run_until_complete(seed_on_loop_a())
            self.assertIs(adapter._http_session, cached)
            self.assertIs(adapter._http_session_loop, loop_a)

            # Loop B: simulate send_message_tool's worker-thread dispatch.
            loop_b = asyncio.new_event_loop()
            try:
                ephemerals = []
                async def use_on_loop_b():
                    async with adapter._session_for_current_loop() as s:
                        ephemerals.append(s)
                        self.assertIsNot(
                            s, cached,
                            "Cross-loop call must NOT yield the cached session",
                        )
                        self.assertFalse(
                            s.closed,
                            "Ephemeral session must be open inside the block",
                        )
                loop_b.run_until_complete(use_on_loop_b())
                self.assertTrue(
                    ephemerals[0].closed,
                    "Ephemeral session must be closed after the block exits",
                )
                # Cache is untouched.
                self.assertIs(adapter._http_session, cached)
                self.assertIs(adapter._http_session_loop, loop_a)
            finally:
                loop_b.close()
        finally:
            # Close the cached session on its own loop before tearing down.
            if adapter._http_session is not None:
                try:
                    loop_a.run_until_complete(adapter._http_session.close())
                except Exception:
                    pass
            loop_a.close()

    def test_cleanup_resets_session_loop_tracking(self):
        adapter = self._make_adapter()

        async def scenario():
            async with adapter._session_for_current_loop():
                pass
            self.assertIsNotNone(adapter._http_session)
            self.assertIsNotNone(adapter._http_session_loop)
            await adapter._cleanup_http()
            self.assertIsNone(adapter._http_session)
            self.assertIsNone(adapter._http_session_loop)

        asyncio.run(scenario())


class ProxyTrustEnvTests(unittest.IsolatedAsyncioTestCase):
    """Every ``aiohttp.ClientSession`` constructed by this plugin must set
    ``trust_env=True`` so that ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``ALL_PROXY``
    are honoured.  aiohttp's default is ``False``, which silently routes
    traffic around the user's proxy and breaks corporate-network deployments.

    Symmetric with the in-tree Yuanbao / WeCom / Weixin / Matrix adapters and
    with upstream commit ``c1ae18ee8`` (the SMS / Slack / Teams / Google-Chat
    sweep).  Tests intentionally cover all five construction sites so a
    refactor cannot silently drop the flag at one of them.
    """

    def _assert_trust_env(self, mock_session_cls) -> None:
        """All calls to the mock ClientSession ctor must include trust_env=True."""
        self.assertTrue(
            mock_session_cls.called,
            "Expected aiohttp.ClientSession to be constructed at least once",
        )
        for call in mock_session_cls.call_args_list:
            _args, kwargs = call
            self.assertTrue(
                kwargs.get("trust_env") is True,
                f"ClientSession constructed without trust_env=True: {call!r}",
            )

    async def test_connect_persistent_session_sets_trust_env(self):
        # Strategy: patch aiohttp.ClientSession to RAISE after recording the
        # call. connect() will bail at the session-construction line, but the
        # mock has already captured kwargs by then. Avoids having to mock the
        # full WS/RPC stack.
        adapter = KimiAdapter(_cfg())
        adapter._bot_token = "test-token"
        sentinel_exc = RuntimeError("test exit after session ctor")
        with patch("kimi_adapter.aiohttp.ClientSession", side_effect=sentinel_exc) as mock_session_cls, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True), \
             patch("kimi_adapter.check_kimi_requirements", return_value=True):
            try:
                await adapter.connect()
            except RuntimeError as exc:
                if exc is not sentinel_exc:
                    raise
            self._assert_trust_env(mock_session_cls)

    async def test_session_for_current_loop_lazy_creation_sets_trust_env(self):
        adapter = KimiAdapter(_cfg())
        self.assertIsNone(adapter._http_session)
        with patch("kimi_adapter.aiohttp.ClientSession") as mock_session_cls:
            # Make the mock instance look like a live, unclosed session so
            # the cross-loop branch doesn't try to close it on cleanup.
            mock_session_cls.return_value.closed = False
            mock_session_cls.return_value.close = AsyncMock()
            async with adapter._session_for_current_loop():
                pass
            self._assert_trust_env(mock_session_cls)

    async def test_session_for_current_loop_stale_closed_replacement_sets_trust_env(self):
        adapter = KimiAdapter(_cfg())
        # Seed a fake closed session bound to a different loop so the helper
        # takes the "stale closed" branch.
        stale = MagicMock()
        stale.closed = True
        adapter._http_session = stale
        adapter._http_session_loop = None
        with patch("kimi_adapter.aiohttp.ClientSession") as mock_session_cls:
            mock_session_cls.return_value.closed = False
            mock_session_cls.return_value.close = AsyncMock()
            async with adapter._session_for_current_loop():
                pass
            self._assert_trust_env(mock_session_cls)

    def test_session_for_current_loop_cross_loop_ephemeral_sets_trust_env(self):
        # Cross-loop branch requires two real event loops, mirroring
        # test_cross_loop_yields_ephemeral_session.
        adapter = KimiAdapter(_cfg())
        loop_a = asyncio.new_event_loop()
        try:
            # Seed a real session on loop_a so the cached-session check passes.
            async def seed():
                async with adapter._session_for_current_loop() as s:
                    return s
            loop_a.run_until_complete(seed())
            self.assertIs(adapter._http_session_loop, loop_a)

            # Now exercise loop_b with the constructor patched.
            loop_b = asyncio.new_event_loop()
            try:
                with patch("kimi_adapter.aiohttp.ClientSession") as mock_session_cls:
                    mock_session_cls.return_value.closed = False
                    mock_session_cls.return_value.close = AsyncMock()
                    mock_session_cls.return_value.__aenter__ = AsyncMock(
                        return_value=mock_session_cls.return_value,
                    )
                    mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)
                    async def use_on_loop_b():
                        async with adapter._session_for_current_loop():
                            pass
                    loop_b.run_until_complete(use_on_loop_b())
                    self._assert_trust_env(mock_session_cls)
            finally:
                loop_b.close()
        finally:
            if adapter._http_session is not None and not adapter._http_session.closed:
                loop_a.run_until_complete(adapter._http_session.close())
            loop_a.close()

    async def test_standalone_send_kimi_message_session_sets_trust_env(self):
        # Exercise the module-level send_kimi_message() helper used by cron
        # and by send_message_tool fallbacks. Use the same _FakeSessionFactory
        # pattern as SendKimiMessageStandalonePolicyTests so we don't depend
        # on the real network or a specific response shape.
        cfg = PlatformConfig(
            enabled=True,
            token="km_b_prod_TRUST_ENV_STANDALONE_TEST",
            extra={"enable_groups": True},
        )

        class _BailPostCtx:
            async def __aenter__(self_inner):
                raise RuntimeError("test exit after post()")

            async def __aexit__(self_inner, *exc):
                return False

        fake_session = MagicMock()
        fake_session.post = MagicMock(return_value=_BailPostCtx())

        class _FakeSessionFactory:
            async def __aenter__(self_inner):
                return fake_session

            async def __aexit__(self_inner, *exc):
                return False

        with patch(
            "kimi_adapter.aiohttp.ClientSession",
            return_value=_FakeSessionFactory(),
        ) as mock_session_cls:
            try:
                await send_kimi_message(
                    cfg,
                    chat_id="room:00000000-0000-0000-0000-000000000000",
                    text="hello",
                )
            except Exception:
                # post() raises by design; we only care that the
                # ClientSession constructor recorded trust_env=True.
                pass
            self._assert_trust_env(mock_session_cls)


class AdapterInitTests(unittest.TestCase):
    def test_config_parsing(self):
        cfg = _cfg(
            enable_dms=True,
            enable_groups=False,
            openclaw_version="2026.5.1",
            claw_id="custom-id-123",
            group_require_mention=True,
            user_message_prefix="FROM KIMI: ",
        )
        adapter = KimiAdapter(cfg)
        self.assertEqual(adapter._bot_token, "km_b_prod_TEST_TOKEN")
        self.assertTrue(adapter._enable_dms)
        self.assertFalse(adapter._enable_groups)
        self.assertEqual(adapter._openclaw_version, "2026.5.1")
        self.assertEqual(adapter._claw_id, "custom-id-123")
        self.assertTrue(adapter._group_require_mention)
        self.assertEqual(adapter._user_message_prefix, "FROM KIMI: ")

    def test_claw_id_auto_generated(self):
        adapter = KimiAdapter(_cfg())
        self.assertTrue(adapter._claw_id.startswith("hermes-kimi-"))
        self.assertEqual(len(adapter._claw_id), len("hermes-kimi-") + 16)

    def test_ws_upgrade_headers_include_runtime_metadata(self):
        adapter = KimiAdapter(_cfg())
        headers = adapter._ws_upgrade_headers()
        self.assertEqual(headers["X-Kimi-Bot-Token"], "km_b_prod_TEST_TOKEN")
        self.assertEqual(headers["X-Kimi-OpenClaw-Version"], "2026.3.13")
        self.assertIn("X-Kimi-Claw-ID", headers)
        self.assertIn("X-Kimi-OpenClaw-Plugins", headers)

    def test_ws_upgrade_headers_omit_empty_runtime_metadata(self):
        adapter = KimiAdapter(_cfg(openclaw_version="", openclaw_skills=""))
        headers = adapter._ws_upgrade_headers()
        self.assertNotIn("X-Kimi-OpenClaw-Version", headers)
        # Skills default is empty — should not be included.
        self.assertNotIn("X-Kimi-OpenClaw-Skills", headers)

    def test_http_headers_unary_vs_streaming(self):
        adapter = KimiAdapter(_cfg())
        unary = adapter._http_headers(streaming=False)
        streaming = adapter._http_headers(streaming=True)
        self.assertEqual(unary["Content-Type"], "application/json")
        self.assertEqual(streaming["Content-Type"], "application/connect+json")
        self.assertEqual(streaming["Accept"], "application/connect+json")
        self.assertEqual(unary["X-Kimi-Bot-Token"], "km_b_prod_TEST_TOKEN")
        self.assertEqual(unary["X-Kimi-Claw-Version"], "0.25.0")
        self.assertEqual(unary["X-Kimi-OpenClaw-Version"], "2026.3.13")
        self.assertIn("X-Kimi-Claw-ID", unary)
        self.assertEqual(
            json.loads(unary["X-Kimi-OpenClaw-Plugins"]),
            [{"id": "kimi-claw", "version": "0.25.0"}],
        )
        self.assertEqual(streaming["X-Kimi-OpenClaw-Plugins"], unary["X-Kimi-OpenClaw-Plugins"])

    def test_legacy_inventory_header_strings_are_normalized_to_json(self):
        adapter = KimiAdapter(_cfg(
            openclaw_plugins="kimi-claw",
            openclaw_skills="kimiim,worker-safety",
        ))

        headers = adapter._http_headers(streaming=False)

        self.assertEqual(
            json.loads(headers["X-Kimi-OpenClaw-Plugins"]),
            [{"id": "kimi-claw", "version": "0.25.0"}],
        )
        self.assertEqual(
            json.loads(headers["X-Kimi-OpenClaw-Skills"]),
            ["kimiim", "worker-safety"],
        )


class EnvelopeCodecTests(unittest.TestCase):
    def test_encode_envelope_data(self):
        adapter = KimiAdapter(_cfg())
        body = adapter._encode_envelope(b'{"hello": "world"}')
        self.assertEqual(body[0], 0)  # data flag
        length = struct.unpack(">I", body[1:5])[0]
        self.assertEqual(length, len(b'{"hello": "world"}'))
        self.assertEqual(body[5:], b'{"hello": "world"}')

    def test_encode_envelope_end_stream(self):
        adapter = KimiAdapter(_cfg())
        body = adapter._encode_envelope(b"{}", end_stream=True)
        self.assertEqual(body[0], _CONNECT_FLAG_END_STREAM)


class EnvelopeParserTests(unittest.IsolatedAsyncioTestCase):
    """Feed synthetic byte streams through the envelope parser."""

    async def _collect(self, adapter: KimiAdapter, stream: bytes):
        reader = _FakeStreamReader(stream)
        out = []
        async for msg in adapter._connect_envelope_parser(reader):
            out.append(msg)
        return out

    async def test_single_data_frame_then_end_stream(self):
        adapter = KimiAdapter(_cfg())
        data = b'{"ping":{}}'
        stream = (
            bytes([0]) + struct.pack(">I", len(data)) + data
            + bytes([_CONNECT_FLAG_END_STREAM]) + struct.pack(">I", 2) + b"{}"
        )
        msgs = await self._collect(adapter, stream)
        self.assertEqual(msgs, [{"ping": {}}])

    async def test_multiple_data_frames(self):
        adapter = KimiAdapter(_cfg())
        parts = [b'{"ping":{}}', b'{"message":{"id":"x"}}']
        stream = b""
        for p in parts:
            stream += bytes([0]) + struct.pack(">I", len(p)) + p
        stream += bytes([_CONNECT_FLAG_END_STREAM]) + struct.pack(">I", 2) + b"{}"
        msgs = await self._collect(adapter, stream)
        self.assertEqual(msgs, [{"ping": {}}, {"message": {"id": "x"}}])

    async def test_end_stream_with_auth_error_raises(self):
        adapter = KimiAdapter(_cfg())
        err_body = json.dumps({
            "error": {"code": "unauthenticated", "message": "token expired"}
        }).encode()
        stream = bytes([_CONNECT_FLAG_END_STREAM]) + struct.pack(">I", len(err_body)) + err_body
        with self.assertRaises(KimiAuthError):
            await self._collect(adapter, stream)

    async def test_end_stream_with_rpc_error_raises(self):
        adapter = KimiAdapter(_cfg())
        err_body = json.dumps({"error": {"code": "invalid_argument", "message": "bad"}}).encode()
        stream = bytes([_CONNECT_FLAG_END_STREAM]) + struct.pack(">I", len(err_body)) + err_body
        with self.assertRaises(KimiRpcError):
            await self._collect(adapter, stream)

    async def test_compressed_frame_rejected(self):
        adapter = KimiAdapter(_cfg())
        stream = bytes([_CONNECT_FLAG_COMPRESSED]) + struct.pack(">I", 2) + b"{}"
        with self.assertRaises(KimiProtocolError):
            await self._collect(adapter, stream)

    async def test_malformed_json_rejected(self):
        adapter = KimiAdapter(_cfg())
        bad = b"{not json"
        stream = bytes([0]) + struct.pack(">I", len(bad)) + bad
        with self.assertRaises(KimiProtocolError):
            await self._collect(adapter, stream)


class _FakeStreamReader:
    """Minimal aiohttp.StreamReader stub for parser tests."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    async def readexactly(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise asyncio.IncompleteReadError(self._data[self._pos:], n)
        chunk = self._data[self._pos:self._pos + n]
        self._pos += n
        return chunk


class ChatIdRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_dm_prefix_routes_to_send_dm(self):
        adapter = KimiAdapter(_cfg())
        adapter._send_dm = AsyncMock(return_value=SendResult(success=True))
        adapter._send_group = AsyncMock()
        result = await adapter.send("dm:im:kimi:main", "hello")
        self.assertTrue(result.success)
        adapter._send_dm.assert_awaited_once()
        adapter._send_group.assert_not_awaited()

    async def test_room_prefix_routes_to_send_group(self):
        adapter = KimiAdapter(_cfg())
        adapter._send_dm = AsyncMock()
        adapter._send_group = AsyncMock(return_value=SendResult(success=True))
        result = await adapter.send("room:abc-def", "hello")
        self.assertTrue(result.success)
        adapter._send_group.assert_awaited_once()
        call_args = adapter._send_group.call_args
        self.assertEqual(call_args.args[0], "abc-def")  # room_id
        # thread_id position (from metadata) should be None
        self.assertIsNone(call_args.args[3])

    async def test_room_thread_suffix_extracts_thread(self):
        adapter = KimiAdapter(_cfg())
        adapter._send_group = AsyncMock(return_value=SendResult(success=True))
        await adapter.send("room:abc-def/thread-xyz", "hello")
        call_args = adapter._send_group.call_args
        self.assertEqual(call_args.args[0], "abc-def")
        self.assertEqual(call_args.args[3], "thread-xyz")

    async def test_thread_id_in_metadata(self):
        adapter = KimiAdapter(_cfg())
        adapter._send_group = AsyncMock(return_value=SendResult(success=True))
        await adapter.send("room:abc-def", "hello", metadata={"thread_id": "t-1"})
        call_args = adapter._send_group.call_args
        self.assertEqual(call_args.args[3], "t-1")

    async def test_unknown_prefix_fails_cleanly(self):
        adapter = KimiAdapter(_cfg())
        result = await adapter.send("weird:foo", "hello")
        self.assertFalse(result.success)
        self.assertFalse(result.retryable)
        self.assertIn("unknown chat_id format", result.error)


class SendArmExceptionPolicyTests(unittest.IsolatedAsyncioTestCase):
    """`send()` translates exceptions into ``SendResult.retryable``.

    Regression for v2.1.4 duplicate-send fix:  asyncio.TimeoutError used to
    be grouped with KimiTransientError and returned ``retryable=True``,
    which let the gateway's retry wrapper re-POST the same SendMessage on
    a fresh TCP connection.  Kimi.com's server-side delivery would already
    have succeeded by then, so the user saw the message twice.

    The diagnostic capture at 2026-05-17 18:46:03-18:46:37 BST proved a
    single client-side ``ClientTimeout(total=30)`` corresponds to a
    server-side accepted delivery (a concurrent SendMessage on the same
    adapter completed in 697 ms during the same 30 s hang window).  The
    fix marks TimeoutError non-retryable; ``KimiTransientError`` keeps
    its retryable=True contract for genuine network failures.
    """

    async def test_timeout_returns_non_retryable(self):
        # asyncio.TimeoutError from _send_group → retryable=False, warning logged.
        adapter = KimiAdapter(_cfg())
        adapter._send_group = AsyncMock(side_effect=asyncio.TimeoutError())
        with self.assertLogs("kimi_adapter", level="WARNING") as cm:
            result = await adapter.send("room:abc-def", "hello")
        self.assertFalse(result.success)
        self.assertFalse(result.retryable)
        self.assertIn("timed out", result.error)
        # WARNING includes chat_id + the configured timeout for ops triage.
        warning_text = "\n".join(cm.output)
        self.assertIn("room:abc-def", warning_text)
        self.assertIn("Marking non-retryable", warning_text)

    async def test_transient_error_remains_retryable(self):
        # KimiTransientError (genuine network failure) → retryable=True.
        adapter = KimiAdapter(_cfg())
        adapter._send_group = AsyncMock(
            side_effect=KimiTransientError("simulated network error"),
        )
        result = await adapter.send("room:abc-def", "hello")
        self.assertFalse(result.success)
        self.assertTrue(result.retryable)
        self.assertIn("simulated network error", result.error)

    async def test_auth_error_remains_non_retryable(self):
        from kimi_adapter import KimiAuthError
        adapter = KimiAdapter(_cfg())
        adapter._send_group = AsyncMock(
            side_effect=KimiAuthError("simulated 401"),
        )
        result = await adapter.send("room:abc-def", "hello")
        self.assertFalse(result.success)
        self.assertFalse(result.retryable)

    async def test_auth_error_surfaces_to_fatal_error_hook(self):
        # v2.1.7: the send-path KimiAuthError handler must call
        # ``_set_fatal_error`` so the gateway's per-platform circuit
        # breaker (upstream commit 518f39557, May 2026) sees the
        # platform as fatally broken and stops dispatching new sends.
        # Without this, the gateway would keep accepting send requests
        # against a dead adapter — silent outage from the operator's
        # perspective.  retryable=True (not False) at this layer because
        # send-time auth failures are ambiguous (transient token expiry
        # vs permanent revoke); reconnect re-evaluates and the connect-
        # loop fatal hook at ~L2269 / ~L2680 sets retryable=False if the
        # new auth attempt also fails.
        from kimi_adapter import KimiAuthError
        adapter = KimiAdapter(_cfg())
        adapter._send_group = AsyncMock(
            side_effect=KimiAuthError("simulated 401 mid-send"),
        )
        fatal_calls: list[tuple] = []

        def _record_fatal(code, message, *, retryable):
            fatal_calls.append((code, message, retryable))

        adapter._set_fatal_error = _record_fatal  # type: ignore[method-assign]

        with self.assertLogs("kimi_adapter", level="ERROR") as cm:
            result = await adapter.send("room:abc-def", "hello")

        # SendResult contract unchanged: retryable=False so the gateway's
        # send-retry wrapper does NOT re-attempt the dispatch (the failure
        # is auth, not network; retry without reconnect would just 401 again).
        self.assertFalse(result.success)
        self.assertFalse(result.retryable)
        self.assertIn("simulated 401 mid-send", result.error)

        # Fatal-error hook was called exactly once with the documented contract.
        self.assertEqual(len(fatal_calls), 1, fatal_calls)
        code, message, retryable = fatal_calls[0]
        self.assertEqual(code, "kimi_send_auth")
        self.assertIn("simulated 401 mid-send", message)
        self.assertTrue(retryable, "Send-path fatal must be retryable so reconnect re-evaluates")

        # ERROR log carries the chat_id for operator triage.
        log_text = "\n".join(cm.output)
        self.assertIn("room:abc-def", log_text)
        self.assertIn("marking platform", log_text.lower())

    async def test_auth_error_on_dm_path_also_surfaces_to_fatal_hook(self):
        # Dispatcher-level coverage: send()'s try/except wraps both
        # _send_dm and _send_group, so a KimiAuthError from either arm
        # must funnel through the same _set_fatal_error call.
        from kimi_adapter import KimiAuthError
        adapter = KimiAdapter(_cfg())
        adapter._send_dm = AsyncMock(
            side_effect=KimiAuthError("simulated 401 on dm arm"),
        )
        fatal_calls: list[tuple] = []
        adapter._set_fatal_error = lambda code, message, *, retryable: fatal_calls.append(  # type: ignore[method-assign]
            (code, message, retryable)
        )

        with self.assertLogs("kimi_adapter", level="ERROR"):
            result = await adapter.send("dm:im:kimi:main", "hello")

        self.assertFalse(result.retryable)
        self.assertEqual(len(fatal_calls), 1)
        self.assertEqual(fatal_calls[0][0], "kimi_send_auth")

    async def test_dm_path_timeout_also_non_retryable(self):
        # Dispatcher-level coverage: send()'s try/except at kimi_adapter.py:1809
        # wraps BOTH the _send_dm and _send_group dispatch arms.  This test
        # exercises the dm:* arm to prove a timeout from either arm collapses
        # to the same non-retryable SendResult contract — i.e. the v2.1.4 fix
        # lives at the dispatcher, not inside _send_group.  In production Kimi
        # delivers user-bot 1:1 conversations as room:<uuid>, so the dm:
        # prefix is rarely exercised live, but the contract guarantee must hold.
        adapter = KimiAdapter(_cfg())
        adapter._send_dm = AsyncMock(side_effect=asyncio.TimeoutError())
        with self.assertLogs("kimi_adapter", level="WARNING"):
            result = await adapter.send("dm:im:kimi:main", "hello")
        self.assertFalse(result.success)
        self.assertFalse(result.retryable)


class DedupTests(unittest.TestCase):
    def test_same_pair_deduped(self):
        adapter = KimiAdapter(_cfg())
        self.assertFalse(adapter._dedup_is_duplicate("group", "chat-1", "msg-1"))
        self.assertTrue(adapter._dedup_is_duplicate("group", "chat-1", "msg-1"))

    def test_different_kinds_not_deduped(self):
        adapter = KimiAdapter(_cfg())
        self.assertFalse(adapter._dedup_is_duplicate("group", "chat-1", "msg-1"))
        self.assertFalse(adapter._dedup_is_duplicate("dm", "chat-1", "msg-1"))

    def test_eviction_at_max(self):
        adapter = KimiAdapter(_cfg())
        maxlen = adapter._processed.maxlen
        # Fill + one overflow
        for i in range(maxlen + 1):
            adapter._dedup_is_duplicate("group", "chat", f"msg-{i}")
        # First key should be evicted now.
        self.assertFalse(adapter._dedup_is_duplicate("group", "chat", "msg-0"))


class MessageEventSynthesisTests(unittest.TestCase):
    def test_text_event(self):
        adapter = KimiAdapter(_cfg())
        event = adapter._build_message_event(
            kind="group",
            text="hello",
            message_id="mid-1",
            chat_id="room:abc",
            chat_name="Test Room",
            user_id="u-1",
            user_name="Alice",
        )
        self.assertEqual(event.text, "hello")
        self.assertEqual(event.message_type, MessageType.TEXT)
        self.assertEqual(event.source.platform, Platform("kimiclaw"))
        self.assertEqual(event.source.chat_id, "room:abc")
        self.assertEqual(event.source.chat_type, "group")
        self.assertFalse(event.internal)

    def test_command_event(self):
        adapter = KimiAdapter(_cfg())
        event = adapter._build_message_event(
            kind="dm",
            text="/reset",
            message_id="mid-2",
            chat_id="dm:im:kimi:main",
            chat_name="Kimi DM",
            user_id="u-2",
            user_name=None,
        )
        self.assertEqual(event.message_type, MessageType.COMMAND)

    def test_photo_event(self):
        adapter = KimiAdapter(_cfg())
        event = adapter._build_message_event(
            kind="group",
            text="look at this",
            message_id="mid-3",
            chat_id="room:abc",
            chat_name=None,
            user_id="u-3",
            user_name=None,
            media_urls=["https://example/img.jpg"],
            media_types=["image/jpeg"],
        )
        self.assertEqual(event.message_type, MessageType.PHOTO)
        self.assertEqual(event.media_urls, ["https://example/img.jpg"])

    def test_auto_skill_passthrough(self):
        adapter = KimiAdapter(_cfg(auto_skill="test-skill"))
        event = adapter._build_message_event(
            kind="group",
            text="hello",
            message_id="mid-4",
            chat_id="room:abc",
            chat_name=None,
            user_id=None,
            user_name=None,
        )
        self.assertEqual(event.auto_skill, "test-skill")


class GroupEventParsingTests(unittest.IsolatedAsyncioTestCase):
    """Kimi Subscribe protobuf-JSON events are converted into Hermes messages."""

    async def test_protobuf_json_chat_message_event_dispatches_summary(self):
        adapter = KimiAdapter(_cfg())
        adapter.handle_message = AsyncMock()  # type: ignore
        # Fix A: legacy summary-fallback test — disable hydration so the
        # cascade lands on the summary path WITHOUT making a real
        # _fetch_group_message → aiohttp call (Commit 6 made hydration
        # the default; without this the test attempts a real network
        # round-trip and only "passes" because the resulting transient
        # error falls through to the same summary fallback).
        adapter._hydrate_missing_text = False

        await adapter._on_group_event({
            "id": "evt-1",
            "chatMessage": {
                "chatId": "chat-1",
                "messageId": "msg-1",
                "status": "STATUS_COMPLETED",
                "senderId": "user-1",
                "senderShortId": "u1",
                "roomId": "room-1",
                "summary": "hello from kimi",
            },
        })

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        # Fix B: summary-fallback path now prepends a truncation marker
        # so the agent knows the body is a preview, not the full message.
        self.assertTrue(
            event.text.startswith("[message truncated"),
            f"expected truncation marker prefix, got: {event.text!r}",
        )
        self.assertIn("hello from kimi", event.text)
        self.assertEqual(event.source.chat_id, "room:chat-1")
        self.assertEqual(event.source.user_id, "user-1")
        self.assertEqual(event.source.user_name, "u1")

    async def test_generated_payload_shape_extracts_text_block(self):
        adapter = KimiAdapter(_cfg())
        adapter.handle_message = AsyncMock()  # type: ignore

        await adapter._on_group_event({
            "id": "evt-2",
            "payload": {
                "case": "chatMessage",
                "value": {
                    "chatId": "chat-2",
                    "messageId": "msg-2",
                    "status": 2,
                    "senderShortId": "u2",
                    "blocks": [
                        {
                            "id": "b1",
                            "content": {
                                "case": "text",
                                "value": {"content": "block text"},
                            },
                        }
                    ],
                },
            },
        })

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        self.assertEqual(event.text, "block text")
        self.assertEqual(event.source.user_id, "kimi:u2")

    async def test_generating_status_is_not_dispatched(self):
        adapter = KimiAdapter(_cfg())
        adapter.handle_message = AsyncMock()  # type: ignore

        await adapter._on_group_event({
            "id": "evt-3",
            "chatMessage": {
                "chatId": "chat-3",
                "messageId": "msg-3",
                "status": "STATUS_GENERATING",
                "summary": "partial",
            },
        })

        adapter.handle_message.assert_not_awaited()


class SendMessageShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_send_uses_kimi_blocks_request_shape(self):
        adapter = KimiAdapter(_cfg())
        adapter._rpc_unary = AsyncMock(return_value={"messageId": "sent-1"})  # type: ignore

        result = await adapter._send_group(
            "chat-1",
            "hello",
            reply_to="ignored",
            thread_id="ignored",
            metadata={},
        )

        self.assertTrue(result.success)
        adapter._rpc_unary.assert_awaited_once()
        method, body = adapter._rpc_unary.await_args.args
        self.assertEqual(method, "SendMessage")
        self.assertEqual(body["chatId"], "chat-1")
        self.assertEqual(body["blocks"][0]["text"]["content"], "hello")
        self.assertNotIn("text", body)
        self.assertNotIn("chat_id", body)

    async def test_group_send_supports_attachment_only_resource_link(self):
        adapter = KimiAdapter(_cfg())
        adapter._rpc_unary = AsyncMock(return_value={"messageId": "sent-2"})  # type: ignore

        result = await adapter._send_group(
            "chat-1",
            "",
            reply_to=None,
            thread_id=None,
            metadata={"attachments": [{
                "uri": "kimi-file://file-123",
                "name": "report.pdf",
                "mimeType": "application/pdf",
                "sizeBytes": 42,
            }]},
        )

        self.assertTrue(result.success)
        _method, body = adapter._rpc_unary.await_args.args
        self.assertEqual(len(body["blocks"]), 1)
        resource = body["blocks"][0]["resourceLink"]
        self.assertEqual(resource["uri"], "kimi-file://file-123")
        self.assertEqual(resource["title"], "report.pdf")
        self.assertNotIn("text", body["blocks"][0])

    async def test_send_document_uploads_to_kimi_file_resource(self):
        adapter = KimiAdapter(_cfg())
        # closed=False so _session_for_current_loop yields the mock instead of
        # falling into the "stale closed session — replace" branch and opening
        # a real aiohttp.ClientSession that nothing closes.
        adapter._http_session = MagicMock(closed=False)
        adapter._send_group = AsyncMock(return_value=SendResult(success=True, message_id="sent-3"))  # type: ignore
        uploaded = [{
            "uri": "kimi-file://file-456",
            "name": "notes.txt",
            "mimeType": "text/plain",
        }]

        with patch("kimi_adapter._upload_kimi_files", new=AsyncMock(return_value=uploaded)) as upload_mock:
            result = await adapter.send_document("room:chat-1", "/tmp/notes.txt", caption="see attached")

        self.assertTrue(result.success)
        upload_mock.assert_awaited_once()
        adapter._send_group.assert_awaited_once()
        args = adapter._send_group.await_args.args
        self.assertEqual(args[0], "chat-1")
        self.assertEqual(args[1], "see attached")
        self.assertEqual(adapter._send_group.await_args.kwargs["metadata"]["attachments"], uploaded)

    async def test_kimi_file_media_resolution_replaces_uri_with_local_path(self):
        adapter = KimiAdapter(_cfg())
        adapter._resolve_kimi_file_uri = AsyncMock(return_value={
            "localPath": "/tmp/kimi-file/report.pdf",
            "contentType": "application/pdf",
        })  # type: ignore

        urls, types = await adapter._resolve_kimi_file_media(
            ["kimi-file://12345678-1234-1234-1234-123456789abc", "https://example/a.png"],
            ["resource_link", "image/png"],
            message_id="msg-1",
        )

        self.assertEqual(urls, ["/tmp/kimi-file/report.pdf", "https://example/a.png"])
        self.assertEqual(types, ["application/pdf", "image/png"])


class GroupRpcHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_chat_info_fetches_room_and_members(self):
        adapter = KimiAdapter(_cfg())
        adapter._rpc_unary = AsyncMock(side_effect=[
            {"room": {"id": "room-1", "name": "Research Room"}},
            {"members": [{"id": "m1", "shortId": "alice"}]},
        ])  # type: ignore

        info = await adapter.get_chat_info("room:room-1")

        self.assertEqual(info["name"], "Research Room")
        self.assertEqual(info["members"], [{"id": "m1", "shortId": "alice"}])
        self.assertEqual(
            adapter._rpc_unary.await_args_list[0].args,
            ("GetRoom", {"roomId": "room-1"}),
        )
        self.assertEqual(adapter._rpc_unary.await_args_list[1].args[0], "ListMembers")
        self.assertEqual(adapter._rpc_unary.await_args_list[1].args[1]["roomId"], "room-1")

    async def test_empty_subscribe_event_hydrates_from_list_messages(self):
        adapter = KimiAdapter(_cfg())
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._rpc_unary = AsyncMock(return_value={
            "messages": [
                {
                    "senderId": "user-9",
                    "senderShortId": "u9",
                    "message": {
                        "id": "msg-9",
                        "blocks": [
                            {
                                "content": {
                                    "case": "text",
                                    "value": {"content": "hydrated text"},
                                },
                            }
                        ],
                    },
                }
            ]
        })  # type: ignore

        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-9",
                "messageId": "msg-9",
                "status": "STATUS_COMPLETED",
            },
        })

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        self.assertEqual(event.text, "hydrated text")
        self.assertEqual(event.source.user_id, "user-9")
        self.assertEqual(event.source.user_name, "u9")
        method, body = adapter._rpc_unary.await_args.args
        self.assertEqual(method, "ListMessages")
        self.assertEqual(body["chatId"], "chat-9")
        self.assertEqual(body["startMessageId"], "msg-9")

    async def test_failed_hydration_does_not_dispatch_or_dedup_empty_event(self):
        adapter = KimiAdapter(_cfg())
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._fetch_group_message = AsyncMock(side_effect=[
            KimiRpcError("temporary bad cursor"),
            {
                "id": "msg-10",
                "blocks": [
                    {
                        "content": {
                            "case": "text",
                            "value": {"content": "second replay text"},
                        },
                    }
                ],
            },
        ])  # type: ignore
        event = {
            "chatMessage": {
                "chatId": "chat-10",
                "messageId": "msg-10",
                "status": "STATUS_COMPLETED",
            },
        }

        await adapter._on_group_event(event)
        adapter.handle_message.assert_not_awaited()

        await adapter._on_group_event(event)
        adapter.handle_message.assert_awaited_once()
        delivered = adapter.handle_message.await_args.args[0]
        self.assertEqual(delivered.text, "second replay text")

    async def test_hydrated_self_message_is_filtered(self):
        adapter = KimiAdapter(_cfg())
        adapter._me_id = "bot-self"
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._fetch_group_message = AsyncMock(return_value={
            "id": "msg-self",
            "senderId": "bot-self",
            "blocks": [
                {
                    "content": {
                        "case": "text",
                        "value": {"content": "echo"},
                    },
                }
            ],
        })  # type: ignore

        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-self",
                "messageId": "msg-self",
                "status": "STATUS_COMPLETED",
            },
        })

        adapter.handle_message.assert_not_awaited()

    async def test_non_user_group_message_role_is_not_dispatched(self):
        adapter = KimiAdapter(_cfg())
        adapter.handle_message = AsyncMock()  # type: ignore
        # Fix A: summary-only event with hydration default-on would
        # otherwise trigger a real _fetch_group_message → aiohttp call.
        # Test asserts a role drop, not text content — disable hydration.
        adapter._hydrate_missing_text = False

        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-bot",
                "messageId": "msg-bot",
                "status": "STATUS_COMPLETED",
                "role": "ROLE_ASSISTANT",
                "senderId": "assistant-1",
                "summary": "assistant echo",
            },
        })

        adapter.handle_message.assert_not_awaited()

    async def test_hydrated_non_user_group_message_role_is_not_dispatched(self):
        adapter = KimiAdapter(_cfg())
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._fetch_group_message = AsyncMock(return_value={
            "id": "msg-hydrated-bot",
            "role": "ASSISTANT",
            "senderId": "assistant-2",
            "blocks": [
                {
                    "content": {
                        "case": "text",
                        "value": {"content": "hydrated assistant echo"},
                    },
                }
            ],
        })  # type: ignore

        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-hydrated-bot",
                "messageId": "msg-hydrated-bot",
                "status": "STATUS_COMPLETED",
            },
        })

        adapter.handle_message.assert_not_awaited()


def _bot_msg(**overrides) -> dict:
    """Build a minimal ROLE_ASSISTANT chatMessage event with a bot-role sender."""
    base = {
        "chatMessage": {
            "chatId": "chat-bot",
            "messageId": "msg-bot",
            "status": "STATUS_COMPLETED",
            "role": "ROLE_ASSISTANT",
            "senderId": "assistant-1",
            "senderShortId": "u_bot",
            "summary": "hello from bot",
        },
    }
    base["chatMessage"].update(overrides)
    return base


class GroupTrustedSenderTests(unittest.IsolatedAsyncioTestCase):
    """group_trusted_senders is an authoritative short_id / id allowlist.

    Fix A: ``_bot_msg()`` is summary-only (no inline blocks). With
    hydration default-on these tests would attempt a real
    ``_fetch_group_message`` → ``aiohttp`` call. Tests assert
    trust/policy behavior, not text content — disable hydration on each
    fixture.
    """

    async def test_group_trusted_sender_bypasses_role_filter(self):
        adapter = KimiAdapter(_cfg(group_trusted_senders=["u_bot"]))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False  # Fix A — see class docstring

        await adapter._on_group_event(_bot_msg())

        adapter.handle_message.assert_awaited_once()

    async def test_group_trusted_sender_by_id_also_matches(self):
        adapter = KimiAdapter(_cfg(group_trusted_senders=["assistant-1"]))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False  # Fix A — see class docstring

        # senderShortId not in allowlist; senderId IS — should still bypass.
        await adapter._on_group_event(_bot_msg(senderShortId="u_somebody_else"))

        adapter.handle_message.assert_awaited_once()


class GroupAllowBotSendersPolicyTests(unittest.IsolatedAsyncioTestCase):
    """group_allow_bot_senders policy: off | trusted_only | mentions | all.

    Fix A: every test in this class feeds ``_bot_msg()`` (summary-only).
    Disable hydration so the cascade doesn't attempt a real
    ``_fetch_group_message`` → ``aiohttp`` call when policy allows
    dispatch.
    """

    async def test_group_allow_bot_senders_off_drops_assistant(self):
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="off"))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False  # Fix A

        await adapter._on_group_event(_bot_msg())

        adapter.handle_message.assert_not_awaited()

    async def test_group_allow_bot_senders_trusted_only_drops_untrusted_assistant(self):
        adapter = KimiAdapter(_cfg(
            group_allow_bot_senders="trusted_only",
            group_trusted_senders=["u_other"],
        ))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False  # Fix A

        await adapter._on_group_event(_bot_msg())

        adapter.handle_message.assert_not_awaited()

    async def test_group_allow_bot_senders_trusted_only_allows_trusted_assistant(self):
        adapter = KimiAdapter(_cfg(
            group_allow_bot_senders="trusted_only",
            group_trusted_senders=["u_bot"],
        ))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False  # Fix A

        await adapter._on_group_event(_bot_msg())

        adapter.handle_message.assert_awaited_once()

    async def test_group_allow_bot_senders_mentions_allows_with_mention(self):
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="mentions"))
        adapter._me_short_id = "u_me"
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False  # Fix A

        await adapter._on_group_event(_bot_msg(
            mentions=[{"short_id": "u_me"}],
        ))

        adapter.handle_message.assert_awaited_once()

    async def test_group_allow_bot_senders_mentions_drops_without_mention(self):
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="mentions"))
        adapter._me_short_id = "u_me"
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False  # Fix A

        await adapter._on_group_event(_bot_msg(
            mentions=[{"short_id": "u_someone_else"}],
        ))

        adapter.handle_message.assert_not_awaited()

    async def test_group_allow_bot_senders_all_allows_unconditionally(self):
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False  # Fix A

        await adapter._on_group_event(_bot_msg())

        adapter.handle_message.assert_awaited_once()


class IsMentionOfMeTests(unittest.TestCase):
    """_is_mention_of_me: pure helper shared by policy + group_require_mention."""

    def _adapter(self) -> KimiAdapter:
        adapter = KimiAdapter(_cfg())
        adapter._me_id = "bot-self-id"
        adapter._me_short_id = "u_me"
        return adapter

    def test_is_mention_of_me_short_id_match(self):
        adapter = self._adapter()
        self.assertTrue(adapter._is_mention_of_me({
            "mentions": [{"short_id": "u_me"}],
        }))
        # shortId variant
        self.assertTrue(adapter._is_mention_of_me({
            "mentions": [{"shortId": "u_me"}],
        }))

    def test_is_mention_of_me_id_match(self):
        adapter = self._adapter()
        self.assertTrue(adapter._is_mention_of_me({
            "mentions": [{"id": "bot-self-id"}],
        }))

    def test_is_mention_of_me_mentioned_flag_fallback(self):
        adapter = self._adapter()
        # No mentions array, but `mentioned: true` → accept
        self.assertTrue(adapter._is_mention_of_me({"mentioned": True}))

    def test_is_mention_of_me_no_match(self):
        adapter = self._adapter()
        self.assertFalse(adapter._is_mention_of_me({}))
        self.assertFalse(adapter._is_mention_of_me({
            "mentions": [{"short_id": "u_someone_else"}],
        }))
        # Malformed inputs return False
        self.assertFalse(adapter._is_mention_of_me({"mentions": "not-a-list"}))
        self.assertFalse(adapter._is_mention_of_me({"mentions": ["not-a-dict"]}))


class GroupPolicyLoggingTests(unittest.IsolatedAsyncioTestCase):
    """Policy drops log at INFO, not DEBUG, for operator observability."""

    async def test_policy_drops_log_at_info_level(self):
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="off"))
        adapter.handle_message = AsyncMock()  # type: ignore
        # Fix A: _bot_msg() is summary-only — disable hydration so the
        # cascade doesn't make a real network call before the role drop.
        adapter._hydrate_missing_text = False

        with self.assertLogs("kimi_adapter", level=logging.INFO) as cm:
            await adapter._on_group_event(_bot_msg())

        adapter.handle_message.assert_not_awaited()
        # Exactly one INFO record, and it's the policy-drop message.
        info_records = [r for r in cm.records if r.levelno == logging.INFO]
        self.assertTrue(
            any("group_allow_bot_senders=off" in r.getMessage() for r in info_records),
            f"expected INFO drop log, got: {[r.getMessage() for r in cm.records]}",
        )


class GroupInvalidPolicyTests(unittest.TestCase):
    """Invalid group_allow_bot_senders values log WARNING and fall back to 'off'."""

    def test_invalid_bot_sender_policy_defaults_to_off(self):
        with self.assertLogs("kimi_adapter", level=logging.WARNING) as cm:
            adapter = KimiAdapter(_cfg(group_allow_bot_senders="nonsense"))
        self.assertEqual(adapter._group_allow_bot_senders, "off")
        self.assertTrue(
            any(
                "invalid group_allow_bot_senders" in r.getMessage()
                and r.levelno == logging.WARNING
                for r in cm.records
            ),
            f"expected WARNING about invalid policy, got: {[r.getMessage() for r in cm.records]}",
        )


class GroupRequireMentionSharedHelperTests(unittest.IsolatedAsyncioTestCase):
    """Existing group_require_mention path now delegates to _is_mention_of_me."""

    async def test_group_require_mention_uses_shared_helper(self):
        # USER-role message in a room — mention gate applies even to humans.
        adapter = KimiAdapter(_cfg(group_require_mention=True))
        adapter._me_short_id = "u_me"
        adapter.handle_message = AsyncMock()  # type: ignore
        # Fix A: events below are summary-only (no inline blocks). Disable
        # hydration so the cascade doesn't make a real _fetch_group_message
        # call before the mention filter runs.
        adapter._hydrate_missing_text = False

        # With a proper @-mention of us via the helper's matching logic → dispatched.
        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-req",
                "messageId": "msg-req",
                "status": "STATUS_COMPLETED",
                "role": "USER",
                "senderId": "user-1",
                "senderShortId": "u_user",
                "summary": "hey @u_me",
                "mentions": [{"short_id": "u_me"}],
            },
        })
        adapter.handle_message.assert_awaited_once()

        # Without a mention → dropped.
        adapter.handle_message.reset_mock()
        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-req",
                "messageId": "msg-req-2",
                "status": "STATUS_COMPLETED",
                "role": "USER",
                "senderId": "user-2",
                "senderShortId": "u_user2",
                "summary": "no mention here",
            },
        })
        adapter.handle_message.assert_not_awaited()

        # Bot-role sender who's in group_trusted_senders still has to mention us
        # (require-mention gate runs AFTER role/policy filters). Confirm the
        # shared helper is the authoritative source by flipping to assistant
        # role with a mention + all policy.
        adapter2 = KimiAdapter(_cfg(
            group_require_mention=True,
            group_allow_bot_senders="all",
        ))
        adapter2._me_short_id = "u_me"
        adapter2.handle_message = AsyncMock()  # type: ignore
        adapter2._hydrate_missing_text = False  # Fix A — _bot_msg is summary-only
        await adapter2._on_group_event(_bot_msg(
            mentions=[{"short_id": "u_me"}],
        ))
        adapter2.handle_message.assert_awaited_once()


class MentionGateExemptionTests(unittest.IsolatedAsyncioTestCase):
    """`kimi_free_response_chats` bypasses the mention gate per chat_id.

    Necessary because Kimi delivers both 1:1 DMs and group rooms as
    `room:<uuid>` with no wire-level distinction. A global
    `group_require_mention=True` would otherwise swallow DM traffic.
    """

    def test_default_exempt_list_is_empty(self):
        adapter = KimiAdapter(_cfg(group_require_mention=True))
        self.assertEqual(adapter._group_require_mention_exempt_rooms, frozenset())

    def test_exempt_list_loaded_from_config(self):
        # Mixed prefixed + raw entries: both normalise to the raw UUID form
        # so they can compare against `chatId` from the subscribe stream.
        # Regression for v2.1.2 deploy bug: README documents the prefixed
        # form (matching KIMI_HOME_CHANNEL convention), but `_on_group_event`
        # extracts `chatId` from the wire envelope as a raw UUID — the v2.1.2
        # gate silently never fired because the set held prefixed strings
        # that never equalled the wire's raw form.
        adapter = KimiAdapter(_cfg(
            group_require_mention=True,
            kimi_free_response_chats=["room:aaa", "bbb"],
        ))
        self.assertEqual(
            adapter._group_require_mention_exempt_rooms,
            frozenset({"aaa", "bbb"}),
        )

    def test_exempt_entries_strip_room_prefix(self):
        # All-prefixed config (the README-documented form) must still match
        # raw-UUID chat_ids from the wire after normalisation.
        adapter = KimiAdapter(_cfg(
            group_require_mention=True,
            kimi_free_response_chats=[
                "room:19e31a29-4722-8804-8000-094a7731741b",
            ],
        ))
        self.assertEqual(
            adapter._group_require_mention_exempt_rooms,
            frozenset({"19e31a29-4722-8804-8000-094a7731741b"}),
        )

    def test_alias_key_recognized(self):
        # group_require_mention_exempt_rooms is accepted as an alias for the
        # documented kimi_free_response_chats key. Useful for users coming
        # from the explicit name. Normalisation applies to the alias too.
        adapter = KimiAdapter(_cfg(
            group_require_mention=True,
            group_require_mention_exempt_rooms=["room:zzz"],
        ))
        self.assertEqual(
            adapter._group_require_mention_exempt_rooms,
            frozenset({"zzz"}),
        )

    def test_non_string_entries_filtered(self):
        # Be tolerant of YAML-side garbage (None, ints, empty strings).
        # Normalisation runs after the type filter; survivors land prefix-free.
        adapter = KimiAdapter(_cfg(
            group_require_mention=True,
            kimi_free_response_chats=["room:ok", "", None, 42, "also-ok"],
        ))
        self.assertEqual(
            adapter._group_require_mention_exempt_rooms,
            frozenset({"ok", "also-ok"}),
        )

    async def test_exempt_room_bypasses_mention_gate(self):
        # Message in an exempt room, no @-mention → still dispatched.
        # This is the DM-as-room case in production.
        adapter = KimiAdapter(_cfg(
            group_require_mention=True,
            kimi_free_response_chats=["chat-dm"],
        ))
        adapter._me_short_id = "u_me"
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False  # Fix A

        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-dm",
                "messageId": "msg-dm-1",
                "status": "STATUS_COMPLETED",
                "role": "USER",
                "senderId": "user-1",
                "senderShortId": "u_user",
                "summary": "no mention here, just talking in DM",
            },
        })
        adapter.handle_message.assert_awaited_once()

    async def test_non_exempt_room_still_gated(self):
        # Same adapter as above, different room, no mention → dropped.
        # Confirms the exempt list is per-chat_id, not a global disable.
        adapter = KimiAdapter(_cfg(
            group_require_mention=True,
            kimi_free_response_chats=["chat-dm"],
        ))
        adapter._me_short_id = "u_me"
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False

        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-other-group",
                "messageId": "msg-grp-1",
                "status": "STATUS_COMPLETED",
                "role": "USER",
                "senderId": "user-1",
                "senderShortId": "u_user",
                "summary": "no mention here either",
            },
        })
        adapter.handle_message.assert_not_awaited()

    async def test_exempt_room_with_mention_still_works(self):
        # Sanity: mention gate ALSO bypassed if the message has a mention —
        # exempt list is an additive bypass, not a replacement of the mention path.
        adapter = KimiAdapter(_cfg(
            group_require_mention=True,
            kimi_free_response_chats=["chat-dm"],
        ))
        adapter._me_short_id = "u_me"
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False

        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-dm",
                "messageId": "msg-dm-2",
                "status": "STATUS_COMPLETED",
                "role": "USER",
                "senderId": "user-1",
                "senderShortId": "u_user",
                "summary": "hey @u_me",
                "mentions": [{"short_id": "u_me"}],
            },
        })
        adapter.handle_message.assert_awaited_once()


class DmAutodetectTests(unittest.IsolatedAsyncioTestCase):
    """v2.2.0 room-distinguishability detector + mention-gate integration.

    Kimi delivers 1:1 DMs as ``room:<uuid>`` indistinguishable from groups
    at the envelope layer the adapter consumes.  Before v2.2.0 the only
    workaround was to add the DM's room UUID to ``kimi_free_response_chats``
    manually.  The ``kimi_dm_autodetect`` flag (default OFF) makes the
    adapter call ``list_group_members`` on rooms about to be dropped by
    the mention gate; rooms with exactly 2 members (bot + 1 user) bypass
    the gate without operator config.

    Default OFF preserves v2.1.x behaviour exactly; the test grouping
    below mirrors that guarantee.

    The detector is the LAST bypass chance — only consulted when the
    message is otherwise about to be dropped.  Messages that already
    qualify for dispatch (no gate, or mentioned, or in the explicit
    exempt list) never incur the detector RPC.

    See ``.review/b1-room-distinguishability-spike.md`` for the design
    document covering cost analysis, race-window handling, and the
    dual-use-case justification for keeping ``kimi_free_response_chats``.
    """

    def _build_msg(self, chat_id: str, with_mention: bool = False) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "chatMessage": {
                "chatId": chat_id,
                "messageId": f"msg-for-{chat_id}",
                "status": "STATUS_COMPLETED",
                "role": "USER",
                "senderId": "user-1",
                "senderShortId": "u_user",
                "summary": "hello bot",
            },
        }
        if with_mention:
            msg["chatMessage"]["mentions"] = [{"short_id": "u_me"}]
        return msg

    # ── _is_dm_room helper ────────────────────────────────────────────────

    async def test_helper_returns_true_for_2_member_room(self):
        adapter = KimiAdapter(_cfg())
        adapter.list_group_members = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"id": "bot"}, {"id": "user-1"}],
        )
        self.assertTrue(await adapter._is_dm_room("room-uuid-1"))

    async def test_helper_returns_false_for_3plus_member_room(self):
        adapter = KimiAdapter(_cfg())
        adapter.list_group_members = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"id": "bot"}, {"id": "user-1"}, {"id": "user-2"}],
        )
        self.assertFalse(await adapter._is_dm_room("room-uuid-2"))

    async def test_helper_returns_none_for_rpc_failure(self):
        from kimi_adapter import KimiTransientError
        adapter = KimiAdapter(_cfg())
        adapter.list_group_members = AsyncMock(  # type: ignore[method-assign]
            side_effect=KimiTransientError("simulated network drop"),
        )
        with self.assertLogs("kimi_adapter", level="WARNING") as cm:
            result = await adapter._is_dm_room("room-uuid-3")
        self.assertIsNone(result)
        warning_text = "\n".join(cm.output)
        self.assertIn("room-uuid-3", warning_text)
        self.assertIn("DM-autodetect classification failed", warning_text)

    async def test_helper_returns_none_for_timeout(self):
        adapter = KimiAdapter(_cfg())
        adapter.list_group_members = AsyncMock(  # type: ignore[method-assign]
            side_effect=asyncio.TimeoutError(),
        )
        with self.assertLogs("kimi_adapter", level="WARNING"):
            self.assertIsNone(await adapter._is_dm_room("room-uuid-4"))

    async def test_helper_uses_cache_within_ttl(self):
        # Second call within the 5-minute TTL must NOT issue a fresh RPC.
        adapter = KimiAdapter(_cfg())
        adapter.list_group_members = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"id": "bot"}, {"id": "user-1"}],
        )
        self.assertTrue(await adapter._is_dm_room("room-uuid-5"))
        self.assertEqual(adapter.list_group_members.await_count, 1)
        self.assertTrue(await adapter._is_dm_room("room-uuid-5"))
        self.assertEqual(adapter.list_group_members.await_count, 1, "Cache hit must not re-RPC")

    async def test_helper_returns_none_for_degenerate_membership(self):
        # Defensive: 0 or 1 member shouldn't happen for a room that sent us
        # a message.  Helper logs a warning and returns None so callers
        # fall back to existing behaviour rather than over-bypassing.
        adapter = KimiAdapter(_cfg())
        adapter.list_group_members = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"id": "lonely"}],
        )
        with self.assertLogs("kimi_adapter", level="WARNING"):
            self.assertIsNone(await adapter._is_dm_room("room-uuid-6"))

    # ── Mention-gate integration ──────────────────────────────────────────

    async def _make_gated_adapter(self, *, dm_autodetect: bool, members: List[Dict[str, Any]]) -> KimiAdapter:
        adapter = KimiAdapter(_cfg(
            group_require_mention=True,
            kimi_dm_autodetect=dm_autodetect,
        ))
        adapter._me_short_id = "u_me"
        adapter._me_id = "bot-self-id"
        adapter.handle_message = AsyncMock()  # type: ignore[method-assign]
        adapter._hydrate_missing_text = False
        adapter.list_group_members = AsyncMock(return_value=members)  # type: ignore[method-assign]
        return adapter

    async def test_gate_default_off_unchanged_v21x_behaviour(self):
        # With kimi_dm_autodetect=False (the v2.2.0 default), a 2-member
        # room without @-mention is still dropped — proving v2.1.x
        # semantics are preserved exactly when the flag is off.
        adapter = await self._make_gated_adapter(
            dm_autodetect=False,
            members=[{"id": "bot"}, {"id": "user-1"}],
        )
        await adapter._on_group_event(self._build_msg("would-be-dm-room"))
        adapter.handle_message.assert_not_awaited()
        # Detector must NOT have been called.
        adapter.list_group_members.assert_not_awaited()

    async def test_gate_with_flag_on_bypasses_for_detected_dm(self):
        adapter = await self._make_gated_adapter(
            dm_autodetect=True,
            members=[{"id": "bot"}, {"id": "user-1"}],  # 2 members → DM
        )
        await adapter._on_group_event(self._build_msg("dm-room-uuid"))
        adapter.handle_message.assert_awaited_once()
        adapter.list_group_members.assert_awaited_once_with("dm-room-uuid")

    async def test_gate_with_flag_on_still_drops_for_detected_group(self):
        adapter = await self._make_gated_adapter(
            dm_autodetect=True,
            members=[{"id": "bot"}, {"id": "user-1"}, {"id": "user-2"}],
        )
        with self.assertLogs("kimi_adapter", level="INFO") as cm:
            await adapter._on_group_event(self._build_msg("real-group-uuid"))
        adapter.handle_message.assert_not_awaited()
        # The augmented drop log must carry the auto-detect annotation so
        # operators can correlate drops with classification outcomes.
        info_records = [r for r in cm.records if r.levelno == logging.INFO]
        self.assertTrue(
            any("auto-detect=group" in r.getMessage() for r in info_records),
            f"expected auto-detect=group annotation, got: {[r.getMessage() for r in cm.records]}",
        )

    async def test_gate_with_flag_on_falls_back_when_classification_fails(self):
        # Classifier returns None → log says auto-detect=unknown, message dropped.
        # Fail-closed behaviour: matches v2.1.x when no explicit exempt exists.
        from kimi_adapter import KimiTransientError
        adapter = await self._make_gated_adapter(
            dm_autodetect=True,
            members=[],  # unused
        )
        adapter.list_group_members = AsyncMock(  # type: ignore[method-assign]
            side_effect=KimiTransientError("rpc died"),
        )
        with self.assertLogs("kimi_adapter", level="INFO") as cm:
            await adapter._on_group_event(self._build_msg("uncertain-room"))
        adapter.handle_message.assert_not_awaited()
        info_records = [r for r in cm.records if r.levelno == logging.INFO]
        self.assertTrue(
            any("auto-detect=unknown" in r.getMessage() for r in info_records),
            f"expected auto-detect=unknown annotation, got: {[r.getMessage() for r in cm.records]}",
        )

    async def test_gate_with_flag_on_explicit_exempt_still_takes_precedence(self):
        # An exempt-listed room must skip the detector entirely (fast path).
        # Proves the detector is the LAST bypass, not a replacement.
        adapter = KimiAdapter(_cfg(
            group_require_mention=True,
            kimi_dm_autodetect=True,
            kimi_free_response_chats=["explicit-policy-group"],
        ))
        adapter._me_short_id = "u_me"
        adapter.handle_message = AsyncMock()  # type: ignore[method-assign]
        adapter._hydrate_missing_text = False
        adapter.list_group_members = AsyncMock()  # type: ignore[method-assign]
        await adapter._on_group_event(self._build_msg("explicit-policy-group"))
        adapter.handle_message.assert_awaited_once()
        # Detector was NOT consulted — the room is in the explicit exempt list.
        adapter.list_group_members.assert_not_awaited()

    async def test_gate_with_flag_on_mentioned_message_skips_detector(self):
        # Sanity: mention path short-circuits before the detector.
        # No RPC, no auto-detect annotation.
        adapter = await self._make_gated_adapter(
            dm_autodetect=True,
            members=[{"id": "bot"}, {"id": "user-1"}],
        )
        await adapter._on_group_event(self._build_msg("any-room", with_mention=True))
        adapter.handle_message.assert_awaited_once()
        adapter.list_group_members.assert_not_awaited()


class ThreadRoutingTests(unittest.IsolatedAsyncioTestCase):
    """Inbound/outbound thread routing preserves thread identity.

    Kimi Claw v0.25.0's ``SendMessageRequest`` has no thread field on the
    wire. Inbound threaded messages are still tagged so gateway sessions
    stay isolated per-thread; outbound sends to a threaded chat_id WARN
    on first occurrence instead of silently collapsing.
    """

    async def test_inbound_thread_chat_id_preserved(self):
        adapter = KimiAdapter(_cfg())
        adapter.handle_message = AsyncMock()  # type: ignore
        # Fix A: summary-only event — disable hydration so routing assertions
        # don't depend on a real network call.
        adapter._hydrate_missing_text = False

        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-t1",
                "messageId": "msg-t1",
                "status": "STATUS_COMPLETED",
                "senderId": "user-1",
                "senderShortId": "u1",
                "threadId": "thread-abc",
                "summary": "hello in a thread",
            },
        })

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        self.assertEqual(event.source.chat_id, "room:chat-t1/thread-abc")
        # SessionSource also carries the raw thread_id for routing helpers.
        self.assertEqual(event.source.thread_id, "thread-abc")

    async def test_inbound_no_thread_chat_id_room_only(self):
        adapter = KimiAdapter(_cfg())
        adapter.handle_message = AsyncMock()  # type: ignore
        # Fix A: summary-only event — disable hydration so routing assertions
        # don't depend on a real network call.
        adapter._hydrate_missing_text = False

        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-t2",
                "messageId": "msg-t2",
                "status": "STATUS_COMPLETED",
                "senderId": "user-2",
                "senderShortId": "u2",
                "summary": "no thread",
            },
        })

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        # No slash — plain room id only.
        self.assertEqual(event.source.chat_id, "room:chat-t2")
        self.assertNotIn("/", event.source.chat_id)

    async def test_inbound_hydrated_thread_id_preserved(self):
        """Thread id on the hydrated wrapper is preserved when the Subscribe
        event itself was a lightweight stub."""
        adapter = KimiAdapter(_cfg())
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._fetch_group_message = AsyncMock(return_value={
            "id": "msg-t3",
            "senderId": "user-3",
            "senderShortId": "u3",
            "threadId": "thread-from-hydration",
            "blocks": [
                {"content": {"case": "text", "value": {"content": "hydrated text"}}},
            ],
        })  # type: ignore

        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-t3",
                "messageId": "msg-t3",
                "status": "STATUS_COMPLETED",
            },
        })

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        self.assertEqual(event.source.chat_id, "room:chat-t3/thread-from-hydration")

    async def test_outbound_thread_suffix_parsed_not_dropped(self):
        """Sending to ``room:<uuid>/<tid>`` warns about thread collapse on
        first occurrence instead of silently targeting the plain room."""
        adapter = KimiAdapter(_cfg())
        adapter._rpc_unary = AsyncMock(return_value={"messageId": "sent-t1"})  # type: ignore

        with self.assertLogs("kimi_adapter", level=logging.WARNING) as cm:
            result = await adapter.send("room:chat-t4/thread-xyz", "hello thread")

        self.assertTrue(result.success)
        # WARNING fired exactly once, references the thread id and the room.
        self.assertTrue(
            any(
                r.levelno == logging.WARNING
                and "thread_id='thread-xyz'" in r.getMessage()
                and "chat-t4" in r.getMessage()
                for r in cm.records
            ),
            f"expected WARNING about thread drop, got: {[r.getMessage() for r in cm.records]}",
        )
        # Payload still hits Kimi at the room level.
        _method, body = adapter._rpc_unary.await_args.args
        self.assertEqual(body["chatId"], "chat-t4")

    async def test_outbound_thread_warning_is_one_shot(self):
        """Second threaded send to the same adapter instance does not re-warn."""
        adapter = KimiAdapter(_cfg())
        adapter._rpc_unary = AsyncMock(return_value={"messageId": "sent-t2"})  # type: ignore

        # Prime: first send consumes the warning.
        with self.assertLogs("kimi_adapter", level=logging.WARNING):
            await adapter.send("room:chat-t5/tid1", "first")

        # Second send: attach a record-capturing handler directly instead of
        # assertLogs (which fails when zero records are emitted).
        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = captured.append  # type: ignore[assignment]
        kimi_logger = logging.getLogger("kimi_adapter")
        kimi_logger.addHandler(handler)
        try:
            await adapter.send("room:chat-t5/tid2", "second")
        finally:
            kimi_logger.removeHandler(handler)
        self.assertFalse(
            any(
                r.levelno >= logging.WARNING
                and "thread_id=" in r.getMessage()
                for r in captured
            ),
            f"thread drop WARNING should be one-shot, got: {[r.getMessage() for r in captured]}",
        )


class OutboundMentionRenderingTests(unittest.IsolatedAsyncioTestCase):
    """Outbound ``metadata['mentions']`` is no longer silently dropped.

    Kimi Claw v0.25.0 has no confirmed mention-block wire shape, so the
    adapter emits a WARNING and falls through to plain text. When the
    surface check confirms a variant, this path can serialize instead.
    """

    async def test_outbound_mentions_metadata_serialized(self):
        adapter = KimiAdapter(_cfg())
        adapter._rpc_unary = AsyncMock(return_value={"messageId": "sent-m1"})  # type: ignore

        with self.assertLogs("kimi_adapter", level=logging.WARNING) as cm:
            result = await adapter._send_group(
                "chat-m1",
                "hey @u_bob",
                reply_to=None,
                thread_id=None,
                metadata={"mentions": ["u_bob"]},
            )

        self.assertTrue(result.success)
        self.assertTrue(
            any(
                r.levelno == logging.WARNING
                and "metadata.mentions" in r.getMessage()
                and "u_bob" in r.getMessage()
                for r in cm.records
            ),
            f"expected WARNING about mention fall-through, got: {[r.getMessage() for r in cm.records]}",
        )
        # Plain text block still goes out — existing send contract preserved.
        _method, body = adapter._rpc_unary.await_args.args
        self.assertEqual(body["chatId"], "chat-m1")
        self.assertEqual(len(body["blocks"]), 1)
        self.assertEqual(body["blocks"][0]["text"]["content"], "hey @u_bob")

    async def test_outbound_mentions_empty_metadata_plain_text_only(self):
        """No mentions → no WARNING, plain text path unchanged."""
        adapter = KimiAdapter(_cfg())
        adapter._rpc_unary = AsyncMock(return_value={"messageId": "sent-m2"})  # type: ignore

        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = captured.append  # type: ignore[assignment]
        kimi_logger = logging.getLogger("kimi_adapter")
        kimi_logger.addHandler(handler)
        try:
            result = await adapter._send_group(
                "chat-m2",
                "hello",
                reply_to=None,
                thread_id=None,
                metadata={},
            )
        finally:
            kimi_logger.removeHandler(handler)

        self.assertTrue(result.success)
        self.assertFalse(
            any(
                "metadata.mentions" in r.getMessage()
                for r in captured
            ),
            f"no mentions means no mention warning, got: {[r.getMessage() for r in captured]}",
        )
        # Also covers metadata=None via empty-dict default in the caller.
        _method, body = adapter._rpc_unary.await_args.args
        self.assertEqual(body["blocks"][0]["text"]["content"], "hello")

    async def test_outbound_mentions_warning_is_one_shot(self):
        adapter = KimiAdapter(_cfg())
        adapter._rpc_unary = AsyncMock(return_value={"messageId": "sent-m3"})  # type: ignore

        with self.assertLogs("kimi_adapter", level=logging.WARNING):
            await adapter._send_group(
                "chat-m3", "first", reply_to=None, thread_id=None,
                metadata={"mentions": ["u_one"]},
            )

        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = captured.append  # type: ignore[assignment]
        kimi_logger = logging.getLogger("kimi_adapter")
        kimi_logger.addHandler(handler)
        try:
            await adapter._send_group(
                "chat-m3", "second", reply_to=None, thread_id=None,
                metadata={"mentions": ["u_two"]},
            )
        finally:
            kimi_logger.removeHandler(handler)
        self.assertFalse(
            any(
                r.levelno >= logging.WARNING
                and "metadata.mentions" in r.getMessage()
                for r in captured
            ),
            f"mention drop WARNING should be one-shot, got: {[r.getMessage() for r in captured]}",
        )


class ConfigIntegrationTests(unittest.TestCase):
    """Plugin-side config bridge: env vars, YAML translation, validate_config.

    Replaces the older Fork-only tests that exercised in-fork
    ``_apply_env_overrides`` and the hardcoded ``Platform.KIMI`` auth env-maps
    in ``gateway/run.py``.  Both of those layers are now plugin-owned via the
    upstream ``PlatformEntry`` registry (see :func:`kimi_adapter.register`).
    """

    def test_platform_kimiclaw_resolves_via_registry(self):
        """After ``setUpModule`` registration, ``Platform('kimiclaw')`` yields a stable pseudo-member."""
        p = Platform("kimiclaw")
        self.assertEqual(p.value, "kimiclaw")
        self.assertIs(p, Platform("kimiclaw"))  # identity-stable

    def test_validate_config_with_env_token(self):
        from kimi_adapter import validate_config
        with patch.dict(os.environ, {"KIMI_BOT_TOKEN": "km_b_prod_TEST"}, clear=False):
            self.assertTrue(validate_config(_cfg()))

    def test_validate_config_with_extra_token(self):
        from kimi_adapter import validate_config
        env_no_token = {k: v for k, v in os.environ.items() if k != "KIMI_BOT_TOKEN"}
        with patch.dict(os.environ, env_no_token, clear=True):
            cfg = PlatformConfig(enabled=True, extra={"bot_token": "km_b_prod_TEST"})
            self.assertTrue(validate_config(cfg))

    def test_validate_config_with_config_token(self):
        """``PlatformConfig.token`` is the third token source the adapter
        ``__init__`` consults; ``validate_config`` must agree."""
        from kimi_adapter import validate_config
        env_no_token = {k: v for k, v in os.environ.items() if k != "KIMI_BOT_TOKEN"}
        with patch.dict(os.environ, env_no_token, clear=True):
            cfg = PlatformConfig(enabled=True, token="km_b_prod_TEST")
            self.assertTrue(validate_config(cfg))

    def test_validate_config_missing_token(self):
        from kimi_adapter import validate_config
        env_no_kimi = {k: v for k, v in os.environ.items()
                       if not k.startswith("KIMI_")}
        with patch.dict(os.environ, env_no_kimi, clear=True):
            cfg = PlatformConfig(enabled=True, extra={})
            self.assertFalse(validate_config(cfg))

    def test_validate_config_rejects_unresolved_env_template(self):
        """``${KIMI_BOT_TOKEN}`` with the env var unset must resolve to empty
        and report not-configured — not a truthy literal that 401s at connect.
        """
        from kimi_adapter import validate_config
        env_no_kimi = {k: v for k, v in os.environ.items()
                       if not k.startswith("KIMI_")}
        with patch.dict(os.environ, env_no_kimi, clear=True):
            cfg_token = PlatformConfig(enabled=True, token="${KIMI_BOT_TOKEN}")
            self.assertFalse(validate_config(cfg_token))
            cfg_extra = PlatformConfig(
                enabled=True, extra={"bot_token": "${KIMI_BOT_TOKEN}"}
            )
            self.assertFalse(validate_config(cfg_extra))

    def test_validate_config_accepts_resolved_env_template(self):
        """A ``${VAR}`` template that resolves to a real value is accepted.

        Uses a non-``KIMI_BOT_TOKEN`` template var so the env short-circuit
        at the top of ``validate_config`` doesn't mask the template path.
        """
        from kimi_adapter import validate_config
        env_no_kimi = {k: v for k, v in os.environ.items()
                       if not k.startswith("KIMI_")}
        env_with = {**env_no_kimi, "KIMI_ALT_TOKEN": "km_b_prod_RESOLVED"}
        with patch.dict(os.environ, env_with, clear=True):
            cfg_token = PlatformConfig(enabled=True, token="${KIMI_ALT_TOKEN}")
            self.assertTrue(validate_config(cfg_token))

    def test_env_enablement_returns_token_dict(self):
        from kimi_adapter import _env_enablement
        with patch.dict(os.environ, {"KIMI_BOT_TOKEN": "km_b_prod_TEST"}, clear=False):
            result = _env_enablement()
        self.assertIsNotNone(result)
        self.assertEqual(result["bot_token"], "km_b_prod_TEST")

    def test_env_enablement_returns_none_without_token(self):
        from kimi_adapter import _env_enablement
        env_no_token = {k: v for k, v in os.environ.items() if k != "KIMI_BOT_TOKEN"}
        with patch.dict(os.environ, env_no_token, clear=True):
            result = _env_enablement()
        self.assertIsNone(result)

    def test_env_enablement_includes_home_channel(self):
        from kimi_adapter import _env_enablement
        with patch.dict(os.environ, {
            "KIMI_BOT_TOKEN": "tok",
            "KIMI_HOME_CHANNEL": "im:kimi:main",
            "KIMI_HOME_CHANNEL_NAME": "Kimi Home",
        }, clear=False):
            result = _env_enablement()
        self.assertEqual(
            result["home_channel"],
            {"chat_id": "im:kimi:main", "name": "Kimi Home"},
        )

    def test_apply_yaml_config_populates_env(self):
        from kimi_adapter import _apply_yaml_config
        env_no_kimi = {k: v for k, v in os.environ.items()
                       if not k.startswith("KIMI_")}
        with patch.dict(os.environ, env_no_kimi, clear=True):
            _apply_yaml_config({}, {
                "bot_token": "km_b_prod_FROM_YAML",
                "home_channel": "room:abc",
                "allowed_users": ["u1", "u2"],
            })
            self.assertEqual(os.environ.get("KIMI_BOT_TOKEN"), "km_b_prod_FROM_YAML")
            self.assertEqual(os.environ.get("KIMI_HOME_CHANNEL"), "room:abc")
            self.assertEqual(os.environ.get("KIMI_ALLOWED_USERS"), "u1,u2")

    def test_apply_yaml_config_respects_env_precedence(self):
        from kimi_adapter import _apply_yaml_config
        with patch.dict(os.environ, {"KIMI_BOT_TOKEN": "km_b_prod_ENV_WINS"}, clear=False):
            _apply_yaml_config({}, {"bot_token": "km_b_prod_YAML_LOSES"})
            self.assertEqual(os.environ["KIMI_BOT_TOKEN"], "km_b_prod_ENV_WINS")

    def test_apply_yaml_config_ignores_non_dict(self):
        from kimi_adapter import _apply_yaml_config
        # platform_cfg can be None or a non-dict when the YAML key is absent
        self.assertIsNone(_apply_yaml_config({}, None))
        self.assertIsNone(_apply_yaml_config({}, "not a dict"))

    def test_load_gateway_config_top_level_kimiclaw_yaml_bridge_seeds_env_and_extra(self):
        from gateway.config import Platform, load_gateway_config
        from gateway.platform_registry import PlatformEntry, platform_registry
        from kimi_adapter import _apply_yaml_config, _env_enablement

        previous = platform_registry.get("kimiclaw")
        platform_registry.unregister("kimiclaw")
        platform_registry.register(PlatformEntry(
            name="kimiclaw",
            label="KimiClaw",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: bool(os.getenv("KIMI_BOT_TOKEN")),
            source="plugin",
            apply_yaml_config_fn=_apply_yaml_config,
            env_enablement_fn=_env_enablement,
        ))
        old_home = os.environ.get("HERMES_HOME")
        env_no_kimi = {k: v for k, v in os.environ.items() if not k.startswith("KIMI_")}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp) / ".hermes"
                home.mkdir()
                (home / "config.yaml").write_text(
                    "kimiclaw:\n"
                    "  bot_token: km_b_prod_FROM_TOP_LEVEL\n"
                    "  home_channel: room:home\n",
                    encoding="utf-8",
                )
                with patch.dict(os.environ, env_no_kimi, clear=True):
                    os.environ["HERMES_HOME"] = str(home)
                    cfg = load_gateway_config()

                    self.assertEqual(
                        os.environ.get("KIMI_BOT_TOKEN"),
                        "km_b_prod_FROM_TOP_LEVEL",
                    )
                    platform = Platform("kimiclaw")
                    self.assertIn(platform, cfg.platforms)
                    self.assertEqual(
                        cfg.platforms[platform].extra.get("bot_token"),
                        "km_b_prod_FROM_TOP_LEVEL",
                    )
                    self.assertIsNotNone(cfg.platforms[platform].home_channel)
                    self.assertEqual(
                        cfg.platforms[platform].home_channel.chat_id,
                        "room:home",
                    )
        finally:
            platform_registry.unregister("kimiclaw")
            if previous is not None:
                platform_registry.register(previous)
            if old_home is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = old_home

    def test_platforms_extra_template_resolves_in_upstream_config_and_runtime_paths(self):
        from gateway.config import Platform, load_gateway_config
        from gateway.platform_registry import PlatformEntry, platform_registry
        from kimi_adapter import _apply_yaml_config, _env_enablement

        previous = platform_registry.get("kimiclaw")
        platform_registry.unregister("kimiclaw")
        platform_registry.register(PlatformEntry(
            name="kimiclaw",
            label="KimiClaw",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            apply_yaml_config_fn=_apply_yaml_config,
            env_enablement_fn=_env_enablement,
        ))
        old_home = os.environ.get("HERMES_HOME")
        sentinel = "km_b_prod_RAW_EXTRA_" + uuid.uuid4().hex
        env_no_kimi = {k: v for k, v in os.environ.items() if not k.startswith("KIMI_")}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp) / ".hermes"
                home.mkdir()
                (home / "config.yaml").write_text(
                    "platforms:\n"
                    "  kimiclaw:\n"
                    "    enabled: true\n"
                    "    extra:\n"
                    "      bot_token: ${KIMI_BOT_TOKEN}\n",
                    encoding="utf-8",
                )
                with patch.dict(os.environ, env_no_kimi, clear=True):
                    os.environ["HERMES_HOME"] = str(home)
                    os.environ["KIMI_BOT_TOKEN"] = sentinel
                    cfg = load_gateway_config()

                    platform = Platform("kimiclaw")
                    pconfig = cfg.platforms[platform]
                    self.assertEqual(pconfig.extra.get("bot_token"), sentinel)
                    adapter = KimiAdapter(pconfig)
                    self.assertEqual(adapter._bot_token, sentinel)
                    with patch(
                        "kimi_adapter.send_kimi_message",
                        new=AsyncMock(return_value=SendResult(success=True)),
                    ) as mock_send:
                        result = asyncio.run(
                            _standalone_send(pconfig, "room:abc", "hello")
                        )

                    self.assertEqual(result, {"success": True})
                    forwarded_cfg = mock_send.await_args.args[0]
                    self.assertEqual(
                        _resolve_env_template(forwarded_cfg.extra["bot_token"]),
                        sentinel,
                    )
        finally:
            platform_registry.unregister("kimiclaw")
            if previous is not None:
                platform_registry.register(previous)
            if old_home is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = old_home


class UserIdentityExtractionTests(unittest.TestCase):
    """_extract_user_identity probes several plausible Kimi wire shapes."""

    def test_flat_userid(self):
        uid, name = _extract_user_identity({"userId": "u-123"})
        self.assertEqual(uid, "u-123")
        self.assertIsNone(name)

    def test_flat_user_id_snake(self):
        uid, _ = _extract_user_identity({"user_id": "u-456"})
        self.assertEqual(uid, "u-456")

    def test_nested_sender(self):
        uid, name = _extract_user_identity(
            {"sender": {"id": "u-789", "name": "Alice"}}
        )
        self.assertEqual(uid, "u-789")
        self.assertEqual(name, "Alice")

    def test_nested_user_with_display_name(self):
        uid, name = _extract_user_identity(
            {"user": {"userId": "u-abc", "display_name": "Bob"}}
        )
        self.assertEqual(uid, "u-abc")
        self.assertEqual(name, "Bob")

    def test_no_identity(self):
        uid, name = _extract_user_identity({"sessionId": "im:kimi:main"})
        self.assertIsNone(uid)
        self.assertIsNone(name)

    def test_non_dict_input(self):
        uid, name = _extract_user_identity(None)
        self.assertIsNone(uid)
        self.assertIsNone(name)
        uid, name = _extract_user_identity("string")
        self.assertIsNone(uid)
        self.assertIsNone(name)


class SenderShortIdPrefixTests(unittest.TestCase):
    """kimi-claw's [sender_short_id: X] text prefix (group-routed-over-ACP)."""

    def test_extracts_short_id_from_prefix_line(self):
        text = (
            "Message From Kimi Group Chat Room:\n"
            "[sender_short_id: u_abc123]\n"
            "hello there"
        )
        self.assertEqual(_extract_short_id_from_text(text), "u_abc123")

    def test_extracts_short_id_anywhere_in_text(self):
        # The kimi-claw injector places it after the group prefix; we accept
        # any line-start match so we're robust to prompt-text transforms.
        text = "prelude\n[sender_short_id: u_xyz]\nactual content"
        self.assertEqual(_extract_short_id_from_text(text), "u_xyz")

    def test_no_prefix_returns_none(self):
        self.assertIsNone(_extract_short_id_from_text("plain message"))
        self.assertIsNone(_extract_short_id_from_text(""))
        self.assertIsNone(_extract_short_id_from_text(None))  # type: ignore

    def test_empty_short_id_returns_none(self):
        # Malformed: empty content between brackets.
        self.assertIsNone(_extract_short_id_from_text("[sender_short_id: ]"))

    def test_strips_surrounding_whitespace(self):
        text = "[sender_short_id:   u_padded   ]"
        self.assertEqual(_extract_short_id_from_text(text), "u_padded")


class EnvelopeLengthCapTests(unittest.IsolatedAsyncioTestCase):
    """Connect envelope parser refuses oversize frames (I4 DoS guard)."""

    async def test_envelope_length_cap_rejects_oversize(self):
        adapter = KimiAdapter(_cfg())
        oversize = _WS_MAX_FRAME_SIZE + 1
        header = bytes([0x00]) + struct.pack(">I", oversize)
        reader = MagicMock()
        reader.readexactly = AsyncMock(return_value=header)
        with self.assertRaises(KimiProtocolError) as ctx:
            async for _ in adapter._connect_envelope_parser(reader):
                self.fail("should not yield")
        self.assertIn("exceeds max frame size", str(ctx.exception))

    async def test_envelope_length_at_cap_is_allowed(self):
        adapter = KimiAdapter(_cfg())
        body = b'{"ping":{}}'
        header = bytes([0x00]) + struct.pack(">I", len(body))
        reads = [header, body, b""]
        reader = MagicMock()

        async def _readexactly(n):
            if not reads:
                raise asyncio.IncompleteReadError(b"", n)
            chunk = reads.pop(0)
            if len(chunk) != n:
                # Simulate end-of-stream for the empty tail read
                raise asyncio.IncompleteReadError(chunk, n)
            return chunk

        reader.readexactly = _readexactly
        yielded = []
        with self.assertRaises(Exception):
            async for msg in adapter._connect_envelope_parser(reader):
                yielded.append(msg)
        self.assertEqual(yielded, [{"ping": {}}])


class DMInflightQueueTests(unittest.IsolatedAsyncioTestCase):
    """Overlapping DM prompts get FIFO end_turn responses (I2)."""

    async def test_send_dm_pops_oldest_inflight(self):
        adapter = KimiAdapter(_cfg())
        # Simulate two in-flight prompts on the same kimi_sid
        sid = "im:kimi:main"
        adapter._dm_inflight[sid] = deque([
            _DMInflight(kimi_sid=sid, req_id=101),
            _DMInflight(kimi_sid=sid, req_id=102),
        ])

        respond_mock = AsyncMock()
        adapter._dm_respond = respond_mock  # type: ignore
        adapter._dm_emit_chunk = AsyncMock()  # type: ignore
        adapter._ws = MagicMock()  # non-None sentinel

        # First reply should pop req_id=101
        await adapter._send_dm(sid, "reply one", reply_to=None, metadata={})
        self.assertEqual(respond_mock.await_args_list[0].args[0], 101)

        # Second reply should pop req_id=102 and leave the queue empty
        await adapter._send_dm(sid, "reply two", reply_to=None, metadata={})
        self.assertEqual(respond_mock.await_args_list[1].args[0], 102)
        self.assertNotIn(sid, adapter._dm_inflight)

    async def test_send_dm_no_inflight_is_harmless(self):
        adapter = KimiAdapter(_cfg())
        adapter._dm_respond = AsyncMock()  # type: ignore
        adapter._dm_emit_chunk = AsyncMock()  # type: ignore
        adapter._ws = MagicMock()
        result = await adapter._send_dm(
            "im:kimi:main", "reply", reply_to=None, metadata={}
        )
        self.assertTrue(result.success)

    async def test_session_cancel_cancels_active_processing_and_clears_inflight(self):
        adapter = KimiAdapter(_cfg())
        sid = "im:kimi:main"
        adapter._dm_inflight[sid] = deque([
            _DMInflight(kimi_sid=sid, req_id=101),
        ])
        adapter.cancel_session_processing = AsyncMock()  # type: ignore

        await adapter._dm_cancel_session(sid)

        adapter.cancel_session_processing.assert_awaited_once()
        self.assertNotIn(sid, adapter._dm_inflight)
        kwargs = adapter.cancel_session_processing.await_args.kwargs
        self.assertTrue(kwargs["release_guard"])
        self.assertTrue(kwargs["discard_pending"])


class DMPromptCounterTests(unittest.IsolatedAsyncioTestCase):
    """DM session/prompt traffic counter increments on real prompts only."""

    async def test_dm_prompt_counter_increments_on_prompt(self):
        adapter = KimiAdapter(_cfg())
        adapter.handle_message = AsyncMock()  # type: ignore
        self.assertEqual(adapter._dm_prompt_count, 0)

        # Minimal session/prompt frame shape — one text block.
        frame = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/prompt",
            "params": {
                "sessionId": "im:kimi:main",
                "prompt": [{"type": "text", "text": "hello"}],
            },
        }
        await adapter._dm_on_inbound_frame(frame)

        self.assertEqual(adapter._dm_prompt_count, 1)
        adapter.handle_message.assert_awaited_once()

        # Two more — counter accumulates.
        await adapter._dm_on_inbound_frame(frame)
        await adapter._dm_on_inbound_frame(frame)
        self.assertEqual(adapter._dm_prompt_count, 3)

    async def test_dm_prompt_counter_unchanged_on_non_prompt(self):
        """$/ping, initialize, session/new etc. are not user prompts."""
        adapter = KimiAdapter(_cfg())
        adapter._dm_respond = AsyncMock()  # type: ignore
        adapter.handle_message = AsyncMock()  # type: ignore

        await adapter._dm_on_inbound_frame({
            "jsonrpc": "2.0", "method": "$/ping", "params": {},
        })
        await adapter._dm_on_inbound_frame({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        await adapter._dm_on_inbound_frame({
            "jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {},
        })
        # Frame with no text block — the prompt branch runs but returns early,
        # so we DO NOT count it as a real prompt.
        await adapter._dm_on_inbound_frame({
            "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
            "params": {"sessionId": "im:kimi:main", "prompt": []},
        })

        self.assertEqual(adapter._dm_prompt_count, 0)

    async def test_dm_prompt_counter_does_not_count_empty_prompt(self):
        """Empty / malformed session/prompt frames short-circuit before counting."""
        adapter = KimiAdapter(_cfg())
        adapter._dm_respond = AsyncMock()  # type: ignore
        adapter.handle_message = AsyncMock()  # type: ignore

        await adapter._dm_on_inbound_frame({
            "jsonrpc": "2.0", "id": 7, "method": "session/prompt",
            "params": {"sessionId": "im:kimi:main"},
        })
        self.assertEqual(adapter._dm_prompt_count, 0)
        adapter.handle_message.assert_not_awaited()


class DMHealthSummaryTests(unittest.IsolatedAsyncioTestCase):
    """One-shot DM traffic tripwire at _dm_health_summary_s."""

    async def test_dm_health_summary_warns_on_zero_traffic(self):
        adapter = KimiAdapter(_cfg(dm_health_summary_s=0.01))
        self.assertEqual(adapter._dm_prompt_count, 0)

        with self.assertLogs("kimi_adapter", level=logging.WARNING) as cm:
            await adapter._log_dm_health_summary()

        warnings = [r for r in cm.records if r.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 1)
        msg = warnings[0].getMessage()
        self.assertIn("zero prompts", msg)
        self.assertIn("enable_dms", msg)

    async def test_dm_health_summary_info_on_traffic(self):
        adapter = KimiAdapter(_cfg(dm_health_summary_s=0.01))
        adapter._dm_prompt_count = 7

        with self.assertLogs("kimi_adapter", level=logging.INFO) as cm:
            await adapter._log_dm_health_summary()

        info = [r for r in cm.records if r.levelno == logging.INFO]
        warnings = [r for r in cm.records if r.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 0)
        self.assertTrue(any("received 7 prompts" in r.getMessage() for r in info))

    async def test_dm_health_summary_cancellation_safe(self):
        """Cancelling the task before the delay elapses logs nothing.

        The coroutine catches CancelledError internally so the task finishes
        cleanly (no unhandled exception noise in logs) — we assert no
        summary record was emitted and the task result is consumed.
        """
        adapter = KimiAdapter(_cfg(dm_health_summary_s=60))  # long delay

        # Collect any records that fire during the task's lifetime.
        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = captured.append  # type: ignore[assignment]
        kimi_logger = logging.getLogger("kimi_adapter")
        kimi_logger.addHandler(handler)
        try:
            task = asyncio.create_task(adapter._log_dm_health_summary())
            await asyncio.sleep(0)  # give it a chance to enter sleep()
            task.cancel()
            # The coroutine swallows CancelledError internally, so it
            # returns normally (and awaiting it produces None).
            await task
        finally:
            kimi_logger.removeHandler(handler)
        health_records = [
            r for r in captured
            if "prompts in first hour" in r.getMessage()
            or "zero prompts in first hour" in r.getMessage()
        ]
        self.assertEqual(health_records, [])

    async def test_dm_health_summary_disabled_by_zero_setting(self):
        """dm_health_summary_s=0 skips arming the task in connect()."""
        adapter = KimiAdapter(_cfg(dm_health_summary_s=0))
        self.assertEqual(adapter._dm_health_summary_s, 0)
        # Task is only created inside connect() — this verifies the knob is
        # read and typed correctly. Lifecycle scheduling is covered indirectly
        # by the cancellation test above.


class SessionKeyConfigHoistingTests(unittest.TestCase):
    """session_key_* config reads live in __init__, not the hot cancel path."""

    def test_session_key_attributes_hoisted_from_config(self):
        adapter = KimiAdapter(_cfg(
            group_sessions_per_user=False,
            thread_sessions_per_user=True,
        ))
        self.assertFalse(adapter._group_sessions_per_user)
        self.assertTrue(adapter._thread_sessions_per_user)

    def test_session_key_attributes_defaults(self):
        adapter = KimiAdapter(_cfg())
        # Defaults match prior behavior (see _dm_cancel_session before hoist).
        self.assertTrue(adapter._group_sessions_per_user)
        self.assertFalse(adapter._thread_sessions_per_user)


class WSUpgradeClassificationTests(unittest.IsolatedAsyncioTestCase):
    """_dm_ws_connect_once special-cases 401/403/409 (C2)."""

    async def test_403_returns_permanent(self):
        adapter = KimiAdapter(_cfg())
        with patch(
            "kimi_adapter.websockets.connect",
            side_effect=_FakeWSStatusError(403),
        ):
            rc = await adapter._dm_ws_connect_once()
        self.assertEqual(rc, 3)

    async def test_409_first_strike_cools_off_60s(self):
        adapter = KimiAdapter(_cfg())
        sleep_mock = AsyncMock()
        with patch(
            "kimi_adapter.websockets.connect",
            side_effect=_FakeWSStatusError(409),
        ), patch("kimi_adapter.asyncio.sleep", sleep_mock):
            rc = await adapter._dm_ws_connect_once()
        self.assertEqual(rc, 0)
        self.assertEqual(adapter._dm_409_strikes, 1)
        sleep_mock.assert_awaited_once_with(60.0)

    async def test_409_second_strike_cools_off_300s(self):
        adapter = KimiAdapter(_cfg())
        adapter._dm_409_strikes = 1  # prior strike
        sleep_mock = AsyncMock()
        with patch(
            "kimi_adapter.websockets.connect",
            side_effect=_FakeWSStatusError(409),
        ), patch("kimi_adapter.asyncio.sleep", sleep_mock):
            rc = await adapter._dm_ws_connect_once()
        self.assertEqual(rc, 0)
        self.assertEqual(adapter._dm_409_strikes, 2)
        sleep_mock.assert_awaited_once_with(300.0)

    async def test_401_returns_permanent(self):
        adapter = KimiAdapter(_cfg())
        with patch(
            "kimi_adapter.websockets.connect",
            side_effect=_FakeWSStatusError(401),
        ):
            rc = await adapter._dm_ws_connect_once()
        self.assertEqual(rc, 3)


class LifecycleStatusTests(unittest.IsolatedAsyncioTestCase):
    """connect/disconnect drive base-class status + permanent auth sets fatal."""

    async def test_permanent_auth_triggers_fatal_error(self):
        adapter = KimiAdapter(_cfg())
        adapter._dm_ws_connect_once = AsyncMock(return_value=3)  # type: ignore
        await adapter._dm_ws_loop()
        self.assertEqual(adapter._fatal_error_code, "kimi_dm_auth")
        self.assertFalse(adapter._fatal_error_retryable)

    async def test_closing_shutdown_does_not_set_fatal(self):
        """If _closing is already True (clean shutdown path), don't leak fatal."""
        adapter = KimiAdapter(_cfg())
        adapter._closing = True
        adapter._dm_ws_connect_once = AsyncMock(return_value=3)  # type: ignore
        await adapter._dm_ws_loop()
        self.assertIsNone(adapter._fatal_error_code)

    async def test_groups_permanent_auth_triggers_fatal_error(self):
        adapter = KimiAdapter(_cfg())
        adapter._group_subscribe_once = AsyncMock(return_value=3)  # type: ignore
        await adapter._group_subscribe_loop()
        self.assertEqual(adapter._fatal_error_code, "kimi_groups_auth")


# ─────────────────────────────────────────────────────────────────────
# Wave-2 hardening (Commit 4)
# ─────────────────────────────────────────────────────────────────────


class TrustedOnlyDropLogLevelTests(unittest.IsolatedAsyncioTestCase):
    """trusted_only drops log at INFO with a redacted sender — balancing
    operator visibility against PII hygiene.

    The prior shape (INFO + full short_id) leaked identifiers into log
    aggregators in kimi-claw groups where every user message has
    role='assistant'. The DEBUG demote that followed removed the operator
    tripwire for misconfigured ``group_trusted_senders``. Current shape:
    INFO with sender redacted to ``prefix + 4 chars + ****`` — enough to
    diagnose drops without bleeding full identities.
    """

    async def test_trusted_only_drop_emits_info_with_redacted_sender(self):
        adapter = KimiAdapter(_cfg(
            group_allow_bot_senders="trusted_only",
            group_trusted_senders=["u_someone_else"],
        ))
        adapter.handle_message = AsyncMock()  # type: ignore
        # Fix A: _bot_msg() is summary-only — disable hydration so the
        # cascade doesn't hit the network before the redacted-drop log fires.
        adapter._hydrate_missing_text = False

        # Use a realistic-looking short_id so we can assert that the
        # redaction preserves only the prefix + first 4 body chars.
        msg = _bot_msg(senderShortId="u_gs5ri2l5dpytlap", senderId="assistant-long-id-xyz")

        with self.assertLogs("kimi_adapter", level=logging.DEBUG) as cm:
            await adapter._on_group_event(msg)

        adapter.handle_message.assert_not_awaited()

        drop_records = [
            r for r in cm.records
            if "not in group_trusted_senders" in r.getMessage()
        ]
        self.assertEqual(len(drop_records), 1, f"expected one drop log, got: {drop_records}")
        # INFO, not DEBUG — operators need a grep-able signal.
        self.assertEqual(drop_records[0].levelno, logging.INFO)
        rendered = drop_records[0].getMessage()
        # Redacted token present; full tail absent.
        self.assertIn("u_gs5r****", rendered)
        self.assertNotIn("gs5ri2l5dpytlap", rendered)

    async def test_trusted_only_drop_redacts_sender_id_when_short_id_absent(self):
        """If only sender_id is set (no short_id), that too must be redacted."""
        adapter = KimiAdapter(_cfg(
            group_allow_bot_senders="trusted_only",
            group_trusted_senders=["u_someone_else"],
        ))
        adapter.handle_message = AsyncMock()  # type: ignore
        # Fix A: _bot_msg() is summary-only — disable hydration.
        adapter._hydrate_missing_text = False

        msg = _bot_msg(senderShortId=None, senderId="assistant-long-id-xyz")

        with self.assertLogs("kimi_adapter", level=logging.DEBUG) as cm:
            await adapter._on_group_event(msg)

        drop_records = [
            r for r in cm.records
            if "not in group_trusted_senders" in r.getMessage()
        ]
        self.assertEqual(len(drop_records), 1)
        self.assertEqual(drop_records[0].levelno, logging.INFO)
        rendered = drop_records[0].getMessage()
        # "assistant-long-id-xyz" has no "u_"/"b_" prefix → first 4 + ****.
        self.assertIn("assi****", rendered)
        self.assertNotIn("assistant-long-id-xyz", rendered)


class SubscribeBackoffStateTests(unittest.IsolatedAsyncioTestCase):
    """Subscribe reconnect backoff is instance-scoped and resets to the
    oscillation-safe floor (10s) on the first processed frame post-connect.

    Pre-fix: backoff was a loop-local int that only grew monotonically.
    After hitting the 60s cap it stayed at 60s forever — messages arriving
    during reconnect windows silently lost.

    The fix resets backoff to ``floor`` (not ``base``) so flap-every-30s
    oscillation doesn't drive reconnect delay back to 2s every cycle,
    which would hammer Kimi's infra.
    """

    def _make_parser(self, events):
        """Return an async generator function that yields the given events."""
        async def _gen(_content):
            for ev in events:
                yield ev
        return _gen

    def _install_fake_session(self, adapter):
        """Replace _http_session.post() with a context manager yielding HTTP 200.

        Also pins ``_http_session_loop`` to the currently running loop so
        ``_session_for_current_loop`` treats the fixture as same-loop and
        yields the mock as-is, instead of creating an ephemeral aiohttp
        session that would bypass the mock.
        """
        resp = MagicMock()
        resp.status = 200

        class _AsyncCtx:
            async def __aenter__(self_inner):
                return resp

            async def __aexit__(self_inner, *exc):
                return False

        session = MagicMock()
        # MagicMock attributes are themselves MagicMocks (truthy by default),
        # so ``cached.closed`` would always test as truthy in
        # ``_session_for_current_loop``'s ``getattr(cached, "closed", False)``
        # check, sending the helper into the "stale session, recreate" branch
        # — which silently replaces this fixture with a real
        # ``aiohttp.ClientSession()`` and dials kimi.com for real.  Pinning
        # the attribute to ``False`` keeps the helper on the cached-yield
        # path so tests actually exercise the mock they installed.  Real
        # production aiohttp sessions expose ``.closed`` as a proper bool,
        # so the helper's check is correct there.
        session.closed = False
        session.post = MagicMock(return_value=_AsyncCtx())
        adapter._http_session = session
        try:
            adapter._http_session_loop = asyncio.get_running_loop()
        except RuntimeError:
            # Called outside a running loop (sync setUp). The helper's
            # ``_http_session_loop is None`` branch treats this as
            # caller-installed and yields as-is, so leave it None.
            pass

    async def test_subscribe_backoff_resets_to_floor_after_first_frame(self):
        adapter = KimiAdapter(_cfg())
        self._install_fake_session(adapter)
        # Pretend we've grown backoff to the cap from prior reconnect churn.
        adapter._group_subscribe_backoff = 60.0
        adapter._connect_envelope_parser = self._make_parser(  # type: ignore
            [{"chatMessage": {"chatId": "c", "messageId": "m"}}]
        )
        adapter._on_group_event = AsyncMock()  # type: ignore

        rc = await adapter._group_subscribe_once()

        self.assertEqual(rc, 0)
        # Reset lands at floor (10s) — NOT base (2s) and NOT the prior 60s.
        self.assertEqual(adapter._group_subscribe_backoff, 10.0)
        self.assertTrue(adapter._group_subscribe_frame_since_connect)

    async def test_subscribe_backoff_respects_floor_under_oscillation(self):
        """Once backoff has grown past the floor, reconnect recoveries must
        clamp to the floor — never below — so a flap-every-30s oscillation
        can't drive the reconnect delay back to the 2s base each cycle.

        Seed at ``floor * 2`` to skip the cold-start no-reset window and
        exercise the oscillation invariant directly.
        """
        adapter = KimiAdapter(_cfg())
        self._install_fake_session(adapter)
        adapter._on_group_event = AsyncMock()  # type: ignore
        # Seed past the floor so every cycle genuinely hits the reset path.
        adapter._group_subscribe_backoff = adapter._group_subscribe_backoff_floor * 2

        observed_backoffs = []
        for cycle in range(3):
            adapter._connect_envelope_parser = self._make_parser(  # type: ignore
                [{"chatMessage": {"chatId": f"c{cycle}", "messageId": f"m{cycle}"}}]
            )
            await adapter._group_subscribe_once()
            observed_backoffs.append(adapter._group_subscribe_backoff)
            # Simulate the loop's post-error grow step between cycles.
            adapter._group_subscribe_backoff = min(
                adapter._group_subscribe_backoff * 2, adapter._reconnect_max_s,
            )

        # Every post-reset value is at least the floor (10s). Without the
        # floor, naive reset to base would yield 2s on every cycle → thrash.
        for value in observed_backoffs:
            self.assertGreaterEqual(value, adapter._group_subscribe_backoff_floor)

    async def test_subscribe_emits_recovered_log_after_frame(self):
        adapter = KimiAdapter(_cfg())
        self._install_fake_session(adapter)
        adapter._group_subscribe_backoff = 32.0  # mid-growth reconnect
        adapter._connect_envelope_parser = self._make_parser(  # type: ignore
            [{"chatMessage": {"chatId": "c", "messageId": "m"}}]
        )
        adapter._on_group_event = AsyncMock()  # type: ignore

        with self.assertLogs("kimi_adapter", level=logging.INFO) as cm:
            await adapter._group_subscribe_once()

        recovered = [r for r in cm.records if "stream recovered" in r.getMessage()]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].levelno, logging.INFO)
        # Log carries the backoff that was in effect at the time of the
        # reconnect — useful operator signal for "how long were we down".
        self.assertIn("32.0s", recovered[0].getMessage())

    async def test_subscribe_no_duplicate_recovered_log(self):
        adapter = KimiAdapter(_cfg())
        self._install_fake_session(adapter)
        adapter._group_subscribe_backoff = 32.0  # past the floor → recovery will log
        adapter._connect_envelope_parser = self._make_parser(  # type: ignore
            [
                {"chatMessage": {"chatId": "c", "messageId": "m1"}},
                {"chatMessage": {"chatId": "c", "messageId": "m2"}},
                {"chatMessage": {"chatId": "c", "messageId": "m3"}},
            ]
        )
        adapter._on_group_event = AsyncMock()  # type: ignore

        with self.assertLogs("kimi_adapter", level=logging.INFO) as cm:
            await adapter._group_subscribe_once()

        recovered = [r for r in cm.records if "stream recovered" in r.getMessage()]
        self.assertEqual(len(recovered), 1, f"expected exactly one 'stream recovered', got: {len(recovered)}")

    async def test_subscribe_backoff_not_reset_on_keepalive_ping(self):
        """A degraded stream emitting only keepalive pings must NOT reset
        backoff — otherwise a ping-only loop would thrash back to the floor
        on every reconnect and hammer Kimi's infra.
        """
        adapter = KimiAdapter(_cfg())
        self._install_fake_session(adapter)
        adapter._group_subscribe_backoff = 32.0  # would be reset if ping counted
        adapter._connect_envelope_parser = self._make_parser(  # type: ignore
            [{"ping": {}}]
        )
        adapter._on_group_event = AsyncMock()  # type: ignore

        records, teardown = _capture_kimi_log_records()
        try:
            await adapter._group_subscribe_once()
        finally:
            teardown()

        # Backoff unchanged, hook not armed, no recovery log emitted.
        self.assertEqual(adapter._group_subscribe_backoff, 32.0)
        self.assertFalse(adapter._group_subscribe_frame_since_connect)
        recovered = [r for r in records if "stream recovered" in r.getMessage()]
        self.assertEqual(recovered, [])

    async def test_subscribe_backoff_not_reset_on_empty_stream(self):
        """Stream opens and closes cleanly with zero events — must NOT flip
        state or emit the recovery log.
        """
        adapter = KimiAdapter(_cfg())
        self._install_fake_session(adapter)
        adapter._group_subscribe_backoff = 32.0
        adapter._connect_envelope_parser = self._make_parser([])  # type: ignore
        adapter._on_group_event = AsyncMock()  # type: ignore

        records, teardown = _capture_kimi_log_records()
        try:
            rc = await adapter._group_subscribe_once()
        finally:
            teardown()

        self.assertEqual(rc, 0)
        self.assertEqual(adapter._group_subscribe_backoff, 32.0)
        self.assertFalse(adapter._group_subscribe_frame_since_connect)
        recovered = [r for r in records if "stream recovered" in r.getMessage()]
        self.assertEqual(recovered, [])

    async def test_subscribe_backoff_not_reset_on_handler_exception(self):
        """If _on_group_event raises on the first chatMessage, the
        operator-facing recovery log must NOT fire — otherwise a
        dispatch-error loop would masquerade as a healthy stream.
        Backoff is untouched too.

        Cumulative review #58 split the pre-dispatch state transitions
        from the post-dispatch operator log: counter + ts + gate flip
        happen synchronously with first-frame detection (so the gap
        log inside dispatch reads consistent ``connect#`` values), but
        the recovery LOG and backoff CLAMP still gate on processing
        success. This test exercises that split — pre-dispatch state
        moves forward; operator-visible "healthy stream" signals do
        NOT fire on dispatch failure.

        The exception still propagates as before (caught by the outer
        handlers into return code 0 / transient retry; the next cycle
        resets the gate at the top of ``_group_subscribe_once``).
        """
        adapter = KimiAdapter(_cfg())
        self._install_fake_session(adapter)
        adapter._group_subscribe_backoff = 32.0
        adapter._connect_envelope_parser = self._make_parser(  # type: ignore
            [{"chatMessage": {"chatId": "c", "messageId": "m"}}]
        )
        adapter._on_group_event = AsyncMock(  # type: ignore
            side_effect=KimiTransientError("simulated handler failure")
        )

        records, teardown = _capture_kimi_log_records()
        try:
            rc = await adapter._group_subscribe_once()
        finally:
            teardown()

        # Transient handler errors are swallowed to rc=0 (retry) by the
        # outer except block, so the caller retries the reconnect.
        self.assertEqual(rc, 0)
        # Backoff: untouched (the clamp is post-dispatch, gated on success).
        self.assertEqual(adapter._group_subscribe_backoff, 32.0)
        # Pre-dispatch state IS now advanced (#58): the counter increments
        # and the gate flips before _on_group_event runs. Their meaning is
        # "WS layer delivered a real chatMessage this cycle", which is
        # true regardless of processing outcome. Cycle-level reset on the
        # next reconnect (line 2500) clears the gate cleanly.
        self.assertTrue(adapter._group_subscribe_frame_since_connect)
        self.assertEqual(adapter._group_subscribe_reconnect_count, 1)
        # The actual operator-visible "healthy stream" signals must
        # remain silent on dispatch failure — both the new "subscribe
        # stream live" and the legacy "stream recovered" log lines.
        live = [r for r in records if "subscribe stream live" in r.getMessage()]
        self.assertEqual(live, [], "stream-live log must not fire on dispatch failure")
        recovered = [r for r in records if "stream recovered" in r.getMessage()]
        self.assertEqual(recovered, [])


class HomeChannelNagGuardTests(unittest.IsolatedAsyncioTestCase):
    """Home-channel nag suppresses when /sethome has persisted a channel.

    Pre-fix: the guard at run.py:4415 checked ``os.getenv(env_key)`` only.
    The /sethome handler (run.py:5952) persists to config.yaml via
    ``HomeChannel`` wiring AND sets the env var, but a gateway restarted
    after /sethome may lose env-var state before config reloads — and
    the authoritative source is the config, not the environment.

    The fix checks ``self.config.get_home_channel(platform)`` first, then
    falls back to env for the true first-run case.

    These tests exercise the guard condition directly because
    ``_handle_message_with_agent`` spans 2000+ lines and requires the
    entire GatewayRunner state graph to reach the nag block. The
    condition itself is a 2-line expression — a focused unit test on
    that expression gives higher signal than a flaky end-to-end.
    """

    def _evaluate_nag_condition(self, config, platform, env_key_present: bool) -> bool:
        """Replicate the guard at gateway/run.py:4425-4438 exactly.

        Returns True iff the nag SHOULD fire under the given config+env.
        """
        env_value = "some-chat-id" if env_key_present else None
        home_channel = config.get_home_channel(platform)
        # Guard from run.py: send nag if neither source has a value.
        return not home_channel and not env_value

    def test_home_channel_nag_suppressed_when_config_has_home_channel(self):
        config = GatewayConfig(
            platforms={
                Platform("kimiclaw"): PlatformConfig(
                    enabled=True,
                    token="tok",
                    home_channel=HomeChannel(
                        platform=Platform("kimiclaw"),
                        chat_id="room:abc",
                        name="Home",
                    ),
                ),
            },
        )
        # Config has a home channel; env is empty — nag must NOT fire.
        self.assertFalse(self._evaluate_nag_condition(config, Platform("kimiclaw"), env_key_present=False))
        # Sanity: the config API used by the fix returns the HomeChannel object.
        self.assertIsNotNone(config.get_home_channel(Platform("kimiclaw")))

    def test_home_channel_nag_fires_when_both_env_and_config_empty(self):
        """True first-run: no env, no config → operator should be prompted."""
        config = GatewayConfig(
            platforms={Platform("kimiclaw"): PlatformConfig(enabled=True, token="tok")},
        )
        self.assertTrue(self._evaluate_nag_condition(config, Platform("kimiclaw"), env_key_present=False))
        # Env-set first-run path still suppresses (preserved behavior).
        self.assertFalse(self._evaluate_nag_condition(config, Platform("kimiclaw"), env_key_present=True))


def _capture_kimi_log_records(level: int = logging.DEBUG):
    """Attach a list-capturing handler to the kimi logger.

    Module-level shared helper used by both ``SubscribeBackoffStateTests``
    and the Probe* test classes. Unlike ``assertLogs``, does not require
    at least one record — suitable for tests that expect *zero* logs
    from a code path. Returns ``(records_list, teardown_callable)``.
    """
    records: list = []

    class _ListHandler(logging.Handler):
        def emit(self_inner, record):
            records.append(record)

    handler = _ListHandler(level=level)
    kimi_logger = logging.getLogger("kimi_adapter")
    prev_level = kimi_logger.level
    kimi_logger.addHandler(handler)
    kimi_logger.setLevel(level)

    def _teardown():
        kimi_logger.removeHandler(handler)
        kimi_logger.setLevel(prev_level)

    return records, _teardown


class Probe1BlockCaseTypeTests(unittest.TestCase):
    """Probe (H-A): DEBUG dump of non-text block shapes from
    ``_extract_blocks_payload``. Discriminates fragmented long messages
    from unknown block variants (resourceLink, mention, code, etc.)
    without behavior change.
    """

    def test_block_probe_fires_for_unknown_block_type(self):
        """resourceLink block with no text → per-block DEBUG log includes
        content_case, and the aggregate summary reports envelope lengths.
        """
        msg = {
            "text": "envelope preview text",
            "summary": "envelope preview summary",
            "blocks": [
                {
                    "id": "b1",
                    "content": {
                        "case": "resourceLink",
                        "value": {"uri": "kimi://file/x", "title": "x"},
                    },
                }
            ],
        }
        records, teardown = _capture_kimi_log_records()
        try:
            text, urls, types = _extract_blocks_payload(msg)
        finally:
            teardown()

        # Behavior unchanged: no text, but URI still extracted.
        self.assertEqual(text, "")
        self.assertEqual(urls, ["kimi://file/x"])
        self.assertEqual(types, ["resource_link"])

        per_block = [
            r for r in records
            if "non-text block (no extracted text)" in r.getMessage()
        ]
        self.assertEqual(len(per_block), 1, f"expected one per-block log, got: {per_block}")
        self.assertEqual(per_block[0].levelno, logging.DEBUG)
        rendered = per_block[0].getMessage()
        self.assertIn("'content_case': 'resourceLink'", rendered)
        self.assertIn("'has_uri': True", rendered)

        # Aggregate summary with envelope lengths (Fix D oracle).
        summary_log = [
            r for r in records
            if "non-text block(s)" in r.getMessage()
        ]
        self.assertEqual(len(summary_log), 1)
        rendered_sum = summary_log[0].getMessage()
        self.assertIn("extracted_text=0", rendered_sum)
        self.assertIn(f"envelope_text={len('envelope preview text')}", rendered_sum)
        self.assertIn(f"envelope_summary={len('envelope preview summary')}", rendered_sum)

    def test_block_probe_silent_for_text_blocks(self):
        """Normal text-bearing block must emit no probe log (neither per-block
        nor aggregate summary).
        """
        msg = {
            "blocks": [
                {
                    "id": "b1",
                    "content": {
                        "case": "text",
                        "value": {"content": "hello world"},
                    },
                }
            ],
        }
        records, teardown = _capture_kimi_log_records()
        try:
            text, _, _ = _extract_blocks_payload(msg)
        finally:
            teardown()

        self.assertEqual(text, "hello world")
        per_block = [
            r for r in records
            if "non-text block (no extracted text)" in r.getMessage()
        ]
        self.assertEqual(per_block, [])
        summary_log = [
            r for r in records
            if "non-text block(s)" in r.getMessage()
        ]
        self.assertEqual(summary_log, [])

    def test_block_probe_silent_for_all_text_blocks(self):
        """Negative-assertion: every block yields text → neither per-block
        probe nor the aggregate summary fires (Fix E trigger invariant).
        """
        msg = {
            "blocks": [
                {"content": {"case": "text", "value": {"content": "alpha"}}},
                {"content": {"case": "text", "value": {"content": "beta"}}},
            ],
        }
        records, teardown = _capture_kimi_log_records()
        try:
            text, _, _ = _extract_blocks_payload(msg)
        finally:
            teardown()

        self.assertEqual(text, "alpha\nbeta")
        probe = [
            r for r in records
            if "non-text block" in r.getMessage()
        ]
        self.assertEqual(probe, [])

    def test_block_probe_handles_non_dict_block(self):
        """Malformed non-dict block → no crash; DEBUG log includes type name."""
        msg = {"blocks": ["not a dict", 42]}
        records, teardown = _capture_kimi_log_records()
        try:
            text, urls, types = _extract_blocks_payload(msg)
        finally:
            teardown()

        self.assertEqual(text, "")
        self.assertEqual(urls, [])
        self.assertEqual(types, [])

        per_block = [
            r for r in records
            if "non-text block (no extracted text)" in r.getMessage()
        ]
        self.assertEqual(len(per_block), 2)
        rendered = " ".join(r.getMessage() for r in per_block)
        self.assertIn("'type': 'str'", rendered)
        self.assertIn("'type': 'int'", rendered)

    def test_block_probe_logs_envelope_lengths(self):
        """Fix D oracle: aggregate log reports envelope text/summary lengths
        + extracted length, so operators can distinguish "legitimate media
        attachment" from "orphaned fragmented tail".
        """
        msg = {
            "text": "short envelope",
            "summary": "much longer envelope preview than the extracted bit",
            "blocks": [
                {"content": {"case": "text", "value": {"content": "hi"}}},
                {
                    "id": "b2",
                    "content": {
                        "case": "resourceLink",
                        "value": {"uri": "kimi://file/y"},
                    },
                },
            ],
        }
        records, teardown = _capture_kimi_log_records()
        try:
            text, _, _ = _extract_blocks_payload(msg)
        finally:
            teardown()

        self.assertEqual(text, "hi")
        summary_log = [
            r for r in records
            if "non-text block(s)" in r.getMessage()
        ]
        self.assertEqual(len(summary_log), 1)
        rendered = summary_log[0].getMessage()
        self.assertIn("extracted_text=2", rendered)
        self.assertIn(f"envelope_text={len('short envelope')}", rendered)
        self.assertIn(
            f"envelope_summary={len('much longer envelope preview than the extracted bit')}",
            rendered,
        )
        # Oracle signature: envelope_summary > extracted_text → operator
        # sees the mismatch immediately.
        self.assertGreater(
            len("much longer envelope preview than the extracted bit"),
            2,
        )


class Probe2TextSourceTests(unittest.IsolatedAsyncioTestCase):
    """Probe (H-B): DEBUG log reports which source populated ``text``
    (blocks / text / summary / none) and the lengths of each candidate.
    Reveals when a short summary silently wins over empty blocks and
    bypasses the hydration gate.
    """

    def _adapter(self):
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        # Disable hydration so the probe log fires on the original text,
        # not on the re-extracted hydrated payload.
        adapter._hydrate_missing_text = False
        return adapter

    async def test_text_source_probe_chooses_blocks(self):
        adapter = self._adapter()
        msg = {
            "chatMessage": {
                "chatId": "room-a",
                "messageId": "01HKCHF4FC4S0W7T3V74SG6AT6",
                "status": "STATUS_COMPLETED",
                "role": "ROLE_USER",
                "senderId": "u1",
                "senderShortId": "u_real",
                "summary": "preview",
                "blocks": [
                    {"content": {"case": "text", "value": {"content": "body text"}}}
                ],
            }
        }
        records, teardown = _capture_kimi_log_records()
        try:
            await adapter._on_group_event(msg)
        finally:
            teardown()

        probe = [r for r in records if "text source for" in r.getMessage()]
        self.assertEqual(len(probe), 1, f"expected one probe log, got: {probe}")
        rendered = probe[0].getMessage()
        self.assertIn("chose=blocks", rendered)
        self.assertIn("blocks=9", rendered)
        # Inline body present → hydration check short-circuits to
        # `skipped:inline` regardless of the _hydrate_missing_text flag
        # (Fix C distinguishes "not needed" from "operator policy off").
        self.assertIn("hydrated=skipped:inline", rendered)

    async def test_text_source_probe_chooses_summary_when_blocks_empty(self):
        adapter = self._adapter()
        msg = {
            "chatMessage": {
                "chatId": "room-b",
                "messageId": "01HKCHF4FC4S0W7T3V74SG6AT7",
                "status": "STATUS_COMPLETED",
                "role": "ROLE_USER",
                "senderId": "u1",
                "senderShortId": "u_real",
                "summary": "preview-only",
                "blocks": [],
            }
        }
        records, teardown = _capture_kimi_log_records()
        try:
            await adapter._on_group_event(msg)
        finally:
            teardown()

        probe = [r for r in records if "text source for" in r.getMessage()]
        self.assertEqual(len(probe), 1)
        rendered = probe[0].getMessage()
        self.assertIn("chose=summary", rendered)
        self.assertIn("blocks=0", rendered)
        self.assertIn(f"summary={len('preview-only')}", rendered)
        # Hydration disabled by fixture (`_hydrate_missing_text=False`) →
        # `skipped:disabled` distinguishes operator policy from "inline
        # body present, hydration not needed" (Fix C).
        self.assertIn("hydrated=skipped:disabled", rendered)

    async def test_text_source_probe_chooses_none_when_all_empty(self):
        adapter = self._adapter()
        msg = {
            "chatMessage": {
                "chatId": "room-c",
                "messageId": "01HKCHF4FC4S0W7T3V74SG6AT8",
                "status": "STATUS_COMPLETED",
                "role": "ROLE_USER",
                "senderId": "u1",
                "senderShortId": "u_real",
            }
        }
        records, teardown = _capture_kimi_log_records()
        try:
            await adapter._on_group_event(msg)
        finally:
            teardown()

        probe = [r for r in records if "text source for" in r.getMessage()]
        self.assertEqual(len(probe), 1)
        rendered = probe[0].getMessage()
        self.assertIn("chose=none", rendered)
        self.assertIn("blocks=0", rendered)
        self.assertIn("text=0", rendered)
        self.assertIn("summary=0", rendered)
        # No miss candidates when every source is empty.
        self.assertIn("miss_candidate=none", rendered)
        # Fix C: `skipped:disabled` (not generic `skipped`) when the
        # hydration flag is off.
        self.assertIn("hydrated=skipped:disabled", rendered)

    async def test_text_source_probe_flags_miss_candidate_when_summary_longer(self):
        """Fix C: when a non-chosen candidate is LONGER than the chosen one,
        `miss_candidate=<name>` flags the hydration-miss signature directly
        so operators don't have to eyeball per-field lengths.
        """
        adapter = self._adapter()
        msg = {
            "chatMessage": {
                "chatId": "room-d",
                "messageId": "01HKCHF4FC4S0W7T3V74SG6AT9",
                "status": "STATUS_COMPLETED",
                "role": "ROLE_USER",
                "senderId": "u1",
                "senderShortId": "u_real",
                "text": "short",
                "summary": "much longer preview",
                "blocks": [],
            }
        }
        records, teardown = _capture_kimi_log_records()
        try:
            await adapter._on_group_event(msg)
        finally:
            teardown()

        probe = [r for r in records if "text source for" in r.getMessage()]
        self.assertEqual(len(probe), 1)
        rendered = probe[0].getMessage()
        self.assertIn("chose=text", rendered)
        self.assertIn("miss_candidate=summary", rendered)
        # Inline body present (text=short) → `skipped:inline` (Fix C),
        # not `skipped:disabled` — the flag is off in this fixture but
        # the inline-present check short-circuits first.
        self.assertIn("hydrated=skipped:inline", rendered)


class HydrateWhenInlineBodyEmptyTests(unittest.IsolatedAsyncioTestCase):
    """H-B fix (Commit 6): when Subscribe ships ``blocks=[]`` and no inline
    ``text`` but a non-empty ``summary`` preview, hydrate from
    ``ListMessages`` rather than dispatching the truncated preview to the
    agent. Summary remains as a last-resort fallback if hydration fails.

    Production trigger (2026-04-26 11:21:36 BST): Probe 2 captured
    ``blocks=0, text=0, summary=50, chose=summary, miss_candidate=none``
    for a ~150-char inbound; agent answered against the 50-char preview.
    """

    def _summary_only_event(self, *, chat_id="chat-hb", message_id="msg-hb",
                            summary="this is a 50-char-ish truncated server-side preview"):
        return {
            "chatMessage": {
                "chatId": chat_id,
                "messageId": message_id,
                "status": "STATUS_COMPLETED",
                "role": "ROLE_USER",
                "senderId": "u1",
                "senderShortId": "u_real",
                "summary": summary,
                "blocks": [],
            }
        }

    async def test_summary_only_event_triggers_hydration(self):
        """Empty blocks + empty text + populated summary + hydration enabled
        → ``_fetch_group_message`` called exactly once with the chat/message
        ids from the event.
        """
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        # Default is True; assert explicitly to make the contract visible.
        self.assertTrue(adapter._hydrate_missing_text)
        adapter._fetch_group_message = AsyncMock(return_value={
            "id": "msg-hb",
            "blocks": [
                {"content": {"case": "text", "value": {"content": "full body"}}}
            ],
        })  # type: ignore

        await adapter._on_group_event(self._summary_only_event())

        adapter._fetch_group_message.assert_awaited_once_with("chat-hb", "msg-hb")

    async def test_hydrated_body_wins_over_summary(self):
        """Dispatched ``MessageEvent.text`` matches the hydrated body, not
        the summary preview — the whole point of the H-B fix.
        """
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        full_body = (
            "please reply with the literal word pineapple so the operator "
            "can verify full delivery (this is a ~150 char message)"
        )
        adapter._fetch_group_message = AsyncMock(return_value={
            "id": "msg-hb",
            "senderId": "u1",
            "senderShortId": "u_real",
            "blocks": [
                {"content": {"case": "text", "value": {"content": full_body}}}
            ],
        })  # type: ignore

        await adapter._on_group_event(
            self._summary_only_event(summary="please reply with the literal word…")
        )

        adapter.handle_message.assert_awaited_once()
        delivered = adapter.handle_message.await_args.args[0]
        self.assertEqual(delivered.text, full_body)
        self.assertNotIn("…", delivered.text)

    async def test_hydration_failure_falls_back_to_summary(self):
        """When hydration raises ``KimiAdapterError`` (transient infra blip)
        we still dispatch with the summary text rather than dropping the
        message entirely. Graceful degradation — better-than-nothing.

        Fix B: summary fallback now prepends a truncation marker so the
        agent can acknowledge the body is a preview rather than answer
        confidently against half a sentence (same H-B failure mode as
        the original bug, just less frequent).
        """
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        summary = "preview-only fallback text"
        adapter._fetch_group_message = AsyncMock(
            side_effect=KimiRpcError("hydration failed")
        )  # type: ignore

        await adapter._on_group_event(self._summary_only_event(summary=summary))

        adapter.handle_message.assert_awaited_once()
        delivered = adapter.handle_message.await_args.args[0]
        self.assertTrue(
            delivered.text.startswith("[message truncated"),
            f"expected truncation marker prefix, got: {delivered.text!r}",
        )
        self.assertTrue(
            delivered.text.endswith(summary),
            f"expected summary at tail, got: {delivered.text!r}",
        )

    async def test_hydration_skipped_when_inline_text_present(self):
        """Happy path: inline blocks/text present → no hydration RPC even
        if a summary also exists. Guards against re-introducing per-event
        ``ListMessages`` overhead in the common case.
        """
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._fetch_group_message = AsyncMock()  # type: ignore

        await adapter._on_group_event({
            "chatMessage": {
                "chatId": "chat-happy",
                "messageId": "msg-happy",
                "status": "STATUS_COMPLETED",
                "role": "ROLE_USER",
                "senderId": "u1",
                "senderShortId": "u_real",
                "summary": "preview",
                "blocks": [
                    {"content": {"case": "text", "value": {"content": "inline body"}}}
                ],
            }
        })

        adapter._fetch_group_message.assert_not_awaited()
        adapter.handle_message.assert_awaited_once()
        delivered = adapter.handle_message.await_args.args[0]
        self.assertEqual(delivered.text, "inline body")

    async def test_probe2_logs_hydrated_field(self):
        """Probe 2 reports ``hydrated=true`` and ``chose=hydrated`` when
        a summary-only inbound is successfully hydrated. Confirms the
        observability surface tracks the new control flow.
        """
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        full_body = "the actual full body fetched via ListMessages"
        adapter._fetch_group_message = AsyncMock(return_value={
            "id": "msg-hb",
            "blocks": [
                {"content": {"case": "text", "value": {"content": full_body}}}
            ],
        })  # type: ignore

        records, teardown = _capture_kimi_log_records()
        try:
            await adapter._on_group_event(
                self._summary_only_event(summary="short preview")
            )
        finally:
            teardown()

        probe = [r for r in records if "text source for" in r.getMessage()]
        self.assertEqual(len(probe), 1, f"expected one probe log, got: {probe}")
        rendered = probe[0].getMessage()
        self.assertIn("hydrated=true", rendered)
        self.assertIn("chose=hydrated", rendered)
        # Original raw-event candidates: blocks=0 text=0 summary>0.
        self.assertIn("blocks=0", rendered)
        self.assertIn("text=0", rendered)

    async def test_summary_fallback_includes_truncation_marker(self):
        """Fix B: hydration failure → summary fallback path prepends a
        clear truncation marker so the agent knows the body is a preview
        and can acknowledge rather than confidently answer against half
        a sentence. Same H-B failure mode as the original bug — just less
        frequent now that hydration is the primary path.
        """
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        summary = "this is the 50-char-ish truncated server-side preview"
        adapter._fetch_group_message = AsyncMock(
            side_effect=KimiRpcError("hydration unavailable")
        )  # type: ignore

        await adapter._on_group_event(self._summary_only_event(summary=summary))

        adapter.handle_message.assert_awaited_once()
        delivered = adapter.handle_message.await_args.args[0]
        self.assertTrue(
            delivered.text.startswith("[message truncated"),
            f"expected truncation marker prefix, got: {delivered.text!r}",
        )
        self.assertIn(summary, delivered.text)

    async def test_probe2_logs_hydrated_skipped_disabled_when_flag_off(self):
        """Fix C: ``_hydrate_missing_text=False`` → ``hydrated=skipped:disabled``
        in Probe 2 (operator policy distinguishable from "inline body
        present, hydration not needed").
        """
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False  # operator policy: no hydration
        # _fetch_group_message intentionally NOT mocked: with the flag off
        # it must never be invoked.

        records, teardown = _capture_kimi_log_records()
        try:
            await adapter._on_group_event(
                self._summary_only_event(summary="short preview")
            )
        finally:
            teardown()

        probe = [r for r in records if "text source for" in r.getMessage()]
        self.assertEqual(len(probe), 1, f"expected one probe log, got: {probe}")
        rendered = probe[0].getMessage()
        self.assertIn("hydrated=skipped:disabled", rendered)
        self.assertNotIn("hydrated=skipped:inline", rendered)

    async def test_probe2_logs_hydrated_skipped_inline_when_inline_text_present(self):
        """Fix C: inline body present (non-empty blocks/text) →
        ``hydrated=skipped:inline``, regardless of the
        ``_hydrate_missing_text`` flag. Distinguishes "happy path, no
        hydration needed" from "operator turned hydration off".
        """
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._fetch_group_message = AsyncMock()  # type: ignore

        records, teardown = _capture_kimi_log_records()
        try:
            await adapter._on_group_event({
                "chatMessage": {
                    "chatId": "chat-inline",
                    "messageId": "msg-inline",
                    "status": "STATUS_COMPLETED",
                    "role": "ROLE_USER",
                    "senderId": "u1",
                    "senderShortId": "u_real",
                    "summary": "preview",
                    "blocks": [
                        {"content": {"case": "text", "value": {"content": "inline body"}}}
                    ],
                }
            })
        finally:
            teardown()

        adapter._fetch_group_message.assert_not_awaited()
        probe = [r for r in records if "text source for" in r.getMessage()]
        self.assertEqual(len(probe), 1, f"expected one probe log, got: {probe}")
        rendered = probe[0].getMessage()
        self.assertIn("hydrated=skipped:inline", rendered)
        self.assertNotIn("hydrated=skipped:disabled", rendered)

    async def test_probe2_logs_hydrated_false_when_hydration_returns_empty_payload(self):
        """Fix E: hydration returns a truthy-but-empty payload (e.g. wrapper
        with no blocks/text) → ``hydrated=false``, NOT ``hydrated=true``.
        Prior shape set ``hydrated=true`` whenever the hydrated dict was
        truthy, before checking whether ``_extract_blocks_payload`` actually
        yielded text — Probe 2 then falsely claimed a hydration win on an
        empty payload.

        Falls through to summary fallback (Fix B annotated).
        """
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        # Truthy but yields no text — wrapper-only payload.
        adapter._fetch_group_message = AsyncMock(return_value={
            "id": "msg-hb",
            "blocks": [],
        })  # type: ignore

        records, teardown = _capture_kimi_log_records()
        try:
            await adapter._on_group_event(
                self._summary_only_event(summary="short preview")
            )
        finally:
            teardown()

        probe = [r for r in records if "text source for" in r.getMessage()]
        self.assertEqual(len(probe), 1, f"expected one probe log, got: {probe}")
        rendered = probe[0].getMessage()
        self.assertIn("hydrated=false", rendered)
        self.assertNotIn("hydrated=true", rendered)
        # Cascade then falls through to Fix B's annotated summary.
        self.assertIn("chose=summary", rendered)


class Probe3MessageIdTimingTests(unittest.IsolatedAsyncioTestCase):
    """Probe (H-C): per-room message_id timing DEBUG log for post-hoc
    burst-drop correlation. Updated BEFORE the filter chain so drops
    still count as "what Kimi sent us" — gaps then map to burst losses.
    """

    def _adapter(self):
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all"))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False
        return adapter

    def _msg(self, chat_id, message_id):
        return {
            "chatMessage": {
                "chatId": chat_id,
                "messageId": message_id,
                "status": "STATUS_COMPLETED",
                "role": "ROLE_USER",
                "senderId": "u1",
                "senderShortId": "u_real",
                "blocks": [
                    {"content": {"case": "text", "value": {"content": "hi"}}}
                ],
            }
        }

    async def test_message_id_probe_first_seen_for_new_room(self):
        adapter = self._adapter()
        records, teardown = _capture_kimi_log_records()
        try:
            await adapter._on_group_event(self._msg("room-X", "01HKCHF4FC4S0W7T3V74SG6AT6"))
        finally:
            teardown()

        probe = [r for r in records if "message_id first-seen" in r.getMessage()]
        self.assertEqual(len(probe), 1, f"expected one first-seen log, got: {probe}")
        rendered = probe[0].getMessage()
        self.assertIn("room=room-X", rendered)
        self.assertIn("id=01HKCHF4FC4S0W7T3V74SG6AT6", rendered)

    async def test_message_id_probe_emits_delta_for_second_message(self):
        adapter = self._adapter()
        # Bump char 10 (end of the 48-bit timestamp prefix) so the two ULIDs
        # decode to different millisecond timestamps — the random tail has
        # no effect on the delta.
        first_id = "01HKCHF4FC4S0W7T3V74SG6AT6"
        second_id = "01HKCHF4FD4S0W7T3V74SG6AT6"
        # DEBUG enabled on both dispatches so the log emission fires; the
        # tracker update itself is hoisted OUT of the DEBUG gate (Fix A)
        # so it populates regardless of level.
        records, teardown = _capture_kimi_log_records()
        try:
            await adapter._on_group_event(self._msg("room-Y", first_id))
            first_records_len = len(records)
            await adapter._on_group_event(self._msg("room-Y", second_id))
        finally:
            teardown()

        # Only inspect records emitted during the second dispatch.
        second_records = records[first_records_len:]
        timing = [r for r in second_records if "message_id timing" in r.getMessage()]
        self.assertEqual(len(timing), 1)
        rendered = timing[0].getMessage()
        self.assertIn("room=room-Y", rendered)
        self.assertIn(f"prev={first_id}", rendered)
        self.assertIn("delta_ms=", rendered)
        # delta must equal the second - first timestamp diff.
        expected_delta = _ulid_time_ms(second_id) - _ulid_time_ms(first_id)
        self.assertIn(f"delta_ms={expected_delta}", rendered)
        # Second dispatch must not emit a first-seen log.
        first_seen = [r for r in second_records if "first-seen" in r.getMessage()]
        self.assertEqual(first_seen, [])

    async def test_message_id_probe_per_room_isolation(self):
        adapter = self._adapter()
        records, teardown = _capture_kimi_log_records()
        try:
            # Room A: two messages.
            await adapter._on_group_event(self._msg("room-A", "01HKCHF4FC4S0W7T3V74SG6AT6"))
            await adapter._on_group_event(self._msg("room-A", "01HKCHF4FD4S0W7T3V74SG6AT6"))
            boundary = len(records)
            # Room B: first message — must log first-seen, not a delta from A.
            await adapter._on_group_event(self._msg("room-B", "01HKCHF4FE4S0W7T3V74SG6AT6"))
        finally:
            teardown()

        b_records = records[boundary:]
        first_seen = [r for r in b_records if "first-seen" in r.getMessage()]
        timing = [r for r in b_records if "message_id timing" in r.getMessage()]
        self.assertEqual(len(first_seen), 1, f"expected first-seen for room-B, got: {b_records}")
        self.assertIn("room=room-B", first_seen[0].getMessage())
        self.assertEqual(timing, [], "room-B must not emit a delta against room-A")

    def test_ulid_time_ms_parses_known_ulid(self):
        """Canonical ULID decode — cross-checked against python-ulid."""
        # '01ARZ3NDEKTSV4RRFFQ69G5FAV' → 1469922850259 ms
        # (2016-07-30T23:54:10.259+00:00 UTC), verified against python-ulid.
        self.assertEqual(
            _ulid_time_ms("01ARZ3NDEKTSV4RRFFQ69G5FAV"),
            1469922850259,
        )
        # Case-insensitive.
        self.assertEqual(
            _ulid_time_ms("01arz3ndektsv4rrffq69g5fav"),
            1469922850259,
        )
        # Monotonic across two close ULIDs — later ULID has larger prefix value.
        earlier = _ulid_time_ms("01HKCHF4FC4S0W7T3V74SG6AT6")
        later = _ulid_time_ms("01HKCHF4FD4S0W7T3V74SG6AT6")
        self.assertIsNotNone(earlier)
        self.assertIsNotNone(later)
        self.assertGreater(later, earlier)

    def test_ulid_time_ms_rejects_short_or_invalid(self):
        """None / empty / short / non-crockford → None, not a crash."""
        self.assertIsNone(_ulid_time_ms(None))
        self.assertIsNone(_ulid_time_ms(""))
        self.assertIsNone(_ulid_time_ms("01ARZ3NDE"))  # 9 chars, too short
        # 'I', 'L', 'O', 'U' are NOT in Crockford base32 — must reject.
        # (These IDs are 10 chars so they don't trip the UUID branch.)
        self.assertIsNone(_ulid_time_ms("01IRZ3NDEK"))
        self.assertIsNone(_ulid_time_ms("01LRZ3NDEK"))
        self.assertIsNone(_ulid_time_ms("01ORZ3NDEK"))
        self.assertIsNone(_ulid_time_ms("01URZ3NDEK"))
        # Non-string input.
        self.assertIsNone(_ulid_time_ms(12345))  # type: ignore[arg-type]

    def test_ulid_time_ms_returns_none_for_uuid_v8_production_format(self):
        """Real Kimi message ids are UUID v8 (per Kimi's own error message:
        ``id_kind=uuidv8``). Their first 48 bits are NOT unix-ms — captured
        deltas show ~16× wall-clock seconds, suggesting a non-standard epoch
        encoding. ``_ulid_time_ms`` deliberately does NOT decode UUID format
        because the resulting magnitudes can't be interpreted as
        milliseconds. Production gap-detection uses wall-clock
        ``time.time()`` arrival tracking instead — see
        ``_last_arrival_time_per_room`` and the Phase 0 #18 gap-candidate
        logger. This test guards against a future "fix" that re-introduces
        UUID decoding without addressing the unit mismatch.
        """
        self.assertIsNone(_ulid_time_ms("19dc9c4f-1262-8c1b-8000-0a4a6c626bbb"))
        self.assertIsNone(_ulid_time_ms("19dc9c4d-93c2-87b8-8000-0a4a0ecd76fe"))

    async def test_message_id_probe_tracker_populates_at_info_level(self):
        """Fix A / Fix E invariant: tracker-update is hoisted out of the
        DEBUG gate, so toggling DEBUG on later does NOT falsely report
        ``first-seen`` for a message that already arrived at INFO. No
        probe-3 log is emitted under INFO either.
        """
        adapter = self._adapter()
        # Capture at WARNING (drops DEBUG) to prove the probe is silent
        # while the tracker still fills.
        records, teardown = _capture_kimi_log_records(level=logging.WARNING)
        try:
            await adapter._on_group_event(
                self._msg("room-K", "01HKCHF4FC4S0W7T3V74SG6AT6")
            )
        finally:
            teardown()

        # Tracker populated regardless of log level.
        self.assertEqual(
            adapter._last_message_id_per_room.get("room-K"),
            "01HKCHF4FC4S0W7T3V74SG6AT6",
        )
        # No probe-3 log emitted at WARNING.
        probe = [r for r in records if "message_id" in r.getMessage()]
        self.assertEqual(probe, [])

    async def test_probes_silent_under_warning_level(self):
        """Fix E: all three probes are DEBUG-gated — at WARNING or above,
        zero probe records are emitted (block, text-source, or message_id).
        """
        adapter = self._adapter()
        msg = {
            "chatMessage": {
                "chatId": "room-W",
                "messageId": "01HKCHF4FC4S0W7T3V74SG6AT6",
                "status": "STATUS_COMPLETED",
                "role": "ROLE_USER",
                "senderId": "u1",
                "senderShortId": "u_real",
                "text": "envelope preview",
                "summary": "envelope summary",
                "blocks": [
                    {
                        "content": {
                            "case": "resourceLink",
                            "value": {"uri": "kimi://file/z"},
                        }
                    }
                ],
            }
        }
        records, teardown = _capture_kimi_log_records(level=logging.WARNING)
        try:
            await adapter._on_group_event(msg)
        finally:
            teardown()

        probe_substrings = ("non-text block", "text source for", "message_id")
        fired = [
            r for r in records
            if any(s in r.getMessage() for s in probe_substrings)
        ]
        self.assertEqual(fired, [], f"expected zero probe logs at WARNING, got: {fired}")

    async def test_probe_msg_id_sample_rate_reduces_log_volume(self):
        """Fix F: ``probe_msg_id_sample_rate=3`` → exactly one probe-3 log
        every three inbound messages in the same room. Tracker still
        updates on every inbound regardless of sampling.
        """
        adapter = KimiAdapter(_cfg(
            group_allow_bot_senders="all",
            probe_msg_id_sample_rate=3,
        ))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False

        ids = [
            "01HKCHF4F14S0W7T3V74SG6AT6",
            "01HKCHF4F24S0W7T3V74SG6AT6",
            "01HKCHF4F34S0W7T3V74SG6AT6",
            "01HKCHF4F44S0W7T3V74SG6AT6",
            "01HKCHF4F54S0W7T3V74SG6AT6",
            "01HKCHF4F64S0W7T3V74SG6AT6",
            "01HKCHF4F74S0W7T3V74SG6AT6",
            "01HKCHF4F84S0W7T3V74SG6AT6",
            "01HKCHF4F94S0W7T3V74SG6AT6",
        ]

        records, teardown = _capture_kimi_log_records()
        try:
            for mid in ids:
                await adapter._on_group_event(self._msg("room-S", mid))
        finally:
            teardown()

        probe3 = [
            r for r in records
            if "message_id timing" in r.getMessage()
            or "message_id first-seen" in r.getMessage()
        ]
        # 9 dispatches / sample_rate=3 → exactly 3 probe-3 logs.
        self.assertEqual(
            len(probe3), 3,
            f"expected 3 probe-3 records at 1-in-3 sampling, got {len(probe3)}: "
            f"{[r.getMessage() for r in probe3]}",
        )
        # Tracker reflects the LAST observed id, not the last SAMPLED id.
        self.assertEqual(
            adapter._last_message_id_per_room.get("room-S"),
            ids[-1],
        )

    def test_probe_msg_id_sample_rate_falls_back_on_invalid_config(self):
        """Fix-up (Codex P2): a non-numeric ``probe_msg_id_sample_rate``
        in ``config.extra`` (e.g. operator typo ``"ten"``) must not crash
        adapter init. Falls back to 1 with a WARNING log naming the bad
        value.
        """
        records, teardown = _capture_kimi_log_records(level=logging.WARNING)
        try:
            adapter = KimiAdapter(_cfg(probe_msg_id_sample_rate="ten"))
        finally:
            teardown()

        self.assertEqual(adapter._probe_msg_id_sample_rate, 1)
        warnings = [
            r for r in records
            if r.levelno == logging.WARNING
            and "probe_msg_id_sample_rate" in r.getMessage()
        ]
        self.assertEqual(
            len(warnings), 1,
            f"expected one WARNING naming the bad value, got: "
            f"{[r.getMessage() for r in records]}",
        )
        self.assertIn("'ten'", warnings[0].getMessage())



# ═══════════════════════════════════════════════════════════════════════════════
# Lift 3a: interrupt-and-drain queue improvements (pending-slot drop-log + TTL)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_message_event(text: str = "hello", chat_id: str = "dm:im:kimi:main") -> "MagicMock":
    """Build a minimal MessageEvent-like object for testing handle_message."""
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from gateway.config import Platform

    source = SessionSource(
        platform=Platform("kimiclaw"),
        chat_id=chat_id,
        chat_type="dm",
        user_id="kimi:user:1",
    )
    event = MagicMock(spec=MessageEvent)
    event.source = source
    event.text = text
    event.message_type = MessageType.TEXT
    event.message_id = "msg-test"
    return event


def _compute_session_key(adapter: "KimiAdapter", event: "MagicMock") -> str:
    """Compute the session key the adapter will derive for a given event."""
    from gateway.session import build_session_key
    return build_session_key(
        event.source,
        group_sessions_per_user=adapter._group_sessions_per_user,
        thread_sessions_per_user=adapter._thread_sessions_per_user,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Issue #18 Phase 0 instrumentation: gap-candidate INFO + reconnect counter +
# ListMessages pagination.
#
# This is the EVIDENCE-GATHERING layer for the future burst-drop recovery work
# (Phase 1+). It does not recover anything; it only surfaces signal that
# operators can use to validate which design wins. Three independent pieces:
#
#   1. ``BurstDropGapLogTests`` — promote suspiciously-large per-room
#      message_id timestamp deltas to INFO so they show up in journalctl
#      regardless of DEBUG state.
#   2. ``GroupSubscribeReconnectCounterTests`` — counter + snapshot log on
#      the first chatMessage post-connect, the candidate Phase 1 hook point.
#   3. ``ListGroupMessagesPaginationTests`` — wrapper now follows
#      ``nextPageToken`` up to ``max_pages``. Required for any recovery
#      design that fetches more than ``limit`` messages from a gap.
# ═══════════════════════════════════════════════════════════════════════════════

class BurstDropGapLogTests(unittest.IsolatedAsyncioTestCase):
    """Phase 0 (#18): suspicious wall-clock gaps logged at INFO.

    Tests use ``time.monotonic`` patching to simulate per-room arrival
    deltas deterministically. Cumulative review #58 switched gap-delta
    tracking from ``time.time`` to ``time.monotonic`` (independent flags
    from Codex and Kimi reviewers — wall-clock is unsafe for delta
    computation under NTP step / leap-second / VM suspend-resume). Gap
    detection is fundamentally about elapsed time between receives, which
    monotonic answers directly without depending on wall-clock stability.

    The message-ids in fixtures are UUID v8 to match Kimi's actual
    production format — see ``test_ulid_time_ms_returns_none_for_uuid_v8_
    production_format`` for the regression guard around id-derived timing.
    """

    def _adapter(self, **cfg):
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all", **cfg))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False
        return adapter

    @staticmethod
    def _msg(chat_id: str, message_id: str) -> "Dict[str, Any]":
        return {
            "chatMessage": {
                "chatId": chat_id,
                "messageId": message_id,
                "status": "STATUS_COMPLETED",
                "role": "ROLE_USER",
                "senderId": "u1",
                "senderShortId": "u_real",
                "blocks": [
                    {"content": {"case": "text", "value": {"content": "hi"}}}
                ],
            }
        }

    async def test_gap_below_threshold_no_info_log(self):
        """Delta < threshold (in seconds) → no INFO log."""
        adapter = self._adapter(burst_drop_gap_log_threshold_s=60.0)
        records, teardown = _capture_kimi_log_records(level=logging.INFO)
        # ``time.time`` is called multiple times per _on_group_event (gap
        # tracking + base-class hooks), so use return_value + manual swap
        # rather than a fixed-length side_effect.
        with patch("kimi_adapter.time.monotonic") as mock_time:
            mock_time.return_value = 1000.0
            try:
                await adapter._on_group_event(
                    self._msg("room-X", "19dc9c4d-93c2-87b8-8000-0a4a0ecd76fe")
                )
                mock_time.return_value = 1001.0
                await adapter._on_group_event(
                    self._msg("room-X", "19dc9c4f-1262-8c1b-8000-0a4a6c626bbb")
                )
            finally:
                teardown()
        gap = [r for r in records if "gap candidate" in r.getMessage()]
        self.assertEqual(gap, [], "small gap should not log gap candidate at INFO")

    async def test_gap_above_threshold_emits_info_log(self):
        """Delta >= threshold (seconds) → INFO log fires with delta_s, prev_id,
        connect#, since_reconnect_s. Production UUID-shaped ids accepted —
        the gap signal is wall-clock-driven so id format is irrelevant."""
        adapter = self._adapter(burst_drop_gap_log_threshold_s=10.0)
        records, teardown = _capture_kimi_log_records(level=logging.INFO)
        with patch("kimi_adapter.time.monotonic") as mock_time:
            mock_time.return_value = 2000.0
            try:
                await adapter._on_group_event(
                    self._msg("room-Y", "19dc9c4d-93c2-87b8-8000-0a4a0ecd76fe")
                )
                mock_time.return_value = 2045.0  # 45s delta, > 10s threshold
                await adapter._on_group_event(
                    self._msg("room-Y", "19dc9c4f-1262-8c1b-8000-0a4a6c626bbb")
                )
            finally:
                teardown()
        gap = [r for r in records if "gap candidate" in r.getMessage()]
        self.assertEqual(len(gap), 1, f"expected one gap-candidate INFO, got {gap}")
        rendered = gap[0].getMessage()
        self.assertIn("room=room-Y", rendered)
        self.assertIn("prev=19dc9c4d-93c2-87b8-8000-0a4a0ecd76fe", rendered)
        self.assertIn("delta_s=45.0", rendered)
        self.assertIn(">=10.0s threshold", rendered)
        self.assertIn("since_reconnect_s=N/A", rendered)
        self.assertIn("connect#=0", rendered)
        self.assertEqual(gap[0].levelno, logging.INFO)

    async def test_gap_log_includes_since_reconnect_correlation(self):
        """When reconnect timestamp is set, the log includes since_reconnect_s
        for in-line correlation (Codex review #2 — operators shouldn't have to
        grep two log streams to correlate)."""
        adapter = self._adapter(burst_drop_gap_log_threshold_s=10.0)
        adapter._group_subscribe_reconnect_count = 5
        adapter._group_subscribe_last_reconnect_ts = 3000.0
        records, teardown = _capture_kimi_log_records(level=logging.INFO)
        with patch("kimi_adapter.time.monotonic") as mock_time:
            mock_time.return_value = 3050.0
            try:
                await adapter._on_group_event(
                    self._msg("room-Z", "19dc9c4d-93c2-87b8-8000-0a4a0ecd76fe")
                )
                mock_time.return_value = 3120.0  # 70s gap, 120s after reconnect
                await adapter._on_group_event(
                    self._msg("room-Z", "19dc9c4f-1262-8c1b-8000-0a4a6c626bbb")
                )
            finally:
                teardown()
        gap = [r for r in records if "gap candidate" in r.getMessage()]
        self.assertEqual(len(gap), 1)
        rendered = gap[0].getMessage()
        self.assertIn("since_reconnect_s=120.0", rendered)
        self.assertIn("connect#=5", rendered)

    async def test_first_message_in_room_no_gap_log(self):
        """First message has no prior arrival anchor; can't compute delta — no log."""
        adapter = self._adapter(burst_drop_gap_log_threshold_s=0.001)
        records, teardown = _capture_kimi_log_records(level=logging.INFO)
        with patch("kimi_adapter.time.monotonic", return_value=4000.0):
            try:
                await adapter._on_group_event(
                    self._msg("room-W", "19dc9c4f-1262-8c1b-8000-0a4a6c626bbb")
                )
            finally:
                teardown()
        gap = [r for r in records if "gap candidate" in r.getMessage()]
        self.assertEqual(gap, [])

    async def test_threshold_zero_disables_info_log(self):
        """Threshold 0 disables the INFO log entirely."""
        adapter = self._adapter(burst_drop_gap_log_threshold_s=0)
        records, teardown = _capture_kimi_log_records(level=logging.INFO)
        with patch("kimi_adapter.time.monotonic") as mock_time:
            mock_time.return_value = 5000.0
            try:
                await adapter._on_group_event(
                    self._msg("room-V", "19dc9c4d-93c2-87b8-8000-0a4a0ecd76fe")
                )
                mock_time.return_value = 5999.0
                await adapter._on_group_event(
                    self._msg("room-V", "19dc9c4f-1262-8c1b-8000-0a4a6c626bbb")
                )
            finally:
                teardown()
        gap = [r for r in records if "gap candidate" in r.getMessage()]
        self.assertEqual(gap, [], "threshold=0 must suppress the INFO log")

    async def test_invalid_threshold_falls_back_to_default(self):
        """Garbage in config doesn't crash startup; default 30.0 applies."""
        adapter = self._adapter(burst_drop_gap_log_threshold_s="not-a-number")
        self.assertEqual(adapter._burst_drop_gap_log_threshold_s, 30.0)

    async def test_negative_threshold_clamped_to_zero(self):
        """Negative values clamp at 0 (which disables the INFO path)."""
        adapter = self._adapter(burst_drop_gap_log_threshold_s=-5.0)
        self.assertEqual(adapter._burst_drop_gap_log_threshold_s, 0.0)

    async def test_monotonic_clock_immune_to_wall_clock_jump(self):
        """Cumulative review #58: gap-delta uses ``time.monotonic`` — patching
        ``time.time`` to simulate an NTP forward jump must NOT trigger a
        spurious gap log if monotonic time hasn't advanced past the threshold.

        This is the regression guard for the wall-clock-vs-monotonic pivot.
        Before the fix, a Pi resuming from suspend (wall-clock catch-up) or
        an NTP step would emit false-positive gap candidates; now only
        true elapsed process-time matters.
        """
        adapter = self._adapter(burst_drop_gap_log_threshold_s=30.0)
        records, teardown = _capture_kimi_log_records(level=logging.INFO)
        try:
            with patch("kimi_adapter.time.monotonic") as mock_mono:
                # Simulate 1 second elapsed in monotonic time, but...
                mock_mono.return_value = 100.0
                await adapter._on_group_event(
                    self._msg("room-Q", "19dc9c4d-93c2-87b8-8000-0a4a0ecd76fe")
                )
                # ...wall-clock would have jumped 60s (NTP / suspend resume).
                # We only patch monotonic — wall-clock is irrelevant to the
                # delta calc now that #58 fixed the timestamp source. If we
                # ever regress to time.time() this test would fail because
                # the test wouldn't be controlling the relevant clock.
                mock_mono.return_value = 101.0  # 1s elapsed monotonic
                await adapter._on_group_event(
                    self._msg("room-Q", "19dc9c4f-1262-8c1b-8000-0a4a6c626bbb")
                )
        finally:
            teardown()
        gap = [r for r in records if "gap candidate" in r.getMessage()]
        self.assertEqual(
            gap, [],
            "1s monotonic elapsed must not trigger 30s threshold even if "
            "wall-clock jumped — proves time.monotonic is the timestamp source",
        )


class GroupSubscribeReconnectCounterTests(unittest.TestCase):
    """Phase 0 (#18): reconnect counter starts at 0 and is exposed on adapter.

    Cumulative review #58 (Claude code-reviewer + Codex challenge) flagged
    an off-by-one in ``_group_subscribe_once``: the counter was incremented
    AFTER ``_on_group_event`` was awaited, so the gap log inside dispatch
    saw the prior cycle's count while the ``subscribe stream live`` log
    emitted moments later carried the new count — defeating in-line
    correlation. Fix: bump pre-dispatch. The end-to-end integration test
    that drives ``_group_subscribe_once`` with mocked WS layers is tracked
    as task #61; the unit invariant below documents the post-bump
    behaviour any regression would fail.
    """

    def test_counter_initialised_to_zero(self):
        adapter = KimiAdapter(_cfg())
        self.assertEqual(adapter._group_subscribe_reconnect_count, 0)
        # Sentinel for "no reconnect observed yet" — used by gap-log
        # since_reconnect_s correlation. Monotonic time is always >= 0,
        # so -1 is safely outside the legitimate value range.
        self.assertEqual(adapter._group_subscribe_last_reconnect_ts, -1.0)

    def test_counter_attribute_exists_for_log_format_consumers(self):
        """Documents that the counter is a public-shape attribute used in the
        ``Kimi groups: subscribe stream live`` INFO log. If anything renames
        or removes this attribute, the log line breaks — operators correlating
        gap-candidate INFOs against this would lose their anchor."""
        adapter = KimiAdapter(_cfg())
        self.assertIsInstance(adapter._group_subscribe_reconnect_count, int)


class ConnectCounterOrderingInvariantTests(unittest.IsolatedAsyncioTestCase):
    """Cumulative review #58: documents the post-bump invariant the
    ``_group_subscribe_once`` fix must preserve.

    The actual subscribe-loop integration test is deferred to #61 (heavy
    WS mocks). This class instead asserts: GIVEN the counter is bumped
    pre-dispatch (as the new code does), THEN ``_on_group_event``'s gap
    log uses the bumped value. A regression that moves the bump back to
    post-dispatch would still pass these tests; it would fail the deferred
    integration test under #61, which is why the sequencing is documented
    in the class docstring rather than spread across files.
    """

    def _adapter(self, **cfg):
        adapter = KimiAdapter(_cfg(group_allow_bot_senders="all", **cfg))
        adapter.handle_message = AsyncMock()  # type: ignore
        adapter._hydrate_missing_text = False
        return adapter

    @staticmethod
    def _msg(chat_id: str, message_id: str):
        return {
            "chatMessage": {
                "chatId": chat_id,
                "messageId": message_id,
                "status": "STATUS_COMPLETED",
                "role": "ROLE_USER",
                "senderId": "u1",
                "senderShortId": "u_real",
                "blocks": [{"content": {"case": "text", "value": {"content": "hi"}}}],
            }
        }

    async def test_gap_log_reads_post_bump_counter(self):
        """If the counter is bumped before dispatch (mirroring the new
        ``_group_subscribe_once`` ordering), the gap-candidate log inside
        ``_on_group_event`` sees the bumped value. With the OLD ordering,
        the log would carry connect#=0 while the ``subscribe stream live``
        log emitted seconds later would carry connect#=1 for the same
        first-frame event."""
        adapter = self._adapter(burst_drop_gap_log_threshold_s=10.0)
        # Simulate the new pre-dispatch bump exactly as
        # _group_subscribe_once now does it.
        adapter._group_subscribe_frame_since_connect = True
        adapter._group_subscribe_reconnect_count += 1  # 0 -> 1
        adapter._group_subscribe_last_reconnect_ts = 100.0  # monotonic

        records, teardown = _capture_kimi_log_records(level=logging.INFO)
        try:
            with patch("kimi_adapter.time.monotonic") as mock_mono:
                # Two arrivals 50s apart on monotonic clock → trips the
                # 10s threshold and emits the gap-candidate INFO log.
                mock_mono.return_value = 100.0
                await adapter._on_group_event(
                    self._msg("room-K", "19dc9c4d-93c2-87b8-8000-0a4a0ecd76fe")
                )
                mock_mono.return_value = 150.0
                await adapter._on_group_event(
                    self._msg("room-K", "19dc9c4f-1262-8c1b-8000-0a4a6c626bbb")
                )
        finally:
            teardown()

        gap = [r for r in records if "gap candidate" in r.getMessage()]
        self.assertEqual(len(gap), 1)
        rendered = gap[0].getMessage()
        # Critical assertion: gap log saw the BUMPED counter, not 0.
        self.assertIn(
            "connect#=1", rendered,
            "Gap log must reflect the post-bump counter — if this asserts "
            "connect#=0 instead, the off-by-one regressed (#58)",
        )
        # since_reconnect_s = 50 (150 - 100) confirms the monotonic ts is
        # also being read post-bump.
        self.assertIn("since_reconnect_s=50.0", rendered)


class ListGroupMessagesPaginationTests(unittest.IsolatedAsyncioTestCase):
    """Phase 0 (#18): list_group_messages now follows nextPageToken up to
    max_pages, with backward-compat default of single-page (max_pages=1).
    """

    def _adapter(self):
        adapter = KimiAdapter(_cfg())
        adapter._rpc_unary = AsyncMock()  # type: ignore
        return adapter

    async def test_default_single_page_backward_compat(self):
        """Default ``max_pages=1`` — single RPC call, ``pageToken`` empty
        in the request, ``nextPageToken`` in response is ignored."""
        adapter = self._adapter()
        adapter._rpc_unary.return_value = {
            "messages": [{"messageId": "m1"}, {"messageId": "m2"}],
            "nextPageToken": "page2",  # ignored in single-page mode
        }
        result = await adapter.list_group_messages("room-1", limit=20)
        self.assertEqual(len(result), 2)
        self.assertEqual(adapter._rpc_unary.await_count, 1)
        first_call_body = adapter._rpc_unary.await_args[0][1]
        # First page omits pageToken entirely (cumulative review #58 fix —
        # was previously sent as empty string; mirrors list_group_files).
        self.assertNotIn("pageToken", first_call_body)

    async def test_multi_page_follows_token(self):
        """``max_pages>1`` follows ``nextPageToken`` until empty."""
        adapter = self._adapter()
        adapter._rpc_unary.side_effect = [
            {"messages": [{"messageId": "m1"}], "nextPageToken": "tok2"},
            {"messages": [{"messageId": "m2"}], "nextPageToken": "tok3"},
            {"messages": [{"messageId": "m3"}], "nextPageToken": ""},
        ]
        result = await adapter.list_group_messages("room-1", max_pages=5)
        self.assertEqual(len(result), 3)
        self.assertEqual(adapter._rpc_unary.await_count, 3)
        # Second call should have sent the first page's token.
        second_body = adapter._rpc_unary.await_args_list[1][0][1]
        self.assertEqual(second_body["pageToken"], "tok2")
        third_body = adapter._rpc_unary.await_args_list[2][0][1]
        self.assertEqual(third_body["pageToken"], "tok3")

    async def test_max_pages_caps_runaway(self):
        """If Kimi keeps returning a token, ``max_pages`` stops the loop."""
        adapter = self._adapter()
        adapter._rpc_unary.return_value = {
            "messages": [{"messageId": "mX"}],
            "nextPageToken": "always-more",
        }
        result = await adapter.list_group_messages("room-1", max_pages=3)
        self.assertEqual(adapter._rpc_unary.await_count, 3)
        self.assertEqual(len(result), 3)

    async def test_empty_pagetoken_breaks_loop_early(self):
        """First-page empty ``nextPageToken`` → stop after one call even with
        ``max_pages=10``."""
        adapter = self._adapter()
        adapter._rpc_unary.return_value = {
            "messages": [{"messageId": "only"}],
            # No nextPageToken at all.
        }
        result = await adapter.list_group_messages("room-1", max_pages=10)
        self.assertEqual(adapter._rpc_unary.await_count, 1)
        self.assertEqual(len(result), 1)

    async def test_max_pages_below_one_raises(self):
        """``max_pages=0`` is a config bug — surface it as ValueError."""
        adapter = self._adapter()
        with self.assertRaises(ValueError):
            await adapter.list_group_messages("room-1", max_pages=0)

    async def test_anchor_only_on_first_page(self):
        """Cumulative review #58 (Kimi review): anchor IDs (start_message_id /
        end_message_id) must be sent ONLY on the first request. Once
        ``pageToken`` is in play, the server-issued cursor encodes
        everything; echoing anchors alongside the cursor risks duplicate
        results or undefined ordering. Mirrors the conditional pageToken
        injection in ``list_group_files``.
        """
        adapter = self._adapter()
        adapter._rpc_unary.side_effect = [
            {"messages": [{"messageId": "m1"}], "nextPageToken": "tok2"},
            {"messages": [{"messageId": "m2"}], "nextPageToken": "tok3"},
            {"messages": [{"messageId": "m3"}], "nextPageToken": ""},
        ]
        await adapter.list_group_messages(
            "room-1",
            max_pages=5,
            start_message_id="anchor-start",
            end_message_id="anchor-end",
        )
        # First call: anchors present, no pageToken.
        first_body = adapter._rpc_unary.await_args_list[0][0][1]
        self.assertEqual(first_body.get("startMessageId"), "anchor-start")
        self.assertEqual(first_body.get("endMessageId"), "anchor-end")
        self.assertNotIn("pageToken", first_body)
        # Subsequent calls: pageToken present, anchors absent.
        for idx in (1, 2):
            body = adapter._rpc_unary.await_args_list[idx][0][1]
            self.assertNotIn(
                "startMessageId", body,
                f"page {idx} must not echo startMessageId alongside pageToken",
            )
            self.assertNotIn(
                "endMessageId", body,
                f"page {idx} must not echo endMessageId alongside pageToken",
            )
            self.assertTrue(body.get("pageToken"), "page 2+ must carry cursor")


class FetchGroupMessagePaginationTests(unittest.IsolatedAsyncioTestCase):
    """#60 fold-in: ``_fetch_group_message`` raises max_pages=2 so a tight
    ``start=end=message_id`` window that Kimi paginates around is still
    found rather than silently returning None. Safe given the
    anchor-only-on-first-page fix above (cumulative review #58)."""

    def _adapter(self):
        adapter = KimiAdapter(_cfg())
        adapter._rpc_unary = AsyncMock()  # type: ignore
        return adapter

    async def test_fetch_passes_max_pages_2_to_list_group_messages(self):
        """``_fetch_group_message`` must call ``list_group_messages`` with
        ``max_pages=2`` so a single follow-up page is permitted."""
        adapter = self._adapter()
        adapter.list_group_messages = AsyncMock(return_value=[])  # type: ignore
        await adapter._fetch_group_message("room-1", "msg-target")
        adapter.list_group_messages.assert_awaited_once()
        kwargs = adapter.list_group_messages.await_args.kwargs
        self.assertEqual(kwargs.get("max_pages"), 2)
        self.assertEqual(kwargs.get("start_message_id"), "msg-target")
        self.assertEqual(kwargs.get("end_message_id"), "msg-target")

    async def test_fetch_finds_message_on_second_page(self):
        """If Kimi paginates the tight-range window such that the target
        appears on page 2, ``_fetch_group_message`` returns it (not None)."""
        adapter = self._adapter()
        # Page 1: unrelated message + nextPageToken.
        # Page 2: the target.
        adapter._rpc_unary.side_effect = [
            {
                "messages": [{
                    "messageId": "msg-other",
                    "message": {"id": "msg-other", "blocks": []},
                }],
                "nextPageToken": "tok-page2",
            },
            {
                "messages": [{
                    "messageId": "msg-target",
                    "senderId": "u1",
                    "message": {"id": "msg-target", "blocks": []},
                }],
                "nextPageToken": "",
            },
        ]
        result = await adapter._fetch_group_message("room-1", "msg-target")
        self.assertIsNotNone(result)
        # Two RPC calls confirm the second-page traversal.
        self.assertEqual(adapter._rpc_unary.await_count, 2)


class HakimiLift3aDropLogTests(unittest.IsolatedAsyncioTestCase):
    """Lift 3a: WARN log when a pending-slot overwrite occurs."""

    async def test_3a_1_drop_log_on_overwrite(self):
        """3a.1 — overwriting an existing pending slot emits a WARNING with
        chat_id and message preview."""
        adapter = KimiAdapter(_cfg())

        # Simulate a session in progress (active-session guard set).
        first_event = _make_message_event("first pending message")
        session_key = _compute_session_key(adapter, first_event)
        guard = asyncio.Event()
        adapter._active_sessions[session_key] = guard

        # Put a first pending message in the slot.
        adapter._pending_messages[session_key] = first_event
        adapter._pending_enqueued_at[session_key] = 1.0  # arbitrary

        # Now call handle_message with a SECOND message — this should overwrite.
        second_event = _make_message_event("second message overwrites first")
        # Patch super().handle_message to avoid real dispatch.
        records, teardown = _capture_kimi_log_records(level=logging.WARNING)
        try:
            with patch.object(
                adapter.__class__.__bases__[0], "handle_message", new=AsyncMock()
            ):
                await adapter.handle_message(second_event)
        finally:
            teardown()

        warnings = [
            r for r in records
            if r.levelno == logging.WARNING and "overwriting pending slot" in r.getMessage()
        ]
        self.assertEqual(len(warnings), 1, f"Expected one overwrite WARNING, got: {[r.getMessage() for r in records]}")
        msg = warnings[0].getMessage()
        self.assertIn("first pending", msg)  # preview of the dropped message

    async def test_3a_2_error_path_drain_preserved(self):
        """3a.2 — when message_handler raises, the pending slot is still drained.

        BasePlatformAdapter already handles this via the late-arrival drain in
        _process_message_background's ``finally`` block. This test documents and
        exercises the contract: after a handler exception, a message queued in
        _pending_messages is NOT silently lost.

        Implementation note: we test the base-class invariant via the KimiAdapter
        since KimiAdapter.handle_message delegates to super().handle_message which
        runs _process_message_background. The test checks that after an exception
        during handler execution, _pending_messages is cleared (either consumed or
        cleaned up).
        """
        adapter = KimiAdapter(_cfg())

        first_event = _make_message_event("first message that will error")
        session_key = _compute_session_key(adapter, first_event)

        # Set up a message handler that raises on the first call, succeeds on second.
        call_count = [0]

        async def _handler(event):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("synthetic processing error")
            return "ok"

        adapter.set_message_handler(_handler)

        second_event = _make_message_event("second message — must not be lost")

        # Queue second event in pending slot before first processes.
        adapter._pending_messages[session_key] = second_event
        adapter._pending_enqueued_at[session_key] = 0.0  # pre-enqueued

        # Now simulate base._process_message_background having set the active guard.
        guard = asyncio.Event()
        guard.set()  # interrupt already signalled
        adapter._active_sessions[session_key] = guard
        adapter._session_tasks[session_key] = asyncio.current_task()

        # Run _process_message_background which includes our exception + drain path.
        try:
            await adapter._process_message_background(first_event, session_key)
        except Exception:
            pass  # exception propagation details not under test

        # After the whole run, _pending_messages for this session should be gone —
        # the pending message was either dispatched (good) or cleaned up (acceptable).
        # The key invariant: it is NOT still sitting unprocessed in the dict
        # while the session is no longer active.
        session_still_active = session_key in adapter._active_sessions
        pending_still_queued = session_key in adapter._pending_messages
        self.assertFalse(
            session_still_active and pending_still_queued,
            "Pending message stranded: session is inactive but pending slot not cleared.",
        )

    async def test_3a_3_ttl_disabled_by_default(self):
        """3a.3 — with no TTL configured, pending slot never expires."""
        adapter = KimiAdapter(_cfg())  # no pending_message_ttl_seconds

        self.assertIsNone(adapter._pending_message_ttl)

        # Simulate an arbitrarily old pending message.
        old_event = _make_message_event("old pending message")
        session_key = _compute_session_key(adapter, old_event)
        guard = asyncio.Event()
        adapter._active_sessions[session_key] = guard

        adapter._pending_messages[session_key] = old_event
        # Enqueued a long time ago (1000 seconds).
        import time as _time
        adapter._pending_enqueued_at[session_key] = _time.monotonic() - 1000.0

        new_event = _make_message_event("newer message")
        records, teardown = _capture_kimi_log_records(level=logging.INFO)
        try:
            with patch.object(
                adapter.__class__.__bases__[0], "handle_message", new=AsyncMock()
            ):
                await adapter.handle_message(new_event)
        finally:
            teardown()

        # No eviction log should appear.
        eviction_logs = [
            r for r in records
            if "evicting expired" in r.getMessage()
        ]
        self.assertEqual(len(eviction_logs), 0, "TTL eviction should not fire when TTL is None")

    async def test_3a_4_ttl_enabled_evicts_expired_pending(self):
        """3a.4 — with TTL set, an expired pending slot is evicted and logged."""
        import time as _time
        adapter = KimiAdapter(_cfg(pending_message_ttl_seconds=5))

        self.assertEqual(adapter._pending_message_ttl, 5.0)

        old_event = _make_message_event("expired pending message")
        session_key = _compute_session_key(adapter, old_event)
        guard = asyncio.Event()
        adapter._active_sessions[session_key] = guard

        adapter._pending_messages[session_key] = old_event
        # Enqueued well past the 5s TTL.
        adapter._pending_enqueued_at[session_key] = _time.monotonic() - 60.0

        new_event = _make_message_event("fresh message after expiry")
        records, teardown = _capture_kimi_log_records(level=logging.INFO)
        try:
            with patch.object(
                adapter.__class__.__bases__[0], "handle_message", new=AsyncMock()
            ):
                await adapter.handle_message(new_event)
        finally:
            teardown()

        eviction_logs = [
            r for r in records
            if "evicting expired" in r.getMessage()
        ]
        self.assertEqual(
            len(eviction_logs), 1,
            f"Expected one eviction INFO log, got: {[r.getMessage() for r in records]}",
        )
        # After eviction of the old slot the new message is NOT double-logged as
        # an "overwriting" drop (the slot was cleared before the drop-log check).
        overwrite_warnings = [
            r for r in records
            if "overwriting pending slot" in r.getMessage()
        ]
        self.assertEqual(len(overwrite_warnings), 0, "Eviction should not also fire a drop warning")


# ═══════════════════════════════════════════════════════════════════════════════
# Issue #33: _pending_enqueued_at cleanup across teardown paths
#
# The Kimi adapter maintains `_pending_enqueued_at` as a parallel TTL dict
# alongside the base class's `_pending_messages` and `_active_sessions`. The
# base clears the latter two during `cancel_background_tasks`; without parallel
# clears in our subclass, the TTL dict leaks across reconnects (the gateway
# reuses the adapter instance). The fix layers three guarantees: (1) the
# `cancel_background_tasks` override mirrors the base's clear; (2) `disconnect`
# also clears for direct-disconnect paths that bypass cancel_background_tasks
# (gateway/run.py:_safe_adapter_disconnect, error-recovery in connect()); (3)
# `handle_message`'s post-super cleanup is wrapped in `try/finally` so any
# unexpected exception from super doesn't leak a stamped timestamp.
# ═══════════════════════════════════════════════════════════════════════════════

class PendingEnqueuedAtCleanupTests(unittest.IsolatedAsyncioTestCase):
    """Issue #33: _pending_enqueued_at must be cleared across teardown paths."""

    async def test_cancel_background_tasks_clears_pending_enqueued_at(self):
        """cancel_background_tasks override must mirror the base's clear()
        behaviour for our parallel TTL state."""
        adapter = KimiAdapter(_cfg())
        adapter._pending_enqueued_at["dm:test:abc"] = 1.0
        adapter._pending_enqueued_at["room:xyz"] = 2.0
        # super().cancel_background_tasks() walks _background_tasks (empty
        # on a fresh adapter) and clears the base's parallel dicts; our
        # override should clear _pending_enqueued_at on top of that.
        await adapter.cancel_background_tasks()
        self.assertEqual(
            adapter._pending_enqueued_at, {},
            "cancel_background_tasks should clear _pending_enqueued_at"
        )

    async def test_disconnect_clears_pending_enqueued_at(self):
        """disconnect() must clear _pending_enqueued_at as defense-in-depth
        for direct-disconnect call sites that bypass cancel_background_tasks."""
        adapter = KimiAdapter(_cfg())
        adapter._pending_enqueued_at["dm:test:abc"] = 1.0
        adapter._pending_enqueued_at["room:xyz"] = 2.0
        # Patch out network/lock teardown — the test only cares about the
        # parallel-state clear; the rest is unrelated infrastructure.
        with patch.object(adapter, "_cleanup_http", new=AsyncMock()), \
             patch.object(adapter, "_release_platform_lock"):
            await adapter.disconnect()
        self.assertEqual(
            adapter._pending_enqueued_at, {},
            "disconnect should clear _pending_enqueued_at"
        )

    async def test_connect_clears_pending_enqueued_at(self):
        """connect() must clear stale TTL state as a belt-and-braces sweep
        before establishing a new session — protects against any path that
        reuses the adapter without going through disconnect first."""
        adapter = KimiAdapter(_cfg())
        adapter._pending_enqueued_at["dm:stale:xyz"] = 99.0  # leftover from prior session

        # connect() reaches the clear() before any network IO. Make connect
        # short-circuit at GetMe so we don't have to mock the whole WS stack.
        from kimi_adapter import KimiAuthError
        with patch.object(adapter, "_acquire_platform_lock", return_value=True), \
             patch.object(adapter, "_rpc_unary", new=AsyncMock(side_effect=KimiAuthError("test"))), \
             patch.object(adapter, "_cleanup_http", new=AsyncMock()), \
             patch.object(adapter, "_release_platform_lock"):
            # Returns False because GetMe raises; the clear() ran before that.
            result = await adapter.connect()
            self.assertFalse(result, "connect() should fail when GetMe raises")
        # _cleanup_http was mocked out above so the aiohttp.ClientSession that
        # connect() opened at session start never gets closed by the SUT.
        # Close it here to keep the suite warning-clean.
        if adapter._http_session is not None and not adapter._http_session.closed:
            await adapter._http_session.close()
        self.assertEqual(
            adapter._pending_enqueued_at, {},
            "connect should clear stale TTL state at session start"
        )

    async def test_handle_message_cleanup_runs_on_cancellation(self):
        """try/finally ensures the post-super cleanup runs on CancelledError,
        so a per-task cancellation outside of full adapter teardown doesn't
        leak a stamped timestamp into the next handler invocation."""
        adapter = KimiAdapter(_cfg())
        event = _make_message_event("test message")
        session_key = _compute_session_key(adapter, event)

        # Simulate an active session so the override stamps _pending_enqueued_at.
        adapter._active_sessions[session_key] = asyncio.Event()

        # Patch super().handle_message to raise CancelledError mid-await, AFTER
        # the override stamped its timestamp. _pending_messages is left empty
        # (the mock does no enqueueing), so the cleanup guard's "if session_key
        # not in _pending_messages" branch should fire.
        with patch.object(
            adapter.__class__.__bases__[0],
            "handle_message",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await adapter.handle_message(event)

        self.assertNotIn(
            session_key,
            adapter._pending_enqueued_at,
            "try/finally should pop the timestamp on CancelledError when the "
            "pending slot was never populated",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Issue #22: room-state eviction policy (bounded LRU on per-room dicts)
#
# The adapter has three dicts keyed on room_id that grew without ceiling on
# long-running deployments: ``_rooms`` (cache), ``_last_message_id_per_room``
# (DEBUG observability anchor), ``_probe_msg_id_room_counts`` (DEBUG counter).
# All three are now backed by ``_BoundedLRU`` with a configurable cap. None of
# them hold replay-dedup correctness state — that's ``_processed_set``, which
# is bounded by ``_DEDUP_MAXLEN`` independently. Eviction is silent because
# every consumer of these dicts handles a missing entry transparently
# (re-fetch / first-seen path).
# ═══════════════════════════════════════════════════════════════════════════════

class BoundedLRUTests(unittest.TestCase):
    """Stand-alone behaviour of the _BoundedLRU primitive."""

    def test_grows_up_to_cap(self):
        """Inserts up to ``maxsize`` are kept; nothing evicted yet."""
        d = _BoundedLRU(maxsize=3)
        for i in range(3):
            d[f"k{i}"] = i
        self.assertEqual(len(d), 3)
        self.assertEqual(list(d.keys()), ["k0", "k1", "k2"])

    def test_evicts_oldest_when_over_cap(self):
        """N+1th insert pops the first-inserted key (FIFO from LRU end)."""
        d = _BoundedLRU(maxsize=3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        d["d"] = 4  # cap exceeded → evict "a"
        self.assertEqual(len(d), 3)
        self.assertNotIn("a", d)
        self.assertEqual(list(d.keys()), ["b", "c", "d"])

    def test_update_moves_key_to_recent_end(self):
        """Re-writing an existing key refreshes its LRU position so it's
        not next to be evicted. This is what protects ``_last_message_id_
        per_room`` for a busy room from being evicted just because the
        room was first inserted long ago."""
        d = _BoundedLRU(maxsize=3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        d["a"] = 99  # touch "a" — it should now be the freshest
        d["d"] = 4   # cap exceeded → evict "b" (oldest), NOT "a"
        self.assertEqual(d["a"], 99)
        self.assertNotIn("b", d)
        self.assertEqual(list(d.keys()), ["c", "a", "d"])

    def test_get_does_not_refresh_lru(self):
        """Reads must not move the key to the recent end. We want
        'least recently *updated*' semantics, not 'least recently
        accessed' — for ``_last_message_id_per_room`` the read happens
        every inbound message, so refreshing on read would defeat the
        cap entirely."""
        d = _BoundedLRU(maxsize=3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        _ = d.get("a")     # read — must NOT touch ordering
        _ = d["a"]         # subscript read — must NOT touch ordering
        d["e"] = 5         # cap exceeded → evict "a" (still oldest)
        self.assertNotIn("a", d)
        self.assertEqual(list(d.keys()), ["b", "c", "e"])

    def test_invalid_maxsize_rejected(self):
        """Cap of 0 or negative is a config bug, not a runtime no-op."""
        with self.assertRaises(ValueError):
            _BoundedLRU(maxsize=0)
        with self.assertRaises(ValueError):
            _BoundedLRU(maxsize=-1)


class RoomCacheCapIntegrationTests(unittest.TestCase):
    """Adapter-level: per-room dicts honour the cap, config knob plumbed."""

    def test_default_cap_applied(self):
        """No config override → all three dicts use _ROOM_CACHE_DEFAULT_MAX."""
        adapter = KimiAdapter(_cfg())
        self.assertIsInstance(adapter._rooms, _BoundedLRU)
        self.assertIsInstance(adapter._last_message_id_per_room, _BoundedLRU)
        self.assertIsInstance(adapter._probe_msg_id_room_counts, _BoundedLRU)
        self.assertEqual(adapter._rooms._maxsize, _ROOM_CACHE_DEFAULT_MAX)
        self.assertEqual(
            adapter._last_message_id_per_room._maxsize, _ROOM_CACHE_DEFAULT_MAX
        )
        self.assertEqual(
            adapter._probe_msg_id_room_counts._maxsize, _ROOM_CACHE_DEFAULT_MAX
        )

    def test_config_override_applied(self):
        """``room_cache_max_entries`` in config.extra propagates to all three."""
        adapter = KimiAdapter(_cfg(room_cache_max_entries=10))
        self.assertEqual(adapter._rooms._maxsize, 10)
        self.assertEqual(adapter._last_message_id_per_room._maxsize, 10)
        self.assertEqual(adapter._probe_msg_id_room_counts._maxsize, 10)

    def test_invalid_config_falls_back_to_default(self):
        """Garbage in config doesn't crash startup; default applies + warning."""
        adapter = KimiAdapter(_cfg(room_cache_max_entries="not-an-int"))
        self.assertEqual(adapter._rooms._maxsize, _ROOM_CACHE_DEFAULT_MAX)

    def test_negative_or_zero_clamped_to_one(self):
        """``max(1, int(...))`` floors any non-positive value at 1, matching
        the same defensive pattern used by ``probe_msg_id_sample_rate``."""
        adapter = KimiAdapter(_cfg(room_cache_max_entries=0))
        self.assertEqual(adapter._rooms._maxsize, 1)
        adapter2 = KimiAdapter(_cfg(room_cache_max_entries=-5))
        self.assertEqual(adapter2._rooms._maxsize, 1)

    def test_per_room_dict_evicts_under_pressure(self):
        """Insert past the cap; oldest room_id is dropped silently."""
        adapter = KimiAdapter(_cfg(room_cache_max_entries=3))
        adapter._last_message_id_per_room["room-1"] = "msg-1"
        adapter._last_message_id_per_room["room-2"] = "msg-2"
        adapter._last_message_id_per_room["room-3"] = "msg-3"
        adapter._last_message_id_per_room["room-4"] = "msg-4"  # evicts room-1
        self.assertEqual(len(adapter._last_message_id_per_room), 3)
        self.assertNotIn("room-1", adapter._last_message_id_per_room)
        self.assertEqual(adapter._last_message_id_per_room["room-4"], "msg-4")

    def test_busy_room_protected_from_eviction(self):
        """A room receiving frequent updates moves to the recent end on each
        write and is NOT evicted just because it was first inserted long ago.
        This is the load-bearing case for ``_last_message_id_per_room`` —
        a single chatty room should be the LAST to evict, not the first."""
        adapter = KimiAdapter(_cfg(room_cache_max_entries=3))
        # Insert busy + idle rooms.
        adapter._last_message_id_per_room["busy"] = "msg-1"
        adapter._last_message_id_per_room["idle-a"] = "msg-2"
        adapter._last_message_id_per_room["idle-b"] = "msg-3"
        # 100 more messages in "busy" — each write refreshes its LRU position.
        for i in range(100):
            adapter._last_message_id_per_room["busy"] = f"msg-{10+i}"
        # Now insert a 4th distinct room. Eviction should hit "idle-a"
        # (oldest among rooms that haven't been touched), NOT "busy".
        adapter._last_message_id_per_room["new-room"] = "msg-fresh"
        self.assertIn("busy", adapter._last_message_id_per_room)
        self.assertNotIn("idle-a", adapter._last_message_id_per_room)


class ArrivalTimeCacheCapTests(unittest.TestCase):
    """Cumulative review #58 (Codex lead): ``_last_arrival_time_per_room``
    is bounded INDEPENDENTLY of the shared ``room_cache_max_entries`` cap.

    Sharing the 500-entry cap with the other per-room dicts would silently
    blind the Phase 0 gap-candidate log under cardinality pressure: an
    evicted arrival-time entry produces ``prev_arrival=None`` on the next
    message, suppressing the INFO log regardless of actual delay. Codex
    framed it as a contradiction with the ``_BoundedLRU`` "no load-bearing
    state" promise — fixed here by giving arrival-time tracking its own
    ceiling that's effectively unbounded for any realistic deployment
    (10000 entries × ~16 bytes ≈ 160KB).
    """

    def test_default_arrival_cap_is_separate_and_larger(self):
        """No config override → arrival dict at 10000, room dict at 500."""
        adapter = KimiAdapter(_cfg())
        self.assertIsInstance(adapter._last_arrival_time_per_room, _BoundedLRU)
        self.assertEqual(
            adapter._last_arrival_time_per_room._maxsize,
            _ARRIVAL_TIME_CACHE_DEFAULT_MAX,
        )
        # Crucially: NOT the same as _ROOM_CACHE_DEFAULT_MAX.
        self.assertNotEqual(
            adapter._last_arrival_time_per_room._maxsize,
            adapter._rooms._maxsize,
            "arrival-time cap must be distinct from shared room cap — "
            "Codex review #58 lead objection",
        )
        self.assertGreater(
            adapter._last_arrival_time_per_room._maxsize,
            adapter._rooms._maxsize,
            "arrival-time cap must be larger so eviction is unreachable "
            "in any realistic deployment",
        )

    def test_arrival_cap_config_override_independent(self):
        """``arrival_time_cache_max_entries`` plumbs to the arrival dict
        without affecting ``room_cache_max_entries``."""
        adapter = KimiAdapter(_cfg(
            room_cache_max_entries=42,
            arrival_time_cache_max_entries=999,
        ))
        self.assertEqual(adapter._rooms._maxsize, 42)
        self.assertEqual(adapter._last_message_id_per_room._maxsize, 42)
        self.assertEqual(adapter._last_arrival_time_per_room._maxsize, 999)

    def test_arrival_cap_invalid_falls_back_to_default(self):
        """Garbage in config doesn't crash startup; default applies + warning."""
        adapter = KimiAdapter(_cfg(arrival_time_cache_max_entries="not-an-int"))
        self.assertEqual(
            adapter._last_arrival_time_per_room._maxsize,
            _ARRIVAL_TIME_CACHE_DEFAULT_MAX,
        )

    def test_arrival_cap_negative_or_zero_clamped_to_one(self):
        """Same defensive ``max(1, int(...))`` floor as the room cap."""
        a0 = KimiAdapter(_cfg(arrival_time_cache_max_entries=0))
        self.assertEqual(a0._last_arrival_time_per_room._maxsize, 1)
        a_neg = KimiAdapter(_cfg(arrival_time_cache_max_entries=-7))
        self.assertEqual(a_neg._last_arrival_time_per_room._maxsize, 1)

    def test_eviction_with_separate_caps_doesnt_cross_dicts(self):
        """Inserting past one dict's cap evicts only from that dict —
        proves the caps are truly independent, not aliased to a single
        shared cap. Regression guard against an accidental refactor that
        re-merges the caps."""
        adapter = KimiAdapter(_cfg(
            room_cache_max_entries=2,
            arrival_time_cache_max_entries=4,
        ))
        # Push past the ROOM cap (2) but stay under the ARRIVAL cap (4).
        for i in range(3):
            adapter._rooms[f"room-{i}"] = object()
            adapter._last_arrival_time_per_room[f"room-{i}"] = float(i)
        self.assertEqual(len(adapter._rooms), 2)  # evicted room-0
        self.assertEqual(len(adapter._last_arrival_time_per_room), 3)  # still has all
        self.assertIn("room-0", adapter._last_arrival_time_per_room,
                      "arrival-dict must NOT be evicted at the room-cap boundary")


# ═══════════════════════════════════════════════════════════════════════════════
# Lift 3b: output_mode flag
# ═══════════════════════════════════════════════════════════════════════════════

class HakimiLift3bOutputModeTests(unittest.IsolatedAsyncioTestCase):
    """Lift 3b: output_mode: passthrough | tool_only."""

    async def test_3b_1_passthrough_mode_delivers_prose(self):
        """3b.1 — passthrough (default) — send() routes through normally."""
        adapter = KimiAdapter(_cfg(output_mode="passthrough"))
        self.assertEqual(adapter._output_mode, "passthrough")

        # Patch the routing methods so we don't need a live WS/HTTP session.
        adapter._send_dm = AsyncMock(return_value=SendResult(success=True))
        adapter._send_group = AsyncMock(return_value=SendResult(success=True))

        result = await adapter.send(
            chat_id="dm:im:kimi:main", content="agent prose response"
        )
        self.assertTrue(result.success)
        adapter._send_dm.assert_awaited_once()

    async def test_3b_2_tool_only_mode_suppresses_prose(self):
        """3b.2 — tool_only — send() is gated, nothing reaches the platform."""
        adapter = KimiAdapter(_cfg(output_mode="tool_only"))
        self.assertEqual(adapter._output_mode, "tool_only")

        adapter._send_dm = AsyncMock(return_value=SendResult(success=True))
        adapter._send_group = AsyncMock(return_value=SendResult(success=True))

        result = await adapter.send(
            chat_id="dm:im:kimi:main", content="agent prose that should be suppressed"
        )
        # Returns success=True (no error) but nothing was sent.
        self.assertTrue(result.success)
        adapter._send_dm.assert_not_awaited()
        adapter._send_group.assert_not_awaited()

    async def test_3b_3_tool_only_mode_suppresses_all_send_targets(self):
        """3b.3 — tool_only suppresses prose to both DM and group targets."""
        adapter = KimiAdapter(_cfg(output_mode="tool_only"))

        adapter._send_dm = AsyncMock(return_value=SendResult(success=True))
        adapter._send_group = AsyncMock(return_value=SendResult(success=True))

        # DM target.
        result_dm = await adapter.send(
            chat_id="dm:im:kimi:main", content="thinking out loud"
        )
        # Group target.
        result_group = await adapter.send(
            chat_id="room:abc123", content="group prose also suppressed"
        )
        self.assertTrue(result_dm.success)
        self.assertTrue(result_group.success)
        adapter._send_dm.assert_not_awaited()
        adapter._send_group.assert_not_awaited()

    async def test_3b_4_tool_only_mode_still_closes_inflight_dm(self):
        """3b.4 — tool_only DM suppression still pops inflight + responds end_turn.

        Regression: without _close_dm_inflight in send()'s short-circuit, the
        DM JSON-RPC reply never fires and Kimi's UI spinner hangs until WS
        reconnect. The inflight deque also grows unbounded over WS lifetime.
        """
        adapter = KimiAdapter(_cfg(output_mode="tool_only"))
        adapter._dm_respond = AsyncMock()
        adapter._send_dm = AsyncMock(return_value=SendResult(success=True))

        # Simulate an in-flight prompt as _dm_handle_prompt would.
        sid = "user-xyz"
        adapter._dm_inflight.setdefault(sid, deque()).append(
            _DMInflight(kimi_sid=sid, req_id=42)
        )

        result = await adapter.send(
            chat_id=f"dm:{sid}", content="agent prose to suppress"
        )

        # tool_only still short-circuits the prose
        self.assertTrue(result.success)
        adapter._send_dm.assert_not_awaited()
        # …but the JSON-RPC end_turn reply fires so the UI spinner closes
        adapter._dm_respond.assert_awaited_once_with(42, {"stopReason": "end_turn"})
        # …and the inflight queue is fully drained (no leak)
        self.assertNotIn(sid, adapter._dm_inflight)

    async def test_3b_5_empty_content_still_closes_inflight_dm(self):
        """3b.5 — empty content send() short-circuit also closes inflight DM."""
        adapter = KimiAdapter(_cfg(output_mode="passthrough"))
        adapter._dm_respond = AsyncMock()

        sid = "user-empty"
        adapter._dm_inflight.setdefault(sid, deque()).append(
            _DMInflight(kimi_sid=sid, req_id=99)
        )

        result = await adapter.send(chat_id=f"dm:{sid}", content="")
        self.assertTrue(result.success)
        adapter._dm_respond.assert_awaited_once_with(99, {"stopReason": "end_turn"})
        self.assertNotIn(sid, adapter._dm_inflight)


class HakimiLift3bOutputModeInitTests(unittest.TestCase):
    """Init-level validation for output_mode."""

    def test_output_mode_default_is_passthrough(self):
        adapter = KimiAdapter(_cfg())
        self.assertEqual(adapter._output_mode, "passthrough")

    def test_output_mode_tool_only_accepted(self):
        adapter = KimiAdapter(_cfg(output_mode="tool_only"))
        self.assertEqual(adapter._output_mode, "tool_only")

    def test_output_mode_invalid_defaults_with_warning(self):
        records, teardown = _capture_kimi_log_records(level=logging.WARNING)
        try:
            adapter = KimiAdapter(_cfg(output_mode="robot_only"))
        finally:
            teardown()
        self.assertEqual(adapter._output_mode, "passthrough")
        warnings = [
            r for r in records
            if r.levelno == logging.WARNING and "output_mode" in r.getMessage()
        ]
        self.assertEqual(len(warnings), 1)
        self.assertIn("robot_only", warnings[0].getMessage())


class PluginRegistrationTests(unittest.TestCase):
    """Verify ``register()`` populates the expected ``PlatformEntry`` fields.

    Replaces the older Fork-only ``AuthorizationIntegrationTests`` which
    scraped ``gateway/run.py`` source for hardcoded ``Platform.KIMI`` map
    entries.  All of that wiring is now plugin-owned via the upstream
    ``ctx.register_platform()`` API — these tests assert the call's kwargs
    instead of poking the gateway's source code.
    """

    def test_register_calls_ctx_with_expected_fields(self):
        from kimi_adapter import register

        ctx = MagicMock()
        register(ctx)

        ctx.register_platform.assert_called_once()
        kwargs = ctx.register_platform.call_args.kwargs

        # Identity
        self.assertEqual(kwargs["name"], "kimiclaw")
        self.assertEqual(kwargs["label"], "KimiClaw")

        # Auth env-var mapping (replaces fork's gateway/run.py auth env-maps).
        self.assertEqual(kwargs["allowed_users_env"], "KIMI_ALLOWED_USERS")
        self.assertEqual(kwargs["allow_all_env"], "KIMI_ALLOW_ALL_USERS")

        # Cron delivery (replaces fork's cron/scheduler.py wiring).
        self.assertEqual(kwargs["cron_deliver_env_var"], "KIMI_HOME_CHANNEL")

        # Required env
        self.assertEqual(kwargs["required_env"], ["KIMI_BOT_TOKEN"])

        # Display flags
        self.assertEqual(kwargs["emoji"], "🌙")
        self.assertTrue(kwargs["pii_safe"])
        self.assertTrue(kwargs["allow_update_command"])

        # Callable hooks must all be present + callable.
        self.assertTrue(callable(kwargs["adapter_factory"]))
        self.assertTrue(callable(kwargs["check_fn"]))
        self.assertTrue(callable(kwargs["validate_config"]))
        self.assertTrue(callable(kwargs["is_connected"]))
        self.assertTrue(callable(kwargs["env_enablement_fn"]))
        self.assertTrue(callable(kwargs["apply_yaml_config_fn"]))
        self.assertTrue(callable(kwargs["standalone_sender_fn"]))
        self.assertTrue(callable(kwargs["setup_fn"]))

        # Platform hint mentions Kimi explicitly.
        self.assertIn("Kimi", kwargs["platform_hint"])
        self.assertIn("kimi.com", kwargs["platform_hint"])

        # Install hint references the upstream commit so operators can find it.
        self.assertIn("2e20f6ae2", kwargs["install_hint"])

        # Message-length guidance present for smart-chunking.
        self.assertGreater(kwargs["max_message_length"], 0)

    def test_register_factory_constructs_adapter(self):
        """The factory passed to ``register_platform`` actually constructs a KimiAdapter."""
        from kimi_adapter import register

        ctx = MagicMock()
        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        adapter = factory(_cfg())
        self.assertIsInstance(adapter, KimiAdapter)

    def test_register_check_fn_passes_in_dev_env(self):
        """``check_fn`` returns True when deps are importable AND
        ``KIMI_BOT_TOKEN`` is set. Mirrors the :class:`CheckForRegistryTests`
        assertion; this variant exercises the actual function reference the
        ``register()`` call hands to ``ctx.register_platform``.
        """
        from kimi_adapter import register

        ctx = MagicMock()
        register(ctx)
        check_fn = ctx.register_platform.call_args.kwargs["check_fn"]
        with patch.dict(os.environ, {"KIMI_BOT_TOKEN": "km_b_prod_TEST"}, clear=False):
            self.assertTrue(check_fn())

    def test_register_check_fn_fails_without_bot_token(self):
        """``check_fn`` returns False when ``KIMI_BOT_TOKEN`` is unset, even
        if deps are importable. Protects against KimiClaw auto-enabling on
        every install that has the ``[messaging]`` extra now that we declare
        ``websockets`` there.
        """
        from kimi_adapter import register

        ctx = MagicMock()
        register(ctx)
        check_fn = ctx.register_platform.call_args.kwargs["check_fn"]
        env_no_kimi = {k: v for k, v in os.environ.items()
                       if not k.startswith("KIMI_")}
        with patch.dict(os.environ, env_no_kimi, clear=True):
            self.assertFalse(check_fn())


if __name__ == "__main__":
    unittest.main()
