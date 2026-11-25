"""Regression tests for Phase 5 Security and Parser Hardening.

Demonstrates pre-fix failure states and post-fix verifications for:
- VULN-01: Unauthorized local file dispatch via LLM text or directives.
- VULN-02: Silent message content destruction in format_message and send.
- VULN-03: Regex Denial of Service (ReDoS) in tag attributes.
- VULN-04: Control marker injection via plain text and paragraph fallback.
"""

import asyncio
import importlib.util
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Optional

try:
    from plugins.platforms.photon.adapter import PhotonAdapter
    PHOTON_AVAILABLE = True
except Exception:  # health: allow BLE001 -- optional adapter import fallback
    PhotonAdapter = None
    PHOTON_AVAILABLE = False

from gateway.airhermes_config import AirHermesConfig
from gateway.bubble_parser import BubbleParser
from gateway.bubble_stream_consumer import BubbleStreamConsumer


class MockSendResult:
    def __init__(self, success: bool = True, error: Optional[str] = None) -> None:
        self.success = success
        self.error = error


class MockPhotonAdapter:
    """Mock PhotonAdapter recording all native and text dispatch calls."""

    def __init__(self, config: Optional[AirHermesConfig] = None) -> None:
        self.airhermes_config = config or AirHermesConfig()
        self.dispatched_texts: list[tuple[str, str]] = []
        self.dispatched_documents: list[tuple[str, str, Optional[str]]] = []
        self.dispatched_image_files: list[tuple[str, str, Optional[str]]] = []
        self.dispatched_images: list[tuple[str, str, Optional[str]]] = []
        self.dispatched_polls: list[tuple[str, str, list[str]]] = []
        self.dispatched_effects: list[tuple[str, str, str]] = []
        self.dispatched_reactions: list[tuple[str, str, Optional[str]]] = []

    async def send(self, chat_id: str, text: str, reply_to: Optional[str] = None, metadata: Optional[dict] = None) -> MockSendResult:
        self.dispatched_texts.append((chat_id, text))
        return MockSendResult(success=True)

    async def send_document(self, chat_id: str, file_path: str, file_name: Optional[str] = None, caption: Optional[str] = None, **kwargs) -> MockSendResult:
        self.dispatched_documents.append((chat_id, file_path, file_name))
        return MockSendResult(success=True)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None, **kwargs) -> MockSendResult:
        self.dispatched_image_files.append((chat_id, image_path, caption))
        return MockSendResult(success=True)

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None, **kwargs) -> MockSendResult:
        self.dispatched_images.append((chat_id, image_url, caption))
        return MockSendResult(success=True)

    async def _sidecar_send_poll(self, space_id: str, title: str, options: list) -> MockSendResult:
        self.dispatched_polls.append((space_id, title, options))
        return MockSendResult(success=True)

    async def _sidecar_send_effect(self, space_id: str, text: str, effect: str) -> MockSendResult:
        self.dispatched_effects.append((space_id, text, effect))
        return MockSendResult(success=True)

    async def add_reaction(self, chat_id: str, emoji: str, message_id: Optional[str] = None) -> dict[str, Any]:
        self.dispatched_reactions.append((chat_id, emoji, message_id))
        return {"ok": True}


class TestPhase5SecurityRegression(unittest.TestCase):
    """Test suite for Phase 5 security regression and verification."""

    def setUp(self) -> None:
        self.mock_adapter = MockPhotonAdapter()
        self.test_dir = tempfile.mkdtemp(prefix="airhermes_sec_test_")
        self.sensitive_file = os.path.join(self.test_dir, "sensitive_host_config.json")
        with open(self.sensitive_file, "w", encoding="utf-8") as f:
            f.write('{"secret_key": "SUPER_SECRET_HOST_TOKEN"}')

    def tearDown(self) -> None:
        try:
            if os.path.exists(self.sensitive_file):
                os.remove(self.sensitive_file)
            if os.path.exists(self.test_dir):
                os.rmdir(self.test_dir)
        except OSError:
            pass

    # =========================================================================
    # VULN-01: Unauthorized Local File Dispatch (INTENTIONALLY FAILING until Stage 2)
    # =========================================================================

    def test_vuln01_untrusted_text_file_marker_exfiltration(self) -> None:
        """Security requirement: Untrusted text referencing arbitrary host files

        must NOT be automatically dispatched as a document over iMessage.
        (Pending VULN-01 Stage 2 authorization gate)
        """
        parser = BubbleParser(paragraph_fallback=True)
        raw_llm_text = f"Here is the requested file:\n\nFILE:{self.sensitive_file}"
        bubbles = parser.feed(raw_llm_text) + parser.finish()

        consumer = BubbleStreamConsumer(self.mock_adapter, "attacker_chat")

        async def run_consumer() -> None:
            for b in bubbles:
                await consumer._send_bubble(b)

        asyncio.run(run_consumer())

        # Expected security assertion: Sensitive host file must NOT have been sent as a document!
        self.assertEqual(
            len(self.mock_adapter.dispatched_documents),
            0,
            f"VULN-01 FAIL: Sensitive file was dispatched to remote chat! Dispatched: {self.mock_adapter.dispatched_documents}",
        )

    def test_vuln01_image_tag_host_file_dispatch(self) -> None:
        """Security requirement: <image path="..."> pointing to host files

        outside authorized boundaries must NOT be dispatched.
        (Pending VULN-01 Stage 2 authorization gate)
        """
        parser = BubbleParser(paragraph_fallback=False)
        raw_llm_text = f'<image path="{self.sensitive_file}"/>'
        bubbles = parser.feed(raw_llm_text) + parser.finish()

        consumer = BubbleStreamConsumer(self.mock_adapter, "attacker_chat")

        async def run_consumer() -> None:
            for b in bubbles:
                await consumer._send_bubble(b)

        asyncio.run(run_consumer())

        # Expected security assertion: No document/image dispatch of unauthorized host file
        self.assertEqual(
            len(self.mock_adapter.dispatched_documents),
            0,
            f"VULN-01 FAIL: <image> tag dispatched host file! Dispatched: {self.mock_adapter.dispatched_documents}",
        )

    # =========================================================================
    # VULN-02: Silent Message Content Destruction (FIXED in Part A)
    # =========================================================================

    @unittest.skipUnless(PHOTON_AVAILABLE, "Requires PhotonAdapter")
    def test_vuln02_format_message_poll_only_not_emptied(self) -> None:
        """Security requirement: A turn containing only a poll tag must preserve

        content and never be stripped to empty string by format_message.
        """
        content = '<poll title="Team Lunch">1. Hotpot\n2. BBQ</poll>'
        adapter = object.__new__(PhotonAdapter)
        formatted = adapter.format_message(content)

        # Formatted content must NOT be wiped to empty string
        self.assertTrue(
            bool(formatted and formatted.strip()),
            "VULN-02 FAIL: format_message stripped entire poll tag and body to empty string!",
        )
        self.assertIn("Team Lunch", formatted)
        self.assertIn("Hotpot", formatted)

    @unittest.skipUnless(PHOTON_AVAILABLE, "Requires PhotonAdapter")
    def test_vuln02_format_message_effect_preserves_inner_text(self) -> None:
        """Security requirement: Named effects must preserve their enclosed user text."""
        content = 'Hello! <effect name="confetti">You won the grand prize!</effect>'
        adapter = object.__new__(PhotonAdapter)
        formatted = adapter.format_message(content)

        self.assertIn(
            "You won the grand prize!",
            formatted,
            f"VULN-02 FAIL: format_message destroyed effect inner text! Result: {formatted!r}",
        )

    @unittest.skipUnless(PHOTON_AVAILABLE, "Requires PhotonAdapter")
    def test_vuln02_format_message_spoiler_preserves_text(self) -> None:
        """Security requirement: Spoiler tags must preserve their secret text."""
        content = "The culprit is <spoiler>Professor Plum</spoiler>."
        adapter = object.__new__(PhotonAdapter)
        formatted = adapter.format_message(content)

        self.assertIn(
            "Professor Plum",
            formatted,
            f"VULN-02 FAIL: format_message destroyed spoiler inner text! Result: {formatted!r}",
        )

    # =========================================================================
    # VULN-03: Regex Denial of Service (ReDoS) (FIXED in Part A)
    # =========================================================================

    def test_vuln03_calendar_redos_bounded_time(self) -> None:
        """Security requirement: Structural tag regex matching on unclosed tag

        with whitespace MUST execute in bounded time (< 25ms), not exponential/polynomial seconds.
        """
        pathological_input = "<calendar " + " " * 500 + "unclosed"

        t0 = time.perf_counter()
        _m = BubbleParser._CALENDAR_RE.search(pathological_input)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.assertLess(
            elapsed_ms,
            25.0,
            f"VULN-03 FAIL: _CALENDAR_RE suffered ReDoS! 500 spaces took {elapsed_ms:.2f}ms (threshold 25ms)",
        )

    def test_vuln03_map_redos_bounded_time(self) -> None:
        """Security requirement: _MAP_RE must execute in bounded time (< 25ms)."""
        pathological_input = "<map " + " " * 500 + "unclosed"

        t0 = time.perf_counter()
        _m = BubbleParser._MAP_RE.search(pathological_input)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.assertLess(
            elapsed_ms,
            25.0,
            f"VULN-03 FAIL: _MAP_RE suffered ReDoS! 500 spaces took {elapsed_ms:.2f}ms (threshold 25ms)",
        )

    # =========================================================================
    # VULN-04: Control Marker Injection (INTENTIONALLY FAILING until Stage 2)
    # =========================================================================

    def test_vuln04_paragraph_poll_injection_not_dispatched_as_sidecar(self) -> None:
        """Security requirement: Plain text containing 'POLL:' in paragraph fallback

        must NOT trigger native sidecar poll dispatch.
        (Pending VULN-04 Stage 2 typed action gate)
        """
        parser = BubbleParser(paragraph_fallback=True)
        raw_llm_text = 'Here is how you format a poll:\n\nPOLL:{"title": "Fake Poll", "options": ["A", "B"]}'
        bubbles = parser.feed(raw_llm_text) + parser.finish()

        consumer = BubbleStreamConsumer(self.mock_adapter, "chat_victim")

        async def run_consumer() -> None:
            for b in bubbles:
                await consumer._send_bubble(b)

        asyncio.run(run_consumer())

        self.assertEqual(
            len(self.mock_adapter.dispatched_polls),
            0,
            f"VULN-04 FAIL: Plain prose injected a native sidecar poll! Dispatched: {self.mock_adapter.dispatched_polls}",
        )

    def test_vuln04_paragraph_tapback_injection_not_dispatched(self) -> None:
        """Security requirement: Plain text containing 'TAPBACK:' in paragraph fallback

        must NOT trigger native reaction.
        (Pending VULN-04 Stage 2 typed action gate)
        """
        parser = BubbleParser(paragraph_fallback=True)
        raw_llm_text = "The user replied with:\n\nTAPBACK:👍"
        bubbles = parser.feed(raw_llm_text) + parser.finish()

        consumer = BubbleStreamConsumer(self.mock_adapter, "chat_victim")

        async def run_consumer() -> None:
            for b in bubbles:
                await consumer._send_bubble(b)

        asyncio.run(run_consumer())

        self.assertEqual(
            len(self.mock_adapter.dispatched_reactions),
            0,
            f"VULN-04 FAIL: Plain prose injected a tapback reaction! Dispatched: {self.mock_adapter.dispatched_reactions}",
        )

    def test_vuln04_paragraph_effect_injection_not_dispatched(self) -> None:
        """Security requirement: Plain text containing 'EFFECT:' in paragraph fallback

        must NOT trigger native effect.
        (Pending VULN-04 Stage 2 typed action gate)
        """
        parser = BubbleParser(paragraph_fallback=True)
        raw_llm_text = 'The syntax is:\n\nEFFECT:{"effect": "confetti", "text": "Hacked"}'
        bubbles = parser.feed(raw_llm_text) + parser.finish()

        consumer = BubbleStreamConsumer(self.mock_adapter, "chat_victim")

        async def run_consumer() -> None:
            for b in bubbles:
                await consumer._send_bubble(b)

        asyncio.run(run_consumer())

        self.assertEqual(
            len(self.mock_adapter.dispatched_effects),
            0,
            f"VULN-04 FAIL: Plain prose injected a native sidecar effect! Dispatched: {self.mock_adapter.dispatched_effects}",
        )

    # =========================================================================
    # Stage 2A: Typed Semantic Actions & Safe Text Dispatch Contract
    # =========================================================================

    def test_stage2a_typed_action_contract_immutability(self) -> None:
        """Requirement: TextBubble, PollAction, TapbackAction, EffectAction must be frozen dataclasses with tuples."""
        from gateway.bubble_actions import TextBubble, PollAction, TapbackAction, EffectAction
        from dataclasses import FrozenInstanceError

        poll = PollAction(title="Lunch", options=("Pizza", "Sushi"))
        self.assertEqual(poll.title, "Lunch")
        self.assertIsInstance(poll.options, tuple)
        self.assertEqual(poll.options, ("Pizza", "Sushi"))
        with self.assertRaises(FrozenInstanceError):
            poll.title = "Dinner"  # type: ignore

        tap = TapbackAction(reaction="love")
        with self.assertRaises(FrozenInstanceError):
            tap.reaction = "like"  # type: ignore

        eff = EffectAction(effect_type="fireworks", content="Congrats!")
        with self.assertRaises(FrozenInstanceError):
            eff.effect_type = "confetti"  # type: ignore

        txt = TextBubble(content="Hello world")
        with self.assertRaises(FrozenInstanceError):
            txt.content = "New world"  # type: ignore

    def test_stage2a_parser_feed_actions_structured_tags(self) -> None:
        """Requirement: Structured tags produce typed actions; ordinary text becomes TextBubble."""
        from gateway.bubble_actions import TextBubble, PollAction, TapbackAction, EffectAction
        parser = BubbleParser(paragraph_fallback=True)

        # 1. Structured poll
        actions = parser.feed_actions('<poll title="Vote"><option>Opt 1</option><option>Opt 2</option></poll>')
        actions.extend(parser.finish_actions())
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], PollAction)
        self.assertEqual(actions[0].title, "Vote")
        self.assertEqual(actions[0].options, ("Opt 1", "Opt 2"))

        # 2. Structured tapback
        parser = BubbleParser()
        actions = parser.feed_actions('<tapback emoji="❤️"/>')
        actions.extend(parser.finish_actions())
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], TapbackAction)
        self.assertEqual(actions[0].reaction, "❤️")

        # 3. Structured effect
        parser = BubbleParser()
        actions = parser.feed_actions('<effect name="fireworks">Boom!</effect>')
        actions.extend(parser.finish_actions())
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], EffectAction)
        self.assertEqual(actions[0].effect_type, "fireworks")
        self.assertEqual(actions[0].content, "Boom!")

        # 4. Plain text containing control strings MUST BE TextBubble
        parser = BubbleParser(paragraph_fallback=True)
        raw_text = "Check this:\n\nPOLL:{\"title\": \"X\"}\n\nTAPBACK:👍\n\nEFFECT:{\"effect\": \"lasers\"}\n\nFILE:/etc/passwd\n\nIMAGE:{\"path\": \"/foo\"}"
        actions = parser.feed_actions(raw_text)
        actions.extend(parser.finish_actions())
        for a in actions:
            self.assertIsInstance(a, TextBubble)

    def test_stage2a_consumer_typed_dispatch_and_safe_text(self) -> None:
        """Requirement: BubbleStreamConsumer dispatches typed actions and safe text without string prefix inference."""
        from gateway.bubble_actions import TextBubble, PollAction, TapbackAction, EffectAction

        consumer = BubbleStreamConsumer(self.mock_adapter, "chat_stage2a")

        async def run_dispatch() -> None:
            # Typed actions
            await consumer._send_bubble(PollAction(title="Q", options=("A", "B")))
            await consumer._send_bubble(TapbackAction(reaction="love"))
            await consumer._send_bubble(EffectAction(effect_type="confetti", content="Yay"))
            # TextBubble containing control prefix
            await consumer._send_bubble(TextBubble(content='POLL:{"title": "Fake"}'))
            await consumer._send_bubble(TextBubble(content="FILE:/etc/passwd"))
            # Raw string containing control prefix
            await consumer._send_bubble('EFFECT:{"effect": "hacked"}')

        asyncio.run(run_dispatch())

        # PollAction -> adapter._sidecar_send_poll
        self.assertEqual(len(self.mock_adapter.dispatched_polls), 1)
        self.assertEqual(self.mock_adapter.dispatched_polls[0][1], "Q")
        # TapbackAction -> adapter.add_reaction
        self.assertEqual(len(self.mock_adapter.dispatched_reactions), 1)
        self.assertEqual(self.mock_adapter.dispatched_reactions[0][1], "love")
        # EffectAction -> adapter._sidecar_send_effect
        self.assertEqual(len(self.mock_adapter.dispatched_effects), 1)
        self.assertEqual(self.mock_adapter.dispatched_effects[0][2], "confetti")
        # TextBubble / string -> plain text send ONLY!
        self.assertEqual(len(self.mock_adapter.dispatched_documents), 0)
        text_contents = [t[1] for t in self.mock_adapter.dispatched_texts]
        self.assertIn('POLL:{"title": "Fake"}', text_contents)
        self.assertIn("FILE:/etc/passwd", text_contents)
        self.assertIn('EFFECT:{"effect": "hacked"}', text_contents)

    def test_stage2a_disabled_features_readable_fallback(self) -> None:
        """Requirement: When semantic feature flags are False, actions degrade to readable text and do not invoke native APIs."""
        from gateway.bubble_actions import PollAction, TapbackAction, EffectAction

        cfg = AirHermesConfig(enable_polls=False, enable_tapbacks=False, enable_effects=False)
        adapter = MockPhotonAdapter(cfg)
        consumer = BubbleStreamConsumer(adapter, "chat_disabled", config=cfg)

        async def run_dispatch() -> None:
            await consumer._send_bubble(PollAction(title="Vote", options=("Option 1", "Option 2")))
            await consumer._send_bubble(TapbackAction(reaction="👍"))
            await consumer._send_bubble(EffectAction(effect_type="lasers", content="Pew pew"))

        asyncio.run(run_dispatch())

        self.assertEqual(len(adapter.dispatched_polls), 0)
        self.assertEqual(len(adapter.dispatched_reactions), 0)
        self.assertEqual(len(adapter.dispatched_effects), 0)
        # Should have delivered readable text fallbacks
        text_contents = [t[1] for t in adapter.dispatched_texts]
        self.assertTrue(any("Vote" in t and "Option 1" in t for t in text_contents))
        self.assertIn("Pew pew", text_contents)

    def test_stage2a_code_masking_collision_safety(self) -> None:
        """Requirement: format_message code-block masking cannot collide with user-supplied text or leave masking placeholders."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")
        adapter = object.__new__(PhotonAdapter)

        tricky_input = "Here is some code:\n```python\nprint('\\x00CODE_0\\x00')\n```\nAnd literal \\x00CODE_1\\x00 in text."
        formatted = adapter.format_message(tricky_input)
        self.assertNotIn("\x00AIRHERMES_CODE_", formatted)
        self.assertIn("print('\\x00CODE_0\\x00')", formatted)

    # =========================================================================
    # Stage 2B: Upstream MEDIA: Bypass & Controlled Attachment Authorization
    # =========================================================================

    def test_stage2b_01_untrusted_media_not_extracted(self) -> None:
        """Scenario 1: Untrusted MEDIA: tag pointing to an existing file must NOT be extracted by PhotonAdapter."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        untrusted_text = f"Look at this file: MEDIA:{self.sensitive_file}"
        extracted, cleaned = PhotonAdapter.extract_media(untrusted_text)
        self.assertEqual(len(extracted), 0, f"VULN-01 BYPASS: Untrusted MEDIA path was extracted: {extracted}")
        self.assertIn(self.sensitive_file, cleaned, "Original text content should be preserved as ordinary text")

    def test_stage2b_02_upstream_media_bypass_with_streaming_enabled(self) -> None:
        """Scenario 2: When bubble_streaming is enabled, upstream _deliver_media_from_response must not dispatch untrusted MEDIA:."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        adapter = object.__new__(PhotonAdapter)
        adapter.airhermes_config = AirHermesConfig(bubble_streaming=True)
        dispatched_attachments: list[tuple[str, str]] = []

        async def mock_sidecar_send_attachment(space_id: str, path: str, **kwargs: Any) -> MockSendResult:
            dispatched_attachments.append((space_id, path))
            return MockSendResult(success=True)

        adapter._sidecar_send_attachment = mock_sidecar_send_attachment

        # Simulate the exact upstream post-stream delivery logic from gateway/run_notifications.py:
        response = f"Finished turn with data: MEDIA:{self.sensitive_file}"
        media_files, _cleaned = adapter.extract_media(response)

        async def run_post_stream_delivery() -> None:
            for p, _is_voice in media_files:
                await adapter.send_document("victim_chat", p)

        asyncio.run(run_post_stream_delivery())
        self.assertEqual(len(dispatched_attachments), 0, f"VULN-01 BYPASS: Streamed turn dispatched untrusted file: {dispatched_attachments}")

    def test_stage2b_03_upstream_media_bypass_with_streaming_disabled(self) -> None:
        """Scenario 3: When bubble_streaming is disabled, non-streamed final response must not dispatch untrusted MEDIA:."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        adapter = object.__new__(PhotonAdapter)
        adapter.airhermes_config = AirHermesConfig(bubble_streaming=False)
        dispatched_attachments: list[tuple[str, str]] = []
        dispatched_texts: list[tuple[str, str]] = []

        async def mock_sidecar_send(space_id: str, text: str, **kwargs: Any) -> MockSendResult:
            dispatched_texts.append((space_id, text))
            return MockSendResult(success=True)

        async def mock_sidecar_send_attachment(space_id: str, path: str, **kwargs: Any) -> MockSendResult:
            dispatched_attachments.append((space_id, path))
            return MockSendResult(success=True)

        adapter._sidecar_send = mock_sidecar_send
        adapter._sidecar_send_attachment = mock_sidecar_send_attachment

        # Simulate non-streamed final delivery:
        response = f"Non-streamed reply: MEDIA:{self.sensitive_file}"
        media_files, cleaned = adapter.extract_media(response)

        async def run_non_stream_delivery() -> None:
            for p, _is_voice in media_files:
                await adapter.send_document("victim_chat", p)
            await adapter.send("victim_chat", cleaned)

        asyncio.run(run_non_stream_delivery())
        self.assertEqual(len(dispatched_attachments), 0, f"VULN-01 BYPASS: Non-streamed turn dispatched untrusted file: {dispatched_attachments}")
        self.assertEqual(len(dispatched_texts), 1)
        self.assertIn(self.sensitive_file, dispatched_texts[0][1])

    def test_stage2b_04_ordinary_text_containing_control_markers(self) -> None:
        """Scenario 4: Ordinary text containing FILE:, IMAGE:, and MEDIA: examples remains ordinary text."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        adapter = object.__new__(PhotonAdapter)
        adapter.airhermes_config = AirHermesConfig()
        dispatched_attachments: list[tuple[str, str]] = []
        dispatched_texts: list[tuple[str, str]] = []

        async def mock_sidecar_send(space_id: str, text: str, **kwargs: Any) -> MockSendResult:
            dispatched_texts.append((space_id, text))
            return MockSendResult(success=True)

        async def mock_sidecar_send_attachment(space_id: str, path: str, **kwargs: Any) -> MockSendResult:
            dispatched_attachments.append((space_id, path))
            return MockSendResult(success=True)

        adapter._sidecar_send = mock_sidecar_send
        adapter._sidecar_send_attachment = mock_sidecar_send_attachment

        prose = f"Syntax examples:\n1. FILE:{self.sensitive_file}\n2. MEDIA:{self.sensitive_file}\n3. IMAGE:{{\"path\": \"{self.sensitive_file}\"}}"

        parser = BubbleParser(paragraph_fallback=True)
        actions = parser.feed_actions(prose) + parser.finish_actions()

        consumer = BubbleStreamConsumer(adapter, "chat_examples")

        async def run_consumer() -> None:
            for a in actions:
                await consumer._send_bubble(a)

        asyncio.run(run_consumer())
        self.assertEqual(len(dispatched_attachments), 0)
        self.assertTrue(len(dispatched_texts) > 0)
        combined_text = "\n".join(t[1] for t in dispatched_texts)
        self.assertIn("FILE:", combined_text)
        self.assertIn("MEDIA:", combined_text)
        self.assertIn("IMAGE:", combined_text)

    def test_stage2b_06_unauthorized_direct_attachment_calls_rejected(self) -> None:
        """Scenario 3: Direct call with unregistered file is rejected without auto-authorization fallback."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        adapter = object.__new__(PhotonAdapter)
        dispatched_attachments: list[tuple[str, str, Optional[str]]] = []

        async def mock_sidecar_send_attachment(space_id: str, path: str, **kwargs: Any) -> MockSendResult:
            dispatched_attachments.append((space_id, path, kwargs.get("name")))
            return MockSendResult(success=True)

        adapter._sidecar_send_attachment = mock_sidecar_send_attachment

        async def run_unauthorized_direct_calls() -> tuple[Any, Any]:
            res1 = await adapter.send_document("chat_direct", self.sensitive_file, file_name="safe_name.txt")
            res2 = await adapter.send_image_file("chat_direct", self.sensitive_file, caption="Photo")
            return res1, res2

        res1, res2 = asyncio.run(run_unauthorized_direct_calls())
        # Both must fail authorization
        self.assertFalse(res1.success)
        self.assertIn("not authorized", (res1.error or "").lower())
        self.assertFalse(res2.success)
        self.assertIn("not authorized", (res2.error or "").lower())
        # Zero attachments must reach the sidecar
        self.assertEqual(len(dispatched_attachments), 0)

    def test_stage2b_06b_authorized_direct_adapter_attachment_calls(self) -> None:
        """Scenario 6: Pre-authorized staged attachments passed to adapter methods succeed."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        from gateway.attachment_registry import registry
        adapter = object.__new__(PhotonAdapter)
        dispatched_attachments: list[tuple[str, str, Optional[str]]] = []

        async def mock_sidecar_send_attachment(space_id: str, path: str, **kwargs: Any) -> MockSendResult:
            dispatched_attachments.append((space_id, path, kwargs.get("name")))
            return MockSendResult(success=True)

        adapter._sidecar_send_attachment = mock_sidecar_send_attachment

        # Pre-authorize document and image
        rec_doc, err_doc = registry.stage_and_authorize(
            "chat_direct_auth", self.sensitive_file, producer="test_auth_direct", file_name="safe_name.txt"
        )
        self.assertIsNotNone(rec_doc)
        self.assertIsNone(err_doc)

        rec_img, err_img = registry.stage_and_authorize(
            "chat_direct_auth", self.sensitive_file, producer="test_auth_direct", file_name="photo.png"
        )
        self.assertIsNotNone(rec_img)
        self.assertIsNone(err_img)

        async def run_authorized_direct_calls() -> tuple[Any, Any]:
            res1 = await adapter.send_document("chat_direct_auth", rec_doc.staged_path)
            res2 = await adapter.send_image_file("chat_direct_auth", rec_img.staged_path, caption="Photo")
            return res1, res2

        res1, res2 = asyncio.run(run_authorized_direct_calls())
        self.assertTrue(res1.success)
        self.assertTrue(res2.success)
        self.assertEqual(len(dispatched_attachments), 2)
        self.assertEqual(dispatched_attachments[0][0], "chat_direct_auth")
        self.assertEqual(dispatched_attachments[0][1], rec_doc.staged_path)
        self.assertEqual(dispatched_attachments[1][0], "chat_direct_auth")
        self.assertEqual(dispatched_attachments[1][1], rec_img.staged_path)

    def test_stage2b_05_legitimate_internally_generated_attachment(self) -> None:
        """Scenario 5: Legitimate internally generated attachments (e.g. .ics) continue to work seamlessly."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        adapter = object.__new__(PhotonAdapter)
        dispatched_docs: list[dict[str, Any]] = []

        async def mock_sidecar_send_attachment(space_id: str, path: str, **kwargs: Any) -> MockSendResult:
            dispatched_docs.append({
                "chat_id": space_id,
                "path": path,
                "name": kwargs.get("name"),
                "caption": kwargs.get("caption"),
            })
            return MockSendResult(success=True)

        adapter._sidecar_send_attachment = mock_sidecar_send_attachment

        calendar_tag = '<calendar title="Phase 5 Review" start="2026-11-25 14:00" end="2026-11-25 15:00" location="Conf Room A" description="Security Gate"/>'
        from gateway.bubble_actions import FileAttachmentAction
        parser = BubbleParser(paragraph_fallback=False, chat_id="chat_legit_turn", turn_id="turn_42")
        actions = parser.feed_actions(calendar_tag) + parser.finish_actions()

        self.assertEqual(len(actions), 1)
        action = actions[0]
        self.assertIsInstance(action, FileAttachmentAction)
        self.assertIn("airhermes_staging", action.staged_path)
        self.assertTrue(action.file_name.endswith(".ics"))
        self.assertTrue(os.path.isfile(action.staged_path))

        with open(action.staged_path, "r", encoding="utf-8") as f:
            ics_text = f.read()
        self.assertIn("BEGIN:VCALENDAR", ics_text)
        self.assertIn("SUMMARY:Phase 5 Review", ics_text)
        self.assertNotIn("SUMMARY;CHARSET=UTF-8:", ics_text)

        consumer = BubbleStreamConsumer(adapter, "chat_legit_turn", metadata={"turn_id": "turn_42"})

        async def deliver() -> None:
            await consumer._send_bubble(action)

        asyncio.run(deliver())
        self.assertEqual(len(dispatched_docs), 1)
        self.assertEqual(dispatched_docs[0]["chat_id"], "chat_legit_turn")
        self.assertEqual(dispatched_docs[0]["path"], action.staged_path)

        # Single-use check: delivering the same staged attachment again should now fail
        from gateway.attachment_registry import registry
        ok, reason, _ = registry.verify_for_delivery("chat_legit_turn", action.staged_path, turn_id="turn_42")
        self.assertFalse(ok)
        self.assertIn("single-use", (reason or "").lower())

    def test_stage2b_07_cross_chat_isolation(self) -> None:
        """Security: An attachment authorized for chat_A must NOT be deliverable to chat_B."""
        from gateway.attachment_registry import registry

        sample_file = os.path.join(self.test_dir, "report.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("confidential report")

        record, err = registry.stage_and_authorize("chat_alice", sample_file, producer="test_isolation")
        self.assertIsNotNone(record)
        self.assertIsNone(err)

        # Verify for authorized chat succeeds
        ok_a, _, _ = registry.verify_for_delivery("chat_alice", record.staged_path)
        self.assertTrue(ok_a)
        # Reset in_flight for subsequent test
        record.in_flight = False

        # Attempt verification for unauthorized chat fails
        ok_b, reason_b, _ = registry.verify_for_delivery("chat_bob", record.staged_path)
        self.assertFalse(ok_b)
        self.assertIn("mismatch", (reason_b or "").lower())

    def test_stage2b_08_cross_turn_isolation(self) -> None:
        """Security: An attachment bound to turn_1 must NOT be deliverable to turn_2."""
        from gateway.attachment_registry import registry

        sample_file = os.path.join(self.test_dir, "turn_bound.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("turn bound payload")

        record, _err = registry.stage_and_authorize(
            "chat_turn_test", sample_file, producer="test_turn_isolation", turn_id="turn_alpha",
        )
        self.assertIsNotNone(record)

        # Same turn succeeds
        ok_1, _, _ = registry.verify_for_delivery("chat_turn_test", record.staged_path, turn_id="turn_alpha")
        self.assertTrue(ok_1)
        record.in_flight = False

        # Different turn fails
        ok_2, reason_2, _ = registry.verify_for_delivery("chat_turn_test", record.staged_path, turn_id="turn_beta")
        self.assertFalse(ok_2)
        self.assertIn("turn mismatch", (reason_2 or "").lower())

    def test_stage2b_09_toctou_hash_tamper_detection(self) -> None:
        """Security: Tampering with the staged file after authorization triggers TOCTOU detection."""
        from gateway.attachment_registry import registry

        sample_file = os.path.join(self.test_dir, "toctou_source.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("clean original content")

        record, _err = registry.stage_and_authorize("chat_toctou", sample_file, producer="test_toctou")
        self.assertIsNotNone(record)

        # Tamper with the staged file on disk
        with open(record.staged_path, "a", encoding="utf-8") as f:
            f.write("\nMALICIOUS APPENDED PAYLOAD")

        ok, reason, _ = registry.verify_for_delivery("chat_toctou", record.staged_path)
        self.assertFalse(ok)
        self.assertIn("toctou", (reason or "").lower())

    def test_stage2b_10_denied_sensitive_source_files(self) -> None:
        """Security: System credentials, .env files, and SSH keys must be rejected at staging time."""
        from gateway.attachment_registry import registry

        denied_files = [".env", ".envrc", "id_rsa", "id_ed25519", "credentials.json", "cert.pem"]
        for dname in denied_files:
            target = os.path.join(self.test_dir, dname)
            with open(target, "w", encoding="utf-8") as f:
                f.write("sensitive_secret=12345")

            record, err = registry.stage_and_authorize("chat_leak_test", target, producer="test_sensitive")
            self.assertIsNone(record, f"Should reject sensitive file {dname}")
            self.assertIsNotNone(err)
            self.assertIn("denied", (err or "").lower())

    def test_stage2b_11_attachment_limits(self) -> None:
        """Security: Enforces 50MB maximum attachment size and 5 attachments per turn."""
        from gateway.attachment_registry import MAX_ATTACHMENTS_PER_TURN, registry

        # Test turn artifact count limit
        turn_file = os.path.join(self.test_dir, "turn_limit_file.txt")
        with open(turn_file, "w", encoding="utf-8") as f:
            f.write("dummy turn artifact")

        turn_id = "turn_overflow_test"
        for i in range(MAX_ATTACHMENTS_PER_TURN):
            rec, err = registry.stage_and_authorize("chat_overflow", turn_file, producer="lim_test", turn_id=turn_id)
            self.assertIsNotNone(rec, f"Attachment {i+1} should succeed")
            self.assertIsNone(err)

        # 6th attachment in the same turn must be rejected
        rec_overflow, err_overflow = registry.stage_and_authorize(
            "chat_overflow", turn_file, producer="lim_test", turn_id=turn_id,
        )
        self.assertIsNone(rec_overflow)
        self.assertIsNotNone(err_overflow)
        self.assertIn("limit", (err_overflow or "").lower())

    def test_stage2b_12_in_flight_cleanup_protection(self) -> None:
        """Lifecycle: In-flight attachments are protected against premature deletion during cleanup."""
        from gateway.attachment_registry import registry

        sample_file = os.path.join(self.test_dir, "inflight.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("in-flight data")

        record, _ = registry.stage_and_authorize("chat_inflight", sample_file, producer="inflight_test", turn_id="turn_if")
        self.assertIsNotNone(record)

        # Verify for delivery marks it in_flight = True
        ok, _, rec = registry.verify_for_delivery("chat_inflight", record.staged_path, turn_id="turn_if")
        self.assertTrue(ok)
        self.assertTrue(rec.in_flight)

        # Calling cleanup_turn while in_flight must NOT delete the file
        registry.cleanup_turn("chat_inflight", "turn_if")
        self.assertTrue(os.path.isfile(record.staged_path), "In-flight file must survive cleanup_turn")

        # After delivery completes
        registry.record_delivery_result(record.token, success=True)
        self.assertFalse(record.in_flight)

        # Now cleanup_turn should safely remove it
        registry.cleanup_turn("chat_inflight", "turn_if")
        self.assertFalse(os.path.isfile(record.staged_path), "Delivered file should be cleaned up")

    def test_stage2b_13_atomic_concurrency_single_use(self) -> None:
        """Scenario 9: Concurrent duplicate redemption is prevented atomically under lock."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        from gateway.attachment_registry import registry
        adapter = object.__new__(PhotonAdapter)
        dispatched: list[str] = []

        async def mock_sidecar_send_attachment(space_id: str, path: str, **kwargs: Any) -> MockSendResult:
            await asyncio.sleep(0.02)  # health: allow HX013 -- queue pacing verification
            dispatched.append(path)
            return MockSendResult(success=True)

        adapter._sidecar_send_attachment = mock_sidecar_send_attachment

        sample_file = os.path.join(self.test_dir, "atomic_conc.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("atomic payload")

        record, _ = registry.stage_and_authorize("chat_atomic", sample_file, producer="test_conc", turn_id="t_conc")

        async def run_concurrent() -> list[Any]:
            return await asyncio.gather(
                adapter.send_document("chat_atomic", record.staged_path, metadata={"turn_id": "t_conc"}),
                adapter.send_document("chat_atomic", record.staged_path, metadata={"turn_id": "t_conc"}),
                return_exceptions=True,
            )

        results = asyncio.run(run_concurrent())
        # Exactly one call should reach the sidecar and succeed
        self.assertEqual(len(dispatched), 1)
        success_count = sum(1 for r in results if getattr(r, "success", False) is True)
        self.assertEqual(success_count, 1)

    def test_stage2b_14_cancellation_lifecycle_handling(self) -> None:
        """Scenario 10: Cancellation during sidecar send sets OUTCOME_UNKNOWN and allows disk recovery."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        from gateway.attachment_registry import AttachmentState, registry
        adapter = object.__new__(PhotonAdapter)

        async def hanging_send(*a: Any, **k: Any) -> MockSendResult:
            await asyncio.sleep(10.0)  # health: allow HX013 -- queue pacing verification
            return MockSendResult(success=True)

        adapter._sidecar_send_attachment = hanging_send

        sample_file = os.path.join(self.test_dir, "cancel_lifecycle.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("cancel test")

        record, _ = registry.stage_and_authorize("chat_cancel_lc", sample_file, producer="test_cancel", turn_id="t_clc")

        async def run_cancel() -> None:
            task = asyncio.create_task(
                adapter.send_document("chat_cancel_lc", record.staged_path, metadata={"turn_id": "t_clc"})
            )
            await asyncio.sleep(0.02)  # health: allow HX013 -- queue pacing verification
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_cancel())

        # State should be OUTCOME_UNKNOWN, in_flight should be False
        self.assertEqual(record.state, AttachmentState.OUTCOME_UNKNOWN)
        self.assertFalse(record.in_flight)

        # Invariant: An OUTCOME_UNKNOWN file must be quarantined and preserved during turn cleanup
        # so an external native sender (e.g. Sidecar) can finish reading it without I/O error
        registry.cleanup_turn("chat_cancel_lc", "t_clc")
        self.assertTrue(os.path.isfile(record.staged_path), "Quarantined file must NOT be deleted immediately during cleanup_turn")

        # After quarantine duration expires, deferred cleanup can safely unlink it
        record.quarantine_duration_seconds = 0.0  # simulate expired quarantine
        registry.cleanup_turn("chat_cancel_lc", "t_clc")
        self.assertFalse(os.path.isfile(record.staged_path), "Expired quarantined file must be cleaned up on disk")

    def test_stage2b_15_extract_local_files_containment(self) -> None:
        """Containment: PhotonAdapter.extract_local_files neutralizes bare host file paths."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        real_file = self.sensitive_file
        untrusted_text = f"Here is the report: {real_file} and another /etc/shadow"

        extracted, cleaned = PhotonAdapter.extract_local_files(untrusted_text)
        self.assertEqual(extracted, [])
        self.assertEqual(cleaned, untrusted_text)

    def test_stage2c_01_cancellation_before_native_dispatch_starts_sets_failed_pre_dispatch(self) -> None:
        """Lifecycle: Cancellation before native dispatch begins transitions to FAILED_PRE_DISPATCH."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        from gateway.attachment_registry import AttachmentState, registry
        adapter = object.__new__(PhotonAdapter)

        dispatch_started = False
        async def mock_send(*a: Any, **k: Any) -> MockSendResult:
            nonlocal dispatch_started
            dispatch_started = True
            return MockSendResult(success=True)

        adapter._sidecar_send_attachment = mock_send

        sample_file = os.path.join(self.test_dir, "cancel_pre.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("pre dispatch cancel")

        record, _ = registry.stage_and_authorize("chat_c_pre", sample_file, producer="test_pre", turn_id="t_pre")
        # Explicitly cancel pre-dispatch
        registry.verify_for_delivery("chat_c_pre", record.staged_path, turn_id="t_pre")
        self.assertEqual(record.state, AttachmentState.CLAIMED)
        registry.mark_cancelled(record.token, in_native_dispatch=False)
        self.assertEqual(record.state, AttachmentState.FAILED_PRE_DISPATCH)
        self.assertFalse(record.in_flight)

        # FAILED_PRE_DISPATCH can be cleaned up immediately since no external reader exists
        registry.cleanup_turn("chat_c_pre", "t_pre")
        self.assertFalse(os.path.isfile(record.staged_path))

    def test_stage2c_02_external_sender_concurrent_read_during_and_after_cancellation(self) -> None:
        """Lifecycle: External sender continuing to read file after coroutine cancellation succeeds."""
        if not PHOTON_AVAILABLE:
            self.skipTest("PhotonAdapter unavailable")

        from gateway.attachment_registry import AttachmentState, registry
        adapter = object.__new__(PhotonAdapter)

        read_content = None
        sample_file = os.path.join(self.test_dir, "concurrent_reader.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("payload for external sender")

        record, _ = registry.stage_and_authorize("chat_ext_read", sample_file, producer="test_ext", turn_id="t_ext")

        async def mock_sidecar_reading(*a: Any, **k: Any) -> MockSendResult:
            # Simulate external process (e.g. Node Sidecar) reading in background
            async def external_sender_reader() -> None:
                await asyncio.sleep(0.03)  # health: allow HX013 -- queue pacing verification
                nonlocal read_content
                read_content = await asyncio.to_thread(lambda: Path(record.staged_path).read_text(encoding="utf-8"))

            asyncio.create_task(external_sender_reader())
            await asyncio.sleep(10.0)  # health: allow HX013 -- queue pacing verification  # Hang until cancelled
            return MockSendResult(success=True)

        adapter._sidecar_send_attachment = mock_sidecar_reading

        async def run_cancel() -> None:
            task = asyncio.create_task(
                adapter.send_document("chat_ext_read", record.staged_path, metadata={"turn_id": "t_ext"})
            )
            await asyncio.sleep(0.01)  # health: allow HX013 -- queue pacing verification
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            # Give background external reader time to read
            await asyncio.sleep(0.05)  # health: allow HX013 -- queue pacing verification

        asyncio.run(run_cancel())

        # Immediately run cleanup_turn
        registry.cleanup_turn("chat_ext_read", "t_ext")

        # Quarantined file must still exist and external reader must have successfully read
        self.assertEqual(record.state, AttachmentState.OUTCOME_UNKNOWN)
        self.assertTrue(os.path.isfile(record.staged_path))
        self.assertEqual(read_content, "payload for external sender")

    def test_stage2c_03_outcome_unknown_forbids_automatic_retry(self) -> None:
        """Security: verify_for_delivery rejects any attempt to redeliver an OUTCOME_UNKNOWN artifact."""
        from gateway.attachment_registry import AttachmentState, registry

        sample_file = os.path.join(self.test_dir, "no_retry.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("no retry payload")

        record, _ = registry.stage_and_authorize("chat_no_retry", sample_file, producer="test_nr", turn_id="t_nr")
        ok1, _, _ = registry.verify_for_delivery("chat_no_retry", record.staged_path, turn_id="t_nr")
        self.assertTrue(ok1)
        registry.mark_dispatching(record.token)
        registry.mark_cancelled(record.token, in_native_dispatch=True)
        self.assertEqual(record.state, AttachmentState.OUTCOME_UNKNOWN)

        # Retry must fail
        ok2, reason2, _ = registry.verify_for_delivery("chat_no_retry", record.staged_path, turn_id="t_nr")
        self.assertFalse(ok2)
        self.assertIn("outcome is unknown", (reason2 or "").lower())

    def test_stage2c_04_terminal_state_immutability(self) -> None:
        """State Machine: Terminal states cannot be regressed or mutated via legacy setters."""
        from gateway.attachment_registry import AttachmentRecord, AttachmentState

        rec = AttachmentRecord(
            token="att_test", chat_id="c", turn_id="t", producer="p",
            original_path="orig", staged_path="staged", sha256="hash", file_size=10, file_name="f",
            state=AttachmentState.SUCCEEDED
        )

        # Regressing SUCCEEDED via delivered = False must raise ValueError
        with self.assertRaises(ValueError):
            rec.delivered = False

        # Setting in_flight = True on SUCCEEDED must raise ValueError
        with self.assertRaises(ValueError):
            rec.in_flight = True

        rec.state = AttachmentState.OUTCOME_UNKNOWN
        with self.assertRaises(ValueError):
            rec.in_flight = True
        with self.assertRaises(ValueError):
            rec.delivered = True

    def test_stage2c_05_cross_process_isolation_and_crash_recovery(self) -> None:
        """Recovery: recover_abandoned_staging protects active PIDs and unlinks dead PIDs."""
        from gateway.attachment_registry import AttachmentRegistry
        import shutil

        temp_staging = tempfile.mkdtemp(prefix="test_staging_rec_")
        try:
            reg = AttachmentRegistry(base_staging_dir=Path(temp_staging))

            # 1. Directory belonging to CURRENT active PID must NOT be touched
            my_pid_dir = Path(temp_staging) / f"proc_{os.getpid()}_abcdef123456"
            my_pid_dir.mkdir(parents=True)
            (my_pid_dir / "test.txt").write_text("my process data")

            # 2. Directory belonging to a dead PID (e.g. 999999) older than min age must be cleaned
            dead_pid_dir = Path(temp_staging) / "proc_999999_deadbeef0000"
            dead_pid_dir.mkdir(parents=True)
            (dead_pid_dir / "dead.txt").write_text("dead process data")
            # Set mtime back by 120 seconds
            old_time = time.time() - 120
            os.utime(dead_pid_dir, (old_time, old_time))

            recovered = reg.recover_abandoned_staging(max_age_seconds=86400, min_abandoned_age_seconds=10.0)
            self.assertEqual(recovered, 1)
            self.assertTrue(my_pid_dir.exists(), "Active process directory must be preserved")
            self.assertFalse(dead_pid_dir.exists(), "Dead process directory must be cleaned up")

            # Idempotency check: running recovery again should be a safe no-op
            recovered_again = reg.recover_abandoned_staging(max_age_seconds=86400, min_abandoned_age_seconds=10.0)
            self.assertEqual(recovered_again, 0)
            self.assertTrue(my_pid_dir.exists())
        finally:
            shutil.rmtree(temp_staging, ignore_errors=True)

    def test_stage2c_06_typed_attachment_action_never_sent_as_plain_text(self) -> None:
        """Boundary: BubbleStreamConsumer._send_bubble refuses to send File/Image actions as plain text."""
        from gateway.bubble_actions import FileAttachmentAction, ImageAttachmentAction
        from gateway.bubble_stream_consumer import BubbleStreamConsumer

        adapter = object.__new__(PhotonAdapter)
        sent_texts: list[str] = []
        async def mock_send(chat_id: str, text: str, **kwargs: Any) -> MockSendResult:
            sent_texts.append(text)
            return MockSendResult(success=True)

        adapter.send = mock_send
        # Adapter does NOT have send_document or send_image_file
        consumer = BubbleStreamConsumer(adapter, "chat_plain")

        async def run() -> None:
            # Pushing FileAttachmentAction or ImageAttachmentAction must NOT result in raw text send
            res_file = await consumer._send_bubble(FileAttachmentAction(staged_path="C:/secret.txt", file_name="secret.txt"))
            res_img = await consumer._send_bubble(ImageAttachmentAction(staged_path="C:/photo.png", caption="Photo"))
            self.assertFalse(res_file["ok"])
            self.assertFalse(res_img["ok"])

        asyncio.run(run())
        self.assertEqual(len(sent_texts), 0, "Attachment action must never fall back to plain text delivery")

    def test_vuln05_map_url_spoofing_containment(self) -> None:
        """VULN-05: Apple Maps URL generation strictly validates domains and prevents arbitrary URL spoofing."""
        parser = BubbleParser()

        # 1. Legitimate supported map URLs are preserved verbatim
        valid_urls = [
            "https://maps.apple.com/?q=Tokyo",
            "http://maps.apple.com/?q=Tokyo",
            "maps://?q=Tokyo",
            "maps://maps.apple.com/?q=Tokyo",
            "https://maps.google.com/?q=Tokyo",
            "https://maps.apple.com/place?coordinate=37.3349,-122.0090&name=Apple%20Park&map=explore",
        ]
        for url in valid_urls:
            res = parser._build_apple_maps_url(query=url)
            self.assertEqual(res, url, f"Legitimate map URL must be preserved: {url}")

        # 2. Deceptive domains and phishing URLs are rejected and converted to safe Apple Maps search
        deceptive_urls = [
            ("https://maps.apple.com.attacker.com/evil?q=Tokyo+Tower", "Tokyo Tower"),
            ("https://attacker-maps.apple.com/steal?name=Secret+Base", "Secret Base"),
            ("https://evil.com/fake_location", "fake_location"),
            ("http://maps.apple.com.sub.phishing.io/login", "login"),
            ("https://attacker.com/maps.apple.com", "maps.apple.com"),
        ]
        for bad_url, expected_keyword in deceptive_urls:
            res = parser._build_apple_maps_url(query=bad_url)
            self.assertNotIn("attacker.com", res)
            self.assertNotIn("phishing.io", res)
            self.assertNotIn("evil.com", res)
            self.assertTrue(res.startswith("https://maps.apple.com/place?"), f"Deceptive URL must fall back to Apple Maps: {res}")
            # Ensure readable location query is preserved in search parameters
            self.assertTrue(
                expected_keyword.replace(" ", "+") in res or expected_keyword.replace(" ", "%20") in res,
                f"Readable location '{expected_keyword}' must be preserved in {res}",
            )

        # 3. Embedded credentials (user:pass@) are rejected
        cred_url = "https://admin:secret@maps.apple.com/?q=Central+Park"
        res_cred = parser._build_apple_maps_url(query=cred_url)
        self.assertNotIn("admin:secret", res_cred)
        self.assertTrue(res_cred.startswith("https://maps.apple.com/place?"))

        # 4. Non-standard ports are rejected
        port_url = "https://maps.apple.com:8080/path?q=Test"
        res_port = parser._build_apple_maps_url(query=port_url)
        self.assertNotIn(":8080", res_port)
        self.assertTrue(res_port.startswith("https://maps.apple.com/place?"))

        # 5. Unsupported schemes (javascript:, file:, data:) are rejected
        unsupported = [
            "javascript:alert(1)",
            "file:///etc/passwd",
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
        ]
        for bad_scheme in unsupported:
            res_scheme = parser._build_apple_maps_url(query=bad_scheme, name="Target")
            self.assertTrue(res_scheme.startswith("https://maps.apple.com/place?"))
            self.assertNotIn("javascript:", res_scheme)
            self.assertNotIn("file:", res_scheme)
            self.assertNotIn("data:", res_scheme)

        # 6. Full parser streaming integration test
        xml_feed = '<map query="https://maps.apple.com.attacker.com/pwn?q=Shanghai+Disney" name="Disney" />'
        bubbles = parser.feed(xml_feed) + parser.finish()
        self.assertEqual(len(bubbles), 1)
        self.assertNotIn("attacker.com", bubbles[0])
        self.assertTrue(bubbles[0].startswith("https://maps.apple.com/place?"))

    def test_vuln06_vcard_and_icalendar_crlf_and_delimiter_injection(self) -> None:
        """VULN-06: vCard and iCalendar generators prevent CRLF and delimiter injection while preserving Unicode."""
        parser = BubbleParser()

        # --- vCard (RFC 2426) Injection and Escaping ---
        vcf_path = parser._generate_vcf(
            name="Alice Smith\r\nTEL;TYPE=CELL:99999999999\r\nBEGIN:VCARD",
            phone="138-0000-0000\r\nTEL;TYPE=HOME:11111",
            email="alice@company.com\r\nBCC:injected@evil.com",
            org="Acme, Corp; Division\\East\r\nInjectedOrg",
            title="Senior Engineer; Architect\r\nInjectedTitle",
            address="100 Silicon Way; Suite 200\r\nTech Park, CA",
            note="Meeting notes:\r\n- Item 1; with, details\\specs\r\n- Item 2",
        )
        self.assertIsNotNone(vcf_path)
        self.assertTrue(os.path.exists(vcf_path))

        with open(vcf_path, "r", encoding="utf-8") as f:
            vcf_content = f.read()

        vcf_lines = [line.strip() for line in vcf_content.splitlines() if line.strip()]

        # Exactly ONE BEGIN:VCARD and ONE END:VCARD block line
        begin_lines = [l for l in vcf_lines if l == "BEGIN:VCARD"]
        end_lines = [l for l in vcf_lines if l == "END:VCARD"]
        self.assertEqual(len(begin_lines), 1, "Must not inject multiple BEGIN:VCARD lines")
        self.assertEqual(len(end_lines), 1, "Must not inject multiple END:VCARD lines")

        # Single-line fields must NOT contain raw newlines or injected extra property lines
        tel_lines = [l for l in vcf_lines if l.startswith("TEL")]
        self.assertEqual(len(tel_lines), 1, f"Multiple TEL lines injected: {tel_lines}")
        self.assertNotIn("11111", tel_lines[0], "Injected phone line must be dropped/sanitized")

        email_lines = [l for l in vcf_lines if l.startswith("EMAIL")]
        self.assertEqual(len(email_lines), 1, f"Multiple EMAIL lines injected: {email_lines}")
        self.assertNotIn("injected@evil.com", email_lines[0], "Injected BCC email must be sanitized")

        # Delimiter escaping in vCard text (semicolons, commas, backslashes)
        org_lines = [l for l in vcf_lines if l.startswith("ORG")]
        self.assertEqual(len(org_lines), 1)
        self.assertIn(r"\;", org_lines[0], "Semicolon in ORG must be escaped")
        self.assertIn(r"\,", org_lines[0], "Comma in ORG must be escaped")
        self.assertIn(r"\\", org_lines[0], "Backslash in ORG must be escaped")

        # Multiline properties (NOTE) preserve line breaks via \n escape
        note_lines = [l for l in vcf_lines if l.startswith("NOTE")]
        self.assertEqual(len(note_lines), 1)
        self.assertIn(r"\n", note_lines[0], "Newlines in NOTE must be RFC-escaped as \\n")
        self.assertIn(r"\;", note_lines[0], "Semicolons in NOTE must be RFC-escaped")
        self.assertIn(r"\,", note_lines[0], "Commas in NOTE must be RFC-escaped")
        self.assertIn(r"\\", note_lines[0], "Backslashes in NOTE must be RFC-escaped")

        # --- iCalendar (RFC 5545) Injection and Escaping ---
        ics_path = parser._generate_ics(
            title="Q4 Sprint Review\r\nBEGIN:VEVENT\r\nSUMMARY:Pwned Event",
            start_str="2026-11-25 14:00:00",
            end_str="2026-11-25 15:00:00",
            location="Room 3B; Building 2, Floor 1\\HQ\r\nInjected Location",
            description="Agenda:\r\n1. Metrics; KPIs, Charts\\Data\r\n2. Next steps\r\nBEGIN:VALARM\r\nACTION:NONE",
        )
        self.assertIsNotNone(ics_path)
        self.assertTrue(os.path.exists(ics_path))

        with open(ics_path, "r", encoding="utf-8") as f:
            ics_content = f.read()

        # Unfold lines per RFC 5545 §3.1 before inspecting logical property lines
        unfolded_ics = re.sub(r"\r\n[ \t]", "", ics_content)
        ics_lines = [line.strip() for line in unfolded_ics.splitlines() if line.strip()]

        # Exactly ONE VEVENT block
        vevent_begin = [l for l in ics_lines if l == "BEGIN:VEVENT"]
        vevent_end = [l for l in ics_lines if l == "END:VEVENT"]
        self.assertEqual(len(vevent_begin), 1, "Must not inject additional VEVENT blocks")
        self.assertEqual(len(vevent_end), 1, "Must not inject additional END:VEVENT lines")

        # SUMMARY must be single-line with collapsed CRLF
        summary_lines = [l for l in ics_lines if l.startswith("SUMMARY")]
        self.assertEqual(len(summary_lines), 1)
        self.assertNotIn("\n", summary_lines[0].replace(r"\n", ""))

        # LOCATION must escape delimiters and collapse newlines
        loc_lines = [l for l in ics_lines if l.startswith("LOCATION")]
        self.assertEqual(len(loc_lines), 1)
        self.assertIn(r"\;", loc_lines[0])
        self.assertIn(r"\,", loc_lines[0])
        self.assertIn(r"\\", loc_lines[0])

        # DESCRIPTION must escape newlines as \n and escape delimiters
        desc_lines = [
            l for l in ics_lines
            if l.startswith("DESCRIPTION") and not l.startswith("DESCRIPTION:提醒") and not l.startswith("DESCRIPTION;CHARSET=UTF-8:提醒")
        ]
        self.assertEqual(len(desc_lines), 1)
        self.assertIn(r"\n", desc_lines[0])
        self.assertIn(r"\;", desc_lines[0])
        self.assertIn(r"\,", desc_lines[0])
        self.assertIn(r"\\", desc_lines[0])

        # --- Unicode Integrity in both formats ---
        u_vcf = parser._generate_vcf(name="王小明 🚀", org="北京未来科技有限公司", note="测试中文字符与符号：【】「」")
        with open(u_vcf, "r", encoding="utf-8") as f:
            u_vcf_text = f.read()
        self.assertIn("王小明 🚀", u_vcf_text)
        self.assertIn("北京未来科技有限公司", u_vcf_text)
        self.assertIn("测试中文字符与符号：【】「」", u_vcf_text)

        u_ics = parser._generate_ics(title="年度战略研讨会 🎯", start_str="2026-11-25 09:00:00", location="西湖国际会议中心", description="战略目标讨论：\n- 目标 1")
        with open(u_ics, "r", encoding="utf-8") as f:
            u_ics_text = f.read()
        self.assertIn("年度战略研讨会 🎯", u_ics_text)
        self.assertIn("西湖国际会议中心", u_ics_text)


if __name__ == "__main__":
    unittest.main()
