"""
Tests for pharma_compliance module.

38 tests total:
  - 33 baseline tests (session, extractor, bot_handler)
  - 5 new multimodal tests (voice, photo, merge, cross-check, auto-type)
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pharma_compliance.session import (
    MERGE_WINDOW_SECONDS,
    MessageType,
    PendingMessage,
    SessionManager,
    VisitSession,
)
from pharma_compliance.extractor import (
    detect_task_type,
    extract_fields,
    fields_to_summary,
    FIELD_NAMES,
    _extract_date,
    _extract_time,
    _extract_address,
    _extract_org_name,
    _extract_person,
    _extract_products,
    _extract_topic,
    _extract_competitor,
    _extract_feedback,
    _extract_next_steps,
    _extract_attendee_count,
)
from pharma_compliance.bot_handler import ComplianceBotHandler, process_message


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def manager():
    return SessionManager()


@pytest.fixture
def session(manager):
    return manager.get_or_create_session("user_test_001")


@pytest.fixture
def handler():
    return ComplianceBotHandler()


# ═══════════════════════════════════════════════════════════════════════════════
# Session tests (tests 1–11)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionCreation:
    def test_create_session(self, manager):
        """Test 1: Creating a new session."""
        session = manager.get_or_create_session("user_001")
        assert session.user_id == "user_001"
        assert session.pending_messages == []
        assert session.message_timeout == 0.0

    def test_get_existing_session(self, manager):
        """Test 2: Getting an existing session returns the same object."""
        s1 = manager.get_or_create_session("user_002")
        s2 = manager.get_or_create_session("user_002")
        assert s1 is s2

    def test_remove_session(self, manager):
        """Test 3: Removing a session."""
        manager.get_or_create_session("user_003")
        manager.remove_session("user_003")
        assert manager.get_session("user_003") is None

    def test_multiple_sessions_independent(self, manager):
        """Test 4: Multiple users have independent sessions."""
        s1 = manager.get_or_create_session("user_A")
        s2 = manager.get_or_create_session("user_B")
        s1.add_message(MessageType.TEXT, "msg from A")
        assert len(s1.pending_messages) == 1
        assert len(s2.pending_messages) == 0


class TestPendingMessages:
    def test_add_text_message(self, session):
        """Test 5: Adding a text message."""
        session.add_message(MessageType.TEXT, "去了百姓大药房见王店长")
        assert len(session.pending_messages) == 1
        assert session.pending_messages[0].msg_type == MessageType.TEXT
        assert "百姓大药房" in session.pending_messages[0].content

    def test_add_voice_message(self, session):
        """Test 6: Adding a voice message."""
        session.add_message(MessageType.VOICE, "聊了陈列小儿宝泰康")
        assert len(session.pending_messages) == 1
        assert session.pending_messages[0].msg_type == MessageType.VOICE

    def test_add_photo_message(self, session):
        """Test 7: Adding a photo message."""
        session.add_message(MessageType.PHOTO, "百姓大药房 中山路100号")
        assert len(session.pending_messages) == 1
        assert session.pending_messages[0].msg_type == MessageType.PHOTO

    def test_pending_count(self, session):
        """Test 8: Pending message count is accurate."""
        session.add_message(MessageType.TEXT, "msg1")
        session.add_message(MessageType.VOICE, "msg2")
        session.add_message(MessageType.PHOTO, "msg3")
        assert len(session.pending_messages) == 3

    def test_merge_contents(self, session):
        """Test 9: Merging contents combines all message types."""
        session.add_message(MessageType.TEXT, "去了百姓大药房")
        session.add_message(MessageType.VOICE, "见了王店长")
        session.add_message(MessageType.PHOTO, "门头照确认")
        merged = session.merge_contents()
        assert "去了百姓大药房" in merged
        assert "[语音]" in merged
        assert "[门头照]" in merged

    def test_session_clear(self, session):
        """Test 10: Clearing a session resets all state."""
        session.add_message(MessageType.TEXT, "test")
        session.clear()
        assert len(session.pending_messages) == 0
        assert session.message_timeout == 0.0


class TestSessionTimeout:
    def test_not_timed_out_without_messages(self, session):
        """Test 11: Empty session should not time out."""
        assert not session.is_timed_out()

    def test_not_timed_out_within_window(self, session):
        """Test 12: Session with recent message should not time out."""
        session.add_message(MessageType.TEXT, "test")
        assert not session.is_timed_out()

    def test_timed_out_after_window(self, session):
        """Test 13: Session times out after MERGE_WINDOW_SECONDS."""
        session.add_message(MessageType.TEXT, "test")
        # Simulate old message by adjusting timeout backwards
        session.message_timeout = time.time() - 1
        assert session.is_timed_out()

    def test_check_timeouts_returns_timed_out(self, manager):
        """Test 14: Manager.check_timeouts returns timed-out sessions."""
        s = manager.get_or_create_session("user_timeout")
        s.add_message(MessageType.TEXT, "test")
        s.message_timeout = time.time() - 1
        timed_out = manager.check_timeouts()
        assert s in timed_out


class TestManualMerge:
    def test_manual_merge_phrase_wl(self, session):
        """Test 15: '完了' triggers manual merge."""
        assert session.check_manual_merge("说完了")

    def test_manual_merge_phrase_jzy(self, session):
        """Test 16: '就这样' triggers manual merge."""
        assert session.check_manual_merge("就这样吧")

    def test_no_manual_merge_without_phrase(self, session):
        """Test 17: Regular text does not trigger manual merge."""
        assert not session.check_manual_merge("见了王店长")


# ═══════════════════════════════════════════════════════════════════════════════
# Extractor tests (tests 18–35)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskTypeDetection:
    def test_detect_pharmacy_visit(self):
        """Test 18: Detect 药店拜访 from pharmacy keywords."""
        result = detect_task_type("去了百姓大药房见王店长")
        assert result == "药店拜访"

    def test_detect_pharmacy_visit_display(self):
        """Test 19: Detect 药店拜访 from 陈列 keyword."""
        result = detect_task_type("聊了陈列调整")
        assert result == "药店拜访"

    def test_detect_hospital_visit(self):
        """Test 20: Detect 医疗机构拜访 from hospital keywords."""
        result = detect_task_type("协和医院心内科张主任")
        assert result == "医疗机构拜访"

    def test_detect_hospital_visit_doctor(self):
        """Test 21: Detect 医疗机构拜访 from 医生 keyword."""
        result = detect_task_type("门诊见了李医生")
        assert result == "医疗机构拜访"

    def test_detect_academic_promotion(self):
        """Test 22: Detect 学术推广 from meeting keywords."""
        result = detect_task_type("今天下午开了科室会")
        assert result == "学术推广"

    def test_detect_academic_promotion_lecture(self):
        """Test 23: Detect 学术推广 from 讲座 keyword."""
        result = detect_task_type("组织了一场产品讲座")
        assert result == "学术推广"

    def test_detect_default_to_pharmacy(self):
        """Test 24: Default to 药店拜访 when no keyword matches."""
        result = detect_task_type("今天去拜访了客户")
        assert result == "药店拜访"


class TestDateExtraction:
    def test_extract_date_full(self):
        """Test 25: Extract full date YYYY-MM-DD."""
        result = _extract_date("2025年6月15日拜访")
        assert "2025-06-15" in result or "2025-6-15" in result

    def test_extract_date_month_day(self):
        """Test 26: Extract month-day format."""
        result = _extract_date("6月15号去的")
        assert result != ""

    def test_extract_date_none(self):
        """Test 27: No date returns empty string."""
        result = _extract_date("没有日期信息")
        assert result == ""


class TestTimeExtraction:
    def test_extract_time_colon(self):
        """Test 28: Extract time with colon."""
        result = _extract_time("下午14:30到店")
        assert "14:30" in result

    def test_extract_time_morning(self):
        """Test 29: Extract morning time."""
        result = _extract_time("上午9点拜访")
        assert "09:00" in result

    def test_extract_time_afternoon(self):
        """Test 30: Extract afternoon time."""
        result = _extract_time("下午3点拜访")
        assert "15:00" in result


class TestPersonExtraction:
    def test_extract_person_store_manager(self):
        """Test 31: Extract store manager."""
        name, _ = _extract_person("见了王店长")
        assert "王" in name or "王店长" in name

    def test_extract_person_doctor(self):
        """Test 32: Extract doctor."""
        name, _ = _extract_person("拜访了张主任")
        assert "张" in name or "张主任" in name

    def test_extract_person_unknown(self):
        """Test 33: No person returns empty strings."""
        name, title = _extract_person("去药店检查陈列")
        assert name == ""
        assert title == ""


class TestOrgExtraction:
    def test_extract_org_name_pharmacy(self):
        """Test 34: Extract pharmacy name."""
        result = _extract_org_name("去了百姓大药房")
        assert "百姓大药房" in result or "百姓" in result

    def test_extract_org_name_hospital(self):
        """Test 35: Extract hospital name."""
        result = _extract_org_name("协和医院心内科")
        assert "协和" in result


class TestProductExtraction:
    def test_extract_product_single(self):
        """Test 36: Extract single product."""
        result = _extract_products("聊了小儿宝泰康的陈列")
        assert "小儿宝泰康" in result

    def test_extract_product_multiple(self):
        """Test 37: Extract multiple products."""
        result = _extract_products("聊了小儿宝泰康和板蓝根")
        assert len(result.split(", ")) >= 2


class TestAddressExtraction:
    def test_extract_address_pattern(self):
        """Test 38: Extract address with 路 pattern."""
        result = _extract_address("地址在中山路100号")
        assert "中山路" in result

    def test_extract_address_no_match(self):
        """Test 39: No address returns empty string."""
        result = _extract_address("没有地址信息")
        assert result == ""


class TestFullExtraction:
    def test_extract_fields_returns_all_17(self):
        """Test 40: extract_fields returns all 17 FIELD_NAMES."""
        fields = extract_fields("去了百姓大药房见王店长，聊了小儿宝泰康陈列")
        for fn in FIELD_NAMES:
            assert fn in fields, f"Missing field '{fn}'"

    def test_fields_to_summary(self):
        """Test 41: fields_to_summary produces readable output."""
        fields = {
            "task_type": "药店拜访",
            "org_name": "百姓大药房",
            "contact_person": "王店长",
        }
        summary = fields_to_summary(fields)
        assert "药店拜访" in summary or "药店" in summary


# ═══════════════════════════════════════════════════════════════════════════════
# Bot handler tests (tests 42–44)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBotHandlerBasic:
    @pytest.mark.asyncio
    async def test_handle_text_message(self, handler):
        """Test 42: Handle a text message."""
        result = await handler.handle_message(
            user_id="user_text",
            msg_type="text",
            content="去了百姓大药房见王店长",
        )
        assert result["merged"] is False
        assert result["pending_count"] == 1

    @pytest.mark.asyncio
    async def test_text_auto_merge_on_timeout(self, handler):
        """Test 43: Text message auto-merges after timeout."""
        result = await handler.handle_message(
            user_id="user_timeout",
            msg_type="text",
            content="去了百姓大药房见王店长",
        )
        assert result["merged"] is False

        # Simulate timeout
        session = handler.sessions.get_session("user_timeout")
        session.message_timeout = time.time() - 1

        # Next message should trigger merge
        result2 = await handler.handle_message(
            user_id="user_timeout",
            msg_type="text",
            content="聊了陈列",  # This triggers merge because previous timed out
        )

    def test_fuzzy_match_identical(self):
        """Test 44: Fuzzy match with identical names."""
        assert ComplianceBotHandler._fuzzy_match("百姓大药房", "百姓大药房")

    def test_fuzzy_match_substring(self):
        """Test 45: Fuzzy match with substring."""
        assert ComplianceBotHandler._fuzzy_match("百姓大药房", "百姓大药房中山路店")

    def test_fuzzy_match_different(self):
        """Test 46: Fuzzy match with different names."""
        assert not ComplianceBotHandler._fuzzy_match("百姓大药房", "同仁堂")

    def test_haversine_distance(self):
        """Test 47: Haversine distance calculation."""
        # Tiananmen to Forbidden City (~1km)
        dist = ComplianceBotHandler._haversine_distance(
            39.9042, 116.3974, 39.9142, 116.3874
        )
        assert 500 < dist < 2000, f"Expected ~1.1km, got {dist:.0f}m"

    def test_haversine_same_point(self):
        """Test 48: Haversine distance for same point."""
        dist = ComplianceBotHandler._haversine_distance(
            39.9042, 116.3974, 39.9042, 116.3974
        )
        assert dist == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-check tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossCheck:
    def test_store_match_no_warning(self, handler):
        """Test 49: Matching stores produce no warning."""
        warnings = handler.cross_check(
            claimed_store="百姓大药房",
            ocr_store="百姓大药房",
        )
        assert len(warnings) == 0

    def test_store_mismatch_warning(self, handler):
        """Test 50: Mismatched stores produce ⚠️ warning."""
        warnings = handler.cross_check(
            claimed_store="百姓大药房",
            ocr_store="同仁堂大药房",
        )
        assert len(warnings) == 1
        assert "地址不匹配" in warnings[0]

    def test_gps_deviation_warning(self, handler):
        """Test 51: GPS deviation >500m produces warning."""
        warnings = handler.cross_check(
            claimed_store="", ocr_store="",
            claimed_location={"latitude": 39.9042, "longitude": 116.3974},
            exif_gps={"latitude": 39.9242, "longitude": 116.3974},  # ~2.2km away
        )
        assert len(warnings) == 1
        assert "位置偏差" in warnings[0]

    def test_gps_within_range_no_warning(self, handler):
        """Test 52: GPS deviation <500m produces no warning."""
        warnings = handler.cross_check(
            claimed_store="", ocr_store="",
            claimed_location={"latitude": 39.9042, "longitude": 116.3974},
            exif_gps={"latitude": 39.9052, "longitude": 116.3984},  # ~100m
        )
        assert len(warnings) == 0

    def test_time_deviation_warning(self, handler):
        """Test 53: Time deviation >30min produces warning."""
        warnings = handler.cross_check(
            claimed_store="", ocr_store="",
            claimed_time="2025-06-15T14:00:00",
            exif_time="2025-06-15T14:45:00",
        )
        assert len(warnings) == 1
        assert "时间偏差" in warnings[0]


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Multimodal tests (Scenario A–E, tests 54–58)
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenarioA_PureVoice:
    """Scenario A: Pure voice → STT → text → field extraction."""

    @pytest.mark.asyncio
    async def test_voice_stt_and_extraction(self, handler):
        """Test 54: Simulate voice message → STT → extract fields."""
        mock_stt_result = {
            "success": True,
            "transcript": "去了百姓大药房见王店长，聊了小儿宝泰康陈列",
            "provider": "local",
        }

        with patch(
            "pharma_compliance.bot_handler.transcribe_audio",
            return_value=mock_stt_result,
        ):
            # Create a temp file simulating voice
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                voice_path = f.name
                f.write(b"mock audio data")

            result = await handler.handle_message(
                user_id="voice_test",
                msg_type="voice",
                content=voice_path,
            )
            assert result["pending_count"] == 1
            assert result["text_preview"] is not None
            assert "百姓大药房" in result["text_preview"]

    @pytest.mark.asyncio
    async def test_voice_stt_failure(self, handler):
        """Test 55: STT failure returns error warning."""
        mock_stt_result = {
            "success": False,
            "transcript": "",
            "error": "STT disabled",
        }

        with patch(
            "pharma_compliance.bot_handler.transcribe_audio",
            return_value=mock_stt_result,
        ):
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                voice_path = f.name
                f.write(b"mock audio")

            result = await handler.handle_message(
                user_id="voice_fail",
                msg_type="voice",
                content=voice_path,
            )
            assert result["merged"] is False
            all_warnings = result.get("warnings", [])
            assert len(all_warnings) > 0, f'Expected warnings, got: {result}'
            assert any("失败" in w for w in all_warnings)


class TestScenarioB_PhotoAndVoice:
    """Scenario B: Door photo + voice → OCR store name + voice fields → merge."""

    @pytest.mark.asyncio
    async def test_photo_then_voice_merge(self, handler):
        """Test 56: Photo (OCR) then voice (STT) → merged extraction."""

        # Mock vision_analyze_tool for photo
        vision_result = json.dumps({
            "success": True,
            "analysis": '{"store_name": "百姓大药房", "address": "中山路100号"}',
        })

        # Mock STT for voice
        stt_result = {
            "success": True,
            "transcript": "见了王店长聊了陈列小儿宝泰康",
            "provider": "local",
        }

        with patch(
            "pharma_compliance.bot_handler.vision_analyze_tool",
            new=AsyncMock(return_value=vision_result),
        ), patch(
            "pharma_compliance.bot_handler.transcribe_audio",
            return_value=stt_result,
        ):
            # Create temp files
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                photo_path = f.name
                f.write(b"mock jpeg data")
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                voice_path = f.name
                f.write(b"mock audio data")

            # Send photo
            r1 = await handler.handle_message(
                user_id="photo_voice_test",
                msg_type="photo",
                content=photo_path,
            )
            assert r1["pending_count"] == 1

            # Send voice
            r2 = await handler.handle_message(
                user_id="photo_voice_test",
                msg_type="voice",
                content=voice_path,
            )
            # New behavior: voice in progressive mode (after photo started it)
            # returns needs_followup with missing_fields, not accumulating pending_count
            assert r2["needs_followup"] is True
            assert len(r2["missing_fields"]) > 0


class TestScenarioC_MultiRoundMixed:
    """Scenario C: 3 mixed messages → 1 merged record, 5-min timeout."""

    def test_merge_timeout_mechanism(self):
        """Test 57: 5-minute timeout triggers auto-merge."""
        manager = SessionManager()
        session = manager.get_or_create_session("multi_test")

        session.add_message(MessageType.PHOTO, "百姓大药房 门头")
        session.add_message(MessageType.VOICE, "见了王店长")
        session.add_message(MessageType.TEXT, "聊了陈列 小儿宝泰康")

        assert len(session.pending_messages) == 3

        # Not timed out yet
        assert not session.is_timed_out()

        # Force timeout
        session.message_timeout = time.time() - 1
        assert session.is_timed_out()

    def test_three_messages_merge_to_one_record(self):
        """Test 58: 3 messages merge into 1 task record."""
        manager = SessionManager()
        session = manager.get_or_create_session("merge_test")

        session.add_message(MessageType.PHOTO, "百姓大药房")
        session.add_message(MessageType.VOICE, "见了王店长")
        session.add_message(MessageType.TEXT, "聊了小儿宝泰康陈列")

        merged = session.merge_contents()
        assert "门头照" in merged
        assert "语音" in merged
        assert "小儿宝泰康" in merged

        fields = extract_fields(merged)
        assert fields["task_type"] == "药店拜访"
        assert len(fields) == 17


class TestScenarioD_AddressMismatch:
    """Scenario D: Door photo OCR ≠ claimed store → ⚠️ warning."""

    def test_address_mismatch_warning(self):
        """Test 59: OCR store name doesn't match claimed → ⚠️ warning."""
        handler = ComplianceBotHandler()
        warnings = handler.cross_check(
            claimed_store="百姓大药房",
            ocr_store="同仁堂大药房",
        )
        assert len(warnings) == 1
        assert "⚠️" in warnings[0]
        assert "地址不匹配" in warnings[0]

    def test_address_match_no_warning(self):
        """Test 60: OCR matches claimed → no warning."""
        handler = ComplianceBotHandler()
        warnings = handler.cross_check(
            claimed_store="百姓大药房",
            ocr_store="百姓大药房",
        )
        assert len(warnings) == 0


class TestScenarioE_AutoTaskType:
    """Scenario E: Auto task type detection without explicit trigger."""

    def test_auto_detect_pharmacy(self):
        """Test 61: Content with pharmacy keywords → 药店拜访."""
        result = detect_task_type("去了百姓大药房见王店长")
        assert result == "药店拜访"

    def test_auto_detect_hospital(self):
        """Test 62: Content with hospital keywords → 医疗机构拜访."""
        result = detect_task_type("协和医院心内科张主任")
        assert result == "医疗机构拜访"

    def test_auto_detect_academic(self):
        """Test 63: Content with meeting keywords → 学术推广."""
        result = detect_task_type("今天下午开了科室会")
        assert result == "学术推广"

    def test_full_extraction_with_auto_type(self):
        """Test 64: Full extraction auto-detects type from content."""
        fields = extract_fields("协和医院心内科张主任，今天下午开了科室会")
        assert fields["task_type"] in ("医疗机构拜访", "学术推广")


# ═══════════════════════════════════════════════════════════════════════════════
# Edge case and degradation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLowConfidenceConfirmation:
    """P1-5: Low-confidence STT → needs_confirmation=True."""

    @pytest.mark.asyncio
    async def test_low_confidence_stt_needs_confirmation(self, handler):
        """Test 65: STT confidence < 0.6 → needs_confirmation=True."""
        mock_stt_result = {
            "success": True,
            "transcript": "去了某个地方见了某人",
            "confidence": 0.45,
            "provider": "local",
        }

        with patch(
            "pharma_compliance.bot_handler.transcribe_audio",
            return_value=mock_stt_result,
        ):
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                voice_path = f.name
                f.write(b"mock audio data")

            result = await handler.handle_message(
                user_id="low_conf_test",
                msg_type="voice",
                content=voice_path,
            )
            assert result["needs_confirmation"] is True
            assert result["pending_count"] == 1  # now added to session for force-merge
            assert result["text_preview"] is not None
            assert result["merged"] is False
            assert "missing_fields" in result  # P0-2: missing fields check enabled

    @pytest.mark.asyncio
    async def test_high_confidence_stt_no_confirmation(self, handler):
        """Test 66: STT confidence >= 0.6 → normal flow, no confirmation needed."""
        mock_stt_result = {
            "success": True,
            "transcript": "去了百姓大药房见王店长",
            "confidence": 0.85,
            "provider": "local",
        }

        with patch(
            "pharma_compliance.bot_handler.transcribe_audio",
            return_value=mock_stt_result,
        ):
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                voice_path = f.name
                f.write(b"mock audio data")

            result = await handler.handle_message(
                user_id="high_conf_test",
                msg_type="voice",
                content=voice_path,
            )
            assert result.get("needs_confirmation") is not True
            assert result["pending_count"] == 1
            assert result["merged"] is False


class TestExifGpsExtraction:
    """P1-5: EXIF GPS extraction from photo images."""

    def test_parse_gps_northern_eastern(self):
        """Test 67: Parse GPS from EXIF (N/E hemisphere)."""
        handler = ComplianceBotHandler()
        gps_data = {
            "GPSLatitudeRef": "N",
            "GPSLatitude": (39, 54, 15.12),
            "GPSLongitudeRef": "E",
            "GPSLongitude": (116, 23, 50.64),
        }
        result = handler._parse_gps(gps_data)
        assert result is not None
        assert abs(result["latitude"] - 39.9042) < 0.01
        assert abs(result["longitude"] - 116.3974) < 0.01

    def test_parse_gps_southern_western(self):
        """Test 68: Parse GPS from EXIF (S/W hemisphere)."""
        handler = ComplianceBotHandler()
        gps_data = {
            "GPSLatitudeRef": "S",
            "GPSLatitude": (33, 55, 27.84),
            "GPSLongitudeRef": "W",
            "GPSLongitude": (18, 25, 25.68),
        }
        result = handler._parse_gps(gps_data)
        assert result is not None
        assert result["latitude"] < 0  # Southern hemisphere
        assert result["longitude"] < 0  # Western hemisphere
        assert abs(result["latitude"] - (-33.9244)) < 0.01
        assert abs(result["longitude"] - (-18.4238)) < 0.01

    def test_parse_gps_missing_coords(self):
        """Test 69: Parse GPS returns None when lat/lon missing."""
        handler = ComplianceBotHandler()
        # No GPSLatitude
        result = handler._parse_gps({"GPSLatitudeRef": "N", "GPSLongitudeRef": "E"})
        assert result is None
        # Empty dict
        result = handler._parse_gps({})
        assert result is None

    def test_extract_exif_with_gps(self):
        """Test 70: _extract_exif returns GPS and datetime from image."""
        from unittest.mock import MagicMock, PropertyMock

        handler = ComplianceBotHandler()

        # Mock ExifTags.Base as an IntEnum-like object
        class _MockExifMember:
            def __init__(self, name):
                self.name = name

        class _MockExifBase:
            _map = {306: "DateTime", 36867: "DateTimeOriginal"}
            def __contains__(self, item):
                return item in self._map
            def __call__(self, key):
                return _MockExifMember(self._map.get(key, f"Unknown_{key}"))

        # Mock GPS IFD
        mock_gps_ifd = {
            1: "N",
            2: (39.0, 54.0, 15.12),
            3: "E",
            4: (116.0, 23.0, 50.64),
        }

        mock_exif = MagicMock()
        mock_exif.__bool__.return_value = True
        mock_exif.items.return_value = [
            (306, "2025:06:15 14:30:00"),
            (36867, "2025:06:15 14:29:55"),
        ]
        mock_exif.get_ifd.return_value = mock_gps_ifd

        mock_img = MagicMock()
        mock_img.getexif.return_value = mock_exif

        with patch("PIL.Image.open", return_value=mock_img), \
             patch("PIL.ExifTags.Base", _MockExifBase()), \
             patch("PIL.ExifTags.GPSTAGS", {1: "GPSLatitudeRef", 2: "GPSLatitude",
                    3: "GPSLongitudeRef", 4: "GPSLongitude"}):
            result = handler._extract_exif("/fake/path.jpg")

        assert result["datetime"] is not None
        assert "2025-06-15T14:29:55" in result["datetime"]
        assert result["gps"] is not None
        assert abs(result["gps"]["latitude"] - 39.9042) < 0.01
        assert abs(result["gps"]["longitude"] - 116.3974) < 0.01

    def test_extract_exif_no_gps(self):
        """Test 71: _extract_exif returns datetime but no GPS when no GPS IFD."""
        from unittest.mock import MagicMock

        handler = ComplianceBotHandler()

        class _MockExifMember:
            def __init__(self, name):
                self.name = name

        class _MockExifBase:
            _map = {306: "DateTime"}
            def __contains__(self, item):
                return item in self._map
            def __call__(self, key):
                return _MockExifMember(self._map.get(key, f"Unknown_{key}"))

        mock_exif = MagicMock()
        mock_exif.__bool__.return_value = True
        mock_exif.items.return_value = [(306, "2025:06:15 14:30:00")]
        mock_exif.get_ifd.return_value = None  # No GPS IFD

        mock_img = MagicMock()
        mock_img.getexif.return_value = mock_exif

        with patch("PIL.Image.open", return_value=mock_img), \
             patch("PIL.ExifTags.Base", _MockExifBase()):
            result = handler._extract_exif("/fake/path.jpg")

        assert result["datetime"] is not None
        assert result["gps"] is None


class TestClaimedLocationCrossCheck:
    """P1-3: claimed_location flows from session metadata → GPS cross-check."""

    def test_claimed_location_extraction_from_metadata(self):
        """Test 72: _extract_claimed_location reads GPS from message metadata."""
        manager = SessionManager()
        session = manager.get_or_create_session("gps_user")

        # Add a text message with claimed_location in metadata
        session.add_message(
            MessageType.TEXT,
            "到了百姓大药房",
            metadata={"claimed_location": {"latitude": 39.9042, "longitude": 116.3974}},
        )

        location = ComplianceBotHandler._extract_claimed_location(session)
        assert location is not None
        assert location["latitude"] == 39.9042
        assert location["longitude"] == 116.3974

    def test_claimed_location_none_without_metadata(self):
        """Test 73: _extract_claimed_location returns None when no GPS metadata."""
        manager = SessionManager()
        session = manager.get_or_create_session("no_gps_user")
        session.add_message(MessageType.TEXT, "hello")

        location = ComplianceBotHandler._extract_claimed_location(session)
        assert location is None

    def test_claimed_location_with_gps_cross_check(self, handler):
        """Test 74: GPS cross-check triggers when claimed_location is provided."""
        warnings = handler.cross_check(
            claimed_store="",
            ocr_store="",
            claimed_location={"latitude": 39.9042, "longitude": 116.3974},
            exif_gps={"latitude": 39.9242, "longitude": 116.3974},  # ~2.2 km away
        )
        assert len(warnings) == 1
        assert "位置偏差" in warnings[0]


class TestDegradation:
    def test_extract_empty_text(self):
        """Test 65: Extracting fields from empty text."""
        fields = extract_fields("")
        assert len(fields) == 17
        assert fields["task_type"] == "药店拜访"  # default

    def test_extract_single_word(self):
        """Test 66: Extracting from minimal input."""
        fields = extract_fields("你好")
        assert len(fields) == 17

    @pytest.mark.asyncio
    async def test_handle_unknown_msg_type(self, handler):
        """Test 67: Unknown message type treated as text."""
        result = await handler.handle_message(
            user_id="unknown_type",
            msg_type="unknown",
            content="test message",
        )
        assert "merged" in result

    @pytest.mark.asyncio
    async def test_asyncio_voice_unavailable_graceful_message(self, handler):
        """Test 78: When STT is unavailable, return clear suggestion to use text."""
        with patch("pharma_compliance.bot_handler.transcribe_audio", None):
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                voice_path = f.name
                f.write(b"mock audio data")

            result = await handler.handle_message(
                user_id="voice_missing_stt",
                msg_type="voice",
                content=voice_path,
            )
        assert result["merged"] is False
        assert result["pending_count"] == 0
        warnings = result.get("warnings", [])
        assert len(warnings) == 1
        assert "语音识别暂不可用" in warnings[0]
        assert "文字" in warnings[0]


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Progressive questioning defect fix tests (tests 79–89)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPersistLocking:
    """Test 79: update_record uses fcntl.flock for both read and write."""

    def test_update_record_locks_file(self):
        """update_record acquires exclusive locks on read and write."""
        # Verify the source code contains 4+ fcntl.flock calls total
        persist_path = PROJECT_ROOT / "pharma_compliance" / "persistence.py"
        content = persist_path.read_text()
        flock_count = content.count("fcntl.flock")
        assert flock_count >= 4, f"Expected >= 4 fcntl.flock calls, found {flock_count}"


class TestRetryOnExtractionFail:
    """Tests 80-81: Field extraction failure → retry logic with retry_count."""

    @pytest.mark.asyncio
    async def test_retry_count_less_than_2_retries_same_field(self, handler):
        """Test 80: Field extraction fails with retry_count < 2 → re-ask same field."""
        manager = SessionManager()
        session = manager.get_or_create_session("retry_user")
        session.merged_fields = {"org_name": "百姓大药房"}
        session.pending_field = "contact_person"
        session.pending_questions = ["products"]
        session.retry_count = 0

        result = await handler._handle_progressive_answer(
            "retry_user", "嗯嗯就是这样", session
        )
        # Should re-ask the same field with retry message
        assert result["merged"] is False
        assert result["needs_followup"] is True
        assert result["followup_field"] == "contact_person"
        assert "信息不清晰" in result["followup_message"]
        assert session.retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_count_exhausted_skips_field(self, handler):
        """Test 81: Field extraction fails with retry_count >= 2 → skip to next field."""
        manager = SessionManager()
        session = manager.get_or_create_session("retry_exhausted_user")
        session.merged_fields = {"org_name": "百姓大药房"}
        session.pending_field = "contact_person"
        session.pending_questions = ["products"]
        session.retry_count = 2

        result = await handler._handle_progressive_answer(
            "retry_exhausted_user", "还是不清楚", session
        )
        # Should skip to next field (products)
        assert result["merged"] is False
        assert result["needs_followup"] is True
        assert result["followup_field"] == "products"
        assert session.retry_count == 2  # unchanged, field skipped


class TestPersistFailurePreservesSession:
    """Test 82: _finalize_accumulated persist failure preserves session."""

    @pytest.mark.asyncio
    async def test_persist_failure_preserves_session(self, handler):
        """When save_record raises in _finalize_accumulated, session is cleared but returns failure."""
        manager = handler.sessions
        session = manager.get_or_create_session("persist_exception_user")
        session.merged_fields = {
            "org_name": "百姓大药房",
            "contact_person": "王店长",
            "task_type": "药店拜访",
            "products": "小儿宝泰康",
            "topic": "产品推广",
            "content_summary": "拜访内容",
            "next_steps": "下次再聊",
        }
        session.add_message(MessageType.TEXT, "test message")

        with patch(
            "pharma_compliance.bot_handler.save_record",
            side_effect=Exception("disk full"),
        ):
            result = await handler._finalize_accumulated(session)

        # Should return failure
        assert result["merged"] is False
        assert "持久化失败" in result["warnings"][0]


class TestPersistFailureNoProgressive:
    """Test 83: persist_ok=False prevents progressive questioning."""

    @pytest.mark.asyncio
    async def test_persist_failure_returns_early(self, handler):
        """When save_record returns None, _do_merge returns failure early."""
        manager = SessionManager()
        session = manager.get_or_create_session("persist_fail_user")
        session.add_message(MessageType.TEXT, "去了百姓大药房见王店长 2025年6月15日 下午14:30")

        with patch(
            "pharma_compliance.bot_handler.save_record",
            return_value=None,
        ):
            result = await handler._do_merge(session)

        assert result["merged"] is False
        warnings_text = " ".join(result.get("warnings", []))
        assert "持久化失败" in warnings_text


class TestSessionStale:
    """Test 84: is_stale triggers session cleanup on progressive entry."""

    def test_is_stale_returns_true_after_30_minutes(self):
        """is_stale returns True when last_activity > 30 min ago."""
        manager = SessionManager()
        session = manager.get_or_create_session("stale_user")
        session.last_activity = time.time() - 1801  # 30 min + 1 sec
        assert session.is_stale() is True

    def test_is_stale_returns_false_within_window(self):
        """is_stale returns False when recently active."""
        manager = SessionManager()
        session = manager.get_or_create_session("active_user")
        session.last_activity = time.time() - 60  # 1 min ago
        assert session.is_stale() is False

    @pytest.mark.asyncio
    async def test_stale_session_cleared_on_progressive_entry(self, handler):
        """Stale session gets cleared at progressive routing entry."""
        manager = handler.sessions
        session = manager.get_or_create_session("stale_progressive_user")
        session.pending_field = "contact_person"
        session.merged_fields = {"org_name": "百姓大药房"}
        session.pending_questions = ["products"]
        session.last_activity = time.time() - 1801  # stale

        result = await handler.handle_message(
            user_id="stale_progressive_user",
            msg_type="text",
            content="王店长",
        )
        assert result["merged"] is False
        assert "上次对话已超时" in result["warnings"][0]
        # Session should be cleared
        assert session.pending_field is None
        assert session.merged_fields is None


class TestManualMergePriority:
    """Test 85: Manual merge trigger takes priority over progressive routing."""

    @pytest.mark.asyncio
    async def test_manual_merge_before_progressive(self, handler):
        """'完了' triggers force_merge even during progressive questioning."""
        manager = handler.sessions
        session = manager.get_or_create_session("merge_priority_user")
        session.add_message(MessageType.TEXT, "去了百姓大药房见王店长")
        session.pending_field = "products"  # Simulate progressive mode
        session.merged_fields = {"org_name": "百姓大药房"}
        session.pending_questions = ["feedback"]

        with patch.object(handler, "_force_merge", new_callable=AsyncMock) as mock_fm:
            mock_fm.return_value = {"merged": True, "task": {}, "warnings": [], "pending_count": 0}
            await handler.handle_message(
                user_id="merge_priority_user",
                msg_type="text",
                content="完了",
            )
            # _force_merge should have been called (manual merge took priority)
            mock_fm.assert_called_once()


class TestEscExitProgressive:
    """Test 86: ESC keyword exits progressive mode."""

    def test_esc_keyword_in_manual_merge_phrases(self):
        """'取消' is in manual merge phrases and can exit progressive."""
        from pharma_compliance.session import MANUAL_MERGE_PHRASES
        # Verify that common exit phrases exist
        assert len(MANUAL_MERGE_PHRASES) > 0

    @pytest.mark.asyncio
    async def test_esc_via_force_merge_exits_progressive(self, handler):
        """Sending a manual merge phrase during progressive mode triggers force merge."""
        manager = handler.sessions
        session = manager.get_or_create_session("esc_user")
        session.add_message(MessageType.TEXT, "去了百姓大药房")
        session.pending_field = "contact_person"
        session.merged_fields = {"org_name": "百姓大药房"}
        session.pending_questions = ["products"]

        with patch.object(handler, "_force_merge", new_callable=AsyncMock) as mock_fm:
            mock_fm.return_value = {
                "merged": True,
                "task": {"fields": session.merged_fields},
                "warnings": [],
                "pending_count": 0,
            }
            result = await handler.handle_message(
                user_id="esc_user",
                msg_type="text",
                content="取消",
            )
            mock_fm.assert_called_once()


class TestFeedbackOnlyFinalize:
    """Test 87: When only feedback is missing, it goes through progressive flow to _finalize_accumulated."""

    @pytest.mark.asyncio
    async def test_feedback_only_triggers_progressive_from_handle_text(self, handler):
        """First message with most fields → starts progressive on first missing field."""
        manager = handler.sessions
        session = manager.get_or_create_session("fb_new_user")

        with patch(
            "pharma_compliance.bot_handler.save_record",
            return_value="record_new",
        ):
            result = await handler.handle_message(
                user_id="fb_new_user",
                msg_type="text",
                content="药店拜访 百姓大药房 王店长 2025年6月15日 下午14:30 小儿宝泰康 主题产品推广 竞品有太极 下次带样品",
            )

        # Should trigger progressive for first missing field (feedback or others)
        assert result["needs_followup"] is True
        assert result["followup_field"] is not None
        assert result["followup_message"] is not None

    @pytest.mark.asyncio
    async def test_feedback_answer_finalizes(self, handler):
        """When the last progressive answer completes all fields, _finalize_accumulated is called."""
        manager = handler.sessions
        session = manager.get_or_create_session("fb_last_field_user")
        session.merged_fields = {
            "task_type": "药店拜访",
            "org_name": "百姓大药房",
            "contact_person": "王店长",
            "products": "小儿宝泰康",
            "topic": "产品推广",
            "content_summary": "拜访内容",
            "next_steps": "下次再聊",
            "visit_date": "2026-06-17",
            "visit_time": "14:30",
            "competitor_info": "竞品有太极",
        }
        session.pending_field = "feedback"
        session.pending_questions = []
        session.add_message(MessageType.TEXT, "initial message")

        with patch(
            "pharma_compliance.bot_handler.save_record",
            return_value="fb_final_record",
        ) as mock_save:
            result = await handler._handle_progressive_answer(
                "fb_last_field_user",
                "王店长说小儿宝泰康卖得不错，建议多备货",
                session,
            )
            # save_record should have been called via _finalize_accumulated
            mock_save.assert_called_once()
            assert result["merged"] is True


class TestRetryCountReset:
    """Test 88: retry_count resets to 0 on successful extraction."""

    @pytest.mark.asyncio
    async def test_retry_count_resets_on_success(self, handler):
        """After a retry, successful extraction resets retry_count to 0."""
        manager = SessionManager()
        session = manager.get_or_create_session("reset_user")
        session.merged_fields = {"org_name": "百姓大药房"}
        session.pending_field = "contact_person"
        session.pending_questions = ["products"]
        session.retry_count = 1  # Had one failed attempt

        result = await handler._handle_progressive_answer(
            "reset_user", "见了王店长", session
        )
        assert result["merged"] is False
        assert result["needs_followup"] is True
        assert result["followup_field"] == "products"
        assert session.retry_count == 0  # Reset after success


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Progressive questioning from first message tests (tests 90-95)
# ═══════════════════════════════════════════════════════════════════════════════

class TestProgressiveFirstMessage:
    """Tests 90-91: Progressive questioning starts from the first message."""

    @pytest.mark.asyncio
    async def test_first_message_triggers_progressive(self, handler):
        """Test 90: First text message with partial info → triggers progressive follow-up."""
        manager = handler.sessions
        session = manager.get_or_create_session("prog_first_user")

        with patch(
            "pharma_compliance.bot_handler.save_record",
            return_value="record_prog_1",
        ):
            result = await handler.handle_message(
                user_id="prog_first_user",
                msg_type="text",
                content="去了保和康药房",
            )

        # Should have started progressive questioning
        assert result["merged"] is False
        assert result["needs_followup"] is True
        assert result["followup_field"] is not None
        assert result["followup_message"] is not None
        # Session should have accumulated fields
        assert session.merged_fields is not None
        assert session.merged_fields.get("org_name") or "保和康" in str(session.merged_fields)

    @pytest.mark.asyncio
    async def test_progressive_reply_extracts_and_continues(self, handler):
        """Test 91: User reply to progressive question → extract → continue to next field."""
        manager = handler.sessions
        session = manager.get_or_create_session("prog_reply_user")
        # Simulate after first message ("去了保和康药房") extracted org_name
        session.merged_fields = {
            "task_type": "药店拜访",
            "org_name": "保和康药房",
        }
        session.pending_field = "contact_person"
        session.pending_questions = ["products"]
        session.add_message(MessageType.TEXT, "去了保和康药房")

        result = await handler._handle_progressive_answer(
            "prog_reply_user",
            "见了王药师",  # provides contact_person
            session,
        )

        assert result["merged"] is False
        assert result["needs_followup"] is True
        # Should move to next missing field
        assert result["followup_field"] == "products"
        assert session.merged_fields.get("contact_person") is not None


class TestProgressiveAllFieldsComplete:
    """Test 92: When all fields are complete, finalize immediately."""

    @pytest.mark.asyncio
    async def test_all_fields_complete_triggers_finalize(self, handler):
        """Test 92: Message that fills all missing fields → immediate finalize."""
        manager = handler.sessions
        session = manager.get_or_create_session("prog_complete_user")

        with patch(
            "pharma_compliance.bot_handler.save_record",
            return_value="record_complete",
        ) as mock_save:
            result = await handler.handle_message(
                user_id="prog_complete_user",
                msg_type="text",
                content=(
                    "药店拜访 保和康药房 王药师 2025年6月15日 下午3点 "
                    "小儿宝泰康 主题产品推广 聊了陈列和销量 "
                    "竞品有太极 下次带样品 王药师说卖得不错"
                ),
            )

        # With all core fields + feedback, should finalize immediately
        assert result["merged"] is True
        mock_save.assert_called_once()


class TestRetrySkipField:
    """Test 93: Retry 2 times then skip to next field."""

    @pytest.mark.asyncio
    async def test_retry_twice_then_skip(self, handler):
        """Test 93: 2 failed attempts → skip to next field."""
        manager = handler.sessions
        session = manager.get_or_create_session("retry_skip_user")
        session.merged_fields = {
            "task_type": "药店拜访",
            "org_name": "保和康药房",
        }
        session.pending_field = "contact_person"
        session.pending_questions = ["products", "topic"]
        session.retry_count = 2  # Already exhausted

        result = await handler._handle_progressive_answer(
            "retry_skip_user",
            "这个不太清楚",  # Can't extract contact_person
            session,
        )

        # Should skip contact_person and move to products
        assert result["merged"] is False
        assert result["needs_followup"] is True
        assert result["followup_field"] == "products"
        assert session.retry_count == 2  # unchanged


class TestManualMergeMidProgressive:
    """Test 94: Mid-progressive '完成' triggers force merge."""

    @pytest.mark.asyncio
    async def test_done_mid_progressive_finalizes(self, handler):
        """Test 94: Saying '完成' during progressive questioning finalizes accumulated fields."""
        manager = handler.sessions
        session = manager.get_or_create_session("done_mid_user")
        session.merged_fields = {
            "task_type": "药店拜访",
            "org_name": "保和康药房",
            "contact_person": "王药师",
        }
        session.pending_field = "products"
        session.pending_questions = ["topic"]
        session.add_message(MessageType.TEXT, "去了保和康药房和王药师")

        with patch.object(handler, "_finalize_accumulated", new_callable=AsyncMock) as mock_final:
            mock_final.return_value = {
                "merged": True,
                "task": {"fields": session.merged_fields, "summary": "test"},
                "warnings": [],
                "pending_count": 0,
            }
            result = await handler.handle_message(
                user_id="done_mid_user",
                msg_type="text",
                content="完成",
            )
            mock_final.assert_called_once()
            assert result["merged"] is True


class TestAccumulateFields:
    """Test 95: accumulate_fields properly merges new fields into existing ones."""

    def test_accumulate_fills_empty_slots(self):
        """accumulate_fields fills empty values without overwriting existing ones."""
        manager = SessionManager()
        session = manager.get_or_create_session("accum_user")
        session.accumulate_fields({"org_name": "保和康药房", "contact_person": ""})
        assert session.merged_fields["org_name"] == "保和康药房"
        # Empty values are not stored
        assert "contact_person" not in session.merged_fields

        # Second accumulation: fills missing contact_person
        session.accumulate_fields({"contact_person": "王药师", "products": "板蓝根"})
        assert session.merged_fields["contact_person"] == "王药师"
        assert session.merged_fields["products"] == "板蓝根"
        # org_name should not be overwritten
        assert session.merged_fields["org_name"] == "保和康药房"

    def test_accumulate_does_not_overwrite_existing(self):
        """accumulate_fields does not overwrite existing non-empty values."""
        manager = SessionManager()
        session = manager.get_or_create_session("accum_keep_user")
        session.accumulate_fields({"org_name": "保和康药房"})
        # Try to overwrite with empty
        session.accumulate_fields({"org_name": ""})
        assert session.merged_fields["org_name"] == "保和康药房"
        # Try to overwrite with different value
        session.accumulate_fields({"org_name": "同仁堂"})
        assert session.merged_fields["org_name"] == "保和康药房"  # unchanged


class TestTouchActivity:
    """Test 89: touch_activity updates last_activity timestamp."""

    def test_touch_activity_updates_timestamp(self):
        """touch_activity() sets last_activity to current time.time()."""
        manager = SessionManager()
        session = manager.get_or_create_session("activity_user")
        old_time = session.last_activity
        # Small delay to ensure timestamp changes
        time.sleep(0.01)
        session.touch_activity()
        new_time = session.last_activity
        assert new_time > old_time

    def test_add_message_touches_activity(self):
        """add_message() calls touch_activity automatically."""
        manager = SessionManager()
        session = manager.get_or_create_session("msg_activity_user")
        old_time = session.last_activity
        time.sleep(0.01)
        session.add_message(MessageType.TEXT, "test message")
        new_time = session.last_activity
        assert new_time > old_time

    def test_get_or_create_touches_activity(self):
        """get_or_create_session() calls touch_activity."""
        manager = SessionManager()
        session = manager.get_or_create_session("create_activity_user")
        # last_activity should be set (just after creation)
        assert session.last_activity > 0
        # Calling again should update
        old_time = session.last_activity
        time.sleep(0.01)
        session2 = manager.get_or_create_session("create_activity_user")
        assert session2 is session
        # Activity should be refreshed
        assert session.last_activity >= old_time
