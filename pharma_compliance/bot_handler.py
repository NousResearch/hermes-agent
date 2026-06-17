"""
QQbot message handler for pharma compliance.

Handles text, voice (STT), and photo (Vision+OCR+EXIF) messages.
Merges multimodal messages within a 5-minute window into one task record.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Lazy imports for Hermes modules (imported at module level so tests can patch)
try:
    from tools.transcription_tools import transcribe_audio
except ImportError:
    transcribe_audio = None

try:
    from tools.vision_tools import vision_analyze_tool
except ImportError:
    vision_analyze_tool = None

from pharma_compliance.session import (
    MERGE_WINDOW_SECONDS,
    MessageType,
    SessionManager,
    VisitSession,
)
from pharma_compliance.extractor import (
    extract_fields,
    fields_to_summary,
    get_missing_core_fields,
    CORE_FIELDS_PRIORITY,
    CORE_FIELD_LABELS,
    _extract_date,
    _extract_time,
    _extract_org_name,
    _extract_person,
    _extract_products,
    _extract_topic,
    _extract_feedback,
    _extract_next_steps,
    _extract_competitor,
    detect_task_type,
)
from pharma_compliance.persistence import save_record

logger = logging.getLogger(__name__)

# ── Progressive field questioning ───────────────────────────────────────────

# Priority order for progressive follow-up: date → time → core fields (deduped)
_PROGRESSIVE_FIELD_PRIORITY = ["visit_date", "visit_time"] + [
    f for f in CORE_FIELDS_PRIORITY if f not in ("visit_date",)
]

# Natural-language question templates for each field
_FIELD_QUESTION_TEMPLATES: Dict[str, str] = {
    "visit_date": "好的，记录下来了～请问这次拜访是什么时候去的呢？",
    "visit_time": "了解～具体是几点左右去的呢？",
    "task_type": "请问这次的任务类型是什么呢？（药店拜访 / 医疗机构拜访 / 学术推广）",
    "org_name": "请问拜访的是哪家药店或机构呢？",
    "contact_person": "请问拜访对象是哪位呢？",
    "products": "主要涉及了哪些产品呢？",
    "topic": "这次拜访主要聊了什么主题呢？",
    "content_summary": "能简单描述一下拜访内容吗？",
    "next_steps": "接下来有什么计划或下一步安排吗？",
    "feedback": "这次的拜访对象有什么具体的反馈吗？比如对产品的评价、销量变化、提出的建议等等。",
    "competitor_info": "有了解到什么竞品信息吗？",
}


def _get_field_question(field_name: str) -> Optional[str]:
    """Return a natural-language follow-up question for a given field."""
    return _FIELD_QUESTION_TEMPLATES.get(field_name)


def _extract_single_field(text: str, field_name: str) -> str:
    """Extract a single field value from user's reply text."""
    if not text or not text.strip():
        return ""
    if field_name == "visit_date":
        return _extract_date(text)
    elif field_name == "visit_time":
        return _extract_time(text)
    elif field_name == "org_name":
        return _extract_org_name(text)
    elif field_name == "contact_person":
        name, _ = _extract_person(text)
        return name
    elif field_name == "products":
        return _extract_products(text)
    elif field_name == "topic":
        return _extract_topic(text)
    elif field_name == "feedback":
        return _extract_feedback(text)
    elif field_name == "next_steps":
        return _extract_next_steps(text)
    elif field_name == "competitor_info":
        return _extract_competitor(text)
    elif field_name == "task_type":
        return detect_task_type(text)
    elif field_name == "content_summary":
        return text.strip()
    else:
        # Generic: use the raw text
        return text.strip()


class ComplianceBotHandler:
    """Handles incoming QQbot messages for pharma compliance tracking.

    Key flows:
    - Text → extract fields directly
    - Voice → download → STT → extract fields
    - Photo → download → Vision+OCR → EXIF → cross-check
    - Multi-message merge within 5-min window
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        on_task_complete: Optional[Callable] = None,
    ):
        self.sessions = session_manager or SessionManager()
        self.on_task_complete = on_task_complete
        self._merge_timers: Dict[str, asyncio.Task] = {}

    # ── Public entry point ──────────────────────────────────────────────────

    async def handle_message(
        self,
        user_id: str,
        msg_type: str,  # "text", "voice", "photo"
        content: str,   # text content or URL/path for voice/photo
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Handle an incoming message and return extracted results.

        Returns:
            dict with:
              - "merged": bool — whether a full task was completed
              - "task": dict or None — extracted fields if merged
              - "warnings": list[str] — any compliance warnings
              - "pending_count": int — messages still pending
              - "text_preview": str or None — preview text for user confirmation
        """
        warnings: List[str] = []
        metadata = metadata or {}
        # ── Check for stale session BEFORE touching activity ──
        # get_or_create_session calls touch_activity() which would reset last_activity,
        # so we must check staleness first using get_session (which does NOT touch activity).
        existing = self.sessions.get_session(user_id)
        if existing and (existing.pending_field is not None or (
            existing.merged_fields is not None and existing.pending_questions
        )):
            if existing.is_stale():
                existing.clear()
                return {
                    "merged": False,
                    "task": None,
                    "warnings": ["上次对话已超时，请重新开始"],
                    "pending_count": 0,
                    "text_preview": None,
                    "missing_fields": [],
                    "needs_followup": False,
                    "followup_field": None,
                    "followup_message": None,
                }
        session = self.sessions.get_or_create_session(user_id)

        # Check for manual merge trigger in text messages FIRST (takes priority over progressive routing)
        if msg_type == "text" and session.check_manual_merge(content):
            return await self._force_merge(session)

        # ── Progressive questioning: user is answering a field follow-up ──
        # Two scenarios trigger this path:
        # 1. session.pending_field is set → actively answering a progressive question
        # 2. merged_fields exists but no pending_field → answering feedback from _do_merge
        #    (feedback was asked in merge response, now transitioning to progressive)
        if session.pending_field is not None or (
            session.merged_fields is not None and session.pending_questions
        ):
            # Check if session has been inactive for too long
            if session.is_stale():
                session.clear()
                return {
                    "merged": False,
                    "task": None,
                    "warnings": ["上次对话已超时，请重新开始"],
                    "pending_count": 0,
                    "text_preview": None,
                    "missing_fields": [],
                    "needs_followup": False,
                    "followup_field": None,
                    "followup_message": None,
                }
            if msg_type != "text":
                # Non-text messages in progressive mode: briefly acknowledge
                return {
                    "merged": False,
                    "task": None,
                    "warnings": [],
                    "pending_count": 0,
                    "text_preview": None,
                    "missing_fields": session.pending_questions,
                    "needs_followup": True,
                    "followup_field": session.pending_field,
                    "followup_message": (
                        f"收到啦～不过刚才在问的「{CORE_FIELD_LABELS.get(session.pending_field or '信息', '信息')}」"
                        f"方便用文字回复一下吗？"
                    ),
                }
            return await self._handle_progressive_answer(user_id, content, session)

        try:
            if msg_type == "voice":
                result = await self._handle_voice(user_id, content, metadata, session)
            elif msg_type == "photo":
                result = await self._handle_photo(user_id, content, metadata, session)
            else:
                result = await self._handle_text(user_id, content, metadata, session)
        except Exception as e:
            logger.error("handle_message failed for user %s: %s", user_id, e)
            return {
                "merged": False,
                "task": None,
                "warnings": [f"处理失败: {e}"],
                "pending_count": len(session.pending_messages),
                "text_preview": None,
                "missing_fields": [],
            }

        # Schedule background merge timer (reset on each new message)
        # Skip if already merged (e.g., progressive finalization)
        if not result.get("merged"):
            self._schedule_merge_timeout(session.user_id)

        # Check if session should auto-merge immediately (5-min timeout from last message)
        if session.should_merge():
            return await self._do_merge(session, warnings)

        result["warnings"] = result.get("warnings", []) + warnings
        return result

    # ── Text handler ────────────────────────────────────────────────────────

    async def _handle_text(
        self,
        user_id: str,
        text: str,
        metadata: Dict[str, Any],
        session: VisitSession,
    ) -> Dict[str, Any]:
        """Handle a text message: extract fields, accumulate, check for missing.

        If core fields are missing after accumulation, triggers progressive
        questioning immediately (does not wait for merge).
        """
        session.add_message(MessageType.TEXT, text, metadata=metadata)
        fields = extract_fields(text)

        # Accumulate into session for progressive completion
        session.accumulate_fields(fields)

        # Check what's still missing
        missing = get_missing_core_fields(session.merged_fields or {})

        if missing:
            # ── Start/continue progressive questioning immediately ──
            # Build the progressive queue (all missing fields, in priority order)
            progressive_missing = self._build_progressive_queue(session.merged_fields or {})
            first_field = progressive_missing.pop(0)
            session.pending_field = first_field
            session.pending_questions = progressive_missing
            session.retry_count = 0
            followup_msg = _get_field_question(first_field)
            if not followup_msg:
                followup_msg = f"好的，再补充一下「{CORE_FIELD_LABELS.get(first_field, first_field)}」的信息吧～"

            logger.info(
                "Progressive: user %s first message, missing=%s, asking '%s', queued=%s",
                user_id, missing, first_field, progressive_missing,
            )
            return {
                "merged": False,
                "task": None,
                "warnings": [],
                "pending_count": len(session.pending_messages),
                "text_preview": text,
                "missing_fields": missing,
                "needs_followup": True,
                "followup_field": first_field,
                "followup_message": followup_msg,
            }

        # ── All core fields present → finalize ──
        logger.info(
            "Progressive: user %s all core fields complete, finalizing",
            user_id,
        )
        return await self._finalize_accumulated(session)

    # ── Voice handler ───────────────────────────────────────────────────────

    async def _handle_voice(
        self,
        user_id: str,
        content: str,  # file path or URL to audio file
        metadata: Dict[str, Any],
        session: VisitSession,
    ) -> Dict[str, Any]:
        transcript = ""
        confidence = 0.0
        stt_used = False
        downloaded = False

        try:
            if not os.path.exists(content):
                # Could be URL — download to temp
                content = await self._download_file(content, suffix=".ogg")
                downloaded = True

            # When STT module is unavailable, give a clear suggestion
            if transcribe_audio is None:
                return {
                    "merged": False,
                    "task": None,
                    "warnings": [
                        "语音识别暂不可用，请用文字描述拜访情况，或发送门头照+文字"
                    ],
                    "pending_count": len(session.pending_messages),
                    "text_preview": None,
                    "missing_fields": [],
                }

            # Use Hermes STT module
            try:
                result = transcribe_audio(content)
                if result.get("success"):
                    transcript = result.get("transcript", "")
                    stt_used = True
                    # Use explicit confidence from result if available, otherwise infer
                    if "confidence" in result:
                        confidence = float(result["confidence"])
                    else:
                        confidence = 0.85 if transcript else 0.0
                    logger.info(
                        "STT success (provider=%s): %s",
                        result.get("provider", "?"), transcript[:80],
                    )
                else:
                    logger.warning("STT failed: %s", result.get("error", "unknown"))
                    return {
                        "merged": False,
                        "task": None,
                        "warnings": [f"语音识别失败: {result.get('error', '')}"],
                        "pending_count": len(session.pending_messages),
                        "text_preview": None,
                        "missing_fields": [],
                    }
            except ImportError:
                logger.warning("Hermes STT not available")
                transcript = "[语音消息 — 无法识别]"
            except Exception as e:
                logger.error("STT call failed: %s", e)
                transcript = "[语音消息 — 识别异常]"

        finally:
            # Only delete temp files that were downloaded, never local files
            if downloaded:
                self._safe_delete(content)

        # Handle long transcript — save raw text in metadata for potential segmentation
        if transcript and len(transcript) > 2000:
            metadata["stt_raw"] = transcript

        # Low-confidence check — ask user to confirm; keep message pending for force-merge
        if confidence < 0.6:
            if transcript:
                session.add_message(MessageType.VOICE, transcript, metadata=metadata)
            fields = extract_fields(transcript) if transcript else {}
            session.accumulate_fields(fields)
            missing = get_missing_core_fields(session.merged_fields or {})
            return {
                "merged": False,
                "task": None,
                "warnings": [],
                "pending_count": len(session.pending_messages),
                "text_preview": transcript,
                "needs_confirmation": True,
                "missing_fields": missing,
            }

        session.add_message(MessageType.VOICE, transcript, metadata=metadata)
        fields = extract_fields(transcript)
        session.accumulate_fields(fields)

        # Check for progressive questioning
        missing = get_missing_core_fields(session.merged_fields or {})
        if missing:
            progressive_missing = self._build_progressive_queue(session.merged_fields or {})
            first_field = progressive_missing.pop(0)
            session.pending_field = first_field
            session.pending_questions = progressive_missing
            session.retry_count = 0
            followup_msg = _get_field_question(first_field)
            if not followup_msg:
                followup_msg = f"好的～再补充一下「{CORE_FIELD_LABELS.get(first_field, first_field)}」的信息吧"
            logger.info(
                "Progressive (voice): user %s missing=%s, asking '%s'",
                user_id, missing, first_field,
            )
            return {
                "merged": False,
                "task": None,
                "warnings": [],
                "pending_count": len(session.pending_messages),
                "text_preview": transcript,
                "missing_fields": missing,
                "needs_followup": True,
                "followup_field": first_field,
                "followup_message": followup_msg,
            }

        # All fields complete
        return await self._finalize_accumulated(session)

    # ── Photo handler ───────────────────────────────────────────────────────

    async def _handle_photo(
        self,
        user_id: str,
        content: str,  # file path or URL to image
        metadata: Dict[str, Any],
        session: VisitSession,
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        ocr_text = ""
        exif_data: Dict[str, Any] = {}
        downloaded = False

        # Step 1: Ensure file is local
        local_path = content
        if content.startswith("http://") or content.startswith("https://"):
            local_path = await self._download_file(content, suffix=".jpg")
            downloaded = True

        # Step 2: Vision + OCR via Hermes vision_tools
        ocr_text = await self._ocr_via_vision(local_path)
        if not ocr_text:
            ocr_text = self._ocr_via_pil_fallback(local_path)

        # Step 3: Extract EXIF (GPS + timestamp)
        exif_data = self._extract_exif(local_path)

        # Step 4: Store photo data in session
        photo_metadata = {
            "ocr_text": ocr_text,
            "exif": exif_data,
            "file_path": local_path,
            "is_temp": downloaded,
        }
        session.add_message(
            MessageType.PHOTO,
            ocr_text or "[门头照]",
            metadata=photo_metadata,
        )

        fields = extract_fields(ocr_text or "[门头照]")
        session.accumulate_fields(fields)

        # Check for progressive questioning
        missing = get_missing_core_fields(session.merged_fields or {})
        if missing:
            progressive_missing = self._build_progressive_queue(session.merged_fields or {})
            first_field = progressive_missing.pop(0)
            session.pending_field = first_field
            session.pending_questions = progressive_missing
            session.retry_count = 0
            followup_msg = _get_field_question(first_field)
            if not followup_msg:
                followup_msg = f"好的～再补充一下「{CORE_FIELD_LABELS.get(first_field, first_field)}」的信息吧"
            logger.info(
                "Progressive (photo): user %s missing=%s, asking '%s'",
                user_id, missing, first_field,
            )
            return {
                "merged": False,
                "task": None,
                "warnings": warnings,
                "pending_count": len(session.pending_messages),
                "text_preview": f"[门头照] {ocr_text}" if ocr_text else "[门头照]",
                "photo_exif": exif_data,
                "missing_fields": missing,
                "needs_followup": True,
                "followup_field": first_field,
                "followup_message": followup_msg,
            }

        # All fields complete
        r = await self._finalize_accumulated(session)
        r["warnings"] = r.get("warnings", []) + warnings
        r["photo_exif"] = exif_data
        return r

    # ── Vision+OCR ──────────────────────────────────────────────────────────

    async def _ocr_via_vision(self, image_path: str) -> str:
        """Use Hermes vision_tools (qwen-vl-max via DashScope) for OCR."""
        try:
            prompt = (
                "请识别这张图片中的所有文字，特别关注：\n"
                "1. 店名/机构名称\n"
                "2. 门牌号\n"
                "3. 任何地址信息\n"
                "4. 电话号码\n\n"
                "请以JSON格式返回：\n"
                '{"store_name": "店名", "address": "地址", "phone": "电话", "other_text": "其他文字"}'
            )
            result = await vision_analyze_tool(
                image_url=image_path,
                user_prompt=prompt,
            )
            # vision_analyze_tool returns JSON string with {"success": bool, "analysis": str}
            parsed = json.loads(result)
            if parsed.get("success"):
                analysis = parsed.get("analysis", "")
                # Try to parse structured JSON from analysis
                try:
                    structured = json.loads(analysis)
                    store_name = structured.get("store_name", "")
                    address = structured.get("address", "")
                    phone = structured.get("phone", "")
                    other = structured.get("other_text", "")
                    parts = [p for p in [store_name, address, phone, other] if p]
                    return " ".join(parts) if parts else analysis
                except (json.JSONDecodeError, TypeError):
                    return analysis
            return ""
        except ImportError:
            logger.warning("Hermes vision_tools not available")
            return ""
        except Exception as e:
            logger.error("Vision+OCR failed: %s", e)
            return ""

    def _ocr_via_pil_fallback(self, image_path: str) -> str:
        """Fallback OCR using PIL/Pillow for basic text extraction."""
        try:
            from PIL import Image

            img = Image.open(image_path)
            return f"[图片: {img.width}x{img.height}]"
        except ImportError:
            return "[图片 — 无法分析]"
        except Exception as e:
            logger.error("PIL fallback failed: %s", e)
            return ""

    # ── EXIF extraction ─────────────────────────────────────────────────────

    def _extract_exif(self, image_path: str) -> Dict[str, Any]:
        """Extract EXIF metadata: GPS coordinates and capture time."""
        result: Dict[str, Any] = {"gps": None, "datetime": None}

        try:
            from PIL import Image
            from PIL.ExifTags import Base as ExifBase, GPSTAGS

            img = Image.open(image_path)
            exif_raw = img.getexif()
            if not exif_raw:
                return result

            exif = {ExifBase(k).name: v for k, v in exif_raw.items() if k in ExifBase}

            # Capture time
            dt_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
            if dt_str:
                try:
                    dt = datetime.strptime(str(dt_str), "%Y:%m:%d %H:%M:%S")
                    result["datetime"] = dt.isoformat()
                except ValueError:
                    result["datetime"] = str(dt_str)

            # GPS coordinates
            gps_info = exif_raw.get_ifd(0x8825)  # GPSInfo IFD
            if gps_info:
                gps = {}
                for tag_id, value in gps_info.items():
                    tag_name = GPSTAGS.get(tag_id, str(tag_id))
                    gps[tag_name] = value
                result["gps"] = self._parse_gps(gps)

        except ImportError:
            logger.debug("PIL/Pillow not available for EXIF")
        except Exception as e:
            logger.warning("EXIF extraction failed: %s", e)

        return result

    def _parse_gps(self, gps: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Parse GPS data from EXIF into lat/lon."""
        try:
            lat_ref = gps.get("GPSLatitudeRef", "N")
            lat = gps.get("GPSLatitude")
            lon_ref = gps.get("GPSLongitudeRef", "E")
            lon = gps.get("GPSLongitude")

            if not lat or not lon:
                return None

            def _to_decimal(dms) -> float:
                if isinstance(dms, (tuple, list)) and len(dms) >= 3:
                    return float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0
                return float(dms)

            lat_dd = _to_decimal(lat)
            lon_dd = _to_decimal(lon)
            if lat_ref == "S":
                lat_dd = -lat_dd
            if lon_ref == "W":
                lon_dd = -lon_dd

            return {"latitude": round(lat_dd, 6), "longitude": round(lon_dd, 6)}
        except Exception as e:
            logger.warning("GPS parse failed: %s", e)
            return None

    # ── Cross-check logic ───────────────────────────────────────────────────

    def cross_check(
        self,
        claimed_store: str,
        ocr_store: str,
        claimed_location: Optional[Dict[str, float]] = None,
        exif_gps: Optional[Dict[str, float]] = None,
        claimed_time: Optional[str] = None,
        exif_time: Optional[str] = None,
    ) -> List[str]:
        """Cross-check claimed info against photo metadata. Returns warnings."""
        warnings: List[str] = []

        # Check 1: store name mismatch
        if ocr_store and claimed_store:
            if not self._fuzzy_match(claimed_store, ocr_store):
                warnings.append(
                    f"⚠️ 地址不匹配: 声称'{claimed_store}' vs OCR'{ocr_store}'"
                )

        # Check 2: GPS deviation > 500m
        if claimed_location and exif_gps:
            dist = self._haversine_distance(
                claimed_location.get("latitude", 0),
                claimed_location.get("longitude", 0),
                exif_gps.get("latitude", 0),
                exif_gps.get("longitude", 0),
            )
            if dist > 500:
                warnings.append(
                    f"⚠️ 位置偏差: {dist:.0f}m"
                )

        # Check 3: time deviation > 30 min
        if claimed_time and exif_time:
            try:
                ct = datetime.fromisoformat(claimed_time)
                et = datetime.fromisoformat(exif_time)
                diff = abs((ct - et).total_seconds())
                if diff > 1800:
                    warnings.append(
                        f"⚠️ 时间偏差: {diff/60:.0f}分钟"
                    )
            except (ValueError, TypeError):
                pass

        return warnings

    @staticmethod
    def _fuzzy_match(a: str, b: str, threshold: float = 0.6) -> bool:
        """Simple fuzzy matching between two names."""
        a_clean = "".join(c for c in a if c.isalnum())
        b_clean = "".join(c for c in b if c.isalnum())
        if not a_clean or not b_clean:
            return False
        # Check if one is a substring of the other
        if a_clean in b_clean or b_clean in a_clean:
            return True
        # Character overlap ratio
        overlap = len(set(a_clean) & set(b_clean))
        max_len = max(len(set(a_clean)), len(set(b_clean)))
        return overlap / max_len >= threshold if max_len > 0 else False

    @staticmethod
    def _haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate haversine distance in meters between two GPS points."""
        import math

        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # ── Merge logic ─────────────────────────────────────────────────────────

    async def _finalize_accumulated(self, session: VisitSession) -> Dict[str, Any]:
        """Finalize when all core fields are complete from progressive accumulation.

        Saves a record from session.merged_fields, does cross-checks with any
        pending photo messages, then cleans up the session.
        """
        merged = session.merged_fields or {}
        warnings: List[str] = []

        # Cross-check photo vs text claims (if photo messages exist)
        photo_metas = self._collect_photo_metadata(session)
        if photo_metas:
            warnings.extend(
                self._run_cross_checks(merged, photo_metas)
            )

        task = {
            "fields": merged,
            "summary": fields_to_summary(merged),
            "warnings": warnings,
            "message_count": len(session.pending_messages),
            "merged_text": session.merge_contents() if session.pending_messages else "",
        }

        # Collect photo paths for persistence
        photo_paths = self._collect_photo_paths(session)

        # Persist record
        persist_ok = False
        record_id = None
        try:
            record_id = save_record(task, photo_paths)
            persist_ok = True
        except Exception as e:
            logger.error("Failed to persist record: %s", e)

        if persist_ok:
            self._archive_photos(session)
            # Only clear session on successful persistence (CX fix: avoid clearing on failure)
            session.clear()
        else:
            logger.warning(
                "Persistence failed for user %s — session preserved for retry",
                session.user_id,
            )

        # Cancel merge timer
        self._cancel_merge_timer(session.user_id)

        if not record_id:
            return {
                "merged": False,
                "task": None,
                "warnings": warnings + ["持久化失败，请重试"],
                "pending_count": 0,
                "text_preview": None,
                "missing_fields": [],
                "needs_followup": False,
                "followup_field": None,
                "followup_message": None,
            }

        # Call on_task_complete callback
        if self.on_task_complete:
            try:
                result = self.on_task_complete(task)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("on_task_complete callback failed: %s", e)

        return {
            "merged": True,
            "task": task,
            "warnings": warnings,
            "pending_count": 0,
            "text_preview": fields_to_summary(merged),
            "missing_fields": [],
            "needs_followup": False,
            "followup_field": None,
            "followup_message": None,
        }

    async def _do_merge(
        self, session: VisitSession, extra_warnings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Merge pending messages into a task record and run extraction."""
        warnings = list(extra_warnings or [])
        merged_text = session.merge_contents()

        # Extract fields from merged text
        fields = extract_fields(merged_text)

        # Extract claimed_location from any message's metadata (rep-shared GPS)
        claimed_location = self._extract_claimed_location(session)

        # Cross-check photo vs text claims
        photo_metas = self._collect_photo_metadata(session)
        if photo_metas:
            warnings.extend(
                self._run_cross_checks(fields, photo_metas, claimed_location)
            )

        task = {
            "fields": fields,
            "summary": fields_to_summary(fields),
            "warnings": warnings,
            "message_count": len(session.pending_messages),
            "merged_text": merged_text,
        }

        # Collect photo paths BEFORE any archival/deletion
        photo_paths = self._collect_photo_paths(session)

        # Persist record + photos to disk FIRST (before archive may delete temp files)
        persist_ok = False
        record_id = None
        try:
            record_id = save_record(task, photo_paths)
            persist_ok = True
        except Exception as e:
            logger.error("Failed to persist record: %s", e)

        if persist_ok:
            # Only archive (which may delete temp files) after successful persistence
            self._archive_photos(session)
        else:
            logger.warning(
                "Persistence failed for user %s — session cleared, data retained in memory only (retry will produce new visit record)",
                session.user_id,
            )

        # Cancel any pending merge timer for this session
        self._cancel_merge_timer(session.user_id)

        # ── Bail out if persist failed ──
        if not record_id:
            session.clear()
            return {
                "merged": False,
                "task": None,
                "warnings": warnings + ["持久化失败，请重试"],
                "pending_count": 0,
                "text_preview": None,
                "missing_fields": [],
                "needs_followup": False,
                "followup_field": None,
                "followup_message": None,
            }

        # ── Build progressive questioning queue ──────────────────────────
        # Check ALL missing fields in priority order (date → time → core fields)
        progressive_missing = self._build_progressive_queue(fields)

        # Feedback is handled specially (existing behavior): if feedback missing,
        # it's asked first via the return value. Remaining fields go to session.
        feedback = fields.get("feedback", "")
        is_feedback_missing = not feedback or (isinstance(feedback, str) and not feedback.strip())
        needs_followup = is_feedback_missing  # Default: needs follow-up if feedback missing

        if progressive_missing:
            # ── Setup progressive state (don't clear session) ──────────

            if is_feedback_missing:
                # Feedback comes first (existing behavior), queue the rest
                # Remove feedback from progressive queue since it's asked separately
                other_missing = [f for f in progressive_missing if f != "feedback"]

                if other_missing:
                    # Both feedback + other fields missing: enter progressive mode
                    session.merged_fields = dict(fields)
                    if record_id:
                        session.merged_record_id = record_id
                    session.pending_questions = other_missing
                    session.pending_field = None  # Will be set after user answers feedback
                    followup_msg = (
                        "好的，记录下来了～再补充一下：这次的拜访对象有什么具体的反馈吗？"
                        "比如对产品的评价、销量变化、提出的建议等等。"
                    )
                    logger.info(
                        "Merge for user %s: feedback missing + %d other fields queued: %s",
                        session.user_id, len(other_missing), other_missing,
                    )
                else:
                    # ONLY feedback missing, no other progressive fields → progressive follow-up
                    session.merged_fields = dict(fields)
                    if record_id:
                        session.merged_record_id = record_id
                    session.pending_questions = []
                    session.pending_field = "feedback"  # Will be answered in progressive flow, then update_record
                    followup_msg = (
                        "好的，记录下来了～再补充一下：这次的拜访对象有什么具体的反馈吗？"
                        "比如对产品的评价、销量变化、提出的建议等等。"
                    )
                    logger.info(
                        "Merge for user %s: only feedback missing, progressive follow-up with record_id=%s",
                        session.user_id, record_id,
                    )
            else:
                # No feedback missing — start progressive immediately
                session.merged_fields = dict(fields)
                if record_id:
                    session.merged_record_id = record_id
                first_field = progressive_missing.pop(0)
                session.pending_field = first_field
                session.pending_questions = progressive_missing
                followup_msg = _get_field_question(first_field)
                needs_followup = True  # Signal needs_followup (non-feedback field when feedback was OK)
                logger.info(
                    "Merge for user %s: progressive start with '%s', %d more queued: %s",
                    session.user_id, first_field, len(progressive_missing), progressive_missing,
                )
        else:
            # No missing fields at all — clean up normally
            session.clear()
            followup_msg = None
            needs_followup = False

        if self.on_task_complete:
            try:
                result = self.on_task_complete(task)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("on_task_complete callback failed: %s", e)

        # Build missing_fields for the response (for backward compat)
        missing_fields = get_missing_core_fields(fields)
        if is_feedback_missing and "feedback" not in missing_fields:
            missing_fields.append("feedback")

        return {
            "merged": True,
            "task": task,
            "warnings": warnings,
            "pending_count": 0,
            "text_preview": fields_to_summary(fields),
            "missing_fields": missing_fields,
            "needs_followup": needs_followup,
            "followup_field": "feedback" if (is_feedback_missing and not session.pending_field) else session.pending_field,
            "followup_message": followup_msg,
        }

    async def _force_merge(self, session: VisitSession) -> Dict[str, Any]:
        """Force merge triggered by manual phrase ('完了' etc.).

        If session has accumulated fields (progressive mode), finalize them.
        Otherwise fall back to _do_merge over pending messages.
        """
        # If we have accumulated fields from progressive mode, finalize them
        if session.merged_fields and any(
            v and (not isinstance(v, str) or v.strip())
            for v in session.merged_fields.values()
        ):
            logger.info(
                "Manual merge triggered for user %s — finalizing %d accumulated fields",
                session.user_id, len(session.merged_fields),
            )
            return await self._finalize_accumulated(session)

        # Fall back to merge over pending messages
        if not session.pending_messages:
            return {
                "merged": False,
                "task": None,
                "warnings": [],
                "pending_count": 0,
                "text_preview": None,
                "missing_fields": [],
            }
        logger.info("Manual merge triggered for user %s", session.user_id)
        return await self._do_merge(session)

    # ── Progressive field questioning (post-merge multi-round) ──────────────

    def _build_progressive_queue(self, fields: Dict[str, Any]) -> List[str]:
        """Build ordered list of missing fields from merged result.

        Priority: visit_date → visit_time → CORE_FIELDS_PRIORITY (deduped).
        Excludes fields already present (non-empty).
        """
        missing: List[str] = []
        for field in _PROGRESSIVE_FIELD_PRIORITY:
            value = fields.get(field, "")
            if not value or (isinstance(value, str) and not value.strip()):
                missing.append(field)
        return missing

    async def _handle_progressive_answer(
        self, user_id: str, text: str, session: VisitSession
    ) -> Dict[str, Any]:
        """Handle user's answer to a progressive field question.

        Extracts the field value from user's reply, updates merged_fields,
        and asks the next pending question. When all fields are collected,
        updates the persisted record and finalizes.
        """
        merged = session.merged_fields or {}
        current_field = session.pending_field

        # ── Phase 1: Answering the current progressive question ──────────
        if current_field:
            extracted = _extract_single_field(text, current_field)
            if extracted:
                merged[current_field] = extracted
                session.retry_count = 0  # reset on success
                logger.info(
                    "Progressive: user %s answered field '%s': %s",
                    user_id, current_field, extracted,
                )
            else:
                if session.retry_count >= 2:
                    # Retry count exhausted, skip this field
                    logger.info(
                        "Progressive: user %s failed field '%s' after 2 retries, skipping",
                        user_id, current_field,
                    )
                    # Notify user that field has been skipped
                    field_label = CORE_FIELD_LABELS.get(current_field, current_field)
                    session._pending_notify = f"{field_label}尝试次数已达上限，将跳过该字段进入汇总"
                else:
                    session.retry_count += 1
                    logger.info(
                        "Progressive: user %s replied to '%s' but no value extracted (retry %d/2): %s",
                        user_id, current_field, session.retry_count, text[:80],
                    )
                    # Re-ask the same field with a retry prompt
                    field_label = CORE_FIELD_LABELS.get(current_field, current_field)
                    retry_msg = (
                        f"信息不清晰，能否重新补充一下{field_label}？"
                        f"比如{CORE_FIELD_LABELS.get(current_field, '具体说明一下')}..."
                    )
                    return {
                        "merged": False,
                        "task": None,
                        "warnings": [],
                        "pending_count": 0,
                        "text_preview": None,
                        "missing_fields": session.pending_questions,
                        "needs_followup": True,
                        "followup_field": current_field,
                        "followup_message": retry_msg,
                    }

        # ── Phase 2: Also extract feedback if we were awaiting it ────────
        # (feedback flagged as missing in _do_merge but no pending_field was set)
        if not current_field and "feedback" in merged and not merged.get("feedback"):
            extracted = _extract_single_field(text, "feedback")
            if extracted:
                merged["feedback"] = extracted
                logger.info(
                    "Progressive: user %s provided feedback: %s", user_id, extracted[:80],
                )

        session.merged_fields = merged

        # ── Phase 3: Move to next question or finalize ───────────────────
        next_field = None
        if session.pending_questions:
            next_field = session.pending_questions.pop(0)

        if next_field:
            session.pending_field = next_field
            question = _get_field_question(next_field) or f"请补充一下「{CORE_FIELD_LABELS.get(next_field, next_field)}」的信息～"
            logger.info(
                "Progressive: user %s → next question for field '%s'", user_id, next_field,
            )
            notify = session._pending_notify
            session._pending_notify = None
            return {
                "merged": False,
                "task": None,
                "warnings": [],
                "pending_count": 0,
                "text_preview": None,
                "missing_fields": session.pending_questions,
                "needs_followup": True,
                "followup_field": next_field,
                "followup_message": question,
                "_notify": notify,
            }

        # ── Phase 4: All questions answered — finalize accumulated fields ──
        session.pending_field = None
        # Re-check whether all core fields are now complete
        still_missing = get_missing_core_fields(merged)
        if still_missing:
            # Still have missing fields — continue progressive questioning
            progressive_missing = self._build_progressive_queue(merged)
            if progressive_missing:
                next_field = progressive_missing.pop(0)
                session.pending_field = next_field
                session.pending_questions = progressive_missing
                session.retry_count = 0
                followup_msg = _get_field_question(next_field)
                if not followup_msg:
                    followup_msg = f"再补充一下「{CORE_FIELD_LABELS.get(next_field, next_field)}」的信息吧～"
                logger.info(
                    "Progressive: user %s still missing %d fields, asking '%s'",
                    user_id, len(still_missing), next_field,
                )
                notify = session._pending_notify
                session._pending_notify = None
                return {
                    "merged": False,
                    "task": None,
                    "warnings": [],
                    "pending_count": 0,
                    "text_preview": None,
                    "missing_fields": still_missing,
                    "needs_followup": True,
                    "followup_field": next_field,
                    "followup_message": followup_msg,
                    "_notify": notify,
                }

        # All done — use _finalize_accumulated to save and clean up
        return await self._finalize_accumulated(session)

    # ── Location helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_claimed_location(session: VisitSession) -> Optional[Dict[str, float]]:
        """Extract claimed GPS location from any pending message's metadata."""
        for msg in session.pending_messages:
            loc = msg.metadata.get("claimed_location")
            if loc and isinstance(loc, dict):
                lat = loc.get("latitude")
                lon = loc.get("longitude")
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    return {"latitude": float(lat), "longitude": float(lon)}
        return None

    def _collect_photo_metadata(self, session: VisitSession) -> List[Dict[str, Any]]:
        return [
            msg.metadata
            for msg in session.pending_messages
            if msg.msg_type == MessageType.PHOTO
        ]

    def _run_cross_checks(
        self,
        fields: Dict[str, Any],
        photo_metas: List[Dict[str, Any]],
        claimed_location: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        warnings: List[str] = []
        claimed_store = fields.get("org_name", "")
        claimed_address = fields.get("org_address", "")
        claimed_date = fields.get("visit_date", "")
        claimed_time = fields.get("visit_time", "")
        claimed_dt = f"{claimed_date}T{claimed_time}" if claimed_date and claimed_time else None

        for pm in photo_metas:
            ocr_text = pm.get("ocr_text", "")
            exif = pm.get("exif", {})

            # Extract store name from OCR text
            ocr_store = self._extract_store_from_ocr(ocr_text)

            cross_warnings = self.cross_check(
                claimed_store=claimed_store,
                ocr_store=ocr_store,
                claimed_location=claimed_location,
                exif_gps=exif.get("gps"),
                claimed_time=claimed_dt,
                exif_time=exif.get("datetime"),
            )
            warnings.extend(cross_warnings)

        return warnings

    @staticmethod
    def _extract_store_from_ocr(ocr_text: str) -> str:
        """Extract store name from OCR output."""
        if not ocr_text:
            return ""
        # OCR text often starts with the store name
        for suffix in ["大药房", "药房", "药店", "医院", "诊所", "连锁"]:
            idx = ocr_text.find(suffix)
            if idx >= 0:
                start = max(0, idx - 10)
                return ocr_text[start:idx + len(suffix)].strip()
        return ""

    def _archive_photos(self, session: VisitSession) -> None:
        """Move photos to session archive directory."""
        archive_dir = Path(tempfile.gettempdir()) / "pharma_archive" / session.user_id
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            for msg in session.pending_messages:
                if msg.msg_type == MessageType.PHOTO:
                    src = msg.metadata.get("file_path", "")
                    if src and os.path.exists(src):
                        dst = archive_dir / os.path.basename(src)
                        import shutil
                        shutil.copy2(src, dst)
                        logger.debug("Archived photo: %s → %s", src, dst)
                        if msg.metadata.get("is_temp"):
                            ComplianceBotHandler._safe_delete(src)
        except Exception as e:
            logger.error("Photo archive failed: %s", e)

    @staticmethod
    def _collect_photo_paths(session: VisitSession) -> List[str]:
        """Collect file paths of all photos in the session for persistence."""
        paths: List[str] = []
        for msg in session.pending_messages:
            if msg.msg_type == MessageType.PHOTO:
                src = msg.metadata.get("file_path", "")
                if src and os.path.exists(src):
                    paths.append(src)
        return paths

    # ── Background merge timer ──────────────────────────────────────────────

    def _schedule_merge_timeout(self, session_id: str, timeout: float = 300.0) -> None:
        """Schedule a background merge after `timeout` seconds of inactivity.

        Cancels any existing timer for the session and creates a new one.
        When the timer fires, triggers merge_and_finalize.
        """
        self._cancel_merge_timer(session_id)

        async def _delayed_merge():
            await asyncio.sleep(timeout)
            session = self.sessions.get_session(session_id)
            if session and session.pending_messages:
                # Don't auto-merge if in progressive questioning mode
                if session.pending_field is not None:
                    logger.debug(
                        "Background merge timer fired but user %s is in progressive mode, skipping",
                        session_id,
                    )
                    return
                logger.info(
                    "Background merge timer fired for user %s (%d pending)",
                    session_id, len(session.pending_messages),
                )
                try:
                    await self._do_merge(session)
                except Exception as e:
                    logger.error("Background merge failed for user %s: %s", session_id, e)

        self._merge_timers[session_id] = asyncio.create_task(_delayed_merge())
        logger.debug("Scheduled merge timer for user %s (%.0fs)", session_id, timeout)

    def _cancel_merge_timer(self, session_id: str) -> None:
        """Cancel the pending merge timer for a session."""
        timer = self._merge_timers.pop(session_id, None)
        if timer and not timer.done():
            timer.cancel()
            logger.debug("Cancelled merge timer for user %s", session_id)

    # ── Utilities ────────────────────────────────────────────────────────────

    async def _download_file(self, url: str, suffix: str = ".tmp") -> str:
        """Download a file from URL to temp directory. Returns local path.

        Cleans up the temp file on failure.
        """
        path = None
        try:
            import aiohttp
        except ImportError:
            aiohttp = None

        if aiohttp is not None:
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            with open(path, "wb") as f:
                                f.write(await resp.read())
                            return path
                        else:
                            raise IOError(f"Download failed: HTTP {resp.status}")
            except Exception:
                self._safe_delete(path)
                raise
        else:
            import urllib.request
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            try:
                urllib.request.urlretrieve(url, path)
                return path
            except Exception:
                self._safe_delete(path)
                raise

    @staticmethod
    def _safe_delete(file_path: str) -> None:
        """Safely delete a temporary file."""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug("Deleted temp file: %s", file_path)
        except Exception as e:
            logger.warning("Failed to delete temp file %s: %s", file_path, e)


# ── Convenience function ────────────────────────────────────────────────────

async def process_message(
    user_id: str,
    msg_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to process a single message."""
    handler = ComplianceBotHandler()
    return await handler.handle_message(user_id, msg_type, content, metadata)
