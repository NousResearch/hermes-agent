"""Vision-backed, schema-normalized semantic analysis for video shots."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from agent.auxiliary_client import call_llm

from .taxonomy import normalize_controlled, taxonomy_prompt


@dataclass(frozen=True)
class SemanticClipResult:
    confidence: float
    controlled_tags: list[str]
    free_tags: list[str]
    model: str
    quality_score: float
    raw: dict[str, Any]
    search_text: str
    summary: str

    def tag_records(self) -> list[dict[str, Any]]:
        controlled = [
            {"confidence": self.confidence, "name": name, "source": "semantic-controlled"}
            for name in self.controlled_tags
        ]
        free = [
            {"confidence": self.confidence, "name": name, "source": "semantic-free"}
            for name in self.free_tags
        ]
        return [*controlled, *free]


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in result:
            result.append(text)
    return result


def normalize_semantic_result(payload: dict[str, Any], *, taxonomy: str) -> SemanticClipResult:
    content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
    cinematography = payload.get("cinematography") if isinstance(payload.get("cinematography"), dict) else {}
    creative = payload.get("creative") if isinstance(payload.get("creative"), dict) else {}
    audio = payload.get("audio") if isinstance(payload.get("audio"), dict) else {}
    quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
    analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
    retrieval = payload.get("retrieval") if isinstance(payload.get("retrieval"), dict) else {}

    mappings = (
        ("主体", content.get("subjects")),
        ("场景", content.get("scene")),
        ("动作", content.get("actions")),
        ("工序", content.get("production_stage")),
        ("景别", [cinematography.get("shot_size")]),
        ("机位", [cinematography.get("angle")]),
        ("运镜", [cinematography.get("camera_motion")]),
        ("画面特点", cinematography.get("visual_features")),
        ("情绪", creative.get("mood")),
        ("用途", creative.get("commercial_functions")),
        ("音频", audio.get("tags")),
        ("门店信息", content.get("brand_elements")),
    )
    controlled: list[str] = []
    for dimension, values in mappings:
        normalized_values = _strings(values if isinstance(values, list) else [])
        for tag in normalize_controlled(dimension, normalized_values, taxonomy):
            if tag not in controlled:
                controlled.append(tag)

    free_tags = _strings(retrieval.get("free_tags") or content.get("free_tags"))
    summary = re.sub(r"\s+", " ", str(content.get("summary") or "")).strip()
    search_text = re.sub(r"\s+", " ", str(retrieval.get("search_text") or "")).strip()
    if not search_text:
        search_text = " ".join(part for part in [summary, *controlled, *free_tags] if part)
    return SemanticClipResult(
        confidence=_clamp(analysis.get("confidence")),
        controlled_tags=controlled,
        free_tags=free_tags,
        model=str(analysis.get("model") or "configured-vision"),
        quality_score=_clamp(quality.get("overall_score")),
        raw=payload,
        search_text=search_text,
        summary=summary,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("vision model did not return a JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("vision model JSON must be an object")
    return payload


def _image_part(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError(f"unsupported keyframe type: {resolved.suffix}")
    data = resolved.read_bytes()
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("keyframe exceeds 8 MB vision input limit")
    mime = "image/png" if resolved.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def analyze_keyframes(
    keyframes: list[Path],
    *,
    taxonomy: str,
    timeout: float = 120,
) -> SemanticClipResult:
    if not keyframes:
        raise ValueError("at least one keyframe is required")
    prompt = f"""你是门店短视频素材分析器。只根据提供的关键帧返回一个 JSON 对象，不要写 Markdown。
无法确认的字段留空，不要猜测人物身份、门店名称或不可见动作。
固定词典如下：
{taxonomy_prompt(taxonomy)}

JSON 必须包含：
{{
  "content": {{"summary":"", "subjects":[], "scene":[], "actions":[], "production_stage":[], "brand_elements":[], "free_tags":[]}},
  "cinematography": {{"shot_size":"", "angle":"", "camera_motion":"", "visual_features":[]}},
  "creative": {{"mood":[], "commercial_functions":[], "suggested_script":[]}},
  "quality": {{"sharpness":0, "stability":0, "exposure":0, "subject_integrity":0, "overall_score":0}},
  "audio": {{"tags":[]}},
  "retrieval": {{"free_tags":[], "search_text":""}},
  "analysis": {{"confidence":0}}
}}
所有分数范围为 0 到 1。camera_motion 仅凭静态关键帧不能确认时留空。"""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(_image_part(path) for path in keyframes[:3])
    response = call_llm(
        messages=[{"role": "user", "content": content}],
        task="vision",
        temperature=0.1,
        timeout=timeout,
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("vision model returned no choices")
    text = str(getattr(choices[0].message, "content", "") or "")
    payload = _extract_json_object(text)
    payload.setdefault("analysis", {})
    if isinstance(payload["analysis"], dict):
        payload["analysis"].setdefault("model", str(getattr(response, "model", "") or "configured-vision"))
    return normalize_semantic_result(payload, taxonomy=taxonomy)


__all__ = ["SemanticClipResult", "analyze_keyframes", "normalize_semantic_result"]
