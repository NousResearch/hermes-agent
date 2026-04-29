"""Hermes-owned prompt compiler for Feishu Image2 jobs.

This is intentionally small and local to Hermes.  It replaces the historical
``marketing-hub/scripts`` prompt-prep boundary for the fast-lane runtime so the
Feishu gateway can create durable, auditable prompt artifacts without shelling
out to marketing-hub.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping


DISH_HINTS = [
    "辣椒小炒肉",
    "剁椒鱼头",
    "糖油粑粑",
    "臭豆腐",
    "酸菜鱼",
    "牛肉粉",
    "小炒肉",
    "米粉",
]

BEVERAGE_HINTS = [
    "三杯",
    "饮品",
    "饮料",
    "果茶",
    "冰柠檬",
    "柠檬饮",
    "鲜果",
    "冷饮",
    "冰饮",
    "草莓",
    "沙棘",
    "蓝莓",
    "薄荷",
]

SOURCE_IMAGE_EDIT_HINTS = [
    "引用",
    "回复",
    "这张",
    "这图",
    "这个海报",
    "原图",
    "图片",
    "修改",
    "改一下",
    "改成",
    "换成",
    "替换",
    "删除",
    "去掉",
    "保留",
    "精修",
    "重设计",
    "重新设计",
    "背景",
    "尺寸",
    "补总标题",
    "补标题",
    "补卖点",
    "标题",
    "卖点",
    "文案",
    "中英文",
]

CONTINUATION_HINTS = [
    "我是说",
    "不是这张",
    "不是这个",
    "你生成的是什么",
    "继续执行",
    "继续在",
    "刚才那张",
    "上一版",
    "那张图",
    "那张海报",
]

SOURCE_IMAGE_MUST_NOT = [
    "辣椒小炒肉",
    "小炒肉",
    "肉片",
    "米饭",
    "餐盘",
    "热菜",
    "锅气",
    "火锅",
    "菜品海报默认模板",
    "火宫殿",
    "T3 到店即点",
    "机场路线",
    "二维码",
    "票据条码",
    "伪Logo",
    "AI假Logo",
    "无关中文标题",
]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".gif", ".avif"}


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def source_files_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return normalized source file dictionaries from a payload/DB row."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("source_files", "source_files_json"):
        value = payload.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = []
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, (str, os.PathLike)):
                record = {"path": str(item)}
            elif isinstance(item, Mapping):
                record = dict(item)
            else:
                continue
            identity = json.dumps(
                {
                    "path": record.get("path") or record.get("file") or record.get("file_path") or record.get("local_path") or record.get("name"),
                    "image_key": record.get("image_key"),
                    "file_key": record.get("file_key"),
                    "mime": record.get("mime_type") or record.get("mime"),
                    "source": record.get("source"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if identity in seen:
                continue
            seen.add(identity)
            result.append(record)
    return result


def _looks_like_image_source(item: Mapping[str, Any]) -> bool:
    mime = _as_text(item.get("mime_type") or item.get("mime") or item.get("content_type")).lower()
    path = _as_text(item.get("path") or item.get("file") or item.get("file_path") or item.get("local_path") or item.get("name"))
    suffix = Path(path.split("?", 1)[0]).suffix.lower()
    return mime.startswith("image/") or suffix in IMAGE_SUFFIXES or bool(item.get("image_key") or item.get("file_key"))


def _has_image_source(payload: Mapping[str, Any]) -> bool:
    return any(_looks_like_image_source(item) for item in source_files_from_payload(payload))


def _extract_aspect_ratio(text: str) -> str:
    ratio_match = re.search(r"(\d{1,3})\s*[:：]\s*(\d{1,3})", text)
    if ratio_match:
        return f"{int(ratio_match.group(1))}:{int(ratio_match.group(2))}"
    size_match = re.search(r"(\d{1,4})(?:\.\d+)?\s*(?:[xX×✕✖]\ufe0f?)\s*(\d{1,4})(?:\.\d+)?", text)
    if size_match:
        width = int(size_match.group(1))
        height = int(size_match.group(2))
        if width > 0 and height > 0:
            divisor = math.gcd(width, height)
            return f"{width // divisor}:{height // divisor}"
    if "竖版" in text or "海报" in text:
        return "4:5"
    return "3:4"




def _extract_explicit_title(text: str) -> str:
    for label in ("主标题", "补总标题", "标题", "大标题", "文案"):
        pattern = re.compile(rf"{re.escape(label)}\s*[:：]\s*([^\n；;，,。]+)")
        match = pattern.search(text)
        if match:
            return match.group(1).strip().strip("『』《》\"\'“”‘’")
    quoted = re.search(r'[『《"“]([^『』《》"“”]{2,16})[』》"”]', text)
    if quoted and any(word in text for word in ("标题", "文案")):
        return quoted.group(1).strip()
    return ""


def _wants_generated_title(text: str) -> bool:
    return any(word in text for word in ("标题你帮我想", "帮我想一个", "想一个好", "起一个", "取一个", "标题文案", "文案")) and any(word in text for word in ("标题", "文案"))


def _headline_limit(text: str) -> int:
    match = re.search(r"(\d{1,2})\s*个?字以内", text)
    if match:
        return max(2, int(match.group(1)))
    return 12


def _fit_headline(value: str, limit: int) -> str:
    clean = re.sub(r"\s+", "", value).strip("，。；;、")
    return clean[:limit] if len(clean) > limit else clean


def _suggest_headline(text: str, subject: str) -> str:
    explicit = _extract_explicit_title(text)
    if explicit:
        return _fit_headline(explicit, max(_headline_limit(text), len(explicit)))
    limit = _headline_limit(text)
    if _wants_generated_title(text):
        if "臭豆腐" in subject or "臭豆腐" in text:
            return _fit_headline("外酥里嫩臭豆腐", limit)
        if any(word in text for word in ("冰柠", "柠檬", "鲜果", "饮品", "果茶")) or any(word in subject for word in ("饮品", "冰柠", "鲜果")):
            return _fit_headline("鲜果冰柠一夏", limit)
        if subject and subject != "本次指定单一主体":
            return _fit_headline(f"{subject}上新", limit)
    return subject or "本次指定单一主体"


def _extract_headline(text: str, fallback: str) -> str:
    explicit = _extract_explicit_title(text)
    if explicit:
        return explicit
    subject = _extract_subject_anchor(text) or fallback
    return _suggest_headline(text, subject)


def _extract_copy_line(text: str, label: str, fallback: str) -> str:
    pattern = re.compile(rf"{re.escape(label)}\s*[:：]\s*([^\n；;]+)")
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return fallback


def _extract_subject_anchor(text: str) -> str:
    for dish in sorted(DISH_HINTS, key=len, reverse=True):
        if dish in text:
            return dish
    flavors = [name for name in ["草莓", "沙棘", "蓝莓"] if name in text]
    if "三杯" in text and any(word in text for word in ("冰柠檬", "饮品", "冰饮", "柠檬饮", "鲜果")):
        suffix = f"（{'、'.join(flavors)}）" if flavors else ""
        return f"三杯夏日冰柠檬饮品{suffix}"
    if any(word in text for word in BEVERAGE_HINTS):
        suffix = f"（{'、'.join(flavors)}）" if flavors else ""
        return f"饮品海报主体{suffix}"
    return ""


def _is_source_image_edit(payload: Mapping[str, Any], text: str) -> bool:
    if not _has_image_source(payload):
        return False
    return any(word in text for word in SOURCE_IMAGE_EDIT_HINTS) or any(word in text for word in BEVERAGE_HINTS)


def _is_continuation_correction(payload: Mapping[str, Any], text: str, subject: str) -> bool:
    if payload.get("chatgpt_continuation_mode") is True:
        return True
    if any(word in text for word in CONTINUATION_HINTS):
        return True
    return bool(subject and any(word in text for word in ("不是", "生成", "继续", "改", "错")))


def _source_main_visual(text: str) -> str:
    subject = _extract_subject_anchor(text)
    if subject:
        if subject.startswith("饮品") or subject.startswith("三杯"):
            return f"引用图中的{subject}"
        return subject
    return "引用图片中的原主体"


def _source_file_summary(source_files: list[dict[str, Any]]) -> str:
    values: list[str] = []
    for item in source_files[:5]:
        path = _as_text(item.get("path") or item.get("file") or item.get("file_path") or item.get("local_path") or item.get("name"))
        if path:
            values.append(path)
        elif item.get("image_key"):
            values.append(f"Feishu image_key:{item.get('image_key')}")
    return "；".join(values) if values else "无"


def build_visual_brief_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact Hermes-owned visual brief from a Feishu Image2 payload."""
    text = _as_text(payload.get("text"))
    source_files = source_files_from_payload(payload)
    aspect_ratio = _extract_aspect_ratio(text)
    subject = _extract_subject_anchor(text)

    if _is_source_image_edit(payload, text):
        main_visual = _source_main_visual(text)
        return {
            "brand": "用户引用图",
            "asset_type": "参考图精修海报",
            "channel": "飞书/Image2 引用图快通道",
            "business_goal": "基于用户引用图片和回复文字完成同主题设计修改",
            "main_visual_object": main_visual,
            "copy": {
                "headline": _extract_headline(text, "沿用参考图里的核心标题/商品名层级"),
                "selling_point": _extract_copy_line(text, "补卖点", "按用户回复的完整修改要求执行，不臆造菜品卖点"),
                "info_line": "保留参考图已有信息位置；不要新增门店/T3/菜品信息",
            },
            "constraints": {
                "aspect_ratio": aspect_ratio,
                "source_image_required": True,
                "must_not": _unique(SOURCE_IMAGE_MUST_NOT),
            },
            "source_edit_mode": True,
            "chatgpt_continuation_mode": False,
            "source_files": source_files,
            "source_text": text,
        }

    if subject and _is_continuation_correction(payload, text, subject):
        return {
            "brand": "同话题 Image2 精修",
            "asset_type": "参考图精修海报",
            "channel": "飞书/Image2 同话题连续精修",
            "business_goal": "在同一 Image2 设计链路里纠正主体漂移并继续精修",
            "main_visual_object": subject,
            "copy": {
                "headline": f"沿用上一版标题层级；主体锁定为{subject}",
                "selling_point": "只按本条反馈收敛，不重写无关卖点",
                "info_line": "保留上一版已有信息和价格区；不要新增门店/T3/菜品信息",
            },
            "constraints": {
                "aspect_ratio": aspect_ratio,
                "source_image_required": False,
                "must_not": _unique(SOURCE_IMAGE_MUST_NOT + ["脱离上一版重新编主题", "新开无上下文生图窗口"]),
            },
            "source_edit_mode": True,
            "chatgpt_continuation_mode": True,
            "source_files": source_files,
            "source_text": text,
        }

    default_subject = subject or "本次指定单一主体"
    return {
        "brand": "火宫殿 T3" if "火宫殿" in text or subject in DISH_HINTS else "用户视觉任务",
        "asset_type": "单品菜品海报" if subject in DISH_HINTS or "菜" in text else "完整海报设计稿",
        "channel": "飞书/Image2 快通道",
        "business_goal": "生成同比例完整海报预览，过视觉 gate 后发送飞书原生图片",
        "main_visual_object": default_subject,
        "copy": {
            "headline": _extract_headline(text, default_subject),
            "selling_point": _extract_copy_line(text, "卖点", "突出食欲、产品利益点和本次业务目标"),
            "info_line": "按用户本条需求克制呈现；不要乱写价格、二维码或小字",
        },
        "constraints": {
            "aspect_ratio": aspect_ratio,
            "logo_policy": "reserve_clean_logo_safe_area_only",
            "source_image_required": False,
            "must_not": _unique(["伪Logo", "AI假Logo", "火宫殿字标", "店招", "门头", "二维码", "乱码小字", "多菜混搭"]),
        },
        "source_edit_mode": False,
        "chatgpt_continuation_mode": False,
        "source_files": source_files,
        "source_text": text,
    }


def _subject_lock_line(main_visual: str, *, source_edit_mode: bool) -> str:
    if main_visual and main_visual != "引用图片中的原主体" and not main_visual.startswith("引用图中的"):
        return f"本次必须仍是「{main_visual}」海报/设计；不得改成任何其他菜品、饮品或泛海报主题。"
    if source_edit_mode:
        return "必须忠实保留引用图片中的原主体/产品类别/版式关系，不得改成火宫殿默认菜品海报或无关主题。"
    return "必须围绕本次用户指定主体生成，不得漂移成无关菜品、饮品或泛海报主题。"


def _build_source_edit_prompt(brief: Mapping[str, Any]) -> str:
    copy = _as_mapping(brief.get("copy"))
    constraints = _as_mapping(brief.get("constraints"))
    source_files = list(brief.get("source_files") or [])
    main_visual = _as_text(brief.get("main_visual_object")) or "引用图片中的原主体"
    subject_lock = _subject_lock_line(main_visual, source_edit_mode=True)
    continuation = "继续编辑同一个 Image2/ChatGPT 生图会话，不要新开无上下文任务。\n" if brief.get("chatgpt_continuation_mode") else ""
    return f"""
基于用户引用图片进行 Image2 完整海报精修，比例 {constraints.get('aspect_ratio', '3:4')}。
{continuation}主视觉对象：{main_visual}。
{subject_lock}
用户原始修改要求：{brief.get('source_text', '')}
文案上图要求：主标题「{copy.get('headline', '沿用参考图标题层级')}」；一句卖点「{copy.get('selling_point', '按用户回复执行')}」；信息线「{copy.get('info_line', '保留参考图已有信息位置')}」。
参考源文件：{_source_file_summary(source_files)}。
不要新增火宫殿/T3/机场/门店信息，除非用户在本条需求里明确要求；不得套用默认辣椒小炒肉/火宫殿单品模板。
最终图必须是完整可审阅海报/配图，不是裸底图、局部补丁或等待脚本后期贴字的半成品。
候选图必须先过 subject/freshness/no-logo gate；P0/P1 不准进入飞书发送。
""".strip()


def _build_default_prompt(brief: Mapping[str, Any]) -> str:
    copy = _as_mapping(brief.get("copy"))
    constraints = _as_mapping(brief.get("constraints"))
    main_visual = _as_text(brief.get("main_visual_object")) or "主视觉对象"
    return f"""
用于华卓/火宫殿 T3 营销物料的同比例设计预览，比例 {constraints.get('aspect_ratio', '3:4')}。
业务归属：{brief.get('brand', '火宫殿 T3')}；但这只是内部归属，不要在画面中生成品牌字标或门头。
物料类型：{brief.get('asset_type')}；业务目标：{brief.get('business_goal')}。
主视觉对象：{main_visual}。
文案上图要求：主标题「{copy.get('headline', main_visual)}」；一句卖点「{copy.get('selling_point', '突出食欲和本次业务目标')}」；信息线「{copy.get('info_line', '按用户本条需求克制呈现')}」。
默认不生成 Logo、火宫殿字标、店招、印章或任何伪品牌标识；只保留自然 Logo 安全区。
画面必须像一次性完整海报设计稿，可审阅，不是裸底图、拼贴图或脚本后期贴字半成品。
""".strip()


def compile_image2_prompt_payload(payload: Mapping[str, Any], *, brief: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compile a Feishu Image2 payload into durable prompt artifacts."""
    built_brief = dict(brief or build_visual_brief_from_payload(payload))
    source_edit_mode = bool(built_brief.get("source_edit_mode"))
    prompt = _build_source_edit_prompt(built_brief) if source_edit_mode else _build_default_prompt(built_brief)
    summary = {
        "brand": built_brief.get("brand"),
        "asset_type": built_brief.get("asset_type"),
        "channel": built_brief.get("channel"),
        "main_visual_object": built_brief.get("main_visual_object"),
        "source_edit_mode": source_edit_mode,
        "chatgpt_continuation_mode": bool(built_brief.get("chatgpt_continuation_mode")),
        "aspect_ratio": _as_mapping(built_brief.get("constraints")).get("aspect_ratio"),
    }
    return {
        "visual_brief_summary": summary,
        "brief": built_brief,
        "one_shot_design_prompt": prompt,
        "compliance_review_checklist": _unique(
            [
                "候选图必须是 fresh image file，不得是旧图库、reference/wiki/cache/gallery 图片，也不得复制源参考图。",
                "检查主体是否与 visual_brief_summary.main_visual_object 一致；主体漂移是 P1。",
                "检查是否没有 AI 生成 Logo、伪Logo、品牌字标、店招、二维码、乱码小字或乱价。",
                "飞书交付必须发送原生图片本体；路径文本不算交付成功。",
            ]
            + list(_as_mapping(built_brief.get("constraints")).get("must_not") or [])
        ),
        "aesthetic_review_checklist": _unique(
            [
                "是否像完整海报/完整设计稿，而不是裸底图、局部补丁、拼贴或底图+脚本贴字。",
                "主标题、卖点、主体、光影、字体和版式是否融在同一构图里。",
                "留白、边距、Logo 安全区是否自然，画面是否有呼吸感。",
            ]
        ),
    }
