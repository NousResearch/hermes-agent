"""OneBot 纯逻辑工具模块（可热加载）。

包含与协议状态无关的纯函数/常量：CQ 码解析、Markdown 剥离、长消息
分段、表情映射、文字图字体链。adapter.py 通过 ``_load_onebot_utils()``
按 mtime 热加载本模块——改这里面的规则无需重启 gateway。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

# ── CQ 码解析 ──────────────────────────────────────────────────────────
_FORWARD_RE = re.compile(r"\[\[qq_forward\]\](.*?)\[\[/qq_forward\]\]", re.S)
_CQ_AT_RE = re.compile(r"\[CQ:at,qq=(\d+|all)\]")
_CQ_IMAGE_RE = re.compile(r"\[CQ:image,[^\]]*?url=([^,\]]+)\]")
_CQ_IMAGE_ALL_RE = re.compile(r"\[CQ:image,([^\]]*)\]")
_CQ_IMAGE_NOURL_RE = re.compile(r"\[CQ:image(?:,[^\]]*)?\]")
_CQ_RECORD_RE = re.compile(r"\[CQ:record,[^\]]*?url=([^,\]]+)\]")
_CQ_RECORD_NOURL_RE = re.compile(r"\[CQ:record(?:,[^\]]*)?\]")
_CQ_REPLY_RE = re.compile(r"\[CQ:reply,id=(\d+)\]")
_CQ_FACE_RE = re.compile(r"\[CQ:face,id=(\d+)\]")
_CQ_ANY_RE = re.compile(r"\[CQ:[^\]]*\]")


def _cq_unescape(s: str) -> str:
    """反转义 CQ 码中的 HTML 实体（& → &amp;，[ → &#91; 等）。

    CQ 码字符串里 url 等参数值会被转义，下载前必须还原，
    否则 URL 里带 &amp; 会导致请求失败（图片获取失败根因）。
    """
    return (
        s.replace("&amp;", "&")
        .replace("&#91;", "[")
        .replace("&#93;", "]")
        .replace("&#44;", ",")
    )


# ── 长消息分段 ─────────────────────────────────────────────────────────
DEFAULT_SPLIT_LENGTH = 100
_SENTENCE_BOUNDS = "。！？!?；;\n"


def _split_reply(content: str, limit: int = DEFAULT_SPLIT_LENGTH) -> List[str]:
    """Split long content into ≤limit-char chunks at sentence boundaries.

    Prefers the nearest sentence-ending punctuation inside the window;
    falls back to a hard cut at the limit only when a chunk has no
    boundary at all (keeps progress guaranteed).
    """
    if not content:
        return []
    if len(content) <= limit:
        return [content]
    parts: List[str] = []
    start = 0
    n = len(content)
    while start < n:
        end = min(start + limit, n)
        if end >= n:
            parts.append(content[start:])
            break
        window = content[start:end]
        cut = -1
        for i in range(len(window) - 1, -1, -1):
            if window[i] in _SENTENCE_BOUNDS:
                cut = i
                break
        if cut == -1:
            # No sentence boundary in the window — hard cut to stay bounded.
            cut = limit - 1
        parts.append(content[start : start + cut + 1])
        start = start + cut + 1
    # Trim stray spaces but keep sentence-boundary newlines intact.
    return [p.rstrip(" \t") for p in parts if p.strip(" \t")]


# ── 表情映射 ───────────────────────────────────────────────────────────
# A few common QQ faces → emoji; anything else collapses to [表情].
_FACE_EMOJI = {
    "0": "😊", "1": "😄", "2": "😁", "3": "😆", "4": "😅", "5": "🤣",
    "14": "😏", "21": "😳", "74": "😪", "107": "🐶", "108": "🐱",
    "110": "👍", "111": "👎", "116": "🎉", "171": "🍺", "173": "👌",
}


# ── Markdown 剥离（QQ 不渲染 Markdown）────────────────────────────────
def _inline_markdown(text: str) -> str:
    """Strip inline Markdown from a single line (QQ shows raw syntax)."""
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"[\1]", text)
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)
    return text


def strip_markdown(text: str) -> str:
    """Convert Markdown to clean QQ-friendly plain text.

    QQ does not render Markdown — raw ``**bold**`` / ``## heading`` would
    appear as literal characters. Common constructs are converted to
    readable Unicode equivalents; fenced code blocks keep their contents.
    """
    lines = text.splitlines()
    out: List[str] = []
    in_code = False
    code_lang = ""
    code_lines: List[str] = []

    for line in lines:
        fence = re.match(r"^(`{3,}|~{3,})(.*)", line.strip())
        if fence:
            if not in_code:
                in_code = True
                code_lang = fence.group(2).strip()
                code_lines = []
            else:
                in_code = False
                label = f"[{code_lang}]" if code_lang else "[代码]"
                out.append(f"┌─{label}─")
                out.extend("│ " + cl for cl in code_lines)
                out.append("└──────")
                code_lines = []
            continue
        if in_code:
            code_lines.append(line)
            continue

        h = re.match(r"^(#{1,6})\s+(.*)", line)
        if h:
            level, title = len(h.group(1)), h.group(2).strip()
            title = _inline_markdown(title)
            out.append(f"【{title}】" if level <= 2 else f"▌ {title}")
            continue

        if re.match(r"^\s*[-*_]{3,}\s*$", line):
            out.append("────────────────")
            continue

        bq = re.match(r"^>\s?(.*)", line)
        if bq:
            out.append("「" + _inline_markdown(bq.group(1)) + "」")
            continue

        ul = re.match(r"^(\s*)[-*+]\s+(.*)", line)
        if ul:
            indent = len(ul.group(1)) // 2
            out.append("  " * indent + "• " + _inline_markdown(ul.group(2)))
            continue

        ol = re.match(r"^(\s*)(\d+)[.)]\s+(.*)", line)
        if ol:
            indent = len(ol.group(1)) // 2
            out.append("  " * indent + ol.group(2) + ". " + _inline_markdown(ol.group(3)))
            continue

        if re.match(r"^\s*\|", line):
            if re.match(r"^\s*\|[\s\-:|]+\|\s*$", line):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            out.append("  ".join(_inline_markdown(c) for c in cells if c))
            continue

        out.append(_inline_markdown(line))

    # Flush an unclosed fenced code block (LLM output truncated mid-block).
    if in_code:
        label = f"[{code_lang}]" if code_lang else "[代码]"
        out.append(f"┌─{label}─")
        out.extend("│ " + cl for cl in code_lines)
        out.append("└──────")

    return "\n".join(out).strip()


# ── 文字图字体链（已基本被 t2i_render 取代，保留作回退）──────────────
_TEXT_IMAGE_FALLBACK_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/unifont/unifont.otf",
    "/usr/share/fonts/opentype/unifont/unifont_upper.otf",
]
_TEXT_IMAGE_WIDTH = 800
_TEXT_IMAGE_FONT_SIZE = 26
_TEXT_IMAGE_MARGIN = 28


def _ttc_sc_index(path: str) -> int:
    """Find the Simplified-Chinese face index inside a .ttc collection."""
    from fontTools.ttLib import TTFont

    for i in range(64):
        try:
            tt = TTFont(path, fontNumber=i, lazy=True)
        except Exception:
            break
        fam = tt["name"].getDebugName(16) or tt["name"].getDebugName(1) or ""
        if "SC" in fam:
            return i
    return 0


def _build_font_chain(size: int):
    """Return [(PIL font, cmap codepoint set)] ordered by preference."""
    from fontTools.ttLib import TTFont
    from PIL import ImageFont

    chain = []
    seen = set()
    for path in _TEXT_IMAGE_FALLBACK_FONTS:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            index = _ttc_sc_index(path) if path.endswith(".ttc") else 0
            pil_font = ImageFont.truetype(path, size, index=index)
            tt = TTFont(path, fontNumber=index, lazy=True)
            cmap = set(tt.getBestCmap().keys())
            chain.append((pil_font, cmap))
        except Exception as e:
            logger.warning("[onebot] font load failed %s: %s", path, e)
    return chain


# ── Chat id 工具 ───────────────────────────────────────────────────────
def _build_chat_id(message_type: str, id_: Any) -> str:
    """Canonical chat_id used by the session store and outbound sends."""
    prefix = "group" if message_type == "group" else "private"
    return f"{prefix}:{id_}"


def _split_chat_id(chat_id: str) -> Tuple[str, str]:
    """Split a canonical chat_id back into (kind, target)."""
    if ":" in chat_id:
        kind, _, target = chat_id.partition(":")
        return kind, target
    return "private", chat_id
