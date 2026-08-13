"""AstrBot 风格 Markdown 文字转图片渲染器（元素化架构 + 表格支持）。

渲染架构借鉴 AstrBot (https://github.com/AstrBotDevs/AstrBot) 的
t2i/local_strategy.py：Markdown 解析为元素 → 两遍流程（先算高再绘制）。

相比 AstrBot 的增强：
- 字形级字体回退链（Noto CJK → 文泉驿 → Unifont → Unifont Upper），
  不再有 emoji/生僻字豆腐块（AstrBot 只有字体文件级回退）
- 新增 TableElement：markdown 表格渲染为带网格线和表头灰底的表格
- 新增有序列表支持
- 页脚品牌换为 Hermes
"""
from __future__ import annotations

import io
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ── 字体字形级回退链（与 adapter.py 一致）───────────────────────────────
_FONT_FALLBACK_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/unifont/unifont.otf",
    "/usr/share/fonts/opentype/unifont/unifont_upper.otf",
]

_FONT_CACHE: dict = {}
_CMAP_CACHE: dict = {}

# ── 彩色 emoji 渲染（NotoColorEmoji CBDT 位图）─────────────────────────
# 该字体是 CBDT/CBLC 彩色位图格式，PIL 不支持 embedded_color，所以直接
# 用 fontTools 从 CBDT 表提取 PNG 位图，缩放到目标字号后按图片绘制。
_EMOJI_FONT_PATH = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
_EMOJI_TTFONT = None
_EMOJI_STRIKE = None
_EMOJI_CMAP = None
_EMOJI_BITMAP_CACHE: dict = {}  # (ch, target_h) -> (img, w, h) | None


def _load_emoji_font():
    global _EMOJI_TTFONT, _EMOJI_STRIKE, _EMOJI_CMAP
    if _EMOJI_TTFONT is not None:
        return
    try:
        from fontTools.ttLib import TTFont

        _EMOJI_TTFONT = TTFont(_EMOJI_FONT_PATH, lazy=True)
        _EMOJI_CMAP = _EMOJI_TTFONT.getBestCmap()
        _EMOJI_STRIKE = _EMOJI_TTFONT["CBDT"].strikeData[0]
    except Exception:
        _EMOJI_TTFONT = False  # 禁用标记


def _is_emoji(ch: str) -> bool:
    """判断字符是否按彩色 emoji 渲染（带 emoji 表现的码点）。"""
    if not ch:
        return False
    cp = ord(ch)
    if cp in (0xFE0F, 0x200D):  # 变体选择符 / ZWJ：零宽
        return True
    if 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x2B00 <= cp <= 0x2BFF:
        return True
    if cp in (0x00A9, 0x00AE, 0x2122):  # © ® ™
        return True
    _load_emoji_font()
    return bool(_EMOJI_CMAP) and cp >= 0xA0 and cp in _EMOJI_CMAP


def _emoji_bitmap(ch: str, target_h: int):
    """提取单个 emoji 的彩色位图，缩放到 target_h 高。

    返回 (RGBA Image, 宽, 高)；失败或字体不可用返回 None（调用方回退普通字体）。
    """
    key = (ch, target_h)
    if key in _EMOJI_BITMAP_CACHE:
        return _EMOJI_BITMAP_CACHE[key]
    result = None
    try:
        _load_emoji_font()
        if not _EMOJI_TTFONT:
            return None
        cp = ord(ch)
        glyph = _EMOJI_CMAP.get(cp)
        if not glyph:
            return None
        data = _EMOJI_STRIKE.get(glyph)
        if data is None:
            return None
        raw = data.data
        idx = raw.find(b"\x89PNG\r\n\x1a\n")
        if idx < 0:
            return None
        from io import BytesIO

        img = Image.open(BytesIO(raw[idx:])).convert("RGBA")
        w, h = img.size
        if h <= 0:
            return None
        nw = max(1, int(round(w * target_h / h)))
        resized = img.resize((nw, target_h), Image.LANCZOS)
        result = (resized, nw, target_h)
    except Exception:
        result = None
    _EMOJI_BITMAP_CACHE[key] = result
    return result


def _char_width(ch: str, chain) -> float:
    """统一字符宽度：emoji 用位图宽，其余用字体度量。"""
    if _is_emoji(ch):
        bm = _emoji_bitmap(ch, chain[0][0].size)
        if bm:
            return bm[1]
        if ord(ch) in (0xFE0F, 0x200D):
            return 0.0
    return _resolve_font(ch, chain).getlength(ch)


def _ttc_sc_index(path: str) -> int:
    """在 ttc 集合里找简体中文（CJK SC）变体的字体索引。

    旧实现误把 name 记录枚举索引当作 ttc 字体索引，导致加载到
    JP/Mono/HK 等错误变体（字形变样 + 宽度度量异常）。
    """
    try:
        from fontTools.ttLib import TTFont

        f0 = TTFont(path, lazy=True, fontNumber=0)
        for i in range(f0.reader.numFonts):
            f = TTFont(path, lazy=True, fontNumber=i)
            names = " ".join((x.toUnicode() or "") for x in f["name"].names)
            if "CJK SC" in names and "Mono" not in names:
                return i
    except Exception:
        pass
    return 0


def build_font_chain(size: int) -> List[Tuple[ImageFont.FreeTypeFont, set]]:
    """加载字体回退链，每个字体附带其字形 cmap 集合。"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    chain: List[Tuple[ImageFont.FreeTypeFont, set]] = []
    seen = set()
    for path in _FONT_FALLBACK_PATHS:
        if path in seen:
            continue
        seen.add(path)
        try:
            index = _ttc_sc_index(path) if path.endswith(".ttc") else 0
            pil_font = ImageFont.truetype(path, size, index=index)
            from fontTools.ttLib import TTFont

            tt = TTFont(path, fontNumber=index, lazy=True)
            cmap = set(tt.getBestCmap().keys())
            chain.append((pil_font, cmap))
        except Exception:
            continue
    _FONT_CACHE[size] = chain
    return chain


def _resolve_font(ch: str, chain) -> ImageFont.FreeTypeFont:
    for pf, cmap in chain:
        if ord(ch) in cmap:
            return pf
    return chain[0][0]


def _text_width(text: str, chain) -> float:
    """按字形回退逐字符测量文本宽度。"""
    w = 0.0
    for ch in text:
        w += _char_width(ch, chain)
    return w


def _seg_width(cls, content: str, chain, font_size: int = 26) -> float:
    """按段类型测量绘制后的实际占宽（含样式附加宽度）。

    表格/换行测量必须与实际绘制一致, 否则 inline code 胶囊等
    会因测量偏窄而溢出单元格（文字超出单元格的根因）。
    """
    if cls is InlineCodeElement:
        mono = _get_mono_font(font_size)
        return _code_width(content, chain, mono) + 14  # pad_x*2 + 2
    if cls is BoldTextElement:
        return _text_width(content, chain) + 1  # 双画偏移
    if cls is ItalicTextElement:
        return _text_width(content, chain) + 8  # 斜体右倾 0.2*font_size≈5.2px + 余量
    return _text_width(content, chain)


# ── 测量工具 ──────────────────────────────────────────────────────────
class TextMeasurer:
    @staticmethod
    def split_to_fit(text: str, chain, max_width: int, width_fn=None) -> List[str]:
        """按宽度贪心拆行（优先在空格处断开，无空格才逐字符硬切）。

        width_fn: 可选单字符宽度函数 (ch, chain) -> float；默认 _char_width。
        InlineCode 段必须传代码字体宽度函数，否则等宽字体比普通字体宽，
        测量偏窄会导致长代码不拆行、绘制溢出。
        """
        if width_fn is None:
            width_fn = _char_width
        if not text:
            return []
        lines = []
        cur = ""
        cur_w = 0.0
        last_space = -1  # cur 内最后一个空格的位置（相对）
        for ch in text:
            w = width_fn(ch, chain)
            if cur and cur_w + w > max_width:
                # 行首禁则: 后置标点不落行首。
                # 若并入会超宽，把标点连同前一个字符（宿主）下移——
                # 上一行少一个字、下一行以"字。"开头，既不落行首也不出界。
                if ch in _HEAD_BANNED:
                    prev = cur[-1]
                    if prev not in _HEAD_BANNED and prev != " ":
                        lines.append(cur[:-1])
                        cur = prev + ch
                        cur_w = width_fn(prev, chain) + w
                        last_space = -1
                        continue
                    # 连续标点/空格结尾: 并入（轻微超宽可接受）
                    cur += ch
                    cur_w += w
                    continue
                # 优先回退到行内最近空格断开（保留完整单词，长代码胶囊不硬切）
                if last_space > 0:
                    head, tail = cur[:last_space], cur[last_space + 1 :]
                    new_w = sum(width_fn(c, chain) for c in tail) + w
                    if head and new_w <= max_width:
                        lines.append(head)
                        cur = tail + ch
                        cur_w = new_w
                        last_space = cur.rfind(" ")  # 新行内重扫最后一个空格
                        continue
                    # 回退无效（head 为空或回退后仍超宽）→ 继续走硬切
                lines.append(cur)
                cur = ch
                cur_w = w
                last_space = -1
            else:
                cur += ch
                cur_w += w
                if ch == " ":
                    last_space = len(cur) - 1
        if cur:
            lines.append(cur)
        return lines


# 行首禁则: 这些后置标点不允许出现在行首（中文排版禁则）
_HEAD_BANNED = set("，。：；！？、）」』】》〉》」』﹂〕%‰℃:;!?.,…—・·")


# ── Markdown 元素 ─────────────────────────────────────────────────────
class MarkdownElement(ABC):
    def __init__(self, content: str):
        self.content = content

    @abstractmethod
    def calculate_height(self, image_width: int, font_size: int, chain) -> int:
        pass

    @abstractmethod
    def render(self, image, draw, x: int, y: int, image_width: int, font_size: int, chain) -> int:
        pass


class TextElement(MarkdownElement):
    """普通文本（含行内样式）: 整行按样式片段统一换行 + 标点禁则。"""

    def _line_parts(self, image_width, chain):
        return split_inline_lines(
            parse_inline(self.content), chain, image_width - 20
        )

    def calculate_height(self, image_width, font_size, chain):
        if not self.content.strip():
            return 10
        return len(self._line_parts(image_width, chain)) * (font_size + 12)

    def render(self, image, draw, x, y, image_width, font_size, chain):
        if not self.content.strip():
            return y + 10
        for parts in self._line_parts(image_width, chain):
            render_inline_parts(image, draw, x, y, parts, chain, font_size=font_size)
            y += font_size + 12
        return y


def _draw_runs(draw, xy, text, chain, fill, skip_emoji=False):
    """按字体分组绘制一行文本，保证回退字形正确混排；emoji 走彩色位图。

    skip_emoji=True 时 emoji 仍正常绘制（宽高一致），供粗体第二遍
    双画时避免位图 1px 偏移产生重影。
    """
    x, y = xy
    i = 0
    while i < len(text):
        ch = text[i]
        if _is_emoji(ch):
            if skip_emoji:
                # 粗体第二遍：跳过位图（避免 1px 重影），但宽度仍占位
                bm = _emoji_bitmap(ch, chain[0][0].size)
                x += bm[1] if bm else 0.0
                i += 1
                continue
            bm = _emoji_bitmap(ch, chain[0][0].size)
            if bm:
                img, nw, nh = bm
                draw._image.paste(img, (int(round(x)), int(round(y))), img)
                x += nw
                i += 1
                continue
            if ord(ch) in (0xFE0F, 0x200D):
                i += 1  # 零宽字符
                continue
        font = _resolve_font(ch, chain)
        j = i + 1
        while j < len(text):
            if _is_emoji(text[j]):
                break
            if _resolve_font(text[j], chain) is not font:
                break
            j += 1
        run = text[i:j]
        draw.text((x, y), run, font=font, fill=fill)
        x += font.getlength(run)
        i = j


class BoldTextElement(MarkdownElement):
    """粗体：双画模拟（对所有字形回退字体都安全，无豆腐块风险）。"""

    def calculate_height(self, image_width, font_size, chain):
        lines = TextMeasurer.split_to_fit(self.content, chain, image_width - 20)
        return len(lines) * (font_size + 12)

    def render(self, image, draw, x, y, image_width, font_size, chain):
        lines = TextMeasurer.split_to_fit(self.content, chain, image_width - 20)
        for line in lines:
            _draw_runs(draw, (x, y), line, chain, fill=(0, 0, 0))
            _draw_runs(draw, (x + 1, y), line, chain, fill=(0, 0, 0), skip_emoji=True)
            y += font_size + 12
        return y


class ItalicTextElement(MarkdownElement):
    """斜体：仿射变换倾斜模拟。"""

    def calculate_height(self, image_width, font_size, chain):
        lines = TextMeasurer.split_to_fit(self.content, chain, image_width - 20)
        return len(lines) * (font_size + 12)

    def render(self, image, draw, x, y, image_width, font_size, chain):
        lines = TextMeasurer.split_to_fit(self.content, chain, image_width - 20)
        for line in lines:
            w = _text_width(line, chain)
            h = font_size + 12
            tmp = Image.new("RGBA", (int(w) + 20, h + 6), (0, 0, 0, 0))
            td = ImageDraw.Draw(tmp)
            _draw_runs(td, (2, 2), line, chain, fill=(0, 0, 0, 255))
            italic = tmp.transform(
                tmp.size,
                Image.Transform.AFFINE,
                (1, 0.2, 0, 0, 1, 0),
                Image.Resampling.BICUBIC,
            )
            image.paste(italic, (x, y), italic)
            y += font_size + 12
        return y


class UnderlineTextElement(MarkdownElement):
    def calculate_height(self, image_width, font_size, chain):
        lines = TextMeasurer.split_to_fit(self.content, chain, image_width - 20)
        return len(lines) * (font_size + 12)

    def render(self, image, draw, x, y, image_width, font_size, chain):
        lines = TextMeasurer.split_to_fit(self.content, chain, image_width - 20)
        for line in lines:
            _draw_runs(draw, (x, y), line, chain, fill=(0, 0, 0))
            w = _text_width(line, chain)
            draw.line((x, y + font_size + 2, x + w, y + font_size + 2), fill=(0, 0, 0), width=1)
            y += font_size + 12
        return y


class StrikethroughTextElement(MarkdownElement):
    def calculate_height(self, image_width, font_size, chain):
        lines = TextMeasurer.split_to_fit(self.content, chain, image_width - 20)
        return len(lines) * (font_size + 12)

    def render(self, image, draw, x, y, image_width, font_size, chain):
        lines = TextMeasurer.split_to_fit(self.content, chain, image_width - 20)
        for line in lines:
            _draw_runs(draw, (x, y), line, chain, fill=(0, 0, 0))
            w = _text_width(line, chain)
            draw.line((x, y + font_size // 2, x + w, y + font_size // 2), fill=(0, 0, 0), width=1)
            y += font_size + 12
        return y


class HeaderElement(MarkdownElement):
    def __init__(self, content: str):
        level = 0
        for ch in content:
            if ch == "#":
                level += 1
            else:
                break
        super().__init__(content[level:].strip())
        self.level = min(level, 6)

    def calculate_height(self, image_width, font_size, chain):
        header_font_size = 42 - (self.level - 1) * 4
        hchain = build_font_chain(header_font_size)
        parts = split_inline_lines(
            parse_inline(self.content), hchain, image_width - 20,
            font_size=header_font_size,
        )
        return max(1, len(parts)) * header_font_size + 30

    def render(self, image, draw, x, y, image_width, font_size, chain):
        header_font_size = 42 - (self.level - 1) * 4
        hchain = build_font_chain(header_font_size)
        y += 10
        parts = split_inline_lines(
            parse_inline(self.content), hchain, image_width - 20,
            font_size=header_font_size,
        )
        if parts:
            render_inline_parts(image, draw, x, y, parts[0], hchain, font_size=header_font_size)
            y += header_font_size + 12
        draw.line((x, y, image_width - 10, y), fill=(230, 230, 230), width=3)
        return y + 10


class QuoteElement(MarkdownElement):
    """引用块：左侧竖线 + 灰色文字。"""

    def __init__(self, content: str):
        super().__init__(content.lstrip(">").strip())

    def calculate_height(self, image_width, font_size, chain):
        parts = split_inline_lines(parse_inline(self.content), chain, image_width - 35)
        return len(parts) * (font_size + 10) + 12

    def render(self, image, draw, x, y, image_width, font_size, chain):
        parts = split_inline_lines(parse_inline(self.content), chain, image_width - 35)
        total_height = len(parts) * (font_size + 10)
        draw.line(
            (x + 3, y + 6, x + 3, y + total_height + 6), fill=(180, 180, 180), width=5
        )
        ty = y + 6
        for line_parts in parts:
            render_inline_parts(image, draw, x + 15, ty, line_parts, chain,
                                fill=(180, 180, 180), font_size=font_size)
            ty += font_size + 10
        return y + total_height + 12


class ListItemElement(MarkdownElement):
    """无序列表项。"""

    def calculate_height(self, image_width, font_size, chain):
        parts = split_inline_lines(parse_inline(self.content), chain, image_width - 45)
        return len(parts) * (font_size + 10) + 16

    def render(self, image, draw, x, y, image_width, font_size, chain):
        parts = split_inline_lines(parse_inline(self.content), chain, image_width - 45)
        y += 8
        draw.text((x + 5, y), "•", font=chain[0][0], fill=(0, 0, 0))
        ty = y
        for line_parts in parts:
            render_inline_parts(image, draw, x + 25, ty, line_parts, chain, font_size=font_size)
            ty += font_size + 10
        return ty + 8


class OrderedListItemElement(MarkdownElement):
    """有序列表项。"""

    def __init__(self, content: str, number: int):
        super().__init__(content)
        self.number = number

    def calculate_height(self, image_width, font_size, chain):
        parts = split_inline_lines(parse_inline(self.content), chain, image_width - 55)
        return len(parts) * (font_size + 10) + 16

    def render(self, image, draw, x, y, image_width, font_size, chain):
        parts = split_inline_lines(parse_inline(self.content), chain, image_width - 55)
        y += 8
        draw.text((x + 5, y), f"{self.number}.", font=chain[0][0], fill=(0, 0, 0))
        ty = y
        for line_parts in parts:
            render_inline_parts(image, draw, x + 35, ty, line_parts, chain, font_size=font_size)
            ty += font_size + 10
        return ty + 8


class CodeBlockElement(MarkdownElement):
    """代码块：圆角灰底 + 等宽优先文本。"""

    def calculate_height(self, image_width, font_size, chain):
        if not self.content.strip():
            return 40
        lines = self.content.split("\n")
        wrapped = []
        for line in lines:
            wrapped.extend(TextMeasurer.split_to_fit(line, chain, image_width - 40))
        return len(wrapped) * (font_size + 4) + 40

    def render(self, image, draw, x, y, image_width, font_size, chain):
        lines = self.content.split("\n")
        wrapped = []
        for line in lines:
            wrapped.extend(TextMeasurer.split_to_fit(line, chain, image_width - 40))
        content_height = len(wrapped) * (font_size + 4)
        total_height = content_height + 30
        draw.rounded_rectangle(
            (x, y + 5, image_width - 10, y + total_height), radius=5,
            fill=(240, 240, 240), width=1,
        )
        ty = y + 15
        for line in wrapped:
            _draw_runs(draw, (x + 15, ty), line, chain, fill=(0, 0, 0))
            ty += font_size + 4
        return y + total_height + 10


class InlineCodeElement(MarkdownElement):
    """行内代码：灰底圆角胶囊。"""

    def calculate_height(self, image_width, font_size, chain):
        return font_size + 16

    def render(self, image, draw, x, y, image_width, font_size, chain):
        w = _text_width(self.content, chain)
        h = font_size
        pad = 4
        draw.rounded_rectangle(
            (x, y + 4, x + w + pad * 2, y + h + pad * 2 + 4), radius=5,
            fill=(230, 230, 230), width=1,
        )
        _draw_runs(draw, (x + pad, y + pad + 4), self.content, chain, fill=(0, 0, 0))
        return y + h + 16


_TABLE_COL_GAP = 4  # 表格列间间隔


class TableElement(MarkdownElement):
    """表格元素（AstrBot 没有，本实现新增）：网格线 + 表头灰底。"""

    def __init__(self, header: List[str], rows: List[List[str]]):
        super().__init__("")
        self.header = header
        self.rows = rows

    def _layout(self, image_width, font_size, chain):
        """计算列宽与每行高度。返回 (col_widths, row_heights, cell_lines, n_cols)。"""
        n_cols = max(len(self.header), max((len(r) for r in self.rows), default=0))
        if n_cols == 0:
            n_cols = 1
        pad_x = 10
        col_gap = _TABLE_COL_GAP  # 列间间隔, 防止胶囊右缘贴相邻列
        # 表格绘制起点 = x+10（x=10 → 20），右缘上限 790 → 可用宽 image_width-30
        max_content_w = image_width - 30 - col_gap * (n_cols - 1) - pad_x * 2 * n_cols

        def _split_cells(col_w_list: List[int]) -> List[List[List]]:
            out: List[List[List]] = []
            for row in [self.header] + self.rows:
                cells = row + [""] * (n_cols - len(row))
                row_lines = []
                for ci, cell in enumerate(cells):
                    row_lines.append(
                        split_inline_lines(
                            parse_inline(cell.strip()), chain,
                            max(20, col_w_list[ci] - pad_x * 2),
                        )
                    )
                out.append(row_lines)
            return out

        # 第一遍: 按等分宽换行, 得列宽
        cell_lines = _split_cells([max_content_w // n_cols] * n_cols)
        col_widths = []
        for ci in range(n_cols):
            max_cw = 0
            for rl in cell_lines:
                for line_parts in rl[ci]:
                    w = sum(_seg_width(cls, seg, chain, font_size) for cls, seg in line_parts)
                    max_cw = max(max_cw, w)
            col_widths.append(int(max_cw) + pad_x * 2)
        # 总宽超限时等比压缩, 再按最终列宽重新换行（防止压缩后文字溢出相邻列重叠）
        total = sum(col_widths) + col_gap * (n_cols - 1)
        if total > image_width - 30:
            ratio = (image_width - 30 - col_gap * (n_cols - 1)) / sum(col_widths)
            col_widths = [max(30, int(cw * ratio)) for cw in col_widths]
            cell_lines = _split_cells(col_widths)
            # 重算列宽（换行后更窄, 不二次压缩）
            new_widths = []
            for ci in range(n_cols):
                max_cw = 0
                for rl in cell_lines:
                    for line_parts in rl[ci]:
                        w = sum(_seg_width(cls, seg, chain, font_size) for cls, seg in line_parts)
                        max_cw = max(max_cw, w)
                new_widths.append(int(max_cw) + pad_x * 2)
            col_widths = new_widths
        # 拉伸列宽占满卡片可用宽度（消除右侧浪费空间）
        avail = image_width - 30 - col_gap * (n_cols - 1)
        cur = sum(col_widths)
        if cur < avail:
            extra = avail - cur
            col_widths = [cw + extra // n_cols for cw in col_widths]
            col_widths[-1] += extra - (extra // n_cols) * n_cols  # 余数给最后一列
        row_h = [font_size + 12] * len(cell_lines)
        return col_widths, row_h, cell_lines, n_cols

    def calculate_height(self, image_width, font_size, chain):
        _, row_h, cell_lines, _ = self._layout(image_width, font_size, chain)
        # 每行按最大单元格行数计算（行距 font_size+10 容纳 inline code 胶囊）
        total = 0
        for ri in range(len(cell_lines)):
            max_lines = max((len(l) for l in cell_lines[ri]), default=1)
            total += max_lines * (font_size + 14) + 12
        return total + 20

    def render(self, image, draw, x, y, image_width, font_size, chain):
        col_widths, _, cell_lines, n_cols = self._layout(image_width, font_size, chain)
        col_gap = _TABLE_COL_GAP
        table_x = x + 10
        table_w = sum(col_widths) + col_gap * (n_cols - 1)
        cy = y + 10
        # 表头
        header_lines = cell_lines[0]
        header_h = max((len(l) for l in header_lines), default=1) * (font_size + 14) + 12
        draw.rectangle(
            (table_x, cy, table_x + table_w, cy + header_h), fill=(240, 240, 240)
        )
        hx = table_x
        for ci in range(n_cols):
            ty = cy + 6
            for line_parts in header_lines[ci]:
                line_w = sum(_seg_width(cls, seg, chain, font_size) for cls, seg in line_parts)
                cx = hx + (col_widths[ci] - line_w) / 2
                render_inline_parts(image, draw, cx, ty, line_parts, chain, font_size=font_size)
                ty += font_size + 14
            hx += col_widths[ci] + col_gap
        cy += header_h
        # 数据行
        for ri in range(1, len(cell_lines)):
            max_lines = max((len(l) for l in cell_lines[ri]), default=1)
            row_h = max_lines * (font_size + 14) + 12
            # 交替行底色
            if ri % 2 == 0:
                draw.rectangle(
                    (table_x, cy, table_x + table_w, cy + row_h), fill=(248, 248, 248)
                )
            hx = table_x
            for ci in range(n_cols):
                ty = cy + 6
                for line_parts in cell_lines[ri][ci]:
                    line_w = sum(_seg_width(cls, seg, chain, font_size) for cls, seg in line_parts)
                    cx = hx + (col_widths[ci] - line_w) / 2
                    render_inline_parts(image, draw, cx, ty, line_parts, chain, font_size=font_size)
                    ty += font_size + 14
                hx += col_widths[ci] + col_gap
            cy += row_h
        # 网格线
        hx = table_x
        for cw in col_widths:
            draw.line((hx, y + 10, hx, cy), fill=(210, 210, 210), width=1)
            hx += cw + col_gap
        # 右边界线: 表格实际右缘 table_x + table_w（勿再加 col_gap）
        draw.line((table_x + table_w, y + 10, table_x + table_w, cy), fill=(210, 210, 210), width=1)
        draw.line((table_x, cy, table_x + table_w, cy), fill=(210, 210, 210), width=1)
        return cy + 10


# ── Markdown 解析器 ───────────────────────────────────────────────────
_INLINE_PATTERNS = [
    (r"\*\*(.*?)\*\*", BoldTextElement),
    (r"\*(?!\*)(.*?)\*", ItalicTextElement),
    (r"__(.*?)__", BoldTextElement),
    (r"_(?!_)(.*?)_", ItalicTextElement),
    (r"~~(.*?)~~", StrikethroughTextElement),
    (r"`(.*?)`", InlineCodeElement),
]

_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def parse_inline(line: str):
    """解析行内样式，返回 [(cls, content)] 序列；cls 为 None 表示普通文本。

    处理 **粗体** / *斜体* / ~~删除线~~ / `行内代码`，并过滤重叠标记
    （**bold** 的内层星号会被 italic 模式二次匹配，只保留最外层）。
    """
    markers = []
    for pattern, cls in _INLINE_PATTERNS:
        for mm in re.finditer(pattern, line):
            markers.append(
                {"start": mm.start(), "end": mm.end(),
                 "text": mm.group(1), "cls": cls}
            )
    if not markers:
        return [(None, line)]
    markers.sort(key=lambda mk: mk["start"])
    filtered = []
    last_end = 0
    for mk in markers:
        if mk["start"] >= last_end:
            filtered.append(mk)
            last_end = mk["end"]
    out = []
    cur_pos = 0
    for mk in filtered:
        if mk["start"] > cur_pos:
            seg = line[cur_pos:mk["start"]]
            if seg:
                out.append((None, seg))
        out.append((mk["cls"], mk["text"]))
        cur_pos = mk["end"]
    if cur_pos < len(line):
        out.append((None, line[cur_pos:]))
    return out


_MONO_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf",
]
_mono_font_cache: dict = {}


def _get_mono_font(size: int):
    """等宽字体（Latin/数字/符号）；无则 None（回退 chain）。"""
    if size in _mono_font_cache:
        return _mono_font_cache[size]
    for p in _MONO_FONT_PATHS:
        try:
            f = ImageFont.truetype(p, size)
            _mono_font_cache[size] = f
            return f
        except Exception:
            continue
    _mono_font_cache[size] = None
    return None


def _code_font_size(chain) -> int:
    """代码场景 CJK 字号：等比缩小（等宽英文 x-height 小，同号中文视觉偏大）。

    26px → 22px（0.85 倍），使中文视觉高度与等宽英文协调。
    """
    full = chain[0][0].size if chain else 26
    return max(12, int(full * 0.85))


def _code_font_for(ch: str, chain, mono) -> ImageFont.FreeTypeFont:
    """代码字体分配: Latin/数字/常见符号用等宽, CJK 回退字形链（缩小一号）。"""
    if mono and ord(ch) < 0x2E80:
        return mono
    small_chain = build_font_chain(_code_font_size(chain))
    if small_chain:
        return _resolve_font(ch, small_chain)
    return _resolve_font(ch, chain)


def _code_width(text: str, chain, mono) -> float:
    w = 0.0
    code_size = _code_font_size(chain)
    for ch in text:
        if _is_emoji(ch):
            bm = _emoji_bitmap(ch, code_size)
            w += bm[1] if bm else 0.0
        else:
            w += _code_font_for(ch, chain, mono).getlength(ch)
    return w


def _code_char_width(ch: str, chain) -> float:
    """单字符代码宽度：等宽/CJK缩小/emoji 与 _code_width 一致。"""
    code_size = _code_font_size(chain)
    if _is_emoji(ch):
        bm = _emoji_bitmap(ch, code_size)
        return bm[1] if bm else 0.0
    if ord(ch) in (0xFE0F, 0x200D):
        return 0.0
    mono = _get_mono_font(chain[0][0].size if chain else 26)
    return _code_font_for(ch, chain, mono).getlength(ch)


def _draw_code_text(draw, x, y, text, chain, mono, fill):
    """按字体分组绘制代码文本（等宽 + CJK 回退混排）。

    mono 字形在 em 框内视觉中心低于 CJK 字形（x-height 偏上），
    下移 2px 使中英文视觉基线对齐。CJK 缩小字号后需垂直居中。
    """
    code_size = _code_font_size(chain)
    full_size = chain[0][0].size if chain else code_size
    cjk_dy = (full_size - code_size) // 2  # 缩小后的 CJK 向下居中
    i = 0
    while i < len(text):
        ch = text[i]
        if _is_emoji(ch):
            bm = _emoji_bitmap(ch, code_size)
            if bm:
                img, nw, nh = bm
                draw._image.paste(img, (int(round(x)), int(round(y)) + cjk_dy), img)
                x += nw
                i += 1
                continue
            if ord(ch) in (0xFE0F, 0x200D):
                i += 1  # 零宽字符
                continue
        font = _code_font_for(ch, chain, mono)
        j = i + 1
        while j < len(text):
            if _is_emoji(text[j]):
                break
            if _code_font_for(text[j], chain, mono) is not font:
                break
            j += 1
        run = text[i:j]
        if font is mono:
            dy = 1
        else:
            dy = cjk_dy if font.size == code_size else 0
        draw.text((x, y + dy), run, font=font, fill=fill)
        x += font.getlength(run)
        i = j


def split_inline_lines(parts, chain, max_width, font_size: int = 26):
    """将行内样式片段序列按宽度整体换行（跨元素禁则吸附）。

    返回 List[List[(cls, content)]]：每行是样式片段序列。
    inline code 内容里的换行符转为空格（PIL 不绘制 \\n）。
    宽度按 _seg_width 测量（含胶囊 padding 等实际占宽）。
    """
    lines: List[List] = [[]]
    cur_w = 0.0
    for cls, content in parts:
        content = content.replace("\n", " ")
        if not content:
            continue
        for seg in TextMeasurer.split_to_fit(
            content, chain,
            max_width
            - (14 if cls is InlineCodeElement else 0)  # 预留胶囊 padding
            - (8 if cls is ItalicTextElement else 0)   # 预留斜体右倾
            - (1 if cls is BoldTextElement else 0),    # 预留粗体双画
            width_fn=_code_char_width if cls is InlineCodeElement else None,
        ):
            w = _seg_width(cls, seg, chain, font_size)
            if lines[-1] and cur_w + w > max_width:
                # 跨元素行首禁则: 后置标点不落行首。
                # 纯文本段: 剥离开头标点前缀吸附到上一行尾（轻微超宽），
                # 正文换新行——避免整段吸附导致大幅超宽。
                # 样式段（inline code 等）: 标点是内容的一部分，整段换行。
                if cls is None and seg and seg[0] in _HEAD_BANNED:
                    i = 0
                    while i < len(seg) and seg[i] in _HEAD_BANNED:
                        i += 1
                    punct, rest = seg[:i], seg[i:]
                    if punct:
                        lines[-1].append((cls, punct))
                        cur_w += _seg_width(cls, punct, chain, font_size)
                    if rest:
                        lines.append([])
                        cur_w = 0.0
                        lines[-1].append((cls, rest))
                        cur_w += _seg_width(cls, rest, chain, font_size)
                    continue
                lines.append([])
                cur_w = 0.0
            lines[-1].append((cls, seg))
            cur_w += w
    # 合并相邻同类型片段（长 inline code 被拆段时粘回单个胶囊，避免接缝）
    merged: List[List] = []
    for line in lines:
        mline: List = []
        for cls, seg in line:
            if mline and mline[-1][0] is cls:
                mline[-1] = (cls, mline[-1][1] + seg)
            else:
                mline.append((cls, seg))
        merged.append(mline)
    return [ln for ln in merged if ln]


def render_inline(image, draw, x, y, text, chain, fill=(0, 0, 0), font_size=26):
    """按行内样式分段绘制一行文本，返回绘制后的 x 坐标。"""
    return render_inline_parts(
        image, draw, x, y, parse_inline(text), chain, fill=fill, font_size=font_size
    )


def render_inline_parts(image, draw, x, y, parts, chain, fill=(0, 0, 0), font_size=26):
    """渲染行内样式片段序列（parts: [(cls, content)]），返回绘制后的 x 坐标。"""
    for cls, content in parts:
        if cls is None:
            _draw_runs(draw, (x, y), content, chain, fill=fill)
            x += _text_width(content, chain)
        elif cls is BoldTextElement:
            _draw_runs(draw, (x, y), content, chain, fill=fill)
            _draw_runs(draw, (x + 1, y), content, chain, fill=fill)
            x += _text_width(content, chain) + 1
        elif cls is ItalicTextElement:
            w = _text_width(content, chain)
            tmp = Image.new("RGBA", (int(w) + 24, font_size + 10), (0, 0, 0, 0))
            td = ImageDraw.Draw(tmp)
            _draw_runs(td, (2, 2), content, chain, fill=(0, 0, 0, 255))
            italic = tmp.transform(
                tmp.size, Image.Transform.AFFINE,
                (1, 0.2, 0, 0, 1, 0), Image.Resampling.BICUBIC,
            )
            image.paste(italic, (int(x), y), italic)
            x += w + 8
        elif cls is StrikethroughTextElement:
            _draw_runs(draw, (x, y), content, chain, fill=fill)
            w = _text_width(content, chain)
            draw.line((x, y + font_size // 2, x + w, y + font_size // 2),
                      fill=fill, width=1)
            x += w
        elif cls is InlineCodeElement:
            mono = _get_mono_font(font_size)
            w = _code_width(content, chain, mono)
            h = font_size
            pad_x, pad_y = 6, 5
            # 浅蓝灰底 + 细边框 + 圆角胶囊（内部高 = h + 2*pad_y = 36px ≤ 行高 font_size+12）
            draw.rounded_rectangle(
                (x, y + 1, x + w + pad_x * 2, y + h + pad_y * 2 + 1), radius=6,
                fill=(238, 241, 248), outline=(206, 215, 235), width=1,
            )
            _draw_code_text(
                draw, x + pad_x, y + pad_y + 3, content, chain, mono,
                fill=(47, 72, 130),
            )
            x += w + pad_x * 2 + 2
    return x


def _is_table_separator(line: str) -> bool:
    if "|" not in line or "-" not in line:
        return False
    return bool(_TABLE_SEP_RE.match(line))


def _parse_table_row(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


class MarkdownParser:
    @staticmethod
    def parse(text: str) -> List[MarkdownElement]:
        elements: List[MarkdownElement] = []
        lines = text.split("\n")
        i = 0
        ordered_counter = 0
        while i < len(lines):
            line = lines[i].rstrip()

            # 表格：当前行含 | 且下一行是分隔行
            if "|" in line and i + 1 < len(lines) and _is_table_separator(lines[i + 1]):
                header = _parse_table_row(line)
                i += 2
                rows = []
                while i < len(lines) and "|" in lines[i] and lines[i].strip():
                    rows.append(_parse_table_row(lines[i]))
                    i += 1
                elements.append(TableElement(header, rows))
                continue

            if not line.strip():
                elements.append(TextElement(""))
                i += 1
                continue

            if line.startswith("#"):
                elements.append(HeaderElement(line))
                i += 1
                continue

            if line.startswith(">"):
                elements.append(QuoteElement(line))
                i += 1
                continue

            if line.startswith("```"):
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                i += 1
                elements.append(CodeBlockElement("\n".join(code_lines)))
                continue

            if re.match(r"^\s*[-*+]\s+", line):
                elements.append(ListItemElement(re.sub(r"^\s*[-*+]\s+", "", line)))
                i += 1
                continue

            m = re.match(r"^\s*(\d+)[.)]\s+", line)
            if m:
                ordered_counter = int(m.group(1))
                elements.append(
                    OrderedListItemElement(line[m.end():], ordered_counter)
                )
                i += 1
                continue

            # 行内样式（粗体/斜体/代码等）: 不再拆分元素，
            # 由 TextElement 在渲染时统一换行 + 禁则处理。
            if any(cls is not None for cls, _ in parse_inline(line)):
                elements.append(TextElement(line))
                i += 1
                continue

            elements.append(TextElement(line))
            i += 1
        return elements


# ── 渲染器 ────────────────────────────────────────────────────────────
class MarkdownRenderer:
    def __init__(self, font_size: int = 26, width: int = 800):
        self.font_size = font_size
        self.width = width

    def render(self, markdown_text: str, title: Optional[str] = None) -> Image.Image:
        chain = build_font_chain(self.font_size)
        if not chain:
            raise RuntimeError("no usable fonts for text-image rendering")
        elements = MarkdownParser.parse(markdown_text)

        # 顶栏占位: 蓝色圆角条(72px, 容纳52px大字) + 上下间距
        topbar_h = 82 if title else 0

        total_height = 20 + topbar_h
        for el in elements:
            total_height += el.calculate_height(self.width, self.font_size, chain)
        footer_height = 40
        total_height += 20 + footer_height

        image = Image.new("RGB", (self.width, max(100, total_height)), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        y = 10
        if title:
            # 顶栏: AstrBot 风格蓝色标题栏（Material blue-500 #2196F3 + 白字 52px，内容的两倍）
            # anchor="lm": 按字体实际行高垂直居中（旧实现按字号估算偏下碰底）
            draw.rounded_rectangle(
                (10, 10, self.width - 10, 10 + 72), radius=8, fill=(33, 150, 243)
            )
            tb_font = build_font_chain(52)[0][0]
            draw.text((24, 10 + 72 / 2), title, font=tb_font, fill=(255, 255, 255), anchor="lm")
            y = 10 + topbar_h

        for el in elements:
            y = el.render(image, draw, 10, y, self.width, self.font_size, chain)

        # 页脚: "Powered by " 灰 + "Hermes" 克莱因蓝
        klein_blue = (0, 47, 167)
        grey = (130, 130, 130)
        footer_font = build_font_chain(20)[0][0]
        pb = "Powered by "
        brand = "Hermes"
        pb_w = footer_font.getlength(pb)
        brand_w = footer_font.getlength(brand)
        total_w = pb_w + brand_w
        x_start = (self.width - total_w) // 2
        fy = total_height - footer_height
        draw.text((x_start, fy), pb, font=footer_font, fill=grey)
        draw.text((x_start + pb_w, fy), brand, font=footer_font, fill=klein_blue)
        return image


_CODE_HOLDER_RE = re.compile(r"\x00CODE(\d+)\x00")


def _literal_n_to_newlines(text: str) -> str:
    """把字面 \\n（反斜杠+n）转为真正的换行符。

    LLM 输出常把换行写成字面 \\n。转换前先保护 inline code 段
    （反引号内的 \\n 由渲染层转为空格，避免反引号配对被破坏）；
    双反斜杠 \\\\n 保留不动。
    """
    codes: List[str] = []

    def _hold(m):
        seg = m.group(0)
        # inline code 内字面 \n → 空格（保留 \\n 双反斜杠）
        seg = re.sub(r"(?<!\\)\\n", " ", seg)
        codes.append(seg)
        return f"\x00CODE{len(codes) - 1}\x00"

    protected = re.sub(r"`[^`\n]*`", _hold, text)
    out = re.sub(r"(?<!\\)\\n", "\n", protected)
    return _CODE_HOLDER_RE.sub(lambda m: codes[int(m.group(1))], out)


def render_text_image(text: str, title: Optional[str] = None) -> bytes:
    """兼容入口：渲染 markdown 文本为 PNG bytes。title 显示为顶部 "To 昵称" 栏。"""
    text = _literal_n_to_newlines(text or "")
    renderer = MarkdownRenderer(font_size=26, width=800)
    img = renderer.render(text, title)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
