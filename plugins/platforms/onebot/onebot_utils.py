"""OneBot 纯逻辑工具模块（可热加载）。

包含与协议状态无关的纯函数/常量：CQ 码解析、Markdown 剥离、长消息
分段、表情映射、文字图字体链。adapter.py 通过 ``_load_onebot_utils()``
按 mtime 热加载本模块——改这里面的规则无需重启 gateway。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── CQ 码解析 ──────────────────────────────────────────────────────────
_FORWARD_RE = re.compile(r"\[\[qq_forward\]\](.*?)\[\[/qq_forward\]\]", re.S)
_CQ_AT_RE = re.compile(r"\[CQ:at,qq=(\d+|all)\]")
_CQ_FORWARD_RE = re.compile(r"\[CQ:forward,id=([^,\]]+)\]")
_CQ_IMAGE_RE = re.compile(r"\[CQ:image,[^\]]*?url=([^,\]]+)\]")
_CQ_IMAGE_ALL_RE = re.compile(r"\[CQ:image,([^\]]*)\]")
_CQ_IMAGE_NOURL_RE = re.compile(r"\[CQ:image(?:,[^\]]*)?\]")
_CQ_RECORD_RE = re.compile(r"\[CQ:record,[^\]]*?url=([^,\]]+)\]")
_CQ_RECORD_NOURL_RE = re.compile(r"\[CQ:record(?:,[^\]]*)?\]")
_CQ_RECORD_ALL_RE = re.compile(r"\[CQ:record,([^\]]*)\]")
_CQ_REPLY_RE = re.compile(r"\[CQ:reply,id=(\d+)\]")
_CQ_FACE_RE = re.compile(r"\[CQ:face,id=(\d+)\]")
# CQ:file —— NapCat 私聊文件属性：file=文件名, file_id=, file_size=, url=
# 无 name 参数（标准 OneBot 的 name 在 NapCat 里不出现），文件名取 file= 值
_CQ_FILE_RE = re.compile(r"\[CQ:file,([^\]]*)\]")
_CQ_VIDEO_RE = re.compile(r"\[CQ:video(?:,[^\]]*)?\]")
_CQ_JSON_RE = re.compile(r"\[CQ:json(?:,[^\]]*)?\]")
_CQ_POKE_RE = re.compile(r"\[CQ:poke(?:,[^\]]*)?\]")
_CQ_ANY_RE = re.compile(r"\[CQ:[^\]]*\]")


def _cq_file_text(m: "re.Match") -> str:
    """把 [CQ:file,...] 转为 '[文件:文件名]'（name 优先，无 name 取 file= 值）。"""
    attrs = {}
    for kv in m.group(1).split(","):
        if "=" in kv:
            k, _, v = kv.partition("=")
            attrs[k.strip()] = _cq_unescape(v)
    name = attrs.get("name") or attrs.get("file") or "文件"
    return f"[文件:{name}]"


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

# ── NapCat API 白名单 ──────────────────────────────────────────────────
# 权威定义在此；tools.py 经 fallback 导入本常量（生产网关以裸模块方式
# 加载插件，plugins.platforms.* 不可导入，tools.py 不能反向依赖 adapter
# 的加载机制，故用双层导入兜底）。
NAPCAT_API_WHITELIST = [
    "get_group_member_list",
    "get_group_member_info",
    "get_stranger_info",
    "get_forward_msg",
    "get_record",
    "get_file",
    "upload_group_file",
    "upload_private_file",
    "get_group_root_files",
    "get_group_files_by_folder",
    "get_group_file_url",
    "get_private_file_url",
    "ocr_image",
    "get_ai_characters",
    "send_group_ai_record",
]


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


# ── 引用目标解析 ───────────────────────────────────────────────────────
def _reply_target_id(raw: str, message: Optional[list] = None) -> Optional[str]:
    """提取被引用消息的 message_id（段数组优先，回退 CQ 字符串）；无引用返回 None。

    纯本地解析、不发网络请求，供提及门预取 sender 用；实际取回原文的
    get_msg 调用在 _process_message 中（结果复用，同一 reply 段至多一次）。
    多 reply 段时与 _parse_message_array 同语义：最后一个非空 id 生效。
    """
    if isinstance(message, list):
        rid: Optional[str] = None
        for seg in message:
            if isinstance(seg, dict) and seg.get("type") == "reply":
                cand = str((seg.get("data") or {}).get("id", "") or "").strip()
                if cand:
                    rid = cand
        return rid
    rm = _CQ_REPLY_RE.search(raw or "")
    return rm.group(1) if rm else None


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


# ── 权限分级 ───────────────────────────────────────────────────────────
def classify_user_role(user_id: str, admin_users: set) -> str:
    """按 user_id 判定角色：'admin' | 'member'。

    管理员集合为空时视为无管理员（安全侧：全员 member）；空 user_id
    一律 member。群内非管理员成员 = member（受限），私聊 member 直接拒。
    """
    if not user_id:
        return "member"
    return "admin" if user_id in (admin_users or set()) else "member"


# 普通用户会话出站回复的敏感意图关键词（软限制兜底审计用，非硬拦截）
SENSITIVE_PATTERNS = [
    r"删(除|掉|了)\b|rm\s+-rf|dd\s+if=",
    r"执行|运行|命令|终端|shell|sh\s+-c|bash\s+-c",
    r"重启|关机|关闭服务|systemctl|kill\b|reboot",
    r"打开.*灯|关灯|空调|窗帘|插座|摄像头|门锁|热水器",   # Home Assistant 设备控制
    r"发送到微信|发微信|发到QQ群|发邮件|发短信",          # 跨平台消息
    r"cron|定时任务|计划任务",
    r"改(配置|文件|设置)|编辑\s*/etc|写入|覆盖.*文件",
]


def scan_sensitive(text: str) -> Optional[str]:
    """命中敏感操作关键词返回首个匹配片段，否则 None。"""
    if not text:
        return None
    for pat in SENSITIVE_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(0)
    return None


# ── 戳一戳（poke）notice 事件 ──────────────────────────────────────────
# per-chat 冷却秒数：bot 被戳后同一会话内的回复防抖窗口
POKE_COOLDOWN_SECONDS = 60.0


def parse_poke_notice(data: dict, self_id: Any) -> Optional[dict]:
    """判定 notice 帧是否为"戳 bot 自己"的戳一戳事件。

    NapCat 语义：post_type=notice, notice_type=notify, sub_type=poke；
    ``user_id`` = 戳人者，``target_id`` = 被戳者。仅当 target_id ==
    self_id（bot 自己被戳）时返回 ``{"user_id", "chat_id"}``；成员互戳、
    字段缺失或 self_id 未知（无法判定被戳者）一律返回 None。
    群聊会话 chat_id 取 group_id；私聊（好友戳）取戳人者 user_id。
    """
    self_id = str(self_id or "").strip()
    if not self_id:
        return None
    if str(data.get("notice_type", "") or "") != "notify":
        return None
    if str(data.get("sub_type", "") or "") != "poke":
        return None
    user = str(data.get("user_id", "") or "").strip()
    target = str(data.get("target_id", "") or "").strip()
    if not user or target != self_id:
        return None
    group_id = str(data.get("group_id", "") or "").strip()
    if group_id:
        return {"user_id": user, "chat_id": f"group:{group_id}"}
    return {"user_id": user, "chat_id": f"private:{user}"}


def poke_cooldown_ok(
    last_ts: float, now: float, cooldown_seconds: float = POKE_COOLDOWN_SECONDS
) -> bool:
    """per-chat 冷却判断：``last_ts <= 0`` 视为从未回复过，直接放行。"""
    if last_ts <= 0.0:
        return True
    return (now - last_ts) >= cooldown_seconds


def poke_reply_text() -> str:
    """戳一戳的轻提示回复文案（纯提示，不做 agent 触发）。"""
    return "戳我干嘛～ 有事请 @我，或发送 /help 查看用法。"


# ── 好友申请/群邀请审批（request 事件）─────────────────────────────────
# 群 request 事件合法 sub_type：add=入群申请，invite=bot 被邀请入群
REQUEST_GROUP_SUB_TYPES = ("add", "invite")
# admin 通知里验证消息的最长摘要长度（防超长刷屏）
REQUEST_COMMENT_SUMMARY_MAX = 80


def parse_request_event(data: dict) -> Optional[dict]:
    """解析 request 事件帧（post_type=request）为审批记录初值。

    request_type=friend → 好友申请；request_type=group 且 sub_type ∈
    (add, invite) → 入群申请/群邀请。flag 或 user_id 缺失、类型不支持
    返回 None（调用方安全忽略）。返回字段均字符串化：
    ``{kind, sub_type, user_id, comment, flag, group_id}``。
    """
    if str(data.get("post_type", "") or "") != "request":
        return None
    request_type = str(data.get("request_type", "") or "")
    flag = str(data.get("flag", "") or "").strip()
    user_id = str(data.get("user_id", "") or "").strip()
    if not flag or not user_id:
        return None
    comment = str(data.get("comment", "") or "")
    if request_type == "friend":
        return {
            "kind": "friend",
            "sub_type": "",
            "user_id": user_id,
            "comment": comment,
            "flag": flag,
            "group_id": "",
        }
    if request_type == "group":
        sub_type = str(data.get("sub_type", "") or "").strip().lower()
        if sub_type not in REQUEST_GROUP_SUB_TYPES:
            return None
        return {
            "kind": "group",
            "sub_type": sub_type,
            "user_id": user_id,
            "comment": comment,
            "flag": flag,
            "group_id": str(data.get("group_id", "") or "").strip(),
        }
    return None


def request_notification_text(seq: int, record: dict) -> str:
    """admin 私聊通知文案：含序号/flag、申请人、验证消息摘要。"""
    if record.get("kind") == "group":
        title = "群邀请" if record.get("sub_type") == "invite" else "入群申请"
        lines = [f"📨 收到{title}（#{seq}）"]
        if record.get("group_id"):
            lines.append(f"群号：{record['group_id']}")
    else:
        lines = [f"📨 收到好友申请（#{seq}）"]
    lines.append(f"申请人：{record.get('user_id', '?')}")
    comment = str(record.get("comment", "") or "").strip()
    if comment:
        summary = comment[:REQUEST_COMMENT_SUMMARY_MAX]
        if len(comment) > REQUEST_COMMENT_SUMMARY_MAX:
            summary += "…"
        lines.append(f"验证消息：{summary}")
    lines.append(f"flag：{record.get('flag', '')}")
    lines.append(f"回复 /approve {seq} 同意，/reject {seq} 拒绝")
    return "\n".join(lines)


def resolve_request_ref(ref: str, records: List[dict]) -> Optional[dict]:
    """解析 /approve /reject 的参数（序号或 flag）为对应审批记录。

    纯数字优先按序号（seq，来自 admin 通知列表）匹配；未命中再按完整
    flag 匹配（兼容碰巧全数字的 flag）。非数字一律按 flag 精确匹配。
    查找范围含已处理记录——便于上层给出"已处理过"的幂等回复，而不是
    误报未找到。无匹配返回 None（调用方报未知 flag/序号）。
    """
    ref = str(ref or "").strip()
    if not ref:
        return None
    if ref.isdigit():
        seq = int(ref)
        for rec in records:
            try:
                if int(rec.get("seq", -1)) == seq:
                    return rec
            except (TypeError, ValueError):
                continue
    for rec in records:
        if str(rec.get("flag", "") or "") == ref:
            return rec
    return None


# ── /model 文字选择器（QQ 无 callback 按钮）────────────────────────────
def render_model_picker_text(
    providers: Any, current_model: str = "", current_provider: str = ""
) -> str:
    """渲染 ``/model`` 文字选择器："序号. provider/模型" 列表，当前项标"← 当前"。

    QQ 没有 inline callback 按钮，文字列表 + 序号回复就是正确的 picker 形态：
    用户回复 ``/model <序号>``，由网关按最近一次 picker 快照解释（快照记录在
    网关 runner 侧，见 gateway/slash_commands_model.py）。

    编号约定与网关 ``_flatten_picker_items`` 严格一致：providers 顺序 × 每个
    provider 的 models 顺序，跳过无模型行——两侧编号一致性由跨层测试锁定。

    返回空串表示无可列项（调用方回退 SendResult 失败，网关走文字列表回退）。
    """
    current_model = str(current_model or "")
    current_provider = str(current_provider or "")
    lines: List[str] = ["📋 可用模型（回复 /model <序号> 切换，列表 5 分钟内有效）："]
    seq = 0
    listed = False
    for p in providers or []:
        slug = str(p.get("slug", "") or "")
        name = str(p.get("name", "") or slug)
        models = p.get("models") or []
        if not models:
            continue  # 与网关展开约定一致：无模型行不占序号
        lines.append("")
        lines.append(f"【{name}】")
        for model in models:
            seq += 1
            model = str(model)
            marker = ""
            if current_model and model == current_model and slug == current_provider:
                marker = " ← 当前"
            lines.append(f"{seq}. {model}{marker}")
            listed = True
    if not listed:
        return ""
    return "\n".join(lines)
