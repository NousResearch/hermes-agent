"""Work Sidebar — 右侧工作进度区 backend。

Mounted at /api/plugins/work-sidebar/ by the desktop plugin system.
Runs inside the gateway process, so it reads state.db directly through
hermes_state.SessionDB (no HTTP indirection).

提供两个能力（快照 + 产物提取）：
  GET /snapshot?session_id=…  → { todos, outputs, title, messageCount }
     todos  = latestSessionTodos 移植（倒序找 todo tool-call part）
     outputs= 消息文本正则提取产物（image|file|link）
     title  = 会话标题
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query

from hermes_state import SessionDB

logger = logging.getLogger(__name__)

router = APIRouter()

TODO_STATUSES = ("pending", "in_progress", "completed", "cancelled")

# SessionDB 单例：快照 2s 一次轮询，避免每次请求重建连接。
# SessionDB 内部用 _lock 串行化连接访问，跨线程安全。
_db_singleton: Optional[SessionDB] = None
_db_singleton_mtime: Optional[float] = None
# B3：单例重建路径整体持锁——检查-替换非原子（并发首次请求/并发
# invalidate 时两个线程都会走到 SessionDB()，先建的实例被覆盖且永不
# close → 连接/atexit hook 泄漏）。此锁只守卫单例生命周期；SessionDB
# 内部 _lock 负责连接访问串行化，互不干扰。
_db_lock = threading.Lock()


def _get_db() -> SessionDB:
    global _db_singleton, _db_singleton_mtime
    with _db_lock:
        if _db_singleton is not None:
            try:
                mtime = _db_singleton.db_path.stat().st_mtime
                if mtime == _db_singleton_mtime:
                    return _db_singleton
            except OSError:
                pass
            # #16:db 文件被替换/备份恢复后 mtime 变化 → 旧连接指向旧库，重建
            try:
                _db_singleton.close()
            except Exception:
                pass
            _db_singleton = None
        _db_singleton = SessionDB()
        try:
            _db_singleton_mtime = _db_singleton.db_path.stat().st_mtime
        except OSError:
            _db_singleton_mtime = None
        return _db_singleton


# #5:快照响应缓存 stored_id → (messageCount, data)。messageCount 不变即复用，
# 2s 轮询不再每次全量 DB 读 + 正则扫描。
_snapshot_cache: dict[str, tuple[int, dict]] = {}
_SNAPSHOT_CACHE_MAX = 20

# ── runtime sid → stored id 持久映射 ────────────────────────────────────
# runtime sid（uuid4().hex[:8]，desktop 的 activeSessionId）只存在于
# tui_gateway.server._sessions 内存 dict，serve 进程重启即失。desktop 恢复
# 会话时若直接拿旧的 runtime sid 来查，_sessions 里已没有对应条目，解析会
# 落空 → title/todos/outputs 全空（“恢复不一致”）。这里把解析成功的结果
# 持久化到插件目录，重启后仍能解析。
_RUNTIME_SID_MAP = Path(__file__).resolve().parent / ".runtime_sid_map.json"
_sid_map_lock = threading.RLock()  # #18:_remember_runtime_sid 整体读改写持锁，_save 内层可重入
# #21：持久映射兜底的有效性窗口。mapped 会话在窗口内有活动才可信——
# 短了误杀「挂机思考」的真实会话（后果仅是显示数据不可用，_sessions 就绪后
# 自愈）；长了防不住 sid 复用 bug 的典型形态（旧会话 2 小时前活跃被误解析）。
_RUNTIME_SID_FRESH_S = 60 * 60


def _load_runtime_sid_map() -> dict:
    try:
        with open(_RUNTIME_SID_MAP, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _save_runtime_sid_map(mapping: dict) -> None:
    try:
        with _sid_map_lock:
            tmp = _RUNTIME_SID_MAP.with_name(_RUNTIME_SID_MAP.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(mapping, fh, ensure_ascii=False, indent=1)
            tmp.replace(_RUNTIME_SID_MAP)
    except OSError:
        pass


def _remember_runtime_sid(runtime_sid: str, stored_id: str) -> None:
    """记录 runtime sid → stored id 映射（幂等，失败不阻塞主流程）。"""
    if not runtime_sid or not stored_id or runtime_sid == stored_id:
        return
    try:
        # #18:读-改-写整体持锁，防并发 _remember_runtime_sid 丢更新
        with _sid_map_lock:
            mapping = _load_runtime_sid_map()
            if mapping.get(runtime_sid) == stored_id:
                return
            mapping[runtime_sid] = stored_id
            # 防无限增长：只保留最近 200 条（每个 stored id 可对应多个 runtime sid）
            if len(mapping) > 200:
                for key in list(mapping)[:-200]:
                    mapping.pop(key, None)
            _save_runtime_sid_map(mapping)
    except Exception:
        pass


def _forget_runtime_sid(runtime_sid: str) -> None:
    """删除 runtime sid → stored id 映射条目（清理脏映射，幂等）。"""
    if not runtime_sid:
        return
    try:
        with _sid_map_lock:
            mapping = _load_runtime_sid_map()
            if runtime_sid not in mapping:
                return
            mapping.pop(runtime_sid, None)
            _save_runtime_sid_map(mapping)
    except Exception:
        pass


# ── todo 解析：apps/desktop/src/lib/todos.ts parse() 的 Python 移植 ──────
def _parse_todos(value: Any, depth: int = 0) -> Optional[List[dict]]:
    if depth > 2:
        return None
    if isinstance(value, list):
        out: List[dict] = []
        for item in value:
            if not isinstance(item, dict) or item.get("status") not in TODO_STATUSES:
                continue
            tid = str(item.get("id", "")).strip()
            content = str(item.get("content", "")).strip()
            if tid and content:
                out.append({"id": tid, "content": content, "status": item["status"]})
        # 空列表保留：表达“todo 已被清空”。None 表示“没有 todo 数据”，
        # [] 表示“显式清空”——两者语义不同，调用方必须区分。
        return out
    if isinstance(value, str) and value.strip():
        try:
            return _parse_todos(json.loads(value), depth + 1)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(value, dict) and "todos" in value:
        return _parse_todos(value["todos"], depth + 1)
    return None


def _todos_from_parts(content: Any) -> tuple[Optional[List[dict]], bool]:
    """返回 (todos, is_todo_message)。

    is_todo_message=True 表示该消息确实是 todo 工具调用——此时空列表
    表达“显式清空”，_latest_session_todos 必须尊重它，而不是跳过并回退
    到更早的非空列表（否则 todo 清空后快照会闪回旧条目）。
    """
    # str → 尝试 JSON 解析（state.db 的 tool 结果/参数都是 JSON 字符串）
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None, False

    # dict 直接带 todos 键（todo 工具的完整结果）
    if isinstance(content, dict) and "todos" in content:
        return _parse_todos(content["todos"]), True

    if not isinstance(content, list):
        return None, False

    latest: Optional[List[dict]] = None
    found = False
    for part in content:
        if (
            not isinstance(part, dict)
            or part.get("type") != "tool-call"
            or part.get("toolName") != "todo"
        ):
            continue
        found = True
        parsed = _parse_todos(part.get("todos"))
        if parsed is None:
            parsed = _parse_todos(part.get("result"))
        if parsed is None:
            parsed = _parse_todos(part.get("args"))
        # todo 调用存在就记录其结果——None（无数据）/ []（显式清空）都算
        latest = parsed
    return latest, found


def _latest_session_todos(messages: List[dict]) -> Optional[List[dict]]:
    """整个会话的当前 todo 状态 —— 最后一条 todo 消息获胜（含显式清空）。"""
    for msg in reversed(messages):
        parsed, is_todo = _todos_from_parts(msg.get("content"))
        if is_todo:
            return parsed
    return None


def _infer_title(messages: List[dict], max_len: int = 40) -> str:
    """无标题会话的兜底标题：第一条非空 user 文本，否则第一条 assistant 文本。

    desktop 恢复会话时首次快照可能早于 _sessions 就绪（runtime sid 暂无法
    解析），此时 title 为空会让侧边栏看起来“没解析到”；有消息推断兜底后
    至少显示会话内容摘要，轮询收敛后由真实标题覆盖。
    """
    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        raw = msg.get("content")
        text = ""
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, list):
            parts = [
                p.get("text")
                for p in raw
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            ]
            text = "\n".join(parts)
        text = re.sub(r"\s+", " ", text).strip()
        if not text or text.startswith(("{", "[", "```")):
            continue
        # 超长消息截断再取头（避免把整段分析当标题）
        return text[:max_len]
    return ""


# ── 产物提取：apps/desktop/src/app/artifacts/artifact-utils.ts 正则移植 ──
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_URL_RE = re.compile(r"https?://[^\s<>\"')]+")
_PATH_RE = re.compile(
    r"(^|[\s(\"'`])((?:[A-Za-z]:[\\/]|/|~\/|\.\.?\/)[^\s\"'`<>，。；、！？()（）\[\]{}]+(?:\.[a-z0-9]{1,8})?)",
    re.I
)
# B1：含空格/括号的路径变体（Windows 上 C:\Program Files\... 极常见）。
# 旧 _PATH_RE 字符类排除 \s 与括号 → C:\Program Files\foo.py 只匹配出
# C:\Program（无扩展名）→ 被 add() 扩展名白名单拦掉，整个路径丢失。
# 此正则允许 [ \t] 分隔的段与半角括号（(x86) 是路径一部分；全角（）仍
# 排除）；段字符集排除 CJK——否则「C:\foo.txt 和 C:\bar.txt」会被吞成
# 一条假产物。扩展名强制（目录不算产物）。无空格的路径仍走 _PATH_RE
# （保留 CJK 路径提取能力）。
_PATH_RE_SPACED = re.compile(
    r"(^|[\s(\"'`])((?:[A-Za-z]:[\\/]|/|~\/|\.\.?\/)"
    r"(?:[^\s\"'`<>，。；、！？（）\[\]{}\u4e00-\u9fff]+[ \t]+)*"
    r"[^\s\"'`<>，。；、！？（）\[\]{}\u4e00-\u9fff]+\.[a-z0-9]{1,8})",
    re.I,
)
_IMAGE_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|webp|svg|bmp)(?:\?.*)?$", re.I)
_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n.*?```", re.S)


def _strip_code_blocks(text: str) -> str:
    """剥离围栏代码块（#17）：块内的示例路径/ls 输出会被路径正则误判为产物。"""
    return _CODE_BLOCK_RE.sub("\n", text)
_FILE_EXT_RE = re.compile(
    r"\.(?:png|jpe?g|gif|webp|svg|bmp|pdf|txt|json|md|csv|zip|tar|gz|mp3|wav|mp4|mov)(?:\?.*)?$", re.I
)
# 结构化产物字段白名单：工具结果 JSON 里出现这些键且值是路径/URL 时，
# 直接作为产物（不依赖扩展名白名单，.py/.js 等脚本产物也不会漏）。
_STRUCTURED_PATH_KEYS = (
    "resolved_path",
    "screenshot_path",
    "image_path",
    "output_path",
    "file_path",
    "artifact_path",
    "pdf_path",
    "docx_path",
    "xlsx_path",
    "pptx_path",
    "result_path",
    "files_modified",
    "written_files",
    "created_files",
    "download_path",
    "export_path",
    "capture_path",
    "thumbnail_path",
    "video_path",
    "audio_path",
    "model_path",
    "output_dir",
    "attachment_path",
)


def _looks_like_artifact(value: str) -> bool:
    return value.startswith(("http://", "https://", "file://", "data:", "@url:")) or value.startswith(
        ("/", "./", "../", "~/")
    ) or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def _iter_structured_paths(payload: Any, depth: int = 0):
    """递归（≤2 层）收集工具结果 JSON 里白名单键的路径/URL 值。

    只认 ``*_path`` 白名单键（write_file 的 resolved_path、浏览器截图的
    screenshot_path 等）且值确实是路径/URL —— 避免把工具结果里无关的
    字符串字段误判为产物。正则兜底仍负责文本扫描。
    """
    if depth > 2 or payload is None:
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in _STRUCTURED_PATH_KEYS:
                if isinstance(value, str) and _looks_like_artifact(value):
                    yield value
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and _looks_like_artifact(item):
                            yield item
            elif isinstance(value, (dict, list)):
                yield from _iter_structured_paths(value, depth + 1)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_structured_paths(item, depth + 1)


def _artifact_kind(value: str) -> str:
    if value.startswith("data:image/") or _IMAGE_EXT_RE.search(value):
        return "image"
    if value.startswith(("/", "./", "../", "~/", "file://")) or re.match(r"^[A-Za-z]:[\\/]", value):
        return "file"
    return "link"


_TOOL_PURPOSE_PATTERNS = (
    ("write_file", "写入的文件"),
    ("patch", "修改的文件"),
    ("copy|cp ", "复制的文件"),
    ("download|curl|wget", "下载的文件"),
    ("screenshot|browser_vision|vision", "页面截图"),
    ("pdf|docx|xlsx|pptx", "生成的文档"),
    ("terminal", "终端产物"),
)


def _tool_purpose(tool_name: Any) -> Optional[str]:
    """工具名 → 用途短语（固定映射，诚实呈现，不猜测语义）。"""
    n = str(tool_name or "").lower()
    for pat, label in _TOOL_PURPOSE_PATTERNS:
        if re.search(pat, n):
            return label
    return "工具产物" if n and n != "tool" else None


def _extract_outputs(messages: List[dict], limit: int = 20) -> List[dict]:
    out: List[dict] = []
    seen = set()

    def add(value: str, kind: Optional[str] = None, ts: Any = 0, structured: bool = False, purpose: Optional[str] = None) -> None:
        value = (value or "").strip().rstrip("),.;")
        if not value or value in seen:
            return
        # @url: 前缀 = 结构化来源显式标注的 URL（如 video_path 存链接）。
        # 剥前缀后按 link 处理，而不是丢弃（旧实现漏掉这类产物）。
        forced_link = False
        if value.startswith("@url:"):
            value = value[len("@url:") :].strip().strip("()").rstrip("),.;")
            if not value or value in seen:
                return
            forced_link = True
        # 接受 POSIX 路径、Windows 盘符路径（C:\...）、URL、data:、file:
        if not value.startswith(("http://", "https://", "file://", "data:")) and not value.startswith(
            ("/", "./", "../", "~/")
        ) and not re.match(r"^[A-Za-z]:[\\/]", value):
            return
        k = "link" if forced_link else (kind or _artifact_kind(value))
        # 结构化来源（write_file 已确认写入成功）信任路径本身；
        # 正则兜底才需要扩展名白名单，避免漏掉 .py/.js 等脚本产物。
        if k == "file" and not structured and not (_IMAGE_EXT_RE.search(value) or _FILE_EXT_RE.search(value)):
            return
        label = value.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or value
        seen.add(value)
        out.append(
            {
                "kind": k,
                "value": value,
                "label": label[:80],
                "ts": ts or 0,
                "purpose": purpose or None,
            }
        )

    for msg in reversed(messages):
        ts = msg.get("timestamp") or msg.get("created_at") or 0

        # ── 结构化来源：工具结果 JSON 白名单字段（write_file 的
        #    resolved_path、browser 截图的 screenshot_path 等），
        #    正则只做文本兜底 ──────────────────────────────────────────
        if msg.get("role") == "tool":
            raw = msg.get("content")
            try:
                payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except (json.JSONDecodeError, TypeError):
                payload = {}
            # 附件线索：消息级 display_metadata 里也可能带路径
            if isinstance(msg.get("display_metadata"), dict):
                payload = {**payload, "_meta": msg["display_metadata"]}
            for value in _iter_structured_paths(payload):
                # 结构化来源不再一律强制 file（#4）：URL/data: 值按 _artifact_kind
                # 判成 link/image，只有路径才标 file——否则 video_path 存链接时
                # 前端会走 revealPath 系统定位，必失败。
                _k = None if value.startswith(("http://", "https://", "data:", "file://")) else "file"
                add(value, _k, ts, structured=True, purpose=_tool_purpose(msg.get("tool_name")))
                if len(out) >= limit:
                    return out

        if msg.get("role") != "assistant":
            continue
        raw = msg.get("content")
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, list):
            # parts 数组：拼 text 型 part 的 text/内容
            parts_txt = []
            for p in raw:
                if isinstance(p, dict):
                    if p.get("type") == "text" and isinstance(p.get("text"), str):
                        parts_txt.append(p["text"])
                    elif isinstance(p.get("text"), str):
                        parts_txt.append(p["text"])
            text = "\n".join(parts_txt)
        else:
            text = json.dumps(raw, ensure_ascii=False) if raw else ""
        # 代码块剥离（#17）：块内的示例路径/ls 输出不该进产物列表
        text = _strip_code_blocks(text)

        candidates: List[tuple[str, Optional[str]]] = []
        for m in _MD_IMAGE_RE.finditer(text):
            candidates.append((m.group(2), "image"))
        for m in _MD_LINK_RE.finditer(text):
            v = m.group(2)
            if v.startswith(("http", "/", "./", "../", "~/", "file:")) or re.match(
                r"^[A-Za-z]:[\\/]", v
            ):
                candidates.append((v, None))
        for m in _URL_RE.finditer(text):
            candidates.append((m.group(0), None))
        for m in _PATH_RE.finditer(text):
            candidates.append((m.group(2).strip(), None))
        for m in _PATH_RE_SPACED.finditer(text):
            candidates.append((m.group(2).strip(), None))

        for value, forced in candidates:
            add(value, forced, ts)
            if len(out) >= limit:
                return out
    return out


_ACTIVITY_PATH_KEYS = ("resolved_path", "file_path", "path", "file", "url", "target")


def _extract_activity(messages: List[dict], limit: int = 8) -> List[dict]:
    """从尾部消息提取最近工具活动，供前端在事件流缺位时兜底
    （被动事件驱动的补充：tool_progress 关闭 / agent 不 emit 时，活动区
    不再永远「暂无活动」）。assistant 消息不取——会与 message.complete
    事件重复；todo 工具排除——已由 todo 面板表达。
    返回 time 为 epoch 毫秒（前端 relativeTime 需要；DB 存的是秒）。
    """
    out: List[dict] = []
    for msg in reversed(messages):
        if msg.get("role") != "tool":
            continue
        name = str(msg.get("tool_name") or "tool")
        if name == "todo":
            continue

        if re.search(r"test|pytest|vitest|check|verify", name, re.I):
            type_ = "test"
        elif re.search(r"read|search|list|get|view|inspect", name, re.I):
            type_ = "read"
        elif re.search(r"write|patch|edit|create|install|build", name, re.I):
            type_ = "write"
        else:
            type_ = "tool"

        # 工具消息 content 是「结果」不是「参数」——JSON 时尽力挑一个
        # 路径/文件名键做上下文（如 skill_view 的 file）；挑不出就退回工具名。
        text = name
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            snippet = None
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    for key in _ACTIVITY_PATH_KEYS:
                        v = parsed.get(key)
                        if isinstance(v, str) and v.strip():
                            snippet = v.strip()[:60]
                            break
            except (ValueError, TypeError):
                cand = content.strip()
                if "\n" not in cand and len(cand) <= 80:
                    snippet = cand
            if snippet:
                text = f"{name}: {snippet}"

        ts = msg.get("timestamp") or 0
        try:
            time_ms = int(float(ts) * 1000)
        except (TypeError, ValueError):
            time_ms = 0

        out.append({"type": type_, "text": text[:80], "time": time_ms})
        if len(out) >= limit:
            break
    return out


def _resolve_stored_session_id(session_id: str, db: Optional[SessionDB] = None) -> str:
    """把插件传入的 session_id 解析为 state.db 的 stored id。

    插件传的是 desktop 的 activeSessionId —— session.create 返回的
    ``session_id``（runtime sid，8 位 hex）。state.db 以 stored_session_id
    （``20260805_...``）为键，直接查会落空。解析链（全部解析到同一套 id，
    title / todos / outputs 共用，保证一致性）：
      1. 直接命中（session_id 本身就是 stored id）
      2. 前缀解析（stored id 的唯一前缀）
      3. tui_gateway ``_sessions`` 运行时映射（桌面运行期间的最新真相，
         命中时写回持久映射，自愈旧条目）
      4. 持久映射文件（覆盖 desktop 重启后 _sessions 为空的恢复场景）
    """
    if not session_id:
        return session_id
    db = db or _get_db()
    # DB 异常直接传播（#10）：state.db 锁/损坏时让 snapshot() 转 500，
    # 前端才能区分「会话真没了」（resolved:false）与「DB 暂时不可用」（500）。
    if db.get_session(session_id):
        return session_id
    resolved = db.resolve_session_id(session_id)
    if resolved:
        return resolved
    # 运行时映射优先于持久映射：桌面运行期间 _sessions 是真相来源，
    # 持久文件只兜底「serve 重启后 _sessions 为空」的恢复场景。
    # 若先查持久文件，会话被重建并复用同一 runtime sid 时，旧映射会在
    # _sessions 已含更新映射的情况下胜出 → 快照读错会话。
    # 命中 _sessions 时写回持久映射，旧条目随 _remember_runtime_sid 自愈。
    try:
        from tui_gateway.server import _sessions  # noqa: PLC0415

        for sid, sess in _sessions.items():
            if sid == session_id:
                stored = sess.get("session_key")
                if stored:
                    stored = str(stored)
                    _remember_runtime_sid(session_id, stored)
                    return stored
            key = sess.get("session_key")
            if key and str(key) == session_id:
                return str(key)
    except Exception:
        pass
    # 持久映射：重启后 _sessions 为空时仍能解析旧 runtime sid
    # 兜底校验（#21）：mapped 会话必须存在且近期活跃，否则丢弃并清理脏条目——
    # desktop 新会话复用旧 runtime sid 时，脏映射会把当前激活会话解析到旧会话
    # （快照/Activity 整面板显示错数据：title/todos/outputs 全来自旧会话）。
    # 宁可返回原 sid（resolved=false，前端显示「数据不可用」）也不显示错误会话；
    # _sessions 就绪后第 3 步正常命中并写回自愈，此兜底只在异常窗口期生效。
    try:
        mapped = _load_runtime_sid_map().get(session_id)
        if mapped:
            sess = db.get_session(str(mapped))
            if sess:
                last_at = sess.get("last_activity_at") or 0
                try:
                    last_at_f = float(last_at)
                except (TypeError, ValueError):
                    last_at_f = 0
                if last_at_f and time.time() - last_at_f < _RUNTIME_SID_FRESH_S:
                    return str(mapped)
            # 会话不存在 / 已不活跃：清理脏条目，防粘滞
            _forget_runtime_sid(session_id)
    except Exception:
        pass
    return session_id


def _max_active_message_id(db: SessionDB, session_id: str) -> Optional[int]:
    """活跃消息的最大 AUTOINCREMENT id（rewind 变化检测）。

    message_count 计全部行（含 soft-deleted），rewind 只把行翻成
    active=0、不删行 → 总数不变但 active 窗口收缩。MAX(id) WHERE
    active=1 在 rewind 后会变小，与 total 互补组成快照失效键。
    复刻 message_count 的 _lock/_conn 模式（同一把锁；走
    idx_messages_session_active 索引的单条聚合查询，轻量），
    不新增 SessionDB 公开 API。失败返回 None（版本戳退化为 total|title）。
    """
    try:
        with db._lock:
            row = db._conn.execute(
                "SELECT MAX(id) FROM messages WHERE session_id = ? AND active = 1",
                (session_id,),
            ).fetchone()
    except Exception:
        return None
    val = row[0] if row else None
    return int(val) if val is not None else None


# ── HTTP 路由 ────────────────────────────────────────────────────────────
@router.get("/snapshot")
def snapshot(session_id: str = Query(...)):
    """一次拿全：todo 快照 + 产物 + 标题。前端事件流为主，此接口兜底。"""
    if not session_id:
        raise HTTPException(400, "session_id required")
    try:
        db = _get_db()
        stored_id = _resolve_stored_session_id(session_id, db)
        # 快照版本戳 = total | last_active_id | title，作为缓存失效键
        # （从「仅 messageCount」升级，见 #5）：
        #   - total          抓新增消息（原主键）
        #   - last_active_id 抓 rewind：soft-delete 不删行，总数不变但
        #     active 窗口收缩（_max_active_message_id）
        #   - title          抓改名（get_session 单行主键查询，轻量）
        # 三者均为轻量查询，不做 tail_hash——读尾部 400 条做哈希会吃掉
        # 缓存收益。
        total = db.message_count(stored_id)
        last_active_id = _max_active_message_id(db, stored_id)
        try:
            session = db.get_session(stored_id)
        except Exception:
            session = None
        resolved = stored_id != session_id or bool(session)
        title = (session.get("title") or "") if session else ""
        snapshot_version = f"{total}|{last_active_id}|{title}"
        # 快照缓存（#5 升级）：版本戳未变 → 直接复用上次响应，
        # 跳过全量 DB 读 + 正则扫描（2s 轮询的空转成本）
        _cached = _snapshot_cache.get(stored_id)
        if _cached and _cached[0] == snapshot_version:
            return _cached[1]
        # 截断策略：只取会话尾部（最新）消息。todo 状态与产物都是“当前
        # 状态”，旧消息里的历史 todo/产物没有展示价值；get_messages 是
        # 插入正序 + LIMIT 从头取，长会话（>400 条）里最新数据会被截掉，
        # 所以用 message_count 做 offset 分页取最后 400 条。
        tail = min(400, total)
        messages = db.get_messages(stored_id, limit=tail, offset=max(0, total - tail))
        if not messages and total > 0:
            # message_count 含 soft-deleted(rewind)行，而 get_messages 默认
            # 只返回 active=1 —— offset 可能越过 active 范围。回退：全量取尾。
            all_msgs = db.get_messages(stored_id, limit=None)
            messages = all_msgs[-tail:]
    except Exception as exc:  # state.db 锁 / 损坏 / 不存在
        logger.warning("work-sidebar snapshot failed (session=%s): %s", session_id, exc)
        raise HTTPException(500, "state.db read failed") from exc

    if not title:
        # 无标题会话：从消息推断，避免恢复/新建会话显示空白
        try:
            title = _infer_title(messages)
        except Exception:
            pass

    # messageCount 必须是真实总数（message_count 计全部行，含 soft-deleted），
    # 不能用 len(messages)：messages 是截断后的尾部窗口（最多 400 条），
    # 长会话（>400 条）里它会恒为 400，前端拿它当「数据是否变化」的指示器
    # 时会误判为没变化 → 快照回填被跳过，面板冻结。真实总数随每条新消息
    # 递增，变化检测才能覆盖长会话（rewind 只 soft-delete，不影响 total）。
    data = {
        "todos": _latest_session_todos(messages),
        "outputs": _extract_outputs(messages),
        # 被动事件驱动的兜底：事件流缺位时前端用此列表填充 Activity
        "activity": _extract_activity(messages),
        "title": title,
        "messageCount": total,
        # B2：前端事件过滤的漂移容错锚点。快照已把 runtime sid 解析成
        # stored id——前端据此学习「runtime sid → stored id」映射：事件
        # 的 sid 与 activeSessionId 不同但映射到同一 stored id 时视为
        # 同一会话（runtime sid 漂移），否则丢弃（跨会话污染防护）。
        "storedId": stored_id,
        # 前端「数据是否变化」的失效键：messageCount 只覆盖新增消息，
        # rewind / 改标题会漏判 → 快照被永久缓存旧值。见 _max_active_message_id。
        "snapshotVersion": snapshot_version,
        "resolved": resolved,
    }
    if len(_snapshot_cache) >= _SNAPSHOT_CACHE_MAX:
        _snapshot_cache.pop(next(iter(_snapshot_cache)), None)
    _snapshot_cache[stored_id] = (snapshot_version, data)
    return data
