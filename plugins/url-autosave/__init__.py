"""url-autosave: 用户发链接时自动将 web_extract 结果落盘到 kaah/inbox/ v1.1

post_tool_call hook 机械执法：
- 拦截 web_extract 返回
- 查询 state.db 获取用户上一条消息
- 含 URL + 提取成功 → 自动写入 kaah/inbox/
- 提取失败 → 不写
- 去重：同一 URL 不重复存
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

INBOX_DIR = Path.home() / "kaah" / "inbox"
STATE_DB = Path.home() / ".hermes" / "state.db"


def _extract_urls(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r'https?://[^\s<>"\')\]》】]+', text)


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _url_already_saved(url: str) -> bool:
    target = _url_hash(url)
    try:
        for f in INBOX_DIR.glob("*.md"):
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                if f"url_hash: {target}" in content:
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _safe_filename(url: str, title: str = "") -> str:
    domain = re.sub(r'https?://(www\.)?', '', url).split('/')[0].replace('.', '_')
    today = datetime.now().strftime("%Y%m%d")
    if title and len(title) > 3:
        safe = re.sub(r'[\\/:*?"<>|\s]+', '_', str(title)[:50]).strip('_')
        return f"{today}_{safe}.md"
    return f"{today}_{domain}_{_url_hash(url)}.md"


def _get_last_user_message(session_id: str) -> str:
    """从 state.db 获取用户最后一条消息"""
    try:
        db = sqlite3.connect(str(STATE_DB))
        row = db.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1",
            (session_id,)
        ).fetchone()
        db.close()
        return row[0] if row else ""
    except Exception as e:
        logger.warning(f"[url-autosave] state.db 查询失败: {e}")
        return ""


def _post_tool_call(
    tool_name: str = "",
    args: dict = None,
    result: str = "",
    session_id: str = "",
    **kwargs,
):
    """拦截 web_extract，用户发了链接就自动落盘。"""
    if tool_name != "web_extract":
        return

    if not session_id:
        return

    # 获取用户上一条消息
    user_msg = _get_last_user_message(session_id)
    urls = _extract_urls(user_msg)
    if not urls:
        return  # 用户没发链接 → Agent 自己搜的

    # 解析 result（可能是 JSON 字符串）
    try:
        result_data = json.loads(result) if isinstance(result, str) else result
    except json.JSONDecodeError:
        return

    results = result_data.get("results", [])

    saved_count = 0
    for url in urls:
        if _url_already_saved(url):
            continue

        # 找对应 URL 的结果
        matched = None
        for r in results:
            if r.get("url", "").rstrip("/") == url.rstrip("/"):
                matched = r
                break
        if not matched:
            for r in results:
                if url.rstrip("/") in r.get("url", ""):
                    matched = r
                    break
        if not matched:
            continue
        if matched.get("error") or not matched.get("content"):
            continue

        title = matched.get("title", "")
        filename = _safe_filename(url, title)
        filepath = INBOX_DIR / filename
        url_hash_val = _url_hash(url)
        content = f"<!-- url: {url} -->\n<!-- url_hash: {url_hash_val} -->\n<!-- plugin: url-autosave -->\n\n"
        content += f"# {title or 'Untitled'}\n\n"
        content += f"> 来源：{url}\n\n"
        content += matched["content"]

        filepath.write_text(content, encoding="utf-8")
        saved_count += 1

    if saved_count:
        logger.warning(f"[url-autosave] 自动保存 {saved_count} 篇原文到 kaah/inbox/")


def register(ctx):
    """注册 post_tool_call hook"""
    ctx.register_hook("post_tool_call", _post_tool_call)
    logger.warning("[url-autosave] 插件已注册 v1.1")
