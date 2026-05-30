"""审批记忆 — 支持 Hindsight / Honcho / none 三种后端。

通过 config.yaml 配置：
  plugin_guard:
    memory:
      backend: hindsight    # hindsight | honcho | none
      bank: approval        # Hindsight bank 或 Honcho user_id

提供两个维度的查询：
  1. session 级 — 查询本 session 的审批历史（理解操作链条）
  2. 模式级 — 查询跨 session 的相似操作（信任度提升）
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "hindsight"
HINDSIGHT_URL = os.getenv("HINDSIGHT_URL", "http://localhost:8421")
HONCHO_URL = os.getenv("HONCHO_URL", "http://localhost:1819")
TIMEOUT = 3

# ── 模式 key 生成：用于跨 session 匹配相似操作 ────────────────────


def _build_pattern_key(tool_name: str, args: Dict[str, Any]) -> str:
    """生成模式 key，用于跨 session 匹配相似操作。

    例: write_file /etc/nginx/nginx.conf → "write_file/etc/nginx/"
         terminal rm -rf node_modules     → "terminal/rm"
    """
    if tool_name in ("write_file", "patch"):
        path = str(args.get("path", ""))
        # 提取目录前缀（如 /etc/nginx/）
        dirs = path.rsplit("/", 1)
        if len(dirs) > 1:
            return f"{tool_name}{dirs[0]}/"
        return f"{tool_name}{path}"

    if tool_name == "terminal":
        cmd = str(args.get("command", "")).strip()
        # 提取第一个词作为命令类型
        first_word = re.split(r"[;\s|&]", cmd)[0].strip()
        if first_word:
            return f"terminal/{first_word}"
        return "terminal/other"

    if tool_name == "delegate_task":
        goal = str(args.get("goal", ""))
        # 提取前 3 个有意义的词
        words = goal.lower().split()
        key_words = [w for w in words if w not in ("the", "a", "an", "is", "in", "to", "of")][:3]
        if key_words:
            return f"delegate_task/ {' '.join(key_words)}"
        return "delegate_task/other"

    return tool_name


# ── 后端选择 ───────────────────────────────────────────────────────


def _get_backend(cfg: Dict[str, Any]) -> str:
    mem_cfg = cfg.get("memory", {})
    if not isinstance(mem_cfg, dict):
        return DEFAULT_BACKEND
    backend = mem_cfg.get("backend", DEFAULT_BACKEND)
    if backend == "none":
        return "none"
    if backend in ("hindsight", "honcho"):
        return backend
    return DEFAULT_BACKEND


# ══════════════════════════════════════════════════════════════════
# Hindsight HTTP API
# ══════════════════════════════════════════════════════════════════


def _hindsight_api(endpoint: str, payload: Dict[str, Any]) -> Optional[Dict]:
    try:
        url = f"{HINDSIGHT_URL}/{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("Hindsight API failed (%s): %s", endpoint, exc)
        return None


def _hindsight_retain(content: str, tags: list, cfg: Dict[str, Any]) -> None:
    bank = cfg.get("memory", {}).get("bank", "approval") if isinstance(cfg.get("memory"), dict) else "approval"
    _hindsight_api("retain", {
        "content": content,
        "context": "approval_decision",
        "tags": tags,
        "bank": bank,
    })


def _hindsight_recall_extended(query_parts: list, cfg: Dict[str, Any], limit: int = 5) -> list:
    """召回并返回完整 memory 列表（含 content + tags）。"""
    bank = cfg.get("memory", {}).get("bank", "approval") if isinstance(cfg.get("memory"), dict) else "approval"
    result = _hindsight_api("recall", {
        "query": " ".join(query_parts),
        "bank": bank,
    })
    if result:
        return result.get("memories", [])[:limit]
    return []


# ══════════════════════════════════════════════════════════════════
# Honcho API
# ══════════════════════════════════════════════════════════════════


def _honcho_api(endpoint: str, payload: Dict[str, Any]) -> Optional[Dict]:
    try:
        url = f"{HONCHO_URL}/{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("Honcho API failed (%s): %s", endpoint, exc)
        return None


def _honcho_retain(content: str, tags: list, cfg: Dict[str, Any]) -> None:
    user_id = cfg.get("memory", {}).get("bank", "approval") if isinstance(cfg.get("memory"), dict) else "approval"
    _honcho_api("memories", {
        "user_id": user_id,
        "content": content,
        "metadata": {"type": "approval_decision", "tags": tags},
    })


def _honcho_recall_extended(query_parts: list, cfg: Dict[str, Any], limit: int = 5) -> list:
    user_id = cfg.get("memory", {}).get("bank", "approval") if isinstance(cfg.get("memory"), dict) else "approval"
    result = _honcho_api(f"memories/{user_id}", {})
    if result and isinstance(result, list):
        query = " ".join(query_parts).lower()
        matches = []
        for m in result:
            content = m.get("content", "").lower()
            if any(q in content for q in query_parts):
                matches.append(m)
        return matches[:limit]
    return []


# ══════════════════════════════════════════════════════════════════
# 公共接口
# ══════════════════════════════════════════════════════════════════


def record_decision(
    tool_name: str,
    args: Dict[str, Any],
    verdict: str,
    reason: str,
    session_id: str = "",
    cfg: Dict[str, Any] = None,
) -> None:
    """记录审批决策到记忆后端。

    存储 content 包含 session_id 以便后续按 session 查询。
    存储 tags 包含 pattern_key 以便跨 session 模式匹配。
    """
    if cfg is None:
        cfg = {}
    backend = _get_backend(cfg)
    if backend == "none":
        return

    pattern_key = _build_pattern_key(tool_name, args)
    args_summary = _summarize_args(args)
    content = (
        f"[{session_id or 'no_session'}] "
        f"approval:{verdict}:{tool_name}:{args_summary} "
        f"reason={reason[:80]} pk={pattern_key}"
    )

    tags = _build_tags(tool_name, args, session_id, pattern_key)

    try:
        if backend == "hindsight":
            _hindsight_retain(content, tags, cfg)
        elif backend == "honcho":
            _honcho_retain(content, tags, cfg)
    except Exception as exc:
        logger.debug("Failed to record decision (%s): %s", backend, exc)


def query_session_history(
    session_id: str,
    cfg: Dict[str, Any] = None,
    limit: int = 5,
) -> str:
    """查询本 session 的审批历史。

    用于 ACP prompt 注入：让 ACP 知道本次会话中之前审批过什么操作。
    返回格式化的文本，可直接注入 prompt；无历史时返回空字符串。
    """
    if not session_id or cfg is None:
        return ""
    cfg = cfg if cfg is not None else {}
    backend = _get_backend(cfg)
    if backend == "none":
        return ""

    try:
        if backend == "hindsight":
            memories = _hindsight_recall_extended([session_id, "approval"], cfg, limit)
        elif backend == "honcho":
            memories = _honcho_recall_extended([session_id, "approval"], cfg, limit)
        else:
            return ""

        if not memories:
            return ""

        lines = []
        for m in memories[:limit]:
            content = m.get("content", "")
            lines.append(f"  - {content}")
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("Failed to query session history: %s", exc)
        return ""


def query_pattern_history(
    tool_name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any] = None,
    limit: int = 5,
) -> str:
    """查询跨 session 的相似操作审批历史。

    用于 ACP prompt 注入：展示历史上相似操作的 ALLOW/DENY 统计。
    返回格式化的文本；无历史时返回空字符串。
    """
    if cfg is None:
        cfg = {}
    backend = _get_backend(cfg)
    if backend == "none":
        return ""

    pattern_key = _build_pattern_key(tool_name, args)
    query_parts = [pattern_key, "approval"]

    try:
        if backend == "hindsight":
            memories = _hindsight_recall_extended(query_parts, cfg, limit)
        elif backend == "honcho":
            memories = _honcho_recall_extended(query_parts, cfg, limit)
        else:
            return ""

        if not memories:
            return ""

        # 统计 ALLOW/DENY 次数
        allows = 0
        denies = 0
        latest = ""
        for m in memories:
            content = m.get("content", "")
            if ":ALLOW:" in content:
                allows += 1
            elif ":DENY:" in content:
                denies += 1
            if not latest:
                latest = content[:120]

        summary = f"{pattern_key}: {allows} 次 ALLOW, {denies} 次 DENY"
        if latest:
            summary += f"\n    最近: {latest}"
        return summary
    except Exception as exc:
        logger.debug("Failed to query pattern history: %s", exc)
        return ""


# ── 辅助函数 ───────────────────────────────────────────────────────


def _build_tags(
    tool_name: str, args: Dict[str, Any],
    session_id: str, pattern_key: str,
) -> list:
    """构建 Hindsight/Honcho 标签。"""
    tags = ["approval", tool_name, pattern_key]
    if session_id:
        tags.append(f"sid:{session_id}")

    if tool_name == "terminal":
        cmd = str(args.get("command", ""))
        for kw in ("rm", "mv", "cp", "chmod", "chown", "sudo", "git"):
            if kw in cmd.lower():
                tags.append(kw)
    elif tool_name in ("write_file", "patch"):
        path = str(args.get("path", ""))
        if "/etc/" in path:
            tags.append("system_path")
        elif "/home/" in path:
            tags.append("home_path")
    return tags


def _summarize_args(args: Dict[str, Any]) -> str:
    if not args:
        return "()"
    parts = []
    for k, v in sorted(args.items()):
        v_str = str(v)
        if len(v_str) > 50:
            v_str = v_str[:50] + "..."
        parts.append(f"{k}={v_str}")
    return "(" + ", ".join(parts[:3]) + ")"
