"""NOESIS-II memory plugin — MemoryProvider for NOESIS long-term memory.

Stores every conversation turn verbatim to NOESIS Working Memory (WM)
via k.capture(), consolidates to Long-Term Memory (LTM) on schedule,
and provides 3-level tiered recall before each response.

Architecture:
  - sync_turn()     — Called by MemoryManager after every LLM response
  - prefetch()      — Level 1: recall summaries before each turn
  - get_tool_schemas()  — Level 2/3 tools for the model to drill down
  - handle_tool_call()  — Dispatch Level 2 (links) and Level 3 (full text)
  - Daily consolidation via cron (noesis-daily-maintenance)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NOESIS kernel lazy import
# ---------------------------------------------------------------------------

_NOESIS_KERNEL = None


def _get_noesis_kernel():
    """Lazy-import and bootstrap the NOESIS kernel (called once on first use)."""
    global _NOESIS_KERNEL
    if _NOESIS_KERNEL is not None:
        return _NOESIS_KERNEL

    noesis_home = "/mnt/e/Project/NOESIS-CLEAN"  # Default for WSL; override via NOESIS_HOME env var
    noesis_home = os.environ.get("NOESIS_HOME", noesis_home)
    if noesis_home not in sys.path:
        sys.path.insert(0, noesis_home)

    try:
        from noesis_ii.kernel import get_kernel
        k = get_kernel()
        if not k._bootstrapped:
            k.bootstrap()
        _NOESIS_KERNEL = k
        logger.info("NOESIS kernel bootstrapped successfully")
    except Exception as e:
        logger.warning("Failed to load NOESIS kernel: %s", e)
        _NOESIS_KERNEL = None

    return _NOESIS_KERNEL


def _get_db_conn() -> Optional[sqlite3.Connection]:
    """Get a SQLite connection to the NOESIS DB for Level 2/3 queries."""
    k = _get_noesis_kernel()
    if k is None or not k.config.db_path:
        return None
    try:
        conn = sqlite3.connect(k.config.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.debug("NOESIS DB connection failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------

_NOESIS_MEMORY_PROVIDER_INSTANCE = None


class NOESISMemoryProvider(MemoryProvider):
    """MemoryProvider with 3-level tiered recall."""

    @property
    def name(self) -> str:
        return "noesis"

    def is_available(self) -> bool:
        """Check if NOESIS is reachable (kernel + db)."""
        try:
            k = _get_noesis_kernel()
            if k is None:
                return False
            conn = sqlite3.connect(k.config.db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM working_memory")
            cur.close()
            conn.close()
            return True
        except Exception as e:
            logger.debug("NOESIS availability check failed: %s", e)
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize the provider for a session. Wires NOESIS kernel."""
        _get_noesis_kernel()
        logger.info("NOESIS memory provider initialized for session %s", session_id)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Level 1 recall: retrieve relevant memory summaries before each turn.

        Called by MemoryManager before each LLM response. Returns formatted
        LTM summaries for the system prompt.
        """
        k = _get_noesis_kernel()
        if k is None:
            return ""

        if not query or len(query.strip()) < 2:
            return ""

        try:
            results = k.recall(query=query.strip(), top_k=5, strategy="ltm")
            if not results:
                return ""

            lines = [
                "Level 1 NOESIS 记忆提炼内容",
                "",
                "以下是记忆中检索到的相关条目摘要。如果需要查看某条记忆的关联网络",
                "或获取原文细节，请使用 noesis_level2 / noesis_level3 工具。",
                "",
            ]
            for r in results:
                node_id = r.get("id", "?")
                score = r.get("relevance_score", r.get("score", 0))
                summary = (r.get("summary", "") or "")[:200]
                lines.append(f"  节点 {node_id}, 相关度 {score:.3f}")
                lines.append(f"  摘要: {summary}")
                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            logger.debug("NOESIS prefetch failed (non-fatal): %s", e)
            return ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a completed turn to NOESIS Working Memory verbatim."""
        k = _get_noesis_kernel()
        if k is None:
            logger.warning("NOESIS kernel not available, skipping capture")
            return

        try:
            turn_text = f"用户: {user_content}\n助手: {assistant_content}"
            k.capture(
                content=turn_text,
                source="conversation",
            )
        except Exception as e:
            logger.warning("NOESIS capture failed (non-fatal): %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Expose Level 2 and Level 3 recall tools to the model."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "noesis_level2",
                    "description": "[NOESIS Level 2] 查询某条记忆的关联链接网络。输入节点ID，返回关联记忆列表（含摘要、关联类型、强度）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "integer",
                                "description": "记忆节点ID（来自 Level 1 返回结果中的 [id]）",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "返回关联数量上限",
                                "default": 20,
                            },
                        },
                        "required": ["node_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "noesis_level3",
                    "description": "[NOESIS Level 3] 获取某条记忆的完整原文。输入节点ID，返回完整原文内容及元数据。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "integer",
                                "description": "记忆节点ID（来自 Level 1/2 返回结果中的 [id]）",
                            },
                        },
                        "required": ["node_id"],
                    },
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch Level 2 (association links) and Level 3 (full text) calls."""
        if tool_name == "noesis_level2":
            return self._handle_level2(args)
        elif tool_name == "noesis_level3":
            return self._handle_level3(args)
        return f"Unknown NOESIS tool: {tool_name}"

    def _handle_level2(self, args: Dict[str, Any]) -> str:
        """Level 2: query associated links for a node."""
        node_id = args.get("node_id")
        top_k = args.get("top_k", 20)

        conn = _get_db_conn()
        if conn is None:
            return "ERROR: NOESIS database not available"

        try:
            # Get the node's own summary
            cur = conn.cursor()
            cur.execute("SELECT summary, category, weight FROM ltm_nodes WHERE id = ?", [node_id])
            node = cur.fetchone()
            if not node:
                return f"ERROR: Node #{node_id} not found"

            lines = [
                f"Level 2 节点 {node_id} 的关联网络",
                "",
                f"节点摘要: {node['summary'] or '(无摘要)'}",
                f"分类: {node['category'] or 'N/A'}, 权重: {node['weight'] or 'N/A'}",
                "",
                "以下是与该节点关联的记忆条目。",
                "",
            ]

            cur.execute("""
                SELECT ll.target_id, ll.relation_type, ll.strength,
                       ll.description, ln.summary AS target_summary
                FROM ltm_links ll
                JOIN ltm_nodes ln ON ll.target_id = ln.id
                WHERE ll.source_id = ?
                ORDER BY ll.strength DESC LIMIT ?
            """, [node_id, top_k])

            links = cur.fetchall()
            if not links:
                lines.append("  (该节点暂无关联链接)")
            else:
                for link in links:
                    lines.append(f"  目标节点 {link['target_id']}, 关系类型 {link['relation_type']}, 强度 {link['strength']:.2f}")
                    desc = link['description'] or ''
                    if desc:
                        lines.append(f"  描述: {desc}")
                    ts = (link['target_summary'] or '')[:150]
                    if ts:
                        lines.append(f"  关联摘要: {ts}")
                    lines.append("")

            return "\n".join(lines)
        except Exception as e:
            logger.warning("NOESIS Level 2 failed: %s", e)
            return f"ERROR: Level 2 query failed: {e}"
        finally:
            conn.close()

    def _handle_level3(self, args: Dict[str, Any]) -> str:
        """Level 3: get full original text for a node."""
        node_id = args.get("node_id")

        conn = _get_db_conn()
        if conn is None:
            return "ERROR: NOESIS database not available"

        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, content, summary, category, weight, source, tags, created_at, access_count "
                "FROM ltm_nodes WHERE id = ?",
                [node_id],
            )
            node = cur.fetchone()
            if not node:
                return f"ERROR: Node #{node_id} not found"

            tags = node['tags'] or ''
            content = node['content'] or '(内容为空)'
            # Truncate content if too long
            if len(content) > 5000:
                content = content[:5000] + "\n\n[... 内容过长，已截断至 5000 字符 ...]"

            lines = [
                f"Level 3 节点 {node_id} 完整原文",
                "",
                f"  分类: {node['category'] or 'N/A'}, 权重: {node['weight'] or 'N/A'}, 访问: {node['access_count'] or 0}次",
                f"  来源: {node['source'] or 'N/A'}",
                f"  标签: {tags}",
                f"  时间: {node['created_at'] or 'N/A'}",
                "",
                "原文:",
                content,
                "",
            ]

            return "\n".join(lines)
        except Exception as e:
            logger.warning("NOESIS Level 3 failed: %s", e)
            return f"ERROR: Level 3 query failed: {e}"
        finally:
            conn.close()

    def system_prompt_block(self) -> str:
        """Light system prompt note about NOESIS availability."""
        k = _get_noesis_kernel()
        if k is None:
            return ""
        try:
            conn = sqlite3.connect(k.config.db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM working_memory")
            wm_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ltm_nodes")
            ltm_count = cur.fetchone()[0]
            conn.close()
            return (
                f"Memory: NOESIS active -- {wm_count} WM entries, {ltm_count} LTM nodes. "
                f"Every conversation turn is stored verbatim. "
                f"Use noesis_level2 and noesis_level3 tools to drill down."
            )
        except Exception:
            return "[Memory: NOESIS active]"

    def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("NOESIS memory provider shut down")


# ---------------------------------------------------------------------------
# Plugin entry point (register pattern)
# ---------------------------------------------------------------------------

def register(ctx):
    """Register the NOESIS memory provider with the plugin context.

    Called by plugins/memory/__init__.py discovery mechanism.
    """
    global _NOESIS_MEMORY_PROVIDER_INSTANCE
    provider = NOESISMemoryProvider()
    _NOESIS_MEMORY_PROVIDER_INSTANCE = provider
    ctx.register_memory_provider(provider)
    logger.info("NOESIS memory provider registered")
