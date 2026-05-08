"""ClickHouse memory provider — multi-tenant memory storage for Hermes Agent.

Stores conversational memory in ClickHouse with PARTITION BY user_id for
full data isolation between users. LLM annotations (topic, importance,
frequency) are computed at write time by the agent itself via system prompt
instructions.

Config
------
Set ``memory.provider: clickhouse`` in config.yaml.

Optional plugin config under ``plugins.clickhouse``:

.. code-block:: yaml

    plugins:
      clickhouse:
        host: localhost                # default: localhost
        port: 8123                     # default: 8123 (HTTP)
        user: default                  # default: default
        password: ''                   # default: '' (use .env for secrets)
        database: hermes_memory        # default: hermes_memory
        table: events                  # default: events
        ttl_days: 30                   # default: 30
        max_prefetch: 50               # max records returned by prefetch

Requirements
------------
- clickhouse-connect (pip install clickhouse-connect)

Schema
------
Auto-created on first initialize() if it doesn't exist.

.. code-block:: sql

    CREATE TABLE IF NOT EXISTS hermes_memory.events (
        user_id          String,
        session_id       String,
        ts               DateTime,
        turn_number      UInt32,
        role             LowCardinality(String),
        content          String,
        topic            String,
        importance       Float32,
        frequency        UInt32,
        is_repeat        Bool,
        related_topics   Array(String),
        emotion          String,
        platform         String,
        channel_id       String,
        model            String,
        parent_session_id String
    ) ENGINE = MergeTree()
    PARTITION BY (user_id, toYYYYMM(ts))
    ORDER BY (user_id, importance * frequency DESC, ts DESC)
    TTL ts + INTERVAL 30 DAY DELETE
    SETTINGS index_granularity = 8192;
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider
from hermes_cli.config import cfg_get
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default patterns for topic/importance extraction (fallback when no LLM annotation)
# ---------------------------------------------------------------------------
TOPIC_PATTERNS: list[tuple[str, str]] = [
    (r"(?:trading|moex|stock|market|futoi|hunter|backtest|rsi|signal|scan|акци)", "trading"),
    (r"(?:memory|memor|knowledge.?graph|hipporag|diary)", "memory-system"),
    (r"(?:devops|nginx|ssh|deploy|docker|systemd|cron|config|firewall)", "devops"),
    (r"(?:vpn|xray|wireguard|reality|proxy|tunnel)", "vpn"),
    (r"(?:telegram|bot|gateway|chat|channel)", "messaging"),
    (r"(?:hermes|agent|skill|tool|mcp|model|llm|openrouter)", "hermes-agent"),
    (r"(?:health|doctor|medkarta|hospital|врач|лечени)", "health"),
    (r"(?:habr|article|стат|пост|блог|write|публикаци)", "writing"),
    (r"(?:clickhouse|database|storage|migrat)", "infrastructure"),
    (r"(?:freelanc|order|project|заказ)", "freelance"),
]

IMPORTANCE_KEYWORDS: dict[str, float] = {
    "security": 0.9, "vulnerability": 0.9, "production": 0.85,
    "release": 0.8, "deploy": 0.75, "migration": 0.75,
    "broken": 0.8, "crash": 0.85, "error": 0.7,
    "plan": 0.6, "feature": 0.6, "architecture": 0.7,
    "question": 0.4, "help": 0.3, "hi": 0.1, "thanks": 0.1,
}

TRIVIAL_WORDS = {"спасибо", "понял", "ok", "okay", "хорошо", "ладно",
                 "понятно", "thanks", "agree", "согласен", "+1", "👍",
                 "давай", "окей", "hello", "hi", "привет"}


def _is_trivial(text: str) -> bool:
    cleaned = text.strip().lower().rstrip(".!?,;:")
    return cleaned in TRIVIAL_WORDS or len(cleaned) < 5


def _extract_topic(text: str) -> str:
    """Rule-based topic extraction — fallback when LLM annotation isn't available."""
    lower = text.lower()
    for pattern, topic in TOPIC_PATTERNS:
        if re.search(pattern, lower):
            return topic
    return "general"


def _estimate_importance(text: str) -> float:
    """Estimate importance (0.0–1.0) from text content."""
    if _is_trivial(text):
        return 0.05
    lower = text.lower()
    score = 0.3  # baseline
    for keyword, weight in IMPORTANCE_KEYWORDS.items():
        if keyword in lower:
            score = max(score, weight)
    # Length bonus: longer messages tend to be more important
    if len(text) > 500:
        score = min(1.0, score + 0.1)
    elif len(text) > 200:
        score = min(1.0, score + 0.05)
    return round(score, 2)


def _is_repeat(current: str, recent: list[str]) -> bool:
    """Check if current message repeats a recent one."""
    if not recent:
        return False
    cur_lower = current.lower().strip()
    for prev in recent[-5:]:
        if not prev:
            continue
        # Simple overlap check
        prev_lower = prev.lower().strip()
        words_cur = set(cur_lower.split())
        words_prev = set(prev_lower.split())
        if not words_cur or not words_prev:
            continue
        overlap = len(words_cur & words_prev) / max(len(words_cur), len(words_prev))
        if overlap > 0.6:
            return True
    return False


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class ClickHouseMemoryProvider(MemoryProvider):
    """Multi-tenant memory provider backed by ClickHouse.

    Each user's data is isolated via PARTITION BY user_id. The provider
    stores every turn with metadata (topic, importance, frequency) and
    retrieves the most relevant past context for prefetch.
    """

    def __init__(self):
        self._name = "clickhouse"
        self._client = None
        self._user_id = ""
        self._session_id = ""
        self._platform = ""
        self._model = ""
        self._recent_contents: list[str] = []
        self._config: dict = {}
        self._table_created = False

    # -- Config helpers -------------------------------------------------------

    def _load_config(self) -> dict:
        """Read clickhouse-specific config from config.yaml."""
        try:
            config_path = get_hermes_home() / "config.yaml"
            if not config_path.exists():
                return {}
            import yaml
            with open(config_path) as f:
                full = yaml.safe_load(f) or {}
            return cfg_get(full, "plugins", "clickhouse", default={}) or {}
        except Exception:
            return {}

    @property
    def name(self) -> str:
        return self._name

    # -- Lifecycle ------------------------------------------------------------

    def is_available(self) -> bool:
        """Check that clickhouse-connect is installed."""
        try:
            import clickhouse_connect  # noqa: F401
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str = "", **kwargs) -> None:
        """Initialize ClickHouse connection and ensure schema exists.

        Receives ``user_id`` from the agent via kwargs.  The user_id
        is stored as ``self._user_id`` and used in all subsequent
        prefetch/sync_turn calls as a **fallback** — the primary
        isolation is that every SQL query includes ``WHERE user_id = ?``,
        so even if user_id is stale, it can only corrupt within that
        user's partition, not across users.

        Kwargs always include:
          - user_id (str): Platform user identifier (gateway sessions)
          - hermes_home (str): Active HERMES_HOME path
          - platform (str): "cli", "telegram", "discord", ...
          - agent_context (str): "primary", "subagent", "cron"
          - agent_identity (str): Profile name
          - model (str): Current model name
        """
        self._config = self._load_config()
        self._session_id = session_id or ""
        self._user_id = kwargs.get("user_id", self._user_id or "")
        self._platform = kwargs.get("platform", self._platform or "")
        self._model = kwargs.get("model", self._model or "")

        # Only reconnect if not already connected
        if self._client is None:
            self._connect()

        # Ensure schema exists (once per process)
        if not self._table_created and self._client:
            self._ensure_schema()
            self._table_created = True

    def _connect(self) -> None:
        """Establish ClickHouse HTTP connection."""
        try:
            import clickhouse_connect
        except ImportError:
            logger.error("clickhouse-connect not installed. Install with: pip install clickhouse-connect")
            return

        cfg = self._config
        host = cfg.get("host", "localhost")
        port = int(cfg.get("port", 8123))
        user = cfg.get("user", "default")
        password = cfg.get("password", "")
        database = cfg.get("database", "hermes_memory")

        try:
            self._client = clickhouse_connect.get_client(
                host=host,
                port=port,
                username=user,
                password=password,
                database=database,
                # Connection pool settings
                connect_timeout=10,
                send_receive_timeout=30,
                retry=2,
            )
            logger.info("Connected to ClickHouse at %s:%s/%s", host, port, database)
        except Exception as e:
            logger.error("Failed to connect to ClickHouse: %s", e)
            self._client = None

    def _ensure_schema(self) -> None:
        """Create the events table if it doesn't exist."""
        if not self._client:
            return

        cfg = self._config
        database = cfg.get("database", "hermes_memory")
        table = cfg.get("table", "events")
        ttl_days = int(cfg.get("ttl_days", 30))

        self._client.command(f"CREATE DATABASE IF NOT EXISTS {database}")

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {database}.{table} (
            user_id          String,
            session_id       String,
            ts               DateTime,
            turn_number      UInt32,
            role             LowCardinality(String),
            content          String,
            topic            String,
            importance       Float32,
            frequency        UInt32,
            is_repeat        UInt8,
            related_topics   Array(String),
            emotion          String,
            platform         String,
            channel_id       String,
            model            String,
            parent_session_id String
        ) ENGINE = MergeTree()
        PARTITION BY (user_id, toYYYYMM(ts))
        ORDER BY (user_id, importance * frequency DESC, ts DESC)
        TTL ts + INTERVAL {ttl_days} DAY DELETE
        SETTINGS index_granularity = 8192
        """
        self._client.command(create_sql)

        # Full-text index
        try:
            self._client.command(
                f"ALTER TABLE {database}.{table} ADD INDEX IF NOT EXISTS fts_content "
                f"content TYPE tokenbf_v1(512, 3, 0) GRANULARITY 1"
            )
        except Exception:
            pass  # may already exist

        logger.info("Schema ensured for %s.%s", database, table)

    def system_prompt_block(self) -> str:
        """Instruct the LLM to use memory and annotate turns.

        This block tells the agent HOW to interact with the ClickHouse
        memory system — including the annotation format for prefetched
        context.
        """
        return (
            "## Memory System (ClickHouse)\n\n"
            "You have persistent memory backed by ClickHouse. "
            "Previous turns are recalled automatically before each response.\n\n"
            "### Memory features:\n"
            "- Each conversation turn is stored with metadata: topic, importance, frequency\n"
            "- Prefetched context shows the most relevant past conversations\n"
            "- You have tools to search and manage memory\n\n"
            "### How to annotate your responses:\n"
            "When responding, include a TM (Topic Marker) block at the end of your response "
            "in this exact JSON format (invisible to the user — stripped before delivery):\n\n"
            '```json\n{\n  "topic": "trading",\n  "importance": 0.8,\n  "emotion": "neutral"\n}\n```\n\n'
            "This annotation is never shown to the user. It helps the memory system "
            "categorize and rank this conversation for future recall."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Retrieve relevant past context from ClickHouse.

        Returns a formatted string with the most important past conversations
        for the current user, ordered by importance* frequency.
        """
        if not self._client:
            return ""

        user_id = self._user_id
        if not user_id:
            return ""

        cfg = self._config
        database = cfg.get("database", "hermes_memory")
        table = cfg.get("table", "events")
        max_prefetch = int(cfg.get("max_prefetch", 50))

        try:
            # Get top records by importance*frequency for this user
            rows = self._client.query_df(
                f"""
                SELECT ts, role, content, topic, importance, frequency, is_repeat, platform
                FROM {database}.{table}
                WHERE user_id = %(user_id)s
                ORDER BY importance * frequency DESC, ts DESC
                LIMIT %(limit)s
                """,
                parameters={"user_id": user_id, "limit": max_prefetch},
            )
        except Exception as e:
            logger.warning("ClickHouse prefetch failed: %s", e)
            return ""

        if rows.empty:
            return ""

        # Format as context block
        lines = ["## Memory Context (past conversations)", ""]
        for _, row in rows.iterrows():
            ts = row["ts"]
            role = row["role"]
            content = str(row["content"])[:200]
            topic = row.get("topic", "")
            imp = row.get("importance", 0)
            platform = row.get("platform", "")
            marker = ""

            lines.append(f"[{ts} | {role} | {topic} | imp:{imp}] {content}")
            if platform:
                lines[-1] += f" ({platform})"

        lines.append("")
        return "\n".join(lines)

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
        """Persist a completed turn to ClickHouse.

        Attempts to extract LLM annotation (topic, importance, emotion)
        from the assistant's response, then stores both user and assistant
        messages with computed or extracted metadata.
        """
        if not self._client:
            return

        user_id = self._user_id
        if not user_id:
            return

        cfg = self._config
        database = cfg.get("database", "hermes_memory")
        table = cfg.get("table", "events")

        # Try to extract LLM annotation from assistant response
        annotation = self._extract_annotation(assistant_content)

        # Clean annotation block from assistant content for storage
        clean_assistant = self._strip_annotation(assistant_content)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Determine topics and importance
        user_topic = annotation.get("topic", "") or _extract_topic(user_content)
        asst_topic = annotation.get("topic", "") or _extract_topic(clean_assistant)
        user_importance = annotation.get("importance", 0) or _estimate_importance(user_content)
        asst_importance = _estimate_importance(clean_assistant)
        emotion = annotation.get("emotion", "")

        # Check for repeats
        user_is_repeat = 1 if _is_repeat(user_content, self._recent_contents) else 0
        self._recent_contents.append(user_content)

        try:
            self._client.insert(
                database + "." + table,
                [
                    [user_id, session_id or self._session_id, now,
                     self._get_turn_number(), "user", user_content,
                     user_topic, user_importance, 1, user_is_repeat,
                     [], emotion,
                     self._platform, "", self._model, ""],
                    [user_id, session_id or self._session_id, now,
                     self._get_turn_number(), "assistant", clean_assistant,
                     asst_topic, asst_importance, 1, 0,
                     [], emotion,
                     self._platform, "", self._model, ""],
                ],
                column_names=[
                    "user_id", "session_id", "ts", "turn_number", "role",
                    "content", "topic", "importance", "frequency", "is_repeat",
                    "related_topics", "emotion",
                    "platform", "channel_id", "model", "parent_session_id",
                ],
            )
        except Exception as e:
            logger.warning("ClickHouse sync_turn failed: %s", e)

    def _get_turn_number(self) -> int:
        """Get the next turn number for the current session."""
        if not self._client or not self._user_id:
            return 1
        try:
            cfg = self._config
            database = cfg.get("database", "hermes_memory")
            table = cfg.get("table", "events")
            result = self._client.query_df(
                f"SELECT max(turn_number) as mx FROM {database}.{table} "
                f"WHERE user_id = %(user_id)s AND session_id = %(session_id)s",
                parameters={"user_id": self._user_id,
                            "session_id": self._session_id or ""},
            )
            mx = result["mx"].iloc[0]
            return (int(mx) + 1) if mx else 1
        except Exception:
            return 1

    def _extract_annotation(self, text: str) -> dict:
        """Extract JSON annotation block from the assistant's response.

        Expected format (invisible to user):
        ```json
        {"topic": "trading", "importance": 0.8, "emotion": "neutral"}
        ```
        """
        # Look for JSON block at the end of the text
        match = re.search(
            r'```json\s*({[^}]+})\s*```\s*$',
            text,
            re.DOTALL,
        )
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, KeyError):
                pass
        return {}

    def _strip_annotation(self, text: str) -> str:
        """Remove the annotation JSON block from the end of the response."""
        return re.sub(
            r'\n?```json\s*{[^}]+}\s*```\s*$',
            '',
            text,
        ).strip()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for memory search and stats."""
        return [
            {
                "name": "memory_search",
                "description": "Search past conversations in ClickHouse memory. "
                               "Full-text search across all stored turns for the current user. "
                               "Use this to recall specific facts, decisions, or discussions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query — keywords or phrase to find",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 10)",
                            "default": 10,
                        },
                        "topic": {
                            "type": "string",
                            "description": "Filter by topic (optional)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_stats",
                "description": "Get memory statistics for the current user: "
                               "total turns, topics distribution, date range.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any],
                         **kwargs) -> str:
        """Handle memory_search and memory_stats tool calls."""
        if tool_name == "memory_search":
            return self._handle_memory_search(args)
        elif tool_name == "memory_stats":
            return self._handle_memory_stats()
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _handle_memory_search(self, args: dict) -> str:
        """Full-text search in ClickHouse memory."""
        if not self._client or not self._user_id:
            return json.dumps({"error": "ClickHouse not connected"})

        query = args.get("query", "")
        limit = min(int(args.get("limit", 10)), 50)
        topic = args.get("topic", "")

        cfg = self._config
        database = cfg.get("database", "hermes_memory")
        table = cfg.get("table", "events")

        params: dict = {"user_id": self._user_id, "limit": limit}
        where = "WHERE user_id = %(user_id)s"
        where += " AND content ILIKE %(search)s"
        params["search"] = f"%{query}%"

        if topic:
            where += " AND topic = %(topic)s"
            params["topic"] = topic

        try:
            rows = self._client.query_df(
                f"""
                SELECT ts, role, content, topic, importance, platform
                FROM {database}.{table}
                {where}
                ORDER BY importance DESC, ts DESC
                LIMIT %(limit)s
                """,
                parameters=params,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

        if rows.empty:
            return json.dumps({"results": [], "message": "No matches found"})

        results = []
        for _, row in rows.iterrows():
            results.append({
                "ts": str(row["ts"]),
                "role": row["role"],
                "content": str(row["content"])[:300],
                "topic": row.get("topic", ""),
                "importance": float(row.get("importance", 0)),
                "platform": row.get("platform", ""),
            })

        return json.dumps({"results": results, "count": len(results)})

    def _handle_memory_stats(self) -> str:
        """Return memory statistics for the current user."""
        if not self._client or not self._user_id:
            return json.dumps({"error": "ClickHouse not connected"})

        cfg = self._config
        database = cfg.get("database", "hermes_memory")
        table = cfg.get("table", "events")

        try:
            stats = self._client.query_df(
                f"""
                SELECT
                    count(*) as total_turns,
                    countDistinct(session_id) as sessions,
                    min(ts) as first_turn,
                    max(ts) as last_turn,
                    avg(importance) as avg_importance
                FROM {database}.{table}
                WHERE user_id = %(user_id)s
                """,
                parameters={"user_id": self._user_id},
            )

            topics = self._client.query_df(
                f"""
                SELECT topic, count(*) as cnt
                FROM {database}.{table}
                WHERE user_id = %(user_id)s AND topic != ''
                GROUP BY topic
                ORDER BY cnt DESC
                LIMIT 10
                """,
                parameters={"user_id": self._user_id},
            )

            result = {
                "total_turns": int(stats["total_turns"].iloc[0]),
                "sessions": int(stats["sessions"].iloc[0]),
                "first_turn": str(stats["first_turn"].iloc[0]) if stats["first_turn"].iloc[0] else "",
                "last_turn": str(stats["last_turn"].iloc[0]) if stats["last_turn"].iloc[0] else "",
                "avg_importance": round(float(stats["avg_importance"].iloc[0]), 2),
                "top_topics": [
                    {"topic": r["topic"], "count": int(r["cnt"])}
                    for _, r in topics.iterrows()
                ],
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def shutdown(self) -> None:
        """Close ClickHouse connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return config fields for ``hermes memory setup``."""
        return [
            {
                "key": "host",
                "description": "ClickHouse server hostname",
                "default": "localhost",
                "required": True,
            },
            {
                "key": "port",
                "description": "ClickHouse HTTP port",
                "default": 8123,
                "required": True,
            },
            {
                "key": "user",
                "description": "ClickHouse username",
                "default": "default",
                "required": True,
            },
            {
                "key": "password",
                "description": "ClickHouse password",
                "secret": True,
                "required": False,
                "env_var": "CLICKHOUSE_PASSWORD",
            },
            {
                "key": "database",
                "description": "ClickHouse database name",
                "default": "hermes_memory",
            },
            {
                "key": "ttl_days",
                "description": "Days to keep memory before auto-deletion",
                "default": 30,
            },
            {
                "key": "max_prefetch",
                "description": "Max records returned by prefetch",
                "default": 50,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write ClickHouse config to config.yaml."""
        try:
            config_path = Path(hermes_home) / "config.yaml"
            import yaml

            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}

            # Ensure plugins.clickhouse section exists
            plugins = config.setdefault("plugins", {})
            plugins["clickhouse"] = {k: v for k, v in values.items() if v is not None}

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            logger.info("ClickHouse config saved to %s", config_path)
        except Exception as e:
            logger.error("Failed to save ClickHouse config: %s", e)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Extract session summary and store as a high-importance memory entry.

        Called when the session ends. We extract key facts from the
        conversation and write them as a single 'session_summary' entry
        with high importance — ensures the session's contributions are
        recalled in future prefetches.
        """
        if not self._client or not self._user_id:
            return

        # Extract user messages for summary
        user_msgs = [
            m.get("content", "") for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
            and isinstance(m.get("content"), str) and len(m["content"]) > 20
        ]

        if not user_msgs:
            return

        # Build a compact summary
        summary = " | ".join(m[:100] for m in user_msgs[-5:])

        if not summary:
            return

        cfg = self._config
        database = cfg.get("database", "hermes_memory")
        table = cfg.get("table", "events")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        try:
            self._client.insert(
                database + "." + table,
                [[
                    self._user_id,
                    self._session_id,
                    now,
                    9999,  # high turn number = summary
                    "session_summary",
                    summary,
                    "session-summary",
                    0.85,  # high importance
                    1, 0,
                    [],
                    "",
                    self._platform, "", self._model, "",
                ]],
                column_names=[
                    "user_id", "session_id", "ts", "turn_number", "role",
                    "content", "topic", "importance", "frequency", "is_repeat",
                    "related_topics", "emotion",
                    "platform", "channel_id", "model", "parent_session_id",
                ],
            )
        except Exception as e:
            logger.debug("ClickHouse on_session_end failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: dict | None = None) -> None:
        """Mirror built-in memory() writes to ClickHouse.

        This ensures that explicit memory saves (via the memory tool)
        are also available in the ClickHouse-backed recall system.
        """
        if not self._client or not self._user_id:
            return

        if action not in ("add", "replace"):
            return

        cfg = self._config
        database = cfg.get("database", "hermes_memory")
        table = cfg.get("table", "events")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        topic = _extract_topic(content)
        importance = _estimate_importance(content)

        try:
            self._client.insert(
                database + "." + table,
                [[
                    self._user_id,
                    self._session_id or "",
                    now,
                    9998,  # explicit memory marker
                    f"memory_{target}",
                    f"[{action}] {content}",
                    topic,
                    importance,
                    1, 0,
                    [],
                    "",
                    self._platform, "", self._model, "",
                ]],
                column_names=[
                    "user_id", "session_id", "ts", "turn_number", "role",
                    "content", "topic", "importance", "frequency", "is_repeat",
                    "related_topics", "emotion",
                    "platform", "channel_id", "model", "parent_session_id",
                ],
            )
        except Exception as e:
            logger.debug("ClickHouse on_memory_write failed: %s", e)
