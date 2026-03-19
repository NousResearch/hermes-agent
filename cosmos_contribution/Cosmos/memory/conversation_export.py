"""
cosmos Conversation Export - Export memory and context as shareable formats

Features:
- Multiple export formats: JSON, Markdown, HTML
- Selective export (date range, tags, topics)
- Human-readable conversation transcripts
- Shareable HTML with styling
- Knowledge graph export
- Memory statistics summary
"""

import json
import base64
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Literal
from enum import Enum
import html

from loguru import logger


class ConversationExportFormat(Enum):
    """Export format options."""
    JSON = "json"           # Raw data format
    MARKDOWN = "markdown"   # Human-readable markdown
    HTML = "html"           # Shareable HTML page
    TEXT = "text"           # Plain text transcript


@dataclass
class ExportOptions:
    """Options for conversation export."""
    format: ConversationExportFormat = ConversationExportFormat.MARKDOWN
    include_memories: bool = True
    include_conversations: bool = True
    include_knowledge_graph: bool = True
    include_statistics: bool = True
    include_metadata: bool = True

    # Filters
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tags_filter: Optional[list[str]] = None
    topics_filter: Optional[list[str]] = None

    # Formatting
    max_memories: int = 100
    max_conversations: int = 500
    truncate_long_content: bool = True
    max_content_length: int = 2000


@dataclass
class ExportResult:
    """Result of an export operation."""
    success: bool
    format: ConversationExportFormat
    file_path: Optional[str] = None
    content: Optional[str] = None
    size_bytes: int = 0

    # Statistics
    memories_exported: int = 0
    conversations_exported: int = 0
    entities_exported: int = 0

    # Metadata
    export_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    checksum: str = ""

    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "format": self.format.value,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "memories_exported": self.memories_exported,
            "conversations_exported": self.conversations_exported,
            "entities_exported": self.entities_exported,
            "export_id": self.export_id,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
            "error": self.error,
        }


class ConversationExporter:
    """
    Export conversations, memories, and context in multiple formats.

    Supports:
    - JSON: Full data export for backup/import
    - Markdown: Human-readable transcript
    - HTML: Shareable styled page
    - Text: Plain text transcript
    """

    def __init__(
        self,
        output_dir: str = "./exports",
        instance_id: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.instance_id = instance_id or "cosmos"

        # Callbacks for data access (set by MemorySystem)
        self.get_memories_fn = None
        self.get_conversations_fn = None
        self.get_entities_fn = None
        self.get_relationships_fn = None
        self.get_statistics_fn = None

    async def export(
        self,
        options: Optional[ExportOptions] = None,
        output_path: Optional[str] = None,
    ) -> ExportResult:
        """
        Export conversation and memory data.

        Args:
            options: Export configuration
            output_path: Optional custom output path

        Returns:
            ExportResult with details
        """
        options = options or ExportOptions()

        try:
            # Gather data
            data = await self._gather_data(options)

            # Generate export based on format
            if options.format == ConversationExportFormat.JSON:
                content = self._export_json(data, options)
                ext = "json"
            elif options.format == ConversationExportFormat.MARKDOWN:
                content = self._export_markdown(data, options)
                ext = "md"
            elif options.format == ConversationExportFormat.HTML:
                content = self._export_html(data, options)
                ext = "html"
            elif options.format == ConversationExportFormat.TEXT:
                content = self._export_text(data, options)
                ext = "txt"
            else:
                raise ValueError(f"Unknown format: {options.format}")

            # Generate export ID and checksum
            export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checksum = hashlib.sha256(content.encode()).hexdigest()[:16]

            # Write to file
            if output_path:
                file_path = Path(output_path)
            else:
                file_path = self.output_dir / f"{export_id}.{ext}"

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')

            result = ExportResult(
                success=True,
                format=options.format,
                file_path=str(file_path),
                content=content if len(content) < 10000 else None,  # Only include small exports
                size_bytes=len(content.encode()),
                memories_exported=len(data.get("memories", [])),
                conversations_exported=len(data.get("conversations", [])),
                entities_exported=len(data.get("entities", [])),
                export_id=export_id,
                checksum=checksum,
            )

            logger.info(f"Exported {result.memories_exported} memories, {result.conversations_exported} conversations to {file_path}")
            return result

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                format=options.format,
                error=str(e),
            )

    async def export_to_string(
        self,
        options: Optional[ExportOptions] = None,
    ) -> ExportResult:
        """
        Export to string without writing to file.

        Returns:
            ExportResult with content in the content field
        """
        options = options or ExportOptions()

        try:
            data = await self._gather_data(options)

            if options.format == ConversationExportFormat.JSON:
                content = self._export_json(data, options)
            elif options.format == ConversationExportFormat.MARKDOWN:
                content = self._export_markdown(data, options)
            elif options.format == ConversationExportFormat.HTML:
                content = self._export_html(data, options)
            elif options.format == ConversationExportFormat.TEXT:
                content = self._export_text(data, options)
            else:
                raise ValueError(f"Unknown format: {options.format}")

            return ExportResult(
                success=True,
                format=options.format,
                content=content,
                size_bytes=len(content.encode()),
                memories_exported=len(data.get("memories", [])),
                conversations_exported=len(data.get("conversations", [])),
                entities_exported=len(data.get("entities", [])),
                export_id=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                checksum=hashlib.sha256(content.encode()).hexdigest()[:16],
            )

        except Exception as e:
            logger.error(f"Export to string failed: {e}")
            return ExportResult(
                success=False,
                format=options.format,
                error=str(e),
            )

    async def _gather_data(self, options: ExportOptions) -> dict:
        """Gather all data for export."""
        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "instance_id": self.instance_id,
                "version": "1.0.0",
                "filters": {
                    "start_date": options.start_date.isoformat() if options.start_date else None,
                    "end_date": options.end_date.isoformat() if options.end_date else None,
                    "tags": options.tags_filter,
                    "topics": options.topics_filter,
                },
            },
            "memories": [],
            "conversations": [],
            "entities": [],
            "relationships": [],
            "statistics": {},
        }

        # Gather memories
        if options.include_memories and self.get_memories_fn:
            memories = await self.get_memories_fn()
            data["memories"] = self._filter_and_limit(
                memories,
                options,
                options.max_memories,
                date_field="created_at"
            )

        # Gather conversations
        if options.include_conversations and self.get_conversations_fn:
            conversations = await self.get_conversations_fn()
            data["conversations"] = self._filter_and_limit(
                conversations,
                options,
                options.max_conversations,
                date_field="timestamp"
            )

        # Gather knowledge graph
        if options.include_knowledge_graph:
            if self.get_entities_fn:
                data["entities"] = await self.get_entities_fn()
            if self.get_relationships_fn:
                data["relationships"] = await self.get_relationships_fn()

        # Gather statistics
        if options.include_statistics and self.get_statistics_fn:
            data["statistics"] = await self.get_statistics_fn()

        return data

    def _filter_and_limit(
        self,
        items: list,
        options: ExportOptions,
        limit: int,
        date_field: str = "created_at",
    ) -> list:
        """Filter and limit items based on options."""
        filtered = []

        for item in items:
            # Date filter
            if options.start_date or options.end_date:
                item_date = item.get(date_field)
                if item_date:
                    if isinstance(item_date, str):
                        item_date = datetime.fromisoformat(item_date)
                    if options.start_date and item_date < options.start_date:
                        continue
                    if options.end_date and item_date > options.end_date:
                        continue

            # Tags filter
            if options.tags_filter:
                item_tags = item.get("tags", [])
                if not any(t in item_tags for t in options.tags_filter):
                    continue

            # Topics filter
            if options.topics_filter:
                item_topics = item.get("topics", [])
                if not any(t in item_topics for t in options.topics_filter):
                    continue

            filtered.append(item)

        # Sort by date (newest first) and limit
        filtered.sort(
            key=lambda x: x.get(date_field, ""),
            reverse=True
        )

        return filtered[:limit]

    def _export_json(self, data: dict, options: ExportOptions) -> str:
        """Export to JSON format."""
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    def _export_markdown(self, data: dict, options: ExportOptions) -> str:
        """Export to Markdown format."""
        lines = []

        # Header
        lines.append("# cosmos Memory Export")
        lines.append("")
        lines.append(f"**Exported:** {data['metadata']['exported_at']}")
        lines.append(f"**Instance:** {data['metadata']['instance_id']}")
        lines.append("")

        # Statistics
        if data.get("statistics") and options.include_statistics:
            lines.append("## Statistics")
            lines.append("")
            stats = data["statistics"]
            if isinstance(stats, dict):
                for key, value in stats.items():
                    lines.append(f"- **{key}:** {value}")
            lines.append("")

        # Memories
        if data.get("memories"):
            lines.append("## Memories")
            lines.append("")
            for i, mem in enumerate(data["memories"], 1):
                content = mem.get("content", "")
                if options.truncate_long_content and len(content) > options.max_content_length:
                    content = content[:options.max_content_length] + "..."

                tags = mem.get("tags", [])
                tags_str = " ".join(f"`{t}`" for t in tags) if tags else ""
                created = mem.get("created_at", "Unknown")

                lines.append(f"### Memory {i}")
                lines.append(f"*{created}* {tags_str}")
                lines.append("")
                lines.append(content)
                lines.append("")
                lines.append("---")
                lines.append("")

        # Conversations
        if data.get("conversations"):
            lines.append("## Conversation History")
            lines.append("")
            for conv in data["conversations"]:
                role = conv.get("role", "unknown")
                content = conv.get("content", "")
                if options.truncate_long_content and len(content) > options.max_content_length:
                    content = content[:options.max_content_length] + "..."

                timestamp = conv.get("timestamp", "")

                role_emoji = "👤" if role == "user" else "🤖" if role == "assistant" else "⚙️"
                lines.append(f"**{role_emoji} {role.capitalize()}** _{timestamp}_")
                lines.append("")
                lines.append(content)
                lines.append("")

        # Knowledge Graph
        if data.get("entities") and options.include_knowledge_graph:
            lines.append("## Knowledge Graph")
            lines.append("")
            lines.append("### Entities")
            lines.append("")
            for entity in data["entities"][:50]:  # Limit to 50 entities
                name = entity.get("name", "Unknown")
                entity_type = entity.get("type", "unknown")
                lines.append(f"- **{name}** ({entity_type})")
            lines.append("")

            if data.get("relationships"):
                lines.append("### Relationships")
                lines.append("")
                for rel in data["relationships"][:50]:
                    source = rel.get("source", "?")
                    target = rel.get("target", "?")
                    rel_type = rel.get("type", "related_to")
                    lines.append(f"- {source} → *{rel_type}* → {target}")
                lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Exported with cosmos - Self-Evolving AI Companion*")

        return "\n".join(lines)

    def _export_html(self, data: dict, options: ExportOptions) -> str:
        """Export to styled HTML format."""

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='UTF-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "  <title>cosmos Memory Export</title>",
            "  <style>",
            self._get_html_styles(),
            "  </style>",
            "</head>",
            "<body>",
            "  <div class='container'>",
        ]

        # Header
        html_parts.extend([
            "    <header>",
            "      <h1>🧠 cosmos Memory Export</h1>",
            f"      <p class='meta'>Exported: {data['metadata']['exported_at']}</p>",
            f"      <p class='meta'>Instance: {data['metadata']['instance_id']}</p>",
            "    </header>",
        ])

        # Statistics
        if data.get("statistics") and options.include_statistics:
            html_parts.append("    <section class='stats'>")
            html_parts.append("      <h2>📊 Statistics</h2>")
            html_parts.append("      <div class='stats-grid'>")
            stats = data["statistics"]
            if isinstance(stats, dict):
                for key, value in stats.items():
                    html_parts.append(f"        <div class='stat-item'><span class='stat-label'>{html.escape(str(key))}</span><span class='stat-value'>{html.escape(str(value))}</span></div>")
            html_parts.append("      </div>")
            html_parts.append("    </section>")

        # Memories
        if data.get("memories"):
            html_parts.append("    <section class='memories'>")
            html_parts.append("      <h2>💾 Memories</h2>")
            for i, mem in enumerate(data["memories"], 1):
                content = html.escape(mem.get("content", ""))
                if options.truncate_long_content and len(content) > options.max_content_length:
                    content = content[:options.max_content_length] + "..."

                tags = mem.get("tags", [])
                tags_html = " ".join(f"<span class='tag'>{html.escape(t)}</span>" for t in tags)
                created = mem.get("created_at", "Unknown")
                importance = mem.get("importance", 0.5)

                html_parts.append("      <div class='memory-card'>")
                html_parts.append(f"        <div class='memory-header'><span class='memory-num'>#{i}</span> <span class='memory-date'>{html.escape(str(created))}</span></div>")
                html_parts.append(f"        <div class='memory-content'>{content}</div>")
                html_parts.append(f"        <div class='memory-footer'>{tags_html} <span class='importance'>Importance: {importance:.1%}</span></div>")
                html_parts.append("      </div>")
            html_parts.append("    </section>")

        # Conversations
        if data.get("conversations"):
            html_parts.append("    <section class='conversations'>")
            html_parts.append("      <h2>💬 Conversation History</h2>")
            html_parts.append("      <div class='chat-container'>")
            for conv in data["conversations"]:
                role = conv.get("role", "unknown")
                content = html.escape(conv.get("content", ""))
                if options.truncate_long_content and len(content) > options.max_content_length:
                    content = content[:options.max_content_length] + "..."
                # Convert newlines to <br> for HTML display
                content = content.replace("\n", "<br>")

                timestamp = conv.get("timestamp", "")
                role_class = "user" if role == "user" else "assistant" if role == "assistant" else "system"

                html_parts.append(f"        <div class='message {role_class}'>")
                html_parts.append(f"          <div class='message-role'>{role.capitalize()}</div>")
                html_parts.append(f"          <div class='message-content'>{content}</div>")
                html_parts.append(f"          <div class='message-time'>{html.escape(str(timestamp))}</div>")
                html_parts.append("        </div>")
            html_parts.append("      </div>")
            html_parts.append("    </section>")

        # Knowledge Graph
        if data.get("entities") and options.include_knowledge_graph:
            html_parts.append("    <section class='knowledge-graph'>")
            html_parts.append("      <h2>🔗 Knowledge Graph</h2>")
            html_parts.append("      <h3>Entities</h3>")
            html_parts.append("      <div class='entities-grid'>")
            for entity in data["entities"][:50]:
                name = html.escape(entity.get("name", "Unknown"))
                entity_type = html.escape(entity.get("type", "unknown"))
                html_parts.append(f"        <div class='entity'><strong>{name}</strong><span class='entity-type'>{entity_type}</span></div>")
            html_parts.append("      </div>")

            if data.get("relationships"):
                html_parts.append("      <h3>Relationships</h3>")
                html_parts.append("      <ul class='relationships'>")
                for rel in data["relationships"][:50]:
                    source = html.escape(rel.get("source", "?"))
                    target = html.escape(rel.get("target", "?"))
                    rel_type = html.escape(rel.get("type", "related_to"))
                    html_parts.append(f"        <li>{source} → <em>{rel_type}</em> → {target}</li>")
                html_parts.append("      </ul>")
            html_parts.append("    </section>")

        # Footer
        html_parts.extend([
            "    <footer>",
            "      <p>Exported with <strong>cosmos</strong> - Self-Evolving AI Companion</p>",
            "      <p><a href='https://github.com/anthropics/cosmos'>GitHub</a></p>",
            "    </footer>",
            "  </div>",
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)

    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML export."""
        return """
    :root {
      --bg: #0d1117;
      --surface: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --user-bg: #1f6feb;
      --assistant-bg: #238636;
      --system-bg: #6e7681;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }

    .container {
      max-width: 900px;
      margin: 0 auto;
      padding: 2rem;
    }

    header {
      text-align: center;
      margin-bottom: 3rem;
      padding-bottom: 2rem;
      border-bottom: 1px solid var(--border);
    }

    h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
    h2 { font-size: 1.5rem; margin: 2rem 0 1rem; color: var(--accent); }
    h3 { font-size: 1.2rem; margin: 1.5rem 0 0.75rem; }

    .meta { color: var(--text-muted); font-size: 0.9rem; }

    /* Stats */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1rem;
    }

    .stat-item {
      background: var(--surface);
      padding: 1rem;
      border-radius: 8px;
      border: 1px solid var(--border);
    }

    .stat-label { display: block; font-size: 0.8rem; color: var(--text-muted); }
    .stat-value { display: block; font-size: 1.5rem; font-weight: bold; color: var(--accent); }

    /* Memories */
    .memory-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1rem;
    }

    .memory-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.5rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    .memory-num { font-weight: bold; color: var(--accent); }

    .memory-content {
      white-space: pre-wrap;
      word-break: break-word;
    }

    .memory-footer {
      margin-top: 0.75rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      align-items: center;
    }

    .tag {
      background: var(--accent);
      color: var(--bg);
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 500;
    }

    .importance {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-left: auto;
    }

    /* Conversations */
    .chat-container {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .message {
      max-width: 80%;
      padding: 0.75rem 1rem;
      border-radius: 12px;
    }

    .message.user {
      background: var(--user-bg);
      align-self: flex-end;
      border-bottom-right-radius: 4px;
    }

    .message.assistant {
      background: var(--assistant-bg);
      align-self: flex-start;
      border-bottom-left-radius: 4px;
    }

    .message.system {
      background: var(--system-bg);
      align-self: center;
      font-style: italic;
      font-size: 0.9rem;
    }

    .message-role {
      font-size: 0.75rem;
      font-weight: bold;
      margin-bottom: 0.25rem;
      opacity: 0.8;
    }

    .message-content {
      word-break: break-word;
    }

    .message-time {
      font-size: 0.7rem;
      opacity: 0.6;
      margin-top: 0.25rem;
      text-align: right;
    }

    /* Knowledge Graph */
    .entities-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.75rem;
    }

    .entity {
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 0.5rem 0.75rem;
      border-radius: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .entity-type {
      font-size: 0.75rem;
      color: var(--text-muted);
      background: var(--border);
      padding: 0.1rem 0.4rem;
      border-radius: 4px;
    }

    .relationships {
      list-style: none;
      columns: 2;
      column-gap: 2rem;
    }

    .relationships li {
      padding: 0.25rem 0;
      border-bottom: 1px solid var(--border);
    }

    /* Footer */
    footer {
      margin-top: 3rem;
      padding-top: 2rem;
      border-top: 1px solid var(--border);
      text-align: center;
      color: var(--text-muted);
      font-size: 0.9rem;
    }

    footer a {
      color: var(--accent);
      text-decoration: none;
    }

    footer a:hover {
      text-decoration: underline;
    }

    @media (max-width: 600px) {
      .container { padding: 1rem; }
      h1 { font-size: 1.75rem; }
      .message { max-width: 90%; }
      .relationships { columns: 1; }
    }
"""

    def _export_text(self, data: dict, options: ExportOptions) -> str:
        """Export to plain text format."""
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("cosmos MEMORY EXPORT")
        lines.append("=" * 60)
        lines.append(f"Exported: {data['metadata']['exported_at']}")
        lines.append(f"Instance: {data['metadata']['instance_id']}")
        lines.append("")

        # Statistics
        if data.get("statistics") and options.include_statistics:
            lines.append("-" * 40)
            lines.append("STATISTICS")
            lines.append("-" * 40)
            stats = data["statistics"]
            if isinstance(stats, dict):
                for key, value in stats.items():
                    lines.append(f"  {key}: {value}")
            lines.append("")

        # Memories
        if data.get("memories"):
            lines.append("-" * 40)
            lines.append("MEMORIES")
            lines.append("-" * 40)
            for i, mem in enumerate(data["memories"], 1):
                content = mem.get("content", "")
                if options.truncate_long_content and len(content) > options.max_content_length:
                    content = content[:options.max_content_length] + "..."

                tags = mem.get("tags", [])
                created = mem.get("created_at", "Unknown")

                lines.append(f"\n[{i}] {created}")
                if tags:
                    lines.append(f"    Tags: {', '.join(tags)}")
                lines.append(f"    {content}")
            lines.append("")

        # Conversations
        if data.get("conversations"):
            lines.append("-" * 40)
            lines.append("CONVERSATION HISTORY")
            lines.append("-" * 40)
            for conv in data["conversations"]:
                role = conv.get("role", "unknown").upper()
                content = conv.get("content", "")
                if options.truncate_long_content and len(content) > options.max_content_length:
                    content = content[:options.max_content_length] + "..."

                timestamp = conv.get("timestamp", "")

                lines.append(f"\n[{role}] {timestamp}")
                lines.append(content)
            lines.append("")

        # Footer
        lines.append("=" * 60)
        lines.append("Exported with cosmos - Self-Evolving AI Companion")
        lines.append("=" * 60)

        return "\n".join(lines)

    def list_exports(self) -> list[dict]:
        """List all exports in the output directory."""
        exports = []

        for ext in ["json", "md", "html", "txt"]:
            for file_path in self.output_dir.glob(f"*.{ext}"):
                stat = file_path.stat()
                exports.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "format": ext,
                })

        exports.sort(key=lambda x: x["created_at"], reverse=True)
        return exports

    def delete_export(self, file_path: str) -> bool:
        """Delete an export file."""
        path = Path(file_path)
        if path.exists() and path.parent == self.output_dir:
            path.unlink()
            return True
        return False
