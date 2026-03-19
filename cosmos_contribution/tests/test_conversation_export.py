"""
Tests for Conversation Export feature.

Tests the ability to export memories, conversations, and knowledge graphs
in multiple formats (JSON, Markdown, HTML, Text).
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from cosmos.memory.conversation_export import (
    ConversationExporter,
    ConversationExportFormat,
    ExportOptions,
    ExportResult,
)


class TestConversationExportFormat:
    """Test export format enum."""

    def test_format_values(self):
        assert ConversationExportFormat.JSON.value == "json"
        assert ConversationExportFormat.MARKDOWN.value == "markdown"
        assert ConversationExportFormat.HTML.value == "html"
        assert ConversationExportFormat.TEXT.value == "text"


class TestExportOptions:
    """Test export options configuration."""

    def test_default_options(self):
        options = ExportOptions()
        assert options.format == ConversationExportFormat.MARKDOWN
        assert options.include_memories is True
        assert options.include_conversations is True
        assert options.include_knowledge_graph is True
        assert options.include_statistics is True
        assert options.max_memories == 100
        assert options.max_conversations == 500

    def test_custom_options(self):
        options = ExportOptions(
            format=ConversationExportFormat.HTML,
            include_memories=False,
            tags_filter=["important", "project"],
            max_memories=50,
        )
        assert options.format == ConversationExportFormat.HTML
        assert options.include_memories is False
        assert options.tags_filter == ["important", "project"]
        assert options.max_memories == 50


class TestExportResult:
    """Test export result dataclass."""

    def test_result_to_dict(self):
        result = ExportResult(
            success=True,
            format=ConversationExportFormat.JSON,
            file_path="/path/to/export.json",
            size_bytes=1024,
            memories_exported=10,
            conversations_exported=25,
            entities_exported=5,
            export_id="export_123",
            checksum="abc123",
        )

        data = result.to_dict()
        assert data["success"] is True
        assert data["format"] == "json"
        assert data["file_path"] == "/path/to/export.json"
        assert data["memories_exported"] == 10
        assert data["checksum"] == "abc123"


class TestConversationExporter:
    """Test the ConversationExporter class."""

    @pytest.fixture
    def temp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def exporter(self, temp_output_dir):
        return ConversationExporter(
            output_dir=temp_output_dir,
            instance_id="test_instance",
        )

    @pytest.fixture
    def sample_data(self):
        """Sample data for export tests."""
        return {
            "memories": [
                {
                    "content": "User prefers Python over JavaScript",
                    "tags": ["preference", "programming"],
                    "created_at": datetime.now().isoformat(),
                    "importance": 0.8,
                },
                {
                    "content": "Project deadline is next Friday",
                    "tags": ["project", "deadline"],
                    "created_at": datetime.now().isoformat(),
                    "importance": 0.9,
                },
            ],
            "conversations": [
                {
                    "role": "user",
                    "content": "Hello, can you help me with Python?",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "role": "assistant",
                    "content": "Of course! I'd be happy to help you with Python. What would you like to know?",
                    "timestamp": datetime.now().isoformat(),
                },
            ],
            "entities": [
                {"name": "Python", "type": "programming_language", "mentions": 5},
                {"name": "User", "type": "person", "mentions": 10},
            ],
            "relationships": [
                {"source": "User", "target": "Python", "type": "prefers"},
            ],
            "statistics": {
                "total_memories": 2,
                "total_conversations": 2,
                "total_entities": 2,
            },
        }

    @pytest.fixture
    def mock_data_callbacks(self, exporter, sample_data):
        """Set up mock callbacks for data access."""
        exporter.get_memories_fn = AsyncMock(return_value=sample_data["memories"])
        exporter.get_conversations_fn = AsyncMock(return_value=sample_data["conversations"])
        exporter.get_entities_fn = AsyncMock(return_value=sample_data["entities"])
        exporter.get_relationships_fn = AsyncMock(return_value=sample_data["relationships"])
        exporter.get_statistics_fn = AsyncMock(return_value=sample_data["statistics"])
        return exporter

    @pytest.mark.asyncio
    async def test_export_json(self, mock_data_callbacks, temp_output_dir):
        """Test JSON export format."""
        options = ExportOptions(format=ConversationExportFormat.JSON)
        result = await mock_data_callbacks.export(options)

        assert result.success is True
        assert result.format == ConversationExportFormat.JSON
        assert result.file_path is not None
        assert result.file_path.endswith(".json")
        assert result.memories_exported == 2
        assert result.conversations_exported == 2

        # Verify file contents
        with open(result.file_path, "r") as f:
            data = json.load(f)
        assert "memories" in data
        assert "conversations" in data
        assert len(data["memories"]) == 2

    @pytest.mark.asyncio
    async def test_export_markdown(self, mock_data_callbacks, temp_output_dir):
        """Test Markdown export format."""
        options = ExportOptions(format=ConversationExportFormat.MARKDOWN)
        result = await mock_data_callbacks.export(options)

        assert result.success is True
        assert result.format == ConversationExportFormat.MARKDOWN
        assert result.file_path.endswith(".md")

        # Verify file contents
        content = Path(result.file_path).read_text()
        assert "# cosmos Memory Export" in content
        assert "## Memories" in content
        assert "## Conversation History" in content
        assert "Python" in content

    @pytest.mark.asyncio
    async def test_export_html(self, mock_data_callbacks, temp_output_dir):
        """Test HTML export format."""
        options = ExportOptions(format=ConversationExportFormat.HTML)
        result = await mock_data_callbacks.export(options)

        assert result.success is True
        assert result.format == ConversationExportFormat.HTML
        assert result.file_path.endswith(".html")

        # Verify file contents
        content = Path(result.file_path).read_text()
        assert "<!DOCTYPE html>" in content
        assert "<title>cosmos Memory Export</title>" in content
        assert "message user" in content or "message assistant" in content
        assert "<style>" in content  # Has CSS styling

    @pytest.mark.asyncio
    async def test_export_text(self, mock_data_callbacks, temp_output_dir):
        """Test plain text export format."""
        options = ExportOptions(format=ConversationExportFormat.TEXT)
        result = await mock_data_callbacks.export(options)

        assert result.success is True
        assert result.format == ConversationExportFormat.TEXT
        assert result.file_path.endswith(".txt")

        # Verify file contents
        content = Path(result.file_path).read_text()
        assert "cosmos MEMORY EXPORT" in content
        assert "MEMORIES" in content
        assert "CONVERSATION HISTORY" in content

    @pytest.mark.asyncio
    async def test_export_to_string(self, mock_data_callbacks):
        """Test export to string without file."""
        options = ExportOptions(format=ConversationExportFormat.MARKDOWN)
        result = await mock_data_callbacks.export_to_string(options)

        assert result.success is True
        assert result.content is not None
        assert result.file_path is None
        assert "# cosmos Memory Export" in result.content

    @pytest.mark.asyncio
    async def test_export_with_date_filter(self, mock_data_callbacks):
        """Test export with date range filter."""
        # Set up memories with different dates
        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        new_date = datetime.now().isoformat()

        mock_data_callbacks.get_memories_fn = AsyncMock(return_value=[
            {"content": "Old memory", "created_at": old_date, "tags": []},
            {"content": "New memory", "created_at": new_date, "tags": []},
        ])

        options = ExportOptions(
            format=ConversationExportFormat.JSON,
            start_date=datetime.now() - timedelta(days=7),
        )
        result = await mock_data_callbacks.export(options)

        assert result.success is True
        # Only new memory should be included
        assert result.memories_exported == 1

    @pytest.mark.asyncio
    async def test_export_with_tags_filter(self, mock_data_callbacks):
        """Test export with tags filter."""
        mock_data_callbacks.get_memories_fn = AsyncMock(return_value=[
            {"content": "Memory 1", "tags": ["important"], "created_at": datetime.now().isoformat()},
            {"content": "Memory 2", "tags": ["misc"], "created_at": datetime.now().isoformat()},
            {"content": "Memory 3", "tags": ["important", "project"], "created_at": datetime.now().isoformat()},
        ])

        options = ExportOptions(
            format=ConversationExportFormat.JSON,
            tags_filter=["important"],
        )
        result = await mock_data_callbacks.export(options)

        assert result.success is True
        assert result.memories_exported == 2  # Only memories with "important" tag

    @pytest.mark.asyncio
    async def test_export_selective_components(self, mock_data_callbacks):
        """Test export with selective component inclusion."""
        options = ExportOptions(
            format=ConversationExportFormat.JSON,
            include_memories=True,
            include_conversations=False,
            include_knowledge_graph=False,
            include_statistics=False,
        )
        result = await mock_data_callbacks.export(options)

        assert result.success is True
        assert result.memories_exported == 2
        assert result.conversations_exported == 0
        assert result.entities_exported == 0

    @pytest.mark.asyncio
    async def test_export_custom_output_path(self, mock_data_callbacks, temp_output_dir):
        """Test export to custom output path."""
        custom_path = Path(temp_output_dir) / "custom_export.json"
        options = ExportOptions(format=ConversationExportFormat.JSON)
        result = await mock_data_callbacks.export(options, output_path=str(custom_path))

        assert result.success is True
        assert result.file_path == str(custom_path)
        assert custom_path.exists()

    @pytest.mark.asyncio
    async def test_export_creates_checksum(self, mock_data_callbacks):
        """Test that export creates content checksum."""
        options = ExportOptions(format=ConversationExportFormat.JSON)
        result = await mock_data_callbacks.export(options)

        assert result.success is True
        assert result.checksum is not None
        assert len(result.checksum) == 16  # SHA256 truncated to 16 chars

    def test_list_exports(self, exporter, temp_output_dir):
        """Test listing exports."""
        # Create some test export files
        (Path(temp_output_dir) / "export_1.json").write_text("{}")
        (Path(temp_output_dir) / "export_2.md").write_text("# Test")
        (Path(temp_output_dir) / "export_3.html").write_text("<html></html>")

        exports = exporter.list_exports()

        assert len(exports) == 3
        assert any(e["format"] == "json" for e in exports)
        assert any(e["format"] == "md" for e in exports)
        assert any(e["format"] == "html" for e in exports)

    def test_delete_export(self, exporter, temp_output_dir):
        """Test deleting an export."""
        export_file = Path(temp_output_dir) / "test_export.json"
        export_file.write_text("{}")

        assert export_file.exists()

        result = exporter.delete_export(str(export_file))

        assert result is True
        assert not export_file.exists()

    def test_delete_nonexistent_export(self, exporter):
        """Test deleting a non-existent export."""
        result = exporter.delete_export("/nonexistent/path.json")
        assert result is False

    @pytest.mark.asyncio
    async def test_export_handles_missing_callbacks(self, exporter):
        """Test export gracefully handles missing data callbacks."""
        # No callbacks set
        options = ExportOptions(format=ConversationExportFormat.JSON)
        result = await exporter.export(options)

        assert result.success is True
        assert result.memories_exported == 0
        assert result.conversations_exported == 0

    @pytest.mark.asyncio
    async def test_export_truncates_long_content(self, mock_data_callbacks):
        """Test that long content is truncated when option is set."""
        long_content = "A" * 5000
        mock_data_callbacks.get_memories_fn = AsyncMock(return_value=[
            {"content": long_content, "tags": [], "created_at": datetime.now().isoformat()},
        ])

        options = ExportOptions(
            format=ConversationExportFormat.MARKDOWN,
            truncate_long_content=True,
            max_content_length=100,
        )
        result = await mock_data_callbacks.export_to_string(options)

        assert result.success is True
        assert "..." in result.content
        assert "A" * 100 in result.content
        assert "A" * 5000 not in result.content


class TestHTMLExportStyling:
    """Test HTML export styling and structure."""

    @pytest.fixture
    def exporter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            return ConversationExporter(output_dir=tmpdir)

    @pytest.mark.asyncio
    async def test_html_has_responsive_design(self, exporter):
        """Test HTML export has responsive CSS."""
        exporter.get_memories_fn = AsyncMock(return_value=[])
        exporter.get_conversations_fn = AsyncMock(return_value=[])
        exporter.get_entities_fn = AsyncMock(return_value=[])
        exporter.get_relationships_fn = AsyncMock(return_value=[])
        exporter.get_statistics_fn = AsyncMock(return_value={})

        options = ExportOptions(format=ConversationExportFormat.HTML)
        result = await exporter.export_to_string(options)

        assert "@media" in result.content  # Has media queries
        assert "max-width" in result.content

    @pytest.mark.asyncio
    async def test_html_escapes_content(self, exporter):
        """Test HTML export properly escapes user content."""
        exporter.get_memories_fn = AsyncMock(return_value=[
            {
                "content": "<script>alert('xss')</script>",
                "tags": [],
                "created_at": datetime.now().isoformat(),
            }
        ])
        exporter.get_conversations_fn = AsyncMock(return_value=[])
        exporter.get_entities_fn = AsyncMock(return_value=[])
        exporter.get_relationships_fn = AsyncMock(return_value=[])
        exporter.get_statistics_fn = AsyncMock(return_value={})

        options = ExportOptions(format=ConversationExportFormat.HTML)
        result = await exporter.export_to_string(options)

        # Script tag should be escaped
        assert "<script>" not in result.content
        assert "&lt;script&gt;" in result.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
