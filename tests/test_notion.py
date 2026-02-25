"""
Tests for tools/notion.py

Run with:
    pytest tests/test_notion.py -v

These tests mock the Notion API — no real API key needed.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_PAGE = {
    "object": "page",
    "id": "page-abc-123",
    "url": "https://notion.so/page-abc-123",
    "properties": {
        "title": {
            "type": "title",
            "title": [{"plain_text": "Test Page"}],
        }
    },
}

MOCK_DATABASE = {
    "object": "database",
    "id": "db-xyz-456",
    "url": "https://notion.so/db-xyz-456",
    "title": [{"plain_text": "My Tasks"}],
}

MOCK_BLOCKS = {
    "results": [
        {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"plain_text": "Hello world"}]
            },
        },
        {
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"plain_text": "Section Title"}]
            },
        },
        {
            "type": "to_do",
            "to_do": {
                "rich_text": [{"plain_text": "A task"}],
                "checked": False,
            },
        },
    ],
    "has_more": False,
    "next_cursor": None,
}


def _mock_req(responses: dict):
    """Return a side_effect function that maps (method, path) to a response."""
    def side_effect(method, path, **kwargs):
        key = (method.upper(), path)
        for k, v in responses.items():
            if path.endswith(k[1]) and method.upper() == k[0]:
                return v
        raise KeyError(f"Unexpected API call: {method} {path}")
    return side_effect


# ---------------------------------------------------------------------------
# notion_search
# ---------------------------------------------------------------------------

class TestNotionSearch:
    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_search_returns_pages_and_databases(self, mock_req):
        mock_req.return_value = {
            "results": [MOCK_PAGE, MOCK_DATABASE],
            "has_more": False,
        }
        from tools.notion import notion_search
        result = notion_search("test")
        assert "Test Page" in result
        assert "My Tasks" in result
        assert "page-abc-123" in result
        assert "db-xyz-456" in result

    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_search_no_results(self, mock_req):
        mock_req.return_value = {"results": []}
        from tools.notion import notion_search
        result = notion_search("nonexistent")
        assert "No results" in result

    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_search_database_filter(self, mock_req):
        mock_req.return_value = {"results": [MOCK_DATABASE]}
        from tools.notion import notion_search
        result = notion_search("", filter_type="database")
        assert "My Tasks" in result
        # Verify filter was passed
        call_kwargs = mock_req.call_args[1]
        assert call_kwargs["json"]["filter"]["value"] == "database"

    def test_search_no_api_key(self):
        import os
        os.environ.pop("NOTION_API_KEY", None)
        from tools.notion import notion_search
        with pytest.raises(ValueError, match="NOTION_API_KEY"):
            notion_search("test")


# ---------------------------------------------------------------------------
# notion_get_page
# ---------------------------------------------------------------------------

class TestNotionGetPage:
    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_get_page_content(self, mock_req):
        mock_req.side_effect = [MOCK_PAGE, MOCK_BLOCKS]
        from tools.notion import notion_get_page
        result = notion_get_page("page-abc-123")
        assert "Test Page" in result
        assert "Hello world" in result
        assert "## Section Title" in result
        assert "[ ] A task" in result

    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_get_page_empty(self, mock_req):
        mock_req.side_effect = [MOCK_PAGE, {"results": [], "has_more": False}]
        from tools.notion import notion_get_page
        result = notion_get_page("page-abc-123")
        assert "No content" in result


# ---------------------------------------------------------------------------
# notion_create_page
# ---------------------------------------------------------------------------

class TestNotionCreatePage:
    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_create_page_in_database(self, mock_req):
        mock_req.return_value = {
            "id": "new-page-id",
            "url": "https://notion.so/new-page-id",
        }
        from tools.notion import notion_create_page
        result = notion_create_page("db-xyz-456", "New Task", "Some content")
        assert "✅" in result
        assert "New Task" in result
        assert "new-page-id" in result

        # Check payload
        payload = mock_req.call_args[1]["json"]
        assert payload["parent"] == {"database_id": "db-xyz-456"}
        assert "New Task" in str(payload["properties"])
        assert len(payload["children"]) == 1  # content block added

    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_create_page_in_page(self, mock_req):
        mock_req.return_value = {"id": "child-page", "url": "https://notion.so/child"}
        from tools.notion import notion_create_page
        result = notion_create_page("page-abc-123", "Sub-page", parent_type="page")
        assert "✅" in result
        payload = mock_req.call_args[1]["json"]
        assert payload["parent"] == {"page_id": "page-abc-123"}
        assert "children" not in payload  # no content = no children block


# ---------------------------------------------------------------------------
# notion_append_blocks
# ---------------------------------------------------------------------------

class TestNotionAppendBlocks:
    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_append_paragraph(self, mock_req):
        mock_req.return_value = {}
        from tools.notion import notion_append_blocks
        result = notion_append_blocks("page-abc-123", "Hello Notion!")
        assert "✅" in result
        assert "1 block" in result
        children = mock_req.call_args[1]["json"]["children"]
        assert children[0]["type"] == "paragraph"

    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_append_multiple_blocks(self, mock_req):
        mock_req.return_value = {}
        from tools.notion import notion_append_blocks
        result = notion_append_blocks("page-abc-123", "Item 1\n\nItem 2\n\nItem 3",
                                      block_type="bulleted_list_item")
        assert "3 block" in result
        children = mock_req.call_args[1]["json"]["children"]
        assert len(children) == 3
        assert all(c["type"] == "bulleted_list_item" for c in children)

    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_append_todo_has_checked_field(self, mock_req):
        mock_req.return_value = {}
        from tools.notion import notion_append_blocks
        notion_append_blocks("page-abc-123", "Do something", block_type="to_do")
        children = mock_req.call_args[1]["json"]["children"]
        assert children[0]["to_do"]["checked"] is False


# ---------------------------------------------------------------------------
# notion_update_page
# ---------------------------------------------------------------------------

class TestNotionUpdatePage:
    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_update_checkbox(self, mock_req):
        mock_req.return_value = {}
        from tools.notion import notion_update_page
        result = notion_update_page("page-abc-123", '{"Done": {"checkbox": true}}')
        assert "✅" in result
        assert "Done" in result
        payload = mock_req.call_args[1]["json"]
        assert payload["properties"]["Done"]["checkbox"] is True

    def test_update_invalid_json(self):
        import os
        os.environ["NOTION_API_KEY"] = "secret_test"
        from tools.notion import notion_update_page
        result = notion_update_page("page-abc-123", "not json")
        assert "❌" in result
        assert "Invalid JSON" in result


# ---------------------------------------------------------------------------
# notion_query_database
# ---------------------------------------------------------------------------

class TestNotionQueryDatabase:
    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_query_returns_entries(self, mock_req):
        mock_req.return_value = {
            "results": [
                {
                    "id": "entry-1",
                    "url": "https://notion.so/entry-1",
                    "properties": {
                        "Name": {
                            "type": "title",
                            "title": [{"plain_text": "Buy Milk"}],
                        },
                        "Done": {"type": "checkbox", "checkbox": False},
                    },
                }
            ],
            "has_more": False,
        }
        from tools.notion import notion_query_database
        result = notion_query_database("db-xyz-456")
        assert "Buy Milk" in result
        assert "entry-1" in result
        assert "☐" in result  # unchecked checkbox

    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_query_empty(self, mock_req):
        mock_req.return_value = {"results": [], "has_more": False}
        from tools.notion import notion_query_database
        result = notion_query_database("db-xyz-456")
        assert "No entries" in result

    def test_query_invalid_filter_json(self):
        import os
        os.environ["NOTION_API_KEY"] = "secret_test"
        from tools.notion import notion_query_database
        result = notion_query_database("db-xyz-456", filter_json="bad json")
        assert "❌" in result


# ---------------------------------------------------------------------------
# notion_delete_block
# ---------------------------------------------------------------------------

class TestNotionDeleteBlock:
    @patch("tools.notion._req")
    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_test"})
    def test_delete_block(self, mock_req):
        mock_req.return_value = {}
        from tools.notion import notion_delete_block
        result = notion_delete_block("block-123")
        assert "✅" in result
        assert "block-123" in result
        mock_req.assert_called_once_with("DELETE", "/blocks/block-123")


# ---------------------------------------------------------------------------
# _summarize_block helper
# ---------------------------------------------------------------------------

class TestSummarizeBlock:
    def test_paragraph(self):
        from tools.notion import _summarize_block
        block = {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Hi"}]}}
        assert _summarize_block(block) == "Hi"

    def test_heading(self):
        from tools.notion import _summarize_block
        block = {"type": "heading_2", "heading_2": {"rich_text": [{"plain_text": "Title"}]}}
        assert _summarize_block(block) == "## Title"

    def test_todo_unchecked(self):
        from tools.notion import _summarize_block
        block = {"type": "to_do", "to_do": {"rich_text": [{"plain_text": "Task"}], "checked": False}}
        assert _summarize_block(block) == "[ ] Task"

    def test_todo_checked(self):
        from tools.notion import _summarize_block
        block = {"type": "to_do", "to_do": {"rich_text": [{"plain_text": "Task"}], "checked": True}}
        assert _summarize_block(block) == "[x] Task"

    def test_divider(self):
        from tools.notion import _summarize_block
        block = {"type": "divider", "divider": {}}
        assert _summarize_block(block) == "---"

    def test_child_page(self):
        from tools.notion import _summarize_block
        block = {"type": "child_page", "child_page": {"title": "My Sub-Page"}}
        assert "Sub-Page" in _summarize_block(block)
