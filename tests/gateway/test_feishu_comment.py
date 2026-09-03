"""Tests for feishu_comment — event filtering, access control integration, wiki reverse lookup."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from plugins.platforms.feishu.feishu_comment import (
    parse_drive_comment_event,
    _ALLOWED_NOTICE_TYPES,
    _sanitize_comment_text,
)


def _make_event(
    comment_id="c1",
    reply_id="r1",
    notice_type="add_reply",
    file_token="docx_token",
    file_type="docx",
    from_open_id="ou_user",
    to_open_id="ou_bot",
    is_mentioned=True,
):
    """Build a minimal drive comment event SimpleNamespace."""
    return SimpleNamespace(event={
        "event_id": "evt_1",
        "comment_id": comment_id,
        "reply_id": reply_id,
        "is_mentioned": is_mentioned,
        "timestamp": "1713200000",
        "notice_meta": {
            "file_token": file_token,
            "file_type": file_type,
            "notice_type": notice_type,
            "from_user_id": {"open_id": from_open_id},
            "to_user_id": {"open_id": to_open_id},
        },
    })


class TestParseEvent(unittest.TestCase):
    def test_parse_valid_event(self):
        evt = _make_event()
        parsed = parse_drive_comment_event(evt)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["comment_id"], "c1")
        self.assertEqual(parsed["file_type"], "docx")
        self.assertEqual(parsed["from_open_id"], "ou_user")
        self.assertEqual(parsed["to_open_id"], "ou_bot")


class TestEventFiltering(unittest.TestCase):
    """Test the filtering logic in handle_drive_comment_event."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule")
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed")
    def test_self_reply_filtered(self, mock_allowed, mock_resolve, mock_load):
        """Events where from_open_id == self_open_id should be dropped."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event

        evt = _make_event(from_open_id="ou_bot", to_open_id="ou_bot")
        self._run(handle_drive_comment_event(Mock(), evt, self_open_id="ou_bot"))
        mock_load.assert_not_called()

    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule")
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed")
    def test_wrong_receiver_filtered(self, mock_allowed, mock_resolve, mock_load):
        """Events where to_open_id != self_open_id should be dropped."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event

        evt = _make_event(to_open_id="ou_other_bot")
        self._run(handle_drive_comment_event(Mock(), evt, self_open_id="ou_bot"))
        mock_load.assert_not_called()


class TestAccessControlIntegration(unittest.TestCase):
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @patch("plugins.platforms.feishu.feishu_comment_rules.has_wiki_keys", return_value=False)
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed", return_value=False)
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule")
    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    def test_denied_user_no_side_effects(self, mock_load, mock_resolve, mock_allowed, mock_wiki_keys):
        """Denied user should not trigger typing reaction or agent."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event
        from plugins.platforms.feishu.feishu_comment_rules import ResolvedCommentRule

        mock_resolve.return_value = ResolvedCommentRule(True, "allowlist", frozenset(), "top")
        mock_load.return_value = Mock()

        client = Mock()
        evt = _make_event()
        self._run(handle_drive_comment_event(client, evt, self_open_id="ou_bot"))

        # No API calls should be made for denied users
        client.request.assert_not_called()


class TestSanitizeCommentText(unittest.TestCase):
    def test_angle_brackets_escaped(self):
        self.assertEqual(_sanitize_comment_text("List<String>"), "List&lt;String&gt;")

    def test_ampersand_escaped_first(self):
        self.assertEqual(_sanitize_comment_text("a & b"), "a &amp; b")

    def test_ampersand_not_double_escaped(self):
        result = _sanitize_comment_text("a < b & c > d")
        self.assertEqual(result, "a &lt; b &amp; c &gt; d")
        self.assertNotIn("&amp;lt;", result)
        self.assertNotIn("&amp;gt;", result)


class TestWikiReverseLookup(unittest.TestCase):
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @patch("plugins.platforms.feishu.feishu_comment._exec_request")
    def test_reverse_lookup_success(self, mock_exec):
        from plugins.platforms.feishu.feishu_comment import _reverse_lookup_wiki_token

        mock_exec.return_value = (0, "Success", {
            "node": {"node_token": "WIKI_TOKEN_123", "obj_token": "docx_abc"},
        })
        result = self._run(_reverse_lookup_wiki_token(Mock(), "docx", "docx_abc"))
        self.assertEqual(result, "WIKI_TOKEN_123")
        # Verify correct API params
        call_args = mock_exec.call_args
        queries = call_args[1].get("queries") or call_args[0][3]
        query_dict = dict(queries)
        self.assertEqual(query_dict["token"], "docx_abc")
        self.assertEqual(query_dict["obj_type"], "docx")


if __name__ == "__main__":
    unittest.main()
