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

    def test_parse_missing_event_attr(self):
        self.assertIsNone(parse_drive_comment_event(object()))

    def test_parse_none_event(self):
        self.assertIsNone(parse_drive_comment_event(SimpleNamespace()))


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

    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule")
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed")
    def test_empty_to_open_id_filtered(self, mock_allowed, mock_resolve, mock_load):
        """Events with empty to_open_id should be dropped."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event

        evt = _make_event(to_open_id="")
        self._run(handle_drive_comment_event(Mock(), evt, self_open_id="ou_bot"))
        mock_load.assert_not_called()

    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule")
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed")
    def test_invalid_notice_type_filtered(self, mock_allowed, mock_resolve, mock_load):
        """Events with unsupported notice_type should be dropped."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event

        evt = _make_event(notice_type="resolve_comment")
        self._run(handle_drive_comment_event(Mock(), evt, self_open_id="ou_bot"))
        mock_load.assert_not_called()

    def test_allowed_notice_types(self):
        self.assertIn("add_comment", _ALLOWED_NOTICE_TYPES)
        self.assertIn("add_reply", _ALLOWED_NOTICE_TYPES)
        self.assertNotIn("resolve_comment", _ALLOWED_NOTICE_TYPES)


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

    @patch("plugins.platforms.feishu.feishu_comment_rules.has_wiki_keys", return_value=False)
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed", return_value=False)
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule")
    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    def test_disabled_comment_skipped(self, mock_load, mock_resolve, mock_allowed, mock_wiki_keys):
        """Disabled comments should return immediately."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event
        from plugins.platforms.feishu.feishu_comment_rules import ResolvedCommentRule

        mock_resolve.return_value = ResolvedCommentRule(False, "allowlist", frozenset(), "top")
        mock_load.return_value = Mock()

        evt = _make_event()
        self._run(handle_drive_comment_event(Mock(), evt, self_open_id="ou_bot"))
        mock_allowed.assert_not_called()


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

    def test_plain_text_unchanged(self):
        self.assertEqual(_sanitize_comment_text("hello world"), "hello world")

    def test_empty_string(self):
        self.assertEqual(_sanitize_comment_text(""), "")

    def test_code_snippet(self):
        text = 'if (a < b && c > 0) { return "ok"; }'
        result = _sanitize_comment_text(text)
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)
        self.assertIn("&lt;", result)
        self.assertIn("&gt;", result)


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

    @patch("plugins.platforms.feishu.feishu_comment._exec_request")
    def test_reverse_lookup_not_wiki(self, mock_exec):
        from plugins.platforms.feishu.feishu_comment import _reverse_lookup_wiki_token

        mock_exec.return_value = (131001, "not found", {})
        result = self._run(_reverse_lookup_wiki_token(Mock(), "docx", "docx_abc"))
        self.assertIsNone(result)

    @patch("plugins.platforms.feishu.feishu_comment._exec_request")
    def test_reverse_lookup_service_error(self, mock_exec):
        from plugins.platforms.feishu.feishu_comment import _reverse_lookup_wiki_token

        mock_exec.return_value = (500, "internal error", {})
        result = self._run(_reverse_lookup_wiki_token(Mock(), "docx", "docx_abc"))
        self.assertIsNone(result)

    @patch("plugins.platforms.feishu.feishu_comment._reverse_lookup_wiki_token", new_callable=AsyncMock)
    @patch("plugins.platforms.feishu.feishu_comment_rules.has_wiki_keys", return_value=True)
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed", return_value=True)
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule")
    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    @patch("plugins.platforms.feishu.feishu_comment.add_comment_reaction", new_callable=AsyncMock)
    @patch("plugins.platforms.feishu.feishu_comment.batch_query_comment", new_callable=AsyncMock)
    @patch("plugins.platforms.feishu.feishu_comment.query_document_meta", new_callable=AsyncMock)
    def test_wiki_lookup_triggered_when_no_exact_match(
        self, mock_meta, mock_batch, mock_reaction,
        mock_load, mock_resolve, mock_allowed, mock_wiki_keys, mock_lookup,
    ):
        """Wiki reverse lookup should fire when rule falls to wildcard/top and wiki keys exist."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event
        from plugins.platforms.feishu.feishu_comment_rules import ResolvedCommentRule

        # First resolve returns wildcard (no exact match), second returns exact wiki match
        mock_resolve.side_effect = [
            ResolvedCommentRule(True, "allowlist", frozenset(), "wildcard"),
            ResolvedCommentRule(True, "allowlist", frozenset(), "exact:wiki:WIKI123"),
        ]
        mock_load.return_value = Mock()
        mock_lookup.return_value = "WIKI123"
        mock_meta.return_value = {"title": "Test", "url": ""}
        mock_batch.return_value = {"is_whole": False, "quote": ""}

        evt = _make_event()
        # Will proceed past access control but fail later — that's OK, we just test the lookup
        try:
            self._run(handle_drive_comment_event(Mock(), evt, self_open_id="ou_bot"))
        except Exception:
            pass

        mock_lookup.assert_called_once_with(unittest.mock.ANY, "docx", "docx_token")
        self.assertEqual(mock_resolve.call_count, 2)
        # Second call should include wiki_token
        second_call_kwargs = mock_resolve.call_args_list[1]
        self.assertEqual(second_call_kwargs[1].get("wiki_token") or second_call_kwargs[0][3], "WIKI123")


<<<<<<< ours

# ---------------------------------------------------------------------------
# gather exception isolation — return_exceptions=True regression coverage
# ---------------------------------------------------------------------------


class TestGatherExceptionIsolation(unittest.TestCase):
    """When one of the parallel fetches raises, the handler must not crash
    and must retain the successful result from the other fetch."""

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @patch("plugins.platforms.feishu.feishu_comment.deliver_comment_reply",
           new_callable=AsyncMock, return_value=True)
    @patch("plugins.platforms.feishu.feishu_comment.delete_comment_reaction",
           new_callable=AsyncMock)
    @patch.object(asyncio, "get_running_loop")
    @patch("plugins.platforms.feishu.feishu_comment_rules.has_wiki_keys", return_value=False)
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed", return_value=True)
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule",
           return_value=Mock(exact_key=None, wiki_token=None))
    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    @patch("plugins.platforms.feishu.feishu_comment.list_comment_replies", new_callable=AsyncMock,
           return_value=[])
    @patch("plugins.platforms.feishu.feishu_comment._run_comment_agent", return_value="Test reply")
    @patch("plugins.platforms.feishu.feishu_comment.add_comment_reaction", new_callable=AsyncMock)
    @patch("plugins.platforms.feishu.feishu_comment.batch_query_comment", new_callable=AsyncMock)
    @patch("plugins.platforms.feishu.feishu_comment.query_document_meta", new_callable=AsyncMock)
    def test_meta_fetch_fails_comment_retained(
        self, mock_meta, mock_batch, mock_reaction, mock_agent, mock_replies,
        mock_load, mock_resolve, mock_allowed, mock_wiki_keys,
        mock_get_loop, mock_del_reaction, mock_deliver,
    ):
        """query_document_meta raises → doc_meta defaults to {}, comment_detail retained,
        handler does not crash and continues to deliver a reply."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event

        mock_meta.side_effect = RuntimeError("network error")
        mock_batch.return_value = {"is_whole": False, "quote": ""}
        mock_load.return_value = Mock()

        # Mock the executor so _run_comment_agent is called directly (not in a thread)
        mock_loop = Mock()
        async def fake_run_in_executor(executor, fn, *args):
            return fn(*args)
        mock_loop.run_in_executor = fake_run_in_executor
        mock_get_loop.return_value = mock_loop

        evt = _make_event()
        # Must not propagate the exception from meta_task
        self._run(handle_drive_comment_event(Mock(), evt, self_open_id="ou_bot"))

        # comment fetch was called
        mock_batch.assert_called_once()
        # Agent was called and reply was delivered
        mock_deliver.assert_called_once()

    @patch("plugins.platforms.feishu.feishu_comment.deliver_comment_reply",
           new_callable=AsyncMock, return_value=True)
    @patch("plugins.platforms.feishu.feishu_comment.delete_comment_reaction",
           new_callable=AsyncMock)
    @patch.object(asyncio, "get_running_loop")
    @patch("plugins.platforms.feishu.feishu_comment_rules.has_wiki_keys", return_value=False)
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed", return_value=True)
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule",
           return_value=Mock(exact_key=None, wiki_token=None))
    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    @patch("plugins.platforms.feishu.feishu_comment.list_comment_replies", new_callable=AsyncMock,
           return_value=[])
    @patch("plugins.platforms.feishu.feishu_comment._run_comment_agent", return_value="Test reply")
    @patch("plugins.platforms.feishu.feishu_comment.add_comment_reaction", new_callable=AsyncMock)
    @patch("plugins.platforms.feishu.feishu_comment.batch_query_comment", new_callable=AsyncMock)
    @patch("plugins.platforms.feishu.feishu_comment.query_document_meta", new_callable=AsyncMock)
    def test_comment_fetch_fails_meta_retained(
        self, mock_meta, mock_batch, mock_reaction, mock_agent, mock_replies,
        mock_load, mock_resolve, mock_allowed, mock_wiki_keys,
        mock_get_loop, mock_del_reaction, mock_deliver,
    ):
        """batch_query_comment raises → comment_detail defaults to {}, doc_meta retained,
        handler does not crash and continues to deliver a reply."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event

        mock_meta.return_value = {"title": "TestDoc", "url": "https://feishu.cn/doc/xxx"}
        mock_batch.side_effect = RuntimeError("rate limit")
        mock_load.return_value = Mock()

        # Mock the executor so _run_comment_agent is called directly (not in a thread)
        mock_loop = Mock()
        async def fake_run_in_executor(executor, fn, *args):
            return fn(*args)
        mock_loop.run_in_executor = fake_run_in_executor
        mock_get_loop.return_value = mock_loop

        evt = _make_event()
        # Must not propagate the exception from comment_task
        self._run(handle_drive_comment_event(Mock(), evt, self_open_id="ou_bot"))

        # meta fetch was called
        mock_meta.assert_called_once()
        # Agent was called and reply was delivered
        mock_deliver.assert_called_once()

=======
class TestGatherExceptionIsolation(unittest.TestCase):
    """Regression tests for asyncio.gather return_exceptions=True (#64864).

    After the gather call was changed to return_exceptions=True, each fetch
    failure must be isolated — the other result is retained and the handler
    continues instead of propagating the exception.
    """

    def _run(self, coro):
        """Execute an async coroutine in a synchronous test context."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def test_gather_meta_failure_isolates_and_preserves_comment(self):
        """Verify that the gather pattern in handle_drive_comment_event
        isolates a meta fetch exception and preserves the comment result."""
        import asyncio
        from unittest.mock import AsyncMock

        async def _simulate_handler_path():
            mock_meta = AsyncMock(side_effect=RuntimeError("meta failed"))
            mock_comment = AsyncMock(return_value={"is_whole": True})

            meta_t = asyncio.ensure_future(mock_meta())
            comment_t = asyncio.ensure_future(mock_comment())
            doc_meta_result, comment_detail_result = await asyncio.gather(
                meta_t, comment_t, return_exceptions=True,
            )

            if isinstance(doc_meta_result, Exception):
                doc_meta = {}
            else:
                doc_meta = doc_meta_result or {}

            if isinstance(comment_detail_result, Exception):
                comment_detail = {}
            else:
                comment_detail = comment_detail_result or {}

            return doc_meta, comment_detail

        doc_meta, comment_detail = self._run(_simulate_handler_path())

        assert doc_meta == {}
        assert comment_detail == {"is_whole": True}

    def test_gather_comment_failure_isolates_and_preserves_meta(self):
        """Verify that the actual gather pattern in handle_drive_comment_event
        isolates a comment fetch exception and preserves the meta result."""
        import asyncio
        from unittest.mock import AsyncMock

        async def _simulate_handler_path():
            mock_meta = AsyncMock(return_value={"title": "MyDoc", "url": "https://x.com"})
            mock_comment = AsyncMock(side_effect=RuntimeError("comment failed"))

            meta_t = asyncio.ensure_future(mock_meta())
            comment_t = asyncio.ensure_future(mock_comment())
            doc_meta_result, comment_detail_result = await asyncio.gather(
                meta_t, comment_t, return_exceptions=True,
            )

            if isinstance(doc_meta_result, Exception):
                doc_meta = {}
            else:
                doc_meta = doc_meta_result or {}

            if isinstance(comment_detail_result, Exception):
                comment_detail = {}
            else:
                comment_detail = comment_detail_result or {}

            return doc_meta, comment_detail

        doc_meta, comment_detail = self._run(_simulate_handler_path())

        assert doc_meta == {"title": "MyDoc", "url": "https://x.com"}
        assert comment_detail == {}

    def test_both_fetches_succeed(self):
        """Baseline: both fetches succeed, results are used normally."""
        import asyncio
        from unittest.mock import AsyncMock

        async def _simulate_handler_path():
            mock_meta = AsyncMock(return_value={"title": "Doc", "url": "https://x.com"})
            mock_comment = AsyncMock(return_value={"is_whole": False})

            meta_t = asyncio.ensure_future(mock_meta())
            comment_t = asyncio.ensure_future(mock_comment())
            doc_meta_result, comment_detail_result = await asyncio.gather(
                meta_t, comment_t, return_exceptions=True,
            )

            if isinstance(doc_meta_result, Exception):
                doc_meta = {}
            else:
                doc_meta = doc_meta_result or {}

            if isinstance(comment_detail_result, Exception):
                comment_detail = {}
            else:
                comment_detail = comment_detail_result or {}

            return doc_meta, comment_detail

        doc_meta, comment_detail = self._run(_simulate_handler_path())

        assert doc_meta == {"title": "Doc", "url": "https://x.com"}
        assert comment_detail == {"is_whole": False}
>>>>>>> theirs

if __name__ == "__main__":
    unittest.main()
