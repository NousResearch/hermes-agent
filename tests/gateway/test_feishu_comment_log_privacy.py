"""Comment diagnostics retain status while the actual request and history stay exact."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from plugins.platforms.feishu import feishu_comment as comment


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [0, 1069302])
@pytest.mark.parametrize("data", [{"private document key": "private response body"}, None])
async def test_api_logs_do_not_duplicate_private_request_or_response(caplog, code, data):
    pytest.importorskip("lark_oapi", reason="request builder integration requires the optional Feishu SDK")
    body = {"content": "private document reply"}
    paths = {"file_token": "private-document-identity"}
    queries = [("user_id", "private-user-identity")]
    response = SimpleNamespace(
        code=code, msg="private echoed error", raw=SimpleNamespace(content=json.dumps({"data": data}))
    )
    client = SimpleNamespace(request=Mock(return_value=response))

    with caplog.at_level(logging.INFO, logger=comment.__name__):
        result = await comment._exec_request(
            client, "POST", comment._REPLIES_URI, paths=paths, queries=queries, body=body
        )

    request = client.request.call_args.args[0]
    assert request.body == body
    assert request.paths == paths
    assert request.queries == queries
    assert result == (code, response.msg, data)
    assert "private" not in caplog.text
    assert f"code={code}" in caplog.text


def test_session_history_retains_raw_messages_without_logging_identity(monkeypatch, caplog):
    monkeypatch.setattr(comment, "_session_cache", {})
    key = "private-document:private-user"
    messages = [{"role": "user", "content": "private document question"}]

    with caplog.at_level(logging.INFO, logger=comment.__name__):
        comment._save_session_history(key, messages)
        restored = comment._load_session_history(key)

    assert restored == messages
    assert "private" not in caplog.text
    assert "1 messages" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["reaction", "meta", "batch", "whole", "replies", "reply", "add", "wiki"])
async def test_api_failure_logs_keep_status_without_echoed_error(monkeypatch, caplog, operation):
    if operation == "reaction":
        pytest.importorskip("lark_oapi", reason="reaction entrypoint checks the optional Feishu SDK")
    monkeypatch.setattr(comment, "_exec_request", AsyncMock(return_value=(1069302, "private echoed error", {})))
    monkeypatch.setattr(comment.asyncio, "sleep", AsyncMock())
    client = Mock()
    calls = {
        "reaction": lambda: comment.update_comment_reaction(client, "add", file_token="private file", file_type="docx", reply_id="private reply"),
        "meta": lambda: comment.query_document_meta(client, "private file", "docx"),
        "batch": lambda: comment.batch_query_comment(client, "private file", "docx", "private comment"),
        "whole": lambda: comment.list_whole_comments(client, "private file", "docx"),
        "replies": lambda: comment.list_comment_replies(client, "private file", "docx", "private comment"),
        "reply": lambda: comment.reply_to_comment(client, "private file", "docx", "private comment", "private reply"),
        "add": lambda: comment.add_whole_comment(client, "private file", "docx", "private reply"),
        "wiki": lambda: comment._reverse_lookup_wiki_token(client, "docx", "private file"),
    }
    with caplog.at_level(logging.DEBUG, logger=comment.__name__):
        await calls[operation]()
    assert "private" not in caplog.text
    assert "1069302" in caplog.text


@pytest.mark.asyncio
async def test_event_logs_do_not_duplicate_prompt_or_delivered_response(monkeypatch, caplog):
    from plugins.platforms.feishu import feishu_comment_rules as rules

    fields = dict(file_token="private file", file_type="docx", comment_id="private comment", reply_id="",
                  from_open_id="private sender", to_open_id="bot", notice_type="add_reply")
    monkeypatch.setattr(comment, "parse_drive_comment_event", lambda event: fields)
    monkeypatch.setattr(rules, "load_config", lambda: {})
    resolve = Mock(return_value=rules.ResolvedCommentRule(True, "allowlist", frozenset(), "top"))
    monkeypatch.setattr(rules, "resolve_rule", resolve)
    monkeypatch.setattr(rules, "has_wiki_keys", lambda config: False)
    authorize = Mock(return_value=True)
    monkeypatch.setattr(rules, "is_user_allowed", authorize)
    monkeypatch.setattr(comment, "query_document_meta", AsyncMock(return_value={"title": "private title"}))
    monkeypatch.setattr(comment, "batch_query_comment", AsyncMock(return_value={"is_whole": False}))
    prompt = "private prompt\nprivate continuation"
    monkeypatch.setattr(comment, "_local_comment_prompt", AsyncMock(return_value=prompt))
    run = Mock(return_value="private response")
    monkeypatch.setattr(comment, "_run_comment_agent", run)
    deliver = AsyncMock(return_value=True)
    monkeypatch.setattr(comment, "deliver_comment_reply", deliver)
    client = Mock()
    with caplog.at_level(logging.DEBUG, logger=comment.__name__):
        await comment.handle_drive_comment_event(client, object(), self_open_id="bot")

    resolve.assert_called_once_with({}, "docx", "private file")
    assert authorize.call_args.args[1] == "private sender"
    assert run.call_args.args == (prompt, client, "comment-doc:docx:private file")
    deliver.assert_awaited_once_with(client, "private file", "docx", "private comment", "private response", False)
    assert "private" not in caplog.text
    assert "Reply delivered successfully" in caplog.text


@pytest.mark.asyncio
async def test_missing_document_title_does_not_break_metadata_lookup(monkeypatch, caplog):
    monkeypatch.setattr(comment, "_exec_request", AsyncMock(return_value=(0, "", {"metas": [{"title": None}]})))
    with caplog.at_level(logging.DEBUG, logger=comment.__name__):
        result = await comment.query_document_meta(Mock(), "private file", "docx")
    assert result == {"title": None, "url": "", "doc_type": "docx"}
    assert "private" not in caplog.text
