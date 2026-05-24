import pytest

from gateway.platforms.feishu import comments


@pytest.mark.asyncio
async def test_reply_to_comment_sends_open_id_mentions_as_person_elements(monkeypatch):
    captured = {}

    async def fake_exec_request(client, method, uri, paths=None, queries=None, body=None):
        captured.update(
            method=method,
            uri=uri,
            paths=paths,
            queries=queries,
            body=body,
        )
        return 0, "ok", {}

    monkeypatch.setattr(comments, "_exec_request", fake_exec_request)

    success, code = await comments.reply_to_comment(
        object(),
        "doxc_token",
        "docx",
        "cm_1",
        "@ou_490767d037764fbccf7f30e504f63af9 看一下 <draft>",
    )

    assert success is True
    assert code == 0
    assert captured["uri"] == "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"
    assert ("file_type", "docx") in captured["queries"]
    assert ("user_id_type", "open_id") in captured["queries"]
    assert captured["body"] == {
        "content": {
            "elements": [
                {
                    "type": "person",
                    "person": {"user_id": "ou_490767d037764fbccf7f30e504f63af9"},
                },
                {
                    "type": "text_run",
                    "text_run": {"text": " 看一下 &lt;draft&gt;"},
                },
            ]
        }
    }


@pytest.mark.asyncio
async def test_add_whole_comment_sends_open_id_mentions_as_person_elements(monkeypatch):
    captured = {}

    async def fake_exec_request(client, method, uri, paths=None, queries=None, body=None):
        captured.update(
            method=method,
            uri=uri,
            paths=paths,
            queries=queries,
            body=body,
        )
        return 0, "ok", {}

    monkeypatch.setattr(comments, "_exec_request", fake_exec_request)

    ok = await comments.add_whole_comment(
        object(),
        "doxc_token",
        "docx",
        "请 @ou_490767d037764fbccf7f30e504f63af9 看一下",
    )

    assert ok is True
    assert captured["uri"] == "/open-apis/drive/v1/files/:file_token/comments"
    assert ("file_type", "docx") in captured["queries"]
    assert ("user_id_type", "open_id") in captured["queries"]
    assert captured["body"] == {
        "reply_list": {
            "replies": [
                {
                    "content": {
                        "elements": [
                            {
                                "type": "text_run",
                                "text_run": {"text": "请 "},
                            },
                            {
                                "type": "person",
                                "person": {"user_id": "ou_490767d037764fbccf7f30e504f63af9"},
                            },
                            {
                                "type": "text_run",
                                "text_run": {"text": " 看一下"},
                            },
                        ]
                    }
                }
            ]
        }
    }
