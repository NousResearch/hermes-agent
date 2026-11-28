"""Removing the bot's @mention from an inbound group message keeps the user's indentation and
blank lines (code pastes, multi-line slash-command args). Port of openclaw/openclaw#183606."""

import json

from plugins.platforms.feishu.adapter import normalize_feishu_message
from plugins.platforms.matrix.adapter import MatrixAdapter

CODE = "def f(x):\n    if x:\n        return 1\n\n    return 2"


def test_feishu_text_and_post_code_block_keep_layout():
    text = normalize_feishu_message(message_type="text", raw_content=json.dumps({"text": "@_user_1 fix:\n" + CODE}))
    assert text.text_content == "fix:\n" + CODE
    post = {"zh_cn": {"title": "", "content": [[{"tag": "text", "text": "fix:"}],
                                               [{"tag": "code_block", "language": "python", "text": CODE}]]}}
    parsed = normalize_feishu_message(message_type="post", raw_content=json.dumps(post))
    assert parsed.text_content == "fix:\n```python\n" + CODE + "\n```"


def test_matrix_mention_strip_keeps_indentation():
    adapter = object.__new__(MatrixAdapter)
    adapter._user_id = "@hermes:example.org"
    adapter._user_localpart = lambda: "hermes"
    assert adapter._strip_mention("@hermes:example.org fix:\n" + CODE) == "fix:\n" + CODE
    assert adapter._strip_mention("hello @hermes, how are you") == "hello, how are you"
    assert adapter._strip_mention("ask @hermes about it") == "ask about it"
