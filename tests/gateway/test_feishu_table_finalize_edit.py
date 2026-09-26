"""Issue #121108: Feishu streamed reply with a Markdown table delivered twice.

A finalize ``edit_message`` builds a ``post`` payload for table content. Feishu
rejects it with code 230001 ("content format of the post type is incorrect"), the
pre-existing retry switched ``msg_type`` to ``text`` — which Feishu refuses on a
message that was born as ``post`` (a message cannot be re-typed) — so
``edit_message`` reported failure and the stream consumer fired a fresh full-text
send next to the frozen preview: two messages.

Pinned lark-oapi 1.6.8 ``Transport.execute`` returns a ``RawResponse`` for every
HTTP status and never raises on an API rejection, so the retry has to happen on
the result object (`BaseResponse.success()` == ``code == 0``). The fix keeps
``msg_type=post`` and drops the GFM table separator rows, which is the shape the
API accepts; a plain-text retry remains the last resort.

These tests drive ``edit_message`` with a stubbed transport that mimics the pinned
SDK: every call returns a result object, and a ``post`` payload still carrying a
table is rejected the way the live API rejects it.
"""
from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("feishu")

TABLE_CONTENT = (
    "Here is the comparison:\n"
    "| col A | col B |\n"
    "| ----- | ----- |\n"
    "| 1     | 2     |"
)

_POST_INVALID_MSG = "content format of the post type is incorrect"
# Independent of the adapter's own regex: a separator row is pipes, dashes,
# colons and spaces only (``| --- | :--: |``).
_DELIMITER_ROW_RE = re.compile(r"^[|\s:\-]*\|[|\s:\-]*$")


def _has_table_row(content: str) -> bool:
    for line in content.splitlines():
        stripped = line.strip()
        if "|" in stripped and "-" in stripped and _DELIMITER_ROW_RE.match(stripped):
            return True
    return False


def _post_text(payload: str) -> str:
    return json.loads(payload)["zh_cn"]["content"][0][0]["text"]


def _make_instance(record: list):
    """Adapter whose update body builder records ``(msg_type, payload)`` per attempt."""
    inst = object.__new__(_adapter.FeishuAdapter)
    # Truthy stub client: the stubbed _run_blocking intercepts before it is touched.
    inst._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(update=lambda req: None)))
    )
    inst._build_update_message_body = lambda **kw: record.append((kw["msg_type"], kw["content"])) or (
        "body",
        kw["msg_type"],
    )
    inst._build_update_message_request = lambda message_id, request_body: ("req", message_id, request_body)
    return inst


def _result_ok():
    return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id=None))


def _result_rejected():
    return SimpleNamespace(success=lambda: False, code=230001, msg=_POST_INVALID_MSG)


def _run(coro):
    return asyncio.run(coro)


def test_finalize_edit_table_post_rejected_retries_as_post_without_table_rows():
    """RED (issue #121108): the retry must stay ``post`` and drop the table rows."""
    record: list = []
    inst = _make_instance(record)

    async def _run_blocking(func, *args):
        # Live transport: result object for every call, 230001 while a table is present.
        msg_type, payload = record[-1]
        if msg_type == "post" and _has_table_row(_post_text(payload)):
            return _result_rejected()
        return _result_ok()

    inst._run_blocking = _run_blocking
    result = _run(inst.edit_message("chat-1", "msg-1", TABLE_CONTENT, finalize=True))

    assert result.success, f"edit_message must succeed without a duplicate send; got {result.error!r}"
    assert [c[0] for c in record] == ["post", "post"], f"expected two post attempts, no re-typing; got {record!r}"
    first_text, retry_text = (_post_text(c[1]) for c in record)
    assert _has_table_row(first_text), "the first attempt must be the table payload (RED on main)"
    assert not _has_table_row(retry_text), f"retry payload still carries a table row: {retry_text!r}"
    assert "| 1     | 2     |" in retry_text, "content must stay visible, only the table syntax goes"
    assert "Here is the comparison:" in retry_text
    assert result.message_id == "msg-1"


def test_finalize_edit_flattened_post_rejected_falls_back_to_text():
    """Last resort: when even the table-free post is rejected, retry as text once."""
    record: list = []
    inst = _make_instance(record)

    async def _run_blocking(func, *args):
        msg_type, _payload = record[-1]
        return _result_rejected() if msg_type == "post" else _result_ok()

    inst._run_blocking = _run_blocking
    result = _run(inst.edit_message("chat-1", "msg-1", TABLE_CONTENT, finalize=True))

    assert result.success, f"text retry must still be attempted; got {result.error!r}"
    assert [c[0] for c in record] == ["post", "post", "text"], f"expected post, post, text; got {record!r}"
    assert result.message_id == "msg-1"


def test_finalize_edit_non_content_error_still_fails_without_retry():
    """Non-content errors must not trigger a fallback (no behavior change)."""
    record: list = []
    inst = _make_instance(record)

    async def _run_blocking(func, *args):
        return SimpleNamespace(success=lambda: False, code=99999, msg="some transient error")

    inst._run_blocking = _run_blocking
    result = _run(inst.edit_message("chat-1", "msg-1", "plain hello", finalize=True))

    assert not result.success
    assert [c[0] for c in record] == ["text"], f"plain content must attempt exactly one update; got {record!r}"


def test_finalize_edit_plain_text_post_rejection_needs_no_flatten_attempt():
    """A prose post (no table) rejected as invalid post: one post attempt, then text."""
    record: list = []
    inst = _make_instance(record)

    async def _run_blocking(func, *args):
        msg_type, _payload = record[-1]
        return _result_rejected() if msg_type == "post" else _result_ok()

    inst._run_blocking = _run_blocking
    result = _run(inst.edit_message("chat-1", "msg-1", "**bold** prose, no table", finalize=True))

    assert result.success
    assert [c[0] for c in record] == ["post", "text"], f"no duplicate post attempt expected; got {record!r}"


def test_drop_gfm_table_delimiters_keeps_content_and_hr():
    drop = _adapter._drop_gfm_table_delimiters
    assert drop(TABLE_CONTENT) == (
        "Here is the comparison:\n"
        "| col A | col B |\n"
        "| 1     | 2     |"
    )
    aligned = "| a | b |\n|:---:|:--:|\n| 1 | 2 |"
    assert drop(aligned) == "| a | b |\n| 1 | 2 |"
    assert drop("text\n\n---\n\nmore") == "text\n\n---\n\nmore"
    assert drop("no table here") == "no table here"
