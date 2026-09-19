"""Issue #101351 — the diverted session transcript must not persist live keys.

When ``state.db`` is replaced under a live process the pending message batch is
appended verbatim to ``HERMES_HOME/sessions/<id>.jsonl``. Tool output in that
batch routinely carries an ``export OPENAI_API_KEY=…`` line or a raw key in a
command body, and the file sits in a path an agent may go on to commit. The
breadcrumb must be written through the repo's existing secret redactor while
keeping the message shape recoverable.
"""

from __future__ import annotations

import json

from hermes_state import divert_session_transcript_jsonl

KEY = "sk-" + "a1B2c3D4e5F6g7H8i9J0kLmN"


def test_diverted_breadcrumb_redacts_keys_but_keeps_structure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    messages = [
        {"role": "tool", "tool_call_id": "call-1", "name": "terminal",
         "content": f"$ export OPENAI_API_KEY={KEY}\nwrote 3 files"},
        {"role": "assistant", "content": [
            {"type": "text", "text": f"the key is {KEY}"},
            {"type": "image_url", "image_url": {"url": "file:///tmp/a.png"}},
        ]},
        "a bare string record",
    ]

    path = divert_session_transcript_jsonl("sess-1", messages)
    raw = path.read_text(encoding="utf-8")

    assert KEY not in raw
    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == 3
    assert records[0]["role"] == "tool"
    assert records[0]["tool_call_id"] == "call-1"
    assert records[0]["name"] == "terminal"
    assert "wrote 3 files" in records[0]["content"]  # non-secret text survives
    # Nested multimodal parts keep their shape and their non-secret fields.
    assert records[1]["content"][0]["type"] == "text"
    assert records[1]["content"][1]["image_url"]["url"] == "file:///tmp/a.png"
    # A non-dict record is wrapped by the (unchanged) writer as {"content": str(msg)}.
    assert records[2] == {"content": "a bare string record"}


def test_diverted_breadcrumb_leaves_secret_free_messages_byte_identical(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    message = {"role": "user", "content": "no secrets here: 42 lines changed in main.py"}

    path = divert_session_transcript_jsonl("sess-2", [message])

    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0]) == message
