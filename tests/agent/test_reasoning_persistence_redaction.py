"""Assistant reasoning fields reach state.db redacted, like ``content`` (#43666 gap 3).

A credential the model quotes in its thinking must not persist verbatim through
``reasoning`` / ``reasoning_content`` / unsigned ``reasoning_details`` text, while
signed and opaque replay material stays byte-exact and redaction-off is a no-op.
"""

import sqlite3

import pytest

from agent.anthropic_thinking_replay import strip_rejected_thinking
from agent.chat_completion_helpers import build_assistant_message
from hermes_state import SessionDB

_SECRET = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
_THOUGHT = f"I will use the token {_SECRET} to call the API."


class _Agent:
    stream_delta_callback = None
    _stream_callback = None
    reasoning_callback = None
    verbose_logging = False

    def _extract_reasoning(self, msg):
        return getattr(msg, "reasoning", None)

    def _strip_think_blocks(self, text):
        return text

    def _needs_thinking_reasoning_pad(self):
        return False


class _Msg:
    def __init__(self, **fields):
        self.content = "done"
        self.tool_calls = None
        self.__dict__.update(fields)

    def __getattr__(self, _name):
        return None


def _signed_detail():
    return {"type": "reasoning.text", "text": _THOUGHT, "signature": "sig-opaque"}


def _assistant_message():
    return _Msg(
        reasoning=_THOUGHT,
        reasoning_content=_THOUGHT,
        reasoning_details=[
            {"type": "reasoning.summary", "summary": _THOUGHT},
            _signed_detail(),
            {"type": "reasoning.encrypted", "data": "opaque-blob"},
            # Bedrock Converse shape for unsigned (non-Claude) reasoning.
            {"type": "thinking", "thinking": _THOUGHT},
            {"type": "thinking", "thinking": _THOUGHT, "signature": "sig-anthropic"},
        ],
    )


def _persist(tmp_path, msg):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("s1", source="test")
    db.append_message("s1", "assistant", msg["content"], reasoning=msg.get("reasoning"),
                      reasoning_content=msg.get("reasoning_content"),
                      reasoning_details=msg.get("reasoning_details"))
    history = db.get_messages_as_conversation("s1")
    db.close()
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT reasoning, reasoning_content, reasoning_details FROM messages").fetchall()
    con.close()
    return history[-1], rows


def test_reasoning_fields_persist_redacted_and_signed_details_stay_exact(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True, raising=False)
    source = _assistant_message()
    msg = build_assistant_message(_Agent(), source, "stop")
    stored, rows = _persist(tmp_path, msg)

    reasoning, reasoning_content, details_json = rows[0]
    assert _SECRET not in reasoning
    assert _SECRET not in reasoning_content
    unsigned_summary, signed, encrypted, unsigned_thinking, signed_thinking = stored["reasoning_details"]
    assert _SECRET not in unsigned_summary["summary"]
    assert _SECRET not in unsigned_thinking["thinking"]
    # Provider-verified replay material is untouched (byte-exact).
    assert signed == _signed_detail()
    assert encrypted == {"type": "reasoning.encrypted", "data": "opaque-blob"}
    assert signed_thinking == {"type": "thinking", "thinking": _THOUGHT, "signature": "sig-anthropic"}
    assert details_json.count(_SECRET) == 2  # only inside the two signed entries
    # Copy-on-write: the SDK response object is not mutated.
    assert source.reasoning_details[0]["summary"] == _THOUGHT


def test_redaction_off_keeps_reasoning_verbatim(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False, raising=False)
    msg = build_assistant_message(_Agent(), _assistant_message(), "stop")
    stored, _rows = _persist(tmp_path, msg)
    assert stored["reasoning"] == _THOUGHT
    assert stored["reasoning_content"] == _THOUGHT
    assert stored["reasoning_details"][0]["summary"] == _THOUGHT


@pytest.mark.parametrize("raw_row", [False, True], ids=["masked-row", "pre-redaction-row"])
def test_rejected_signed_thinking_drops_its_masked_mirror(monkeypatch, raw_row):
    """A rejected signed block's readable mirror is dropped whether it was stored masked or raw."""
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True, raising=False)
    signed = {"type": "thinking", "thinking": _THOUGHT, "signature": "sig-anthropic"}
    msg = build_assistant_message(_Agent(), _Msg(reasoning=_THOUGHT, reasoning_details=[signed]), "stop")
    if raw_row:
        msg["reasoning"] = msg["reasoning_content"] = _THOUGHT
    assert strip_rejected_thinking(msg, None) == 1
    assert "reasoning" not in msg
    assert "reasoning_content" not in msg
