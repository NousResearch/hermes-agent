"""Credentials never reach the durable transcript in cleartext.

The log path and the request-dump path both run ``agent.redact``; the
persistence path used to write the conversation as given, so a key pasted into
a turn (or printed by a tool) stayed verbatim in ``state.db`` for the life of
the session. The write funnel ``_message_row_params`` now runs the same
redactor over message text.

The second test pins the other half of the contract: ordinary prose and code
are stored byte-for-byte. A false positive silently corrupts a transcript, so
that direction is guarded too.
"""
from __future__ import annotations

import sqlite3

import pytest

from hermes_state import SessionDB
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session

# Invented credentials in the shape of real ones — never a live value.
FAKE_GITHUB = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
FAKE_ROUTER = "sk-or-v1-" + "0123456789abcdef" * 4


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-redact", source="cli")
    yield d, tmp_path / "state.db"
    d.close()


def _stored(db_path, role):
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT role, content FROM messages WHERE session_id = 'sess-redact' AND role = ? "
            "ORDER BY id DESC LIMIT 1",
            (role,),
        ).fetchone()
    finally:
        con.close()
    assert row is not None, f"no {role} row was written"
    return row[0], row[1]


def test_pasted_credential_is_stored_redacted(db):
    d, db_path = db
    prompt = f"deploy with {FAKE_GITHUB} then set ROUTER={FAKE_ROUTER} please"
    d.append_message(session_id="sess-redact", role="user", content=prompt)

    role, stored = _stored(db_path, "user")
    assert role == "user"
    assert FAKE_GITHUB not in stored, "a pasted token reached the transcript verbatim"
    assert FAKE_ROUTER not in stored, "a pasted token reached the transcript verbatim"
    # The row survives with its prose: redaction masks the token, not the message.
    assert stored.startswith("deploy with ") and stored.endswith(" please")
    assert len(stored) < len(prompt)


def test_tool_output_is_stored_redacted(db):
    d, db_path = db
    output = f"curl -H 'Authorization: Bearer {FAKE_ROUTER}' https://api.example.com\n200 OK"
    d.append_message(
        session_id="sess-redact", role="tool", content=output,
        tool_name="terminal", tool_call_id="call_1",
    )

    _, stored = _stored(db_path, "tool")
    assert FAKE_ROUTER not in stored, "tool output reached the transcript verbatim"
    assert stored.endswith("200 OK")


def test_ordinary_prose_and_code_are_stored_verbatim(db):
    d, db_path = db
    message = (
        "Here is the fix — rename the variable and rerun the suite.\n\n"
        "```python\n"
        "def main(tokens: dict[str, int]) -> None:\n"
        "    MAX_TOKENS = 100\n"
        "    api_key_length = len(tokens)\n"
        "    print(f'done: {api_key_length}')\n"
        "```\n\n"
        "See https://example.com/docs/setup for the background."
    )
    d.append_message(session_id="sess-redact", role="assistant", content=message)

    _, stored = _stored(db_path, "assistant")
    assert stored == message, "redaction rewrote text that held no credential"


def test_multimodal_parts_keep_their_shape(db):
    d, db_path = db
    parts = [
        {"type": "text", "text": f"screenshot of the failure, token {FAKE_GITHUB}"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ]
    d.append_message(session_id="sess-redact", role="user", content=parts)

    _, stored = _stored(db_path, "user")
    assert FAKE_GITHUB not in stored
    assert "image/png;base64,iVBORw0KGgo=" in stored, "the image part was rewritten"


# --- the sidecar writers -------------------------------------------------
#
# ``api_content`` is the provider-facing copy of a user message. It reaches the
# column through a bare ``UPDATE`` in ``_store/api_content.py`` (and again in the
# worker-authority duplicate in ``workers/_lifecycle.py``), never through
# ``_message_row_params``, so masking on the insert path alone leaves this door
# open. Probed before the fix: the sidecar landed verbatim.

SIDECAR_CONTENT = "deploy with the token please"
SIDECAR_TEXT = f"deploy with the token {FAKE_GITHUB} please"


def _sidecar(db_path, row_id):
    con = sqlite3.connect(db_path)
    try:
        return con.execute("SELECT api_content FROM messages WHERE id = ?", (row_id,)).fetchone()[0]
    finally:
        con.close()


@pytest.fixture()
def store(tmp_path):
    s = open_persistence(profile_scope(home=tmp_path))
    seed_session(s, "s1", "cli")
    yield s, next(tmp_path.rglob("*.db"))
    s.close()


def test_row_addressed_sidecar_is_stored_redacted(store):
    s, db_path = store
    [row_id] = seed_messages(s, "s1", {"role": "user", "content": SIDECAR_CONTENT})

    assert s.messages.set_message_api_content("s1", row_id, SIDECAR_CONTENT, SIDECAR_TEXT)

    stored = _sidecar(db_path, row_id)
    assert stored is not None, "the sidecar was not written at all"
    assert FAKE_GITHUB not in stored, "a credential reached the sidecar verbatim"
    assert stored.startswith("deploy with the token ") and stored.endswith(" please")


def test_positional_sidecar_is_stored_redacted(store):
    s, db_path = store
    [row_id] = seed_messages(s, "s1", {"role": "user", "content": SIDECAR_CONTENT})

    assert s.messages.set_latest_user_api_content("s1", SIDECAR_CONTENT, SIDECAR_TEXT)

    stored = _sidecar(db_path, row_id)
    assert stored is not None
    assert FAKE_GITHUB not in stored, "a credential reached the sidecar verbatim"


def test_worker_sidecar_handlers_store_redacted_text(store):
    """The worker-authority path keeps its own copy of the SQL; it must mask too."""
    s, db_path = store
    [row_id] = seed_messages(s, "s1", {"role": "user", "content": SIDECAR_CONTENT})
    from persistence.sqlite.workers._lifecycle import worker_api_content, worker_message_api_content

    con = sqlite3.connect(db_path)
    try:
        addressed = worker_message_api_content(
            None, con, "s1", {"row_id": row_id, "content": SIDECAR_CONTENT, "api_content": SIDECAR_TEXT})
        assert addressed["value"] == 1
        positional = worker_api_content(
            None, con, "s1", {"content": SIDECAR_CONTENT, "api_content": SIDECAR_TEXT})
        assert positional["value"] == 1
        con.commit()
    finally:
        con.close()

    stored = _sidecar(db_path, row_id)
    assert FAKE_GITHUB not in stored, "a credential reached the sidecar verbatim through the worker path"
