"""Dashboard-only projection of gateway vision pre-analysis rows.

Text-mode image routing (`_enrich_with_attached_images` in
`tui_gateway/server.py`) prepends a synthetic `vision_analyze` description to
the next user turn and persists the whole thing as one `user` row. The
dashboard endpoint re-projects those rows so the vision output shows up as a
tool result while the user's own trailing text stays a user message. The
stored rows are never touched.
"""

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.web_server import _normalize_dashboard_message_roles


# Mirror the exact strings emitted by `_enrich_with_attached_images` so these
# tests fail if the gateway wording drifts.
def _hint(path: str) -> str:
    return f"[You can examine it with vision_analyze using image_url: {path}]"


def _success_block(desc: str, path: str) -> str:
    return f"[The user attached an image:\n{desc}]\n{_hint(path)}"


def _failure_block(path: str) -> str:
    return f"[The user attached an image but analysis failed.]\n{_hint(path)}"


def test_success_preamble_with_user_text_splits_into_tool_and_user():
    block = _success_block("A cat asleep on a chair.", "/tmp/cat.png")
    messages = [
        {"role": "user", "content": f"{block}\n\nwhat breed is this?", "tool_name": None}
    ]

    out = _normalize_dashboard_message_roles(messages)

    assert [m["role"] for m in out] == ["tool", "user"]
    assert out[0]["tool_name"] == "vision_analyze"
    assert out[0]["content"] == block
    assert out[1]["content"] == "what breed is this?"
    # Stored row is left untouched.
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].endswith("what breed is this?")
    assert messages[0]["tool_name"] is None


def test_success_preamble_without_user_text_is_single_tool_row():
    block = _success_block("A bar chart of Q3 revenue.", "/tmp/chart.png")
    messages = [{"role": "user", "content": block}]

    out = _normalize_dashboard_message_roles(messages)

    assert len(out) == 1
    assert out[0]["role"] == "tool"
    assert out[0]["tool_name"] == "vision_analyze"
    assert out[0]["content"] == block


def test_failure_preamble_still_labeled_as_tool():
    block = _failure_block("/tmp/broken.png")
    messages = [{"role": "user", "content": f"{block}\n\ncan you read it?"}]

    out = _normalize_dashboard_message_roles(messages)

    assert [m["role"] for m in out] == ["tool", "user"]
    assert out[0]["content"] == block
    assert out[0]["tool_name"] == "vision_analyze"
    assert out[1]["content"] == "can you read it?"


def test_multiple_image_blocks_stay_in_one_tool_row():
    b1 = _success_block("First: a diagram.", "/tmp/a.png")
    b2 = _success_block("Second: some code.", "/tmp/b.png")
    prefix = f"{b1}\n\n{b2}"
    messages = [{"role": "user", "content": f"{prefix}\n\ncompare these"}]

    out = _normalize_dashboard_message_roles(messages)

    assert [m["role"] for m in out] == ["tool", "user"]
    assert out[0]["content"] == prefix
    assert out[1]["content"] == "compare these"


def test_existing_tool_name_is_preserved():
    block = _success_block("A screenshot.", "/tmp/s.png")
    messages = [{"role": "user", "content": block, "tool_name": "custom_vision"}]

    out = _normalize_dashboard_message_roles(messages)

    assert out[0]["role"] == "tool"
    assert out[0]["tool_name"] == "custom_vision"


def test_plain_user_and_other_roles_pass_through_as_copies():
    messages = [
        {"role": "user", "content": "please describe this image"},
        {"role": "assistant", "content": "[The user attached an image: not really]"},
        {"role": "user", "content": None},
    ]

    out = _normalize_dashboard_message_roles(messages)

    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[0]["content"] == "please describe this image"
    # Copies, not the same dict objects.
    for original, projected in zip(messages, out):
        assert projected is not original


class _FakeDB:
    """Minimal stand-in for SessionDB covering the messages endpoint path."""

    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def resolve_session_id(self, session_id):
        return session_id

    def resolve_resume_session_id(self, session_id):
        return session_id

    def get_messages(self, session_id, limit=None, offset=0):
        rows = self._rows[offset:]
        return rows[:limit] if limit is not None else rows

    def close(self):
        self.closed = True


def test_messages_endpoint_projects_and_counts_stored_rows(monkeypatch):
    block = _success_block("A landscape photo.", "/tmp/land.png")
    stored = [{"role": "user", "content": f"{block}\n\nwhere is this?", "tool_name": None}]
    fake = _FakeDB(stored)
    monkeypatch.setattr(web_server, "_open_session_db_for_profile", lambda profile: fake)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    resp = client.get("/api/sessions/sess-1/messages?limit=10")

    assert resp.status_code == 200
    body = resp.json()
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["tool", "user"]
    assert body["messages"][0]["tool_name"] == "vision_analyze"
    assert body["messages"][1]["content"] == "where is this?"
    # One stored row → `returned` stays 1 even though projection emitted two.
    assert body["pagination"]["returned"] == 1
    assert body["pagination"]["limit"] == 10
    # Endpoint left the stored row untouched.
    assert stored[0]["role"] == "user"
    assert fake.closed is True
