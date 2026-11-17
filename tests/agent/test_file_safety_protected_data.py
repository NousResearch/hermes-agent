"""Tests for the cross-user session/memory data-store matcher."""
from pathlib import Path

from agent.file_safety import is_protected_data_path, get_read_block_error
from hermes_constants import get_hermes_home


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")
    return p


def test_state_db_is_protected():
    p = _touch(get_hermes_home() / "state.db")
    assert is_protected_data_path(str(p)) is not None


def test_memory_store_db_is_protected():
    p = _touch(get_hermes_home() / "memory_store.db")
    assert is_protected_data_path(str(p)) is not None


def test_holographic_scope_db_is_protected():
    p = _touch(get_hermes_home() / "memories" / "holographic" / "U123.db")
    assert is_protected_data_path(str(p)) is not None


def test_session_json_snapshot_is_protected():
    p = _touch(get_hermes_home() / "sessions" / "session_abc.json")
    assert is_protected_data_path(str(p)) is not None


def test_session_jsonl_is_protected():
    p = _touch(get_hermes_home() / "sessions" / "abc.jsonl")
    assert is_protected_data_path(str(p)) is not None


def test_request_dump_is_protected():
    p = _touch(get_hermes_home() / "sessions" / "request_dump_abc_1.json")
    assert is_protected_data_path(str(p)) is not None


def test_global_memory_markdown_is_not_protected():
    p = _touch(get_hermes_home() / "memories" / "MEMORY.md")
    assert is_protected_data_path(str(p)) is None
    p2 = _touch(get_hermes_home() / "memories" / "USER.md")
    assert is_protected_data_path(str(p2)) is None


def test_arbitrary_project_db_is_not_protected(tmp_path):
    # A user's own project file named state.db, NOT under HERMES_HOME.
    p = _touch(tmp_path / "myproject" / "state.db")
    assert is_protected_data_path(str(p)) is None


def test_arbitrary_sessions_dir_outside_hermes_is_not_protected(tmp_path):
    p = _touch(tmp_path / "proj" / "sessions" / "session_x.json")
    assert is_protected_data_path(str(p)) is None


def test_sibling_profile_state_db_is_protected():
    root = get_hermes_home()  # in tests HERMES_HOME is the root
    p = _touch(root / "profiles" / "other" / "state.db")
    assert is_protected_data_path(str(p)) is not None


def test_read_block_error_covers_protected_db():
    p = _touch(get_hermes_home() / "state.db")
    msg = get_read_block_error(str(p))
    assert msg is not None
    assert "security boundary" in msg.lower()


def test_autonomy_state_is_protected():
    for rel in ("autonomy/.hmac_key", "autonomy/pending/zendesk_create_draft/0123456789ab.json",
                "autonomy/ledger/zendesk_create_draft/abc.json", "autonomy/locks/x/y.lock",
                "autonomy"):
        p = get_hermes_home() / rel
        if rel != "autonomy":
            _touch(p)
        reason = is_protected_data_path(str(p))
        assert reason is not None and "autonomy" in reason, rel


def test_autonomy_dir_outside_hermes_is_not_protected(tmp_path):
    p = _touch(tmp_path / "proj" / "autonomy" / "notes.md")
    assert is_protected_data_path(str(p)) is None


def test_file_tools_refuse_autonomy_state():
    import json
    from tools.file_tools import read_file_tool, write_file_tool
    p = _touch(get_hermes_home() / "autonomy" / "pending" / "zendesk_create_draft" / "0123456789ab.json")
    assert "autonomy" in json.loads(read_file_tool(str(p)))["error"]
    out = json.loads(write_file_tool(str(p), "{}"))
    assert "error" in out
    assert p.read_text(encoding="utf-8") == "x"
