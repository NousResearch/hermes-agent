from pathlib import Path

import pytest

from hermes_cli.session_export_md import (
    append_manifest_entry,
    render_session_markdown,
    safe_session_filename,
    verify_export_file,
    write_session_markdown,
)


def _session(**overrides):
    data = {
        "id": "20260706_123456_abcd1234",
        "title": "Export Test",
        "source": "telegram",
        "model": "gpt-5.5",
        "billing_provider": "openai-codex",
        "cwd": "/tmp/project",
        "started_at": 1783331696.0,
        "last_active": 1783331705.0,
        "ended_at": 1783331710.0,
        "message_count": 3,
        "tool_call_count": 1,
        "archived": 0,
        "messages": [
            {"role": "user", "content": "Hello", "created_at": 1783331697.0},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "terminal", "arguments": "{\"command\": \"pwd\"}"}}
                ],
                "created_at": 1783331698.0,
            },
            {"role": "tool", "name": "terminal", "content": "output", "created_at": 1783331699.0},
        ],
    }
    data.update(overrides)
    return data




def test_safe_session_filename_is_deterministic_and_path_safe():
    filename = safe_session_filename(
        _session(id="20260706_123456_abcd1234", title="Bad / title: * ?"), fmt="qmd"
    )

    assert filename.startswith("20260706_123456_abcd1234-")
    assert filename.endswith(".qmd")
    assert "/" not in filename
    assert ":" not in filename
    assert "*" not in filename
    assert "?" not in filename






def test_verify_export_file_checks_count_and_sha(tmp_path):
    session = _session()
    path = write_session_markdown(session, tmp_path)

    ok, reason = verify_export_file(path, session)
    assert ok is True
    assert reason == "ok"

    path.write_text(path.read_text(encoding="utf-8").replace("Hello", "Tampered"), encoding="utf-8")
    ok, reason = verify_export_file(path, session)
    assert ok is False
    assert "sha256" in reason


def _self_referential_session(content: str) -> dict:
    """A session whose own messages contain a marker line the exporter also writes."""
    return _session(
        messages=[
            {"role": "user", "content": content, "created_at": 1783331697.0},
            {"role": "assistant", "content": "noted", "created_at": 1783331698.0},
        ],
        message_count=2,
    )


def test_export_quoting_the_placeholder_token_still_verifies(tmp_path):
    """Mechanism A of issue #157707: the in-body token must not be rewritten with the digest."""
    session = _self_referential_session("what is `__SHA256_PLACEHOLDER__` for?")
    path = write_session_markdown(session, tmp_path)

    assert verify_export_file(path, session) == (True, "ok")
    assert "`__SHA256_PLACEHOLDER__`" in path.read_text(encoding="utf-8")


def test_export_quoting_an_earlier_footer_still_verifies(tmp_path):
    """Mechanism B: a quoted SHA line inside the messages is not the marker to recompute from."""
    quoted = f"- SHA256 of exported body: `{'0' * 64}`"
    session = _self_referential_session(quoted)
    path = write_session_markdown(session, tmp_path)

    assert verify_export_file(path, session) == (True, "ok")
    assert path.read_text(encoding="utf-8").count(quoted) == 1


def test_self_referential_export_still_detects_tampering(tmp_path):
    session = _self_referential_session("what is `__SHA256_PLACEHOLDER__` for?")
    path = write_session_markdown(session, tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("noted", "Tampered"), encoding="utf-8")

    ok, reason = verify_export_file(path, session)
    assert ok is False
    assert "sha256" in reason
