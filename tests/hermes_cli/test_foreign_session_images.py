import json

import pytest

from hermes_cli.foreign_sessions import (
    gather_foreign_sessions,
    parse_claude_session,
    parse_codex_session,
)


def _write_session(tmp_path, monkeypatch, source, messages):
    root = tmp_path / source
    if source == "codex":
        monkeypatch.setenv("CODEX_HOME", str(root))
        path = root / "sessions" / "2026" / "10" / "26" / "rollout-images.jsonl"
        rows = [{"type": "session_meta", "payload": {"id": "images", "cwd": str(tmp_path)}}]
        for message in messages:
            rows.append({"type": "response_item", "payload": {"type": "message", **message}})
        parser = parse_codex_session
    else:
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(root))
        path = root / "projects" / "project" / "images.jsonl"
        rows = []
        for message in messages:
            rows.append({"type": message["role"], "sessionId": "images", "cwd": str(tmp_path),
                         "message": message})
        parser = parse_claude_session
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return parser(path)


def _image_block(source):
    if source == "codex":
        return {"type": "input_image", "image_url": "data:image/png;base64,dGVzdA=="}
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "dGVzdA=="}}


@pytest.mark.parametrize("source", ["codex", "claude"])
def test_image_only_session_remains_discoverable(tmp_path, monkeypatch, source):
    image = _image_block(source)
    parsed = _write_session(tmp_path, monkeypatch, source, [{"role": "user", "content": [image]}])

    assert parsed["turns"] == [{"role": "user", "content": "[image]"}]
    sessions = gather_foreign_sessions(source)
    assert len(sessions) == 1
    assert sessions[0].session_id == "images"
    assert sessions[0].turn_count == 1


@pytest.mark.parametrize("source", ["codex", "claude"])
def test_image_user_turn_keeps_assistant_replies_separate(tmp_path, monkeypatch, source):
    user_text = "input_text" if source == "codex" else "text"
    assistant_text = "output_text" if source == "codex" else "text"
    messages = [
        {"role": "user", "content": [{"type": user_text, "text": "First question."}]},
        {"role": "assistant", "content": [{"type": assistant_text, "text": "First answer."}]},
        {"role": "user", "content": [_image_block(source)]},
        {"role": "assistant", "content": [{"type": assistant_text, "text": "Image answer."}]},
    ]

    parsed = _write_session(tmp_path, monkeypatch, source, messages)

    assert parsed["turns"] == [
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "[image]"},
        {"role": "assistant", "content": "Image answer."},
    ]
