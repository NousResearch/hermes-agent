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


def test_render_session_markdown_includes_frontmatter_messages_and_verification():
    rendered = render_session_markdown(_session())

    assert rendered.startswith("---\n")
    assert 'session_id: "20260706_123456_abcd1234"' in rendered
    assert 'title: "Export Test"' in rendered
    assert 'source: "telegram"' in rendered
    assert 'model: "gpt-5.5"' in rendered
    assert 'provider: "openai-codex"' in rendered
    assert "# Export Test" in rendered
    assert "## Messages" in rendered
    assert "### User" in rendered
    assert "Hello" in rendered
    assert "### Assistant" in rendered
    assert "## Tool calls" in rendered
    assert '"name": "terminal"' in rendered
    assert "### Tool — terminal" in rendered
    assert "output" in rendered
    assert "## Export verification" in rendered
    assert "Exported messages: `3`" in rendered
    assert "SHA256 of exported body:" in rendered


def test_render_session_markdown_renders_structured_content_as_json_fence():
    rendered = render_session_markdown(
        _session(messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    )

    assert "```json" in rendered
    assert '"type": "text"' in rendered


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


def test_safe_session_filename_sanitizes_traversal_id():
    # Session ids can come from the untrusted X-Hermes-Session-Id header and
    # are interpolated raw into the export filename; a traversal-shaped id must
    # collapse to a single path-free segment.
    filename = safe_session_filename(
        _session(id="../../../../tmp/pwned", title="x"), fmt="md"
    )

    assert "/" not in filename
    assert ".." not in filename
    assert filename.endswith(".md")


def test_write_session_markdown_contains_traversal_id_within_output_dir(tmp_path):
    path = write_session_markdown(_session(id="../pwned", title="x"), tmp_path)

    resolved = path.resolve()
    assert resolved != tmp_path.resolve()
    assert tmp_path.resolve() in resolved.parents


def test_render_session_markdown_includes_logical_lineage_segments():
    rendered = render_session_markdown(
        _session(
            id="tip",
            title="Logical",
            lineage_session_ids=["root", "tip"],
            segments=[
                _session(id="root", messages=[{"role": "user", "content": "root text"}]),
                _session(id="tip", messages=[{"role": "assistant", "content": "tip text"}]),
            ],
        )
    )

    assert 'lineage_session_ids: ["root", "tip"]' in rendered
    assert "## Compression segment: root" in rendered
    assert "root text" in rendered
    assert "## Compression segment: tip" in rendered
    assert "tip text" in rendered
    assert "Exported messages: `2`" in rendered


def test_write_session_markdown_refuses_to_overwrite_without_force(tmp_path):
    session = _session()
    first = write_session_markdown(session, tmp_path)

    assert first.exists()
    with pytest.raises(FileExistsError):
        write_session_markdown(session, tmp_path)

    second = write_session_markdown(session, tmp_path, force=True)
    assert second == first


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


def test_append_manifest_entry_writes_jsonl_with_sha(tmp_path):
    session = _session()
    path = write_session_markdown(session, tmp_path)
    manifest = append_manifest_entry(tmp_path, session, path, fmt="md")

    text = manifest.read_text(encoding="utf-8")
    assert '"session_id": "20260706_123456_abcd1234"' in text
    assert '"path":' in text
    assert '"sha256":' in text
