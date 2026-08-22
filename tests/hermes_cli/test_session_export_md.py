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


def test_safe_session_filename_strips_non_ascii_session_id():
    # ``\w`` is Unicode-aware in Python, so a naive ``[^\w-]`` class would let
    # Unicode homoglyphs (e.g. a Cyrillic "а") survive into the filename. The
    # sanitizer must be ASCII-only so only ``[A-Za-z0-9_-]`` pass through.
    filename = safe_session_filename(_session(id="аdmin", title="x"), fmt="md")

    assert "а" not in filename
    # The Cyrillic char collapses to "_" (stripped from the head), the ASCII
    # tail survives, and a disambiguating hash is appended since the id changed.
    assert filename.startswith("dmin_")


def test_write_session_markdown_contains_traversal_id_within_output_dir(tmp_path):
    path = write_session_markdown(_session(id="../pwned", title="x"), tmp_path)

    resolved = path.resolve()
    assert resolved.is_file()
    assert tmp_path.resolve() in resolved.parents






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


