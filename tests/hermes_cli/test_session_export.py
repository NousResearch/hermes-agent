import json
import os
import sys

from hermes_cli.session_export import export_record_count, render_sessions_export
from hermes_cli.session_export_html import (
    _generate_messages_html,
    generate_multi_session_html_export,
)


def _sample_session():
    return {
        "id": "sess-123",
        "source": "cli",
        "model": "test/model",
        "title": "Debug auth flow",
        "started_at": 1700000000,
        "message_count": 5,
        "messages": [
            {
                "id": 1,
                "role": "system",
                "content": "hidden system context",
                "timestamp": 1700000000,
            },
            {
                "id": 2,
                "role": "user",
                "content": "Why is login broken?",
                "timestamp": 1700000001,
                "platform_message_id": "evt-2",
            },
            {
                "id": 3,
                "role": "assistant",
                "content": "I will inspect the auth middleware.",
                "timestamp": 1700000002,
            },
            {
                "id": 4,
                "role": "tool",
                "tool_name": "read_file",
                "content": "def redirect_after_login(): pass",
                "timestamp": 1700000003,
            },
            {
                "id": 5,
                "role": "user",
                "content": [{"type": "text", "text": "Only show me the prompts."}],
                "timestamp": 1700000004,
            },
        ],
    }










def test_html_export_escapes_tool_call_names():
    payload = '<img src=x onerror="alert(document.domain)">'

    rendered = _generate_messages_html(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": payload, "arguments": "<b>x</b>"},
                    }
                ],
            }
        ]
    )

    assert payload not in rendered
    assert '&lt;img src=x onerror=&quot;alert(document.domain)&quot;&gt;' in rendered
    assert "&lt;b&gt;x&lt;/b&gt;" in rendered




def test_export_record_count_switches_unit_for_prompt_only_exports():
    assert export_record_count([_sample_session()]) == (1, "session")
    assert export_record_count([_sample_session()], only="user-prompts") == (
        2,
        "prompt",
    )


def test_sessions_export_cli_prompt_only_stdout(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"] = session_id
            return "sess-123"

        def export_session(self, session_id):
            captured["exported"] = session_id
            return _sample_session()

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "export", "-", "--session-id", "sess", "--only", "user-prompts"],
    )

    main_mod.main()

    output = capsys.readouterr().out
    records = [json.loads(line) for line in output.splitlines()]
    assert [record["text"] for record in records] == [
        "Why is login broken?",
        "Only show me the prompts.",
    ]
    assert captured == {
        "resolved_from": "sess",
        "exported": "sess-123",
        "closed": True,
    }


# ─── `sessions export <directory>` ────────────────────────────────────────
#
# `hermes sessions export --help` documents the positional as an output
# *directory* for md/qmd, and the bulk md/qmd branch honours that. Every
# single-file writer in `cmd_sessions` instead called `open(args.output, "w")`
# on the raw positional, so handing it a directory raised an uncaught
# `IsADirectoryError` — for `--only` and `--format html` only after the whole
# export had already been rendered. A missing parent directory raised an
# equally uncaught `FileNotFoundError` at the same lines.


class _FakeSessionDB:
    """Minimal ``SessionDB`` stand-in covering the export command's reads."""

    def __init__(self):
        self.closed = False

    def resolve_session_id(self, session_id):
        return "sess-123"

    def export_session(self, session_id):
        return _sample_session()

    def export_all(self, source=None):
        return [_sample_session()]

    def get_session(self, session_id):
        return {"id": session_id, "model": "test/model"}

    def get_messages_as_conversation(self, session_id):
        return [
            {"role": "user", "content": "Why is login broken?"},
            {"role": "assistant", "content": "I will inspect the auth middleware."},
        ]

    def close(self):
        self.closed = True


def _run_sessions_export(monkeypatch, argv):
    """Drive ``hermes sessions export`` end-to-end against a stub database."""
    import hermes_cli.main as main_mod
    import hermes_state

    db = _FakeSessionDB()
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: db)
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "export"] + argv)
    main_mod.main()
    return db


def test_prompt_only_md_export_into_directory(monkeypatch, capsys, tmp_path):
    """``--format md --only user-prompts <dir>`` is the documented invocation.

    The positional is documented as a directory for md and ``--only`` is
    documented as supported for md, yet the same command line only worked
    once ``--only`` was dropped.
    """
    _run_sessions_export(
        monkeypatch,
        [
            str(tmp_path),
            "--format",
            "md",
            "--only",
            "user-prompts",
            "--session-id",
            "sess",
        ],
    )

    written = tmp_path / "user-prompts.md"
    assert written.is_file()
    assert "Why is login broken?" in written.read_text(encoding="utf-8")
    assert str(written) in capsys.readouterr().out


def test_prompt_only_jsonl_export_into_directory(monkeypatch, capsys, tmp_path):
    _run_sessions_export(
        monkeypatch,
        [
            str(tmp_path),
            "--format",
            "jsonl",
            "--only",
            "user-prompts",
            "--session-id",
            "sess",
        ],
    )

    written = tmp_path / "user-prompts.jsonl"
    assert written.is_file()
    records = [
        json.loads(line) for line in written.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["text"] for record in records] == [
        "Why is login broken?",
        "Only show me the prompts.",
    ]
    assert str(written) in capsys.readouterr().out


def test_html_export_into_directory(monkeypatch, capsys, tmp_path):
    _run_sessions_export(
        monkeypatch,
        [str(tmp_path), "--format", "html", "--session-id", "sess"],
    )

    written = tmp_path / "sessions.html"
    assert written.is_file()
    assert "<html" in written.read_text(encoding="utf-8").lower()
    assert str(written) in capsys.readouterr().out


def test_trace_single_session_export_into_directory(monkeypatch, capsys, tmp_path):
    """``--format trace`` accepted a directory for many sessions but not one.

    Five lines apart in the same function, the multi-session trace branch
    does ``Path(args.output)`` + ``mkdir(parents=True, exist_ok=True)`` while
    the single-session branch opened the same positional as a file.
    """
    _run_sessions_export(
        monkeypatch,
        [str(tmp_path), "--format", "trace", "--session-id", "sess"],
    )

    written = tmp_path / "sess-123.trace.jsonl"
    assert written.is_file()
    assert written.read_text(encoding="utf-8").strip()
    assert str(written) in capsys.readouterr().out


def test_jsonl_session_export_into_directory(monkeypatch, capsys, tmp_path):
    _run_sessions_export(
        monkeypatch,
        [str(tmp_path), "--format", "jsonl", "--session-id", "sess"],
    )

    written = tmp_path / "sess-123.jsonl"
    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8"))["id"] == "sess-123"
    assert str(written) in capsys.readouterr().out


def test_jsonl_bulk_export_into_directory(monkeypatch, capsys, tmp_path):
    _run_sessions_export(monkeypatch, [str(tmp_path), "--format", "jsonl"])

    written = tmp_path / "sessions.jsonl"
    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8"))["id"] == "sess-123"
    assert str(written) in capsys.readouterr().out


def test_export_creates_missing_parent_directory(monkeypatch, tmp_path):
    """A not-yet-existing parent raised a bare ``FileNotFoundError``."""
    target = tmp_path / "nested" / "deep" / "prompts.md"

    _run_sessions_export(
        monkeypatch,
        [
            str(target),
            "--format",
            "md",
            "--only",
            "user-prompts",
            "--session-id",
            "sess",
        ],
    )

    assert target.is_file()
    assert "Why is login broken?" in target.read_text(encoding="utf-8")


def test_export_trailing_separator_creates_new_directory(monkeypatch, capsys, tmp_path):
    """A trailing separator asks for a directory that need not exist yet.

    The multi-session trace and bulk md/qmd branches ``mkdir`` a
    not-yet-existing output directory, so the single-session writers need a
    way to say "directory" about a path that is not one on disk. ``open()``
    rejects a trailing-separator path outright, so honouring it here cannot
    change the meaning of any invocation that works today.
    """
    target_dir = tmp_path / "new-exports"

    _run_sessions_export(
        monkeypatch,
        [str(target_dir) + os.sep, "--format", "trace", "--session-id", "sess"],
    )

    written = target_dir / "sess-123.trace.jsonl"
    assert target_dir.is_dir()
    assert written.is_file()
    assert str(written) in capsys.readouterr().out


def test_export_bare_nonexistent_path_stays_a_file(monkeypatch, tmp_path):
    """Boundary guard: a bare non-existent path is still a file name.

    ``sessions export ~/my-export --format jsonl --session-id X`` writes a
    file called ``my-export`` today, so a path that is neither an existing
    directory nor separator-terminated must not be silently reinterpreted.
    """
    target = tmp_path / "my-export"

    _run_sessions_export(
        monkeypatch,
        [str(target), "--format", "jsonl", "--session-id", "sess"],
    )

    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8"))["id"] == "sess-123"


def test_prompt_only_md_stdout_dash_unchanged(monkeypatch, capsys):
    """Behaviour-preservation guard: ``-`` still streams to stdout."""
    _run_sessions_export(
        monkeypatch,
        ["-", "--format", "md", "--only", "user-prompts", "--session-id", "sess"],
    )

    out = capsys.readouterr().out
    assert "Why is login broken?" in out
    assert "Exported" not in out


def test_prompt_only_md_plain_file_path_unchanged(monkeypatch, capsys, tmp_path):
    """Behaviour-preservation guard: a file path still writes that exact file."""
    target = tmp_path / "prompts.md"

    _run_sessions_export(
        monkeypatch,
        [
            str(target),
            "--format",
            "md",
            "--only",
            "user-prompts",
            "--session-id",
            "sess",
        ],
    )

    assert target.is_file()
    assert "Why is login broken?" in target.read_text(encoding="utf-8")
    assert not (tmp_path / "user-prompts.md").exists()
    assert str(target) in capsys.readouterr().out


