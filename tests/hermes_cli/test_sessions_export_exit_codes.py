"""Regression tests: `hermes sessions export` error paths return non-zero.

``aca40d1d63`` fixed delete/rename/prune/import so scripting/CI could detect
a failure (SES-04/SES-10) — every one of those actions' error paths now
``return 1`` instead of a bare ``return`` (exit 0). The ``export`` action was
left untouched: every one of its many error paths (session-not-found, a bad
``build_prune_filters`` argument — the exact same try/except shape fixed for
prune/archive, ``--only``/format mismatches, a missing ``--output`` for
HTML/JSONL, a missing/empty trace session, refusing a bulk export without a
filter, etc.) still printed an error and returned exit 0. A script doing
``hermes sessions export --session-id bad_id ... || retry`` could not detect
the failure.

These tests cover a representative sample across every export format
(jsonl/md/html/trace) plus the one subtlety in the fix: a ``--dry-run``
*preview* (session list printed, nothing exported, by design) must NOT
return 1 — only a genuine error should. ``_collect_sessions()``'s internal
``_collect_error`` flag is what tells the two callers (``--only`` and
``--format html``) which of the two a ``None`` result means.
"""

from argparse import Namespace

import hermes_cli.sessions_cmd as sc
from hermes_state import SessionDB


def _export_args(**kw):
    base = dict(
        sessions_action="export",
        output=None,
        format="jsonl",
        upload=False,
        public=False,
        no_redact=False,
        only=None,
        session_id=None,
        redact=False,
        lineage="single",
        delete_after_verified=False,
        force=False,
        dry_run=False,
        yes=False,
        # Filter args recognized by _any_filters / build_prune_filters.
        older_than=None, newer_than=None, before=None, after=None,
        source=None, title=None, end_reason=None, cwd=None,
        min_messages=None, max_messages=None, model=None, provider=None,
        user=None, chat_id=None, chat_type=None, branch=None,
        min_tokens=None, max_tokens=None, min_cost=None, max_cost=None,
        min_tool_calls=None, max_tool_calls=None,
    )
    base.update(kw)
    return Namespace(**base)


def _init_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    SessionDB(tmp_path / "state.db")  # initialize an empty store


class TestExportErrorReturns1:
    def test_bad_filter_returns_1(self, tmp_path, monkeypatch, capsys):
        """Same try/except shape aca40d1d63 fixed for prune/archive."""
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(
            _export_args(output=str(tmp_path / "out.jsonl"), older_than="not-a-real-duration")
        )
        assert rc == 1
        assert "error" in capsys.readouterr().out.lower()

    def test_jsonl_missing_output_returns_1(self, tmp_path, monkeypatch):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(_export_args(format="jsonl", output=None))
        assert rc == 1

    def test_jsonl_session_not_found_returns_1(self, tmp_path, monkeypatch, capsys):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(
            _export_args(format="jsonl", output=str(tmp_path / "out.jsonl"), session_id="nope_xyz")
        )
        assert rc == 1
        assert "not found" in capsys.readouterr().out.lower()

    def test_only_bad_format_returns_1(self, tmp_path, monkeypatch):
        """--only user-prompts requires --format jsonl or md."""
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(_export_args(format="html", only="user-prompts", output=str(tmp_path / "out.html")))
        assert rc == 1

    def test_html_missing_output_returns_1(self, tmp_path, monkeypatch):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(_export_args(format="html", output=None))
        assert rc == 1

    def test_html_session_not_found_returns_1(self, tmp_path, monkeypatch, capsys):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(
            _export_args(format="html", output=str(tmp_path / "out.html"), session_id="nope_xyz")
        )
        assert rc == 1
        assert "not found" in capsys.readouterr().out.lower()

    def test_trace_no_session_found_returns_1(self, tmp_path, monkeypatch, capsys):
        """No --session-id, no filters, and no sessions exist at all."""
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(_export_args(format="trace"))
        assert rc == 1
        assert "no session found" in capsys.readouterr().out.lower()

    def test_trace_session_not_found_returns_1(self, tmp_path, monkeypatch, capsys):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(_export_args(format="trace", session_id="nope_xyz"))
        assert rc == 1
        assert "not found" in capsys.readouterr().out.lower()

    def test_trace_upload_missing_session_id_returns_1(self, tmp_path, monkeypatch, capsys):
        """A filter (not a session-id) is set, so the "no session found"
        auto-resolve-to-most-recent shortcut is skipped and --upload's own
        "needs exactly one session" check is reached."""
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(_export_args(format="trace", upload=True, older_than="1d"))
        assert rc == 1
        assert "--upload exports one session" in capsys.readouterr().out

    def test_md_stdout_not_supported_returns_1(self, tmp_path, monkeypatch):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(_export_args(format="md", output="-"))
        assert rc == 1

    def test_md_bulk_without_filter_returns_1(self, tmp_path, monkeypatch, capsys):
        """Refusing a bulk export without --session-id or a filter."""
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(_export_args(format="md", output=str(tmp_path / "out")))
        assert rc == 1
        assert "refusing bulk export" in capsys.readouterr().out.lower()

    def test_md_session_not_found_returns_1(self, tmp_path, monkeypatch, capsys):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(
            _export_args(format="md", output=str(tmp_path / "out"), session_id="nope_xyz")
        )
        assert rc == 1
        assert "not found" in capsys.readouterr().out.lower()

    def test_delete_after_verified_requires_yes_returns_1(self, tmp_path, monkeypatch):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(
            _export_args(
                format="md", output=str(tmp_path / "out"),
                session_id="whatever", delete_after_verified=True, yes=False,
            )
        )
        assert rc == 1

    def test_delete_after_verified_requires_session_id_returns_1(self, tmp_path, monkeypatch):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(
            _export_args(
                format="md", output=str(tmp_path / "out"),
                delete_after_verified=True, yes=True, older_than="1d",
            )
        )
        assert rc == 1

    def test_jsonl_dry_run_without_filter_returns_1(self, tmp_path, monkeypatch, capsys):
        """--dry-run with no --session-id and no filter is a usage error,
        not a preview — must not silently exit 0."""
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(_export_args(format="jsonl", output=str(tmp_path / "out.jsonl"), dry_run=True))
        assert rc == 1
        assert "requires at least one filter" in capsys.readouterr().out.lower()


class TestExportDryRunPreviewIsNotAnError:
    """A --dry-run preview with a real filter prints and exits 0 — it did
    exactly what was asked, unlike the usage-error case above. This is the
    one place _collect_sessions()'s None-but-not-an-error distinction
    (_collect_error) has to get right."""

    def test_only_dry_run_with_filter_does_not_return_1(self, tmp_path, monkeypatch, capsys):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(
            _export_args(format="jsonl", only="user-prompts", dry_run=True, older_than="1d")
        )
        assert rc != 1
        assert "would export" in capsys.readouterr().out.lower()

    def test_html_dry_run_with_filter_does_not_return_1(self, tmp_path, monkeypatch, capsys):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(
            _export_args(format="html", output=str(tmp_path / "out.html"), dry_run=True, older_than="1d")
        )
        assert rc != 1
        assert "would export" in capsys.readouterr().out.lower()

    def test_jsonl_dry_run_with_filter_does_not_return_1(self, tmp_path, monkeypatch, capsys):
        _init_store(tmp_path, monkeypatch)
        rc = sc.cmd_sessions(
            _export_args(format="jsonl", output=str(tmp_path / "out.jsonl"), dry_run=True, older_than="1d")
        )
        assert rc != 1
        assert "would export" in capsys.readouterr().out.lower()
