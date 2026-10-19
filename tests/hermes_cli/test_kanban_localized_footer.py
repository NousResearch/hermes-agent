"""A localized CLI exit footer never replaces the current invocation's failure output."""
from datetime import datetime
from types import SimpleNamespace
import contextlib
import io

import pytest
import cli  # Bootstrap at collection, like the existing CLI test modules.
from agent import i18n
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli.cli_session_mixin import CLISessionMixin


@pytest.mark.parametrize("language", ["en", "de", "zh", "ja"])
@pytest.mark.parametrize("header", [False, True])
@pytest.mark.parametrize("trailing_crash", [False, True])
def test_real_localized_footer_attribution(tmp_path, monkeypatch, language, header, trailing_crash):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"display:\n  language: {language}\n")
    i18n.reset_language_cache()
    try:
        session = SimpleNamespace(conversation_history=[{"role": "user", "content": "fixture"}],
                                  session_start=datetime.now(), _session_db=None, session_id="fixture")
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            CLISessionMixin._print_exit_summary(session, clear_screen=False)
        prefix = "=== HERMES_KANBAN_RUN task=t_fixture run=2 started_at=1 ===\n" if header else "Query: current\n"
        text = prefix + "CURRENT FAILURE: missing artifact\n" + capture.getvalue()
        if trailing_crash:
            text += "Unknown skill(s): sdlc-review\n"
        monkeypatch.setattr(dispatch._kb, "read_worker_log", lambda *a, **kw: text)
        result = dispatch._worker_final_output("t_fixture")
        if trailing_crash:
            assert "Unknown skill(s): sdlc-review" in result
            assert "CURRENT FAILURE" not in result
        else:
            assert result == "CURRENT FAILURE: missing artifact"
    finally:
        i18n.reset_language_cache()
