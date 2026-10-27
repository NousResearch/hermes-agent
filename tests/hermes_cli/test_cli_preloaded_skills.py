"""Regression tests for --skills preload finalization in HermesCLI (#54088).

``finalize_preloaded_skills`` joins the background preload thread and folds the
result into the session. A fully-missing skill set still fails loudly for
direct CLI runs, but a dispatcher-owned kanban worker (``HERMES_KANBAN_DB``
pinned) must warn and keep running instead of stranding its card.
"""

import logging
import threading

import pytest

from cli import HermesCLI


def _worker_cli(result):
    """A bare HermesCLI whose preload thread already finished with *result*."""
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    cli = HermesCLI.__new__(HermesCLI)
    cli._preload_skills_thread = thread
    cli._preload_skills_error = None
    cli._auto_load_skills_result = None
    cli._preload_skills_result = result
    cli.preloaded_skills = []
    cli.system_prompt = "base prompt"
    return cli


class TestFinalizePreloadedSkills:
    def test_all_missing_kanban_worker_warns_and_continues(self, monkeypatch, caplog):
        monkeypatch.setenv("HERMES_KANBAN_DB", "/tmp/fake-kanban.db")
        cli = _worker_cli(("", [], ["missing-skill"]))

        with caplog.at_level(logging.WARNING):
            cli.finalize_preloaded_skills()

        assert "continuing without preloaded skills" in caplog.text
        assert cli.preloaded_skills == []
        assert cli.system_prompt == "base prompt"

    def test_all_missing_direct_cli_still_raises(self, monkeypatch, caplog):
        monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
        cli = _worker_cli(("", [], ["missing-skill"]))

        with pytest.raises(ValueError, match="Unknown skill"):
            cli.finalize_preloaded_skills()

    def test_mixed_kanban_worker_continues_with_loaded(self, monkeypatch, caplog):
        monkeypatch.setenv("HERMES_KANBAN_DB", "/tmp/fake-kanban.db")
        cli = _worker_cli(("loaded prompt", ["good-skill"], ["missing-skill"]))

        with caplog.at_level(logging.WARNING):
            cli.finalize_preloaded_skills()

        assert "Continuing with: good-skill" in caplog.text
        assert cli.preloaded_skills == ["good-skill"]
        assert cli.system_prompt == "base prompt\n\nloaded prompt"
