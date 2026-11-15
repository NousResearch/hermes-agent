"""A one-shot session (``hermes chat -q`` / ``-z`` / ``--oneshot``) gets the instant title but no title model call.

The model upgrade is a second paid request (a second CLI spawn on a process provider such as the Claude
subscription DirectSDK plugin), so a one-shot skips it. The instant title is derived from the query with no
model call, and ``hermes sessions list`` shows one-shot sessions, so a one-shot still gets it (#156048).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_state import SessionDB


class _Agent:
    def __init__(self, db):
        self.platform = "cli"
        self.session_id = "sess-1"
        self.model, self.provider, self.base_url = "m", "openrouter", "https://openrouter.ai/api/v1"
        self.api_key, self.api_mode = "k", "chat_completions"
        self._session_db = db
        self._session_db_created = True


@pytest.mark.parametrize("oneshot", [True, False])
def test_first_turn_titles_the_session_and_only_interactive_calls_the_model(oneshot, tmp_path, monkeypatch):
    from agent import turn_context
    from agent.title_generator import wait_for_title_upgrades

    if oneshot:
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    else:
        monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="sess-1", source="oneshot" if oneshot else "cli")
    with patch("agent.title_generator.generate_title", return_value="Login fix") as model_title:
        turn_context._maybe_title_session_at_turn_start(_Agent(db), [{"role": "user", "content": "Fix the login"}])
        wait_for_title_upgrades()

    assert model_title.called is not oneshot
    if oneshot:
        assert (db.get_session_title("sess-1"), db.get_session_title_source("sess-1")) == ("Fix the login", "derived")
    else:
        assert db.get_session_title("sess-1") == "Login fix"
