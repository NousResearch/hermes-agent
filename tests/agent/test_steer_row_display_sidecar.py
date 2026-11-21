"""A stored mid-turn steer shows the user's own words, even when its ``content`` holds the turn's prompt.

Rows written while the current-turn anchor sat on a steer (#132165) carry the opening prompt in ``content``
and the steer marker only in the ``api_content`` sidecar. The Desktop projection and the TUI history read the
steer text from the sidecar then, so those rows read right without a migration.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agent.prompt_builder import format_steer_marker
from hermes_cli.web_routers.sessions import _project_for_display
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import conversation, transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session

PROMPT = "Continue the work on the two open epics from the kanban boards"
STEER = "reassign all the cards on the first board to the second agent"


@pytest.fixture()
def db():
    tmp = tempfile.mkdtemp(prefix="steer_sidecar_")
    store = open_persistence(profile_scope(home=Path(tmp)))
    try:
        yield store
    finally:
        store.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _seed_overwritten_steer(db, session_id):
    seed_session(db, session_id, "desktop", model="test-model")
    seed_messages(
        db, session_id,
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": "on it"},
        {"role": "user", "content": PROMPT, "display_kind": "steer",
         "api_content": format_steer_marker(STEER).lstrip()},
    )


def test_desktop_projection_shows_the_steer_text_from_the_sidecar(db):
    _seed_overwritten_steer(db, "stored")

    projected = _project_for_display(transcript_rows(db, "stored"))

    assert projected[2]["display_content"] == STEER


def test_tui_history_shows_the_steer_text_from_the_sidecar(db):
    from tui_gateway import server  # session_history's bodies are bound onto server's globals

    _seed_overwritten_steer(db, "stored-tui")

    shown = json.dumps(server._history_to_messages(conversation(db, "stored-tui")))

    assert STEER in shown
    assert shown.count(PROMPT) == 1


def test_a_steer_row_with_its_marker_in_content_is_unchanged(db):
    seed_session(db, "intact", "desktop", model="test-model")
    seed_messages(
        db, "intact",
        {"role": "user", "content": PROMPT},
        {"role": "user", "content": format_steer_marker(STEER).lstrip(), "display_kind": "steer"},
    )

    assert _project_for_display(transcript_rows(db, "intact"))[1]["display_content"] == STEER
