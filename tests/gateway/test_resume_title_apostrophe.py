"""An apostrophe in a session title must not break the gateway's ``/resume`` and ``/sessions``.

Both parsed their argument with ``shlex.split``, which reads the ``'`` of ``Mom's birthday plan`` as
an unclosed quote: the reply was a parse error (with a hint about spaces, which were never the
problem) instead of the session. The classic CLI's ``/resume`` takes the raw text and resumes it.
"""

import pytest

from hermes_cli.session_listing import parse_session_listing_args, split_session_args
from tests.gateway.test_resume_command import _make_event, _make_runner, _session_key_for_event


@pytest.fixture()
def db(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    lane = _session_key_for_event(_make_event())
    db.create_session("moms_plan", "telegram", session_key=lane, user_id="12345", chat_id="67890")
    db.set_session_title("moms_plan", "Mom's birthday plan")
    db.create_session("current_session_001", "telegram", session_key=lane, user_id="12345", chat_id="67890")
    yield db
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [
    "/resume Mom's birthday plan",
    "/resume \"Mom's birthday plan\"",
    "/sessions Mom's birthday plan",
])
async def test_title_with_an_apostrophe_resumes(db, command):
    event = _make_event(text=command)
    runner = _make_runner(session_db=db, event=event)

    handler = runner._handle_sessions_command if command.startswith("/sessions") else runner._handle_resume_command
    reply = await handler(event)

    assert "Could not parse" not in reply
    assert runner.session_store.switch_session.call_args.args[1] == "moms_plan"


@pytest.mark.asyncio
async def test_sessions_search_with_an_apostrophe_finds_the_title(db):
    event = _make_event(text="/sessions search Mom's")
    runner = _make_runner(session_db=db, event=event)

    reply = await runner._handle_sessions_command(event)

    assert "Mom's birthday plan" in reply
    assert "moms_plan" in reply


def test_quotes_still_group_words_and_flags_still_parse():
    assert split_session_args('--all "Project A Plan"') == ["--all", "Project A Plan"]
    assert split_session_args("Mom's birthday plan --all") == ["Mom's", "birthday", "plan", "--all"]
    assert parse_session_listing_args("all search don't") == (True, False, "", "don't")
