"""Renaming a compressed chat by an ended segment's id renames the chat the listing shows.

A compressed conversation lists under its live compression tip, so `hermes sessions rename`
and the Hermes Console's `sessions rename`, given the id the chat was created with, must write
the tip, as the authority's sidebar rename already does (#152624).
"""
from argparse import Namespace

from hermes_cli import sessions_cmd
from hermes_cli.console_engine import HermesConsoleEngine
from hermes_constants import get_hermes_home
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession, SessionQuery, TitleSource
from persistence_assembly import open_persistence, profile_scope


def _compressed_chat():
    """`chat` compressed into `chat-2`, which carries the automatic title the listing shows."""
    with open_persistence(profile_scope(home=get_hermes_home())) as store:
        store.sessions.create(NewSession(session_id="chat", source="cli"))
        store.sessions.end("chat", END_REASON_COMPRESSION)
        store.sessions.create(NewSession(session_id="chat-2", source="cli", parent_session_id="chat"))
        store.sessions.set_auto_title("chat-2", "Plugin para sessoes", source=TitleSource.LLM)


def _listed_title():
    with open_persistence(profile_scope(home=get_hermes_home())) as store:
        [entry] = [entry for entry in store.sessions.list_sessions(SessionQuery())
                   if "chat" in (entry.lineage_ids or (entry.session.session_id,))]
        return entry.session.title


def test_cli_rename_by_the_original_id_renames_the_listed_chat():
    _compressed_chat()

    rc = sessions_cmd.cmd_sessions(Namespace(sessions_action="rename", session_id="chat", title=["SESSION", "TALK"]))

    assert rc in (None, 0)
    assert _listed_title() == "SESSION TALK"


def test_console_rename_by_the_original_id_renames_the_listed_chat():
    _compressed_chat()

    result = HermesConsoleEngine().execute("sessions rename chat SESSION TALK", confirmed=True)

    assert result.status == "ok", result.output
    assert _listed_title() == "SESSION TALK"
