"""Read rows back through the storage interface as the dicts tests compare: the session row, the stored message rows
and the OpenAI-format replay.

The other reading operations are plain calls on the handle that return typed records: ``admissions.list``,
``admissions.get``, ``bindings.receipt`` and ``sessions.list_sessions``.
"""
from persistence.conversation.messages import ConversationRequest, Reaction, TranscriptRead


def session_row(store, session_id):
    """The session's row dict (``SessionRecord.as_session_row``), or None when the store has no such session."""
    record = store.sessions.get(session_id)
    return None if record is None else record.as_session_row()


def transcript_rows(store, session_id, request=TranscriptRead()):
    """The session's stored message rows as dicts (``TranscriptRow.as_wire_row``), in transcript order."""
    return [row.as_wire_row() for row in store.messages.read_transcript(session_id, request)]


def conversation(store, session_id, request=ConversationRequest()):
    """The session's replay messages as OpenAI-format dicts (``ConversationMessage.as_openai_message``)."""
    return [message.as_openai_message() for message in store.messages.read_conversation(session_id, request)]


def message_reactions(store, session_id, message_id):
    """The reactions stored on row *message_id* of the session's display resume (the rows ``messages.set_reaction``
    accepts), as the ``Reaction`` records that verb returns; empty for any other row."""
    for message in store.messages.read_resume_conversations(session_id).display:
        if message.row_id == message_id:
            return tuple(Reaction(**reaction) for reaction in (message.display_metadata or {}).get("reactions", ()))
    return ()
