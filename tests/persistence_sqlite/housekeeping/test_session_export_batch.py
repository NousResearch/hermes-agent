from hermes_cli.session_export import to_legacy_export_dict
from persistence.housekeeping.maintenance import SessionExport
from persistence.sqlite.conversation._store.messages import append_messages_batch, get_messages
from persistence.sqlite.conversation._store.sessions import create_session, search_sessions
from persistence.sqlite.housekeeping._portability import _export_row, _transcript_row
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder


def test_export_all_batches_message_reads_without_changing_export_rows(tmp_path, monkeypatch):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        db = raw_holder(store)
        for index, source in enumerate(("cli", "telegram", "cli", "cli")):
            session_id = f"session-{index}"
            create_session(db, session_id=session_id, source=source)
            append_messages_batch(
                db,
                session_id,
                [
                    {"role": "user", "content": f"question {index}"},
                    {
                        "role": "assistant",
                        "content": f"answer {index}",
                        "tool_calls": [{"id": f"call-{index}", "type": "function"}],
                    },
                ],
            )

        sessions = search_sessions(db, source="cli", limit=100000)
        expected = [
            SessionExport(session=_export_row(session),
                          messages=tuple(map(_transcript_row, get_messages(db, session["id"]))))
            for session in sessions
        ]

        original_read_all = db._read_all
        read_calls = 0

        def counted_read_all(*args, **kwargs):
            nonlocal read_calls
            read_calls += 1
            return original_read_all(*args, **kwargs)

        monkeypatch.setattr(db, "_read_all", counted_read_all)

        exported = store.maintenance.export_all(source="cli")
        assert list(exported) == expected
        # The export dict carries a derived `timings` block on top of the session + messages.
        assert all("timings" in to_legacy_export_dict(row) for row in exported)
        assert read_calls <= 2
