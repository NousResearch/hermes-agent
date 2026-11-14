"""A cron delivery into a local (Webapp/Desktop) chat must not drop MEDIA attachments silently.

Field report: a job created from a local chat resolves ``deliver=origin`` to a ``local`` target.
That branch hands ``deliver_to_local_chat`` the tag-stripped text and has no attachment lane, so a
``MEDIA:`` image never reaches the chat — and because the branch ``continue``s before any media
accounting, the run still reports a clean ``delivered`` with ``last_delivery_error`` unset.

Every other lane records the drop (media-path policy, live-adapter confirmation timeout, the
reconnect ledger), which is what lets a user tell a text-only reply from one whose image vanished.
The local branch owes the same record: the note itself stays text-only (a local chat renders the
delivered note as a user-role row, where a ``MEDIA:`` tag is plain text, not an inline image).
"""

from cron.scheduler_delivery import _deliver_result


def _local_job():
    return {
        "id": "job-local-media",
        "name": "local-media",
        "deliver": "origin",
        "origin": {"platform": "local", "chat_id": "local-abc"},
    }


def test_local_delivery_records_the_attachment_it_cannot_carry(tmp_path, monkeypatch):
    media = tmp_path / "card.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

    error = _deliver_result(_local_job(), f"Today's card.\n\nMEDIA:{media}", adapters=None, loop=None)

    assert error is not None
    assert "1 media attachment(s) not delivered" in error, error
    assert "local chat local-abc" in error, error


def test_text_only_local_delivery_gains_no_attachment_note(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

    error = _deliver_result(_local_job(), "Today's summary.", adapters=None, loop=None)

    # No gateway owns the chat in this process, so the delivery itself is still refused — but the
    # refusal must not invent an attachment loss.
    assert error is not None
    assert "media attachment" not in error, error


def test_local_note_text_still_has_the_media_tag_stripped(tmp_path, monkeypatch):
    """The fix reports the drop; it must not start forwarding the path into the note."""
    media = tmp_path / "card.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    seen = []

    from gateway import session_cron

    def fake_deliver(home, chat_id, text, *, diagnostic=False):
        seen.append(text)

    monkeypatch.setattr(session_cron, "deliver_to_local_chat", fake_deliver)

    error = _deliver_result(_local_job(), f"Today's card.\n\nMEDIA:{media}", adapters=None, loop=None)

    assert seen and str(media) not in seen[0]
    assert "1 media attachment(s) not delivered" in (error or ""), error
