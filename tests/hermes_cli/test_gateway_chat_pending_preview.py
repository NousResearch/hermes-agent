"""A resumed classic CLI shows what each pending admission will run, safely, from the real snapshot."""
from dataclasses import asdict


def test_resume_lists_pending_admission_text_capped_and_terminal_safe(capsys):
    from gateway.session_authority import SessionAuthority
    from hermes_cli.gateway_chat_view import GatewayChatView

    authority = object.__new__(SessionAuthority)
    authority.profile_id, authority.epoch = "profile", 3
    authority.launch_sources = {"stored": "cli"}  # the resumed classic CLI's launch source
    row = {"target_session_id": "stored", "outcome": None, "owner_epoch": 3}
    hostile = "\x1b]0;pwned\x07\x1b[2Jdeploy\rhidden\x08\u202e\x9b31m now " + "x" * 200
    rows = [
        {**row, "admission_id": "adm-unknown", "seq": 1, "status": "unknown", "generation": 4,
         "request_id": "in-1", "payload": {"native_text_v1": {"event": {"text": "summarise the logs"}}}},
        {**row, "admission_id": "adm-queued", "seq": 2, "status": "queued", "generation": None,
         "request_id": "in-2", "payload": {"text": hostile}},
    ]
    # The authority's own snapshot projection, exactly what session.resume ships to the CLI.
    snapshot = {"stored_session_id": "stored",
                "pending": [asdict(authority._pending_receipt(r)) for r in rows]}

    GatewayChatView(None, snapshot).show_pending()

    err = capsys.readouterr().err
    lines = err.splitlines()
    assert lines.index('    "summarise the logs"') < lines.index("/discard adm-unknown")
    queued = next(line for line in lines if "deploy hidden" in line)
    assert "now xxx" in queued
    assert queued.startswith('    "') and queued.endswith('…"')
    assert len(queued[5:-1]) <= 80
    assert not [ch for ch in err if ch != "\n" and not ch.isprintable()], repr(err)
