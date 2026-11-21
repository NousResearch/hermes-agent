"""Regression (NousResearch/hermes-agent#145022 field data): SessionAuthority.receipt must
not run the admission read on the gateway event loop.

Kanban workers poll this route while a card runs; on a multiplex host (21 profiles, one
gateway in the field report) the synchronous store read held the shared owner loop long
enough to miss the liveness probe and trigger exit-75 self-kills. The sibling claim/cancel
paths already read through asyncio.to_thread; a regression that pulls receipt back on-loop
must fail here."""
import asyncio
import threading
import types

from gateway.session_authority import SessionAuthority


def test_receipt_reads_the_admission_off_the_event_loop(monkeypatch):
    seen = {}

    class _Admissions:
        def get(self, *, admission_id):
            seen['thread'] = threading.get_ident()
            seen['admission_id'] = admission_id
            return {'target_session_id': 's1', 'principal_id': 'p1', 'status': 'terminal'}

    authority = types.SimpleNamespace(
        authorize=lambda *args: None,
        persistence=types.SimpleNamespace(admissions=_Admissions()),
        sessions={'s1': types.SimpleNamespace(settling={},
                                              event_stream=types.SimpleNamespace(lock=threading.Lock()))},
        _receipt=lambda row: row)
    # Patch the bindings the method body reads, keeping the test about thread affinity.
    monkeypatch.setattr('gateway.session_authority.admission_row', lambda record: record)
    actor = types.SimpleNamespace(subject='p1', capabilities={'session:submit', 'session:control'})
    ref = types.SimpleNamespace(session_id='s1')

    receipt = asyncio.run(SessionAuthority.receipt(authority, actor, ref, 'adm-1'))

    assert receipt['status'] == 'terminal'
    assert seen['admission_id'] == 'adm-1'
    # asyncio.run drives the coroutine on THIS thread, so a to_thread read lands on a
    # different ident; a synchronous read (the stall) would record this thread's ident.
    assert seen['thread'] != threading.get_ident()
