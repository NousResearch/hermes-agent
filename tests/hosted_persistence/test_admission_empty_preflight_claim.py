"""An empty preflight claims nothing that lands before its own claim transaction.

The drain that read an empty FIFO validated nothing: a row admitted after that read but before the claim's write
commits has had no preclaim checks of its own, so that pass claims nothing (the claim names the head it validated,
none) and lets the drain re-read; the new head earns its own authorization.
"""

from __future__ import annotations

import pytest

from gateway import session_api_turn, session_finite
from gateway.config import Platform
from persistence.core.errors import AuthorityRefusal
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority


def _admit_before_the_claim(monkeypatch, authority, text):
    """The authority's claim admits ``text`` first, once: the row commits while the claim waits for the writer,
    after the preflight read."""
    admissions = authority.persistence.admissions
    original, inserted = admissions.claim, []

    def racing(**kw):
        if not inserted:
            inserted.append(admissions.admit(epoch=authority.epoch, principal_id="u", session_id="s",
                                             request_id="late", payload={"text": text}))
        return original(**kw)
    monkeypatch.setattr(admissions, "claim", racing)
    return inserted


@pytest.mark.asyncio
async def test_a_head_landing_before_the_claim_transaction_is_not_claimed_by_it(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.API_SERVER)
    inserted = _admit_before_the_claim(monkeypatch, authority, "late")
    with db:
        result, first = await authority._claim_next(REF, authority.sessions["s"])
        assert inserted and first is None
        assert result is False, "a new head must earn its own preclaim checks"
        assert [row.status for row in authority.persistence.admissions.list(session_id="s")] == ["queued"]


@pytest.mark.asyncio
async def test_a_late_revoked_head_earns_its_own_preclaim_authorization(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.API_SERVER)
    _admit_before_the_claim(monkeypatch, authority, "revoked")
    checked, executed = [], []

    def deny(authority, ref, payload):
        checked.append(payload["text"])
        raise AuthorityRefusal("permission_denied")

    async def execute(authority, ref, row):
        executed.append(row["payload"]["text"])
        return "done"
    monkeypatch.setattr(session_api_turn, "check_api_turn", deny)
    monkeypatch.setattr(session_finite, "execute_finite_admission", execute)
    with db:
        await authority._drain(REF)
        assert executed == [], "revoked input bypassed its preclaim authorization"
        assert checked == ["revoked"]
