"""Repair ACP rows whose workspace lives only in the model_config JSON.

ACP sessions minted before the ``cwd`` column was populated still carry the
workspace inside ``model_config``, so promoting it is a lossless repair rather
than a guess. On a real install every ACP session predating the fix was
affected, and every one was recoverable this way.
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    yield store
    store.close()


def test_backfill_promotes_cwd_from_model_config(db):
    """Promotes only rows that need it, and only once."""
    db.create_session(session_id="legacy", source="acp", model="m",
                      model_config={"cwd": "/work/hs-wwd"})
    # A non-ACP row is not this repair's business even if it has a JSON cwd,
    # and an ACP row with no JSON cwd has nothing to promote.
    db.create_session(session_id="cli", source="cli", model="m",
                      model_config={"cwd": "/somewhere"})
    db.create_session(session_id="bare", source="acp", model="m",
                      model_config={"provider": "anthropic"})

    assert db.backfill_acp_session_cwd() == 1
    assert db.get_session("legacy")["cwd"] == "/work/hs-wwd"
    assert db.get_session("cli")["cwd"] in (None, "")
    assert db.get_session("bare")["cwd"] in (None, "")

    # Idempotent: a second run is a no-op, not a rewrite.
    assert db.backfill_acp_session_cwd() == 0


def test_backfill_never_overwrites_an_existing_cwd(db):
    """An explicit column value always wins over the JSON copy."""
    db.create_session(session_id="fine", source="acp", model="m",
                      cwd="/real/path", model_config={"cwd": "/stale/json/path"})

    assert db.backfill_acp_session_cwd() == 0
    assert db.get_session("fine")["cwd"] == "/real/path"
