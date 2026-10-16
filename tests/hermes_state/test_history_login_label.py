"""A dashboard login may act on the ``<provider>:<user>`` rows Desktop stamps for it.

Desktop stamps its rows as ``<provider>:<user>`` (tui_gateway/session_transports.py), while a
forwarded dashboard actor carries ``authenticated_subject``'s ``auth:v1:[provider, "", user]``
form (gateway/session_identity.py). ``authorize_history`` compared the two byte for byte, so the
right logged-in user got ``permission_denied`` on rows it owned (#138950 review). The label maps
only for an issuer-less login — the label drops the issuer, so an issuer-bearing subject must not
claim one — and never to anything that itself reads as a subject.
"""

import pytest

from gateway.session_contract import Principal
from hermes_state import SessionDB
from hermes_state_mutation_binding import _login_label, authorize_history

LOGIN = 'auth:v1:["basic","","e2e"]'


def _actor(subject, tmp_path):
    return Principal(subject, str(tmp_path), frozenset({"session:read", "session:create"}), "native-http:x")


def _auth(db, subject, row_id):
    with db._read_ctx() as conn:
        return authorize_history(conn, _actor(subject, db.db_path.parent), row_id)


@pytest.fixture
def db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


def test_the_login_label_maps_for_an_issuer_less_login():
    assert _login_label(LOGIN) == "basic:e2e"


def test_a_desktop_labelled_row_takes_its_login(db):
    db.create_session("desktop-row", source="desktop", user_id="basic:e2e")
    assert _auth(db, LOGIN, "desktop-row") is True


def test_a_foreign_account_still_refused(db):
    db.create_session("mallory-row", source="desktop", user_id="basic:mallory")
    with pytest.raises(Exception) as raised:
        _auth(db, LOGIN, "mallory-row")
    assert getattr(raised.value, "reason", None) == "permission_denied"


def test_the_exact_subject_still_matches(db):
    db.create_session("subject-row", source="desktop", user_id=LOGIN)
    assert _auth(db, LOGIN, "subject-row") is True


@pytest.mark.parametrize("subject", [
    'auth:v1:["basic","https://idp.example","e2e"]',  # issuer-bearing: the label drops the issuer
    'auth:v1:["basic","",""]',                        # empty user
    'auth:v1:["basic","","  "]',                      # blank user
    'auth:v1:["ba:sic","","e2e"]',                    # a ':' in the provider would forge a third form
    'auth:v1:["","","e2e"]',                          # empty provider
    'auth:v1:["auth","","v1:[\\"x\\"]"]',             # label would read as a subject itself
    'auth:v1:"basic"',                                # non-list payload
    'uid:501',                                        # not a login subject at all
])
def test_labels_that_must_not_map(subject):
    assert _login_label(subject) is None


def test_an_issuer_bearing_login_cannot_touch_the_labelled_row(db):
    db.create_session("desktop-row", source="desktop", user_id="basic:e2e")
    with pytest.raises(Exception) as raised:
        _auth(db, 'auth:v1:["basic","https://idp.example","e2e"]', "desktop-row")
    assert getattr(raised.value, "reason", None) == "permission_denied"


def test_an_unowned_row_still_admits_any_local_actor(db):
    db.create_session("unowned-row", source="desktop")
    assert _auth(db, LOGIN, "unowned-row") is True
