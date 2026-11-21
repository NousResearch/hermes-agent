"""``same_history_owner``: whether a chain member is history the authorizer of the chain's anchor controls."""

from persistence.sqlite.authority._binding_history import same_history_owner
from tests.fixtures.storage_seed import seed_session


def test_rows_the_resume_adopts_take_sidebar_edits(store):
    for sid, source, user_id in (('old-desktop', 'desktop', None), ('old-oneshot', 'oneshot', None),
                                 ('old-desktop-nous', 'desktop', 'nous:account'),
                                 ('old-desktop-foreign', 'desktop', 'telegram:42'),
                                 ('old-cli-nous', 'cli', 'nous:account'), ('api-row', 'api_server', None)):
        seed_session(store, sid, source, user_id=user_id)
    with store.holder()._read_ctx() as conn:
        assert same_history_owner(conn, 'old-desktop', 'old-desktop-nous') is True


def test_a_login_labelled_desktop_row_belongs_to_that_login_only(store):
    seed_session(store, 'alice-row', 'desktop', user_id='basic:alice')
    seed_session(store, 'oidc-row', 'desktop', user_id='oidc:alice')
    with store.holder()._read_ctx() as conn:
        # Not an anonymous chain member either: a no-owner anchor does not reach it.
        assert same_history_owner(conn, 'alice-row', 'alice-row') is True
    seed_session(store, 'plain', 'desktop')
    with store.holder()._read_ctx() as conn:
        assert same_history_owner(conn, 'plain', 'alice-row') is False
