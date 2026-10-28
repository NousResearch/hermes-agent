"""Regression for #145075: the owning profile's home comes from ``authority.profile_id``,
never from ``Path(authority.db.db_path).resolve().parent`` — ``resolve()`` follows a
symlinked ``state.db`` to its physical target, so a store outside the profile home
made cron (and every config read) scope into the wrong directory.
"""
import asyncio
from types import SimpleNamespace


def _symlinked_db(tmp_path):
    home = tmp_path / 'home'
    store = tmp_path / 'store'
    home.mkdir()
    store.mkdir()
    db_path = home / 'state.db'
    db_path.symlink_to(store / 'state.db')
    return home, db_path


def test_cron_owner_keyed_by_profile_home_not_db_symlink_parent(tmp_path):
    from gateway import session_cron

    home, db_path = _symlinked_db(tmp_path)
    authority = SimpleNamespace(
        profile_id=str(home), db=SimpleNamespace(db_path=str(db_path)))

    async def probe():
        session_cron.bind_owner(authority)
        try:
            bound = session_cron.owner_for_home(home)
            assert bound is not None and bound[0] is authority
        finally:
            session_cron.unbind_owner(authority)
        assert session_cron.owner_for_home(home) is None

    asyncio.run(probe())


def test_authority_for_home_uses_profile_id_not_symlinked_db(tmp_path):
    from gateway.session_authorities import authority_for_home

    home, db_path = _symlinked_db(tmp_path)
    authority = SimpleNamespace(
        profile_id=str(home), db=SimpleNamespace(db_path=str(db_path)))
    runner = SimpleNamespace(session_authority=authority)

    assert authority_for_home(runner, home) is authority
    assert authority_for_home(runner, tmp_path / 'other') is None
