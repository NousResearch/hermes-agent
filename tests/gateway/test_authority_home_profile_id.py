"""Regression for #145075: the owning profile's home comes from ``authority.profile_id``,
never from ``Path(authority.db.db_path).resolve().parent`` — ``resolve()`` follows a
symlinked ``state.db`` to its physical target, so a store outside the profile home
made cron (and every config read) scope into the wrong directory.
"""
import asyncio
import json
from types import SimpleNamespace


def _symlinked_db(tmp_path):
    home = tmp_path / 'home'
    store = tmp_path / 'store'
    home.mkdir()
    store.mkdir()
    db_path = home / 'state.db'
    db_path.symlink_to(store / 'state.db')
    return home, db_path


def _authority(home, db_path):
    return SimpleNamespace(
        profile_id=str(home), db=SimpleNamespace(db_path=str(db_path)))


def test_cron_owner_keyed_by_profile_home_not_db_symlink_parent(tmp_path):
    from gateway import session_cron

    home, db_path = _symlinked_db(tmp_path)
    authority = _authority(home, db_path)

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
    authority = _authority(home, db_path)
    runner = SimpleNamespace(session_authority=authority)

    assert authority_for_home(runner, home) is authority
    assert authority_for_home(runner, tmp_path / 'other') is None


def test_profile_config_reads_the_profile_home(tmp_path):
    """session_policy.profile_config must read home/config.yaml, not the symlink target's dir."""
    from gateway.session_policy import profile_config

    home, db_path = _symlinked_db(tmp_path)
    (home / 'config.yaml').write_text('marker_section:\n  flag: true\n', encoding='utf-8')

    assert 'marker_section' in profile_config(_authority(home, db_path))


def test_orphan_close_grace_reads_the_profile_home(tmp_path):
    """session_viewer_close.orphan_close_grace scopes its config read to the owning home."""
    from gateway.session_viewer_close import orphan_close_grace

    home, db_path = _symlinked_db(tmp_path)
    (home / 'config.yaml').write_text('sessions:\n  orphan_end_grace_s: 42\n', encoding='utf-8')

    assert orphan_close_grace(_authority(home, db_path)) == 42.0


def test_config_reference_binds_the_profile_home(tmp_path):
    """session_policy_credentials.config_reference records the profile home, not the db parent."""
    from gateway.session_policy import LocalSessionPolicy
    from gateway.session_policy_credentials import PREFIX, config_reference

    home, db_path = _symlinked_db(tmp_path)
    policy = LocalSessionPolicy(
        source='cli', platform='cli', cwd=str(tmp_path), model=None, toolsets=(),
        config_json='{}', request_json='{}', terminal_json='{}')

    ref = config_reference(_authority(home, db_path), 'sess-1', policy, {})
    payload = json.loads(ref[len(PREFIX):])
    assert payload['home'] == str(home)
    assert payload['profile'] == str(home)


def test_provider_resolution_reads_the_profile_home(tmp_path):
    """session_mutation_model._provider_resolution_config merges the owning home's providers."""
    from gateway.session_mutation_model import _provider_resolution_config

    home, db_path = _symlinked_db(tmp_path)
    (home / 'config.yaml').write_text(
        'providers:\n  marker_provider:\n    key_env: MARKER_KEY\n', encoding='utf-8')

    resolved = _provider_resolution_config(_authority(home, db_path), {}, bypass=False)
    assert 'marker_provider' in resolved.get('providers', {})
