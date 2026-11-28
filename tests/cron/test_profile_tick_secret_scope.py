"""Tick housekeeping must resolve the owning profile's delivery secrets."""
import pytest

from agent.secret_scope import (
    current_secret_scope_home,
    get_secret,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from cron.scheduler_delivery import _env_home_target_chat_id
from cron.scheduler_provider import _profile_cron_scope
from hermes_constants import get_hermes_home


@pytest.mark.parametrize('fail_inside', [False, True])
def test_tick_binds_own_secrets_and_restores_parent(tmp_path, monkeypatch, fail_inside):
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'launch'))
    monkeypatch.setenv('SLACK_HOME_CHANNEL', 'ambient-wrong-channel')
    monkeypatch.setenv('SLACK_BOT_TOKEN', 'ambient-wrong-token')
    home = tmp_path / 'ops'
    home.mkdir()
    (home / '.env').write_text('SLACK_HOME_CHANNEL=ops-channel\n')
    set_multiplex_active(True)
    parent = set_secret_scope({'SLACK_HOME_CHANNEL': 'parent-channel'}, profile_home='parent')
    try:
        try:
            with _profile_cron_scope(home):
                assert get_hermes_home() == home
                assert current_secret_scope_home() == str(home)
                assert _env_home_target_chat_id('slack') == 'ops-channel'
                assert get_secret('SLACK_BOT_TOKEN', '') == ''
                if fail_inside:
                    raise RuntimeError('tick failed')
        except RuntimeError:
            if not fail_inside:
                raise
        assert get_secret('SLACK_HOME_CHANNEL') == 'parent-channel'
        assert current_secret_scope_home() == 'parent'
        assert get_hermes_home() == tmp_path / 'launch'
    finally:
        reset_secret_scope(parent)
        set_multiplex_active(False)
