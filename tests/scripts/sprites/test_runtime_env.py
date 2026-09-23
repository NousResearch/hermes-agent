import os

from dotenv import dotenv_values

from scripts.sprites.runtime_env import sync_environment


def test_rotation_and_removal_survive_repeated_configure_without_erasing_user_values(tmp_path):
    env = tmp_path / '.env'
    env.write_text("USER_KEY=user-value\nMANAGED_KEY=old-user-value\n")
    sync_environment(tmp_path, {'MANAGED_KEY': "rotated'\\value"}, os.getuid(), os.getgid())
    assert dotenv_values(env) == {'USER_KEY': 'user-value', 'MANAGED_KEY': "rotated'\\value"}
    sync_environment(tmp_path, {}, os.getuid(), os.getgid())
    sync_environment(tmp_path, {}, os.getuid(), os.getgid())
    assert dotenv_values(env) == {'USER_KEY': 'user-value', 'MANAGED_KEY': ''}
    assert env.stat().st_mode & 0o777 == 0o600


def test_named_profiles_receive_relay_settings_without_other_profiles_credentials(tmp_path):
    profile = tmp_path / 'profiles' / 'work'
    profile.mkdir(parents=True)
    (profile / '.env').write_text('VENDOR_KEY=profile-owned\n')
    sync_environment(tmp_path, {'VENDOR_KEY': 'default-only', 'HERMES_AUTH_JSON_BOOTSTRAP': 'private-bootstrap',
                              'GATEWAY_RELAY_URL': 'https://connector.example'}, os.getuid(), os.getgid())
    root = dotenv_values(tmp_path / '.env')
    named = dotenv_values(profile / '.env')
    assert 'HERMES_AUTH_JSON_BOOTSTRAP' not in root
    assert named['VENDOR_KEY'] == 'profile-owned'
    assert named['GATEWAY_RELAY_URL'] == 'https://connector.example'
    assert 'private-bootstrap' not in (profile / '.env').read_text()
