"""Surface retired reset settings at update completion, without changing policy."""
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from hermes_cli.backup import _sibling_profile_homes
from hermes_cli.config_effective import load_user_config_effective
from hermes_cli.profiles import get_active_profile_name
from hermes_cli.session_reset_retirement import PLUGIN_NAME, reset_plugin_enabled, retired_reset_policy
from hermes_cli.update_cmd_common import _best_effort


def print_session_reset_update_notice() -> None:
    """Check each profile independently: an enabled root plugin does not cover siblings."""
    active_home = get_hermes_home()
    homes = [(get_active_profile_name(), active_home), *_sibling_profile_homes(active_home)]
    for name, home in homes:
        with _best_effort('Session reset policy notice failed: %s'):
            token = set_hermes_home_override(home)
            try:
                config = load_user_config_effective()
                policy = retired_reset_policy(config)
                if policy is not None and not reset_plugin_enabled():
                    _print_profile_notice(name, config, *policy)
            finally:
                reset_hermes_home_override(token)


def _print_profile_notice(name: str, config: dict, path: str, mode: str) -> None:
    block = config['session_reset'] if path == 'session_reset' else config['gateway']['session_reset']
    settings = [f'{path}.mode: {mode}']
    settings.extend(f'{key}: {block[key]}' for key in ('idle_minutes', 'at_hour') if key in block)
    command = 'hermes' if name == 'custom' else f'hermes -p {name}'
    print(f"\n⚠ Session reset policy needs attention (profile '{name}')")
    print('  Configured: ' + ', '.join(settings))
    print('  Core no longer applies this policy; gateway conversations reset only on /new or /reset.')
    print('  To keep idle/daily resets, review the third-party catalog plugin, then run:')
    print(f'    {command} plugins install {PLUGIN_NAME}')
    if path != 'session_reset':
        print('  Move the block to top-level session_reset before using the plugin.')
    print('  Configuration preserved; no plugin was installed by this notice.')
