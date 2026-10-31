#!/usr/bin/env python3
"""Interactive, local-only setup. Does not launch a server or publish a port."""
import argparse
import copy
import getpass
import os
from pathlib import Path
import secrets
import sys
from datetime import datetime, timezone
from urllib.parse import urlsplit


def validate_origin(value):
    url = urlsplit(value)
    if (url.scheme != 'https' or not url.hostname or not url.hostname.endswith('.ts.net')
            or url.username or url.password or url.query or url.fragment
            or url.path not in ('', '/') or url.port not in (None, 443)):
        raise ValueError('Use the exact private HTTPS .ts.net origin, without a path.')
    return 'https://' + url.hostname


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', required=True, type=validate_origin)
    parser.add_argument('--hermes-source', type=Path, required=True)
    parser.add_argument('--hermes-home', type=Path, required=True)
    args = parser.parse_args()
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError('Run this setup directly in Terminal; passwords must not be piped or logged.')
    source = args.hermes_source.resolve(strict=True)
    home = args.hermes_home.resolve(strict=True)
    os.environ['HERMES_HOME'] = str(home)
    sys.path.insert(0, str(source))
    from hermes_cli.config import atomic_config_write, require_readable_config_before_write, is_managed
    from hermes_cli.managed_scope import managed_config_keys
    from plugins.dashboard_auth.basic import hash_password
    if is_managed() or any(str(k).startswith(('dashboard', 'plugins')) for k in managed_config_keys()):
        raise RuntimeError('Dashboard or plugin configuration is managed; use your administrator setup.')
    path = home / 'config.yaml'
    initial = path.read_bytes()
    original = require_readable_config_before_write(path)
    existing = original.get('dashboard') or {}
    if existing.get('basic_auth'):
        raise RuntimeError('Existing password settings found. They have been preserved; review before replacing.')
    if existing.get('public_url') not in (None, '', args.origin):
        raise RuntimeError('A different dashboard URL is already configured; preserved without changes.')
    disabled = (original.get('plugins') or {}).get('disabled') or []
    if set(disabled) & {'basic', 'dashboard_auth/basic'}:
        raise RuntimeError('The password plugin is explicitly disabled; review before enabling it.')
    print('\nHermes iPhone private login setup')
    print('Mac address: ' + args.origin)
    print('Only dashboard password settings and its trusted HTTPS origin will change.')
    print('The password is hidden while typing and is never stored as plaintext.\n')
    username = input('Choose Hermes username: ').strip()
    if not username:
        raise RuntimeError('Username cannot be empty.')
    password = getpass.getpass('Choose Hermes password (at least 8 characters): ')
    confirm = getpass.getpass('Confirm password: ')
    if len(password) < 8 or password != confirm:
        raise RuntimeError('Password is too short or confirmation does not match. Nothing changed.')
    digest = hash_password(password)
    password = confirm = ''
    if input('Save this private login configuration? Type yes: ').strip().lower() != 'yes':
        print('Cancelled; nothing changed.')
        return
    if path.read_bytes() != initial:
        raise RuntimeError('Configuration changed during setup. Nothing written; run setup again.')
    updated = copy.deepcopy(original)
    dashboard = updated.setdefault('dashboard', {})
    dashboard['public_url'] = args.origin
    dashboard['basic_auth'] = {'username': username, 'password_hash': digest,
                               'secret': secrets.token_urlsafe(32)}
    backups = home / 'companion-backups'
    backups.mkdir(mode=0o700, exist_ok=True)
    if backups.stat().st_mode & 0o077:
        raise RuntimeError('Backup directory is not private; nothing changed.')
    backup = backups / ('config-before-private-login-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.yaml')
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(initial)
    atomic_config_write(path, updated, sort_keys=False)
    if require_readable_config_before_write(path) != updated:
        raise RuntimeError('Saved configuration verification failed. Preserve the backup and inspect before restart.')
    print('\nSAVED. Tell Codex setup is saved. The backend has not been restarted.')
    print('No Tailscale port has been published by this setup.')


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('\nSetup stopped: ' + str(error), file=sys.stderr)
        sys.exit(1)
