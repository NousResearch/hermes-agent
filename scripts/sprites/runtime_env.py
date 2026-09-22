"""Apply NAS-owned environment values without replacing user configuration.

Hermes loads .env with override=True. The final managed block therefore owns
rotation/removal, including tombstones for removed keys. Only relay settings are
shared with named profiles; OAuth bootstrap and vendor secrets stay scoped.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile

BEGIN = '# BEGIN NAS SPRITES\n'
END = '# END NAS SPRITES\n'
RELAY_KEYS = frozenset(('GATEWAY_RELAY_URL', 'GATEWAY_RELAY_INSTANCE_ID', 'GATEWAY_RELAY_ID',
    'GATEWAY_RELAY_PLATFORMS', 'GATEWAY_RELAY_BOT_IDS', 'GATEWAY_MULTIPLEX_PROFILES',
    'HERMES_SCALE_TO_ZERO', 'GATEWAY_RELAY_WAKE_URL', 'GATEWAY_RELAY_SLEEP_URL'))
BOOTSTRAP_KEYS = frozenset(('HERMES_AUTH_JSON_BOOTSTRAP', 'HERMES_GATEWAY_BOOTSTRAP_STATE'))


def write_managed(path: Path, values: dict[str, str], uid: int, gid: int) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Unsafe managed environment path')
    existing = path.read_text(encoding='utf-8') if path.exists() else ''
    blocks = re.findall(re.escape(BEGIN) + r'(.*?)' + re.escape(END), existing, flags=re.S)
    prior = set(re.findall(r'^([A-Za-z_][A-Za-z0-9_]*)=', ''.join(blocks), flags=re.M))
    existing = re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END), '', existing, flags=re.S)
    lines = []
    for name in sorted(prior | values.keys()):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            raise ValueError('Invalid environment name')
        value = values.get(name, '')
        if '\0' in value:
            raise ValueError('Invalid environment value')
        quoted = value.replace('\\', '\\\\').replace("'", "\\'")
        lines.append(name + "='" + quoted + "'\n")
    fd, temp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(existing.rstrip('\n') + '\n' + BEGIN + ''.join(lines) + END)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchown(stream.fileno(), uid, gid)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def sync_environment(home: Path, values: dict[str, str], uid: int, gid: int) -> None:
    write_managed(home / '.env', {k: v for k, v in values.items() if k not in BOOTSTRAP_KEYS}, uid, gid)
    relay = {k: values.get(k, '') for k in RELAY_KEYS}
    for profile in (home / 'profiles').glob('*'):
        if profile.is_dir() and not profile.is_symlink():
            write_managed(profile / '.env', relay, uid, gid)
