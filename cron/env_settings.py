"""Profile-scoped reads of the ``HERMES_*`` tuning settings cron honours from ``.env``.

A standalone ``hermes -p X gateway run`` loads X's ``.env`` into ``os.environ``, so a bare
``os.getenv("HERMES_CRON_TIMEOUT")`` is X's value. Under ``gateway.multiplex_profiles`` the same
tick runs inside the default profile's process, where ``os.environ`` holds the DEFAULT profile's
``.env``. With a secret scope installed (job run + delivery) the scope is authoritative; the tick
loop itself (due-job scan, pool sizing) runs under the profile's home override only, so the
setting is read from that home's ``.env``. Outside multiplex the read is the plain environ.
"""

from __future__ import annotations

import os

from agent.secret_scope import current_secret_scope, is_multiplex_active, load_env_file
from hermes_constants import get_hermes_home


def cron_env_setting(name: str, default: str = "") -> str:
    if not is_multiplex_active():
        return os.getenv(name) or default
    scope = current_secret_scope()
    if scope is None:
        scope = load_env_file(get_hermes_home() / ".env")
    value = scope.get(name)
    return default if value is None else str(value)
