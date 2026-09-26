"""Dashboard secret-scope boundary (audit 2026-09-25 / repair-phase fix).

The dashboard's own HTTP routes (portal status, subscription features, auth checks, the
memory-providers probe) call into credential-reading code without ever binding a profile
secret scope. Under multiplex this either raises ``UnscopedSecretError`` (loud sites:
auth.py, cron delivery) or gets silently swallowed by a broad ``except Exception`` (portal
status), degrading to a misleadingly empty result instead of the real one.

These routes are NOT per-request profile-parameterized like the cron dashboard endpoints
(``web_server_cron.py::_cron_profile_home``) — they report on the profile this dashboard
process is currently homed to, exactly like ``/api/profiles/active``'s ``"current"`` field
already does. So the correct scope boundary is simply: bind the CURRENT effective
``HERMES_HOME`` (``get_hermes_home()``, which already respects any active
``set_hermes_home_override``) — no new profile-resolution logic, reusing the existing
``build_profile_secret_scope`` primitive the same way ``gateway/run.py::
_load_profile_secret_scope`` does for turns.
"""
from __future__ import annotations

import contextlib


@contextlib.contextmanager
def dashboard_profile_secret_scope():
    """Bind secret scope to the dashboard's current effective profile home for one block.

    Only touches secret scope (never ``HERMES_HOME`` itself — the caller's effective home is
    already correct); safe to nest inside ``_cron_store_scope``, which additionally redirects
    the home for admin cron routes that operate on a DIFFERENT profile than the one serving
    the dashboard.
    """
    from agent.secret_scope import (
        build_profile_secret_scope,
        reset_secret_scope,
        set_secret_scope,
    )
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    secrets = build_profile_secret_scope(home)
    token = set_secret_scope(secrets)
    try:
        yield home
    finally:
        reset_secret_scope(token)
