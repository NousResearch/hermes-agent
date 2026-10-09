"""Dashboard access tiers for a Telegram user whose identity is already verified.

Three tiers for the Mini App dashboard —

  1. unauthenticated — Telegram ``initData`` missing/invalid/expired. 403 on
     every route. Not represented here; this module only runs on a user_id
     that already passed HMAC verification (see ``initdata.py``).
  2. paired/allowlisted — the same trust decision the Telegram bot itself
     uses for DM traffic (the gateway's own ``_is_user_authorized``: the
     pairing store or a ``TELEGRAM_ALLOWED_USERS``-style env allowlist).
     Read-only: status/skills/cron + the user's own DM sessions.
  3. admin — also listed in ``TELEGRAM_DASHBOARD_ADMIN_USERS``. Full access.

``TELEGRAM_DASHBOARD_ADMIN_USERS`` is fail-closed by default: unset or empty
means the admin tier does not exist for anyone, even a paired/allowlisted
user — the Mini App never silently promotes ordinary pairing into admin.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

Tier = Literal["admin", "paired", "unauthorized"]


def dashboard_admin_user_ids() -> frozenset[str]:
    """Parse ``TELEGRAM_DASHBOARD_ADMIN_USERS`` into a set of user IDs.

    Comma-separated, matching the convention of every other ``*_ALLOWED_USERS``
    env var in this codebase. No ``"*"`` wildcard support: unlike the
    inbound-message allowlists, admin is the highest-privilege dashboard tier
    and is opt-in per user ID only — a wildcard here would silently promote
    every paired user to admin.

    Reads the machine's ``.env`` file fresh per call, not the dashboard
    process's ``os.environ``, so an admin added by editing ``.env`` takes
    effect without restarting the dashboard.
    """
    raw = _read_machine_env_var("TELEGRAM_DASHBOARD_ADMIN_USERS")
    if not raw:
        return frozenset()
    return frozenset(uid.strip() for uid in raw.split(",") if uid.strip())


def is_dashboard_admin(user_id: str) -> bool:
    """True iff *user_id* is explicitly listed in ``TELEGRAM_DASHBOARD_ADMIN_USERS``.

    Fail-closed: an unset/empty env var means nobody is admin, full stop.
    """
    if not user_id:
        return False
    return user_id in dashboard_admin_user_ids()


def _machine_hermes_home() -> Path:
    """Resolve ``~/.hermes`` for the *machine's default profile*, deliberately
    bypassing any per-request profile-scope override active on this task.

    Telegram bot pairing/allowlisting is a single, process-global concept
    (see this module's docstring: there is one Telegram gateway per machine,
    not one per profile). A dashboard request scoped to a different profile
    (``?profile=worker``) runs inside a ``_profile_scope``/``_config_profile_scope``
    context (``hermes_cli/web_server.py``) that overrides
    ``hermes_constants.get_hermes_home()`` via a contextvar for the duration
    of that request. Calling that function here would silently read a
    *different* profile's ``.env`` instead of the one the Telegram gateway
    itself consumes — reintroducing a multiplex-profile leak in the exact
    place this module's per-request re-read exists to avoid one. Replicates
    only the override-free half of ``get_hermes_home()``'s own resolution
    (``HERMES_HOME`` env var, then the platform-native default) so it never
    consults the profile-scope override.
    """

    from hermes_constants import _get_platform_default_hermes_home

    override_free = os.environ.get("HERMES_HOME", "").strip()
    if override_free:
        return Path(override_free)
    return _get_platform_default_hermes_home()


def _read_machine_env_var(key: str) -> str:
    """Read *key* directly from the machine's ``~/.hermes/.env``, per call.

    No ``os.environ`` mutation, no ``load_hermes_dotenv()`` call (which would
    also pull external secret sources and the managed-scope overlay — more
    than this needs, and both process-global side effects this function must
    not have), and no profile-scope interaction (see :func:`_machine_hermes_home`).

    Reuses :func:`hermes_cli.config.load_env` — the same sanitizing,
    quote-aware ``.env`` parser every other reader in this codebase uses —
    but by temporarily clearing the profile-scope override (restored in the
    ``finally``) rather than calling ``get_hermes_home()`` as that function
    normally would, so it reads the machine default regardless of what
    profile the calling dashboard request happens to be scoped to.
    ``load_env()`` is mtime+size-cached, so calling this on every dashboard
    auth check re-parses the file only when it has actually changed.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(_machine_hermes_home()))
    try:
        from hermes_cli.config import load_env

        return load_env().get(key, "") or ""
    finally:
        reset_hermes_home_override(token)


def is_paired_or_allowlisted(user_id: str, *, pairing_store) -> bool:
    """Same trust decision the Telegram bot uses for this user's DMs.

    Runs the gateway's own ``GatewayAuthorizationMixin._is_user_authorized`` on
    a synthetic Telegram DM source (``chat_id`` is the user's id, as for a real
    private chat) rather than reimplementing pairing/allowlist logic here. No
    live adapter receives a Mini App HTTP call, so the decision resolves as it
    does for a platform with no adapter registered: the pairing store and the
    env allowlists — exactly what a Telegram DM is checked against.

    Env allowlists are read from this process's environment. Allowlist edits
    made through the dashboard (``save_env_value`` / ``remove_env_value``)
    are published into it immediately.
    """
    from gateway.authz_mixin import GatewayAuthorizationMixin
    from gateway.config import Platform
    from gateway.session import SessionSource

    if not user_id:
        return False
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=user_id, chat_type="dm", user_id=user_id, user_name=None)
    # A bare instance with only the pairing store: no adapters, as the mixin supports.
    authorizer = object.__new__(GatewayAuthorizationMixin)
    authorizer.pairing_store = pairing_store
    return authorizer._is_user_authorized(source)


def resolve_tier(user_id: str, *, pairing_store) -> Tier:
    """Return the dashboard tier for a verified Telegram *user_id*.

    ``"unauthorized"`` covers both "not paired/allowlisted at all" and
    "no admin allowlist" cases — the Mini App auth layer treats it the same
    as a failed ``initData`` check (403), it is just distinguished here for
    logging/audit clarity.
    """
    if not is_paired_or_allowlisted(user_id, pairing_store=pairing_store):
        return "unauthorized"
    if is_dashboard_admin(user_id):
        return "admin"
    return "paired"
