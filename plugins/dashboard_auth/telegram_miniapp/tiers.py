"""Dashboard access tiers for a Telegram user whose identity is already verified.

Spec §1: three tiers for the Mini App dashboard —

  1. unauthenticated — Telegram ``initData`` missing/invalid/expired. 403 on
     every route. Not represented here; this module only runs on a user_id
     that already passed HMAC verification (see ``initdata.py``).
  2. paired/allowlisted — the same trust decision the Telegram bot itself
     uses for DM traffic (:func:`gateway.authz_mixin.is_authorized`, via the
     pairing store or a ``TELEGRAM_ALLOWED_USERS``-style env allowlist).
     Read-only: status/skills/cron + the user's own DM sessions.
  3. admin — also listed in ``TELEGRAM_DASHBOARD_ADMIN_USERS``. Full access.

``TELEGRAM_DASHBOARD_ADMIN_USERS`` is fail-closed by default: unset or empty
means the admin tier does not exist for anyone, even a paired/allowlisted
user — the Mini App never silently promotes ordinary pairing into admin.
"""
from __future__ import annotations

import os
from typing import Literal

Tier = Literal["admin", "paired", "unauthorized"]


def dashboard_admin_user_ids() -> frozenset[str]:
    """Parse ``TELEGRAM_DASHBOARD_ADMIN_USERS`` into a set of user IDs.

    Comma-separated, matching the convention of every other ``*_ALLOWED_USERS``
    env var in this codebase. No ``"*"`` wildcard support: unlike the
    inbound-message allowlists in ``gateway.authz_mixin.is_authorized``,
    admin is the highest-privilege dashboard tier and is opt-in per user ID
    only — a wildcard here would silently promote every paired user to admin.

    Reads the machine's ``.env`` file fresh per call (:func:`_dashboard_env_get`),
    not the dashboard process's own ``os.environ`` snapshot — the same
    staleness fix ``is_paired_or_allowlisted`` already applies to
    ``TELEGRAM_ALLOWED_USERS``/``TELEGRAM_GROUP_ALLOWED_USERS``. An earlier
    version of this function read ``os.environ`` directly, meaning an admin
    added via ``PUT /api/env`` (or a live-edited ``.env``) wouldn't take
    effect until the dashboard process itself restarted — the exact
    dashboard-side staleness problem this whole per-request-read design
    exists to avoid, just never wired up for this one variable.
    """
    raw = _dashboard_env_get("TELEGRAM_DASHBOARD_ADMIN_USERS")
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


def _machine_hermes_home() -> "Path":
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
    from pathlib import Path

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


def _dashboard_env_get(key: str, default: str = "") -> str:
    """``os.getenv``-shaped adapter over :func:`_read_machine_env_var`.

    Passed as ``is_authorized(..., env_get=...)`` so the trust-classification
    logic itself is not duplicated here (see :func:`is_paired_or_allowlisted`)
    — only the source of the raw env strings changes, from the dashboard
    process's own stale ``os.environ`` snapshot to a fresh per-request file
    read.
    """
    return _read_machine_env_var(key) or default


def is_paired_or_allowlisted(user_id: str, *, pairing_store) -> bool:
    """Same trust decision the Telegram bot uses for this user's DMs.

    Builds a synthetic Telegram DM :class:`~gateway.session.SessionSource`
    for *user_id* and runs it through the pure
    :func:`gateway.authz_mixin.is_authorized` — the identical check the live
    Telegram adapter applies to an inbound DM from this user — rather than
    reimplementing pairing/allowlist logic here and risking divergence.
    ``chat_id`` is set to *user_id*: Telegram's private-chat ID for a user's
    DM with the bot is that user's own numeric ID.

    No adapter-policy callables are passed: there is no live Telegram adapter
    "receiving" a Mini App HTTP call, so this resolves exactly as
    ``is_authorized`` does for a platform with no adapter registered (its
    documented default) — env allowlist / pairing store only, which is what
    Telegram (unlike the own-policy adapters such as WeCom) already uses.

    ``env_get=_dashboard_env_get`` closes the dashboard-side staleness
    problem: the dashboard process loads ``.env`` into its own ``os.environ``
    once, at its own startup, exactly like the gateway process does — so a
    ``TELEGRAM_ALLOWED_USERS``/``TELEGRAM_GROUP_ALLOWED_USERS`` edit via
    ``PUT /api/env`` would otherwise only take effect here after a *dashboard*
    restart, a second stale-consumer problem independent of the gateway's own
    (which needs a *gateway* restart and is out of this function's scope).
    Reading fresh per call structurally avoids that without needing a
    dashboard-restart banner.
    """
    from gateway.authz_mixin import is_authorized
    from gateway.config import Platform
    from gateway.session import SessionSource

    if not user_id:
        return False

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=user_id,
        chat_type="dm",
        user_id=user_id,
        user_name=None,
    )
    return is_authorized(
        source,
        pairing_is_approved=lambda platform_name, uid: pairing_store.is_approved(platform_name, uid),
        env_get=_dashboard_env_get,
    )


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
