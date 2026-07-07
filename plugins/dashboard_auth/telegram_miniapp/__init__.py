"""TelegramMiniAppProvider — dashboard auth for the Telegram Mini App (spec §1-§4).

Exposes the local Hermes dashboard inside a Telegram Mini App. The frontend
sends Telegram's signed ``initData`` string as the bearer credential
(``Authorization: Bearer <initData>`` — see ``initdata.py``'s module
docstring for the spec §8 caveat on this transport assumption);
``verify_token`` HMAC-verifies it against ``TELEGRAM_BOT_TOKEN``, extracts
the Telegram user id, and resolves it to one of the tiers in ``tiers.py``
(unauthorized / paired / admin) using the exact same pairing/allowlist
decision the Telegram bot itself uses for DM traffic — not a parallel
authorization system.

Non-goals (do not reintroduce — see the spec this plugin implements):
  * No Chat/PTY-over-WebSocket exposure through the Mini App.
  * No per-message sender attribution.
  * No "fixing" cross-profile authorization into being profile-aware —
    ``PairingStore`` and the env allowlists this defers to
    (``gateway.authz_mixin.is_authorized``) are process-global by design,
    same as for every other platform; a multi-profile Hermes instance has
    ONE pairing/allowlist surface, not one per profile.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    LoginStart,
    Session,
    TokenPrincipal,
)

from .initdata import DEFAULT_MAX_AGE_SECONDS, extract_user_id, verify_init_data
from .tiers import resolve_tier

logger = logging.getLogger(__name__)

LAST_SKIP_REASON: str = ""


class TelegramMiniAppProvider(DashboardAuthProvider):
    """Non-interactive provider: verifies Telegram ``initData``, not a login flow."""

    name = "telegram-miniapp"
    display_name = "Telegram Mini App"
    supports_token = True
    supports_session = False

    def __init__(
        self,
        *,
        bot_token: str,
        pairing_store,
        max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    ) -> None:
        if not bot_token:
            raise ValueError("TelegramMiniAppProvider requires a non-empty bot_token")
        self._bot_token = bot_token
        self._pairing_store = pairing_store
        self._max_age_seconds = max_age_seconds

    # ---- token capability (the only thing this provider implements) --------

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        """Verify Telegram ``initData`` (passed as *token*) and resolve a tier.

        Returns ``None`` — never raises — for: a bad/expired/malformed
        ``initData`` (``verify_init_data`` failure), a missing user id, or a
        verified user who isn't paired/allowlisted (``tiers.resolve_tier``
        returns ``"unauthorized"``). All three collapse to the same 401 at
        the seam; this method's job is only to say yes/no + attach scopes,
        not to distinguish rejection reasons to the caller (that's what the
        audit log the seam already writes is for).
        """
        fields = verify_init_data(
            token, bot_token=self._bot_token, max_age_seconds=self._max_age_seconds
        )
        if fields is None:
            # Deliberately logged (without the raw token): during live
            # rollout this was the only observable difference between "bad
            # credential" and "valid credential, unpaired user" -- both
            # collapse to the same 401 by design, and this line is what let
            # the initData check-string bug be found at all.
            logger.warning(
                "dashboard-auth-telegram-miniapp: verify_init_data rejected the "
                "token (bad signature, malformed, or outside the %ss replay "
                "window)",
                self._max_age_seconds,
            )
            return None

        user_id = extract_user_id(fields)
        if not user_id:
            return None

        tier = resolve_tier(user_id, pairing_store=self._pairing_store)
        if tier == "unauthorized":
            return None

        scopes = ("dashboard:read", "dashboard:admin") if tier == "admin" else ("dashboard:read",)
        return TokenPrincipal(
            principal=f"telegram:{user_id}",
            provider=self.name,
            scopes=scopes,
        )

    # ---- interactive methods: unsupported (non-interactive credential only) --

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        raise NotImplementedError(
            "TelegramMiniAppProvider is a non-interactive token credential; "
            "there is no login flow."
        )

    def complete_login(
        self, *, code: str, state: str, code_verifier: str, redirect_uri: str
    ) -> Session:
        raise NotImplementedError(
            "TelegramMiniAppProvider is a non-interactive token credential."
        )

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        # Not a cookie-session provider — never mints a Session, so it can
        # never recognise a session cookie. Return None (don't raise) so it
        # stacks harmlessly in the cookie-verify loop, mirroring DrainSecretProvider.
        return None

    def refresh_session(self, *, refresh_token: str) -> Session:
        raise NotImplementedError(
            "TelegramMiniAppProvider is a non-interactive token credential."
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def _load_config_section() -> dict:
    """Return ``dashboard.telegram_miniapp`` from config.yaml, or ``{}``."""
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        logger.debug(
            "dashboard-auth-telegram-miniapp: load_config() raised %s; "
            "falling back to disabled",
            exc,
        )
        return {}
    section = cfg_get(cfg, "dashboard", "telegram_miniapp", default=None)
    return section if isinstance(section, dict) else {}


def register(ctx) -> None:
    """Plugin entry — registers TelegramMiniAppProvider when explicitly enabled.

    Off by default (spec §6): a no-op unless ``dashboard.telegram_miniapp.enabled:
    true`` in config.yaml. When enabled, fails closed (records a skip reason,
    does not register) if ``TELEGRAM_BOT_TOKEN`` is unset — there's nothing to
    verify ``initData`` against without it. The admin-allowlist fail-closed
    behavior (``TELEGRAM_DASHBOARD_ADMIN_USERS`` unset => no admin tier for
    anyone) lives in ``tiers.py`` and needs no gating here: an empty admin
    allowlist is a valid, safe configuration (paired/allowlisted-only access),
    not a startup error.
    """
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    section = _load_config_section()
    if not bool(section.get("enabled", False)):
        LAST_SKIP_REASON = (
            "dashboard.telegram_miniapp.enabled is not set to true in config.yaml; "
            "the Telegram Mini App dashboard is disabled by default."
        )
        logger.debug("dashboard-auth-telegram-miniapp: %s", LAST_SKIP_REASON)
        return

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        LAST_SKIP_REASON = (
            "dashboard.telegram_miniapp.enabled is true but TELEGRAM_BOT_TOKEN is "
            "not set. The Mini App dashboard stays disabled (fail-closed) — "
            "initData can't be verified without the bot token that signed it."
        )
        logger.warning("dashboard-auth-telegram-miniapp: %s", LAST_SKIP_REASON)
        return

    try:
        max_age_seconds = int(section.get("max_age_seconds", DEFAULT_MAX_AGE_SECONDS))
    except (TypeError, ValueError):
        max_age_seconds = DEFAULT_MAX_AGE_SECONDS

    from gateway.pairing import PairingStore

    try:
        provider = TelegramMiniAppProvider(
            bot_token=bot_token,
            pairing_store=PairingStore(),
            max_age_seconds=max_age_seconds,
        )
    except ValueError as exc:
        LAST_SKIP_REASON = f"TelegramMiniAppProvider construction failed: {exc}"
        logger.warning("dashboard-auth-telegram-miniapp: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)
    _register_miniapp_token_routes()
    logger.info("dashboard-auth-telegram-miniapp: registered Telegram Mini App dashboard provider")


# ---------------------------------------------------------------------------
# Route registration (spec §1/§4: read-only status/skills/cron + own DM
# sessions). Deliberately a short, explicit allowlist — NOT a blanket prefix
# on /api/sessions/. `token_auth.is_token_route()` has no HTTP-method
# awareness, so a bare `match="prefix"` registration of "/api/sessions/"
# would also make POST /api/sessions/bulk-delete, POST /api/sessions/prune,
# and the broad-read GET /api/sessions/search, /stats, /empty/count
# token-authable — none of which this plugin intends to expose. Session ids
# are registered with `match="regex"` instead, anchored to the two shapes
# actually emitted by session_id generation (gateway/session.py's 8-hex new
# sessions, gateway/slash_commands.py + hermes_cli/cli_commands_mixin.py's
# 6-hex branch/thread sessions — both `YYYYMMDD_HHMMSS_<hex>`), so a literal
# sibling path can never satisfy the pattern by construction.
#
# READ-ONLY (methods=("GET","HEAD")): this is the spec §1 read-only guarantee,
# and it is load-bearing for safety, not cosmetic. Path matching is method-
# blind, and several of these paths ALSO mount a mutating handler on the exact
# same path — GET vs POST /api/skills (create-skill, which is agent-executed,
# i.e. code execution), GET vs PUT /api/skills/content, GET vs POST
# /api/cron/jobs, GET vs DELETE/PATCH /api/sessions/{id}. Registering these
# for token auth WITHOUT the method restriction would make a paired Mini App
# principal able to authenticate those write verbs too (the token seam would
# set token_authenticated=True and the cookie gate would then skip), turning a
# "read-only" tier into full write + RCE. Restricting to the read verbs means
# the seam never authenticates a write request here; a write attempt falls
# through to the cookie gate and is rejected for a caller with no cookie.
#
# This registers DISPATCH eligibility only — "a token could plausibly
# authenticate here" — not authorization. The DM-ownership check on
# GET /api/sessions/{id} and .../messages (ensuring the requester only sees
# their own sessions) is enforced separately in the route handlers.
#
# All registrations use required=False: every one of these routes is also
# reachable today by the existing cookie-authenticated desktop dashboard.
# required=True (the default) would make this seam token-EXCLUSIVE for the
# path — a desktop request with no Authorization header would get a hard 401
# from this seam and never reach the cookie gate, locking out the actual
# dashboard operator the day this plugin is enabled. required=False means a
# request with no bearer token at all is untouched by this seam and falls
# through to the cookie gate exactly as before; a request that DOES present
# a bearer token (e.g. the Mini App) is still fully authenticated/rejected
# by this seam alone, unchanged from required=True's contract for a
# token-bearing caller. See token_auth.py's module + register_token_route
# docstrings for the full design.
# ---------------------------------------------------------------------------

# Two id shapes: an ordinary interactive session (started_at timestamp +
# short uuid4 hex, see gateway/session.py and slash_commands.py's /branch) or
# a cron run (cron/scheduler.py's run_job: "cron_{job_id}_{timestamp}", where
# job_id is cron/jobs.py create_job's uuid4().hex[:12]). Opening a cron-run
# session used to fall through this seam entirely (fullmatch failed, so the
# request reached the downstream cookie/session gate instead, which has no
# cookie to check for a Mini App caller and returned a 401) -- surfaced in
# the Mini App as an instant, incorrect "session expired" screen, since
# api.ts treats any post-establishment 401 as auth expiry.
_SESSION_ID_SHAPE = r"(?:\d{8}_\d{6}_[0-9a-f]{6,8}|cron_[0-9a-f]{12}_\d{8}_\d{6})"
_SESSION_ID_RE = rf"/api/sessions/{_SESSION_ID_SHAPE}"
_SESSION_MESSAGES_RE = rf"/api/sessions/{_SESSION_ID_SHAPE}/messages"

# Cron job ids are uuid4().hex[:12] (cron/jobs.py's create_job) -- 12 lowercase
# hex chars, never a shape that could collide with a sibling literal path.
_CRON_JOB_ACTION_RE = r"/api/cron/jobs/[0-9a-f]{12}/(pause|resume|trigger)"
_CRON_JOB_ID_RE = r"/api/cron/jobs/[0-9a-f]{12}"

# Same numeric-id shape POST /api/telegram/allowlist validates server-side
# (hermes_cli/web_server.py's _TELEGRAM_USER_ID_RE) -- anchors the DELETE
# path so it can't be satisfied by any other sibling route.
_TELEGRAM_ALLOWLIST_ENTRY_RE = r"/api/telegram/allowlist/\d{5,15}"


_READ_METHODS = ("GET", "HEAD")


def _register_miniapp_token_routes() -> None:
    """Register every Mini App-reachable route: read-only for every tier,
    plus admin-tier mutating actions the Users/Cron/Status/Sessions/Skills
    screens need.

    The read-only half (unchanged from the original spec §1/§4 read-only
    guarantee) is method-restricted to GET/HEAD for the reasons in the
    module comment above. The admin-tier mutations added below are each a
    DIFFERENT method on paths that share no literal collision with anything
    not meant to be exposed (verified per-endpoint: cron's pause/resume/
    trigger/delete via regexes anchored to the real job-id shape, not a
    prefix that would also sweep in PUT /api/cron/jobs/{id} (job-detail-edit
    -- no Mini App editor exists for it); sessions' PATCH/DELETE via the SAME
    session-id regex already registered read-only, method sets unioned
    across the two registrations).
    Every one of these mutating routes has its own handler-level
    `_require_dashboard_admin(request)` call (hermes_cli/web_server.py) —
    registering a route here only makes it DISPATCH-eligible for a bearer
    token, exactly like the read-only half; it is not by itself an
    authorization decision. A non-admin paired token presenting itself to
    any of these gets 403 from the handler, not a silent pass-through.
    """
    from hermes_cli.dashboard_auth.token_auth import register_token_route

    for path in (
        "/api/status",
        "/api/skills",
        "/api/skills/content",
        "/api/cron/jobs",
        "/api/cron/delivery-targets",
        "/api/cron/blueprints",
        "/api/sessions",
    ):
        register_token_route(path, required=False, methods=_READ_METHODS)
    register_token_route(_SESSION_ID_RE, match="regex", required=False, methods=_READ_METHODS)
    register_token_route(_SESSION_MESSAGES_RE, match="regex", required=False, methods=_READ_METHODS)

    # /api/miniapp/me: the frontend's own tier-detection call. Read-only,
    # but not registered above since it's Mini-App-specific, not shared with
    # the desktop dashboard's existing GET surface.
    register_token_route("/api/miniapp/me", required=False, methods=_READ_METHODS)

    # ---- admin-tier mutating actions -------------------------------------
    # Cron: pause/resume/trigger, plus DELETE the job itself -- NOT a prefix
    # (would also expose PUT /api/cron/jobs/{id}, job-detail-edit, which the
    # Mini App's Cron screen has no editor for). _CRON_JOB_ID_RE is
    # deliberately registered for DELETE only here; GET on the same shape
    # (single-job detail) is not a Mini App route at all yet.
    register_token_route(_CRON_JOB_ACTION_RE, match="regex", required=False, methods=("POST",))
    register_token_route(_CRON_JOB_ID_RE, match="regex", required=False, methods=("DELETE",))

    # Skills: toggle only -- NOT create/edit (POST /api/skills, PUT
    # /api/skills/content -- agent-executed skill writes stay out of the
    # Mini App entirely, admin or not).
    register_token_route("/api/skills/toggle", required=False, methods=("PUT",))

    # Sessions: PATCH (archive) + DELETE, same path already registered
    # read-only above -- methods union, required stays False either way.
    register_token_route(_SESSION_ID_RE, match="regex", required=False, methods=("PATCH", "DELETE"))

    # Telegram allowlist: the Users tab's own narrow surface (see
    # hermes_cli/web_server.py's module comment on why this exists instead
    # of exposing the generic /api/env here).
    register_token_route("/api/telegram/allowlist", required=False, methods=("GET", "HEAD", "POST"))
    register_token_route(_TELEGRAM_ALLOWLIST_ENTRY_RE, match="regex", required=False, methods=("DELETE",))

    # Gateway lifecycle: restart + update, the Status tab's admin buttons.
    register_token_route("/api/gateway/restart", required=False, methods=("POST",))
    register_token_route("/api/hermes/update", required=False, methods=("POST",))

    # Sessions search: admin-tier only (see search_sessions's own docstring
    # in hermes_cli/web_server.py) -- a raw global FTS search with no
    # DM-ownership scoping, unlike GET /api/sessions.
    register_token_route("/api/sessions/search", required=False, methods=_READ_METHODS)
