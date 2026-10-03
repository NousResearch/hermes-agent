"""Outbound-only delivery grants (``gateway.delivery_grants``): a profile lends its OWN bot to
another profile's cron deliveries for an exact, enumerated set of targets.

Why this exists (#128411). Under ``gateway.multiplex_profiles`` each platform is its own
credential boundary, so a satellite profile with no messaging adapter of its own cannot deliver
cron output — or, worse, its ``failure_deliver`` alerts — to a channel on a bot a *different*
profile owns. A ``profile_routes`` entry only lends the bot for targets that route INBOUND to the
satellite, which is a different (and much smaller) set than "where my alerts should go". The
common topology — one ``ops`` profile owning the Discord bot, ~10 worker profiles owning cron
monitors — left every worker's failure alerts undeliverable after the fold to multiplex.

What a grant is, and what it is NOT. A grant authorizes exactly one thing: an OUTBOUND send
through the GRANTOR's already-connected adapter, to one of the grantor-declared targets, on behalf
of a profile the grantor named. It is a delivery capability, not a credential delegation:

- **Never inbound.** A grant creates no ``ProfileRoute``; no message arriving on the grantor's
  bot is routed to the grantee's profile, and ``gateway/authz_mixin.py`` never consults a grant
  (its ``bot_profile is None`` shared-bot fallback stays route-driven).
- **Never session routing.** Nothing keys a session, a runtime profile, or an agent turn to the
  grantee because of a grant.
- **Never tool access.** A grant does not appear in the toolset resolver or ``check_fn``; the
  grantee gains no new secret, and its ``.env`` is still its own.
- **Never a prefix or a pattern.** ``targets`` are exact ``platform:chat_id[:thread_id]`` tokens
  compared for full string equality after the same normalization a job's ``deliver`` value gets.
  A grant naming one channel never reaches another, a thread, or a DM of the same bot.
- **Never implicit.** There is no default grant and no wildcard target; ``outbound_only: false``
  is rejected at parse time (an inbound grant is not a thing this surface offers), and a
  malformed or unreadable grant is a fail-closed ``[]`` — never a fallback to the default bot.

Because it is outbound-only, the grant cannot weaken the per-platform credential boundary: the
grantor is deliberately publishing a channel on its own bot, and the grantee can only write into
the container the grantor named.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ``targets`` entries are written exactly as a cron job's ``deliver`` value writes a channel, so
# both sides of a comparison go through one normalizer.
# ``to_profiles: ["*"]`` names every served profile. A wildcard is about WHICH PROFILES the
# grantor trusts, never about WHICH TARGETS: ``targets`` always stays exact.
_ALL_PROFILES = "*"


def _as_bool(value: Any, default: bool) -> bool:
    """YAML-truthy coercion tolerant of a quoted ``"true"``/``"no"``."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "yes", "on", "1"):
        return True
    if text in ("false", "no", "off", "0"):
        return False
    return default


@dataclass(frozen=True)
class DeliveryGrant:
    """One grantor's outbound-only permission for one platform's bot, to named profiles, for an
    exact target set.

    ``targets`` holds ``(platform, chat_id, thread_id)`` triples already normalized through the
    same target resolver a job's ``deliver`` value uses, so comparison is exact string equality
    on both sides. There is deliberately no ``outbound_only`` field: an entry that does not assert
    it is dropped at parse time, so every surviving grant is outbound-only by construction and no
    consumer can read the flag and honor an inbound reading.
    """

    name: str
    bot_platform: str
    to_profiles: tuple
    targets: tuple
    enabled: bool = True

    def authorizes(self, platform_name: Any, chat_id: Any, thread_id: Any = None) -> bool:
        """True when this grant covers exactly this platform + chat (+ thread, when the target
        carries one). A grant on a channel does not cover its threads, and vice versa."""
        if not self.enabled:
            return False
        platform_key = str(getattr(platform_name, "value", platform_name) or "").strip().lower()
        if not platform_key or platform_key != self.bot_platform:
            return False
        if not chat_id:
            return False
        wanted_chat = str(chat_id)
        wanted_thread = str(thread_id) if thread_id else None
        return (
            platform_key,
            wanted_chat,
            wanted_thread,
        ) in self.targets

    def names_profile(self, matches) -> bool:
        """True when this grant names a served profile, per the caller's own home predicate.

        ``matches(profile_name) -> bool`` is supplied by the caller because the only sound answer
        to "is this name the profile being served?" is a comparison against the serving HOME, not
        against a name this module would re-derive (and get wrong for a home outside
        ``profiles/``). ``"*"`` short-circuits to every served profile."""
        if _ALL_PROFILES in self.to_profiles:
            return True
        return any(matches(name) for name in self.to_profiles)


def _normalize_grant_target(token: Any) -> Optional[tuple]:
    """``platform:chat_id[:thread_id]`` -> ``(platform, chat_id, thread_id)``, or ``None``.

    Resolved through the same target resolver cron delivery uses, so a grant written against a
    channel directory name matches the id a job's ``deliver`` resolves to. A reference that will
    not resolve is kept verbatim rather than dropped: matching is exact string equality on both
    sides, so a grant written against a raw id still matches, and a grant written against a name
    that does not resolve simply never matches (fail closed)."""
    text = str(token or "").strip()
    if ":" not in text:
        return None
    platform, rest = text.split(":", 1)
    platform = platform.strip().lower()
    rest = rest.strip()
    if not platform or not rest:
        return None
    try:
        from tools.send_message_tool import prepare_send_message_platforms, resolve_send_target

        prepare_send_message_platforms()
        chat_id, thread_id, error = resolve_send_target(
            platform, rest, pass_unresolved_references=True)
        if not error and chat_id:
            return (platform, str(chat_id), str(thread_id) if thread_id else None)
    except Exception:
        logger.debug("delivery grant target %r unresolvable; matching it verbatim", text, exc_info=True)
    return (platform, rest, None)


def _normalize_profile_list(value: Any) -> tuple:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(
        name for name in (str(item or "").strip() for item in value) if name
    )


def parse_delivery_grants(raw: Any) -> list:
    """Parse ``gateway.delivery_grants`` from config.yaml. Every malformed entry is dropped with a
    WARNING naming it; a list with no surviving entry is ``[]``, which authorizes nothing."""
    if not isinstance(raw, list):
        return []
    grants: list = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        platform = str(entry.get("bot_platform") or "").strip().lower()
        if not platform:
            logger.warning(
                "Skipping delivery grant %s: missing bot_platform (the granted platform is the "
                "credential boundary — a grant never guesses it)", name or entry)
            continue
        # An inbound grant is not a thing this surface offers, so refuse it loudly rather than
        # shipping a flag some future consumer could read the wrong way (#128411).
        if not _as_bool(entry.get("outbound_only", True), default=True):
            logger.warning(
                "Skipping delivery grant %r: outbound_only must be true. A delivery grant is "
                "outbound-only by construction — inbound routing is configured with "
                "gateway.profile_routes, which grants nothing beyond receiving.", name or platform)
            continue
        to_profiles = _normalize_profile_list(entry.get("to_profiles"))
        if not to_profiles:
            logger.warning(
                "Skipping delivery grant %r: to_profiles is empty — a grant must name who may "
                "use the grantor's bot (or \"*\" for every served profile)", name or platform)
            continue
        if _ALL_PROFILES not in to_profiles:
            # Reuse the profile-name validator: the name becomes a path component when matched
            # against a served home, so `../` must not survive parsing (#90418 parity).
            try:
                from hermes_cli.profiles import normalize_profile_name, validate_profile_name

                validated = []
                for profile_name in to_profiles:
                    canon = normalize_profile_name(profile_name)
                    validate_profile_name(canon)
                    validated.append(canon)
                to_profiles = tuple(validated)
            except (ValueError, ImportError):
                logger.warning(
                    "Skipping delivery grant %r: invalid profile name in to_profiles %s",
                    name or platform, to_profiles)
                continue
        raw_targets = entry.get("targets")
        if isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        if not isinstance(raw_targets, list):
            logger.warning(
                "Skipping delivery grant %r: targets must be a list of exact "
                "`platform:chat_id[:thread_id]` targets", name or platform)
            continue
        targets = tuple(
            normalized
            for normalized in (_normalize_grant_target(token) for token in raw_targets)
            if normalized is not None
        )
        if not targets:
            logger.warning(
                "Skipping delivery grant %r: no usable target in %s — a grant with no target "
                "grants nothing and is a config error, not an open invitation",
                name or platform, raw_targets)
            continue
        grants.append(DeliveryGrant(
            name=name,
            bot_platform=platform,
            to_profiles=to_profiles,
            targets=targets,
            enabled=_as_bool(entry.get("enabled", True), default=True),
        ))
    return grants


def grant_for_target(
    grants, platform_name: Any, chat_id: Any, thread_id: Any = None,
) -> Optional[DeliveryGrant]:
    """The first enabled grant authorizing exactly this platform/chat/thread, else ``None``."""
    for grant in grants or ():
        if grant.authorizes(platform_name, chat_id, thread_id):
            return grant
    return None


def grant_for_deliver_token(grants, token: str) -> Optional[DeliveryGrant]:
    """The grant covering a raw ``deliver``-style token (``"discord:<channel>"``).

    The preflight gate reads the job's configured token, not a resolved target, so it asks the
    grant through the same normalizer — one predicate, so the gate and the send cannot disagree."""
    normalized = _normalize_grant_target(token)
    if normalized is None:
        return None
    platform, chat_id, thread_id = normalized
    return grant_for_target(grants, platform, chat_id, thread_id)
