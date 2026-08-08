"""Per-user profile derivation and lazy provisioning for a single-bot gateway.

``gateway.multiplex_profiles`` isolates one *bot credential* per profile: the
profile is stamped by the adapter that received the message
(``_make_profile_message_handler``), and two profiles cannot poll the same bot
token. So every user of a single bot lands in the one ``default`` profile and
shares its ``HERMES_HOME`` (skills, cron ``jobs.json``, native ``MEMORY.md`` /
``USER.md``, config, SOUL) — one user's tool calls can then read or clobber
another's workspace.

``gateway.per_user_profiles`` closes that gap by deriving the profile from the
message **sender** instead of the adapter, then reusing the exact same runtime
machinery (``_profile_runtime_scope`` / ``_resolve_profile_home_for_source`` and
the ``agent:<profile>:…`` session-key namespace). Two deliberate differences
from multiplex, both handled by the caller (``gateway.run``):

  1. Sender-derived, not adapter-derived — this module.
  2. Shared service credentials — per-user turns run under a *home-only* scope
     (``get_hermes_home()`` → the user's home) while ``get_secret`` keeps
     reading the process-global ``os.environ``, so the user profile never needs
     its own ``.env`` API keys.

This module owns only the two pure-ish pieces: deriving a valid, collision-safe
profile id from a ``SessionSource``, and creating that profile on first contact
(seeded from a template) in a race-safe way.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Mirror hermes_cli.profiles._PROFILE_ID_RE: an on-disk profile dir name must be
# ``[a-z0-9][a-z0-9_-]{0,63}`` (≤64 chars, starts alphanumeric). We validate our
# derived names against the public ``validate_profile_name`` at call time, but
# keep the regex here too so derivation can pick the hashed fallback *before*
# constructing an invalid name.
_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_MAX_LEN = 64

# One lock per profile name so two concurrent first-contact messages from the
# same new user don't both run create_profile(). Guarded by a module lock so the
# defaultdict itself stays thread-safe.
_provision_locks: "defaultdict[str, threading.Lock]" = defaultdict(threading.Lock)
_locks_guard = threading.Lock()


def _sanitize_segment(value: str) -> str:
    """Lowercase and strip a string down to the profile-id charset.

    Non ``[a-z0-9_-]`` characters are dropped (not replaced) so different raw
    ids can't collide on a shared separator, and a leading run of ``-``/``_`` is
    trimmed so the joined name still starts alphanumeric.
    """
    out = re.sub(r"[^a-z0-9_-]+", "", str(value).strip().lower())
    return out.lstrip("-_")


def derive_user_profile_name(source: Any, prefix: str = "u") -> Optional[str]:
    """Derive a deterministic, valid profile id from a message sender.

    Returns ``<prefix>-<platform>-<uid>`` (e.g. ``u-telegram-1271274566``), or a
    hashed form ``<prefix>-<platform>-<sha1[:16]>`` when the raw uid doesn't fit
    the profile-id charset/length. The platform segment namespaces ids across
    platforms so a telegram uid and a spacechat uid never collide.

    Returns ``None`` when the source carries no stable sender id — the caller
    then leaves the turn in the shared default namespace rather than bucketing
    unknown senders together.
    """
    raw = getattr(source, "user_id_alt", None) or getattr(source, "user_id", None)
    if raw is None:
        return None
    uid = str(raw).strip()
    if not uid:
        # Whitespace-only id is not a stable sender → shared default namespace,
        # never a degenerate ``<prefix>-<platform>-`` profile shared by all of them.
        return None

    prefix_seg = _sanitize_segment(prefix) or "u"

    platform = getattr(source, "platform", None)
    platform_val = getattr(platform, "value", None) or str(platform or "")
    plat_seg = _sanitize_segment(platform_val) or "x"

    uid_seg = _sanitize_segment(uid)
    name = f"{prefix_seg}-{plat_seg}-{uid_seg}"

    # Use the readable name ONLY when sanitizing the uid was LOSSLESS — the uid was
    # already a non-empty lowercase profile-id charset (e.g. a numeric platform
    # id). Compare against the case-PRESERVING uid, because sanitization also
    # lowercases and drops characters: without this, distinct senders collide
    # (``Bob``/``bob`` → ``bob``; ``a.b``/``ab`` → ``ab``; ``!!!`` → ``""``) and
    # silently share one profile. In every lossy case hash the raw uid so each
    # distinct sender gets a distinct, collision-safe, case-preserving id.
    lossless = bool(uid_seg) and uid_seg == uid
    if not lossless or not _PROFILE_ID_RE.match(name) or len(name) > _MAX_LEN:
        digest = hashlib.sha1(uid.encode("utf-8")).hexdigest()[:16]
        name = f"{prefix_seg}-{plat_seg}-{digest}"

    if len(name) > _MAX_LEN:
        # Pathological prefix/platform lengths — fall back to a fully hashed tail.
        # Reserve room for ``-<digest>`` BEFORE truncating, then bound only the
        # prefix. A blind ``[:_MAX_LEN]`` here would slice the sender-specific
        # digest off a max-length prefix and collapse every sender onto one
        # profile; bounding the prefix keeps the full digest, so distinct senders
        # stay distinct while the name still fits ``_MAX_LEN``.
        digest = hashlib.sha1(f"{plat_seg}:{uid}".encode("utf-8")).hexdigest()[:16]
        prefix_bounded = prefix_seg[: _MAX_LEN - len(digest) - 1].rstrip("-_") or "u"
        name = f"{prefix_bounded}-{digest}"

    return name


def is_user_profile_name(name: Optional[str], prefix: str = "u") -> bool:
    """True if *name* looks like a derived per-user profile id (``<prefix>-``).

    Lets the home resolver auto-provision derived names while still warning +
    falling back for an explicit but missing operator profile.
    """
    if not name:
        return False
    prefix_seg = _sanitize_segment(prefix) or "u"
    return str(name).startswith(f"{prefix_seg}-")


def ensure_user_profile(name: str, template: Optional[str] = None) -> bool:
    """Create the per-user profile dir on first contact; idempotent + race-safe.

    Seeds from *template* (falls back to the active profile) via
    ``create_profile(clone_config=True)`` so the user inherits the operator's
    config.yaml / SOUL.md / base skills. The seeded ``.env`` is then removed:
    per-user turns resolve credentials from the shared ``os.environ`` (home-only
    scope), so a copied secret file would be dead weight at rest.

    Returns ``True`` if it created the profile, ``False`` if it already existed
    (the hot path after first contact) or creation lost a race.
    """
    from hermes_cli.profiles import (
        create_profile,
        get_active_profile_name,
        get_profile_dir,
        profile_exists,
    )

    import shutil

    if profile_exists(name):
        return False

    with _locks_guard:
        lock = _provision_locks[name]
    try:
        with lock:
            # Re-check under the lock — another turn may have created it while we waited.
            if profile_exists(name):
                return False
            tmpl = (template or "").strip() or get_active_profile_name() or "default"
            try:
                create_profile(name, clone_from=tmpl, clone_config=True, no_alias=True)
            except FileExistsError:
                return False
            except Exception:
                # A partial create_profile (e.g. disk full mid-copytree) can leave a
                # half-built dir behind. profile_exists() only checks the dir's
                # presence, so a corrupt remnant would be treated as fully
                # provisioned forever after — every future turn would run against an
                # incomplete home. Remove it (best-effort) before propagating so the
                # next message retries a clean provision.
                try:
                    shutil.rmtree(get_profile_dir(name), ignore_errors=True)
                except Exception:
                    logger.debug("per_user_profiles: cleanup of partial %r failed", name, exc_info=True)
                logger.warning(
                    "per_user_profiles: failed to provision profile %r from template %r",
                    name, tmpl, exc_info=True,
                )
                raise
            # Drop the cloned .env — credentials come from the shared os.environ.
            try:
                env_path = get_profile_dir(name) / ".env"
                if env_path.exists():
                    env_path.unlink()
            except Exception:
                logger.debug("per_user_profiles: could not remove seeded .env for %r", name, exc_info=True)
            logger.info("per_user_profiles: provisioned profile %r from template %r", name, tmpl)
            return True
    finally:
        # Evict the per-name lock so _provision_locks doesn't grow one entry per
        # distinct user for the process lifetime. Safe: after the first successful
        # provision the fast-path profile_exists() short-circuits before ever
        # taking a lock again, and a concurrent waiter re-checks profile_exists().
        with _locks_guard:
            _provision_locks.pop(name, None)
