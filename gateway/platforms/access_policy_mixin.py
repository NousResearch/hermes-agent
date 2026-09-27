"""Config-driven DM / group intake policy shared by the adapters that gate access themselves
(Weixin, WeCom, QQBot, WhatsApp, Yuanbao's ``AccessPolicy``).

Host contract — the adapter sets these on ``self`` before any predicate runs: ``_dm_policy`` /
``_group_policy`` (``open`` | ``allowlist`` | ``disabled`` | ``pairing``), ``_allow_from`` /
``_group_allow_from`` (iterable of allowlist entries), and ``ALLOW_ALL_ENV_PREFIX`` (class
attribute, e.g. ``"WEIXIN"``). Platform-specific allowlist matching goes through
``_entry_matches``; env-seeded allowlists that must be re-read live go through
``_live_dm_allow_from``.

Every env read is profile-scoped and fails closed: under multiplexing ``os.environ`` holds the
DEFAULT profile's opt-in, which must never open a secondary bot's DMs (#93522).
"""

from __future__ import annotations

from typing import Iterable

from gateway.platforms._shared import get_scoped_secret

OPTIN_TRUTHY = frozenset({"true", "1", "yes"})


class OwnAccessPolicyMixin:
    ALLOW_ALL_ENV_PREFIX: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # A host that forgets its prefix would silently read ``_ALLOW_ALL_USERS`` and deny every
        # open-DM deployment whose setup wrote ``<PLATFORM>_ALLOW_ALL_USERS`` — refuse at class creation.
        if not str(cls.ALLOW_ALL_ENV_PREFIX or "").strip():
            raise TypeError(f"{cls.__qualname__} mixes in OwnAccessPolicyMixin but sets no ALLOW_ALL_ENV_PREFIX")

    @property
    def enforces_own_access_policy(self) -> bool:
        return True

    def _allow_all_env_names(self) -> tuple[str, ...]:
        return ("GATEWAY_ALLOW_ALL_USERS", f"{self.ALLOW_ALL_ENV_PREFIX}_ALLOW_ALL_USERS")

    def _open_dm_opted_in(self) -> bool:
        return any(str(get_scoped_secret(name, "") or "").strip().lower() in OPTIN_TRUTHY
                   for name in self._allow_all_env_names())

    def _entry_matches(self, entries: Iterable[str], target: str) -> bool:
        return target in entries

    def _live_dm_allow_from(self) -> Iterable[str]:
        return self._allow_from

    def _is_dm_allowed(self, sender_id: str) -> bool:
        """Strict DM authorization — pairing does not imply access."""
        if self._dm_policy == "allowlist":
            return self._entry_matches(self._live_dm_allow_from(), sender_id)
        return self._dm_policy == "open" and self._open_dm_opted_in()

    def _is_dm_intake_allowed(self, sender_id: str) -> bool:
        """Whether a DM may reach gateway intake; ``pairing`` admits everyone with a principal so the
        handshake can run (the pairing gate itself runs later). A blank principal is never admitted."""
        principal = str(sender_id or "").strip()
        if not principal:
            return False
        return self._dm_policy == "pairing" or self._is_dm_allowed(principal)

    def _is_group_allowed(self, chat_id: str, sender_id: str = "") -> bool:
        """``disabled``/``pairing``/unknown policies never forward group traffic at intake."""
        if self._group_policy == "allowlist":
            return self._entry_matches(self._group_allow_from, chat_id)
        return self._group_policy == "open"
