"""Adapter-facing pairing: issue a code without a chat reply (``BasePlatformAdapter.request_pairing``)
and tell adapters when one of their senders is approved or revoked (``on_pairing_changed``).

Both serve screen-first platforms, where the person pairing reads a device rather than a DM. The code
comes from the same path an unauthorized DM takes (``run_inbound.py::_hm_admit_event`` →
``_hm_offer_pairing_code``): authorization check, ``unauthorized_dm_behavior``, pairing store, rate
limit and pending cap. Approvals are written by other processes (``hermes pairing approve``, the
dashboard) that notify nothing, so ``_pairing_watcher`` polls the stores of the adapters that override
``on_pairing_changed``; with none, it reads nothing.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from gateway.pairing import CODE_TTL_SECONDS
from gateway.platforms.base import BasePlatformAdapter, PairingOffer
from gateway.run_inbound_unauthorized import pairing_approve_command, pairing_profile_arg
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

_PAIRING_WATCH_INTERVAL_SECS = 2.0


def _watches_pairing(adapter) -> bool:
    """Adapters opt in by overriding ``on_pairing_changed``; the rest never cost a store read."""
    return isinstance(adapter, BasePlatformAdapter) and (
        type(adapter).on_pairing_changed is not BasePlatformAdapter.on_pairing_changed)


class GatewayPairingMixin:
    """Runner side of ``request_pairing`` and ``on_pairing_changed``."""

    def _hm_request_pairing(self, source: SessionSource) -> Optional[PairingOffer]:
        """The code :meth:`_hm_offer_pairing_code` would send an unauthorized DM sender, returned
        instead of sent. ``None`` wherever the DM path would send no code."""
        if not source.user_id or source.chat_type != "dm" or getattr(source, "is_bot", False):
            return None
        if getattr(source, "profile_route_rejected", False) is True:
            return None
        if self._is_user_authorized_for_source(source):
            return None
        if self._get_unauthorized_dm_behavior(source.platform, profile=source.profile) != "pair":
            return None
        platform_name = source.platform.value
        store = self._pairing_store_for(source)
        if store is None or store._is_rate_limited(platform_name, source.user_id):
            return None
        code = store.generate_code(platform_name, source.user_id, source.user_name or "")
        if not code:
            # As on the DM path: a sender refused once stays quiet for the rate-limit window.
            store._record_rate_limit(platform_name, source.user_id)
            return None
        return PairingOffer(code=code, expires_in=CODE_TTL_SECONDS,
                            command=pairing_approve_command(platform_name, code, pairing_profile_arg(store)))

    def _make_pairing_requester(self, profile_name: Optional[str] = None):
        """Bind :meth:`_hm_request_pairing` to one adapter the way its message handler is bound:
        canonicalize FIRST, then decide under the profile that owns the decision (a secondary's own
        bot; the routed profile for the shared primary under multiplexing; else the launch profile)."""
        from gateway.run import _async_profile_runtime_scope, get_hermes_home
        if profile_name is not None:
            profile_home = self._routed_profile_home(profile_name)

            async def _secondary(source: SessionSource) -> Optional[PairingOffer]:
                self._canonicalize(source, transport_profile=profile_name)
                async with self._async_scope_or_null(_async_profile_runtime_scope, profile_home):
                    return self._hm_request_pairing(source)

            return _secondary
        if self._multiplex_on():
            default_home = Path(get_hermes_home())

            async def _primary(source: SessionSource) -> Optional[PairingOffer]:
                runtime_home = self._admit_primary_source(source, default_home)
                if runtime_home is None:
                    return None  # rejected route: same disposition as the message ingress gate
                async with _async_profile_runtime_scope(runtime_home):
                    return self._hm_request_pairing(source)

            return _primary

        async def _standalone(source: SessionSource) -> Optional[PairingOffer]:
            with self._standalone_launch_scope():
                return self._hm_request_pairing(source)

        return _standalone

    async def _pairing_watcher(self, interval: float = _PAIRING_WATCH_INTERVAL_SECS) -> None:
        """Supervised: report approvals and revocations to adapters that override ``on_pairing_changed``."""
        seen: Dict[Tuple[Optional[str], str], frozenset] = {}
        while self._running:
            await asyncio.sleep(interval)
            if not self._running:
                return
            try:
                await self._pairing_watch_tick(seen)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Pairing watch failed; retrying next cycle", exc_info=True)

    async def _pairing_watch_tick(self, seen: Dict[Tuple[Optional[str], str], frozenset]) -> None:
        """One poll of each watching adapter's pairing store (the primary's store for a primary adapter,
        the profile's for a secondary). The first look at a (profile, platform) only records the approved
        set: approvals from before the watch started are not news."""
        targets = [(None, adapter) for adapter in (getattr(self, "adapters", None) or {}).values()]
        for profile, adapters in (getattr(self, "_profile_adapters", None) or {}).items():
            targets += [(profile, adapter) for adapter in (adapters or {}).values()]
        for profile, adapter in targets:
            if not _watches_pairing(adapter):
                continue
            store = ((getattr(self, "pairing_stores", None) or {}).get(profile) if profile
                     else getattr(self, "pairing_store", None))
            if store is None:
                continue
            platform_name = adapter.platform.value
            approved = store.approved_user_ids(platform_name)
            previous = seen.get((profile, platform_name))
            seen[(profile, platform_name)] = approved
            if previous is None:
                continue
            changes = [(user_id, True) for user_id in sorted(approved - previous)]
            changes += [(user_id, False) for user_id in sorted(previous - approved)]
            for user_id, now_approved in changes:
                await self._notify_pairing_changed(adapter, profile, user_id, now_approved)

    async def _notify_pairing_changed(
            self, adapter, profile: Optional[str], user_id: str, approved: bool) -> None:
        """Run one adapter's hook under its profile's scope; a failing hook never stops the watch."""
        from gateway.run import _async_profile_runtime_scope
        profile_home = self._routed_profile_home(profile) if profile else None
        try:
            async with self._async_scope_or_null(_async_profile_runtime_scope, profile_home):
                await adapter.on_pairing_changed(user_id, approved)
        except Exception:
            logger.warning("[%s] on_pairing_changed failed", adapter.platform.value, exc_info=True)
