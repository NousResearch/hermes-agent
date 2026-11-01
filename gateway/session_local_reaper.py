"""Finalize expired UI rows at canonical-owner bootstrap, before any viewer can attach.

Peer refresh stores TUI/Desktop routes as ``local``. Their private receipts, rather than
that shared source label, retain which lifecycle owns them. No producer or CLI/ACP sweep.
"""
import asyncio
import math
from pathlib import Path
import threading


def _startup_ttl(authority):
    from gateway.run import _load_gateway_config
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway._env import env_float
    from utils import is_truthy_value

    token = set_hermes_home_override(str(Path(authority.db.db_path).resolve().parent))
    try:
        dashboard = (_load_gateway_config() or {}).get('dashboard') or {}
    finally:
        reset_hermes_home_override(token)
    dashboard = dashboard if isinstance(dashboard, dict) else {}
    if not is_truthy_value(dashboard.get('startup_orphan_sweep'), default=True):
        return 0
    try:
        grace = float(dashboard.get('ws_orphan_reap_grace_s', 20))
    except (ValueError, TypeError):
        grace = 20
    grace = env_float('HERMES_TUI_WS_ORPHAN_REAP_GRACE_S', grace)
    ttl = env_float('HERMES_TUI_SESSION_TTL_S', 6 * 3600)
    return ttl if math.isfinite(ttl) and ttl > 0 and grace > 0 else 0


def _reap_unpublished(authority, stopped):
    from gateway.session_local_recovery import local_receipt_ids, owned_receipt
    from gateway.session_policy import restore_policy
    from hermes_state_local_reap import _reap_idle_local_session
    from hermes_state_runtime import RuntimeStoreError

    ttl = _startup_ttl(authority)
    if not ttl:
        return
    for sid in local_receipt_ids(authority):
        if stopped():
            return
        try:
            receipt = owned_receipt(authority, sid)[0]
            policy = restore_policy(receipt['policy'])
            if policy.source in ('tui', 'gui'):
                _reap_idle_local_session(authority.db, epoch=authority.epoch, receipt=receipt,
                                        ttl=ttl, stopped=stopped)
        except RuntimeStoreError as exc:
            if exc.reason not in ('profile_mismatch', 'storage_unavailable'):
                raise  # a superseded epoch must never publish an old authority
        except (KeyError, TypeError, ValueError, AttributeError):
            continue  # the regular restore logs and refuses malformed/foreign receipts


async def reap_startup_local_sessions(authority):
    """Join the off-loop writer even on cancellation: an unpublished owner cannot retire it.

The initializer holds exclusive profile ownership and has not published this authority;
there are no current viewers. Epoch, receipt, activity and work checks share the end txn.
"""
    stopped = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(_reap_unpublished, authority, stopped.is_set))
    try:
        await asyncio.shield(worker)
    except asyncio.CancelledError:
        stopped.set()
        joined = asyncio.gather(worker, return_exceptions=True)
        while not joined.done():
            try:
                await asyncio.shield(joined)
            except asyncio.CancelledError:
                continue  # repeated cancellation still cannot abandon an unpublished writer
        raise
