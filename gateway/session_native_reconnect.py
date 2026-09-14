"""Recheck original native queues when their receiving adapter is published."""
import logging

from gateway.session_authorities import all_authorities, owner_scope

logger = logging.getLogger(__name__)


async def recover_adapter_native_inputs(runner, platform, adapter, *, profile=None):
    """Reconnect is a preflight trigger, never a new admission or outcome decision.

    A receiving bot can route to another served home. Select by saved transport
    provenance, then let each existing authority validate its original queue in
    its own scope. Do not run startup's local/hosted/passive recovery here.
    Unknown work stays paused; a successful preflight only asks the existing
    scheduler to inspect its FIFO and never promises execution or delivery.
    """
    def current():
        mapping = runner.adapters if profile is None else getattr(runner, '_profile_adapters', {}).get(profile, {})
        return mapping.get(platform) is adapter and not getattr(runner, '_draining', False)

    if not current():
        return
    for authority in all_authorities(runner):
        if authority is None or not current():
            return
        try:
            with owner_scope(authority):
                # Inspect already-owned ledgers only. This transport's native
                # rows are the candidates, not every queue on the same platform.
                rows = authority.db._read_all("""SELECT DISTINCT target_session_id,
                    json_extract(payload_json, '$.native_text_v1.route') AS route
                    FROM session_admissions WHERE status='queued'
                    AND json_extract(payload_json, '$.native_text_v1.source.platform')=?
                    AND json_extract(payload_json, '$.native_text_v1.provenance.transport_profile') IS ?
                    AND json_type(payload_json, '$.native_text_v1.automation') IS NULL""",
                    (platform.value, profile))
                bindings = []
                for row in rows:
                    entry = runner.session_store.lookup_by_session_key(row['route'])
                    if (entry is not None and entry.origin is not None
                            and entry.origin.platform == platform
                            and authority.logical_owner(entry.session_id) == row['target_session_id']):
                        bindings.append((row['target_session_id'], entry.origin, adapter))
                if bindings and current():
                    # check_native_route also checks this exact adapter AFTER
                    # awaited sender/role authorization, fencing superseded hooks.
                    outcomes = await authority.recover_native_sessions(bindings)
                    refused = sum(verdict not in {'ready', 'active'} for verdict in outcomes.values())
                    if refused:
                        logger.debug('Native reconnect left %d candidate queues paused', refused)
        except Exception as exc:
            # Refusal or missing state must not undo a successful adapter install.
            # No payload-bearing exception text, retry loop or synthesized input.
            logger.debug('Native reconnect recovery deferred (%s)', type(exc).__name__)
