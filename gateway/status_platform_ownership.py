"""Which platform entries of a gateway runtime record belong to the CURRENT process life.

``gateway_state.json`` outlives the process that wrote it, so every reader of a platform map has to
answer one question before it trusts a verdict: did *this* life write it? The writer enforces the
rule once, at the chokepoint every write passes (``platform_map_for_write``); the two readers filter
a map they already hold (``runtime_platforms_owned_by`` for a caller-named live process,
``own_profile_platforms_owned_by`` for a record's own declared identity). One predicate, so the
writer, ``/api/status`` and the messaging card can never disagree about the same record.
"""

from __future__ import annotations

from typing import Any, Optional


def platform_map_for_write(
    platforms: Any, current_record: dict[str, Any], previous_payload: Optional[dict[str, Any]],
    *, clear_profile_platforms: bool = False, drop_profile_platforms: Optional[str] = None,
) -> dict[str, Any]:
    """The platform map a runtime-status write keeps, in the order both rules were always applied.

    ``<profile>:``-keyed clears run first (they only ever reach prefixed entries), then the
    life-boundary rule downgrades whatever still cannot prove it belongs to ``current_record``'s
    process life — so a clear and a boundary in the same write behave exactly as they did inline.
    """
    if not isinstance(platforms, dict):
        platforms = {}
    if clear_profile_platforms or drop_profile_platforms:
        drop_prefix = f"{drop_profile_platforms}:" if drop_profile_platforms else None
        platforms = {
            k: v for k, v in platforms.items()
            if not isinstance(k, str) or ":" not in k
            or (drop_prefix is not None and not k.startswith(drop_prefix))
        }
    return _downgrade_foreign_life_platforms(platforms, current_record, previous_payload)


def _downgrade_foreign_life_platforms(
    platforms: dict[str, Any], current_record: dict[str, Any],
    loaded_record: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """B1: keep a platform verdict only if it can prove it belongs to THIS process life.

    ``gateway_state.json`` outlives its writer, so the first write of a new life merges into the
    PREVIOUS life's record. ``clear_profile_platforms`` only ever dropped ``<profile>:``-keyed
    entries, so the default profile's OWN (flat) adapters — ``api_server`` among them — survived a
    restart carrying the DEAD life's verdict beside the NEW life's ``pid``/``start_time``/
    ``updated_at``: a false green no reader could detect from the file alone.

    At a life boundary — the record being merged into names a different ``(pid, start_time)`` —
    every inherited entry that cannot name THIS life is re-stamped ``starting`` (never a healthy
    state; see ``web_routers.status._HEALTHY_PLATFORM_STATES``) with the old verdict parked under
    ``stale_*``. The entry's writer identity moves to the current life so readers serve the honest
    ``starting`` state instead of dropping the platform entirely. Entries with NO writer stamps
    fail closed. Colon-keyed and flat keys are treated identically. When there is no boundary (the
    record is this life's, or carries no identity at all) every entry is left untouched, which
    keeps ``clear_profile_platforms`` and the stopped-gateway behaviour byte-for-byte.
    """
    if not isinstance(platforms, dict):
        return {}
    loaded = loaded_record if isinstance(loaded_record, dict) else {}
    current_pid = current_record.get("pid")
    current_start = current_record.get("start_time")
    has_loaded_identity = loaded.get("pid") is not None or loaded.get("start_time") is not None
    if not has_loaded_identity or (
        loaded.get("pid") == current_pid and loaded.get("start_time") == current_start
    ):
        return platforms
    for value in platforms.values():
        if not isinstance(value, dict):
            continue
        if (value.get("writer_pid") == current_pid
                and value.get("writer_start_time") == current_start):
            continue
        old_state = value.get("state")
        old_pid = value.get("writer_pid")
        old_start = value.get("writer_start_time")
        value.update(
            state="starting", error_code=None, error_message=None, needs_attention=False,
            stale_state=old_state, stale_writer_pid=old_pid,
            stale_writer_start_time=old_start,
            writer_pid=current_pid, writer_start_time=current_start,
        )
    return platforms


def runtime_platforms_owned_by(platforms: Any, live_pid: Any, live_start: Any) -> dict[str, Any]:
    """Platform entries of ``platforms`` written by the process ``(live_pid, live_start)``.

    The single home of the gateway's writer-identity predicate. Gateway startup preserves plain
    platform entries across restarts, so a runtime-status platform map can carry a DEAD life's
    verdict; ownership is exact ``(writer_pid, writer_start_time)`` equality — the same PID-reuse
    fingerprint the liveness checks use — never a clock heuristic. Fails closed: an entry with no
    writer stamps, or a caller that cannot name the live process, is not owned.
    """
    if not isinstance(platforms, dict) or live_pid is None or live_start is None:
        return {}
    return {
        key: value for key, value in platforms.items()
        if isinstance(value, dict)
        and value.get("writer_pid") == live_pid
        and value.get("writer_start_time") == live_start}


def own_profile_platforms_owned_by(runtime: Any, live_pid: Any) -> dict[str, Any]:
    """A runtime record's OWN platform map, filtered to entries the live process wrote.

    Own-profile readers project the RAW map, so a live-pid record could carry a previous, dead
    life's flat-platform verdict (``api_server: connected`` for a server the current process never
    bound) and certify it healthy. Ownership is the record's OWN declared ``(pid, start_time)`` —
    the exact pair ``write_runtime_status`` stamps into every platform entry — trusted only while
    the resolved live pid agrees, so a lagging record can never drop a live process's entries.
    Identity that cannot be certified (unknown live pid, missing/None declared identity,
    cross-container record) returns the map unchanged: the callers' dead/startup_failed arms
    already cover the not-running cases, and a false drop is the worse failure mode.
    """
    platforms = runtime.get("platforms") if isinstance(runtime, dict) else None
    if not isinstance(platforms, dict):
        return {}
    if not platforms:
        return platforms
    declared_pid = runtime.get("pid")
    declared_start = runtime.get("start_time")
    if (live_pid is None or declared_pid is None or declared_start is None
            or declared_pid != live_pid):
        return platforms
    return runtime_platforms_owned_by(platforms, declared_pid, declared_start)
