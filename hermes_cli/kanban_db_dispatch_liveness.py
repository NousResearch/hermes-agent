"""Worker liveness probes: the restart-stable process fingerprint and the claim/signalling
verdicts built on it (exact recycle detection, sleep-drift and read-disagreement rescue).

Split out of ``hermes_cli.kanban_db_dispatch`; ``kanban_db_dispatch`` re-exports every name so
monkeypatching ``kanban_db_dispatch.<name>`` keeps working. The origin module is reached
late-bound inside ``_worker_alive`` (import-cycle breaking).

The split is deliberate about which side may be permissive: claim liveness (``_worker_alive``)
tolerates a drifted or unreadable start-time reading, while the signalling path
(``_pid_recycled``) stays exact — a stranger is never signalled (#118326, #99558).
"""

from __future__ import annotations

from typing import Optional


# ``worker_started_at`` value for a spawn whose fingerprint could not be captured. Distinct from the
# NULL legacy row (pre-fingerprint spawn): such a worker is held (its claim is never released beside
# the live PID) but NEVER signalled — missing process identity is refusal, not permission (#99558).
UNVERIFIED_WORKER_FINGERPRINT = "unverified"

# psutil's macOS ``create_time()`` reading of an *unchanged* live process drifts forward across every
# sleep/wake by the accumulated sleep (#118326: after 27 min asleep the stale-claim reaper read the
# drifted fingerprint as a recycled PID, released a live worker's claim, and a duplicate worker spawned
# beside it). A start-time gap proves recycling only when gross — PID allocation is sequential, so a
# reused number's start lands minutes-to-hours away (the reasoning #111454 applies to the cron ledger).
# Claim liveness tolerates the drift; the signalling path (``_pid_recycled``) stays exact so a stranger
# is never signalled, and a claim wrongly held against a recycled PID is still released once the
# heartbeat goes stale (#29747 backstop).
_SAME_WORKER_START_DRIFT_CEILING = 12 * 3600 * 100  # centiseconds; ~ a long overnight sleep


def _process_fingerprint(pid: int) -> Optional[str]:
    """Restart-stable identity of a live process: ``"<instantiation epoch>|<start time>"``. The start
    time alone (``/proc/<pid>/stat`` field 22 on Linux) is clock ticks since THIS boot, so a row that
    survives a reboot could match an unrelated process with the same PID and the same tick value;
    ``gateway.drain_control.current_instantiation_epoch`` (``boot_id`` + PID-1 start) changes on every
    reboot / container recreate, so the composed value never survives one. ``None`` when unreadable."""
    from gateway.drain_control import current_instantiation_epoch
    from gateway.status import get_process_start_time
    start = get_process_start_time(int(pid))
    if start is None:
        return None
    return f"{current_instantiation_epoch()}|{start}"


def _worker_alive(pid: Optional[int], started_at) -> bool:
    """True when ``pid`` is live AND is still the worker we spawned. ``started_at`` is the fingerprint
    recorded by ``_set_worker_pid``; after a reboot (or any PID recycle) an unrelated process can own
    the number, so bare existence is never enough to extend a claim or to signal. A legacy row without
    a fingerprint keeps the existence answer: killing it is the pre-fingerprint behaviour and the row is
    rewritten with a fingerprint on its next spawn. An UNVERIFIED spawn also keeps the existence answer
    (a claim is never released beside a possibly-live worker) but ``_terminate_reclaimed_worker``
    refuses to signal it. A start-time reading that drifted within ``_SAME_WORKER_START_DRIFT_CEILING``
    under a macOS sleep/wake (forward) or within the shared read-disagreement tolerance (either
    direction) is still our worker (#118326) — unless two readable epoch witnesses disagree."""
    from hermes_cli import kanban_db as _kb
    if not _kb._pid_alive(pid):
        return False
    if started_at == UNVERIFIED_WORKER_FINGERPRINT:
        return True
    return not _pid_recycled(pid, started_at) or _drifted_spawn_still_matches(pid, started_at)


def _drifted_spawn_still_matches(pid: int, started_at) -> bool:
    """Same incarnation despite a start-time reading drift: no readable-epoch disagreement and either
    a forward delta within ``_SAME_WORKER_START_DRIFT_CEILING`` (accumulated macOS sleep/wake) or a
    near-match within the read-disagreement tolerance every other liveness surface already shares
    (``gateway.status.start_time_fingerprints_match``) — the reaper can read the same live worker's
    start time ~1 s above OR below the spawn-time reading (#118326). Exact matches and legacy
    integer fingerprints are decided by ``_pid_recycled`` before this runs; only a live near-match
    on the same boot is rescued. The signalling path never consults this helper — a drifted
    stranger is still not signalled."""
    if not (isinstance(started_at, str) and "|" in started_at):
        return False
    current = _process_fingerprint(int(pid))
    if current is None:
        return False
    recorded_epoch, _, recorded_start = started_at.partition("|")
    current_epoch, _, current_start = current.partition("|")
    # Two readable witnesses that disagree prove a reboot. An EMPTY half on either side is a
    # missing witness, not a differing one — a row written before this host had one (the
    # ``"|<start>"`` rows in flight across an upgrade) or a reader that cannot read any today
    # (``""`` disables the epoch check by contract, never fail-closed). Neither releases a claim
    # beside a live worker; neither authorizes a signal, which ``_pid_recycled`` decides exactly.
    if recorded_epoch and current_epoch and recorded_epoch != current_epoch:
        return False
    try:
        delta = int(current_start) - int(recorded_start)
    except ValueError:
        return False
    if 0 < delta <= _SAME_WORKER_START_DRIFT_CEILING:
        return True
    # A recycled PID re-starts after the recorded worker died, so it reads newer, never older;
    # only the same live worker can read below its spawn-time record. The negative half is the
    # cross-process read disagreement, bounded by the shared comparator's tolerance instead of
    # the 12 h sleep-drift ceiling.
    from gateway.status import start_time_fingerprints_match
    return start_time_fingerprints_match(int(recorded_start), int(current_start))


def _pid_recycled(pid: Optional[int], started_at) -> bool:
    """True when a live ``pid`` is NOT the process fingerprinted at spawn (or the fingerprint can no
    longer be read). Signalling it would hit a stranger. ``None`` fingerprint = legacy row, never
    recycled; the UNVERIFIED marker is always foreign. An integer fingerprint (rows written before the
    boot witness was added) compares the start time only."""
    if started_at is None or not pid:
        return False
    if started_at == UNVERIFIED_WORKER_FINGERPRINT:
        return True
    if isinstance(started_at, str) and "|" in started_at:
        return _process_fingerprint(int(pid)) != started_at
    from gateway.status import _start_times_agree, get_process_start_time
    current = get_process_start_time(int(pid))
    if current is None:
        return True
    try:
        return not _start_times_agree(current, started_at)
    except (TypeError, ValueError):
        return True
