"""Explicit fail-closed recovery policy; no service effects on import.

This candidate requires a separately approved offline window. A failure after
stop intentionally leaves writers stopped for operator evidence review. It never
rolls back, retries optimize, resumes a CLI, or restarts on a failure path.
"""
import argparse
from dataclasses import asdict
from datetime import UTC, datetime

import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from threading import Event
import uuid
from storage_safety import ensure_directory, maintenance_lock, sync_directory


ROOT = Path(__file__).resolve().parent



def orchestrate(*, initially_active, preflight, stop, maintain, verified, start, healthy, record):
    phase = 'preflight'
    stop_attempted = False
    maintenance_verified = False
    try:
        if type(initially_active) is not bool:
            raise RuntimeError('Initial gateway activity must be known explicitly')
        record({'phase': phase, 'initially_active': initially_active})
        preflight()
        if initially_active:
            phase = 'stopping-gateway'
            record({'phase': phase})
            stop_attempted = True
            stop()
        phase = 'offline-maintenance'
        record({'phase': phase})
        result = maintain()
        if type(result) is not int or result != 0:
            raise RuntimeError('Maintenance did not explicitly return success')
        if verified() is not True:
            raise RuntimeError('Durable maintenance postconditions are not verified')
        maintenance_verified = True
        record({'phase': 'verified-offline', 'maintenance_verified': True})
        if initially_active:
            phase = 'restarting-verified-gateway'
            record({'phase': phase})
            start()
            if healthy() is not True:
                raise RuntimeError('Gateway not active after verified restart')
        record({'status': 'complete', 'maintenance_verified': True,
                'gateway': 'active' if initially_active else 'left-inactive'})
        return 0
    except BaseException as error:
        logging.getLogger(__name__).exception('Verified maintenance/recovery blocked')
        event = {'status': 'blocked', 'phase': phase, 'maintenance_verified': maintenance_verified,
                 'stop_attempted': stop_attempted, 'error_type': type(error).__name__,
                 'recovery': 'manual-review-required; no automatic restart or rollback'}
        try:
            record(event)
        finally:
            print('BLOCKED: inspect preserved receipts and service state. No automatic recovery. '
                  'If gateway stop was attempted, it may remain offline until operator review.', file=sys.stderr)
        return 1


def command(args):
    return subprocess.run(args, text=True, encoding='utf-8', errors='replace',
                          capture_output=True, check=True, timeout=180).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-offline', action='store_true')
    parser.add_argument('--service', required=True, help='Exact user-systemd unit serving this HERMES_HOME')
    parser.add_argument('--uid-budget', type=Path, required=True)
    parser.add_argument('--search', action='append', required=True)
    platforms = parser.add_mutually_exclusive_group(required=True)
    platforms.add_argument('--expect-platform', action='append', help='Independently approved adapter roster (repeatable)')
    platforms.add_argument('--no-platforms', action='store_true', help='Explicitly expect a platform-less gateway')
    args = parser.parse_args()
    if not args.execute_offline:
        parser.error('Requires --execute-offline and an explicitly approved offline window')
    if len(set(args.search)) < 2:
        parser.error('At least two distinct existing search probes are required')
    expected_platforms = frozenset(args.expect_platform or ())
    if any(not name or any(c.isspace() for c in name) for name in expected_platforms):
        parser.error('Expected platform names must be nonempty identifiers without whitespace')
    os.umask(0o077)
    # Importing this module never opens SessionDB or changes any service.
    import offline_rebuild as worker
    from bounded_verification import UIDBudget, validate_budget
    budget = UIDBudget(**json.loads(args.uid_budget.read_text(encoding='utf-8-sig')))
    validate_budget(budget)
    worker.SEARCH_TERMS = tuple(dict.fromkeys(args.search))
    os.environ['HERMES_HOME'] = str(worker.HOME)
    def interrupted(signum, frame):
        raise InterruptedError('Signal interrupted offline maintenance')
    signal.signal(signal.SIGTERM, interrupted)
    root = worker.HOME / 'maintenance'
    with maintenance_lock(worker.DB) as lock:
        ensure_directory(root)
        worker.ROOT = root / ('run-' + uuid.uuid4().hex)
        ensure_directory(worker.ROOT)
        path = worker.ROOT / 'recovery.jsonl'
        with path.open('x', encoding='utf-8') as log:
            def record(event):
                event.update(expected_platforms=sorted(expected_platforms), approved_uid_budget=asdict(budget),
                             service=args.service, target_home=str(worker.HOME))
                event['at'] = datetime.now(UTC).isoformat()
                log.write(json.dumps(event) + '\n')
                log.flush()
                os.fsync(log.fileno())
                sync_directory(path.parent)
            state = command(['systemctl', '--user', 'show', args.service, '-p', 'ActiveState', '--value'])
            if state not in ('active', 'inactive'):
                record({'status': 'blocked', 'reason': 'Unstable or failed initial gateway state; no service action'})
                return 1
            pid = int(command(['systemctl', '--user', 'show', args.service, '-p', 'MainPID', '--value']))
            def preflight():
                if state == 'active' and pid <= 0:
                    raise RuntimeError('Active gateway has no valid MainPID')
                other = [p for p in worker.holders(worker.DB) if p != pid]
                if other:
                    raise RuntimeError('Other database holders remain; quit all CLIs first')
                # Never stop a guessed unit: the requested active service must own this DB.
                if state == 'active' and pid not in worker.holders(worker.DB):
                    raise RuntimeError('Requested service is not a holder of the selected database')
                if state == 'active' and not ready_once(worker.HOME, args.service, 0, expected_platforms):
                    raise RuntimeError('Target must be a healthy standalone launch home with the approved adapter roster; multiplexed profiles are unsupported')
            print('Recovery log: ' + str(path), flush=True)
            generation = {}
            def start():
                generation['not_before'] = datetime.now(UTC).timestamp()
                command(['systemctl', '--user', 'start', args.service])
            return orchestrate(
                initially_active=state == 'active', preflight=preflight,
                stop=lambda: command(['systemctl', '--user', 'stop', args.service]),
                maintain=lambda: worker.main(execute_offline=True, budget=budget, lock=lock),
                verified=lambda: verify_receipt(worker.ROOT, budget),
                start=start,
                healthy=lambda: wait_ready(worker.HOME, args.service, generation['not_before'], expected_platforms),
                record=record,
            )


def process_start(pid):
    from gateway.status import get_process_start_time
    return get_process_start_time(pid)


def fresh(record, not_before):
    try:
        stamp = datetime.fromisoformat(record['updated_at'])
        return stamp.tzinfo is not None and stamp.timestamp() >= not_before
    except (TypeError, ValueError, KeyError):
        return False


def platform_ready(record, pid, started, not_before):
    return (isinstance(record, dict) and record.get('state') == 'connected'
            and not record.get('needs_attention') and not record.get('error_code')
            and record.get('writer_pid') == pid and record.get('writer_start_time') == started
            and fresh(record, not_before))


def ready_once(home, service, not_before, expected_platforms):
    """Require a standalone launch-home receipt for the actual new service generation."""
    if command(['systemctl', '--user', 'is-active', service]) != 'active':
        return False
    pid = int(command(['systemctl', '--user', 'show', service, '-p', 'MainPID', '--value']))
    started = process_start(pid)
    try:
        state = json.loads((home / 'gateway_state.json').read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return False
    if not isinstance(state, dict) or not isinstance(state.get('platforms'), dict):
        return False
    platforms = state['platforms']
    return (pid > 0 and type(started) is int and started > 0
            and state.get('pid') == pid and state.get('start_time') == started
            and state.get('kind') == 'hermes-gateway' and state.get('served_profiles') == []
            and fresh(state, not_before) and state.get('gateway_state') == 'running'
            and state.get('session_store', {}).get('status') == 'ok'
            and set(platforms) == set(expected_platforms)
            and all(platform_ready(p, pid, started, not_before) for p in platforms.values()))


def wait_ready(home, service, not_before, expected_platforms):
    deadline = time.monotonic() + 90
    delay = Event()
    while time.monotonic() < deadline:
        if ready_once(home, service, not_before, expected_platforms):
            return True
        delay.wait(1)
    return False


def verify_receipt(root, budget):
    """Require the single completed receipt of this exclusive run, not exit code zero alone."""
    from dataclasses import asdict
    paths = list(root.glob('offline-*/receipt.json'))
    if len(paths) != 1:
        return False
    report = json.loads(paths[0].read_text(encoding='utf-8-sig'))
    before = report.get('search_before', {})
    verification = report.get('verification', {})
    return (report.get('status') == 'complete'
            and report.get('phase') == 'complete-offline-checks'
            and report.get('backup_verified') is True
            and report.get('conversation_content_and_protected_fields_unchanged') is True
            and verification.get('status') == 'passed'
            and verification.get('independent_uid_budget') == asdict(budget)
            and report.get('fts_health', {}).get('status') == 'passed'
            and report.get('layout_current') is True
            and report.get('layout_drift') == []
            and report.get('pending_markers') == []
            and report.get('quick_check') == ['ok']
            and len(before) >= 2 and all(before.values())
            and before == report.get('search_after'))


if __name__ == '__main__':
    sys.exit(main())
