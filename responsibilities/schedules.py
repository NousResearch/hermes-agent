"""Reconcile file-owned declarations into the native cron store."""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from cron import jobs
from hermes_time import now
from responsibilities.schedule_policy import _guard_rejection
from responsibilities.common import get_responsibilities_root
from responsibilities.packages import scan_workspace_responsibilities

logger = logging.getLogger(__name__)


def parsed_schedule(declaration):
    text = declaration['schedule']
    schedule = jobs.parse_schedule(text, bare_duration_is_once=True)
    zone = declaration.get('timezone')
    if zone:
        schedule['timezone'] = zone
        if schedule['kind'] == 'once':
            # A declared zone only applies to a naive ISO wall time (validator enforces it).
            schedule['run_at'] = datetime.fromisoformat(text).replace(tzinfo=ZoneInfo(zone)).isoformat()
    return schedule


def reconcile():
    root = get_responsibilities_root()
    snapshot = scan_workspace_responsibilities(root)
    entries = {entry.name: entry for entry in snapshot.entries}
    errors = dict(snapshot.package_errors)
    with jobs._jobs_lock():
        rows = jobs.load_jobs()
        owned = {}
        retired = set()
        for job in rows:
            owner = job.get('responsibility')
            if not owner:
                continue
            job['workdir'] = None  # The agent uses its profile workspace; only guards run beside scripts.
            if job.get('state') != 'completed' and job.get('failure_streak', 0) >= 3:
                job.update(enabled=False, state='paused', next_run_at=None,
                           responsibility_failure_paused=True, paused_reason='Three consecutive responsibility failures')
            name, trigger = owner['name'], owner['trigger']
            owned[name, trigger] = job
            entry = entries.get(name)
            if name in snapshot.package_errors:
                continue
            if entry is None:
                retired.add(job['id'])
                continue
            if 'schedules' in entry.schedule_errors or trigger + '.yaml' in entry.schedule_errors:
                continue
            if trigger not in entry.schedules:
                retired.add(job['id'])
        for name, entry in entries.items():
            for filename, error in entry.schedule_errors.items():
                errors[f'{name}/schedules/{filename}'] = error
            for trigger, frozen in entry.schedules.items():
                declaration = dict(frozen)
                old = owned.get((name, trigger))
                signature = [declaration['schedule'], declaration.get('timezone')]
                same_schedule = old and old['responsibility']['schedule_signature'] == signature
                try:
                    schedule = old['schedule'] if same_schedule else parsed_schedule(declaration)
                except (ValueError, OverflowError) as exc:
                    errors[f'{name}/schedules/{trigger}.yaml'] = str(exc)
                    continue
                package = root / name
                rejection = _guard_rejection(f'{name}/{trigger}', schedule, declaration, entry.scripts, now=now())
                if rejection:
                    errors[f'{name}/schedules/{trigger}.yaml'] = rejection
                    if old:
                        old.update(enabled=False, responsibility_disarmed=True)
                    continue
                if old and old.pop('responsibility_disarmed', False):
                    repeat = old.get('repeat') or {}
                    exhausted = repeat.get('times') is not None and repeat.get('completed', 0) >= repeat['times']
                    if not exhausted and old.get('state') == 'scheduled':
                        old['next_run_at'] = jobs.compute_next_run(schedule)
                        old['enabled'] = old['next_run_at'] is not None
                if old and old['responsibility']['content_hash'] == declaration['content_hash']:
                    continue
                if old and old.pop('responsibility_failure_paused', False) and old.get('state') != 'completed':
                    next_run = jobs.compute_next_run(schedule)
                    old.update(state='scheduled', enabled=next_run is not None, next_run_at=next_run,
                               failure_streak=0, paused_reason=None, paused_at=None)
                owner = dict(name=name, trigger=trigger,
                    content_hash=declaration['content_hash'], schedule_signature=signature,
                    report=declaration.get('deliver', 'muted'))
                fields = dict(
                    prompt=declaration['prompt'], name=f'{name}/{trigger}',
                    deliver='local' if declaration.get('deliver') == 'muted' else declaration.get('deliver', 'local'),
                    script=str(package / declaration['script']) if declaration.get('script') else None,
                    workdir=None,
                )
                try:
                    if old is None or not same_schedule:
                        native_schedule = schedule.get('run_at') or schedule.get('expr') or f"every {schedule['minutes']}m"
                        job = jobs.create_job(schedule=native_schedule, repeat=declaration.get('repeat'),
                            responsibility=owner, attach_to_session=True, schedule_timezone=schedule.get('timezone'),
                            replace_job_id=old['id'] if old else None, **fields)
                        if old:
                            retired.add(old['id'])
                        job.update(schedule=schedule, next_run_at=jobs.compute_next_run(schedule))
                        rows.append(job)
                    else:
                        job = old
                        job.update(fields)
                        job['repeat']['times'] = declaration.get('repeat', 1 if schedule['kind'] == 'once' else None)
                    job['attach_to_session'] = True
                    job['responsibility'] = owner
                    job['updated_at'] = now().isoformat()
                except ValueError as exc:
                    errors[f'{name}/schedules/{trigger}.yaml'] = str(exc)
        jobs.save_jobs([job for job in rows if job['id'] not in retired], removed_ids=retired)
    for path, error in errors.items():
        logger.warning('Responsibility %s: %s', path, error)
    return errors
