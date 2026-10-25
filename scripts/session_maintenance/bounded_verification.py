"""Explicitly bounded cold-snapshot verification. No mutations or service effects.

Caller must supply standalone cold snapshots and a pre-approved, independently
recorded UID budget. This supplements (does not weaken) the strict verifier.
"""
from contextlib import closing
from dataclasses import dataclass, asdict
import verification as base


@dataclass(frozen=True)
class UIDBudget:
    before_cursor: int
    max_id: int
    max_changes: int
    expected_after_cursor: int | None = None
    expected_changes: int | None = None


def protected_catalog(c):
    families = ('messages_fts', 'messages_fts_trigram', 'messages_fts_cjk')
    allowed = base.FTS_TABLES | {f + suffix for f in families
                                for suffix in ('_src', '_insert', '_update', '_delete')}
    return sorted(r for r in c.execute(
        'SELECT type,name,tbl_name,sql FROM sqlite_master') if r[1] not in allowed)


def validate_budget(budget):
    base.require(type(budget.before_cursor) is int and type(budget.max_id) is int
                 and type(budget.max_changes) is int and
                 0 <= budget.before_cursor <= budget.max_id and budget.max_changes >= 0,
                 'Invalid independent UID budget')
    for value in (budget.expected_after_cursor, budget.expected_changes):
        base.require(value is None or type(value) is int and value >= 0,
                     'Invalid exact UID expectation')
    base.require(budget.expected_after_cursor is None or
                 budget.before_cursor <= budget.expected_after_cursor <= budget.max_id,
                 'Expected cursor outside approved range')
    base.require(budget.expected_changes is None or budget.expected_changes <= budget.max_changes,
                 'Expected changes exceed approved ceiling')


def verify_bounded(before_path, after_path, budget):
    validate_budget(budget)
    with closing(base.open_snapshot(before_path)) as a, closing(base.open_snapshot(after_path)) as b:
        old = dict(a.execute('SELECT key,value FROM state_meta'))
        new = dict(b.execute('SELECT key,value FROM state_meta'))
        raw = old.get(base.CURSOR, '0')
        base.require(isinstance(raw, str) and base.re.fullmatch(r'0|[1-9][0-9]*', raw),
                     'Invalid baseline UID cursor')
        base.require(int(raw) == budget.before_cursor, 'Unexpected baseline UID cursor')
        if base.DONE in new and base.DONE not in old:
            endpoint = max(budget.before_cursor, a.execute('SELECT COALESCE(MAX(id),0) FROM messages').fetchone()[0])
        else:
            raw = new.get(base.CURSOR, '0')
            base.require(isinstance(raw, str) and base.re.fullmatch(r'0|[1-9][0-9]*', raw),
                         'Invalid final UID cursor')
            endpoint = int(raw)
        base.require(endpoint <= budget.max_id, 'UID range exceeds independent budget')
        if budget.expected_after_cursor is not None:
            base.require(endpoint == budget.expected_after_cursor, 'Unexpected final UID cursor')
        base.require(protected_catalog(a) == protected_catalog(b), 'Protected catalog changed')
    result = base.verify_snapshots(before_path, after_path)
    delta = result['uid_backfill']
    base.require(delta['rows'] <= budget.max_changes, 'UID row count exceeds independent budget')
    if delta['rows']:
        base.require(budget.before_cursor < delta['first_id'] <= delta['last_id'] <= budget.max_id,
                     'UID changed outside independent range')
    if budget.expected_changes is not None:
        base.require(delta['rows'] == budget.expected_changes, 'Unexpected UID change count')
    result['independent_uid_budget'] = asdict(budget)
    result['contract_version'] = 'bounded-maintenance-1'
    return result
