"""Bound pending operations recover under the same real writer exclusion."""
import hashlib
import json
import time

import pytest

from hermes_cli.skill_catalog.validation import CatalogError
from tests.hermes_cli.test_skill_catalog_mutations import admitted, operation


def tree_digest(files):
    facts = {name: 'sha256:' + hashlib.sha256(data).hexdigest() for name, data in files.items()}
    return hashlib.sha256(json.dumps(facts, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def test_post_effect_pre_marker_recovery_records_local_commit_without_second_write(tmp_path):
    with admitted(tmp_path) as (projection, owner, api, admission, op):
        target = tmp_path / 'skills/new'
        with pytest.raises(RuntimeError):
            with api.commit(admission, op):
                target.mkdir(parents=True)
                (target / 'SKILL.md').write_bytes(b'approved')
                raise RuntimeError('death before marker')
        before = (target / 'SKILL.md').stat()
        with api.recover(op) as recovery:
            assert recovery.target_relative_path == 'new'
            assert recovery.pre_revision_matches is False
            assert recovery.observed_revision == tree_digest({'SKILL.md': b'approved'})
            recovery.decide(dict(kind='reconciled', journal_digest=recovery.observed_revision,
                                 result=operation(owner, op, state='pending', outcome=None)))
        assert (target / 'SKILL.md').stat() == before
        assert projection.read()['reservations'][admission['reservation_id']]['local_result']['state'] == 'pending'
        assert projection.read()['targets']['new']['installation_revision'] == 'revision-1'


def test_pre_effect_recovery_retries_original_operation_and_settles(tmp_path):
    with admitted(tmp_path) as (projection, owner, api, admission, op):
        with pytest.raises(RuntimeError):
            with api.commit(admission, op):
                raise RuntimeError('death before first rename')
        with api.recover(op) as recovery:
            assert recovery.pre_revision_matches is True
            assert recovery.observed_revision is None
            recovery.decide(dict(kind='retry_effect'))
            (tmp_path / 'skills/new').mkdir(parents=True)
            recovery.settle(operation(owner, op, state='pending', outcome=None))
        assert projection.read()['reservations'][admission['reservation_id']]['operation_id'] == op


def test_mismatched_reconciliation_fails_without_adopting_replacement(tmp_path):
    with admitted(tmp_path) as (projection, _owner, api, _admission, op):
        target = tmp_path / 'skills/new'
        target.mkdir(parents=True)
        (target / 'SKILL.md').write_bytes(b'replacement')
        before = (target / 'SKILL.md').stat()
        with pytest.raises(CatalogError) as caught:
            with api.recover(op) as recovery:
                recovery.decide(dict(kind='reconciled', journal_digest=tree_digest({'SKILL.md': b'approved'}),
                    result=operation(_owner, op, state='pending', outcome=None)))
        assert caught.value.code == 'review_changed'
        assert (target / 'SKILL.md').read_bytes() == b'replacement'
        assert (target / 'SKILL.md').stat() == before
        assert projection.read()['reservations'] == {}
        assert projection.read()['targets'] == {}
        assert api.settlement(op) == 'settled'


@pytest.mark.parametrize('decision', [None, {}, {'kind': 'unknown'}])
def test_missing_or_invalid_recovery_answer_never_authorizes_an_effect(tmp_path, decision):
    with admitted(tmp_path) as (projection, _owner, api, admission, op):
        with pytest.raises(CatalogError) as caught:
            with api.recover(op) as recovery:
                if decision is not None:
                    recovery.decide(decision)
        assert caught.value.code == 'review_changed'
        assert not (tmp_path / 'skills/new').exists()
        assert projection.read()['reservations'][admission['reservation_id']]['local_result'] is None


def test_abandon_keeps_bytes_and_settles_failed(tmp_path):
    with admitted(tmp_path) as (projection, _owner, api, _admission, op):
        target = tmp_path / 'skills/new'
        target.mkdir(parents=True)
        (target / 'SKILL.md').write_bytes(b'foreign')
        with api.recover(op) as recovery:
            recovery.decide(dict(kind='abandon', reason='journal identity differs'))
        assert (target / 'SKILL.md').read_bytes() == b'foreign'
        assert projection.read()['reservations'] == {}
        assert projection.read()['targets'] == {}


def test_recovery_excludes_generic_same_ancestor_and_descendant_writers(tmp_path):
    with admitted(tmp_path) as (projection, owner, api, _admission, op):
        with api.recover(op) as recovery:
            for path in ('new', 'new/child'):
                with pytest.raises(CatalogError):
                    with projection.generic([path], deadline=time.monotonic()):
                        pytest.fail('generic writer entered a recovering target')
            recovery.decide(dict(kind='retry_effect'))
            (tmp_path / 'skills/new').mkdir(parents=True)
            recovery.settle(operation(owner, op, state='pending', outcome=None))


def test_saved_core_local_marker_allows_only_identical_reconciliation(tmp_path):
    with admitted(tmp_path) as (projection, owner, api, admission, op):
        with api.commit(admission, op) as commit:
            target = tmp_path / 'skills/new'
            target.mkdir(parents=True)
            (target / 'SKILL.md').write_bytes(b'approved')
            pending = operation(owner, op, state='pending', outcome=None)
            commit.settle(pending)
        with api.recover(op) as recovery:
            recovery.decide(dict(kind='reconciled', journal_digest=recovery.observed_revision, result=pending))
        assert projection.read()['reservations'][admission['reservation_id']]['local_result'] == pending
        with pytest.raises(CatalogError):
            with api.recover(op) as recovery:
                recovery.decide(dict(kind='retry_effect'))
        assert (target / 'SKILL.md').read_bytes() == b'approved'


def test_terminal_or_foreign_operation_cannot_get_recovery_hold(tmp_path):
    with admitted(tmp_path) as (_projection, owner, api, admission, op):
        with pytest.raises(CatalogError) as caught:
            with api.recover('foreign_operation_0001'):
                pytest.fail('foreign operation entered')
        assert caught.value.code == 'review_changed'
        with api.commit(admission, op) as commit:
            (tmp_path / 'skills/new').mkdir(parents=True)
            commit.settle(operation(owner, op))
        with pytest.raises(CatalogError) as caught:
            with api.recover(op):
                pytest.fail('completed operation entered')
        assert caught.value.code == 'review_changed'
