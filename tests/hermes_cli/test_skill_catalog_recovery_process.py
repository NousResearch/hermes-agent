"""Recovery after real writer exit retains one exact phase and admission."""
import json
import multiprocessing
import os
import uuid
from pathlib import Path

import pytest

from hermes_cli.skill_catalog.mutations import Mutations
from hermes_cli.skill_catalog.projection import Projection
from hermes_cli.skill_catalog.validation import CatalogError
from tests.hermes_cli.test_skill_catalog_mutations import operation
from tests.hermes_cli.test_skill_catalog_projection import entry, record, review
from tests.hermes_cli.test_skill_catalog_recovery import tree_digest


def interrupted_writer(home, action, phase, pipe):
    """A separate serving process exits without running any context cleanup."""
    home = Path(home)
    target, staging, backup = home / 'skills/new', home / 'staged', home / 'backup'
    target.parent.mkdir(parents=True, exist_ok=True)
    owner = entry(home)
    projection = Projection(home)
    projection.start(owner)
    if action != 'install':
        target.mkdir(parents=True)
        (target / 'SKILL.md').write_bytes(b'old')
    if action != 'uninstall':
        staging.mkdir()
        (staging / 'SKILL.md').write_bytes(b'new')
    records = [] if action == 'install' else [record(owner, 'new')]
    projection.publish(owner, records)
    plan = review(owner, action=action, installation_revision=None if action == 'install' else 'receipt-1')
    op = uuid.uuid4().hex
    api = Mutations(owner, projection)
    admission = projection.reserve(owner, plan, operation_id=op)
    with api.commit(admission, op):
        if phase != 'before':
            if action != 'install':
                os.replace(target, backup)
            if phase == 'after' and action != 'uninstall':
                os.replace(staging, target)
        pipe.send(dict(op=op, plan=plan, plugin_key=owner.plugin_key))
        pipe.close()
        os._exit(0)


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('action,phase', [
    ('install', 'before'), ('install', 'after'), ('update', 'before'),
    ('update', 'between'), ('update', 'after'), ('uninstall', 'before'), ('uninstall', 'after'),
])
def test_process_death_reregistration_recovers_only_remaining_phase(tmp_path, action, phase):
    ctx = multiprocessing.get_context('spawn')
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=interrupted_writer, args=(str(tmp_path), action, phase, send))
    process.start()
    send.close()
    try:
        assert receive.poll(15)
        saved = receive.recv()
        process.join(15)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        receive.close()
    projection = Projection(tmp_path)
    owner = entry(tmp_path, saved['plugin_key'])
    handle = projection.start(owner)
    api = Mutations(owner, projection)
    target, staging = tmp_path / 'skills/new', tmp_path / 'staged'
    try:
        projection.publish(owner, [] if not target.exists() else [record(owner, 'new')])
        admission = projection.reserve(owner, saved['plan'], operation_id=saved['op'], recovery=True)
        with pytest.raises(CatalogError):
            with api.commit(admission, saved['op']):
                pytest.fail('new-generation recovery bypassed journal hold')
        with api.recover(saved['op']) as recovery:
            expected = None if not target.exists() else tree_digest({'SKILL.md': b'old' if phase == 'before' else b'new'})
            assert recovery.observed_revision == expected
            local = operation(owner, saved['op'], action=action, state='pending', outcome=None,
                managed_state='removed' if action == 'uninstall' else 'installed',
                installation_revision=None if action == 'uninstall' else 'revision-1')
            if phase == 'after':
                recovery.decide(dict(kind='reconciled', journal_digest=expected, result=local))
            else:
                recovery.decide(dict(kind='retry_effect'))
                if phase == 'before' and action != 'install':
                    os.replace(target, tmp_path / 'backup')
                if action != 'uninstall':
                    os.replace(staging, target)
                recovery.settle(local)
        if action == 'uninstall':
            assert not target.exists()
        else:
            assert (target / 'SKILL.md').read_bytes() == b'new'
        assert len(projection.read()['reservations']) == 1
        assert projection.read()['reservations'][admission['reservation_id']]['local_result'] == local
    finally:
        api.close()
        owner.stopped = True
        projection.stop(owner)
        handle.close()
