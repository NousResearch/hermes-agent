"""r2 regression: mutable demand must not erase exact attempted delivery evidence."""
import pytest
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_db_surface as s

@pytest.mark.parametrize('expired', [False, True])
def test_attempt_revision_survives_new_demand(tmp_path, monkeypatch, expired):
    path = tmp_path/'board.db'
    monkeypatch.setenv('HERMES_KANBAN_DB', str(path))
    with kbc.connect(path) as c:
        task = kb.create_task(c, title='synthetic')
        rev = s.get_task_source(c, task).current_revision
        receipt = s.ensure_delivery_receipt(c, task_id=task, platform='telegram', chat_id='-100', notifier_profile='a', desired_revision=rev)
        lease = s.claim_delivery_receipt(c, receipt.id, owner_id='first', now=100, lease_seconds=5)
        kb.add_comment(c, task, 'test', 'new demand')
        new = s.get_task_source(c, task).current_revision
        s.ensure_delivery_receipt(c, task_id=task, platform='telegram', chat_id='-100', notifier_profile='a', desired_revision=new)
        args = dict(owner_id=lease.owner_id, owner_epoch=lease.owner_epoch, attempt_id=lease.attempt_id, desired_revision=rev, state='sent', message_id='731', destination_profile='a', delivered_revision=rev, now=106 if expired else 101)
        if expired:
            with pytest.raises(s.DeliveryReceiptUnknown):
                s.claim_delivery_receipt(c, receipt.id, owner_id='next', now=106)
            result = s.reconcile_delivery_outcome(c, receipt.id, **args)
        else:
            result = s.record_delivery_outcome(c, receipt.id, **args)
        assert result.delivered_revision == rev
        assert result.desired_revision == new
        assert result.destination_message_id == '731'
        next_lease = s.claim_delivery_receipt(c, receipt.id, owner_id='next', now=107)
        assert next_lease.receipt.destination_message_id == '731'
        assert next_lease.desired_revision == new
        with pytest.raises(s.DeliveryReceiptLeaseLost):
            s.reconcile_delivery_outcome(c, receipt.id, **args)
