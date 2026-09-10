import json

import pytest

from hermes_cli.web_server_ssh_batch import SshSpawnBatchLease


BATCH_ID = "a" * 32
FIRST_MEMBER = "b" * 16
SECOND_MEMBER = "c" * 16


def test_fresh_active_sibling_blocks_idle_exit_until_it_becomes_inactive(tmp_path):
    clock = {"now": 1_000.0}
    first = SshSpawnBatchLease(BATCH_ID, FIRST_MEMBER, root=tmp_path, now=lambda: clock["now"])
    second = SshSpawnBatchLease(BATCH_ID, SECOND_MEMBER, root=tmp_path, now=lambda: clock["now"])

    assert first.publish(active=True) is True
    assert second.publish(active=False) is True
    assert second.has_active_peer(ttl_s=30.0) is True

    clock["now"] += 1
    assert first.publish(active=False) is True
    assert second.has_active_peer(ttl_s=30.0) is False


def test_stale_sibling_lease_does_not_keep_an_idle_batch_alive(tmp_path):
    clock = {"now": 1_000.0}
    first = SshSpawnBatchLease(BATCH_ID, FIRST_MEMBER, root=tmp_path, now=lambda: clock["now"])
    second = SshSpawnBatchLease(BATCH_ID, SECOND_MEMBER, root=tmp_path, now=lambda: clock["now"])

    assert first.publish(active=True) is True
    clock["now"] += 31.0

    assert second.has_active_peer(ttl_s=30.0) is False


def test_unreadable_peer_record_fails_closed_instead_of_retiring_a_live_sibling(tmp_path):
    clock = {"now": 1_000.0}
    second = SshSpawnBatchLease(BATCH_ID, SECOND_MEMBER, root=tmp_path, now=lambda: clock["now"])

    assert second.publish(active=False) is True
    peer_path = tmp_path / "batches" / BATCH_ID / f"{FIRST_MEMBER}.json"
    peer_path.write_text(json.dumps({"active": "not-a-bool", "updated_at": clock["now"]}))

    assert second.has_active_peer(ttl_s=30.0) is None


def test_closing_a_member_removes_its_lease_without_touching_a_sibling(tmp_path):
    clock = {"now": 1_000.0}
    first = SshSpawnBatchLease(BATCH_ID, FIRST_MEMBER, root=tmp_path, now=lambda: clock["now"])
    second = SshSpawnBatchLease(BATCH_ID, SECOND_MEMBER, root=tmp_path, now=lambda: clock["now"])

    assert first.publish(active=True) is True
    assert second.publish(active=True) is True
    first.close()

    assert not (tmp_path / "batches" / BATCH_ID / f"{FIRST_MEMBER}.json").exists()
    assert (tmp_path / "batches" / BATCH_ID / f"{SECOND_MEMBER}.json").exists()
    assert second.has_active_peer(ttl_s=30.0) is False


@pytest.mark.parametrize("batch_id, owner_nonce", [("bad", FIRST_MEMBER), (BATCH_ID, "bad")])
def test_batch_identifiers_are_strictly_validated(tmp_path, batch_id, owner_nonce):
    with pytest.raises(ValueError):
        SshSpawnBatchLease(batch_id, owner_nonce, root=tmp_path)
