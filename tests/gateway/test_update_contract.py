"""Update-helper lifecycle lock contract tests."""

from gateway.update_contract import UpdateHelperLock, acquire_update_helper_probe


def test_probe_returns_held_lock_when_helper_lock_is_free(tmp_path):
    lock_path = tmp_path / ".update_helper.lock"

    probe = acquire_update_helper_probe(lock_path)

    assert probe is not None
    assert acquire_update_helper_probe(lock_path) is None
    probe.release()


def test_probe_returns_none_while_helper_owns_lock(tmp_path):
    lock_path = tmp_path / ".update_helper.lock"

    with UpdateHelperLock(lock_path):
        assert acquire_update_helper_probe(lock_path) is None

    probe = acquire_update_helper_probe(lock_path)
    assert probe is not None
    probe.release()
