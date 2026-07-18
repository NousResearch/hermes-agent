import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from plugins.platforms.a2a import client_state, config, setup


def _peer(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    setup.ensure_a2a_platform_config(public_url="https://self.example/a2a")
    setup.add_peer("peer", url="https://peer.example/a2a", token="t" * 40)
    return config.load_a2a_settings().peers["peer"]["generation"]


def test_state_revision_round_trip_clear_and_permissions(tmp_path, monkeypatch):
    generation = _peer(tmp_path, monkeypatch)
    claim = client_state.try_begin_request(
        "peer", generation, "owner", new_context=False
    )
    assert claim is not None and claim.context_id is None
    assert client_state.complete_request(
        "peer", generation, claim, context_id="context", task_id="task"
    )
    state = client_state.get_peer_state("peer")
    assert state == {
        "generation": generation,
        "revision_epoch": claim.epoch,
        "revision": claim.revision,
        "context_id": "context",
        "task_id": "task",
    }
    assert oct(client_state.state_path().stat().st_mode & 0o777) == "0o600"
    assert "token" not in client_state.state_path().read_text()
    client_state.clear_peer_state("peer")
    assert client_state.get_peer_state("peer") == {}


def test_state_rejects_symlink_unsafe_lock_and_oversized_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = client_state.state_path()
    path.parent.mkdir(parents=True)
    target = tmp_path / "target"
    target.write_text(json.dumps({"version": 1, "peers": {}}))
    path.symlink_to(target)
    with pytest.raises(RuntimeError, match="regular file"):
        client_state.get_peer_state("peer")
    path.unlink()
    client_state._lock_path().unlink()
    client_state._lock_path().symlink_to(target)
    with pytest.raises(RuntimeError, match="lock is unsafe"):
        client_state.get_peer_state("peer")
    with pytest.raises(RuntimeError):
        client_state._bounded_id("x" * 257, required=True)


def test_state_concurrent_lease_is_exclusive_and_abort_allows_next(
    tmp_path, monkeypatch
):
    generation = _peer(tmp_path, monkeypatch)
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(
            pool.map(
                lambda index: client_state.try_begin_request(
                    "peer", generation, f"owner-{index}", new_context=False
                ),
                range(24),
            )
        )
    acquired = [claim for claim in claims if claim is not None]
    assert len(acquired) == 1
    client_state.abort_request("peer", generation, acquired[0])
    next_claim = client_state.try_begin_request(
        "peer", generation, "next-owner", new_context=False
    )
    assert next_claim is not None


def test_expired_lease_recovery_and_revision_rollover(tmp_path, monkeypatch):
    generation = _peer(tmp_path, monkeypatch)
    claim = client_state.try_begin_request("peer", generation, "old", new_context=False)
    assert claim is not None
    with client_state._state_lock() as directory_fd:
        data = client_state._load_unlocked(directory_fd)
        entry = data["peers"]["peer"]
        entry["lease_expires_at"] = 0
        entry["revision"] = client_state._MAX_REVISION
        old_epoch = entry["revision_epoch"]
        client_state._save_unlocked(data, directory_fd)
    recovered = client_state.try_begin_request(
        "peer", generation, "recovered", new_context=False
    )
    assert recovered is not None
    assert recovered.revision == 1
    assert recovered.epoch != old_epoch


def test_state_rejects_oversized_file_and_unknown_entry_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = client_state.state_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * (client_state._MAX_FILE_BYTES + 1))
    with pytest.raises(RuntimeError, match="too large"):
        client_state.get_peer_state("peer")
    path.write_text(json.dumps({"version": 1, "peers": {"peer": {"generation": "g", "revision": 0, "token": "forbidden"}}}))
    with pytest.raises(RuntimeError, match="invalid"):
        client_state.get_peer_state("peer")


def test_post_flock_lock_replacement_is_detected(tmp_path, monkeypatch):
    import fcntl

    generation = _peer(tmp_path, monkeypatch)
    client_state.try_begin_request("peer", generation, "owner", new_context=False)
    original = fcntl.flock
    replaced = False

    def replace_after_lock(fd, operation):
        nonlocal replaced
        original(fd, operation)
        if not replaced:
            replaced = True
            replacement = client_state._lock_path().with_suffix(".replacement")
            replacement.write_bytes(b"")
            os.replace(replacement, client_state._lock_path())

    monkeypatch.setattr(fcntl, "flock", replace_after_lock)
    with pytest.raises(RuntimeError, match="lock is unsafe"):
        client_state.get_peer_state("peer")


def test_parent_rename_and_symlink_swap_is_detected(tmp_path, monkeypatch):
    import fcntl

    generation = _peer(tmp_path, monkeypatch)
    client_state.try_begin_request("peer", generation, "owner", new_context=False)
    original = fcntl.flock
    parent = client_state.state_path().parent
    moved = parent.with_name("a2a-moved")
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    swapped = False

    def swap_parent_after_lock(fd, operation):
        nonlocal swapped
        original(fd, operation)
        if not swapped:
            swapped = True
            parent.rename(moved)
            parent.symlink_to(attacker, target_is_directory=True)

    monkeypatch.setattr(fcntl, "flock", swap_parent_after_lock)
    try:
        with pytest.raises(RuntimeError, match="lock is unsafe"):
            client_state.get_peer_state("peer")
    finally:
        if parent.is_symlink():
            parent.unlink()
        if moved.exists():
            moved.rename(parent)
