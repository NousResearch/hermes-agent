"""Verify stale-lock cleanup really works on this Windows host.

NOT a unit test — a live integration check that exercises the actual
_cleanup_stale_gateway_lock + acquire_gateway_runtime_lock path with
real PIDs on the real filesystem.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# Make sure gateway/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.status import (
    _cleanup_stale_gateway_lock,
    _get_gateway_lock_path,
    _pid_exists,
    _read_pid_record,
    _write_gateway_lock_record,
    acquire_gateway_runtime_lock,
    is_gateway_runtime_lock_active,
    release_gateway_runtime_lock,
)


def _write_fake_lock(lock_path, pid, kind="hermes-gateway"):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"pid": pid, "kind": kind, "argv": ["fake"], "start_time": 123456}))
    print(f"  ✓ 写入假锁: {lock_path} (PID={pid})")


def test_dead_pid_cleanup(lock_path):
    """Scenario: crash residue — lock file exists but PID is dead."""
    print("\n=== 测试1：死 PID → 锁应被清理 ===")

    # Pick a PID that almost certainly doesn't exist
    dead_pid = 99999
    while _pid_exists(dead_pid):
        dead_pid += 1
    print(f"  死 PID={dead_pid}, _pid_exists={_pid_exists(dead_pid)}")

    _write_fake_lock(lock_path, dead_pid)

    # _cleanup_stale_gateway_lock should delete it
    assert lock_path.exists()
    _cleanup_stale_gateway_lock(lock_path)
    result = lock_path.exists()
    print(f"  清理后锁存在: {result}")
    assert not result, f"FAIL: 死 PID 的锁未被清理！"
    print("  ✅ 通过")


def test_live_pid_preserved(lock_path):
    """Scenario: lock file belongs to a live process — leave it alone."""
    print("\n=== 测试2：活 PID → 锁应保留 ===")

    # Clean up from test1 first
    lock_path.unlink(missing_ok=True)

    live_pid = os.getpid()
    assert _pid_exists(live_pid), f"自己的 PID {live_pid} 应该存在"
    print(f"  活 PID={live_pid}, _pid_exists={_pid_exists(live_pid)}")

    _write_fake_lock(lock_path, live_pid)

    assert lock_path.exists()
    _cleanup_stale_gateway_lock(lock_path)
    result = lock_path.exists()
    print(f"  清理后锁存在: {result}")
    assert result, f"FAIL: 活 PID 的锁被误删！"
    print("  ✅ 通过")


def test_is_lock_active_with_dead_pid(lock_path):
    """Scenario: dead PID lock file — is_gateway_runtime_lock_active should return False."""
    print("\n=== 测试3：死 PID 锁 → is_gateway_runtime_lock_active 应为 False ===")

    lock_path.unlink(missing_ok=True)
    dead_pid = 99998
    while _pid_exists(dead_pid):
        dead_pid += 1
    _write_fake_lock(lock_path, dead_pid)

    active = is_gateway_runtime_lock_active(lock_path)
    print(f"  is_gateway_runtime_lock_active = {active}")
    assert active == False, f"FAIL: 死 PID 锁被误判为活跃！"
    print("  ✅ 通过")


def test_acquire_after_stale_lock(lock_path):
    """End-to-end: create stale lock, call acquire, verify our PID is now in the file."""
    print("\n=== 测试4：端到端 —— 死锁→获取锁→验证 PID ===")

    # Clean up
    lock_path.unlink(missing_ok=True)

    dead_pid = 99997
    while _pid_exists(dead_pid):
        dead_pid += 1
    _write_fake_lock(lock_path, dead_pid)

    # Now try to acquire
    result = acquire_gateway_runtime_lock()
    print(f"  acquire_gateway_runtime_lock() = {result}")
    assert result, "FAIL: 获取锁失败！"

    # Verify our PID is recorded
    record = _read_pid_record(lock_path)
    recorded_pid = record["pid"] if record else None
    print(f"  锁文件 PID={recorded_pid}, 实际 PID={os.getpid()}")
    assert recorded_pid == os.getpid(), f"FAIL: 锁文件 PID 不匹配！"

    # Verify lock is active
    assert is_gateway_runtime_lock_active(lock_path), "FAIL: 获取后锁不活跃！"

    # Release
    release_gateway_runtime_lock()
    print("  ✅ 通过")


def main():
    with tempfile.TemporaryDirectory(prefix="stale_lock_test_") as tmpdir:
        # Use a temp HERMES_HOME so we don't touch real state
        os.environ["HERMES_HOME"] = tmpdir
        lock_path = _get_gateway_lock_path()
        print(f"临时 HERMES_HOME={tmpdir}")
        print(f"锁路径={lock_path}")

        try:
            test_dead_pid_cleanup(lock_path)
            test_live_pid_preserved(lock_path)
            test_is_lock_active_with_dead_pid(lock_path)
            test_acquire_after_stale_lock(lock_path)
        finally:
            # Clean up env
            os.environ.pop("HERMES_HOME", None)

    print("\n" + "=" * 50)
    print("全部 4 个测试通过 ✅")


if __name__ == "__main__":
    main()
