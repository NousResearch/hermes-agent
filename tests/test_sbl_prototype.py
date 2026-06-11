"""SBL prototype test — run on HQ server."""
import sys
import os
# Try to find the project root dynamically
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from plugins.sbl import (
    _on_pre_tool_call, _classify_path, _take_snapshot,
    _has_snapshot, _lookup_dependencies,
)
from plugins.sbl import _on_transform_tool_result

# ── Test 1: FHS Classification ──────────────────────────────────────────
print("=== Test 1: FHS Classification ===")
FHS_CASES = {
    "/etc/nginx/nginx.conf": "SYSTEM",
    "/home/user/test.txt": "USER",
    "/tmp/test": "USER",
    "/opt/hermes/test": "SYSTEM",
    "/var/log/syslog": "SYSTEM",
    "/unknown/path": "UNKNOWN",
}
for path, expected in FHS_CASES.items():
    cls = _classify_path(path)
    assert cls == expected, f"{path}: expected {expected}, got {cls}"
    print(f"  ✅ {path} -> {cls}")
print("  ✅ FHS classification: ALL PASSED")

# ── Test 2: Snapshot ────────────────────────────────────────────────────
print()
print("=== Test 2: Snapshot ===")
sm = _take_snapshot()
assert hasattr(sm, "services"), "ServiceMap missing services"
assert hasattr(sm, "port_owners"), "ServiceMap missing port_owners"
assert hasattr(sm, "file_owners"), "ServiceMap missing file_owners"
assert len(sm.services) > 0, f"Expected >0 services, got {len(sm.services)}"
assert len(sm.port_owners) > 0, f"Expected >0 ports, got {len(sm.port_owners)}"
print(f"  ✅ Services: {len(sm.services)}")
print(f"  ✅ Ports: {len(sm.port_owners)}")
print(f"  ✅ Config deps: {len(sm.file_owners)}")

# Show ports (informational)
print("  Listening ports:")
for port, owners in sorted(sm.port_owners.items()):
    print(f"    :{port} -> {', '.join(sorted(owners))}")

# ── Test 3: Dependency Lookup ───────────────────────────────────────────
print()
print("=== Test 3: Dependency Lookup ===")
# nginx.conf should find nginx dependency
nginx_deps = _lookup_dependencies("/etc/nginx/nginx.conf")
assert nginx_deps is not None, "nginx.conf should have deps"
nginx_names = [d["service"] for d in nginx_deps]
assert "nginx" in nginx_names, f"nginx not in deps: {nginx_names}"
print(f"  ✅ /etc/nginx/nginx.conf affects: {', '.join(nginx_names)}")

# stalwart config may or may not be tracked — just verify no crash
stalwart_deps = _lookup_dependencies("/etc/stalwart/config.toml")
print(f"  ✅ /etc/stalwart/config.toml -> {'deps tracked' if stalwart_deps else 'no deps tracked'}")

# ── Test 4: Pre-write on SYSTEM path ────────────────────────────────────
print()
print("=== Test 4: Pre-write on SYSTEM path ===")
result = _on_pre_tool_call(
    tool_name="write_file",
    args={"path": "/etc/nginx/nginx.conf", "content": "bad config"},
)
assert result is not None, "SYSTEM path should trigger SBL warning"
assert "SBL" in result, f"Expected SBL marker in result: {result}"
assert "nginx" in result, f"Expected nginx in result: {result}"
print(f"  ✅ Result: {result[:80]}...")

# ── Test 5: Pre-write on USER path ──────────────────────────────────────
print()
print("=== Test 5: Pre-write on USER path ===")
result = _on_pre_tool_call(
    tool_name="write_file",
    args={"path": "/home/user/project/test.txt", "content": "ok"},
)
assert result is None, f"USER path should not trigger SBL, got: {result}"
print(f"  ✅ Result: None (correct)")

# ── Test 6: Pre-write on UNKNOWN path ──────────────────────────────────
print()
print("=== Test 6: Pre-write on UNKNOWN path ===")
result = _on_pre_tool_call(
    tool_name="write_file",
    args={"path": "/weird/location/lib.so", "content": "bad"},
)
assert result is not None, "UNKNOWN path should trigger SBL warning"
assert "blocked" in result.lower() or "unclassified" in result.lower(), \
    f"Expected blocked/unclassified in result: {result}"
print(f"  ✅ Result: blocked")

# ── Test 7: Terminal with systemctl ─────────────────────────────────────
print()
print("=== Test 7: Terminal with systemctl ===")
result = _on_pre_tool_call(
    tool_name="terminal",
    args={"command": "systemctl restart nginx"},
)
assert result is not None, "systemctl should trigger SBL warning"
assert "nginx" in result, f"Expected nginx in result: {result}"
print(f"  ✅ Result: {result[:80]}...")

# ── Test 8: Terminal with echo redirect ─────────────────────────────────
print()
print("=== Test 8: Terminal with echo redirect ===")
result = _on_pre_tool_call(
    tool_name="terminal",
    args={"command": "echo '127.0.0.1 test' >> /etc/hosts"},
)
assert result is not None, "echo redirect to SYSTEM should trigger SBL"
print(f"  ✅ Result: {result[:80]}...")

# ── Test 9: Non-write tool ──────────────────────────────────────────────
print()
print("=== Test 9: Non-write tool ===")
result = _on_pre_tool_call(
    tool_name="read_file",
    args={"path": "/etc/nginx/nginx.conf"},
)
assert result is None, f"read_file should not trigger SBL, got: {result}"
print(f"  ✅ Result: None (correct)")

# ── Test 10: Transform tool result hook (learns, doesn't annotate) ────
print()
print("=== Test 10: Transform tool result (learns internally) ===")
# SBL's transform hook learns new paths, never annotates output
# (awareness, not blocking — by design)
result = _on_transform_tool_result(
    tool_name="write_file",
    args={"path": "/etc/nginx/nginx.conf", "content": "new config"},
    result='{"status": "ok"}',
)
assert result is None, f"transform_tool_result should return None (learn-only), got: {result}"
print(f"  ✅ SYSTEM write: learns silently (returns None)")

result = _on_transform_tool_result(
    tool_name="read_file",
    args={"path": "/etc/nginx/nginx.conf"},
    result='{"content": "...", "total_lines": 100}',
)
assert result is None, f"read_file should return None, got: {result}"
print(f"  ✅ read_file: returns None (correct)")

result = _on_transform_tool_result(
    tool_name="write_file",
    args={"path": "/tmp/test.txt", "content": "test"},
    result='{"status": "ok"}',
)
assert result is None, f"USER write should return None, got: {result}"
print(f"  ✅ USER write: returns None (correct)")

# Verify learn happens: second write to same path should find it in change_log
from plugins.sbl import _change_log
recent = [c for c in _change_log if "/etc/nginx/nginx.conf" in c.get("path", "")]
assert len(recent) >= 1, f"Expected change_log entry for nginx.conf, got: {_change_log[-3:]}"
print(f"  ✅ Change log updated: {len(recent)} entries for nginx.conf")

print()
print("=== ✅ ALL 10 TESTS PASSED (with assertions) ===")
