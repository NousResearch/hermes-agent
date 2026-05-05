#!/usr/bin/env python3
"""
Integration test: Verify CompoundCommandSplitter behavior in realistic scenarios
"""
from tools.command_compressors import CompoundCommandSplitter, CompressorRegistry

def test_all():
    splitter = CompoundCommandSplitter()
    registry = CompressorRegistry()
    passed = 0
    failed = 0

    print("=" * 70)
    print("COMPOUND COMMAND SPLITTER - INTEGRATION TEST")
    print("=" * 70)

    # Test 1: Simple command (should NOT split)
    print("\n1. Simple command (should return 1 segment):")
    cmd = "ls -la"
    segments = splitter.split(cmd)
    result = len(segments) == 1 and segments[0] == cmd
    print(f"   Command: {cmd}")
    print(f"   Segments: {segments}")
    print(f"   {'✓ PASS' if result else '✗ FAIL'}")
    passed += result; failed += not result

    # Test 2: Compound with && (should split)
    print("\n2. Compound with && (should split into 2):")
    cmd = "git status && git diff"
    segments = splitter.split(cmd)
    result = len(segments) == 2
    print(f"   Command: {cmd}")
    print(f"   Segments: {segments}")
    print(f"   {'✓ PASS' if result else '✗ FAIL'}")
    passed += result; failed += not result

    # Test 3: Compound with multiple operators (should split)
    print("\n3. Compound with multiple operators (should split into 3):")
    cmd = "npm install && npm test || npm run dev"
    segments = splitter.split(cmd)
    result = len(segments) == 3
    print(f"   Command: {cmd}")
    print(f"   Segments: {segments}")
    print(f"   {'✓ PASS' if result else '✗ FAIL'}")
    passed += result; failed += not result

    # Test 4: Subshell protection (should NOT split)
    print("\n4. Subshell protection (should NOT split):")
    cmd = "echo $(date) && ls"
    segments = splitter.split(cmd)
    result = len(segments) == 1
    print(f"   Command: {cmd}")
    print(f"   Segments: {segments}")
    print(f"   {'✓ PASS' if result else '✗ FAIL'}")
    passed += result; failed += not result

    # Test 5: compress_segments on single command (should return None, delegate)
    print("\n5. compress_segments on single command (should return None):")
    result = splitter.compress_segments(registry, "ls -la", "file1\nfile2", "", 0)
    is_none = result is None
    print(f"   Command: ls -la")
    print(f"   Result: {result}")
    print(f"   {'✓ PASS' if is_none else '✗ FAIL'}")
    passed += is_none; failed += not is_none

    # Test 6: compress_segments on compound (should return None)
    print("\n6. compress_segments on compound command (should return None):")
    result = splitter.compress_segments(registry, "ls && pwd", "output", "", 0)
    is_none = result is None
    print(f"   Command: ls && pwd")
    print(f"   Result: {result}")
    print(f"   {'✓ PASS' if is_none else '✗ FAIL'}")
    passed += is_none; failed += not is_none

    # Test 7: Backslash at EOF (safety test)
    print("\n7. Backslash at EOF (should not crash):")
    cmd = "echo \\"
    try:
        segments = splitter.split(cmd)
        print(f"   Command: {repr(cmd)}")
        print(f"   Segments: {segments}")
        print(f"   ✓ PASS (no crash)")
        passed += 1
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        failed += 1

    # Test 8: Backtick substitution (should NOT split)
    print("\n8. Backtick substitution (should NOT split):")
    cmd = "echo `date` && ls"
    segments = splitter.split(cmd)
    result = len(segments) == 1
    print(f"   Command: {cmd}")
    print(f"   Segments: {segments}")
    print(f"   {'✓ PASS' if result else '✗ FAIL'}")
    passed += result; failed += not result

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    import sys
    sys.exit(0 if test_all() else 1)
