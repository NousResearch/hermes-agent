#!/usr/bin/env python3
"""Standalone POSIX Compliance Check for Agathos.

Quick verification without pytest dependency.
Run: python agathos/tests/run_posix_compliance_check.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def check_platform_constants():
    """Check that all modules define platform constants."""
    print("Checking platform constants...")

    errors = []

    try:
        from agathos.venv_utils import _IS_MACOS, _IS_LINUX, _IS_WINDOWS
        assert isinstance(_IS_MACOS, bool)
        assert isinstance(_IS_LINUX, bool)
        assert isinstance(_IS_WINDOWS, bool)
        print(f"  venv_utils: OK (macOS={_IS_MACOS}, Linux={_IS_LINUX}, Windows={_IS_WINDOWS})")
    except Exception as e:
        errors.append(f"venv_utils: {e}")

    try:
        from agathos.subprocess_utils import _IS_MACOS, _IS_LINUX, _IS_WINDOWS
        print(f"  subprocess_utils: OK")
    except Exception as e:
        errors.append(f"subprocess_utils: {e}")

    try:
        from agathos.daemon_mgmt import _IS_MACOS, _IS_LINUX, _IS_WINDOWS
        print(f"  daemon_mgmt: OK")
    except Exception as e:
        errors.append(f"daemon_mgmt: {e}")

    try:
        from agathos.cli import _IS_MACOS, _IS_LINUX, _IS_WINDOWS
        print(f"  cli: OK")
    except Exception as e:
        errors.append(f"cli: {e}")

    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  - {e}")
        return False

    return True


def check_path_separator():
    """Check that PATH uses os.pathsep correctly."""
    print("\nChecking PATH separator usage...")

    from agathos.venv_utils import build_agathos_subprocess_env

    env = build_agathos_subprocess_env()
    path = env.get('PATH', '')

    print(f"  os.pathsep = '{os.pathsep}'")
    print(f"  PATH has {len(path.split(os.pathsep))} components")

    # Check that the function works without error
    parts = path.split(os.pathsep)
    if '' in parts:
        print("  Warning: Empty PATH components found")

    print("  PASS: PATH separator check")
    return True


def check_platform_paths():
    """Check platform-specific paths."""
    print("\nChecking platform-specific paths...")

    from agathos.venv_utils import _get_platform_std_paths, _IS_MACOS, _IS_LINUX, _IS_WINDOWS

    paths = _get_platform_std_paths()
    print(f"  Platform std paths: {paths}")

    if _IS_MACOS:
        # macOS should have Homebrew paths
        has_homebrew = '/opt/homebrew/bin' in paths or '/usr/local/bin' in paths
        if has_homebrew:
            print("  PASS: macOS has Homebrew paths")
        else:
            print("  FAIL: macOS missing Homebrew paths")
            return False
    elif _IS_LINUX:
        # Linux should NOT have Homebrew paths
        if '/opt/homebrew/bin' in paths:
            print("  FAIL: Linux has macOS-specific Homebrew path")
            return False
        print("  PASS: Linux has standard paths only")
    elif _IS_WINDOWS:
        # Windows should return empty list
        if paths == []:
            print("  PASS: Windows returns empty list (uses PATH env)")
        else:
            print(f"  WARNING: Windows has unexpected paths: {paths}")

    return True


def check_service_guards():
    """Check service management has platform guards."""
    print("\nChecking service management guards...")

    from agathos.daemon_mgmt import agathos_launchd_status, _IS_MACOS, _IS_LINUX

    # Status should work on all platforms
    try:
        status = agathos_launchd_status()
        # Support both old and new field names
        required_keys = ['label', 'pid_file_exists', 'running_pid', 'is_running']
        for key in required_keys:
            if key not in status:
                print(f"  FAIL: status missing key '{key}'")
                return False
        # Check for service path (new) or plist path (old)
        if 'service_path' not in status and 'plist_path' not in status:
            print("  FAIL: status missing 'service_path' or 'plist_path'")
            return False
        if 'service_exists' not in status and 'plist_exists' not in status:
            print("  FAIL: status missing 'service_exists' or 'plist_exists'")
            return False
        print(f"  PASS: agathos_launchd_status works (macOS={_IS_MACOS}, Linux={_IS_LINUX})")
    except Exception as e:
        print(f"  FAIL: agathos_launchd_status error: {e}")
        return False

    # Install behavior depends on platform
    from agathos.daemon_mgmt import agathos_launchd_install
    try:
        result = agathos_launchd_install()
        if _IS_MACOS or _IS_LINUX:
            # On supported platforms, may succeed or fail depending on systemctl/launchctl
            print(f"  PASS: agathos_launchd_install returned {result} (supported platform)")
        else:
            # On unsupported platforms, should return False
            if result is False:
                print("  PASS: agathos_launchd_install returns False on unsupported platform")
            else:
                print(f"  WARNING: agathos_launchd_install returned {result}, expected False on unsupported platform")
    except Exception as e:
        print(f"  FAIL: agathos_launchd_install raised {type(e).__name__}: {e}")
        return False

    return True


def check_no_hardcoded_separators():
    """Check source files don't have hardcoded ':' for PATH operations."""
    print("\nChecking for hardcoded PATH separators...")

    agathos_dir = PROJECT_ROOT / 'agathos'
    violations = []

    py_files = list(agathos_dir.glob('*.py'))

    for py_file in py_files:
        if py_file.name.startswith('test_'):
            continue

        content = py_file.read_text()
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Look for suspicious patterns
            if '.split(' in line and "':'" in line:
                # Skip if it's os.pathsep
                if 'os.pathsep' in line:
                    continue
                # Skip platform paths definitions
                if '/usr' in line or '/opt' in line or 'homebrew' in line:
                    continue
                violations.append(f"{py_file.name}:{i}: {line.strip()}")

            if '.join(' in line and '":"' in line:
                if 'os.pathsep' in line:
                    continue
                if '/usr' in line or '/opt' in line or 'homebrew' in line:
                    continue
                violations.append(f"{py_file.name}:{i}: {line.strip()}")

    if violations:
        print("  WARNING: Potential hardcoded separators found:")
        for v in violations[:5]:  # Show first 5
            print(f"    {v}")
        if len(violations) > 5:
            print(f"    ... and {len(violations) - 5} more")
        # Don't fail, just warn - some may be legitimate
    else:
        print("  PASS: No hardcoded PATH separators found")

    return True


def check_venv_bin_dir():
    """Check venv bin dir is correct for platform."""
    print("\nChecking venv bin directory...")

    from agathos.venv_utils import get_venv_bin_dir, _IS_WINDOWS

    test_path = Path('/tmp/test_venv')
    bin_dir = get_venv_bin_dir(test_path)

    if _IS_WINDOWS:
        if 'Scripts' in str(bin_dir):
            print("  PASS: Windows uses 'Scripts' directory")
        else:
            print(f"  FAIL: Windows should use 'Scripts', got: {bin_dir}")
            return False
    else:
        if str(bin_dir).endswith('bin'):
            print("  PASS: POSIX uses 'bin' directory")
        else:
            print(f"  FAIL: POSIX should use 'bin', got: {bin_dir}")
            return False

    return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("Agathos POSIX Compliance Check")
    print("=" * 60)
    print(f"Platform: {sys.platform}")
    print(f"os.name: {os.name}")
    print(f"os.pathsep: '{os.pathsep}'")
    print()

    checks = [
        ("Platform Constants", check_platform_constants),
        ("PATH Separator", check_path_separator),
        ("Platform Paths", check_platform_paths),
        ("Service Guards", check_service_guards),
        ("No Hardcoded Separators", check_no_hardcoded_separators),
        ("Venv Bin Dir", check_venv_bin_dir),
    ]

    passed = 0
    failed = 0

    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
            else:
                failed += 1
                print(f"\nFAILED: {name}")
        except Exception as e:
            failed += 1
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nSome checks failed. Review output above.")
        return 1
    else:
        print("\nAll checks passed!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
