# POSIX Compliance Hunt Summary

**Date:** April 9, 2026  
**Status:** All macOS-specific features identified and made POSIX-compliant

## Hunt Results

### Code-Level macOS Features Found (All Properly Guarded)

| Feature | Location | Status | Notes |
|---------|----------|--------|-------|
| `launchctl` calls | `daemon_mgmt.py:272,300` | ✅ Guarded | Inside `_IS_MACOS` check |
| `gui/{os.getuid()}` | `daemon_mgmt.py:272,300` | ✅ Guarded | macOS launchctl domain |
| `os.getuid()` | `daemon_mgmt.py`, `cli.py` | ✅ Guarded | POSIX-only calls wrapped |
| `launchctl print` | `cli.py:396` | ✅ Guarded | Inside `_IS_MACOS` check |
| `/opt/homebrew/bin` | `venv_utils.py:143`, `subprocess_utils.py:40` | ✅ Conditional | Only in `_IS_MACOS` block |
| `~/Library/LaunchAgents` | `daemon_mgmt.py:149` | ✅ Conditional | Only returned on macOS |
| Signal handling | `agathos.py` (inline) | ✅ Cross-platform | Python stdlib, works on all |
| `os.kill(pid, 0)` | `daemon_mgmt.py`, bash scripts | ✅ Cross-platform | PID check works on all |

### Documentation References (Acceptable)

| File | References | Status |
|------|------------|--------|
| `VENV_SETUP.md` | Homebrew, launchd, `~/Projects/` paths | ✅ Acceptable - Examples only |
| `README.md` | macOS service instructions | ✅ Acceptable - Primary platform |
| `*.py` docstrings | macOS feature descriptions | ✅ Acceptable - Documentation |

### Bash Scripts (`agathos/bin/`)

| Script | Issue | Status |
|--------|-------|--------|
| `agathos-control` | Hardcoded dev path `~/Projects/hermes-dev` | ✅ Low priority - Dev helper |
| `agathos-control` | launchctl references in help | ✅ Acceptable - macOS primary |
| `check-agathos` | POSIX paths in commands | ✅ Cross-platform compatible |

## What Was Fixed in This Round

### 1. `daemon_mgmt.py` - PATH separator (Line ~190)
**Before:**
```python
sane_path = ":".join(
    dict.fromkeys(
        priority_dirs + [p for p in os.environ.get("PATH", "").split(":") if p]
    )
)
```

**After:**
```python
sane_path = os.pathsep.join(
    dict.fromkeys(
        priority_dirs + [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    )
)
```

## What Was Already Compliant

The following macOS-specific features are **correctly implemented** with platform guards:

1. **Service Management** (`daemon_mgmt.py`):
   - `agathos_launchd_install()` returns `False` on non-macOS
   - `agathos_launchd_uninstall()` handles all platforms
   - `_get_service_directory()` returns platform-appropriate paths

2. **CLI** (`cli.py`):
   - Service commands check `_IS_MACOS` before using launchctl
   - Platform-appropriate help messages

3. **Setup** (`setup.py`):
   - `_is_macos = sys.platform == 'darwin'` check at entry
   - Graceful fallback for non-macOS platforms

4. **Path Utilities** (`venv_utils.py`, `subprocess_utils.py`):
   - `_get_platform_std_paths()` returns appropriate paths per platform
   - Uses `os.pathsep` consistently

## Test Results

```
Standalone check: 6 passed, 0 failed
pytest: 24 passed, 3 skipped (platform-specific)
```

## Remaining Acceptable macOS-Specific Code

These are **intentionally** platform-specific and properly guarded:

1. **launchctl integration** - macOS-only feature by design
2. **Homebrew paths** - Only included on macOS
3. **Documentation examples** - Showing primary platform usage
4. **Docstring comments** - Explaining platform behavior

## Summary

The Agathos codebase is now **fully POSIX-compliant**:

- ✅ Path separators use `os.pathsep`
- ✅ Platform detection uses `sys.platform` and `os.name`
- ✅ Service management has platform guards
- ✅ PATH construction is cross-platform
- ✅ All modules import successfully
- ✅ Tests pass on macOS (Linux/Windows tests skip appropriately)

No further POSIX compliance work is required at the code level.
