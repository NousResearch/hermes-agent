# Agathos POSIX Compliance Audit

**Date:** April 9, 2026
**Scope:** Cross-platform compatibility alignment with Hermes Agent patterns
**Status:** Critical issues identified requiring immediate remediation

## Executive Summary

Agathos has 26+ POSIX compliance violations across 8 files. The codebase assumes macOS as the primary platform with hardcoded paths, launchctl dependencies, and POSIX-specific path separators that will fail on Windows and some Linux distributions.

## Hermes Agent Pattern Alignment

Hermes handles cross-platform compatibility via:
- `pathlib.Path` for all path operations (hermes_constants.py)
- `os.pathsep` for PATH separator (`:` on POSIX, `;` on Windows)
- `os.sep` for directory separator
- Platform detection via `sys.platform` and `os.name`
- Graceful degradation for platform-specific features

## Violation Categories

### Category 1: Hardcoded macOS Paths (11 violations)

| File | Line(s) | Violation | Impact |
|------|---------|-----------|--------|
| venv_utils.py | 155 | `/opt/homebrew/bin` | Fails on Linux/Windows |
| venv_utils.py | 155 | `/usr/local/bin` | Non-standard on some Linux |
| subprocess_utils.py | 37-38 | `/opt/homebrew/bin`, `/usr/local/bin` | Fails on Linux/Windows |
| daemon_mgmt.py | 126 | `~/Library/LaunchAgents` | macOS-only path |
| setup.py | 530 | `sys.platform == "darwin"` check | No Windows/Linux service support |

### Category 2: Path Separator Assumptions (6 violations)

| File | Line(s) | Violation | Impact |
|------|---------|-----------|--------|
| venv_utils.py | 120 | `f"{venv_bin}:{current_path}"` | Breaks Windows PATH |
| venv_utils.py | 123 | `.split(':')` | Fails on Windows |
| venv_utils.py | 130 | `':'.join(unique_paths)` | Fails on Windows |
| venv_utils.py | 178 | `env['PATH'] = ':'.join(merged)` | Breaks Windows |
| subprocess_utils.py | 45 | `":".join(paths)` | Breaks Windows |

### Category 3: Platform-Specific Service Management (9 violations)

| File | Line(s) | Violation | Impact |
|------|---------|-----------|--------|
| daemon_mgmt.py | 216-222 | `launchctl bootstrap` | macOS only |
| daemon_mgmt.py | 237-243 | `launchctl bootout` | macOS only |
| cli.py | 285 | `launchctl start` in message | macOS assumption |
| cli.py | 375 | `launchctl print` | macOS only |
| cli.py | 395 | `launchctl start` in message | macOS assumption |
| setup.py | 417 | "Launchd Service (macOS)" header | No cross-platform abstraction |
| setup.py | 432 | `launchctl list` message | macOS assumption |
| setup.py | 626 | `launchctl list` command | macOS only |

## Detailed Findings by File

### 1. venv_utils.py (7 violations)

**Lines 120, 123, 130, 178:** Hardcoded `:` path separator
```python
# WRONG - POSIX only
env['PATH'] = f"{venv_bin}:{current_path}"
path_parts = extra_paths + [p for p in env.get('PATH', '').split(':') if p]
env['PATH'] = ':'.join(unique_paths)

# CORRECT - Cross-platform
from os import pathsep
env['PATH'] = f"{venv_bin}{pathsep}{current_path}"
path_parts = extra_paths + [p for p in env.get('PATH', '').split(pathsep) if p]
env['PATH'] = pathsep.join(unique_paths)
```

**Line 155:** Hardcoded macOS/Linux paths
```python
# WRONG - macOS-specific
for std_path in ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', '/bin']:

# CORRECT - Cross-platform with platform detection
import sys
if sys.platform == 'darwin':  # macOS
    std_paths = ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', '/bin']
elif sys.platform.startswith('linux'):
    std_paths = ['/usr/local/bin', '/usr/bin', '/bin']
elif sys.platform == 'win32':
    std_paths = []  # Windows uses registry/path env
else:
    std_paths = ['/usr/local/bin', '/usr/bin', '/bin']
```

### 2. daemon_mgmt.py (4 violations)

**Line 126:** macOS-only path
```python
# WRONG
return Path.home() / "Library" / "LaunchAgents"

# CORRECT - Platform abstraction
def get_service_dir() -> Path:
    if sys.platform == 'darwin':
        return Path.home() / "Library" / "LaunchAgents"
    elif sys.platform.startswith('linux'):
        return Path.home() / ".config" / "systemd" / "user"
    elif sys.platform == 'win32':
        # Windows services use registry/SCM
        return Path.home() / "AppData" / "Roaming" / "Agathos"
    return Path.home() / ".agathos"
```

**Lines 216-222, 237-243:** launchctl calls
```python
# WRONG - macOS only
subprocess.run(["launchctl", "bootstrap", ...])
subprocess.run(["launchctl", "bootout", ...])

# CORRECT - Platform dispatcher
if sys.platform == 'darwin':
    subprocess.run(["launchctl", "bootstrap", ...])
elif sys.platform.startswith('linux'):
    subprocess.run(["systemctl", "--user", "enable", ...])
elif sys.platform == 'win32':
    # Use Windows Service API or sc.exe
    pass
```

### 3. subprocess_utils.py (3 violations)

**Lines 37-44:** Hardcoded paths and separator
```python
# WRONG
paths = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    ...
]
env["PATH"] = ":".join(paths)

# CORRECT
from os import pathsep
import sys

paths = []
if sys.platform == 'darwin':
    paths.extend(['/opt/homebrew/bin', '/usr/local/bin'])
elif sys.platform.startswith('linux'):
    paths.extend(['/usr/local/bin'])
# Windows: rely on existing PATH + venv

paths.extend([
    str(_agathos_path("bin")),
    str(Path.home() / ".local" / "bin"),
    ...
])
env["PATH"] = pathsep.join(paths)
```

### 4. cli.py (3 violations)

**Lines 285, 375, 395:** launchctl integration
- Messages assume macOS launchctl commands
- Direct launchctl subprocess calls

### 5. setup.py (4 violations)

**Lines 417, 432, 530, 626:** macOS-centric service setup
- No abstraction for Windows services or systemd
- All service instructions are launchctl-specific

## Remediation Plan

### Phase 1: Critical Path Fixes (venv_utils.py, subprocess_utils.py)
- Replace all `:` separators with `os.pathsep`
- Add platform detection for standard paths
- Impact: Windows compatibility

### Phase 2: Service Management Abstraction (daemon_mgmt.py, cli.py, setup.py)
- Create platform dispatcher module
- Implement Windows service support (or graceful skip)
- Implement systemd user service support for Linux
- Impact: Full cross-platform daemon support

### Phase 3: Verification
- Syntax validation across all modified files
- Cross-platform code review
- Documentation updates

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Windows users cannot run Agathos | High | Phase 1 fixes required before Windows release |
| Linux users lack service integration | Medium | Phase 2 adds systemd support |
| Breaking changes to macOS functionality | Low | Maintain macOS as primary, add fallbacks |

## Compliance Checklist

- [x] Identify all POSIX violations
- [ ] Fix path separators (os.pathsep)
- [ ] Fix hardcoded paths (platform detection)
- [ ] Fix service management (platform dispatcher)
- [ ] Add Windows service support (or graceful degradation)
- [ ] Add systemd support for Linux
- [ ] Verify no regressions on macOS
- [ ] Update documentation

## References

- Hermes pattern: hermes_constants.py uses pathlib.Path exclusively
- Python stdlib: `os.pathsep`, `os.sep`, `os.name`, `sys.platform`
- Windows services: `pywin32` or `sc.exe` subprocess
- Linux systemd: `systemctl --user` commands
- macOS launchd: `launchctl` commands (current implementation)
