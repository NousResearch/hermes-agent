# Linux and WSL Support

**Date:** April 9, 2026  
**Status:** Full Linux and WSL support implemented

## Overview

Agathos now provides first-class support for Linux (via systemd) and Windows Subsystem for Linux (WSL).

## Platform-Specific Features

### Linux (systemd)

| Feature | Implementation | Status |
|---------|---------------|--------|
| Service Installation | `systemctl --user enable/start` | ✅ Full Support |
| Service Uninstall | `systemctl --user disable/stop` | ✅ Full Support |
| Service Status | `systemctl --user is-active` | ✅ Full Support |
| PATH | Standard paths + snap + flatpak | ✅ Full Support |
| XDG Directories | `~/.config/systemd/user/` | ✅ Full Support |

### WSL (Windows Subsystem for Linux)

| Feature | Implementation | Status |
|---------|---------------|--------|
| WSL Detection | `/proc/version` + env vars | ✅ Full Support |
| Service Management | Same as Linux (systemd) | ✅ Full Support |
| Windows Interop | Automatic via WSL | ✅ Works |
| PATH | Linux paths + Windows (auto) | ✅ Full Support |

### macOS (launchd)

| Feature | Implementation | Status |
|---------|---------------|--------|
| Service Installation | `launchctl bootstrap` | ✅ Full Support |
| Service Uninstall | `launchctl bootout` | ✅ Full Support |
| Service Status | `launchctl list` | ✅ Full Support |
| PATH | Homebrew + standard paths | ✅ Full Support |

## Service Management

### Linux Installation

```bash
# Install as systemd user service
python -m agathos.cli service install

# Or programmatically
from agathos import agathos_launchd_install
agathos_launchd_install()  # Returns True on success
```

This creates:
- Service file: `~/.config/systemd/user/com.hermes.agathos.service`
- Enabled for user session
- Started immediately

### Linux Uninstall

```bash
python -m agathos.cli service uninstall
```

This:
- Stops the service (`systemctl --user stop`)
- Disables the service (`systemctl --user disable`)
- Removes the service file

### Linux Status

```bash
python -m agathos.cli service status
```

Shows:
- Service active/inactive status
- PID if running
- WSL detection (if applicable)

## WSL Detection

Agathos automatically detects WSL environments:

```python
from agathos import _is_wsl

if _is_wsl():
    print("Running under WSL")
    # WSL-specific handling if needed
```

Detection methods:
1. `WSL_DISTRO_NAME` environment variable
2. `WSL_INTEROP` environment variable
3. `/proc/version` containing "microsoft" or "wsl"

## PATH Configuration

### Linux PATH Priority

1. Virtual environment bin (if in venv)
2. Hermes venv bin (if found)
3. Snap packages (`~/snap/bin` if exists)
4. Standard paths (`/usr/local/bin`, `/usr/bin`, `/bin`)
5. Flatpak exports (`/var/lib/flatpak/exports/bin` if exists)

### WSL PATH

WSL automatically handles Windows PATH interop. Agathos uses standard Linux paths within WSL.

## Systemd Service File

Generated service file (`~/.config/systemd/user/com.hermes.agathos.service`):

```ini
[Unit]
Description=Agathos - Agent Guardian & Health Oversight System
After=network.target

[Service]
Type=simple
ExecStart=/path/to/python /path/to/agathos.py
Restart=on-failure
RestartSec=10
Environment="PATH=/venv/bin:/usr/local/bin:..."
Environment="HERMES_HOME=~/.hermes"
StandardOutput=append:~/hermes/logs/agathos/agathos.stdout.log
StandardError=append:~/hermes/logs/agathos/agathos.stderr.log

[Install]
WantedBy=default.target
```

## Testing

Run Linux/WSL-specific tests:

```bash
# All POSIX tests
python -m pytest agathos/tests/test_posix_compliance.py -v

# Linux-specific tests (skip if not on Linux)
python -m pytest agathos/tests/test_posix_compliance.py::TestLinuxSpecific -v

# Standalone check
python agathos/tests/run_posix_compliance_check.py
```

## Troubleshooting

### "systemctl not found"

Your Linux distribution may not use systemd. Agathos will fall back to manual daemon mode:

```bash
# Run manually instead of as service
python -m agathos.agathos
```

### WSL not detected

Check detection manually:

```bash
cat /proc/version
# Should contain "microsoft" or "WSL"

echo $WSL_DISTRO_NAME
# Should show your distro name (e.g., "Ubuntu")
```

### Snap/Flatpak paths not found

Agathos only adds these paths if the directories exist. Verify:

```bash
ls ~/snap/bin  # Snap user binaries
ls /var/lib/flatpak/exports/bin  # Flatpak system exports
```

## API Reference

### WSL Detection

```python
from agathos import _is_wsl
from agathos.daemon_mgmt import _is_wsl
from agathos.venv_utils import _is_wsl

# All three are the same function
is_wsl = _is_wsl()  # True if running under WSL
```

### Systemd Service Generation

```python
from agathos import generate_systemd_service

service_content = generate_systemd_service()
# Returns systemd service file content as string
```

### Cross-Platform Service Status

```python
from agathos import agathos_service_status

status = agathos_service_status()
# Returns dict with:
#   - platform: sys.platform
#   - is_wsl: bool
#   - service_type: 'launchd' | 'systemd' | 'windows' | 'unknown'
#   - service_exists: bool
#   - is_running: bool
#   - running_pid: int | None
#   - Plus platform-specific fields
```

## Comparison Matrix

| Feature | macOS | Linux | WSL | Windows |
|---------|-------|-------|-----|---------|
| Service Install | ✅ launchd | ✅ systemd | ✅ systemd | ⏳ Planned |
| Service Uninstall | ✅ | ✅ | ✅ | ⏳ |
| Service Status | ✅ | ✅ | ✅ | ⏳ |
| Auto-start | ✅ | ✅ | ✅ | ⏳ |
| PATH Management | ✅ | ✅ | ✅ | ⏳ |
| WSL Detection | N/A | N/A | ✅ | N/A |

## Migration Notes

If you were previously running Agathos on Linux manually:

```bash
# Install as service (new feature)
python -m agathos.cli service install

# Check it's running
python -m agathos.cli service status

# Logs now go to systemd journal + log files
journalctl --user -u com.hermes.agathos -f  # Follow logs
```
