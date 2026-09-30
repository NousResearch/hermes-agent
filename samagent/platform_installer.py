"""Local Platform Installer & Desktop / VS Code Bridge Setup for SamAgent.

Installs SamAgent on the user's machine as a persistent local platform (not a CLI workflow):
1. Creates default local projects directory (~/SamAgentProjects) where every app is stored as a real Git + .vscode workspace.
2. Installs the bundled VS Code / Cursor extension into ~/.vscode/extensions and ~/.cursor/extensions.
3. Creates native OS desktop application launchers & background service manifests:
   - macOS: ~/Applications/SamAgent.app + ~/Library/LaunchAgents/com.samjuniors.samagent.plist
   - Linux: ~/.local/share/applications/samagent.desktop + ~/.config/systemd/user/samagent.service
   - Windows: Start Menu launcher script
"""
from __future__ import annotations

import os
import platform
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
VSCODE_EXT_SOURCE = REPO_ROOT / "integrations" / "vscode-samagent"
EXTENSION_DIR_NAME = "samjuniors.samagent-vscode-0.1.0"


def get_default_projects_root(home_dir: Path | None = None) -> Path:
    base = Path(home_dir) if home_dir else Path.home()
    projects_dir = base / "SamAgentProjects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    return projects_dir


def install_vscode_extension(home_dir: Path | None = None) -> Dict[str, Any]:
    """Copy integrations/vscode-samagent into ~/.vscode/extensions and ~/.cursor/extensions."""
    base = Path(home_dir) if home_dir else Path.home()
    installed_targets: List[str] = []

    if not VSCODE_EXT_SOURCE.exists():
        return {"ok": False, "error": f"Extension source missing at {VSCODE_EXT_SOURCE}", "targets": []}

    for editor_dir in (".vscode", ".cursor", ".vscode-oss"):
        ext_root = base / editor_dir / "extensions"
        target = ext_root / EXTENSION_DIR_NAME
        ext_root.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(VSCODE_EXT_SOURCE, target)
        installed_targets.append(str(target))

    return {
        "ok": True,
        "extension_id": "samjuniors.samagent-vscode",
        "version": "0.1.0",
        "targets": installed_targets,
    }


def install_os_desktop_platform(
    *,
    home_dir: Path | None = None,
    port: int = 8080,
) -> Dict[str, Any]:
    """Install OS-native desktop launcher, background service unit, and VS Code extension."""
    base = Path(home_dir) if home_dir else Path.home()
    projects_root = get_default_projects_root(base)
    ext_result = install_vscode_extension(base)
    python_exe = sys.executable or "python3"
    os_name = platform.system().lower()
    artifacts: List[str] = []

    # 1. Linux desktop entry + systemd user service
    apps_dir = base / ".local" / "share" / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)
    desktop_file = apps_dir / "samagent-platform.desktop"
    desktop_file.write_text(
        f"""[Desktop Entry]
Type=Application
Name=SamAgent Local Platform
Comment=Local Spec-Driven Software Factory with Live VS Code & Pre-Prod Testing
Exec=sh -c "{python_exe} -m samagent.ui_server --host 127.0.0.1 --port {port} & xdg-open http://127.0.0.1:{port}"
Path={REPO_ROOT}
Terminal=false
Categories=Development;IDE;
StartupNotify=true
""",
        encoding="utf-8",
    )
    artifacts.append(str(desktop_file))

    systemd_dir = base / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)
    service_file = systemd_dir / "samagent-platform.service"
    service_file.write_text(
        f"""[Unit]
Description=SamAgent Local Platform Daemon (Mission Control & VS Code Bridge)
After=network.target

[Service]
Type=simple
WorkingDirectory={REPO_ROOT}
Environment=PYTHONPATH={REPO_ROOT}
ExecStart={python_exe} -m samagent.ui_server --host 127.0.0.1 --port {port}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
""",
        encoding="utf-8",
    )
    artifacts.append(str(service_file))

    # 2. macOS .app bundle + LaunchAgent plist
    mac_app_contents = base / "Applications" / "SamAgent.app" / "Contents"
    mac_macos_dir = mac_app_contents / "MacOS"
    mac_macos_dir.mkdir(parents=True, exist_ok=True)
    info_plist = mac_app_contents / "Info.plist"
    info_plist.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDisplayName</key>
  <string>SamAgent Platform</string>
  <key>CFBundleExecutable</key>
  <string>SamAgentLauncher</string>
  <key>CFBundleIdentifier</key>
  <string>com.samjuniors.samagent</string>
  <key>CFBundleName</key>
  <string>SamAgent</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
</dict>
</plist>
""",
        encoding="utf-8",
    )
    launcher_bin = mac_macos_dir / "SamAgentLauncher"
    launcher_bin.write_text(
        f"""#!/usr/bin/env bash
cd "{REPO_ROOT}"
PYTHONPATH="{REPO_ROOT}" "{python_exe}" -m samagent.ui_server --host 127.0.0.1 --port {port} >/tmp/samagent-platform.log 2>&1 &
sleep 0.8
open "http://127.0.0.1:{port}"
""",
        encoding="utf-8",
    )
    launcher_bin.chmod(launcher_bin.stat().st_mode | stat.S_IEXEC)
    artifacts.append(str(base / "Applications" / "SamAgent.app"))

    launch_agents = base / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    plist_path = launch_agents / "com.samjuniors.samagent.plist"
    plist_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.samjuniors.samagent</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python_exe}</string>
    <string>-m</string>
    <string>samagent.ui_server</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>{port}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>{REPO_ROOT}</string>
  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>
""",
        encoding="utf-8",
    )
    artifacts.append(str(plist_path))

    return {
        "ok": True,
        "os": os_name,
        "platform_url": f"http://127.0.0.1:{port}",
        "projects_root": str(projects_root),
        "vscode_extension": ext_result,
        "installed_artifacts": artifacts,
    }


def get_platform_install_status(home_dir: Path | None = None, port: int = 8080) -> Dict[str, Any]:
    """Return current installation status of the local platform, projects folder, and VS Code bridge."""
    base = Path(home_dir) if home_dir else Path.home()
    projects_root = base / "SamAgentProjects"
    vscode_ext = base / ".vscode" / "extensions" / EXTENSION_DIR_NAME
    cursor_ext = base / ".cursor" / "extensions" / EXTENSION_DIR_NAME
    desktop_linux = base / ".local" / "share" / "applications" / "samagent-platform.desktop"
    desktop_mac = base / "Applications" / "SamAgent.app"
    systemd_unit = base / ".config" / "systemd" / "user" / "samagent-platform.service"

    installed = vscode_ext.exists() or desktop_linux.exists() or desktop_mac.exists()
    return {
        "installed": installed,
        "platform_url": f"http://127.0.0.1:{port}",
        "projects_root": str(projects_root),
        "projects_root_exists": projects_root.exists(),
        "vscode_extension_installed": vscode_ext.exists(),
        "cursor_extension_installed": cursor_ext.exists(),
        "desktop_app_installed": desktop_linux.exists() or desktop_mac.exists(),
        "background_service_installed": systemd_unit.exists() or (base / "Library" / "LaunchAgents" / "com.samjuniors.samagent.plist").exists(),
        "vscode_extension_path": str(vscode_ext),
        "install_script": str(REPO_ROOT / "scripts" / "install-samagent-platform.sh"),
    }
