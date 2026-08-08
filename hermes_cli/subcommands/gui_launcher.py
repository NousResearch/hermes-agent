"""``hermes desktop launcher install`` implementation.

Creates a macOS Spotlight-searchable .app wrapper and optional .command
Desktop launcher that calls ``hermes desktop --skip-build --cwd <path>``.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


def _resolve_app_bundle(hermes_agent_dir: Path) -> Path | None:
    """Find the built Hermes.app bundle under apps/desktop/release/mac-*/."""
    release_dir = hermes_agent_dir / "apps" / "desktop" / "release"
    if not release_dir.exists():
        return None
    for candidate in sorted(release_dir.glob("mac-*/Hermes.app")):
        if candidate.is_dir():
            return candidate
    return None


def _build_desktop_if_needed(hermes_agent_dir: Path, cwd: Path) -> Path | None:
    """Build the desktop app if no bundle exists. Returns the bundle path."""
    bundle = _resolve_app_bundle(hermes_agent_dir)
    if bundle is not None:
        return bundle

    print("  ⚠ Hermes Desktop app bundle not found; building with hermes desktop --build-only...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "desktop", "--build-only", "--cwd", str(cwd)],
            cwd=hermes_agent_dir,
            check=False,
        )
        if result.returncode != 0:
            print("  ✗ Desktop build failed")
            return None
    except Exception as e:
        print(f"  ✗ Failed to build desktop app: {e}")
        return None

    return _resolve_app_bundle(hermes_agent_dir)


def _generate_launcher_script(launch_cwd: Path, hermes_agent_dir: Path) -> str:
    """Generate the .command launcher script content."""
    return textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

        LAUNCH_CWD="{launch_cwd}"
        HERMES_AGENT_DIR="${{HERMES_AGENT_DIR:-{hermes_agent_dir}}}"

        APP_BUNDLE=""
        for candidate in "$HERMES_AGENT_DIR"/apps/desktop/release/mac-*/Hermes.app; do
          if [[ -d "$candidate" ]]; then
            APP_BUNDLE="$candidate"
            break
          fi
        done

        if [[ -n "$APP_BUNDLE" ]]; then
          APP_EXECUTABLE="$APP_BUNDLE/Contents/MacOS/Hermes"
          if pgrep -f "$APP_EXECUTABLE" >/dev/null 2>&1; then
            echo "Hermes Desktop is already running; not launching another instance."
            exit 0
          fi
        fi

        exec hermes desktop --skip-build --cwd "$LAUNCH_CWD"
    """)


def _generate_app_launcher_script(launch_cwd: Path, hermes_agent_dir: Path) -> str:
    """Generate the .app internal launcher script content."""
    return textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

        LAUNCH_CWD="{launch_cwd}"
        HERMES_AGENT_DIR="${{HERMES_AGENT_DIR:-{hermes_agent_dir}}}"

        APP_BUNDLE=""
        for candidate in "$HERMES_AGENT_DIR"/apps/desktop/release/mac-*/Hermes.app; do
          if [[ -d "$candidate" ]]; then
            APP_BUNDLE="$candidate"
            break
          fi
        done

        if [[ -n "$APP_BUNDLE" ]]; then
          APP_EXECUTABLE="$APP_BUNDLE/Contents/MacOS/Hermes"
          if pgrep -f "$APP_EXECUTABLE" >/dev/null 2>&1; then
            exit 0
          fi
        fi

        nohup hermes desktop --skip-build --cwd "$LAUNCH_CWD" >/tmp/hermes-desktop-launcher.log 2>&1 &
        exit 0
    """)


def _generate_info_plist(name: str) -> str:
    """Generate Info.plist content for the .app wrapper."""
    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
          <key>CFBundleDisplayName</key>
          <string>{name}</string>
          <key>CFBundleExecutable</key>
          <string>{name}</string>
          <key>CFBundleIconFile</key>
          <string>icon</string>
          <key>CFBundleIdentifier</key>
          <string>com.nousresearch.hermes-desktop-launcher</string>
          <key>CFBundleInfoDictionaryVersion</key>
          <string>6.0</string>
          <key>CFBundleName</key>
          <string>{name}</string>
          <key>CFBundlePackageType</key>
          <string>APPL</string>
          <key>CFBundleShortVersionString</key>
          <string>1.0.0</string>
          <key>CFBundleVersion</key>
          <string>1.0.0</string>
        </dict>
        </plist>
    """)


def install_launcher(
    *,
    cwd: Path | None = None,
    name: str = "Hermes Desktop",
    hermes_agent_dir: Path | None = None,
) -> bool:
    """Install a macOS Spotlight-searchable launcher.

    Creates:
      - ~/Desktop/<name>.command  (Desktop launcher)
      - ~/Applications/<name>.app (Spotlight-searchable wrapper)

    Returns True on success, False on failure.
    """
    if platform.system() != "Darwin":
        print("✗ Launcher installer is macOS-only.")
        return False

    if hermes_agent_dir is None:
        hermes_agent_dir = Path.home() / ".hermes" / "hermes-agent"

    if cwd is None:
        cwd = Path.home()

    cwd = Path(cwd).expanduser().resolve()
    if not cwd.exists():
        print(f"✗ Launch folder does not exist: {cwd}")
        return False

    desktop_dir = Path.home() / "Desktop"
    if not desktop_dir.exists():
        print(f"✗ Desktop folder not found: {desktop_dir}")
        return False

    user_app_dir = Path.home() / "Applications"
    user_app_dir.mkdir(parents=True, exist_ok=True)

    # Resolve or build the app bundle for icon
    bundle = _build_desktop_if_needed(hermes_agent_dir, cwd)
    if bundle is None:
        print("✗ Could not find or build Hermes.app bundle")
        return False

    icon_path = bundle / "Contents" / "Resources" / "icon.icns"

    # 1. Create .command Desktop launcher
    launcher_path = desktop_dir / f"{name}.command"
    launcher_content = _generate_launcher_script(cwd, hermes_agent_dir)
    launcher_path.write_text(launcher_content)
    launcher_path.chmod(0o755)
    print(f"  ✓ Created Desktop launcher: {launcher_path}")

    # 2. Create .app wrapper for Spotlight
    app_wrapper_path = user_app_dir / f"{name}.app"
    if app_wrapper_path.exists():
        shutil.rmtree(app_wrapper_path)

    app_contents = app_wrapper_path / "Contents"
    app_macos = app_contents / "MacOS"
    app_resources = app_contents / "Resources"
    app_macos.mkdir(parents=True, exist_ok=True)
    app_resources.mkdir(parents=True, exist_ok=True)

    # Copy icon if available
    if icon_path.exists():
        shutil.copy2(icon_path, app_resources / "icon.icns")
        print(f"  ✓ Copied icon from {icon_path}")

    # Write Info.plist
    (app_contents / "Info.plist").write_text(_generate_info_plist(name))

    # Write PkgInfo
    (app_contents / "PkgInfo").write_text("APPL????")

    # Write executable script
    app_executable = app_macos / name
    app_executable.write_text(_generate_app_launcher_script(cwd, hermes_agent_dir))
    app_executable.chmod(0o755)

    print(f"  ✓ Created Spotlight app: {app_wrapper_path}")

    # 3. Try to set icon on .command file (optional, requires Rez/SetFile)
    if icon_path.exists() and shutil.which("Rez") and shutil.which("SetFile"):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copy2(icon_path, tmp_path / "icon.icns")
            (tmp_path / "icon.r").write_text("read 'icns' (-16455) \"icon.icns\";\n")
            try:
                subprocess.run(
                    ["Rez", "-append", "icon.r", "-o", str(launcher_path)],
                    cwd=tmp_path,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["SetFile", "-a", "C", str(launcher_path)],
                    check=True,
                    capture_output=True,
                )
                launcher_path.touch()
                print("  ✓ Set custom icon on Desktop launcher")
            except subprocess.CalledProcessError:
                pass  # Non-fatal

    # 4. Reindex for Spotlight
    if shutil.which("mdimport"):
        try:
            subprocess.run(
                ["mdimport", str(app_wrapper_path)],
                check=False,
                capture_output=True,
            )
        except Exception:
            pass

    print()
    print("✓ Hermes Desktop launcher installed!")
    print(f"  Desktop: {launcher_path}")
    print(f"  Spotlight: {app_wrapper_path}")
    print(f"  Launch folder: {cwd}")
    print()
    print("You can now:")
    print(f"  • Press ⌘Space and type '{name}' to launch from Spotlight")
    print(f"  • Double-click the Desktop launcher to start")
    return True


def uninstall_launcher(name: str = "Hermes Desktop") -> bool:
    """Remove the macOS launcher files.

    Returns True on success, False on failure.
    """
    if platform.system() != "Darwin":
        print("✗ Launcher uninstaller is macOS-only.")
        return False

    desktop_dir = Path.home() / "Desktop"
    user_app_dir = Path.home() / "Applications"

    launcher_path = desktop_dir / f"{name}.command"
    app_wrapper_path = user_app_dir / f"{name}.app"

    removed = False

    if launcher_path.exists():
        launcher_path.unlink()
        print(f"  ✓ Removed Desktop launcher: {launcher_path}")
        removed = True

    if app_wrapper_path.exists():
        shutil.rmtree(app_wrapper_path)
        print(f"  ✓ Removed Spotlight app: {app_wrapper_path}")
        removed = True

    if removed:
        print()
        print(f"✓ Hermes Desktop launcher '{name}' uninstalled.")
    else:
        print(f"  ⚠ No launcher found with name '{name}'")

    return removed
