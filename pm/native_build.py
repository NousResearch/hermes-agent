"""Native compiler environment for Windows ARM64 product bundle builds.

Source installs use locked native wheels and do not provision a compiler.
Desktop and bundle builders still prepare MSVC, Clang, Rust and static OpenSSL
when compiling their own native product dependencies.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from pm.progress import run_contained

_PROVIDER = Path("scripts/build/windows-deps.ps1")


def prepare_windows_environment(*, source: Path, state: Path, env: Mapping[str, str]) -> dict[str, str]:
    """The distribution adapter decides whether this target needs ARM64 tools."""
    shell = shutil.which("powershell", path=env.get("PATH")) or shutil.which("pwsh", path=env.get("PATH"))
    if shell is None:
        raise FileNotFoundError("PowerShell is required to prepare Windows ARM64 build dependencies")
    state.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="environment-", dir=state) as scratch:
        output = Path(scratch) / "environment.json"
        # A cold vcpkg clone and OpenSSL build print thousands of lines; the
        # user needs the step and its failure, not the patch log.
        run_contained(
            [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
             str(source / _PROVIDER), "-StateRoot", str(state),
             "-EnvironmentFile", str(output)],
            "Preparing Windows ARM64 build tools", indent="  ",
            cwd=source, env=dict(env), stdin=subprocess.DEVNULL,
        )
        prepared = json.loads(output.read_text(encoding="utf-8-sig"))
    if not isinstance(prepared, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in prepared.items()):
        raise ValueError("Windows build dependency provider returned an invalid environment")
    return prepared


def plugin_build_environment(source: Path) -> dict[str, str] | None:
    """Prepare compilers only after a native plugin-member build fails.

    Bundle builders use prepare_windows_environment directly. Core source
    installs have locked wheels and never call this function.
    """
    from pm.paths import store_root
    from pm.store import current_target

    if current_target() != "win32-arm64" or not (source / _PROVIDER).is_file():
        return None
    from pm.index_config import bridged_index_settings

    prepared = prepare_windows_environment(source=source, state=store_root().parent, env=os.environ)
    # This environment replaces the ambient one; carry configured index
    # bridges into the retried plugin resolution.
    prepared.update(bridged_index_settings(os.environ))
    return prepared
