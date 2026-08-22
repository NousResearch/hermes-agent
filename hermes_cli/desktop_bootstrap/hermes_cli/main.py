"""Authority-first wrapper for ``python -m hermes_cli.main`` Desktop launches."""

from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType

import hermes_cli as package


def _load_authority_module(name: str, path: Path) -> ModuleType:
    qualified = f"hermes_cli.{name}"
    existing = sys.modules.get(qualified)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Desktop authority module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


real_package = Path(package.DESKTOP_REAL_PACKAGE)
mode = (os.environ.get("HERMES_DESKTOP_PROCESS_AUTHORITY") or "").strip()

if mode == "windows-job-v1":
    authority = _load_authority_module(
        "windows_process_authority",
        real_package / "windows_process_authority.py",
    )
    authority.install_windows_process_authority()
elif mode == "posix-session-v1":
    authority = _load_authority_module(
        "posix_process_authority",
        real_package / "posix_process_authority.py",
    )
    authority.install_posix_process_authority()
elif mode:
    raise RuntimeError(f"unsupported Desktop process authority mode: {mode!r}")

# Only the authority-owning backend child reaches this point on POSIX; the
# supervisor remains the Electron ChildProcess and never imports application
# code. Restore the real package before loading the real CLI entrypoint.
bootstrap_root = Path(package.DESKTOP_BOOTSTRAP_ROOT).resolve()
sys.path[:] = [
    entry
    for entry in sys.path
    if not entry or Path(entry).resolve() != bootstrap_root
]
package.__file__ = package.DESKTOP_REAL_INIT
package.__path__ = [str(real_package)]
if package.__spec__ is not None:
    package.__spec__.submodule_search_locations = list(package.__path__)

real_init = Path(package.DESKTOP_REAL_INIT)
exec(compile(real_init.read_bytes(), str(real_init), "exec"), package.__dict__)
runpy.run_module("hermes_cli.main", run_name="__main__", alter_sys=True)
