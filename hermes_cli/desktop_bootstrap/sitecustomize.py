"""Earliest process-authority bootstrap for Desktop-owned backends.

Electron prepends this directory to ``PYTHONPATH`` only for marked Desktop
backend launches. Python imports ``sitecustomize`` before ``hermes_cli.main``,
so platform authority owns the complete execution scope before Hermes imports
spawn-capable code. Marked setup failures deliberately abort startup.
"""

import os
import sys

_AUTHORITY_ENV = "HERMES_DESKTOP_PROCESS_AUTHORITY"
_DESCENDANT_GUARD_ENV = "_HERMES_DESKTOP_POSIX_DESCENDANT_GUARD"
_POSIX_AUTHORITY_MODE = "posix-session-v1"

mode = (os.environ.get(_AUTHORITY_ENV) or "").strip()
descendant_mode = (os.environ.get(_DESCENDANT_GUARD_ENV) or "").strip()

if mode == "windows-job-v1":
    from hermes_cli.windows_process_authority import install_windows_process_authority

    install_windows_process_authority()
elif mode == _POSIX_AUTHORITY_MODE:
    from hermes_cli.posix_process_authority import install_posix_process_authority

    install_posix_process_authority()
elif mode:
    raise RuntimeError(f"unsupported Desktop process authority mode: {mode!r} on {sys.platform}")
elif descendant_mode:
    from hermes_cli.posix_process_authority import install_posix_descendant_guard

    install_posix_descendant_guard()
