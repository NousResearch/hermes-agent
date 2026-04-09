"""Subprocess utilities with Hermes environment awareness."""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("argus.subprocess")

# === HERMES INTEGRATION ===
try:
    from hermes_constants import get_hermes_home
    _HERMES_HOME = get_hermes_home()
except ImportError:
    _HERMES_HOME = Path(os.path.expanduser("~/.hermes"))

_ARGUS_HOME = Path(os.path.expanduser("~/hermes"))


def _hermes_path(*parts: str) -> Path:
    """Build a path under HERMES_HOME (~/.hermes)."""
    return _HERMES_HOME.joinpath(*parts)


def _argus_path(*parts: str) -> Path:
    """Build a path under ARGUS_HOME (~/hermes)."""
    return _ARGUS_HOME.joinpath(*parts)


def build_argus_subprocess_env() -> Dict[str, str]:
    """Build a full environment dict for subprocess calls in sandboxed contexts."""
    env = os.environ.copy()

    # Ensure PATH includes all critical tool locations
    paths = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        str(_argus_path("bin")),  # ~/hermes/bin
        str(Path.home() / ".local" / "bin"),  # ~/.local/bin
        str(_hermes_path("credentials")),  # ~/.hermes/credentials
        "/usr/bin",
        "/bin",
    ]
    env["PATH"] = ":".join(paths)

    # Ensure HOME is set (some launchd contexts may not have it)
    env["HOME"] = os.path.expanduser("~")

    return env


def safe_subprocess(
    cmd: List[str], timeout: int = 10, **kwargs
) -> Optional[subprocess.CompletedProcess]:
    """Run a subprocess with full env and error handling. Never raises."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=build_argus_subprocess_env(),
            **kwargs,
        )
    except FileNotFoundError:
        logger.warning("Command not found: %s (check PATH)", cmd[0])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out after %ss: %s", timeout, " ".join(cmd))
        return None
    except Exception as e:
        logger.error("Subprocess error for %s: %s", cmd[0], e, exc_info=True)
        return None
