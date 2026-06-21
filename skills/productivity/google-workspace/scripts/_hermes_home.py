"""Resolve HERMES_HOME for standalone skill scripts.

Skill scripts may run outside the Hermes process (e.g. system Python,
nix env, CI) where ``hermes_constants`` is not importable.  This module
provides the same ``get_hermes_home()`` and ``display_hermes_home()``
contracts as ``hermes_constants`` without requiring it on ``sys.path``.

When ``hermes_constants`` IS available it is used directly so that any
future enhancements (profile resolution, Docker detection, etc.) are
picked up automatically.  The fallback path replicates the core logic
from ``hermes_constants.py`` using only the stdlib.

All scripts under ``google-workspace/scripts/`` should import from here
instead of duplicating the ``HERMES_HOME = Path(os.getenv(...))`` pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from hermes_constants import display_hermes_home as display_hermes_home
    from hermes_constants import get_default_hermes_root as get_default_hermes_root
    from hermes_constants import get_hermes_home as get_hermes_home
except (ModuleNotFoundError, ImportError):

    def get_hermes_home() -> Path:
        """Return the Hermes home directory (default: ~/.hermes).

        Mirrors ``hermes_constants.get_hermes_home()``."""
        val = os.environ.get("HERMES_HOME", "").strip()
        return Path(val) if val else Path.home() / ".hermes"

    def get_default_hermes_root() -> Path:
        """Return the root Hermes dir for profile-level / host-wide files.

        Mirrors ``hermes_constants.get_default_hermes_root()``: when
        ``HERMES_HOME`` is a profile path (``<root>/profiles/<name>``) or sits
        under ``~/.hermes``, return the native root so host-wide files (the
        shared OAuth client secret) are found regardless of active profile.
        A custom root that uses profiles returns its own root
        (``<root>/profiles/<name>`` → ``<root>``); a custom non-profile
        ``HERMES_HOME`` is returned unchanged.
        """
        native_home = Path.home() / ".hermes"
        env_home = os.environ.get("HERMES_HOME", "").strip()
        if not env_home:
            return native_home
        env_path = Path(env_home)
        try:
            env_path.resolve().relative_to(native_home.resolve())
            return native_home
        except ValueError:
            pass
        if env_path.parent.name == "profiles":
            return env_path.parent.parent
        return env_path

    def display_hermes_home() -> str:
        """Return a user-friendly ``~/``-shortened display string.

        Mirrors ``hermes_constants.display_hermes_home()``."""
        home = get_hermes_home()
        try:
            return "~/" + str(home.relative_to(Path.home()))
        except ValueError:
            return str(home)


def client_secret_path() -> Path:
    """Resolve the shared OAuth ``google_client_secret.json`` (host-wide app
    credential).

    Anchored at the default Hermes root so a gateway running under a named
    profile sees the one-time host setup; a profile-local copy wins when it is
    an actual file. Centralizes the precedence so the standalone scripts
    (``setup.py``/``google_api.py``) cannot drift. Mirrors the google_chat fix
    in commit fff056144.

    ``is_file()`` (not ``exists()``) gates the profile-local override so a
    same-named *directory* never shadows the real default-root secret.
    """
    profile_local = get_hermes_home() / "google_client_secret.json"
    if profile_local.is_file():
        return profile_local
    return get_default_hermes_root() / "google_client_secret.json"
