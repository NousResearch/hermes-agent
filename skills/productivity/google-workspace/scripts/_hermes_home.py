from __future__ import annotations

import os
from pathlib import Path

try:
    from hermes_constants import display_hermes_home, get_hermes_home
except Exception:

    def get_hermes_home() -> Path:
        env_home = os.environ.get("HERMES_HOME", "").strip()
        if env_home:
            return Path(env_home).expanduser()
        return Path.home() / ".hermes"

    def display_hermes_home() -> str:
        home = Path.home()
        hermes_home = get_hermes_home()
        try:
            rel = hermes_home.relative_to(home)
        except ValueError:
            env_home = os.environ.get("HERMES_HOME", "").strip()
            if env_home:
                return env_home
            return str(hermes_home)
        if str(rel) == ".":
            return "~"
        return "~/" + rel.as_posix()
