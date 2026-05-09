"""
Early HERMES_HOME resolution for multi-profile deployments.

``hermes_cli.main`` historically applied profile selection before importing the
rest of the CLI. Background services that import ``gateway.run`` directly must
reuse the same rules so systemd can keep ``HERMES_HOME`` anchored at the
canonical root while sticky ``active_profile`` still redirects into
``profiles/<name>`` (Issue #22502).

Child processes keep an explicit ``HERMES_HOME`` that already points outside
that canonical anchor (typically a concrete profile directory) — we bail out
quickly then so subprocess relaunches inherit the parent's choice.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Optional

_PROFILE_ID_TAIL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def apply_profile_env_override(cli_argv: Optional[List[str]] = None) -> None:
    """Pre-parse profile flags / ``active_profile`` before ``get_hermes_home()`` reads env."""
    mutate_sys_argv = cli_argv is None
    argv: List[str] = list(sys.argv[1:] if mutate_sys_argv else cli_argv)

    profile_name = None
    consume = 0

    for i, arg in enumerate(argv):
        if arg in ("--profile", "-p") and i + 1 < len(argv):
            profile_name = argv[i + 1]
            consume = 2
            break
        if arg.startswith("--profile="):
            profile_name = arg.split("=", 1)[1]
            consume = 1
            break

    if profile_name is not None and consume == 2:
        if not _PROFILE_ID_TAIL_RE.match(profile_name):
            profile_name = None
            consume = 0

    if profile_name is None and os.environ.get("HERMES_HOME"):
        from hermes_cli.profiles import get_profile_dir

        try:
            env_home = Path(os.environ["HERMES_HOME"]).expanduser().resolve()
            canon_anchor = get_profile_dir("default").expanduser().resolve()
        except OSError:
            return
        if env_home != canon_anchor:
            return

    if profile_name is None:
        try:
            from hermes_constants import get_default_hermes_root

            active_path = get_default_hermes_root() / "active_profile"
            if active_path.exists():
                name = active_path.read_text(encoding="utf-8").strip()
                if name and name != "default":
                    profile_name = name
                    consume = 0
        except (UnicodeDecodeError, OSError):
            pass

    if profile_name is None:
        return

    try:
        from hermes_cli.profiles import resolve_profile_env

        hermes_home = resolve_profile_env(profile_name)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(
            f"Warning: profile override failed ({exc}), using default",
            file=sys.stderr,
        )
        return

    os.environ["HERMES_HOME"] = hermes_home

    if consume > 0 and mutate_sys_argv:
        for i, arg in enumerate(argv):
            if arg in ("--profile", "-p"):
                start = i + 1
                sys.argv = sys.argv[:start] + sys.argv[start + consume :]
                break
            if arg.startswith("--profile="):
                start = i + 1
                sys.argv = sys.argv[:start] + sys.argv[start + 1 :]
                break
