"""Which commands may run in the default home when the sticky ``active_profile`` is gone.

``_apply_profile_override`` fails closed when ``active_profile`` names a deleted profile:
running an ordinary command in the default profile would read or write the wrong profile's
state. The commands here are the way out of that state, so they must still reach their handler.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import contextlib
import io
from pathlib import Path


def session_or_scoped_home_override(
    profile_name: str | None, environ: Mapping[str, str],
) -> tuple[str | None, bool]:
    """Resolve an active session or scoped ``HERMES_HOME`` before sticky ``active_profile``.

    Returns ``(profile_name, keep_current_home)``:
    - Explicit ``-p``/``--profile`` (*profile_name* already set) always wins.
    - ``HERMES_HOME`` pointing at ``.../profiles/<name>`` is already profile-scoped
      (including a profile-routed subagent targeting a named profile) and is kept as-is.
    - A delegated/worker descendant (``HERMES_DELEGATED_CHILD_CONTEXT``) with an explicit
      ``HERMES_HOME`` (e.g. a subagent routed back to ``default`` while the parent chat's
      ``HERMES_SESSION_PROFILE`` names another profile) keeps that ``HERMES_HOME`` without
      consulting ``HERMES_SESSION_PROFILE`` or the sticky ``active_profile``.
    - Otherwise a non-empty ``HERMES_SESSION_PROFILE`` pins bare CLI invocations inside a
      session to that session's profile (``"custom"`` keeps the custom ``HERMES_HOME``;
      ``"default"`` or a named profile resolves via ``resolve_profile_env``), preventing
      ``active_profile`` from re-homing mutating commands into another profile (#159008).
    """
    if profile_name is not None:
        return profile_name, False
    hermes_home_env = environ.get("HERMES_HOME", "")
    if hermes_home_env != "" and Path(hermes_home_env).parent.name == "profiles":
        return None, True
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER

    if environ.get(DELEGATED_CHILD_ENV_MARKER, "") != "" and hermes_home_env != "":
        return None, True
    session_profile = environ.get("HERMES_SESSION_PROFILE", "").strip()
    if session_profile == "":
        return None, False
    if session_profile == "custom":
        return None, True
    return session_profile, False


def _uninstall_keeps_data(args: list[str]) -> bool:
    """True when ``uninstall <args>`` removes no user data.

    Parsed with the real ``uninstall`` subparser so abbreviations (``--dat``, ``--fu``) resolve
    exactly as they will for the handler. ``--full`` wipes the default root, which also holds
    every other profile under ``profiles/``, without asking, so it is refused like ``--data``.
    The plain interactive menu still offers a full wipe, but only behind its own confirmations.
    """
    from hermes_cli.subcommands.uninstall import build_uninstall_parser

    parser = argparse.ArgumentParser(add_help=False)
    build_uninstall_parser(parser.add_subparsers(dest="command"), cmd_uninstall=lambda _args: None)
    # The real parser prints help/usage errors once the handler runs; this probe stays silent.
    # --help exits 0 (harmless, let it through); an unparseable argv fails closed.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            ns, _ = parser.parse_known_args(["uninstall", *args])
        except SystemExit as exit_:
            return exit_.code == 0
    return not (ns.data or ns.full)


def is_stale_profile_recovery_command(argv: list[str]) -> bool:
    """``profile list``, ``profile use default``, or an ``uninstall`` that keeps user data."""
    from hermes_cli._parser import command_argv

    command = command_argv(argv)
    if command[:2] == ["profile", "list"]:
        return True
    if command[:2] == ["profile", "use"]:
        return len(command) > 2 and command[2].casefold() == "default"
    return command[:1] == ["uninstall"] and _uninstall_keeps_data(command[1:])
