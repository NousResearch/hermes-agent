"""Public spawn helper for kanban workers.

Thin, importable wrapper around the ``hermes -p <profile> chat -q ...`` cmd
that the kanban dispatcher builds for every worker subprocess. Exposed so:

  * ad-hoc scripts and tests can build the same argv the dispatcher would,
    without copying the construction logic,
  * the dispatcher's internal ``_default_spawn`` can share the canonical
    builder, so a future change to the spawn shape (extra flag, new
    placeholder for skill preload, etc.) lands in one place,
  * downstream tooling that wants to invoke a profile the way the
    dispatcher would (e.g. local rehearsals, scenario-runner helpers,
    ``hermes-demo`` style manual fan-outs) has a single source of truth.

The function is intentionally narrow: it returns the argv list, the same
shape ``subprocess.Popen`` expects. It does NOT spawn the process, set up
the per-task log, or pin the board — those are dispatch concerns and stay
in ``hermes_cli.kanban_db._default_spawn``. The split is the same as
``argparse`` building a parser and the CLI handler executing it: builders
build, dispatchers dispatch.

Why a separate module (not a helper in ``kanban_db``): ``kanban_db`` is the
hot-path DB module imported at every CLI invocation; adding new public
imports there tightens the import graph for everyone. Keeping the helper
in its own file also lets the KPI tests assert against the public shape
without touching the DB.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence


def _safe_which_no_cwd(name: str) -> Optional[str]:
    """``shutil.which`` that ignores the current directory on Windows.

    Background: on Windows, ``shutil.which`` will return a path with a
    leading ``.\\`` when the command is found in CWD. That is unsafe as
    ``argv[0]`` for ``Popen`` because it lets a same-directory file win
    over a system-installed command. Mirrors the helper in
    ``hermes_cli.kanban_db`` for the same reason.
    """
    import shutil
    import sys

    found = shutil.which(name)
    if not found:
        return None
    if sys.platform.startswith("win") and (found.startswith(".\\") or found.startswith("./")):
        return None
    return found


def _looks_like_path(value: str) -> bool:
    """True if ``value`` is filesystem-path-shaped (has a separator or a
    drive letter). Bare command names like ``"hermes"`` return False so the
    PATH lookup still applies."""
    if not value:
        return False
    if value[0] in ("\\", "/") or (len(value) >= 2 and value[1] == ":"):
        return True
    if "\\" in value or "/" in value:
        return True
    return False


def _module_hermes_argv() -> list[str]:
    """Fallback argv when no ``hermes`` shim is on PATH: launch the current
    Python with ``-m hermes_cli.main`` so the result is independent of
    ``$PATH`` (cron, systemd ``User=`` services, detached jobs, etc.)."""
    import sys
    return [sys.executable, "-m", "hermes_cli.main"]


def _hermes_path_argv(path: str) -> list[str]:
    """Normalize a resolved ``hermes`` shim path to an absolute argv.

    Path-like values are passed through absolute. Bare command names keep
    normal PATH semantics.
    """
    p = Path(path)
    if p.is_absolute():
        return [str(p)]
    return [str(p)]


def resolve_hermes_argv() -> list[str]:
    """Resolve the ``hermes`` invocation as argv parts for ``Popen``.

    Tries in order:
      1. ``$HERMES_BIN`` — explicit operator override.
      2. ``shutil.which("hermes")`` — the console-script shim.
      3. ``sys.executable -m hermes_cli.main`` — fallback for setups
         where the shim is not on PATH.

    Mirrors ``hermes_cli.kanban_db._resolve_hermes_argv``; kept as a public
    re-export so callers do not have to reach into the private module.
    """
    env_bin = os.environ.get("HERMES_BIN", "").strip()
    if env_bin:
        if _looks_like_path(env_bin):
            return _hermes_path_argv(env_bin)
        resolved_env_bin = _safe_which_no_cwd(env_bin)
        if resolved_env_bin:
            return _hermes_path_argv(resolved_env_bin)
        return _module_hermes_argv()

    hermes_bin = _safe_which_no_cwd("hermes")
    if hermes_bin:
        return _hermes_path_argv(hermes_bin)
    return _module_hermes_argv()


def build_spawn_cmd(
    profile: str,
    prompt: str,
    *,
    model: Optional[str] = None,
    skills: Optional[Sequence[str]] = None,
    accept_hooks: bool = True,
    extra_args: Optional[Sequence[str]] = None,
) -> list[str]:
    """Build the argv list for ``hermes -p <profile> chat -q <prompt>``.

    This is the canonical command shape used by the kanban dispatcher for
    every worker subprocess, factored out so callers and tests can build
    the same argv without going through the dispatcher.

    Args:
      profile: profile name (becomes the ``-p`` value).
      prompt: the worker's opening prompt (becomes the ``-q`` value).
      model: optional per-task model override. When set, ``-m <model>`` is
        appended, and the dispatched worker uses that model id in place of
        the profile's default. Mirrors ``hermes kanban create --model``.
      skills: optional iterable of skill names to force-load into the
        worker via repeated ``--skills <name>`` flags. ``kanban-worker``
        is added automatically when the home under which the worker will
        run actually ships the bundled skill (the dispatcher checks the
        same precondition to avoid a fatal "Unknown skill" error).
      accept_hooks: when True (default), pass ``--accept-hooks`` so
        profile-local hook allowlists are honored. Profile-scoped workers
        use the profile's allowlist, not the dispatcher's root allowlist,
        so the flag is needed to register the hooks again.
      extra_args: optional extra argv parts appended after the standard
        shape (before ``chat -q <prompt>``). Escape hatch for one-off
        flags; prefer the named kwargs for the supported ones.

    Returns:
      A list of argv strings, suitable for ``subprocess.Popen(cmd, ...)``.

    Examples:
      >>> cmd = build_spawn_cmd("coder", "work kanban task t_abc",
      ...                       model="kimi-k2.7-code",
      ...                       skills=["test-driven-development"])
      >>> "coder" in cmd and "kimi-k2.7-code" in cmd
      True
    """
    profile = (profile or "").strip()
    if not profile:
        raise ValueError("profile is required")
    if prompt is None:
        raise ValueError("prompt is required")

    cmd: list[str] = list(resolve_hermes_argv())
    cmd.extend(["-p", profile])
    if accept_hooks:
        cmd.append("--accept-hooks")
    # Per-task force-loaded skills. Same shape as the dispatcher:
    # one ``--skills X`` pair per name, easy to grep in ``ps`` output.
    skill_list = list(skills or [])
    for sk in skill_list:
        if sk and sk != "kanban-worker":
            cmd.extend(["--skills", sk])
    if model and model.strip():
        cmd.extend(["-m", model.strip()])
    if extra_args:
        cmd.extend(list(extra_args))
    cmd.extend(["chat", "-q", prompt])
    return cmd
