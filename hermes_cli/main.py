#!/usr/bin/env python3
"""Hermes CLI - Main entry point.

Usage:
    hermes                     # Interactive chat (default)
    hermes chat / gateway / setup / status / cron / doctor / update / ...
    hermes --version           # Show version and update status
    hermes <cmd> --help        # Per-command help
"""

# hermes_bootstrap must be the very first import — it sets up UTF-8 stdio on
# Windows (no-op on POSIX). Guarded: after a ``git pull`` / interrupted
# ``hermes update`` the editable install's ``.pth`` may not list it yet; crashing
# here would block ``hermes update``.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap":
        raise  # the bootstrap exists but cannot load: skipping it would skip PM activation

# A `hermes update` killed while git was writing the new tree leaves a mix of old and new files that
# fails at the next import, whichever it is — put the old tree back before importing anything else
# from the checkout, then rerun the command (this module may itself be one of the new files).
# ``_early_recovery`` is stdlib-only and imported unguarded on purpose: same package
# dir, so if IT can't import nothing in hermes_cli can.
from hermes_cli import _early_recovery as _early_recovery_mod

if _early_recovery_mod.restore_interrupted_pull():
    _early_recovery_mod.relaunch_after_restore()

# Windows: neutralize CPython's ``platform._syscmd_ver`` before anything else
# imports — it shells out ``cmd /c ver`` and flashes a console when this
# process is windowless (pythonw gateway, kanban workers). No-op on POSIX.
from hermes_cli._subprocess_compat import suppress_platform_ver_console

suppress_platform_ver_console()

import os
import re
import sys

# Inline path math so ``python hermes_cli/main.py`` (script mode: sys.path[0]
# is hermes_cli/, not the repo root) can import hermes_cli._startup_fast.
_bootstrap_root = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
if _bootstrap_root not in sys.path:
    sys.path.insert(0, _bootstrap_root)
from hermes_cli import _startup_fast  # noqa: E402

# A literal ``~``/``$VAR`` in HERMES_HOME (fish, or any quoted value) must become absolute
# before the first reader — otherwise it resolves against cwd and scaffolds <cwd>/~/.hermes.
_startup_fast.normalize_hermes_home_env()

# Startup-liveness watchdog: for gateway runs, arm BEFORE the heavy import
# graph below — an import-time deadlock (native-extension init, contended
# import lock) is exactly the "wedged before the event loop, no logs, live
# PID" class it exists for. ``hermes_startup_watchdog`` is stdlib-only so it
# cannot itself wedge. The match requires the ADJACENT pair ``gateway run``
# (wherever global flags like ``-p <profile>`` put it) so unrelated commands
# mentioning both words never arm a 300s hard-exit timer. Foreground runs arm
# too — a pre-loop wedge is just as dead without a supervisor; GatewayRunner
# disarms once the event loop is live.
def _argv_is_gateway_run(argv: list) -> bool:
    return any(a == "gateway" and b == "run" for a, b in zip(argv, argv[1:]))


if _argv_is_gateway_run(sys.argv[1:]):
    try:
        from hermes_startup_watchdog import arm_startup_watchdog as _arm_sw

        _arm_sw()
        del _arm_sw
    except Exception:
        pass


def _exit_after_oneshot(rc: object) -> None:
    """Exit one-shot mode without letting late native finalizers change rc.

    The SIGABRT this guards against fires in a native-extension finalizer
    during ``Py_FinalizeEx``, *after* the response printed. Flush, shut down
    file logging, then ``os._exit`` past finalization. The ``atexit`` chain is
    deliberately skipped — several handlers re-enter native code that may be
    the abort source; stateful cleanup lives in ``_cleanup_oneshot_runtime``.

    See #30387, #43055.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    try:
        logging.shutdown()
    except Exception:
        pass
    os._exit(rc if isinstance(rc, int) else (0 if rc is None else 1))


_oneshot_cleanup_done = False
# (module, attr, kwargs, exceptions swallowed). MCP shutdown may raise
# BaseException-derived errors from executor teardown; the rest are Exception.
_ONESHOT_CLEANUPS = (
    ("tools.terminal_tool", "cleanup_all_environments", {}, Exception),
    ("tools.async_delegation", "interrupt_all", {"reason": "oneshot shutdown"}, Exception),
    ("tools.browser_tool_lifecycle", "_emergency_cleanup_all_sessions", {}, Exception),
    ("tools.mcp_tool_lifecycle", "shutdown_mcp_servers", {}, BaseException),
    ("agent.auxiliary_client", "shutdown_cached_clients", {}, Exception),
)


def _cleanup_oneshot_runtime() -> None:
    """Best-effort process-global cleanup before one-shot hard exit.

    ``run_oneshot`` owns the agent-local cleanup (memory provider, agent.close,
    session_db.close — all in ``_run_agent``'s finally block). This mirrors the
    process-global pieces from ``cli.py:_run_cleanup()`` that would otherwise
    be skipped by ``os._exit``.
    """
    global _oneshot_cleanup_done
    if _oneshot_cleanup_done:
        return
    _oneshot_cleanup_done = True
    import importlib

    for module, attr, kwargs, swallow in _ONESHOT_CLEANUPS:
        try:
            getattr(importlib.import_module(module), attr)(**kwargs)
        except swallow:
            pass


def _run_and_exit_oneshot(
    prompt: str,
    *,
    model: object = None,
    provider: object = None,
    toolsets: object = None,
    skills: object = None,
    usage_file: object = None,
    resume: object = None,
    reasoning: object = None,
) -> None:
    try:
        from hermes_cli.oneshot import run_oneshot

        rc = run_oneshot(
            prompt,
            model=model,
            provider=provider,
            toolsets=toolsets,
            skills=skills,
            usage_file=usage_file,
            resume=resume,
            reasoning=reasoning,
        )
    except KeyboardInterrupt:
        rc = 130
    except SystemExit as exc:
        if exc.code is not None and not isinstance(exc.code, int):
            print(exc.code, file=sys.stderr)
            rc = 1
        else:
            rc = exc.code
    except BaseException:
        # ``run_oneshot`` already maps agent failures to an int rc; anything
        # still escaping means it malfunctioned. Print it but never fall
        # through to interpreter teardown (the SIGABRT path this routine fixes).
        import traceback
        try:
            traceback.print_exc()
        except Exception:
            pass
        rc = 1
    try:
        _cleanup_oneshot_runtime()
    finally:
        # Even an interrupt during cleanup must not fall back into interpreter
        # finalization, where the native SIGABRT occurs.
        # The hard exit is the safety boundary for #43055.
        _exit_after_oneshot(rc)


def _warn_if_unsupervised_pid1(pid: "int | None" = None) -> None:
    """Warn when this process is PID 1 with nothing above it to reap orphans.

    The official image's ENTRYPOINT (``docker/entrypoint-dispatch.sh`` -> s6-overlay's
    ``/init``) is the reaper for orphaned grandchildren (browser tooling, MCP servers, shell
    children). A Compose service that overrides ``entrypoint:`` to invoke hermes directly makes
    hermes itself PID 1 — nothing then ``wait()``s on those orphans and they accumulate as
    zombies without bound (#111577). Outside a container a user process is never PID 1, so
    this is quiet everywhere else; it mirrors the dispatcher's own non-PID-1 warning.
    """
    if (pid if pid is not None else os.getpid()) != 1:
        return
    print(
        "[hermes] WARNING: this process is PID 1 with no init above it "
        "(entrypoint override?). Orphaned child processes will not be "
        "reaped and will accumulate as zombies. Use the image's default "
        "ENTRYPOINT (docker/entrypoint-dispatch.sh) instead of overriding "
        "it, or run with `docker run --init` / `init: true` in Compose.",
        file=sys.stderr,
    )


def _set_process_title() -> None:
    """Cosmetic: show 'hermes' instead of 'python3.xx' in ps/top/htop.

    Order: opt-in ``setproctitle`` dep; ctypes ``prctl(PR_SET_NAME)`` (Linux,
    15-char limit); ``pthread_setname_np`` (macOS — lldb/top only, not ``ps
    aux``); no-op on Windows (the .exe is already ``hermes.exe``). Never fatal.
    """
    try:
        import setproctitle  # type: ignore[import-untyped]

        setproctitle.setproctitle("hermes")
        return
    except ImportError:
        pass

    import ctypes
    import platform

    try:
        system = platform.system()
        if system == "Linux":
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(15, b"hermes", 0, 0, 0)  # PR_SET_NAME = 15
        elif system == "Darwin":
            libc = ctypes.CDLL("libc.dylib", use_errno=True)
            libc.pthread_setname_np(b"hermes")
    except Exception:
        pass


# Cheap read of `display.interface` for the earliest hot-path decisions
# (mouse-residue suppression, Termux fast launch) that run before
# hermes_cli.config is importable. Cached per config path so early callers
# don't re-parse YAML, and so the answer follows the home the process ends up
# in: mouse-residue suppression reads this BEFORE `_apply_profile_override()`
# sets HERMES_HOME, and a cache keyed on nothing pinned every later caller to
# the default home's interface for the whole run (#116902).
_EARLY_INTERFACE_CACHE: "tuple[str, str] | None" = None


def _early_interface_config_path() -> str:
    """config.yaml of the home this process is currently pointed at."""
    home = os.environ.get("HERMES_HOME")
    if home:
        return os.path.join(home, "config.yaml")
    return os.path.join(os.path.expanduser("~"), ".hermes", "config.yaml")


def _config_default_interface_early() -> str:
    """Return the configured default interface ("cli"/"tui") via a minimal
    YAML read. Best-effort: any error falls back to "cli" (legacy behavior)."""
    global _EARLY_INTERFACE_CACHE
    cfg_path = _early_interface_config_path()
    if _EARLY_INTERFACE_CACHE is not None and _EARLY_INTERFACE_CACHE[0] == cfg_path:
        return _EARLY_INTERFACE_CACHE[1]
    value = "cli"
    try:
        if os.path.exists(cfg_path):
            import hermes_yaml as _yaml_iface

            with open(cfg_path, encoding="utf-8-sig") as _f:
                raw = _yaml_iface.safe_load(_f) or {}
            disp = raw.get("display", {})
            if isinstance(disp, dict):
                iface = disp.get("interface")
                if isinstance(iface, str) and iface.strip().lower() == "tui":
                    value = "tui"
    except Exception:
        value = "cli"  # best-effort — default to classic REPL on any error
    _EARLY_INTERFACE_CACHE = (cfg_path, value)
    return value


def _wants_tui_early(argv: "list[str] | None" = None) -> bool:
    """Earliest TUI decision, usable before argparse/config imports.

    Precedence: ``--cli`` wins, then ``--tui``/``HERMES_TUI=1``, then a
    real-TTY gate, then ``display.interface``. The TTY gate is load-bearing
    for headless spawners (kanban workers, cron, pipes running ``chat -q``):
    a ``display.interface: tui`` default used to boot the TUI here, whose
    no-TTY bail-out exits 0 without doing the task. An explicit ``--tui``
    still reaches that informative bail-out.
    """
    if argv is None:
        argv = sys.argv[1:]
    if "--cli" in argv:
        return False
    if os.environ.get("HERMES_TUI") == "1" or any(flag in argv for flag in ("--tui", "--native", "--tui-native")):
        return True
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except Exception:
        return False
    return _config_default_interface_early() == "tui"


# Mouse-tracking residue suppression — runs BEFORE every other import on the
# TUI hot path: while the launcher is still importing (~100-300ms, cooked+echo
# mode, before the Node TUI takes stdin raw) incoming SGR/X10 mouse reports
# echo into the shell scrollback as ``^[[<…M``. entry.tsx's
# `resetTerminalModes()` is the later cousin. ``HERMES_TUI_NO_EARLY_DISABLE``
# escapes the behaviour for diagnostics.
def _suppress_mouse_residue_early() -> None:
    if os.environ.get("HERMES_TUI_NO_EARLY_DISABLE") == "1":
        return
    if not _wants_tui_early():
        return
    try:
        if not os.isatty(1):  # redirected stdout: raw CSI would pollute the log
            return
        # Every mouse-tracking variant we know about; idempotent.
        os.write(
            1,
            b"\x1b[?1003l\x1b[?1002l\x1b[?1001l\x1b[?1000l\x1b[?9l"
            b"\x1b[?1006l\x1b[?1005l\x1b[?1015l\x1b[?1016l\x1b[?2029l",
        )
    except OSError:
        pass


_suppress_mouse_residue_early()


_startup_fast.ensure_project_root_on_path()

# ``hermes --version`` is answered before config/logging imports.
if _startup_fast.try_fast_version():
    raise SystemExit(0)

import argparse
import contextlib
import json
import shutil
import subprocess
import tempfile
import secrets
from pathlib import Path
from typing import Optional


from hermes_cli.subcommands.cron import build_cron_parser
from hermes_cli.subcommands.sync import build_sync_parser
from hermes_cli.subcommands.gateway import build_gateway_parser
from hermes_cli.subcommands.profile import build_profile_parser
from hermes_cli.subcommands.model import build_model_parser
from hermes_cli.subcommands.setup import build_setup_parser

from hermes_cli.subcommands.whatsapp import build_whatsapp_parser, build_whatsapp_cloud_parser
from hermes_cli.subcommands.slack import build_slack_parser
from hermes_cli.subcommands.login import build_login_parser
from hermes_cli.subcommands.logout import build_logout_parser
from hermes_cli.subcommands.auth import build_auth_parser
from hermes_cli.subcommands.status import build_status_parser
from hermes_cli.subcommands.pause import build_pause_parser
from hermes_cli.subcommands.webhook import build_webhook_parser
from hermes_cli.subcommands.hooks import build_hooks_parser
from hermes_cli.subcommands.doctor import build_doctor_parser
from hermes_cli.subcommands.verify import build_verify_parser
from hermes_cli.subcommands.security import build_security_parser
from hermes_cli.subcommands.approvals import build_approvals_parser
from hermes_cli.subcommands.dump import build_dump_parser
from hermes_cli.subcommands.debug import build_debug_parser
from hermes_cli.subcommands.backup import build_backup_parser
from hermes_cli.subcommands.import_cmd import build_import_cmd_parser
from hermes_cli.subcommands.import_agent import build_import_agent_parser
from hermes_cli.subcommands.config import build_config_parser
from hermes_cli.subcommands.skin import build_skin_parser
from hermes_cli.subcommands.console import build_console_parser
from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.subcommands.uninstall import build_uninstall_parser
from hermes_cli.subcommands.desktop_web import build_desktop_web_parser
from hermes_cli.subcommands.dashboard import build_dashboard_parser, build_serve_parser
from hermes_cli.subcommands.gui import build_gui_parser
from hermes_cli.subcommands.logs import build_logs_parser
from hermes_cli.subcommands.prompt_size import build_prompt_size_parser
from hermes_cli.subcommands.memory import build_memory_parser
from hermes_cli.subcommands.acp import build_acp_parser
from hermes_cli.subcommands.tools import build_tools_parser
from hermes_cli.subcommands.insights import build_insights_parser
from hermes_cli.subcommands.usage import build_usage_parser
from hermes_cli.subcommands.monitoring import build_monitoring_parser
from hermes_cli.subcommands.skills import build_skills_parser
from hermes_cli.subcommands.pairing import build_pairing_parser
from hermes_cli.subcommands.plugins import build_plugins_parser
from hermes_cli.subcommands.mcp import build_mcp_parser
from hermes_cli.subcommands.claw import build_claw_parser
from hermes_cli.subcommands.vault import build_vault_parser
from hermes_cli.subcommands.moa import build_moa_parser
from hermes_cli.subcommands.fallback import build_fallback_parser
from hermes_cli.subcommands.worktree import build_worktree_parser
from hermes_cli.subcommands.browser import build_browser_parser
from hermes_cli.subcommands.secrets import build_secrets_parser
from hermes_cli.subcommands.codex_runtime import build_codex_runtime_parser
from hermes_cli.subcommands.egress import build_egress_parser
from hermes_cli.subcommands.migrate import build_migrate_parser
from hermes_cli.subcommands.checkpoints import build_checkpoints_parser
from hermes_cli.subcommands.bundles import build_bundles_parser
from hermes_cli.subcommands.curator import build_curator_parser
from hermes_cli.subcommands.pets import build_pets_parser
from hermes_cli.subcommands.journey import build_journey_parser
from hermes_cli.subcommands.computer_use import build_computer_use_parser
from hermes_cli.subcommands.sessions import build_sessions_parser
from hermes_cli.subcommands.completion import build_completion_parser


def _require_tty(command_name: str) -> None:
    """Exit 1 if stdin is not a terminal: curses/input() prompts spin at 100% CPU on a pipe."""
    if not sys.stdin.isatty():
        print(
            f"Error: 'hermes {command_name}' requires an interactive terminal.\n"
            f"It cannot be run through a pipe or non-interactive subprocess.\n"
            f"Run it directly in your terminal instead.",
            file=sys.stderr,
        )
        sys.exit(1)


PROJECT_ROOT = Path(_startup_fast.project_root_str())
_startup_fast.ensure_project_root_on_path()


# Profile override — MUST happen before any hermes module import: many modules
# cache HERMES_HOME at import time. --profile/-p is pre-parsed from sys.argv,
# HERMES_HOME set, and the flag stripped so argparse never sees it. Falls back
# to ~/.hermes/active_profile for the sticky default.
_PROFILE_NAME_RE = r"^[a-z0-9][a-z0-9_-]{0,63}$"  # mirrors hermes_cli.profiles._PROFILE_ID_RE
# Set only when -p/--profile was on argv. Sticky active_profile must not count:
# `hermes desktop` with no flag must not overwrite Desktop's stored profile.
_explicit_cli_profile: str | None = None


def _inside_mcp_add_args(argv: list, index: int) -> bool:
    """True once argv reaches `hermes mcp add ... --args <command argv>`.

    ``mcp add --args`` is command-argv passthrough. Flags after that point
    belong to the child MCP command (for example Docker MCP Toolkit's
    ``--profile``), not to Hermes' own profile selector.
    """
    try:
        mcp_index = argv.index("mcp", 0, index)
        argv.index("add", mcp_index + 1, index)
    except ValueError:
        return False
    return True


def _looks_like_hermes_invocation() -> bool:
    """False when ``sys.argv`` belongs to a test runner rather than a ``hermes`` run.

    pytest's own ``-p no:xdist`` reaches ``_scan_profile_flag`` through ``sys.argv`` at import
    time; it must stay a silent skip, while a real ``hermes -p 'Work Bot'`` must fail loudly.
    """
    return "pytest" not in (sys.argv[0] or "")


def _exit_invalid_profile_name(value: str) -> None:
    from hermes_cli.profiles import _invalid_profile_name_error

    print(f"Error: {_invalid_profile_name_error(value)}", file=sys.stderr)
    print("Run `hermes profile list` to see your profiles.", file=sys.stderr)
    sys.exit(2)


def _looks_like_option_value(value: str) -> bool:
    """A ``-p`` value that clearly belongs to some other tool (pytest's ``-p no:xdist``, a
    third-party ``-p --flag``), never a mistyped profile name."""
    return value.startswith("-") or ":" in value


def _scan_profile_flag(argv: list) -> tuple:
    """Find -p/--profile/--profile= in argv -> (name, tokens_consumed, index).

    Historically the flag worked even after the subcommand (`hermes chat -p
    coder`), so scan broadly; stop at ``--`` and at the `mcp add --args`
    passthrough region. The value is normalised (strip + casefold, matching
    ``profiles.normalize_profile_name``) before validation so ``-p Work`` selects
    ``work``. A value that cannot be a profile name is rejected so
    resolve_profile_env never sys.exits on it; the rejection is explained (exit 2)
    only when the flag comes BEFORE the first subcommand token under a real
    ``hermes`` run — after a subcommand, ``-p`` may belong to that subcommand or a
    plugin (`hermes kanban ... -p 8080`), and option-looking values (``no:xdist``,
    ``--flag``) are always a silent skip.
    """
    from hermes_cli._parser import top_level_value_flag_sets

    value_flags, optional_value_flags = top_level_value_flag_sets()
    i = 0
    saw_subcommand = False
    while i < len(argv):
        arg = argv[i]
        if arg == "--" or (arg == "--args" and _inside_mcp_add_args(argv, i)):
            break
        if arg in {"--profile", "-p"} and i + 1 < len(argv):
            raw = argv[i + 1]
            value = raw.strip().casefold()
            if re.match(_PROFILE_NAME_RE, value):
                return value, 2, i
            if not saw_subcommand and not _looks_like_option_value(raw) and _looks_like_hermes_invocation():
                _exit_invalid_profile_name(raw)
            break
        if arg.startswith("--profile="):
            return arg.split("=", 1)[1].strip().casefold(), 1, i
        takes_value = "=" not in arg and i + 1 < len(argv) and (
            arg in value_flags
            or (arg in optional_value_flags and not argv[i + 1].startswith("-"))
        )
        if not takes_value and not arg.startswith("-"):
            saw_subcommand = True
        i += 2 if takes_value else 1
    return None, 0, None


def _resolve_sudo_user_profile_env(name: str) -> str | None:
    """Resolve `sudo hermes -p <name>` against the invoking user's home.

    This runs before argparse, so `--run-as-user` is not available yet. For
    sudo invocations the best signal is SUDO_USER: root is only doing the
    privileged install/start action; the profile store belongs to the user.
    """
    if name == "default":
        return None
    from hermes_constants import named_profile_is_live, sudo_invoker_default_home

    sudo_home = sudo_invoker_default_home()
    if sudo_home is None:
        return None
    candidate = sudo_home / "profiles" / name
    return str(candidate) if named_profile_is_live(candidate) else None


def _under_gateway_supervisor(argv: list) -> bool:
    """A supervisor-launched gateway child must NOT follow the sticky active_profile.

    Each supervised slot has a fixed profile identity: named slots pass
    ``-p <name>`` or pin HERMES_HOME to the profile dir; a bare invocation
    means "the root HERMES_HOME profile". If a supervised default-profile
    child read active_profile, switching the active profile (dashboard,
    ``hermes profile use``) would silently redirect the default gateway into
    that profile — adopting its credentials and double-polling a Telegram
    token already owned by that profile's own gateway (#74872).

    Markers (see gateway/restart.py ``is_gateway_supervisor_process``):
    HERMES_SUPERVISED_CHILD (systemd unit / launchd plist / Windows task),
    HERMES_S6_SUPERVISED_CHILD (legacy s6 container), INVOCATION_ID (systemd
    service children only — consulted ONLY for gateway commands because it is
    inherited by every descendant of a systemd-launched process, e.g.
    self-hosted CI runners), HERMES_GATEWAY_EXTERNAL_SUPERVISOR (explicit
    opt-in). XPC_SERVICE_NAME is deliberately NOT consulted: interactive macOS
    terminals set it too.
    """
    if os.environ.get("HERMES_SUPERVISED_CHILD") or os.environ.get("HERMES_S6_SUPERVISED_CHILD"):
        return True
    is_gateway_cmd = next((a for a in argv if not a.startswith("-")), None) == "gateway"
    if is_gateway_cmd and os.environ.get("INVOCATION_ID"):
        return True
    return os.environ.get(
        "HERMES_GATEWAY_EXTERNAL_SUPERVISOR", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _s6_supervised_gateway_run(argv: list) -> bool:
    """A bare ``gateway run`` inside the s6 image names the ``gateway-default`` slot too.

    ``_maybe_redirect_run_to_s6_supervision`` turns it into a start of the supervised slot for the
    current profile, and it is the image's own CMD. Following the sticky ``active_profile`` there
    started that profile's named slot on every container boot: the one the boot reconciler just
    registered down, because a started named slot is a second gateway beside the multiplexer.
    ``--no-supervise`` keeps the foreground run, which follows ``active_profile`` as before (#22502).
    """
    words = [a for a in argv if not a.startswith("-")]
    if words[:2] != ["gateway", "run"] or "--no-supervise" in argv:
        return False
    if os.environ.get("HERMES_GATEWAY_NO_SUPERVISE", "").lower() in ("1", "true", "yes"):
        return False
    from hermes_cli.service_manager import _s6_running
    return _s6_running()


def explicit_cli_profile() -> str | None:
    """Profile named by a consumed ``-p``/``--profile`` flag, else None.

    Sticky ``active_profile`` is not explicit. Desktop launch must not overwrite
    its stored profile when the user omitted the flag.
    """
    return _explicit_cli_profile


def _apply_profile_override() -> None:
    """Pre-parse --profile/-p and set HERMES_HOME before imports."""
    global _explicit_cli_profile
    _explicit_cli_profile = None
    argv = sys.argv[1:]
    profile_name, consume, profile_index = _scan_profile_flag(argv)
    from_sticky_profile = False

    # HERMES_HOME already set with no explicit flag: trust it only when it
    # points at a specific profile dir ("profiles" as immediate parent). If it
    # points at the hermes root (systemd hardcodes HERMES_HOME=/root/.hermes)
    # we must still read active_profile — the user may have run
    # `hermes profile use` and the gateway should honour it (#22502).
    hermes_home_env = os.environ.get("HERMES_HOME", "")
    if profile_name is None and hermes_home_env and Path(hermes_home_env).parent.name == "profiles":
        return

    if (profile_name is None and not _under_gateway_supervisor(argv)
            and not _startup_fast.is_desktop_ssh_backend_argv(argv)
            and not _s6_supervised_gateway_run(argv)):
        try:
            from hermes_constants import get_default_hermes_root

            active_path = get_default_hermes_root() / "active_profile"
            if active_path.exists():
                name = active_path.read_text(encoding="utf-8-sig").strip()
                if name and name != "default":
                    profile_name = name  # consume stays 0: nothing to strip
                    from_sticky_profile = True
        except (UnicodeDecodeError, OSError):
            pass  # corrupted file, skip

    if profile_name is None:
        return
    try:
        from hermes_cli.profiles import resolve_profile_env

        hermes_home = resolve_profile_env(profile_name)
    except FileNotFoundError as exc:
        hermes_home = _resolve_sudo_user_profile_env(profile_name)
        error = str(exc)
        if not hermes_home and from_sticky_profile:
            from hermes_cli.main_profile_recovery import is_stale_profile_recovery_command

            if is_stale_profile_recovery_command(argv):
                hermes_home = resolve_profile_env("default")
                print(
                    f"Warning: saved profile '{profile_name}' no longer exists; "
                    "running this recovery command in the default profile.",
                    file=sys.stderr,
                )
            else:
                error = f"Saved profile '{profile_name}' no longer exists. Switch back with: hermes profile use default"
        if not hermes_home:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        # A bug in profiles.py must NEVER prevent hermes from starting
        print(f"Warning: profile override failed ({exc}), using default", file=sys.stderr)
        return
    os.environ["HERMES_HOME"] = hermes_home
    if consume > 0:
        _explicit_cli_profile = profile_name
    # Strip the flag from argv so argparse doesn't choke
    if consume > 0 and profile_index is not None:
        start = profile_index + 1  # +1 because argv is sys.argv[1:]
        sys.argv = sys.argv[:start] + sys.argv[start + consume :]


_apply_profile_override()
# ``-p``/active_profile re-homed the process after hermes_bootstrap ran: re-point the temp vars
# at THIS home's scratch dir (a user-set TMPDIR is still left alone).
try:
    from hermes_constants import export_scratch_tmp_env as _export_scratch_tmp_env

    _export_scratch_tmp_env()
except Exception:
    pass  # an unwritable home leaves the system temp dir in place; never block startup

# PM runs after profile resolution but before application dependency imports.
if sys.argv[1:2] == ["pm"]:
    from pm.cli import main as _pm_main

    raise SystemExit(_pm_main(sys.argv[2:]))

# Windows launcher self-heal — the ``hermes`` command is a COPY of the venv
# console script staged into the managed bin dir (outside the checkout, since
# ``hermes update``'s autostash once swept ``<checkout>\bin`` copies off disk;
# venv\Scripts must stay off PATH as it shadows the user's ``python``).
# Re-staging at process start reaches already-broken installs via the desktop
# app's ``python -m hermes_cli.main`` spawn. Gates fail toward inaction. Sits
# AFTER the profile override on purpose — no hermes module may import before
# profiles resolve; the helper anchors on the DEFAULT root, so profile
# sessions heal the same shared dir.
# That dir lives OUTSIDE the git checkout precisely because an earlier layout staged the copies at
# ``<checkout>\bin``, where ``hermes update``'s autostash (``git stash push --include-untracked``) swept
# them off disk; with the desktop updater's ``--keep-stash`` nothing restored them and ``hermes`` stopped
# resolving in every new terminal (venv\Scripts itself must stay off PATH — it shadows the user's
# ``python``, #83797). Costs a few stat calls when healthy; gates fail toward inaction so source checkouts
# are untouched.
if sys.platform == "win32":
    try:
        from hermes_cli import _install_repair as _install_repair_mod

        _install_repair_mod.ensure_windows_bin_launchers(_bootstrap_root)
    except Exception:
        pass

# Load .env from ~/.hermes/.env first, then project root as dev fallback.
# User-managed env files should override stale shell exports on restart.
from hermes_cli.config import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv

# ``update`` must not resolve external secret sources (Windows self-lock via cryptography, slow
# helpers inside the import probe) — ``_early_recovery._should_skip_external_secret_sources``
# owns that argv check for every dotenv load in the process. See #73381.
load_hermes_dotenv(project_env=PROJECT_ROOT / ".env")

# Bridge security.redact_secrets → HERMES_REDACT_SECRETS BEFORE hermes_logging
# imports agent.redact, which snapshots the flag exactly once at import. A
# .env value still wins — this is config.yaml fallback only. network.force_ipv4
# is read from the same parse to avoid a second full load_config() (~17ms).
_FORCE_IPV4_EARLY = False
try:
    # The effective-config cache (shared raw parse with read_raw_config()) means this SAME parse
    # serves hermes_logging, hermes_time and later raw reads: 3-4 config.yaml parses become one.
    # Managed overlay included: administrator-pinned redact_secrets / force_ipv4 win here too.
    from hermes_cli.config_effective import load_user_config_effective as _load_effective_early

    _cfg_path = get_hermes_home() / "config.yaml"
    if _cfg_path.exists():
        _early_cfg_raw = _load_effective_early(_cfg_path)
        if "HERMES_REDACT_SECRETS" not in os.environ:
            _early_sec_cfg = _early_cfg_raw.get("security", {})
            if isinstance(_early_sec_cfg, dict):
                _early_redact = _early_sec_cfg.get("redact_secrets")
                if _early_redact is not None:
                    os.environ["HERMES_REDACT_SECRETS"] = str(_early_redact).lower()
        _early_net_cfg = _early_cfg_raw.get("network", {})
        if isinstance(_early_net_cfg, dict) and _early_net_cfg.get("force_ipv4"):
            _FORCE_IPV4_EARLY = True
        del _early_cfg_raw
    del _cfg_path
except Exception:
    pass  # best-effort — redaction stays at default (enabled) on config errors

# Centralized file logging for every subcommand (agent.log + errors.log).
# Dashboard entrypoints use GUI mode so gui.log captures pre-dispatch failures.
try:
    from hermes_logging import setup_logging as _setup_logging

    _setup_logging(
        mode=(
            "gui"
            if next((arg for arg in sys.argv[1:] if not arg.startswith("-")), "")
            in {"dashboard", "serve", "gui", "desktop"}
            else "cli"
        )
    )
except Exception:
    pass  # best-effort — don't crash the CLI if logging setup fails

# Apply IPv4 preference before any HTTP client is created.
if _FORCE_IPV4_EARLY:
    try:
        from hermes_constants import apply_ipv4_preference as _apply_ipv4

        _apply_ipv4(force=True)
    except Exception:
        pass  # best-effort — don't crash if hermes_constants not importable yet

import logging
import threading
from datetime import datetime

from hermes_cli.model_setup_flows import (
    _model_flow_openrouter,
    _model_flow_nous,
    _model_flow_openai_codex,
    _model_flow_xai_oauth,
    _model_flow_qwen_oauth,
    _model_flow_minimax_oauth,
    _model_flow_custom,
    _model_flow_azure_foundry,
    _model_flow_named_custom,
    _model_flow_copilot,
    _model_flow_copilot_acp,
    _model_flow_kimi,
    _model_flow_stepfun,
    _model_flow_bedrock,
    _model_flow_vertex,
    _model_flow_api_key_provider,
    _model_flow_anthropic,
    _model_flow_moa,
    _model_flow_ai_gateway,
    _model_flow_plugin_provider,
    _is_profile_plugin_flow_provider,
)
logger = logging.getLogger(__name__)
from hermes_cli.main_agent_cmds import (
    cmd_acp,
    cmd_insights,
    cmd_memory,
    cmd_monitoring,
    cmd_skills,
    cmd_tools,
)
from hermes_cli.main_platform_setup import (
    cmd_slack,
    cmd_sync,
    cmd_whatsapp,
    cmd_whatsapp_cloud,
)
from hermes_cli.process_identity import is_desktop_owned_backend as _is_desktop_owned_backend
from hermes_cli.main_dashboard import (
    _attach_to_host_backend,
    _finalize_update_output,
    _find_stale_dashboard_pids,
    _install_hangup_protection,
    _is_electron_packaged_web_dist,
    _maybe_setup_dashboard_auth_interactively,
    _read_ssh_session_token_file,
    _report_dashboard_status,
    _resolve_dashboard_web_dist,
    _route_named_profile_dashboard,
)
from hermes_cli.main_dashboard import (  # frozen updater surface: update_cmd*.py resolve these via _m()
    _respawn_dashboard_processes,
)
from hermes_cli.main_provider_setup import (
    _GENERIC_API_KEY_PROVIDERS,
    _aux_config_menu,
    _build_provider_picker_rows,
    _clear_stale_openai_base_url,
    _is_profile_api_key_provider,
    _named_custom_provider_map,
    _offer_reasoning_after_pick,
    _prompt_main_reasoning_effort,
    _prompt_provider_choice,
    _remove_custom_provider,
)
# Frozen external updater API: old in-memory siblings still import these names
# after a checkout swap. Keep their inert shims separate from live launch helpers.
from hermes_cli.old_updater_main import (
    ShimQuarantineError,
    _BYTECODE_FINGERPRINT_FILE,
    _desktop_stamp_path,
    _detect_broken_lazy_refresh_imports,
    _expected_windows_pe_machines,
    _hermes_exe_shims,
    _insert_python_pin,
    _interpreter_scripts_dir,
    _load_installable_optional_extras,
    _parse_pe_machine,
    _quarantine_running_hermes_exe,
    _repair_broken_lazy_refresh_imports,
    _resolve_install_target_python,
    _restore_quarantined_exes,
    _run_install_with_heartbeat,
    _run_package_only_install,
    _run_quarantined_install,
    _run_with_idle_timeout,
    _self,
    _verify_console_scripts_installed,
    _verify_core_dependencies_installed,
    _web_ui_build_needed,
    _windows_native_machine,
    _windows_shim_in_process_chain,
    _write_web_ui_build_stamp,
)
from hermes_cli.main_install_repair import _cleanup_quarantined_exes
from hermes_cli.main_install_repair import (  # frozen updater surface: update_cmd*.py resolve these via _m()
    _UPDATE_REEXEC_ENV,
    _clear_lazy_refresh_incomplete_marker,
    _clear_marker_file,
    _clear_update_incomplete_marker,
    _is_termux_env,
    _is_windows,
    _is_windows_npm_path,
    _lazy_refresh_marker_path,
    _pytest_owns_live_checkout,
    _reexec_dependency_sync_off_windows_shim,
    _resolve_node_runtime_npm,
    _resolve_update_branch,
    _update_marker_path,
    _venv_scripts_dir,
)
from hermes_cli.main_desktop import (
    cmd_gui,
)
from hermes_cli.main_desktop import (  # frozen updater surface: update_cmd*.py resolve these via _m()
    _desktop_build_needed,
    _desktop_dist_exists,
    _desktop_macos_relaunchable_fixup,
    _desktop_packaged_executable,
    _install_rebuilt_desktop_app,
    _installed_desktop_apps,
)
from hermes_cli.main_web_build import (
    _sweep_stale_bytecode_if_checkout_changed,
)
from hermes_cli.main_web_build import (  # frozen updater surface: update_cmd*.py resolve these via _m()
    _build_web_ui,
    _nixos_build_env,
    _record_bytecode_fingerprint,
    _run_npm_install_deterministic,
)
from hermes_cli.main_tui_launch import (
    _launch_tui,
    _pin_kanban_board_env,
    _resolve_use_tui,
    _sync_bundled_skills_quietly,
)


def _is_termux_startup_environment(env: dict[str, str] | None = None) -> bool:
    """Import-safe Termux check for cold-start-sensitive CLI paths."""
    check = env or os.environ
    prefix = str(check.get("PREFIX", ""))
    return bool(
        check.get("TERMUX_VERSION")
        or "com.termux/files/usr" in prefix
        or prefix.startswith("/data/data/com.termux/")
    )


def _read_packed_ref(common_dir: Path, ref: str) -> str | None:
    """Look up a ref in .git/packed-refs without spawning git.

    packed-refs lines look like ``<sha> <ref>`` with optional ``^<sha>``
    peel lines and ``#``-prefixed comments / ``# pack-refs with:`` header.
    """
    try:
        text = (common_dir / "packed-refs").read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            return parts[0].strip()
    return None


def _read_git_revision_fingerprint(repo_root: Path) -> str | None:
    """Return a cheap checkout fingerprint without spawning git."""
    git_dir = repo_root / ".git"
    try:
        if git_dir.is_file():
            for line in git_dir.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                key, _, value = line.partition(":")
                if key.strip() == "gitdir" and value.strip():
                    git_dir = (repo_root / value.strip()).resolve()
                    break
        # Worktrees point HEAD at a per-worktree gitdir but pack their refs
        # in the main repo's gitdir (referenced via ``commondir``). Resolve
        # that up front so packed-refs lookups hit the right file.
        common_dir = git_dir
        commondir_file = git_dir / "commondir"
        if commondir_file.exists():
            try:
                rel = commondir_file.read_text(encoding="utf-8-sig", errors="replace").strip()
                if rel:
                    common_dir = (git_dir / rel).resolve()
            except OSError:
                pass
        head = (git_dir / "HEAD").read_text(encoding="utf-8-sig", errors="replace").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            # Loose refs may live in the worktree gitdir OR the common dir
            # (branches created via `git worktree add` typically live in the
            # common dir's refs/heads/).
            for candidate in (git_dir, common_dir):
                ref_file = candidate / ref
                if ref_file.exists():
                    return f"git:{ref}:{ref_file.read_text(encoding='utf-8-sig', errors='replace').strip()}"
            packed_sha = _read_packed_ref(common_dir, ref)
            if packed_sha:
                return f"git:{ref}:{packed_sha}"
            # Ref name is known but unresolved — still stable across launches,
            # and the version/release fallback in the caller will invalidate
            # after `hermes update`.
            return f"git:{ref}:unresolved"
        return f"git:HEAD:{head}"
    except OSError:
        return None


def _termux_bundled_skills_fingerprint() -> str:
    """Cheap invalidation key for Termux bundled-skill startup sync."""
    git_fp = _read_git_revision_fingerprint(PROJECT_ROOT)
    if git_fp:
        return git_fp
    from hermes_cli.version_info import get_version_info

    version_info = get_version_info()
    code_identity = version_info.commit or version_info.derived_version
    skills_dir = PROJECT_ROOT / "skills"
    try:
        stat = skills_dir.stat()
        return f"skills:{code_identity}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return f"skills:{code_identity}:missing"


def _termux_bundled_skills_stamp_path() -> Path:
    return get_hermes_home() / "skills" / ".termux_bundled_sync_stamp"


def _termux_bundled_skills_sync_needed() -> bool:
    if not _is_termux_startup_environment():
        return True
    if os.environ.get("HERMES_TERMUX_FORCE_SKILLS_SYNC") == "1":
        return True
    try:
        stamp = _termux_bundled_skills_stamp_path()
        return stamp.read_text(encoding="utf-8-sig").strip() != _termux_bundled_skills_fingerprint()
    except OSError:
        return True


def _mark_termux_bundled_skills_synced() -> None:
    if not _is_termux_startup_environment():
        return
    try:
        stamp = _termux_bundled_skills_stamp_path()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(_termux_bundled_skills_fingerprint() + "\n", encoding="utf-8")
    except OSError:
        pass


def _sync_bundled_skills_for_startup() -> bool:
    """Sync bundled skills, but skip unchanged Termux checkouts cheaply.

    Hashing every bundled skill is safe but expensive on older Android
    storage. The git/ref stamp keeps post-update correctness: a changed
    checkout revision forces one real sync, then later starts skip it.
    """
    if _is_termux_startup_environment() and not _termux_bundled_skills_sync_needed():
        return False

    from tools.skills_sync import sync_skills

    sync_skills(quiet=True)
    _mark_termux_bundled_skills_synced()
    return True


def _termux_should_prefetch_update_check() -> bool:
    if not _is_termux_startup_environment():
        return True
    return os.environ.get("HERMES_TERMUX_PREFETCH_UPDATES") == "1"


def _dotenv_has_provider_key(env_file: Path, provider_env_vars: set) -> bool:
    """True if ~/.hermes/.env assigns a non-empty value to any provider key."""
    if not env_file.exists():
        return False
    try:
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                # Strip the bash-compatible ``export `` prefix so lines like ``export API_KEY=...`` parse as
                # ``API_KEY`` rather than being stored under the wrong key ``"export API_KEY"`` (#6659).
                line = line[7:]
            key, _, val = line.partition("=")
            if key.strip() in provider_env_vars and val.strip().strip("'\""):
                return True
    except Exception:
        pass
    return False


def _auth_store_logged_in(auth_file: Path, registry, strict_profile_scope: bool) -> bool:
    """True if auth.json's active provider is logged in (api_key providers ignored under strict scope)."""
    from hermes_cli.auth import get_auth_status

    if not auth_file.exists():
        return False
    try:
        auth = json.loads(auth_file.read_text(encoding="utf-8-sig"))
        active = auth.get("active_provider")
        active_config = registry.get(str(active or "").strip().lower())
        if active and not (
            strict_profile_scope and active_config and active_config.auth_type == "api_key"
        ):
            return bool(get_auth_status(active).get("logged_in"))
    except Exception:
        pass
    return False


def _has_any_provider_configured(*, strict_profile_scope: bool = False) -> bool:
    """Check if at least one inference provider is usable. Never creates one: the Nous free tier
    counts only once its identity exists, and the boot bootstrap (``hermes_cli.free_tier_bootstrap``)
    is the only thing that creates it; ``cmd_chat`` runs the bootstrap before asking.

    ``strict_profile_scope``: the caller has bound a NAMED profile's home and
    secret scope and wants an answer for that profile only — launch-process
    env and host-wide fallbacks (gh auth, Claude Code credentials) must not
    make it appear ready. Unscoped callers keep the legacy behavior.
    """
    from hermes_cli.config import DEFAULT_CONFIG, get_env_path, get_hermes_home, load_config
    from hermes_cli.auth import PROVIDER_REGISTRY, get_auth_status

    cfg = load_config()
    model_cfg = cfg.get("model")
    _model_name = model_cfg if isinstance(model_cfg, str) else ""
    if isinstance(model_cfg, dict):
        _model_name = model_cfg.get("default") or ""
        if isinstance(_model_name, dict):
            from hermes_cli.config import split_model_config_default
            _model_name, _ = split_model_config_default(_model_name)
    _model_name = str(_model_name).strip()
    # "Explicitly configured" = model differs from the hardcoded default; gates
    # Claude Code credentials so they don't skip setup on a fresh install.
    _has_hermes_config = _model_name and _model_name != DEFAULT_CONFIG.get("model", "")

    # Env vars (.env or shell). OPENAI_BASE_URL alone counts — local models
    # (vLLM, llama.cpp) often need no API key.
    provider_env_vars = {
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "OPENAI_BASE_URL",
    }
    for pconfig in PROVIDER_REGISTRY.values():
        if pconfig.auth_type == "api_key":
            provider_env_vars.update(pconfig.api_key_env_vars)
    if strict_profile_scope:
        from agent.secret_scope import current_secret_scope

        read_provider_env = (current_secret_scope() or {}).get
    else:
        read_provider_env = os.getenv
    if any(read_provider_env(v) for v in provider_env_vars):
        return True
    if _dotenv_has_provider_key(get_env_path(), provider_env_vars):
        return True

    # Cheap on-disk checks (auth.json, config.yaml) first: the PROVIDER_REGISTRY
    # sweep below spawns subprocesses (gh) and can take 15-20s — long enough
    # that desktop setup.status calls time out.
    if _auth_store_logged_in(get_hermes_home() / "auth.json", PROVIDER_REGISTRY, strict_profile_scope):
        return True

    # model as a dict with provider/base_url/api_key means setup ran (fresh
    # installs have a plain string); also covers custom endpoints kept in config.
    if isinstance(model_cfg, dict) and any(
        (model_cfg.get(k) or "").strip() for k in ("provider", "base_url", "api_key")
    ):
        return True

    # Provider-specific auth fallbacks (e.g. Copilot via gh auth).
    if not strict_profile_scope:
        try:
            if any(
                get_auth_status(pid).get("logged_in")
                for pid, pconfig in PROVIDER_REGISTRY.items()
                if pconfig.auth_type == "api_key"
            ):
                return True
        except Exception:
            pass

    # Claude Code OAuth credentials count only once Hermes is explicitly
    # configured — having Claude Code installed isn't consent to use its tokens.
    if _has_hermes_config and not strict_profile_scope:
        try:
            from agent.anthropic_credentials import read_claude_code_credentials, is_claude_code_token_valid

            creds = read_claude_code_credentials()
            if creds and (
                is_claude_code_token_valid(creds) or creds.get("refreshToken")
            ):
                return True
        except Exception:
            pass

    # Nothing explicit anywhere: an existing Nous free-tier identity counts while the tier is on.
    try:
        from hermes_cli.anon_auth import guest_enabled, has_guest
        return guest_enabled() and has_guest()
    except Exception as exc:
        logger.debug("free tier check on first run skipped: %s", exc)
    return False


def _confirm_startup_expensive_model_override(args) -> None:
    """Guard startup -m/--provider overrides before the first API call."""
    explicit_model = (getattr(args, "model", None) or "").strip()
    explicit_provider = (getattr(args, "provider", None) or "").strip()
    if not explicit_model and not explicit_provider:
        return

    try:
        from hermes_cli.config import load_config
        from hermes_cli.model_selection_guards import (
            combined_message,
            selection_warnings,
        )
    except Exception as exc:
        logger.warning("startup model cost guard unavailable: %s", exc)
        return

    try:
        config = load_config()
    except Exception as exc:
        logger.warning("startup model cost guard could not load config: %s", exc)
        config = {}
    _dict = lambda v: v if isinstance(v, dict) else {}  # noqa: E731
    config = _dict(config)
    model_cfg = _dict(config.get("model"))
    security_cfg = _dict(config.get("security"))

    model = explicit_model or (model_cfg.get("default") or "").strip()
    if not model:
        return
    provider = (explicit_provider or model_cfg.get("provider") or "").strip()
    try:
        # Unified registry: cost guard + id-keyed guards (e.g. the
        # data-training-tier warning) all fire at startup too.
        warnings = selection_warnings(
            model,
            provider=provider,
            base_url=(model_cfg.get("base_url") or ""),
            api_key=(model_cfg.get("api_key") or ""),
        )
    except Exception as exc:
        logger.warning("startup model cost guard failed for %s/%s: %s", provider, model, exc)
        return
    if not warnings:
        return

    # Intentionally independent of --yolo / --accept-hooks: those approve local
    # command risk, not paid aggregator spend or a surprising provider route.
    is_interactive = sys.stdin.isatty()
    if not is_interactive and security_cfg.get("allow_data_training_tiers_noninteractive") is True:
        acknowledged = [w for w in warnings if w.kind == "data_policy"]
        if acknowledged:
            sys.stderr.write(combined_message(acknowledged) + "\n")
            sys.stderr.write(
                "Proceeding in non-interactive mode because "
                "security.allow_data_training_tiers_noninteractive is true.\n"
            )
            warnings = [w for w in warnings if w.kind != "data_policy"]
            if not warnings:
                return

    message = combined_message(warnings)
    if not is_interactive:
        sys.stderr.write(message + "\n")
        if any(warning.kind == "data_policy" for warning in warnings):
            sys.stderr.write(
                "To acknowledge data-training tiers for unattended runs, set "
                "security.allow_data_training_tiers_noninteractive to true "
                "in config.yaml.\n"
            )
        sys.stderr.write(
            "Refusing this startup model override in non-interactive mode. "
            "Run interactively and confirm if you intend to use it.\n"
        )
        raise SystemExit(1)

    sys.stderr.write(message + "\n")
    try:
        reply = input("Use this model for this invocation? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        reply = ""
    if reply not in {"y", "yes"}:
        sys.stderr.write("Model override cancelled.\n")
        raise SystemExit(1)


def _resolve_workspace_key() -> Optional[str]:
    """The current workspace identity for cwd-scoped resume.

    Git repo root when CWD is inside a repo (so all sessions across its
    subdirs/worktrees group together), else the CWD itself. Returns None when
    neither can be determined — callers fall back to the global MRU then.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return os.path.abspath(result.stdout.strip())
    except Exception:
        pass
    try:
        return os.getcwd()
    except Exception:
        return None


@contextlib.contextmanager
def _session_db():
    """Yield a read-only ``SessionDB`` (lazy import, so test patches on ``hermes_state``
    intercept). Every caller is a lookup (last session, title → id, recorded cwd), so it
    never opens a writer beside the one the CLI acquires from the registry a moment later.
    Open failures yield None and any error raised by the ``with`` body is swallowed —
    callers fall through to their ``return None``."""
    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB(read_only=True)
    except Exception:
        pass
    try:
        yield db  # body errors (incl. AttributeError on a None db) are swallowed
    except Exception:
        pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _latest_session_id(use_tui: bool) -> Optional[str]:
    """MRU session for the active interface; a TUI launch falls back to the CLI MRU."""
    last_id = _resolve_last_session(source="tui" if use_tui else "cli")
    if not last_id and use_tui:
        last_id = _resolve_last_session(source="cli")
    return last_id


def _resolve_last_session(source: str = "cli") -> Optional[str]:
    """Look up the most recently-used session ID for a source.

    Scoped to the current workspace first (git repo root, else cwd) so
    ``hermes -c`` from repo A continues repo A's last session rather than the
    global MRU. Falls back to the unscoped MRU when no session matches the
    current workspace, preserving the old behaviour for fresh directories.
    """
    # A finite `hermes -z`/`chat -q` run is CLI history too: `hermes -z … --resume latest` chains on it.
    if source == "cli":
        from run_agent import CLI_FAMILY_SOURCES
        source = sorted(CLI_FAMILY_SOURCES)
    with _session_db() as db:
        ws_key = _resolve_workspace_key()
        if ws_key:
            sessions = db.search_sessions(source=source, limit=1, workspace_key=ws_key)
            if sessions:
                return sessions[0]["id"]
        # Fallback: global MRU for this source.
        sessions = db.search_sessions(source=source, limit=1)
        return sessions[0]["id"] if sessions else None
    return None


def _probe_container(cmd: list, backend: str, via_sudo: bool = False):
    """Run a container inspect probe, returning the CompletedProcess.

    Catches TimeoutExpired specifically for a human-readable message;
    all other exceptions propagate naturally.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
    except subprocess.TimeoutExpired:
        label = f"sudo {backend}" if via_sudo else backend
        print(
            f"Error: timed out waiting for {label} to respond.\n"
            f"The {backend} daemon may be unresponsive or starting up.",
            file=sys.stderr,
        )
        sys.exit(1)


def _exec_in_container(container_info: dict, cli_args: list):
    """Replace the current process with a command inside the managed container.

    Probes whether sudo is needed (rootful containers), then os.execvp
    into the container. On success the Python process is replaced entirely
    and the container's exit code becomes the process exit code (OS semantics).
    On failure, OSError propagates naturally.

    Args:
        container_info: dict with backend, container_name, exec_user, hermes_bin
        cli_args: the original CLI arguments (everything after 'hermes')
    """

    backend = container_info["backend"]
    container_name = container_info["container_name"]
    exec_user = container_info["exec_user"]
    hermes_bin = container_info["hermes_bin"]

    runtime = shutil.which(backend)
    if not runtime:
        print(
            f"Error: {backend} not found on PATH. Cannot route to container.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Rootful containers (NixOS systemd service) are invisible to unprivileged
    # users — Podman uses per-user namespaces, Docker needs group access.
    # Probe whether the runtime can see the container; if not, try via sudo.
    inspect_cmd = [runtime, "inspect", "--format", "ok", container_name]
    cmd_prefix = [runtime]
    if _probe_container(inspect_cmd, backend).returncode != 0:
        sudo_path = shutil.which("sudo")
        if not sudo_path:
            print(
                f"Error: container '{container_name}' not found via {backend}.\n"
                f"The container may be running under root. Try: sudo hermes {' '.join(cli_args)}",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd_prefix = [sudo_path, "-n", runtime]
        if _probe_container(cmd_prefix[:2] + inspect_cmd, backend, via_sudo=True).returncode != 0:
            print(
                f"Error: container '{container_name}' not found via {backend}.\n"
                f"\n"
                f"The container is likely running as root. Your user cannot see it\n"
                f"because {backend} uses per-user namespaces. Grant passwordless\n"
                f"sudo for {backend} — the -n (non-interactive) flag is required\n"
                f"because a password prompt would hang or break piped commands.\n"
                f"\n"
                f"On NixOS:\n"
                f"\n"
                f"  security.sudo.extraRules = [{{\n"
                f'    users = [ "{os.getenv("USER", "your-user")}" ];\n'
                f'    commands = [{{ command = "{runtime}"; options = [ "NOPASSWD" ]; }}];\n'
                f"  }}];\n"
                f"\n"
                f"Or run: sudo hermes {' '.join(cli_args)}",
                file=sys.stderr,
            )
            sys.exit(1)

    env_flags = []
    for var in ("TERM", "COLORTERM", "LANG", "LC_ALL"):
        val = os.environ.get(var)
        if val:
            env_flags.extend(["-e", f"{var}={val}"])

    exec_cmd = (
        cmd_prefix
        + ["exec", "-it" if sys.stdin.isatty() else "-i", "-u", exec_user]
        + env_flags
        + [container_name, hermes_bin]
        + cli_args
    )
    os.execvp(exec_cmd[0], exec_cmd)


def _resolve_session_by_name_or_id(name_or_id: str) -> Optional[str]:
    """Resolve a session title or ID to a session ID (None if neither matches).

    A compression root is followed forward to its latest continuation so an
    old root ID (exit summary, notes) resumes at the live tip.
    """
    with _session_db() as db:
        # Exact session ID first, then title (with auto-latest for lineage).
        session = db.get_session(name_or_id)
        resolved_id = session["id"] if session else db.resolve_session_by_title(name_or_id)
        if resolved_id:
            # Project forward through compression chain so resumes land on
            # the live tip instead of a dead compressed parent.
            try:
                resolved_id = db.get_compression_tip(resolved_id) or resolved_id
            except Exception:
                pass
        return resolved_id
    return None


def _create_titled_session(title: str) -> Optional[str]:
    """Create a fresh titled session (``chat -c <title> --create-if-missing``).

    Same timestamp+uuid id shape the CLI uses; the title is recorded with
    user provenance so auto-titling never overwrites it.

    Used by ``chat -c <title> --create-if-missing`` (#86794): programmatic callers (plugins, scripts) that
    want "send to this named thread, making it if needed" get a deterministic outcome instead of a silent
    no-op.
    """
    db = None
    try:
        from hermes_state_ids import new_session_id as mint_session_id
        from hermes_state_registry import acquire

        new_session_id = mint_session_id()
        # The CLI acquires the registry handle for this same path moments later; share it
        # instead of minting a second writer for one INSERT (close() releases the refcount).
        db = acquire()
        db.create_session(new_session_id, source="cli")
        db.set_session_title(new_session_id, title)
        return new_session_id
    except Exception:
        # Programmatic callers rely on --create-if-missing being deterministic;
        # swallow the failure but log the cause so it lands in errors.log
        # (DB lock, I/O error, import error — all otherwise invisible).
        # See #86794.
        logger.exception("Failed to create titled session %r", title)
        return None
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _resolve_continue_arg(args, *, use_tui: bool) -> None:
    """Resolve ``-c/--continue`` into ``args.resume``.

    ``-c <name>``: resolve by title/ID; on miss fail loudly on **stderr** (exit
    1) so programmatic callers see it even under quiet mode, or with
    ``--create-if-missing`` create a fresh titled session. Bare ``-c``: this
    terminal's breadcrumb session if valid, else the MRU session.

    Handles both forms: See #86794.
    """
    continue_val = getattr(args, "continue_last", None)
    if continue_val and not getattr(args, "resume", None):
        if isinstance(continue_val, str):
            resolved = _resolve_session_by_name_or_id(continue_val)
            if resolved:
                args.resume = resolved
            elif getattr(args, "create_if_missing", False):
                # "send to this named thread, making it if needed" — without it
                # a quiet send to a not-yet-existing session silently no-ops.
                # --create-if-missing: no session matches the title — create a new session with that title
                # and proceed. See #86794.
                new_sid = _create_titled_session(continue_val)
                if new_sid:
                    args.resume = new_sid
                else:
                    print(
                        f"No session found matching '{continue_val}' and "
                        "a new titled session could not be created.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            else:
                print(f"No session found matching '{continue_val}'.", file=sys.stderr)
                print(
                    "Use 'hermes sessions list' to see available sessions, or "
                    "pass --create-if-missing to start a new session with that title.",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            # Bare -c: this terminal's breadcrumb (so side-by-side terminals
            # each continue their own conversation), else the MRU session
            # (also when session.terminal_continue is false).
            if getattr(args, "create_if_missing", False):
                # Nothing to create without a name — surface the no-op.
                print(
                    "--create-if-missing requires a session name: "
                    "`-c <name> --create-if-missing`",
                    file=sys.stderr,
                )
            try:
                from hermes_cli.terminal_breadcrumbs import resolve_breadcrumb_session

                _crumb_id = resolve_breadcrumb_session()
            except Exception:
                _crumb_id = None
            if _crumb_id:
                args.resume = _crumb_id
            else:
                # No valid breadcrumb — continue the most recent session
                last_id = _latest_session_id(use_tui)
                if last_id:
                    args.resume = last_id
                else:
                    kind = "TUI" if use_tui else "CLI"
                    print(
                        f"No previous {kind} session to continue. Start a new one with "
                        "`hermes`, or list sessions with `hermes sessions list`.",
                        file=sys.stderr,
                    )
                    sys.exit(1)


def _apply_in_dir(args) -> None:
    """--in DIR: chdir first so workspace-scoped lookups key off DIR; pins the session there."""
    in_dir = getattr(args, "in_dir", None)
    if not in_dir:
        return
    # Git Bash / MSYS hands us POSIX-style paths (`--in ~` → `/c/Users/x`);
    # translate drive-root spellings to native Windows form. No-op elsewhere.
    from tools.environments.local import _msys_to_windows_path

    _target_dir = os.path.abspath(os.path.expanduser(_msys_to_windows_path(in_dir)))
    if not os.path.isdir(_target_dir):
        print(f"Error: --in directory not found: {in_dir}")
        sys.exit(1)
    try:
        os.chdir(_target_dir)
    except OSError as e:
        print(f"Error: cannot enter --in directory {in_dir}: {e}")
        sys.exit(1)
    # Every cwd consumer (resolve_agent_cwd -> Codex app-server thread cwd, the
    # terminal tool, context-file discovery) prefers TERMINAL_CWD over the process
    # cwd, so a value inherited from a parent surface, the shell or .env outlives
    # this chdir and re-homes the session in the old directory (#106220). Refresh
    # it. An unset variable stays unset: the backends then derive from the new
    # process cwd (local exports it at cli import, docker mounts it, ssh and
    # container backends keep their own remote/sandbox default).
    if os.environ.get("TERMINAL_CWD", "").strip():
        os.environ["TERMINAL_CWD"] = _target_dir
    args.no_restore_cwd = True


def _import_foreign_resume(args) -> None:
    """--resume @claude / @codex: import a foreign session and resume it."""
    _resume_foreign = getattr(args, "resume", None)
    if not (isinstance(_resume_foreign, str) and _resume_foreign.strip().lower() in ("@claude", "@codex")):
        return
    from hermes_cli.foreign_sessions import import_foreign_session, pick_foreign_session

    _picked = pick_foreign_session(_resume_foreign.strip().lower().lstrip("@"))
    if _picked is None:
        sys.exit(1)
    try:
        _imported_id = import_foreign_session(_picked.source, _picked.path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"✓ Imported as {_imported_id} — resuming it now.")
    print(f"  (later: hermes --resume {_imported_id})")
    args.resume = _imported_id


def _resolve_chat_session_args(args, use_tui: bool) -> None:
    """Normalize --in / --resume / --continue on ``args`` before agent init.

    Order matters: ``--in DIR`` chdirs first so workspace-scoped "latest"/-c
    lookups key off DIR (and pins the session there, skipping cwd restore);
    then ``--resume latest`` → MRU id, ``--continue`` → ``--resume``,
    ``--resume @claude/@codex`` → imported session id, title → id; finally
    cd back into a resumed session's recorded cwd (best-effort, opt-out via
    --no-restore-cwd, skipped under --worktree).
    """
    _apply_in_dir(args)

    # --resume latest: same resolution as bare `-c`. The keyword wins over a
    # session literally titled "latest" (still reachable by ID or `-c latest`).
    _resume_raw = getattr(args, "resume", None)
    if isinstance(_resume_raw, str) and _resume_raw.strip().lower() == "latest":
        _last_id = _latest_session_id(use_tui)
        if _last_id:
            args.resume = _last_id
        else:
            kind = "TUI" if use_tui else "CLI"
            print(f"No previous {kind} session found to resume.")
            print("Use 'hermes sessions list' to see available sessions.")
            sys.exit(1)

    _resolve_continue_arg(args, use_tui=use_tui)

    _import_foreign_resume(args)

    resume_val = getattr(args, "resume", None)
    if resume_val:
        # On miss keep the original so _init_agent reports "Session not found" with it.
        args.resume = _resolve_session_by_name_or_id(resume_val) or resume_val

    # cd back into a resumed session's recorded cwd (opt out: --no-restore-cwd;
    # --worktree owns its own dir). A missing dir warns and stays put.
    if (
        getattr(args, "resume", None)
        and not getattr(args, "no_restore_cwd", False)
        and not getattr(args, "worktree", False)
    ):
        with _session_db() as db:  # never let cwd-restore break a resume
            _saved_cwd = ((db.get_session(args.resume) or {}).get("cwd") or "").strip()
            if _saved_cwd and not os.path.isdir(_saved_cwd):
                print(f"⚠ session's recorded dir is gone ({_saved_cwd}); staying in {os.getcwd()}")
            elif _saved_cwd and os.path.realpath(_saved_cwd) != os.path.realpath(os.getcwd()):
                os.chdir(_saved_cwd)
                print(f"↪ restored workspace dir: {_saved_cwd}")


def _warn_retired_xai_models() -> None:
    """One-shot xAI retirement warning on stderr; non-blocking, never fails startup."""
    try:
        from hermes_cli.xai_retirement import (
            MIGRATION_GUIDE_URL,
            RETIREMENT_DATE,
            find_retired_xai_refs,
            format_issue,
        )
        from hermes_cli.config import load_config as _load_config_for_xai_check

        _retired_xai_refs = find_retired_xai_refs(_load_config_for_xai_check())
        if _retired_xai_refs:
            sys.stderr.write(
                f"\033[33m⚠ xAI retires {len(_retired_xai_refs)} model(s) "
                f"in your config on {RETIREMENT_DATE}:\033[0m\n"
            )
            for _ref in _retired_xai_refs:
                sys.stderr.write(f"  \033[33m⚠\033[0m {format_issue(_ref)}\n")
            sys.stderr.write(f"  \033[2mMigration guide: {MIGRATION_GUIDE_URL}\033[0m\n")
            sys.stderr.write("  \033[2mRun 'hermes doctor' for details.\033[0m\n\n")
    except Exception:
        pass


def _start_chat_background_prefetch() -> None:
    """Kick off the update-check/banner prefetch and the bundled-skills sync.

    Update check is opt-in on Termux (it imports rich/prompt_toolkit in the
    foreground and competes for CPU on single-core devices). The skills sync
    is idempotent and hash-gated (~120-170ms of rglob/hashing) so it normally
    runs in a daemon thread — skill loading happens at agent init, long after.
    The ONE exception is an unseeded ~/.hermes/skills: there the banner
    prefetch races the sync and caches an empty index ("No skills installed"
    on the very first launch), so the first run syncs in the foreground and
    drops the banner's skills cache.
    """
    if _termux_should_prefetch_update_check():
        try:
            from hermes_cli.banner import prefetch_banner_data, prefetch_update_check

            prefetch_update_check()
            prefetch_banner_data()  # git banner state + skills index off-thread
        except Exception:
            pass

    def _skills_dir_is_unseeded() -> bool:
        try:
            from hermes_cli.config import get_hermes_home
            skills_dir = Path(get_hermes_home()) / "skills"
            if not skills_dir.is_dir():
                return True
            return next(skills_dir.rglob("SKILL.md"), None) is None
        except Exception:
            return False

    def _skills_sync_bg() -> None:
        try:
            _sync_bundled_skills_for_startup()
        except Exception:
            pass

    if _skills_dir_is_unseeded():
        _skills_sync_bg()
        # Drop the banner's possibly-empty skills cache so it recomputes.
        try:
            import hermes_cli.banner as _banner_mod
            _banner_mod._available_skills_cache = None
        except Exception:
            pass
    else:
        threading.Thread(
            target=_skills_sync_bg, name="bundled-skills-sync", daemon=True
        ).start()


def _first_run_setup_guard(args) -> None:
    """No provider configured: offer `hermes setup` (TTY) or exit 1 with guidance."""
    print()
    print(
        "It looks like Hermes isn't configured yet -- no API keys or providers found."
    )
    print()
    print("  Run:  hermes setup")
    print()

    from hermes_cli.setup import (
        is_interactive_stdin,
        print_noninteractive_setup_guidance,
    )

    if not is_interactive_stdin():
        print_noninteractive_setup_guidance(
            "No interactive TTY detected for the first-run setup prompt."
        )
        sys.exit(1)

    try:
        reply = input("Run setup now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        reply = "n"
    if reply in {"", "y", "yes"}:
        cmd_setup(args)
        return
    print()
    print("You can run 'hermes setup' at any time to configure.")
    sys.exit(1)


def _read_query_file(args) -> None:
    """--query-file: read the single query from a file (or stdin via '-').

    Callers never have to shell-quote message bodies — this is the transport
    the Bot Mode DM protocol uses; interpolating arbitrary text into a
    double-quoted shell argument truncates on quotes and executes $(...)
    (see tools/bot_mode_probe.py).
    """
    _qfile = getattr(args, "query_file", None)
    if not _qfile:
        return
    if args.query:
        # argparse's mutually-exclusive group catches the normal CLI path;
        # this guards programmatic callers that fill the namespace directly.
        print("Error: -q/--query and --query-file are mutually exclusive", file=sys.stderr)
        sys.exit(2)
    try:
        if _qfile == "-":
            args.query = sys.stdin.read()
        else:
            with open(_qfile, "r", encoding="utf-8-sig", errors="replace") as _fh:
                args.query = _fh.read()
    except OSError as _e:
        print(f"Error: cannot read --query-file {_qfile}: {_e}", file=sys.stderr)
        sys.exit(2)
    if not (args.query or "").strip():
        print(f"Error: --query-file {_qfile} is empty", file=sys.stderr)
        sys.exit(2)


# args attr -> (kwarg, default) passed through to _launch_tui / cli.main.
_CHAT_PASSTHROUGH = (
    ("provider", None), ("toolsets", None), ("skills", None), ("verbose", None),
    ("quiet", False), ("query", None), ("image", None), ("resume", None),
    ("worktree", False), ("checkpoints", False), ("pass_session_id", False),
    ("max_turns", None),
)


def cmd_chat(args):
    """Run interactive chat CLI."""
    _apply_safe_mode(args)
    _apply_user_config_bypass(args)
    _guard_noninteractive_user_config(args)
    from hermes_cli.stream_json import stream_json_requested
    # Structured stdout is a non-interactive protocol: it overrides HERMES_TUI/display.interface too.
    use_tui = False if stream_json_requested(args) else _resolve_use_tui(args)

    _resolve_chat_session_args(args, use_tui)

    _warn_retired_xai_models()

    # First-run guard: the free-tier bootstrap runs first (synchronously here; it is the only thing
    # that may create the identity), then the inventory decides whether setup is needed.
    from hermes_cli.free_tier_bootstrap import run_bootstrap

    run_bootstrap(announce=False)
    if not _has_any_provider_configured():
        _first_run_setup_guard(args)
        return

    _start_chat_background_prefetch()

    # --yolo: bypass all dangerous command approvals. main() also sets this
    # before _prepare_agent_startup() — the authoritative site, since it runs
    # before tool imports freeze _YOLO_MODE_FROZEN. This is a safety net for
    # callers that invoke cmd_chat directly (e.g. subcommand dispatch).
    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"
    # --ignore-rules: skip AGENTS.md/SOUL.md/.cursorrules injection, memory
    # entries and preloaded skills (AIAgent(skip_context_files, skip_memory)).
    if getattr(args, "ignore_rules", False):
        os.environ["HERMES_IGNORE_RULES"] = "1"
    # --source: tag session source for filtering (e.g. 'tool' for integrations)
    if getattr(args, "source", None):
        os.environ["HERMES_SESSION_SOURCE"] = args.source
        # Explicit flag, not a label inherited from a parent TUI/Desktop session — one-shot
        # runs must keep it (see run_agent._session_source_for_agent).
        os.environ["HERMES_SESSION_SOURCE_EXPLICIT"] = "1"

    _pin_kanban_board_env()
    from hermes_cli.observability.shared_metrics_consent import offer_consent_before_chat

    offer_consent_before_chat(args)
    _confirm_startup_expensive_model_override(args)

    passthrough = {k: getattr(args, k, d) for k, d in _CHAT_PASSTHROUGH}
    if use_tui:
        _launch_tui(
            passthrough.pop("resume"),
            tui_dev=getattr(args, "tui_dev", False),
            native_mode=getattr(args, "tui_native", False) or None,
            model=getattr(args, "model", None),
            accept_hooks=getattr(args, "accept_hooks", False),
            **passthrough,
        )

    _read_query_file(args)

    safe_mode = getattr(args, "safe_mode", False)
    kwargs = {
        "model": args.model,
        "reasoning": getattr(args, "reasoning", None),
        "toolsets": args.toolsets,
        "query": args.query,
        "oneshot": bool(getattr(args, "oneshot_exit", False)),
        "run_budget": getattr(args, "run_budget", None),
        "output_format": getattr(args, "output_format", "text"),
        "ignore_rules": getattr(args, "ignore_rules", False) or safe_mode,
        "ignore_user_config": getattr(args, "ignore_user_config", False) or safe_mode,
        "compact": getattr(args, "compact", False),
        **{k: getattr(args, k, d) for k, d in _CHAT_PASSTHROUGH},
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        from cli import main as cli_main
        from hermes_cli.observability.shared_metrics_process import begin_process

        begin_process("cli")
        cli_main(**kwargs)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ImportError as e:
        # Mixed-version installs (new cli.py, older hermes_cli.config) crash
        # here — e.g. missing resolve_turn_limit / split_model_config_default
        # (#96900). The agent-setup mixin prints this hint too late: HermesCLI
        # construction already failed. Fast-chat launch also goes through
        # cmd_chat, so this one catch covers `hermes` / `hermes chat`.
        from hermes_constants import emit_partial_update_hint

        if emit_partial_update_hint(e):
            sys.exit(1)
        raise


def cmd_gateway(args):
    """Gateway management commands."""
    _sync_bundled_skills_quietly()

    from hermes_cli.gateway import gateway_command

    gateway_command(args)


def cmd_proxy(args):
    """Local OpenAI-compatible proxy to OAuth providers."""
    # aiohttp is an extras install; keep it off the common path.
    from hermes_cli.proxy.cli import cmd_proxy as _cmd_proxy

    rc = _cmd_proxy(args)
    if isinstance(rc, int) and rc != 0:
        raise SystemExit(rc)


def _forward_command(name: str, module: str, attr: str, *, forward_return: bool = False, doc: str = ""):
    """A ``hermes <cmd>`` handler that hands ``args`` to ``<module>.<attr>``.

    Imports at CALL time so fast paths never pay for it and
    ``patch("<module>.<attr>")`` keeps intercepting. ``forward_return``
    surfaces the return code to ``main()`` (kanban/project/mcp propagate).
    """

    def _cmd(args):
        import importlib

        result = getattr(importlib.import_module(module), attr)(args)
        return result if forward_return else None

    _cmd.__name__ = _cmd.__qualname__ = name
    _cmd.__doc__ = doc or None
    return _cmd


cmd_setup = _forward_command("cmd_setup", "hermes_cli.setup", "run_setup_wizard", doc='Interactive setup wizard.')
cmd_login = _forward_command("cmd_login", "hermes_cli.auth", "login_command", doc='Authenticate Hermes CLI with a provider.')
cmd_logout = _forward_command("cmd_logout", "hermes_cli.auth", "logout_command", doc='Clear provider authentication.')
cmd_auth = _forward_command("cmd_auth", "hermes_cli.auth_commands", "auth_command", doc='Manage pooled credentials.')
cmd_status = _forward_command("cmd_status", "hermes_cli.status", "show_status", doc='Show status of all components.')
cmd_cron = _forward_command("cmd_cron", "hermes_cli.cron", "cron_command", forward_return=True, doc='Cron job management.')
cmd_webhook = _forward_command("cmd_webhook", "hermes_cli.webhook", "webhook_command", doc='Webhook subscription management.')
cmd_kanban = _forward_command("cmd_kanban", "hermes_cli.kanban", "kanban_command", forward_return=True, doc='Multi-profile collaboration board.')
cmd_project = _forward_command("cmd_project", "hermes_cli.projects_cmd", "projects_command", forward_return=True, doc='Manage projects (named, multi-folder workspaces).')
cmd_hooks = _forward_command("cmd_hooks", "hermes_cli.hooks", "hooks_command", doc='Shell-hook inspection and management.')
cmd_doctor = _forward_command("cmd_doctor", "hermes_cli.doctor", "run_doctor", forward_return=True, doc='Check configuration and dependencies.')
cmd_dump = _forward_command("cmd_dump", "hermes_cli.dump", "run_dump", doc='Dump setup summary for support/debugging.')
cmd_debug = _forward_command("cmd_debug", "hermes_cli.debug", "run_debug", doc='Debug tools (share report, etc.).')
cmd_skin = _forward_command("cmd_skin", "hermes_cli.skin_cmd", "skin_command", doc='Skin management (list / use / set).')
cmd_import = _forward_command("cmd_import", "hermes_cli.backup", "run_import", forward_return=True, doc='Restore a Hermes backup from a zip file.')
cmd_dashboard_register = _forward_command("cmd_dashboard_register", "hermes_cli.dashboard_register", "cmd_dashboard_register", doc='Register a self-hosted dashboard OAuth client with Nous Portal.')
cmd_gateway_enroll = _forward_command("cmd_gateway_enroll", "hermes_cli.gateway_enroll", "cmd_gateway_enroll", doc='Enroll a self-hosted gateway with a relay connector.')
cmd_prompt_size = _forward_command("cmd_prompt_size", "hermes_cli.prompt_size", "cmd_prompt_size", doc='Show a byte/char breakdown of the system prompt + tool schemas.')
cmd_pairing = _forward_command("cmd_pairing", "hermes_cli.pairing", "pairing_command")
cmd_plugins = _forward_command("cmd_plugins", "hermes_cli.plugins_cmd", "plugins_command")
cmd_mcp = _forward_command("cmd_mcp", "hermes_cli.mcp_config", "mcp_command", forward_return=True)
cmd_claw = _forward_command("cmd_claw", "hermes_cli.claw", "claw_command")
cmd_import_agent = _forward_command("cmd_import_agent", "hermes_cli.agent_import", "import_agent_command")


def cmd_model(args):
    """Select default model — starts with provider selection, then model picker."""
    _require_tty("model")
    if getattr(args, "refresh", False):
        try:
            from hermes_cli.models import clear_provider_models_cache
            clear_provider_models_cache()
            print("  Cleared model picker cache.")
        except Exception:
            pass
    from hermes_cli.observability.shared_metrics_setup import provider_setup_surface
    from hermes_cli.setup import run_setup_action_with_navigation

    with provider_setup_surface("cli_model"):
        run_setup_action_with_navigation(
            "Model & Provider", lambda: select_provider_and_model(args=args), cancelled_message="No change.",
        )


# Provider id -> flow(config, current_model, args). Lambdas resolve the
# _model_flow_* names at call time so test monkeypatches keep intercepting.
# ``custom:*``, remove-custom and the generic API-key set are the fallthrough
# branches in select_provider_and_model.
_PROVIDER_MODEL_FLOWS = {
    "openrouter": lambda c, m, a: _model_flow_openrouter(c, m),
    "moa": lambda c, m, a: _model_flow_moa(c, m),
    "ai-gateway": lambda c, m, a: _model_flow_ai_gateway(c, m),
    "nous": lambda c, m, a: _model_flow_nous(c, m, args=a),
    "openai-codex": lambda c, m, a: _model_flow_openai_codex(c, m),
    "xai-oauth": lambda c, m, a: _model_flow_xai_oauth(c, m, args=a),
    "qwen-oauth": lambda c, m, a: _model_flow_qwen_oauth(c, m),
    "minimax-oauth": lambda c, m, a: _model_flow_minimax_oauth(c, m, args=a),
    "copilot-acp": lambda c, m, a: _model_flow_copilot_acp(c, m),
    "copilot": lambda c, m, a: _model_flow_copilot(c, m),
    "custom": lambda c, m, a: _model_flow_custom(c),
    "anthropic": lambda c, m, a: _model_flow_anthropic(c, m),
    "kimi-coding": lambda c, m, a: _model_flow_kimi(c, m),
    "stepfun": lambda c, m, a: _model_flow_stepfun(c, m),
    "bedrock": lambda c, m, a: _model_flow_bedrock(c, m),
    "vertex": lambda c, m, a: _model_flow_vertex(c, m),
    "azure-foundry": lambda c, m, a: _model_flow_azure_foundry(c, m),
}


def _norm_base_url(url: str) -> str:
    return str(url or "").strip().rstrip("/").lower()


def _resolve_active_provider(config, model_cfg, effective_provider, custom_provider_map):
    """Provider slug currently in effect (the picker's default row), or None.

    Order: a saved custom provider whose base_url matches model.base_url →
    the configured/env provider (named custom → canonical map key) → auto
    detection. Unknown/unauthenticated providers warn and fall back to auto.
    """
    from hermes_cli.auth import AuthError, format_auth_error, resolve_provider
    from hermes_cli.config import get_compatible_custom_providers, get_env_value
    from hermes_cli.providers import custom_provider_aliases, resolve_provider_full

    active = ""
    if effective_provider == "custom" and isinstance(model_cfg, dict):
        current_base = _norm_base_url(model_cfg.get("base_url", ""))
        if current_base:
            active = next(
                (k for k, info in custom_provider_map.items()
                 if _norm_base_url(info.get("base_url", "")) == current_base),
                "",
            )
    if not active and effective_provider != "auto":
        active_def = resolve_provider_full(
            effective_provider,
            config.get("providers"),
            get_compatible_custom_providers(config),
        )
        if active_def is not None:
            active = active_def.id
            if active_def.source == "user-config":
                requested = str(active or "").strip().lower()
                active = next(
                    (k for k, info in custom_provider_map.items()
                     if requested in custom_provider_aliases(
                         info.get("name", ""), info.get("provider_key", ""))),
                    active,
                )
        else:
            print(
                f"Warning: Unknown provider '{effective_provider}'. Check 'hermes model' for "
                "available providers, or run 'hermes doctor' to diagnose config "
                "issues. Falling back to auto provider detection."
            )
    if not active:
        try:
            active = resolve_provider("auto")
        except AuthError as exc:
            if exc.code == "no_provider_configured":
                # The picker that is about to open IS the fix; a warning that says
                # "run `hermes model`" from inside `hermes model` is circular.
                print("No provider is set up yet — pick one below. (Nous Portal works without an API key.)")
            elif effective_provider == "auto":
                print(f"Warning: {format_auth_error(exc)} Falling back to auto provider detection.")
            active = None  # no provider yet; default to first in list

    # Detect custom endpoint
    if active == "openrouter" and get_env_value("OPENAI_BASE_URL"):
        active = "custom"
    return active


def _pick_provider(config, active, provider_labels, custom_provider_map):
    """Provider picker (+ group member sub-picker) -> concrete slug, or None on cancel."""
    # Group rows drill into a member sub-picker that resolves back to a
    # concrete slug, so the flow dispatch is unchanged.
    ordered, default_idx = _build_provider_picker_rows(
        config, active, provider_labels, custom_provider_map
    )
    provider_idx = _prompt_provider_choice(
        [label for _, label, _ in ordered],
        default=default_idx,
    )
    if provider_idx is None or ordered[provider_idx][0] == "cancel":
        return None
    selected_key, group_label, selected_members = ordered[provider_idx]
    if not selected_members:
        return selected_key
    # Default to the active member when it lives in this group. The group row
    # carries the descriptive text, so member rows show only their short label.
    member_idx = _prompt_provider_choice(
        [provider_labels.get(m, m) for m in selected_members],
        default=selected_members.index(active) if active in selected_members else 0,
        title=f"Select {group_label.split(' ▸', 1)[0]} provider:",
    )
    return None if member_idx is None else selected_members[member_idx]


def select_provider_and_model(args=None):
    """Core provider selection + model picking logic.

    Shared by ``cmd_model`` (``hermes model``) and the setup wizard
    (``setup_model_provider`` in setup.py).  Handles the full flow:
    provider picker, credential prompting, model selection, and config
    persistence.
    """
    from hermes_cli.config import load_config

    config = load_config()
    model_cfg = config.get("model")
    current_model = model_cfg.get("default", "") if isinstance(model_cfg, dict) else model_cfg
    current_model = current_model or "(not set)"

    # Effective provider the same way the CLI resolves it at startup:
    # config.yaml model.provider > env var > auto-detect
    config_provider = model_cfg.get("provider") if isinstance(model_cfg, dict) else None
    effective_provider = config_provider or os.getenv("HERMES_INFERENCE_PROVIDER") or "auto"

    # User-defined custom providers from config.yaml: key → {name, base_url, api_key}
    _custom_provider_map = _named_custom_provider_map(config)
    active = _resolve_active_provider(config, model_cfg, effective_provider, _custom_provider_map)

    from hermes_cli.models import _PROVIDER_LABELS

    provider_labels = dict(_PROVIDER_LABELS)  # derive from canonical list
    if active and active in _custom_provider_map:
        active_label = _custom_provider_map[active]["name"]
    else:
        active_label = provider_labels.get(active, active) if active else "none"

    print()
    print(f"  Current model:    {current_model}")
    print(f"  Active provider:  {active_label}")
    print()

    selected_provider = _pick_provider(config, active, provider_labels, _custom_provider_map)
    if selected_provider is None:
        print("No change.")
        return
    if selected_provider == "aux-config":
        _aux_config_menu()
        return
    if selected_provider == "reasoning":
        # Effort for the CURRENT default model, no model change.
        _prompt_main_reasoning_effort(current_model, active or "")
        return

    # Provider-specific setup + model selection. Flows resolve the
    # _model_flow_* names at call time so test monkeypatches on
    # hermes_cli.main keep intercepting.
    from hermes_cli.observability.shared_metrics_setup import cli_provider_setup
    with cli_provider_setup(selected_provider):
        flow = _PROVIDER_MODEL_FLOWS.get(selected_provider)
        if flow is None and _is_profile_plugin_flow_provider(selected_provider):
            # Registered plugin profile with no bespoke flow: the generic one, keyed by its auth_type.
            flow = lambda c, m, a: _model_flow_plugin_provider(c, selected_provider, m)  # noqa: E731
        if flow is not None:
            flow(config, current_model, args)
        elif (
            selected_provider.startswith("custom:")
            or selected_provider in _custom_provider_map
        ):
            provider_info = _named_custom_provider_map(load_config()).get(selected_provider)
            if provider_info is None:
                print(
                    "Warning: the selected saved custom provider is no longer available. "
                    "It may have been removed from config.yaml. No change."
                )
                return
            _model_flow_named_custom(config, provider_info)
        elif selected_provider == "remove-custom":
            _remove_custom_provider(config)
        elif (
            selected_provider in _GENERIC_API_KEY_PROVIDERS
            or _is_profile_api_key_provider(selected_provider)
        ):
            _model_flow_api_key_provider(config, selected_provider, current_model)

    # Every flow persists through _save_model_choice; a changed model.default means a pick
    # landed, so offer its reasoning effort here once instead of inside each flow.
    _offer_reasoning_after_pick(current_model)

    # Post-switch cleanup: switching to a named provider (anything except
    # "custom") leaves a stale OPENAI_BASE_URL in ~/.hermes/.env that poisons
    # auxiliary clients using provider:auto — clear it proactively. (#5161)
    if selected_provider not in {
        "custom",
        "cancel",
        "remove-custom",
    } and not selected_provider.startswith("custom:"):
        _clear_stale_openai_base_url()


def _detect_venv_python_processes(*, exclude_pids: set[int] | None = None) -> list[tuple[int, str, str]]:
    # Shim to stop the old updater doing work until relaunch. Do not scan or kill.
    # Current Windows checks use the real detector in update_cmd_windows instead.
    return []


# Retired hooks must bypass update_cmd: its current dispatcher captures PM helpers
# before swapping code, but a historical caller may first resolve these afterward.
# Frozen updater surface (PEP 562 ``__getattr__`` below): the frozen
# ``hermes_cli/update_cmd*.py`` files resolve these names via ``_m().<name>``
# on hermes_cli.main; importing update_cmd eagerly would cost every ``hermes``
# invocation ~50-100ms, so they resolve on first read. Nothing else may be
# added here — internal import paths are not a stable API.
_FROZEN_UPDATER_SURFACE: dict[str, tuple[str, ...]] = {
    "hermes_cli.update_cmd": (
        "_assess_parked_branch_switch",
        "_cold_start_windows_gateway_after_update", "_discard_stashed_changes",
        "_filter_non_gateway_concurrent_instances", "_fleet_probe_expected_runtimes",
        "_get_origin_url", "_park_stashed_changes",
        "_pause_windows_gateways_for_update", "_print_parked_branch_kept_notice",
        "_print_parked_branch_skip_warning",
        "_refresh_bootstrap_cache_scripts", "_refresh_windows_gateway_launchers",
        "_restore_stashed_changes",
        "_resume_windows_gateways_after_update", "_run_logged_subprocess", "_run_pre_update_backup",
        "_stash_local_changes_if_needed", "_sync_with_upstream_if_needed",
        "_venv_launcher_ancestors",
        "_wait_for_windows_update_gateway_exit", "_warn_orphaned_update_autostashes",
    ),
    "hermes_cli.old_updater_deps": (
        "_capture_active_lazy_features", "_handoff_reapable_backend_pids",
        "_ledger_manual_serve_holders", "_ledger_reapable_backend_pids",
        "_leftover_pausable_gateway_pids", "_npm_lockfile_changed",
        "_orphaned_desktop_backend_pids", "_refresh_active_lazy_features",
        "_refresh_active_memory_provider_dependencies", "_relaunch_stopped_serves",
        "_stop_process_trees",
    ),
    "hermes_cli.update_cmd_maint": (
        "_purge_stale_hermes_modules", "_reload_updated_runtime_modules",
    ),
    "hermes_cli.dashboard_procs": (
        "_detect_concurrent_hermes_instances", "_kill_stale_dashboard_processes",
    ),
}
_FROZEN_ATTR_SOURCES: dict[str, str] = {
    attr: module for module, attrs in _FROZEN_UPDATER_SURFACE.items() for attr in attrs
}


def __getattr__(name):
    """Resolve the frozen updater surface on first read (see _FROZEN_UPDATER_SURFACE)."""
    if name == "_write_update_incomplete_marker":
        from hermes_cli._old_updater import stop_for_relaunch as value
    else:
        module = _FROZEN_ATTR_SOURCES.get(name)
        if module is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        import importlib

        value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # cache: later accesses skip __getattr__
    return value


def cmd_verify(args):
    """Detect a project's run recipe and smoke-test it."""
    from hermes_cli.verify_cmd import run_verify_command

    sys.exit(run_verify_command(args))


def cmd_security(args):
    """Dispatch `hermes security <subcmd>`."""
    sub = getattr(args, "security_command", None)
    if sub in ("audit", None):
        from hermes_cli.security_audit import cmd_security_audit

        # Default subcommand is `audit` when no subcmd is given.
        code = cmd_security_audit(args)
        sys.exit(int(code or 0))
    print(f"unknown security subcommand: {sub}", file=sys.stderr)
    sys.exit(2)


def cmd_approvals(args):
    """Dispatch `hermes approvals <subcmd>`."""
    from hermes_cli.approvals_suggest import approvals_command

    status = approvals_command(args)
    if status:
        sys.exit(status)
    return status


def cmd_config(args):
    """Configuration management."""
    from hermes_cli.config import config_command

    try:
        config_command(args)
    except RuntimeError as exc:
        # Fail-closed config write guard (require_readable_config_before_write);
        # covers migrate and future write subcommands so none end in a traceback.
        print(f"✗ {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_backup(args):
    """Back up Hermes home directory to a zip file."""
    from hermes_cli import backup

    if getattr(args, "quick", False):
        backup.run_quick_backup(args)
    elif not backup.run_backup(args):
        raise SystemExit(1)  # archive written but incomplete: never shell-success for a timer


def _print_version_info(*, check_updates: bool = True) -> None:
    # Shared with the `hermes --version` pre-import fast path.
    _startup_fast.print_fast_version_info(check_updates=check_updates)


def cmd_version(args):
    """Show version (--version/-V flag)."""
    _print_version_info(check_updates=True)


def cmd_uninstall(args):
    """Uninstall Hermes Agent (or just the Chat GUI with --gui).

    ``--yes`` paths run from the desktop app's non-interactive cleanup scripts,
    so the TTY gate applies only when we actually need to prompt.
    """
    # Machine-readable snapshot for the desktop uninstall UI; before any TTY gate.
    if getattr(args, "gui_summary", False):
        from hermes_cli.gui_uninstall import gui_install_summary

        print(json.dumps(gui_install_summary()))
        return

    if getattr(args, "data", False):
        if not getattr(args, "yes", False) and not getattr(args, "dry_run", False):
            _require_tty("uninstall --data")
        from hermes_cli.uninstall import run_data_uninstall

        run_data_uninstall(args)
        return

    if getattr(args, "gui", False):
        if not getattr(args, "yes", False) and not getattr(args, "dry_run", False):
            _require_tty("uninstall --gui")
        from hermes_cli.uninstall import run_gui_uninstall

        run_gui_uninstall(args)
        return

    if not getattr(args, "yes", False) and not getattr(args, "dry_run", False):
        _require_tty("uninstall")
    from hermes_cli.uninstall import run_uninstall

    run_uninstall(args)


def _clear_bytecode_cache(root: Path) -> int:
    """Remove all __pycache__ dirs under *root* (stale .pyc → ImportError after updates).

    Returns the number of directories removed.
    """
    removed = 0
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {"venv", ".venv", "node_modules", ".git", ".worktrees"}
        ]
        if os.path.basename(dirpath) == "__pycache__":
            try:
                shutil.rmtree(dirpath)
                removed += 1
            except OSError:
                pass
            dirnames.clear()  # nothing left to recurse into
    return removed


def _finalize_update_receipt(code: int, reason: str) -> None:
    """Best-effort receipt close at the command boundary; no-op if already finalized."""
    try:
        # Receipt boundary (#91283 review): the impl has many early sys.exit paths (concurrent-instance
        # preflight, venv-holder refusal, head-pinned no-op, fetch failure) that never reach an inner
        # finalize. Persist any still-open receipt with the real exit code, then let the exit proceed
        # unchanged. No-op when an inner path already finalized (exactly-once by construction).
        from hermes_cli.update_receipt import finalize_pending_update_receipt

        finalize_pending_update_receipt(code, reason)
    except Exception:
        pass


def _update_preflight_handled(args) -> bool:
    """Managed-install refusal, --plan, admission gate, --check. True = nothing more to do."""
    from hermes_cli.config import is_managed, managed_error
    from hermes_cli.update_channel import handle_metadata_args

    if handle_metadata_args(args, PROJECT_ROOT):
        sys.exit(0)
    if is_managed():
        managed_error("update Hermes Agent")
        return True

    # --plan is read-only and deployment-kind aware, so it runs BEFORE the
    # docker/nix/apt refusal gates: on an image/package-managed install the
    # plan itself reports "not updatable in place" plus the right mechanism.
    if getattr(args, "plan", False):
        # Read-only plan phase (#91277 Phase 2): inventory every running Hermes runtime across profiles, its
        # supervisor, and its running code version — without mutating anything. Safe on a live fleet.
        from hermes_cli.update_inventory import (
            collect_runtime_inventory,
            print_update_plan,
        )

        print_update_plan(collect_runtime_inventory())
        return True

    if getattr(args, "list_venv_holders", False):
        # Read-only twin of the Windows venv-holder refusal (#117246): same scan and classifiers,
        # machine-readable, exit 3 when holders remain so automation can stop those PIDs and retry.
        import json

        from hermes_cli.update_cmd_windows import VENV_HOLDERS_EXIT, list_venv_holders

        holders = list_venv_holders()
        print(json.dumps(holders, indent=2))
        if holders:
            sys.exit(VENV_HOLDERS_EXIT)
        return True

    # Image/package-managed admission gate: baked provenance marker first
    # (fail-closed on malformed), then docker/nix/apt heuristics. Records a
    # `refused` receipt and exits 2 (refused-by-contract, distinct from errors).
    # Image-managed / package-managed admission gate (#91277 Phase 3): one shared decision for every
    # mutation surface. Prints the real update command, records a `refused` receipt so fleet tooling sees
    # the blocked attempt, and exits 2 (refused-by-contract, distinct from exit 1 errors).
    # Shared admission gate (#91277 Phase 3): same marker-first decision as the apply path, so --check can
    # never report git state for an install whose real update mechanism is an image pull.
    # The response keeps the pre-existing per-kind error codes the dashboard UI already keys on. See #91277.
    from hermes_cli.update_contract import (
        evaluate_update_admission,
        record_refusal_receipt,
    )

    refusal = evaluate_update_admission(PROJECT_ROOT)
    if refusal is not None:
        print(refusal.message)
        record_refusal_receipt(refusal)
        sys.exit(2)

    if getattr(args, "check", False):
        # --check honors --branch so its answer matches what update would pull.
        branch = _resolve_update_branch(args)
        from hermes_cli.update_cmd import _cmd_update_check

        _cmd_update_check(
            branch=branch,
            branch_explicit=bool(getattr(args, "branch", None)),
            **({"channel": args.channel} if getattr(args, "channel", None) else {}),
        )
        return True
    return False


from hermes_cli.update_receipt import update_receipt_scope


@update_receipt_scope()
def cmd_update(args):
    """Update Hermes Agent: hangup protection + update lock around ``_cmd_update_impl``."""
    # Marks this frame as the CURRENT updater for
    # _old_updater.in_historical_update(); historical on-disk updaters do not
    # declare this local, so only they hand off through retired shims.
    _hermes_current_updater_frame = True
    from hermes_cli.update_owning_install import retarget_to_owning_install

    retarget_to_owning_install(PROJECT_ROOT)
    if _update_preflight_handled(args):
        return
    gateway_mode = getattr(args, "gateway", False)

    _update_io_state = _install_hangup_protection(gateway_mode=gateway_mode)
    # Cross-process mutual exclusion: dashboard Update button, Tauri updater
    # and this command all mutate one checkout; two at once strand it
    # half-updated. Shares the marker the Tauri/Electron updaters already use.
    from hermes_cli.update_lock import (
        UPDATE_EXIT_CONCURRENT,
        UpdateLock,
        describe_holder,
    )

    _update_lock = UpdateLock()
    if not _update_lock.acquire():
        print(describe_holder(_update_lock.holder))
        _finalize_update_output(_update_io_state)
        sys.exit(UPDATE_EXIT_CONCURRENT)


    from hermes_cli.update_cmd import _cmd_update_impl
    from pm import InstallError

    try:
        _cmd_update_impl(args, gateway_mode=gateway_mode)
    except (InstallError, OSError, subprocess.SubprocessError) as exc:
        print(f"✗ Update failed: {exc}")
        _finalize_update_receipt(1, f"{type(exc).__name__}: {exc}")
        if gateway_mode:
            from hermes_cli.update_cmd_fleet import _write_gateway_update_exit_code
            _write_gateway_update_exit_code(False)

        raise SystemExit(1) from exc
    except SystemExit as _update_exit:
        # Receipt boundary: the impl has many early sys.exit paths that never
        # reach an inner finalize. Persist any still-open receipt with the real
        # exit code (no-op if already finalized), then let the exit proceed.
        _code = _update_exit.code if isinstance(_update_exit.code, int) else 1
        _finalize_update_receipt(_code, f"sys.exit({_code})")
        if gateway_mode and _code:
            from hermes_cli.update_cmd_fleet import _write_gateway_update_exit_code
            _write_gateway_update_exit_code(False)

        raise
    except BaseException as _update_exc:
        if gateway_mode:
            from hermes_cli.update_cmd_fleet import _write_gateway_update_exit_code
            _write_gateway_update_exit_code(False)
        _finalize_update_receipt(1, f"{type(_update_exc).__name__}: {_update_exc}")
        raise
    else:
        from hermes_cli.update_receipt import COMMAND_BOUNDARY_STOP_REASON

        _finalize_update_receipt(0, COMMAND_BOUNDARY_STOP_REASON)

    finally:
        _update_lock.release()
        _finalize_update_output(_update_io_state)



def _coalesce_session_name_args(argv: list) -> list:
    """Join unquoted multi-word session names after -c/--continue and -r/--resume.

    ``hermes -c Pokemon Agent Dev`` → ``['-c', 'Pokemon Agent Dev']``; tokens
    are collected until the next flag (``-*``) or known top-level subcommand.
    """
    _SUBCOMMANDS = {
        "chat", "model", "gateway", "setup", "whatsapp", "whatsapp-cloud", "login", "logout",
        "auth", "status", "cron", "doctor", "config", "pairing", "skills", "tools", "mcp",
        "sessions", "insights", "update", "uninstall", "profile", "dashboard", "serve",
        "desktop", "gui", "honcho", "claw", "plugins", "security", "acp", "webhook", "peer",
        "memory", "dump", "debug", "backup", "import", "completion", "logs", "usage",
    }
    _SESSION_FLAGS = {"-c", "--continue", "-r", "--resume"}

    result = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _SESSION_FLAGS:
            result.append(token)
            i += 1
            # Collect subsequent non-flag, non-subcommand tokens as one name
            parts: list = []
            while (
                i < len(argv)
                and not argv[i].startswith("-")
                and argv[i] not in _SUBCOMMANDS
            ):
                parts.append(argv[i])
                i += 1
            if parts:
                result.append(" ".join(parts))
        else:
            result.append(token)
            i += 1
    return result


from hermes_cli.profile_cmd import cmd_profile


def _dashboard_lifecycle_flags(args, token_file) -> None:
    """--status / --stop: report or kill running dashboards and exit (no deps needed)."""
    if token_file and (getattr(args, "status", False) or getattr(args, "stop", False)):
        raise SystemExit("--ssh-session-token-file cannot be used with --status or --stop")
    if getattr(args, "status", False):
        _report_dashboard_status()
        sys.exit(0)  # status is informational, always 0
    if getattr(args, "stop", False):
        # Scoped to the invoking home (`-p` applied by _apply_profile_override): another
        # install's or profile's backend on this machine is never a target (#113978).
        from hermes_constants import get_hermes_home

        own_home = str(get_hermes_home())
        if not _find_stale_dashboard_pids(scope_home=own_home):
            print("No hermes dashboard processes running for this profile.")
            sys.exit(0)
        # Reuse the same SIGTERM-grace-SIGKILL path used after `hermes update`;
        # it prints outcomes itself. Exit 1 only if a pid was unkillable — judged
        # from the kill result, not a re-scan: a launchd KeepAlive job respawns
        # its backend on a fresh PID, which is not a failed stop.
        from hermes_cli.dashboard_procs import _kill_stale_dashboard_processes

        result = _kill_stale_dashboard_processes(reason="requested via --stop", scope_home=own_home)
        sys.exit(1 if result["failed"] else 0)


def _dashboard_validate_serve_args(args, headless_backend, token_file):
    """Headless-serve argument checks -> ssh_owner_nonce (or None)."""
    # `hermes serve` is headless/non-interactive: fail closed on a corrupt
    # config.yaml instead of silently starting on defaults where provider
    # auto-detection can adopt unnamed .env credentials (issue #81952).
    # Same policy + escape hatch as _guard_noninteractive_user_config.
    if headless_backend:
        from hermes_cli.config import (
            InvalidUserConfigError,
            require_parseable_user_config,
        )

        try:
            require_parseable_user_config(
                ignore_user_config=bool(getattr(args, "ignore_user_config", False))
            )
            entries.append((task_key, label))
        entries.append(("__reset__", "Reset all to auto"))
        entries.append(("__back__", "Back"))

        idx = _prompt_provider_choice(
            [label for _, label in entries],
            default=0,
        )
        if idx is None:
            return
        key = entries[idx][0]
        if key == "__back__":
            return
        if key == "__reset__":
            n = _reset_aux_to_auto()
            if n:
                print(f"Reset {n} auxiliary task(s) to auto.")
            else:
                print("All auxiliary tasks were already set to auto.")
            print()
            continue
        # Otherwise configure the specific task
        _aux_select_for_task(key)


def _aux_select_for_task(task: str) -> None:
    """Pick a provider + model for a single auxiliary task and persist it.

    Provider rows come from ``build_aux_picker_rows()`` — the shared aux-picker
    substrate — so this surface shows exactly what every other aux picker
    shows: authenticated built-ins, the user's own ``providers:`` /
    ``custom_providers:`` endpoints, and providers whose credential pool is
    temporarily exhausted. Only already-configured providers appear; users set
    up new ones through the normal ``hermes model`` flow, then route aux tasks
    to them here.
    """
    from hermes_cli.config import load_config
    from hermes_cli.inventory import build_aux_picker_rows, format_aux_picker_entries

    cfg = load_config()
    if task == _DELEGATION_TASK_KEY:
        task_cfg = _delegation_cfg_as_task(cfg)
    else:
        aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
        task_cfg = aux.get(task, {}) if isinstance(aux.get(task), dict) else {}
    current_provider = str(task_cfg.get("provider") or "auto").strip() or "auto"
    current_model = str(task_cfg.get("model") or "").strip()
    current_base_url = str(task_cfg.get("base_url") or "").strip()

    display_name = _aux_task_display_name(task)

    # Gather authenticated providers (has credentials + curated model list)
    try:
        providers = build_aux_picker_rows(
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
        )
    except Exception as exc:
        print(f"Could not detect authenticated providers: {exc}")
        providers = []

    entries: list[tuple[str, str, list[str]]] = []  # (slug, label, models)
    # "auto" always first
    auto_marker = (
        "  ← current" if current_provider == "auto" and not current_base_url else ""
    )
    auto_label = (
        "auto (inherit main agent)"
        if task == _DELEGATION_TASK_KEY
        else "auto (recommended)"
    )
    entries.append(("__auto__", f"{auto_label}{auto_marker}", []))

    entries.extend(
        format_aux_picker_entries(
            providers,
            current_provider=current_provider,
            current_base_url=current_base_url,
        )
    )

    # Custom endpoint (raw base_url)
    custom_marker = "  ← current" if current_base_url else ""
    entries.append(("__custom__", f"Custom endpoint (direct URL){custom_marker}", []))
    entries.append(("__back__", "Back", []))

    print()
    print(f"  Configure {display_name} — current: {_format_aux_current(task_cfg)}")
    print()

    idx = _prompt_provider_choice([label for _, label, _ in entries], default=0)
    if idx is None:
        return
    slug, _label, models = entries[idx]

    if slug == "__back__":
        return

    if slug == "__auto__":
        _save_aux_choice(task, provider="auto", model="", base_url="", api_key="")
        print(f"{display_name}: reset to auto.")
        return

    if slug == "__custom__":
        _aux_flow_custom_endpoint(task, task_cfg)
        return

    # Regular provider — pick a model from its curated list
    _aux_flow_provider_model(task, slug, models, current_model)


def _aux_flow_provider_model(
    task: str,
    provider_slug: str,
    curated_models: list,
    current_model: str = "",
) -> None:
    """Prompt for a model under an already-authenticated provider, save to aux."""
    from hermes_cli.auth import _prompt_model_selection
    from hermes_cli.models import get_pricing_for_provider

    display_name = _aux_task_display_name(task)

    # Fetch live pricing for this provider (non-blocking)
    pricing: dict = {}
    try:
        pricing = get_pricing_for_provider(provider_slug) or {}
    except Exception:
        pricing = {}

    model_list = list(curated_models)

    # Let the user pick a model. _prompt_model_selection supports "Enter custom
    # model name" and cancel.  When there's no curated list (rare), fall back
    # to a raw input prompt.
    if not model_list:
        print(f"No curated model list for {provider_slug}.")
        print("Enter a model slug manually (blank = use provider default):")
        try:
            val = line_input("Model: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        selected = val or ""
    else:
        selected = _prompt_model_selection(
            model_list,
            current_model=current_model,
            pricing=pricing,
            confirm_provider=provider_slug,
        )
        if selected is None:
            print("No change.")
            return

    _save_aux_choice(
        task, provider=provider_slug, model=selected or "", base_url="", api_key=""
    )
    if selected:
        print(f"{display_name}: {provider_slug} · {selected}")
    else:
        print(f"{display_name}: {provider_slug} (provider default model)")


def _aux_flow_custom_endpoint(task: str, task_cfg: dict) -> None:
    """Prompt for a direct OpenAI-compatible base_url + optional api_key/model."""
    from hermes_cli.secret_prompt import masked_secret_prompt

    display_name = _aux_task_display_name(task)
    current_base_url = str(task_cfg.get("base_url") or "").strip()
    current_model = str(task_cfg.get("model") or "").strip()

    print()
    print(f"  Custom endpoint for {display_name}")
    print("  Provide an OpenAI-compatible base URL (e.g. http://localhost:11434/v1)")
    print()
    try:
        url_prompt = (
            f"Base URL [{current_base_url}]: " if current_base_url else "Base URL: "
        )
        url = line_input(url_prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return
    url = url or current_base_url
    if not url:
        print("No URL provided. No change.")
        return
    try:
        model_prompt = (
            f"Model slug (optional) [{current_model}]: "
            if current_model
            else "Model slug (optional): "
        )
        model = line_input(model_prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return
    model = model or current_model
    try:
        api_key = masked_secret_prompt(
            "API key (optional, blank = use OPENAI_API_KEY): "
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return

    _save_aux_choice(
        task,
        provider="custom",
        model=model,
        base_url=url,
        api_key=api_key,
    )
    short_url = url.replace("https://", "").replace("http://", "").rstrip("/")
    print(f"{display_name}: custom ({short_url})" + (f" · {model}" if model else ""))


def _prompt_provider_choice(choices, *, default=0, title="Select provider:"):
    """Show provider selection menu with curses arrow-key navigation.

    Falls back to a numbered list when curses is unavailable (e.g. piped
    stdin, non-TTY environments).  Returns the selected index, or None
    if the user cancels.
    """
    try:
        from hermes_cli.setup import _curses_prompt_choice

        idx = _curses_prompt_choice(title, choices, default)
        if idx >= 0:
            print()
            return idx
    except Exception:
        pass

    # Fallback: numbered list
    print(title)
    for i, c in enumerate(choices, 1):
        marker = "→" if i - 1 == default else " "
        print(f"  {marker} {i}. {c}")
    print()
    while True:
        try:
            val = input(f"Choice [1-{len(choices)}] ({default + 1}): ").strip()
            if not val:
                return default
            idx = int(val) - 1
            if 0 <= idx < len(choices):
                return idx
            print(f"Please enter 1-{len(choices)}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            print()
            return None










_DEFAULT_QWEN_PORTAL_MODELS = [
    "qwen3-coder-plus",
    "qwen3-coder",
]


def _prompt_custom_api_mode_selection(base_url: str, current_api_mode: str = "") -> Optional[str]:
    """Prompt for a custom provider API mode.

    Returns an explicit mode string, or None to keep auto-detect behavior.
    """
    from hermes_cli.runtime_provider import _detect_api_mode_for_url

    detected_mode = _detect_api_mode_for_url(base_url)
    normalized_current = str(current_api_mode or "").strip().lower()
    default_mode = normalized_current or detected_mode or ""

    mode_options = [
        (
            "",
            "Auto-detect",
            "Use Hermes URL heuristics; best for standard OpenAI-compatible endpoints.",
        ),
        (
            "chat_completions",
            "Chat Completions",
            "Use /chat/completions for standard OpenAI-compatible servers.",
        ),
        (
            "codex_responses",
            "Responses / Codex",
            "Use /responses for Codex-compatible tool-calling backends.",
        ),
        (
            "anthropic_messages",
            "Anthropic Messages",
            "Use /v1/messages for Anthropic-compatible endpoints.",
        ),
    ]

    print()
    print("Select API compatibility mode:")
    for idx, (value, label, description) in enumerate(mode_options, 1):
        markers = []
        if value == detected_mode:
            markers.append("detected")
        if value == default_mode:
            markers.append("current")
        suffix = f" [{' / '.join(markers)}]" if markers else ""
        print(f"  {idx}. {label}{suffix}")
        print(f"     {description}")

    try:
        raw = input(
            "Choice [1-4, Enter to keep current/detected]: "
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        raise

    if not raw:
        return default_mode or None

    if raw in {"1", "auto", "detect", "auto-detect"}:
        return None
    if raw in {"2", "chat", "chat_completions", "completions"}:
        return "chat_completions"
    if raw in {"3", "responses", "codex", "codex_responses"}:
        return "codex_responses"
    if raw in {"4", "anthropic", "anthropic_messages", "messages"}:
        return "anthropic_messages"

    print(f"Invalid API mode choice: {raw}. Falling back to auto-detect.")
    return None


def _auto_provider_name(base_url: str) -> str:
    """Generate a display name from a custom endpoint URL.

    Returns a human-friendly label like "Local (localhost:11434)" or
    "RunPod (xyz.runpod.io)".  Used as the default when prompting the
    user for a display name during custom endpoint setup.
    """
    import re

    clean = base_url.replace("https://", "").replace("http://", "").rstrip("/")
    clean = re.sub(r"/v1/?$", "", clean)
    name = clean.split("/")[0]
    if "localhost" in name or "127.0.0.1" in name:
        name = f"Local ({name})"
    elif "runpod" in name.lower():
        name = f"RunPod ({name})"
    else:
        name = name.capitalize()
    return name


def _custom_provider_api_key_config_value(provider_info, resolved_api_key=""):
    """Return the value that should be persisted for a custom provider key."""
    api_key_ref = str(provider_info.get("api_key_ref", "") or "").strip()
    if api_key_ref:
        return api_key_ref

    key_env = str(provider_info.get("key_env", "") or "").strip()
    if key_env and not str(provider_info.get("api_key", "") or "").strip():
        return f"${{{key_env}}}"

    return str(resolved_api_key or "").strip()


def _custom_provider_base_url_config_value(provider_info, resolved_base_url=""):
    """Return the value that should be persisted for a custom provider URL."""
    base_url_ref = str(provider_info.get("base_url_ref", "") or "").strip()
    if base_url_ref:
        return base_url_ref
    return str(resolved_base_url or "").strip()


def _save_custom_provider(
    base_url, api_key="", model="", context_length=None, name=None, api_mode=None,
    key_env=""
):
    """Save a custom endpoint to custom_providers in config.yaml.

    Deduplicates by base_url — if the URL already exists, updates the
    model name, context_length, and api_mode but doesn't add a duplicate entry.
    Uses *name* when provided, otherwise auto-generates from the URL.

    When *key_env* is set the caller has already written the key to ``.env``,
    so the entry references it instead of inlining the secret (#69449).
    """
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    providers = cfg.get("custom_providers") or []
    if not isinstance(providers, list):
        providers = []

    # Check if this URL is already saved — update model/context_length if so
    for entry in providers:
        if isinstance(entry, dict) and entry.get("base_url", "").rstrip(
            "/"
        ) == base_url.rstrip("/"):
            changed = False
            if model and entry.get("model") != model:
                entry["model"] = model
                changed = True
            if model and context_length:
                models_cfg = entry.get("models", {})
                if not isinstance(models_cfg, dict):
                    models_cfg = {}
                models_cfg[model] = {"context_length": context_length}
                entry["models"] = models_cfg
                changed = True
            if api_mode:
                if entry.get("api_mode") != api_mode:
                    entry["api_mode"] = api_mode
                    changed = True
            elif "api_mode" in entry:
                entry.pop("api_mode", None)
                changed = True
            if key_env and (entry.get("key_env") != key_env or entry.get("api_key")):
                entry["key_env"] = key_env
                entry.pop("api_key", None)
                changed = True
            if changed:
                cfg["custom_providers"] = providers
                save_config(cfg)
            return  # already saved, updated if needed

    # Use provided name or auto-generate from URL
    if not name:
        name = _auto_provider_name(base_url)

    entry = {"name": name, "base_url": base_url}
    if key_env:
        entry["key_env"] = key_env
    elif api_key:
        entry["api_key"] = api_key
    if model:
        entry["model"] = model
    if api_mode:
        entry["api_mode"] = api_mode
    if model and context_length:
        entry["models"] = {model: {"context_length": context_length}}

    providers.append(entry)
    cfg["custom_providers"] = providers
    save_config(cfg)
    print(f'  💾 Saved to custom providers as "{name}" (edit in config.yaml)')




def _remove_custom_provider(config):
    """Let the user remove a saved custom provider from config.yaml."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    providers = cfg.get("custom_providers") or []
    if not isinstance(providers, list) or not providers:
        print("No custom providers configured.")
        return

    print("Remove a custom provider:\n")

    choices = []
    for entry in providers:
        if isinstance(entry, dict):
            name = entry.get("name", "unnamed")
            url = entry.get("base_url", "")
            short_url = url.replace("https://", "").replace("http://", "").rstrip("/")
            choices.append(f"{name} ({short_url})")
        else:
            choices.append(str(entry))
    choices.append("Cancel")

    try:
        from hermes_cli.curses_ui import curses_radiolist

        idx = curses_radiolist(
            "Select provider to remove:",
            list(choices),
            selected=0,
            cancel_returns=-1,
        )
        print()
        if idx < 0:
            idx = None
    except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")
        print()
        try:
            val = input(f"Choice [1-{len(choices)}]: ").strip()
            idx = int(val) - 1 if val else None
        except (ValueError, KeyboardInterrupt, EOFError):
            idx = None

    if idx is None or idx >= len(providers):
        print("No change.")
        return

    removed = providers.pop(idx)
    cfg["custom_providers"] = providers
    save_config(cfg)
    removed_name = (
        removed.get("name", "unnamed") if isinstance(removed, dict) else str(removed)
    )
    print(f'✅ Removed "{removed_name}" from custom providers.')




# Lazy-export the model catalog at module level. Tests and a handful of
# downstream call sites read `hermes_cli.main._PROVIDER_MODELS` directly,
# so the symbol needs to be reachable as a module attribute. But importing
# the catalog eagerly costs ~55ms on every `hermes` invocation — including
# fast paths like `hermes --version` and slash-command dispatch that never
# touch the catalog. PEP 562 module-level __getattr__ defers the import
# until first attribute access, so the cost is only paid by callers that
# actually look up the catalog. Termux already defers via the same
# mechanism (its model-selection handlers do their own function-local
# imports), so the explicit termux branch from before is no longer needed.
_LAZY_MODEL_EXPORTS = ("_PROVIDER_MODELS",)


# The main.py decomposition moved the sessions/update/dashboard command
# implementations into their own modules, but main.py still re-exports their
# surface so argparse wiring and test monkeypatches on hermes_cli.main.<name>
# keep resolving unchanged. Importing those modules eagerly costs ~50ms on
# every `hermes` invocation, including fast paths like `hermes --version`
# that never run a subcommand. Resolve the re-exports through the module
# __getattr__ below instead, so each module is only imported when one of its
# names is actually touched. Monkeypatching keeps working: patch.object sets
# a real module attribute, which shadows __getattr__.
_LAZY_COMMAND_EXPORTS = {
    "hermes_cli.sessions_cmd": (
        "cmd_sessions",
    ),
    "hermes_cli.dashboard_procs": (
        "_detect_concurrent_hermes_instances",
        "_filter_dashboard_respawn_candidates",
        "_kill_stale_dashboard_processes",
        "_scan_dashboard_processes",
    ),
    "hermes_cli.update_cmd": (
        "_abort_dependency_sync_if_self_locked",
        "_add_upstream_remote",
        "_apply_pending_fleet_restart_catchup",
        "_atomic_replace_dir",
        "_capture_active_lazy_features",
        "_capture_active_tool_dependencies",
        "_capture_head_sha",
        "_classify_concurrent_instance",
        "_clear_fleet_restart_pending_marker",
        "_assess_parked_branch_switch",
        "_branch_head_label",
        "_branch_head_suffix",
        "_cmd_update_check",
        "_cmd_update_impl",
        "_cold_start_windows_gateway_after_update",
        "_count_commits_between",
        "_dependency_sync_would_rewrite",
        "_detect_self_loaded_native_modules",
        "_detect_venv_python_processes",
        "_desktop_owns_gateway_lifecycle",
        "_defer_update_for_self_lock",
        "_discard_lockfile_churn",
        "_discard_stashed_changes",
        "_park_stashed_changes",
        "_ensure_acp_launcher",
        "_ensure_fhs_path_guard",
        "_ensure_uv_for_termux",
        "_finish_dashboard_update_cleanup",
        "_fleet_probe_expected_runtimes",
        "_fleet_restart_pending_marker_path",
        "_filter_non_gateway_concurrent_instances",
        "_for_each_systemd_gateway_unit",
        "_format_concurrent_instances_message",
        "_format_time_ago",
        "_gateway_service_matches_profile",
        "_gateway_recovery_partition",
        "_gateway_restart_recovery_profiles",
        "_handoff_reapable_backend_pids",
        "_ledger_reapable_backend_pids",
        "_purge_stale_hermes_modules",
        "_format_venv_python_holders_message",
        "_gateway_prompt",
        "_get_origin_url",
        "_has_upstream_remote",
        "_install_psutil_android_compat",
        "_invalidate_update_cache",
        "_is_android_python",
        "_is_fork",
        "_leftover_pausable_gateway_pids",
        "_ledger_manual_serve_holders",
        "_relaunch_stopped_serves",
        "_serve_relaunch_commands",
        "_log_only_write",
        "_mark_skip_upstream_prompt",
        "_npm_bin_exists",
        "_npm_lockfile_changed",
        "_npm_manifest_paths",
        "_npm_manifests_digest",
        "_orphaned_desktop_backend_pids",
        "_pending_fleet_restart_needed",
        "_pause_windows_gateways_for_update",
        "_print_curator_first_run_notice",
        "_print_curator_recent_run_notice",
        "_print_fts_optimize_available_notice",
        "_print_parked_branch_kept_notice",
        "_print_parked_branch_skip_warning",
        "_print_stash_cleanup_guidance",
        "_print_update_completion",
        "_record_npm_lockfile_hash",
        "_refresh_active_lazy_features",
        "_refresh_active_memory_provider_dependencies",
        "_refresh_bootstrap_cache_scripts",
        "_refresh_windows_gateway_launchers",
        "_recover_gateway_restart_after_abort",
        "_reload_updated_runtime_modules",
        "_resolve_pre_update_backup_mode",
        "_resolve_stash_selector",
        "_restart_phase_failure_is_incomplete",
        "_restore_active_tool_dependencies",
        "_restore_stashed_changes",
        "_resume_windows_gateways_after_update",
        "_run_pending_fleet_restart",
        "_run_logged_subprocess",
        "_run_pre_update_backup",
        "_service_unit_supports_graceful_sigusr1_restart",
        "_should_skip_upstream_prompt",
        "_stash_apply_failed_only_on_existing_untracked",
        "_stash_local_changes_if_needed",
        "_stop_process_trees",
        "_surviving_gateway_pids_after_failed_restart",
        "_sync_fork_with_upstream",
        "_sync_with_upstream_if_needed",
        "_update_node_dependencies",
        "_update_via_zip",
        "_upgrade_pip_before_lazy_refresh",
        "_validate_critical_files_syntax",
        "_validate_critical_modules_import",
        "_venv_core_imports_healthy",
        "_venv_launcher_ancestors",
        "_wait_for_windows_update_gateway_exit",
        "_warn_gateway_restart_phase_aborted",
        "_warn_incomplete_gateway_fleet_restart",
        "_warn_pending_fleet_restart_on_startup",
        "_web_build_toolchain_ready",
        "_web_toolchain_roots",
        "_write_fleet_restart_pending_marker",
        "_write_lazy_refresh_incomplete_marker",
        "_write_marker_file",
        "_write_update_incomplete_marker",
        "_write_update_planned_stop_marker",
        "_UPDATE_RUNTIME_RELOAD_MODULES",
        "_UPDATE_CRITICAL_FILES",
        "_UPDATE_CRITICAL_MODULES",
        "OFFICIAL_REPO_URLS",
        "OFFICIAL_REPO_URL",
        "SKIP_UPSTREAM_PROMPT_FILE",
        "_PRE_UPDATE_SNAPSHOT_KEEP",
        "_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE",
    ),
}

_LAZY_COMMAND_ATTR_TO_MODULE = {
    attr: module for module, attrs in _LAZY_COMMAND_EXPORTS.items() for attr in attrs
}

# Back-compat alias: some tests and external callers import the old warn-only
# name. The kill behaviour replaced it; resolve to the new name lazily.
_LAZY_COMMAND_ALIASES = {
    "_warn_stale_dashboard_processes": (
        "hermes_cli.dashboard_procs",
        "_kill_stale_dashboard_processes",
    ),
}


def _self():
    """This module, for attribute access at call time.

    Bare-name global lookups inside this module do not go through the PEP 562
    __getattr__ below, so internal callers of the lazily re-exported names use
    _self().<name> instead. That resolves the lazy re-export on first use and
    keeps monkeypatches on hermes_cli.main.<name> working, exactly like a
    globals lookup did. ``sys`` is imported locally because some tests patch
    this module's ``sys`` attribute.
    """
    import sys as _sys

    return _sys.modules[__name__]


def __getattr__(name):
    """Defer the model-catalog and command-module imports until first read."""
    if name in _LAZY_MODEL_EXPORTS:
        from hermes_cli.models import _PROVIDER_MODELS
        # Cache on the module so subsequent accesses skip the import machinery.
        globals()[name] = _PROVIDER_MODELS
        return _PROVIDER_MODELS
    module = _LAZY_COMMAND_ATTR_TO_MODULE.get(name)
    if module is not None:
        import importlib

        value = getattr(importlib.import_module(module), name)
        globals()[name] = value
        return value
    alias = _LAZY_COMMAND_ALIASES.get(name)
    if alias is not None:
        import importlib

        module_name, attr = alias
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _current_reasoning_effort(config) -> str:
    agent_cfg = config.get("agent")
    if isinstance(agent_cfg, dict):
        return str(agent_cfg.get("reasoning_effort") or "").strip().lower()
    return ""


def _set_reasoning_effort(config, effort: str) -> None:
    agent_cfg = config.get("agent")
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}
        config["agent"] = agent_cfg
    agent_cfg["reasoning_effort"] = effort


def _prompt_reasoning_effort_selection(efforts, current_effort=""):
    """Prompt for a reasoning effort. Returns effort, 'none', or None to keep current."""
    deduped = list(
        dict.fromkeys(
            str(effort).strip().lower() for effort in efforts if str(effort).strip()
        )
    )
    canonical_order = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
    ordered = [effort for effort in canonical_order if effort in deduped]
    ordered.extend(effort for effort in deduped if effort not in canonical_order)
    if not ordered:
        return None

    def _label(effort):
        if effort == current_effort:
            return f"{effort}  ← currently in use"
        return effort

    disable_label = "Disable reasoning"
    skip_label = "Skip (keep current)"

    if current_effort == "none":
        default_idx = len(ordered)
    elif current_effort in ordered:
        default_idx = ordered.index(current_effort)
    elif "medium" in ordered:
        default_idx = ordered.index("medium")
    else:
        default_idx = 0

    try:
        from hermes_cli.curses_ui import curses_radiolist

        choices = [_label(effort) for effort in ordered]
        choices.append(disable_label)
        choices.append(skip_label)
        idx = curses_radiolist(
            "Select reasoning effort:",
            choices,
            selected=default_idx,
            cancel_returns=-1,
        )
        if idx < 0:
            return None
        print()
        if idx < len(ordered):
            return ordered[idx]
        if idx == len(ordered):
            return "none"
        return None
    except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
        pass

    print("Select reasoning effort:")
    for i, effort in enumerate(ordered, 1):
        print(f"  {i}. {_label(effort)}")
    n = len(ordered)
    print(f"  {n + 1}. {disable_label}")
    print(f"  {n + 2}. {skip_label}")
    print()

    while True:
        try:
            choice = input(f"Choice [1-{n + 2}] (default: keep current): ").strip()
            if not choice:
                return None
            idx = int(choice)
            if 1 <= idx <= n:
                return ordered[idx - 1]
            if idx == n + 1:
                return "none"
            if idx == n + 2:
                return None
            print(f"Please enter 1-{n + 2}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            return None






def _prompt_api_key(
    pconfig,
    existing_key: str,
    provider_id: str = "",
    existing_source: str = "",
) -> tuple:
    """Shared API-key entry point for ``hermes setup`` / ``hermes model``.

    Handles both first-time entry and the already-configured case.  When a key
    is already present, offers [K]eep / [R]eplace / [C]lear so the user can
    recover from a malformed paste without editing ``~/.hermes/.env`` by hand.

    Returns ``(resolved_key, abort)``.  ``abort=True`` means the caller should
    ``return`` immediately — the user cancelled entry, declined to replace, or
    cleared the key and is now unconfigured.
    """
    from hermes_cli.auth import LMSTUDIO_NOAUTH_PLACEHOLDER
    from hermes_cli.config import save_env_value
    from hermes_cli.secret_prompt import masked_secret_prompt

    key_env = pconfig.api_key_env_vars[0] if pconfig.api_key_env_vars else ""

    def _prompt_new_key(*, allow_lmstudio_default: bool) -> str:
        if provider_id == "lmstudio" and allow_lmstudio_default:
            prompt = f"{key_env} (Enter for no-auth default {LMSTUDIO_NOAUTH_PLACEHOLDER!r}): "
        else:
            prompt = f"{key_env} (or Enter to cancel): "
        try:
            entered = masked_secret_prompt(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return ""
        if not entered and provider_id == "lmstudio" and allow_lmstudio_default:
            return LMSTUDIO_NOAUTH_PLACEHOLDER
        return entered

    # First-time entry ────────────────────────────────────────────────────
    if not existing_key:
        print(f"No {pconfig.name} API key configured.")
        if not key_env:
            return "", True
        new_key = _prompt_new_key(allow_lmstudio_default=True)
        if not new_key:
            print("Cancelled.")
            return "", True
        save_env_value(key_env, new_key)
        print("API key saved.")
        print()
        return new_key, False

    # Already configured — offer K / R / C ────────────────────────────────
    from hermes_cli.env_loader import format_secret_source_suffix

    source_suffix = format_secret_source_suffix(key_env) if key_env else ""
    print(f"  {pconfig.name} API key: {existing_key[:8]}... ✓{source_suffix}")
    if not key_env:
        # Nothing we can rewrite; just acknowledge and move on.
        print()
        return existing_key, False
    pool_backed = existing_source.startswith("credential_pool:")
    menu = (
        "  [K]eep / [R]eplace (default K): "
        if pool_backed
        else "  [K]eep / [R]eplace / [C]lear (default K): "
    )
    try:
        choice = input(menu).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        choice = "k"

    if choice.startswith("r"):
        new_key = _prompt_new_key(allow_lmstudio_default=False)
        if not new_key:
            print("  No change.")
            print()
            return existing_key, False
        save_env_value(key_env, new_key)
        print("  API key updated.")
        print()
        return new_key, False

    if choice.startswith("c") and not pool_backed:
        save_env_value(key_env, "")
        print(
            f"  API key cleared.  Re-run `hermes setup` to configure {pconfig.name} again."
        )
        return "", True

    # Keep (default, or any other input)
    print()
    return existing_key, False




def _infer_stepfun_region(base_url: str) -> str:
    """Infer the current StepFun region from the configured endpoint."""
    normalized = (base_url or "").strip().lower()
    if "api.stepfun.com" in normalized:
        return "china"
    return "international"


def _stepfun_base_url_for_region(region: str) -> str:
    from hermes_cli.auth import (
        STEPFUN_STEP_PLAN_CN_BASE_URL,
        STEPFUN_STEP_PLAN_INTL_BASE_URL,
    )

    return (
        STEPFUN_STEP_PLAN_CN_BASE_URL
        if region == "china"
        else STEPFUN_STEP_PLAN_INTL_BASE_URL
    )










def _run_anthropic_oauth_flow(save_env_value):
    """Run the Claude OAuth setup-token flow. Returns True if credentials were saved."""
    from agent.anthropic_adapter import (
        run_oauth_setup_token,
        read_claude_code_credentials,
        is_claude_code_token_valid,
    )
    from hermes_cli.config import (
        save_anthropic_oauth_token,
        use_anthropic_claude_code_credentials,
    )

    def _activate_claude_code_credentials_if_available() -> bool:
        try:
            creds = read_claude_code_credentials()
        except Exception:
            creds = None
        if creds and (
            is_claude_code_token_valid(creds) or bool(creds.get("refreshToken"))
        ):
            use_anthropic_claude_code_credentials(save_fn=save_env_value)
            print("  ✓ Claude Code credentials linked.")
            from hermes_constants import display_hermes_home as _dhh_fn

            print(
                f"    Hermes will use Claude's credential store directly instead of copying a setup-token into {_dhh_fn()}/.env."
            )
            return True
        return False

    try:
        print()
        print("  Running 'claude setup-token' — follow the prompts below.")
        print("  A browser window will open for you to authorize access.")
        print()
        token = run_oauth_setup_token()
        if token:
            if _activate_claude_code_credentials_if_available():
                return True
            save_anthropic_oauth_token(token, save_fn=save_env_value)
            print("  ✓ OAuth credentials saved.")
            return True

        # Subprocess completed but no token auto-detected — ask user to paste
        print()
        print("  If the setup-token was displayed above, paste it here:")
        print()
        from hermes_cli.secret_prompt import masked_secret_prompt

        try:
            manual_token = masked_secret_prompt(
                "  Paste setup-token (or Enter to cancel): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return False
        if manual_token:
            save_anthropic_oauth_token(manual_token, save_fn=save_env_value)
            print("  ✓ Setup-token saved.")
            return True

        print("  ⚠ Could not detect saved credentials.")
        return False

    except FileNotFoundError:
        # Claude CLI not installed — guide user through manual setup
        print()
        print("  The 'claude' CLI is required for OAuth login.")
        print()
        print("  To install and authenticate:")
        print()
        print("    1. Install Claude Code:  npm install -g @anthropic-ai/claude-code")
        print("    2. Run:                  claude setup-token")
        print("    3. Follow the browser prompts to authorize")
        print("    4. Re-run:               hermes model")
        print()
        print("  Or paste an existing setup-token now (sk-ant-oat-...):")
        print()
        from hermes_cli.secret_prompt import masked_secret_prompt

        try:
            token = masked_secret_prompt("  Setup-token (or Enter to cancel): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return False
        if token:
            save_anthropic_oauth_token(token, save_fn=save_env_value)
            print("  ✓ Setup-token saved.")
            return True
        print("  Cancelled — install Claude Code and try again.")
        return False




def cmd_login(args):
    """Authenticate Hermes CLI with a provider."""
    from hermes_cli.auth import login_command

    login_command(args)


def cmd_logout(args):
    """Clear provider authentication."""
    from hermes_cli.auth import logout_command

    logout_command(args)


def cmd_auth(args):
    """Manage pooled credentials."""
    from hermes_cli.auth_commands import auth_command

    auth_command(args)


def cmd_status(args):
    """Show status of all components."""
    from hermes_cli.status import show_status

    show_status(args)


def cmd_cron(args):
    """Cron job management."""
    from hermes_cli.cron import cron_command

    cron_command(args)


def cmd_sync(args):
    """Skill Sync — personal sync across devices, plus sharing with your org."""
    import json as _json

    sub = getattr(args, "sync_command", None)

    if sub in {None, ""}:
        print(
            "usage: hermes sync "
            "<status|pull|push|now|enable|disable|device|propose>\n"
            "\n"
            "Your skills, across your devices:\n"
            "  status            Show what is synced, and from where\n"
            "  pull              Pull your synced skills\n"
            "  push              Push your opted-in skills\n"
            "  now               Reconcile now: pull then push\n"
            "  enable <skill>    Include a skill in your sync\n"
            "  disable <skill>   Exclude a skill from your sync\n"
            "  device [--name N] Show or set this device's label\n"
            "\n"
            "Shared with your team:\n"
            "  propose <skill>   Share a skill with your organisation",
            file=sys.stderr,
        )
        return 1

    if sub == "device":
        from tools import skills_sync_client as ssc

        name = getattr(args, "device_name", None)
        if name is not None:
            try:
                stored = ssc.set_device_name(name)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            print(f"device label set to '{stored}'.")
            print(
                "New commits from this device will use this label; existing "
                "commits keep their previous one.",
                file=sys.stderr,
            )
            return 0
        # No --name: print the current (creating a default on first use).
        print(ssc.stable_device_id())
        return 0

    if sub == "propose":
        from tools import skills_sync_client as ssc

        name = args.name
        try:
            result = ssc.propose_skill(name, message=args.message)
        except ssc.SyncInertError as e:
            print(f"cannot share this skill: {e}", file=sys.stderr)
            return 1
        except ssc.SyncError as e:
            print(f"could not share '{name}': {e}", file=sys.stderr)
            return 1
        if result.get("proposal_pending"):
            print(
                f"Shared '{name}' with your organisation — an admin needs to "
                f"approve it (proposal #{result.get('proposal_id')}). It is "
                f"not live for the team until then."
            )
        else:
            print(f"Added '{name}' to your organisation's shared skills.")
        return 0

    if sub in {"enable", "disable"}:
        from tools.skill_usage import set_sync, is_curation_eligible

        skill = args.skill
        if not is_curation_eligible(skill):
            print(
                f"'{skill}' is not sync-eligible (bundled, hub-installed, "
                f"external, or not found). Only agent-created / user-authored "
                f"skills under ~/.hermes/skills/ can sync.",
                file=sys.stderr,
            )
            return 1
        set_sync(skill, sub == "enable")
        print(f"sync {'enabled' if sub == 'enable' else 'disabled'} for '{skill}'.")
        return 0

    from tools import skills_sync_client as ssc

    if sub == "status":
        status = ssc.sync_status()
        print(_json.dumps(status, indent=2, ensure_ascii=False))
        if status.get("org_available"):
            n = len(status.get("org_skills") or [])
            modified = status.get("org_skills_modified") or []
            print(
                f"\nOrg skills: {n} shared skill(s) from your organisation "
                f"(your role: {status.get('org_role')}). They load alongside "
                f"your own, labeled by origin, and you can edit them.",
                file=sys.stderr,
            )
            if modified:
                print(
                    f"  {len(modified)} with local edits not yet shared: "
                    f"{', '.join(modified)}\n"
                    f"  Share them back with `hermes sync propose <skill>`. "
                    f"Org updates will not overwrite them.",
                    file=sys.stderr,
                )
        elif status.get("logged_in"):
            print(
                "\nOrg skills: not applicable — this account isn't a member "
                "of a shared organisation.",
                file=sys.stderr,
            )
        if not status.get("logged_in"):
            print("\nNot logged into Nous Portal — sync is inert.", file=sys.stderr)
        elif not status.get("nous_admin"):
            print(
                "\nSync is not enabled for your account yet.",
                file=sys.stderr,
            )
        elif not status.get("feature_enabled"):
            print(
                "\nSync feature is off for this instance (set HERMES_SYNC_ENABLED=1 "
                "or config.yaml sync.enabled: true). Sync is inert.",
                file=sys.stderr,
            )
        elif not status.get("base_url"):
            print(
                "\nNo sync base URL configured (config.yaml sync.base_url or "
                "HERMES_SYNC_BASE_URL). Sync is inert.",
                file=sys.stderr,
            )
        return 0

    # pull / push / now — enforce the gate up front with a clear message.
    try:
        identity = ssc.resolve_identity()
    except ssc.SyncInertError as e:
        print(f"sync inert: {e}", file=sys.stderr)
        return 1
    if not identity.get("nous_admin"):
        print(
            "sync unavailable: not enabled for your account yet.",
            file=sys.stderr,
        )
        return 1
    if not ssc.resolve_sync_base_url():
        print(
            "sync inert: no sync base URL configured (config.yaml sync.base_url "
            "or HERMES_SYNC_BASE_URL).",
            file=sys.stderr,
        )
        return 1

    try:
        if sub == "pull":
            result = ssc.pull_skills(identity=identity)
            # Refresh the org mirror too when this account belongs to an
            # organisation (no-op otherwise), so one pull covers both.
            org_result = ssc.maybe_pull_org_skills()
            if org_result:
                n = len(org_result.get("updated") or [])
                print(
                    f"org: refreshed {n} shared skill(s) from your "
                    f"organisation.",
                    file=sys.stderr,
                )
                clashes = org_result.get("conflicted") or []
                if clashes:
                    print(
                        f"org: {len(clashes)} skill(s) have BOTH local edits "
                        f"and org updates, so they were left as-is: "
                        f"{', '.join(clashes)}\n"
                        f"     Your local version is intact. Review it, then "
                        f"either propose it or delete the local copy and pull "
                        f"again to take the org version.",
                        file=sys.stderr,
                    )
        elif sub == "push":
            result = ssc.push_skills(identity=identity, message="hermes sync push")
        elif sub == "now":
            pull_res = ssc.pull_skills(identity=identity)
            push_res = ssc.push_skills(identity=identity, message="hermes sync now")
            result = {"pull": pull_res, "push": push_res}
        else:
            print(f"Unknown sync subcommand: {sub}", file=sys.stderr)
            return 1
    except ssc.SyncError as e:
        print(f"sync failed: {e}", file=sys.stderr)
        return 1

    print(_json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_webhook(args):
    """Webhook subscription management."""
    from hermes_cli.webhook import webhook_command

    webhook_command(args)


def cmd_slack(args):
    """Slack integration helpers.

    Dispatches ``hermes slack <subcommand>``. Currently supports:
      manifest — print or write a Slack app manifest with every gateway
                 command registered as a first-class slash.
    """
    sub = getattr(args, "slack_command", None)
    if sub in {None, ""}:
        # No subcommand — print usage hint.
        print(
            "usage: hermes slack <subcommand>\n"
            "\n"
            "subcommands:\n"
            "  manifest   Generate a Slack app manifest with every gateway\n"
            "             command registered as a native slash\n"
            "\n"
            "Run `hermes slack manifest -h` for details.",
            file=sys.stderr,
        )
        return 1

    if sub == "manifest":
        from hermes_cli.slack_cli import slack_manifest_command

        status = slack_manifest_command(args)
        if status:
            raise SystemExit(status)
        return status

    print(f"Unknown slack subcommand: {sub}", file=sys.stderr)
    return 1


def cmd_kanban(args):
    """Multi-profile collaboration board."""
    from hermes_cli.kanban import kanban_command

    return kanban_command(args)


def cmd_project(args):
    """Manage projects (named, multi-folder workspaces)."""
    from hermes_cli.projects_cmd import projects_command

    return projects_command(args)


def cmd_hooks(args):
    """Shell-hook inspection and management."""
    from hermes_cli.hooks import hooks_command

    hooks_command(args)


def cmd_doctor(args):
    """Check configuration and dependencies."""
    from hermes_cli.doctor import run_doctor

    run_doctor(args)


def cmd_verify(args):
    """Detect a project's run recipe and smoke-test it."""
    from hermes_cli.verify_cmd import run_verify_command

    sys.exit(run_verify_command(args))


def cmd_security(args):
    """Dispatch `hermes security <subcmd>`."""
    sub = getattr(args, "security_command", None)
    if sub in ("audit", None):
        from hermes_cli.security_audit import cmd_security_audit

        # Default subcommand is `audit` when no subcmd is given.
        code = cmd_security_audit(args)
        sys.exit(int(code or 0))
    print(f"unknown security subcommand: {sub}", file=sys.stderr)
    sys.exit(2)


def cmd_approvals(args):
    """Dispatch `hermes approvals <subcmd>`."""
    from hermes_cli.approvals_suggest import approvals_command

    status = approvals_command(args)
    if status:
        sys.exit(status)
    return status


def cmd_dump(args):
    """Dump setup summary for support/debugging."""
    from hermes_cli.dump import run_dump

    run_dump(args)


def cmd_debug(args):
    """Debug tools (share report, etc.)."""
    from hermes_cli.debug import run_debug

    run_debug(args)


def cmd_config(args):
    """Configuration management."""
    from hermes_cli.config import config_command

    try:
        config_command(args)
    except RuntimeError as exc:
        # Safety net for the fail-closed config write guard (unparseable /
        # non-mapping / unreadable config.yaml raises RuntimeError from
        # require_readable_config_before_write). set/unset already surface
        # this per-branch; this covers migrate and future write subcommands
        # so no path ends in a raw traceback.
        print(f"✗ {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_skin(args):
    """Skin management (list / use / set)."""
    from hermes_cli.skin_cmd import skin_command

    skin_command(args)


def cmd_backup(args):
    """Back up Hermes home directory to a zip file."""
    if getattr(args, "quick", False):
        from hermes_cli.backup import run_quick_backup

        run_quick_backup(args)
    else:
        from hermes_cli.backup import run_backup

        run_backup(args)


def cmd_import(args):
    """Restore a Hermes backup from a zip file."""
    from hermes_cli.backup import run_import

    run_import(args)


def _print_version_info(*, check_updates: bool = True) -> None:
    # Single source of truth for version output — shared with the
    # `hermes --version` pre-import fast path (the `version` subcommand
    # was consolidated into `--version`).
    _startup_fast.print_fast_version_info(check_updates=check_updates)


def cmd_version(args):
    """Show version (--version/-V flag)."""
    _print_version_info(check_updates=True)


def cmd_uninstall(args):
    """Uninstall Hermes Agent (or just the Chat GUI with --gui)."""
    # Machine-readable install snapshot for the desktop app's uninstall UI.
    # Must run before any TTY gate — it's called from a non-interactive child.
    if getattr(args, "gui_summary", False):
        from hermes_cli.gui_uninstall import gui_install_summary

        print(json.dumps(gui_install_summary()))
        return

    # GUI-only uninstall. The desktop app shells out to this non-interactively
    # with --yes, so only gate on a TTY when we actually need to prompt.
    if getattr(args, "gui", False):
        if not getattr(args, "yes", False):
            _require_tty("uninstall --gui")
        from hermes_cli.uninstall import run_gui_uninstall

        run_gui_uninstall(args)
        return

    # Full/keep-data uninstall. ``--yes`` runs non-interactively (the desktop
    # app's lite/full modes drive this from a detached cleanup script), so only
    # gate on a TTY when we actually need to prompt for the option + confirm.
    if not getattr(args, "yes", False):
        _require_tty("uninstall")
    from hermes_cli.uninstall import run_uninstall

    run_uninstall(args)


def _clear_bytecode_cache(root: Path) -> int:
    """Remove all __pycache__ directories under *root*.

    Stale .pyc files can cause ImportError after code updates when Python
    loads a cached bytecode file that references names that no longer exist
    (or don't yet exist) in the updated source.  Clearing them forces Python
    to recompile from the .py source on next import.

    Returns the number of directories removed.
    """
    removed = 0
    for dirpath, dirnames, _ in os.walk(root):
        # Skip venv / node_modules / .git entirely
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {"venv", ".venv", "node_modules", ".git", ".worktrees"}
        ]
        if os.path.basename(dirpath) == "__pycache__":
            try:
                shutil.rmtree(dirpath)
                removed += 1
            except OSError:
                pass
            dirnames.clear()  # nothing left to recurse into
    return removed


# Update pipeline lives in hermes_cli/update_cmd.py (main.py decomposition,
# mechanical move). Its names are re-exported lazily through the module-level
# __getattr__ above (see _LAZY_COMMAND_EXPORTS) so argparse wiring and test
# monkeypatches on hermes_cli.main.<name> keep resolving unchanged without
# paying the update_cmd import cost on every CLI invocation.

# Stamp file recording the checkout fingerprint the bytecode cache was last
# validated against. Lives next to the checkout (NOT in HERMES_HOME) because
# __pycache__ is per-checkout state shared by every profile.
_BYTECODE_FINGERPRINT_FILE = ".bytecode-fingerprint"


def _record_bytecode_fingerprint() -> None:
    """Persist the current checkout fingerprint after a bytecode sweep.

    Never raises. A failed write just means the next launch re-sweeps —
    safe, merely redundant.
    """
    try:
        fingerprint = _read_git_revision_fingerprint(PROJECT_ROOT)
        if not fingerprint:
            return
        stamp_path = PROJECT_ROOT / _BYTECODE_FINGERPRINT_FILE
        tmp_path = stamp_path.with_name(stamp_path.name + ".tmp")
        tmp_path.write_text(fingerprint, encoding="utf-8")
        tmp_path.replace(stamp_path)
    except OSError as exc:
        logger.debug("Could not record bytecode fingerprint: %s", exc)


def _sweep_stale_bytecode_if_checkout_changed() -> None:
    """Clear ``__pycache__`` at launch when the checkout changed underneath us.

    The stale-bytecode bug class (issues #6207, #60242; Dhruv's WhatsApp
    ``cannot import name 'parse_model_flags_detailed'`` report) has one
    shared shape: the checkout's ``.py`` files change (git pull inside
    ``hermes update``, a manual ``git pull``, a ZIP update, a file-sync
    restore) while ``__pycache__`` retains bytecode from the previous
    revision, and a later process trusts the stale ``.pyc`` instead of the
    fresh source.

    Update-time clears alone can never close this class: ``hermes update``
    always executes the PRE-pull updater code, so any hardening added to it
    only takes effect one update late, and manual ``git pull`` never runs
    the updater at all. This launch-time guard closes the loop: every
    ``hermes`` entry point compares the checkout fingerprint (cheap file
    reads, no git subprocess) against the last-validated stamp and sweeps
    the bytecode cache once when they diverge.

    Never raises — a failure here must not block launch.
    """
    try:
        fingerprint = _read_git_revision_fingerprint(PROJECT_ROOT)
        if not fingerprint:
            return  # non-git install — the ZIP update path clears explicitly
        stamp_path = PROJECT_ROOT / _BYTECODE_FINGERPRINT_FILE
        try:
            recorded = stamp_path.read_text(encoding="utf-8").strip()
        except OSError:
            recorded = ""
        if recorded == fingerprint:
            return
        removed = _clear_bytecode_cache(PROJECT_ROOT)
        if removed:
            logger.info(
                "Checkout changed since last launch (%s -> %s): cleared %d stale __pycache__ director%s",
                recorded or "unknown",
                fingerprint,
                removed,
                "y" if removed == 1 else "ies",
            )
        _record_bytecode_fingerprint()
    except Exception as exc:
        logger.debug("Stale-bytecode launch sweep failed: %s", exc)


def _web_ui_build_needed(web_dir: Path) -> bool:
    """Return True if the web UI dist is missing or its source content changed.

    Uses a SHA-256 content hash of the web source tree (the same approach
    ``_desktop_build_needed()`` already uses for the Electron build), NOT
    mtime comparison. ``git checkout`` / ``git pull`` / ``hermes update``
    rewrite source mtimes without changing content, which made the old
    mtime check unreliable in both directions: it could skip a rebuild when
    source had genuinely changed (serving a stale dashboard) and force a
    rebuild when nothing had. A content hash is stable across mtime churn.

    The dashboard source lives under ``web/`` but Vite outputs to
    ``hermes_cli/web_dist/`` (per vite.config.ts outDir), NOT ``web/dist/``,
    so the dist directory is never part of the hashed source tree.
    """
    project_root = web_dir.parent.parent if web_dir.parent.name == "apps" else web_dir.parent
    dist_dir = project_root / "hermes_cli" / "web_dist"
    sentinel = dist_dir / ".vite" / "manifest.json"
    if not sentinel.exists():
        sentinel = dist_dir / "index.html"
    if not sentinel.exists():
        return True
    stamp_file = _web_ui_stamp_path()
    if not stamp_file.is_file():
        return True
    try:
        stamp_data = json.loads(stamp_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if not isinstance(stamp_data, dict):
        return True
    saved_hash = stamp_data.get("contentHash")
    if not saved_hash:
        return True
    return _compute_web_ui_content_hash(project_root, web_dir) != saved_hash


def _compute_web_ui_content_hash(project_root: Path, web_dir: Path) -> str:
    """Return a SHA-256 hex digest of the web UI source tree.

    Covers ``web_dir`` (the dashboard frontend source) plus the root
    ``package.json`` / ``package-lock.json`` (workspace config that
    determines dependency resolution). Mirrors
    ``_compute_desktop_content_hash()``: ignored paths (``node_modules/``,
    ``dist/``, ``*.pyc``, ...) are skipped via the repo-root ``.gitignore``
    so build output never feeds back into its own staleness check.
    """
    h = hashlib.sha256()

    def _hash_file(path: Path) -> None:
        rel = str(path.relative_to(project_root))
        h.update(rel.encode())
        h.update(b"\0")
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except OSError:
            pass
        h.update(b"\0")

    from pathspec import PathSpec

    gitignore = project_root / ".gitignore"
    lines: list[str] = []
    if gitignore.is_file():
        lines = gitignore.read_text(encoding="utf-8").splitlines()
    spec = PathSpec.from_lines("gitignore", lines)

    # Root workspace config (single package-lock.json covers all workspaces).
    for name in ("package.json", "package-lock.json"):
        p = project_root / name
        if p.is_file():
            rel = str(p.relative_to(project_root))
            if not spec.match_file(rel):
                _hash_file(p)

    # Walk the web source tree, pruning ignored directories in-place so we
    # never descend into node_modules/ or a stray dist/. Sort filenames for
    # a deterministic, order-independent digest.
    for dirpath, dirnames, filenames in os.walk(web_dir, topdown=True):
        dirnames[:] = [
            d for d in dirnames
            if not spec.match_file(str((Path(dirpath) / d).relative_to(project_root)))
        ]
        for fn in sorted(filenames):
            fp = Path(dirpath) / fn
            rel = str(fp.relative_to(project_root))
            if not spec.match_file(rel):
                _hash_file(fp)

    return h.hexdigest()


def _web_ui_stamp_path() -> Path:
    """Return the path to the web UI build stamp file under $HERMES_HOME."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "web-ui-build-stamp.json"


def _write_web_ui_build_stamp(project_root: Path, web_dir: Path) -> None:
    """Write the web UI build stamp after a successful build."""
    stamp_file = _web_ui_stamp_path()
    try:
        stamp_file.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        stamp_data = {
            "contentHash": _compute_web_ui_content_hash(project_root, web_dir),
            "builtAt": datetime.now(timezone.utc).isoformat(),
        }
        stamp_file.write_text(json.dumps(stamp_data, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        # Never let stamp-writing block or fail a build.
        logger.debug("Failed to write web UI build stamp: %s", exc)


def _run_with_idle_timeout(
    cmd: list[str],
    cwd: Path,
    *,
    idle_timeout_seconds: int = 180,
    indent: str = "    ",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess that streams output, with an idle-output timeout.

    Issue #33788: ``npm run build`` (Vite) was invoked with
    ``capture_output=True`` and no timeout. On low-memory hosts (notably
    WSL2 with the default 4 GB cap) the build can stall or sit silent for
    minutes; users see a frozen terminal, assume the update is hung, and
    reboot — leaving the editable install in a half-state with the
    ``hermes`` launcher present but ``hermes_cli`` not importable.

    This helper fixes both halves: stdout is streamed (so the user sees
    progress), and if no bytes have appeared on stdout/stderr for
    ``idle_timeout_seconds``, the process is terminated and the call
    returns with a non-zero ``returncode``. The caller's existing
    stale-dist fallback (#23817) takes over from there.

    Returns a ``CompletedProcess`` with merged stdout (text), empty
    stderr, and an integer returncode. Never raises on idle timeout —
    propagation of failure is via the returncode.
    """
    merged_chunks: list[str] = []
    last_output_ts = _time.monotonic()
    lock = threading.Lock()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
    except OSError as exc:
        # E.g. npm not on PATH between the which() check and now.
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))

    def _reader() -> None:
        nonlocal last_output_ts
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                print(f"{indent}{line.rstrip()}", flush=True)
            except UnicodeEncodeError:
                # Windows cp1252 fallback — same pattern as _say().
                enc = getattr(sys.stdout, "encoding", None) or "ascii"
                safe = line.rstrip().encode(enc, errors="replace").decode(enc, errors="replace")
                print(f"{indent}{safe}", flush=True)
            with lock:
                merged_chunks.append(line)
                last_output_ts = _time.monotonic()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    idle_killed = False
    while True:
        try:
            rc = proc.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            with lock:
                idle = _time.monotonic() - last_output_ts
            if idle > idle_timeout_seconds:
                idle_killed = True
                proc.terminate()
                try:
                    rc = proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()
                break

    # Drain reader so we don't leak the stdout file descriptor.
    reader_thread.join(timeout=2)

    combined = "".join(merged_chunks)
    if idle_killed:
        msg = (
            f"\n  ⚠ Build produced no output for {idle_timeout_seconds}s — terminated.\n"
            "    Common causes: out-of-memory on a low-RAM host (WSL/container),\n"
            "    a stuck Node process, or an antivirus scan stalling I/O.\n"
        )
        combined += msg
        # Force a non-zero rc even if terminate() raced with a clean exit.
        if rc == 0:
            rc = 124  # GNU `timeout` convention
    return subprocess.CompletedProcess(cmd, rc, stdout=combined, stderr="")


def _nixos_build_env() -> dict[str, str] | None:
    """Return extra env vars for native module builds on NixOS.

    On NixOS, python3 is typically not on the system PATH (it lives in
    the Nix store and only enters PATH inside a nix-shell or when
    explicitly installed as a system package).  node-gyp uses Python to
    compile native addons like ``node-pty`` and its ``find-python.js``
    does a bare ``PATH`` lookup — which fails on NixOS.

    Two-tier resolution:
    1. Fast path — the hermes venv's python3 (present in managed installs)
    2. Fallback — resolves the absolute python3 path via ``nix-shell``

    Returns an env dict suitable for ``subprocess.run(env=...)`` or
    ``None`` when we are not on NixOS or python3 is already on PATH.
    """
    import re

    try:
        os_release = Path("/etc/os-release").read_text(encoding="utf-8")
    except OSError:
        return None
    if not re.search(r"^ID=nixos$", os_release, re.M):
        return None

    # python3 already on PATH — nothing to do
    if shutil.which("python3"):
        return None

    # Tier 1: fast path — hermes venv python3, no nix-shell overhead
    for venv_name in ("venv", ".venv"):
        venv_python = PROJECT_ROOT / venv_name / "bin" / "python3"
        if venv_python.exists():
            return {**os.environ, "PYTHON": str(venv_python)}

    # Tier 2: nix-shell fallback — resolves the absolute python3 path once.
    # Slower (~2–5 s for the nix-shell eval) but always works, even without
    # a hermes venv (pip / non-managed / bare-git installs).  The resolved
    # path is a self-contained Nix store binary (all deps via RPATH) so it
    # stays valid even after the nix-shell exits.
    try:
        result = subprocess.run(
            ["nix-shell", "-p", "python3", "--run", "which python3"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=15,
        )
        if result.returncode == 0:
            python3_path = result.stdout.strip()
            if python3_path and Path(python3_path).exists():
                return {**os.environ, "PYTHON": python3_path}
    except Exception:
        pass  # nix-shell not available — caller will get None

    return None
def _run_npm_install_deterministic(
    npm: str,
    cwd: Path,
    *,
    extra_args: tuple[str, ...] = (),
    capture_output: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a deterministic npm install that does not mutate ``package-lock.json``.

    Prefers ``npm ci`` (strict, lockfile-preserving) when a lockfile is present;
    falls back to ``npm install`` only if ``npm ci`` fails (e.g. lockfile out of
    sync on a WIP checkout).  Without this, ``npm install`` on npm ≥ 10 silently
    rewrites committed lockfiles (stripping ``"peer": true`` etc.), which leaves
    the working tree dirty and causes the next ``hermes update`` to stash the
    lockfile — repeatedly.

    ``--include=dev`` is forced on every invocation: the callers are frontend
    builds (web UI / TUI / desktop workspaces), and those builds need the dev
    toolchain (``tsc``, ``vite``, ``electron-builder`` — all
    ``devDependencies``).  If the caller's environment has
    ``NODE_ENV=production`` (or npm config ``omit=dev``) — which leaks in from
    a shell profile, a container image, or the bundled TUI launcher that sets
    ``NODE_ENV=production`` on its subprocess env — npm silently omits
    devDependencies (exit 0, no error), so the build toolchain never installs
    and the subsequent build dies with ``tsc: command not found`` (exit 127).
    The flag overrides both the env var and npm config, unlike scrubbing
    ``NODE_ENV`` from the environment which only fixes the env-leak case.

    ``--no-save`` on the ``npm install`` fallback keeps it true to this
    function's contract: never mutate ``package-lock.json``.  Without it, an
    out-of-sync lockfile gets rewritten by the fallback, which drifts the
    committed lockfile and makes every future ``npm ci`` fail — a
    self-reinforcing cycle where web devDeps never install and a stale dist
    is served on every update (PR #65595).
    """
    # unicode-animations' postinstall animates to /dev/tty (bypasses
    # --silent/capture_output). It no-ops when CI is set — same as the TUI
    # install path and nix/lib.nix npm ci hooks.
    run_env = _npm_lifecycle_env(env)

    def _run(cmd: list[str]) -> subprocess.CompletedProcess:
        return _run_npm_watching_for_engine_failure(
            cmd,
            cwd=cwd,
            env=run_env,
            capture_output=capture_output,
        )

    def _attempt(npm_exe: str) -> subprocess.CompletedProcess:
        lockfile = cwd / "package-lock.json"
        if lockfile.exists():
            ci_result = _run([npm_exe, "ci", "--include=dev", *extra_args])
            if ci_result.returncode == 0:
                return ci_result
            # Fall through to `npm install` — lockfile may be out of sync on a
            # WIP fork/branch, or `npm ci` may not be available on very old npm.
        return _run([npm_exe, "install", "--no-save", "--include=dev", *extra_args])

    result = _attempt(npm)
    if result.returncode == 0:
        return result

    # An npm outside the root package.json's `engines.npm` range fails every
    # command here identically (the `npm install` fallback included), so the
    # failure is worth exactly one repair attempt. `maybe_repair_npm_engine`
    # returns the npm to retry with — the same one after an in-place upgrade
    # of a Hermes-managed install, or a freshly provisioned managed npm when
    # the failing npm belongs to the user's own toolchain.
    from hermes_cli.npm_engine import maybe_repair_npm_engine

    combined = f"{result.stdout or ''}\n{result.stderr or ''}"
    repaired_npm = maybe_repair_npm_engine(npm, combined)
    if not repaired_npm:
        return result
    # The repaired npm may be a freshly provisioned managed one whose shebang
    # and lifecycle scripts resolve `node` from PATH — put the managed tree
    # first so they find the managed Node, not the mismatched system one.
    from hermes_constants import with_hermes_node_path

    run_env["PATH"] = with_hermes_node_path(run_env)["PATH"]
    return _attempt(repaired_npm)


def _run_npm_watching_for_engine_failure(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    capture_output: bool,
) -> subprocess.CompletedProcess:
    """Run *cmd*, always retaining stderr so ``EBADENGINE`` stays detectable.

    ``capture_output=False`` callers stream npm's progress live and would
    otherwise hand back a ``CompletedProcess`` with ``stderr=None``, leaving the
    engine-failure recovery nothing to read. Tee stderr instead: each line is
    forwarded to this process's stderr as it arrives (so live output is
    unchanged) and accumulated for the caller.
    """
    if capture_output:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    captured: list[str] = []
    with subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        if proc.stderr is not None:
            for line in proc.stderr:
                captured.append(line)
                sys.stderr.write(line)
            sys.stderr.flush()
        returncode = proc.wait()
    return subprocess.CompletedProcess(cmd, returncode, None, "".join(captured))


def _missing_web_build_tool(output: str) -> str | None:
    """Return the build tool a failed ``npm run build`` could not resolve.

    Each shell words this differently: ``sh: 1: tsc: not found`` (dash),
    ``vite: command not found`` (bash/zsh), and ``'tsc' is not recognized as
    an internal or external command`` (cmd.exe).
    """
    lowered = output.lower()
    for tool in ("tsc", "vite"):
        if any(
            phrase in lowered
            for phrase in (
                f"{tool}: not found",
                f"{tool}: command not found",
                f"'{tool}' is not recognized",
            )
        ):
            return tool
    return None


def _build_web_ui(web_dir: Path, *, fatal: bool = False) -> bool:
    """Build the web UI frontend if npm is available, serializing across processes.

    Concurrent dashboard boots (e.g. the desktop app's retry loop after a
    readiness timeout) used to each spawn their own ``npm install`` +
    ``vite build`` over the same tree; the parallel builds starved each
    other, none finished, the dist sentinel never advanced, and every new
    boot re-triggered the build. One process builds under an exclusive
    flock; the rest serve the existing dist (stale is acceptable) or, when
    no dist exists yet, block until the builder finishes.

    Staleness is checked once, inside :func:`_do_build_web_ui`, after the
    lock is held — so a process that queued behind the builder skips the
    rebuild, and the (os.walk-based) check runs at most once per boot.
    """
    if not (web_dir / "package.json").exists():
        return True
    try:
        import fcntl
    except ImportError:
        # Windows: no flock — fall through to the unserialized build.
        return _do_build_web_ui(web_dir, fatal=fatal)
    project_root = web_dir.parent.parent if web_dir.parent.name == "apps" else web_dir.parent
    dist_index = project_root / "hermes_cli" / "web_dist" / "index.html"
    try:
        lock_file = open(project_root / ".web_ui_build.lock", "a", encoding="utf-8")
    except OSError:
        return _do_build_web_ui(web_dir, fatal=fatal)
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if dist_index.exists():
                # Another process is already building — serve the current
                # dist instead of piling a second build onto the same tree.
                return True
            # No dist at all (first-ever build): wait for the builder.
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return _do_build_web_ui(web_dir, fatal=fatal)
    finally:
        lock_file.close()


def _do_build_web_ui(web_dir: Path, *, fatal: bool = False) -> bool:
    """Build the web UI frontend if npm is available.

    Args:
        web_dir: Path to the dashboard frontend source directory.
        fatal: If True, print error guidance and return False on failure
               instead of a soft warning (used by ``hermes web``).

    Returns True if the build succeeded or was skipped (no package.json).
    """
    if not (web_dir / "package.json").exists():
        return True

    if not _web_ui_build_needed(web_dir):
        return True

    # Console-encoding-safe print: Windows consoles default to cp1252
    # (or similar) and will raise UnicodeEncodeError on arrow / check
    # glyphs unless PYTHONIOENCODING=utf-8 is set. Routing every print
    # in this function through _say() with errors="replace" keeps the
    # build path usable on a stock `py -m hermes_cli.main web` invocation.
    def _say(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))

    from hermes_constants import with_hermes_node_path

    npm = _resolve_node_runtime_npm()
    if not npm:
        if fatal:
            _say("Web UI frontend not built and npm is not available.")
            _say("Install Node.js, then run:  cd web && npm install && npm run build")
        return not fatal
    build_env = _npm_lifecycle_env(with_hermes_node_path())
    _say("→ Building web UI...")

    def _relay(result: "subprocess.CompletedProcess") -> None:
        """Print captured npm output so users can see *why* a step failed.

        Windows users hitting `rm -rf` / `cp -r` errors (or any other
        sync-assets / Vite failure) would otherwise see only ``Web UI
        build failed`` with no hint of the underlying cause, because
        the npm calls run with ``capture_output=True``.
        """
        for blob in (result.stdout, result.stderr):
            if not blob:
                continue
            text = blob.decode("utf-8", errors="replace").rstrip() if isinstance(blob, bytes) else blob.rstrip()
            if text:
                _say(text)

    npm_cwd = _workspace_root(web_dir)
    # Scope the install to the web workspace only so that the full workspace
    # graph (including apps/desktop with its Electron + node-pty deps) is never
    # resolved here.  Without --workspace the root package.json's apps/* glob
    # would pull in desktop on every web build. See #38772.
    # When web/ has its own package-lock.json, _workspace_root() returns
    # web_dir itself and --workspace would fail.  See #42973.
    #
    # When running from the workspace root, this must name the SAME closure
    # as `hermes update`'s _update_node_dependencies() (ui-tui + web +
    # --include-workspace-root): the helper prefers `npm ci`, which deletes
    # node_modules before reifying the requested tree, so a narrower closure
    # here silently prunes everything the update step just installed (root
    # devDependencies and the ui-tui workspace) while still exiting 0 —
    # and since the manifests digest was already recorded, later no-op
    # updates skip the repair. See #43564/#64354.
    npm_workspace_args: tuple[str, ...]
    if npm_cwd == web_dir:
        npm_workspace_args = ()
    else:
        npm_workspace_args = ("--workspace", "web", "--include-workspace-root")
        # Prebuilt/partial checkouts can lack the ui-tui workspace; naming a
        # missing workspace makes npm fail hard, so only include it when
        # present (same guard as _update_node_dependencies()).
        if (npm_cwd / "ui-tui" / "package.json").exists():
            npm_workspace_args = ("--workspace", "ui-tui", *npm_workspace_args)
    if _is_termux_startup_environment():
        npm_cwd, npm_workspace_args = _termux_workspace_install_context(web_dir)

    def _install_web_deps(*, silent: bool) -> "subprocess.CompletedProcess":
        return _run_npm_install_deterministic(
            npm,
            npm_cwd,
            extra_args=(*npm_workspace_args, "--silent", "--prefer-offline") if silent else (*npm_workspace_args, "--prefer-offline"),
            env=build_env,
        )

    r1 = _install_web_deps(silent=True)
    if r1.returncode != 0:
        _say(
            f"  {'✗' if fatal else '⚠'} Web UI npm install failed"
            + ("" if fatal else " (hermes web will not be available)")
        )
        _relay(r1)
        if fatal:
            _say("  Run manually:  npm install --workspace web && npm run build -w web")
        return False
    # First attempt — stream output via idle-timeout helper (issue #33788).
    # capture_output=True on a long Vite build looks identical to a hang;
    # users react by rebooting, which leaves the editable install in a
    # half-state. Streaming + idle-kill makes failures observable AND
    # recoverable (the stale-dist fallback below handles the kill path).
    r2 = _run_with_idle_timeout([npm, "run", "build"], cwd=web_dir, env=build_env)
    if r2.returncode != 0:
        # The install above can exit 0 while leaving the tree without a build
        # toolchain — a lockfile-hash skip over a half-installed tree, or an
        # interrupted link step. The generic retry below just reruns the same
        # command, so `tsc: not found` survives it and the stale dist is
        # served forever. Reinstall (non-silent, so the user sees it) first.
        missing_tool = _missing_web_build_tool((r2.stdout or "") + (r2.stderr or ""))
        if missing_tool:
            _say(f"  ⚠ Build could not resolve {missing_tool} — reinstalling web dependencies...")
            _install_web_deps(silent=False)
            r2 = _run_with_idle_timeout([npm, "run", "build"], cwd=web_dir, env=build_env)
        if r2.returncode != 0:
            # Retry once after a short delay — covers boot-time races on Windows
            # (antivirus scanning Node.js binaries, npm cache not ready, transient
            # I/O when launched via Scheduled Task at logon). See issue #23817.
            _time.sleep(3)
            r2 = _run_with_idle_timeout([npm, "run", "build"], cwd=web_dir, env=build_env)

    if r2.returncode != 0:
        # _run_with_idle_timeout merges stderr into stdout; older callers
        # using subprocess.run kept them split. Pull from whichever has
        # content so the error surfaces regardless of which path produced
        # the CompletedProcess.
        build_output = (r2.stderr or "") + (r2.stdout or "")
        stderr_preview = build_output.strip()
        stderr_tail = "\n  ".join(stderr_preview.splitlines()[-10:]) if stderr_preview else ""
        project_root = web_dir.parent.parent if web_dir.parent.name == "apps" else web_dir.parent
        dist_dir = project_root / "hermes_cli" / "web_dist"
        dist_index = dist_dir / "index.html"

        # If a stale dist exists, serve it as a fallback instead of failing.
        # A stale UI is far better than no UI for non-interactive callers
        # (Windows Scheduled Tasks, CI) — issue #23817.
        if dist_index.exists():
            _say("  ⚠ Web UI build failed — serving stale dist as fallback")
            if stderr_tail:
                _say(f"  Build error:\n  {stderr_tail}")
            return True

        _say(
            f"  {'✗' if fatal else '⚠'} Web UI build failed"
            + ("" if fatal else " (hermes web will not be available)")
        )
        _relay(r2)
        if fatal:
            _say("  Run manually:  npm install --workspace web && npm run build -w web")
        return False
    _say("  ✓ Web UI built")
    project_root = web_dir.parent.parent if web_dir.parent.name == "apps" else web_dir.parent
    _write_web_ui_build_stamp(project_root, web_dir)
    return True


def _desktop_dist_exists(desktop_dir: Path) -> bool:
    """Return True when a local desktop renderer build is present."""
    return (desktop_dir / "dist" / "index.html").exists()


# ---------------------------------------------------------------------------
# Desktop build stamp — content-hash based skip logic
# ---------------------------------------------------------------------------
# The desktop Electron build is expensive.
# Unlike the web UI (which uses mtime comparison), the desktop uses a
# SHA-256 content hash of the source tree so that:
#   - ``git checkout`` / ``git pull`` that touch mtimes but not content
#     don't trigger a rebuild
#   - ``hermes update`` can unconditionally call ``hermes desktop --build-only``
#     and it will skip if nothing actually changed
#   - ``hermes desktop`` (interactive launch) skips the build when the
#     stamp matches, making repeated launches fast
#
# Stamp file: $HERMES_HOME/desktop-build-stamp.json
# Schema:
#   {
#     "contentHash": "<sha256 hex of source files>",
#     "sourceMode": true | false,
#     "builtAt": "<ISO 8601>"
#   }

def _compute_desktop_content_hash(project_root: Path) -> str:
    """Return a SHA-256 hex digest of all source files that feed the desktop build.

    Covers ``apps/desktop/`` (excluding anything matched by .gitignore)
    plus the root ``package.json`` / ``package-lock.json`` (workspace config
    that determines dependency resolution for the desktop workspace).

    Parses the repo-root ``.gitignore`` via *pathspec* so we automatically
    skip ``node_modules/``, ``dist/``, ``*.pyc``, etc. without maintaining
    a hardcoded skip-list.
    """
    h = hashlib.sha256()

    def _hash_file(path: Path) -> None:
        rel = str(path.relative_to(project_root))
        h.update(rel.encode())
        h.update(b"\0")
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except (OSError, IOError):
            pass
        h.update(b"\0")


    from pathspec import PathSpec

    gitignore = project_root / ".gitignore"
    lines: list[str] = []
    if gitignore.is_file():
        lines = gitignore.read_text(encoding="utf-8").splitlines()
    spec = PathSpec.from_lines("gitignore", lines)

    # Root workspace config
    for name in ("package.json", "package-lock.json"):
        p = project_root / name
        if p.is_file():
            rel = str(p.relative_to(project_root))
            if not spec.match_file(rel):
                _hash_file(p)

    # Walk apps/desktop/ — prune ignored directories in-place
    desktop_dir = project_root / "apps" / "desktop"
    for dirpath, dirnames, filenames in os.walk(desktop_dir, topdown=True):
        # Prune ignored directories so we never descend into them
        dirnames[:] = [
            d for d in dirnames
            if not spec.match_file(str((Path(dirpath) / d).relative_to(project_root)))
        ]

        for fn in sorted(filenames):
            fp = Path(dirpath) / fn
            rel = str(fp.relative_to(project_root))
            if not spec.match_file(rel):
                _hash_file(fp)

    return h.hexdigest()


def _desktop_stamp_path() -> Path:
    """Return the path to the desktop build stamp file under $HERMES_HOME."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "desktop-build-stamp.json"


def _renderer_bundle_dir(desktop_dir: Path, *, source_mode: bool) -> Optional[Path]:
    """The renderer ``dist`` directory a launch loads, when it is inspectable.

    Source mode builds to ``apps/desktop/dist``. A packaged app ships the same
    bundle twice — inside ``app.asar`` and, because ``asarUnpack`` lists
    ``dist/**``, beside it in ``app.asar.unpacked``. Only the unpacked copy is
    a real directory; that is also the one an interrupted replace tears, so
    checking it catches the failure we care about.
    """
    if source_mode:
        return desktop_dir / "dist"

    executable = _desktop_packaged_executable(desktop_dir)
    if executable is None:
        return None

    # macOS: …/Hermes.app/Contents/MacOS/Hermes → …/Contents/Resources
    resources = (
        executable.parent.parent / "Resources"
        if sys.platform == "darwin"
        else executable.parent / "resources"
    )
    return resources / "app.asar.unpacked" / "dist"


# The module files the renderer fetches before any app code runs: Vite emits
# them as `<script type="module" src>` plus `<link rel="modulepreload" href>`.
_HTML_TAG_WITH_URL = re.compile(r"""<(?:script|link)\b[^>]*\b(?:src|href)=["']([^"']+)["'][^>]*>""", re.IGNORECASE)
_MODULE_TAG = re.compile(r"""\btype=["']module["']|\brel=["']modulepreload["']""", re.IGNORECASE)


def _renderer_bundle_torn(dist_dir: Path) -> bool:
    """True when ``index.html`` names hashed module files that aren't there.

    ``index.html`` and the hashed chunks under ``assets/`` are ONE generation.
    An update that replaces the app while its files are locked (antivirus, a
    still-running instance, an interrupted Windows replace) can leave the two
    behind from different generations. The app then launches and dies on the
    first lazy import with ``Failed to fetch dynamically imported module:
    …/assets/<chunk>-<hash>.js`` — and because the content stamp still matches
    the intact SOURCE tree, ``hermes desktop`` skips the rebuild that would fix
    it, so every relaunch reproduces the crash and reinstalling looks like the
    only way out. Detecting the tear turns it into a normal rebuild.

    Conservative: an unreadable index, or one naming nothing checkable, is NOT
    reported as torn — the missing-bundle guards own those cases.
    """
    try:
        html = (dist_dir / "index.html").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    for match in _HTML_TAG_WITH_URL.finditer(html):
        href = match.group(1)
        # Absolute/CDN URLs aren't part of this bundle's generation.
        if not _MODULE_TAG.search(match.group(0)) or re.match(r"^[a-z]+:|^//", href, re.IGNORECASE):
            continue
        rel = href.split("?", 1)[0].split("#", 1)[0].lstrip("./")
        if rel and not (dist_dir / rel).exists():
            return True

    return False


def _desktop_build_needed(desktop_dir: Path, project_root: Path, *, source_mode: bool) -> bool:
    """Return True when the desktop build output is stale, missing, or torn.

    Compares the current content hash against the saved stamp. Also returns
    True if the expected build artifact doesn't exist (e.g. first run after
    ``hermes update`` that pulled new source but hasn't built yet).
    """
    # If there's no build output at all, we definitely need to build
    if source_mode:
        if not _desktop_dist_exists(desktop_dir):
            return True
    else:
        if _desktop_packaged_executable(desktop_dir) is None:
            return True

    # A torn renderer bundle is stale no matter what the stamp says: the hash
    # describes the SOURCE tree, which is intact, while the built output is the
    # half-replaced one that crashes on its first lazy import.
    dist_dir = _renderer_bundle_dir(desktop_dir, source_mode=source_mode)
    if dist_dir is not None and _renderer_bundle_torn(dist_dir):
        print(f"  ⚠ A previous update left the desktop bundle incomplete ({dist_dir}); rebuilding it")
        return True

    stamp_file = _desktop_stamp_path()
    if not stamp_file.is_file():
        return True

    try:
        stamp_data = json.loads(stamp_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError):
        return True

    # If the mode changed (source vs packaged), force a rebuild
    if stamp_data.get("sourceMode") != source_mode:
        return True

    saved_hash = stamp_data.get("contentHash")
    if not saved_hash:
        return True

    current_hash = _compute_desktop_content_hash(project_root)
    return current_hash != saved_hash


def _write_desktop_build_stamp(project_root: Path, *, source_mode: bool) -> None:
    """Write the desktop build stamp after a successful build."""
    stamp_file = _desktop_stamp_path()
    try:
        stamp_file.parent.mkdir(parents=True, exist_ok=True)
        content_hash = _compute_desktop_content_hash(project_root)
        from datetime import datetime, timezone
        stamp_data = {
            "contentHash": content_hash,
            "sourceMode": source_mode,
            "builtAt": datetime.now(timezone.utc).isoformat(),
        }
        stamp_file.write_text(json.dumps(stamp_data, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        # Never let stamp-writing block or fail a build
        logger.debug("Failed to write desktop build stamp: %s", exc)


def _desktop_packaged_executable(desktop_dir: Path) -> Optional[Path]:
    """Return the current platform's unpacked Electron app executable."""
    release_dir = desktop_dir / "release"
    if sys.platform == "darwin":
        candidates = list(release_dir.glob("mac*/Hermes.app/Contents/MacOS/Hermes"))
    elif sys.platform == "win32":
        candidates = [
            release_dir / "win-unpacked" / "Hermes.exe",
            release_dir / "win-ia32-unpacked" / "Hermes.exe",
            release_dir / "win-arm64-unpacked" / "Hermes.exe",
        ]
    else:
        candidates = [
            release_dir / "linux-unpacked" / "hermes",
            release_dir / "linux-unpacked" / "Hermes",
            release_dir / "linux-arm64-unpacked" / "hermes",
            release_dir / "linux-arm64-unpacked" / "Hermes",
        ]

    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    if sys.platform == "win32" and len(existing) > 1:
        # Multiple unpacked trees can coexist (e.g. a stale win-arm64-unpacked
        # left behind by a cross-arch experiment next to the real win-unpacked).
        # Picking purely by mtime can then hand a wrong-architecture Hermes.exe
        # to the launcher, which Windows rejects with "This app can't run on
        # your computer" (#69179). Prefer candidates whose PE machine field
        # matches the host; fall back to mtime when none can be parsed.
        expected = _expected_windows_pe_machines()
        matching = [p for p in existing if _pe_machine_or_none(p) in expected]
        if matching:
            existing = matching
    return max(existing, key=lambda p: p.stat().st_mtime)


# ─── Desktop exe integrity gate (#69179) ────────────────────────────────────
#
# The desktop self-update chain (Desktop → hermes-setup --update →
# `hermes update` → `hermes desktop --build-only` → relaunch) rebuilds
# Hermes.exe on the end user's machine and used to verify only that the file
# EXISTS before declaring success. A corrupt cached Electron zip whose
# extraction produced a truncated electron.exe, an interrupted rcedit resource
# rewrite, a disk-full pack, or a wrong-arch unpacked tree therefore shipped a
# broken binary that Windows refuses to load ("This app can't run on your
# computer" / 此应用无法在你的电脑上运行). These helpers parse the PE header —
# no signature infrastructure required — so a structurally broken or
# wrong-architecture Hermes.exe is caught BEFORE the updater replaces the
# working app, and the previous build can be restored from the .bak tree that
# apps/desktop/scripts/before-pack.mjs now preserves.

_PE_MACHINE_I386 = 0x014C
_PE_MACHINE_AMD64 = 0x8664
_PE_MACHINE_ARM64 = 0xAA64

_PE_MACHINE_NAMES = {
    _PE_MACHINE_I386: "x86 (32-bit)",
    _PE_MACHINE_AMD64: "x64 (AMD64)",
    _PE_MACHINE_ARM64: "ARM64",
}

_PE_MACHINE_TO_NAME = {
    _PE_MACHINE_ARM64: "ARM64",
    _PE_MACHINE_AMD64: "AMD64",
    _PE_MACHINE_I386: "X86",
}

# MACHINE_ATTRIBUTES bits (processthreadsapi.h). UserEnabled means the host
# can run user-mode code of that machine type — natively or under emulation.
_MACHINE_ATTRIBUTE_USER_ENABLED = 0x00000001


def _windows_native_machine_from_iswow64() -> Optional[str]:
    """Ask IsWow64Process2 for the OS-native machine (None if unavailable/fail).

    ctypes defaults ``GetCurrentProcess``'s restype to ``c_int``, so the
    current-process pseudo-handle ``(HANDLE)-1`` is truncated to
    ``0xFFFFFFFF`` and zero-extended into a 64-bit invalid handle. On Win64
    that makes ``IsWow64Process2`` fail with ``ERROR_INVALID_HANDLE`` (6),
    which is exactly the residual Windows-on-ARM failure after #71218: the
    gate fell through to ``PROCESSOR_ARCHITECTURE=AMD64`` (the emulated
    process arch) and rejected a correctly-built ARM64 ``Hermes.exe``.
    Binding ``restype``/``argtypes`` to ``wintypes.HANDLE`` keeps the full
    ``0xFFFFFFFFFFFFFFFF`` pseudo-handle.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.IsWow64Process2.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.USHORT),
        ctypes.POINTER(wintypes.USHORT),
    ]
    kernel32.IsWow64Process2.restype = wintypes.BOOL

    process_machine = wintypes.USHORT(0)
    native_machine = wintypes.USHORT(0)
    if not kernel32.IsWow64Process2(
        kernel32.GetCurrentProcess(),
        ctypes.byref(process_machine),
        ctypes.byref(native_machine),
    ):
        return None
    return _PE_MACHINE_TO_NAME.get(native_machine.value)


def _windows_user_runnable_pe_machines() -> Optional[set]:
    """PE machines this host can run in user mode, via GetMachineTypeAttributes.

    This asks the question the integrity gate actually cares about — "can this
    Windows host load a PE of machine X?" — instead of inferring it from a
    host-architecture name. It is also the only documented API that reports
    AMD64-on-ARM64 emulation support; ``IsWow64GuestMachineSupported`` only
    answers for 32-bit guests.

    Returns None when the API is unavailable (pre-Windows-11 build 22000) or
    reports nothing runnable, so callers fall back to name-based detection.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetMachineTypeAttributes.argtypes = [
        wintypes.USHORT,
        ctypes.POINTER(ctypes.c_int),
    ]
    kernel32.GetMachineTypeAttributes.restype = ctypes.c_long

    runnable = set()
    for machine in (_PE_MACHINE_ARM64, _PE_MACHINE_AMD64, _PE_MACHINE_I386):
        attributes = ctypes.c_int(0)
        # HRESULT: zero is success, any nonzero value is a failure.
        if kernel32.GetMachineTypeAttributes(machine, ctypes.byref(attributes)):
            continue
        if attributes.value & _MACHINE_ATTRIBUTE_USER_ENABLED:
            runnable.add(machine)
    return runnable or None


def _windows_native_machine() -> str:
    """The Windows host OS's NATIVE machine architecture, normalized upper.

    ``platform.machine()`` reports the PROCESS architecture, which lies under
    emulation: the desktop update chain runs an x64 hermes-setup.exe (and thus
    x64 Python) on Windows-on-ARM devices, where ``platform.machine()``
    returns ``AMD64`` even though the OS is ARM64. The #71119 integrity gate
    then rejected the CORRECT ARM64 rebuild as an "architecture mismatch"
    (#69179 follow-up report). Probe order:

    1. ``IsWow64Process2`` with a correctly-typed current-process HANDLE
       (#71218 + HANDLE-truncation fix). This is the only API that tells the
       truth from an x64 process emulated on ARM64.
    2. ``PROCESSOR_ARCHITEW6432`` / ``PROCESSOR_ARCHITECTURE`` — WOW64
       (32-bit) hosts and pre-1511 Windows 10 without the newer API.
    3. ``platform.machine()``.

    Note ``GetNativeSystemInfo`` is deliberately NOT used: Microsoft documents
    that it "also returns emulated processor details when run from an app
    under emulation", so on the very WoA hosts this function exists to serve
    it reports AMD64 — no better than the env-var rung below it.
    """
    if sys.platform == "win32":
        try:
            name = _windows_native_machine_from_iswow64()
        except (OSError, AttributeError, TypeError, ValueError):
            # API missing (pre-1511), DLL load failure in tests, or a
            # mistyped ctypes binding — fall through to the env vars.
            name = None
        if name:
            return name
        env_arch = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get(
            "PROCESSOR_ARCHITECTURE"
        )
        if env_arch:
            return env_arch.upper()
    import platform as _platform

    return (_platform.machine() or "").upper()


def _expected_windows_pe_machines() -> set:
    """PE machine values the current Windows host can natively load.

    Preferred source is ``GetMachineTypeAttributes``, which answers this
    question directly (including AMD64-on-ARM64 emulation) instead of
    inferring it from an architecture name.

    Fallback is name-based: AMD64 hosts run x64 and (via WOW64) x86. ARM64
    hosts run ARM64 and (Windows 11 emulation) x64. 32-bit x86 hosts run only
    x86. Unknown machines return the permissive full set so the integrity gate
    can never brick launch on exotic hosts. Host detection uses the OS-native
    machine (see ``_windows_native_machine``), not the process architecture.
    """
    if sys.platform == "win32":
        try:
            runnable = _windows_user_runnable_pe_machines()
        except (OSError, AttributeError, TypeError, ValueError):
            runnable = None
        if runnable:
            return runnable
    machine = _windows_native_machine().upper()
    if machine in ("AMD64", "X86_64", "X64"):
        return {_PE_MACHINE_AMD64, _PE_MACHINE_I386}
    if machine in ("ARM64", "AARCH64"):
        return {_PE_MACHINE_ARM64, _PE_MACHINE_AMD64}
    if machine in ("X86", "I386", "I486", "I586", "I686"):
        return {_PE_MACHINE_I386}
    return {_PE_MACHINE_AMD64, _PE_MACHINE_ARM64, _PE_MACHINE_I386}


def _parse_pe_machine(path: Path) -> int:
    """Parse ``path`` as a PE executable and return its COFF machine field.

    Raises ``ValueError`` with a human-readable reason when the file is not a
    structurally complete PE: missing MZ/PE magic (an HTML error page or JSON
    body saved as .exe), header truncation, or raw section data extending past
    the end of the file (the truncated-download / interrupted-extraction
    shape). Purely a header walk — cheap even on a 200 MB Electron exe.
    """
    import struct

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"unreadable: {exc}")
    if file_size < 512:
        raise ValueError(
            f"file is only {file_size} bytes — far too small to be a Windows executable"
        )
    with path.open("rb") as fh:
        head = fh.read(64)
        if len(head) < 64 or head[:2] != b"MZ":
            raise ValueError(
                "missing MZ header — not a Windows executable "
                "(a truncated or non-binary file saved as .exe?)"
            )
        e_lfanew = struct.unpack_from("<I", head, 0x3C)[0]
        if e_lfanew <= 0 or e_lfanew + 24 > file_size:
            raise ValueError("corrupt DOS header: PE header offset points past end of file")
        fh.seek(e_lfanew)
        pe_head = fh.read(24)
        if len(pe_head) < 24 or pe_head[:4] != b"PE\x00\x00":
            raise ValueError("missing PE signature — corrupt executable header")
        machine, n_sections = struct.unpack_from("<HH", pe_head, 4)
        size_of_optional = struct.unpack_from("<H", pe_head, 20)[0]
        fh.seek(e_lfanew + 24 + size_of_optional)
        max_section_end = 0
        for _ in range(n_sections):
            section = fh.read(40)
            if len(section) < 40:
                raise ValueError("truncated PE section table")
            size_of_raw, pointer_to_raw = struct.unpack_from("<II", section, 16)
            max_section_end = max(max_section_end, pointer_to_raw + size_of_raw)
        if file_size < max_section_end:
            raise ValueError(
                f"truncated executable: file is {file_size} bytes but its PE "
                f"sections extend to {max_section_end} bytes"
            )
    return machine


def _pe_machine_or_none(path: Path) -> Optional[int]:
    try:
        return _parse_pe_machine(path)
    except ValueError:
        return None


def _desktop_exe_integrity_error(path: Path) -> Optional[str]:
    """Return a human-readable reason ``path`` cannot run on this Windows host,
    or ``None`` when the exe parses as a complete PE of a loadable architecture.
    """
    try:
        machine = _parse_pe_machine(path)
    except ValueError as exc:
        return str(exc)
    expected = _expected_windows_pe_machines()
    if machine not in expected:
        got = _PE_MACHINE_NAMES.get(machine, f"unknown machine 0x{machine:04X}")
        return (
            f"architecture mismatch: built a {got} executable but this is a "
            f"{_windows_native_machine()} Windows host"
        )
    return None


def _desktop_backup_unpacked_dir(packaged_executable: Path) -> Path:
    """The rollback tree before-pack.mjs preserves: ``<unpacked-dir>.bak``."""
    unpacked = packaged_executable.parent
    return unpacked.parent / (unpacked.name + ".bak")


def _rollback_desktop_from_backup(packaged_executable: Path) -> Optional[Path]:
    """Restore the previous unpacked desktop app from its ``.bak`` tree.

    Returns the restored executable path, or ``None`` when no usable backup
    exists (missing, or its exe fails the same integrity probe). The corrupt
    tree is kept alongside as ``<unpacked-dir>.corrupt`` for diagnostics.
    Best-effort: never raises.
    """
    unpacked = packaged_executable.parent
    backup_dir = _desktop_backup_unpacked_dir(packaged_executable)
    backup_exe = backup_dir / packaged_executable.name
    if not backup_exe.exists():
        return None
    if _desktop_exe_integrity_error(backup_exe) is not None:
        return None
    corrupt_dir = unpacked.parent / (unpacked.name + ".corrupt")
    try:
        shutil.rmtree(corrupt_dir, ignore_errors=True)
        try:
            unpacked.rename(corrupt_dir)
        except OSError:
            shutil.rmtree(unpacked, ignore_errors=True)
        backup_dir.rename(unpacked)
    except OSError:
        return None
    restored = unpacked / packaged_executable.name
    return restored if restored.exists() else None


def _ensure_desktop_exe_launchable(
    desktop_dir: Path, packaged_executable: Optional[Path]
) -> tuple:
    """Windows post-build integrity gate for the self-update rebuild (#69179).

    Returns ``(verified_exe_or_None, rolled_back)``:

    - exe passed the probe → ``(exe, False)``
    - exe corrupt/wrong-arch, previous build restored → ``(old_exe, True)``
    - exe corrupt and nothing restorable → ``(None, False)``

    On any integrity failure the corrupt cached Electron zip is purged and the
    desktop build stamp invalidated, so the updater's retry-once rebuild pulls
    a fresh, SHASUM-verified Electron download instead of re-staging the same
    corrupt bytes. No-op off Windows and when there is no executable to check.
    """
    if packaged_executable is None or sys.platform != "win32":
        return packaged_executable, False

    error = _desktop_exe_integrity_error(packaged_executable)
    if error is None:
        return packaged_executable, False

    print(f"✗ The built Hermes.exe failed its integrity check: {error}")
    print(f"    at: {packaged_executable}")

    # Self-heal setup for the retry: drop the (likely corrupt) cached Electron
    # zip and the content stamp so the next rebuild is a genuine re-download +
    # re-stage rather than a replay of the same broken extraction.
    _purge_electron_build_cache(desktop_dir)
    try:
        _desktop_stamp_path().unlink()
    except OSError:
        pass

    restored = _rollback_desktop_from_backup(packaged_executable)
    if restored is not None:
        print("  ↩ Update aborted — restored the previous working Hermes.exe from backup.")
        print("    Your existing version was kept and still works. Run `hermes desktop`")
        print("    (or the in-app update) again to retry with a fresh Electron download.")
        return restored, True

    print("  ✗ No usable backup was found to restore.")
    print("    Run `hermes desktop --force-build` to rebuild, or re-run the Hermes")
    print("    installer to repair the install.")
    return None, False


def _electron_download_cache_dirs() -> list[Path]:
    """Return the per-user Electron download cache directories for this OS.

    electron-builder's ``app-builder unpack-electron`` extracts the Electron
    distribution from a zip stored in this cache (NOT from node_modules), so a
    corrupt zip here — not a bad workspace install — is what poisons the build.
    Honors the ``electron_config_cache`` / ``ELECTRON_CACHE`` overrides that
    ``@electron/get`` respects, then falls back to the platform defaults.
    """
    home = Path.home()
    candidates: list[Path] = []
    override = os.environ.get("electron_config_cache") or os.environ.get("ELECTRON_CACHE")
    if override:
        candidates.append(Path(override))
    if sys.platform == "darwin":
        candidates.append(home / "Library" / "Caches" / "electron")
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.append(Path(local) / "electron" / "Cache")
        candidates.append(home / "AppData" / "Local" / "electron" / "Cache")
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            candidates.append(Path(xdg) / "electron")
        candidates.append(home / ".cache" / "electron")

    seen: set[Path] = set()
    out: list[Path] = []
    for c in candidates:
        rc = c.expanduser()
        if rc not in seen:
            seen.add(rc)
            out.append(rc)
    return out


def _purge_electron_build_cache(desktop_dir: Path) -> list[Path]:
    """Clear the cached Electron download + half-written unpacked dir so the
    next ``pack`` re-downloads and re-stages from scratch.

    Root cause of the ``ENOENT … rename '…/linux-unpacked/electron' ->
    '…/linux-unpacked/Hermes'`` desktop build failure: a corrupt zip in the
    per-user Electron download cache (a partial download resumed into the same
    file leaves prepended/concatenated junk, or an interrupted write truncates
    it). electron-builder's ``app-builder unpack-electron`` extracts the
    distribution from that cached zip (NOT from node_modules); a bad zip yields
    a partial tree MISSING the 193 MB ``electron`` binary, so the final rename
    dies. Re-running repeats the same broken extraction forever.

    We deliberately do NOT try to detect corruption ourselves. stdlib
    ``zipfile`` silently tolerates the prepended/concatenated junk that is the
    most common corruption here — it reads from the end-of-central-directory
    backward, so ``testzip()`` returns clean on exactly the zips ``unzip -t``
    and ``@electron/get`` reject. Gating the purge on a self-rolled validator
    would therefore skip the real-world case and never self-heal. Instead, on a
    packaged-build failure we unconditionally remove the version's cached zips
    and the stale unpacked dir, then let the caller retry once: ``@electron/get``
    re-downloads with its own SHASUM verification (the real source of truth),
    and ``before-pack.cjs`` re-wipes the unpacked dir. If the failure was
    unrelated, a clean re-download is harmless and the retry fails the same way.

    Best-effort: never raises. Returns the paths removed so the caller can log
    them and decide whether a retry is worthwhile (empty list ⇒ nothing to
    clear, so no point retrying).
    """
    removed: list[Path] = []

    for cache_dir in _electron_download_cache_dirs():
        if not cache_dir.is_dir():
            continue
        for zip_path in sorted(cache_dir.rglob("electron-*.zip")):
            try:
                zip_path.unlink()
                removed.append(zip_path)
            except OSError:
                # Locked/permission-denied entry is out of our hands; let the
                # build report its own error rather than masking it.
                pass

    # Drop the half-written unpacked dir too: an interrupted prior pack leaves
    # a partial tree that poisons the rename even after the zip is fixed.
    # (before-pack.cjs also handles this, but clearing it here makes the retry
    # robust even if the hook is somehow skipped.)
    release_dir = desktop_dir / "release"
    if release_dir.is_dir():
        for unpacked in release_dir.glob("*-unpacked"):
            try:
                shutil.rmtree(unpacked, ignore_errors=True)
                removed.append(unpacked)
            except OSError:
                pass

    return removed


# Last-resort Electron mirror after GitHub download fails (#47266). Only used
# when the user hasn't pinned ELECTRON_MIRROR.
_ELECTRON_FALLBACK_MIRROR = "https://npmmirror.com/mirrors/electron/"


def _electron_dir(project_root: Path) -> Path:
    """Return the Electron package directory the desktop workspace installs.

    npm may keep workspace-only dev dependencies under
    ``apps/desktop/node_modules`` instead of hoisting them to the repo root.
    Which layout you get depends on the npm version and what else is installed,
    so a build path that assumes one or the other breaks intermittently across
    machines. ``apps/desktop/package.json`` points electron-builder's
    ``electronDist`` at ``node_modules/electron/dist`` relative to the desktop
    project, so prefer the workspace-local package and fall back to the root
    hoist when that's where npm landed it.
    """
    desktop_local = project_root / "apps" / "desktop" / "node_modules" / "electron"
    if desktop_local.exists():
        return desktop_local
    return project_root / "node_modules" / "electron"


def _electron_dist_binary(project_root: Path) -> Path:
    """Return the path to the Electron main binary inside the installed package.

    electron-builder reads the binary from ``build.electronDist`` since #38673,
    so this is the exact file whose absence makes a pack fail with "The
    specified electronDist does not exist". The basename differs per OS (the
    platform Electron is named for the host the build runs on).
    """
    dist = _electron_dir(project_root) / "dist"
    if sys.platform == "darwin":
        return dist / "Electron.app" / "Contents" / "MacOS" / "Electron"
    if sys.platform == "win32":
        return dist / "electron.exe"
    return dist / "electron"


def _electron_dist_ok(project_root: Path) -> bool:
    """True when ``node_modules/electron/dist`` holds a usable Electron binary.

    A directory that exists but is missing the binary (a partial extraction from
    a corrupt cached zip, or an interrupted postinstall) counts as NOT ok, since
    that is exactly the shape that makes electron-builder throw on the pinned
    electronDist.
    """
    try:
        return _electron_dist_binary(project_root).exists()
    except OSError:
        return False


def _electron_pkg_staged_missing_dist(project_root: Path) -> bool:
    """electron staged (package.json + install.js) but dist missing — blocked postinstall."""
    electron_dir = _electron_dir(project_root)
    return (
        (electron_dir / "package.json").is_file()
        and (electron_dir / "install.js").is_file()
        and not _electron_dist_ok(project_root)
    )


def _redownload_electron_dist(
    project_root: Path,
    env: dict,
    *,
    mirror: Optional[str] = None,
) -> bool:
    """Best-effort: run electron's install.js to populate dist/ (optional mirror)."""
    if _electron_dist_ok(project_root):
        return True

    electron_dir = _electron_dir(project_root)
    installer = electron_dir / "install.js"
    if not installer.is_file():
        return False
    from hermes_constants import find_node_executable, with_hermes_node_path

    node = find_node_executable("node")
    if not node:
        return False

    dist_dir = electron_dir / "dist"
    shutil.rmtree(dist_dir, ignore_errors=True)
    try:
        (electron_dir / "path.txt").unlink()
    except OSError:
        pass

    dl_env = with_hermes_node_path(env)
    if mirror:
        dl_env["ELECTRON_MIRROR"] = mirror
    try:
        subprocess.run([node, str(installer)], cwd=str(electron_dir), env=dl_env, check=False)
    except OSError:
        return False
    return _electron_dist_ok(project_root)


def _try_redownload_electron_dist(project_root: Path, env: dict) -> bool:
    """Canonical download, then fallback mirror unless the user pinned one."""
    if _redownload_electron_dist(project_root, env):
        return True
    if env.get("ELECTRON_MIRROR"):
        return False
    return _redownload_electron_dist(project_root, env, mirror=_ELECTRON_FALLBACK_MIRROR)


def _stop_desktop_processes_locking_build(desktop_dir: Path) -> list[int]:
    """Terminate any running desktop app executing from this build's ``release``
    dir so a rebuild can replace its (otherwise locked) executable.

    On Windows a running ``Hermes.exe`` keeps an exclusive lock on
    ``release/win-unpacked/Hermes.exe``. electron-builder's pack then can't
    delete the stale binary and dies with ``remove …\\Hermes.exe: Access is
    denied`` / ``ERR_ELECTRON_BUILDER_CANNOT_EXECUTE`` (before-pack hits the same
    EPERM cleaning the dir). The retry path repeats the failure because the lock
    is still held. POSIX lets you unlink a running binary, so this is a no-op
    off-Windows.

    Scope is deliberately narrow: only processes whose executable lives *inside*
    this desktop's ``release`` tree are stopped — a packaged install elsewhere or
    an unrelated "Hermes" process is never touched. Best-effort: never raises.
    Returns the PIDs we asked to stop.
    """
    if sys.platform != "win32":
        return []
    try:
        import psutil
    except Exception:
        return []
    try:
        release_dir = (desktop_dir / "release").resolve()
    except OSError:
        return []
    if not release_dir.is_dir():
        return []

    me = os.getpid()
    victims = []
    try:
        proc_iter = psutil.process_iter(["pid", "exe"])
    except Exception:
        return []
    for proc in proc_iter:
        try:
            info = proc.info
        except Exception:
            continue
        pid = info.get("pid")
        exe = info.get("exe")
        if not exe or pid is None or pid == me:
            continue
        try:
            exe_path = Path(exe).resolve()
        except (OSError, ValueError):
            continue
        if release_dir in exe_path.parents:
            victims.append(proc)

    stopped: list[int] = []
    for proc in victims:
        try:
            proc.terminate()
            stopped.append(int(proc.pid))
        except Exception:
            continue
    if stopped:
        # Wait for the handles (and thus the file locks) to actually release.
        try:
            _, alive = psutil.wait_procs(victims, timeout=5)
            for proc in alive:
                try:
                    proc.kill()
                except Exception:
                    continue
        except Exception:
            pass
    return stopped


def _desktop_macos_bundle_id(bundle: Path) -> Optional[str]:
    """Return a bundle/framework CFBundleIdentifier for local macOS signing."""
    import plistlib

    info = bundle / "Contents" / "Info.plist"
    if not info.exists() and bundle.suffix == ".framework":
        candidates = list(bundle.glob("Versions/*/Resources/Info.plist")) + list(
            bundle.glob("Resources/Info.plist")
        )
        if candidates:
            info = candidates[0]
    if not info.exists():
        return None
    try:
        data = plistlib.loads(info.read_bytes())
    except Exception:
        return None
    ident = data.get("CFBundleIdentifier")
    return str(ident) if ident else None


def _desktop_macos_local_signing_identity() -> Optional[str]:
    """Return the opt-in keychain identity for local macOS desktop signing.

    ``desktop.macos_signing_identity`` in config.yaml names a persistent
    code-signing certificate in the user's login keychain (a self-signed
    "Code Signing" cert made in Keychain Access is enough — no Apple Developer
    account needed). Signing with any identity gives the app a
    certificate-anchored Designated Requirement, which is the strongest way to
    keep macOS TCC grants (Full Disk Access, Accessibility, Automation, Files
    and Folders) stable across local rebuilds. Empty/unset keeps the default
    identifier-pinned ad-hoc signing.
    """
    if sys.platform != "darwin":
        return None
    try:
        from hermes_cli.config import load_config

        desktop = load_config().get("desktop", {})
        if not isinstance(desktop, dict):
            return None
        identity = desktop.get("macos_signing_identity")
        if not isinstance(identity, str):
            return None
        return identity.strip() or None
    except Exception as exc:
        print(
            "  (warning: could not load desktop.macos_signing_identity: "
            f"{exc}; falling back to ad-hoc signing)"
        )
        return None


def _desktop_macos_has_valid_real_signature(app: Path) -> bool:
    """True when the bundle carries an intact non-ad-hoc (Team ID) signature.

    Used to make the relaunch fixup a no-op on properly signed/notarized
    builds even when CSC_LINK / APPLE_SIGNING_IDENTITY aren't in the
    environment (e.g. a release DMG install being repaired) — clobbering a
    Developer ID signature with an ad-hoc one would reset TCC grants and can
    break the hardened runtime. A *stale* real signature (in-place rebuild
    tampered with the bundle) fails --verify and returns False so the fixup
    can repair it.
    """
    codesign = shutil.which("codesign")
    if not codesign:
        return False
    try:
        info = subprocess.run(
            [codesign, "-dv", str(app)], check=False, capture_output=True, text=True
        )
        output = f"{info.stdout}\n{info.stderr}"
        if info.returncode != 0 or "TeamIdentifier=" not in output \
                or "TeamIdentifier=not set" in output:
            return False
        verify = subprocess.run(
            [codesign, "--verify", "--deep", "--strict", str(app)],
            check=False, capture_output=True,
        )
        return verify.returncode == 0
    except Exception:
        return False


def _desktop_macos_local_codesign(
    app: Path, *, desktop_dir: Path, identity: str = "-"
) -> bool:
    """Re-sign a local Desktop build so macOS TCC grants survive rebuilds.

    A plain ``codesign --deep --sign -`` leaves the bundle with a cdhash-only
    Designated Requirement and strips electron-builder's entitlements. Every
    rebuild changes the cdhash, so TCC (Full Disk Access, Accessibility,
    Automation, Files and Folders: Desktop/Downloads/Documents, microphone)
    treats the rebuilt app as different code and the user must re-grant
    everything — and the lost entitlements break microphone/JIT under the
    hardened runtime.

    Instead, sign inside-out (standalone Mach-O binaries, then nested
    frameworks/helper apps, then the main bundle), preserving the repo's
    entitlement plists, and pin an explicit identifier-based Designated
    Requirement when signing ad-hoc. With a real ``identity`` the certificate
    anchors the DR, so no explicit requirement is needed. Raises on signing
    failure; returns True after strict verification passes.
    """
    codesign = shutil.which("codesign")
    if not codesign:
        return False

    ent_main = desktop_dir / "electron" / "entitlements.mac.plist"
    ent_inherit = desktop_dir / "electron" / "entitlements.mac.inherit.plist"
    if not (ent_main.exists() and ent_inherit.exists()):
        # Hardened-runtime restrictions are enforced even for ad-hoc
        # signatures. Signing with --options runtime but WITHOUT the allow-jit
        # entitlements would leave Electron/V8 crashing on launch — strictly
        # worse than the legacy plain ad-hoc sign. Bail out so the caller
        # falls back to that legacy path instead.
        raise FileNotFoundError(
            f"desktop entitlement plists missing under {desktop_dir / 'electron'}"
        )

    def sign_path(
        path: Path,
        *,
        entitlements: Optional[Path] = None,
        identifier: Optional[str] = None,
        runtime: bool = True,
    ) -> None:
        args = [codesign, "--force", "--sign", identity, "--timestamp=none"]
        if runtime:
            args += ["--options", "runtime"]
        if entitlements is not None and entitlements.exists():
            args += ["--entitlements", str(entitlements)]
        if identifier and identity == "-":
            # Ad-hoc signatures get a cdhash-only DR by default; pin an
            # identifier-based DR so TCC has something stable to persist.
            args += ["--requirements", f'=designated => identifier "{identifier}"']
        args.append(str(path))
        subprocess.run(args, check=True, capture_output=True)

    # 1) Standalone Mach-O files (native modules, dylibs, crashpad handler).
    #    Compare paths relative to the app root — the absolute path always
    #    contains the outer Hermes.app component, so an absolute-parts check
    #    would skip every file.
    contents = app / "Contents"
    standalone: list[Path] = []
    for root, _dirs, files in os.walk(contents):
        root_path = Path(root)
        rel_parts = root_path.relative_to(app).parts
        if any(part.endswith(".app") for part in rel_parts):
            continue  # nested helper apps are signed as bundles below
        for name in files:
            fp = root_path / name
            if name in {"chrome_crashpad_handler", "spawn-helper"} or fp.suffix in {
                ".node",
                ".dylib",
            }:
                standalone.append(fp)
    for fp in sorted(standalone, key=lambda p: len(p.parts), reverse=True):
        sign_path(fp, runtime=False)

    # 2) Nested frameworks and helper apps, deepest first.
    bundles: list[Path] = []
    frameworks_dir = contents / "Frameworks"
    if frameworks_dir.exists():
        for root, _dirs, _files in os.walk(frameworks_dir):
            p = Path(root)
            if p.suffix in {".framework", ".app"}:
                bundles.append(p)
    for bundle in sorted(set(bundles), key=lambda p: len(p.parts), reverse=True):
        ent = ent_inherit if bundle.suffix == ".app" and "Helper" in bundle.name else None
        sign_path(bundle, entitlements=ent, identifier=_desktop_macos_bundle_id(bundle))

    # 3) The main bundle, with the app's own entitlements.
    sign_path(app, entitlements=ent_main, identifier=_desktop_macos_bundle_id(app))
    subprocess.run(
        [codesign, "--verify", "--deep", "--strict", str(app)],
        check=True, capture_output=True,
    )
    return True


def _desktop_macos_relaunchable_fixup(
    desktop_dir: Path,
    *,
    publisher_signing_configured: Optional[bool] = None,
) -> bool:
    """Make a locally-built macOS desktop app survive in-place self-update
    without resetting the user's TCC permission grants.

    An ad-hoc-signed .app has no stable Designated Requirement, so when the
    self-updater rebuilds the bundle in place (new cdhash) Gatekeeper reports
    "Hermes is damaged and can't be opened" — and macOS TCC forgets every
    permission the user granted (Full Disk Access, Desktop/Downloads/Documents,
    Accessibility, Automation, microphone), re-prompting on every launch after
    every update.

    Clear the quarantine xattrs, then re-sign with a stable identity:
    ``desktop.macos_signing_identity`` (a persistent keychain cert — strongest)
    when configured, else ad-hoc with identifier-pinned Designated Requirements,
    preserving the repo's entitlement plists either way. No-op when a real
    publisher identity is configured (CSC_LINK / APPLE_SIGNING_IDENTITY) or the
    bundle already carries an intact Developer ID signature, so a properly
    signed/notarized build is never clobbered. Callers that already made the
    publisher-signing decision may pass it explicitly so a later dotenv load
    can't reverse it. Falls back to the legacy deep ad-hoc sign if the
    entitlement-preserving path fails. Best-effort: never raises. Returns True
    when no work was needed or signing + strict verification succeeded.
    """
    if sys.platform != "darwin":
        return True
    if publisher_signing_configured is None:
        publisher_signing_configured = bool(
            os.environ.get("CSC_LINK") or os.environ.get("APPLE_SIGNING_IDENTITY")
        )
    if publisher_signing_configured:
        return True
    exe = _desktop_packaged_executable(desktop_dir)
    if exe is None:
        return True
    # exe = .../Hermes.app/Contents/MacOS/Hermes  ->  app bundle = .../Hermes.app
    app = exe.parents[2]
    if not str(app).endswith(".app") or not app.is_dir():
        return True
    codesign = shutil.which("codesign")
    if not codesign:
        return False
    if _desktop_macos_has_valid_real_signature(app):
        return True
    subprocess.run(["xattr", "-cr", str(app)], check=False)
    identity = _desktop_macos_local_signing_identity() or "-"
    try:
        if _desktop_macos_local_codesign(app, desktop_dir=desktop_dir, identity=identity):
            label = "keychain identity" if identity != "-" else "stable ad-hoc identity"
            print(f"  → macOS desktop signed with {label}; TCC grants persist across rebuilds")
            return True
    except Exception as exc:
        if identity != "-":
            print(
                f"  (warning: configured macOS signing identity failed: {identity!r}; "
                "falling back to ad-hoc — TCC grants may need to be re-granted)"
            )
        print(f"  (warning: stable macOS signing failed ({exc}); using legacy ad-hoc sign)")
    try:
        # Legacy ad-hoc fallback: re-sign, but NEVER delete the safeStorage
        # keychain item. Deleting it would permanently orphan every
        # credential encrypted under it (gateway token, native OAuth access/
        # refresh tokens) — and this path is reached exactly when the
        # entitlement-preserving signer failed, so there is no verified
        # successor identity to hand the key to. The keychain prompt macOS
        # shows instead is recoverable ("Always Allow" updates the item's ACL
        # partition list and preserves the key); deletion is not. The real
        # fix (proof-carrying rotation/migration) belongs in Electron, where
        # safeStorage can read the old key. Tracked as follow-up.
        result = subprocess.run(
            [codesign, "--force", "--deep", "--sign", "-", str(app)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(
                f"  (warning: legacy ad-hoc re-sign failed (exit {result.returncode}); "
                "leaving safeStorage keychain item untouched)"
            )
            return False
        verify = subprocess.run(
            [codesign, "--verify", "--deep", "--strict", str(app)],
            check=False, capture_output=True, text=True,
        )
        if verify.returncode != 0:
            print(
                f"  (warning: legacy ad-hoc re-sign did not pass strict verification; "
                "leaving safeStorage keychain item untouched)"
            )
            return False
        print("  → macOS desktop re-signed (legacy ad-hoc); safeStorage keychain item left untouched")
        return True
    except Exception as exc:
        print(f"  (warning: macOS relaunch fixup skipped: {exc})")
    return False


def _macos_codesigning_identity_valid(security: str, identity: str) -> bool:
    """True when `identity` appears among VALID code-signing identities.

    ``security find-identity -p codesigning`` (without ``-v``) also lists
    certificates macOS will refuse to sign with — e.g. a self-signed cert that
    was imported but never trusted for the codeSign policy. Only the ``-v``
    listing proves codesign can actually use it, so this is both the
    idempotency probe and the success postcondition for
    ``--setup-tcc-identity``. Never raises.
    """
    try:
        result = subprocess.run(
            [security, "find-identity", "-v", "-p", "codesigning"],
            capture_output=True, text=True, check=False,
        )
    except Exception:
        return False

    return f'"{identity}"' in (result.stdout or "")


def _desktop_macos_setup_tcc_identity(identity: str = "Hermes Local Signing") -> bool:
    """Create/import a self-signed code-signing cert and configure Hermes to use it.

    One-shot setup for ``hermes desktop --setup-tcc-identity``. Creates a
    self-signed "Code Signing" certificate in the login keychain (the same
    artifact the docs describe creating manually via Keychain Access), grants
    ``codesign`` access to it, writes ``desktop.macos_signing_identity`` to
    config.yaml, and re-signs the already-packaged app so the next launch uses
    the certificate-anchored identity.

    Why this matters: macOS TCC grants (Full Disk Access, Accessibility,
    Automation, Files and Folders, microphone) persist against the app's
    code-signing identity, not its path. A plain ad-hoc signature gets a
    cdhash-only Designated Requirement, so every rebuild looks like a new app
    and the user must re-grant everything. A certificate-anchored identity is
    stable across rebuilds — the same mechanism yabai/skhd users rely on.

    Idempotent: re-running after an update finds the existing certificate and
    only re-points the config + re-signs. Returns True on success (or when
    already configured), False on failure. Never raises.
    """
    if sys.platform != "darwin":
        print("  (--setup-tcc-identity is macOS-only; skipping)")
        return False

    openssl = shutil.which("openssl")
    security = shutil.which("security")
    codesign = shutil.which("codesign")
    if not (openssl and security and codesign):
        print(
            "  (--setup-tcc-identity requires openssl, security, and codesign; "
            f"found openssl={bool(openssl)} security={bool(security)} codesign={bool(codesign)})"
        )
        return False

    keychain = str(Path.home() / "Library" / "Keychains" / "login.keychain-db")
    # A certificate that merely EXISTS in the keychain is not enough — macOS
    # only treats it as a code-signing identity once it is trusted for the
    # codeSign policy. Probe with `-v` (valid identities only) so a previously
    # imported-but-untrusted cert is repaired rather than reported as done.
    already_imported = _macos_codesigning_identity_valid(security, identity)

    if not already_imported:
        # Create a self-signed code-signing cert (valid 10 years) and import it
        # into the login keychain with codesign access so signing works without
        # an interactive unlock prompt.
        tmp_dir = Path(tempfile.mkdtemp(prefix="hermes-tcc-"))
        try:
            key = tmp_dir / "sign.key"
            crt = tmp_dir / "sign.crt"
            p12 = tmp_dir / "sign.p12"
            subprocess.run(
                [
                    openssl, "req", "-x509", "-newkey", "rsa:2048",
                    "-keyout", str(key), "-out", str(crt),
                    "-days", "3650", "-nodes",
                    "-subj", f"/CN={identity}",
                    "-addext", "basicConstraints=critical,CA:TRUE",
                    "-addext", "keyUsage=critical,digitalSignature,keyCertSign",
                    "-addext", "extendedKeyUsage=codeSigning",
                ],
                capture_output=True, check=True,
            )
            # OpenSSL 3 defaults to AES/SHA-2 PKCS#12 encryption that macOS
            # `security import` rejects with "MAC verification failed during
            # PKCS12 import (wrong password?)". The `-legacy` flag restores the
            # RC2/SHA-1 format the importer accepts, but only exists on
            # OpenSSL 3 — so try the plain export first and fall back to
            # `-legacy` when the IMPORT fails with that signature. (Verified
            # E2E on macOS 26.3.1 / OpenSSL 3.6.3 by @ctaylor86 on PR #77189.)
            def _export_p12(extra_args: list) -> None:
                subprocess.run(
                    [
                        openssl, "pkcs12", "-export", *extra_args,
                        "-inkey", str(key), "-in", str(crt),
                        "-out", str(p12), "-passout", "pass:hermeslocal",
                    ],
                    capture_output=True, check=True,
                )

            def _import_p12():
                return subprocess.run(
                    [
                        security, "import", str(p12), "-k", keychain,
                        "-P", "hermeslocal",
                        "-T", codesign, "-T", "/usr/bin/codesign_allocate",
                    ],
                    capture_output=True, text=True, check=False,
                )

            _export_p12([])
            imported = _import_p12()
            if imported.returncode != 0 and "MAC verification failed" in (imported.stderr or ""):
                try:
                    _export_p12(["-legacy"])
                    imported = _import_p12()
                except subprocess.CalledProcessError:
                    # Older OpenSSL without -legacy: keep the original failure.
                    pass
            if imported.returncode != 0:
                print(f"  (could not import signing identity into keychain: {imported.stderr.strip()})")
                return False

            # Importing is still not enough: without explicit trust for the
            # codeSign policy, `security find-identity -v -p codesigning`
            # reports 0 valid identities and codesign refuses the cert. Trust
            # the self-signed root for code signing. This writes to the user's
            # trust settings, so macOS may prompt for the login password ONCE
            # here — that is the one-time setup cost this command exists to
            # front-load.
            trusted = subprocess.run(
                [security, "add-trusted-cert", "-r", "trustRoot", "-p", "codeSign", "-k", keychain, str(crt)],
                capture_output=True, text=True, check=False,
            )
            if trusted.returncode != 0:
                print(
                    "  (could not trust the certificate for code signing: "
                    f"{(trusted.stderr or trusted.stdout).strip()})"
                )
                return False
            print(f"  → created, imported, and trusted self-signed identity: {identity!r}")
        except Exception as exc:
            print(f"  (certificate creation failed: {exc})")
            return False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"  → identity {identity!r} already valid in keychain")

    # Postcondition gate: only report success once macOS actually agrees the
    # identity is usable for code signing. Name-in-output checks pass for
    # invalid identities; this is the check that failed silently before.
    if not _macos_codesigning_identity_valid(security, identity):
        print(
            f"  (identity {identity!r} was imported but is not a VALID code-signing identity; "
            "run `security find-identity -v -p codesigning` to inspect, and see the manual "
            "Keychain Access steps in the desktop docs)"
        )
        return False

    # Point Hermes at the identity (config.yaml, not .env — it's not a secret).
    try:
        from hermes_cli.config import set_config_value

        set_config_value("desktop.macos_signing_identity", identity)
        print(f"  → set desktop.macos_signing_identity = {identity!r}")
    except Exception as exc:
        print(f"  (could not write desktop.macos_signing_identity: {exc})")
        return False

    # Re-sign the packaged app so the current build already uses the identity.
    desktop_dir = PROJECT_ROOT / "apps" / "desktop"
    if _desktop_packaged_executable(desktop_dir) is not None:
        try:
            if _desktop_macos_relaunchable_fixup(desktop_dir):
                print(
                    "  → packaged app re-signed with certificate-anchored identity; "
                    "TCC grants persist across rebuilds"
                )
        except Exception as exc:
            print(f"  (could not re-sign packaged app: {exc})")

    print(
        "\n  Note: macOS will re-prompt for permissions ONE final time (the identity "
        "changed). Grant them and they persist from then on. If a permission gets "
        "stuck, reset it with:  tccutil reset All com.nousresearch.hermes"
    )
    return True


def _force_adhoc_macos_signing(env: dict, *, source_mode: bool) -> bool:
    """Stop electron-builder grabbing a random keychain identity on self-update.

    The desktop self-updater rebuilds *and re-signs the .app on the end user's
    machine* (``hermes desktop --build-only`` → electron-builder ``--dir``).
    With ``CSC_IDENTITY_AUTO_DISCOVERY`` on (its default), electron-builder
    signs the ``type=distribution``, hardened-runtime bundle with whatever it
    finds in that user's keychain — typically a personal "Apple Development"
    cert. That stalls/fails the sign step (no Developer ID + no provisioning
    profile) or clobbers your real notarized signature with an unusable one, so
    every post-update launch trips Gatekeeper.

    Force ad-hoc signing for the local packaged rebuild instead: deterministic,
    and exactly what ``_desktop_macos_relaunchable_fixup`` already finishes off.
    No-op for source runs, off-macOS, when a real identity is configured
    (``CSC_LINK`` / ``APPLE_SIGNING_IDENTITY``), or when the caller already
    pinned the flag. Mutates ``env``; returns True when it set the flag.
    """
    if sys.platform != "darwin" or source_mode:
        return False
    if env.get("CSC_LINK") or env.get("APPLE_SIGNING_IDENTITY"):
        return False
    if "CSC_IDENTITY_AUTO_DISCOVERY" in env:
        return False
    env["CSC_IDENTITY_AUTO_DISCOVERY"] = "false"
    return True


def _desktop_linux_needs_no_sandbox() -> bool:
    """Return True when Chromium/Electron should bypass the Linux sandbox.

    Ubuntu 23.10+ can enable AppArmor's
    ``apparmor_restrict_unprivileged_userns`` hardening, which breaks
    Chromium/Electron's user-namespace sandbox for normal users unless the app
    ships a working root-owned 4755 ``chrome-sandbox`` helper. In headless or
    non-interactive CLI contexts we may be unable to ``sudo chown/chmod`` that
    helper, so detect the host restriction and fall back to ``--no-sandbox``
    rather than hard-failing the launcher.

    We intentionally do NOT return True for root users here: running Electron as
    root without a sandbox is a qualitatively riskier path than launching as an
    unprivileged desktop user on an AppArmor-restricted host. The root case
    should remain an explicit user choice.
    """
    if os.environ.get("ELECTRON_DISABLE_SANDBOX", 0) == "1":
        return True

    if sys.platform != "linux":
        return False
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return False
    try:
        with open("/proc/sys/kernel/apparmor_restrict_unprivileged_userns", encoding="utf-8") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def _desktop_linux_sandbox_helper_is_regular_file(packaged_executable: Path) -> bool:
    """Return True when ``chrome-sandbox`` exists as a regular file."""
    if sys.platform != "linux":
        return False
    sandbox = packaged_executable.parent / "chrome-sandbox"
    try:
        sandbox_lstat = sandbox.lstat()
    except OSError:
        return False
    return stat.S_ISREG(sandbox_lstat.st_mode)



def _desktop_linux_sandbox_fixup(packaged_executable: Path) -> bool:
    """Configure Electron's Linux SUID sandbox helper when required."""
    if sys.platform != "linux":
        return True

    sandbox = packaged_executable.parent / "chrome-sandbox"
    if not sandbox.exists():
        print(f"✗ Hermes Desktop is missing Electron's Linux sandbox helper: {sandbox}")
        return False

    # Reject symlinks — chown/chmod must not follow an attacker-controlled
    # link to an arbitrary path.  Use lstat() so we inspect the link itself
    # rather than the target, and require a regular file.
    try:
        sandbox_lstat = sandbox.lstat()
    except OSError:
        print(f"✗ Cannot stat Electron's Linux sandbox helper: {sandbox}")
        return False
    if not stat.S_ISREG(sandbox_lstat.st_mode):
        print(f"✗ Electron's Linux sandbox helper is not a regular file: {sandbox}")
        return False

    if sandbox_lstat.st_uid == 0 and stat.S_IMODE(sandbox_lstat.st_mode) == 0o4755:
        return True

    sudo = shutil.which("sudo")
    if not sudo:
        print("✗ Hermes Desktop requires sudo to configure Electron's Linux sandbox helper.")
        return False

    print("→ Configuring Electron Linux sandbox helper (sudo required)...")
    for command in ([sudo, "chown", "root:root", str(sandbox)], [sudo, "chmod", "4755", str(sandbox)]):
        if subprocess.run(command, check=False).returncode != 0:
            print(f"✗ Failed to configure Electron's Linux sandbox helper: {sandbox}")
            return False
    return True


_LINUX_PASSWORD_STORES = frozenset({"gnome-libsecret", "kwallet", "kwallet5", "kwallet6", "basic"})


def _detect_linux_password_store() -> str | None:
    """Detect the Chromium password-store backend for the current Linux session.

    Electron's safeStorage only reports encryption as available when Chromium
    selects the right keychain backend, and Chromium's own detection routinely
    fails under `hermes desktop` because the launcher environment doesn't look
    like a full desktop session. Probe order: KDE session env vars, GNOME
    Keyring's control socket, then a D-Bus ping of org.freedesktop.secrets
    (covers any Secret Service implementation, e.g. KeePassXC). Returns None
    when no keychain daemon is reachable.
    """
    kde_version = os.environ.get("KDE_SESSION_VERSION", "").strip()
    if kde_version == "6":
        return "kwallet6"
    if kde_version == "5":
        return "kwallet5"
    if kde_version:
        return "kwallet"
    if os.environ.get("KDE_FULL_SESSION"):
        return "kwallet"
    if os.environ.get("GNOME_KEYRING_CONTROL"):
        return "gnome-libsecret"
    try:
        result = subprocess.run(
            [
                "dbus-send", "--session", "--print-reply", "--reply-timeout=2000",
                "--dest=org.freedesktop.secrets",
                "/org/freedesktop/secrets",
                "org.freedesktop.DBus.Peer.Ping",
            ],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "gnome-libsecret"
    except Exception:
        pass
    return None


def _desktop_launch_options() -> tuple[list[str], str, str, str]:
    """Read `desktop.*` launch options from config.yaml.

    Returns ``(electron_flags, disable_gpu, password_store, ozone_hint)`` where
    ``electron_flags`` is a list of extra Electron CLI flags, ``disable_gpu``
    is one of "auto"/"1"/"0" (normalized for the HERMES_DESKTOP_DISABLE_GPU
    env var the Electron app reads), ``password_store`` is "auto" or one
    of the Chromium password-store backends (unknown values normalize to
    "auto"), and ``ozone_hint`` is one of "auto"/"x11"/"wayland" (normalized
    for ``ELECTRON_OZONE_PLATFORM_HINT``). Best-effort: any config error
    yields the safe defaults ``([], "auto", "auto", "auto")`` so a malformed
    config never blocks the launch.
    """
    flags: list[str] = []
    disable_gpu = "auto"
    password_store = "auto"
    ozone_hint = "auto"
    try:
        from hermes_cli.config import load_config

        desktop_cfg = (load_config() or {}).get("desktop") or {}
    except Exception:
        return flags, disable_gpu, password_store, ozone_hint

    raw_flags = desktop_cfg.get("electron_flags")
    if isinstance(raw_flags, str):
        flags = shlex.split(raw_flags, posix=(os.name != "nt"))
    elif isinstance(raw_flags, (list, tuple)):
        flags = [str(f) for f in raw_flags if str(f).strip()]

    raw_gpu = desktop_cfg.get("disable_gpu", "auto")
    if isinstance(raw_gpu, bool):
        disable_gpu = "1" if raw_gpu else "0"
    elif isinstance(raw_gpu, str):
        low = raw_gpu.strip().lower()
        if low in ("1", "true", "yes", "on"):
            disable_gpu = "1"
        elif low in ("0", "false", "no", "off"):
            disable_gpu = "0"
        else:
            disable_gpu = "auto"

    raw_store = desktop_cfg.get("password_store", "auto")
    if isinstance(raw_store, str):
        low_store = raw_store.strip().lower()
        if low_store in _LINUX_PASSWORD_STORES:
            password_store = low_store

    raw_ozone = desktop_cfg.get("ozone_platform_hint", "auto")
    if isinstance(raw_ozone, str):
        low_ozone = raw_ozone.strip().lower()
        if low_ozone in ("auto", "x11", "wayland"):
            ozone_hint = low_ozone
    return flags, disable_gpu, password_store, ozone_hint


def _register_linux_desktop_entry() -> None:
    """Install the XDG desktop entry for Hermes Desktop (Linux only, best-effort).

    Gives the Electron app a launcher presence: a menu item and an icon.
    ``Exec`` and ``Icon`` are absolute, so the entry works outside a login
    shell. ``hermes uninstall --gui`` removes it.
    """
    try:
        from hermes_cli.linux_desktop_entry import install_desktop_entry, is_supported

        if not is_supported():
            return
        entry = install_desktop_entry(PROJECT_ROOT)
        if entry:
            print(f"✓ Desktop launcher entry installed: {entry}")
    except Exception as exc:  # never block a launch on launcher plumbing
        print(f"⚠ Could not install the desktop launcher entry: {exc}")


def cmd_gui(args: argparse.Namespace):
    """Build and launch the native Electron desktop GUI."""
    desktop_dir = PROJECT_ROOT / "apps" / "desktop"
    if not (desktop_dir / "package.json").exists():
        print(f"Desktop GUI source not found at: {desktop_dir}")
        sys.exit(1)
    
    desktop_web_dir = PROJECT_ROOT / "apps" / "desktop" / "web"
    if not (desktop_web_dir / "package.json").exists():
        print(f"Desktop WEB source not found at: {desktop_web_dir}")
        sys.exit(1)

    try:
        from hermes_logging import setup_logging as _setup_logging_gui
        _setup_logging_gui(mode="gui")
    except Exception:
        pass

    from hermes_constants import with_hermes_node_path

    # with_hermes_node_path() copies os.environ when called with no arg.
    env = with_hermes_node_path()
    if getattr(args, "fake_boot", False):
        env["HERMES_DESKTOP_BOOT_FAKE"] = "1"
    if getattr(args, "ignore_existing", False):
        env["HERMES_DESKTOP_IGNORE_EXISTING"] = "1"
    if getattr(args, "hermes_root", None):
        env["HERMES_DESKTOP_HERMES_ROOT"] = str(Path(args.hermes_root).expanduser().resolve())
    if getattr(args, "cwd", None):
        env["HERMES_DESKTOP_CWD"] = str(Path(args.cwd).expanduser().resolve())
    else:
        env["HERMES_DESKTOP_CWD"] = os.getcwd()

    # Desktop launch options from config.yaml (`desktop.electron_flags`,
    # `desktop.disable_gpu`, `desktop.ozone_platform_hint`). The GPU policy
    # and ozone hint are bridged to env vars the Electron/Chromium process
    # already reads; an explicit env var still wins over config so
    # `HERMES_DESKTOP_DISABLE_GPU=... hermes desktop` and
    # `ELECTRON_OZONE_PLATFORM_HINT=... hermes desktop` keep working.
    config_electron_flags, config_disable_gpu, config_password_store, config_ozone_hint = (
        _desktop_launch_options()
    )
    if config_disable_gpu != "auto" and "HERMES_DESKTOP_DISABLE_GPU" not in os.environ:
        env["HERMES_DESKTOP_DISABLE_GPU"] = config_disable_gpu
    if config_ozone_hint != "auto" and "ELECTRON_OZONE_PLATFORM_HINT" not in os.environ:
        env["ELECTRON_OZONE_PLATFORM_HINT"] = config_ozone_hint

    # Linux keychain backend for safeStorage (`desktop.password_store`).
    # Chromium needs the --password-store switch to pick the right keychain;
    # without it safeStorage.isEncryptionAvailable() is often false and the
    # desktop app refuses to persist remote gateway tokens. Config wins over
    # detection; an explicit env var wins over both so
    # `HERMES_DESKTOP_PASSWORD_STORE=... hermes desktop` keeps working.
    if sys.platform == "linux" and "HERMES_DESKTOP_PASSWORD_STORE" not in os.environ:
        password_store = (
            config_password_store
            if config_password_store != "auto"
            else _detect_linux_password_store()
        )
        if password_store:
            env["HERMES_DESKTOP_PASSWORD_STORE"] = password_store

    source_mode = getattr(args, "source", False)
    skip_build = getattr(args, "skip_build", False)
    force_build = getattr(args, "force_build", False)

    # macOS-only one-shot: create a self-signed code-signing identity so TCC
    # grants survive rebuilds, then exit without building/launching.
    if getattr(args, "setup_tcc_identity", False):
        identity = getattr(args, "identity", None) or "Hermes Local Signing"
        ok = _desktop_macos_setup_tcc_identity(identity)
        sys.exit(0 if ok else 1)

    packaged_executable = _desktop_packaged_executable(desktop_dir)

    if source_mode or not skip_build:
        npm = _resolve_node_runtime_npm()
        if not npm:
            print("Desktop GUI requires Node.js/npm, but npm was not found on PATH.")
            print("Install Node.js, then run:  hermes gui")
            sys.exit(1)
    else:
        npm = None

    if skip_build:
        if source_mode:
            if not _desktop_dist_exists(desktop_dir):
                print(f"✗ --skip-build --source was passed but no desktop dist found at: {desktop_dir / 'dist'}")
                print("  Pre-build first:  cd apps/desktop && npm run build")
                print("  Or drop --skip-build to install dependencies and build automatically.")
                sys.exit(1)
            if not (_electron_dir(PROJECT_ROOT) / "package.json").exists():
                print("✗ --skip-build --source requires existing desktop workspace dependencies.")
                print(f"  Install first:  cd {PROJECT_ROOT} && npm ci")
                print("  Or drop --skip-build to install dependencies and build automatically.")
                sys.exit(1)
            print(f"→ Skipping desktop source build (--skip-build --source); using dist at {desktop_dir / 'dist'}")
        elif packaged_executable is None:
            print(f"✗ --skip-build was passed but no packaged desktop app was found at: {desktop_dir / 'release'}")
            print("  Pre-build first:  cd apps/desktop && npm run pack")
            print("  Or drop --skip-build to package automatically.")
            sys.exit(1)
        else:
            print(f"→ Skipping desktop package build (--skip-build); using {packaged_executable}")
    else:
        # Check the content-hash stamp before doing any build work.
        # If the source tree hasn't changed since the last successful build,
        # skip the npm install + build entirely (saves a ton of useless work).
        # --force-build overrides the stamp and always rebuilds.
        build_needed = force_build or _desktop_build_needed(
            desktop_dir, PROJECT_ROOT, source_mode=source_mode
        )
        if not build_needed:
            build_label = "source build" if source_mode else "packaged app"
            print(f"✓ Desktop {build_label} is up to date (content stamp matches)")
        else:
            print("→ Installing desktop workspace dependencies...")
            # Put the Hermes-managed Node on PATH so npm's child scripts (which
            # shell out to bare `node`, e.g. electron-winstaller's
            # select-7z-arch.js) resolve it even when the parent PATH is
            # stripped — the desktop updater chain (Desktop → hermes-setup →
            # hermes update) loses shell PATH customizations. Wrapping the
            # NixOS build env keeps its PYTHON hint while restoring managed Node
            # ahead of a bare PATH (same idiom as the `hermes update` path).
            nixos_env = with_hermes_node_path(_nixos_build_env())
            install_result = _run_npm_install_deterministic(npm, PROJECT_ROOT, capture_output=False, env=nixos_env)
            if install_result.returncode != 0:
                if not _electron_pkg_staged_missing_dist(PROJECT_ROOT):
                    print("✗ Desktop dependency install failed")
                    print(f"  Run manually:  cd {PROJECT_ROOT} && npm ci")
                    sys.exit(install_result.returncode or 1)
                repaired = _try_redownload_electron_dist(PROJECT_ROOT, env)
                if repaired:
                    print("  ⚠ Dependency install failed with a missing Electron dist; "
                          "repopulated it and continuing.")
                else:
                    print("  ⚠ Dependency install failed with a missing Electron dist; "
                          "continuing to the build so electron-builder can attempt "
                          "the Electron fetch itself.")

            build_label = "source build" if source_mode else "packaged app"
            print(f"→ Building desktop {build_label}...")
            build_script = "build" if source_mode else "pack"
            if _force_adhoc_macos_signing(env, source_mode=source_mode):
                print("  → No Developer ID configured; ad-hoc signing this local rebuild "
                      "(CSC_IDENTITY_AUTO_DISCOVERY=false)")
            npm_build_env = _npm_lifecycle_env(env)
            if not source_mode:
                # A running desktop instance launched from release/win-unpacked
                # holds Hermes.exe locked on Windows, so the pack can't replace
                # it ("Access is denied" / ERR_ELECTRON_BUILDER_CANNOT_EXECUTE).
                # Stop it first so the rebuild — including the installer's
                # headless --update rebuild — succeeds instead of failing cryptically.
                stopped = _stop_desktop_processes_locking_build(desktop_dir)
                if stopped:
                    print(f"  ⚠ Stopped running desktop app to free the build output (pid {', '.join(map(str, stopped))})")
            build_result = subprocess.run(
                [npm, "run", build_script], cwd=desktop_dir, env=npm_build_env, check=False
            )
            if (
                build_result.returncode != 0
                and not source_mode
                and _desktop_packaged_executable(desktop_dir) is None
            ):
                # Corrupt cached Electron zip → partial unpack → ENOENT on rename.
                # stdlib zipfile won't catch the common concat-junk case, so purge
                # and retry once; @electron/get SHASUM is the real gate.
                #
                # Gate on a MISSING packaged executable: that is the signature of
                # the corrupt-download class this recovery exists for. A late
                # failure such as macOS code signing leaves the executable in
                # place — redownloading Electron can't repair it, so the purge +
                # retry would only add another slow, identical failure (#40187).
                purged: list[Path] = []
                restored = False
                if not _electron_dist_ok(PROJECT_ROOT):
                    purged = _purge_electron_build_cache(desktop_dir)
                    restored = _redownload_electron_dist(PROJECT_ROOT, env)
                if restored:
                    print("  ⚠ Desktop build failed; refreshed the Electron download and retrying once...")
                    for p in purged:
                        print(f"    - {p}")
                    # The purge can't remove a win-unpacked tree whose Hermes.exe
                    # is still locked by a running instance; stop it before retry.
                    _stop_desktop_processes_locking_build(desktop_dir)
                    build_result = subprocess.run(
                        [npm, "run", build_script], cwd=desktop_dir, env=npm_build_env, check=False
                    )
            if (
                build_result.returncode != 0
                and not source_mode
                and not env.get("ELECTRON_MIRROR")
                and _desktop_packaged_executable(desktop_dir) is None
            ):
                print("  ⚠ Desktop build still failing; the Electron download from "
                      "GitHub looks blocked. Re-downloading via a public mirror "
                      "(npmmirror.com)... (set ELECTRON_MIRROR to use another mirror)")
                mirror = _ELECTRON_FALLBACK_MIRROR
                mirror_env = dict(npm_build_env)
                mirror_env["ELECTRON_MIRROR"] = mirror
                if not _electron_dist_ok(PROJECT_ROOT):
                    _redownload_electron_dist(PROJECT_ROOT, env, mirror=mirror)
                _stop_desktop_processes_locking_build(desktop_dir)
                build_result = subprocess.run([npm, "run", build_script], cwd=desktop_dir, env=mirror_env, check=False)
            if build_result.returncode != 0:
                print("✗ Desktop GUI build failed")
                print(f"  Run manually:  cd apps/desktop && npm run {build_script}")
                if sys.platform == "win32":
                    print("  If this says \"Access is denied\" on Hermes.exe, close any")
                    print("  running Hermes desktop window and retry.")
                print("  If the log shows Electron download retries, rebuild via a mirror:")
                print("    ELECTRON_MIRROR=<mirror-base-url> hermes desktop --force-build")
                sys.exit(build_result.returncode or 1)
            packaged_executable = _desktop_packaged_executable(desktop_dir)
            if not source_mode:
                # Locally-built apps are ad-hoc signed; make them relaunchable after
                # an in-place self-update (otherwise macOS reports "Hermes is
                # damaged"). No-op on non-macOS and on real-identity builds.
                _desktop_macos_relaunchable_fixup(desktop_dir)

                # Windows integrity gate (#69179): never declare the rebuild a
                # success on a Hermes.exe Windows cannot load (truncated PE from
                # a corrupt cached Electron zip, wrong-arch tree, interrupted
                # rcedit rewrite). Roll back to the .bak tree preserved by
                # before-pack.mjs when possible, then fail loudly so the
                # updater's retry-once rebuilds from a fresh Electron download
                # instead of silently shipping the broken exe.
                verified_executable, rolled_back = _ensure_desktop_exe_launchable(
                    desktop_dir, packaged_executable
                )
                if packaged_executable is not None and (
                    rolled_back or verified_executable is None
                ):
                    sys.exit(1)
                packaged_executable = verified_executable

            # Build succeeded — write the stamp so next run can skip
            _write_desktop_build_stamp(PROJECT_ROOT, source_mode=source_mode)

    # Linux: register the app in the desktop launcher, so Hermes shows up
    # in the application menu with its icon. Best-effort and idempotent.
    # A failure must never stop the app from launching.
    _register_linux_desktop_entry()

    # --build-only: produce the artifact but do NOT launch. The installer's
    # --update flow drives the rebuild headlessly and then launches the desktop
    # itself (detached, after the old exe has exited), so the launch must NOT
    # happen here — it would block the installer and, on Windows, the old exe
    # is still being replaced. Verify the expected artifact exists so a silent
    # "built nothing" can't slip past, then return success.
    if getattr(args, "build_only", False):
        if source_mode:
            if not _desktop_dist_exists(desktop_dir):
                print(f"✗ --build-only --source produced no dist at: {desktop_dir / 'dist'}")
                sys.exit(1)
            print(f"✓ Desktop source build ready at {desktop_dir / 'dist'} (not launching; --build-only)")
        elif packaged_executable is None:
            print(f"✗ --build-only produced no launchable app at: {desktop_dir / 'release'}")
            print("  Expected an unpacked Electron app for the current OS.")
            sys.exit(1)
        else:
            print(f"✓ Desktop packaged app ready: {packaged_executable} (not launching; --build-only)")
        return

    if source_mode:
        print("→ Launching Hermes Desktop from source build...")
        launch_result = subprocess.run([npm, "exec", "--", "electron", "."], cwd=desktop_dir, env=env, check=False)
        sys.exit(launch_result.returncode)

    if packaged_executable is None:
        print(f"✗ Desktop package build completed but no launchable app was found at: {desktop_dir / 'release'}")
        print("  Expected an unpacked Electron app for the current OS.")
        sys.exit(1)

    launch_command = [str(packaged_executable)]
    if not _desktop_linux_sandbox_fixup(packaged_executable):
        if _desktop_linux_needs_no_sandbox() and _desktop_linux_sandbox_helper_is_regular_file(packaged_executable):
            print("⚠ Falling back to --no-sandbox because this Linux host restricts unprivileged user namespaces and the Electron sandbox helper could not be configured.")
            launch_command.append("--no-sandbox")
        else:
            sys.exit(1)

    launch_command.extend(config_electron_flags)
    print(f"→ Launching packaged Hermes Desktop: {' '.join(launch_command)}")
    launch_result = subprocess.run(launch_command, cwd=desktop_dir, env=env, check=False)
    sys.exit(launch_result.returncode)



# Dashboard process-hygiene helpers live in hermes_cli/dashboard_procs.py
# (main.py decomposition, mechanical move). Re-exported lazily through the
# module-level __getattr__ above so callers and test monkeypatches on
# hermes_cli.main.<name> keep resolving unchanged.

def _find_stale_dashboard_pids(
    *,
    exclude_pids: set[int] | None = None,
) -> list[int]:
    """Return PIDs of stale ``dashboard``/``serve`` processes for update cleanup."""
    return [pid for pid, _cmd in _self()._scan_dashboard_processes(exclude_pids=exclude_pids)]


def _parse_dashboard_runtime(command: str) -> tuple[str, str, int] | None:
    """Best-effort parse of a dashboard/server cmdline into mode, host, and port."""
    mode = None
    if any(
        pattern in command
        for pattern in (
            "hermes dashboard",
            "hermes_cli.main dashboard",
            "hermes_cli/main.py dashboard",
        )
    ):
        mode = "dashboard"
    elif any(
        pattern in command
        for pattern in (
            "hermes serve",
            "hermes_cli.main serve",
            "hermes_cli/main.py serve",
        )
    ):
        mode = "serve"
    if mode is None:
        return None

    port = 9119
    host = "127.0.0.1"

    port_match = re.search(r"(?:^|\s)--port(?:=|\s+)(\d+)", command)
    if port_match:
        try:
            port = int(port_match.group(1))
        except ValueError:
            return None

    host_match = re.search(r"(?:^|\s)--host(?:=|\s+)(\"[^\"]+\"|'[^']+'|\S+)", command)
    if host_match:
        host = host_match.group(1).strip("\"'") or "127.0.0.1"

    return mode, host, port


def _dashboard_probe_host(host: str | None) -> str:
    """Map wildcard binds to a loopback address suitable for local probing."""
    normalized = (host or "127.0.0.1").strip().strip("[]")
    if normalized in {"", "0.0.0.0", "::"}:
        return "127.0.0.1"
    return normalized


_DASHBOARD_SYSTEMD_UNIT = "hermes-dashboard.service"


def _restart_managed_dashboard_service(
    reason: str,
    unit: str = _DASHBOARD_SYSTEMD_UNIT,
) -> bool:
    """Restart a systemd-managed dashboard instead of raw-killing its PID.

    Returns True when a dashboard unit was found and handled (successfully or
    with a printed actionable failure).  Returning True deliberately prevents
    the caller from falling back to ``os.kill``: systemd treats a direct
    SIGTERM of the service's main PID as a clean stop, so ``Restart=on-failure``
    will not bring the dashboard back.
    """
    if sys.platform == "win32":
        return False

    def _systemctl(*args: str, timeout: int = 10) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["systemctl", *args],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )

    # Probe the user manager first: Hermes installs Linux services in the
    # user's systemd scope by default.  Only fall back to the system manager
    # when the unit is not present there, preserving root/system deployments.
    # Crucially, keep the selected scope for *all* probes and the restart — a
    # user unit must never be restarted through the system manager (or raw-killed).
    scope: tuple[str, ...] | None = None
    listed: subprocess.CompletedProcess | None = None
    for candidate in (("--user",), ()):
        try:
            result = _systemctl(
                *candidate, "list-unit-files", unit, "--no-legend", "--no-pager"
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode != 0:
            continue
        unit_rows = (result.stdout or "").splitlines()
        if any(row.split()[0:1] == [unit] for row in unit_rows if row.split()):
            scope = candidate
            listed = result
            break

    if scope is None or listed is None:
        return False

    try:
        active = _systemctl(*scope, "is-active", unit)
        enabled = _systemctl(*scope, "is-enabled", unit)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False

    active_state = (active.stdout or "").strip()
    enabled_state = (enabled.stdout or "").strip()
    if active_state != "active" and enabled_state not in {
        "enabled",
        "enabled-runtime",
        "linked",
        "linked-runtime",
        "static",
        "generated",
    }:
        return False

    print()
    print(f"⟲ Restarting managed dashboard service ({reason})")

    scope_label = "systemctl --user" if scope else "sudo systemctl"
    restart = ("systemctl", *scope, "restart", unit)
    commands = [restart]
    if not scope:
        # System units may require privilege escalation; user units must use
        # the user manager directly and never prompt for sudo.
        commands.append(("sudo", "-n", "systemctl", "restart", unit))

    errors: list[str] = []
    for command in commands:
        try:
            result = subprocess.run(
                list(command),
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=60,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            errors.append(f"{' '.join(command)}: {e}")
            continue
        if result.returncode == 0:
            print(f"    ✓ restarted {unit}")
            return True
        errors.append(
            f"{' '.join(command)}: {(result.stderr or result.stdout or '').strip()}"
        )

    print(f"    ✗ failed to restart {unit}")
    for err in errors:
        if err.strip():
            print(f"      {err}")
    print(
        "  Dashboard is managed by systemd; not raw-killing its PID because "
        "systemd would treat that as a clean stop."
    )
    print(f"  Restart manually: {scope_label} restart {unit}")
    return True


def _get_systemd_service_for_pid(pid: int) -> str | None:
    """If *pid* belongs to a systemd service unit, return the unit name.

    Reads ``/proc/<pid>/cgroup`` and extracts the service name (e.g.
    ``hermes-serve.service``).  Returns ``None`` when the PID is not
    part of a systemd service, when the file is unreadable, or on
    non-Linux platforms.
    """
    try:
        cgroup_path = Path(f"/proc/{pid}/cgroup")
        if not cgroup_path.is_file():
            return None
        text = cgroup_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            # Format: 0::/system.slice/hermes-serve.service
            #         0::/user.slice/user-1000.slice/session-42.scope
            parts = line.split("::", 1)
            if len(parts) != 2:
                continue
            cg_path = parts[1]
            if cg_path.endswith(".service"):
                svc_name = cg_path.rsplit("/", 1)[-1]
                if svc_name:
                    return svc_name
    except (OSError, PermissionError):
        pass
    return None


def _extract_scope_from_cgroup(cgroup_entry: str) -> str | None:
    """Extract the systemd scope (``user`` or ``system``) from a cgroup path.

    The cgroup path format is ``/system.slice/<name>.service`` for system
    services and ``/user.slice/user-<uid>.slice/<name>.service`` for user
    services.  Returns ``None`` when the scope cannot be determined.
    """
    if "/system.slice/" in cgroup_entry:
        return "system"
    if "/user.slice/" in cgroup_entry:
        return "user"
    return None


def _get_pid_cgroup_path(pid: int) -> str | None:
    """Return the cgroup path from ``/proc/<pid>/cgroup``, or ``None``.

    Only the unified (``0::``) hierarchy cgroup entry is examined.
    """
    try:
        cgroup_path = Path(f"/proc/{pid}/cgroup")
        if not cgroup_path.is_file():
            return None
        text = cgroup_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            parts = line.split("::", 1)
            if len(parts) == 2:
                return parts[1]
    except (OSError, PermissionError):
        pass
    return None


def _try_restart_systemd_service(svc_name: str, cgroup_path: str | None = None) -> bool:
    """Attempt to restart *svc_name* via systemctl.

    Uses ``systemctl --user`` for user-scope services and ``systemctl``
    for system-scope services.  Returns ``True`` on success.
    """
    scope = _extract_scope_from_cgroup(cgroup_path) if cgroup_path else None
    if scope == "user":
        cmd = ["systemctl", "--user", "restart", svc_name]
    elif scope == "system":
        cmd = ["systemctl", "restart", svc_name]
    else:
        # Unknown scope — try system first, then user
        cmd = None
        for candidate in (
            ["systemctl", "restart", svc_name],
            ["systemctl", "--user", "restart", svc_name],
        ):
            try:
                r = subprocess.run(
                    candidate,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=15,
                )
                if r.returncode == 0:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
        return False

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=15,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _dashboard_cmdline_for_pid(pid: int) -> list[str] | None:
    """Return the exact argv of a running process, when recoverable.

    Linux: reads ``/proc/<pid>/cmdline`` (NUL-separated, lossless).
    macOS: falls back to ``ps -o command=`` + shlex (best effort — quoting
    is reconstructed, but hermes launch commands don't embed exotic args).
    Windows: returns ``None``; taskkill /F gives no graceful window and the
    desktop app manages its own backend there.
    """
    if sys.platform == "win32":
        return None
    try:
        cmdline_path = f"/proc/{pid}/cmdline"
        if os.path.exists(cmdline_path):
            with open(cmdline_path, "rb") as f:
                raw = f.read()
            argv = [
                part.decode("utf-8", errors="replace")
                for part in raw.split(b"\x00")
                if part
            ]
            return argv or None
        # macOS (no /proc): best-effort via ps.
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        if result.returncode != 0:
            return None
        command = (result.stdout or "").strip()
        if not command:
            return None
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = command.split()
        return argv or None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _respawn_dashboard_processes(commands: list[list[str]]) -> list[list[str]]:
    """Best-effort respawn of manually-started dashboards after ``hermes update``.

    Spawns each recovered argv detached (new session, output to the profile's
    ``logs/dashboard-restart.log``).  Returns the commands that failed to
    spawn; the caller prints the manual hint for those.

    Callers must pre-filter via ``_filter_dashboard_respawn_candidates`` so
    Desktop ``serve|dashboard --port 0`` backends are not replayed and
    duplicates are capped per profile (#78821).
    """
    from hermes_constants import get_hermes_home

    respawned: list[list[str]] = []
    failed: list[tuple[list[str], str]] = []
    log_path = get_hermes_home() / "logs" / "dashboard-restart.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    for command in commands:
        try:
            # Keep restarted dashboards headless; reopening a browser after a
            # background update is noisy and fails in SSH/headless sessions.
            if "dashboard" in command and "--no-open" not in command:
                command = [*command, "--no-open"]
            with open(log_path, "ab") as log_f:
                subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            respawned.append(command)
        except (OSError, ValueError) as exc:
            failed.append((command, str(exc)))

    for command in respawned:
        print(f"    ✓ restarted: {shlex.join(command)}")
    for command, err_msg in failed:
        print(f"    ✗ failed to restart ({shlex.join(command)}): {err_msg}")
    return [command for command, _ in failed]


# Back-compat alias: some tests and any external callers may import the old
# warn-only name.  The new behaviour (kill stale processes) replaces it.
# Resolved lazily via _LAZY_COMMAND_ALIASES near the module __getattr__.


# =========================================================================
# Fork detection and upstream management for `hermes update`
# =========================================================================


def _load_installable_optional_extras(group: str = "all") -> list[str]:
    """Return optional extras referenced by a dependency group.

    ``group`` is usually ``all`` (desktop/server broad install) or
    ``termux-all`` (Termux-compatible broad install).
    """
    try:
        import tomllib

        with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
    except Exception:
        return []

    optional_deps = project.get("optional-dependencies", {})
    if not isinstance(optional_deps, dict):
        return []

    refs = optional_deps.get(group, [])
    referenced: list[str] = []
    for ref in refs:
        if "[" in ref and "]" in ref:
            name = ref.split("[", 1)[1].split("]", 1)[0]
            if name in optional_deps:
                referenced.append(name)

    return referenced


# Install-scoped breadcrumbs live next to the venv (not under $HERMES_HOME)
# because the venv is shared across profiles.
#
# ``.update-incomplete`` — generic core ``.[all]`` install was interrupted.
# Cleared only after a confirmed full dependency reinstall/recovery.
#
# ``.lazy-refresh-incomplete`` — lazy-backend refresh phase may have corrupted
# packages. Cleared only after import-probe repair confirms healthy (not when
# probes are unavailable/indeterminate). Narrow lazy probes must NEVER clear
# the generic core marker (#58004 review).
def _update_marker_path() -> Path:
    return PROJECT_ROOT / ".update-incomplete"


def _lazy_refresh_marker_path() -> Path:
    return PROJECT_ROOT / ".lazy-refresh-incomplete"


def _pytest_owns_live_checkout(root: Path) -> bool:
    """True when running under pytest AND ``root`` is this checkout itself.

    Tests that drive update/recovery without sandboxing ``PROJECT_ROOT``
    must neither litter the live repo root with recovery breadcrumbs
    (a leftover ``.lazy-refresh-incomplete`` / ``.update-incomplete``
    false-arms recovery on the developer's next real launch) nor run a real
    reinstall against the executing venv. Sandboxed tests point at a
    tmp_path and are unaffected (same posture as
    ``managed_scope._under_pytest``)."""
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        and root == Path(__file__).resolve().parent.parent
    )


def _clear_marker_file(path: Path, *, label: str) -> None:
    """Remove an update-recovery breadcrumb. Never raises."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Could not clear %s marker: %s", label, exc)


def _clear_update_incomplete_marker() -> None:
    """Remove the interrupted core-install breadcrumb. Never raises."""
    _clear_marker_file(_update_marker_path(), label="update-incomplete")


def _clear_lazy_refresh_incomplete_marker() -> None:
    """Remove the interrupted lazy-refresh breadcrumb. Never raises."""
    _clear_marker_file(_lazy_refresh_marker_path(), label="lazy-refresh-incomplete")


def _recover_from_interrupted_install() -> None:
    """Finish update work left half-done by a prior ``hermes update``.

    Handles two independent breadcrumbs:

    - ``.update-incomplete`` — core ``.[all]`` install interrupted. Recovers
      via full quarantined reinstall. Never cleared by the narrow lazy-refresh
      import probes alone.
    - ``.lazy-refresh-incomplete`` — lazy-backend refresh may have corrupted
      packages. Recovers via package-only import probes; cleared only when
      probes confirm healthy/repaired (indeterminate keeps the marker).

    Never raises: a recovery failure must not block launch.  If it can't
    self-heal it prints the manual command and leaves the relevant marker so
    the next launch tries again.

    Concurrency: markers live next to the shared venv, so a gateway start
    plus a CLI launch (or two profiles starting at once) can both see them.
    An ``O_EXCL`` lockfile ensures only one process runs recovery; the
    others skip and let the winner clear markers.

    Output: everything — our status lines AND the streamed pip/uv install
    (which inherits fd 1) — is routed to stderr.  Launches whose stdout is a
    protocol stream (``hermes acp`` speaks JSON-RPC on stdout) must never get
    install noise on stdout.
    """
    if _pytest_owns_live_checkout(PROJECT_ROOT):
        return
    core_marker = _update_marker_path().exists()
    lazy_marker = _lazy_refresh_marker_path().exists()
    if not core_marker and not lazy_marker:
        return

    # Skip in managed/Docker installs and on PyPI installs with no git checkout:
    # those don't run the source-tree update path, so a stray marker is not ours
    # to act on. Just clear it.
    if not (PROJECT_ROOT / "pyproject.toml").is_file():
        _clear_update_incomplete_marker()
        _clear_lazy_refresh_incomplete_marker()
        return

    # Single-flight guard: atomically claim the recovery lock. If another
    # process holds it, skip — it is running the same reinstall into the same
    # shared venv right now. A crashed holder leaves a stale lock; break it
    # after an hour (well past any realistic install) so recovery can't be
    # wedged forever.
    lock_path = PROJECT_ROOT / ".update-incomplete.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError:
        try:
            if _time.time() - lock_path.stat().st_mtime > 3600:
                lock_path.unlink()
        except OSError:
            pass
        return
    except OSError as exc:
        # Couldn't create the lock (read-only fs, perms). Proceed unlocked —
        # the install itself will surface the real problem.
        logger.debug("Could not create install-recovery lock: %s", exc)

    saved_stdout_fd = None
    saved_sys_stdout = sys.stdout
    try:
        # Route Python-level prints AND subprocess-inherited fd 1 to stderr
        # for the duration of recovery (see docstring: ACP stdout safety).
        try:
            saved_stdout_fd = os.dup(1)
            os.dup2(2, 1)
        except OSError:
            saved_stdout_fd = None
        sys.stdout = sys.stderr

        if lazy_marker:
            _recover_lazy_refresh_marker_locked()

        if _update_marker_path().exists():
            _recover_core_update_marker_locked()
    finally:
        sys.stdout = saved_sys_stdout
        if saved_stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, 1)
                os.close(saved_stdout_fd)
            except OSError:
                pass
        try:
            lock_path.unlink()
        except OSError:
            pass


def _recover_lazy_refresh_marker_locked() -> None:
    """Heal ``.lazy-refresh-incomplete`` via confirmed import-probe repair."""
    print(
        "⚠ A previous lazy-backend refresh may have left the venv unhealthy — "
        "running import-based package repair..."
    )
    install_prefix, install_env = _default_venv_install_target()
    status = _repair_venv_via_import_probes(install_prefix, env=install_env)
    if status in ("healthy", "repaired"):
        _clear_lazy_refresh_incomplete_marker()
        print("✓ Lazy-refresh venv recovery confirmed — install is healthy again.")
        return
    if status == "indeterminate":
        print(
            "  ⚠ Import probes unavailable — cannot confirm venv health. "
            "Leaving `.lazy-refresh-incomplete` for the next launch."
        )
    else:
        print(
            "  ⚠ Lazy-refresh package repair incomplete. "
            "Leaving `.lazy-refresh-incomplete` for the next launch."
        )
        print("  Recover manually with:")
        all_specs = _lazy_refresh_repair_specs(
            sorted(set(_LAZY_REFRESH_REPAIR_PACKAGES.values()))
        )
        print(
            f"    {' '.join(install_prefix)} install --force-reinstall "
            + " ".join(shlex.quote(s) for s in all_specs)
        )


def _recover_core_update_marker_locked() -> None:
    """Heal ``.update-incomplete`` via full ``.[all]`` reinstall only.

    Narrow lazy-refresh import probes are not sufficient proof that a generic
    interrupted core install finished — a missing dep outside that probe set
    would otherwise look healthy and clear the breadcrumb too early.
    """
    print(
        "⚠ A previous `hermes update` was interrupted mid-install — "
        "finishing dependency installation now..."
    )

    # Windows: a normal ``hermes.exe`` launch always has the launcher as an
    # ancestor. Full editable reinstall uses quarantine so the live shim can
    # still be replaced. Package-only import repair may help as first aid but
    # must NEVER clear this core marker on its own (#58004 review).
    self_locked = _windows_running_hermes_launcher_locked()
    if self_locked:
        install_prefix, install_env = _default_venv_install_target()
        print(
            "  → Running from hermes.exe; applying package-only first aid, "
            "then quarantined full reinstall (core marker stays until that "
            "succeeds)..."
        )
        _repair_venv_via_import_probes(install_prefix, env=install_env)

    try:
        from hermes_cli import _install_repair as _ir

        # ensure_uv bootstraps the installer itself when missing (the early
        # pass's stdlib-only lookup cannot); keeping it here means the late
        # path still self-heals a venv whose uv vanished mid-update.
        from hermes_cli.managed_uv import ensure_uv

        ensure_uv()

        # Delegate the install itself to the shared stdlib executor so both
        # this late path and the pre-import early pass run exactly the same
        # reinstall.  Called inside the same stdout→stderr redirect already
        # established by _recover_from_interrupted_install, so
        # run_core_install's own redirect nests harmlessly.
        _ir.run_core_install(PROJECT_ROOT)

        _clear_update_incomplete_marker()
        print("✓ Dependency installation recovered — your install is healthy again.")
    except Exception as exc:
        # Leave the marker in place so the next launch retries. Give the user
        # the exact manual recovery command in the meantime.
        logger.debug("Interrupted-install recovery failed: %s", exc)
        print("✗ Could not auto-recover the interrupted install.")
        if self_locked:
            print(
                "  Hermes is still running from the launcher that needs "
                "replacing. Close other Hermes windows, restart from a "
                "different terminal, then run:"
            )
            print(f'    cd /d "{PROJECT_ROOT}"')
            print(
                f'    "{sys.executable}" -m pip install -e ".[all]"'
            )
        else:
            print("  Recover manually with:")
            print(f"    cd {PROJECT_ROOT}")
            print(f"    {sys.executable} -m ensurepip --upgrade")
            print(f"    {sys.executable} -m pip install -e '.[all]'")


def _norm_exe_path(path) -> str:
    """Case-folded resolved path, for comparing executables on Windows."""
    try:
        return str(Path(path).resolve()).lower()
    except OSError:
        return str(path).lower()


def _windows_shim_in_process_chain() -> Path | None:
    """The venv console shim this process runs from or under, if any.

    ``venv\\Scripts\\hermes.exe`` is a launcher that runs the interpreter with
    the shim itself as its script, and that keeps the shim open — without
    ``FILE_SHARE_DELETE`` — for the whole process lifetime. So every
    ``hermes ...`` command holds its own shim, and an editable install run
    from one can never rewrite it (#88838, #89599).

    Two independent probes, because either can come up empty. Process
    ancestry finds the launcher when it is a separate parent process, but
    needs psutil. This process's own launch paths (``sys.argv[0]``,
    ``__main__.__file__``, the module spec origin) cover the rest — the
    runpy/zipapp launch puts ``<shim>\\__main__.py`` there, which a plain
    argv[0] check misses.

    Candidates are intersected with the project venv's own shims, so a
    ``hermes.exe`` belonging to some other install never matches.
    """
    if not _is_windows():
        return None
    scripts_dir = _venv_scripts_dir()
    if scripts_dir is None:
        return None
    shims = {_norm_exe_path(shim): shim for shim in _hermes_exe_shims(scripts_dir)}
    if not shims:
        return None

    def _match(candidate) -> Path | None:
        path = Path(candidate)
        if path.name.lower() == "__main__.py":
            path = path.parent
        return shims.get(_norm_exe_path(path))

    candidates: list[str] = list(sys.argv[:1])
    main_mod = sys.modules.get("__main__")
    for attr in (getattr(main_mod, "__file__", None),
                 getattr(getattr(main_mod, "__spec__", None), "origin", None)):
        if attr:
            candidates.append(attr)
    for candidate in candidates:
        matched = _match(candidate)
        if matched is not None:
            return matched

    try:
        import psutil

        me = psutil.Process()
        for proc in [me] + list(me.parents()):
            try:
                matched = _match(proc.exe())
            except Exception:
                continue
            if matched is not None:
                return matched
    except Exception:
        return None
    return None


def _windows_running_hermes_launcher_locked() -> bool:
    """True when a venv ``hermes*.exe`` shim is this process or an ancestor.

    Best-effort: returns False when psutil is unavailable or inspection fails.
    """
    return _windows_shim_in_process_chain() is not None


# Set on the re-exec'd child so it can never spawn another one.
_UPDATE_REEXEC_ENV = "HERMES_UPDATE_REEXEC"


def _reexec_dependency_sync_off_windows_shim() -> bool:
    """Hand the dependency sync to the venv interpreter, off the console shim.

    Returns True when a child was spawned and the caller must exit at once,
    releasing the shim before the child reaches ``pip install -e .``. Returns
    False to continue the sync in-process.

    Called at the dependency-sync boundary, NOT at the top of the command —
    the same placement rule as the native-module deferral beside it, and for
    the same reason (#86735): a hand-off that fires before the fetch detaches
    every run, including the ``Already up to date!`` no-op that never touches
    the venv at all, and it takes the interactive prompts with it. By the time
    we reach here the code swap is done and every question — stash, branch
    switch, config migration — has already been asked and answered in the
    user's own console. Only the venv rewrite is left, and that is the single
    step that genuinely cannot run from inside the shim.

    ``venv\\Scripts\\hermes.exe`` is a launcher that runs the interpreter with
    the shim as its script and holds it open without ``FILE_SHARE_DELETE`` for
    the whole command, so the quarantine rename is refused and uv fails to
    replace it with os error 32 (#88838, #89599).

    A child is required, and waiting on it cannot work: this process holds the
    handle the child needs released, so a parent that waits deadlocks against
    the work it is waiting for. Windows has no exec to escape with either.
    The shell therefore returns while the install runs on; the child keeps the
    console and prints its own result, and ``--gateway`` writes the true exit
    code to ``.update_exit_code`` for the gateway watcher.

    The child re-runs ``hermes update``, so the whole remaining flow — the
    dependency sync and the node/web/lazy-refresh tail behind it — still
    happens exactly once. ``_UPDATE_REEXEC_ENV`` marks it so it cannot spawn
    another child, and so the "already up to date" early return does not
    swallow the sync it was spawned to perform (the checkout is current by
    now; that is the point).

    The caller has already written ``.update-incomplete``, so a child that
    dies mid-install is finished by the next launch's recovery instead of
    leaving a half-synced venv. Anything that stops the hand-off (no venv
    python, spawn refused) returns False and syncs in-process, where the
    pre-existing os-error-32 path and its marker recovery still apply.
    """
    if os.environ.get(_UPDATE_REEXEC_ENV) == "1":
        return False
    shim = _windows_shim_in_process_chain()
    if shim is None:
        return False

    from hermes_constants import venv_python_path

    python_exe = venv_python_path(shim.parent.parent, windows=True)
    cmd = [str(python_exe), "-m", "hermes_cli.main", *sys.argv[1:]]
    if python_exe.is_file():
        try:
            subprocess.Popen(
                cmd,
                env={**os.environ, _UPDATE_REEXEC_ENV: "1"},
                stdin=subprocess.DEVNULL,
            )
            print(
                f"→ Windows: {shim.name} cannot replace itself while it runs; "
                "finishing the dependency install under the venv Python."
            )
            print(
                "  The code update is already applied. The install continues "
                "below and this shell returns right away."
            )
            return True
        except OSError as exc:
            logger.debug("Dependency-sync hand-off via %s failed: %s", python_exe, exc)
        print(f"  ⚠ Could not hand the dependency install off {shim.name}.")
        print("    Continuing in-process; if it cannot replace the shim, run:")
        print(f"    {subprocess.list2cmdline(cmd)}")
    return False


def _default_venv_install_target() -> tuple[list[str], dict[str, str] | None]:
    """Return ``(install_cmd_prefix, env)`` for the project venv when possible."""
    try:
        from hermes_cli.managed_uv import ensure_uv

        uv_bin = ensure_uv()
    except Exception:
        uv_bin = None
    if uv_bin:
        from hermes_constants import project_venv_dir

        venv_dir = project_venv_dir(PROJECT_ROOT) or PROJECT_ROOT / "venv"
        env = {**os.environ, "VIRTUAL_ENV": str(venv_dir)}
        if _is_termux_env(env):
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
        return [uv_bin, "pip"], env
    return [sys.executable, "-m", "pip"], None


def _run_install_with_heartbeat(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    heartbeat_interval_seconds: int = 30,
) -> None:
    """Run dependency install command with periodic heartbeat output.

    Some resolvers/build backends (especially when compiling Rust/C extensions)
    can stay quiet for minutes. Emit a simple elapsed-time heartbeat so users
    know ``hermes update`` is still progressing even if pip/uv itself is silent.
    """
    done = threading.Event()
    start = _time.time()

    def _heartbeat() -> None:
        # Wait first, then print, so short installs don't emit noise.
        while not done.wait(heartbeat_interval_seconds):
            elapsed = int(_time.time() - start)
            print(
                f"  … still installing dependencies ({elapsed}s elapsed)"
                " — compiling Rust/C extensions can take several minutes",
                flush=True,
            )

    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    try:
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            env=env,
        )
    finally:
        done.set()
        t.join(timeout=0.2)


def _is_windows() -> bool:
    return sys.platform == "win32"


def _venv_scripts_dir() -> Path | None:
    """Return the venv Scripts directory if we're running inside the project venv."""
    from hermes_constants import project_venv_dir, venv_bin_dir

    venv_dir = project_venv_dir(PROJECT_ROOT)
    if venv_dir is None:
        return None

    scripts = venv_bin_dir(venv_dir, windows=_is_windows())
    return scripts if scripts.is_dir() else None


def _hermes_exe_shims(scripts_dir: Path) -> list[Path]:
    """Entry-point shims that uv may try to rewrite during ``pip install -e .``.

    On Windows these are .exe launchers generated by setuptools/uv. On POSIX
    they're regular Python scripts which can be replaced atomically — no
    self-replacement hazard exists outside Windows.
    """
    if not _is_windows():
        return []

    names = set(_load_console_script_names()) or {"hermes", "hermes-agent", "hermes-acp"}
    # The gateway shim is not a [project.scripts] entry point, but older
    # update/install paths still rewrite and quarantine it.
    names.add("hermes-gateway")
    return [scripts_dir / f"{name}.exe" for name in sorted(names)]


def _quarantine_running_hermes_exe(
    scripts_dir: Path, *, max_attempts: int = 4,
    failed_out: list[str] | None = None,
) -> list[tuple[Path, Path]]:
    """Pre-empt Windows file lock on the running ``hermes.exe``.

    Windows allows RENAMING a mapped/running executable (the kernel tracks the
    file by handle, not path), but blocks DELETE/REPLACE while it's loaded. uv
    needs to overwrite the entry-point shims during ``pip install -e .``;
    when ``hermes update`` runs, ``hermes.exe`` IS the live process, and uv
    fails with ``Access is denied. (os error 5)``.

    We rename live shims to ``hermes.exe.old.<unix-ms>`` first. uv then writes
    fresh shims at the original paths. The ``.old`` files are cleaned up on
    the next hermes invocation by ``_cleanup_quarantined_exes``.

    Rename can still fail when *another* process has opened the .exe without
    ``FILE_SHARE_DELETE`` — typically AV real-time scanners with transient
    handles (recovers in <1s), or the Hermes Desktop backend child process
    (won't recover until the user closes it). We mitigate:

    1. Retry up to ``max_attempts`` times with exponential backoff
       (100/250/500/1000 ms). Handles the AV-scanner case.
    2. If all retries fail, print a clear warning naming the most likely
       culprit (running Hermes Desktop / gateway / REPL).

    The updater's own launcher is no longer one of those culprits: an update
    started from ``hermes.exe`` re-runs itself under the venv Python before
    reaching here (``_reexec_dependency_sync_off_windows_shim``).

    Returns the list of (original, quarantined) pairs so the caller can roll
    back if the install itself fails before uv writes a replacement.

    ``failed_out``: when provided, the names of shims whose rename failed on
    every attempt are appended — callers that must not mutate a contended
    venv (the update dependency sync, #87331) check it and refuse instead of
    letting the install run into a half-broken state.
    """
    moved: list[tuple[Path, Path]] = []
    if not _is_windows():
        return moved

    import time

    stamp = int(time.time() * 1000)
    # Backoff schedule: first attempt is immediate, subsequent ones sleep.
    # 100ms / 250ms / 500ms covers the typical AV scanner re-scan window.
    backoff_ms = [0, 100, 250, 500, 1000]
    attempts = max(1, min(max_attempts, len(backoff_ms)))

    for shim in _hermes_exe_shims(scripts_dir):
        if not shim.exists():
            continue
        target = shim.with_suffix(shim.suffix + f".old.{stamp}")

        last_exc: OSError | None = None
        for attempt in range(attempts):
            delay = backoff_ms[attempt] / 1000.0
            if delay:
                time.sleep(delay)
            try:
                shim.rename(target)
                moved.append((shim, target))
                last_exc = None
                break
            except OSError as e:
                last_exc = e
                continue

        if last_exc is None:
            continue

        # Every rename failed. Deferring one to next boot via
        # MOVEFILE_DELAY_UNTIL_REBOOT used to be the fallback here, but it
        # cannot help: it needs elevation we don't have, and when it does
        # land it frees nothing for the install running right now while
        # queueing an operation that will move a later, freshly repaired shim
        # aside at next boot. Report and let uv try its luck instead —
        # sometimes its own retry handling pulls through.
        print(
            f"  ⚠ Could not quarantine {shim.name} ({last_exc.__class__.__name__}: "
            f"another process is holding it open)."
        )
        print(
            "    Close Hermes Desktop, exit other `hermes` REPLs, stop the "
            "gateway, or pause AV scanning, then re-run `hermes update`."
        )
        if failed_out is not None:
            failed_out.append(shim.name)

    return moved


_PENDING_RENAME_KEY = r"SYSTEM\CurrentControlSet\Control\Session Manager"
_PENDING_RENAME_VALUE = "PendingFileRenameOperations"


def _filter_pending_shim_renames(
    entries: list[str], shims: list[Path]
) -> tuple[list[str], int]:
    """Drop shim-quarantine pairs from a PendingFileRenameOperations value.

    The value is a flat REG_MULTI_SZ of (source, target) pairs, and other
    installers share it, so only pairs matching our own
    ``<shim>`` -> ``<shim>.old.<stamp>`` naming are removed. Returns the
    entries to keep and how many pairs were dropped.
    """
    import ntpath

    def _norm(value: str) -> str:
        path = str(value).lstrip("!")
        if path.startswith("\\??\\"):
            path = path[4:]
        return ntpath.normcase(ntpath.normpath(path))

    shim_paths = {_norm(str(shim)) for shim in shims}
    kept: list[str] = []
    removed = 0
    for index in range(0, len(entries) - 1, 2):
        source, target = entries[index], entries[index + 1]
        source_norm = _norm(source)
        if source_norm in shim_paths and _norm(target).startswith(f"{source_norm}.old."):
            removed += 1
        else:
            kept.extend((source, target))
    if len(entries) % 2:
        kept.append(entries[-1])
    return kept, removed


def _cleanup_pending_shim_renames(scripts_dir: Path) -> int:
    """Drop reboot renames older Hermes versions queued for our shims.

    Hermes used to fall back to ``MoveFileExW(MOVEFILE_DELAY_UNTIL_REBOOT)``
    when the quarantine rename failed. Those entries outlive the update that
    queued them, so at the next boot they move away whatever now sits at the
    shim path — including a shim a later repair just wrote. Needs elevation
    to remove (same as it needed to create); a no-op otherwise.
    """
    if not _is_windows():
        return 0
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            _PENDING_RENAME_KEY,
            0,
            winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE,
        ) as key:
            entries, value_type = winreg.QueryValueEx(key, _PENDING_RENAME_VALUE)
            if value_type != winreg.REG_MULTI_SZ or not isinstance(entries, list):
                return 0
            kept, removed = _filter_pending_shim_renames(
                entries, _hermes_exe_shims(scripts_dir)
            )
            if not removed:
                return 0
            if kept:
                winreg.SetValueEx(key, _PENDING_RENAME_VALUE, 0, winreg.REG_MULTI_SZ, kept)
            else:
                winreg.DeleteValue(key, _PENDING_RENAME_VALUE)
            return removed
    except (OSError, ValueError):
        return 0


def _restore_quarantined_exes(moved: list[tuple[Path, Path]]) -> None:
    """Roll back ``_quarantine_running_hermes_exe`` if uv didn't write replacements.

    This is the safety-critical direction. A failed *quarantine* only aborts an
    update; a failed *restore* leaves the install with no ``hermes`` on PATH,
    and therefore no way to run the command that would repair it (#75584). The
    outbound rename already retries a lock, so this one must too rather than
    swallow the first ``OSError`` in silence.

    Delegates to the stdlib-only helper that the early-recovery copy in
    ``_install_repair`` also uses, so the two cannot drift apart.
    """
    _early_recovery_mod.restore_quarantined_shims(moved)


class ShimQuarantineError(RuntimeError):
    """A live ``hermes*.exe`` shim could not be renamed aside (#87331).

    Raised by :func:`_run_quarantined_install` in ``strict_quarantine`` mode
    BEFORE the install command runs. A shim that cannot even be renamed means
    another process holds the venv hard enough that the dependency sync would
    die partway and strand the install half-updated — the update must refuse,
    not warn-and-continue.
    """

    def __init__(self, failed_shims: list[str]):
        self.failed_shims = list(failed_shims)
        super().__init__(
            "could not quarantine live shim(s): " + ", ".join(self.failed_shims)
        )


def _run_quarantined_install(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    scripts_dir: Path | None = None,
    strict_quarantine: bool = False,
) -> None:
    """Run an editable install, quarantining the running ``hermes.exe`` first.

    Any ``pip install -e .`` (or ``--reinstall``) rewrites the entry-point
    shims, and on Windows the live ``hermes.exe`` is the running process —
    pip can neither delete nor overwrite it, so without quarantine the shim
    is left missing and ``hermes`` drops off PATH. This wraps
    :func:`_run_install_with_heartbeat` with the same rename-out-of-the-way /
    restore-on-failure dance that the primary install path uses, so EVERY
    install that touches the shims is protected — including the
    verification-repair reinstalls in
    :func:`_verify_core_dependencies_installed`, which previously called
    ``_run_install_with_heartbeat`` directly and bypassed quarantine.

    ``strict_quarantine=True`` (the update dependency sync, #87331): a shim
    whose rename failed every retry means a process is holding the venv
    without ``FILE_SHARE_DELETE`` — the install WILL hit the same lock on
    .pyd files and strand the venv between versions. Roll the successful
    renames back and raise :class:`ShimQuarantineError` WITHOUT running the
    install. Non-strict callers (post-sync entry-point repair) keep the old
    warn-and-try behavior: their venv is already mutated, so refusing buys
    nothing.

    Off-Windows (``scripts_dir is None``) this is a thin pass-through.
    """
    moved: list[tuple[Path, Path]] = []
    failed: list[str] = []
    if scripts_dir is not None:
        moved = _quarantine_running_hermes_exe(scripts_dir, failed_out=failed)
    if strict_quarantine and failed:
        _restore_quarantined_exes(moved)
        raise ShimQuarantineError(failed)
    try:
        _run_install_with_heartbeat(cmd, env=env)
    finally:
        # Restore shims when the installer didn't write replacements — on
        # FAILURE (install died before the entry-points step) and on SUCCESS
        # too: uv audits an already-satisfied editable install as a no-op and
        # rewrites no entry points, which would otherwise leave the shims
        # quarantined aside and `hermes` missing from PATH after a green
        # install (#75584). _restore_quarantined_exes skips any shim the
        # installer actually replaced, so this never clobbers fresh output.
        # Errors are not swallowed — the finally re-raises whatever escaped.
        if scripts_dir is not None:
            _restore_quarantined_exes(moved)


# A quarantine file younger than this may belong to an update running RIGHT
# NOW in another process, whose restore step still needs it. Deleting one
# mid-flight destroys the only copy of that shim.
_QUARANTINE_GRACE_SECONDS = 15 * 60


def _quarantine_stamp_ms(stale: Path) -> int | None:
    """The ``.old.<unix-ms>`` stamp in a quarantine filename, or ``None``.

    ``None`` means the name was not produced by
    :func:`_quarantine_running_hermes_exe`. We neither rescue nor delete those:
    the sweep should not destroy files whose provenance it cannot establish, and
    they are not ours to put back.

    Parsed from the NAME rather than ``st_mtime`` because ``rename`` preserves
    the original shim's mtime, which records when uv wrote the shim — days
    earlier, in general — not when it was quarantined.
    """
    try:
        return int(stale.name.rsplit(".old.", 1)[1])
    except (IndexError, ValueError):
        return None


def _cleanup_quarantined_exes(scripts_dir: Path | None = None) -> None:
    """Sweep — and where necessary RESCUE — ``hermes.exe.old.*`` from updates.

    Called early on every hermes invocation. Two cases the old unconditional
    ``unlink()`` got wrong, both ending with ``hermes`` gone from PATH:

    1. **Orphan rescue.** If ``hermes.exe`` is missing while
       ``hermes.exe.old.*`` is present, that .old file is the ONLY surviving
       copy of the shim — an update died, or its restore failed, between
       the rename and uv writing a replacement (#75584). Deleting it converts a
       one-rename recovery into a full reinstall. Put it back instead, through
       the same retry-and-report helper the update-time restore uses.
    2. **Concurrency.** A fresh quarantine file may belong to an update in
       flight in another process (the desktop update button racing a shell
       ``hermes update`` does exactly this). Leave anything inside the grace
       window alone; a later run sweeps it.

    Silent no-op on non-Windows, when there is nothing to do, or on
    file-locked / permission errors.
    """
    if not _is_windows():
        return
    if scripts_dir is None:
        scripts_dir = _venv_scripts_dir()
    if scripts_dir is None:
        return
    _cleanup_pending_shim_renames(scripts_dir)

    now = _time.time()

    try:
        candidates = [
            (stamp, stale)
            for stale, stamp in (
                (p, _quarantine_stamp_ms(p)) for p in scripts_dir.glob("*.exe.old.*")
            )
            if stamp is not None
        ]
    except OSError:
        return

    # Newest first by PARSED stamp. Sorting the raw filenames lexicographically
    # only tracks recency while every stamp shares a digit width: a stray
    # ``.old.999`` sorts above a 13-digit epoch-ms stamp and would be the copy
    # rescued onto the live shim name.
    candidates.sort(key=lambda pair: pair[0], reverse=True)

    for stamp, stale in candidates:
        try:
            original = stale.with_name(stale.name.rsplit(".old.", 1)[0])

            if not original.exists():
                # Orphan rescue: this is the last copy of the shim, so it gets
                # the retry ladder and the recovery message, not a bare rename.
                _early_recovery_mod.restore_quarantined_shims([(original, stale)])
                continue

            if now - stamp / 1000.0 < _QUARANTINE_GRACE_SECONDS:
                continue  # may be a live quarantine from a concurrent update

            stale.unlink()
        except OSError:
            pass  # still locked or in use — try again next run


# Import probes for venv corruption after a failed lazy ``uv pip install``.
# Metadata can look fine while ``.py`` files were removed mid-install (#57828).
# Canonical tables live in the stdlib-only ``_early_recovery`` module (which
# also probes/repairs BEFORE this module's third-party imports can run) so the
# early and full recovery layers can never drift apart.
_LAZY_REFRESH_IMPORT_PROBES: tuple[tuple[str, str], ...] = (
    _early_recovery_mod.LAZY_REFRESH_IMPORT_PROBES
)

_LAZY_REFRESH_REPAIR_PACKAGES: dict[str, str] = (
    _early_recovery_mod.LAZY_REFRESH_REPAIR_PACKAGES
)


def _run_package_only_install(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Run a package-only pip/uv install without quarantining entry-point shims.

    ``pip install --upgrade pip`` and ``--force-reinstall <pkg>`` do not
    rewrite ``hermes.exe``. The editable-install quarantine path would rename
    shims without uv recreating them on Windows (#57828).
    """
    _run_install_with_heartbeat(cmd, env=env)


def _lazy_refresh_repair_specs(packages: list[str]) -> list[str]:
    """Map repair package names to their declared pin specs in pyproject.toml."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover
        return packages

    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return packages

    try:
        with open(pyproject, "rb") as f:
            raw_deps = tomllib.load(f).get("project", {}).get("dependencies", []) or []
    except Exception as exc:
        logger.debug("lazy refresh repair spec lookup failed: %s", exc)
        return packages

    name_to_spec: dict[str, str] = {}
    try:
        from packaging.requirements import Requirement  # type: ignore

        for spec in raw_deps:
            try:
                req = Requirement(spec)
                name_to_spec[req.name.lower()] = spec.split(";", 1)[0].strip()
            except Exception:
                continue
    except Exception:
        for spec in raw_deps:
            head = spec.split(";", 1)[0].strip()
            bare = head
            for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
                if op in bare:
                    bare = bare.split(op, 1)[0]
                    break
            key = bare.strip().split("[", 1)[0].strip().lower()
            if key:
                name_to_spec[key] = head

    return [name_to_spec.get(pkg.lower(), pkg) for pkg in packages]


def _detect_broken_lazy_refresh_imports(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> list[str] | None:
    """Probe lazy-refresh packages via real imports.

    Returns:
      - ``[]`` when probes ran and every package imported cleanly
      - ``[dist, ...]`` when probes ran and some packages failed
      - ``None`` when the probe could not run (missing venv Python, subprocess
        failure, non-zero probe exit) — this is *indeterminate*, not healthy
    """
    venv_python = _resolve_install_target_python(install_cmd_prefix, env)
    if venv_python is None:
        return None

    probe_lines = "\n".join(
        f"    ({mod!r}, {attr!r})," for mod, attr in _LAZY_REFRESH_IMPORT_PROBES
    )
    check_script = (
        "import os\n"
        "import sys\n"
        "probes = [\n"
        f"{probe_lines}\n"
        "]\n"
        "broken = []\n"
        "for mod, attr in probes:\n"
        "    try:\n"
        "        imported = __import__(mod)\n"
        "        if not hasattr(imported, attr):\n"
        "            broken.append(mod)\n"
        "        elif mod == 'certifi':\n"
        "            # The module can import cleanly while cacert.pem is\n"
        "            # missing/corrupt (brew Python upgrade, interrupted venv\n"
        "            # rebuild) - every TLS call then fails (#29866).\n"
        "            bundle = imported.where()\n"
        "            if not os.path.isfile(bundle) or os.path.getsize(bundle) < 1024:\n"
        "                broken.append(mod)\n"
        "    except Exception:\n"
        "        broken.append(mod)\n"
        "print('\\n'.join(broken))\n"
    )
    try:
        result = subprocess.run(
            [str(venv_python), "-c", check_script],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            check=False,
            env=env,
        )
    except Exception as exc:
        logger.debug("lazy refresh import probe failed: %s", exc)
        return None

    if result.returncode != 0:
        logger.debug(
            "lazy refresh import probe exited %s: %s",
            result.returncode,
            (result.stderr or "")[:200],
        )
        return None

    broken_modules = [
        line.strip() for line in result.stdout.splitlines() if line.strip()
    ]
    packages: list[str] = []
    seen: set[str] = set()
    for mod in broken_modules:
        pkg = _LAZY_REFRESH_REPAIR_PACKAGES.get(mod)
        if pkg and pkg not in seen:
            seen.add(pkg)
            packages.append(pkg)
    return packages


def _repair_broken_lazy_refresh_imports(
    install_cmd_prefix: list[str],
    packages: list[str],
    *,
    env: dict[str, str] | None = None,
) -> bool:
    """Force-reinstall ``packages`` and re-probe imports. Never raises."""
    if not packages:
        return True

    specs = _lazy_refresh_repair_specs(packages)
    try:
        _run_package_only_install(
            install_cmd_prefix + ["install", "--force-reinstall", *specs],
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning("lazy refresh venv repair failed: %s", exc)
        return False

    after = _detect_broken_lazy_refresh_imports(install_cmd_prefix, env=env)
    # Indeterminate re-probe is not confirmed success.
    return after == []


def _repair_venv_via_import_probes(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> str:
    """Probe imports and force-reinstall any broken lazy-refresh packages.

    Uses real ``import`` checks (not distribution metadata) so a venv where
    METADATA remains but ``.py`` files were wiped mid-install is still
    detected (#57828). Package-only reinstall — never rewrites ``hermes.exe``.

    Never raises. Returns one of:
      - ``"healthy"`` — probes ran and found nothing broken
      - ``"repaired"`` — probes found breakage and force-reinstall confirmed clean
      - ``"failed"`` — probes found breakage and repair did not confirm clean
      - ``"indeterminate"`` — probes could not run; do NOT treat as healthy
    """
    broken = _detect_broken_lazy_refresh_imports(install_cmd_prefix, env=env)
    if broken is None:
        print(
            "  ⚠ Import probes unavailable — cannot confirm venv package health."
        )
        return "indeterminate"
    if not broken:
        return "healthy"
    print(
        "  → Detected corrupted venv packages via import probes: "
        f"{', '.join(broken)}; repairing..."
    )
    if _repair_broken_lazy_refresh_imports(
        install_cmd_prefix, broken, env=env
    ):
        print("  ✓ Venv repair succeeded")
        return "repaired"
    manual = " ".join(
        shlex.quote(s) for s in _lazy_refresh_repair_specs(broken)
    )
    print("  ⚠ Venv repair incomplete. Run manually, then `hermes update`:")
    print(
        f"    {' '.join(install_cmd_prefix)} install --force-reinstall {manual}"
    )
    return "failed"


def _is_uv_command(install_cmd_prefix: list[str]) -> bool:
    """True when the install command is a uv/uvx invocation.

    Handles a bare uv binary (``uv`` / ``uvx``, any extension), a path to
    one, and ``python -m uv`` / ``python -m uvx`` — the naive basename check
    misses the module form and launcher wrappers whose name does not contain
    "uv".
    """
    if not install_cmd_prefix:
        return False
    first = str(install_cmd_prefix[0]).lower()
    if "uv" in Path(first).name:
        return True
    # python -m uv / python -m uvx
    if len(install_cmd_prefix) >= 3 and first.endswith(("python", "python.exe")):
        return install_cmd_prefix[1] == "-m" and install_cmd_prefix[2] in (
            "uv",
            "uvx",
        )
    return False


def _insert_python_pin(args: list[str]) -> list[str]:
    """Insert ``--python <sys.executable>`` into a uv command line.

    If the caller already passed ``--python``, its value wins (uv's last-wins
    semantics are ambiguous; the explicit caller intent should not be
    overridden by the fallback pin).
    """
    if "--python" in args:
        return args
    return [args[0], "--python", str(sys.executable), *args[1:]]


def _interpreter_scripts_dir() -> Path | None:
    """Scripts/bin directory of the running interpreter (sys.executable).

    Used when pinning an install to ``sys.executable`` on a site-packages
    install where ``PROJECT_ROOT / "venv"`` does not exist: the entry-point
    shims uv rewrites live next to the interpreter, not under a project venv.
    Layout comes from the canonical ``venv_bin_dir`` helper (#76105 —
    hand-rolling Scripts/bin is lint-tested against).
    """
    from hermes_constants import venv_bin_dir

    exe = Path(sys.executable)
    # sys.executable lives IN the bin/Scripts dir; its parent.parent is the
    # env root venv_bin_dir derives from.
    cand = venv_bin_dir(exe.parent.parent, windows=_is_windows())
    if cand.is_dir():
        return cand
    return exe.parent if exe.parent.is_dir() else None


def _install_python_dependencies_with_optional_fallback(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
    group: str = "all",
) -> None:
    """Install base deps plus as many optional extras as the environment supports.

    By default this targets ``.[all]``; Termux callers can pass
    ``group='termux-all'`` to use the curated Android-compatible profile.

    On Windows, pre-renames live ``hermes.exe`` / ``hermes-gateway.exe`` shims
    in the venv Scripts dir before each install attempt so uv can write fresh
    copies (Windows blocks REPLACE on a running .exe but allows RENAME). See
    ``_quarantine_running_hermes_exe`` for the rationale.

    When ``env`` carries a ``VIRTUAL_ENV`` that does not exist (a pip /
    site-packages install whose ``PROJECT_ROOT`` is the interpreter's
    ``site-packages`` directory, where ``PROJECT_ROOT / "venv"`` is never
    created), ``uv pip`` fails with ``Failed to inspect Python interpreter from
    active virtual environment`` before doing any work.  Pin the install at the
    running interpreter instead so the update/recovery path succeeds on those
    installs (#71510 fixed the ZIP path, #83335 fixed lazy-deps; this closes the
    shared helper for the remaining callers).
    """
    scripts_dir = _venv_scripts_dir() if _is_windows() else None

    # A pip / site-packages install has no PROJECT_ROOT/venv; the caller still
    # passes VIRTUAL_ENV=PROJECT_ROOT/venv, which does not exist. uv would fail
    # before installing anything ("Failed to inspect Python interpreter from
    # active virtual environment"). Detect the stale pointer and pin the target
    # interpreter explicitly instead of trusting the nonexistent venv.
    pin_python = False
    if (
        env
        and env.get("VIRTUAL_ENV")
        and not Path(env["VIRTUAL_ENV"]).is_dir()
        and install_cmd_prefix
        and _is_uv_command(install_cmd_prefix)
    ):
        # Only uv needs the explicit pin; pip resolves the target from
        # sys.executable itself and has no --python flag.
        pin_python = True
        env = {**env}
        env.pop("VIRTUAL_ENV", None)
        # When we pin to sys.executable, the entry-point shims that uv will
        # rewrite live in that interpreter's Scripts/bin directory, NOT in
        # PROJECT_ROOT/venv (which does not exist on a site-packages install).
        # Quarantining the wrong dir means the running hermes.exe stays locked
        # on Windows and the install fails exactly like the original bug. Only
        # override when the venv-derived dir is missing; otherwise keep it.
        if scripts_dir is None and _is_windows():
            scripts_dir = _interpreter_scripts_dir()

    def _install(args: list[str]) -> None:
        if pin_python:
            args = _insert_python_pin(args)
        # strict_quarantine: this is the UPDATE dependency sync. A shim that
        # cannot be renamed aside proves a hard venv hold; running uv anyway
        # is how installs strand half-updated (#87331). ShimQuarantineError
        # propagates to the update's sync boundary, which defers via the
        # update-incomplete marker instead of mutating a contended venv.
        _run_quarantined_install(
            install_cmd_prefix + args, env=env, scripts_dir=scripts_dir,
            strict_quarantine=True,
        )

    try:
        _install(["install", "-e", f".[{group}]"])
        _verify_console_scripts_installed(install_cmd_prefix, env=env)
        return
    except subprocess.CalledProcessError:
        print(
            "  ⚠ Optional extras failed, reinstalling base dependencies and retrying extras individually..."
        )

    _install(["install", "-e", "."])

    failed_extras: list[str] = []
    installed_extras: list[str] = []
    for extra in _load_installable_optional_extras(group=group):
        try:
            _install(["install", "-e", f".[{extra}]"])
            installed_extras.append(extra)
        except subprocess.CalledProcessError:
            failed_extras.append(extra)

    if installed_extras:
        print(
            f"  ✓ Reinstalled optional extras individually: {', '.join(installed_extras)}"
        )
    if failed_extras:
        print(
            f"  ⚠ Skipped optional extras that still failed: {', '.join(failed_extras)}"
        )

    # Belt-and-suspenders: verify every declared core dependency from
    # pyproject.toml's [project.dependencies] is actually importable in the
    # target venv. uv's incremental resolver has — in the wild — produced
    # partial installs where a newly added base dep (e.g. ``pathspec``)
    # silently fails to land on top of a half-stale venv, and the only
    # symptom is a downstream subprocess crashing with ModuleNotFoundError
    # hours later inside ``hermes update``'s desktop-rebuild or skill-sync
    # stage. Reinstall with --reinstall to force resolution if anything is
    # missing, then re-verify so the failure surfaces here instead of
    # downstream.
    _verify_core_dependencies_installed(install_cmd_prefix, env=env, group=group)
    _verify_console_scripts_installed(install_cmd_prefix, env=env)


def _load_console_script_names() -> list[str]:
    """Return ``[project.scripts]`` entry-point names from pyproject.toml."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover
        return []

    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return []

    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        scripts = data.get("project", {}).get("scripts", {}) or {}
        return [str(name) for name in scripts if name]
    except Exception as e:
        logger.debug("console script verification: failed to read pyproject.toml: %s", e)
        return []


def _verify_console_scripts_installed(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Ensure every declared console_script shim exists on disk after install.

    On Windows, ``uv pip install -e .`` can register ``hermes.exe`` in the
    wheel RECORD while the file never lands on disk — typically when the live
    ``hermes.exe`` shim is locked during ``hermes update``, or when uv/distlib
    skips a launcher write. The symptom is ``hermes-agent.exe`` and
    ``hermes-acp.exe`` present but ``hermes.exe`` missing, so ``hermes`` drops
    off PATH even though the install reported success (issue #52931).

    If any shim is missing we reinstall with ``--reinstall -e .`` under the
    same quarantine dance as the primary install path, then re-check.
    """
    if not _is_windows():
        return

    scripts_dir = _venv_scripts_dir()
    if scripts_dir is None:
        return

    names = _load_console_script_names()
    if not names:
        return

    def _missing() -> list[str]:
        return [
            name
            for name in names
            if not (scripts_dir / f"{name}.exe").is_file()
        ]

    missing = _missing()
    if not missing:
        return

    print(
        f"  ⚠ Verification: {len(missing)} console script(s) missing on disk: "
        f"{', '.join(missing)}"
    )
    print("  → Reinstalling entry points with --reinstall...")

    try:
        _run_quarantined_install(
            install_cmd_prefix + ["install", "--reinstall", "-e", "."],
            env=env,
            scripts_dir=scripts_dir,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("console script verification: repair install failed: %s", e)
        print(
            "  ⚠ Entry point repair failed; try `hermes update --force` after "
            "closing other hermes processes."
        )
        return

    still_missing = _missing()
    if still_missing:
        print(
            f"  ⚠ Still missing after repair: {', '.join(still_missing)}. "
            "Workaround: python -m hermes_cli.main <command>"
        )
    else:
        print("  ✓ All console entry points restored")


def _verify_core_dependencies_installed(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
    group: str = "all",
) -> None:
    """Check that every base dep from pyproject.toml is importable; if not, retry.

    Reads ``pyproject.toml`` directly (so we don't trust the venv's stale
    metadata), filters out deps gated by ``;`` environment markers that don't
    apply to this platform, and runs ``importlib.metadata.version()`` in the
    venv interpreter for each one. If anything is missing we reinstall the
    base group with ``--reinstall`` to force uv to re-resolve, then check
    again. We treat the final state as a warning rather than a hard failure
    so a single broken-on-PyPI dep can't block an otherwise-successful
    update — but the warning makes the partial install visible at the spot
    that caused it, instead of hours later in a downstream subprocess.
    """
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover — Python < 3.11 unsupported but be safe
        return

    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return

    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        raw_deps = data.get("project", {}).get("dependencies", []) or []
    except Exception as e:
        logger.debug("dep verification: failed to read pyproject.toml: %s", e)
        return

    # Parse each "name OP version ; marker" string into (dist_name, marker_obj).
    # We use packaging.requirements when available (it ships with pip/uv envs),
    # falling back to a naive split that's good enough for the canonical
    # ``name==version[; marker]`` style this repo uses.
    deps: list[tuple[str, "object | None"]] = []
    try:
        from packaging.requirements import Requirement  # type: ignore

        for spec in raw_deps:
            try:
                req = Requirement(spec)
                deps.append((req.name, req.marker))
            except Exception:
                continue
    except Exception:
        for spec in raw_deps:
            head = spec.split(";", 1)[0]
            for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
                if op in head:
                    head = head.split(op, 1)[0]
                    break
            name = head.strip().split("[", 1)[0].strip()
            if name:
                deps.append((name, None))

    # Apply environment markers to drop deps that don't apply on this platform
    # (e.g. ``ptyprocess ; sys_platform != 'win32'`` is correctly skipped on
    # Windows). Without markers we'd false-positive every cross-platform exclusion.
    applicable: list[str] = []
    for name, marker in deps:
        if marker is None:
            applicable.append(name)
            continue
        try:
            if marker.evaluate():  # type: ignore[union-attr]
                applicable.append(name)
        except Exception:
            applicable.append(name)

    if not applicable:
        return

    # Run the check inside the venv Python — sys.executable here may be the
    # outer Python that drove ``hermes update``, not the venv we just wrote
    # to. The uv install_cmd_prefix encodes which environment we targeted
    # (either ``[uv, pip]`` with VIRTUAL_ENV in env, or
    # ``[sys.executable, -m, pip]`` for the in-process Python); resolve the
    # right interpreter for the verification.
    venv_python = _resolve_install_target_python(install_cmd_prefix, env)
    if venv_python is None:
        return

    def _missing_deps() -> list[str]:
        check_script = (
            "import importlib.metadata as md, sys\n"
            "missing=[]\n"
            "for name in sys.argv[1:]:\n"
            "    try: md.version(name)\n"
            "    except md.PackageNotFoundError: missing.append(name)\n"
            "print('\\n'.join(missing))\n"
        )
        try:
            result = subprocess.run(
                [str(venv_python), "-c", check_script, *applicable],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                check=False,
                env=env,
            )
        except Exception as e:
            logger.debug("dep verification: subprocess failed: %s", e)
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    missing = _missing_deps()
    if not missing:
        return

    print(
        f"  ⚠ Verification: {len(missing)} declared dep(s) missing after install: "
        f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}"
    )
    print("  → Reinstalling base group with --reinstall to repair...")

    # Reinstall base group with --reinstall so uv re-resolves from scratch
    # against the current pyproject. We don't pass ``[{group}]`` here on
    # purpose — the missing dep is in *base* deps; rerunning the full all-
    # extras install can cost minutes and trips on whatever optional extra
    # was already broken upstream. Base is fast and is what's actually wrong.
    #
    # Quarantine the running ``hermes.exe`` first: ``--reinstall -e .``
    # rewrites the entry-point shims, and on Windows pip can't overwrite the
    # live launcher, which would leave ``hermes`` off PATH.
    scripts_dir = _venv_scripts_dir() if _is_windows() else None
    repair_args = ["install", "--reinstall", "-e", "."]
    try:
        _run_quarantined_install(
            install_cmd_prefix + repair_args, env=env, scripts_dir=scripts_dir
        )
    except subprocess.CalledProcessError as e:
        logger.warning("dep verification: repair install failed: %s", e)
        print("  ⚠ Repair install failed; check `hermes update` output above.")
        return

    still_missing = _missing_deps()
    if not still_missing:
        print("  ✓ All declared core dependencies now installed")
        return

    # Last-ditch: install each remaining missing dep with its pin directly.
    # Useful when uv's resolver thinks the env is satisfied but the on-disk
    # package metadata says otherwise (rare but observed).
    name_to_spec = {}
    for spec in raw_deps:
        head = spec.split(";", 1)[0].strip()
        bare = head
        for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
            if op in bare:
                bare = bare.split(op, 1)[0]
                break
        name_to_spec[bare.strip().split("[", 1)[0].strip()] = head

    specs = [name_to_spec.get(n, n) for n in still_missing]
    print(
        f"  → Force-installing remaining missing dep(s): {', '.join(specs)}"
    )
    try:
        _run_install_with_heartbeat(
            install_cmd_prefix + ["install", "--reinstall", *specs], env=env
        )
    except subprocess.CalledProcessError as e:
        logger.warning("dep verification: per-package repair failed: %s", e)
        print(
            f"  ⚠ Could not install: {', '.join(still_missing)}. "
            "Run `hermes update --force` after closing other hermes processes."
        )
        return

    final_missing = _missing_deps()
    if final_missing:
        print(
            f"  ⚠ Still missing after repair: {', '.join(final_missing)}. "
            "Run `hermes update --force` after closing other hermes processes."
        )
    else:
        print("  ✓ All declared core dependencies now installed")


def _resolve_install_target_python(
    install_cmd_prefix: list[str], env: dict[str, str] | None
) -> Path | None:
    """Figure out which Python interpreter the install just targeted.

    ``_install_python_dependencies_with_optional_fallback`` is called with
    either ``[uv, pip]`` (and a ``VIRTUAL_ENV`` env var pointing at the
    target venv) or ``[sys.executable, -m, pip]`` (the in-process Python).
    The verification step needs the *resulting* environment's Python so
    ``importlib.metadata`` queries the right site-packages.
    """
    if env and "VIRTUAL_ENV" in env:
        from hermes_constants import venv_python_path

        venv_root = Path(env["VIRTUAL_ENV"])
        candidate = venv_python_path(venv_root, windows=_is_windows())
        if candidate.exists():
            return candidate

    # Fallback: assume install_cmd_prefix[0] is the python interpreter (the
    # ``[sys.executable, -m, pip]`` shape). Skip if it looks like ``uv``.
    if install_cmd_prefix:
        first = Path(install_cmd_prefix[0])
        if first.exists() and "uv" not in first.name.lower():
            return first

    return None


def _is_termux_env(env: dict[str, str] | None = None) -> bool:
    return _is_termux_startup_environment(env)


def _is_windows_npm_path(npm_path: str) -> bool:
    """Return True if ``npm_path`` points at a Windows npm shim.

    On WSL the Windows install dir is exposed through the ``/mnt/c`` drive
    mount and PATH interop, so ``shutil.which("npm")`` can hand back
    ``/mnt/c/Program Files/nodejs/npm`` (or the ``npm.cmd`` / ``npm.exe``
    shim). Those are detected here by their ``.exe``/``.cmd``/``.bat``
    suffix, a ``/mnt/`` drive-mount prefix, or an embedded backslash (a UNC
    path). Callers use this only on a POSIX host — on native Windows an
    ``npm.cmd`` shim is the correct executable.
    """
    low = npm_path.lower()
    return (
        low.endswith((".exe", ".cmd", ".bat"))
        or low.startswith("/mnt/")
        or "\\" in npm_path
    )


def _resolve_node_runtime_npm() -> str | None:
    """Resolve an npm executable that belongs to the host's Node runtime.

    On WSL/Linux ``shutil.which("npm")`` may resolve a Windows npm exposed
    through PATH interop. Running that Windows npm against the Linux checkout
    operates over ``\\wsl.localhost\\...`` UNC paths and fails with EISDIR /
    symlink errors in symlink-heavy trees like ``ui-tui`` (#30271). Refuse a
    Windows npm on a POSIX host and re-scan PATH (skipping ``/mnt/*`` interop
    entries) for a Linux-native npm. Returns the npm path, or ``None`` when
    no suitable npm is reachable.
    """
    from hermes_constants import find_node_executable

    npm = find_node_executable("npm")

    # On native Windows the platform npm (``npm.cmd``) is exactly what we
    # want — only reject Windows shims when we're a POSIX/WSL process.
    if _is_windows():
        return npm

    if not npm:
        return None

    if not _is_windows_npm_path(npm):
        return npm

    # The first resolution was a Windows npm. Re-scan PATH skipping the
    # ``/mnt/*`` Windows drive mounts WSL injects, so a Linux-native npm that
    # came later on PATH is still found.
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory or directory.lower().startswith("/mnt/"):
            continue
        candidate = shutil.which("npm", path=directory)
        if candidate and not _is_windows_npm_path(candidate):
            return candidate
    return None


class _UpdateOutputStream:
    """Stream wrapper used during ``hermes update`` to survive terminal loss.

    Wraps the process's original stdout/stderr so that:

    * Every write is also mirrored to an append-only log file
      (``~/.hermes/logs/update.log``) that users can inspect after the
      terminal disconnects.
    * Writes to the original stream that fail with ``BrokenPipeError`` /
      ``OSError`` / ``ValueError`` (closed file) no longer cascade into
      process exit — the update keeps going, only the on-screen output
      stops.

    Combined with ``SIGHUP -> SIG_IGN`` installed by
    ``_install_hangup_protection``, this makes ``hermes update`` safe to
    run in a plain SSH session that might disconnect mid-install.
    """

    def __init__(self, original, log_file):
        self._original = original
        self._log = log_file
        self._original_broken = False

    def write(self, data):
        # Mirror to the log file first — it's the most reliable destination.
        if self._log is not None:
            try:
                self._log.write(data)
            except Exception:
                # Log errors should never abort the update.
                pass

        if self._original_broken:
            return len(data) if isinstance(data, (str, bytes)) else 0

        try:
            return self._original.write(data)
        except (BrokenPipeError, OSError, ValueError):
            # Terminal vanished (SSH disconnect, shell close).  Stop trying
            # to write to it, but keep the update running.
            self._original_broken = True
            return len(data) if isinstance(data, (str, bytes)) else 0

    def flush(self):
        if self._log is not None:
            try:
                self._log.flush()
            except Exception:
                pass
        if self._original_broken:
            return
        try:
            self._original.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._original_broken = True

    def isatty(self):
        if self._original_broken:
            return False
        try:
            return self._original.isatty()
        except Exception:
            return False

    def fileno(self):
        # Some tools probe fileno(); defer to the underlying stream and let
        # callers handle failures (same behaviour as the unwrapped stream).
        return self._original.fileno()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _install_hangup_protection(gateway_mode: bool = False):
    """Protect ``cmd_update`` from SIGHUP and broken terminal pipes.

    Users commonly run ``hermes update`` in an SSH session or a terminal
    that may close mid-install.  Without protection, ``SIGHUP`` from the
    terminal kills the Python process during ``pip install`` and leaves
    the venv half-installed; the documented workaround ("use screen /
    tmux") shouldn't be required for something as routine as an update.

    Protections installed:

    1. ``SIGHUP`` is set to ``SIG_IGN``.  POSIX preserves ``SIG_IGN``
       across ``exec()``, so pip and git subprocesses also stop dying on
       hangup.
    2. ``sys.stdout`` / ``sys.stderr`` are wrapped to mirror output to
       ``~/.hermes/logs/update.log`` and to silently absorb
       ``BrokenPipeError`` when the terminal vanishes.

    ``SIGINT`` (Ctrl-C) and ``SIGTERM`` (systemd shutdown) are
    **intentionally left alone** — those are legitimate cancellation
    signals the user or OS sent on purpose.

    In gateway mode (``hermes update --gateway``) the update is already
    spawned detached from a terminal, so this function is a no-op.

    Returns a dict that ``cmd_update`` can pass to
    ``_finalize_update_output`` on exit.  Returning a dict rather than a
    tuple keeps the call site forward-compatible with future additions.
    """
    state = {
        "prev_stdout": sys.stdout,
        "prev_stderr": sys.stderr,
        "log_file": None,
        "installed": False,
    }

    if gateway_mode:
        return state

    import signal as _signal

    # (1) Ignore SIGHUP for the remainder of this process.
    if hasattr(_signal, "SIGHUP"):
        try:
            _signal.signal(_signal.SIGHUP, _signal.SIG_IGN)
        except (ValueError, OSError):
            # Called from a non-main thread — not fatal.  The update still
            # runs, just without hangup protection.
            pass

    # (2) Mirror output to update.log and wrap stdio for broken-pipe
    # tolerance.  Any failure here is non-fatal; we just skip the wrap.
    try:
        # Late-bound import so tests can monkeypatch
        # hermes_cli.config.get_hermes_home to simulate setup failure.
        from hermes_cli.config import get_hermes_home as _get_hermes_home

        logs_dir = _get_hermes_home() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "update.log"
        log_file = open(log_path, "a", buffering=1, encoding="utf-8")

        import datetime as _dt

        log_file.write(
            f"\n=== hermes update started "
            f"{_dt.datetime.now().isoformat(timespec='seconds')} ===\n"
        )

        state["log_file"] = log_file
        sys.stdout = _UpdateOutputStream(state["prev_stdout"], log_file)
        sys.stderr = _UpdateOutputStream(state["prev_stderr"], log_file)
        state["installed"] = True
    except Exception:
        # Leave stdio untouched on any setup failure.  Update continues
        # without mirroring.
        state["log_file"] = None

    return state


def _finalize_update_output(state):
    """Restore stdio and close the update.log handle opened by ``_install_hangup_protection``."""
    if not state:
        return
    if state.get("installed"):
        try:
            sys.stdout = state.get("prev_stdout", sys.stdout)
        except Exception:
            pass
        try:
            sys.stderr = state.get("prev_stderr", sys.stderr)
        except Exception:
            pass
    log_file = state.get("log_file")
    if log_file is not None:
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass


def _resolve_update_branch(args) -> str:
    """Normalize ``args.branch`` into a non-empty branch name.

    Centralizes the "default to main, accept --branch override, treat empty
    or whitespace-only values as the default" parsing so every consumer of
    ``--branch`` (check path, git-update path, ZIP-fallback path) agrees on
    the same answer.
    """
    return (getattr(args, "branch", None) or "main").strip() or "main"


def _size_delta_label(saved_mb: float) -> str:
    """Human label for a before/after database size delta, in MB.

    A negative delta means the file GREW — concurrent session writes during a
    long optimize can outweigh what the rebuild freed. Printing
    "reclaimed -163.0 MB" for that reads as data loss, so say "grew by"
    instead.
    """
    if saved_mb >= 0:
        return f"reclaimed {saved_mb:.1f} MB"
    return f"grew by {-saved_mb:.1f} MB"


def cmd_update(args):
    """Update Hermes Agent to the latest version.

    Thin wrapper around ``_cmd_update_impl``: installs hangup protection,
    runs the update, then restores stdio on the way out (even on
    ``sys.exit`` or unhandled exceptions).
    """
    from hermes_cli.config import (
        is_managed,
        managed_error,
    )

    if is_managed():
        managed_error("update Hermes Agent")
        return

    # --plan is read-only and deployment-kind aware, so it runs BEFORE the
    # docker/nix/apt refusal gates: on an image-managed or package-managed
    # install the plan itself reports "not updatable in place" plus the
    # right mechanism — strictly more useful than the bare refusal text.
    if getattr(args, "plan", False):
        # Read-only plan phase (#91277 Phase 2): inventory every running
        # Hermes runtime across profiles, its supervisor, and its running
        # code version — without mutating anything. Safe on a live fleet.
        from hermes_cli.update_inventory import (
            collect_runtime_inventory,
            print_update_plan,
        )

        print_update_plan(collect_runtime_inventory())
        return

    # Image-managed / package-managed admission gate (#91277 Phase 3): one
    # shared decision for every mutation surface. Consults the baked image
    # provenance marker first (authoritative, fail-closed on malformed),
    # then the pre-existing docker/nix/apt heuristics. Prints the real
    # update command, records a `refused` receipt so fleet tooling sees the
    # blocked attempt, and exits 2 (refused-by-contract, distinct from
    # exit 1 errors).
    from hermes_cli.update_contract import (
        evaluate_update_admission,
        record_refusal_receipt,
    )

    refusal = evaluate_update_admission(PROJECT_ROOT)
    if refusal is not None:
        print(refusal.message)
        record_refusal_receipt(refusal)
        sys.exit(2)

    if getattr(args, "check", False):
        # --check honors --branch so the "any new commits?" answer matches
        # what a subsequent `hermes update --branch=<x>` would actually pull.
        branch = _resolve_update_branch(args)
        _self()._cmd_update_check(
            branch=branch,
            branch_explicit=bool(getattr(args, "branch", None)),
        )
        return

    gateway_mode = getattr(args, "gateway", False)

    # Protect against mid-update terminal disconnects (SIGHUP) and tolerate
    # writes to a closed stdout.  No-op in gateway mode.  See
    # _install_hangup_protection for rationale.
    _update_io_state = _install_hangup_protection(gateway_mode=gateway_mode)
    # Cross-process mutual exclusion. The dashboard's Update button spawns
    # this same command detached, and the desktop hands off to the Tauri
    # updater / install-mode bootstrap — all three mutate one checkout. Two of
    # them running together rewrite source under a live interpreter and strand
    # the tree half-updated. Share the marker the Tauri updater and Electron
    # already use rather than inventing a second lock.
    from hermes_cli.update_lock import (
        UPDATE_EXIT_CONCURRENT,
        UpdateLock,
        describe_holder,
    )

    _update_lock = UpdateLock()
    if not _update_lock.acquire():
        print(describe_holder(_update_lock.holder))
        _finalize_update_output(_update_io_state)
        sys.exit(UPDATE_EXIT_CONCURRENT)

    # Exit code for the Windows hand-off child's hard exit (see finally).
    # None = not a SystemExit-shaped outcome; real exceptions keep the
    # normal raise path so their traceback still prints.
    _update_handoff_exit_code: int | None = None
    try:
        _self()._cmd_update_impl(args, gateway_mode=gateway_mode)
    except SystemExit as _update_exit:
        # Receipt boundary (#91283 review): the impl has many early
        # sys.exit paths (concurrent-instance preflight, venv-holder
        # refusal, head-pinned no-op, fetch failure) that never reach an
        # inner finalize. Persist any still-open receipt with the real
        # exit code, then let the exit proceed unchanged. No-op when an
        # inner path already finalized (exactly-once by construction).
        try:
            from hermes_cli.update_receipt import finalize_pending_update_receipt

            _code = _update_exit.code if isinstance(_update_exit.code, int) else 1
            finalize_pending_update_receipt(_code, f"sys.exit({_code})")
        except Exception:
            pass
        _update_handoff_exit_code = (
            _update_exit.code if isinstance(_update_exit.code, int) else 0
        )
        raise
    except BaseException as _update_exc:
        try:
            from hermes_cli.update_receipt import finalize_pending_update_receipt

            finalize_pending_update_receipt(
                1, f"{type(_update_exc).__name__}: {_update_exc}"
            )
        except Exception:
            pass
        raise
    else:
        try:
            from hermes_cli.update_receipt import finalize_pending_update_receipt

            finalize_pending_update_receipt(0, "completed at command boundary")
        except Exception:
            pass
        _update_handoff_exit_code = 0
    finally:
        _update_lock.release()
        _finalize_update_output(_update_io_state)
        # Windows hand-off child (#93581): the re-exec'd venv child cannot
        # rely on graceful interpreter shutdown — a leftover non-daemon
        # thread from the update tail keeps the console busy long after
        # the receipt is durable (success, exit 0, "completed at command
        # boundary"), freezing the PowerShell window for minutes. By this
        # point every durable step is done (receipt finalized above, lock
        # released, stdio restored), so on the hand-off path only, flush
        # and exit hard instead of waiting for the interpreter to unwind
        # — the same treatment #79040's cron workaround applies. No-op on
        # every non-hand-off invocation: the marker env is set solely by
        # _reexec_dependency_sync_off_windows_shim when it spawns the child.
        if _update_handoff_exit_code is not None and os.environ.get(_UPDATE_REEXEC_ENV) == "1":
            logger.debug(
                "Update hand-off child %s exiting via os._exit(%s)",
                os.getpid(), _update_handoff_exit_code,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(_update_handoff_exit_code)


def _coalesce_session_name_args(argv: list) -> list:
    """Join unquoted multi-word session names after -c/--continue and -r/--resume.

    When a user types ``hermes -c Pokemon Agent Dev`` without quoting the
    session name, argparse sees three separate tokens.  This function merges
    them into a single argument so argparse receives
    ``['-c', 'Pokemon Agent Dev']`` instead.

    Tokens are collected after the flag until we hit another flag (``-*``)
    or a known top-level subcommand.
    """
    _SUBCOMMANDS = {
        "chat",
        "model",
        "gateway",
        "setup",
        "whatsapp",
        "whatsapp-cloud",
        "login",
        "logout",
        "auth",
        "status",
        "cron",
        "doctor",
        "config",
        "pairing",
        "skills",
        "tools",
        "mcp",
        "sessions",
        "insights",
        "update",
        "uninstall",
        "profile",
        "dashboard",
        "serve",
        "desktop",
        "gui",
        "honcho",
        "claw",
        "plugins",
        "security",
        "acp",
        "webhook",
        "peer",
        "memory",
        "dump",
        "debug",
        "backup",
        "import",
        "completion",
        "logs",
    }
    _SESSION_FLAGS = {"-c", "--continue", "-r", "--resume"}

    result = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _SESSION_FLAGS:
            result.append(token)
            i += 1
            # Collect subsequent non-flag, non-subcommand tokens as one name
            parts: list = []
            while (
                i < len(argv)
                and not argv[i].startswith("-")
                and argv[i] not in _SUBCOMMANDS
            ):
                parts.append(argv[i])
                i += 1
            if parts:
                result.append(" ".join(parts))
        else:
            result.append(token)
            i += 1
    return result


def cmd_profile(args):
    """Profile management — create, delete, list, switch, alias."""
    from hermes_cli.profiles import (
        list_profiles,
        create_profile,
        delete_profile,
        seed_profile_skills,
        set_active_profile,
        get_active_profile_name,
        check_alias_collision,
        create_wrapper_script,
        remove_wrapper_script,
        _is_wrapper_dir_in_path,
        _get_wrapper_dir,
    )
    from hermes_constants import display_hermes_home

    action = getattr(args, "profile_action", None)

    if action is None:
        # Bare `hermes profile` — show current profile status
        from hermes_cli.profiles import format_profile_label

        profile_name = get_active_profile_name()
        dhh = display_hermes_home()

        profiles = list_profiles()
        current = next(
            (
                p
                for p in profiles
                if p.name == profile_name
                or (profile_name == "default" and p.is_default)
            ),
            None,
        )
        label = format_profile_label(
            profile_name, current.display_name if current else ""
        )
        print(f"\nActive profile: {label}")
        print(f"Path:           {dhh}")

        if current is not None:
            p = current
            if p.model:
                print(
                    f"Model:          {p.model}"
                    + (f" ({p.provider})" if p.provider else "")
                )
            print(
                f"Gateway:        {'running' if p.gateway_running else 'stopped'}"
            )
            print(f"Skills:         {p.skill_count} installed")
            if p.alias_path:
                alias_display = p.alias_name or p.name
                print(f"Alias:          {alias_display} → hermes -p {p.name}")
        print()
        return

    if action == "list":
        from hermes_cli.profiles import format_profile_label

        profiles = list_profiles()
        active = get_active_profile_name()

        if not profiles:
            print("No profiles found.")
            return

        # Header
        print(
            f"\n {'Profile':<16} {'Model':<28} {'Gateway':<12} "
            f"{'Alias':<12} {'Distribution'}"
        )
        print(
            f" {'─' * 15}    {'─' * 27}    {'─' * 11}    "
            f"{'─' * 11}    {'─' * 20}"
        )

        for p in profiles:
            marker = (
                " ◆"
                if (p.name == active or (active == "default" and p.is_default))
                else "  "
            )
            name = format_profile_label(p.name, p.display_name)
            model = (p.model or "—")[:26]
            gw = "running" if p.gateway_running else "stopped"
            alias = (p.alias_name or p.name) if p.alias_path else "—"
            if p.is_default:
                alias = "—"
            if p.distribution_name:
                dist = f"{p.distribution_name}@{p.distribution_version or '?'}"
                dist = dist[:30]
            else:
                dist = "—"
            print(f"{marker}{name:<15} {model:<28} {gw:<12} {alias:<12} {dist}")
        print()

    elif action == "use":
        name = args.profile_name
        try:
            set_active_profile(name)
            if name == "default":
                print("Switched to: default (~/.hermes)")
            else:
                print(f"Switched to: {name}")
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "create":
        name = args.profile_name
        clone = getattr(args, "clone", False)
        clone_all = getattr(args, "clone_all", False)
        no_alias = getattr(args, "no_alias", False)
        no_skills = getattr(args, "no_skills", False)

        try:
            clone_from = getattr(args, "clone_from", None)
            clone_config = clone or clone_from is not None

            profile_dir = create_profile(
                name=name,
                clone_from=clone_from,
                clone_all=clone_all,
                clone_config=clone_config,
                no_alias=no_alias,
                no_skills=no_skills,
                description=getattr(args, "description", None),
            )
            print(f"\nProfile '{name}' created at {profile_dir}")

            if clone_config or clone_all:
                source_label = (
                    getattr(args, "clone_from", None) or get_active_profile_name()
                )
                if clone_all:
                    print(
                        f"Full copy from {source_label} "
                        "(excluding session history, backups, and snapshots)."
                    )
                else:
                    print(
                        f"Cloned config, .env, SOUL.md, and skills from {source_label}."
                    )

            # Auto-clone Honcho config for the new profile (only with clone operations)
            if clone_config or clone_all:
                try:
                    from plugins.memory.honcho.cli import clone_honcho_for_profile

                    if clone_honcho_for_profile(name):
                        print(f"Honcho config cloned (peer: {name})")
                except Exception:
                    pass  # Honcho plugin not installed or not configured

            # Seed bundled skills for fresh profiles only. Clone operations
            # already copied the source profile's skills, including any
            # user-installed or intentionally removed skills.
            if not (clone_config or clone_all):
                result = seed_profile_skills(profile_dir)
                if result and result.get("skipped_opt_out"):
                    print(
                        "No bundled skills seeded (--no-skills). "
                        "Delete .no-bundled-skills in the profile to opt back in."
                    )
                elif result:
                    copied = len(result.get("copied", []))
                    print(f"{copied} bundled skills synced.")
                else:
                    print(
                        "⚠ Skills could not be seeded. Run `{} update` to retry.".format(
                            name
                        )
                    )

            # Create wrapper alias
            if not no_alias:
                collision = check_alias_collision(name)
                if collision:
                    print(f"\n⚠ Cannot create alias '{name}' — {collision}")
                    print(
                        f"  Choose a custom alias:  hermes profile alias {name} --name <custom>"
                    )
                    print(f"  Or access via flag:     hermes -p {name} chat")
                else:
                    wrapper_path = create_wrapper_script(name)
                    if wrapper_path:
                        print(f"Wrapper created: {wrapper_path}")
                        if not _is_wrapper_dir_in_path():
                            print(f"\n⚠ {_get_wrapper_dir()} is not in your PATH.")
                            print(
                                "  Add to your shell config (~/.bashrc or ~/.zshrc):"
                            )
                            print('    export PATH="$HOME/.local/bin:$PATH"')

            # Profile dir for display
            try:
                profile_dir_display = "~/" + profile_dir.relative_to(Path.home()).as_posix()
            except ValueError:
                profile_dir_display = str(profile_dir)

            # Next steps
            print("\nNext steps:")
            print(f"  {name} setup              Configure API keys and model")
            print(f"  {name} chat               Start chatting")
            print(f"  {name} gateway start      Start the messaging gateway")
            if clone or clone_all:
                print(f"\n  Edit {profile_dir_display}/.env for different API keys")
                print(f"  Edit {profile_dir_display}/SOUL.md for different personality")
            else:
                print(
                    f"\n  ⚠ This profile has no API keys yet. Run '{name} setup' first,"
                )
                print("    or it will inherit keys from your shell environment.")
                print(f"  Edit {profile_dir_display}/SOUL.md to customize personality")
            print()

        except (ValueError, FileExistsError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "delete":
        name = args.profile_name
        yes = getattr(args, "yes", False)
        try:
            delete_profile(name, yes=yes)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "describe":
        # Read or write a profile's description. The description is
        # consumed by the kanban decomposer to route tasks based on
        # role instead of name alone.
        from hermes_cli import profiles as _profiles_mod

        all_flag = bool(getattr(args, "all_missing", False))
        auto_flag = bool(getattr(args, "auto", False))
        overwrite_flag = bool(getattr(args, "overwrite", False))
        text_value = getattr(args, "text", None)
        name = getattr(args, "profile_name", None)

        if all_flag and not auto_flag:
            print("profile describe: --all requires --auto", file=sys.stderr)
            sys.exit(2)
        if all_flag and (text_value or name):
            print(
                "profile describe: --all is mutually exclusive with a profile name / --text",
                file=sys.stderr,
            )
            sys.exit(2)
        if not all_flag and not name:
            print("profile describe: profile name is required (or --all --auto)", file=sys.stderr)
            sys.exit(2)
        if text_value and auto_flag:
            print(
                "profile describe: --text is mutually exclusive with --auto",
                file=sys.stderr,
            )
            sys.exit(2)

        # Show current description if no operation requested.
        if name and not text_value and not auto_flag:
            try:
                if _profiles_mod.normalize_profile_name(name) == "default":
                    from hermes_constants import get_hermes_home as _hh
                    profile_dir = Path(_hh())
                else:
                    profile_dir = _profiles_mod.get_profile_dir(name)
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            if not profile_dir.is_dir():
                print(f"Error: profile '{name}' not found", file=sys.stderr)
                sys.exit(1)
            meta = _profiles_mod.read_profile_meta(profile_dir)
            desc = meta.get("description") or ""
            if not desc:
                print(f"(no description set for '{name}')")
            else:
                tag = "[auto] " if meta.get("description_auto") else ""
                print(f"{tag}{desc}")
            sys.exit(0)

        # --text path: just write the user-authored description.
        if text_value:
            try:
                if _profiles_mod.normalize_profile_name(name) == "default":
                    from hermes_constants import get_hermes_home as _hh
                    profile_dir = Path(_hh())
                else:
                    profile_dir = _profiles_mod.get_profile_dir(name)
                _profiles_mod.write_profile_meta(
                    profile_dir,
                    description=text_value,
                    description_auto=False,
                )
                print(f"Description updated for '{name}'.")
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            sys.exit(0)

        # --auto path: invoke the LLM describer.
        from hermes_cli import profile_describer as _pd

        if all_flag:
            targets = _pd.list_describable_profiles(missing_only=True)
            if not targets:
                print("All profiles already have descriptions.")
                sys.exit(0)
        else:
            targets = [name]

        ok_count = 0
        fail_count = 0
        for tgt in targets:
            outcome = _pd.describe_profile(tgt, overwrite=overwrite_flag)
            if outcome.ok:
                ok_count += 1
                print(f"Described '{outcome.profile_name}': {outcome.description}")
            else:
                fail_count += 1
                print(
                    f"profile describe {outcome.profile_name}: {outcome.reason}",
                    file=sys.stderr,
                )
        if not all_flag:
            sys.exit(0 if ok_count == 1 else 1)
        sys.exit(0 if ok_count > 0 else 1)

    elif action == "show":
        name = args.profile_name
        from hermes_cli.profiles import (
            get_profile_dir,
            profile_exists,
            _read_config_model,
            _check_gateway_running,
            _count_skills,
            _read_distribution_meta,
            _get_wrapper_dir,
            find_alias_for_profile,
            format_profile_label,
            read_profile_meta,
        )

        if not profile_exists(name):
            print(f"Error: Profile '{name}' does not exist.")
            sys.exit(1)
        profile_dir = get_profile_dir(name)
        model, provider = _read_config_model(profile_dir)
        gw = _check_gateway_running(profile_dir)
        skills = _count_skills(profile_dir)
        dist_name, dist_version, dist_source = _read_distribution_meta(profile_dir)
        alias_name = find_alias_for_profile(name)
        display = read_profile_meta(profile_dir).get("display_name", "")

        print(f"\nProfile: {format_profile_label(name, display)}")
        print(f"Path:    {profile_dir}")
        if model:
            print(f"Model:   {model}" + (f" ({provider})" if provider else ""))
        print(f"Gateway: {'running' if gw else 'stopped'}")
        print(f"Skills:  {skills}")
        print(
            f".env:    {'exists' if (profile_dir / '.env').exists() else 'not configured'}"
        )
        print(
            f"SOUL.md: {'exists' if (profile_dir / 'SOUL.md').exists() else 'not configured'}"
        )
        if dist_name:
            print(f"Distribution: {dist_name}@{dist_version or '?'}")
            if dist_source:
                print(f"Installed from: {dist_source}")
            print(f"  (run `hermes profile info {name}` for full manifest)")
        if alias_name:
            is_windows = sys.platform == "win32"
            wrapper = _get_wrapper_dir() / (f"{alias_name}.bat" if is_windows else alias_name)
            print(f"Alias:   {alias_name} → hermes -p {name}  ({wrapper})")
        print()

    elif action == "alias":
        name = args.profile_name
        remove = getattr(args, "remove", False)
        custom_name = getattr(args, "alias_name", None)

        from hermes_cli.profiles import profile_exists, validate_alias_name

        if not profile_exists(name):
            print(f"Error: Profile '{name}' does not exist.")
            sys.exit(1)

        alias_name = custom_name or name

        try:
            validate_alias_name(alias_name)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

        if remove:
            if remove_wrapper_script(alias_name):
                print(f"✓ Removed alias '{alias_name}'")
            else:
                print(f"No alias '{alias_name}' found to remove.")
        else:
            collision = check_alias_collision(alias_name)
            if collision:
                print(f"Error: {collision}")
                sys.exit(1)
            wrapper_path = create_wrapper_script(
                alias_name, target=name if custom_name else None
            )
            if wrapper_path:
                print(f"✓ Alias created: {wrapper_path}")
                if not _is_wrapper_dir_in_path():
                    print(f"⚠ {_get_wrapper_dir()} is not in your PATH.")

    elif action == "rename":
        from hermes_cli.profiles import normalize_profile_name, rename_profile

        try:
            new_dir = rename_profile(args.old_name, args.new_name)
            if normalize_profile_name(args.old_name) != "default":
                print(f"\nProfile renamed: {args.old_name} → {args.new_name}")
                print(f"Path: {new_dir}\n")
        except (ValueError, FileExistsError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "export":
        from hermes_cli.profiles import export_profile

        name = args.profile_name
        output = args.output or f"{name}.tar.gz"
        try:
            result_path = export_profile(name, output)
            print(f"✓ Exported '{name}' to {result_path}")
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "import":
        from hermes_cli.profiles import import_profile

        try:
            profile_dir = import_profile(
                args.archive, name=getattr(args, "import_name", None)
            )
            name = profile_dir.name
            print(f"✓ Imported profile '{name}' at {profile_dir}")

            # Offer to create alias
            collision = check_alias_collision(name)
            if not collision:
                wrapper_path = create_wrapper_script(name)
                if wrapper_path:
                    print(f"  Wrapper created: {wrapper_path}")
            print()
        except (ValueError, FileExistsError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "install":
        import tempfile
        from hermes_cli.profile_distribution import (
            plan_install,
            install_distribution,
            DistributionError,
        )

        try:
            # Preview: stage the distribution into a scratch dir, show the
            # manifest, then do the real install.  The double-stage avoids
            # any side-effects if the user declines.
            with tempfile.TemporaryDirectory(prefix="hermes_dist_preview_") as tmp:
                plan = plan_install(
                    args.source,
                    Path(tmp),
                    override_name=getattr(args, "install_name", None),
                )
                _render_distribution_plan(plan)

                if not getattr(args, "yes", False):
                    try:
                        answer = input("\nProceed with install? [y/N] ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        answer = ""
                    if answer not in {"y", "yes"}:
                        print("Install cancelled.")
                        return

            plan = install_distribution(
                args.source,
                name=getattr(args, "install_name", None),
                force=getattr(args, "force", False),
                create_alias=getattr(args, "alias", False),
            )
            print(f"\n✓ Installed '{plan.manifest.name}' v{plan.manifest.version}")
            print(f"  Profile path: {plan.target_dir}")
            if plan.manifest.env_requires:
                print(
                    f"  Next: copy .env.EXAMPLE to .env and fill in required keys:\n"
                    f"    {plan.target_dir}/.env.EXAMPLE"
                )
            if plan.has_cron:
                print(
                    "  Cron jobs were included but are NOT scheduled automatically.\n"
                    f"  Review them with:  hermes -p {plan.manifest.name} cron list"
                )
            print(f"\n  Use with:      hermes -p {plan.manifest.name} chat")
        except (DistributionError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "update":
        from hermes_cli.profile_distribution import (
            update_distribution,
            read_manifest,
            DistributionError,
        )
        from hermes_cli.profiles import get_profile_dir, normalize_profile_name

        name = args.profile_name
        try:
            canon = normalize_profile_name(name)
            current = read_manifest(get_profile_dir(canon))
            if current is None:
                print(
                    f"Error: Profile '{canon}' is not a distribution (no distribution.yaml). "
                    "Only profiles installed via `hermes profile install` can be updated."
                )
                sys.exit(1)

            force_config = getattr(args, "force_config", False)
            if not getattr(args, "yes", False):
                print(f"\nUpdate '{canon}' from: {current.source or '(no source)'}")
                print(f"  Currently at version {current.version}")
                if force_config:
                    print("  --force-config set: config.yaml WILL be overwritten.")
                else:
                    print("  config.yaml will be preserved (pass --force-config to overwrite).")
                print("  User data (memories, sessions, auth, .env) will NOT be touched.")
                try:
                    answer = input("\nProceed? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = ""
                if answer not in {"y", "yes"}:
                    print("Update cancelled.")
                    return

            plan = update_distribution(canon, force_config=force_config)
            print(f"\n✓ Updated '{plan.manifest.name}' → v{plan.manifest.version}")
            if plan.has_cron:
                print(
                    "  Cron files were refreshed.  Review with:  "
                    f"hermes -p {plan.manifest.name} cron list"
                )
        except (DistributionError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "info":
        from hermes_cli.profile_distribution import describe_distribution, DistributionError

        try:
            data = describe_distribution(args.profile_name)
        except (DistributionError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        if not data:
            print(
                f"Profile '{args.profile_name}' is not a distribution "
                "(no distribution.yaml)."
            )
            return
        print(f"\nDistribution: {data.get('name')}")
        print(f"Version:      {data.get('version', '?')}")
        if data.get("description"):
            print(f"Description:  {data['description']}")
        if data.get("author"):
            print(f"Author:       {data['author']}")
        if data.get("license"):
            print(f"License:      {data['license']}")
        if data.get("hermes_requires"):
            print(f"Requires:     Hermes {data['hermes_requires']}")
        if data.get("source"):
            print(f"Source:       {data['source']}")
        if data.get("installed_at"):
            print(f"Installed:    {data['installed_at']}")
        env_reqs = data.get("env_requires") or []
        if env_reqs:
            print("\nEnvironment variables:")
            for er in env_reqs:
                tag = "required" if er.get("required", True) else "optional"
                line = f"  {er['name']} ({tag})"
                if er.get("description"):
                    line += f" — {er['description']}"
                print(line)
                if er.get("default") is not None:
                    print(f"      default: {er['default']}")
        print()


def _render_distribution_plan(plan) -> None:
    """Print a human-readable summary of a pending distribution install."""
    from hermes_cli.profile_distribution import MANIFEST_FILENAME
    mf = plan.manifest
    print(f"\nDistribution: {mf.name} v{mf.version}")
    if mf.description:
        print(f"  {mf.description}")
    if mf.author:
        print(f"  Author:   {mf.author}")
    if mf.hermes_requires:
        print(f"  Requires: Hermes {mf.hermes_requires}")
    print(f"  Source:   {plan.provenance}")
    print(f"  Target:   {plan.target_dir}")
    if plan.existing:
        # Distinguish "updating an existing distribution" (well-understood
        # semantics — dist-owned overwritten, config preserved, user data
        # untouched) from "overwriting a hand-built plain profile" (same
        # mechanics but the user didn't sign up for this when they created
        # the profile manually).
        existing_is_distribution = (plan.target_dir / MANIFEST_FILENAME).is_file()
        if existing_is_distribution:
            print("  (profile exists — will overwrite distribution-owned files only)")
        else:
            print(
                "  ⚠ Profile exists but is NOT a distribution.  Installing here will\n"
                "    overwrite its SOUL.md, skills/, cron/, and mcp.json.\n"
                "    Your memories, sessions, auth.json, and .env will be preserved,\n"
                "    but any hand-edits to distribution-owned files will be lost."
            )
    if mf.env_requires:
        print("\n  Env vars:")
        for er in mf.env_requires:
            tag = "required" if er.required else "optional"
            # Check both the current shell environment and the target profile's
            # .env file so we don't nag about keys the user already has set up.
            already = os.environ.get(er.name) is not None
            if not already and plan.target_dir.is_dir():
                env_path = plan.target_dir / ".env"
                if env_path.is_file():
                    try:
                        # .env is written as UTF-8 everywhere in the codebase,
                        # but a Notepad-edited file can carry a BOM — read as
                        # utf-8-sig so the first key isn't hidden behind
                        # U+FEFF (#62617).
                        for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
                            line = raw.strip()
                            if not line or line.startswith("#"):
                                continue
                            key = line.split("=", 1)[0].strip()
                            if key == er.name:
                                already = True
                                break
                    except (OSError, UnicodeDecodeError):
                        # UnicodeDecodeError is a ValueError, not an OSError, so
                        # the old guard let a mis-encoded .env abort the whole
                        # install preview. Skip the pre-check instead.
                        pass
            status = "✓ set" if already else ("needs setting" if er.required else "—")
            line = f"    • {er.name} ({tag}, {status})"
            if er.description:
                line += f" — {er.description}"
            print(line)
    if plan.has_cron:
        print(
            "\n  ⚠ This distribution ships cron jobs.  They will NOT run "
            "automatically — review and enable manually."
        )


def _report_dashboard_status() -> int:
    """Print live listening dashboard/serve processes and return the count.

    Serve-mode backends are INCLUDED (#81564): `--stop` kills them, so
    `--status` hiding them left Desktop SSH backends invisible to the CLI —
    an operator could kill what they couldn't see. Ledger-registered serves
    (profiled launches the argv scan can't match) surface via the
    spawn-ledger augmentation in _scan_dashboard_processes.
    """
    from gateway.status import _pid_exists

    live: list[tuple[int, str, str]] = []
    for pid, command in _self()._scan_dashboard_processes():
        runtime = _parse_dashboard_runtime(command)
        if runtime is None:
            continue
        mode, host, port = runtime
        if port <= 0 or not _pid_exists(pid):
            continue
        if not _dashboard_listening(host, port):
            continue
        live.append((pid, command, mode))

    if not live:
        print("No hermes dashboard or serve processes running.")
        return 0

    print(f"{len(live)} hermes dashboard/serve process(es) running:")
    for pid, command, mode in live:
        print(f"    PID {pid} [{mode}]: {command}")
    return len(live)


def _dashboard_listening(host: str, port: int) -> bool:
    """True when something is accepting TCP connections at host:port.

    Any listener counts — even a 401 response proves a dashboard is up.
    Used by the unified profile-launch routing to decide attach-vs-start.
    """
    import socket

    try:
        with socket.create_connection((_dashboard_probe_host(host), port), timeout=1.5):
            return True
    except OSError:
        return False


def _maybe_setup_dashboard_auth_interactively(args) -> None:
    """Offer to configure dashboard auth when the gate engages and none exists.

    Called from ``cmd_dashboard`` just before ``start_server``. The auth
    gate engages on every non-loopback bind (``--insecure`` is a no-op since
    the June 2026 hardening) and whenever ``dashboard.public_url`` declares a
    non-loopback browser-facing hostname. ``start_server`` fails closed when no
    ``DashboardAuthProvider`` is registered. Rather than greet an interactive
    operator with that hard error, prompt them to set up the bundled password
    provider on the spot — or point them at ``hermes dashboard register`` for
    OAuth.

    No-ops (so the existing fail-closed ``SystemExit`` remains the backstop)
    when:
      * neither the bind nor configured public URL engages the gate, or
      * a provider is already registered, or
      * stdin/stdout isn't a TTY (Docker/s6, CI, piped ``--no-open`` runs).
    """
    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"

    try:
        from hermes_cli.web_server import should_require_dashboard_auth
        if not should_require_dashboard_auth(host):
            return  # local-only bind and URL — gate does not engage
    except Exception:
        return  # if we can't tell, defer to start_server's own gate

    try:
        from hermes_cli.dashboard_auth import list_providers
        if list_providers():
            return  # a provider is already configured/registered
    except Exception:
        return

    # Only prompt an interactive operator. Non-TTY callers fall through to
    # start_server's fail-closed SystemExit (with the corrected fix hint).
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return

    print()
    print(f"⚠ Dashboard authentication is required for this configuration ({host}).")
    print(
        "  Non-loopback binds and configured external dashboard.public_url "
        "values require authentication (--insecure does not bypass this)."
    )
    print()
    print("  How do you want to authenticate the dashboard?")
    print("    [1] Username & password (quickest; for a trusted LAN / VPN)")
    print("    [2] OAuth via Nous Portal (run `hermes dashboard register`)")
    print("    [3] Cancel")
    print()

    try:
        choice = input("  Choice [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        sys.exit(1)

    if choice == "2":
        print()
        print(
            "  Run this on the host where the dashboard lives, then start "
            "the dashboard again:\n"
            "    hermes dashboard register\n"
            "  It provisions a Nous Portal OAuth client and writes "
            "HERMES_DASHBOARD_OAUTH_CLIENT_ID into ~/.hermes/.env for you.\n"
            "  Docs: https://hermes-agent.nousresearch.com/docs/"
            "user-guide/features/web-dashboard#authentication-gated-mode"
        )
        sys.exit(0)

    if choice not in ("1",):
        print("  Cancelled.")
        sys.exit(1)

    # ── Username/password setup ──────────────────────────────────────────
    import getpass
    import secrets

    print()
    try:
        username = line_input("  Username [admin]: ").strip() or "admin"
        password = getpass.getpass("  Password: ")
        confirm = getpass.getpass("  Confirm password: ")
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        sys.exit(1)

    if not password:
        print("  ✗ Empty password — aborting.")
        sys.exit(1)
    if password != confirm:
        print("  ✗ Passwords don't match — aborting.")
        sys.exit(1)

    try:
        from plugins.dashboard_auth.basic import hash_password
    except Exception as exc:
        print(f"  ✗ Could not load the password provider: {exc}")
        sys.exit(1)

    password_hash = hash_password(password)
    # A stable token-signing secret so sessions survive a dashboard restart.
    secret = secrets.token_urlsafe(32)

    try:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.plugins_cmd import ensure_basic_auth_plugin_enabled_in_config

        cfg = load_config()
        dash = cfg.setdefault("dashboard", {})
        basic = dash.setdefault("basic_auth", {})
        basic["username"] = username
        basic["password_hash"] = password_hash
        # Never persist plaintext: clear any stale plaintext password key.
        basic["password"] = ""
        if not str(basic.get("secret", "") or "").strip():
            basic["secret"] = secret
        # The bundled basic provider is a backend plugin that still honours
        # plugins.disabled. Unblock it when we just wrote basic_auth so the
        # discover_plugins(force=True) call below can register the provider
        # (#54489). Surface the mutation so an operator who deliberately
        # disabled it isn't surprised.
        if ensure_basic_auth_plugin_enabled_in_config(cfg):
            print(
                "  ✓ Re-enabled the bundled 'basic' auth plugin "
                "(was in plugins.disabled)"
            )
        save_config(cfg)
    except Exception as exc:
        print(f"  ✗ Failed to write config.yaml: {exc}")
        sys.exit(1)

    # Re-run plugin discovery so the basic provider registers from the
    # just-written config before start_server's gate check runs.
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins(force=True)
    except Exception as exc:
        print(f"  ⚠ Plugin re-discovery failed ({exc}); the gate may still "
              "fail closed. Set the password again or restart the dashboard.")

    print()
    print(f"  ✓ Username/password auth configured (user: {username}).")
    print("    Saved to config.yaml under dashboard.basic_auth.")
    print("    Sign in at the dashboard with these credentials.")
    print()


def _maybe_setup_desktop_web_auth_interactively(args) -> None:
    """Offer Dashboard-style first-run password setup for Desktop Web.

    Desktop Web has its own host-level gate because its backend is a private,
    launcher-owned child and must not borrow Dashboard sessions.  Keep the
    first-run experience familiar, though: when a public authority or a
    non-loopback bind means the gate will engage, an interactive operator can
    configure the bundled username/password provider before the Node host
    starts.  Non-TTY launches deliberately fall through to the Node host's
    fail-closed response.
    """
    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    loopback_hosts = {"localhost", "127.0.0.1", "::1"}

    try:
        from urllib.parse import urlparse

        from hermes_cli.config import load_config, save_config

        config = load_config()
        desktop_web = config.get("desktop_web")
        if not isinstance(desktop_web, dict):
            desktop_web = {}
        basic_auth = desktop_web.get("basic_auth")
        if not isinstance(basic_auth, dict):
            basic_auth = {}

        configured_public_url = (
            os.environ.get("HERMES_DESKTOP_WEB_PUBLIC_URL", "").strip()
            or str(desktop_web.get("public_url", "") or "").strip()
        )
        public_hostname = ""
        if configured_public_url:
            parsed = urlparse(configured_public_url)
            if parsed.scheme in {"http", "https"}:
                public_hostname = (parsed.hostname or "").lower()

        auth_required = host.strip("[]").lower() not in loopback_hosts
        auth_required = auth_required or (
            public_hostname and public_hostname not in loopback_hosts
        )
        if not auth_required:
            return

        env_username = os.environ.get("HERMES_DESKTOP_WEB_BASIC_AUTH_USERNAME", "").strip()
        env_password = os.environ.get("HERMES_DESKTOP_WEB_BASIC_AUTH_PASSWORD", "").strip()
        env_hash = os.environ.get("HERMES_DESKTOP_WEB_BASIC_AUTH_PASSWORD_HASH", "").strip()
        username = env_username or str(basic_auth.get("username", "") or "").strip()
        has_password = bool(
            env_password
            or env_hash
            or str(basic_auth.get("password_hash", "") or "").strip()
            or str(basic_auth.get("password", "") or "").strip()
        )
        if username and has_password:
            return
    except Exception:
        # Do not make startup less safe when config discovery fails.  The Node
        # host remains the final fail-closed authority.
        return

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return

    print()
    print(f"⚠ Desktop Web authentication is required for this configuration ({host}).")
    print(
        "  A public URL or non-loopback bind requires a username and password."
    )
    print()
    print("  Configure Desktop Web username/password now?")
    print("    [1] Username & password (recommended)")
    print("    [2] Cancel")
    print()

    try:
        choice = input("  Choice [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        sys.exit(1)
    if choice != "1":
        print("  Cancelled.")
        sys.exit(1)

    import getpass

    print()
    try:
        username = line_input("  Username [admin]: ").strip() or "admin"
        password = getpass.getpass("  Password: ")
        confirm = getpass.getpass("  Confirm password: ")
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        sys.exit(1)

    if not password:
        print("  ✗ Empty password — aborting.")
        sys.exit(1)
    if password != confirm:
        print("  ✗ Passwords don't match — aborting.")
        sys.exit(1)

    try:
        from plugins.dashboard_auth.basic import hash_password

        password_hash = hash_password(password)
        basic_auth["username"] = username
        basic_auth["password_hash"] = password_hash
        # Never persist plaintext, even though the server supports it as a
        # compatibility fallback for manually managed configurations.
        basic_auth["password"] = ""
        if not str(basic_auth.get("secret", "") or "").strip():
            basic_auth["secret"] = secrets.token_urlsafe(32)
        desktop_web["basic_auth"] = basic_auth
        config["desktop_web"] = desktop_web
        save_config(config)
    except Exception as exc:
        print(f"  ✗ Failed to configure Desktop Web authentication: {exc}")
        sys.exit(1)

    print()
    print(f"  ✓ Username/password auth configured (user: {username}).")
    print("    Saved to config.yaml under desktop_web.basic_auth.")
    print("    Start Desktop Web again if this process was not launched interactively.")
    print()


def _read_ssh_session_token_file(path: str) -> str:
    """Read and unlink a Desktop SSH token from its private runtime directory."""
    if sys.platform == "win32":
        from hermes_cli.windows_ssh_runtime import read_token
        return read_token(path)

    import stat as _stat
    from pathlib import Path as _Path

    if not os.path.isabs(path):
        raise SystemExit("--ssh-session-token-file must be absolute")

    token_path = _Path(path)
    # The Desktop client writes the token under $HOME/.hermes/desktop-ssh: a
    # literal "~/.hermes/desktop-ssh" in apps/desktop/electron/remote-lifecycle.ts
    # expanded against the account's $HOME, independent of HERMES_HOME and the
    # active profile. Anchor validation to that same OS-home path, NOT to
    # get_hermes_home(): a non-default sticky profile (or any HERMES_HOME pointing
    # elsewhere, e.g. a Docker /opt/data root) re-homes get_hermes_home() and
    # would otherwise reject every token the client legitimately wrote (#69551).
    token_root = _Path.home() / ".hermes" / "desktop-ssh"
    try:
        relative = token_path.relative_to(token_root)
    except ValueError as exc:
        raise SystemExit("--ssh-session-token-file must be under the desktop-ssh directory") from exc
    if len(relative.parts) != 2 or not re.fullmatch(r"[0-9a-f]{32}", relative.parts[0]):
        raise SystemExit("--ssh-session-token-file has an invalid runtime path")
    if not re.fullmatch(r"[0-9a-f]{16}\.token", relative.parts[1]):
        raise SystemExit("--ssh-session-token-file has an invalid filename")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    root_fd = -1
    directory_fd = -1
    file_fd = -1
    try:
        try:
            root_fd = os.open(token_root, directory_flags)
            root_stat = os.fstat(root_fd)
            if not _stat.S_ISDIR(root_stat.st_mode):
                raise SystemExit("--ssh-session-token-file has an unsafe runtime root")
            if hasattr(os, "getuid") and root_stat.st_uid != os.getuid():
                raise SystemExit("--ssh-session-token-file runtime root has the wrong owner")
            directory_fd = os.open(relative.parts[0], directory_flags, dir_fd=root_fd)
            directory_stat = os.fstat(directory_fd)
            if not _stat.S_ISDIR(directory_stat.st_mode):
                raise SystemExit("--ssh-session-token-file has an unsafe parent directory")
            if hasattr(os, "getuid") and directory_stat.st_uid != os.getuid():
                raise SystemExit("--ssh-session-token-file parent has the wrong owner")
            if (directory_stat.st_mode & 0o777) != 0o700:
                raise SystemExit("--ssh-session-token-file parent has unsafe permissions")
            file_fd = os.open(relative.parts[1], file_flags, dir_fd=directory_fd)
        except SystemExit:
            raise
        except OSError as exc:
            if exc.errno == getattr(__import__("errno"), "ELOOP", -1):
                raise SystemExit("--ssh-session-token-file is a symlink") from exc
            raise SystemExit("--ssh-session-token-file is not accessible") from exc

        file_stat = os.fstat(file_fd)
        if not _stat.S_ISREG(file_stat.st_mode):
            raise SystemExit("--ssh-session-token-file is not a regular file")
        if file_stat.st_size != 64:
            raise SystemExit("--ssh-session-token-file contains an invalid token")
        if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
            raise SystemExit("--ssh-session-token-file has the wrong owner")
        if hasattr(os, "getuid") and (file_stat.st_mode & 0o777) & ~0o600:
            raise SystemExit("--ssh-session-token-file has unsafe permissions")

        with os.fdopen(file_fd, "r", encoding="utf-8") as token_stream:
            file_fd = -1
            token = token_stream.read(65)

        if not re.fullmatch(r"[0-9a-f]{64}", token):
            raise SystemExit("--ssh-session-token-file contains an invalid token")
        return token
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd >= 0:
            try:
                os.unlink(relative.parts[1], dir_fd=directory_fd)
            except OSError:
                pass
            os.close(directory_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _is_electron_packaged_web_dist(path: str) -> bool:
    """True when *path* looks like an Electron-packaged renderer dist.

    Packaged Desktop sets ``HERMES_WEB_DIST`` to ``.../app.asar/dist`` or
    ``.../app.asar.unpacked/dist``. A standalone ``hermes dashboard`` that
    inherits that value serves the desktop frontend in the browser
    (issue #52945 — "Desktop IPC bridge is unavailable").
    """
    if not path:
        return False
    # Both app.asar and app.asar.unpacked contain this marker; normalize
    # separators so Windows paths match too.
    return "app.asar" in path.replace("\\", "/")


def _desktop_web_processes() -> list[tuple[int, str]]:
    """Return processes launched through the canonical desktop-web command."""
    try:
        import psutil
    except Exception:
        return []

    current = os.getpid()
    candidates = {}
    for process in psutil.process_iter(["pid", "ppid", "cmdline"]):
        try:
            argv = process.info.get("cmdline") or []
            candidates[process.info["pid"]] = (process.info.get("ppid"), argv, process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    roots = set()
    for pid, (_ppid, argv, _process) in candidates.items():
        if pid == current:
            continue
        is_launcher = "-m" in argv and "hermes_cli.main" in argv and "desktop-web" in argv
        is_server = any(str(item).endswith("apps/desktop/web/server.mjs") for item in argv) and "--desktop-web" in argv
        if is_launcher or is_server:
            roots.add(pid)

    owned = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _argv, _process) in candidates.items():
            if ppid in owned and pid not in owned:
                owned.add(pid)
                changed = True

    return [(pid, " ".join(candidates[pid][1])) for pid in sorted(owned)]


def _stop_desktop_web_processes() -> bool:
    processes = _desktop_web_processes()
    if not processes:
        print("No hermes desktop-web processes running.")
        return True

    import signal
    import time

    for pid, _command in processes:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _desktop_web_processes():
        time.sleep(0.1)

    remaining = _desktop_web_processes()
    for pid, _command in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return not _desktop_web_processes()


def cmd_desktop_web(args):
    """Build and serve the standalone Desktop Web host."""
    web_dir = PROJECT_ROOT / "apps" / "desktop" / "web"
    dist_dir = web_dir / "dist"
    if not (web_dir / "package.json").exists():
        print(f"Desktop Web source not found at: {web_dir}", file=sys.stderr)
        return 1

    if args.status:
        processes = _desktop_web_processes()
        if not processes:
            print("No hermes desktop-web processes running.")
        else:
            for pid, command in processes:
                print(f"{pid}: {command}")
        return 0

    if args.stop:
        return 0 if _stop_desktop_web_processes() else 1

    if args.port < 0 or args.port > 65535:
        print("desktop-web port must be between 0 and 65535", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["HERMES_DESKTOP_WEB"] = "1"

    if args.skip_build:
        if not (dist_dir / "index.html").exists():
            print(f"✗ --skip-build was passed but no web dist found at: {dist_dir}", file=sys.stderr)
            print("  Run: npm run build --workspace apps/desktop/web", file=sys.stderr)
            return 1
        print(f"→ Skipping Desktop Web build; using {dist_dir}")
    else:
        npm = _resolve_node_runtime_npm()
        if not npm:
            print("Desktop Web requires Node.js/npm, but npm was not found on PATH.", file=sys.stderr)
            return 1
        from hermes_constants import with_hermes_node_path
        npm_env = _npm_lifecycle_env(with_hermes_node_path(env))
        print("→ Installing Desktop Web workspace dependencies...")
        install = subprocess.run(
            [npm, "install", "--workspace", "apps/desktop/web"],
            cwd=PROJECT_ROOT,
            env=npm_env,
            check=False,
        )
        if install.returncode != 0:
            print("✗ Desktop Web dependency install failed", file=sys.stderr)
            return install.returncode or 1
        print("→ Building Desktop Web renderer...")
        build = subprocess.run(
            [npm, "run", "build", "--workspace", "apps/desktop/web"],
            cwd=PROJECT_ROOT,
            env=npm_env,
            check=False,
        )
        if build.returncode != 0 or not (dist_dir / "index.html").exists():
            print("✗ Desktop Web build failed or produced no dist/index.html", file=sys.stderr)
            return build.returncode or 1

    # Match Dashboard's first-run behavior: interactive operators can create
    # the independent Desktop Web gate before the host starts. Background and
    # service launches remain fail-closed in server.mjs when it is unconfigured.
    _maybe_setup_desktop_web_auth_interactively(args)

    node = shutil.which("node")
    if not node:
        print("Desktop Web requires Node.js, but node was not found on PATH.", file=sys.stderr)
        return 1

    import selectors
    import time

    backend_env = os.environ.copy()
    backend_env.pop("HERMES_DESKTOP_WEB", None)
    backend_env.pop("HERMES_DESKTOP_WEB_PUBLIC_URL", None)
    backend_env.pop("HERMES_WEB_DIST", None)
    backend_env["HERMES_DESKTOP_WEB_CHILD"] = "1"
    # Keep the native Desktop backend contract: this credential is generated
    # for this child only and is never exposed to the browser-facing renderer.
    backend_token = secrets.token_hex(32)
    backend_env["HERMES_DASHBOARD_SESSION_TOKEN"] = backend_token
    backend_command = [
        sys.executable, "-m", "hermes_cli.main", "serve",
        "--host", "127.0.0.1", "--port", "0", "--no-open", "--isolated",
    ]
    print("→ Starting dedicated Hermes backend child...")
    backend = subprocess.Popen(
        backend_command,
        cwd=PROJECT_ROOT,
        env=backend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    backend_port = None
    backend_deadline = time.monotonic() + 90
    backend_selector = selectors.DefaultSelector()
    assert backend.stdout is not None
    backend_selector.register(backend.stdout, selectors.EVENT_READ)
    try:
        while time.monotonic() < backend_deadline:
            if backend.poll() is not None:
                break
            for key, _events in backend_selector.select(timeout=0.25):
                line = backend.stdout.readline()
                if not line:
                    continue
                print(f"[backend] {line.rstrip()}")
                match = re.search(r"HERMES_BACKEND_READY port=(\d+)", line)
                if match:
                    backend_port = int(match.group(1))
                    break
            if backend_port is not None:
                break
        if backend_port is None:
            print("✗ Dedicated Hermes backend did not become ready.", file=sys.stderr)
            return 1

        command = [
            node,
            str(web_dir / "server.mjs"),
            "--desktop-web",
            "--host", args.host,
            "--port", str(args.port),
            "--backend-url", f"http://127.0.0.1:{backend_port}",
        ]
        web_env = env.copy()
        # The Node host needs the token to authenticate only its owned backend
        # proxy. It remains process-local and is never included in page data.
        web_env["HERMES_DESKTOP_WEB_BACKEND_TOKEN"] = backend_token
        print(f"→ Starting standalone Desktop Web on {args.host}:{args.port}")
        web = subprocess.Popen(command, cwd=web_dir, env=web_env)
        try:
            if not args.no_open:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and web.poll() is None:
                    if _dashboard_listening(args.host, args.port):
                        try:
                            import webbrowser
                            webbrowser.open(f"http://{args.host}:{args.port}")
                        except Exception:
                            pass
                        break
                    time.sleep(0.1)
            return web.wait()
        except KeyboardInterrupt:
            web.terminate()
            return web.wait()
        finally:
            if web.poll() is None:
                web.terminate()
                try:
                    web.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    web.kill()
    finally:
        backend_selector.close()
        if backend.poll() is None:
            backend.terminate()
            try:
                backend.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend.kill()
                backend.wait()


def cmd_dashboard(args):
    """Start the web UI server, or (with --stop/--status) manage running ones."""
    _token_file = getattr(args, "ssh_session_token_file", None)
    if _token_file and (
        getattr(args, "status", False) or getattr(args, "stop", False)
    ):
        raise SystemExit("--ssh-session-token-file cannot be used with --status or --stop")

    # --status: report running dashboards and exit, no deps needed.
    if getattr(args, "status", False):
        count = _report_dashboard_status()
        sys.exit(0 if count == 0 else 0)  # status is informational, always 0

    # --stop: kill any running dashboards and exit, no deps needed.
    if getattr(args, "stop", False):
        pids = _find_stale_dashboard_pids()
        if not pids:
            print("No hermes dashboard processes running.")
            sys.exit(0)
        # Reuse the same SIGTERM-grace-SIGKILL path used after `hermes update`.
        _self()._kill_stale_dashboard_processes(reason="requested via --stop")
        # _kill_stale_dashboard_processes prints outcomes itself.  Exit 0 if
        # we killed at least one, 1 if they were all unkillable.
        remaining = _find_stale_dashboard_pids()
        sys.exit(1 if remaining else 0)

    # `serve` is the headless backend: no UI build, no SPA mount, neutral
    # ready sentinel. Resolved once and threaded through the re-exec, the
    # build gate, and start_server.
    _headless_backend = getattr(args, "headless_backend", False)
    _ssh_owner_nonce = getattr(args, "ssh_owner_nonce", None)
    if _ssh_owner_nonce and not re.fullmatch(r"[0-9a-f]{16}", _ssh_owner_nonce):
        raise SystemExit("--ssh-owner-nonce must be 16 lowercase hex characters")
    _ssh_session_token = None
    if _token_file and not _headless_backend:
        raise SystemExit("--ssh-session-token-file is only valid with hermes serve")

    # ── Sanitize Desktop-inherited env that hijacks a standalone launch ─
    # Desktop Electron spawns its backend with HERMES_DESKTOP=1 plus
    # HERMES_WEB_DIST=<packaged app.asar[/unpacked]/dist> (and often
    # HERMES_SERVE_HEADLESS=1 on the serve path). A shell that inherits
    # those vars then runs `hermes dashboard` would otherwise:
    #   - serve the desktop renderer → "Desktop IPC bridge is unavailable"
    #     (issue #52945), or
    #   - disable the SPA via inherited HERMES_SERVE_HEADLESS.
    # Only strip Electron-packaged WEB_DIST contamination — caller-managed
    # HERMES_WEB_DIST overrides (dev / custom builds) must still work.
    # The desktop-spawned backend itself (HERMES_DESKTOP=1) keeps its dist.
    # Intentionally headless `serve` re-sets HERMES_SERVE_HEADLESS below.
    if os.environ.get("HERMES_DESKTOP") != "1":
        _inherited_web_dist = os.environ.get("HERMES_WEB_DIST", "")
        if _is_electron_packaged_web_dist(_inherited_web_dist):
            os.environ.pop("HERMES_WEB_DIST", None)
    if not _headless_backend:
        os.environ.pop("HERMES_SERVE_HEADLESS", None)

    # ── Unified profile launch routing ────────────────────────────────
    # The dashboard is a MACHINE management surface: it can read/write any
    # profile via the per-request ?profile= scoping. Running one dashboard
    # per profile just fragments that (port collisions, N processes, and a
    # "which dashboard am I on?" guessing game). So when a NAMED profile
    # launches the dashboard (`worker dashboard` → HERMES_HOME points into
    # profiles/), default to the machine dashboard:
    #   - already running → open the browser at ?profile=<name> and exit
    #   - not running     → re-exec as the machine dashboard (pinned to the
    #     default profile so _apply_profile_override can't re-route through
    #     the sticky active_profile file) with the launching profile
    #     preselected in the UI's switcher.
    # `--isolated` opts out and preserves the old per-profile behavior.
    try:
        from hermes_cli.profiles import get_active_profile_name
        _launch_profile = get_active_profile_name()
    except Exception:
        _launch_profile = "default"

    if (
        _launch_profile not in ("default", "custom")
        and not getattr(args, "isolated", False)
        and not getattr(args, "open_profile", "")
        # Desktop pool backends are intentionally per-profile.
        and os.environ.get("HERMES_DESKTOP") != "1"
    ):
        url = f"http://{args.host or '127.0.0.1'}:{args.port}/?profile={_launch_profile}"
        if _dashboard_listening(args.host, args.port):
            print(f"Machine dashboard already running on port {args.port}.")
            print(f"  Managing profile '{_launch_profile}': {url}")
            if not args.no_open:
                try:
                    import webbrowser
                    webbrowser.open(url)
                except Exception:
                    pass
            sys.exit(0)

        print(
            f"Routing to the machine dashboard (profile '{_launch_profile}' "
            f"preselected). Use --isolated for a dedicated per-profile server."
        )
        reexec_argv = [
            sys.executable, "-m", "hermes_cli.main",
            "-p", "default",
            # Preserve the lean serve path across the re-exec so a named-profile
            # `serve` doesn't silently rebuild the UI as `dashboard`.
            "serve" if _headless_backend else "dashboard",
            "--port", str(args.port),
            "--host", args.host,
            "--open-profile", _launch_profile,
        ]
        if _ssh_owner_nonce:
            reexec_argv.extend(["--ssh-owner-nonce", _ssh_owner_nonce])
        if _token_file:
            reexec_argv.extend(["--ssh-session-token-file", _token_file])
        if args.no_open:
            reexec_argv.append("--no-open")
        if getattr(args, "insecure", False):
            reexec_argv.append("--insecure")
        if getattr(args, "skip_build", False):
            reexec_argv.append("--skip-build")
        from tools.environments.local import build_subprocess_env
        # Exact env preservation: HERMES_HOME is explicitly pinned to the
        # machine root below — the factory must not re-inject a profile home.
        env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
        # Pin the child to the machine ROOT, not the launching profile's
        # HERMES_HOME.  We must resolve the root explicitly instead of just
        # dropping HERMES_HOME: in the Docker layout the machine root is
        # /opt/data (set via `ENV HERMES_HOME=/opt/data`), so an unset
        # HERMES_HOME falls back to $HOME/.hermes = /opt/data/.hermes — an
        # empty, auto-seeded home where the dashboard sees only the default
        # profile and the install-method stamp is missing (so the Docker
        # update-button guard also misfires).  get_default_hermes_root()
        # returns the root for both layouts: ~/.hermes for a standard install
        # and /opt/data for Docker (it strips a trailing profiles/<name>).
        # See the support report for the double-mount workaround this avoids.
        try:
            from hermes_constants import get_default_hermes_root
            env["HERMES_HOME"] = str(get_default_hermes_root())
        except Exception:
            # Best-effort: if root resolution fails, fall back to the prior
            # behaviour (drop HERMES_HOME) rather than block the reroute.
            env.pop("HERMES_HOME", None)
        # On Windows, os.execvpe() does not truly replace the process — it
        # spawns via CreateProcess then the parent exits.  Under Python 3.14+
        # this can crash with STATUS_ACCESS_VIOLATION (0xC0000005) when
        # re-executing the dashboard for a non-default profile.  Use
        # subprocess.Popen + sys.exit() on Windows to avoid the crash.
        if sys.platform == "win32":
            proc = subprocess.Popen(reexec_argv, env=env)
            sys.exit(proc.wait())
        else:
            os.execvpe(sys.executable, reexec_argv, env)

    # Apply the final process/profile policy after dashboard routing, but before
    # importing the web server or opening dashboard state. Applying it before a
    # named-profile re-exec could leak that profile's higher limit into the
    # machine/default dashboard, whose lower policy intentionally cannot undo it.
    # This also covers Desktop SSH's isolated `serve` child, which does not route.
    from hermes_cli.resource_limits import apply_nofile_soft_limit

    apply_nofile_soft_limit()

    if _token_file:
        _ssh_session_token = _read_ssh_session_token_file(_token_file)

    # Attach gui.log early so dashboard startup/build failures are captured in
    # the same logs directory as every other Hermes surface.
    try:
        from hermes_logging import setup_logging as _setup_logging_gui
        _setup_logging_gui(mode="gui")
    except Exception:
        pass

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as e:
        print("Web UI dependencies not installed (need fastapi + uvicorn).")
        print(
            f"Re-install the package into this interpreter so metadata updates apply:\n"
            f"  cd {PROJECT_ROOT}\n"
            f"  {sys.executable} -m pip install -e .\n"
            "If `pip` is missing in this venv, use:  uv pip install -e ."
        )
        print(f"Import error: {e}")
        sys.exit(1)

    # Seed bundled skills on first dashboard launch so the desktop GUI's
    # skills picker / agent skill discovery sees the bundled library.
    # cmd_chat does this in its own pre-dispatch block; the dashboard
    # backend is the desktop's primary entrypoint and needs the same.
    _sync_bundled_skills_quietly()

    # Bridge terminal.* config into the TERMINAL_* env vars for THIS process,
    # mirroring the CLI (cli.py env_mappings) and gateway (gateway/run.py
    # _terminal_env_map) startup bridges. The dashboard/serve backend runs
    # agents in-process (tui_gateway.ws → server._make_agent) and ticks cron
    # jobs itself when desktop-spawned — without this bridge those consumers
    # saw an unset TERMINAL_ENV and silently ran every command on the host
    # even when config.yaml selects `terminal.backend: docker`
    # (#63141, #54449, #61115, #65696). PTY chat spawns already bridge their
    # child env copy; this covers the in-process consumers.
    try:
        from hermes_cli.config import apply_terminal_config_to_env

        apply_terminal_config_to_env()
    except Exception:
        logger.debug("terminal config → env bridge failed for dashboard/serve",
                     exc_info=True)

    if _headless_backend:
        # Don't build the SPA, and tell mount_spa() (read at web_server import
        # below) to disable it even if a stray dist exists. Set it first.
        os.environ["HERMES_SERVE_HEADLESS"] = "1"
    elif "HERMES_WEB_DIST" not in os.environ and not getattr(args, "skip_build", False):
        if not _build_web_ui(PROJECT_ROOT / "web", fatal=True):
            sys.exit(1)
    elif getattr(args, "skip_build", False):
        # --build-mode skip trusts the caller to have pre-built the web UI.
        # Verify the dist actually exists; otherwise the server will start
        # and serve 404s with no obvious cause (issue #23817).
        _dist_root = (
            Path(os.environ["HERMES_WEB_DIST"])
            if "HERMES_WEB_DIST" in os.environ
            else PROJECT_ROOT / "hermes_cli" / "web_dist"
        )
        if not (_dist_root / "index.html").exists():
            # The caller promised a pre-built dist but there isn't one.
            # Instead of hard-failing (issue #59288 — desktop launches with
            # --build-mode skip after a wipe of web_dist), warn and attempt
            # ONE recovery build through the normal build path. Only the
            # default dist location is recoverable: a custom HERMES_WEB_DIST
            # points at a caller-managed directory the build cannot populate.
            _recoverable = "HERMES_WEB_DIST" not in os.environ
            if _recoverable:
                print(f"⚠ --skip-build was passed but no web dist found at: {_dist_root}")
                print("  Attempting one recovery build of the web UI...")
                _build_web_ui(PROJECT_ROOT / "web", fatal=True)
            if not (_dist_root / "index.html").exists():
                print(f"✗ --skip-build was passed but no web dist found at: {_dist_root}")
                if _recoverable:
                    print("  The recovery build did not produce a usable dist.")
                print("  Pre-build first:  npm install --workspace web && npm run build -w web")
                print("  Or drop --skip-build to build automatically.")
                sys.exit(1)
            print("  ✓ Recovery build produced a web dist")
        print(f"→ Skipping web UI build (--skip-build); using dist at {_dist_root}")
    else:
        # HERMES_WEB_DIST is set without --skip-build: the build is skipped
        # (the env var points at a caller-managed dist), so validate it the
        # same way the --skip-build branch does — otherwise the server starts
        # and serves 404s with no obvious cause (same failure mode as #23817,
        # via the env-var path).
        _dist_root = Path(os.environ["HERMES_WEB_DIST"]).expanduser()
        if not (_dist_root / "index.html").exists():
            print(f"✗ HERMES_WEB_DIST is set but no web dist found at: {_dist_root}")
            print("  Pre-build first:  npm install --workspace web && npm run build -w web")
            print("  Or unset HERMES_WEB_DIST to build and use the default web UI dist.")
            sys.exit(1)
        # Write the expanded path back: web_server reads HERMES_WEB_DIST raw
        # at import (no expanduser), so a validated "~/dist" would otherwise
        # pass here and still 404 there.
        os.environ["HERMES_WEB_DIST"] = str(_dist_root)
        print(f"→ Using web dist from HERMES_WEB_DIST: {_dist_root}")

    # Discover and load plugins so any DashboardAuthProvider plugin
    # (e.g. plugins/dashboard_auth/nous) registers BEFORE start_server's
    # fail-closed gate check runs. The top-level argparse setup skips
    # plugin discovery for built-in subcommands like ``dashboard`` to
    # save ~500ms startup; we have to trigger it explicitly here because
    # the dashboard's server-side runtime depends on plugin-registered
    # providers (image_gen, web, dashboard_auth, …).
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    except Exception as exc:
        # Discovery failures must not block dashboard startup outright —
        # log and proceed; the gate's fail-closed branch will surface
        # the missing-provider state if it matters.
        print(f"⚠ Plugin discovery failed: {exc}", file=sys.stderr)

    # Desktop chat uses the dashboard's in-process /api/ws gateway, which builds
    # agents via tui_gateway.server._make_agent.  That path only snapshots the
    # tool registry — it never starts MCP discovery (the stdio TUI does that in
    # tui_gateway/entry.py, which the dashboard process doesn't run).  Without
    # this, a profile's configured MCP servers never connect, so desktop
    # sessions show no MCP tools.  Spawn discovery in the background here so a
    # slow/dead server can't block dashboard startup.
    try:
        from hermes_cli.mcp_startup import start_background_mcp_discovery

        start_background_mcp_discovery(
            logger=logger,
            thread_name="dashboard-mcp-discovery",
        )
    except Exception:
        logger.debug(
            "Background MCP tool discovery failed at dashboard startup",
            exc_info=True,
        )

    from hermes_cli.web_server import start_server

    # Interactive auth setup: if this bind will engage the auth gate but no
    # provider is registered yet, offer to configure one here (TTY only)
    # instead of hard-failing inside start_server. Non-interactive callers
    # (Docker/s6, CI, --no-open pipelines) fall through to start_server's
    # fail-closed SystemExit unchanged.
    _maybe_setup_dashboard_auth_interactively(args)

    # The in-browser Chat tab (the embedded TUI over PTY/WebSocket) is always
    # available — the desktop app and the dashboard's own Chat tab both rely on
    # the `/api/ws` + `/api/pty` sockets, so there is no reason to gate them.
    start_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        allow_public=getattr(args, "insecure", False),
        initial_profile=getattr(args, "open_profile", "") or "",
        headless=_headless_backend,
        ssh_session_token=_ssh_session_token,
        ssh_owner_nonce=_ssh_owner_nonce,
    )


def cmd_dashboard_register(args):
    """Register a self-hosted dashboard OAuth client with Nous Portal."""
    from hermes_cli.dashboard_register import cmd_dashboard_register as _impl

    _impl(args)


def cmd_gateway_enroll(args):
    """Enroll a self-hosted gateway with a relay connector."""
    from hermes_cli.gateway_enroll import cmd_gateway_enroll as _impl

    _impl(args)


def cmd_completion(args, parser=None):
    """Print shell completion script."""
    from hermes_cli.completion import generate_bash, generate_zsh, generate_fish

    shell = getattr(args, "shell", "bash")
    if shell == "zsh":
        print(generate_zsh(parser))
    elif shell == "fish":
        print(generate_fish(parser))
    else:
        print(generate_bash(parser))


def cmd_prompt_size(args):
    """Show a byte/char breakdown of the system prompt + tool schemas."""
    from hermes_cli.prompt_size import cmd_prompt_size as _impl

    _impl(args)


def cmd_logs(args):
    """View and filter Hermes log files."""
    from hermes_cli.logs import tail_log, list_logs

    log_name = getattr(args, "log_name", "agent") or "agent"

    if log_name == "list":
        list_logs()
        return

    tail_log(
        log_name,
        num_lines=getattr(args, "lines", 50),
        follow=getattr(args, "follow", False),
        level=getattr(args, "level", None),
        session=getattr(args, "session", None),
        since=getattr(args, "since", None),
        component=getattr(args, "component", None),
    )


def cmd_console(args):
    """Open the safe Hermes command console."""
    from hermes_cli.console_engine import run_console_repl

    return run_console_repl()


def _build_provider_choices() -> list[str]:
    """Build the --provider choices list from CANONICAL_PROVIDERS + 'auto'."""
    try:
        from hermes_cli.models import CANONICAL_PROVIDERS as _cp
        return ["auto"] + [p.slug for p in _cp]
    except Exception:
        # Fallback: static list guarantees the CLI always works
        return [
            "auto", "openrouter", "nous", "openai-codex", "xai-oauth", "copilot-acp", "copilot",
            "anthropic", "gemini", "vertex", "xai", "bedrock", "azure-foundry",
            "ollama-cloud", "huggingface", "zai", "kimi-coding", "kimi-coding-cn",
            "stepfun", "minimax", "minimax-cn", "kilocode", "novita", "xiaomi", "arcee",
            "nvidia", "deepseek", "alibaba", "qwen-oauth", "opencode-zen", "opencode-go",
        ]


# Top-level subcommands that argparse knows about WITHOUT running plugin
# discovery.  Used to short-circuit eager plugin imports (which can take
# 500ms+ pulling in google.cloud.pubsub_v1, aiohttp, grpc, etc.) when the
# user's invocation clearly doesn't need any plugin-registered subcommand.
#
# Keep this in sync with the ``subparsers.add_parser("NAME", ...)`` calls
# below in ``main()``. Missing an entry here only costs a one-time
# discovery; extra entries here would let a plugin command silently fail
# to parse.
_BUILTIN_SUBCOMMANDS = frozenset(
    {
        "acp", "approvals", "auth", "backup", "bundles", "checkpoints", "claw", "completion",
        "computer-use",
        "config", "console", "cron", "curator", "dashboard", "serve", "debug", "doctor",
        "dump", "egress", "fallback", "gateway", "hooks", "import", "import-agent", "insights",
        "gui", "desktop", "kanban", "login", "logout", "logs", "lsp", "mcp", "memory", "migrate", "moa",
        "journey", "memory-graph", "learning",
        "model", "monitoring", "pairing", "pause", "peer", "pets", "plugins", "portal", "profile",
        "project", "proxy",
        "prompt-size",
        "resume",
        "send", "sessions", "setup",
        "skin", "skills", "slack", "status", "sync", "tools", "uninstall", "update",
        "webhook", "whatsapp", "whatsapp-cloud", "worktree", "chat", "secrets", "security",
        "browser",
        "verify",
        # Help-ish invocations — plugin commands not being listed in
        # top-level --help is an acceptable trade-off for skipping an
        # expensive eager import of every bundled plugin module.
        "help",
    }
)


def _first_positional_argv() -> str | None:
    """Return the first non-flag, non-flag-value token in ``sys.argv[1:]``.

    Used by ``main()`` to decide whether plugin discovery has to run at
    argparse-setup time. Handles common invocations like
    ``hermes -m gpt5 --provider openai chat "msg"`` by skipping the
    values attached to known top-level flags.

    Does NOT fully simulate argparse — unknown ``--foo=bar`` / ``--foo
    bar`` flags degrade gracefully (``bar`` may be wrongly classified as
    a positional, which at worst forces a one-time plugin discovery).
    """
    from hermes_cli._parser import top_level_value_flag_sets

    required_value_flags, optional_value_flags = top_level_value_flag_sets()
    value_flags = required_value_flags | optional_value_flags
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            # Everything after ``--`` is positional.
            if i + 1 < len(argv):
                return argv[i + 1]
            return None
        if tok.startswith("-"):
            # ``--flag=value`` carries its value inline — single token.
            if "=" in tok:
                i += 1
                continue
            if tok in value_flags and i + 1 < len(argv):
                i += 2
                continue
            i += 1
            continue
        return tok
    return None


def _plugin_cli_discovery_needed() -> bool:
    """True when the CLI might be invoking a plugin-registered subcommand.

    Returning False lets ``main()`` skip plugin discovery entirely during
    argparse setup, saving ~500-650ms per invocation for users whose
    enabled plugins don't contribute any CLI command.
    """
    first = _first_positional_argv()
    if first is None:
        # Bare ``hermes`` or only flags → defaults to ``chat``.
        return False
    if first in _BUILTIN_SUBCOMMANDS:
        return False
    # Unknown token — could be a plugin subcommand, OR a chat prompt
    # starting with a non-flag word. Either way we need discovery: if it
    # IS a plugin command, argparse needs the subparser; if it's a chat
    # prompt, argparse will route it via positional handling and the
    # extra discovery cost is amortized over a full agent run anyway.
    return True


def _resolve_deferred_platform_cli_command(command_name: str | None) -> None:
    """Materialize the deferred platform whose top-level CLI command matches.

    Bundled platform plugins are cheap-registered as *deferred* entries to
    avoid importing every gateway SDK during normal startup. A platform that
    registers a top-level ``hermes <name>`` command (e.g. Photon ->
    ``ctx.register_cli_command(name="photon", ...)``) only runs that side
    effect when its module is imported. On the unknown-top-level-command slow
    path, ``discover_plugins()`` records the deferred loader but does not
    import it, so the CLI registration never happens and ``hermes photon``
    fails with argparse ``invalid choice`` (issue #54678).

    Resolving only the platform whose name matches the first positional token
    keeps normal startup cheap while making the targeted command available.
    """
    if not command_name:
        return
    try:
        from gateway.platform_registry import platform_registry

        platform_registry.get(command_name)
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Deferred platform CLI resolution failed for %s: %s",
            command_name,
            exc,
        )


_AGENT_COMMANDS = {None, "chat", "acp", "rl"}
_AGENT_SUBCOMMANDS = {
    "cron": ("cron_command", {"run", "tick"}),
    "gateway": ("gateway_command", {"run"}),
    "mcp": ("mcp_action", {"serve"}),
}


def _is_tui_chat_launch(args) -> bool:
    return bool(getattr(args, "tui", False) or os.environ.get("HERMES_TUI") == "1")


def _command_has_dedicated_mcp_startup(args) -> bool:
    if args.command == "acp":
        return True
    if args.command == "gateway" and getattr(args, "gateway_command", None) == "run":
        return True
    if args.command == "cron" and getattr(args, "cron_command", None) in {"run", "tick"}:
        return True
    return False


def _should_background_mcp_startup(args) -> bool:
    if _is_tui_chat_launch(args):
        return False
    return args.command in {None, "chat", "rl"}


def _prepare_agent_startup(args) -> None:
    """Discover plugins/MCP/hooks for commands that can run an agent turn."""
    # --yolo: chokepoint guarantee that HERMES_YOLO_MODE is set before ANY
    # plugin/tool discovery below imports tools.approval, which freezes
    # _YOLO_MODE_FROZEN at import time (PR #7994 security design).  main()'s
    # dispatch path also sets this earlier, but _prepare_agent_startup() is
    # reachable from other launchers too (e.g. the Termux fast-CLI path),
    # so the guarantee lives here where the import is actually triggered
    # (#60328).
    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"
    _apply_safe_mode(args)

    _sub_attr, _sub_set = _AGENT_SUBCOMMANDS.get(args.command, (None, None))
    if not (
        args.command in _AGENT_COMMANDS
        or (_sub_attr and getattr(args, _sub_attr, None) in _sub_set)
    ):
        return

    _accept_hooks = bool(getattr(args, "accept_hooks", False))
    if not _is_tui_chat_launch(args):
        # The TUI backend process does its own plugin discovery; the launcher
        # only spawns Node, so discovery here would be thrown-away work.
        try:
            from hermes_cli.plugins import start_background_plugin_discovery

            # Discovery runs in a daemon thread so its ~150ms of manifest
            # scanning + plugin imports overlaps the rest of startup (cli /
            # prompt_toolkit imports, worktree git calls). Correctness is
            # unchanged: every synchronous reader goes through
            # discover_plugins(), which joins this thread first — including
            # the discover_plugins() call model_tools makes at import time,
            # which happens before any tool list is built.
            start_background_plugin_discovery()
        except Exception:
            logger.warning(
                "plugin discovery failed at CLI startup",
                exc_info=True,
            )
    _run_inline_mcp_discovery = True
    if _is_tui_chat_launch(args):
        # The TUI launcher hands off to a dedicated startup path that already
        # backgrounds MCP discovery with a bounded join before the first tool
        # snapshot.
        _run_inline_mcp_discovery = False
    elif _command_has_dedicated_mcp_startup(args):
        # These entrypoints already do their own MCP startup later on the real
        # runtime path (gateway executor, ACP launcher, cron job runner).
        _run_inline_mcp_discovery = False
    elif _should_background_mcp_startup(args):
        try:
            from hermes_cli.mcp_startup import start_background_mcp_discovery

            start_background_mcp_discovery(
                logger=logger,
                thread_name="cli-mcp-discovery",
            )
        except Exception:
            logger.debug(
                "Background MCP tool discovery failed at CLI startup",
                exc_info=True,
            )
        _run_inline_mcp_discovery = False
    if _run_inline_mcp_discovery:
        try:
            # MCP tool discovery remains synchronous for entrypoints that do
            # not own a later bounded/executor startup path.
            from tools.mcp_tool import discover_mcp_tools

            discover_mcp_tools()
        except Exception:
            logger.debug(
                "MCP tool discovery failed at CLI startup",
                exc_info=True,
            )
    try:
        from hermes_cli.config import load_config
        from agent.shell_hooks import register_from_config

        _hooks_cfg = load_config()
        register_from_config(_hooks_cfg, accept_hooks=_accept_hooks)

        from agent.outbound_webhooks import (
            register_from_config as register_outbound_webhooks,
        )

        register_outbound_webhooks(_hooks_cfg)
    except Exception:
        logger.debug(
            "shell-hook registration failed at CLI startup",
            exc_info=True,
        )


def _apply_safe_mode(args) -> None:
    if not getattr(args, "safe_mode", False):
        return
    os.environ["HERMES_SAFE_MODE"] = "1"
    os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"
    os.environ["HERMES_IGNORE_RULES"] = "1"


def _set_chat_arg_defaults(args) -> None:
    for attr, default in [
        ("query", None),
        ("model", None),
        ("provider", None),
        ("toolsets", None),
        ("verbose", False),
        ("resume", None),
        ("continue_last", None),
        ("worktree", False),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)


def _try_fast_chat_launch() -> bool:
    """Fast path for unambiguous interactive chat launches (all hosts).

    ``hermes`` / ``hermes -w -s foo --yolo`` / ``hermes chat`` don't need the
    full argparse tree: building all ~40 subcommand parsers costs ~140ms of
    pure-Python argparse setup plus their module imports, none of which the
    chat path uses. Parse the lightweight top-level/chat parser instead and
    dispatch straight to ``cmd_chat``.

    Bails out (returns False) whenever the invocation is not certainly a
    chat launch — a subcommand positional, ``--help``, unknown flags — so
    every other path still goes through the full parser unchanged. Mirrors
    ``_try_termux_fast_cli_launch`` minus the Termux-specific deferred
    startup; kept separate so phone-tuned behavior doesn't leak to desktops.
    """
    if os.environ.get("HERMES_DISABLE_FAST_CHAT_LAUNCH") == "1":
        return False
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        return False
    # Container-aware routing must win: when NixOS container mode is
    # active, EVERY invocation is forwarded into the managed container.
    try:
        from hermes_cli.config import get_container_exec_info
        if get_container_exec_info():
            return False
    except Exception:
        return False
    # TUI launches have their own startup path (bounded MCP joins etc.) —
    # keep them on full dispatch outside Termux.
    if _wants_tui_early(argv):
        return False
    if _first_positional_argv() not in {None, "chat"}:
        return False

    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)
    try:
        args, unknown = parser.parse_known_args(_coalesce_session_name_args(argv))
    except SystemExit:
        return False
    if unknown:
        # Flags the light parser doesn't know — could belong to a plugin
        # subcommand or a newer full-parser flag. Fall back to full dispatch.
        return False
    if getattr(args, "version", False):
        return False
    if getattr(args, "command", None) not in {None, "chat"}:
        return False

    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"
    _prepare_agent_startup(args)

    if getattr(args, "oneshot", None):
        _confirm_startup_expensive_model_override(args)
        _run_and_exit_oneshot(
            args.oneshot,
            model=getattr(args, "model", None),
            provider=getattr(args, "provider", None),
            toolsets=getattr(args, "toolsets", None),
            skills=getattr(args, "skills", None),
            usage_file=getattr(args, "usage_file", None),
        )

    if (args.resume or args.continue_last) and args.command is None:
        args.command = "chat"

    _set_chat_arg_defaults(args)
    cmd_chat(args)
    return True


def _try_termux_fast_cli_launch() -> bool:
    """Run obvious Termux non-TUI chat/oneshot/version paths on a light parser."""
    if not _is_termux_startup_environment():
        return False
    if os.environ.get("HERMES_TERMUX_DISABLE_FAST_CLI") == "1":
        return False

    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        return False
    # Let the TUI fast path (or full dispatch) handle anything that resolves to
    # the TUI — explicit --tui/env or display.interface=tui. `--cli` forces this
    # to stay False so the classic fast path still runs.
    if _wants_tui_early(argv):
        return False

    if _is_termux_fast_version_argv(argv):
        _print_version_info(check_updates=True)
        return True

    first = _first_positional_argv()
    has_oneshot = any(
        arg == "-z" or arg == "--oneshot" or arg.startswith("--oneshot=")
        for arg in argv
    )
    if not has_oneshot and first not in {None, "chat"}:
        return False

    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)
    args = parser.parse_args(_coalesce_session_name_args(argv))

    if getattr(args, "version", False):
        _print_version_info(check_updates=True)
        return True

    if getattr(args, "oneshot", None):
        _prepare_agent_startup(args)
        _confirm_startup_expensive_model_override(args)
        _run_and_exit_oneshot(
            args.oneshot,
            model=getattr(args, "model", None),
            provider=getattr(args, "provider", None),
            toolsets=getattr(args, "toolsets", None),
            skills=getattr(args, "skills", None),
            usage_file=getattr(args, "usage_file", None),
        )

    if (args.resume or args.continue_last) and args.command is None:
        args.command = "chat"

    if args.command in {None, "chat"}:
        _set_chat_arg_defaults(args)
        interactive_prompt = not getattr(args, "query", None) and not getattr(args, "image", None)
        if interactive_prompt:
            # Bare Termux CLI should reach the prompt first and do agent-only
            # discovery on the first submitted turn instead of before input.
            setattr(args, "compact", True)
            os.environ["HERMES_DEFER_AGENT_STARTUP"] = "1"
            os.environ["HERMES_FAST_STARTUP_BANNER"] = "1"
            if getattr(args, "accept_hooks", False):
                os.environ["HERMES_ACCEPT_HOOKS"] = "1"
        else:
            _prepare_agent_startup(args)
        cmd_chat(args)
        return True

    return False


def _try_termux_fast_tui_launch() -> bool:
    """Launch obvious Termux TUI invocations before building every subparser.

    `hermes --tui` is the hot path on phones. The full parser setup imports
    command modules for model, fallback, migrate, kanban, bundles, plugins,
    etc. even though the TUI immediately execs Node. On Termux only, parse the
    lightweight top-level/chat parser and hand off to ``cmd_chat`` when the
    invocation is unambiguously the built-in TUI/chat path.
    """
    if not _is_termux_startup_environment():
        return False

    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        return False

    wants_tui = _wants_tui_early(sys.argv[1:])
    if not wants_tui:
        return False

    first = _first_positional_argv()
    if first not in {None, "chat"}:
        return False

    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)
    args = parser.parse_args(_coalesce_session_name_args(sys.argv[1:]))

    # Preserve top-level behaviours whose semantics are not "launch chat/TUI".
    if getattr(args, "version", False) or getattr(args, "oneshot", None):
        return False
    if getattr(args, "command", None) not in {None, "chat"}:
        return False
    if not _resolve_use_tui(args):
        return False

    cmd_chat(args)
    return True
        except InvalidUserConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    ssh_owner_nonce = getattr(args, "ssh_owner_nonce", None)
    if ssh_owner_nonce and not re.fullmatch(r"[0-9a-f]{16}", ssh_owner_nonce):
        raise SystemExit("--ssh-owner-nonce must be 16 lowercase hex characters")
    if token_file and not headless_backend:
        raise SystemExit("--ssh-session-token-file is only valid with hermes serve")
    return ssh_owner_nonce


def _dashboard_sanitize_desktop_env(headless_backend) -> None:
    """Strip Desktop-inherited env that hijacks a standalone launch.

    Desktop Electron spawns its backend with HERMES_DESKTOP=1 plus
    HERMES_WEB_DIST=<packaged app.asar[/unpacked]/dist> (and often
    HERMES_SERVE_HEADLESS=1). A shell inheriting those then running
    `hermes dashboard` would serve the desktop renderer ("Desktop IPC bridge
    is unavailable", #52945) or disable the SPA. Only Electron-packaged
    WEB_DIST contamination is stripped from browser dashboards — caller-managed
    overrides (dev / custom builds) must still work, while headless `serve`
    keeps the packaged path used by the Desktop backend. Headless `serve`
    re-sets HERMES_SERVE_HEADLESS itself.

    The Desktop's legacy fallback spawn (`dashboard --no-open`, taken when the
    `serve --help` probe times out on a cold host) is not headless yet must keep
    its packaged dist: it is told apart by the per-spawn
    HERMES_DASHBOARD_SESSION_TOKEN, which the terminal pane never receives and
    the terminal tool's env policy strips from agent children.
    """
    desktop_owned_child = _is_desktop_owned_backend()
    if (
        not headless_backend
        and not desktop_owned_child
        and _is_electron_packaged_web_dist(os.environ.get("HERMES_WEB_DIST", ""))
    ):
        os.environ.pop("HERMES_WEB_DIST", None)
    if not headless_backend:
        os.environ.pop("HERMES_SERVE_HEADLESS", None)


def _require_dashboard_web_deps() -> None:
    """Exit with the right message when the dashboard's web-server packages can't import.

    A plain missing-package ImportError gets the standard repair guidance; the
    ``DLL load failed ... _ssl`` signature of Windows Smart App Control blocking the
    embedded runtime gets the policy guidance instead, so users stop looping on
    repair for a block repair can never lift (#63796)."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as e:
        from hermes_cli.main_dep_hints import (
            missing_optional_deps_message,
            smart_app_control_block_message,
        )

        print(smart_app_control_block_message(e) or missing_optional_deps_message(
            "dashboard", "its web-server packages (fastapi, uvicorn)", "all"))
        print(f"Details: {e}")
        sys.exit(1)


def _dashboard_prepare_runtime(args, headless_backend) -> bool:
    """Deps check, skills seed, terminal env bridge, plugins, MCP discovery.

    Returns ``start_mcp_discovery_after_bind`` for start_server.
    """
    # Attach gui.log early so dashboard startup/build failures are captured in
    # the same logs directory as every other Hermes surface.
    try:
        from hermes_logging import setup_logging as _setup_logging_gui
        _setup_logging_gui(mode="gui")
    except Exception:
        pass

    _require_dashboard_web_deps()

    # Seed bundled skills on first dashboard launch so the desktop GUI's
    # skills picker / agent skill discovery sees the bundled library.
    _sync_bundled_skills_quietly()

    # Bridge terminal.* config into TERMINAL_* env for THIS process, like the
    # CLI (cli.py env_mappings) and gateway (_terminal_env_map) do. The
    # dashboard/serve backend runs agents in-process (tui_gateway.ws →
    # server._make_agent) and ticks cron itself when desktop-spawned; without
    # this those consumers saw an unset TERMINAL_ENV and ran every command on
    # the host even under `terminal.backend: docker` (#63141, #54449).
    try:
        # PTY chat spawns already bridge their child env copy; this covers the in-process consumers. See
        # #61115, #65696.
        from hermes_cli.config import apply_terminal_config_to_env

        apply_terminal_config_to_env()
    except Exception:
        logger.debug("terminal config → env bridge failed for dashboard/serve",
                     exc_info=True)

    _resolve_dashboard_web_dist(args, headless_backend)
    # Load plugins so any DashboardAuthProvider plugin registers BEFORE
    # start_server's fail-closed gate check. Argparse setup skips discovery
    # for built-in subcommands (~500ms), but the dashboard's server-side
    # runtime depends on plugin-registered providers (image_gen, web,
    # dashboard_auth, …).
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    except Exception as exc:
        # Must not block startup; the gate's fail-closed branch surfaces a
        # missing provider if it matters.
        print(f"⚠ Plugin discovery failed: {exc}", file=sys.stderr)

    # Desktop chat uses the in-process /api/ws gateway (tui_gateway.server
    # ._make_agent), which only snapshots the tool registry and never starts
    # MCP discovery — so configured MCP servers would never connect. Spawn
    # discovery in the background here so a slow/dead server can't block
    # startup. Desktop-spawned headless backends start it AFTER the socket
    # binds instead (start_server's ready path): the thread's first act is the
    # ~350ms `mcp` SDK import, which holds the GIL against the web_server
    # import and delays the READY sentinel; _make_agent's bounded
    # wait_for_mcp_discovery covers a server still connecting at first turn.
    # A standalone (non-Desktop) dashboard may sit idle and unvisited for days
    # (#58733): it arms discovery instead and the first /api/ws client fires it.
    desktop = _is_desktop_owned_backend()
    if headless_backend and desktop:
        return True
    try:
        from hermes_cli.mcp_startup import (
            defer_background_mcp_discovery,
            start_background_mcp_discovery,
        )

        if desktop:
            start_background_mcp_discovery(logger=logger, thread_name="dashboard-mcp-discovery")
        else:
            defer_background_mcp_discovery(logger=logger, thread_name="dashboard-mcp-discovery", delay=None)
    except Exception:
        logger.debug(
            "Background MCP tool discovery failed at dashboard startup",
            exc_info=True,
        )
    return False


def cmd_dashboard(args):
    """Start the web UI server, or (with --stop/--status) manage running ones."""
    _token_file = getattr(args, "ssh_session_token_file", None)
    _dashboard_lifecycle_flags(args, _token_file)

    # `serve` is the headless backend: no UI build, no SPA mount, neutral
    # ready sentinel. Resolved once and threaded through the re-exec, the
    # build gate, and start_server.
    _headless_backend = getattr(args, "headless_backend", False)
    _ssh_owner_nonce = _dashboard_validate_serve_args(args, _headless_backend, _token_file)
    _dashboard_sanitize_desktop_env(_headless_backend)

    _attach_to_host_backend(args, _headless_backend)
    _route_named_profile_dashboard(args, _headless_backend, _ssh_owner_nonce, _token_file)

    # Apply the final process/profile policy after dashboard routing, but before
    # importing the web server or opening dashboard state. Applying it before a
    # named-profile re-exec could leak that profile's higher limit into the
    # machine/default dashboard, whose lower policy intentionally cannot undo it.
    # This also covers Desktop SSH's isolated `serve` child, which does not route.
    from hermes_cli.resource_limits import apply_nofile_soft_limit

    apply_nofile_soft_limit()

    _ssh_session_token = _read_ssh_session_token_file(_token_file) if _token_file else None
    _mcp_discovery_after_bind = _dashboard_prepare_runtime(args, _headless_backend)

    from hermes_cli.web_server import start_server

    # Interactive auth setup: if this bind will engage the auth gate but no
    # provider is registered yet, offer to configure one here (TTY only)
    # instead of hard-failing inside start_server. Non-interactive callers
    # (Docker/s6, CI, --no-open pipelines) fall through to start_server's
    # fail-closed SystemExit unchanged.
    _maybe_setup_dashboard_auth_interactively(args)
    if _headless_backend:
        from hermes_cli.observability.shared_metrics_process import begin_process
        begin_process("serve")

    # The in-browser Chat tab (embedded TUI over PTY/WebSocket) is always
    # available — desktop and dashboard both rely on `/api/ws` + `/api/pty`.
    start_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        allow_public=getattr(args, "insecure", False),
        initial_profile=getattr(args, "open_profile", "") or "",
        headless=_headless_backend,
        isolated=getattr(args, "isolated", False),
        ssh_session_token=_ssh_session_token,
        ssh_owner_nonce=_ssh_owner_nonce,
        start_mcp_discovery_after_bind=_mcp_discovery_after_bind,
    )


def cmd_completion(args, parser=None):
    """Print shell completion script."""
    from hermes_cli import completion

    shell = getattr(args, "shell", "bash")
    generate = {"zsh": completion.generate_zsh, "fish": completion.generate_fish}.get(
        shell, completion.generate_bash
    )
    print(generate(parser))


def cmd_logs(args):
    """View and filter Hermes log files."""
    from hermes_cli.logs import tail_log, list_logs

    log_name = getattr(args, "log_name", "agent") or "agent"

    if log_name == "list":
        list_logs()
        return

    tail_log(
        log_name,
        num_lines=getattr(args, "lines", 50),
        follow=getattr(args, "follow", False),
        level=getattr(args, "level", None),
        session=getattr(args, "session", None),
        since=getattr(args, "since", None),
        component=getattr(args, "component", None),
    )


def cmd_console(args):
    """Open the safe Hermes command console."""
    from hermes_cli.console_engine import run_console_repl

    return run_console_repl()


# Top-level subcommands known WITHOUT plugin discovery (which costs 500ms+ of
# eager plugin imports). Keep in sync with the add_parser calls in
# _build_cli_parser: a missing entry only costs a one-time discovery; an extra
# entry would let a plugin command silently fail to parse.
_BUILTIN_SUBCOMMANDS = frozenset(
    {
        "acp", "approvals", "auth", "backup", "bundles", "checkpoints", "claw", "codex-runtime", "completion",
        "computer-use",
        "config", "console", "cron", "curator", "dashboard", "serve", "debug", "doctor",
        "dump", "egress", "fallback", "gateway", "hooks", "import", "import-agent", "insights",
        "gui", "desktop", "kanban", "login", "logout", "logs", "lsp", "mcp", "memory", "migrate", "moa",
        "journey", "memory-graph", "learning",
        "model", "monitoring", "pairing", "pause", "peer", "pets", "plugins", "portal", "profile",
        "project", "proxy",
        "prompt-size",
        "resume",
        "send", "sessions", "setup",
        "skin", "skills", "slack", "status", "sync", "tools", "uninstall", "update",
        "usage", "vault",
        "webhook", "whatsapp", "whatsapp-cloud", "worktree", "chat", "secrets", "security",
        "browser",
        "verify",
        # Plugin commands missing from top-level --help is an accepted trade-off.
        "help",
    }
)


def _first_positional_argv() -> str | None:
    """First non-flag, non-flag-value token in ``sys.argv[1:]`` (skips values of known flags).

    Not a full argparse simulation: an unknown ``--foo bar`` may classify
    ``bar`` as positional, which at worst forces a one-time plugin discovery.
    """
    from hermes_cli._parser import command_argv

    args = command_argv(sys.argv[1:])
    return args[0] if args else None


def _plugin_cli_discovery_needed() -> bool:
    """True when the CLI might be invoking a plugin-registered subcommand.

    False skips plugin discovery at argparse setup (~500-650ms). An unknown
    first token could be a plugin command OR a chat prompt — either way
    discovery is needed; for a prompt its cost amortizes over the agent run.
    """
    first = _first_positional_argv()  # None = bare ``hermes`` → chat
    return first is not None and first not in _BUILTIN_SUBCOMMANDS


def _resolve_deferred_platform_cli_command(command_name: str | None) -> None:
    """Materialize the deferred platform whose top-level CLI command matches.

    Bundled platforms are *deferred* entries (no gateway SDK imports at
    startup), so a platform's ``register_cli_command`` side effect only runs
    on import; ``discover_plugins()`` alone leaves ``hermes photon`` failing
    with ``invalid choice``. Importing just the matching platform keeps
    startup cheap.

    On the unknown-top-level-command slow path, ``discover_plugins()`` records the deferred loader but does
    not import it, so the CLI registration never happens and ``hermes photon`` fails with argparse ``invalid
    choice`` (issue #54678).
    """
    if not command_name:
        return
    try:
        from gateway.platform_registry import platform_registry

        platform_registry.get(command_name)
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Deferred platform CLI resolution failed for %s: %s",
            command_name,
            exc,
        )


_AGENT_COMMANDS = {None, "chat", "acp", "rl"}
_AGENT_SUBCOMMANDS = {
    "cron": ("cron_command", {"run", "tick"}),
    "gateway": ("gateway_command", {"run"}),
    "mcp": ("mcp_action", {"serve"}),
}


def _is_tui_chat_launch(args) -> bool:
    if getattr(args, "tui", False) or os.environ.get("HERMES_TUI") == "1":
        return True
    # The chat path decides TUI-vs-classic via _resolve_use_tui (--cli/--tui
    # flags, TTY gate, HERMES_TUI env, display.interface config). Bare
    # `hermes`/`hermes chat` with a TUI display config was previously missed
    # here, so the wrapper pre-warmed its own MCP discovery while the TUI
    # gateway (spawned moments later) ran a second one — an idle stdio MCP
    # server copy held dead for the whole session. Only chat commands can
    # launch the TUI; other commands (mcp serve, gateway, acp, cron) keep
    # their own discovery behavior untouched.
    if getattr(args, "command", None) not in {None, "chat"}:
        return False
    return _resolve_use_tui(args)


def _agent_subcommand_selected(args) -> bool:
    """True for ``cron run/tick``, ``gateway run``, ``mcp serve`` (see _AGENT_SUBCOMMANDS)."""
    _sub_attr, _sub_set = _AGENT_SUBCOMMANDS.get(args.command, (None, None))
    return bool(_sub_attr and getattr(args, _sub_attr, None) in _sub_set)


def _command_has_dedicated_mcp_startup(args) -> bool:
    """acp / gateway run / cron run|tick own their MCP startup on the runtime path."""
    return args.command == "acp" or (
        args.command != "mcp" and _agent_subcommand_selected(args)
    )


def _should_background_mcp_startup(args) -> bool:
    return not _is_tui_chat_launch(args) and args.command in {None, "chat", "rl"}


def _prepare_agent_startup(args) -> None:
    """Discover plugins/MCP/hooks for commands that can run an agent turn."""
    # --yolo chokepoint: HERMES_YOLO_MODE must be set before any discovery
    # below imports tools.approval, which freezes _YOLO_MODE_FROZEN at import.
    # main() sets it earlier too, but other launchers (Termux fast-CLI) reach
    # here directly, so the guarantee lives where the import is triggered.
    # See #7994.
    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"
    _apply_safe_mode(args)
    _apply_user_config_bypass(args)
    _guard_noninteractive_user_config(args)

    if not (args.command in _AGENT_COMMANDS or _agent_subcommand_selected(args)):
        return

    _accept_hooks = bool(getattr(args, "accept_hooks", False))
    if not _is_tui_chat_launch(args):
        # The TUI backend does its own discovery; the launcher only spawns Node.
        try:
            from hermes_cli.plugins import start_background_plugin_discovery

            # Daemon thread: ~150ms of manifest scanning overlaps the rest of
            # startup. Every synchronous reader goes through discover_plugins(),
            # which joins this thread first (incl. model_tools at import time).
            start_background_plugin_discovery()
        except Exception:
            logger.warning(
                "plugin discovery failed at CLI startup",
                exc_info=True,
            )
    # -t/--toolsets narrows which configured MCP servers get spawned, on
    # every discovery path (inline below, background thread, TUI/desktop
    # deferred start). Built-in toolset names never match a server key, so
    # `-t terminal` simply spawns nothing; `-t all` keeps the full set.
    try:
        from hermes_cli.mcp_startup import set_mcp_server_filter

        set_mcp_server_filter(getattr(args, "toolsets", None))
    except Exception:
        logger.debug("MCP server filter setup failed", exc_info=True)

    # TUI launches hand off to a startup path that backgrounds MCP discovery
    # with a bounded join; acp/gateway/cron do their own on the runtime path.
    _run_inline_mcp_discovery = not (
        _is_tui_chat_launch(args) or _command_has_dedicated_mcp_startup(args)
    )
    if _run_inline_mcp_discovery and _should_background_mcp_startup(args):
        try:
            from hermes_cli.mcp_startup import start_background_mcp_discovery

            start_background_mcp_discovery(
                logger=logger,
                thread_name="cli-mcp-discovery",
            )
        except Exception:
            logger.debug(
                "Background MCP tool discovery failed at CLI startup",
                exc_info=True,
            )
        _run_inline_mcp_discovery = False
    if _run_inline_mcp_discovery:
        try:  # synchronous for entrypoints without a later bounded startup path
            from hermes_cli.mcp_startup import get_mcp_server_filter
            from tools.mcp_tool_discovery import discover_mcp_tools

            _mcp_filter = get_mcp_server_filter()
            if _mcp_filter is None:
                discover_mcp_tools()
            else:
                discover_mcp_tools(allowed_mcp_names=_mcp_filter)
        except Exception:
            logger.debug(
                "MCP tool discovery failed at CLI startup",
                exc_info=True,
            )
    try:
        from hermes_cli.config import load_config
        from agent.shell_hooks import register_from_config

        _hooks_cfg = load_config()
        register_from_config(_hooks_cfg, accept_hooks=_accept_hooks)

        from agent.outbound_webhooks import (
            register_from_config as register_outbound_webhooks,
        )

        register_outbound_webhooks(_hooks_cfg)
    except Exception:
        logger.debug(
            "shell-hook registration failed at CLI startup",
            exc_info=True,
        )


def _apply_safe_mode(args) -> None:
    if not getattr(args, "safe_mode", False):
        return
    os.environ["HERMES_SAFE_MODE"] = "1"
    os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"
    os.environ["HERMES_IGNORE_RULES"] = "1"


def _apply_user_config_bypass(args) -> None:
    """Apply the explicit config bypass before any startup config reads."""
    if getattr(args, "ignore_user_config", False):
        os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"


def _guard_noninteractive_user_config(args) -> None:
    """Fail closed before a non-interactive invocation initializes providers."""
    if getattr(args, "_noninteractive_config_validated", False):
        return

    is_noninteractive = (
        bool(getattr(args, "oneshot", None))
        or bool(getattr(args, "query", None))
    )
    if not is_noninteractive:
        return

    from hermes_cli.config import (
        InvalidUserConfigError,
        require_parseable_user_config,
    )

    try:
        require_parseable_user_config(
            ignore_user_config=bool(
                getattr(args, "ignore_user_config", False)
                or getattr(args, "safe_mode", False)
            )
        )
    except InvalidUserConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    setattr(args, "_noninteractive_config_validated", True)


def _set_chat_arg_defaults(args) -> None:
    """Fill the chat-parser attrs cmd_chat reads when chat was not parsed."""
    for attr, default in [
        ("query", None),
        ("model", None),
        ("provider", None),
        ("toolsets", None),
        ("verbose", False),
        ("resume", None),
        ("continue_last", None),
        ("worktree", False),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)


def _run_oneshot_from_args(args) -> None:
    """Top-level --oneshot / -z: single-shot mode, stdout = final response only.

    Bypasses cli.py entirely; _run_and_exit_oneshot never returns.
    """
    _confirm_startup_expensive_model_override(args)
    # -z honors --resume/-c/--in exactly like chat (#105892): normalize BEFORE the
    # oneshot exit path takes over, else the flags parse fine but silently do nothing
    # and the turn starts a fresh session (every wire request loses all history).
    _resolve_chat_session_args(args, use_tui=False)
    _run_and_exit_oneshot(
        args.oneshot,
        model=getattr(args, "model", None),
        provider=getattr(args, "provider", None),
        toolsets=getattr(args, "toolsets", None),
        skills=getattr(args, "skills", None),
        usage_file=getattr(args, "usage_file", None),
        resume=getattr(args, "resume", None),
        reasoning=getattr(args, "reasoning", None),
    )


def _light_chat_parser():
    """Top-level + chat parser only (no subcommand tree); chat dispatches to cmd_chat."""
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)
    return parser


def _promote_top_level_resume(args) -> None:
    """Top-level --resume/--continue with no subcommand is a chat shortcut."""
    if (args.resume or args.continue_last) and args.command is None:
        args.command = "chat"


def _try_fast_serve_launch() -> bool:
    """Dispatch an unambiguous built-in ``serve`` without the full CLI tree.

    Desktop runs this on every cold start; building every other parser costs
    thousands of filesystem-backed lookups on Windows. Unknown or global
    arguments fall back to normal parsing so error reporting is unchanged.
    """
    if os.environ.get("HERMES_DISABLE_FAST_SERVE_LAUNCH") == "1":
        return False

    argv = sys.argv[1:]
    if not argv or argv[0] != "serve" or "-h" in argv or "--help" in argv:
        return False

    # Container routing is top-level policy and must run before host dispatch.
    try:
        from hermes_cli.config import get_container_exec_info

        if get_container_exec_info():
            return False
    except Exception:
        return False

    parser = build_serve_parser(
        cmd_dashboard=cmd_dashboard,
        add_help=False,
        exit_on_error=False,
    )
    try:
        args, unknown = parser.parse_known_args(argv[1:])
    except (argparse.ArgumentError, ValueError):
        return False
    if unknown:
        return False

    cmd_dashboard(args)
    return True


def _try_fast_chat_launch() -> bool:
    """Fast path for unambiguous interactive chat launches (all hosts).

    Building all ~40 subcommand parsers costs ~140ms the chat path never
    uses. Bails out (False) whenever the invocation is not certainly a chat
    launch — subcommand positional, ``--help``, unknown flags. Mirrors
    ``_try_termux_fast_cli_launch`` minus the Termux deferred startup; kept
    separate so phone-tuned behavior doesn't leak to desktops.
    """
    if os.environ.get("HERMES_DISABLE_FAST_CHAT_LAUNCH") == "1":
        return False
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        return False
    # Container routing must win: NixOS container mode forwards EVERY invocation.
    try:
        from hermes_cli.config import get_container_exec_info
        if get_container_exec_info():
            return False
    except Exception:
        return False
    # TUI launches keep full dispatch outside Termux (own startup path).
    if _wants_tui_early(argv):
        return False
    if _first_positional_argv() not in {None, "chat"}:
        return False

    parser = _light_chat_parser()
    try:
        args, unknown = parser.parse_known_args(_coalesce_session_name_args(argv))
    except SystemExit:
        return False
    if unknown:  # plugin subcommand or full-parser-only flag → full dispatch
        return False
    if getattr(args, "version", False):
        return False
    if getattr(args, "command", None) not in {None, "chat"}:
        return False

    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"
    _prepare_agent_startup(args)

    if getattr(args, "oneshot", None):
        _run_oneshot_from_args(args)

    _promote_top_level_resume(args)
    _set_chat_arg_defaults(args)
    cmd_chat(args)
    return True


def _try_termux_fast_cli_launch() -> bool:
    """Run obvious Termux non-TUI chat/oneshot/version paths on a light parser."""
    if not _is_termux_startup_environment():
        return False
    if os.environ.get("HERMES_TERMUX_DISABLE_FAST_CLI") == "1":
        return False

    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        return False
    if _wants_tui_early(argv):  # TUI fast path / full dispatch owns those
        return False

    if _startup_fast.is_global_fast_version_argv(argv):
        _print_version_info(check_updates=True)
        return True

    first = _first_positional_argv()
    has_oneshot = any(
        arg == "-z" or arg == "--oneshot" or arg.startswith("--oneshot=")
        for arg in argv
    )
    if not has_oneshot and first not in {None, "chat"}:
        return False

    parser = _light_chat_parser()
    args = parser.parse_args(_coalesce_session_name_args(argv))

    if getattr(args, "version", False):
        _print_version_info(check_updates=True)
        return True

    if getattr(args, "oneshot", None):
        _prepare_agent_startup(args)
        _run_oneshot_from_args(args)

    _promote_top_level_resume(args)
    if args.command in {None, "chat"}:
        _set_chat_arg_defaults(args)
        interactive_prompt = not getattr(args, "query", None) and not getattr(args, "image", None)
        if interactive_prompt:
            # Reach the prompt first; agent-only discovery on the first turn.
            setattr(args, "compact", True)
            os.environ["HERMES_DEFER_AGENT_STARTUP"] = "1"
            os.environ["HERMES_FAST_STARTUP_BANNER"] = "1"
            if getattr(args, "accept_hooks", False):
                os.environ["HERMES_ACCEPT_HOOKS"] = "1"
        else:
            _prepare_agent_startup(args)
        cmd_chat(args)
        return True

    return False


def _try_termux_fast_tui_launch() -> bool:
    """Launch obvious Termux TUI invocations before building every subparser.

    `hermes --tui` is the hot path on phones and the TUI immediately execs
    Node, so the full parser's command-module imports are pure waste there.
    """
    if not _is_termux_startup_environment():
        return False

    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        return False

    wants_tui = _wants_tui_early(sys.argv[1:])
    if not wants_tui:
        return False

    first = _first_positional_argv()
    if first not in {None, "chat"}:
        return False

    parser = _light_chat_parser()
    args = parser.parse_args(_coalesce_session_name_args(sys.argv[1:]))

    # Preserve top-level behaviours whose semantics are not "launch chat/TUI".
    if getattr(args, "version", False) or getattr(args, "oneshot", None):
        return False
    if getattr(args, "command", None) not in {None, "chat"}:
        return False
    if not _resolve_use_tui(args):
        return False

    cmd_chat(args)
    return True


def _advertise_agent_env() -> None:
    """Advertise the agent harness to child processes.

    ``AI_AGENT`` is the cross-agent standard (huggingface_hub reads it); the
    value must be our id in the public agent-harness registry
    (``hermes-agent``) — matching is exact. ``HERMES_AGENT`` is the
    Hermes-specific marker. setdefault: never clobber an outer harness.

    ``AI_AGENT`` is the emerging cross-agent standard (huggingface_hub's agent detection reads it; pi and
    other agents set it — earendil-works/pi#7493) so generic tooling can attribute subprocesses to the
    harness that spawned them. Hermes running inside another agent's terminal).
    """
    os.environ.setdefault("AI_AGENT", "hermes-agent")
    os.environ.setdefault("HERMES_AGENT", "true")


def _attach_plugin_cli_command(subparsers, cmd_info) -> None:
    """Register one plugin-provided top-level command from its descriptor."""
    plugin_parser = subparsers.add_parser(
        cmd_info["name"],
        help=cmd_info["help"],
        description=cmd_info.get("description", ""),
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
    )
    cmd_info["setup_fn"](plugin_parser)
    if cmd_info.get("handler_fn") is not None:
        plugin_parser.set_defaults(func=cmd_info["handler_fn"])


def _register_plugin_cli_commands(subparsers) -> None:
    """Register plugin-provided top-level commands (each plugin builds its own argparse tree).

    Skipped when the invocation targets a known built-in — eagerly importing
    every bundled plugin module costs 500-650ms.
    """
    if not _plugin_cli_discovery_needed():
        return
    try:
        from plugins.memory import discover_plugin_cli_commands
        from hermes_cli.plugins import discover_plugins, get_plugin_manager

        seen_plugin_commands = set()
        for cmd_info in discover_plugin_cli_commands():
            _attach_plugin_cli_command(subparsers, cmd_info)
            seen_plugin_commands.add(cmd_info["name"])

        discover_plugins()
        # The invoked platform may still be a deferred entry; import it so its
        # register_cli_command side effect runs before we read _cli_commands.
        # See #54678.
        _resolve_deferred_platform_cli_command(_first_positional_argv())
        for cmd_info in get_plugin_manager()._cli_commands.values():
            if cmd_info["name"] not in seen_plugin_commands:
                _attach_plugin_cli_command(subparsers, cmd_info)
    except Exception as _exc:
        logging.getLogger(__name__).debug("Plugin CLI discovery failed: %s", _exc)


def _cmd_sessions_lazy(args, **kwargs):
    """``hermes sessions`` handler; sessions_cmd imports only when the subcommand runs."""
    from hermes_cli.sessions_cmd import cmd_sessions

    return cmd_sessions(args, **kwargs)


def _build_cli_parser():
    """Build the full ``hermes`` argparse tree -> ``(parser, subparsers)``.

    Registration ORDER is the ``hermes --help`` order; keep it stable. Groups
    live in ``hermes_cli/subcommands/<group>.py`` with handlers injected so
    those modules never import main.
    """
    from hermes_cli._parser import build_top_level_parser

    parser, subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)

    build_model_parser(subparsers, cmd_model=cmd_model)
    build_moa_parser(subparsers)
    build_fallback_parser(subparsers)
    build_worktree_parser(subparsers)
    build_browser_parser(subparsers)
    build_secrets_parser(subparsers)
    # OUTBOUND egress firewall; ``hermes proxy`` (gateway group) is the INBOUND one.
    build_egress_parser(subparsers)
    build_migrate_parser(subparsers)
    build_codex_runtime_parser(subparsers)
    build_gateway_parser(
        subparsers, cmd_gateway=cmd_gateway, cmd_proxy=cmd_proxy, cmd_gateway_enroll=cmd_gateway_enroll
    )

    # LSP is optional — a registration failure must not break the CLI.
    try:
        from agent.lsp.cli import register_subparser as _lsp_register
        _lsp_register(subparsers)
    except Exception as _lsp_err:  # noqa: BLE001
        logger.debug("LSP CLI registration failed: %s", _lsp_err)

    build_setup_parser(subparsers, cmd_setup=cmd_setup)
    build_whatsapp_parser(subparsers, cmd_whatsapp=cmd_whatsapp)
    build_whatsapp_cloud_parser(subparsers, cmd_whatsapp_cloud=cmd_whatsapp_cloud)
    build_slack_parser(subparsers, cmd_slack=cmd_slack)

    from hermes_cli.send_cmd import register_send_subparser
    register_send_subparser(subparsers)

    build_login_parser(subparsers, cmd_login=cmd_login)
    build_logout_parser(subparsers, cmd_logout=cmd_logout)
    build_auth_parser(subparsers, cmd_auth=cmd_auth)
    build_status_parser(subparsers, cmd_status=cmd_status)
    build_pause_parser(subparsers)
    build_cron_parser(subparsers, cmd_cron=cmd_cron)
    build_sync_parser(subparsers, cmd_sync=cmd_sync)
    build_webhook_parser(subparsers, cmd_webhook=cmd_webhook)

    from hermes_cli.subcommands.peer import build_peer_parser
    build_peer_parser(subparsers)

    from hermes_cli.portal_cli import add_parser as _add_portal_parser
    _add_portal_parser(subparsers)

    from hermes_cli.kanban import build_parser as _build_kanban_parser
    _build_kanban_parser(subparsers).set_defaults(func=cmd_kanban)

    from hermes_cli.projects_cmd import build_parser as _build_project_parser
    _build_project_parser(subparsers).set_defaults(func=cmd_project)

    build_hooks_parser(subparsers, cmd_hooks=cmd_hooks)
    build_doctor_parser(subparsers, cmd_doctor=cmd_doctor)
    build_verify_parser(subparsers, cmd_verify=cmd_verify)
    build_security_parser(subparsers, cmd_security=cmd_security)
    build_approvals_parser(subparsers, cmd_approvals=cmd_approvals)
    build_dump_parser(subparsers, cmd_dump=cmd_dump)
    build_debug_parser(subparsers, cmd_debug=cmd_debug)
    build_backup_parser(subparsers, cmd_backup=cmd_backup)
    build_checkpoints_parser(subparsers)
    build_import_cmd_parser(subparsers, cmd_import=cmd_import)
    build_import_agent_parser(subparsers, cmd_import_agent=cmd_import_agent)
    build_config_parser(subparsers, cmd_config=cmd_config)
    build_skin_parser(subparsers, cmd_skin=cmd_skin)
    build_console_parser(subparsers, cmd_console=cmd_console)
    build_pairing_parser(subparsers, cmd_pairing=cmd_pairing)
    build_skills_parser(subparsers, cmd_skills=cmd_skills)
    build_bundles_parser(subparsers)
    build_plugins_parser(subparsers, cmd_plugins=cmd_plugins)

    _register_plugin_cli_commands(subparsers)

    build_curator_parser(subparsers)
    build_pets_parser(subparsers)
    build_journey_parser(subparsers)
    build_memory_parser(subparsers, cmd_memory=cmd_memory)
    build_tools_parser(subparsers, cmd_tools=cmd_tools)
    build_computer_use_parser(subparsers)
    build_mcp_parser(subparsers, cmd_mcp=cmd_mcp)
    build_sessions_parser(subparsers, cmd_sessions=_cmd_sessions_lazy)
    build_insights_parser(subparsers, cmd_insights=cmd_insights)
    build_usage_parser(subparsers)
    build_monitoring_parser(subparsers, cmd_monitoring=cmd_monitoring)
    build_claw_parser(subparsers, cmd_claw=cmd_claw)
    build_vault_parser(subparsers)
    build_update_parser(subparsers, cmd_update=cmd_update)
    build_uninstall_parser(subparsers, cmd_uninstall=cmd_uninstall)
    build_acp_parser(subparsers, cmd_acp=cmd_acp)
    build_profile_parser(subparsers, cmd_profile=cmd_profile)
    build_completion_parser(subparsers, cmd_completion=cmd_completion, parser=parser)
    build_dashboard_parser(
        subparsers,
        cmd_dashboard=cmd_dashboard,
        cmd_dashboard_register=cmd_dashboard_register,
    )
    build_desktop_web_parser(subparsers, cmd_desktop_web=cmd_desktop_web)


    # =========================================================================
    # desktop (a.k.a. gui) command
    #
    # The canonical name is "desktop"; "gui" is kept as a deprecated alias
    # for one release. The Hermes-Setup.exe success screen tells users to
    # run `hermes desktop` from a terminal, so the canonical name needs
    # to be the one that appears in --help (argparse promotes the primary
    # name; aliases stay hidden).
    # =========================================================================
    # gui command  (parser built in hermes_cli/subcommands/gui.py)
    # =========================================================================
    # "desktop" is canonical (Hermes-Setup.exe tells users to run it, so it
    # must be the name --help shows); "gui" is a deprecated alias.
    build_gui_parser(subparsers, cmd_gui=cmd_gui)
    build_logs_parser(subparsers, cmd_logs=cmd_logs)
    build_prompt_size_parser(subparsers, cmd_prompt_size=cmd_prompt_size)
    return parser, subparsers


def _parse_cli_args(parser, subparsers, argv):
    """Parse ``argv`` with the bpo-9338 subparser-routing workaround.

    On Python <3.11 argparse fails to route subcommand tokens when the parent
    has nargs='?' optionals (--continue): "unrecognized arguments: model". When
    argv holds a known subcommand token, set subparsers.required=True to force
    routing; if that fails (``hermes -c model`` — 'model' is the session name)
    fall back to the default behaviour.
    """
    import io as _io

    _processed_argv = _coalesce_session_name_args(argv)
    _known_cmds = (
        set(subparsers.choices.keys()) if hasattr(subparsers, "choices") else set()
    )
    _has_cmd_token = any(
        t in _known_cmds for t in _processed_argv if not t.startswith("-")
    )
    if not _has_cmd_token:
        subparsers.required = False
        return parser.parse_args(_processed_argv)

    subparsers.required = True
    _saved_stderr = sys.stderr
    try:
        sys.stderr = _io.StringIO()
        args = parser.parse_args(_processed_argv)
        sys.stderr = _saved_stderr
    except SystemExit as exc:
        sys.stderr = _saved_stderr
        if exc.code == 0:  # help/version already printed; don't print twice
            raise
        # Subcommand consumed as a flag value (e.g. -c model): normal parse.
        subparsers.required = False
        args = parser.parse_args(_processed_argv)
    return args


def _default_to_chat(args) -> None:
    """No subcommand given: run chat."""
    _promote_top_level_resume(args)
    _set_chat_arg_defaults(args)
    cmd_chat(args)


def main():
    """Main entry point for hermes CLI."""
    _set_process_title()
    _warn_if_unsupervised_pid1()
    _advertise_agent_env()

    # Force UTF-8 stdio on Windows before anything prints.  No-op elsewhere.
    try:
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()
    except Exception:
        pass
    # A non-UTF-8 locale that the package import had to repair would crash Python children the
    # same way. Only on that host, so a healthy UTF-8 locale keeps its children untouched.
    from hermes_cli import _stdio_repaired
    if _stdio_repaired:
        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # One TLS authority: trust the OS store before any outbound call resolves a
    # CA bundle (agent/ssl_verify.py). Never raises; False just means OpenSSL's paths.
    from agent.ssl_verify import install_truststore

    install_truststore()

    # Sweep stale ``hermes.exe.old.*`` quarantine files from previous Windows
    # updates. No-op elsewhere.
    try:
        _cleanup_quarantined_exes()
    except Exception:
        pass

    # Checkout changed since last launch → sweep stale __pycache__ once so no
    # process resolves fresh source against old bytecode. Never raises.
    _sweep_stale_bytecode_if_checkout_changed()

    # Dependency recovery already ran before imports. Report any fleet restart
    # still owed by a previous update without restarting services here.
    if "update" not in sys.argv[1:]:
        try:
            from hermes_cli.update_cmd_fleet import _warn_pending_fleet_restart_on_startup

            _warn_pending_fleet_restart_on_startup()
        except Exception:
            pass

    if _first_positional_argv() != "update":
        from hermes_cli.boot_bootstrap import maybe_run_boot_bootstrap
        from pm.paths import install_root
        maybe_run_boot_bootstrap(install_root())

    # Every dispatch, including fast chat/serve, gets one passive PM verdict.
    try:
        from hermes_cli.venv_sync import check_runtime
        from pm.paths import install_root

        problem = check_runtime(install_root())
        if problem:
            print(f"⚠ {problem}", file=sys.stderr)
    except Exception:
        import logging

        logging.getLogger(__name__).debug("pm startup check failed", exc_info=True)

    if _try_termux_fast_tui_launch():
        return
    if _try_termux_fast_cli_launch():
        return
    if _try_fast_serve_launch():
        return
    if _try_fast_chat_launch():
        return

    parser, subparsers = _build_cli_parser()

    # NixOS container mode routes ALL invocations into the managed container.
    # MUST run before parse_args() so --help, unrecognised flags and every
    # subcommand are forwarded instead of intercepted by argparse on the host.
    from hermes_cli.config import get_container_exec_info

    container_info = get_container_exec_info()
    if container_info:
        _exec_in_container(container_info, sys.argv[1:])
        sys.exit(1)  # unreachable: execvp replaces the process or raises

    args = _parse_cli_args(parser, subparsers, sys.argv[1:])

    if args.version:
        cmd_version(args)
        return

    # --yolo must be set *before* plugin discovery: tools.approval freezes
    # _YOLO_MODE_FROZEN at import; set later (inside cmd_chat) it does nothing.
    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"

    # Plugin discovery + shell hooks once, gated so introspection commands
    # (hooks list, cron list, gateway status, ...) pay no discovery cost and
    # trigger no consent prompts for hooks the user is still inspecting.
    _prepare_agent_startup(args)

    if getattr(args, "oneshot", None):
        _run_oneshot_from_args(args)

    from hermes_cli.observability.shared_metrics_disabled import set_process_surface
    set_process_surface(args.command)
    # No subcommand (optionally with top-level --resume / --continue) → chat.
    if args.command is None:
        _default_to_chat(args)
        return

    # A handler's int return code becomes the exit code (None = success).
    if hasattr(args, "func"):
        rc = args.func(args)
        if isinstance(rc, int) and rc != 0:
            sys.exit(rc)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
