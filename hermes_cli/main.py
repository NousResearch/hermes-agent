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
except ModuleNotFoundError:
    pass

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

# Early venv self-heal — MUST run before any third-party import below. A prior
# ``hermes update`` may have left a recovery marker with a core package wiped;
# the hermes_cli.config/env_loader imports further down would then crash before
# main() reaches _recover_from_interrupted_install(). ``_early_recovery`` is
# stdlib-only (safe on a corrupted venv) and repairs just enough to finish this
# import; the marker lifecycle stays with the full recovery path. Its own
# import is unguarded on purpose: same package dir, so if IT can't import
# nothing in hermes_cli can.
# It is also the canonical home of the probe/repair tables reused by the full recovery path below. See
# #57828.
from hermes_cli import _early_recovery as _early_recovery_mod

try:
    _early_recovery_mod.recover_if_needed()
except Exception:
    pass


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
# hermes_cli.config is importable. Cached so early callers don't re-parse YAML.
_EARLY_INTERFACE_CACHE: "list | None" = None


def _config_default_interface_early() -> str:
    """Return the configured default interface ("cli"/"tui") via a minimal
    YAML read. Best-effort: any error falls back to "cli" (legacy behavior)."""
    global _EARLY_INTERFACE_CACHE
    if _EARLY_INTERFACE_CACHE is not None:
        return _EARLY_INTERFACE_CACHE[0]
    value = "cli"
    try:
        home = os.environ.get("HERMES_HOME")
        if home:
            cfg_path = os.path.join(home, "config.yaml")
        else:
            cfg_path = os.path.join(os.path.expanduser("~"), ".hermes", "config.yaml")
        if os.path.exists(cfg_path):
            import yaml as _yaml_iface

            with open(cfg_path, encoding="utf-8") as _f:
                raw = _yaml_iface.load(
                    _f, Loader=getattr(_yaml_iface, "CSafeLoader", None) or _yaml_iface.SafeLoader
                ) or {}
            disp = raw.get("display", {})
            if isinstance(disp, dict):
                iface = disp.get("interface")
                if isinstance(iface, str) and iface.strip().lower() == "tui":
                    value = "tui"
    except Exception:
        value = "cli"  # best-effort — default to classic REPL on any error
    _EARLY_INTERFACE_CACHE = [value]
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
    if os.environ.get("HERMES_TUI") == "1" or "--tui" in argv:
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
from hermes_cli.subcommands.dashboard import build_dashboard_parser, build_serve_parser
from hermes_cli.subcommands.gui import build_gui_parser
from hermes_cli.subcommands.logs import build_logs_parser
from hermes_cli.subcommands.prompt_size import build_prompt_size_parser
from hermes_cli.subcommands.memory import build_memory_parser
from hermes_cli.subcommands.acp import build_acp_parser
from hermes_cli.subcommands.tools import build_tools_parser
from hermes_cli.subcommands.insights import build_insights_parser
from hermes_cli.subcommands.monitoring import build_monitoring_parser
from hermes_cli.subcommands.skills import build_skills_parser
from hermes_cli.subcommands.pairing import build_pairing_parser
from hermes_cli.subcommands.plugins import build_plugins_parser
from hermes_cli.subcommands.mcp import build_mcp_parser
from hermes_cli.subcommands.claw import build_claw_parser
from hermes_cli.subcommands.moa import build_moa_parser
from hermes_cli.subcommands.fallback import build_fallback_parser
from hermes_cli.subcommands.worktree import build_worktree_parser
from hermes_cli.subcommands.browser import build_browser_parser
from hermes_cli.subcommands.secrets import build_secrets_parser
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


def _scan_profile_flag(argv: list) -> tuple:
    """Find -p/--profile/--profile= in argv -> (name, tokens_consumed, index).

    Historically the flag worked even after the subcommand (`hermes chat -p
    coder`), so scan broadly; stop at ``--`` and at the `mcp add --args`
    passthrough region. Values that can't be profile names (pytest's
    ``-p no:xdist``) are rejected so resolve_profile_env never sys.exits on them.
    """
    from hermes_cli._parser import top_level_value_flag_sets

    value_flags, optional_value_flags = top_level_value_flag_sets()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--" or (arg == "--args" and _inside_mcp_add_args(argv, i)):
            break
        if arg in {"--profile", "-p"} and i + 1 < len(argv):
            if re.match(_PROFILE_NAME_RE, argv[i + 1]):
                return argv[i + 1], 2, i
            break
        if arg.startswith("--profile="):
            return arg.split("=", 1)[1], 1, i
        takes_value = "=" not in arg and i + 1 < len(argv) and (
            arg in value_flags
            or (arg in optional_value_flags and not argv[i + 1].startswith("-"))
        )
        i += 2 if takes_value else 1
    return None, 0, None


def _resolve_sudo_user_profile_env(name: str) -> str | None:
    """Resolve `sudo hermes -p <name>` against the invoking user's home.

    This runs before argparse, so `--run-as-user` is not available yet. For
    sudo invocations the best signal is SUDO_USER: root is only doing the
    privileged install/start action; the profile store belongs to the user.
    """
    if name == "default" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        return None
    sudo_user = os.environ.get("SUDO_USER", "").strip()
    if not sudo_user or sudo_user == "root":
        return None
    try:
        import pwd

        candidate = Path(pwd.getpwnam(sudo_user).pw_dir) / ".hermes" / "profiles" / name
        return str(candidate) if candidate.is_dir() else None
    except Exception:
        return None


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


def _desktop_ssh_backend(argv: list) -> bool:
    """A Desktop-owned ``serve --ssh-session-token-file`` child has a fixed identity too.

    The Desktop client names the remote profile explicitly (``--profile <name>``, or none for
    the root home). Following the remote host's sticky ``active_profile`` instead silently
    re-homes the backend into a profile the UI never asked for, so Settings read one
    ``config.yaml`` and the user edits another (KC's "nothing sticks over SSH").
    """
    return "--ssh-session-token-file" in argv


def _apply_profile_override() -> None:
    """Pre-parse --profile/-p and set HERMES_HOME before imports."""
    argv = sys.argv[1:]
    profile_name, consume, profile_index = _scan_profile_flag(argv)

    # HERMES_HOME already set with no explicit flag: trust it only when it
    # points at a specific profile dir ("profiles" as immediate parent). If it
    # points at the hermes root (systemd hardcodes HERMES_HOME=/root/.hermes)
    # we must still read active_profile — the user may have run
    # `hermes profile use` and the gateway should honour it (#22502).
    hermes_home_env = os.environ.get("HERMES_HOME", "")
    if profile_name is None and hermes_home_env and Path(hermes_home_env).parent.name == "profiles":
        return

    if profile_name is None and not _under_gateway_supervisor(argv) and not _desktop_ssh_backend(argv):
        try:
            from hermes_constants import get_default_hermes_root

            active_path = get_default_hermes_root() / "active_profile"
            if active_path.exists():
                name = active_path.read_text(encoding="utf-8").strip()
                if name and name != "default":
                    profile_name = name  # consume stays 0: nothing to strip
        except (UnicodeDecodeError, OSError):
            pass  # corrupted file, skip

    if profile_name is None:
        return
    try:
        from hermes_cli.profiles import resolve_profile_env

        hermes_home = resolve_profile_env(profile_name)
    except FileNotFoundError as exc:
        hermes_home = _resolve_sudo_user_profile_env(profile_name)
        if not hermes_home:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        # A bug in profiles.py must NEVER prevent hermes from starting
        print(f"Warning: profile override failed ({exc}), using default", file=sys.stderr)
        return
    os.environ["HERMES_HOME"] = hermes_home
    # Strip the flag from argv so argparse doesn't choke
    if consume > 0 and profile_index is not None:
        start = profile_index + 1  # +1 because argv is sys.argv[1:]
        sys.argv = sys.argv[:start] + sys.argv[start + consume :]


_apply_profile_override()

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

# ``update`` must not import optional secret-manager libs before ``uv``
# replaces the environment: on Windows Bitwarden's cryptography import maps
# ``_rust.pyd`` and the parent updater then blocks its own child installer.
# Profile flags are already stripped, so argv[1] is the authoritative subcommand.
# Profile flags have already been stripped above, so the first remaining argument is the authoritative
# argparse subcommand. Dotenv/managed config still loads; only external secret fetches are unnecessary for
# installation maintenance. See #73381.
load_hermes_dotenv(
    project_env=PROJECT_ROOT / ".env",
    load_external_secrets=sys.argv[1:2] != ["update"],
)

# Bridge security.redact_secrets → HERMES_REDACT_SECRETS BEFORE hermes_logging
# imports agent.redact, which snapshots the flag exactly once at import. A
# .env value still wins — this is config.yaml fallback only. network.force_ipv4
# is read from the same parse to avoid a second full load_config() (~17ms).
_FORCE_IPV4_EARLY = False
try:
    # read_raw_config()'s (mtime, size)-keyed cache means this SAME parse serves
    # hermes_logging and later raw reads: 3-4 config.yaml parses become one.
    from hermes_cli.config import read_raw_config as _read_raw_early

    _cfg_path = get_hermes_home() / "config.yaml"
    if _cfg_path.exists():
        _early_cfg_raw = _read_raw_early() or {}
        # Managed scope overlay: administrator-pinned redact_secrets /
        # force_ipv4 must win here too (load_config isn't usable yet). Fail-open.
        try:
            from hermes_cli import managed_scope
            _early_cfg_raw = managed_scope.apply_managed_overlay(_early_cfg_raw)
        except Exception:
            pass
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

from hermes_cli import __version__, __release_date__

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
from hermes_cli.main_dashboard import (
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
    _prompt_provider_choice,
    _remove_custom_provider,
)
from hermes_cli.main_install_repair import (
    _cleanup_quarantined_exes,
    _recover_from_interrupted_install,
)
from hermes_cli.main_install_repair import (  # frozen updater surface: update_cmd*.py resolve these via _m()
    ShimQuarantineError,
    _UPDATE_REEXEC_ENV,
    _clear_lazy_refresh_incomplete_marker,
    _clear_marker_file,
    _clear_update_incomplete_marker,
    _install_python_dependencies_with_optional_fallback,
    _is_termux_env,
    _is_windows,
    _is_windows_npm_path,
    _lazy_refresh_marker_path,
    _pytest_owns_live_checkout,
    _reexec_dependency_sync_off_windows_shim,
    _repair_venv_via_import_probes,
    _resolve_install_target_python,
    _resolve_node_runtime_npm,
    _resolve_update_branch,
    _run_install_with_heartbeat,
    _run_package_only_install,
    _update_marker_path,
    _venv_scripts_dir,
    _verify_console_scripts_installed,
    _verify_core_dependencies_installed,
)
from hermes_cli.main_desktop import (
    cmd_gui,
)
from hermes_cli.main_desktop import (  # frozen updater surface: update_cmd*.py resolve these via _m()
    _desktop_build_needed,
    _desktop_dist_exists,
    _desktop_macos_relaunchable_fixup,
    _desktop_packaged_executable,
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
        text = (common_dir / "packed-refs").read_text(encoding="utf-8", errors="replace")
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
            for line in git_dir.read_text(encoding="utf-8", errors="replace").splitlines():
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
                rel = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                if rel:
                    common_dir = (git_dir / rel).resolve()
            except OSError:
                pass
        head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            # Loose refs may live in the worktree gitdir OR the common dir
            # (branches created via `git worktree add` typically live in the
            # common dir's refs/heads/).
            for candidate in (git_dir, common_dir):
                ref_file = candidate / ref
                if ref_file.exists():
                    return f"git:{ref}:{ref_file.read_text(encoding='utf-8', errors='replace').strip()}"
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
    skills_dir = PROJECT_ROOT / "skills"
    try:
        stat = skills_dir.stat()
        return f"skills:{__version__}:{__release_date__}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return f"skills:{__version__}:{__release_date__}:missing"


def _termux_bundled_skills_stamp_path() -> Path:
    return get_hermes_home() / "skills" / ".termux_bundled_sync_stamp"


def _termux_bundled_skills_sync_needed() -> bool:
    if not _is_termux_startup_environment():
        return True
    if os.environ.get("HERMES_TERMUX_FORCE_SKILLS_SYNC") == "1":
        return True
    try:
        stamp = _termux_bundled_skills_stamp_path()
        return stamp.read_text(encoding="utf-8").strip() != _termux_bundled_skills_fingerprint()
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
        for line in env_file.read_text(encoding="utf-8").splitlines():
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
    """Check if at least one inference provider is usable.

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
    """Yield a ``SessionDB`` (lazy import, so test patches on ``hermes_state``
    intercept). Open failures yield None and any error raised by the ``with``
    body is swallowed — callers fall through to their ``return None``."""
    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB()
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
        import uuid as _uuid

        from hermes_state import SessionDB

        new_session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:6]}"
        db = SessionDB()
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
                    print(f"No previous {kind} session found to continue.")
                    sys.exit(1)



def _read_tui_active_session_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sid = str(data.get("session_id") or "").strip()
        return sid or None
    except Exception:
        return None


def _print_tui_exit_summary(
    session_id: Optional[str], active_session_file: Optional[str] = None
) -> None:
    """Print a shell-visible epilogue after TUI exits."""
    target = (
        _read_tui_active_session_file(active_session_file)
        or session_id
        or _resolve_last_session(source="tui")
    )
    if not target:
        return

    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        session = db.get_session(target)
        if not session:
            return

        title = db.get_session_title(target)
        message_count = int(session.get("message_count") or 0)
        if message_count == 0:
            return  # No real conversation — don't show resume info
        input_tokens = int(session.get("input_tokens") or 0)
        output_tokens = int(session.get("output_tokens") or 0)
        cache_read_tokens = int(session.get("cache_read_tokens") or 0)
        cache_write_tokens = int(session.get("cache_write_tokens") or 0)
        reasoning_tokens = int(session.get("reasoning_tokens") or 0)
        total_tokens = (
            input_tokens
            + output_tokens
            + cache_read_tokens
            + cache_write_tokens
            + reasoning_tokens
        )
    except Exception:
        return
    finally:
        if db is not None:
            db.close()

    print()
    print("Resume this session with:")
    print(f"  hermes --tui --resume {target}")
    if title:
        print(f'  hermes --tui -c "{title}"')
    print()
    print(f"Session:        {target}")
    if title:
        print(f"Title:          {title}")
    print(f"Messages:       {message_count}")
    print(
        "Tokens:         "
        f"{total_tokens} (in {input_tokens}, out {output_tokens}, "
        f"cache {cache_read_tokens + cache_write_tokens}, reasoning {reasoning_tokens})"
    )


_NPM_LOCK_RUNTIME_KEYS = frozenset(
    {
        "ideallyInert",
        "peer",
        # npm writes these boolean annotation fields non-deterministically
        # between the declarative package-lock.json and the hidden actualized
        # .package-lock.json.  The intersection comparison (see
        # _tui_need_npm_install) already handles the "field present in root
        # but absent in hidden" case for structured fields like version,
        # dependencies, license, etc.  These boolean flags need explicit
        # exclusion because when present in *both* lockfiles they may still
        # differ (e.g. dev: true → stripped in hidden).
        "dev",
        "extraneous",
        "hasInstallScript",
        "optional",
    }
)
"""Lockfile fields npm writes non-deterministically at install time.

``ideallyInert`` is npm's runtime annotation for packages it skipped installing
(per-platform opt-outs).  ``peer`` is dropped from the hidden ``.package-lock.json``
on dev-dependencies that are *also* declared as peers — the canonical
``package-lock.json`` records the dual role, but npm 9's actualized tree strips
it.  Neither key represents a real skew between what was declared and what was
installed, so we exclude them from the comparison in :func:`_tui_need_npm_install`
to avoid false-positive reinstalls on every launch.

``dev``, ``optional``, ``extraneous``, and ``hasInstallScript`` are boolean
annotations that npm populates differently in the hidden lock (npm >= 10/11
writes ``extraneous`` into the hidden lock only, and ``dev: true`` from the
root lock may be absent or ``false`` in the hidden actualized tree).
They never indicate a changed dependency — the authoritative check is the
``resolved``/``integrity`` pair, which the intersection comparison always
catches.
"""


def _workspace_root(dir: Path) -> Path:
    """Return the npm workspace root for *dir*.

    In a workspace checkout the single ``package-lock.json`` and hoisted
    ``node_modules/`` live at the workspace root (the parent of the
    sub-package directory).  Heuristic: if *dir* has a ``package.json``
    but **no** ``package-lock.json``, and its **parent** has a
    ``package-lock.json``, the parent is the workspace root.
    Otherwise *dir* itself is the root (standalone project or
    prebuilt-bundle layout).

    Used by ``_tui_need_npm_install``, ``_make_tui_argv``, and
    ``_build_web_ui`` so that lockfile/node_modules resolution and
    ``npm install`` cwd stay consistent — a single helper prevents
    the checks from diverging if someone accidentally creates a
    sub-package lockfile (e.g. running ``npm install`` in the wrong
    directory).
    """
    if (
        (dir / "package.json").is_file()
        and not (dir / "package-lock.json").is_file()
        and (dir.parent / "package-lock.json").is_file()
    ):
        return dir.parent
    return dir


def _termux_workspace_install_context(
    dir: Path, *, include_child_workspaces: bool = False
) -> tuple[Path, tuple[str, ...]]:
    """Return Termux-only ``(cwd, npm_args)`` for installing deps for *dir* only."""
    ws_root = _workspace_root(dir)
    if ws_root == dir:
        return dir, ()

    try:
        workspace = dir.relative_to(ws_root).as_posix()
    except ValueError:
        return ws_root, ()

    workspace_args: list[str] = ["--workspace", workspace]
    if include_child_workspaces:
        packages_dir = dir / "packages"
        if packages_dir.is_dir():
            for child in sorted(packages_dir.iterdir()):
                if child.is_dir() and (child / "package.json").is_file():
                    workspace_args.extend(
                        ["--workspace", child.relative_to(ws_root).as_posix()]
                    )
    workspace_args.append("--include-workspace-root=false")
    return ws_root, tuple(workspace_args)


def _npm_lock_workspace_closure(packages: dict, starts) -> Optional[set]:
    """Package-map keys reachable from the selected workspaces via npm resolution.

    *starts* is the set of workspace keys the launch install explicitly scopes
    to (a single str is accepted for convenience).  ``devDependencies`` are
    followed for **each** of those workspaces, since ``npm install`` installs
    the dev toolchain for every workspace it selects.  Returns ``None`` when
    none of *starts* are present in *packages* so callers fall back to the
    full-lockfile comparison.

    The launch install is scoped with ``npm install --workspace ui-tui`` (see
    ``_make_tui_argv``), so only the ui-tui workspace's dependency closure is
    written to the hidden ``.package-lock.json``.  On Termux it additionally
    selects ui-tui's child ``packages/*`` workspaces, so their devDependencies
    join the closure too.  The shared root ``package-lock.json`` additionally
    lists every *other* workspace's deps (``apps/desktop``, ``web``, …);
    comparing the two in full reports those unrelated packages as "missing" and
    reinstalls on every launch (#66978).

    Keys follow npm's v3 ``packages`` map (``""`` root, ``ui-tui`` /
    ``apps/desktop`` workspace members, ``node_modules/<name>`` hoisted deps,
    ``<dir>/node_modules/<name>`` nested deps).  Dependency names resolve to a
    key by walking up ``node_modules`` ancestors, mirroring node resolution, and
    workspace symlinks (``link: true``) are followed to their real entry so a
    linked workspace's own deps join the closure.
    """
    start_set = {starts} if isinstance(starts, str) else {s for s in starts if s}
    present = [s for s in start_set if s in packages]
    if not present:
        return None

    def resolve(from_key: str, dep: str) -> Optional[str]:
        base = from_key
        while True:
            prefix = f"{base}/" if base else ""
            candidate = f"{prefix}node_modules/{dep}"
            if candidate in packages:
                return candidate
            if not base:
                return None
            base = base.rsplit("/", 1)[0] if "/" in base else ""

    seen: set = set()
    stack = list(present)
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        entry = packages.get(key)
        if not isinstance(entry, dict):
            continue
        # Workspace symlink (e.g. node_modules/@hermes/ink → ui-tui/packages/…):
        # follow to the real package entry so its dependencies join the closure.
        resolved = entry.get("resolved")
        if entry.get("link") and isinstance(resolved, str) and resolved in packages:
            stack.append(resolved)
        # devDependencies are installed for each explicitly-selected workspace
        # (its build toolchain), but not for transitive deps.
        fields = ["dependencies", "optionalDependencies", "peerDependencies"]
        if key in start_set:
            fields.append("devDependencies")
        for field in fields:
            deps = entry.get(field)
            if not isinstance(deps, dict):
                continue
            for dep in deps:
                target = resolve(key, dep)
                if target is not None:
                    stack.append(target)
    return seen


def _tui_selected_workspace_keys(tui_dir: Path, ws_root: Path) -> set:
    """Lock-map keys for the workspaces the launch install scopes to.

    Mirrors ``_make_tui_argv``: always the ui-tui workspace, plus its child
    ``packages/*`` workspaces on Termux (where ``include_child_workspaces=True``
    in ``_termux_workspace_install_context``).  ``npm install`` installs the
    devDependencies of every workspace it selects, so the freshness closure must
    treat each as a dev-included root — otherwise a devDependency unique to a
    selected child is dropped from the closure and a genuine missing package
    slips past the check.  Returns an empty set when ui-tui can't be located
    under *ws_root*, so the caller falls back to the full comparison.
    """
    try:
        primary = tui_dir.relative_to(ws_root).as_posix()
    except ValueError:
        return set()
    keys = {primary}
    if _is_termux_startup_environment():
        packages_dir = tui_dir / "packages"
        if packages_dir.is_dir():
            for child in sorted(packages_dir.iterdir()):
                if child.is_dir() and (child / "package.json").is_file():
                    try:
                        keys.add(child.relative_to(ws_root).as_posix())
                    except ValueError:
                        continue
    return keys


def _tui_need_npm_install(root: Path) -> bool:
    """True when @hermes/ink is missing or node_modules is behind package-lock.json.

    Prebuilt bundle mode: when ``dist/entry.js`` exists and there is no
    ``package-lock.json`` (nix install layout only ships ``dist/`` +
    ``package.json``), skip reinstall entirely — the bundle is self-contained
    and there is nothing to install.

    With npm workspaces the single ``package-lock.json`` and the hoisted
    ``node_modules/`` live at the workspace root (the parent of the
    ``ui-tui/`` directory).  The lockfile / ink / marker checks use that
    workspace root; only the prebuilt-bundle sentinel stays relative to
    *root* (``ui-tui/dist/entry.js``).

    Compares ``package-lock.json`` against ``node_modules/.package-lock.json``
    (npm's hidden lockfile) by **content**, not mtime: git checkouts and npm
    rewrites can bump the root lockfile's timestamp even when installed deps
    already match, which used to trigger a spurious "Installing TUI
    dependencies" on every launch.

    For each entry in the root lock's ``packages`` map:
      - missing from hidden lock → reinstall (unless the entry is marked
        ``optional`` or ``peer``, which npm may intentionally skip per platform)
      - present in both → compare only the **intersection** of fields (after
        stripping ``_NPM_LOCK_RUNTIME_KEYS``).  npm's hidden lock
        intentionally omits many metadata fields (version, license, engines,
        dependencies, funding, etc.) — those one-side-only fields are normal
        npm artefacts, not real skew.  A real version/dependency change will
        change ``resolved``/``integrity``, which are present in both locks
        and will be caught by the intersection comparison.

    Extra entries that exist only in the hidden lock are ignored — stale
    transitives left over from a removed dependency don't break runtime and
    we'd rather not force a reinstall for them. Falls back to mtime
    comparison if either lockfile is unparseable.
    """
    # Prebuilt self-contained bundle (nix / packaged release): no lockfile
    # shipped, dist/entry.js is the single runtime artefact.
    entry = root / "dist" / "entry.js"
    # With npm workspaces the lockfile lives at the workspace root.
    ws_root = _workspace_root(root)
    lock = ws_root / "package-lock.json"
    if entry.is_file() and not lock.is_file():
        return False

    ink = ws_root / "node_modules" / "@hermes" / "ink" / "package.json"
    if not ink.is_file():
        return True
    if not lock.is_file():
        return False
    marker = ws_root / "node_modules" / ".package-lock.json"
    if not marker.is_file():
        return True

    # Compare lockfile contents, not mtimes: git checkouts and npm rewrites
    # can bump the root lockfile timestamp even when installed deps already
    # match. Fall back to mtime when either file is unparseable.
    try:
        wanted = json.loads(lock.read_text(encoding="utf-8")).get("packages") or {}
        installed = json.loads(marker.read_text(encoding="utf-8")).get("packages") or {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return lock.stat().st_mtime > marker.stat().st_mtime

    def entries_differ(pkg: dict, installed_pkg: dict) -> bool:
        # Only compare keys present in *both* lockfiles with non-null values.
        # npm's hidden .package-lock.json intentionally omits many metadata
        # fields the root lock records (version, dependencies, license,
        # engines, bin, ...), and npm >= 10/11 writes a further *reduced*
        # hidden lockfile that stores some of them as null.  Missing- or
        # null-on-one-side is a normal npm artefact, not a real skew.  The
        # authoritative fields "resolved" and "integrity" are present in both
        # locks for installed packages, so a genuinely stale install (root
        # lockfile bumped while node_modules is behind) still differs on them.
        a = {k: v for k, v in pkg.items() if k not in _NPM_LOCK_RUNTIME_KEYS}
        b = {
            k: v
            for k, v in installed_pkg.items()
            if k not in _NPM_LOCK_RUNTIME_KEYS
        }
        for k in a.keys() & b.keys():
            if a[k] is None or b[k] is None:
                continue
            if a[k] != b[k]:
                return True
        return False

    # In a shared workspace checkout the launch install is scoped to the ui-tui
    # workspace (plus its child packages/* workspaces on Termux), so only that
    # dependency closure lands in the hidden lock.  Limit the comparison to the
    # same selected-workspace closure so unrelated workspace deps (apps/desktop,
    # web, …) don't force a reinstall every launch (#66978).  Standalone /
    # own-lockfile layouts (ws_root == root) do a full install, so keep the full
    # comparison; a missing/unlocatable workspace falls back to it too.
    closure: Optional[set] = None
    if ws_root != root:
        selected = _tui_selected_workspace_keys(root, ws_root)
        if selected:
            closure = _npm_lock_workspace_closure(wanted, selected)

    for name, pkg in wanted.items():
        if not name:
            continue

        if closure is not None and name not in closure:
            continue

        if not isinstance(pkg, dict):
            continue

        if name not in installed:
            # Workspace link entries (`"link": true`, paths outside
            # node_modules/ like `apps/desktop`, `node_modules/web`) are never
            # materialized by a partial `npm install --workspace ui-tui` —
            # they're deliberately skipped (see #38772) and would otherwise
            # force a reinstall on every launch.
            if pkg.get("optional") or pkg.get("peer") or pkg.get("link"):
                continue
            if not name.startswith("node_modules/"):
                continue
            return True

        if isinstance(installed[name], dict) and entries_differ(
            pkg, installed[name]
        ):
            return True

    return False


_TUI_BUILD_INPUT_DIRS = (
    "src",
    "packages/hermes-ink/src",
)

_TUI_BUILD_INPUT_FILES = (
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "tsconfig.build.json",
    "babel.compiler.config.cjs",
    "scripts/build.mjs",
    "packages/hermes-ink/package.json",
    "packages/hermes-ink/index.js",
    "packages/hermes-ink/text-input.js",
)

_TUI_BUILD_INPUT_SUFFIXES = frozenset(
    {".cjs", ".js", ".jsx", ".json", ".mjs", ".ts", ".tsx"}
)


def _iter_tui_build_inputs(root: Path):
    """Yield source/config files that affect ``ui-tui/dist/entry.js``."""
    for rel in _TUI_BUILD_INPUT_FILES:
        path = root / rel
        if path.is_file():
            yield path

    for rel in _TUI_BUILD_INPUT_DIRS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in _TUI_BUILD_INPUT_SUFFIXES:
                yield path


def _tui_need_rebuild(root: Path) -> bool:
    """True when ``dist/entry.js`` is missing or older than TUI inputs.

    The TUI bundle is self-contained. Rebuilding it on every launch adds a
    visible cold-start tax on slow Termux CPUs, while a simple mtime freshness
    check still rebuilds immediately after source updates, dependency updates,
    or local edits. Set ``HERMES_TUI_FORCE_BUILD=1`` to force the old behaviour.
    """
    force = (os.environ.get("HERMES_TUI_FORCE_BUILD") or "").strip().lower()
    if force in {"1", "true", "yes", "on"}:
        return True

    entry = root / "dist" / "entry.js"
    try:
        output_mtime = entry.stat().st_mtime
    except OSError:
        return True

    for path in _iter_tui_build_inputs(root):
        try:
            if path.stat().st_mtime > output_mtime:
                return True
        except OSError:
            return True
    return False


def _ensure_tui_node() -> None:
    """Make sure `node` + `npm` are on PATH for the TUI.

    If either is missing and scripts/lib/node-bootstrap.sh is available, source
    it and call `ensure_node` (fnm/nvm/proto/brew/bundled cascade). After
    install, capture the resolved node binary path from the bash subprocess
    and prepend its directory to os.environ["PATH"] so shutil.which finds the
    new binaries in this Python process — regardless of which version manager
    was used (nvm, fnm, proto, brew, or the bundled fallback).

    Idempotent no-op when node+npm are already discoverable. Set
    ``HERMES_SKIP_NODE_BOOTSTRAP=1`` to disable auto-install.
    """
    if shutil.which("node") and shutil.which("npm"):
        return
    if os.environ.get("HERMES_SKIP_NODE_BOOTSTRAP"):
        return

    helper = PROJECT_ROOT / "scripts" / "lib" / "node-bootstrap.sh"
    if not helper.is_file():
        return

    from hermes_constants import get_hermes_home

    hermes_home = str(get_hermes_home())
    try:
        # Helper writes logs to stderr; we ask bash to print `command -v node`
        # on stdout once ensure_node succeeds. Subshell PATH edits don't leak
        # back into Python, so the stdout capture is the bridge.
        result = subprocess.run(
            [
                "bash",
                "-c",
                f'source "{helper}" >&2 && ensure_node >&2 && command -v node',
            ],
            env={**os.environ, "HERMES_HOME": hermes_home},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return

    parts = os.environ.get("PATH", "").split(os.pathsep)
    extras: list[Path] = []

    resolved = (result.stdout or "").strip()
    if resolved:
        extras.append(Path(resolved).resolve().parent)

    extras.extend([Path(hermes_home) / "node" / "bin", Path.home() / ".local" / "bin"])

    for extra in extras:
        s = str(extra)
        if extra.is_dir() and s not in parts:
            parts.insert(0, s)
    os.environ["PATH"] = os.pathsep.join(parts)


def _find_bundled_tui(hermes_cli_dir: Path | None = None) -> Path | None:
    """Find a pre-built TUI entry.js bundled in the wheel."""
    if hermes_cli_dir is None:
        hermes_cli_dir = Path(__file__).parent
    bundled = hermes_cli_dir / "tui_dist" / "entry.js"
    return bundled if bundled.is_file() else None


def _restore_tui_workspace(tui_dir: Path) -> bool:
    """Try to restore a missing ``ui-tui/`` from git, returning True on success.

    On Windows an antivirus / NTFS filter driver can leave tracked ``ui-tui/``
    files deleted in the working tree after ``hermes update`` (HEAD stays
    intact; the files just vanish — see issue #49145). Those files are tracked,
    so ``git restore`` puts them back deterministically. Best-effort: returns
    False (rather than raising) when git is unavailable, this isn't a checkout,
    or the restore leaves the directory still missing — the caller then prints
    the manual-recovery message.
    """
    git = shutil.which("git")
    if not git or not (tui_dir.parent / ".git").exists():
        return False
    try:
        subprocess.run(
            [git, "restore", "--", tui_dir.name],
            cwd=str(tui_dir.parent),
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            check=False,
        )
    except OSError:
        return False
    return tui_dir.is_dir()


def _ensure_tui_workspace(tui_dir: Path) -> None:
    """Ensure ``ui-tui/`` exists before any npm/node subprocess uses it as cwd.

    Without this, a missing workspace falls through to ``subprocess.run(...,
    cwd=<missing ui-tui>)``, which crashes with ``NotADirectoryError``
    (``WinError 267`` on Windows) instead of a usable message (#49145). We
    first try to self-heal via ``git restore``; only if that can't recover the
    directory do we abort with concrete manual-recovery steps.
    """
    if tui_dir.is_dir():
        return

    if _restore_tui_workspace(tui_dir):
        if not os.environ.get("HERMES_QUIET"):
            print(f"Restored missing TUI workspace: {tui_dir}")
        return

    print(
        "Error: the TUI workspace is missing from this Hermes checkout.\n"
        f"Expected directory: {tui_dir}\n"
        "This usually means `hermes update` left tracked ui-tui files deleted.\n"
        "Recovery:\n"
        "  1. From the Hermes checkout, run `git restore -- ui-tui`\n"
        "  2. Run `npm install --silent --no-fund --no-audit --progress=false`\n"
        "  3. Retry `hermes --tui`\n"
        "If the checkout is still inconsistent, run `hermes update --force`.",
        file=sys.stderr,
    )
    sys.exit(1)


def _npm_lifecycle_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Build a clean environment for the pinned UI toolchain lifecycle."""
    run_env = {**os.environ, **(env or {}), "CI": "1"}
    # esbuild treats this as an executable override. If a shell points it at a
    # different release, the pinned package's postinstall rejects that binary.
    run_env.pop("ESBUILD_BINARY_PATH", None)
    return run_env


def _make_tui_argv(tui_dir: Path, tui_dev: bool) -> tuple[list[str], Path]:
    """TUI: --dev → tsx src; else node dist (HERMES_TUI_DIR prebuilt or esbuild)."""
    _ensure_tui_node()

    def _node_bin(bin: str) -> str:
        if bin == "node":
            env_node = os.environ.get("HERMES_NODE")
            if env_node and os.path.isfile(env_node) and os.access(env_node, os.X_OK):
                return env_node
        # find_node_executable() prefers the managed $HERMES_HOME/node tree,
        # which is not on PATH — a bare which() would declare "node not found"
        # and exit on an install whose only Node is the one Hermes installed,
        # and would pick a system Node over the managed one when both exist.
        from hermes_constants import find_node_executable

        path = find_node_executable(bin)
        if not path and bin == "node":
            try:
                from hermes_cli.dep_ensure import ensure_dependency
                if ensure_dependency("node"):
                    path = find_node_executable("node")
            except Exception:
                pass
        if not path:
            print(f"{bin} not found — install Node.js to use the TUI.")
            sys.exit(1)
        return path

    # Footgun: --dev against a prebuilt bundle that has no source/node_modules.
    ext_dir = os.environ.get("HERMES_TUI_DIR")
    if tui_dev and ext_dir:
        print(
            f"Error: --dev is incompatible with HERMES_TUI_DIR={ext_dir}\n"
            f"The prebuilt TUI has no source code to hot-reload.\n"
            f"Unset HERMES_TUI_DIR (e.g. `unset HERMES_TUI_DIR`) to use --dev from a checkout.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Prebuilt bundle (nix / packaged release / Docker image): just run it.
    #
    # This must run BEFORE _ensure_tui_workspace() below. A prebuilt install
    # (Docker image, Nix build, or prior `npm run build`) ships
    # hermes_cli/tui_dist/entry.js but never ships ui-tui/ at all (that
    # directory only exists in a git checkout) — so requiring the workspace
    # to exist first made every prebuilt dashboard Chat tab connection
    # hard-exit before it ever got a chance to try the bundled entry.js it
    # already has. See #56665.
    if not tui_dev:
        if ext_dir:
            p = Path(ext_dir)
            if (p / "dist" / "entry.js").is_file():
                node = _node_bin("node")
                return [node, "--expose-gc", str(p / "dist" / "entry.js")], p

        # 1b. Bundled prebuilt TUI (Docker image, Nix build, or prior npm build)
        bundled = _find_bundled_tui()
        if bundled is not None:
            node = _node_bin("node")
            return [node, "--expose-gc", str(bundled)], bundled.parent

    # No prebuilt bundle available (or --dev, which never uses one) — we're
    # about to npm install/build from source, so the workspace must exist.
    if not ext_dir:
        _ensure_tui_workspace(tui_dir)

    # 2. Normal flow: npm install if needed, always esbuild, then node dist/entry.js.
    #    --dev flow: npm install if needed, then tsx src/entry.tsx.
    #    Existing desktop behaviour runs npm from the workspace root.  Termux
    #    scopes the install to ui-tui so launch does not pull desktop/web
    #    dependencies into the hot path.
    did_install = False
    termux_startup = _is_termux_startup_environment()
    termux_need_rebuild = False
    if termux_startup and not tui_dev:
        termux_need_rebuild = _tui_need_rebuild(tui_dir)

    skip_install_for_fresh_termux_bundle = (
        termux_startup and not tui_dev and not termux_need_rebuild
    )
    if (
        not skip_install_for_fresh_termux_bundle
        and _tui_need_npm_install(tui_dir)
    ):
        npm = _node_bin("npm")
        if not os.environ.get("HERMES_QUIET"):
            print("Installing TUI dependencies…")
        npm_cwd = _workspace_root(tui_dir)
        # --workspace ui-tui avoids resolving apps/desktop (Electron + node-pty).
        # See #38772.
        # When ui-tui/ has its own package-lock.json (e.g. curl install),
        # _workspace_root() returns tui_dir itself.  Passing --workspace in
        # that case fails because npm cannot find a workspace named "ui-tui"
        # inside ui-tui/.  See #42973.
        npm_workspace_args: tuple[str, ...] = () if npm_cwd == tui_dir else ("--workspace", "ui-tui")
        if termux_startup:
            npm_cwd, npm_workspace_args = _termux_workspace_install_context(
                tui_dir,
                include_child_workspaces=True,
            )
        npm_install_cmd = [
            npm,
            "install",
            *npm_workspace_args,
            # --include=dev: ui-tui's build toolchain (esbuild, typescript)
            # lives in devDependencies. An inherited NODE_ENV=production
            # (e.g. from a container shell or a parent TUI launch) or an
            # npm `omit=dev` config would silently skip them and the TUI
            # build would fail. See _run_npm_install_deterministic.
            "--include=dev",
            "--silent",
            "--no-fund",
            "--no-audit",
            "--progress=false",
        ]

        def _run_tui_install() -> subprocess.CompletedProcess:
            from hermes_constants import with_hermes_node_path

            # Managed tree first on PATH: if the EBADENGINE repair below
            # provisioned a managed Node, npm's shebang/lifecycle scripts must
            # resolve that node, not the mismatched system one.
            return subprocess.run(
                npm_install_cmd,
                cwd=str(npm_cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_npm_lifecycle_env(with_hermes_node_path()),
            )

        result = _run_tui_install()
        if result.returncode != 0:
            # An npm outside the root package.json's `engines.npm` range fails
            # here before doing any work; repair once (upgrade a Hermes-managed
            # npm in place, or provision a managed runtime when the npm belongs
            # to the user) and retry rather than dumping EBADENGINE at the user.
            from hermes_cli.npm_engine import maybe_repair_npm_engine

            combined_output = f"{result.stdout or ''}\n{result.stderr or ''}"
            repaired_npm = maybe_repair_npm_engine(npm, combined_output)
            if repaired_npm:
                npm = repaired_npm
                npm_install_cmd[0] = repaired_npm
                result = _run_tui_install()
        if result.returncode != 0:
            combined = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
            preview = "\n".join(combined.splitlines()[-30:])
            print("npm install failed.")
            if preview:
                print(preview)
            sys.exit(1)
        did_install = True

    if tui_dev:
        # Keep the local @hermes/ink package exports in sync with source.
        # --dev runs src/entry.tsx directly, but @hermes/ink resolves through
        # packages/hermes-ink/dist/entry-exports.js. If that dist bundle is
        # stale after a pull, newer hooks/components can exist in src while
        # being missing at runtime (e.g. useCursorAdvance). Prebuild it here.
        npm = _node_bin("npm")
        ink_dir = tui_dir / "packages" / "hermes-ink"
        result = subprocess.run(
            [npm, "run", "build"],
            cwd=str(ink_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_npm_lifecycle_env(),
        )
        if result.returncode != 0:
            combined = f"{result.stdout or ''}{result.stderr or ''}".strip()
            preview = "\n".join(combined.splitlines()[-30:])
            print("TUI dev prebuild failed.")
            if preview:
                print(preview)
            sys.exit(1)

        tsx = tui_dir / "node_modules" / ".bin" / "tsx"
        if tsx.exists():
            return [str(tsx), "src/entry.tsx"], tui_dir
        return [npm, "start"], tui_dir

    # Skip esbuild when dist/entry.js is newer than TUI inputs unless npm
    # install just ran (deps may have changed) or HERMES_TUI_FORCE_BUILD=1.
    # The build subprocess captures stdout, so skipping stale rebuilds avoids
    # long silent PTY periods before Ink starts (local dev + CI smoke).
    should_build = did_install or _tui_need_rebuild(tui_dir)

    if should_build:
        npm = _node_bin("npm")
        result = subprocess.run(
            [npm, "run", "build"],
            cwd=str(tui_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_npm_lifecycle_env(),
        )
        if result.returncode != 0:
            combined = f"{result.stdout or ''}{result.stderr or ''}".strip()
            preview = "\n".join(combined.splitlines()[-30:])
            print("TUI build failed.")
            if preview:
                print(preview)
            sys.exit(1)

    node = _node_bin("node")
    return [node, "--expose-gc", str(tui_dir / "dist" / "entry.js")], tui_dir


def _normalize_tui_toolsets(toolsets: object) -> list[str]:
    """Normalize argparse/Fire-style toolset input for the TUI subprocess."""
    try:
        from hermes_cli.oneshot import _normalize_toolsets

        return _normalize_toolsets(toolsets) or []
    except (AttributeError, ImportError):
        if not toolsets:
            return []

        raw_items = [toolsets] if isinstance(toolsets, str) else toolsets
        if not isinstance(raw_items, (list, tuple)):
            raw_items = [raw_items]

        normalized: list[str] = []
        for item in raw_items:
            if isinstance(item, str):
                normalized.extend(part.strip() for part in item.split(","))
            else:
                normalized.append(str(item).strip())

        return [item for item in normalized if item]


def _read_cgroup_memory_limit() -> Optional[int]:
    """Return the container memory limit in bytes, or None if unconstrained.

    Node's V8 heap is NOT cgroup-aware: with a flat ``--max-old-space-size=8192``
    it happily grows the heap toward 8GB regardless of the container's real
    memory limit.  In a Docker/k8s container capped below ~9-10GB, the cgroup
    OOM-killer SIGKILLs Node before V8's own heap monitor ever fires — which
    runs no JS handler, writes no ``[tui-parent]`` breadcrumb, and the user
    sees only a bare gateway ``stdin EOF``.  Reading the real cgroup limit lets
    us size the heap cap below it so V8 GCs/exits gracefully instead of being
    reaped silently.

    Checks cgroup v2 (``/sys/fs/cgroup/memory.max``) then v1
    (``/sys/fs/cgroup/memory/memory.limit_in_bytes``).  A literal ``max`` (v2)
    or the v1 "unlimited" sentinel (a huge near-INT64 value) means no limit.
    """
    candidates = (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    )
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
        except (OSError, ValueError):
            continue
        if raw == "max":
            return None
        if not raw:
            # Blank/empty file: no usable value here. Fall through to the next
            # candidate (don't mistake an empty v2 file for "unlimited").
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        if limit <= 0:
            continue
        # cgroup v1 reports "unlimited" as a huge value (often
        # 0x7FFFFFFFFFFFF000 ≈ 9.2 EB, sometimes PAGE_COUNTER_MAX). Anything
        # at/above ~1 PB is effectively unconstrained — treat as no limit.
        if limit >= (1 << 50):
            return None
        return limit
    return None


def _resolve_tui_heap_mb(default_mb: int = 8192) -> int:
    """Pick a V8 ``--max-old-space-size`` (MB) that fits the container.

    Returns ``default_mb`` (8192) when unconstrained or when the box is large
    enough that 8GB fits.  In a memory-limited container, returns ~75% of the
    cgroup limit so the heap + non-heap RSS stays under the cgroup ceiling,
    clamped to a sane floor (1536MB — below this V8 GC-thrashes and the TUI
    is barely usable).  Never exceeds ``default_mb``.
    """
    limit = _read_cgroup_memory_limit()
    if not limit:
        return default_mb
    limit_mb = limit // (1024 * 1024)
    # Leave headroom for non-heap RSS (Node internals, buffers, the Python
    # gateway child shares the same cgroup): cap the heap at 75% of the limit.
    sized = int(limit_mb * 0.75)
    if sized >= default_mb:
        return default_mb
    # Floor so a tiny limit doesn't drive V8 into constant GC. If the container
    # is smaller than the floor, honor the limit-derived value anyway (better a
    # graceful V8 exit than a silent cgroup kill).
    return max(1536, sized) if limit_mb > 2048 else sized


def _safe_tui_cwd(env: Optional[dict] = None) -> str:
    """Return a stable cwd value for the Node TUI child environment."""
    try:
        return os.getcwd()
    except FileNotFoundError:
        candidate = ((env or {}).get("PWD") or os.environ.get("PWD") or "").strip()
        if candidate and Path(candidate).is_dir():
            return candidate
        return str(PROJECT_ROOT)


def _apply_tui_python_env(env: dict) -> None:
    """Seed/repair Python-related env vars shared by CLI and dashboard TUI launches."""
    src_root = str(env.get("HERMES_PYTHON_SRC_ROOT") or "").strip()
    if not src_root or not Path(src_root).is_dir():
        env["HERMES_PYTHON_SRC_ROOT"] = str(PROJECT_ROOT)

    cwd = str(env.get("HERMES_CWD") or "").strip()
    if not cwd or not Path(cwd).is_dir():
        env["HERMES_CWD"] = _safe_tui_cwd(env)

    python = str(env.get("HERMES_PYTHON") or "").strip()
    if os.path.dirname(python):
        python_path = Path(python)
        if not python_path.is_absolute():
            python_path = Path(env["HERMES_CWD"]) / python_path
        python_is_executable = python_path.is_file() and os.access(python_path, os.X_OK)
    else:
        python_is_executable = bool(shutil.which(python, path=env.get("PATH")))
    if not python_is_executable:
        env["HERMES_PYTHON"] = sys.executable


def _launch_tui(
    resume_session_id: Optional[str] = None,
    tui_dev: bool = False,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    toolsets: object = None,
    skills: object = None,
    verbose: Optional[bool] = None,
    quiet: bool = False,
    query: Optional[str] = None,
    image: Optional[str] = None,
    worktree: bool = False,
    checkpoints: bool = False,
    pass_session_id: bool = False,
    max_turns: Optional[int] = None,
    accept_hooks: bool = False,
):
    """Replace current process with the TUI."""
    tui_dir = PROJECT_ROOT / "ui-tui"

    import tempfile

    # TUI child is a hermes process: propagate the profile-home contract via
    # the single factory; keep secrets (the TUI/agent needs provider creds).
    from tools.environments.local import build_subprocess_env
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
    try:
        from hermes_cli.config import apply_terminal_config_to_env
        apply_terminal_config_to_env(env=env)
    except Exception:
        logger.debug("Failed to apply terminal config bridge for TUI launch", exc_info=True)
    active_session_fd, active_session_file = tempfile.mkstemp(
        prefix="hermes-tui-active-session-", suffix=".json"
    )
    os.close(active_session_fd)
    env["HERMES_TUI_ACTIVE_SESSION_FILE"] = active_session_file
    env.setdefault("NODE_ENV", "development" if tui_dev else "production")

    wt_info = None
    if worktree:
        try:
            from cli import (
                _cleanup_worktree,
                _git_repo_root,
                _maintain_pack_health,
                _prune_stale_worktrees,
                _setup_worktree,
            )

            repo = _git_repo_root()
            if repo:
                _prune_stale_worktrees(repo)
                # Same maintenance pass as the CLI path: repack on pack
                # sprawl so `worktree add` never crawls on a multi-agent box
                # (cli._maintain_pack_health is a cheap no-op below the
                # threshold). Runs on a thread — the TUI path calls the
                # pruner synchronously, and a repack must not block launch.
                import threading as _threading

                _threading.Thread(
                    target=_maintain_pack_health,
                    args=(repo,),
                    name="pack-maintenance",
                    daemon=True,
                ).start()
            wt_info = _setup_worktree()
        except Exception as exc:
            print(f"✗ Failed to create TUI worktree: {exc}", file=sys.stderr)
            wt_info = None
        if not wt_info:
            sys.exit(1)
        env["HERMES_CWD"] = wt_info["path"]
        env["TERMINAL_CWD"] = wt_info["path"]

    _apply_tui_python_env(env)

    if model:
        env["HERMES_MODEL"] = model
        env["HERMES_INFERENCE_MODEL"] = model
    if provider:
        env["HERMES_TUI_PROVIDER"] = provider
        env["HERMES_INFERENCE_PROVIDER"] = provider
    tui_toolsets = _normalize_tui_toolsets(toolsets)
    if tui_toolsets:
        env["HERMES_TUI_TOOLSETS"] = ",".join(tui_toolsets)
    if skills:
        if isinstance(skills, (list, tuple)):
            flattened = []
            for item in skills:
                flattened.extend(
                    part.strip() for part in str(item).split(",") if part.strip()
                )
            if flattened:
                env["HERMES_TUI_SKILLS"] = ",".join(flattened)
        else:
            value = str(skills).strip()
            if value:
                env["HERMES_TUI_SKILLS"] = value
    if query:
        env["HERMES_TUI_QUERY"] = query
    if image:
        env["HERMES_TUI_IMAGE"] = image
    if checkpoints:
        env["HERMES_TUI_CHECKPOINTS"] = "1"
    if pass_session_id:
        env["HERMES_TUI_PASS_SESSION_ID"] = "1"
    if max_turns is not None:
        env["HERMES_TUI_MAX_TURNS"] = str(max_turns)
    if verbose:
        env["HERMES_TUI_TOOL_PROGRESS"] = "verbose"
    elif quiet:
        env["HERMES_TUI_TOOL_PROGRESS"] = "off"
    if accept_hooks:
        env["HERMES_ACCEPT_HOOKS"] = "1"
    # Guarantee a generous V8 heap for the TUI. Default node cap is ~1.5–4GB
    # depending on version and can fatal-OOM on long sessions with large
    # transcripts / reasoning blobs. We target 8GB on an unconstrained host,
    # but V8 is NOT cgroup-aware: in a memory-limited Docker/k8s container a
    # flat 8GB heap grows past the container limit and the cgroup OOM-killer
    # SIGKILLs Node — running no JS handler, writing no breadcrumb, leaving the
    # user with only a bare gateway `stdin EOF`. _resolve_tui_heap_mb() reads
    # the real cgroup limit and sizes the cap below it so V8 GCs/exits
    # gracefully (and the memory monitor's onCritical breadcrumb can fire)
    # instead of being reaped silently. Token-level merge: respect any
    # user-supplied --max-old-space-size (they may have set it higher).
    # --expose-gc is *not* added here: Node rejects it in NODE_OPTIONS
    # ("--expose-gc is not allowed in NODE_OPTIONS") and refuses to start.
    # It is passed as a direct argv flag in _make_tui_argv() instead.
    _tokens = env.get("NODE_OPTIONS", "").split()
    if not any(t.startswith("--max-old-space-size=") for t in _tokens):
        _tokens.append(f"--max-old-space-size={_resolve_tui_heap_mb()}")
    env["NODE_OPTIONS"] = " ".join(_tokens)
    # HERMES_TUI_RESUME is an internal hand-off from the Python wrapper to the
    # Ink app.  Because we start from a full os.environ snapshot (via
    # build_subprocess_env), an exported/stale value
    # in the user's shell would otherwise make a plain `hermes --tui` try to
    # resume a non-existent session and leave the UI at "error: session not
    # found" with no live session.  Only forward a resume id that argparse
    # resolved for this invocation; direct `node ui-tui/dist/entry.js` users can
    # still set HERMES_TUI_RESUME themselves.
    env.pop("HERMES_TUI_RESUME", None)
    if resume_session_id:
        env["HERMES_TUI_RESUME"] = resume_session_id

    argv, cwd = _make_tui_argv(tui_dir, tui_dev)
    code: Optional[int] = None
    try:
        try:
            code = subprocess.call(argv, cwd=str(cwd), env=env)
        except KeyboardInterrupt:
            code = 130

        if code in {0, 130}:
            _print_tui_exit_summary(resume_session_id, active_session_file)
    finally:
        try:
            os.unlink(active_session_file)
        except OSError:
            pass
        if wt_info:
            try:
                _cleanup_worktree(wt_info)
            except Exception:
                pass

    # Exit code 42 = TUI requested an update. Relaunch as `hermes update` so
    # the user sees update output directly and gets the new version.
    # preserve_inherited=False ensures --tui and other flags are NOT carried
    # into the update subcommand.
    if code == 42:
        from hermes_cli.relaunch import relaunch

        print()
        print("⚕ Launching update...")
        print()
        relaunch(["update"], preserve_inherited=False)

    sys.exit(code)


def _pin_kanban_board_env() -> None:
    """Pin the active kanban board into ``HERMES_KANBAN_BOARD`` for the chat session.

    Without this, in-process tools (``kanban_*``) and shelled-out CLI calls
    (``hermes kanban …``) resolve the board on different paths: the env-pin if
    set, otherwise the global ``<root>/kanban/current`` file. A concurrent
    ``hermes kanban boards switch`` from another session can flip the file
    mid-turn, so the same chat sees its tool calls hit board A while its shell
    calls hit board B (#20074). Pinning at chat boot mirrors what the
    dispatcher already does for spawned workers.
    """
    if os.environ.get("HERMES_KANBAN_BOARD"):
        return
    try:
        from hermes_cli.kanban_db import get_current_board

        os.environ["HERMES_KANBAN_BOARD"] = get_current_board()
    except Exception:
        pass


def _sync_bundled_skills_quietly() -> None:
    """Seed ``~/.hermes/skills/`` with the bundled skill library on first launch.

    Called from any CLI entrypoint that the user might use as their first
    interaction with Hermes — chat, dashboard (the desktop GUI's backend),
    and gateway. The skills_sync module is manifest-based and idempotent:
    skipped skills cost ~milliseconds, so calling this repeatedly is fine.

    Failures are swallowed because skills are an enhancement, not a hard
    dependency. Hermes still functions without them; the user just sees an
    empty skills library.
    """
    try:
        from tools.skills_sync import sync_skills

        sync_skills(quiet=True)
    except Exception:
        pass


def _resolve_use_tui(args) -> bool:
    """Decide whether to launch the TUI for a chat/bare invocation.

    Precedence (highest first):
      1. ``--cli`` flag         → always classic REPL
      2. ``--tui`` flag         → always TUI (explicit ask)
      3. no TTY                 → always classic (ambient prefs don't apply)
      4. ``HERMES_TUI=1`` env   → TUI
      5. ``display.interface`` config value ("cli" | "tui")
      6. default → classic REPL

    Explicit flags always win over config so muscle memory and scripts keep
    working regardless of the configured default.

    The TTY gate (3) is load-bearing: ambient TUI preferences (env var or
    config default) must never hijack a NON-interactive invocation. Kanban
    workers, cron jobs, and pipelines run ``hermes … chat -q`` with stdout
    on a pipe; booting the Ink TUI there hits its no-TTY bail-out, which
    prints a resume hint and exits 0 — a kanban worker then dies with
    "exited cleanly without calling kanban_complete — protocol violation"
    on every attempt (found dogfooding the desktop kanban board). A user
    who *explicitly* passes ``--tui`` still gets the informative bail-out.
    """
    if getattr(args, "cli", False):
        return False
    if getattr(args, "tui", False):
        return True
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except Exception:
        return False
    if os.environ.get("HERMES_TUI") == "1":
        return True
    try:
        from hermes_cli.config import load_config

        iface = (load_config().get("display", {}) or {}).get("interface", "cli")
        return isinstance(iface, str) and iface.strip().lower() == "tui"
    except Exception:
        return False


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
            with open(_qfile, "r", encoding="utf-8", errors="replace") as _fh:
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
    use_tui = _resolve_use_tui(args)

    _resolve_chat_session_args(args, use_tui)

    _warn_retired_xai_models()

    # First-run guard: check if any provider is configured before launching
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

    _pin_kanban_board_env()
    _confirm_startup_expensive_model_override(args)

    passthrough = {k: getattr(args, k, d) for k, d in _CHAT_PASSTHROUGH}
    if use_tui:
        _launch_tui(
            passthrough.pop("resume"),
            tui_dev=getattr(args, "tui_dev", False),
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
        "ignore_rules": getattr(args, "ignore_rules", False) or safe_mode,
        "ignore_user_config": getattr(args, "ignore_user_config", False) or safe_mode,
        "compact": getattr(args, "compact", False),
        **{k: getattr(args, k, d) for k, d in _CHAT_PASSTHROUGH},
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        from cli import main as cli_main

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
    surfaces the return code to ``main()`` (only kanban/project propagate).
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
cmd_doctor = _forward_command("cmd_doctor", "hermes_cli.doctor", "run_doctor", doc='Check configuration and dependencies.')
cmd_dump = _forward_command("cmd_dump", "hermes_cli.dump", "run_dump", doc='Dump setup summary for support/debugging.')
cmd_debug = _forward_command("cmd_debug", "hermes_cli.debug", "run_debug", doc='Debug tools (share report, etc.).')
cmd_skin = _forward_command("cmd_skin", "hermes_cli.skin_cmd", "skin_command", doc='Skin management (list / use / set).')
cmd_import = _forward_command("cmd_import", "hermes_cli.backup", "run_import", doc='Restore a Hermes backup from a zip file.')
cmd_dashboard_register = _forward_command("cmd_dashboard_register", "hermes_cli.dashboard_register", "cmd_dashboard_register", doc='Register a self-hosted dashboard OAuth client with Nous Portal.')
cmd_gateway_enroll = _forward_command("cmd_gateway_enroll", "hermes_cli.gateway_enroll", "cmd_gateway_enroll", doc='Enroll a self-hosted gateway with a relay connector.')
cmd_prompt_size = _forward_command("cmd_prompt_size", "hermes_cli.prompt_size", "cmd_prompt_size", doc='Show a byte/char breakdown of the system prompt + tool schemas.')
cmd_pairing = _forward_command("cmd_pairing", "hermes_cli.pairing", "pairing_command")
cmd_plugins = _forward_command("cmd_plugins", "hermes_cli.plugins_cmd", "plugins_command")
cmd_mcp = _forward_command("cmd_mcp", "hermes_cli.mcp_config", "mcp_command")
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
    from hermes_cli.setup import run_setup_action_with_navigation

    run_setup_action_with_navigation(
        "Model & Provider",
        lambda: select_provider_and_model(args=args),
        cancelled_message="No change.",
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
            if effective_provider == "auto":
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

    # Provider-specific setup + model selection. Flows resolve the
    # _model_flow_* names at call time so test monkeypatches on
    # hermes_cli.main keep intercepting.
    flow = _PROVIDER_MODEL_FLOWS.get(selected_provider)
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

    # Post-switch cleanup: switching to a named provider (anything except
    # "custom") leaves a stale OPENAI_BASE_URL in ~/.hermes/.env that poisons
    # auxiliary clients using provider:auto — clear it proactively. (#5161)
    if selected_provider not in {
        "custom",
        "cancel",
        "remove-custom",
    } and not selected_provider.startswith("custom:"):
        _clear_stale_openai_base_url()


# Frozen updater surface (PEP 562 ``__getattr__`` below): the frozen
# ``hermes_cli/update_cmd*.py`` files resolve these names via ``_m().<name>``
# on hermes_cli.main; importing update_cmd eagerly would cost every ``hermes``
# invocation ~50-100ms, so they resolve on first read. Nothing else may be
# added here — internal import paths are not a stable API.
_FROZEN_UPDATER_SURFACE: dict[str, tuple[str, ...]] = {
    "hermes_cli.update_cmd": (
        "_abort_dependency_sync_if_self_locked", "_assess_parked_branch_switch",
        "_capture_active_lazy_features", "_capture_active_tool_dependencies",
        "_cold_start_windows_gateway_after_update", "_defer_update_for_self_lock",
        "_dependency_sync_would_rewrite", "_detect_self_loaded_native_modules",
        "_detect_venv_python_processes", "_discard_stashed_changes",
        "_filter_non_gateway_concurrent_instances", "_fleet_probe_expected_runtimes",
        "_get_origin_url", "_handoff_reapable_backend_pids", "_ledger_manual_serve_holders",
        "_ledger_reapable_backend_pids", "_leftover_pausable_gateway_pids", "_npm_lockfile_changed",
        "_orphaned_desktop_backend_pids", "_park_stashed_changes",
        "_pause_windows_gateways_for_update", "_print_parked_branch_kept_notice",
        "_print_parked_branch_skip_warning", "_purge_stale_hermes_modules",
        "_refresh_active_lazy_features", "_refresh_active_memory_provider_dependencies",
        "_refresh_bootstrap_cache_scripts", "_refresh_windows_gateway_launchers",
        "_relaunch_stopped_serves", "_reload_updated_runtime_modules",
        "_restore_active_tool_dependencies", "_restore_stashed_changes",
        "_resume_windows_gateways_after_update", "_run_logged_subprocess", "_run_pre_update_backup",
        "_stash_local_changes_if_needed", "_stop_process_trees", "_sync_with_upstream_if_needed",
        "_upgrade_pip_before_lazy_refresh", "_venv_launcher_ancestors",
        "_wait_for_windows_update_gateway_exit", "_warn_orphaned_update_autostashes",
        "_write_update_incomplete_marker",
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

    (backup.run_quick_backup if getattr(args, "quick", False) else backup.run_backup)(args)


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

    if getattr(args, "gui", False):
        if not getattr(args, "yes", False):
            _require_tty("uninstall --gui")
        from hermes_cli.uninstall import run_gui_uninstall

        run_gui_uninstall(args)
        return

    if not getattr(args, "yes", False):
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
        )
        return True
    return False


def cmd_update(args):
    """Update Hermes Agent: hangup protection + update lock around ``_cmd_update_impl``."""
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

    # Exit code for the Windows hand-off child's hard exit (see finally); None
    # = not SystemExit-shaped, so real exceptions keep their traceback.
    _update_handoff_exit_code: int | None = None
    from hermes_cli.update_cmd import _cmd_update_impl

    try:
        _cmd_update_impl(args, gateway_mode=gateway_mode)
    except SystemExit as _update_exit:
        # Receipt boundary: the impl has many early sys.exit paths that never
        # reach an inner finalize. Persist any still-open receipt with the real
        # exit code (no-op if already finalized), then let the exit proceed.
        _code = _update_exit.code if isinstance(_update_exit.code, int) else 1
        _finalize_update_receipt(_code, f"sys.exit({_code})")
        _update_handoff_exit_code = (
            _update_exit.code if isinstance(_update_exit.code, int) else 0
        )
        raise
    except BaseException as _update_exc:
        _finalize_update_receipt(1, f"{type(_update_exc).__name__}: {_update_exc}")
        raise
    else:
        from hermes_cli.update_receipt import COMMAND_BOUNDARY_STOP_REASON

        _finalize_update_receipt(0, COMMAND_BOUNDARY_STOP_REASON)
        _update_handoff_exit_code = 0
    finally:
        _update_lock.release()
        _finalize_update_output(_update_io_state)
        # Windows hand-off child: a leftover non-daemon thread from the update
        # tail would freeze the PowerShell window for minutes after the receipt
        # is durable. Every durable step is done by now, so on the hand-off
        # path only (marker env set solely by
        # _reexec_dependency_sync_off_windows_shim) flush and exit hard.
        # By this point every durable step is done (receipt finalized above, lock released, stdio restored),
        # so on the hand-off path only, flush and exit hard instead of waiting for the interpreter to unwind
        # — the same treatment #79040's cron workaround applies.
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

    ``hermes -c Pokemon Agent Dev`` → ``['-c', 'Pokemon Agent Dev']``; tokens
    are collected until the next flag (``-*``) or known top-level subcommand.
    """
    _SUBCOMMANDS = {
        "chat", "model", "gateway", "setup", "whatsapp", "whatsapp-cloud", "login", "logout",
        "auth", "status", "cron", "doctor", "config", "pairing", "skills", "tools", "mcp",
        "sessions", "insights", "update", "uninstall", "profile", "dashboard", "serve",
        "desktop", "gui", "honcho", "claw", "plugins", "security", "acp", "webhook", "peer",
        "memory", "dump", "debug", "backup", "import", "completion", "logs",
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
        if not _find_stale_dashboard_pids():
            print("No hermes dashboard processes running.")
            sys.exit(0)
        # Reuse the same SIGTERM-grace-SIGKILL path used after `hermes update`;
        # it prints outcomes itself. Exit 1 only if every pid was unkillable.
        from hermes_cli.dashboard_procs import _kill_stale_dashboard_processes

        _kill_stale_dashboard_processes(reason="requested via --stop")
        sys.exit(1 if _find_stale_dashboard_pids() else 0)


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
    WEB_DIST contamination is stripped — caller-managed overrides (dev /
    custom builds) must still work, and the desktop-spawned backend itself
    (HERMES_DESKTOP=1) keeps its dist. Headless `serve` re-sets
    HERMES_SERVE_HEADLESS itself.
    """
    if os.environ.get("HERMES_DESKTOP") != "1":
        if _is_electron_packaged_web_dist(os.environ.get("HERMES_WEB_DIST", "")):
            os.environ.pop("HERMES_WEB_DIST", None)
    if not headless_backend:
        os.environ.pop("HERMES_SERVE_HEADLESS", None)


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
    mcp_discovery_after_bind = headless_backend and os.environ.get("HERMES_DESKTOP") == "1"
    if not mcp_discovery_after_bind:
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
    return mcp_discovery_after_bind


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

    # The in-browser Chat tab (embedded TUI over PTY/WebSocket) is always
    # available — desktop and dashboard both rely on `/api/ws` + `/api/pty`.
    start_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        allow_public=getattr(args, "insecure", False),
        initial_profile=getattr(args, "open_profile", "") or "",
        headless=_headless_backend,
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
        # Plugin commands missing from top-level --help is an accepted trade-off.
        "help",
    }
)


def _first_positional_argv() -> str | None:
    """First non-flag, non-flag-value token in ``sys.argv[1:]`` (skips values of known flags).

    Not a full argparse simulation: an unknown ``--foo bar`` may classify
    ``bar`` as positional, which at worst forces a one-time plugin discovery.
    """
    from hermes_cli._parser import top_level_value_flag_sets

    required_value_flags, optional_value_flags = top_level_value_flag_sets()
    value_flags = required_value_flags | optional_value_flags
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":  # everything after is positional
            return argv[i + 1] if i + 1 < len(argv) else None
        if not tok.startswith("-"):
            return tok
        # ``--flag=value`` is a single token; a known value flag consumes the next.
        i += 2 if ("=" not in tok and tok in value_flags and i + 1 < len(argv)) else 1
    return None


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

    if _startup_fast.is_termux_fast_version_argv(argv):
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
    build_monitoring_parser(subparsers, cmd_monitoring=cmd_monitoring)
    build_claw_parser(subparsers, cmd_claw=cmd_claw)
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
    _advertise_agent_env()

    # Force UTF-8 stdio on Windows before anything prints.  No-op elsewhere.
    try:
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()
    except Exception:
        pass

    # Sweep stale ``hermes.exe.old.*`` quarantine files from previous Windows
    # updates (see ``_quarantine_running_hermes_exe``). No-op elsewhere.
    try:
        _cleanup_quarantined_exes()
    except Exception:
        pass

    # Checkout changed since last launch → sweep stale __pycache__ once so no
    # process resolves fresh source against old bytecode. Never raises.
    _sweep_stale_bytecode_if_checkout_changed()

    # Self-heal a venv left half-built by an interrupted ``hermes update``, and
    # hint (never restart) about a fleet the interrupted update never
    # restarted. Both skipped while the user is *running* update — that flow
    # owns its marker and a recovery install must not race the real one. The
    # substring match is deliberately loose: over-matching (``hermes skills
    # install update``) only defers recovery one launch; under-matching
    # (``hermes -p work update``) would race. Never raises.
    # See #95294.
    if "update" not in sys.argv[1:]:
        try:
            _recover_from_interrupted_install()
        except Exception:
            pass
        try:
            from hermes_cli.update_cmd_fleet import _warn_pending_fleet_restart_on_startup

            _warn_pending_fleet_restart_on_startup()
        except Exception:
            pass

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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import hashlib  # noqa: F401,E402
import shlex  # noqa: F401,E402
import stat  # noqa: F401,E402
import tempfile  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'line_input': ('hermes_cli.cli_output', 'line_input'),
}

_plugin_compat_prev_getattr = __getattr__


def __getattr__(name):  # PEP 562 — chained onto the module's own __getattr__
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        return _plugin_compat_prev_getattr(name)
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
