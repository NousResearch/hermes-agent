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
    if profile_name is None and hermes_home_env:
        if Path(hermes_home_env).parent.name == "profiles":
            return
        if os.environ.get("HERMES_TUI_DASHBOARD") == "1":
            return

    if profile_name is None and not _under_gateway_supervisor(argv):
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
cmd_cron = _forward_command("cmd_cron", "hermes_cli.cron", "cron_command", doc='Cron job management.')
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
        _finalize_update_receipt(0, "completed at command boundary")
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
    _run_and_exit_oneshot(
        args.oneshot,
        model=getattr(args, "model", None),
        provider=getattr(args, "provider", None),
        toolsets=getattr(args, "toolsets", None),
        skills=getattr(args, "skills", None),
        usage_file=getattr(args, "usage_file", None),
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
