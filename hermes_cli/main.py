#!/usr/bin/env python3
"""
Hermes CLI - Main entry point.

Usage:
    hermes                     # Interactive chat (default)
    hermes chat                # Interactive chat
    hermes gateway             # Run gateway in foreground
    hermes gateway start       # Start gateway as service
    hermes gateway stop        # Stop gateway service
    hermes gateway status      # Show gateway status
    hermes gateway install     # Install gateway service
    hermes gateway uninstall   # Uninstall gateway service
    hermes setup               # Interactive setup wizard
    hermes logout              # Clear stored authentication
    hermes status              # Show status of all components
    hermes cron                # Manage cron jobs
    hermes cron list           # List cron jobs
    hermes cron status         # Check if cron scheduler is running
    hermes doctor              # Check configuration and dependencies
    hermes honcho setup                    # Configure Honcho AI memory integration
    hermes honcho status                   # Show Honcho config and connection status
    hermes honcho sessions                 # List directory → session name mappings
    hermes honcho map <name>               # Map current directory to a session name
    hermes honcho peer                     # Show peer names and dialectic settings
    hermes honcho peer --user NAME         # Set user peer name
    hermes honcho peer --ai NAME           # Set AI peer name
    hermes honcho peer --reasoning LEVEL   # Set dialectic reasoning level
    hermes honcho mode                     # Show current memory mode
    hermes honcho mode [hybrid|honcho|local]  # Set memory mode
    hermes honcho tokens                   # Show token budget settings
    hermes honcho tokens --context N       # Set session.context() token cap
    hermes honcho tokens --dialectic N     # Set dialectic result char cap
    hermes honcho identity                 # Show AI peer identity representation
    hermes honcho identity <file>          # Seed AI peer identity from a file (SOUL.md etc.)
    hermes honcho migrate                  # Step-by-step migration guide: OpenClaw native → Hermes + Honcho
    hermes version             Show version
    hermes update              Update to latest version
    hermes uninstall           Uninstall Hermes Agent
    hermes acp                 Run as an ACP server for editor integration
    hermes sessions browse     Interactive session picker with search

    hermes claw migrate --dry-run  # Preview migration without changes
"""

# IMPORTANT: hermes_bootstrap must be the very first import — it sets up
# UTF-8 stdio on Windows so print()/subprocess children don't hit
# UnicodeEncodeError with non-ASCII characters.  No-op on POSIX.
#
# Guarded against ModuleNotFoundError because ``hermes_bootstrap`` is a
# top-level module registered via pyproject.toml's ``py-modules`` list.
# When the user upgrades code via ``git pull`` (or ``hermes update``
# crashes between ``git reset --hard`` and ``uv pip install -e .``), the
# new code references ``hermes_bootstrap`` but the editable install's
# ``.pth`` file still points at the old set of top-level modules.  Without
# this guard, hermes crashes on import and the user can't run
# ``hermes update`` to recover.  Missing the bootstrap means UTF-8 stdio
# setup is skipped on Windows — degraded, not broken.  POSIX is unaffected.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

# Windows: neutralize CPython's ``platform._syscmd_ver`` before anything else
# imports — it shells out ``cmd /c ver`` (shell=True, no CREATE_NO_WINDOW), so
# any dependency touching ``platform.uname()`` at import time flashes a
# visible console when this process is windowless (pythonw gateway + every
# kanban worker).  No-op on POSIX; never raises.
from hermes_cli._subprocess_compat import suppress_platform_ver_console

suppress_platform_ver_console()

import os
import sys

# ── Startup fast-path bootstrap ─────────────────────────────────────────
# Two lines of inline path math so ``python hermes_cli/main.py`` (script
# mode — sys.path[0] is hermes_cli/, not the repo root) can import the
# canonical helpers; everything else lives in hermes_cli._startup_fast.
_bootstrap_root = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
if _bootstrap_root not in sys.path:
    sys.path.insert(0, _bootstrap_root)
from hermes_cli import _startup_fast  # noqa: E402

# Early venv self-heal — MUST run before any third-party import below.  When
# a prior ``hermes update`` left a recovery marker and a core package's import
# files were wiped (#57828 — failed lazy backend refresh), the module-level
# ``from hermes_cli.env_loader import ...`` / ``from hermes_cli.config import
# ...`` imports further down would crash before ``main()`` ever reaches
# ``_recover_from_interrupted_install()``.  ``_early_recovery`` is stdlib-only
# (safe to import on a corrupted venv), repairs just enough for this module to
# finish importing, and leaves the marker lifecycle to the full recovery path.
# The module import itself is unguarded on purpose: it lives in this same
# package directory, so if IT can't import, nothing else in hermes_cli can
# either. It is also the canonical home of the probe/repair tables reused by
# the full recovery path below.
from hermes_cli import _early_recovery as _early_recovery_mod

try:
    _early_recovery_mod.recover_if_needed()
except Exception:
    pass


def _exit_after_oneshot(rc: object) -> None:
    """Exit one-shot mode without letting late native finalizers change rc.

    The SIGABRT this guards against (#30387, #43055) fires in a
    native-extension finalizer during CPython's ``Py_FinalizeEx``, *after*
    the response has printed. Flush streams, shut down file logging, then
    ``os._exit`` past interpreter finalization. The ``atexit`` chain is
    deliberately skipped — several handlers re-enter native code that may
    be the abort source. Stateful cleanup is handled in ``_run_agent`` and
    ``_cleanup_oneshot_runtime``.
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
    if rc is None:
        exit_code = 0
    elif isinstance(rc, int):
        exit_code = rc
    else:
        exit_code = 1
    os._exit(exit_code)


_oneshot_cleanup_done = False


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
    try:
        from tools.terminal_tool import cleanup_all_environments
        cleanup_all_environments()
    except Exception:
        pass
    try:
        from tools.async_delegation import interrupt_all
        interrupt_all(reason="oneshot shutdown")
    except Exception:
        pass
    try:
        from tools.browser_tool import _emergency_cleanup_all_sessions
        _emergency_cleanup_all_sessions()
    except Exception:
        pass
    try:
        from tools.mcp_tool import shutdown_mcp_servers
        shutdown_mcp_servers()
    except BaseException:
        pass
    try:
        from agent.auxiliary_client import shutdown_cached_clients
        shutdown_cached_clients()
    except Exception:
        pass


def _run_and_exit_oneshot(
    prompt: str,
    *,
    model: object = None,
    provider: object = None,
    toolsets: object = None,
    usage_file: object = None,
) -> None:
    try:
        from hermes_cli.oneshot import run_oneshot

        rc = run_oneshot(
            prompt,
            model=model,
            provider=provider,
            toolsets=toolsets,
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
        # Defense-in-depth. ``run_oneshot`` already converts agent failures
        # into an int return code and only re-raises KeyboardInterrupt /
        # SystemExit (handled above). Anything still escaping here means
        # ``run_oneshot`` itself malfunctioned — surface it on stderr but never
        # fall through to normal interpreter teardown, which is the exact path
        # that aborts with SIGABRT on AL2023 (the bug this routine fixes).
        import traceback
        try:
            traceback.print_exc()
        except Exception:
            pass
        rc = 1
    try:
        _cleanup_oneshot_runtime()
    finally:
        # The hard exit is the safety boundary for #43055. Even an interrupt
        # during best-effort cleanup must not fall back into interpreter
        # finalization, where the reported native SIGABRT occurs.
        _exit_after_oneshot(rc)


def _project_root_str_fast() -> str:
    return _startup_fast.project_root_str()


def _ensure_project_root_on_path_fast() -> None:
    _startup_fast.ensure_project_root_on_path()


def _set_process_title() -> None:
    """Set the process title to 'hermes' so tools like 'ps', 'top', and
    'htop' show the app name instead of 'python3.xx'.

    Purely cosmetic — non-fatal on any platform.

    Strategy (try in order):
      1. ``setproctitle`` (opt-in dep — installed via ``hermes tools`` or
         ``pip install setproctitle``, or bundled in a future release).
      2. ctypes ``prctl(PR_SET_NAME)`` (Linux only, 15-char limit).
      3. ctypes ``pthread_setname_np`` (macOS only, kernel thread name —
         changes lldb/top but not ``ps aux``).
      4. No-op on Windows (the .exe name is already ``hermes.exe``).
    """
    # Strategy 1: setproctitle (best — works on macOS, Linux, BSD)
    try:
        import setproctitle  # type: ignore[import-untyped]

        setproctitle.setproctitle("hermes")
        return
    except ImportError:
        pass

    # Strategy 2/3: platform-specific ctypes fallback
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
        # Windows: the .exe name is already ``hermes.exe`` — nothing to do.
    except Exception:
        pass


# Cheap, dependency-free read of `display.interface` from config.yaml for the
# earliest hot-path decisions (mouse-residue suppression, Termux fast launch)
# that run *before* hermes_cli.config is importable. Mirrors the explicit
# precedence used everywhere else: `--cli` always wins, then `--tui`/env, then
# this config value. Cached so the multiple early callers don't re-parse YAML.
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

    Precedence: explicit ``--cli`` wins (forces classic REPL), then
    explicit ``--tui``/``HERMES_TUI=1``, then a real-TTY gate (a
    non-interactive stdio can't host the Ink UI, so ambient config never
    boots it there), then ``display.interface`` in config.

    The TTY gate is load-bearing for headless spawners — kanban workers,
    cron jobs, pipes run ``hermes … chat -q`` with stdio on a pipe. This
    is the earliest launch decision (it runs before ``cmd_chat`` /
    ``_resolve_use_tui``), so a ``display.interface: tui`` default used to
    boot the TUI here — whose no-TTY bail-out exits 0 without doing the
    task → "protocol violation" on every attempt. An explicit ``--tui``
    still reaches the informative bail-out.
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
# TUI hot path so the terminal stops emitting SGR/X10 mouse reports while the
# Python launcher is still doing imports (≈100–300ms in cooked + echo mode,
# before the Node TUI takes stdin into raw mode). During that window any
# incoming bytes are echoed straight back to the user's shell scrollback as
# ``^[[<…M`` text. The TUI itself runs `resetTerminalModes()` again in
# `entry.tsx`; this is just the earlier cousin. ``HERMES_TUI_NO_EARLY_DISABLE``
# escapes the behaviour for diagnostics.
def _suppress_mouse_residue_early() -> None:
    if os.environ.get("HERMES_TUI_NO_EARLY_DISABLE") == "1":
        return
    if not _wants_tui_early():
        return
    try:
        # Skip when stdout is redirected (`hermes --tui … >log`, CI capture):
        # the bytes can't reach the terminal anyway and would just pollute
        # the log with raw CSI.
        if not os.isatty(1):
            return
        # Disable every mouse-tracking variant we know about. Idempotent and
        # safe to send even when no tracking is currently asserted.
        os.write(
            1,
            b"\x1b[?1003l\x1b[?1002l\x1b[?1001l\x1b[?1000l\x1b[?9l"
            b"\x1b[?1006l\x1b[?1005l\x1b[?1015l\x1b[?1016l\x1b[?2029l",
        )
    except OSError:
        pass


_suppress_mouse_residue_early()


def _is_termux_startup_environment_fast() -> bool:
    """Tiny Termux check for pre-import startup shortcuts."""
    return _startup_fast.is_termux_env()


def _is_termux_fast_version_argv(argv: list[str]) -> bool:
    return _startup_fast.is_termux_fast_version_argv(argv)


def _is_global_fast_version_argv(argv: list[str]) -> bool:
    return _startup_fast.is_global_fast_version_argv(argv)


def _is_container_startup_environment_fast() -> bool:
    return _startup_fast.is_container_startup_environment()


def _active_profile_may_override_home_fast(hermes_root: str) -> bool:
    return _startup_fast.active_profile_may_override_home(hermes_root)


def _container_mode_may_be_active_fast() -> bool:
    return _startup_fast.container_mode_may_be_active()


def _read_openai_version_fast() -> str | None:
    """Read OpenAI SDK version without importing ``importlib.metadata``."""
    return _startup_fast.read_openai_version()


def _print_fast_version_info() -> None:
    _startup_fast.print_fast_version_info()


def _try_ultrafast_version() -> bool:
    """Handle ``hermes --version`` before config/logging imports."""
    return _startup_fast.try_fast_version()


def _try_termux_ultrafast_version() -> bool:
    """Backward-compatible test hook for the Termux startup fast path."""
    if not _is_termux_startup_environment_fast():
        return False
    return _try_ultrafast_version()


_ensure_project_root_on_path_fast()

if _try_ultrafast_version():
    raise SystemExit(0)

import argparse
import hashlib
import json
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Optional


import functools as _functools

from hermes_cli.subcommands._shared import add_accept_hooks_flag as _add_accept_hooks_flag
from hermes_cli.subcommands.cron import build_cron_parser
from hermes_cli.subcommands.sync import build_sync_parser
from hermes_cli.subcommands.gateway import build_gateway_parser
from hermes_cli.subcommands.profile import build_profile_parser
from hermes_cli.subcommands.model import build_model_parser
from hermes_cli.subcommands.setup import build_setup_parser

from hermes_cli.subcommands.whatsapp import build_whatsapp_parser
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
from hermes_cli.subcommands.version import build_version_parser
from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.subcommands.uninstall import build_uninstall_parser
from hermes_cli.subcommands.dashboard import build_dashboard_parser
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


def _require_tty(command_name: str) -> None:
    """Exit with a clear error if stdin is not a terminal.

    Interactive TUI commands (hermes tools, hermes setup, hermes model) use
    curses or input() prompts that spin at 100% CPU when stdin is a pipe.
    This guard prevents accidental non-interactive invocation.
    """
    if not sys.stdin.isatty():
        print(
            f"Error: 'hermes {command_name}' requires an interactive terminal.\n"
            f"It cannot be run through a pipe or non-interactive subprocess.\n"
            f"Run it directly in your terminal instead.",
            file=sys.stderr,
        )
        sys.exit(1)


# Add project root to path
PROJECT_ROOT = Path(_project_root_str_fast())
_ensure_project_root_on_path_fast()


# ---------------------------------------------------------------------------
# Profile override — MUST happen before any hermes module import.
#
# Many modules cache HERMES_HOME at import time (module-level constants).
# We intercept --profile/-p from sys.argv here and set the env var so that
# every subsequent ``os.getenv("HERMES_HOME", ...)`` resolves correctly.
# The flag is stripped from sys.argv so argparse never sees it.
# Falls back to ~/.hermes/active_profile for sticky default.
# ---------------------------------------------------------------------------
def _apply_profile_override() -> None:
    """Pre-parse --profile/-p and set HERMES_HOME before imports."""
    argv = sys.argv[1:]
    profile_name = None
    consume = 0
    profile_index = None

    def _inside_mcp_add_args(index: int) -> bool:
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

    def _resolve_sudo_user_profile_env(name: str) -> str | None:
        """Resolve `sudo hermes -p <name>` against the invoking user's home.

        `_apply_profile_override()` runs before argparse, so `--run-as-user`
        is not available yet. For sudo invocations, the best available signal
        is SUDO_USER: root is only doing the privileged install/start action,
        while the profile store normally belongs to the user who invoked sudo.
        """
        if name == "default":
            return None
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            return None
        sudo_user = os.environ.get("SUDO_USER", "").strip()
        if not sudo_user or sudo_user == "root":
            return None

        try:
            import pwd

            home = Path(pwd.getpwnam(sudo_user).pw_dir)
        except Exception:
            return None

        candidate = home / ".hermes" / "profiles" / name
        try:
            if candidate.is_dir():
                return str(candidate)
        except OSError:
            return None
        return None

    # 1. Check for explicit -p / --profile flag. Historically this worked even
    # after the subcommand (`hermes chat -p coder`), so keep scanning broadly.
    # The exception is command-argv passthrough regions such as `mcp add --args`.
    value_flags = {
        "-z", "--oneshot",
        "-m", "--model",
        "--provider",
        "-t", "--toolsets",
        "-r", "--resume",
        "-s", "--skills",
        "--usage-file",
        "--in",
    }
    optional_value_flags = {"-c", "--continue"}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            break
        if arg == "--args" and _inside_mcp_add_args(i):
            break
        if arg in {"--profile", "-p"} and i + 1 < len(argv):
            profile_name = argv[i + 1]
            consume = 2
            profile_index = i
            break
        if arg.startswith("--profile="):
            profile_name = arg.split("=", 1)[1]
            consume = 1
            profile_index = i
            break
        if "=" not in arg and arg in value_flags and i + 1 < len(argv):
            i += 2
        elif (
            "=" not in arg
            and arg in optional_value_flags
            and i + 1 < len(argv)
            and not argv[i + 1].startswith("-")
        ):
            i += 2
        else:
            i += 1

    # 1b. Reject values that can't be valid profile names (e.g. pytest's
    # "-p no:xdist" would be misread as profile "no:xdist" otherwise).
    # Mirrors hermes_cli.profiles._PROFILE_ID_RE so we never call
    # resolve_profile_env() with a value it must reject + sys.exit on.
    if profile_name is not None and consume == 2:
        import re as _re

        if not _re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", profile_name):
            profile_name = None
            consume = 0
            profile_index = None

    # 1.5 If HERMES_HOME is already set and no explicit flag was given, trust it
    # only when it already points to a specific profile directory.  The
    # distinguishing heuristic: a profile path has "profiles" as its immediate
    # parent directory name (e.g. ~/.hermes/profiles/coder or
    # /opt/data/profiles/coder).  If HERMES_HOME points to the hermes root
    # instead (e.g. systemd hardcodes HERMES_HOME=/root/.hermes), we must
    # still read active_profile — the user may have switched profiles via
    # `hermes profile use` and the gateway should honour that choice.
    # See issue #22502.
    hermes_home_env = os.environ.get("HERMES_HOME", "")
    if profile_name is None and hermes_home_env:
        if Path(hermes_home_env).parent.name == "profiles":
            return

    # 2. If no flag, check active_profile in the hermes root.
    #
    # EXCEPTION: a supervised s6 gateway child (exported by the container
    # run-script as HERMES_S6_SUPERVISED_CHILD=1) must NOT follow the sticky
    # active_profile. Each supervised slot has a fixed profile identity: named
    # slots pass ``-p <name>`` explicitly (handled in step 1 above), and the
    # reserved ``gateway-default`` slot runs bare ``hermes gateway run`` to mean
    # "the root HERMES_HOME profile". If the reserved default child read
    # active_profile here, switching the active profile (e.g. via the dashboard)
    # would silently redirect the default gateway into that profile — yielding a
    # duplicate gateway for the active profile and no real default gateway. See
    # the "Docker & Profiles & Dashboard" report.
    if profile_name is None and not os.environ.get("HERMES_S6_SUPERVISED_CHILD"):
        try:
            from hermes_constants import get_default_hermes_root

            active_path = get_default_hermes_root() / "active_profile"
            if active_path.exists():
                name = active_path.read_text(encoding="utf-8").strip()
                if name and name != "default":
                    profile_name = name
                    consume = 0  # don't strip anything from argv
        except (UnicodeDecodeError, OSError):
            pass  # corrupted file, skip

    # 3. If we found a profile, resolve and set HERMES_HOME
    if profile_name is not None:
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
            print(
                f"Warning: profile override failed ({exc}), using default",
                file=sys.stderr,
            )
            return
        os.environ["HERMES_HOME"] = hermes_home
        # Strip the flag from argv so argparse doesn't choke
        if consume > 0 and profile_index is not None:
            start = profile_index + 1  # +1 because argv is sys.argv[1:]
            sys.argv = sys.argv[:start] + sys.argv[start + consume :]


_apply_profile_override()

# Load .env from ~/.hermes/.env first, then project root as dev fallback.
# User-managed env files should override stale shell exports on restart.
from hermes_cli.config import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv

load_hermes_dotenv(project_env=PROJECT_ROOT / ".env")

# Bridge security.redact_secrets from config.yaml → HERMES_REDACT_SECRETS env
# var BEFORE hermes_logging imports agent.redact (which snapshots the flag at
# module-import time). Without this, config.yaml's toggle is ignored because
# the setup_logging() call below imports agent.redact, which reads the env var
# exactly once. Env var in .env still wins — this is config.yaml fallback only.
#
# We also read network.force_ipv4 from the same yaml load to avoid two
# separate config.yaml reads (saves ~17ms on every CLI startup — the second
# `load_config()` was doing a full deep-merge for one boolean lookup).
_FORCE_IPV4_EARLY = False
try:
    # Reuse read_raw_config()'s (mtime, size)-keyed cache instead of a bespoke
    # yaml.load — the SAME parse then serves hermes_logging's
    # _read_logging_config and any later raw reads in this process, collapsing
    # 3-4 config.yaml parses per invocation into one.
    from hermes_cli.config import read_raw_config as _read_raw_early

    _cfg_path = get_hermes_home() / "config.yaml"
    if _cfg_path.exists():
        _early_cfg_raw = _read_raw_early() or {}
        # Managed scope: overlay administrator-pinned values so a managed
        # security.redact_secrets / network.force_ipv4 wins here too. This early
        # bridge reads config.yaml directly (before load_config is usable), so
        # without the overlay a managed redact_secrets toggle would be ignored.
        # Fail-open via the shared helper.
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

# Initialize centralized file logging early — all `hermes` subcommands
# (chat, setup, gateway, config, etc.) write to agent.log + errors.log.
# Dashboard entrypoints bootstrap with GUI mode so gui.log is always present
# during GUI testing, including pre-dispatch startup failures.
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

# Apply IPv4 preference early, before any HTTP clients are created.
# We already determined whether to force IPv4 from the raw yaml read above —
# this just calls the toggle without a redundant load_config() round trip.
if _FORCE_IPV4_EARLY:
    try:
        from hermes_constants import apply_ipv4_preference as _apply_ipv4

        _apply_ipv4(force=True)
    except Exception:
        pass  # best-effort — don't crash if hermes_constants not importable yet

import logging
import threading
import time as _time
from datetime import datetime

from hermes_cli import __version__, __release_date__

# Provider model-selection wizard flows extracted to hermes_cli/model_setup_flows.py
# (god-file decomposition Phase 2). Re-imported here so select_provider_and_model and
# existing test monkeypatches (hermes_cli.main._model_flow_*) keep resolving unchanged.
from hermes_cli.model_setup_flows import (
    _prompt_auth_credentials_choice,
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
    _model_flow_bedrock_api_key,
    _model_flow_bedrock,
    _model_flow_vertex,
    _model_flow_api_key_provider,
    _model_flow_anthropic,
    _model_flow_moa,
    _model_flow_ai_gateway,
)
logger = logging.getLogger(__name__)


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
        head_file = git_dir / "HEAD"
        head = head_file.read_text(encoding="utf-8", errors="replace").strip()
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


def _relative_time(ts) -> str:
    """Format a timestamp as relative time (e.g., '2h ago', 'yesterday').

    Thin wrapper kept for backward compatibility; the implementation lives
    in :mod:`hermes_cli.timefmt` so lightweight consumers don't have to
    import the whole CLI surface.
    """
    from hermes_cli.timefmt import relative_time

    return relative_time(ts)


def _has_any_provider_configured() -> bool:
    """Check if at least one inference provider is usable."""
    from hermes_cli.config import get_env_path, get_hermes_home, load_config
    from hermes_cli.auth import get_auth_status

    # Determine whether Hermes itself has been explicitly configured (model
    # in config that isn't the hardcoded default). Used below to gate external
    # tool credentials (Claude Code, Codex CLI) that shouldn't silently skip
    # the setup wizard on a fresh install.
    from hermes_cli.config import DEFAULT_CONFIG

    _DEFAULT_MODEL = DEFAULT_CONFIG.get("model", "")
    cfg = load_config()
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        _default = model_cfg.get("default")
        if isinstance(_default, dict):
            from hermes_cli.config import split_model_config_default
            _model_name, _ = split_model_config_default(_default)
        else:
            _model_name = (_default or "")
        _model_name = (str(_model_name) if not isinstance(_model_name, str) else _model_name).strip()
    elif isinstance(model_cfg, str):
        _model_name = model_cfg.strip()
    else:
        _model_name = ""
    _has_hermes_config = _model_name and _model_name != _DEFAULT_MODEL

    # Check env vars (may be set by .env or shell).
    # OPENAI_BASE_URL alone counts — local models (vLLM, llama.cpp, etc.)
    # often don't require an API key.
    from hermes_cli.auth import PROVIDER_REGISTRY

    # Collect all provider env vars
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
    if any(os.getenv(v) for v in provider_env_vars):
        return True

    # Check .env file for keys
    env_file = get_env_path()
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:]
                key, _, val = line.partition("=")
                val = val.strip().strip("'\"")
                if key.strip() in provider_env_vars and val:
                    return True
        except Exception:
            pass

    # Cheap local checks first: auth.json and config.yaml are on-disk lookups,
    # while the PROVIDER_REGISTRY sweep below spawns subprocesses (gh) and can
    # take 15-20s — long enough that desktop setup.status calls time out.

    # Check for Nous Portal OAuth credentials
    auth_file = get_hermes_home() / "auth.json"
    if auth_file.exists():
        try:
            import json

            auth = json.loads(auth_file.read_text(encoding="utf-8-sig"))
            active = auth.get("active_provider")
            if active:
                status = get_auth_status(active)
                if status.get("logged_in"):
                    return True
        except Exception:
            pass

    # Check config.yaml — if model is a dict with an explicit provider set,
    # the user has gone through setup (fresh installs have model as a plain
    # string).  Also covers custom endpoints that store api_key/base_url in
    # config rather than .env.
    if isinstance(model_cfg, dict):
        cfg_provider = (model_cfg.get("provider") or "").strip()
        cfg_base_url = (model_cfg.get("base_url") or "").strip()
        cfg_api_key = (model_cfg.get("api_key") or "").strip()
        if cfg_provider or cfg_base_url or cfg_api_key:
            return True

    # Check provider-specific auth fallbacks (for example, Copilot via gh auth).
    try:
        for provider_id, pconfig in PROVIDER_REGISTRY.items():
            if pconfig.auth_type != "api_key":
                continue
            status = get_auth_status(provider_id)
            if status.get("logged_in"):
                return True
    except Exception:
        pass

    # Check for Claude Code OAuth credentials (~/.claude/.credentials.json)
    # Only count these if Hermes has been explicitly configured — Claude Code
    # being installed doesn't mean the user wants Hermes to use their tokens.
    if _has_hermes_config:
        try:
            from agent.anthropic_adapter import (
                read_claude_code_credentials,
                is_claude_code_token_valid,
            )

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
        from hermes_cli.model_selection_guards import combined_selection_warning
    except Exception as exc:
        logger.warning("startup model cost guard unavailable: %s", exc)
        return

    try:
        model_cfg = (load_config().get("model") or {})
    except Exception as exc:
        logger.warning("startup model cost guard could not load config: %s", exc)
        model_cfg = {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    model = explicit_model or (model_cfg.get("default") or "").strip()
    if not model:
        return
    provider = (explicit_provider or model_cfg.get("provider") or "").strip()
    try:
        # Unified registry: cost guard + id-keyed guards (e.g. the
        # data-training-tier warning) all fire at startup too.
        warning = combined_selection_warning(
            model,
            provider=provider,
            base_url=(model_cfg.get("base_url") or ""),
            api_key=(model_cfg.get("api_key") or ""),
        )
    except Exception as exc:
        logger.warning("startup model cost guard failed for %s/%s: %s", provider, model, exc)
        return
    if warning is None:
        return

    # Cost and provider-routing confirmation is intentionally independent of
    # --yolo / --accept-hooks: those flags approve local command/tool risk, not
    # paid aggregator spend or a surprising provider route.
    message = warning.message
    if not sys.stdin.isatty():
        sys.stderr.write(message + "\n")
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


def _session_browse_picker(sessions: list) -> Optional[str]:
    """Interactive curses-based session browser with live search filtering.

    Returns the selected session ID, or None if cancelled.
    """
    if not sessions:
        print("No sessions found.")
        return None

    # Try curses-based picker first
    try:
        import curses

        result_holder = [None]

        def _format_row(s, max_x):
            """Format a session row for display."""
            title = (s.get("title") or "").strip()
            preview = (s.get("preview") or "").strip()
            source = s.get("source", "")[:6]
            last_active = _relative_time(s.get("last_active"))
            sid = s["id"][:18]

            # Adaptive column widths based on terminal width
            # Layout: [arrow 3] [title/preview flexible] [active 12] [src 6] [id 18]
            fixed_cols = 3 + 12 + 6 + 18 + 6  # arrow + active + src + id + padding
            name_width = max(20, max_x - fixed_cols)

            if title:
                name = title[:name_width]
            elif preview:
                name = preview[:name_width]
            else:
                name = sid

            return f"{name:<{name_width}}  {last_active:<10}  {source:<5} {sid}"

        def _match(s, query):
            """Check if a session matches the search query (case-insensitive)."""
            q = query.lower()
            return (
                q in (s.get("title") or "").lower()
                or q in (s.get("preview") or "").lower()
                or q in s.get("id", "").lower()
                or q in (s.get("source") or "").lower()
            )

        def _curses_browse(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)  # selected
                curses.init_pair(2, curses.COLOR_YELLOW, -1)  # header
                curses.init_pair(3, curses.COLOR_CYAN, -1)  # search
                curses.init_pair(4, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1)  # dim

            cursor = 0
            scroll_offset = 0
            search_text = ""
            filtered = list(sessions)

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                if max_y < 5 or max_x < 40:
                    # Terminal too small
                    try:
                        stdscr.addstr(0, 0, "Terminal too small")
                    except curses.error:
                        pass
                    stdscr.refresh()
                    stdscr.getch()
                    return

                # Header line
                if search_text:
                    header = f"  Browse sessions — filter: {search_text}█"
                    header_attr = curses.A_BOLD
                    if curses.has_colors():
                        header_attr |= curses.color_pair(3)
                else:
                    header = "  Browse sessions — ↑↓ navigate  Enter select  Type to filter  Esc quit"
                    header_attr = curses.A_BOLD
                    if curses.has_colors():
                        header_attr |= curses.color_pair(2)
                try:
                    stdscr.addnstr(0, 0, header, max_x - 1, header_attr)
                except curses.error:
                    pass

                # Column header line
                fixed_cols = 3 + 12 + 6 + 18 + 6
                name_width = max(20, max_x - fixed_cols)
                col_header = f"   {'Title / Preview':<{name_width}}  {'Active':<10}  {'Src':<5} {'ID'}"
                try:
                    dim_attr = (
                        curses.color_pair(4) if curses.has_colors() else curses.A_DIM
                    )
                    stdscr.addnstr(1, 0, col_header, max_x - 1, dim_attr)
                except curses.error:
                    pass

                # Compute visible area
                visible_rows = max_y - 4  # header + col header + blank + footer
                visible_rows = max(visible_rows, 1)

                # Clamp cursor and scroll
                if not filtered:
                    try:
                        msg = "  No sessions match the filter."
                        stdscr.addnstr(3, 0, msg, max_x - 1, curses.A_DIM)
                    except curses.error:
                        pass
                else:
                    if cursor >= len(filtered):
                        cursor = len(filtered) - 1
                    cursor = max(cursor, 0)
                    if cursor < scroll_offset:
                        scroll_offset = cursor
                    elif cursor >= scroll_offset + visible_rows:
                        scroll_offset = cursor - visible_rows + 1

                    for draw_i, i in enumerate(
                        range(
                            scroll_offset,
                            min(len(filtered), scroll_offset + visible_rows),
                        )
                    ):
                        y = draw_i + 3
                        if y >= max_y - 1:
                            break
                        s = filtered[i]
                        arrow = " → " if i == cursor else "   "
                        row = arrow + _format_row(s, max_x - 3)
                        attr = curses.A_NORMAL
                        if i == cursor:
                            attr = curses.A_BOLD
                            if curses.has_colors():
                                attr |= curses.color_pair(1)
                        try:
                            stdscr.addnstr(y, 0, row, max_x - 1, attr)
                        except curses.error:
                            pass

                # Footer
                footer_y = max_y - 1
                if filtered:
                    footer = f"  {cursor + 1}/{len(filtered)} sessions"
                    if len(filtered) < len(sessions):
                        footer += f" (filtered from {len(sessions)})"
                else:
                    footer = f"  0/{len(sessions)} sessions"
                try:
                    stdscr.addnstr(
                        footer_y,
                        0,
                        footer,
                        max_x - 1,
                        curses.color_pair(4) if curses.has_colors() else curses.A_DIM,
                    )
                except curses.error:
                    pass

                stdscr.refresh()
                key = stdscr.getch()

                if key in {curses.KEY_UP,}:
                    if filtered:
                        cursor = (cursor - 1) % len(filtered)
                elif key in {curses.KEY_DOWN,}:
                    if filtered:
                        cursor = (cursor + 1) % len(filtered)
                elif key in {curses.KEY_ENTER, 10, 13}:
                    if filtered:
                        result_holder[0] = filtered[cursor]["id"]
                    return
                elif key == 27:  # Esc
                    if search_text:
                        # First Esc clears the search
                        search_text = ""
                        filtered = list(sessions)
                        cursor = 0
                        scroll_offset = 0
                    else:
                        # Second Esc exits
                        return
                elif key in {curses.KEY_BACKSPACE, 127, 8}:
                    if search_text:
                        search_text = search_text[:-1]
                        if search_text:
                            filtered = [s for s in sessions if _match(s, search_text)]
                        else:
                            filtered = list(sessions)
                        cursor = 0
                        scroll_offset = 0
                elif key == ord("q") and not search_text:
                    return
                elif 32 <= key <= 126:
                    # Printable character → add to search filter
                    search_text += chr(key)
                    filtered = [s for s in sessions if _match(s, search_text)]
                    cursor = 0
                    scroll_offset = 0

        curses.wrapper(_curses_browse)
        return result_holder[0]

    except Exception:
        pass

    # Fallback: numbered list (Windows without curses, etc.)
    print("\n  Browse sessions  (enter number to resume, q to cancel)\n")
    for i, s in enumerate(sessions):
        title = (s.get("title") or "").strip()
        preview = (s.get("preview") or "").strip()
        label = title or preview or s["id"]
        if len(label) > 50:
            label = label[:47] + "..."
        last_active = _relative_time(s.get("last_active"))
        src = s.get("source", "")[:6]
        print(f"  {i + 1:>3}. {label:<50}  {last_active:<10}  {src}")

    while True:
        try:
            val = input(f"\n  Select [1-{len(sessions)}]: ").strip()
            if not val or val.lower() in {"q", "quit", "exit"}:
                return None
            idx = int(val) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]["id"]
            print(f"  Invalid selection. Enter 1-{len(sessions)} or q to cancel.")
        except ValueError:
            print("  Invalid input. Enter a number or q to cancel.")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


def _resolve_workspace_key() -> Optional[str]:
    """The current workspace identity for cwd-scoped resume.

    Git repo root when CWD is inside a repo (so all sessions across its
    subdirs/worktrees group together), else the CWD itself. Returns None when
    neither can be determined — callers fall back to the global MRU then.
    """
    try:
        import subprocess

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


def _resolve_last_session(source: str = "cli") -> Optional[str]:
    """Look up the most recently-used session ID for a source.

    Scoped to the current workspace first (git repo root, else cwd) so
    ``hermes -c`` from repo A continues repo A's last session rather than the
    global MRU. Falls back to the unscoped MRU when no session matches the
    current workspace, preserving the old behaviour for fresh directories.
    """
    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        ws_key = _resolve_workspace_key()
        if ws_key:
            sessions = db.search_sessions(source=source, limit=1, workspace_key=ws_key)
            if sessions:
                return sessions[0]["id"]
        # Fallback: global MRU for this source.
        sessions = db.search_sessions(source=source, limit=1)
        return sessions[0]["id"] if sessions else None
    except Exception:
        pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
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
    sudo_path = None
    probe = _probe_container(
        [runtime, "inspect", "--format", "ok", container_name],
        backend,
    )
    if probe.returncode != 0:
        sudo_path = shutil.which("sudo")
        if sudo_path:
            probe2 = _probe_container(
                [sudo_path, "-n", runtime, "inspect", "--format", "ok", container_name],
                backend,
                via_sudo=True,
            )
            if probe2.returncode != 0:
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
        else:
            print(
                f"Error: container '{container_name}' not found via {backend}.\n"
                f"The container may be running under root. Try: sudo hermes {' '.join(cli_args)}",
                file=sys.stderr,
            )
            sys.exit(1)

    is_tty = sys.stdin.isatty()
    tty_flags = ["-it"] if is_tty else ["-i"]

    env_flags = []
    for var in ("TERM", "COLORTERM", "LANG", "LC_ALL"):
        val = os.environ.get(var)
        if val:
            env_flags.extend(["-e", f"{var}={val}"])

    cmd_prefix = [sudo_path, "-n", runtime] if sudo_path else [runtime]
    exec_cmd = (
        cmd_prefix
        + ["exec"]
        + tty_flags
        + ["-u", exec_user]
        + env_flags
        + [container_name, hermes_bin]
        + cli_args
    )

    os.execvp(exec_cmd[0], exec_cmd)


def _resolve_session_by_name_or_id(name_or_id: str) -> Optional[str]:
    """Resolve a session name (title) or ID to a session ID.

    - If it looks like a session ID (contains underscore + hex), try direct lookup first.
    - Otherwise, treat it as a title and use resolve_session_by_title (auto-latest).
    - Falls back to the other method if the first doesn't match.
    - If the resolved session is a compression root, follow the chain forward
      to the latest continuation. Users who remember the old root ID (e.g.
      from an exit summary printed before the bug fix, or from notes) get
      resumed at the live tip instead of a stale parent with no messages.
    """
    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB()

        # Try as exact session ID first
        session = db.get_session(name_or_id)
        resolved_id: Optional[str] = None
        if session:
            resolved_id = session["id"]
        else:
            # Try as title (with auto-latest for lineage)
            resolved_id = db.resolve_session_by_title(name_or_id)

        if resolved_id:
            # Project forward through compression chain so resumes land on
            # the live tip instead of a dead compressed parent.
            try:
                resolved_id = db.get_compression_tip(resolved_id) or resolved_id
            except Exception:
                pass

        return resolved_id
    except Exception:
        pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
    return None


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


_NPM_LOCK_RUNTIME_KEYS = frozenset({"ideallyInert", "peer"})
"""Lockfile fields npm writes non-deterministically at install time.

``ideallyInert`` is npm's runtime annotation for packages it skipped installing
(per-platform opt-outs).  ``peer`` is dropped from the hidden ``.package-lock.json``
on dev-dependencies that are *also* declared as peers — the canonical
``package-lock.json`` records the dual role, but npm 9's actualized tree strips
it.  Neither key represents a real skew between what was declared and what was
installed, so we exclude them from the comparison in :func:`_tui_need_npm_install`
to avoid false-positive reinstalls on every launch.
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
      - present but with differing fields (excluding npm-written runtime
        annotations like ``ideallyInert``) → reinstall

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

    def comparable(pkg: dict) -> dict:
        return {k: v for k, v in pkg.items() if k not in _NPM_LOCK_RUNTIME_KEYS}

    for name, pkg in wanted.items():
        if not name:
            continue

        if not isinstance(pkg, dict):
            continue

        if name not in installed:
            if pkg.get("optional") or pkg.get("peer"):
                continue
            return True

        if isinstance(installed[name], dict) and comparable(pkg) != comparable(
            installed[name]
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
                env={**with_hermes_node_path(), "CI": "1"},
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

    # Desktop/dev launches retain the historical "always rebuild" behaviour.
    # Termux cold starts use the freshness check because esbuild startup is
    # expensive on old mobile CPUs.
    should_build = True
    if termux_startup:
        should_build = did_install or termux_need_rebuild

    if should_build:
        npm = _node_bin("npm")
        result = subprocess.run(
            [npm, "run", "build"],
            cwd=str(tui_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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
                _prune_stale_worktrees,
                _setup_worktree,
            )

            repo = _git_repo_root()
            if repo:
                _prune_stale_worktrees(repo)
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


def cmd_chat(args):
    """Run interactive chat CLI."""
    use_tui = _resolve_use_tui(args)

    _apply_safe_mode(args)

    # --in DIR: run in DIR. Must happen before any session resolution so the
    # workspace-scoped "latest"/-c lookups key off DIR, and it pins the
    # session there — an explicit --in wins over a resumed session's
    # recorded cwd (so the restore step below is skipped).
    in_dir = getattr(args, "in_dir", None)
    if in_dir:
        # Git Bash / MSYS hands the CLI POSIX-style paths (`--in ~` expands to
        # `/c/Users/x` before Python ever sees it; MSYS2's path conversion is
        # disabled for native executables). Translate the MSYS/Cygwin/WSL
        # drive-root spellings to native Windows form first — no-op elsewhere.
        from tools.environments.local import _msys_to_windows_path

        _target_dir = os.path.abspath(
            os.path.expanduser(_msys_to_windows_path(in_dir))
        )
        if not os.path.isdir(_target_dir):
            print(f"Error: --in directory not found: {in_dir}")
            sys.exit(1)
        try:
            os.chdir(_target_dir)
        except OSError as e:
            print(f"Error: cannot enter --in directory {in_dir}: {e}")
            sys.exit(1)
        args.no_restore_cwd = True

    # --resume latest: keyword for "most recent session" — same resolution
    # as `-c` with no name (workspace-scoped MRU, then global fallback).
    # The keyword wins over a session literally titled "latest"; that
    # session stays reachable via its ID or `-c latest` (title match).
    _resume_raw = getattr(args, "resume", None)
    if isinstance(_resume_raw, str) and _resume_raw.strip().lower() == "latest":
        _source = "tui" if use_tui else "cli"
        _last_id = _resolve_last_session(source=_source)
        if not _last_id and _source == "tui":
            _last_id = _resolve_last_session(source="cli")
        if _last_id:
            args.resume = _last_id
        else:
            kind = "TUI" if use_tui else "CLI"
            print(f"No previous {kind} session found to resume.")
            print("Use 'hermes sessions list' to see available sessions.")
            sys.exit(1)

    # Resolve --continue into --resume with the latest session or by name
    continue_val = getattr(args, "continue_last", None)
    if continue_val and not getattr(args, "resume", None):
        if isinstance(continue_val, str):
            # -c "session name" — resolve by title or ID
            resolved = _resolve_session_by_name_or_id(continue_val)
            if resolved:
                args.resume = resolved
            else:
                print(f"No session found matching '{continue_val}'.")
                print("Use 'hermes sessions list' to see available sessions.")
                sys.exit(1)
        else:
            # -c with no argument — continue the most recent session
            source = "tui" if use_tui else "cli"
            last_id = _resolve_last_session(source=source)
            if not last_id and source == "tui":
                last_id = _resolve_last_session(source="cli")
            if last_id:
                args.resume = last_id
            else:
                kind = "TUI" if use_tui else "CLI"
                print(f"No previous {kind} session found to continue.")
                sys.exit(1)

    # Resolve --resume by title if it's not a direct session ID
    resume_val = getattr(args, "resume", None)
    if resume_val:
        resolved = _resolve_session_by_name_or_id(resume_val)
        if resolved:
            args.resume = resolved
        # If resolution fails, keep the original value — _init_agent will
        # report "Session not found" with the original input

    # Session<->workspace binding: cd back into a resumed session's recorded cwd
    # so it resumes in the repo it belonged to. Opt out with --no-restore-cwd;
    # skipped under --worktree (that path owns its own dir). Best-effort — a
    # missing dir warns and stays put rather than failing the resume.
    if (
        getattr(args, "resume", None)
        and not getattr(args, "no_restore_cwd", False)
        and not getattr(args, "worktree", False)
    ):
        _resume_db = None
        try:
            from hermes_state import SessionDB

            _resume_db = SessionDB()
            _saved_cwd = ((_resume_db.get_session(args.resume) or {}).get("cwd") or "").strip()
            if _saved_cwd and not os.path.isdir(_saved_cwd):
                print(f"⚠ session's recorded dir is gone ({_saved_cwd}); staying in {os.getcwd()}")
            elif _saved_cwd and os.path.realpath(_saved_cwd) != os.path.realpath(os.getcwd()):
                os.chdir(_saved_cwd)
                print(f"↪ restored workspace dir: {_saved_cwd}")
        except Exception:
            pass  # never let cwd-restore break a resume
        finally:
            if _resume_db is not None:
                try:
                    _resume_db.close()
                except Exception:
                    pass

    # xAI retirement warning — one-shot, non-blocking, never fails startup
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

    # First-run guard: check if any provider is configured before launching
    if not _has_any_provider_configured():
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

    # Start update check in background (runs while other init happens).
    # On Termux this imports rich/prompt_toolkit in the foreground and then
    # competes for CPU on single-core devices, so keep it opt-in there.
    if _termux_should_prefetch_update_check():
        try:
            from hermes_cli.banner import prefetch_banner_data, prefetch_update_check

            prefetch_update_check()
            # Warm git banner state + skills index off-thread too — their
            # subprocess/file-I/O waits overlap the CPU-bound cli import.
            prefetch_banner_data()
        except Exception:
            pass

    # Sync bundled skills on every CLI launch. Runs in a background daemon
    # thread: the sync is idempotent, hash-gated (unchanged skills are
    # skipped), and nothing on the banner path depends on it, yet the scan
    # alone costs ~120-170ms of rglob/hashing on the startup path. Skill
    # loading happens at agent init (first message), by which point the
    # sync has long finished; a same-instant race would only matter in the
    # rare launch right after `hermes update` changed a bundled skill.
    def _skills_sync_bg() -> None:
        try:
            _sync_bundled_skills_for_startup()
        except Exception:
            pass

    threading.Thread(
        target=_skills_sync_bg, name="bundled-skills-sync", daemon=True
    ).start()

    # --yolo: bypass all dangerous command approvals.
    # Also set in main() before _prepare_agent_startup() — that is the
    # authoritative site because it runs before tool imports freeze
    # _YOLO_MODE_FROZEN.  This redundant set is a safety net for callers
    # that invoke cmd_chat directly (e.g. subcommand dispatch).
    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"

    # --ignore-user-config: make load_cli_config() / load_config() skip the
    # user's ~/.hermes/config.yaml and return built-in defaults. Set BEFORE
    # importing cli (which runs `CLI_CONFIG = load_cli_config()` at module
    # import time). Credentials in .env are still loaded — this flag only
    # ignores behavioral/config settings.
    if getattr(args, "ignore_user_config", False):
        os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"

    # --ignore-rules: skip auto-injection of AGENTS.md/SOUL.md/.cursorrules
    # (rules), memory entries, and any preloaded skills coming from user config.
    # Maps to AIAgent(skip_context_files=True, skip_memory=True).
    if getattr(args, "ignore_rules", False):
        os.environ["HERMES_IGNORE_RULES"] = "1"

    # --source: tag session source for filtering (e.g. 'tool' for third-party integrations)
    if getattr(args, "source", None):
        os.environ["HERMES_SESSION_SOURCE"] = args.source

    _pin_kanban_board_env()
    _confirm_startup_expensive_model_override(args)

    if use_tui:
        _launch_tui(
            getattr(args, "resume", None),
            tui_dev=getattr(args, "tui_dev", False),
            model=getattr(args, "model", None),
            provider=getattr(args, "provider", None),
            toolsets=getattr(args, "toolsets", None),
            skills=getattr(args, "skills", None),
            verbose=getattr(args, "verbose", None),
            quiet=getattr(args, "quiet", False),
            query=getattr(args, "query", None),
            image=getattr(args, "image", None),
            worktree=getattr(args, "worktree", False),
            checkpoints=getattr(args, "checkpoints", False),
            pass_session_id=getattr(args, "pass_session_id", False),
            max_turns=getattr(args, "max_turns", None),
            accept_hooks=getattr(args, "accept_hooks", False),
        )

    # Import and run the CLI
    from cli import main as cli_main

    # Build kwargs from args
    kwargs = {
        "model": args.model,
        "provider": getattr(args, "provider", None),
        "reasoning": getattr(args, "reasoning", None),
        "toolsets": args.toolsets,
        "skills": getattr(args, "skills", None),
        "verbose": getattr(args, "verbose", None),
        "quiet": getattr(args, "quiet", False),
        "query": args.query,
        "image": getattr(args, "image", None),
        "resume": getattr(args, "resume", None),
        "worktree": getattr(args, "worktree", False),
        "checkpoints": getattr(args, "checkpoints", False),
        "pass_session_id": getattr(args, "pass_session_id", False),
        "max_turns": getattr(args, "max_turns", None),
        "ignore_rules": getattr(args, "ignore_rules", False) or getattr(args, "safe_mode", False),
        "ignore_user_config": getattr(args, "ignore_user_config", False) or getattr(args, "safe_mode", False),
        "compact": getattr(args, "compact", False),
    }
    # Filter out None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        cli_main(**kwargs)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_gateway(args):
    """Gateway management commands."""
    _sync_bundled_skills_quietly()

    from hermes_cli.gateway import gateway_command

    gateway_command(args)


def cmd_proxy(args):
    """Local OpenAI-compatible proxy to OAuth providers."""
    # Lazy import — pulls in aiohttp, which is gated behind an extras install
    # for users who don't run the proxy or the messaging gateway.
    from hermes_cli.proxy.cli import cmd_proxy as _cmd_proxy

    rc = _cmd_proxy(args)
    if isinstance(rc, int) and rc != 0:
        raise SystemExit(rc)


def cmd_whatsapp(args):
    """Set up WhatsApp: choose mode, configure, install bridge, pair via QR."""
    _require_tty("whatsapp")
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_constants import find_node_executable, with_hermes_node_path

    print()
    print("⚕ WhatsApp Setup")
    print("=" * 50)

    # ── Step 1: Choose mode ──────────────────────────────────────────────
    current_mode = get_env_value("WHATSAPP_MODE") or ""
    if not current_mode:
        print()
        print("How will you use WhatsApp with Hermes?")
        print()
        print("  1. Separate bot number (recommended)")
        print("     People message the bot's number directly — cleanest experience.")
        print(
            "     Requires a second phone number with WhatsApp installed on a device."
        )
        print()
        print("  2. Personal number (self-chat)")
        print("     You message yourself to talk to the agent.")
        print("     Quick to set up, but the UX is less intuitive.")
        print()
        try:
            choice = input("  Choose [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            return

        if choice == "1":
            save_env_value("WHATSAPP_MODE", "bot")
            wa_mode = "bot"
            print("  ✓ Mode: separate bot number")
            print()
            print("  ┌─────────────────────────────────────────────────┐")
            print("  │  Getting a second number for the bot:           │")
            print("  │                                                 │")
            print("  │  Easiest: Install WhatsApp Business (free app)  │")
            print("  │  on your phone with a second number:            │")
            print("  │    • Dual-SIM: use your 2nd SIM slot            │")
            print("  │    • Google Voice: free US number (voice.google) │")
            print("  │    • Prepaid SIM: $3-10, verify once            │")
            print("  │                                                 │")
            print("  │  WhatsApp Business runs alongside your personal │")
            print("  │  WhatsApp — no second phone needed.             │")
            print("  └─────────────────────────────────────────────────┘")
        else:
            save_env_value("WHATSAPP_MODE", "self-chat")
            wa_mode = "self-chat"
            print("  ✓ Mode: personal number (self-chat)")
    else:
        wa_mode = current_mode
        mode_label = (
            "separate bot number" if wa_mode == "bot" else "personal number (self-chat)"
        )
        print(f"\n✓ Mode: {mode_label}")

    # ── Step 2: Mode is selected, will enable WhatsApp only after pairing ──
    # We intentionally don't write WHATSAPP_ENABLED=true here.  If the user
    # aborts the wizard later (Ctrl+C, failed npm install, missed QR scan),
    # we'd otherwise leave .env claiming WhatsApp is ready when the bridge
    # has no creds.json.  Every subsequent `hermes gateway` then paid a 30s
    # bridge-bootstrap timeout and queued WhatsApp for indefinite retries.
    # Now: aborted setup leaves WHATSAPP_ENABLED unset → gateway skips it.
    # Re-runs that already have WHATSAPP_ENABLED=true (from a prior
    # successful pairing) stay enabled — we just don't write it pre-emptively.
    print()
    if (get_env_value("WHATSAPP_ENABLED") or "").lower() == "true":
        print("✓ WhatsApp is already enabled")

    # ── Step 3: Allowed users ────────────────────────────────────────────
    current_users = get_env_value("WHATSAPP_ALLOWED_USERS") or ""
    if current_users:
        print(f"✓ Allowed users: {current_users}")
        try:
            response = input("\n  Update allowed users? [y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in {"y", "yes"}:
            if wa_mode == "bot":
                phone = input(
                    "  Phone numbers that can message the bot (comma-separated): "
                ).strip()
            else:
                phone = input("  Your phone number (e.g. 15551234567): ").strip()
            if phone:
                save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
                print(f"  ✓ Updated to: {phone}")
    else:
        print()
        if wa_mode == "bot":
            print("  Who should be allowed to message the bot?")
            phone = input(
                "  Phone numbers (comma-separated, or * for anyone): "
            ).strip()
        else:
            phone = input("  Your phone number (e.g. 15551234567): ").strip()
        if phone:
            save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
            print(f"  ✓ Allowed users set: {phone}")
        else:
            print("  ⚠ No allowlist — the agent will respond to ALL incoming messages")

    # ── Step 4: Install bridge dependencies ──────────────────────────────
    from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
    bridge_dir = resolve_whatsapp_bridge_dir()
    bridge_script = bridge_dir / "bridge.js"

    if not bridge_script.exists():
        print(f"\n✗ Bridge script not found at {bridge_script}")
        return

    if not (bridge_dir / "node_modules").exists():
        print(
            "\n→ Installing WhatsApp bridge dependencies (this can take a few minutes)..."
        )
        npm = find_node_executable("npm")
        if not npm:
            print("  ✗ npm not found on PATH — install Node.js first")
            return
        try:
            result = subprocess.run(
                [npm, "install", "--no-fund", "--no-audit", "--progress=false"],
                cwd=str(bridge_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=with_hermes_node_path(),
            )
        except KeyboardInterrupt:
            print("\n  ✗ Install cancelled")
            return
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            preview = "\n".join(err.splitlines()[-30:]) if err else "(no output)"
            print("  ✗ npm install failed:")
            print(preview)
            return
        print("  ✓ Dependencies installed")
    else:
        print("✓ Bridge dependencies already installed")

    # ── Step 5: Check for existing session ───────────────────────────────
    session_dir = get_hermes_home() / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    if (session_dir / "creds.json").exists():
        print("✓ Existing WhatsApp session found")
        try:
            response = input(
                "\n  Re-pair? This will clear the existing session. [y/N] "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in {"y", "yes"}:
            shutil.rmtree(session_dir, ignore_errors=True)
            session_dir.mkdir(parents=True, exist_ok=True)
            print("  ✓ Session cleared")
        else:
            # Existing pairing — ensure WHATSAPP_ENABLED reflects that.
            # (Older installs may have lost the env var; covers re-runs
            # where the user picked "no, keep my session" but the var
            # was never set or got removed.)
            if (get_env_value("WHATSAPP_ENABLED") or "").lower() != "true":
                save_env_value("WHATSAPP_ENABLED", "true")
            print("\n✓ WhatsApp is configured and paired!")
            print("  Start the gateway with: hermes gateway")
            return

    # ── Step 6: QR code pairing ──────────────────────────────────────────
    print()
    print("─" * 50)
    if wa_mode == "bot":
        print("📱 Open WhatsApp (or WhatsApp Business) on the")
        print("   phone with the BOT's number, then scan:")
    else:
        print("📱 Open WhatsApp on your phone, then scan:")
    print()
    print("   Settings → Linked Devices → Link a Device")
    print("─" * 50)
    print()

    try:
        subprocess.run(
            [
                find_node_executable("node") or "node",
                str(bridge_script),
                "--pair-only",
                "--session",
                str(session_dir),
            ],
            cwd=str(bridge_dir),
            env=with_hermes_node_path(),
        )
    except KeyboardInterrupt:
        pass

    # ── Step 7: Post-pairing ─────────────────────────────────────────────
    print()
    if (session_dir / "creds.json").exists():
        # Only enable WhatsApp now that pairing actually succeeded.  If the
        # user Ctrl+C'd at any earlier step, WHATSAPP_ENABLED stays unset
        # and `hermes gateway` skips it cleanly instead of paying a 30s
        # bridge timeout + queueing the platform for indefinite retries.
        save_env_value("WHATSAPP_ENABLED", "true")
        print("✓ WhatsApp paired successfully!")
        print()
        if wa_mode == "bot":
            print("  Next steps:")
            print("    1. Start the gateway:  hermes gateway")
            print("    2. Send a message to the bot's WhatsApp number")
            print("    3. The agent will reply automatically")
            print()
            print("  Tip: Agent responses are prefixed with '⚕ Hermes Agent'")
        else:
            print("  Next steps:")
            print("    1. Start the gateway:  hermes gateway")
            print("    2. Open WhatsApp → Message Yourself")
            print("    3. Type a message — the agent will reply")
            print()
            print("  Tip: Agent responses are prefixed with '⚕ Hermes Agent'")
            print("  so you can tell them apart from your own messages.")
        print()
        print("  Or install as a service: hermes gateway install")
    else:
        print("⚠ Pairing may not have completed. Run 'hermes whatsapp' to try again.")


def cmd_whatsapp_cloud(args):
    """Set up WhatsApp Business Cloud API (official Meta integration).

    Walks the user through the Meta-side credentials (Phone Number ID,
    Access Token, App Secret, optional App/WABA IDs) plus webhook
    configuration. Includes field-shape validators that catch the most
    common setup mistakes (e.g. pasting a phone number into the Phone
    Number ID field).

    Distinct from ``hermes whatsapp`` (the Baileys bridge wizard) — the
    two adapters are complementary, not alternatives. See
    ``hermes_cli/setup_whatsapp_cloud.py``.
    """
    _require_tty("whatsapp-cloud")
    from hermes_cli.setup_whatsapp_cloud import run_whatsapp_cloud_setup

    return run_whatsapp_cloud_setup()


def cmd_setup(args):
    """Interactive setup wizard."""
    from hermes_cli.setup import run_setup_wizard

    run_setup_wizard(args)


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
    select_provider_and_model(args=args)


def _is_profile_api_key_provider(provider_id: str) -> bool:
    """Return True when provider_id maps to a profile with auth_type='api_key'.

    Used as a catch-all in select_provider_and_model() so that new providers
    declared in plugins/model-providers/<name>/ automatically dispatch to _model_flow_api_key_provider
    without requiring an explicit elif branch here.
    """
    try:
        from providers import get_provider_profile
        _p = get_provider_profile(provider_id)
        return _p is not None and _p.auth_type == "api_key"
    except Exception:
        return False


def select_provider_and_model(args=None):
    """Core provider selection + model picking logic.

    Shared by ``cmd_model`` (``hermes model``) and the setup wizard
    (``setup_model_provider`` in setup.py).  Handles the full flow:
    provider picker, credential prompting, model selection, and config
    persistence.
    """
    from hermes_cli.auth import (
        resolve_provider,
        AuthError,
        format_auth_error,
    )
    from hermes_cli.config import (
        get_compatible_custom_providers,
        load_config,
        get_env_value,
    )
    from hermes_cli.providers import (
        custom_provider_aliases,
        custom_provider_slug,
        resolve_provider_full,
    )

    config = load_config()
    current_model = config.get("model")
    if isinstance(current_model, dict):
        current_model = current_model.get("default", "")
    current_model = current_model or "(not set)"

    # Read effective provider the same way the CLI does at startup:
    # config.yaml model.provider > env var > auto-detect
    config_provider = None
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        config_provider = model_cfg.get("provider")

    effective_provider = (
        config_provider or os.getenv("HERMES_INFERENCE_PROVIDER") or "auto"
    )
    compatible_custom_providers = get_compatible_custom_providers(config)
    def _named_custom_provider_map(cfg) -> dict[str, dict[str, str]]:
        from hermes_cli.config import read_raw_config

        # Build lookups of raw (un-expanded) templates keyed by a
        # stable identity. We intentionally bypass
        # ``get_compatible_custom_providers(read_raw_config())`` here because
        # its ``_normalize_custom_provider_entry`` step calls ``urlparse()``
        # on ``base_url`` and drops any entry whose ``base_url`` is itself an
        # env-ref template (e.g. ``${NEURALWATT_API_BASE}``). Dropping those
        # entries is exactly how env-ref preservation fails for the user
        # config that motivated this fix.
        raw_api_key_refs: dict[tuple, str] = {}
        raw_base_url_refs: dict[tuple, str] = {}
        raw_cfg = read_raw_config()

        def _record_raw(
            name: str,
            provider_key: str,
            model: str,
            api_key: str,
            base_url: str,
        ) -> None:
            template = str(api_key or "").strip()
            base_template = str(base_url or "").strip()
            name = str(name or "").strip()
            provider_key = str(provider_key or "").strip()
            model = str(model or "").strip()
            # Index by every plausible identity the loaded (expanded) config
            # might present: (name), (name, model), (provider_key), and
            # (provider_key, model). Case-insensitive on name/provider_key so
            # the loaded entry matches regardless of display casing.
            identities = []
            if name:
                identities.extend(((name.lower(),), (name.lower(), model)))
            if provider_key:
                identities.extend(
                    ((provider_key.lower(),), (provider_key.lower(), model))
                )
            if "${" in template:
                for identity in identities:
                    raw_api_key_refs.setdefault(identity, template)
            if "${" in base_template:
                for identity in identities:
                    raw_base_url_refs.setdefault(identity, base_template)

        raw_list = raw_cfg.get("custom_providers")
        if isinstance(raw_list, list):
            for raw_entry in raw_list:
                if not isinstance(raw_entry, dict):
                    continue
                _record_raw(
                    raw_entry.get("name", ""),
                    "",
                    raw_entry.get("model", "") or raw_entry.get("default_model", ""),
                    raw_entry.get("api_key", ""),
                    raw_entry.get("base_url", "")
                    or raw_entry.get("url", "")
                    or raw_entry.get("api", ""),
                )
        raw_providers = raw_cfg.get("providers")
        if isinstance(raw_providers, dict):
            for raw_key, raw_entry in raw_providers.items():
                if not isinstance(raw_entry, dict):
                    continue
                _record_raw(
                    raw_entry.get("name", "") or raw_key,
                    raw_key,
                    raw_entry.get("model", "") or raw_entry.get("default_model", ""),
                    raw_entry.get("api_key", ""),
                    raw_entry.get("base_url", "")
                    or raw_entry.get("url", "")
                    or raw_entry.get("api", ""),
                )

        def _lookup_ref(
            refs: dict[tuple, str],
            name: str,
            provider_key: str,
            model: str,
        ) -> str:
            name_lc = str(name or "").strip().lower()
            pkey_lc = str(provider_key or "").strip().lower()
            model = str(model or "").strip()
            for identity in (
                (pkey_lc, model),
                (pkey_lc,),
                (name_lc, model),
                (name_lc,),
            ):
                if identity[0] and identity in refs:
                    return refs[identity]
            return ""

        custom_provider_map = {}
        for entry in get_compatible_custom_providers(cfg):
            if not isinstance(entry, dict):
                continue
            name = (entry.get("name") or "").strip()
            base_url = (entry.get("base_url") or "").strip()
            if not name or not base_url:
                continue
            provider_key = (entry.get("provider_key") or "").strip()
            key = custom_provider_slug(name, provider_key)
            custom_provider_map[key] = {
                "name": name,
                "base_url": base_url,
                "api_key": entry.get("api_key", ""),
                "key_env": entry.get("key_env", ""),
                "model": entry.get("model", ""),
                "models": entry.get("models", {}),
                "discover_models": entry.get("discover_models", True),
                "api_mode": entry.get("api_mode", ""),
                "provider_key": provider_key,
                "api_key_ref": _lookup_ref(
                    raw_api_key_refs, name, provider_key, entry.get("model", "")
                ),
                "base_url_ref": _lookup_ref(
                    raw_base_url_refs, name, provider_key, entry.get("model", "")
                ),
            }
        return custom_provider_map

    def _norm_base_url(url: str) -> str:
        return str(url or "").strip().rstrip("/").lower()

    # Add user-defined custom providers from config.yaml
    _custom_provider_map = _named_custom_provider_map(
        config
    )  # key → {name, base_url, api_key}

    def _canonical_named_custom_key(provider_id: str) -> str:
        requested = str(provider_id or "").strip().lower()
        for key, provider_info in _custom_provider_map.items():
            if requested in custom_provider_aliases(
                provider_info.get("name", ""),
                provider_info.get("provider_key", ""),
            ):
                return key
        return provider_id

    def _active_custom_key_from_base_url() -> str:
        if effective_provider != "custom" or not isinstance(model_cfg, dict):
            return ""
        current_base = _norm_base_url(model_cfg.get("base_url", ""))
        if not current_base:
            return ""
        for key, provider_info in _custom_provider_map.items():
            if _norm_base_url(provider_info.get("base_url", "")) == current_base:
                return key
        return ""

    active = _active_custom_key_from_base_url()
    if active is None:
        active = ""
    if not active and effective_provider != "auto":
        active_def = resolve_provider_full(
            effective_provider,
            config.get("providers"),
            compatible_custom_providers,
        )
        if active_def is not None:
            active = active_def.id
            if active_def.source == "user-config":
                active = _canonical_named_custom_key(active)
        else:
            warning = (
                f"Unknown provider '{effective_provider}'. Check 'hermes model' for "
                "available providers, or run 'hermes doctor' to diagnose config "
                "issues."
            )
            print(f"Warning: {warning} Falling back to auto provider detection.")
    if not active:
        try:
            active = resolve_provider("auto")
        except AuthError as exc:
            if effective_provider == "auto":
                warning = format_auth_error(exc)
                print(f"Warning: {warning} Falling back to auto provider detection.")
            active = None  # no provider yet; default to first in list

    # Detect custom endpoint
    if active == "openrouter" and get_env_value("OPENAI_BASE_URL"):
        active = "custom"

    from hermes_cli.models import (
        CANONICAL_PROVIDERS,
        _PROVIDER_LABELS,
        _PROVIDER_ALIASES,
        group_providers,
        provider_group_for_slug,
    )

    provider_labels = dict(_PROVIDER_LABELS)  # derive from canonical list
    if active and active in _custom_provider_map:
        active_label = _custom_provider_map[active]["name"]
    else:
        active_label = provider_labels.get(active, active) if active else "none"

    print()
    print(f"  Current model:    {current_model}")
    print(f"  Active provider:  {active_label}")
    print()

    # Step 1: Provider selection.
    #
    # Canonical providers are folded into top-level groups (display only — see
    # PROVIDER_GROUPS in hermes_cli/models.py). A multi-member group shows one
    # row ("Kimi / Moonshot ▸"); picking it opens a member sub-picker that
    # resolves back to a concrete slug, so the dispatch chain below is
    # unchanged. Custom providers and the trailing actions stay flat.
    canonical_descs = {p.slug: p.tui_desc for p in CANONICAL_PROVIDERS}
    # Honor ``model_catalog.excluded_providers`` so the CLI ``hermes model``
    # picker hides the same providers the gateway/TUI pickers do. A canonical
    # provider is hidden if its slug OR any of its aliases appears in the
    # exclusion list (case-insensitive), matching list_authenticated_providers'
    # matching against hermes_id / alias / canonical slug.
    _cli_excluded = {
        str(p).strip().lower()
        for p in (config.get("model_catalog", {}) or {}).get("excluded_providers") or []
        if p
    }
    if _cli_excluded:
        _alias_to_canon = _PROVIDER_ALIASES
        _names_for: dict[str, set[str]] = {}
        for _p in CANONICAL_PROVIDERS:
            _names_for[_p.slug] = {_p.slug.lower()}
        for _alias, _canon in _alias_to_canon.items():
            _names_for.setdefault(_canon, {_canon.lower()}).add(_alias.lower())
        _visible_slugs = [
            p.slug for p in CANONICAL_PROVIDERS
            if not _names_for.get(p.slug, {p.slug.lower()}) & _cli_excluded
        ]
    else:
        _visible_slugs = [p.slug for p in CANONICAL_PROVIDERS]
    grouped_rows = group_providers(_visible_slugs)

    # The group/slug that should be pre-selected: the active provider's group
    # if it's grouped, otherwise the active slug itself.
    active_group = provider_group_for_slug(active) if active else ""

    # ordered entries: (key, label, members)
    #   members == [] → leaf row, key is a provider slug / action
    #   members != [] → group row, key is "group:<gid>"
    ordered: list[tuple[str, str, list[str]]] = []
    default_idx = 0
    for row in grouped_rows:
        if row["kind"] == "group":
            gid = row["group_id"]
            group_desc = row.get("description", "")
            label = f"{row['label']} ▸ ({group_desc})" if group_desc else f"{row['label']} ▸"
            key = f"group:{gid}"
            is_active = bool(active_group) and gid == active_group
            members = row["members"]
        else:
            slug = row["slug"]
            label = canonical_descs.get(slug, provider_labels.get(slug, slug))
            key = slug
            is_active = bool(active) and slug == active
            members = []
        if is_active:
            ordered.append((key, f"{label}  ← currently active", members))
            default_idx = len(ordered) - 1
        else:
            ordered.append((key, label, members))

    for key, provider_info in _custom_provider_map.items():
        name = provider_info["name"]
        base_url = provider_info["base_url"]
        short_url = base_url.replace("https://", "").replace("http://", "").rstrip("/")
        saved_model = provider_info.get("model", "")
        model_hint = f" — {saved_model}" if saved_model else ""
        label = f"{name} ({short_url}){model_hint}"
        if active and key == active:
            ordered.append((key, f"{label}  ← currently active", []))
            default_idx = len(ordered) - 1
        else:
            ordered.append((key, label, []))

    ordered.append(("custom", "Custom endpoint (enter URL manually)", []))
    _has_saved_custom_list = isinstance(config.get("custom_providers"), list) and bool(
        config.get("custom_providers")
    )
    if _has_saved_custom_list:
        ordered.append(("remove-custom", "Remove a saved custom provider", []))
    ordered.append(("aux-config", "Configure auxiliary models...", []))
    ordered.append(("cancel", "Leave unchanged", []))

    provider_idx = _prompt_provider_choice(
        [label for _, label, _ in ordered],
        default=default_idx,
    )
    if provider_idx is None or ordered[provider_idx][0] == "cancel":
        print("No change.")
        return

    selected_key = ordered[provider_idx][0]
    selected_members = ordered[provider_idx][2]

    # Group row → drill into a member sub-picker. Default to the active member
    # if the active provider lives in this group. The descriptive text lives on
    # the group row itself, so member rows show only their short label here.
    if selected_members:
        member_default = 0
        if active in selected_members:
            member_default = selected_members.index(active)
        member_labels = [
            provider_labels.get(m, m) for m in selected_members
        ]
        group_label = ordered[provider_idx][1].split(" ▸", 1)[0]
        member_idx = _prompt_provider_choice(
            member_labels,
            default=member_default,
            title=f"Select {group_label} provider:",
        )
        if member_idx is None:
            print("No change.")
            return
        selected_provider = selected_members[member_idx]
    else:
        selected_provider = selected_key

    if selected_provider == "aux-config":
        _aux_config_menu()
        return

    # Step 2: Provider-specific setup + model selection
    if selected_provider == "openrouter":
        _model_flow_openrouter(config, current_model)
    elif selected_provider == "moa":
        _model_flow_moa(config, current_model)
    elif selected_provider == "ai-gateway":
        _model_flow_ai_gateway(config, current_model)
    elif selected_provider == "nous":
        _model_flow_nous(config, current_model, args=args)
    elif selected_provider == "openai-codex":
        _model_flow_openai_codex(config, current_model)
    elif selected_provider == "xai-oauth":
        _model_flow_xai_oauth(config, current_model, args=args)
    elif selected_provider == "qwen-oauth":
        _model_flow_qwen_oauth(config, current_model)
    elif selected_provider == "minimax-oauth":
        _model_flow_minimax_oauth(config, current_model, args=args)
    elif selected_provider == "copilot-acp":
        _model_flow_copilot_acp(config, current_model)
    elif selected_provider == "copilot":
        _model_flow_copilot(config, current_model)
    elif selected_provider == "custom":
        _model_flow_custom(config)
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
    elif selected_provider == "anthropic":
        _model_flow_anthropic(config, current_model)
    elif selected_provider == "kimi-coding":
        _model_flow_kimi(config, current_model)
    elif selected_provider == "stepfun":
        _model_flow_stepfun(config, current_model)
    elif selected_provider == "bedrock":
        _model_flow_bedrock(config, current_model)
    elif selected_provider == "vertex":
        _model_flow_vertex(config, current_model)
    elif selected_provider == "azure-foundry":
        _model_flow_azure_foundry(config, current_model)
    elif selected_provider in {
        "openai-api",
        "gemini",
        "deepseek",
        "xai",
        "zai",
        "kimi-coding-cn",
        "minimax",
        "minimax-cn",
        "kilocode",
        "opencode-zen",
        "opencode-go",
        "alibaba",
        "huggingface",
        "xiaomi",
        "arcee",
        "gmi",
        "nvidia",
        "ollama-cloud",
        "tencent-tokenhub",
        "lmstudio",
    } or _is_profile_api_key_provider(selected_provider):
        _model_flow_api_key_provider(config, selected_provider, current_model)

    # ── Post-switch cleanup: clear stale OPENAI_BASE_URL ──────────────
    # When the user switches to a named provider (anything except "custom"),
    # a leftover OPENAI_BASE_URL in ~/.hermes/.env can poison auxiliary
    # clients that use provider:auto. Clear it proactively.  (#5161)
    if selected_provider not in {
        "custom",
        "cancel",
        "remove-custom",
    } and not selected_provider.startswith("custom:"):
        _clear_stale_openai_base_url()


def _clear_stale_openai_base_url():
    """Remove OPENAI_BASE_URL from ~/.hermes/.env if the active provider is not 'custom'.

    After a provider switch, a leftover OPENAI_BASE_URL causes auxiliary
    clients (compression, vision, delegation) with provider:auto to route
    requests to the old custom endpoint instead of the newly selected
    provider.  See issue #5161.
    """
    from hermes_cli.config import get_env_value, save_env_value, load_config

    cfg = load_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        provider = (model_cfg.get("provider") or "").strip().lower()
    else:
        provider = ""

    if provider == "custom" or not provider:
        return  # custom provider legitimately uses OPENAI_BASE_URL

    stale_url = get_env_value("OPENAI_BASE_URL")
    if stale_url:
        save_env_value("OPENAI_BASE_URL", "")
        print(
            f"Cleared stale OPENAI_BASE_URL from .env (was: {stale_url[:40]}...)"
            if len(stale_url) > 40
            else f"Cleared stale OPENAI_BASE_URL from .env (was: {stale_url})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary model configuration
#
# Hermes uses lightweight "auxiliary" models for side tasks (vision analysis,
# context compression, web extraction, session search, etc.). Each task has
# its own provider+model pair in config.yaml under `auxiliary.<task>`.
#
# The UI lives behind "Configure auxiliary models..." at the bottom of the
# `hermes model` provider picker. It does NOT re-run credential setup — it
# only routes already-authenticated providers to specific aux tasks. Users
# configure new providers through the normal `hermes model` flow first.
# ─────────────────────────────────────────────────────────────────────────────

# (task_key, display_name, short_description)
_AUX_TASKS: list[tuple[str, str, str]] = [
    ("vision", "Vision", "image/screenshot analysis"),
    ("compression", "Compression", "context summarization"),
    ("web_extract", "Web extract", "web page summarization"),
    ("approval", "Approval", "smart command approval"),
    ("mcp", "MCP", "MCP tool reasoning"),
    ("title_generation", "Title generation", "session titles"),
    ("memory_query_rewrite", "Memory query rewrite", "memory retrieval queries"),
    ("tts_audio_tags", "TTS audio tags", "Gemini TTS tag insertion"),
    ("skills_hub", "Skills hub", "skills search/install"),
    ("triage_specifier", "Triage specifier", "kanban spec fleshing"),
    ("kanban_decomposer", "Kanban decomposer", "task decomposition"),
    ("profile_describer", "Profile describer", "auto profile descriptions"),
    ("curator", "Curator", "skill-usage review pass"),
]


def _all_aux_tasks() -> list[tuple[str, str, str]]:
    """Return built-in + plugin-registered auxiliary tasks for picker/menu use.

    Built-in tasks come first (preserving order), followed by plugin tasks
    sorted by key. Used by ``_aux_config_menu``, ``_reset_aux_to_auto``, and
    display-name lookups so plugin-registered tasks (registered via
    :meth:`hermes_cli.plugins.PluginContext.register_auxiliary_task`) appear
    in the same surfaces as built-in ones without core knowing about them.
    """
    tasks = list(_AUX_TASKS)
    try:
        from hermes_cli.plugins import get_plugin_auxiliary_tasks
        for entry in get_plugin_auxiliary_tasks():
            tasks.append((entry["key"], entry["display_name"], entry["description"]))
    except Exception:
        # Plugin discovery failure must not break the aux config UI.
        # Built-in tasks remain available.
        pass
    return tasks


def _format_aux_current(task_cfg: dict) -> str:
    """Render the current aux config for display in the task menu."""
    if not isinstance(task_cfg, dict):
        return "auto"
    base_url = str(task_cfg.get("base_url") or "").strip()
    provider = str(task_cfg.get("provider") or "auto").strip() or "auto"
    model = str(task_cfg.get("model") or "").strip()
    if base_url:
        short = base_url.replace("https://", "").replace("http://", "").rstrip("/")
        return f"custom ({short})" + (f" · {model}" if model else "")
    if provider == "auto":
        return "auto" + (f" · {model}" if model else "")
    if model:
        return f"{provider} · {model}"
    return provider


def _save_aux_choice(
    task: str,
    *,
    provider: str,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
) -> None:
    """Persist an auxiliary task's provider/model to config.yaml.

    Only writes the four routing fields — timeout, download_timeout, and any
    other task-specific settings are preserved untouched. The main model
    config (``model.default``/``model.provider``) is never modified.
    """
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    aux = cfg.setdefault("auxiliary", {})
    if not isinstance(aux, dict):
        aux = {}
        cfg["auxiliary"] = aux
    entry = aux.setdefault(task, {})
    if not isinstance(entry, dict):
        entry = {}
        aux[task] = entry
    entry["provider"] = provider
    entry["model"] = model or ""
    entry["base_url"] = base_url or ""
    entry["api_key"] = api_key or ""
    save_config(cfg)


def _reset_aux_to_auto() -> int:
    """Reset every known aux task back to auto/empty. Returns number reset.

    Includes plugin-registered tasks (via ``_all_aux_tasks``) so a plugin
    that contributed an auxiliary task gets reset alongside built-ins.
    """
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    aux = cfg.setdefault("auxiliary", {})
    if not isinstance(aux, dict):
        aux = {}
        cfg["auxiliary"] = aux
    count = 0
    for task, _name, _desc in _all_aux_tasks():
        entry = aux.setdefault(task, {})
        if not isinstance(entry, dict):
            entry = {}
            aux[task] = entry
        changed = False
        if entry.get("provider") not in {None, "", "auto"}:
            entry["provider"] = "auto"
            changed = True
        for field in ("model", "base_url", "api_key"):
            if entry.get(field):
                entry[field] = ""
                changed = True
        # Preserve timeout/download_timeout — those are user-tuned, not routing
        if changed:
            count += 1
    save_config(cfg)
    return count


def _aux_config_menu() -> None:
    """Top-level auxiliary-model picker — choose a task to configure.

    Loops until the user picks "Back" so multiple tasks can be configured
    without returning to the main provider menu.
    """
    from hermes_cli.config import load_config

    while True:
        cfg = load_config()
        aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}

        print()
        print("  Auxiliary models — side-task routing")
        print()
        print("  Side tasks (vision, compression, web extraction, etc.) default")
        print('  to your main chat model.  "auto" means "use my main model" —')
        print("  Hermes only falls back to a lightweight backend (OpenRouter,")
        print("  Nous Portal) if the main model is unavailable.  Override a")
        print("  task below if you want it pinned to a specific provider/model.")
        print()

        # Build the task menu with current settings inline
        all_tasks = _all_aux_tasks()
        name_col = max(len(name) for _, name, _ in all_tasks) + 2
        desc_col = max(len(desc) for _, _, desc in all_tasks) + 4
        entries: list[tuple[str, str]] = []
        for task_key, name, desc in all_tasks:
            task_cfg = (
                aux.get(task_key, {}) if isinstance(aux.get(task_key), dict) else {}
            )
            current = _format_aux_current(task_cfg)
            label = (
                f"{name.ljust(name_col)}{('(' + desc + ')').ljust(desc_col)}{current}"
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
    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    task_cfg = aux.get(task, {}) if isinstance(aux.get(task), dict) else {}
    current_provider = str(task_cfg.get("provider") or "auto").strip() or "auto"
    current_model = str(task_cfg.get("model") or "").strip()
    current_base_url = str(task_cfg.get("base_url") or "").strip()

    display_name = next((name for key, name, _ in _all_aux_tasks() if key == task), task)

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
    entries.append(("__auto__", f"auto (recommended){auto_marker}", []))

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

    display_name = next((name for key, name, _ in _all_aux_tasks() if key == task), task)

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
            val = input("Model: ").strip()
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

    display_name = next((name for key, name, _ in _all_aux_tasks() if key == task), task)
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
        url = input(url_prompt).strip()
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
        model = input(model_prompt).strip()
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
        "_add_upstream_remote",
        "_atomic_replace_dir",
        "_capture_active_lazy_features",
        "_capture_active_tool_dependencies",
        "_capture_head_sha",
        "_cmd_update_check",
        "_cmd_update_impl",
        "_cold_start_windows_gateway_after_update",
        "_count_commits_between",
        "_detect_self_loaded_native_modules",
        "_detect_venv_python_processes",
        "_defer_update_for_self_lock",
        "_discard_lockfile_churn",
        "_discard_stashed_changes",
        "_ensure_acp_launcher",
        "_ensure_fhs_path_guard",
        "_ensure_uv_for_termux",
        "_finish_dashboard_update_cleanup",
        "_for_each_systemd_gateway_unit",
        "_format_concurrent_instances_message",
        "_format_time_ago",
        "_format_venv_python_holders_message",
        "_gateway_prompt",
        "_get_origin_url",
        "_has_upstream_remote",
        "_install_psutil_android_compat",
        "_invalidate_update_cache",
        "_is_android_python",
        "_is_fork",
        "_leftover_pausable_gateway_pids",
        "_log_only_write",
        "_mark_skip_upstream_prompt",
        "_npm_bin_exists",
        "_npm_lockfile_changed",
        "_npm_manifest_paths",
        "_npm_manifests_digest",
        "_orphaned_desktop_backend_pids",
        "_pause_windows_gateways_for_update",
        "_print_curator_first_run_notice",
        "_print_curator_recent_run_notice",
        "_print_fts_optimize_available_notice",
        "_print_stash_cleanup_guidance",
        "_print_update_completion",
        "_record_npm_lockfile_hash",
        "_refresh_active_lazy_features",
        "_refresh_active_memory_provider_dependencies",
        "_refresh_bootstrap_cache_scripts",
        "_refresh_windows_gateway_launchers",
        "_reload_updated_runtime_modules",
        "_resolve_pre_update_backup_mode",
        "_resolve_stash_selector",
        "_restart_phase_failure_is_incomplete",
        "_restore_active_tool_dependencies",
        "_restore_stashed_changes",
        "_resume_windows_gateways_after_update",
        "_run_logged_subprocess",
        "_run_pre_update_backup",
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
        "_web_build_toolchain_ready",
        "_web_toolchain_roots",
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

    config_command(args)


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
    from hermes_cli.config import detect_install_method
    from hermes_cli.slash_exec import CommandContext, execute_command

    # Core version line is registry-owned (shared with the gateway /version);
    # the install/python/SDK detail below is CLI-only decoration.
    print(execute_command("version", CommandContext(surface="cli")).text)
    print(f"Install directory: {PROJECT_ROOT}")
    print(f"Install method: {detect_install_method(PROJECT_ROOT)}")

    # Show Python version
    print(f"Python: {sys.version.split()[0]}")

    # Check for key dependencies.  Use importlib.metadata rather than
    # ``import openai`` — the SDK drags in ~800ms of pydantic-backed type
    # modules just to expose ``__version__``.  Metadata lookup is ~2ms.
    try:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError

        try:
            print(f"OpenAI SDK: {_pkg_version('openai')}")
        except PackageNotFoundError:
            print("OpenAI SDK: Not installed")
    except ImportError:
        print("OpenAI SDK: Not installed")

    if not check_updates:
        return

    # Show update status (synchronous — acceptable since user asked for version info)
    try:
        from hermes_cli.banner import UPDATE_AVAILABLE_NO_COUNT, check_for_updates
        from hermes_cli.config import recommended_update_command

        behind = check_for_updates()
        if behind == UPDATE_AVAILABLE_NO_COUNT:
            print(
                f"Update available — run '{recommended_update_command()}'"
            )
        elif behind and behind > 0:
            commits_word = "commit" if behind == 1 else "commits"
            print(
                f"Update available: {behind} {commits_word} behind — "
                f"run '{recommended_update_command()}'"
            )
        elif behind == 0:
            print("Up to date")
    except Exception:
        pass


def cmd_version(args):
    """Show version."""
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
    run_env = {**os.environ, **(env or {}), "CI": "1"}

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
    build_env = with_hermes_node_path()
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
        subprocess.run([codesign, "--force", "--deep", "--sign", "-", str(app)], check=False)
    except Exception as exc:
        print(f"  (warning: macOS relaunch fixup skipped: {exc})")
    return False


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


def _desktop_launch_options() -> tuple[list[str], str, str]:
    """Read `desktop.*` launch options from config.yaml.

    Returns ``(electron_flags, disable_gpu, password_store)`` where
    ``electron_flags`` is a list of extra Electron CLI flags, ``disable_gpu``
    is one of "auto"/"1"/"0" (normalized for the HERMES_DESKTOP_DISABLE_GPU
    env var the Electron app reads), and ``password_store`` is "auto" or one
    of the Chromium password-store backends (unknown values normalize to
    "auto"). Best-effort: any config error yields the safe defaults
    ``([], "auto", "auto")`` so a malformed config never blocks the launch.
    """
    flags: list[str] = []
    disable_gpu = "auto"
    password_store = "auto"
    try:
        from hermes_cli.config import load_config

        desktop_cfg = (load_config() or {}).get("desktop") or {}
    except Exception:
        return flags, disable_gpu, password_store

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
    return flags, disable_gpu, password_store


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
    # `desktop.disable_gpu`). The GPU policy is bridged to the env var the
    # Electron app already reads; an explicit env var still wins over config so
    # `HERMES_DESKTOP_DISABLE_GPU=... hermes desktop` keeps working.
    config_electron_flags, config_disable_gpu, config_password_store = _desktop_launch_options()
    if config_disable_gpu != "auto" and "HERMES_DESKTOP_DISABLE_GPU" not in os.environ:
        env["HERMES_DESKTOP_DISABLE_GPU"] = config_disable_gpu

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
            if not source_mode:
                # A running desktop instance launched from release/win-unpacked
                # holds Hermes.exe locked on Windows, so the pack can't replace
                # it ("Access is denied" / ERR_ELECTRON_BUILDER_CANNOT_EXECUTE).
                # Stop it first so the rebuild — including the installer's
                # headless --update rebuild — succeeds instead of failing cryptically.
                stopped = _stop_desktop_processes_locking_build(desktop_dir)
                if stopped:
                    print(f"  ⚠ Stopped running desktop app to free the build output (pid {', '.join(map(str, stopped))})")
            build_result = subprocess.run([npm, "run", build_script], cwd=desktop_dir, env=env, check=False)
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
                    build_result = subprocess.run([npm, "run", build_script], cwd=desktop_dir, env=env, check=False)
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
                mirror_env = dict(env)
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


def _windows_running_hermes_launcher_locked() -> bool:
    """True when a venv ``hermes*.exe`` shim is this process or an ancestor.

    Best-effort: returns False when psutil is unavailable or inspection fails.
    """
    if not _is_windows():
        return False
    scripts_dir = _venv_scripts_dir()
    if scripts_dir is None:
        return False
    shims = _hermes_exe_shims(scripts_dir)
    if not shims:
        return False
    shim_set: set[str] = set()
    for shim in shims:
        try:
            shim_set.add(str(shim.resolve()).lower())
        except OSError:
            shim_set.add(str(shim).lower())
    try:
        import psutil

        me = psutil.Process()
        for proc in [me] + list(me.parents()):
            try:
                exe_norm = str(Path(proc.exe()).resolve()).lower()
            except Exception:
                continue
            if exe_norm in shim_set:
                return True
    except Exception:
        return False
    return False


def _default_venv_install_target() -> tuple[list[str], dict[str, str] | None]:
    """Return ``(install_cmd_prefix, env)`` for the project venv when possible."""
    try:
        from hermes_cli.managed_uv import ensure_uv

        uv_bin = ensure_uv()
    except Exception:
        uv_bin = None
    if uv_bin:
        env = {**os.environ, "VIRTUAL_ENV": str(PROJECT_ROOT / "venv")}
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
    venv_dir = PROJECT_ROOT / "venv"
    if not venv_dir.is_dir():
        return None
    from hermes_constants import venv_bin_dir

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
    scripts_dir: Path, *, max_attempts: int = 4
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
    2. If all retries fail, schedule the .exe for replacement on next
       reboot via ``MoveFileExW(MOVEFILE_DELAY_UNTIL_REBOOT)``. This still
       lets uv create a fresh shim at the original path (Windows will keep
       the old file's content under a new name until the reboot), so the
       update can complete; the user just needs to reboot to fully unload
       the stale image.
    3. Print a clear warning naming the most likely culprit (running
       Hermes Desktop / gateway / REPL) and pointing to ``--force``.

    Returns the list of (original, quarantined) pairs so the caller can roll
    back if the install itself fails before uv writes a replacement. Pairs
    where we used ``MOVEFILE_DELAY_UNTIL_REBOOT`` are NOT returned — they
    are already deferred and roll-back is meaningless.
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

        # All in-process renames failed. Try MoveFileEx with
        # MOVEFILE_DELAY_UNTIL_REBOOT as a last resort. This succeeds in the
        # exact case where the inline rename failed (another process holds
        # the handle without share-delete), at the cost of requiring a
        # reboot to fully reclaim the old .exe.
        scheduled = _schedule_replace_on_reboot(shim, target)
        if scheduled:
            print(
                f"  ⚠ {shim.name} is locked by another process; scheduled "
                f"replacement on next reboot."
            )
            print(
                "    The new shim was written at the same path, but a "
                "reboot is needed to fully unload the old one."
            )
            # Do NOT append to ``moved``: we don't want roll-back to undo a
            # reboot-deferred operation.
            continue

        # Truly couldn't budge the .exe. Print an actionable warning and let
        # uv try its luck — sometimes uv's own retry handling pulls through.
        print(
            f"  ⚠ Could not quarantine {shim.name} ({last_exc.__class__.__name__}: "
            f"another process is holding it open)."
        )
        print(
            "    Close Hermes Desktop, exit other `hermes` REPLs, stop the "
            "gateway, or pause AV scanning, then re-run `hermes update`."
        )

    return moved


def _schedule_replace_on_reboot(shim: Path, quarantine_target: Path) -> bool:
    """Schedule ``shim`` -> ``quarantine_target`` via PendingFileRenameOperations.

    Uses Win32 ``MoveFileExW`` with ``MOVEFILE_REPLACE_EXISTING |
    MOVEFILE_DELAY_UNTIL_REBOOT``. The OS persists the rename in
    ``HKLM\\System\\CurrentControlSet\\Control\\Session Manager\\
    PendingFileRenameOperations`` and applies it before any user-mode code
    runs on next boot — at which point no process can hold the .exe.

    Returns ``True`` if the schedule call succeeded, ``False`` otherwise
    (non-Windows, ctypes failure, lack of privilege, etc.). Never raises.
    """
    if not _is_windows():
        return False
    try:
        import ctypes
        from ctypes import wintypes

        MOVEFILE_REPLACE_EXISTING = 0x1
        MOVEFILE_DELAY_UNTIL_REBOOT = 0x4

        MoveFileExW = ctypes.windll.kernel32.MoveFileExW
        MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        MoveFileExW.restype = wintypes.BOOL

        ok = MoveFileExW(
            str(shim),
            str(quarantine_target),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_DELAY_UNTIL_REBOOT,
        )
        return bool(ok)
    except Exception:
        return False


def _restore_quarantined_exes(moved: list[tuple[Path, Path]]) -> None:
    """Roll back ``_quarantine_running_hermes_exe`` if uv didn't write replacements."""
    for original, quarantined in moved:
        try:
            if not original.exists() and quarantined.exists():
                quarantined.rename(original)
        except OSError:
            pass


def _run_quarantined_install(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    scripts_dir: Path | None = None,
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

    Off-Windows (``scripts_dir is None``) this is a thin pass-through.
    """
    moved: list[tuple[Path, Path]] = []
    if scripts_dir is not None:
        moved = _quarantine_running_hermes_exe(scripts_dir)
    try:
        _run_install_with_heartbeat(cmd, env=env)
    except BaseException:
        # Restore shims if pip/uv didn't write replacements (e.g. install
        # failed before the entry-points step). Don't swallow the error.
        if scripts_dir is not None:
            _restore_quarantined_exes(moved)
        raise


def _cleanup_quarantined_exes(scripts_dir: Path | None = None) -> None:
    """Sweep ``hermes.exe.old.*`` left by prior updates.

    Called early on every hermes invocation. The .old files are unlocked once
    their owning process exited, so deletion succeeds the next run. Silent
    no-op when nothing's there or on file-locked / permission errors.
    """
    if not _is_windows():
        return
    if scripts_dir is None:
        scripts_dir = _venv_scripts_dir()
    if scripts_dir is None:
        return
    try:
        for stale in scripts_dir.glob("*.exe.old.*"):
            try:
                stale.unlink()
            except OSError:
                pass  # still locked or in use — try again next run
    except OSError:
        pass


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
    """
    scripts_dir = _venv_scripts_dir() if _is_windows() else None

    def _install(args: list[str]) -> None:
        _run_quarantined_install(
            install_cmd_prefix + args, env=env, scripts_dir=scripts_dir
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
        detect_install_method,
        format_docker_update_message,
        is_managed,
        managed_error,
        recommended_update_command_for_method,
    )

    if is_managed():
        managed_error("update Hermes Agent")
        return

    # Docker users can't ``git pull`` — the image excludes ``.git`` from
    # the build context.  Bail with a friendly explanation pointing at
    # ``docker pull`` BEFORE any of the apply-path / check-path branches
    # below get a chance to error out with misleading "Not a git
    # repository" text.  See format_docker_update_message() for the full
    # rationale and tag-pinning / config-persistence notes.
    install_method = detect_install_method(PROJECT_ROOT)
    if install_method == "docker":
        print(format_docker_update_message())
        sys.exit(1)

    if install_method in {"nix", "nixos"}:
        print(recommended_update_command_for_method(install_method))
        sys.exit(1)

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

    try:
        _self()._cmd_update_impl(args, gateway_mode=gateway_mode)
    finally:
        _update_lock.release()
        _finalize_update_output(_update_io_state)


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
        "version",
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
        profile_name = get_active_profile_name()
        dhh = display_hermes_home()
        print(f"\nActive profile: {profile_name}")
        print(f"Path:           {dhh}")

        profiles = list_profiles()
        for p in profiles:
            if p.name == profile_name or (profile_name == "default" and p.is_default):
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
                break
        print()
        return

    if action == "list":
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
            name = p.name
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
                profile_dir_display = "~/" + str(profile_dir.relative_to(Path.home()))
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

        print(f"\nProfile: {name}")
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
        from hermes_cli.profiles import rename_profile

        try:
            new_dir = rename_profile(args.old_name, args.new_name)
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
    """Print live listening dashboard processes and return the count."""
    from gateway.status import _pid_exists

    live: list[tuple[int, str]] = []
    for pid, command in _self()._scan_dashboard_processes():
        runtime = _parse_dashboard_runtime(command)
        if runtime is None:
            continue
        mode, host, port = runtime
        if mode != "dashboard":
            continue
        if port <= 0 or not _pid_exists(pid):
            continue
        if not _dashboard_listening(host, port):
            continue
        live.append((pid, command))

    if not live:
        print("No hermes dashboard processes running.")
        return 0

    print(f"{len(live)} hermes dashboard process(es) running:")
    for pid, command in live:
        print(f"    PID {pid}: {command}")
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
    """Offer to configure dashboard auth when a non-loopback bind has none.

    Called from ``cmd_dashboard`` just before ``start_server``. The auth
    gate engages on every non-loopback bind (``--insecure`` is a no-op since
    the June 2026 hardening), and ``start_server`` fails closed when no
    ``DashboardAuthProvider`` is registered. Rather than greet an interactive
    operator with that hard error, prompt them to set up the bundled
    username/password provider on the spot — or point them at
    ``hermes dashboard register`` for OAuth.

    No-ops (so the existing fail-closed ``SystemExit`` remains the backstop)
    when:
      * the bind is loopback (gate never engages), or
      * a provider is already registered, or
      * stdin/stdout isn't a TTY (Docker/s6, CI, piped ``--no-open`` runs).
    """
    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"

    try:
        from hermes_cli.web_server import should_require_auth
        if not should_require_auth(host):
            return  # loopback bind — gate never engages
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
    print(
        f"⚠ The dashboard is binding to a non-loopback address ({host}) and "
        f"needs an auth provider."
    )
    print(
        "  Non-loopback binds always require authentication "
        "(--insecure no longer bypasses this)."
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
        username = input("  Username [admin]: ").strip() or "admin"
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
        "model", "monitoring", "pairing", "pause", "pets", "plugins", "portal", "profile",
        "project", "proxy",
        "prompt-size",
        "resume",
        "send", "sessions", "setup",
        "skin", "skills", "slack", "status", "sync", "tools", "uninstall", "update",
        "version", "webhook", "whatsapp", "whatsapp-cloud", "chat", "secrets", "security",
        "verify",
        # Help-ish invocations — plugin commands not being listed in
        # top-level --help is an acceptable trade-off for skipping an
        # expensive eager import of every bundled plugin module.
        "help",
    }
)


# Top-level flags that take a value. Needed by ``_first_positional_argv``
# so that in ``hermes -m gpt5 chat``, ``gpt5`` is correctly skipped as a
# flag value rather than misclassified as a subcommand. Kept in sync with
# the top-level flags declared in ``hermes_cli/_parser.py``.
#
# Correctness-safe either way: missing an entry here only makes the
# fast-path bail out too eagerly (we run plugin discovery when we didn't
# need to); extra entries would make us skip a real positional.
_TOP_LEVEL_VALUE_FLAGS = frozenset(
    {
        "-z", "--oneshot",
        "-m", "--model",
        "--provider",
        "-t", "--toolsets",
        "-r", "--resume",
        "-s", "--skills",
        "--usage-file",
        "--in",
        # ``-c / --continue`` is nargs='?' (optional value). Treat it as
        # value-taking: if the next token is a subcommand-looking word
        # the user almost certainly meant it as the session name, and
        # either interpretation keeps us on the safe side.
        "-c", "--continue",
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
            if tok in _TOP_LEVEL_VALUE_FLAGS and i + 1 < len(argv):
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
        _print_version_info(check_updates=False)
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
        _print_version_info(check_updates=False)
        return True

    if getattr(args, "oneshot", None):
        _prepare_agent_startup(args)
        _confirm_startup_expensive_model_override(args)
        _run_and_exit_oneshot(
            args.oneshot,
            model=getattr(args, "model", None),
            provider=getattr(args, "provider", None),
            toolsets=getattr(args, "toolsets", None),
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


def cmd_memory(args):
    sub = getattr(args, "memory_command", None)
    if sub == "off":
        from hermes_cli.config import load_config, save_config

        config = load_config()
        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = ""
        save_config(config)
        print("\n  ✓ Memory provider: built-in only")
        print("  Saved to config.yaml\n")
    elif sub == "reset":
        from hermes_constants import get_hermes_home, display_hermes_home

        mem_dir = get_hermes_home() / "memories"
        target = getattr(args, "target", "all")
        files_to_reset = []
        if target in {"all", "memory"}:
            files_to_reset.append(("MEMORY.md", "agent notes"))
        if target in {"all", "user"}:
            files_to_reset.append(("USER.md", "user profile"))

        # Check what exists
        existing = [
            (f, desc) for f, desc in files_to_reset if (mem_dir / f).exists()
        ]
        if not existing:
            print(
                f"\n  Nothing to reset — no memory files found in {display_hermes_home()}/memories/\n"
            )
            return

        print("\n  This will permanently erase the following memory files:")
        for f, desc in existing:
            path = mem_dir / f
            size = path.stat().st_size
            print(f"    ◆ {f} ({desc}) — {size:,} bytes")

        if not getattr(args, "yes", False):
            try:
                answer = input("\n  Type 'yes' to confirm: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.\n")
                return
            if answer != "yes":
                print("  Cancelled.\n")
                return

        for f, desc in existing:
            (mem_dir / f).unlink()
            print(f"  ✓ Deleted {f} ({desc})")

        print(
            "\n  Memory reset complete. New sessions will start with a blank slate."
        )
        print(f"  Files were in: {display_hermes_home()}/memories/\n")
    else:
        from hermes_cli.memory_setup import memory_command

        memory_command(args)


def cmd_acp(args):
    """Launch Hermes Agent as an ACP server."""
    try:
        from acp_adapter.entry import main as acp_main

        acp_argv = []
        if getattr(args, "acp_version", False):
            acp_argv.append("--version")
        if getattr(args, "check", False):
            acp_argv.append("--check")
        if getattr(args, "setup", False):
            acp_argv.append("--setup")
        if getattr(args, "setup_browser", False):
            acp_argv.append("--setup-browser")
        if getattr(args, "assume_yes", False):
            acp_argv.append("--yes")
        acp_main(acp_argv)
    except ImportError:
        print("ACP dependencies not installed.", file=sys.stderr)
        print("Install them with:  pip install -e '.[acp]'", file=sys.stderr)
        sys.exit(1)


def cmd_tools(args):
    action = getattr(args, "tools_action", None)
    if action in {"list", "disable", "enable"}:
        from hermes_cli.tools_config import tools_disable_enable_command

        tools_disable_enable_command(args)
    elif action == "post-setup":
        from hermes_cli.tools_config import run_post_setup_command

        sys.exit(run_post_setup_command(args))
    else:
        _require_tty("tools")
        from hermes_cli.tools_config import tools_command

        tools_command(args)


def cmd_insights(args):
    db = None
    try:
        from hermes_state import SessionDB
        from agent.insights import InsightsEngine

        db = SessionDB()
        engine = InsightsEngine(db)
        report = engine.generate(days=args.days, source=args.source)
        print(engine.format_terminal(report))
    except Exception as e:
        print(f"Error generating insights: {e}")
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def cmd_monitoring(args):
    """Gateway monitoring status: health & diagnostics export posture."""
    from hermes_cli.config import load_config

    action = getattr(args, "monitoring_action", None) or "status"
    config = load_config()
    mon_raw = config.get("monitoring")
    mon: dict = mon_raw if isinstance(mon_raw, dict) else {}

    if action == "status":
        from agent.monitoring import otlp_exporter

        gh_raw = mon.get("gateway_health_export")
        gh: dict = gh_raw if isinstance(gh_raw, dict) else {}
        export_raw = mon.get("export")
        export_cfg: dict = export_raw if isinstance(export_raw, dict) else {}
        otlp_raw = export_cfg.get("otlp")
        otlp: dict = otlp_raw if isinstance(otlp_raw, dict) else {}

        print("Gateway monitoring")
        print(f"  Health export:  {'enabled' if gh.get('enabled') else 'disabled'} "
              f"(monitoring.gateway_health_export.enabled)")
        if gh.get("enabled"):
            print(f"    Metrics:            {'on' if gh.get('metrics_enabled', True) else 'off'} "
                  f"(interval {gh.get('export_interval_seconds', 60)}s)")
            print(f"    Diagnostic events:  {'on' if gh.get('diagnostic_events_enabled', True) else 'off'}")
            print(f"    Warning/error logs: {'on' if gh.get('warning_error_events_enabled', True) else 'off'} "
                  f"(interval {gh.get('logs_export_interval_seconds', 5)}s)")
            print("    Content safety:     always on "
                  "(rendered messages are never exported; not configurable)")
        endpoint = otlp.get("endpoint") or ""
        if otlp.get("enabled") and endpoint:
            print(f"  OTLP endpoint:  {endpoint}")
        else:
            print("  OTLP endpoint:  not configured (monitoring.export.otlp)")
        print(f"  OTel SDK:       {'installed' if otlp_exporter.is_available() else 'not installed'} "
              f"(optional extra: hermes-agent[otlp])")
        print("\n  Scope: gateway service health + redacted diagnostics only.")
        print("  No prompts, messages, tool args/results, usage analytics, or traces.")
        return

    print(f"Unknown monitoring action: {action}", file=sys.stderr)
    sys.exit(2)


def cmd_skills(args):
    # Route 'config' action to skills_config module
    if getattr(args, "skills_action", None) == "config":
        _require_tty("skills config")
        from hermes_cli.skills_config import skills_command as skills_config_command

        skills_config_command(args)
    else:
        from hermes_cli.skills_hub import skills_command

        skills_command(args)


def cmd_pairing(args):
    from hermes_cli.pairing import pairing_command

    pairing_command(args)


def cmd_plugins(args):
    from hermes_cli.plugins_cmd import plugins_command

    plugins_command(args)


def cmd_mcp(args):
    from hermes_cli.mcp_config import mcp_command

    mcp_command(args)


def cmd_claw(args):
    from hermes_cli.claw import claw_command

    claw_command(args)


def _advertise_agent_env() -> None:
    """Advertise the agent harness to child processes.

    ``AI_AGENT`` is the emerging cross-agent standard (huggingface_hub's agent
    detection reads it; pi and other agents set it — earendil-works/pi#7493)
    so generic tooling can attribute subprocesses to the harness that spawned
    them. The value must be our id in the public agent-harness registry
    (``hermes-agent`` in huggingface.js ``agent-harnesses.ts``): standard-var
    matching is exact, so any other value is counted as "unknown".
    ``HERMES_AGENT`` is the Hermes-specific marker. setdefault: never
    clobber an outer harness (e.g. Hermes running inside another agent's
    terminal).
    """
    os.environ.setdefault("AI_AGENT", "hermes-agent")
    os.environ.setdefault("HERMES_AGENT", "true")


def main():
    """Main entry point for hermes CLI."""
    # Cosmetic: make the process show up as 'hermes' instead of 'python3.11'
    # in ps/top/htop.  Non-fatal — just a nicer UX.
    _set_process_title()

    # Let child processes (and tools like huggingface_hub) detect they run
    # under an AI agent harness.
    _advertise_agent_env()

    # Force UTF-8 stdio on Windows before anything prints.  No-op elsewhere.
    try:
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()
    except Exception:
        pass

    # Sweep stale ``hermes.exe.old.*`` quarantine files left by previous
    # ``hermes update`` runs on Windows. Silent no-op on non-Windows or when
    # there's nothing to clean. See ``_quarantine_running_hermes_exe``.
    try:
        _cleanup_quarantined_exes()
    except Exception:
        pass

    # If the checkout changed since the last launch (hermes update, manual
    # git pull, old-updater update that predates newer clears), sweep stale
    # __pycache__ once so no process — this one's lazy imports included —
    # resolves fresh source against old bytecode. Never raises.
    _sweep_stale_bytecode_if_checkout_changed()

    # Self-heal a venv left half-built by an interrupted ``hermes update``
    # (Ctrl-C, terminal close, WSL OOM mid-install). Skip when the user is
    # *running* update — that flow writes and clears its own marker, and we
    # don't want a recovery install racing the real one. Never raises.
    #
    # The substring match is deliberately loose: argv isn't parsed yet at this
    # point, and the failure modes are asymmetric. Over-matching (e.g.
    # ``hermes skills install update``) merely defers recovery one launch;
    # under-matching (missing ``hermes -p work update``) would race a recovery
    # install against the real one. Loose wins.
    try:
        if "update" not in sys.argv[1:]:
            _recover_from_interrupted_install()
    except Exception:
        pass

    if _try_termux_fast_tui_launch():
        return
    if _try_termux_fast_cli_launch():
        return
    if _try_fast_chat_launch():
        return

    from hermes_cli._parser import build_top_level_parser

    parser, subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)

    # =========================================================================
    # model command  (parser built in hermes_cli/subcommands/model.py)
    # =========================================================================
    build_model_parser(subparsers, cmd_model=cmd_model)

    from hermes_cli.moa_cmd import cmd_moa

    moa_parser = subparsers.add_parser(
        "moa",
        help="Configure Mixture of Agents provider/model slots",
        description="Configure the provider/model set used by /moa <prompt>.",
    )
    moa_subparsers = moa_parser.add_subparsers(dest="moa_command")
    moa_subparsers.add_parser("list", aliases=["ls"], help="Show current MoA model slots")
    moa_configure = moa_subparsers.add_parser("configure", aliases=["config"], help="Interactively pick MoA models")
    moa_configure.add_argument("name", nargs="?", help="Preset name to create or update")
    moa_delete = moa_subparsers.add_parser("delete", aliases=["rm"], help="Delete a MoA preset")
    moa_delete.add_argument("name", help="Preset name to delete")
    moa_parser.set_defaults(func=cmd_moa)

    # =========================================================================
    # fallback command — manage the fallback provider chain
    # =========================================================================
    from hermes_cli.fallback_cmd import cmd_fallback

    fallback_parser = subparsers.add_parser(
        "fallback",
        help="Manage fallback providers (tried when the primary model fails)",
        description=(
            "Manage the fallback provider chain.  Fallback providers are tried "
            "in order when the primary model fails with rate-limit, overload, or "
            "connection errors.  See: "
            "https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers"
        ),
    )
    fallback_subparsers = fallback_parser.add_subparsers(dest="fallback_command")
    fallback_subparsers.add_parser(
        "list",
        aliases=["ls"],
        help="Show the current fallback chain (default when no subcommand)",
    )
    fallback_subparsers.add_parser(
        "add",
        help="Pick a provider + model (same picker as `hermes model`) and append to the chain",
    )
    fallback_subparsers.add_parser(
        "remove",
        aliases=["rm"],
        help="Pick an entry to delete from the chain",
    )
    fallback_subparsers.add_parser(
        "clear",
        help="Remove all fallback entries",
    )
    fallback_parser.set_defaults(func=cmd_fallback)

    # =========================================================================
    # secrets command — external secret managers (Bitwarden, 1Password)
    # =========================================================================
    secrets_parser = subparsers.add_parser(
        "secrets",
        help="Manage external secret sources (Bitwarden, 1Password)",
        description=(
            "Pull API keys from an external secret manager at process startup "
            "instead of storing them in ~/.hermes/.env.  Supports Bitwarden "
            "Secrets Manager and 1Password.  See: "
            "https://hermes-agent.nousresearch.com/docs/user-guide/secrets/"
        ),
    )
    secrets_subparsers = secrets_parser.add_subparsers(dest="secrets_command")

    secrets_bw = secrets_subparsers.add_parser(
        "bitwarden",
        aliases=["bw"],
        help="Bitwarden Secrets Manager integration",
    )

    secrets_op = secrets_subparsers.add_parser(
        "onepassword",
        aliases=["op", "1password"],
        help="1Password (op:// references) integration",
    )

    # The secrets_cli and onepassword_secrets_cli modules use lazy-imported
    # backends (agent.secret_sources.bitwarden / onepassword), so registering
    # their parsers here does NOT load cryptography._rust.pyd.  The imports
    # happen inside each cmd_* handler at first use.
    from hermes_cli import secrets_cli as _secrets_cli
    from hermes_cli import onepassword_secrets_cli as _op_secrets_cli

    _secrets_cli.register_cli(secrets_bw)
    _op_secrets_cli.register_cli(secrets_op)

    def _dispatch_secrets(args):  # noqa: ANN001
        sub = getattr(args, "secrets_command", None)
        if sub is None:
            secrets_parser.print_help()
            return 0
        return args.func(args)

    secrets_parser.set_defaults(func=_dispatch_secrets)

    # =========================================================================
    # egress command — iron-proxy outbound credential-injection firewall
    # =========================================================================
    # NOTE: this is the OUTBOUND egress firewall (ironsh/iron-proxy).
    # `hermes proxy` (defined elsewhere in this file) is a separate INBOUND
    # OAuth-aggregator reverse proxy.  Different direction, different purpose.
    egress_parser = subparsers.add_parser(
        "egress",
        help="Manage the iron-proxy egress credential-injection firewall",
        description=(
            "Manage iron-proxy, the optional TLS-intercepting egress firewall "
            "that swaps proxy tokens for real API credentials before outbound "
            "requests leave a sandbox.  Disabled by default.  See: "
            "https://hermes-agent.nousresearch.com/docs/user-guide/egress/iron-proxy"
        ),
    )

    from hermes_cli import proxy_cli as _proxy_cli
    _proxy_cli.register_cli(egress_parser)

    def _dispatch_egress(args):  # noqa: ANN001
        # The egress subparser uses dest='egress_command' to stay disjoint
        # from the inbound OAuth ``hermes proxy`` subparser (dest='proxy_command').
        sub = getattr(args, "egress_command", None)
        if sub is not None and hasattr(args, "func") and args.func is not _dispatch_egress:
            return args.func(args)
        egress_parser.print_help()
        return 0

    egress_parser.set_defaults(func=_dispatch_egress)

    # =========================================================================
    # migrate command
    # =========================================================================
    from hermes_cli.migrate import cmd_migrate, cmd_migrate_xai

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate configuration for retired models or deprecated settings",
        description=(
            "Diagnose and (optionally) rewrite the active config.yaml to "
            "replace references to retired models or deprecated settings."
        ),
    )
    migrate_subparsers = migrate_parser.add_subparsers(dest="migrate_type")

    migrate_xai = migrate_subparsers.add_parser(
        "xai",
        help="Migrate xAI models scheduled for retirement on May 15, 2026",
        description=(
            "Scan config.yaml for references to xAI models retiring on "
            "May 15, 2026 and, with --apply, rewrite them in-place to the "
            "official replacements per the xAI migration guide. The original "
            "config.yaml is backed up before any rewrite."
        ),
    )
    migrate_xai.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite config.yaml in-place (default: dry-run, no writes)",
    )
    migrate_xai.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the timestamped backup of config.yaml when applying",
    )
    migrate_xai.set_defaults(func=cmd_migrate_xai)
    migrate_parser.set_defaults(func=cmd_migrate)

    # =========================================================================
    # gateway + proxy commands  (parsers built in hermes_cli/subcommands/gateway.py)
    # =========================================================================
    build_gateway_parser(
        subparsers, cmd_gateway=cmd_gateway, cmd_proxy=cmd_proxy, cmd_gateway_enroll=cmd_gateway_enroll
    )

    # =========================================================================
    # lsp command
    # =========================================================================
    try:
        from agent.lsp.cli import register_subparser as _lsp_register
        _lsp_register(subparsers)
    except Exception as _lsp_err:  # noqa: BLE001
        # LSP is optional infrastructure — never let a registration
        # failure break the CLI overall.
        logger.debug("LSP CLI registration failed: %s", _lsp_err)

    # =========================================================================
    # setup command  (parser built in hermes_cli/subcommands/setup.py)
    # =========================================================================
    build_setup_parser(subparsers, cmd_setup=cmd_setup)


    # =========================================================================
    # whatsapp command  (parser built in hermes_cli/subcommands/whatsapp.py)
    # =========================================================================
    build_whatsapp_parser(subparsers, cmd_whatsapp=cmd_whatsapp)

    # =========================================================================
    # whatsapp-cloud command (official Meta Cloud API; complement to Baileys)
    # =========================================================================
    whatsapp_cloud_parser = subparsers.add_parser(
        "whatsapp-cloud",
        help="Set up WhatsApp Business Cloud API integration",
        description=(
            "Configure the official Meta WhatsApp Business Cloud API "
            "adapter (Business account required, public webhook URL "
            "required). Distinct from `hermes whatsapp` which sets up "
            "the Baileys bridge for personal accounts."
        ),
    )
    whatsapp_cloud_parser.set_defaults(func=cmd_whatsapp_cloud)

    # =========================================================================
    # slack command  (parser built in hermes_cli/subcommands/slack.py)
    # =========================================================================
    build_slack_parser(subparsers, cmd_slack=cmd_slack)

    # =========================================================================
    # send command — pipe shell-script output to any configured platform
    # =========================================================================
    from hermes_cli.send_cmd import register_send_subparser
    register_send_subparser(subparsers)

    # =========================================================================
    # login command  (parser built in hermes_cli/subcommands/login.py)
    # =========================================================================
    build_login_parser(subparsers, cmd_login=cmd_login)

    # =========================================================================
    # logout command  (parser built in hermes_cli/subcommands/logout.py)
    # =========================================================================
    build_logout_parser(subparsers, cmd_logout=cmd_logout)

    # =========================================================================
    # auth command  (parser built in hermes_cli/subcommands/auth.py)
    # =========================================================================
    build_auth_parser(subparsers, cmd_auth=cmd_auth)

    # =========================================================================
    # status command  (parser built in hermes_cli/subcommands/status.py)
    # =========================================================================
    build_status_parser(subparsers, cmd_status=cmd_status)

    # =========================================================================
    # pause / resume commands  (parser built in hermes_cli/subcommands/pause.py)
    # =========================================================================
    build_pause_parser(subparsers)

    # =========================================================================
    # cron command  (parser built in hermes_cli/subcommands/cron.py)
    # =========================================================================
    build_cron_parser(subparsers, cmd_cron=cmd_cron)
    build_sync_parser(subparsers, cmd_sync=cmd_sync)

    # =========================================================================
    # webhook command  (parser built in hermes_cli/subcommands/webhook.py)
    # =========================================================================
    build_webhook_parser(subparsers, cmd_webhook=cmd_webhook)

    # =========================================================================
    # portal command — Nous Portal status + Tool Gateway routing
    # =========================================================================
    from hermes_cli.portal_cli import add_parser as _add_portal_parser
    _add_portal_parser(subparsers)

    # =========================================================================
    # kanban command — multi-profile collaboration board
    # =========================================================================
    from hermes_cli.kanban import build_parser as _build_kanban_parser

    kanban_parser = _build_kanban_parser(subparsers)
    kanban_parser.set_defaults(func=cmd_kanban)

    # =========================================================================
    # project command — named, multi-folder workspaces
    # =========================================================================
    from hermes_cli.projects_cmd import build_parser as _build_project_parser

    project_parser = _build_project_parser(subparsers)
    project_parser.set_defaults(func=cmd_project)

    # =========================================================================
    # hooks command — shell-hook inspection and management
    # =========================================================================
    # hooks command  (parser built in hermes_cli/subcommands/hooks.py)
    # =========================================================================
    build_hooks_parser(subparsers, cmd_hooks=cmd_hooks)

    # =========================================================================
    # doctor command  (parser built in hermes_cli/subcommands/doctor.py)
    # =========================================================================
    build_doctor_parser(subparsers, cmd_doctor=cmd_doctor)

    # =========================================================================
    # verify command  (parser built in hermes_cli/subcommands/verify.py)
    # =========================================================================
    build_verify_parser(subparsers, cmd_verify=cmd_verify)

    # =========================================================================
    # security command — on-demand supply-chain audit
    # =========================================================================
    # security command  (parser built in hermes_cli/subcommands/security.py)
    # =========================================================================
    build_security_parser(subparsers, cmd_security=cmd_security)

    # =========================================================================
    # approvals command  (parser built in hermes_cli/subcommands/approvals.py)
    # =========================================================================
    build_approvals_parser(subparsers, cmd_approvals=cmd_approvals)

    # =========================================================================
    # dump command  (parser built in hermes_cli/subcommands/dump.py)
    # =========================================================================
    build_dump_parser(subparsers, cmd_dump=cmd_dump)

    # =========================================================================
    # debug command  (parser built in hermes_cli/subcommands/debug.py)
    # =========================================================================
    build_debug_parser(subparsers, cmd_debug=cmd_debug)

    # =========================================================================
    # backup command  (parser built in hermes_cli/subcommands/backup.py)
    # =========================================================================
    build_backup_parser(subparsers, cmd_backup=cmd_backup)

    # =========================================================================
    # checkpoints command
    # =========================================================================
    checkpoints_parser = subparsers.add_parser(
        "checkpoints",
        help="Inspect / prune / clear ~/.hermes/checkpoints/",
        description="Manage the filesystem checkpoint store — the shadow git "
        "repo hermes uses to snapshot working directories before "
        "write_file/patch/terminal calls. Lets you see how much "
        "space checkpoints occupy, force a prune, or wipe the base.",
    )
    from hermes_cli.checkpoints import register_cli as _register_checkpoints_cli
    _register_checkpoints_cli(checkpoints_parser)

    # =========================================================================
    # import command  (parser built in hermes_cli/subcommands/import_cmd.py)
    # =========================================================================
    build_import_cmd_parser(subparsers, cmd_import=cmd_import)

    # =========================================================================
    # import-agent command  (parser: hermes_cli/subcommands/import_agent.py)
    # =========================================================================
    def cmd_import_agent(args):
        from hermes_cli.agent_import import import_agent_command
        import_agent_command(args)

    build_import_agent_parser(subparsers, cmd_import_agent=cmd_import_agent)

    # =========================================================================
    # config command  (parser built in hermes_cli/subcommands/config.py)
    # =========================================================================
    build_config_parser(subparsers, cmd_config=cmd_config)

    # =========================================================================
    # skin command  (parser built in hermes_cli/subcommands/skin.py)
    # =========================================================================
    build_skin_parser(subparsers, cmd_skin=cmd_skin)

    # =========================================================================
    # console command  (parser built in hermes_cli/subcommands/console.py)
    # =========================================================================
    build_console_parser(subparsers, cmd_console=cmd_console)

    # =========================================================================
    # pairing command  (parser built in hermes_cli/subcommands/pairing.py)
    # =========================================================================
    build_pairing_parser(subparsers, cmd_pairing=cmd_pairing)

    # =========================================================================
    # skills command  (parser built in hermes_cli/subcommands/skills.py)
    # =========================================================================
    build_skills_parser(subparsers, cmd_skills=cmd_skills)

    # =========================================================================
    # bundles command — skill bundles (alias /<name> for multiple skills)
    # =========================================================================
    bundles_parser = subparsers.add_parser(
        "bundles",
        help="Create, list, and manage skill bundles (aliases for multiple skills)",
        description=(
            "Skill bundles let you load several skills under one slash "
            "command. `/<bundle>` from the CLI or gateway loads every "
            "referenced skill at once."
        ),
    )
    from hermes_cli.bundles import register_cli as _bundles_register, bundles_command
    _bundles_register(bundles_parser)
    bundles_parser.set_defaults(func=bundles_command)

    # =========================================================================
    # plugins command  (parser built in hermes_cli/subcommands/plugins.py)
    # =========================================================================
    build_plugins_parser(subparsers, cmd_plugins=cmd_plugins)

    # =========================================================================
    # Plugin CLI commands — dynamically registered by memory/general plugins.
    # Plugins provide a register_cli(subparser) function that builds their
    # own argparse tree.  No hardcoded plugin commands in main.py.
    #
    # Skipped when the invocation is already targeting a known built-in
    # subcommand — ``hermes --help``, ``hermes version``, ``hermes logs``,
    # etc.  This avoids eagerly importing every bundled plugin module
    # (google.cloud.pubsub_v1, aiohttp, grpc, PIL …) which costs
    # 500-650ms on typical installs.
    # =========================================================================
    if _plugin_cli_discovery_needed():
        try:
            from plugins.memory import discover_plugin_cli_commands
            from hermes_cli.plugins import discover_plugins, get_plugin_manager

            seen_plugin_commands = set()
            for cmd_info in discover_plugin_cli_commands():
                plugin_parser = subparsers.add_parser(
                    cmd_info["name"],
                    help=cmd_info["help"],
                    description=cmd_info.get("description", ""),
                    formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
                )
                cmd_info["setup_fn"](plugin_parser)
                if cmd_info.get("handler_fn") is not None:
                    plugin_parser.set_defaults(func=cmd_info["handler_fn"])
                seen_plugin_commands.add(cmd_info["name"])

            discover_plugins()
            # A bundled platform whose top-level CLI command is the one being
            # invoked is still only a deferred entry at this point; import it
            # so its register_cli_command side effect runs before we read
            # _cli_commands (issue #54678).
            _resolve_deferred_platform_cli_command(_first_positional_argv())
            for cmd_info in get_plugin_manager()._cli_commands.values():
                if cmd_info["name"] in seen_plugin_commands:
                    continue
                plugin_parser = subparsers.add_parser(
                    cmd_info["name"],
                    help=cmd_info["help"],
                    description=cmd_info.get("description", ""),
                    formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
                )
                cmd_info["setup_fn"](plugin_parser)
                if cmd_info.get("handler_fn") is not None:
                    plugin_parser.set_defaults(func=cmd_info["handler_fn"])
        except Exception as _exc:
            logging.getLogger(__name__).debug("Plugin CLI discovery failed: %s", _exc)

    # =========================================================================
    # curator command — background skill maintenance
    # =========================================================================
    curator_parser = subparsers.add_parser(
        "curator",
        help="Background skill maintenance (curator) — status, run, pause, pin",
        description=(
            "The curator is an auxiliary-model background task that "
            "periodically reviews agent-created skills, prunes stale ones, "
            "consolidates overlaps, and archives obsolete skills. "
            "Bundled and hub-installed skills are never touched. "
            "Archives are recoverable; auto-deletion never happens."
        ),
    )
    try:
        from hermes_cli.curator import register_cli as _register_curator_cli

        _register_curator_cli(curator_parser)
    except Exception as _exc:
        logging.getLogger(__name__).debug("curator CLI wiring failed: %s", _exc)

    # =========================================================================
    # pets command — petdex animated mascots (CLI / TUI / desktop display)
    # =========================================================================
    pets_parser = subparsers.add_parser(
        "pets",
        help="Browse, install, and select petdex animated pets",
        description=(
            "Petdex (https://github.com/crafter-station/petdex) is a public "
            "gallery of animated sprite pets for coding agents. Install one "
            "and Hermes shows it reacting to agent activity across the CLI, "
            "TUI, and desktop app."
        ),
    )
    try:
        from hermes_cli.pets import register_cli as _register_pets_cli

        _register_pets_cli(pets_parser)
    except Exception as _exc:
        logging.getLogger(__name__).debug("pets CLI wiring failed: %s", _exc)

    # =========================================================================
    # journey command — learned skills + memories over time, in the terminal
    # =========================================================================
    journey_parser = subparsers.add_parser(
        "journey",
        aliases=["learning", "memory-graph"],
        help="Timeline of learned skills + memories over time",
        description=(
            "A terminal rendition of the desktop Star Map / Memory Graph: a "
            "timeline bar chart of learned skills and memories over time "
            "(oldest at top, newest at bottom) plus a playable constellation "
            "scrubber. Mirrors the TUI `/journey` overlay and the desktop panel."
        ),
    )
    try:
        from hermes_cli.journey import register_cli as _register_journey_cli

        _register_journey_cli(journey_parser)
    except Exception as _exc:
        logging.getLogger(__name__).debug("journey CLI wiring failed: %s", _exc)

    # =========================================================================
    # memory command  (parser built in hermes_cli/subcommands/memory.py)
    # =========================================================================
    build_memory_parser(subparsers, cmd_memory=cmd_memory)

    # =========================================================================
    # tools command  (parser built in hermes_cli/subcommands/tools.py)
    # =========================================================================
    build_tools_parser(subparsers, cmd_tools=cmd_tools)

    # =========================================================================
    # computer-use command — manage Computer Use (cua-driver) on macOS
    # =========================================================================
    computer_use_parser = subparsers.add_parser(
        "computer-use",
        help="Manage the Computer Use (cua-driver) backend (macOS/Windows/Linux)",
        description=(
            "Install or check the cua-driver binary used by the\n"
            "`computer_use` toolset. Supported on macOS, Windows, and\n"
            "Linux.\n\n"
            "Use `hermes computer-use install` to fetch and run the\n"
            "upstream cua-driver installer. This is equivalent to the\n"
            "post-setup hook that `hermes tools` runs when you first\n"
            "enable the Computer Use toolset, and is a stable target\n"
            "for re-running the install if it didn't fire (e.g. when\n"
            "toggling the toolset on a returning-user setup).\n\n"
            "Use `hermes computer-use doctor` to run cua-driver's\n"
            "`health_report` MCP tool and surface its check matrix\n"
            "(TCC, bundle identity, version, platform support, ...)\n"
            "in human-readable form."
        ),
    )
    computer_use_sub = computer_use_parser.add_subparsers(dest="computer_use_action")

    computer_use_install = computer_use_sub.add_parser(
        "install",
        help="Install or repair the cua-driver binary (macOS/Windows/Linux)",
    )
    computer_use_install.add_argument(
        "--upgrade",
        action="store_true",
        help=(
            "Re-run the upstream installer even if cua-driver is already on "
            "PATH. The upstream install.sh always pulls the latest release, "
            "so this performs an in-place upgrade."
        ),
    )
    computer_use_sub.add_parser(
        "status",
        help="Print whether cua-driver is installed and on PATH",
    )
    computer_use_doctor = computer_use_sub.add_parser(
        "doctor",
        help="Run cua-driver `health_report` and surface the check matrix",
        description=(
            "Drive cua-driver's stable `health_report` MCP tool and render\n"
            "its check matrix (TCC permissions, bundle identity, version,\n"
            "platform support, screenshot probe, …) as human-readable\n"
            "output. cua-driver owns the health model; this command stays\n"
            "thin so new checks added upstream surface here without code\n"
            "changes. Exits 0 when overall=ok, 1 when degraded/failed, 2\n"
            "when the binary is missing or unreachable."
        ),
    )
    computer_use_doctor.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="CHECK",
        help=(
            "Run only the listed checks. Repeat for multiple "
            "(e.g. --include tcc_accessibility --include bundle_identity). "
            "Unknown names are reported by cua-driver."
        ),
    )
    computer_use_doctor.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="CHECK",
        help="Skip the listed checks. Repeat for multiple. Wins over --include.",
    )
    computer_use_doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw structured payload as JSON (same shape as `tools/call`).",
    )
    computer_use_perms = computer_use_sub.add_parser(
        "permissions",
        help="Check or grant macOS Accessibility + Screen Recording (macOS)",
        description=(
            "Computer Use drives the Mac through cua-driver, whose TCC grants\n"
            "attach to cua-driver's own identity (com.trycua.driver) — not the\n"
            "terminal or the Hermes app. `status` reports the driver's grant\n"
            "state; `grant` launches CuaDriver via LaunchServices so the macOS\n"
            "permission dialog is attributed to the process that does the work."
        ),
    )
    computer_use_perms_sub = computer_use_perms.add_subparsers(
        dest="computer_use_perms_action"
    )
    computer_use_perms_status = computer_use_perms_sub.add_parser(
        "status",
        help="Report Accessibility + Screen Recording grant state (read-only)",
    )
    computer_use_perms_status.add_argument(
        "--json",
        action="store_true",
        help="Emit the normalized permission payload as JSON.",
    )
    computer_use_perms_sub.add_parser(
        "grant",
        help="Request the grants (opens the dialog attributed to CuaDriver)",
    )

    def cmd_computer_use(args):
        action = getattr(args, "computer_use_action", None)
        if action == "install":
            from hermes_cli.tools_config import install_cua_driver
            install_cua_driver(upgrade=bool(getattr(args, "upgrade", False)))
            return
        if action == "status":
            import subprocess
            from tools.computer_use.cua_backend import (
                cua_driver_update_check,
                resolve_cua_driver_cmd,
            )
            # Must match the runtime resolver: Desktop/TUI processes can omit
            # ~/.local/bin even though the official installer put the driver there.
            path = resolve_cua_driver_cmd()
            if path:
                version = ""
                try:
                    from hermes_cli.tools_config import _cua_driver_env
                    version = subprocess.run(
                        [path, "--version"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
                        env=_cua_driver_env(),
                    ).stdout.strip()
                except Exception:
                    pass
                if version:
                    print(f"cua-driver: installed at {path} ({version})")
                else:
                    print(f"cua-driver: installed at {path}")
                try:
                    st = cua_driver_update_check()
                    if st and st.get("update_available"):
                        latest = st.get("latest_version") or "?"
                        print(f"  ⬆ Update available: cua-driver {latest}.")
                        print("    Run: hermes computer-use install --upgrade")
                    elif st:
                        print("  ✓ Up to date.")
                    else:
                        # Older driver (no check-update verb) or offline.
                        print("  Refresh to latest: hermes computer-use install --upgrade")
                except Exception:
                    print("  Refresh to latest: hermes computer-use install --upgrade")
                return
            print("cua-driver: not installed")
            print("  Run: hermes computer-use install")
            return
        if action == "doctor":
            from tools.computer_use.doctor import run_doctor
            code = run_doctor(
                include=list(getattr(args, "include", []) or []),
                skip=list(getattr(args, "skip", []) or []),
                json_output=bool(getattr(args, "json", False)),
            )
            sys.exit(code)
        if action == "permissions":
            perms_action = getattr(args, "computer_use_perms_action", None)
            if perms_action == "grant":
                from tools.computer_use.permissions import request_permissions_grant
                sys.exit(request_permissions_grant())
            if perms_action == "status":
                import json as _json
                from tools.computer_use.permissions import computer_use_status
                st = computer_use_status()
                if bool(getattr(args, "json", False)):
                    print(_json.dumps(st, indent=2, sort_keys=True))
                    sys.exit(0 if st["ready"] else 1)
                if not st["platform_supported"]:
                    print(f"Computer Use is not supported on {st['platform']}.")
                    sys.exit(1)
                if not st["installed"]:
                    print("cua-driver: not installed. Run: hermes computer-use install")
                    sys.exit(1)
                glyph = lambda v: "✅" if v is True else ("❌" if v is False else "•")  # noqa: E731
                print(f"cua-driver: {st['version'] or 'installed'} ({st['platform']})")
                if st["can_grant"]:  # macOS TCC permissions
                    print(f"  {glyph(st['accessibility'])} Accessibility")
                    print(f"  {glyph(st['screen_recording'])} Screen Recording")
                    if not st["ready"]:
                        print("  Grant: hermes computer-use permissions grant")
                else:  # no TCC model — readiness is driver health
                    print(f"  {glyph(st['ready'])} driver health (no permission toggles on {st['platform']})")
                for c in st["checks"]:
                    if c["status"] != "ok":
                        print(f"  ⚠ {c['label']}: {c['message']}")
                if st["error"]:
                    print(f"  ⚠ {st['error']}")
                sys.exit(0 if st["ready"] else 1)
            computer_use_perms.print_help()
            return
        # No subcommand → show help
        computer_use_parser.print_help()

    computer_use_parser.set_defaults(func=cmd_computer_use)
    # =========================================================================
    # mcp command  (parser built in hermes_cli/subcommands/mcp.py)
    # =========================================================================
    build_mcp_parser(subparsers, cmd_mcp=cmd_mcp)

    # =========================================================================
    # sessions command
    # =========================================================================
    sessions_parser = subparsers.add_parser(
        "sessions",
        help="Manage session history (list, rename, export, prune, delete)",
        description="View and manage the SQLite session store",
    )
    sessions_subparsers = sessions_parser.add_subparsers(dest="sessions_action")

    sessions_list = sessions_subparsers.add_parser("list", help="List recent sessions")
    sessions_list.add_argument(
        "--source", help="Filter by source (cli, telegram, discord, etc.)"
    )
    sessions_list.add_argument(
        "--limit", type=int, default=20, help="Max sessions to show"
    )
    sessions_list.add_argument(
        "--workspace",
        metavar="NEEDLE",
        help="Only sessions in one workspace: a git repo root or project dir "
        "(matched by path substring or basename).",
    )

    def _add_session_filter_args(p, default_older_help):
        p.add_argument(
            "--older-than",
            metavar="AGE",
            help=default_older_help,
        )
        p.add_argument(
            "--newer-than",
            metavar="AGE",
            help="Only match sessions active within the last AGE "
            "(e.g. '5h', '2d') or after an ISO timestamp",
        )
        p.add_argument(
            "--before",
            metavar="TIME",
            help="Only match sessions started before TIME "
            "(duration ago like '5h', or ISO timestamp like '2026-07-05 14:30')",
        )
        p.add_argument(
            "--after",
            metavar="TIME",
            help="Only match sessions started at/after TIME "
            "(duration ago like '5h', or ISO timestamp)",
        )
        p.add_argument("--source", help="Only match sessions from this source")
        p.add_argument(
            "--title", help="Only match sessions whose title contains this substring"
        )
        p.add_argument(
            "--end-reason", help="Only match sessions with this end reason"
        )
        p.add_argument(
            "--cwd", help="Only match sessions whose working directory is under this path"
        )
        p.add_argument(
            "--min-messages", type=int, help="Only match sessions with >= N messages"
        )
        p.add_argument(
            "--max-messages", type=int, help="Only match sessions with <= N messages"
        )
        p.add_argument(
            "--model",
            help="Only match sessions whose model name contains this substring "
            "(e.g. 'sonnet', 'gpt-5', 'hermes')",
        )
        p.add_argument(
            "--provider",
            help="Only match sessions billed through this provider "
            "(e.g. openrouter, anthropic, nous)",
        )
        p.add_argument(
            "--user", help="Only match sessions from this user ID"
        )
        p.add_argument(
            "--chat-id", help="Only match sessions from this chat/channel ID"
        )
        p.add_argument(
            "--chat-type",
            help="Only match sessions with this chat type (e.g. dm, group)",
        )
        p.add_argument(
            "--branch",
            help="Only match sessions whose git branch contains this substring",
        )
        p.add_argument(
            "--min-tokens", type=int,
            help="Only match sessions with >= N total tokens (input+output)",
        )
        p.add_argument(
            "--max-tokens", type=int,
            help="Only match sessions with <= N total tokens (input+output)",
        )
        p.add_argument(
            "--min-cost", type=float,
            help="Only match sessions costing >= N USD (actual or estimated)",
        )
        p.add_argument(
            "--max-cost", type=float,
            help="Only match sessions costing <= N USD (actual or estimated)",
        )
        p.add_argument(
            "--min-tool-calls", type=int,
            help="Only match sessions with >= N tool calls",
        )
        p.add_argument(
            "--max-tool-calls", type=int,
            help="Only match sessions with <= N tool calls",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="List matching sessions without changing anything",
        )
        p.add_argument(
            "--yes", "-y", action="store_true", help="Skip confirmation"
        )

    sessions_export = sessions_subparsers.add_parser(
        "export", help="Export sessions to JSONL, Markdown, or QMD"
    )
    sessions_export.add_argument(
        "output",
        nargs="?",
        help=(
            "Output path. JSONL: file path (use - for stdout, required). "
            "md/qmd: output directory (default: <hermes home>/session-exports)"
        ),
    )
    sessions_export.add_argument(
        "--format",
        choices=["jsonl", "md", "qmd", "html", "trace"],
        default="jsonl",
        help=(
            "Export format (default: jsonl). 'trace' emits Claude Code JSONL "
            "for the Hugging Face Agent Trace Viewer"
        ),
    )
    sessions_export.add_argument(
        "--upload",
        action="store_true",
        help=(
            "trace only: upload to your Hugging Face traces dataset instead "
            "of writing a local file (needs HF_TOKEN)"
        ),
    )
    sessions_export.add_argument(
        "--public",
        action="store_true",
        help="trace --upload only: create/update a public dataset instead of private",
    )
    sessions_export.add_argument(
        "--no-redact",
        action="store_true",
        help=(
            "trace only: skip the forced secret redaction; "
            "only use after manual review"
        ),
    )
    sessions_export.add_argument(
        "--only",
        choices=["user-prompts"],
        help=(
            "Export only a filtered view (user-prompts: one prompt record "
            "per line for jsonl, headed sections for md)"
        ),
    )
    sessions_export.add_argument(
        "--session-id", help="Session ID or unique prefix to export"
    )
    _add_session_filter_args(
        sessions_export,
        "Only export sessions older than AGE (duration like '5h'/'2d', "
        "bare number of days, or an ISO timestamp)",
    )
    sessions_export.add_argument(
        "--redact",
        action="store_true",
        help="Redact secrets (API keys, tokens, credentials) from exported content",
    )
    sessions_export.add_argument(
        "--lineage",
        choices=["single", "logical"],
        default="single",
        help="md/qmd only: export one row or its compression lineage",
    )
    sessions_export.add_argument(
        "--delete-after-verified",
        action="store_true",
        help="md/qmd only: after verified single-session export, delete that session (needs --yes)",
    )
    sessions_export.add_argument(
        "--force",
        action="store_true",
        help="md/qmd only: overwrite an existing export file",
    )

    sessions_delete = sessions_subparsers.add_parser(
        "delete", help="Delete a specific session"
    )
    sessions_delete.add_argument("session_id", help="Session ID to delete")
    sessions_delete.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation"
    )

    sessions_prune = sessions_subparsers.add_parser(
        "prune",
        help="Delete old sessions (filterable by time window, source, title, ...)",
    )
    _add_session_filter_args(
        sessions_prune,
        "Delete sessions older than AGE — days if bare number, or a duration "
        "like '5h'/'2d'/'1w', or an ISO timestamp (bare prune with no filters "
        "defaults to 90 days; any filter matches all ages)",
    )
    sessions_prune.add_argument(
        "--include-archived",
        action="store_true",
        help="Also delete archived sessions (excluded by default)",
    )

    sessions_archive = sessions_subparsers.add_parser(
        "archive",
        help="Bulk-archive (soft-hide) sessions matching filters — no deletion",
    )
    _add_session_filter_args(
        sessions_archive,
        "Only archive sessions older than AGE (duration like '5h'/'2d', "
        "bare number of days, or ISO timestamp)",
    )

    sessions_subparsers.add_parser(
        "optimize",
        help="Reclaim disk space: merge FTS5 segments + VACUUM (no data change)",
    )

    sessions_clean_markers = sessions_subparsers.add_parser(
        "clean-markers",
        help="Permanently clear stale tool-call marker content left by sessions from before #78148",
        description=(
            "Before the #78148 fix, a local tool-call template could persist a "
            "bare bracketed marker (e.g. \"[memory]\") as an assistant turn's "
            "content instead of real text. This is already repaired in memory "
            "on every session load, so running this is optional — it rewrites "
            "the affected rows once, in place, so long-lived sessions stop "
            "re-scanning/re-repairing the same rows on every resume. Only the "
            "content column is touched; tool_calls and every other column on "
            "the row are left untouched."
        ),
    )
    sessions_clean_markers.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Report the affected row count without writing",
    )
    sessions_clean_markers.add_argument(
        "--no-backup",
        action="store_true",
        default=False,
        help="Skip the timestamped state.db backup taken before writing (not recommended)",
    )

    sessions_optimize_storage = sessions_subparsers.add_parser(
        "optimize-storage",
        help="Migrate the search index to the compact v23 layout (reclaims disk on large DBs)",
        description=(
            "Rebuild the full-text search index in the compact v23 "
            "external-content layout. On large databases this reclaims a "
            "large fraction of state.db (the old layout stored duplicate "
            "copies of every message and indexed tool output). Runs "
            "foreground with a progress bar, throttles so a running gateway "
            "stays responsive, and VACUUMs at the end. Safe to interrupt and "
            "re-run — it resumes where it left off. No conversation data is "
            "changed; only the search index is rebuilt."
        ),
    )
    sessions_optimize_storage.add_argument(
        "--no-vacuum",
        action="store_true",
        default=False,
        help="Skip the final VACUUM (index is rebuilt but freed pages aren't returned to the OS until a later VACUUM)",
    )
    sessions_optimize_storage.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip the disk-space confirmation prompt",
    )

    sessions_repair = sessions_subparsers.add_parser(
        "repair",
        help="Repair a malformed state.db schema so hidden sessions reappear",
        description=(
            "Recover a state.db whose schema is malformed (e.g. 'table "
            "messages_fts already exists'), which makes Desktop/Dashboard show "
            "no sessions. A backup is made first; sessions and messages are "
            "preserved and the FTS search index is rebuilt if needed."
        ),
    )
    sessions_repair.add_argument(
        "--check-only",
        action="store_true",
        help="Only report whether the database opens cleanly; do not modify it",
    )
    sessions_repair.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the timestamped backup copy (not recommended)",
    )

    sessions_repair_routing = sessions_subparsers.add_parser(
        "repair-routing",
        help="Re-stamp gateway sessions that lost their routing identity",
        description=(
            "Find gateway conversations stranded in session rows whose "
            "routing identity (session_key/chat_id/origin) was never "
            "written — the damage a corrupt state.db write path leaves "
            "behind (#82616). Such a row is invisible to restart recovery, "
            "so the chat resumes an older session instead. Re-stamps each "
            "orphan from the keyed predecessor it continues, and only when "
            "that predecessor is unambiguous. Reports without touching the "
            "database unless --apply is given."
        ),
    )
    sessions_repair_routing.add_argument(
        "--apply",
        action="store_true",
        help="Perform the adoptions (default: report only)",
    )
    sessions_repair_routing.add_argument(
        "--max-gap-seconds",
        type=float,
        default=None,
        help=(
            "Window between a keyed predecessor's last activity and an "
            "orphan's start for them to count as the same conversation "
            "(default: 900)"
        ),
    )

    sessions_recover = sessions_subparsers.add_parser(
        "recover",
        help="Rebuild canonical session data into a separate clean database",
        description=(
            "Offline, non-destructive recovery for a damaged state.db. The "
            "source database and its WAL/SHM/rollback-journal sidecars are "
            "copied before SQLite opens anything. Canonical rows are rebuilt "
            "into a new output database; derived search indexes are recreated "
            "and the active database is never replaced automatically."
        ),
    )
    sessions_recover.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source state.db or preserved backup to inspect/recover",
    )
    sessions_recover.add_argument(
        "--output",
        type=Path,
        help="New recovery database path (required unless --inspect-only)",
    )
    sessions_recover.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only report canonical table readability; do not create an output database",
    )
    sessions_recover.add_argument(
        "--work-dir",
        type=Path,
        help="Existing directory for the disposable source copy (defaults beside the output)",
    )
    sessions_recover.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Rows committed per recovery batch (default: 1000)",
    )
    sessions_recover.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Best-effort salvage across damaged row ranges; the output remains "
            "separate and every skipped range is recorded"
        ),
    )
    sessions_recover.add_argument(
        "--report",
        type=Path,
        help="JSON report path (defaults to <output>.recovery.json)",
    )

    sessions_subparsers.add_parser("stats", help="Show session store statistics")

    sessions_rename = sessions_subparsers.add_parser(
        "rename", help="Set or change a session's title"
    )
    sessions_rename.add_argument("session_id", help="Session ID to rename")
    sessions_rename.add_argument("title", nargs="+", help="New title for the session")

    sessions_retitle = sessions_subparsers.add_parser(
        "retitle-skills",
        help="Re-title sessions whose auto-title came from a /skill's own text",
        description=(
            "Sessions opened with a /skill were auto-titled from the expanded "
            "message, which embeds the whole skill body — so the title "
            "describes the SKILL, not the request. This regenerates those "
            "titles from what the user actually typed. Lists what it would "
            "change unless --apply is passed."
        ),
    )
    sessions_retitle.add_argument(
        "--apply",
        action="store_true",
        help="Write the new titles (default: dry run)",
    )
    sessions_retitle.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum sessions to examine (default: 200)",
    )

    sessions_browse = sessions_subparsers.add_parser(
        "browse",
        help="Interactive session picker — browse, search, and resume sessions",
    )
    sessions_browse.add_argument(
        "--source", help="Filter by source (cli, telegram, discord, etc.)"
    )
    sessions_browse.add_argument(
        "--limit", type=int, default=500, help="Max sessions to load (default: 500)"
    )


    # cmd_sessions lives in hermes_cli/sessions_cmd.py (main.py decomposition).
    # sessions_parser is threaded in via functools.partial because the
    # fallthrough branch calls sessions_parser.print_help() (formerly a
    # closure capture of this main()-local). The indirection through _self()
    # keeps the sessions_cmd import lazy until the subcommand actually runs
    # and lets monkeypatches on hermes_cli.main.cmd_sessions keep working.
    def _dispatch_sessions(_args, *, sessions_parser=sessions_parser):
        return _self().cmd_sessions(_args, sessions_parser=sessions_parser)

    sessions_parser.set_defaults(func=_dispatch_sessions)

    # =========================================================================
    # insights command  (parser built in hermes_cli/subcommands/insights.py)
    # =========================================================================
    build_insights_parser(subparsers, cmd_insights=cmd_insights)
    build_monitoring_parser(subparsers, cmd_monitoring=cmd_monitoring)

    # =========================================================================
    # claw command  (parser built in hermes_cli/subcommands/claw.py)
    # =========================================================================
    build_claw_parser(subparsers, cmd_claw=cmd_claw)

    # =========================================================================
    # version command  (parser built in hermes_cli/subcommands/version.py)
    # =========================================================================
    build_version_parser(subparsers, cmd_version=cmd_version)

    # =========================================================================
    # update command  (parser built in hermes_cli/subcommands/update.py)
    # =========================================================================
    build_update_parser(subparsers, cmd_update=cmd_update)

    # =========================================================================
    # uninstall command  (parser built in hermes_cli/subcommands/uninstall.py)
    # =========================================================================
    build_uninstall_parser(subparsers, cmd_uninstall=cmd_uninstall)

    # =========================================================================
    # acp command  (parser built in hermes_cli/subcommands/acp.py)
    # =========================================================================
    build_acp_parser(subparsers, cmd_acp=cmd_acp)

    # =========================================================================
    # profile command  (parser built in hermes_cli/subcommands/profile.py)
    # =========================================================================
    build_profile_parser(subparsers, cmd_profile=cmd_profile)

    # =========================================================================
    # completion command
    # =========================================================================
    completion_parser = subparsers.add_parser(
        "completion",
        help="Print shell completion script (bash, zsh, or fish)",
    )
    completion_parser.add_argument(
        "shell",
        nargs="?",
        default="bash",
        choices=["bash", "zsh", "fish"],
        help="Shell type (default: bash)",
    )
    completion_parser.set_defaults(func=lambda args: cmd_completion(args, parser))

    # =========================================================================
    # dashboard command  (parser built in hermes_cli/subcommands/dashboard.py)
    # =========================================================================
    build_dashboard_parser(
        subparsers,
        cmd_dashboard=cmd_dashboard,
        cmd_dashboard_register=cmd_dashboard_register,
    )


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
    build_gui_parser(subparsers, cmd_gui=cmd_gui)

    # =========================================================================
    # logs command  (parser built in hermes_cli/subcommands/logs.py)
    # =========================================================================
    build_logs_parser(subparsers, cmd_logs=cmd_logs)

    # =========================================================================
    # prompt-size command  (parser built in hermes_cli/subcommands/prompt_size.py)
    # =========================================================================
    build_prompt_size_parser(subparsers, cmd_prompt_size=cmd_prompt_size)

    # =========================================================================
    # Parse and execute
    # =========================================================================
    # Pre-process argv so unquoted multi-word session names after -c / -r
    # are merged into a single token before argparse sees them.
    # e.g. ``hermes -c Pokemon Agent Dev`` → ``hermes -c 'Pokemon Agent Dev'``
    # ── Container-aware routing ────────────────────────────────────────
    # When NixOS container mode is active, route ALL subcommands into
    # the managed container.  This MUST run before parse_args() so that
    # --help, unrecognised flags, and every subcommand are forwarded
    # transparently instead of being intercepted by argparse on the host.
    from hermes_cli.config import get_container_exec_info

    container_info = get_container_exec_info()
    if container_info:
        _exec_in_container(container_info, sys.argv[1:])
        # Unreachable: os.execvp never returns on success (process is replaced)
        # and raises OSError on failure (which propagates as a traceback).
        sys.exit(1)

    _processed_argv = _coalesce_session_name_args(sys.argv[1:])

    # ── Defensive subparser routing (bpo-9338 workaround) ───────────
    # On some Python versions (notably <3.11), argparse fails to route
    # subcommand tokens when the parent parser has nargs='?' optional
    # arguments (--continue).  The symptom: "unrecognized arguments: model"
    # even though 'model' is a registered subcommand.
    #
    # Fix: when argv contains a token matching a known subcommand, set
    # subparsers.required=True to force deterministic routing.  If that
    # fails (e.g. 'hermes -c model' where 'model' is consumed as the
    # session name for --continue), fall back to the default behaviour.
    import io as _io

    _known_cmds = (
        set(subparsers.choices.keys()) if hasattr(subparsers, "choices") else set()
    )
    _has_cmd_token = any(
        t in _known_cmds for t in _processed_argv if not t.startswith("-")
    )

    if _has_cmd_token:
        subparsers.required = True
        _saved_stderr = sys.stderr
        try:
            sys.stderr = _io.StringIO()
            args = parser.parse_args(_processed_argv)
            sys.stderr = _saved_stderr
        except SystemExit as exc:
            sys.stderr = _saved_stderr
            # Help/version flags (exit code 0) already printed output —
            # re-raise immediately to avoid a second parse_args printing
            # the same help text again (#10230).
            if exc.code == 0:
                raise
            # Subcommand name was consumed as a flag value (e.g. -c model).
            # Fall back to optional subparsers so argparse handles it normally.
            subparsers.required = False
            args = parser.parse_args(_processed_argv)
    else:
        subparsers.required = False
        args = parser.parse_args(_processed_argv)

    # Handle --version flag
    if args.version:
        cmd_version(args)
        return

    # --yolo: set HERMES_YOLO_MODE *before* plugin discovery.  The call to
    # _prepare_agent_startup() below triggers discover_plugins() → tool
    # imports, and tools.approval freezes _YOLO_MODE_FROZEN at module
    # import time (PR #7994, security hardening against prompt-injection).
    # If the env var is set only later (e.g. inside cmd_chat), the frozen
    # value is already False and --yolo silently does nothing.
    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"

    # Discover Python plugins and register shell hooks once, before any
    # command that can fire lifecycle hooks.  Both are idempotent; gated
    # so introspection/management commands (hermes hooks list, cron
    # list, gateway status, mcp add, ...) don't pay discovery cost or
    # trigger consent prompts for hooks the user is still inspecting.
    _prepare_agent_startup(args)

    # Handle top-level --oneshot / -z: single-shot mode, stdout = final
    # response only, nothing else. Bypasses cli.py entirely.
    if getattr(args, "oneshot", None):
        _confirm_startup_expensive_model_override(args)
        _run_and_exit_oneshot(
            args.oneshot,
            model=getattr(args, "model", None),
            provider=getattr(args, "provider", None),
            toolsets=getattr(args, "toolsets", None),
            usage_file=getattr(args, "usage_file", None),
        )

    # Handle top-level --resume / --continue as shortcut to chat
    if (args.resume or args.continue_last) and args.command is None:
        args.command = "chat"
        for attr, default in [
            ("query", None),
            ("model", None),
            ("provider", None),
            ("toolsets", None),
            ("verbose", None),
            ("worktree", False),
        ]:
            if not hasattr(args, attr):
                setattr(args, attr, default)
        cmd_chat(args)
        return

    # Default to chat if no command specified
    if args.command is None:
        for attr, default in [
            ("query", None),
            ("model", None),
            ("provider", None),
            ("toolsets", None),
            ("verbose", None),
            ("resume", None),
            ("continue_last", None),
            ("worktree", False),
        ]:
            if not hasattr(args, attr):
                setattr(args, attr, default)
        cmd_chat(args)
        return

    # Execute the command.  Propagate the handler's return code as the
    # process exit code so subcommands that signal failure (e.g.
    # ``hermes egress start`` refusing when credential_source=bitwarden
    # is misconfigured) actually exit non-zero.  Handlers that return
    # None are treated as success (exit 0).
    if hasattr(args, "func"):
        rc = args.func(args)
        if isinstance(rc, int) and rc != 0:
            sys.exit(rc)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
