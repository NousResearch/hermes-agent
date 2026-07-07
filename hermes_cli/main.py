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

import os
import sys


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
    ``--tui``/``HERMES_TUI=1``, then ``display.interface`` in config.
    """
    if argv is None:
        argv = sys.argv[1:]
    if "--cli" in argv:
        return False
    if os.environ.get("HERMES_TUI") == "1" or "--tui" in argv:
        return True
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
    prefix = os.environ.get("PREFIX", "")
    return bool(
        os.environ.get("TERMUX_VERSION")
        or "com.termux/files/usr" in prefix
        or prefix.startswith("/data/data/com.termux/")
    )


def _is_termux_fast_version_argv(argv: list[str]) -> bool:
    return argv in (["--version"], ["-V"], ["version"])


def _read_openai_version_fast() -> str | None:
    """Read OpenAI SDK version without importing ``importlib.metadata``."""
    for base in sys.path:
        if not base:
            base = os.getcwd()
        version_file = os.path.join(base, "openai", "_version.py")
        try:
            with open(version_file, encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped.startswith("__version__"):
                        continue
                    _key, _sep, value = stripped.partition("=")
                    value = value.split("#", 1)[0].strip().strip("\"'")
                    return value or None
        except OSError:
            continue
    return None


def _print_fast_version_info() -> None:
    from hermes_cli import __release_date__, __version__

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    print(f"Hermes Agent v{__version__} ({__release_date__})")
    print(f"Project: {project_root}")
    print(f"Python: {sys.version.split()[0]}")

    openai_version = _read_openai_version_fast()
    print(f"OpenAI SDK: {openai_version}" if openai_version else "OpenAI SDK: Not installed")


def _try_termux_ultrafast_version() -> bool:
    """Handle ``hermes --version`` before config/logging imports on Termux."""
    if os.environ.get("HERMES_TERMUX_DISABLE_FAST_CLI") == "1":
        return False
    if not _is_termux_startup_environment_fast():
        return False
    if not _is_termux_fast_version_argv(sys.argv[1:]):
        return False

    _print_fast_version_info()
    return True


if _try_termux_ultrafast_version():
    raise SystemExit(0)

import argparse
import hashlib
import json
import shlex
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Optional


from hermes_cli.subcommands._shared import add_accept_hooks_flag as _add_accept_hooks_flag
from hermes_cli.subcommands.cron import build_cron_parser
from hermes_cli.subcommands.gateway import build_gateway_parser
from hermes_cli.subcommands.profile import build_profile_parser
from hermes_cli.subcommands.model import build_model_parser
from hermes_cli.subcommands.setup import build_setup_parser
from hermes_cli.subcommands.postinstall import build_postinstall_parser
from hermes_cli.subcommands.whatsapp import build_whatsapp_parser
from hermes_cli.subcommands.slack import build_slack_parser
from hermes_cli.subcommands.login import build_login_parser
from hermes_cli.subcommands.logout import build_logout_parser
from hermes_cli.subcommands.auth import build_auth_parser
from hermes_cli.subcommands.status import build_status_parser
from hermes_cli.subcommands.webhook import build_webhook_parser
from hermes_cli.subcommands.hooks import build_hooks_parser
from hermes_cli.subcommands.doctor import build_doctor_parser
from hermes_cli.subcommands.security import build_security_parser
from hermes_cli.subcommands.dump import build_dump_parser
from hermes_cli.subcommands.debug import build_debug_parser
from hermes_cli.subcommands.backup import build_backup_parser
from hermes_cli.subcommands.import_cmd import build_import_cmd_parser
from hermes_cli.subcommands.config import build_config_parser
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
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


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
                name = active_path.read_text().strip()
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
    import yaml as _yaml_early

    _cfg_path = get_hermes_home() / "config.yaml"
    if _cfg_path.exists():
        with open(_cfg_path, encoding="utf-8") as _f:
            _early_cfg_raw = _yaml_early.load(
                _f, Loader=getattr(_yaml_early, "CSafeLoader", None) or _yaml_early.SafeLoader
            ) or {}
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
    """Format a timestamp as relative time (e.g., '2h ago', 'yesterday')."""
    if not ts:
        return "?"
    delta = _time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    if delta < 172800:
        return "yesterday"
    if delta < 604800:
        return f"{int(delta / 86400)}d ago"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


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
        _model_name = (model_cfg.get("default") or "").strip()
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

    # Check for Nous Portal OAuth credentials
    auth_file = get_hermes_home() / "auth.json"
    if auth_file.exists():
        try:
            import json

            auth = json.loads(auth_file.read_text())
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


def _session_browse_picker(sessions: list) -> Optional[str]:
    """Interactive curses-based session browser with live search filtering.

    Returns the selected session ID, or None if cancelled.
    Uses curses (not simple_term_menu) to avoid the ghost-duplication rendering
    bug in tmux/iTerm when arrow keys are used.
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


def _resolve_last_session(source: str = "cli") -> Optional[str]:
    """Look up the most recently-used session ID for a source."""
    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB()
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
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15)
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

        db.close()
        return resolved_id
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

    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
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
            text=True,
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
        path = shutil.which(bin)
        if not path and bin == "node":
            try:
                from hermes_cli.dep_ensure import ensure_dependency
                if ensure_dependency("node"):
                    path = shutil.which("node")
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

    if not ext_dir:
        _ensure_tui_workspace(tui_dir)

    # 1. Prebuilt bundle (nix / packaged release): just run it.
    if not tui_dev:
        if ext_dir:
            p = Path(ext_dir)
            if (p / "dist" / "entry.js").is_file():
                node = _node_bin("node")
                return [node, "--expose-gc", str(p / "dist" / "entry.js")], p

        # 1b. Bundled in wheel (pip install)
        bundled = _find_bundled_tui()
        if bundled is not None:
            node = _node_bin("node")
            return [node, "--expose-gc", str(bundled)], bundled.parent

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
        result = subprocess.run(
            [
                npm,
                "install",
                *npm_workspace_args,
                "--silent",
                "--no-fund",
                "--no-audit",
                "--progress=false",
            ],
            cwd=str(npm_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "CI": "1"},
        )
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

    env = os.environ.copy()
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
    env["HERMES_PYTHON_SRC_ROOT"] = os.environ.get(
        "HERMES_PYTHON_SRC_ROOT", str(PROJECT_ROOT)
    )
    env.setdefault("HERMES_PYTHON", sys.executable)
    env.setdefault("HERMES_CWD", os.getcwd())
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
    # Ink app.  Because we start from os.environ.copy(), an exported/stale value
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
      2. ``--tui`` flag / ``HERMES_TUI=1`` → always TUI
      3. ``display.interface`` config value ("cli" | "tui")
      4. default → classic REPL

    Explicit flags always win over config so muscle memory and scripts keep
    working regardless of the configured default.
    """
    if getattr(args, "cli", False):
        return False
    if getattr(args, "tui", False) or os.environ.get("HERMES_TUI") == "1":
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
            from hermes_cli.banner import prefetch_update_check

            prefetch_update_check()
        except Exception:
            pass

    # Sync bundled skills on every CLI launch (fast -- skips unchanged skills)
    try:
        _sync_bundled_skills_for_startup()
    except Exception:
        pass

    # --yolo: bypass all dangerous command approvals
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


def cmd_postinstall(args):
    """One-shot bootstrap for pip users: install non-Python deps + run setup."""
    from hermes_cli.config import stamp_install_method
    from hermes_cli.dep_ensure import ensure_dependency

    stamp_install_method("pip")

    print("⚕ Hermes post-install bootstrap")
    print()

    for dep in ("node", "browser", "ripgrep", "ffmpeg"):
        ensure_dependency(dep)

    if not _has_any_provider_configured():
        print()
        cmd_setup(args)
    else:
        print()
        print("✓ Post-install complete.")


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
    from hermes_cli.providers import resolve_provider_full

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
            key = "custom:" + name.lower().replace(" ", "-")
            provider_key = (entry.get("provider_key") or "").strip()
            if provider_key:
                try:
                    resolve_provider(provider_key)
                except AuthError:
                    key = provider_key
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
    grouped_rows = group_providers([p.slug for p in CANONICAL_PROVIDERS])

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

    Uses ``list_authenticated_providers()`` to only show providers the user
    has already configured. This avoids re-running OAuth/credential flows
    inside the aux picker — users set up new providers through the normal
    ``hermes model`` flow, then route aux tasks to them here.
    """
    from hermes_cli.config import load_config
    from hermes_cli.model_switch import list_authenticated_providers

    cfg = load_config()
    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    task_cfg = aux.get(task, {}) if isinstance(aux.get(task), dict) else {}
    current_provider = str(task_cfg.get("provider") or "auto").strip() or "auto"
    current_model = str(task_cfg.get("model") or "").strip()
    current_base_url = str(task_cfg.get("base_url") or "").strip()

    display_name = next((name for key, name, _ in _all_aux_tasks() if key == task), task)

    # Gather authenticated providers (has credentials + curated model list)
    try:
        providers = list_authenticated_providers(
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

    for p in providers:
        slug = p.get("slug", "")
        name = p.get("name") or slug
        total = p.get("total_models", 0)
        models = p.get("models") or []
        model_hint = f" — {total} models" if total else ""
        marker = (
            "  ← current" if slug == current_provider and not current_base_url else ""
        )
        entries.append((slug, f"{name}{model_hint}{marker}", list(models)))

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
    base_url, api_key="", model="", context_length=None, name=None, api_mode=None
):
    """Save a custom endpoint to custom_providers in config.yaml.

    Deduplicates by base_url — if the URL already exists, updates the
    model name, context_length, and api_mode but doesn't add a duplicate entry.
    Uses *name* when provided, otherwise auto-generates from the URL.
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
            if changed:
                cfg["custom_providers"] = providers
                save_config(cfg)
            return  # already saved, updated if needed

    # Use provided name or auto-generate from URL
    if not name:
        name = _auto_provider_name(base_url)

    entry = {"name": name, "base_url": base_url}
    if api_key:
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


def __getattr__(name):
    """Defer the model-catalog import until something actually reads it."""
    if name in _LAZY_MODEL_EXPORTS:
        from hermes_cli.models import _PROVIDER_MODELS
        # Cache on the module so subsequent accesses skip the import machinery.
        globals()[name] = _PROVIDER_MODELS
        return _PROVIDER_MODELS
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
    canonical_order = ("minimal", "low", "medium", "high", "xhigh")
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






def _prompt_api_key(pconfig, existing_key: str, provider_id: str = "") -> tuple:
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
    try:
        choice = input("  [K]eep / [R]eplace / [C]lear (default K): ").strip().lower()
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

    if choice.startswith("c"):
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

        return slack_manifest_command(args)

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
    from hermes_cli.banner import format_banner_version_label

    print(format_banner_version_label())
    print(f"Project: {PROJECT_ROOT}")

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
        from hermes_cli.banner import check_for_updates
        from hermes_cli.config import recommended_update_command

        behind = check_for_updates()
        if behind and behind > 0:
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


_UPDATE_RUNTIME_RELOAD_MODULES = (
    "hermes_constants",
    "tools.environments.local",
    "tools.lazy_deps",
)


def _reload_updated_runtime_modules() -> None:
    """Reload update-sensitive modules after the checkout changes in-place.

    ``hermes update`` keeps running in the pre-pull Python process. After a
    large update, modules already present in ``sys.modules`` can still expose
    old symbols even though their source files on disk are new. Refresh the
    small module set used by lazy-backend refresh before that step imports
    newly-updated code paths.
    """
    try:
        import importlib

        importlib.invalidate_caches()
        for module_name in _UPDATE_RUNTIME_RELOAD_MODULES:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            try:
                importlib.reload(module)
            except Exception as exc:
                logger.debug("Could not reload updated module %s: %s", module_name, exc)
    except Exception as exc:
        logger.debug("Could not refresh update runtime modules: %s", exc)


# Critical files that Hermes must be able to import immediately after an
# update/install. Most are imported on every CLI startup; ``web_server.py``
# is the desktop/dashboard backend path that a fresh Windows install launches
# right away. If any of these fail to parse after a pull, the user can be
# left with a bricked CLI or desktop backend. The post-pull syntax guard
# validates these and auto-rolls-back on failure.
_UPDATE_CRITICAL_FILES = (
    "hermes_cli/main.py",
    "hermes_cli/config.py",
    "hermes_cli/__init__.py",
    "hermes_cli/web_server.py",
    "cli.py",
    "run_agent.py",
    "model_tools.py",
    "toolsets.py",
    "hermes_constants.py",
)


def _capture_head_sha(git_cmd, cwd) -> str | None:
    """Return the current HEAD SHA, or None if it can't be resolved."""
    try:
        result = subprocess.run(
            git_cmd + ["rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, OSError):
        return None


def _validate_critical_files_syntax(root) -> tuple[bool, str | None, str | None]:
    """Compile each file in ``_UPDATE_CRITICAL_FILES`` to catch SyntaxErrors.

    These are the files imported on every ``hermes`` startup; if any of them
    has a syntax error (orphan merge-conflict markers, bad ref to a name
    that no longer exists, etc.) the CLI can't bootstrap at all. We validate
    them after a successful ``git pull`` so we can auto-roll-back instead of
    leaving the user with a bricked install.

    The compiled ``.pyc`` is written to a temp directory rather than the
    source tree's ``__pycache__/`` so we don't race with concurrent test
    workers that walk the same dir, and so we don't leave a stale pyc
    behind in production if the next interpreter run picks a different
    Python version. The pyc is discarded on function return either way —
    we only care about the compile-or-not signal.

    Returns ``(ok, failing_path, error_message)``. ``ok=True`` means every
    file parsed cleanly.
    """
    import py_compile
    import tempfile

    root = Path(root)
    with tempfile.TemporaryDirectory(prefix="hermes-syntax-check-") as tmpdir:
        for relpath in _UPDATE_CRITICAL_FILES:
            path = root / relpath
            if not path.exists():
                # Missing file is suspicious but not necessarily fatal — a future
                # refactor may legitimately remove one of these. Skip and move on.
                continue
            # Mirror the relative path under the tmpdir so two different
            # files with the same basename don't collide on the cfile name.
            cfile = Path(tmpdir) / (relpath.replace("/", "__") + "c")
            try:
                py_compile.compile(str(path), cfile=str(cfile), doraise=True)
            except py_compile.PyCompileError as exc:
                return False, str(path), str(exc)
            except OSError as exc:
                return False, str(path), f"could not read: {exc}"
    return True, None, None


def _gateway_prompt(prompt_text: str, default: str = "", timeout: float = 300.0) -> str:
    """File-based IPC prompt for gateway mode.

    Writes a prompt marker file so the gateway can forward the question to the
    user, then polls for a response file.  Falls back to *default* on timeout.

    Used by ``hermes update --gateway`` so interactive prompts (stash restore,
    config migration) are forwarded to the messenger instead of being silently
    skipped.
    """
    import json as _json
    import uuid as _uuid
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    prompt_path = home / ".update_prompt.json"
    response_path = home / ".update_response"

    # Clean any stale response file
    response_path.unlink(missing_ok=True)

    payload = {
        "prompt": prompt_text,
        "default": default,
        "id": str(_uuid.uuid4()),
    }
    tmp = prompt_path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(payload))
    tmp.replace(prompt_path)

    # Poll for response
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if response_path.exists():
            try:
                answer = response_path.read_text().strip()
                response_path.unlink(missing_ok=True)
                prompt_path.unlink(missing_ok=True)
                return answer if answer else default
            except (OSError, ValueError):
                pass
        _time.sleep(0.5)

    # Timeout — clean up and use default
    prompt_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)
    print(f"  (no response after {int(timeout)}s, using default: {default!r})")
    return default


def _web_ui_build_needed(web_dir: Path) -> bool:
    """Return True if the web UI dist is missing or stale.

    Mirrors the staleness logic used by ``_tui_build_needed()`` for the TUI.
    The dashboard source lives under ``web/``, but the Vite build
    still outputs to ``hermes_cli/web_dist/`` (per vite.config.ts
    outDir: "../hermes_cli/web_dist"), NOT to ``web/dist/``, so Python
    packaging can continue serving the same static asset directory. Uses the
    Vite manifest as the sentinel because it is written last and therefore
    has the newest mtime of any build output.
    """
    project_root = web_dir.parent.parent if web_dir.parent.name == "apps" else web_dir.parent
    dist_dir = project_root / "hermes_cli" / "web_dist"
    sentinel = dist_dir / ".vite" / "manifest.json"
    if not sentinel.exists():
        sentinel = dist_dir / "index.html"
    if not sentinel.exists():
        return True
    dist_mtime = sentinel.stat().st_mtime
    skip = frozenset({"node_modules", "dist"})
    for dirpath, dirnames, filenames in os.walk(web_dir, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            if fn.endswith((".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".vue")):
                if os.path.getmtime(os.path.join(dirpath, fn)) > dist_mtime:
                    return True
    for meta in (
        "package.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "vite.config.ts",
        "vite.config.js",
    ):
        mp = web_dir / meta
        if mp.exists() and mp.stat().st_mtime > dist_mtime:
            return True
    # Workspace root lockfile (single package-lock.json covers all workspaces).
    root_lock = project_root / "package-lock.json"
    if root_lock.exists() and root_lock.stat().st_mtime > dist_mtime:
        return True
    return False


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
            capture_output=True, text=True, check=False, timeout=15,
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
    """
    # unicode-animations' postinstall animates to /dev/tty (bypasses
    # --silent/capture_output). It no-ops when CI is set — same as the TUI
    # install path and nix/lib.nix npm ci hooks.
    run_env = {**os.environ, **(env or {}), "CI": "1"}

    lockfile = cwd / "package-lock.json"
    if lockfile.exists():
        ci_cmd = [npm, "ci", *extra_args]
        ci_result = subprocess.run(
            ci_cmd,
            cwd=cwd,
            env=run_env,
            capture_output=capture_output,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if ci_result.returncode == 0:
            return ci_result
        # Fall through to `npm install` — lockfile may be out of sync on a
        # WIP fork/branch, or `npm ci` may not be available on very old npm.
    install_cmd = [npm, "install", *extra_args]
    return subprocess.run(
        install_cmd,
        cwd=cwd,
        env=run_env,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _build_web_ui(web_dir: Path, *, fatal: bool = False) -> bool:
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

    from hermes_constants import find_node_executable, with_hermes_node_path

    npm = find_node_executable("npm")
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
    npm_workspace_args: tuple[str, ...] = () if npm_cwd == web_dir else ("--workspace", "web")
    if _is_termux_startup_environment():
        npm_cwd, npm_workspace_args = _termux_workspace_install_context(web_dir)
    r1 = _run_npm_install_deterministic(
        npm,
        npm_cwd,
        extra_args=(*npm_workspace_args, "--silent"),
        env=build_env,
    )
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


def _desktop_build_needed(desktop_dir: Path, project_root: Path, *, source_mode: bool) -> bool:
    """Return True when the desktop build output is stale or missing.

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
    return max(existing, key=lambda p: p.stat().st_mtime)


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


def _desktop_macos_relaunchable_fixup(desktop_dir: Path) -> None:
    """Make a locally-built (unsigned) macOS desktop app survive in-place self-update.

    An ad-hoc-signed .app has no stable Designated Requirement (no Team ID), so
    when the self-updater rebuilds the bundle in place with a fresh build (a new,
    different cdhash) Gatekeeper/LaunchServices treats the changed code as
    tampering and macOS reports "Hermes is damaged and can't be opened." The
    bundle also inherits the com.apple.quarantine flag from the downloaded
    installer process chain. Both make the relaunch fail.

    Clearing the quarantine xattrs and re-applying a clean deep ad-hoc signature
    (omitting the hardened-runtime flag, which is meaningless without a real
    Developer ID) lets the rebuilt app relaunch. No-op when a real signing
    identity is configured (CSC_LINK / APPLE_SIGNING_IDENTITY) so a properly
    signed/notarized build is never clobbered. Best-effort: never raises.
    """
    if sys.platform != "darwin":
        return
    if os.environ.get("CSC_LINK") or os.environ.get("APPLE_SIGNING_IDENTITY"):
        return
    exe = _desktop_packaged_executable(desktop_dir)
    if exe is None:
        return
    # exe = .../Hermes.app/Contents/MacOS/Hermes  ->  app bundle = .../Hermes.app
    app = exe.parents[2]
    if not str(app).endswith(".app") or not app.is_dir():
        return
    codesign = shutil.which("codesign")
    if not codesign:
        return
    try:
        subprocess.run(["xattr", "-cr", str(app)], check=False)
        subprocess.run([codesign, "--force", "--deep", "--sign", "-", str(app)], check=False)
    except Exception as exc:
        print(f"  (warning: macOS relaunch fixup skipped: {exc})")


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


def _desktop_launch_options() -> tuple[list[str], str]:
    """Read `desktop.*` launch options from config.yaml.

    Returns ``(electron_flags, disable_gpu)`` where ``electron_flags`` is a list
    of extra Electron CLI flags and ``disable_gpu`` is one of "auto"/"1"/"0"
    (normalized for the HERMES_DESKTOP_DISABLE_GPU env var the Electron app
    reads). Best-effort: any config error yields the safe defaults
    ``([], "auto")`` so a malformed config never blocks the launch.
    """
    flags: list[str] = []
    disable_gpu = "auto"
    try:
        from hermes_cli.config import load_config

        desktop_cfg = (load_config() or {}).get("desktop") or {}
    except Exception:
        return flags, disable_gpu

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
    return flags, disable_gpu


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

    from hermes_constants import find_node_executable, with_hermes_node_path

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
    config_electron_flags, config_disable_gpu = _desktop_launch_options()
    if config_disable_gpu != "auto" and "HERMES_DESKTOP_DISABLE_GPU" not in os.environ:
        env["HERMES_DESKTOP_DISABLE_GPU"] = config_disable_gpu

    source_mode = getattr(args, "source", False)
    skip_build = getattr(args, "skip_build", False)
    force_build = getattr(args, "force_build", False)

    packaged_executable = _desktop_packaged_executable(desktop_dir)

    if source_mode or not skip_build:
        npm = find_node_executable("npm")
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
            nixos_env = _nixos_build_env()
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

            # Build succeeded — write the stamp so next run can skip
            _write_desktop_build_stamp(PROJECT_ROOT, source_mode=source_mode)

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


def _find_stale_dashboard_pids(
    *,
    exclude_pids: set[int] | None = None,
) -> list[int]:
    """Return PIDs of ``hermes dashboard`` processes other than ourselves.

    ``hermes dashboard`` is a long-lived server process commonly started and
    forgotten.  When ``hermes update`` replaces files on disk, the running
    process keeps the old Python backend in memory while the JS bundle on
    disk is updated, causing a silent frontend/backend mismatch (e.g. new
    auth headers the old backend doesn't recognise → every API call 401s).

    The dashboard has no service manager (systemd / launchd), no PID file,
    and we can't know the original launch args — so the only sane action
    after an update is to kill the stale process and let the user restart
    it.  This helper is just the detection step; see
    ``_kill_stale_dashboard_processes`` for the kill.

    *exclude_pids* is an optional set of PIDs that must never be returned.
    This is used by the Hermes Desktop Electron app to protect its own
    backend child process: when the desktop spawns ``hermes serve`` as
    a backend and triggers an auto-update, the update must not kill the
    backend that the desktop itself manages.  The desktop sets the
    environment variable ``HERMES_DESKTOP_CHILD_PID`` on the spawned
    backend process; ``_kill_stale_dashboard_processes`` reads it and
    passes it here.  (#37532)

    Returns an empty list on any scan error (missing ps/wmic, timeout, etc.).
    """
    patterns = [
        "hermes dashboard",
        "hermes_cli.main dashboard",
        "hermes_cli/main.py dashboard",
        # The headless backend (`hermes serve`) is the same long-lived server
        # under a different command name — the desktop app spawns it. Reap it
        # on update for the same frontend/backend-mismatch reason.
        "hermes serve",
        "hermes_cli.main serve",
        "hermes_cli/main.py serve",
    ]
    self_pid = os.getpid()
    dashboard_pids: list[int] = []

    try:
        if sys.platform == "win32":
            # wmic may emit text in the system code page (for example cp936
            # on zh-CN systems), not UTF-8. In text mode, subprocess output
            # decoding depends on Python's configuration (locale-dependent
            # by default, or UTF-8 in UTF-8 mode). The important protection
            # here is errors="ignore": it prevents a reader-thread
            # UnicodeDecodeError from leaving result.stdout=None and turning
            # the later .split() into an AttributeError (#17049).
            # CREATE_NO_WINDOW hides the conhost flash: this scan can run from
            # the windowless pythonw.exe desktop/gateway backend during an
            # update, where a bare wmic spawn would pop a console window.
            from hermes_cli._subprocess_compat import windows_hide_flags

            result = subprocess.run(
                ["wmic", "process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="ignore",
                creationflags=windows_hide_flags(),
            )
            if result.returncode != 0 or result.stdout is None:
                return []
            current_cmd = ""
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line.startswith("CommandLine="):
                    current_cmd = line[len("CommandLine=") :]
                elif line.startswith("ProcessId="):
                    pid_str = line[len("ProcessId=") :]
                    if (
                        any(p in current_cmd for p in patterns)
                        and int(pid_str) != self_pid
                    ):
                        try:
                            dashboard_pids.append(int(pid_str))
                        except ValueError:
                            pass
        else:
            # Linux / macOS: scan the process table via ps and match against
            # the same explicit patterns list used on Windows.  Using ps
            # (rather than `pgrep -f "hermes.*dashboard"`) keeps us consistent
            # with `hermes_cli.gateway._scan_gateway_pids` and avoids the
            # greedy regex matching unrelated cmdlines that merely contain
            # both words (e.g. a chat session discussing "dashboard").
            result = subprocess.run(
                ["ps", "-A", "-o", "pid=,command="],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in getattr(result, "stdout", "").split("\n"):
                    stripped = line.strip()
                    if not stripped or "grep" in stripped:
                        continue
                    parts = stripped.split(None, 1)
                    if len(parts) != 2:
                        continue
                    try:
                        pid = int(parts[0])
                    except ValueError:
                        continue
                    command = parts[1]
                    if any(p in command for p in patterns) and pid != self_pid:
                        dashboard_pids.append(pid)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []

    if exclude_pids:
        dashboard_pids = [p for p in dashboard_pids if p not in exclude_pids]
    return dashboard_pids


def _print_curator_first_run_notice() -> None:
    """Print a short heads-up about the skill curator after `hermes update`.

    Only fires when the curator is enabled AND has no recorded run yet, which
    is exactly the window where the gateway ticker used to fire Curator
    against a fresh skill library immediately after an update. We defer the
    first real pass by one ``interval_hours``; this notice tells the user how
    to preview or disable before then. Silent on steady state.
    """
    try:
        from agent import curator
    except Exception:
        return
    try:
        if not curator.is_enabled():
            return
        state = curator.load_state()
    except Exception:
        return
    if state.get("last_run_at"):
        # Curator has run before (real or already seeded) — no notice needed.
        return
    try:
        hours = curator.get_interval_hours()
    except Exception:
        hours = 24 * 7
    days = max(1, hours // 24)
    print()
    print("ℹ Skill curator")
    print(
        f"  Background skill maintenance is enabled. First pass is deferred "
        f"~{days}d after installation; only agent-created skills are in "
        f"scope and nothing is ever auto-deleted (archive is recoverable)."
    )
    print("  Preview now:  hermes curator run --dry-run")
    print("  Pause it:     hermes curator pause")
    print(
        "  Docs:         https://hermes-agent.nousresearch.com/docs/user-guide/features/curator"
    )


def _print_curator_recent_run_notice() -> None:
    """Print the most recent curator run summary, exactly once.

    The curator runs in the background (gateway tick + CLI session start),
    so users learn about skill consolidations only by stumbling into a
    rename. ``hermes update`` is a high-attention surface — surface the
    most recent run's rename map here, once.

    Show-once: state stamps ``last_run_summary_shown_at`` after printing.
    Subsequent ``hermes update`` invocations skip the block until a newer
    curator run lands. Silent when the curator has never run, when the
    most recent summary has already been shown, or when the summary has
    no rename information to display (no archives).
    """
    try:
        from agent import curator
    except Exception:
        return
    try:
        state = curator.load_state()
    except Exception:
        return

    last_run_at = state.get("last_run_at")
    if not last_run_at:
        return  # no curator run yet — first-run notice handles this case

    if state.get("last_run_summary_shown_at") == last_run_at:
        return  # already shown for this run

    summary = state.get("last_run_summary") or ""
    if not summary:
        return

    # Only print when there's something interesting to show — i.e. the
    # rename map block was appended (multi-line summary). A bare "auto:
    # no changes; llm: no change" doesn't warrant interrupting the
    # update flow.
    if "\n" not in summary:
        # Still stamp it shown so we don't reconsider it on every update.
        try:
            state["last_run_summary_shown_at"] = last_run_at
            curator.save_state(state)
        except Exception:
            pass
        return

    # Format the timestamp as "Xh ago" for readability.
    when = _format_time_ago(last_run_at)
    print()
    print(f"ℹ Skill curator — last run {when}")
    for line in summary.splitlines():
        print(f"  {line}")
    print(
        "  (This message shows once per curator run. "
        "View anytime: hermes curator status)"
    )

    # Stamp shown so we don't repeat on the next update.
    try:
        state["last_run_summary_shown_at"] = last_run_at
        curator.save_state(state)
    except Exception:
        pass


def _format_time_ago(iso_ts: str) -> str:
    """Render an ISO timestamp as `Xh ago` / `Xd ago` / `Xm ago`. Best effort."""
    try:
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        secs = int(delta.total_seconds())
        if secs < 60:
            return "just now"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return "recently"


def _kill_stale_dashboard_processes(
    reason: str = "the running backend no longer matches the updated frontend",
) -> None:
    """Kill running ``hermes dashboard`` processes.

    Called at the end of ``hermes update`` (default ``reason``) and also
    from ``hermes dashboard --stop`` (which overrides ``reason``).  The
    dashboard has no service manager, so after a code update the running
    process is guaranteed to be serving stale Python against a
    freshly-updated JS bundle.  Leaving it alive produces silent
    frontend/backend mismatches (new auth headers the old backend doesn't
    recognise → every API call 401s).

    POSIX: SIGTERM, wait up to ~3s for graceful exit, SIGKILL any survivors.
    Windows: ``taskkill /PID <pid> /F`` since there's no clean SIGTERM
    equivalent for background console apps.

    The dashboard isn't auto-restarted because we don't know the original
    launch args (--host, --port, --insecure, --tui, --no-open).  The user
    restarts it manually; a hint is printed.
    """
    # When the Hermes Desktop Electron app spawns this dashboard as a
    # backend child, it sets HERMES_DESKTOP_CHILD_PID so that the update
    # path can skip killing the desktop-managed process.  (#37532)
    exclude: set[int] | None = None
    raw_pid = os.environ.get("HERMES_DESKTOP_CHILD_PID")
    if raw_pid:
        # The desktop may manage several backends (one per active profile) and
        # passes them comma-separated; a lone int still parses for back-compat.
        parsed: set[int] = set()
        for part in raw_pid.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.add(int(part))
            except (ValueError, TypeError):
                pass
        if parsed:
            exclude = parsed

    pids = _find_stale_dashboard_pids(exclude_pids=exclude)
    if not pids:
        return

    print()
    print(f"⟲ Stopping {len(pids)} dashboard process(es) ({reason})")

    killed: list[int] = []
    failed: list[tuple[int, str]] = []

    if sys.platform == "win32":
        for pid in pids:
            try:
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    killed.append(pid)
                else:
                    failed.append((pid, (result.stderr or result.stdout or "").strip()))
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
                failed.append((pid, str(e)))
    else:
        import signal as _signal
        import time as _time

        # SIGTERM first — give each process a chance to shut down cleanly
        # (uvicorn closes its socket, flushes logs, etc.).
        for pid in pids:
            try:
                os.kill(pid, _signal.SIGTERM)
            except ProcessLookupError:
                # Already gone — count as killed.
                killed.append(pid)
            except (PermissionError, OSError) as e:
                failed.append((pid, str(e)))

        # Poll for exit up to ~3s total.
        deadline = _time.monotonic() + 3.0
        pending = [
            p for p in pids if p not in killed and p not in {f[0] for f in failed}
        ]
        while pending and _time.monotonic() < deadline:
            _time.sleep(0.1)
            still_pending = []
            # On Windows, os.kill(pid, 0) is NOT a no-op. Route through
            # the cross-platform existence check.
            from gateway.status import _pid_exists
            for pid in pending:
                if _pid_exists(pid):
                    still_pending.append(pid)
                else:
                    killed.append(pid)
            pending = still_pending

        # SIGKILL any survivors.
        for pid in pending:
            try:
                os.kill(pid, _signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                killed.append(pid)
            except (PermissionError, OSError) as e:
                failed.append((pid, str(e)))

    for pid in killed:
        print(f"    ✓ stopped PID {pid}")
    for pid, err_msg in failed:
        print(f"    ✗ failed to stop PID {pid}: {err_msg}")

    if killed:
        print("  Restart the dashboard when you're ready:")
        print("    hermes dashboard --port <port>")


# Back-compat alias: some tests and any external callers may import the old
# warn-only name.  The new behaviour (kill stale processes) replaces it.
_warn_stale_dashboard_processes = _kill_stale_dashboard_processes


def _atomic_replace_dir(src: str, dst: str) -> None:
    """Replace directory *dst* with *src* without leaving *dst* half-deleted.

    The naive ``rmtree(dst); copytree(src, dst)`` has a destructive window: if
    the copy fails partway (common on the Windows ZIP-update path, which only
    runs because file I/O is already flaky on that machine), the old directory
    is already gone and nothing replaced it — the install is left with a
    deleted tree (issue #49145, where ``ui-tui/`` vanished and broke the TUI).

    Instead, stage the new copy into a sibling temp dir first; only once that
    fully succeeds do we swap it in. A failure during staging raises with the
    original *dst* still intact.
    """
    staging = f"{dst}.hermes-update-staging"
    backup = f"{dst}.hermes-update-old"
    # Clear any leftovers from a previously-interrupted update.
    for leftover in (staging, backup):
        if os.path.exists(leftover):
            shutil.rmtree(leftover, ignore_errors=True)

    # 1. Stage the new copy. If this fails, dst is untouched.
    shutil.copytree(src, staging)
    # 2. Swap: move the live dir aside, move staging into place. Both moves are
    #    same-filesystem renames; if the second fails we restore the backup.
    if os.path.exists(dst):
        os.rename(dst, backup)
    try:
        os.rename(staging, dst)
    except OSError:
        if os.path.exists(backup) and not os.path.exists(dst):
            os.rename(backup, dst)  # roll back to the original
        raise
    # 3. New dir is in place; drop the old one (best-effort — never fatal).
    if os.path.exists(backup):
        shutil.rmtree(backup, ignore_errors=True)


def _update_via_zip(args):
    """Update Hermes Agent by downloading a ZIP archive.

    Used on Windows when git file I/O is broken (antivirus, NTFS filter
    drivers causing 'Invalid argument' errors on file creation).
    """
    import tempfile
    import zipfile
    from urllib.request import urlretrieve

    # The ZIP fallback exists for Windows git-file-I/O breakage. It pulls a
    # static archive from GitHub, which is fine for the default "main"
    # channel but would silently ignore --branch and update from main even
    # if the user asked for something else — exactly the silent-divergence
    # bug --branch was added to prevent. Refuse to proceed in that case
    # rather than lie.
    branch = _resolve_update_branch(args)
    if branch != "main":
        print(
            f"✗ --branch={branch} is not supported on the Windows ZIP-fallback "
            "update path."
        )
        print(
            "  This path runs when git file I/O is broken on the system. "
            "Either resolve the git-side breakage (typically an antivirus "
            "or NTFS filter holding files open) and rerun `hermes update "
            f"--branch {branch}`, or update against main with `hermes update`."
        )
        sys.exit(1)
    zip_url = (
        f"https://github.com/NousResearch/hermes-agent/archive/refs/heads/{branch}.zip"
    )

    print("→ Downloading latest version...")
    tmp_dir = tempfile.mkdtemp(prefix="hermes-update-")
    try:
        zip_path = os.path.join(tmp_dir, f"hermes-agent-{branch}.zip")
        urlretrieve(zip_url, zip_path)

        print("→ Extracting...")
        import stat as _stat
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Validate paths to prevent zip-slip (path traversal) AND reject
            # symlink members. A GitHub source ZIP for hermes-agent itself
            # should never contain symlinks — they'd point outside the
            # extracted tree and let an attacker who can compromise the
            # update mirror plant arbitrary files via the update path.
            tmp_dir_real = os.path.realpath(tmp_dir)
            for member in zf.infolist():
                member_path = os.path.realpath(os.path.join(tmp_dir, member.filename))
                if (
                    not member_path.startswith(tmp_dir_real + os.sep)
                    and member_path != tmp_dir_real
                ):
                    raise ValueError(
                        f"Zip-slip detected: {member.filename} escapes extraction directory"
                    )
                # Unix mode lives in the upper 16 bits of external_attr;
                # mask to the file-type bits.
                mode = (member.external_attr >> 16) & 0o170000
                if _stat.S_ISLNK(mode):
                    raise ValueError(
                        f"ZIP contains unsupported symlink member: {member.filename}"
                    )
            zf.extractall(tmp_dir)

        # GitHub ZIPs extract to hermes-agent-<branch>/
        extracted = os.path.join(tmp_dir, f"hermes-agent-{branch}")
        if not os.path.isdir(extracted):
            # Try to find it
            for d in os.listdir(tmp_dir):
                candidate = os.path.join(tmp_dir, d)
                if os.path.isdir(candidate) and d != "__MACOSX":
                    extracted = candidate
                    break

        # Copy updated files over existing installation, preserving venv/node_modules/.git
        preserve = {"venv", "node_modules", ".git", ".env"}
        update_count = 0
        for item in os.listdir(extracted):
            if item in preserve:
                continue
            src = os.path.join(extracted, item)
            dst = os.path.join(str(PROJECT_ROOT), item)
            if os.path.isdir(src):
                # Atomic-ish replace: never leave dst half-deleted if the copy
                # fails partway (the failure mode behind #49145 on Windows).
                _atomic_replace_dir(src, dst)
            else:
                shutil.copy2(src, dst)
            update_count += 1

        print(f"✓ Updated {update_count} items from ZIP")

    except Exception as e:
        print(f"✗ ZIP update failed: {e}")
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Clear stale bytecode after ZIP extraction
    removed = _clear_bytecode_cache(PROJECT_ROOT)
    if removed:
        print(
            f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}"
        )

    # Reinstall Python dependencies. Prefer .[all], but if one optional extra
    # breaks on this machine, keep base deps and reinstall the remaining extras
    # individually so update does not silently strip working capabilities.
    print("→ Updating Python dependencies...")

    from hermes_cli.managed_uv import ensure_uv, update_managed_uv

    # Keep managed uv current — runs `uv self update` if we already have one.
    update_managed_uv()

    uv_bin = ensure_uv()

    pip_cmd = [sys.executable, "-m", "pip"]
    if not uv_bin:
        uv_bin = _ensure_uv_for_termux(pip_cmd)
    if uv_bin:
        uv_env = {**os.environ, "VIRTUAL_ENV": str(PROJECT_ROOT / "venv")}
        if _is_termux_env(uv_env):
            uv_env.pop("PYTHONPATH", None)
            uv_env.pop("PYTHONHOME", None)
        _install_python_dependencies_with_optional_fallback([uv_bin, "pip"], env=uv_env)
    else:
        # Use sys.executable to explicitly call the venv's pip module,
        # avoiding PEP 668 'externally-managed-environment' errors on Debian/Ubuntu.
        # Some environments lose pip inside the venv; bootstrap it back with
        # ensurepip before trying the editable install.
        try:
            subprocess.run(
                pip_cmd + ["--version"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                cwd=PROJECT_ROOT,
                check=True,
            )
        _install_python_dependencies_with_optional_fallback(pip_cmd)

    _update_node_dependencies()
    _build_web_ui(PROJECT_ROOT / "web")

    # Sync skills
    try:
        from tools.skills_sync import sync_skills

        print("→ Syncing bundled skills...")
        result = sync_skills(quiet=True)
        if result["copied"]:
            print(f"  + {len(result['copied'])} new: {', '.join(result['copied'])}")
        if result.get("updated"):
            print(
                f"  ↑ {len(result['updated'])} updated: {', '.join(result['updated'])}"
            )
        if result.get("user_modified"):
            print(f"  ~ {len(result['user_modified'])} user-modified (kept)")
            print(
                "    → see them: hermes skills list-modified  "
                "(diff/reset to resume updates)"
            )
        if result.get("cleaned"):
            print(f"  − {len(result['cleaned'])} removed from manifest")
        if not result["copied"] and not result.get("updated"):
            print("  ✓ Skills are up to date")
    except Exception:
        pass

    # Seed the model-catalog disk cache from the freshly-unpacked checkout
    # (same rationale as the git-pull path in _cmd_update_impl). Non-fatal.
    try:
        from hermes_cli.model_catalog import seed_cache_from_checkout

        if seed_cache_from_checkout(PROJECT_ROOT):
            print("  ✓ Model catalog cache refreshed from checkout")
    except Exception as e:
        logger.debug("Model catalog seed during zip update failed: %s", e)

    print()
    print("✓ Update complete!")
    try:
        _print_curator_first_run_notice()
    except Exception as e:
        logger.debug("Curator first-run notice failed: %s", e)
    try:
        _print_curator_recent_run_notice()
    except Exception as e:
        logger.debug("Curator recent-run notice failed: %s", e)
    _kill_stale_dashboard_processes()


def _stash_local_changes_if_needed(git_cmd: list[str], cwd: Path) -> Optional[str]:
    status = subprocess.run(
        git_cmd + ["status", "--porcelain"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    if not status.stdout.strip():
        return None

    # If the index has unmerged entries (e.g. from an interrupted merge/rebase),
    # git stash will fail with "needs merge / could not write index".  Clear the
    # conflict state with `git reset` so the stash can proceed.  Working-tree
    # changes are preserved; only the index conflict markers are dropped.
    unmerged = subprocess.run(
        git_cmd + ["ls-files", "--unmerged"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if unmerged.stdout.strip():
        print("→ Clearing unmerged index entries from a previous conflict...")
        subprocess.run(git_cmd + ["reset"], cwd=cwd, capture_output=True)

    from datetime import datetime, timezone

    stash_name = datetime.now(timezone.utc).strftime(
        "hermes-update-autostash-%Y%m%d-%H%M%S"
    )
    print("→ Local changes detected — stashing before update...")
    subprocess.run(
        git_cmd + ["stash", "push", "--include-untracked", "-m", stash_name],
        cwd=cwd,
        check=True,
    )
    stash_ref = subprocess.run(
        git_cmd + ["rev-parse", "--verify", "refs/stash"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return stash_ref


def _resolve_stash_selector(
    git_cmd: list[str], cwd: Path, stash_ref: str
) -> Optional[str]:
    stash_list = subprocess.run(
        git_cmd + ["stash", "list", "--format=%gd %H"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in stash_list.stdout.splitlines():
        selector, _, commit = line.partition(" ")
        if commit.strip() == stash_ref:
            return selector.strip()
    return None


def _print_stash_cleanup_guidance(
    stash_ref: str, stash_selector: Optional[str] = None
) -> None:
    print(
        "  Check `git status` first so you don't accidentally reapply the same change twice."
    )
    print("  Find the saved entry with: git stash list --format='%gd %H %s'")
    if stash_selector:
        print(f"  Remove it with: git stash drop {stash_selector}")
    else:
        print(
            f"  Look for commit {stash_ref}, then drop its selector with: git stash drop stash@{{N}}"
        )


def _restore_stashed_changes(
    git_cmd: list[str],
    cwd: Path,
    stash_ref: str,
    prompt_user: bool = False,
    input_fn=None,
) -> bool:
    if prompt_user:
        print()
        print("⚠ Local changes were stashed before updating.")
        print(
            "  Restoring them may reapply local customizations onto the updated codebase."
        )
        print("  Review the result afterward if Hermes behaves unexpectedly.")
        print("Restore local changes now? [Y/n]")
        if input_fn is not None:
            response = input_fn("Restore local changes now? [Y/n]", "y")
        else:
            response = input().strip().lower()
        if response not in {"", "y", "yes"}:
            print("Skipped restoring local changes.")
            print("Your changes are still preserved in git stash.")
            print(f"Restore manually with: git stash apply {stash_ref}")
            return False

    print("→ Restoring local changes...")
    restore = subprocess.run(
        git_cmd + ["stash", "apply", stash_ref],
        cwd=cwd,
        capture_output=True,
        text=True,
    )

    # Check for unmerged (conflicted) files — can happen even when returncode is 0
    unmerged = subprocess.run(
        git_cmd + ["diff", "--name-only", "--diff-filter=U"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    has_conflicts = bool(unmerged.stdout.strip())

    if restore.returncode != 0 or has_conflicts:
        print("✗ Update pulled new code, but restoring local changes hit conflicts.")
        if restore.stdout.strip():
            print(restore.stdout.strip())
        if restore.stderr.strip():
            print(restore.stderr.strip())

        # Show which files conflicted
        conflicted_files = unmerged.stdout.strip()
        if conflicted_files:
            print("\nConflicted files:")
            for f in conflicted_files.splitlines():
                print(f"  • {f}")

        print("\nYour stashed changes are preserved — nothing is lost.")
        print(f"  Stash ref: {stash_ref}")

        # Always reset to clean state — leaving conflict markers in source
        # files makes hermes completely unrunnable (SyntaxError on import).
        # The user's changes are safe in the stash for manual recovery.
        subprocess.run(
            git_cmd + ["reset", "--hard", "HEAD"],
            cwd=cwd,
            capture_output=True,
        )
        print("Working tree reset to clean state.")
        print(f"Restore your changes later with: git stash apply {stash_ref}")
        # Don't sys.exit — the code update itself succeeded, only the stash
        # restore had conflicts.  Let cmd_update continue with pip install,
        # skill sync, and gateway restart.
        return False

    stash_selector = _resolve_stash_selector(git_cmd, cwd, stash_ref)
    if stash_selector is None:
        print(
            "⚠ Local changes were restored, but Hermes couldn't find the stash entry to drop."
        )
        print(
            "  The stash was left in place. You can remove it manually after checking the result."
        )
        _print_stash_cleanup_guidance(stash_ref)
    else:
        drop = subprocess.run(
            git_cmd + ["stash", "drop", stash_selector],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if drop.returncode != 0:
            print(
                "⚠ Local changes were restored, but Hermes couldn't drop the saved stash entry."
            )
            if drop.stdout.strip():
                print(drop.stdout.strip())
            if drop.stderr.strip():
                print(drop.stderr.strip())
            print(
                "  The stash was left in place. You can remove it manually after checking the result."
            )
            _print_stash_cleanup_guidance(stash_ref, stash_selector)

    print("⚠ Local changes were restored on top of the updated codebase.")
    print("  Review `git diff` / `git status` if Hermes behaves unexpectedly.")
    return True


def _discard_stashed_changes(
    git_cmd: list[str],
    cwd: Path,
    stash_ref: str,
) -> bool:
    """Throw away a stash created before an update, without applying it.

    Used only on a NON-interactive update when the user has set
    ``updates.non_interactive_local_changes: discard`` — i.e. they've opted out
    of keeping local source edits on this machine. Drops the stash entry
    instead of re-applying it, so the working tree stays clean at the freshly
    pulled HEAD. Unlike ``git reset --hard`` + ``git clean -fd``, this only
    affects what was stashed (tracked changes + the untracked files we
    explicitly captured) — ignored paths like node_modules/venv/build outputs
    are never touched, since they were never stashed.

    Returns True if the stash was dropped, False on a git failure (in which
    case the stash is left in place for safety).
    """
    stash_selector = _resolve_stash_selector(git_cmd, cwd, stash_ref)
    if stash_selector is None:
        print(
            "⚠ Configured to discard local changes on non-interactive update, "
            "but Hermes couldn't find the stash entry to drop."
        )
        _print_stash_cleanup_guidance(stash_ref)
        return False

    drop = subprocess.run(
        git_cmd + ["stash", "drop", stash_selector],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if drop.returncode != 0:
        print(
            "⚠ Configured to discard local changes, but Hermes couldn't drop "
            "the saved stash entry."
        )
        if drop.stderr.strip():
            print(f"  {drop.stderr.strip().splitlines()[0]}")
        _print_stash_cleanup_guidance(stash_ref, stash_selector)
        return False

    print("→ Discarded local source changes (updates.non_interactive_local_changes=discard).")
    return True


# =========================================================================
# Fork detection and upstream management for `hermes update`
# =========================================================================

OFFICIAL_REPO_URLS = {
    "https://github.com/NousResearch/hermes-agent.git",
    "git@github.com:NousResearch/hermes-agent.git",
    "https://github.com/NousResearch/hermes-agent",
    "git@github.com:NousResearch/hermes-agent",
}
OFFICIAL_REPO_URL = "https://github.com/NousResearch/hermes-agent.git"
SKIP_UPSTREAM_PROMPT_FILE = ".skip_upstream_prompt"


def _get_origin_url(git_cmd: list[str], cwd: Path) -> Optional[str]:
    """Get the URL of the origin remote, or None if not set."""
    try:
        result = subprocess.run(
            git_cmd + ["remote", "get-url", "origin"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _is_fork(origin_url: Optional[str]) -> bool:
    """Check if the origin remote points to a fork (not the official repo)."""
    if not origin_url:
        return False
    # Normalize URL for comparison (strip trailing .git if present)
    normalized = origin_url.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    for official in OFFICIAL_REPO_URLS:
        official_normalized = official.rstrip("/")
        if official_normalized.endswith(".git"):
            official_normalized = official_normalized[:-4]
        if normalized == official_normalized:
            return False
    return True


def _has_upstream_remote(git_cmd: list[str], cwd: Path) -> bool:
    """Check if an 'upstream' remote already exists."""
    try:
        result = subprocess.run(
            git_cmd + ["remote", "get-url", "upstream"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _add_upstream_remote(git_cmd: list[str], cwd: Path) -> bool:
    """Add the official repo as the 'upstream' remote. Returns True on success."""
    try:
        result = subprocess.run(
            git_cmd + ["remote", "add", "upstream", OFFICIAL_REPO_URL],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _count_commits_between(git_cmd: list[str], cwd: Path, base: str, head: str) -> int:
    """Count commits on `head` that are not on `base`. Returns -1 on error."""
    try:
        result = subprocess.run(
            git_cmd + ["rev-list", "--count", f"{base}..{head}"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return -1


def _should_skip_upstream_prompt() -> bool:
    """Check if user previously declined to add upstream."""
    from hermes_constants import get_hermes_home

    return (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).exists()


def _mark_skip_upstream_prompt():
    """Create marker file to skip future upstream prompts."""
    try:
        from hermes_constants import get_hermes_home

        (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).touch()
    except Exception:
        pass


def _sync_fork_with_upstream(git_cmd: list[str], cwd: Path) -> bool:
    """Attempt to push updated main to origin (sync fork).

    Returns True if push succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            git_cmd + ["push", "origin", "main", "--force-with-lease"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _sync_with_upstream_if_needed(git_cmd: list[str], cwd: Path) -> None:
    """Check if fork is behind upstream and sync if safe.

    This implements the fork upstream sync logic:
    - If upstream remote doesn't exist, ask user if they want to add it
    - Compare origin/main with upstream/main
    - If origin/main is strictly behind upstream/main, pull from upstream
    - Try to sync fork back to origin if possible
    """
    has_upstream = _has_upstream_remote(git_cmd, cwd)

    if not has_upstream:
        # Check if user previously declined
        if _should_skip_upstream_prompt():
            return

        # Ask user if they want to add upstream
        print()
        print("ℹ Your fork is not tracking the official Hermes repository.")
        print("  This means you may miss updates from NousResearch/hermes-agent.")
        print()
        try:
            response = (
                input("Add official repo as 'upstream' remote? [Y/n]: ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt):
            print()
            response = "n"

        if response in {"", "y", "yes"}:
            print("→ Adding upstream remote...")
            if _add_upstream_remote(git_cmd, cwd):
                print(
                    "  ✓ Added upstream: https://github.com/NousResearch/hermes-agent.git"
                )
                has_upstream = True
            else:
                print("  ✗ Failed to add upstream remote. Skipping upstream sync.")
                return
        else:
            print(
                "  Skipped. Run 'git remote add upstream https://github.com/NousResearch/hermes-agent.git' to add later."
            )
            _mark_skip_upstream_prompt()
            return

    # Fetch upstream main only. This sync compares upstream/main with
    # origin/main, so there's no reason to pull every upstream ref — and a bare
    # fetch drags in thousands of auto-generated branches.
    print()
    print("→ Fetching upstream...")
    try:
        subprocess.run(
            git_cmd + ["fetch", "upstream", "main", "--quiet"],
            cwd=cwd,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("  ✗ Failed to fetch upstream. Skipping upstream sync.")
        return

    # Compare origin/main with upstream/main
    origin_ahead = _count_commits_between(git_cmd, cwd, "upstream/main", "origin/main")
    upstream_ahead = _count_commits_between(
        git_cmd, cwd, "origin/main", "upstream/main"
    )

    if origin_ahead < 0 or upstream_ahead < 0:
        print("  ✗ Could not compare branches. Skipping upstream sync.")
        return

    # If origin/main has commits not on upstream, don't trample
    if origin_ahead > 0:
        print()
        print(f"ℹ Your fork has {origin_ahead} commit(s) not on upstream.")
        print("  Skipping upstream sync to preserve your changes.")
        print("  If you want to merge upstream changes, run:")
        print("    git pull upstream main")
        return

    # If upstream is not ahead, fork is up to date
    if upstream_ahead == 0:
        print("  ✓ Fork is up to date with upstream")
        return

    # origin/main is strictly behind upstream/main (can fast-forward)
    print()
    print(f"→ Fork is {upstream_ahead} commit(s) behind upstream")
    print("→ Pulling from upstream...")

    try:
        subprocess.run(
            git_cmd + ["pull", "--ff-only", "upstream", "main"],
            cwd=cwd,
            check=True,
        )
    except subprocess.CalledProcessError:
        print(
            "  ✗ Failed to pull from upstream. You may need to resolve conflicts manually."
        )
        return

    print("  ✓ Updated from upstream")

    # Try to sync fork back to origin
    print("→ Syncing fork...")
    if _sync_fork_with_upstream(git_cmd, cwd):
        print("  ✓ Fork synced with upstream")
    else:
        print(
            "  ℹ Got updates from upstream but couldn't push to fork (no write access?)"
        )
        print("    Your local repo is updated, but your fork on GitHub may be behind.")


def _invalidate_update_cache():
    """Delete the update-check cache for ALL profiles so no banner
    reports a stale "commits behind" count after a successful update.

    The git repo is shared across profiles — when one profile runs
    ``hermes update``, every profile is now current.
    """
    homes = []
    # Default profile home (Docker-aware — uses /opt/data in Docker)
    from hermes_constants import get_default_hermes_root

    default_home = get_default_hermes_root()
    homes.append(default_home)
    # Named profiles under <root>/profiles/
    profiles_root = default_home / "profiles"
    if profiles_root.is_dir():
        for entry in profiles_root.iterdir():
            if entry.is_dir():
                homes.append(entry)
    for home in homes:
        try:
            cache_file = home / ".update_check"
            if cache_file.exists():
                cache_file.unlink()
        except Exception:
            pass


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


# Install-scoped breadcrumb dropped right before ``hermes update`` mutates the
# venv and cleared only after the dependency install verifies clean.  If a user
# kills the update mid-install (Ctrl-C, terminal close, WSL OOM), the marker
# survives and the next ``hermes`` launch finishes the install instead of
# limping along on a half-built venv (e.g. pip wiped, a core dep like Pillow
# never landed).  Lives next to the venv (not under $HERMES_HOME) because the
# venv is shared across all profiles, so a single marker covers every profile.
def _update_marker_path() -> Path:
    return PROJECT_ROOT / ".update-incomplete"


def _write_update_incomplete_marker() -> None:
    """Drop the interrupted-install breadcrumb. Never raises."""
    try:
        _update_marker_path().write_text(
            f"started={_time.time()}\npid={os.getpid()}\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.debug("Could not write update-incomplete marker: %s", exc)


def _clear_update_incomplete_marker() -> None:
    """Remove the interrupted-install breadcrumb. Never raises."""
    try:
        _update_marker_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Could not clear update-incomplete marker: %s", exc)


def _recover_from_interrupted_install() -> None:
    """Finish a dependency install that a prior ``hermes update`` left half-done.

    Triggered on launch when ``.update-incomplete`` is present — meaning the
    code was pulled but the dep install was killed before it verified clean.
    Unconditionally bootstraps pip via ``ensurepip`` (a killed ``pip install``
    can wipe pip from the venv entirely, which blocks the venv from recovering
    on its own), then re-runs the editable ``.[all]`` install + core-dependency
    verification, then clears the marker.

    Never raises: a recovery failure must not block launch.  If it can't
    self-heal it prints the one-line manual command and leaves the marker so
    the next launch tries again.

    Concurrency: the marker lives next to the shared venv, so a gateway start
    plus a CLI launch (or two profiles starting at once) can both see it.  An
    ``O_EXCL`` lockfile ensures only one process runs the reinstall; the
    others skip and let the winner clear the marker.

    Output: everything — our status lines AND the streamed pip/uv install
    (which inherits fd 1) — is routed to stderr.  Launches whose stdout is a
    protocol stream (``hermes acp`` speaks JSON-RPC on stdout) must never get
    install noise on stdout.
    """
    if not _update_marker_path().exists():
        return

    # Skip in managed/Docker installs and on PyPI installs with no git checkout:
    # those don't run the source-tree update path, so a stray marker is not ours
    # to act on. Just clear it.
    if not (PROJECT_ROOT / "pyproject.toml").is_file():
        _clear_update_incomplete_marker()
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

    # Windows self-lock guard: if hermes.exe is the launcher that spawned
    # this Python process, any attempt to pip-install will fail with
    # "拒绝访问 / WinError 32" because the running .exe cannot be replaced.
    # Rather than entering the permanent retry loop described in issue
    # #45542, clear the marker and give the user an offline recovery command.
    if _is_windows():
        scripts_dir = _venv_scripts_dir()
        if scripts_dir is not None:
            shims = _hermes_exe_shims(scripts_dir)
            if shims:
                _shim_set: set[str] = set()
                for _s in shims:
                    try:
                        _shim_set.add(str(_s.resolve()).lower())
                    except OSError:
                        _shim_set.add(str(_s).lower())
                try:
                    import psutil
                    _me = psutil.Process()
                    for _anc in [_me] + list(_me.parents()):
                        try:
                            _anc_exe = _anc.exe()
                            _anc_norm = str(Path(_anc_exe).resolve()).lower()
                        except Exception:
                            continue
                        if _anc_norm in _shim_set:
                            print(
                                "✗ Hermes is running from the binary that "
                                "needs to be replaced — the auto-recovery "
                                "cannot overwrite a running executable."
                            )
                            print(
                                "  Restart Hermes from a different terminal, "
                                "then run the manual recovery command below:"
                            )
                            print(f'    cd /d "{PROJECT_ROOT}"')
                            print(
                                f'    "{sys.executable}" -m pip install '
                                '-e ".[all]"'
                            )
                            _clear_update_incomplete_marker()
                            try:
                                lock_path.unlink()
                            except OSError:
                                pass
                            return
                except Exception:
                    pass  # psutil is best-effort; fall through to install

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

        print(
            "⚠ A previous `hermes update` was interrupted mid-install — "
            "finishing dependency installation now..."
        )

        try:
            from hermes_cli.managed_uv import ensure_uv

            # Always bootstrap pip first: a killed install can leave the venv with
            # no pip module at all, and uv may also be gone. ensurepip restores a
            # known-good pip so at least the plain-pip path below can proceed.
            try:
                subprocess.run(
                    [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                )
            except Exception as exc:
                logger.debug("ensurepip during install recovery failed: %s", exc)

            uv_bin = ensure_uv()
            if uv_bin:
                uv_env = {**os.environ, "VIRTUAL_ENV": str(PROJECT_ROOT / "venv")}
                if _is_termux_env(uv_env):
                    uv_env.pop("PYTHONPATH", None)
                    uv_env.pop("PYTHONHOME", None)
                _install_python_dependencies_with_optional_fallback(
                    [uv_bin, "pip"],
                    env=uv_env,
                    group="termux-all" if _is_termux_env(uv_env) else "all",
                )
            else:
                _install_python_dependencies_with_optional_fallback(
                    [sys.executable, "-m", "pip"],
                    group="termux-all" if _is_termux_env() else "all",
                )

            _clear_update_incomplete_marker()
            print("✓ Dependency installation recovered — your install is healthy again.")
        except Exception as exc:
            # Leave the marker in place so the next launch retries. Give the user
            # the exact manual recovery command in the meantime.
            logger.debug("Interrupted-install recovery failed: %s", exc)
            print("✗ Could not auto-recover the interrupted install.")
            print("  Recover manually with:")
            print(f"    cd {PROJECT_ROOT}")
            print(f"    {sys.executable} -m ensurepip --upgrade")
            print(f"    {sys.executable} -m pip install -e '.[all]'")
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
    scripts = venv_dir / ("Scripts" if _is_windows() else "bin")
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


def _detect_concurrent_hermes_instances(
    scripts_dir: Path, *, exclude_pid: int | None = None
) -> list[tuple[int, str]]:
    """Find other live processes whose .exe is one of our entry-point shims.

    Windows blocks DELETE/REPLACE on a running .exe — and even RENAME on the
    same .exe when another process opened it without ``FILE_SHARE_DELETE``.
    The Hermes Desktop Electron app spawns ``hermes.EXE`` as a backend child,
    so during ``hermes update`` the user-invoked process and the desktop's
    child both hold the same file. The quarantine rename then fails with
    ``[WinError 32]`` and uv inherits the lock.

    This helper enumerates processes whose ``exe`` matches one of the venv's
    shims (``hermes.exe`` / ``hermes-gateway.exe``) and returns ``(pid,
    process_name)`` pairs. The caller's own PID and its entire ancestor
    chain are excluded so the running ``hermes update`` invocation never
    reports itself — this matters on Windows where the setuptools .exe
    launcher (``hermes.exe``) is a separate process from the Python
    interpreter it loads (``python.exe``).

    Returns an empty list off-Windows, on missing psutil, or when no other
    instances exist. Never raises — process enumeration is best-effort.
    """
    if not _is_windows():
        return []

    try:
        import psutil
    except Exception:
        return []

    # Resolve every shim path to its canonical form once for cheap comparison.
    shim_paths: set[str] = set()
    for shim in _hermes_exe_shims(scripts_dir):
        try:
            shim_paths.add(str(shim.resolve()).lower())
        except OSError:
            shim_paths.add(str(shim).lower())
    if not shim_paths:
        return []

    # Build a set of PIDs to exclude: the Python process itself plus every
    # ancestor whose executable is one of our shims. On Windows the
    # setuptools-generated hermes.exe launcher is a separate native process
    # that spawns python.exe (the interpreter that runs our code).
    # os.getpid() returns the Python PID, but the launcher (which holds the
    # file lock) is the parent. Without excluding it, every ``hermes update``
    # reports its own launcher as a concurrent instance — a false positive
    # (issues #29341, #34795).
    #
    # Two robustness points learned from the field:
    #   1. Use ``proc.parents()`` — it returns the WHOLE ancestor list in one
    #      call. The earlier per-hop ``current.parent()`` loop bailed on the
    #      first psutil error (AccessDenied/NoSuchProcess is common on Windows
    #      across session/elevation boundaries), leaving the launcher shim in
    #      the candidate set and re-triggering the false positive.
    #   2. Only exclude ancestors whose exe is itself a shim. A genuine second
    #      hermes.exe sitting *under* a non-Hermes parent (e.g. a Hermes
    #      Desktop backend child) must still be flagged, so we don't blanket-
    #      exclude unrelated ancestors like the shell or terminal.
    # Broad ``except Exception`` guards against partially-stubbed psutil in
    # unit tests; this helper is documented as "never raises".
    if exclude_pid is not None:
        exclude_pids: set[int] = {int(exclude_pid)}
    else:
        exclude_pids = {os.getpid()}
    try:
        seed = next(iter(exclude_pids))
        try:
            ancestors = psutil.Process(seed).parents()
        except Exception:
            ancestors = []
        for ancestor in ancestors:
            try:
                anc_exe = ancestor.exe()
            except Exception:
                continue
            if not anc_exe:
                continue
            try:
                anc_norm = str(Path(anc_exe).resolve()).lower()
            except (OSError, ValueError):
                anc_norm = str(anc_exe).lower()
            if anc_norm in shim_paths:
                try:
                    exclude_pids.add(int(ancestor.pid))
                except Exception:
                    continue
    except Exception:
        pass

    matches: list[tuple[int, str]] = []
    try:
        proc_iter = psutil.process_iter(["pid", "exe", "name"])
    except Exception:
        return []

    for proc in proc_iter:
        try:
            info = proc.info
        except Exception:
            continue
        pid = info.get("pid")
        exe = info.get("exe")
        if not exe or pid is None or pid in exclude_pids:
            continue
        try:
            exe_norm = str(Path(exe).resolve()).lower()
        except (OSError, ValueError):
            exe_norm = str(exe).lower()
        if exe_norm in shim_paths:
            name = info.get("name") or Path(exe).name
            matches.append((int(pid), str(name)))

    return matches


def _format_concurrent_instances_message(
    matches: list[tuple[int, str]], scripts_dir: Path
) -> str:
    """Build a human-readable explanation + remediation hint for the user."""
    shim = scripts_dir / "hermes.exe"
    lines = ["✗ Another hermes.exe is running:"]
    for pid, name in matches:
        lines.append(f"    PID {pid}  {name}")
    lines.append("")
    lines.append(f"  Updating now would fail to overwrite {shim} because")
    lines.append("  Windows blocks REPLACE on a running executable.")
    lines.append("")
    lines.append("  Close Hermes Desktop, exit any open `hermes` REPLs, and")
    lines.append("  stop the gateway (`hermes gateway stop`) before retrying.")
    lines.append("")
    if matches:
        pid_args = " ".join(f"/PID {pid}" for pid, _ in matches)
        lines.append("  If you've already closed everything and these PIDs are")
        lines.append("  stale, terminate them directly, then retry the update:")
        lines.append(f"      taskkill {pid_args} /F")
        lines.append("")
    lines.append("  Override with `hermes update --force` if you've already")
    lines.append("  confirmed those processes will not write to the venv.")
    return "\n".join(lines)


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


def _refresh_active_lazy_features() -> None:
    """Refresh lazy-installed backends after a code update.

    When pyproject.toml's ``[all]`` extra was slimmed down (May 2026), most
    optional backends moved to ``tools/lazy_deps.py`` and only install on
    first use. ``hermes update`` runs ``uv pip install -e .[all]`` which
    leaves those packages untouched — so if we bump a pin in
    :data:`LAZY_DEPS` (CVE response, transitive bug fix), users who already
    activated the backend keep the stale version forever.

    This function asks lazy_deps which features the user has previously
    activated and reinstalls them under the current pins. Features the
    user never enabled stay quiet — no churn for cold backends.

    Never raises. A failure here must not block the rest of the update.
    """
    try:
        from tools import lazy_deps
    except Exception as exc:
        logger.debug("Lazy refresh skipped (import failed): %s", exc)
        return

    try:
        active = lazy_deps.active_features()
    except Exception as exc:
        logger.debug("Lazy refresh skipped (active_features failed): %s", exc)
        return

    if not active:
        return

    print()
    print(f"→ Refreshing {len(active)} active lazy backend(s)...")

    try:
        results = lazy_deps.refresh_active_features(prompt=False)
    except Exception as exc:
        # refresh_active_features is documented as never-raise, but defend
        # the update flow against future regressions.
        print(f"  ⚠ Lazy refresh failed unexpectedly: {exc}")
        return

    refreshed = [f for f, s in results.items() if s == "refreshed"]
    current = [f for f, s in results.items() if s == "current"]
    failed = [(f, s) for f, s in results.items() if s.startswith("failed:")]
    skipped = [(f, s) for f, s in results.items() if s.startswith("skipped:")]

    if refreshed:
        print(f"  ↑ {len(refreshed)} refreshed: {', '.join(refreshed)}")
    if current:
        print(f"  ✓ {len(current)} already current")
    if skipped:
        # Most common reason: security.allow_lazy_installs=false. Show one
        # line so the user knows why; not an error.
        names = ", ".join(f for f, _ in skipped)
        reason = skipped[0][1].split(": ", 1)[-1]
        print(f"  · {len(skipped)} skipped ({reason}): {names}")
    if failed:
        for feature, status in failed:
            reason = status.split(": ", 1)[-1]
            # Clip noisy pip stderr to keep update output legible.
            if len(reason) > 200:
                reason = reason[:200] + "..."
            print(f"  ⚠ {feature} failed to refresh: {reason}")
        print("  Backends keep their previously-installed version; rerun")
        print("  `hermes update` once the upstream issue is resolved.")


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
                text=True,
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
        venv_root = Path(env["VIRTUAL_ENV"])
        scripts = venv_root / ("Scripts" if _is_windows() else "bin")
        candidate = scripts / ("python.exe" if _is_windows() else "python")
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


def _is_android_python() -> bool:
    return sys.platform == "android"


def _install_psutil_android_compat(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Install psutil on Android by patching upstream platform detection.

    psutil's setup currently gates Linux sources behind
    ``sys.platform.startswith('linux')``. On Termux Python reports
    ``sys.platform == 'android'``, so setup aborts with
    "platform android is not supported" despite compiling fine when using the
    Linux source path.

    We patch only the extracted build tree used for this install attempt;
    nothing is persisted in the repository.

    Stopgap: remove this once https://github.com/giampaolo/psutil/pull/2762
    merges and ships in a release. The standalone installer script uses the
    same shared helper and should be removed together.
    """
    import tempfile
    import urllib.request
    from hermes_cli.psutil_android import PSUTIL_URL, prepare_patched_psutil_sdist

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "psutil.tar.gz"
        urllib.request.urlretrieve(PSUTIL_URL, archive)
        src_root = prepare_patched_psutil_sdist(archive, tmp_path)

        _run_install_with_heartbeat(
            install_cmd_prefix + ["install", "--no-build-isolation", str(src_root)],
            env=env,
        )


def _ensure_uv_for_termux(pip_cmd: list[str]) -> str | None:
    """Best-effort uv bootstrap on Termux for faster update installs.

    The normal path (``ensure_uv()`` in managed_uv) installs the managed
    standalone uv into ``$HERMES_HOME/bin/uv``, but on Termux the official
    installer may not work (glibc vs bionic).  Prefer a uv already on PATH
    (e.g. ``pkg install uv``); only if there is none do we fall back to a
    wheel-only ``pip install uv`` so we never source-build the Rust crate.
    """
    from hermes_cli.managed_uv import resolve_uv

    existing = resolve_uv()
    if existing:
        return existing
    if not _is_termux_env():
        return None
    # A Termux-packaged uv lands on PATH but not in the managed bin dir, so
    # resolve_uv() misses it. Use it before pip, which has no Android wheel and
    # would otherwise build uv from source on a low-memory device.
    system_uv = shutil.which("uv")
    if system_uv:
        return system_uv
    try:
        print("  → Termux detected: trying to install uv for faster dependency updates...")
        result = subprocess.run(
            pip_cmd + ["install", "uv", "--only-binary", ":all:"],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            return None
    except Exception:
        pass
    # After pip install, check managed path first, then PATH
    return resolve_uv() or shutil.which("uv")


def _update_node_dependencies() -> None:
    from hermes_constants import find_node_executable, with_hermes_node_path

    npm = find_node_executable("npm")
    if not npm:
        return

    if not (PROJECT_ROOT / "package.json").exists():
        return

    # With a single workspace lockfile the root install would cover ALL
    # workspaces — but apps/desktop pulls in Electron as a devDependency,
    # and its postinstall downloads a ~200MB binary.  Most users don't
    # need desktop during `hermes update`, so we install root-only first
    # then add just the workspaces the CLI/TUI/web build actually requires.
    # Desktop deps are installed on demand by the desktop launcher
    # (see _desktop_build_needed).
    print("→ Updating Node.js dependencies...")
    extra_args = ["--no-fund", "--no-audit", "--progress=false"]

    nixos_env = with_hermes_node_path(_nixos_build_env())

    # Step 1: root install (no workspace recursion).
    # NOTE: capture_output=False here is deliberate (#18840) — optional
    # postinstall scripts (e.g. @askjo/camofox-browser's browser-binary fetch)
    # print download progress, and capturing it makes a long download look
    # hung. The chatty npm-deprecation noise during `hermes update` comes from
    # the *desktop* build, not this step; that one is captured to update.log.
    root_args = [*extra_args, "--workspaces=false"]
    root_result = _run_npm_install_deterministic(
        npm,
        PROJECT_ROOT,
        extra_args=tuple(root_args),
        capture_output=False,
        env=nixos_env,
    )
    if root_result.returncode != 0:
        print("  ⚠ npm install failed in repo root")
        stderr = (root_result.stderr or "").strip() if root_result.stderr else ""
        if stderr:
            print(f"    {stderr.splitlines()[-1]}")
        return

    # Step 2: install only the workspaces update needs (ui-tui, web).
    # --workspace selects specific workspaces; the rest (desktop) are skipped.
    ws_args = [*extra_args, "--workspace", "ui-tui", "--workspace", "web"]
    ws_result = _run_npm_install_deterministic(
        npm,
        PROJECT_ROOT,
        extra_args=tuple(ws_args),
        capture_output=False,
        env=nixos_env,
    )
    if ws_result.returncode == 0:
        print("  ✓ repo root + ui-tui, web workspaces (desktop skipped)")
    else:
        print("  ⚠ npm workspace install failed")
        stderr = (ws_result.stderr or "").strip() if ws_result.stderr else ""
        if stderr:
            print(f"    {stderr.splitlines()[-1]}")


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


def _log_only_write(text: str) -> None:
    """Write ``text`` to ``~/.hermes/logs/update.log`` only, never the terminal.

    During ``hermes update`` ``sys.stdout`` is an ``_UpdateOutputStream`` that
    mirrors to both the terminal and ``update.log``. Loud, low-signal
    subprocess output (npm installs, the Electron/vite build, the cua-driver
    installer's "Next steps" wall) should be captured and tucked into the log
    so failures stay debuggable, without flooding the user's terminal. This
    reaches past the mirroring stream straight to the underlying log handle.
    """
    if not text:
        return
    stream = sys.stdout
    log_file = getattr(stream, "_log", None)
    if log_file is None:
        return
    try:
        log_file.write(text if text.endswith("\n") else text + "\n")
        log_file.flush()
    except Exception:
        pass


def _run_logged_subprocess(cmd, *, cwd=None, env=None):
    """Run ``cmd`` capturing combined output into update.log (not the terminal).

    Returns the ``CompletedProcess`` (with ``stdout`` populated) so the caller
    can decide whether to surface the captured output on failure.
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _log_only_write(result.stdout or "")
    return result


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


def _cmd_update_check(branch: str = "main", *, branch_explicit: bool = False):
    """Implement ``hermes update --check``: fetch and report without installing.

    ``branch`` selects which branch the check compares against. Default is
    "main"; callers can pass another branch to ask "are there new commits
    on origin/<branch>?" without performing the update.

    ``branch_explicit`` is True iff the caller passed --branch on the CLI.
    PyPI installs can't honor non-default branches, so when this is True
    on a PyPI install we surface a one-line notice instead of silently
    dropping the flag.
    """
    from hermes_cli.config import detect_install_method
    method = detect_install_method(PROJECT_ROOT)
    if method == "docker":
        # Docker can't ``git fetch`` from within the container.  Surface the
        # same long-form ``docker pull`` guidance ``hermes update`` (apply
        # path) uses — telling the user to "reinstall via curl" or that
        # ".git is missing" would point them at the wrong remediation.
        from hermes_cli.config import format_docker_update_message
        print(format_docker_update_message())
        sys.exit(1)
    if method == "pip":
        from hermes_cli.config import recommended_update_command
        from hermes_cli.banner import check_via_pypi
        if branch_explicit and branch != "main":
            print(f"⚠ --branch is ignored for PyPI installs (would have checked '{branch}').")
        result = check_via_pypi()
        if result is None:
            print("✗ Could not reach PyPI to check for updates.")
            sys.exit(1)
        elif result == 0:
            print("✓ Already up to date.")
        else:
            print("⚕ Update available on PyPI.")
            print(f"  Run '{recommended_update_command()}' to install.")
        return

    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.exists():
        print("✗ Not a git repository — cannot check for updates.")
        sys.exit(1)

    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]

    # Fetch only the branch we compare against; prefer upstream as the canonical
    # reference. A bare `git fetch <remote>` pulls every ref, and this repo has
    # thousands of auto-generated branches, so scope the fetch to <branch>.
    # Note: upstream/<branch> may not exist for non-main branches (a fork's
    # bb/gui has no upstream counterpart), so when the caller picks a
    # non-default branch we skip the upstream probe and use origin directly.
    # Installer checkouts are shallow (`git clone --depth 1`). A plain
    # `git fetch` would unshallow the repo (dragging in the whole history —
    # the exact cost the shallow clone avoided) and the rev-list count below
    # would then report a huge bogus "behind" number. Detect shallow up front:
    # fetch with --depth 1 to preserve the boundary and report presence-only.
    is_shallow = (
        subprocess.run(
            git_cmd + ["rev-parse", "--is-shallow-repository"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == "true"
    )
    depth_args = ["--depth", "1"] if is_shallow else []

    if branch == "main":
        print("→ Fetching from upstream...")
        fetch_result = subprocess.run(
            git_cmd + ["fetch"] + depth_args + ["upstream", branch],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if fetch_result.returncode != 0:
            # Fallback to origin if upstream doesn't exist
            print("→ Fetching from origin...")
            fetch_result = subprocess.run(
                git_cmd + ["fetch"] + depth_args + ["origin", branch],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            upstream_exists = False
            compare_branch = f"origin/{branch}"
        else:
            upstream_exists = True
            compare_branch = f"upstream/{branch}"
    else:
        # Non-default branch: compare against origin/<branch> directly.
        print("→ Fetching from origin...")
        fetch_result = subprocess.run(
            git_cmd + ["fetch"] + depth_args + ["origin", branch],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        upstream_exists = False
        compare_branch = f"origin/{branch}"

    if fetch_result.returncode != 0:
        stderr = fetch_result.stderr.strip()
        if "Could not resolve host" in stderr or "unable to access" in stderr:
            print("✗ Network error — cannot reach the remote repository.")
        elif "Authentication failed" in stderr or "could not read Username" in stderr:
            print("✗ Authentication failed — check your git credentials or SSH key.")
        else:
            print("✗ Failed to fetch.")
            if stderr:
                print(f"  {stderr.splitlines()[0]}")
        sys.exit(1)

    # Verify the compare ref actually exists before asking rev-list about it.
    # Without this, `git rev-list HEAD..origin/<bogus> --count` exits 128 and
    # (with check=True) raises CalledProcessError, surfacing a Python
    # traceback. Friendlier to detect-and-report.
    verify_result = subprocess.run(
        git_cmd + ["rev-parse", "--verify", "--quiet", compare_branch],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if verify_result.returncode != 0:
        print(f"✗ Branch '{branch}' not found on {compare_branch.split('/', 1)[0]}.")
        sys.exit(1)

    if is_shallow:
        # No history to count across the shallow boundary. Compare tip SHAs and
        # report presence-only (mirrors the banner's _check_via_local_git).
        head_sha = subprocess.run(
            git_cmd + ["rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        ).stdout.strip()
        target_sha = subprocess.run(
            git_cmd + ["rev-parse", compare_branch],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        ).stdout.strip()
        if head_sha and target_sha and head_sha == target_sha:
            print("✓ Already up to date.")
        else:
            print(f"⚕ Update available (behind {compare_branch}).")
            from hermes_cli.config import recommended_update_command

            print(f"  Run '{recommended_update_command()}' to install.")
        return

    rev_result = subprocess.run(
        git_cmd + ["rev-list", f"HEAD..{compare_branch}", "--count"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    behind = int(rev_result.stdout.strip())

    if behind == 0:
        print("✓ Already up to date.")
    else:
        commits_word = "commit" if behind == 1 else "commits"
        print(f"⚕ Update available: {behind} {commits_word} behind {compare_branch}.")
        from hermes_cli.config import recommended_update_command

        print(f"  Run '{recommended_update_command()}' to install.")


def _ensure_fhs_path_guard() -> None:
    """Ensure /usr/local/bin is on PATH for RHEL-family root non-login shells.

    Mirrors the post-symlink probe added to ``scripts/install.sh`` so that
    existing FHS-layout root installs on RHEL/CentOS/Rocky/Alma 8+ get
    repaired on ``hermes update`` without requiring a reinstall.  The
    installer's assumption that ``/usr/local/bin`` is on PATH for every
    standard shell breaks on those distros in non-login interactive shells
    (su, sudo -s, tmux panes, some web terminals): /etc/bashrc doesn't
    add /usr/local/bin and /root/.bash_profile doesn't either.  Symptom:
    ``hermes`` prints ``command not found`` even though the symlink lives
    at /usr/local/bin/hermes.

    Silent no-op on: non-Linux, non-root, non-FHS installs, and any system
    where ``bash -i -c 'command -v hermes'`` already resolves.  Idempotent.
    """
    if sys.platform != "linux":
        return
    try:
        if os.geteuid() != 0:  # windows-footgun: ok — Linux FHS helper, guarded by sys.platform == "linux" above + AttributeError catch
            return
    except AttributeError:
        return
    # Only act when this is actually an FHS-layout install (command link at
    # /usr/local/bin/hermes, code at /usr/local/lib/hermes-agent).
    fhs_link = Path("/usr/local/bin/hermes")
    if not fhs_link.is_symlink() and not fhs_link.exists():
        return

    # Probe a fresh non-login interactive bash the way the user will use it.
    # ``bash -i -c`` sources ~/.bashrc but NOT ~/.bash_profile or /etc/profile,
    # which is the exact scenario where RHEL root loses /usr/local/bin.
    home = os.environ.get("HOME") or "/root"
    try:
        probe = subprocess.run(
            [
                "env",
                "-i",
                f"HOME={home}",
                f"TERM={os.environ.get('TERM', 'dumb')}",
                "bash",
                "-i",
                "-c",
                "command -v hermes",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return  # no bash or probe hung — don't block update on this
    if probe.returncode == 0:
        return  # already on PATH, nothing to do

    path_line = 'export PATH="/usr/local/bin:$PATH"'
    path_comment = (
        "# Hermes Agent — ensure /usr/local/bin is on PATH " "(RHEL non-login shells)"
    )
    wrote_any = False
    for candidate in (".bashrc", ".bash_profile"):
        cfg = Path(home) / candidate
        if not cfg.is_file():
            continue
        try:
            existing = cfg.read_text(errors="replace")
        except OSError:
            continue
        # Idempotency: skip if any uncommented PATH= line already references
        # /usr/local/bin.  Mirrors the grep pattern used by install.sh.
        already_guarded = any(
            "/usr/local/bin" in line
            and "PATH" in line
            and not line.lstrip().startswith("#")
            for line in existing.splitlines()
        )
        if already_guarded:
            continue
        try:
            with cfg.open("a", encoding="utf-8") as f:
                f.write("\n" + path_comment + "\n" + path_line + "\n")
        except OSError as e:
            print(f"  ⚠ Could not update {cfg}: {e}")
            continue
        print(f"  ✓ Added /usr/local/bin to PATH in {cfg}")
        wrote_any = True
    if wrote_any:
        print("    (reload your shell or run 'source ~/.bashrc' to pick it up)")


def _run_pre_update_backup(args) -> None:
    """Create a full zip backup of HERMES_HOME before running the update.

    Gated on ``updates.pre_update_backup`` in config (default false).  Off
    by default because the zip can add minutes to every update on large
    HERMES_HOME directories.  The ``--backup`` flag on ``hermes update``
    opts in for a single run; ``--no-backup`` forces it off when config
    has it enabled.  Never raises — a backup failure should not block the
    update itself.
    """
    # CLI flags win over config.  --no-backup beats --backup if both are set.
    if getattr(args, "no_backup", False):
        print("◆ Pre-update backup: skipped (--no-backup)")
        print()
        return

    force_backup = bool(getattr(args, "backup", False))

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Could not load config for pre-update backup: %s", exc
        )
        cfg = {}

    updates_cfg = cfg.get("updates", {}) if isinstance(cfg, dict) else {}
    # The default config ships with ``pre_update_backup: false`` (see
    # ``hermes_cli/config.py``). Fall back to false if the key is missing
    # so the default behaviour matches the shipped config: zipping a large
    # HERMES_HOME can add minutes to every update. Users who want the
    # #48200 safety net opt in via the config knob or ``--backup``.
    enabled = updates_cfg.get("pre_update_backup", False)
    keep = updates_cfg.get("backup_keep", 5)

    if not enabled and not force_backup:
        # Silent by default — the backup is off, most users don't need to
        # hear about it on every update.  They can opt in via --backup
        # or by flipping the config knob.
        return

    try:
        from hermes_cli.backup import create_pre_update_backup
    except Exception as exc:
        print(
            f"⚠ Pre-update backup: could not load backup module ({exc}); continuing update."
        )
        print()
        return

    print("◆ Creating pre-update backup...")
    t0 = _time.monotonic()
    try:
        out_path = create_pre_update_backup(keep=int(keep))
    except Exception as exc:  # defensive — helper already swallows, but just in case
        print(f"  ⚠ Backup failed: {exc}")
        print("  Continuing with update.")
        print()
        return

    elapsed = _time.monotonic() - t0

    if out_path is None:
        print("  ⚠ Backup skipped (no files found or write failed); continuing update.")
        print()
        return

    try:
        size_bytes = out_path.stat().st_size
    except OSError:
        size_bytes = 0

    # Human-readable size
    size_str = f"{size_bytes} B"
    for unit in ("KB", "MB", "GB"):
        if size_bytes < 1024:
            break
        size_bytes /= 1024
        size_str = f"{size_bytes:.1f} {unit}"

    # Render path using display_hermes_home so the user sees ~/.hermes/...
    try:
        from hermes_constants import get_hermes_home, display_hermes_home

        home = get_hermes_home()
        try:
            display_path = f"{display_hermes_home()}/{out_path.relative_to(home)}"
        except ValueError:
            display_path = str(out_path)
    except Exception:
        display_path = str(out_path)

    print(f"  Saved:    {display_path} ({size_str}, {elapsed:.1f}s)")
    print(f"  Restore:  hermes import {out_path}")
    print("  Disable:  omit --backup (backups are off by default)")
    print("            set updates.pre_update_backup: false in config.yaml")
    print()


def _write_update_planned_stop_marker(profile_path: Path, pid: int) -> bool:
    """Write a planned-stop marker into a specific profile home."""
    try:
        from datetime import timezone

        from gateway.status import _get_process_start_time
        from utils import atomic_json_write

        record = {
            "target_pid": pid,
            "target_start_time": _get_process_start_time(pid),
            "stopper_pid": os.getpid(),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json_write(
            Path(profile_path) / ".gateway-planned-stop.json",
            record,
            indent=None,
            separators=(",", ":"),
        )
        return True
    except (OSError, PermissionError):
        return False


def _wait_for_windows_update_gateway_exit(
    pids: list[int], *, timeout: float
) -> set[int]:
    """Wait for the given gateway PIDs to exit, returning survivors."""
    if not pids:
        return set()

    from gateway.status import _pid_exists

    remaining = set(pids)
    deadline = _time.monotonic() + max(timeout, 0.0)
    while remaining and _time.monotonic() < deadline:
        for pid in list(remaining):
            try:
                if not _pid_exists(pid):
                    remaining.discard(pid)
            except Exception:
                remaining.discard(pid)
        if remaining:
            _time.sleep(0.25)

    survivors: set[int] = set()
    for pid in remaining:
        try:
            if _pid_exists(pid):
                survivors.add(pid)
        except Exception:
            pass
    return survivors


def _venv_core_imports_healthy() -> tuple[bool, str]:
    """Probe the project venv for the core imports the backend needs to boot.

    Runs a tiny import check inside the venv interpreter (NOT this process —
    ``hermes update`` may be driven by a different Python). Catches the
    half-updated-venv state: git checkout current but a dependency sync that
    failed or was killed partway (e.g. Windows access-denied on a loaded
    .pyd), leaving imports like ``fastapi``'s new transitive deps missing.
    Without this probe, ``hermes update`` on a current checkout prints
    "Already up to date!" and returns without ever re-syncing dependencies —
    the user's install stays broken no matter how many times they update
    (ryanc's incident, July 2026).

    Returns ``(healthy, detail)``. Never raises; unknown states report
    healthy so a probe failure can't force needless reinstalls.
    """
    venv_dir = PROJECT_ROOT / "venv"
    python_name = "python.exe" if _is_windows() else "python"
    bin_dir = "Scripts" if _is_windows() else "bin"
    venv_python = venv_dir / bin_dir / python_name
    if not venv_python.exists():
        # No venv interpreter at all. In a dev checkout that's normal (the
        # dev may run hermes from any interpreter), so report healthy to
        # avoid forcing reinstalls. But on a MANAGED install (the Windows
        # installer / desktop bootstrap stamps `.hermes-bootstrap-complete`,
        # and an interrupted update leaves `.update-incomplete`), the venv
        # IS the install — its absence means a repair got interrupted after
        # the old venv was moved aside, and "Already up to date!" would
        # gaslight the user while nothing can run.
        managed_markers = (
            PROJECT_ROOT / ".hermes-bootstrap-complete",
            _update_marker_path(),
        )
        if any(m.exists() for m in managed_markers):
            return False, f"venv python missing ({venv_python})"
        return True, ""

    # Core web/serve imports plus their newest transitive deps. Import (not
    # just metadata) — a package can have intact dist-info but a missing
    # module after an interrupted uninstall/install cycle.
    check = (
        "import importlib\n"
        "mods = ['fastapi', 'uvicorn', 'pydantic', 'openai', 'yaml']\n"
        "missing = []\n"
        "for m in mods:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception as e: missing.append(f'{m}: {e}')\n"
        "print('\\n'.join(missing))\n"
    )
    try:
        result = subprocess.run(
            [str(venv_python), "-c", check],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=PROJECT_ROOT,
        )
    except Exception as exc:
        logger.debug("venv health probe failed to run: %s", exc)
        return True, ""

    missing = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if result.returncode != 0 and not missing:
        # Interpreter itself is broken (e.g. deleted stdlib) — that IS unhealthy.
        detail = (result.stderr or "").strip().splitlines()
        return False, detail[0] if detail else "venv python failed to run"
    if missing:
        return False, "; ".join(missing[:4])
    return True, ""


def _detect_venv_python_processes(
    *, exclude_pids: set[int] | None = None
) -> list[tuple[int, str, str]]:
    """Find live processes running from the project venv's interpreter.

    The hermes.exe shim guard misses the biggest lock-holder class on
    Windows: the Desktop app's backend (``python.exe -m hermes_cli.main
    serve``) and anything else running straight off ``venv\\Scripts\\python
    (w).exe``. Those processes keep native ``.pyd`` extensions mapped, so a
    dependency sync mid-update dies with access-denied and strands the venv
    half-updated (ryanc's brotlicffi/_sodium.pyd incidents, July 2026).

    Killing them from here is pointless — the Desktop app supervises its
    backend and respawns it within seconds — so the caller should refuse and
    tell the user to close the app instead. Returns ``(pid, name, cmdline)``
    tuples; empty off-Windows / without psutil / when nothing matches. The
    calling process and its ancestors are always excluded (a CLI ``hermes
    update`` itself runs from the venv python). Never raises.
    """
    if not _is_windows():
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_dir = PROJECT_ROOT / "venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep
    try:
        root_prefix = str(PROJECT_ROOT.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        root_prefix = str(PROJECT_ROOT).lower().rstrip(os.sep) + os.sep

    skip: set[int] = set(exclude_pids or set())
    skip.add(os.getpid())
    try:
        for anc in psutil.Process().parents():
            skip.add(int(anc.pid))
    except Exception:
        pass

    matches: list[tuple[int, str, str]] = []
    try:
        proc_iter = psutil.process_iter(["pid", "exe", "name", "cmdline", "cwd"])
    except Exception:
        return []
    for proc in proc_iter:
        try:
            info = proc.info
        except Exception:
            continue
        pid = info.get("pid")
        exe = info.get("exe")
        if not exe or pid is None or int(pid) in skip:
            continue
        try:
            exe_norm = str(Path(exe).resolve()).lower()
        except (OSError, ValueError):
            exe_norm = str(exe).lower()
        cmdline_raw = " ".join(info.get("cmdline") or [])
        cmdline_low = cmdline_raw.lower()
        cwd_low = str(info.get("cwd") or "").lower().rstrip(os.sep) + os.sep

        # Primary match: the executable itself lives under this venv
        # (venv\Scripts\python(w).exe — the desktop backend / gateway case).
        is_holder = exe_norm.startswith(venv_prefix)
        # Fallback: uv/base-interpreter trampolines run a python whose exe is
        # OUTSIDE the venv but which still imports from it and holds its .pyd
        # files. Catch those by what they're running: a cmdline that references
        # this venv's path, or a `-m hermes_cli.main ...` invocation tied to
        # this install (install root in the cmdline or as the working dir).
        if not is_holder and venv_prefix in cmdline_low:
            is_holder = True
        if not is_holder and "hermes_cli.main" in cmdline_low:
            if root_prefix in cmdline_low or cwd_low.startswith(root_prefix):
                is_holder = True
        if not is_holder:
            continue
        name = info.get("name") or Path(exe).name
        matches.append((int(pid), str(name), cmdline_raw[:120]))
    return matches


def _format_venv_python_holders_message(matches: list[tuple[int, str, str]]) -> str:
    """Explain which venv processes block the update and how to clear them."""
    lines = [
        "✗ Other Hermes processes are running from this install's venv:",
    ]
    for pid, name, cmdline in matches[:6]:
        hint = ""
        low = cmdline.lower()
        if "serve" in low or "dashboard" in low:
            hint = "  ← Hermes Desktop backend (close the desktop app)"
        elif "gateway" in low:
            hint = "  ← gateway"
        lines.append(f"  PID {pid}  {name}  {cmdline}{hint}")
    if len(matches) > 6:
        lines.append(f"  ... and {len(matches) - 6} more")
    lines.append("")
    lines.append(
        "  On Windows these keep native extension files (.pyd) locked, so the"
    )
    lines.append(
        "  dependency update would fail partway and leave a broken install."
    )
    lines.append(
        "  Close the Hermes desktop app / other Hermes terminals, then re-run:"
    )
    lines.append("    hermes update")
    lines.append("  (or use `hermes update --force-venv` to proceed anyway at your own risk)")
    return "\n".join(lines)


def _pause_windows_gateways_for_update() -> dict | None:
    """Stop running Windows gateways before mutating the checkout or venv.

    Windows scheduled/startup gateways run through pythonw.exe, so the generic
    hermes.exe concurrent-instance guard does not see them. They still import
    from the checkout and can keep files locked while ``git`` or ``uv`` updates
    the install. Stop only PIDs that the gateway discovery code identifies.
    """
    if not _is_windows():
        return None

    try:
        from gateway.status import terminate_pid
        from hermes_cli.gateway import (
            _capture_gateway_argv,
            _get_restart_drain_timeout,
            find_gateway_pids,
            find_profile_gateway_processes,
        )
    except Exception as exc:
        logger.debug("Could not prepare Windows gateway pause for update: %s", exc)
        return None

    try:
        running_pids = list(dict.fromkeys(find_gateway_pids(all_profiles=True)))
    except Exception as exc:
        logger.debug("Could not discover Windows gateway PIDs before update: %s", exc)
        return None
    if not running_pids:
        # No gateway is running right now, but the user may have installed an
        # autostart entry (Scheduled Task or Startup-folder login item) — that
        # is an explicit "I want a gateway" signal. A gateway that died between
        # updates (e.g. the spawning terminal/TUI closed, taking its child with
        # it) would otherwise never come back: the autostart entry only fires on
        # the next login, and the update flow's resume path only relaunched
        # gateways that were running when the update began. Cold-start one after
        # the update so an installed gateway is actually up post-update. Users
        # who run gateway-less (no autostart entry) get nothing forced on them.
        try:
            from hermes_cli import gateway_windows

            if gateway_windows.is_installed():
                return {
                    "resume_needed": True,
                    "profiles": {},
                    "unmapped_pids": [],
                    "unmapped": [],
                    "cold_start_if_installed": True,
                }
        except Exception as exc:
            logger.debug(
                "Could not check Windows gateway autostart state before update: %s",
                exc,
            )
        return None

    profile_processes = {}
    try:
        profile_processes = {
            proc.pid: proc for proc in find_profile_gateway_processes()
        }
    except Exception as exc:
        logger.debug("Could not map Windows gateway PIDs to profiles: %s", exc)

    profiles: dict[str, int] = {}
    mapped_pids = []
    for pid in running_pids:
        proc = profile_processes.get(pid)
        if proc is None:
            continue
        profiles[str(proc.profile)] = int(pid)
        mapped_pids.append(int(pid))
        _write_update_planned_stop_marker(Path(proc.path), int(pid))

    print("→ Stopping Windows gateway process(es) before updating Hermes...")
    try:
        drain_timeout = max(float(_get_restart_drain_timeout()), 1.0)
    except Exception:
        drain_timeout = 10.0
    survivors = _wait_for_windows_update_gateway_exit(
        mapped_pids,
        timeout=drain_timeout,
    )
    unmapped_pids = [pid for pid in running_pids if pid not in profile_processes]

    # Snapshot each unmapped gateway's command line *before* we force-kill it,
    # so ``_resume_windows_gateways_after_update`` can respawn it by replaying
    # its own argv. Unmapped gateways are ones with no profile→PID-file mapping
    # — e.g. a Windows Scheduled Task running ``pythonw.exe -m hermes_cli.main
    # gateway run``. Without this snapshot they were force-killed and never
    # restarted (the "Restart manually after update" dead-end from #50090).
    unmapped: list[dict] = []
    for pid in unmapped_pids:
        argv = None
        try:
            argv = _capture_gateway_argv(int(pid))
        except Exception as exc:
            logger.debug("Could not capture argv for unmapped gateway %s: %s", pid, exc)
        unmapped.append({"pid": int(pid), "argv": argv})

    force_killed = []
    for pid in sorted(set(survivors).union(unmapped_pids)):
        try:
            terminate_pid(int(pid), force=True)
            force_killed.append(int(pid))
        except (ProcessLookupError, PermissionError, OSError):
            pass

    if profiles:
        print(f"  ✓ Paused gateway profile(s): {', '.join(sorted(profiles))}")
    if force_killed:
        print(f"  → Force-stopped {len(force_killed)} gateway process(es)")

    if unmapped_pids:
        respawnable = sum(1 for u in unmapped if u.get("argv"))
        print(
            f"  → Stopped {len(unmapped_pids)} gateway process(es) without profile mapping"
        )
        if respawnable < len(unmapped_pids):
            # Some had no recoverable command line (psutil missing, access
            # denied, already gone): those still need a manual restart.
            print("    Restart manually after update: hermes gateway run")

    return {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": unmapped_pids,
        "unmapped": unmapped,
    }


def _cold_start_windows_gateway_after_update() -> None:
    """Start a fresh detached gateway after update when one is installed but down.

    Invoked from ``_resume_windows_gateways_after_update`` for the
    ``cold_start_if_installed`` case: no gateway was running when the update
    began, but an autostart entry (Scheduled Task / Startup-folder login item)
    is installed, signalling the user wants a gateway. Unlike the relaunch
    paths — which watch an old PID and respawn once it exits — this is a direct
    fresh spawn via the same windowless ``pythonw`` + breakaway path that
    ``hermes gateway start`` uses (``gateway_windows._spawn_detached``).

    Best-effort and idempotent: re-checks that nothing is running first so a
    concurrent start (e.g. the autostart entry firing) can't produce a
    duplicate gateway.
    """
    if not _is_windows():
        return
    try:
        from hermes_cli import gateway_windows
        from hermes_cli.gateway import find_gateway_pids
    except Exception as exc:
        logger.debug("Could not load Windows gateway cold-start helpers: %s", exc)
        return

    # Re-check liveness right before spawning — between pause and resume the
    # autostart entry may have already brought a gateway up, or a leftover
    # process may have re-registered. Don't double-start.
    try:
        if list(find_gateway_pids(all_profiles=True)):
            return
    except Exception as exc:
        logger.debug("Could not re-check gateway liveness before cold-start: %s", exc)
        return

    try:
        pid = gateway_windows._spawn_detached()
    except Exception as exc:
        logger.debug("Could not cold-start Windows gateway after update: %s", exc)
        return

    if pid:
        print()
        print(f"  ✓ Starting Windows gateway after update (PID {pid})")


def _resume_windows_gateways_after_update(token: dict | None) -> None:
    """Restart Windows profile gateways previously paused for update."""
    if not token or not token.get("resume_needed"):
        return
    token["resume_needed"] = False
    if not _is_windows():
        return

    profiles = token.get("profiles") or {}
    unmapped = token.get("unmapped") or []
    cold_start = bool(token.get("cold_start_if_installed"))
    if not profiles and not any(u.get("argv") for u in unmapped):
        if cold_start:
            _cold_start_windows_gateway_after_update()
        return

    try:
        from hermes_cli.gateway import (
            launch_detached_gateway_restart_by_cmdline,
            launch_detached_profile_gateway_restart,
        )
    except Exception as exc:
        logger.debug("Could not load Windows gateway restart helper: %s", exc)
        return

    relaunched = []
    for profile, old_pid in sorted(profiles.items()):
        try:
            if launch_detached_profile_gateway_restart(str(profile), int(old_pid)):
                relaunched.append(str(profile))
        except Exception as exc:
            logger.debug(
                "Could not restart Windows gateway profile %s after update: %s",
                profile,
                exc,
            )

    # Respawn unmapped gateways (no profile→PID-file mapping, e.g. a Scheduled
    # Task) by replaying the argv we snapshotted before force-killing them.
    unmapped_relaunched = 0
    for entry in unmapped:
        argv = entry.get("argv")
        old_pid = entry.get("pid")
        if not argv or not old_pid:
            continue
        try:
            if launch_detached_gateway_restart_by_cmdline(int(old_pid), list(argv)):
                unmapped_relaunched += 1
        except Exception as exc:
            logger.debug(
                "Could not restart unmapped Windows gateway (pid %s) after update: %s",
                old_pid,
                exc,
            )

    if relaunched:
        print()
        print(f"  ✓ Restarting Windows gateway profile(s): {', '.join(relaunched)}")
    if unmapped_relaunched:
        if not relaunched:
            print()
        print(
            f"  ✓ Restarting {unmapped_relaunched} unmapped Windows gateway process(es)"
        )


def _discard_lockfile_churn(git_cmd, repo_root):
    """Restore tracked ``package-lock.json`` files that npm dirtied locally.

    npm rewrites lockfiles non-deterministically at install/build time. On a
    managed install those diffs are never intentional, so we discard them so
    ``hermes update`` sees a clean tree instead of autostashing every run.
    Best-effort; only ever touches files named ``package-lock.json``.
    """
    try:
        diff = subprocess.run(
            git_cmd + ["diff", "--name-only"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if diff.returncode != 0:
            return
        dirty_package_dirs = {
            Path(line.strip()).parent
            for line in diff.stdout.splitlines()
            if line.strip().endswith("package.json")
        }
        dirty = [
            line.strip()
            for line in diff.stdout.splitlines()
            if line.strip().endswith("package-lock.json")
            and Path(line.strip()).parent not in dirty_package_dirs
        ]
        if not dirty:
            return
        subprocess.run(
            git_cmd + ["checkout", "--", *dirty],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"→ Discarded npm lockfile churn ({len(dirty)} file(s))")
    except Exception:
        # Never let lockfile cleanup block an update.
        pass


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
    if detect_install_method(PROJECT_ROOT) == "docker":
        print(format_docker_update_message())
        sys.exit(1)

    if getattr(args, "check", False):
        # --check honors --branch so the "any new commits?" answer matches
        # what a subsequent `hermes update --branch=<x>` would actually pull.
        branch = _resolve_update_branch(args)
        _cmd_update_check(
            branch=branch,
            branch_explicit=bool(getattr(args, "branch", None)),
        )
        return

    gateway_mode = getattr(args, "gateway", False)

    # Protect against mid-update terminal disconnects (SIGHUP) and tolerate
    # writes to a closed stdout.  No-op in gateway mode.  See
    # _install_hangup_protection for rationale.
    _update_io_state = _install_hangup_protection(gateway_mode=gateway_mode)
    try:
        _cmd_update_impl(args, gateway_mode=gateway_mode)
    finally:
        _finalize_update_output(_update_io_state)


def _cmd_update_pip(args):
    """Update Hermes via pip (for PyPI installs)."""
    from hermes_cli import __version__
    from hermes_cli.config import is_uv_tool_install

    print(f"→ Current version: {__version__}")
    print("→ Checking PyPI for updates...")

    from hermes_cli.managed_uv import ensure_uv, update_managed_uv

    # Keep managed uv current before using it.
    update_managed_uv()

    uv = ensure_uv()
    in_venv = sys.prefix != sys.base_prefix
    # pipx-managed installs live under .../pipx/venvs/<name>/...
    pipx_managed = "pipx" in sys.prefix.split(os.sep)
    pipx = shutil.which("pipx") if pipx_managed else None

    # Only the ``uv pip install`` path inside a venv needs VIRTUAL_ENV
    # exported (uv refuses to install without it when the launcher shim
    # didn't activate the venv). ``uv tool upgrade`` / ``pipx upgrade``
    # operate on a named environment and ignore VIRTUAL_ENV, so we don't
    # set it for them.
    export_virtualenv = False

    if is_uv_tool_install():
        if not uv:
            print("✗ Detected a uv-tool install but managed uv install failed.")
            print("  Install uv manually: https://docs.astral.sh/uv/getting-started/installation/")
            sys.exit(1)
        cmd = [uv, "tool", "upgrade", "hermes-agent"]
    elif pipx_managed and pipx:
        # pipx owns its own venv; ``pipx upgrade`` is the only correct path.
        # Matches scripts/auto-update.sh, which already uses pipx upgrade.
        cmd = [pipx, "upgrade", "hermes-agent"]
    elif uv:
        cmd = [uv, "pip", "install", "--upgrade", "hermes-agent"]
        if in_venv:
            # Launcher shim runs the venv interpreter but doesn't export
            # VIRTUAL_ENV; without it uv errors "No virtual environment found".
            export_virtualenv = True
        else:
            # Outside any venv, ``--system`` lets uv target the active
            # interpreter, matching pip's default behaviour.
            cmd.insert(3, "--system")
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "hermes-agent"]

    print(f"→ Running: {' '.join(cmd)}")
    run_kwargs = {}
    if export_virtualenv:
        run_kwargs["env"] = {**os.environ, "VIRTUAL_ENV": sys.prefix}
    result = subprocess.run(cmd, **run_kwargs)
    if result.returncode != 0:
        print("✗ Update failed")
        sys.exit(1)

    print("✓ Update complete! Restart hermes to use the new version.")


def _cmd_update_impl(args, gateway_mode: bool):
    """Body of ``cmd_update`` — kept separate so the wrapper can always
    restore stdio even on ``sys.exit``."""
    # In gateway mode, use file-based IPC for prompts instead of stdin
    gw_input_fn = (
        (lambda prompt, default="": _gateway_prompt(prompt, default))
        if gateway_mode
        else None
    )
    assume_yes = bool(getattr(args, "yes", False))

    # Whether this update is running without a human at the keyboard.
    # Interactive terminal updates always stash-and-ask (unchanged behavior);
    # only non-interactive updates (desktop/chat app, gateway, `--yes`) consult
    # the `updates.non_interactive_local_changes` config setting to decide
    # whether to auto-restore stashed local source changes or throw them away.
    _non_interactive_update = (
        gateway_mode
        or assume_yes
        or not (sys.stdin.isatty() and sys.stdout.isatty())
    )
    discard_local_changes = False
    if _non_interactive_update:
        try:
            from hermes_cli.config import load_config

            _update_cfg = (load_config() or {}).get("updates", {})
            if isinstance(_update_cfg, dict):
                _mode = str(_update_cfg.get("non_interactive_local_changes", "stash")).lower()
                discard_local_changes = _mode == "discard"
        except Exception as exc:
            # Never let a config read failure change the safe default.
            logger.debug("Could not read updates.non_interactive_local_changes: %s", exc)
            discard_local_changes = False

    print("⚕ Updating Hermes Agent...")
    print()

    # On Windows, abort early if another hermes.exe is holding the venv shim
    # open. Continuing would result in a string of WinError 32 warnings and
    # then either a deferred-rename leftover or a failed git-pull fast path
    # that silently falls back to the slower ZIP route. See issue #26670.
    if _is_windows() and not getattr(args, "force", False):
        scripts_dir = _venv_scripts_dir()
        if scripts_dir is not None:
            concurrent = _detect_concurrent_hermes_instances(scripts_dir)
            if concurrent:
                print(_format_concurrent_instances_message(concurrent, scripts_dir))
                sys.exit(2)

    # Pre-update backup — runs before any git/file mutation so users can
    # always roll back to the exact state they had before this update.
    _run_pre_update_backup(args)

    _windows_gateway_resume = _pause_windows_gateways_for_update()
    if _windows_gateway_resume:
        import atexit as _atexit

        _atexit.register(
            _resume_windows_gateways_after_update,
            _windows_gateway_resume,
        )

    # With gateways paused, anything still running from the venv interpreter
    # (most commonly the Desktop app's `hermes serve` backend) will keep .pyd
    # files locked and corrupt the dependency sync below. Refuse rather than
    # race: killing the desktop backend is futile (the app supervises and
    # respawns it), so the user must close the app. Deliberately NOT bypassed
    # by plain --force: the desktop bootstrap updater passes --force to skip
    # the hermes.exe shim guard above, but its lock probe only checks the shim
    # and app.asar — a non-desktop venv python holding a .pyd would sail
    # through and corrupt the sync (the exact failure this guard exists for).
    # --force-venv is the explicit escape hatch.
    if _is_windows() and not getattr(args, "force_venv", False):
        _venv_holders = _detect_venv_python_processes()
        if _venv_holders:
            print(_format_venv_python_holders_message(_venv_holders))
            _resume_windows_gateways_after_update(_windows_gateway_resume)
            sys.exit(2)

    # Try git-based update first, fall back to ZIP download on Windows
    # when git file I/O is broken (antivirus, NTFS filter drivers, etc.)
    use_zip_update = False
    git_dir = PROJECT_ROOT / ".git"

    if not git_dir.exists():
        if sys.platform == "win32":
            use_zip_update = True
        else:
            from hermes_cli.config import detect_install_method
            method = detect_install_method(PROJECT_ROOT)
            if method == "pip":
                _cmd_update_pip(args)
                return
            print("✗ Not a git repository. Please reinstall:")
            print(
                "  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash"
            )
            sys.exit(1)

    # On Windows, git can fail with "unable to write loose object file: Invalid argument"
    # due to filesystem atomicity issues. Set the recommended workaround.
    if sys.platform == "win32" and git_dir.exists():
        subprocess.run(
            [
                "git",
                "-c",
                "windows.appendAtomically=false",
                "config",
                "windows.appendAtomically",
                "false",
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
        )

    # Build git command once — reused for fork detection and the update itself.
    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]

    # Discard npm lockfile churn before any stash/branch logic. npm rewrites
    # tracked package-lock.json files non-deterministically at install/build
    # time (platform-specific optional deps, ideallyInert annotations, etc.),
    # which is never an intentional edit on a managed install but leaves the
    # tree dirty — forcing an autostash on every update and making branch
    # switches fragile. Restoring them first lets the common case (only
    # lockfile churn) update with a clean tree.
    _discard_lockfile_churn(git_cmd, PROJECT_ROOT)

    # Detect if we're updating from a fork (before any branch logic)
    origin_url = _get_origin_url(git_cmd, PROJECT_ROOT)
    is_fork = _is_fork(origin_url)

    if is_fork:
        print("⚠ Updating from fork:")
        print(f"  {origin_url}")
        print()

    if use_zip_update:
        # ZIP-based update for Windows when git is broken
        try:
            _update_via_zip(args)
        finally:
            _resume_windows_gateways_after_update(_windows_gateway_resume)
        return

    # Fetch and pull
    try:

        # Resolve the target branch up front so the fetch can be scoped to it.
        # A bare `git fetch origin` pulls every ref, and this repo carries
        # thousands of auto-generated branches — an unscoped fetch can stall for
        # minutes on a non-single-branch checkout. Fetch only what we update
        # against.
        branch = _resolve_update_branch(args)

        print("→ Fetching updates...")
        fetch_result = subprocess.run(
            git_cmd + ["fetch", "origin", branch],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if fetch_result.returncode != 0:
            stderr = fetch_result.stderr.strip()
            if "Could not resolve host" in stderr or "unable to access" in stderr:
                print("✗ Network error — cannot reach the remote repository.")
                print(f"  {stderr.splitlines()[0]}" if stderr else "")
            elif (
                "Authentication failed" in stderr or "could not read Username" in stderr
            ):
                print(
                    "✗ Authentication failed — check your git credentials or SSH key."
                )
            else:
                print("✗ Failed to fetch updates from origin.")
                if stderr:
                    print(f"  {stderr.splitlines()[0]}")
            sys.exit(1)

        # Get current branch (returns literal "HEAD" when detached)
        result = subprocess.run(
            git_cmd + ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        current_branch = result.stdout.strip()

        # If user is on a different branch than the update target, switch
        # to the target. When the target is "main" this is the historical
        # "always update against main" behavior; for any other target it's
        # the same thing — get HEAD onto the requested branch first, then
        # fast-forward.
        if current_branch != branch:
            label = (
                "detached HEAD"
                if current_branch == "HEAD"
                else f"branch '{current_branch}'"
            )
            print(f"  ⚠ Currently on {label} — switching to {branch} for update...")
            # Stash before checkout so uncommitted work isn't lost
            auto_stash_ref = _stash_local_changes_if_needed(git_cmd, PROJECT_ROOT)
            checkout_result = subprocess.run(
                git_cmd + ["checkout", branch],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if checkout_result.returncode != 0:
                # Local checkout doesn't have this branch yet. Try to set
                # it up as a tracking branch of origin/<branch>. This is
                # the common case when the requested branch exists upstream
                # but was never checked out locally.
                track_result = subprocess.run(
                    git_cmd + ["checkout", "-B", branch, f"origin/{branch}"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                )
                if track_result.returncode != 0:
                    # Restore the user's prior branch + stash before bailing
                    # so we don't leave them stranded in a weird state.
                    if auto_stash_ref is not None:
                        _restore_stashed_changes(
                            git_cmd,
                            PROJECT_ROOT,
                            auto_stash_ref,
                            prompt_user=False,
                            input_fn=gw_input_fn,
                        )
                    print(f"✗ Branch '{branch}' does not exist locally or on origin.")
                    if track_result.stderr.strip():
                        print(f"  {track_result.stderr.strip().splitlines()[0]}")
                    sys.exit(1)
        else:
            auto_stash_ref = _stash_local_changes_if_needed(git_cmd, PROJECT_ROOT)

        prompt_for_restore = (
            auto_stash_ref is not None
            and not assume_yes
            and (gateway_mode or (sys.stdin.isatty() and sys.stdout.isatty()))
        )

        # Check if there are updates
        result = subprocess.run(
            git_cmd + ["rev-list", f"HEAD..origin/{branch}", "--count"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        commit_count = int(result.stdout.strip())

        if commit_count == 0:
            _invalidate_update_cache()

            # Even if origin is up to date, the fork may be behind upstream
            if is_fork and branch == "main":
                _sync_with_upstream_if_needed(git_cmd, PROJECT_ROOT)

            # Restore stash and switch back to original branch if we moved
            if auto_stash_ref is not None:
                _restore_stashed_changes(
                    git_cmd,
                    PROJECT_ROOT,
                    auto_stash_ref,
                    prompt_user=prompt_for_restore,
                    input_fn=gw_input_fn,
                )
            if current_branch not in {branch, "HEAD"}:
                subprocess.run(
                    git_cmd + ["checkout", current_branch],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            # A current checkout does NOT imply a healthy install: a previous
            # dependency sync may have failed partway (classic on Windows,
            # where a running gateway/desktop backend keeps .pyd files locked
            # and uv/pip dies with access-denied, stranding the venv between
            # versions). Probe the venv's core imports and repair if broken —
            # otherwise "Already up to date!" gaslights the user while their
            # install stays bricked.
            healthy, detail = _venv_core_imports_healthy()
            if not healthy:
                print("⚠ Checkout is current, but the venv is unhealthy:")
                print(f"  {detail}")
                print("→ Repairing Python dependencies...")
                _write_update_incomplete_marker()
                from hermes_cli.managed_uv import ensure_uv

                repair_uv = ensure_uv()
                # A managed install whose venv is gone entirely (interrupted
                # repair after the old venv was moved aside) needs the venv
                # recreated before dependencies can be installed into it.
                venv_python_missing = not (
                    PROJECT_ROOT
                    / "venv"
                    / ("Scripts" if _is_windows() else "bin")
                    / ("python.exe" if _is_windows() else "python")
                ).exists()
                if venv_python_missing and repair_uv:
                    print("→ Recreating virtual environment...")
                    subprocess.run(
                        [repair_uv, "venv", "venv"],
                        cwd=PROJECT_ROOT,
                        check=False,
                    )
                if repair_uv:
                    repair_env = {**os.environ, "VIRTUAL_ENV": str(PROJECT_ROOT / "venv")}
                    _install_python_dependencies_with_optional_fallback(
                        [repair_uv, "pip"], env=repair_env, group="all"
                    )
                else:
                    _install_python_dependencies_with_optional_fallback(
                        [sys.executable, "-m", "pip"], group="all"
                    )
                _clear_update_incomplete_marker()
                healthy_after, detail_after = _venv_core_imports_healthy()
                if healthy_after:
                    print("✓ Dependencies repaired!")
                else:
                    print(f"⚠ Venv still unhealthy after repair: {detail_after}")
                    print("  Close all Hermes windows/gateways and re-run: hermes update")
            else:
                print("✓ Already up to date!")
            _resume_windows_gateways_after_update(_windows_gateway_resume)
            return

        print(f"→ Found {commit_count} new commit(s)")

        # Snapshot critical state (state.db, config, pairing JSONs, etc.)
        # before pulling so a user can recover if something goes wrong.
        # Issue #15733 reported missing pairing data after an update; even
        # though `git pull` can't touch $HERMES_HOME, this is cheap
        # belt-and-suspenders insurance and gives the user something to
        # restore from via `/snapshot list` / `/snapshot restore <id>`.
        pre_update_snapshot_id = None
        try:
            from hermes_cli.backup import create_quick_snapshot

            pre_update_snapshot_id = create_quick_snapshot(label="pre-update", keep=1)
            if pre_update_snapshot_id:
                print(f"  ✓ Pre-update snapshot: {pre_update_snapshot_id}")
        except Exception as exc:
            # Never let a snapshot failure block an update.
            logger.debug("Pre-update snapshot failed: %s", exc)

        print("→ Pulling updates...")
        update_succeeded = False
        # Capture the pre-pull SHA so we can auto-roll-back if the new code
        # has a syntax error in a critical-path file (PR #28452 incident:
        # orphan merge-conflict markers in hermes_cli/config.py bricked
        # every user who ran ``hermes update`` for the 7 minutes between
        # the bad commit and the fix landing).
        pre_pull_sha = _capture_head_sha(git_cmd, PROJECT_ROOT)
        try:
            pull_result = subprocess.run(
                git_cmd + ["pull", "--ff-only", "origin", branch],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if pull_result.returncode != 0:
                # ff-only failed — local and remote have diverged (e.g. upstream
                # force-pushed or rebase).  Since local changes are already
                # stashed, reset to match the remote exactly.
                print(
                    "  ⚠ Fast-forward not possible (history diverged), resetting to match remote..."
                )
                reset_result = subprocess.run(
                    git_cmd + ["reset", "--hard", f"origin/{branch}"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                )
                if reset_result.returncode != 0:
                    print(f"✗ Failed to reset to origin/{branch}.")
                    if reset_result.stderr.strip():
                        print(f"  {reset_result.stderr.strip()}")
                    print(
                        f"  Try manually: git fetch origin && git reset --hard origin/{branch}"
                    )
                    sys.exit(1)

            # Post-pull syntax guard: validate critical-path files actually
            # parse before declaring the update successful. If a bad commit
            # made it through CI (e.g. admin-merge bypass of a failing
            # ruff check), this catches it on the user side and rolls back
            # so the CLI stays bootable. The user can then retry ``hermes
            # update`` later once a fix lands upstream.
            syntax_ok, failing_path, syntax_error = _validate_critical_files_syntax(
                PROJECT_ROOT
            )
            if not syntax_ok:
                print()
                print("✗ Pulled code has a syntax error in a critical file:")
                print(f"  {failing_path}")
                if syntax_error:
                    # py_compile errors can be multi-line; show the first
                    # ~6 lines so the user sees the actual SyntaxError text.
                    for line in str(syntax_error).splitlines()[:6]:
                        print(f"    {line}")
                if pre_pull_sha:
                    print()
                    print(f"→ Rolling back to {pre_pull_sha[:10]}...")
                    rollback_result = subprocess.run(
                        git_cmd + ["reset", "--hard", pre_pull_sha],
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                        text=True,
                    )
                    if rollback_result.returncode == 0:
                        print("  ✓ Rollback complete — your install is unchanged.")
                        print("  Try ``hermes update`` again later once a fix lands.")
                    else:
                        print("  ✗ Rollback failed. Recover manually with:")
                        print(f"    cd {PROJECT_ROOT} && git reset --hard {pre_pull_sha}")
                        if rollback_result.stderr.strip():
                            print(f"    ({rollback_result.stderr.strip().splitlines()[0]})")
                else:
                    print()
                    print("  Could not capture pre-pull SHA — recover manually with:")
                    print(f"    cd {PROJECT_ROOT} && git reflog && git reset --hard <prev-sha>")
                sys.exit(1)

            update_succeeded = True
        finally:
            if auto_stash_ref is not None:
                # Don't attempt stash restore if the code update itself failed —
                # working tree is in an unknown state.
                if not update_succeeded:
                    print(
                        f"  ℹ️  Local changes preserved in stash (ref: {auto_stash_ref})"
                    )
                    print("  Restore manually with: git stash apply")
                elif discard_local_changes:
                    # Non-interactive update + user opted into discarding local
                    # source edits (updates.non_interactive_local_changes:
                    # discard). Throw the stash away instead of re-applying it.
                    _discard_stashed_changes(
                        git_cmd,
                        PROJECT_ROOT,
                        auto_stash_ref,
                    )
                else:
                    _restore_stashed_changes(
                        git_cmd,
                        PROJECT_ROOT,
                        auto_stash_ref,
                        prompt_user=prompt_for_restore,
                        input_fn=gw_input_fn,
                    )

        _invalidate_update_cache()

        # Clear stale .pyc bytecode cache — prevents ImportError on gateway
        # restart when updated source references names that didn't exist in
        # the old bytecode (e.g. get_hermes_home added to hermes_constants).
        removed = _clear_bytecode_cache(PROJECT_ROOT)
        if removed:
            print(
                f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}"
            )

        # Fork upstream sync logic (only for main branch on forks)
        if is_fork and branch == "main":
            _sync_with_upstream_if_needed(git_cmd, PROJECT_ROOT)

        # Reinstall Python dependencies. Prefer .[all], but if one optional extra
        # breaks on this machine, keep base deps and reinstall the remaining extras
        # individually so update does not silently strip working capabilities.
        #
        # Drop the interrupted-install breadcrumb BEFORE touching the venv. If
        # the install is killed mid-flight (Ctrl-C, terminal close, WSL OOM),
        # the marker survives and the next ``hermes`` launch finishes the
        # install via ``_recover_from_interrupted_install``. Cleared only after
        # the install + core-dependency verification completes below.
        _write_update_incomplete_marker()
        print("→ Updating Python dependencies...")
        from hermes_cli.managed_uv import ensure_uv, update_managed_uv

        # Keep managed uv current — runs `uv self update` if we already have one.
        update_managed_uv()

        uv_bin = ensure_uv()

        pip_cmd = [sys.executable, "-m", "pip"]
        if not uv_bin:
            uv_bin = _ensure_uv_for_termux(pip_cmd)
        install_group = "all"

        if uv_bin:
            uv_env = {**os.environ, "VIRTUAL_ENV": str(PROJECT_ROOT / "venv")}
            if _is_termux_env(uv_env):
                uv_env.pop("PYTHONPATH", None)
                uv_env.pop("PYTHONHOME", None)
                install_group = "termux-all"
                print("  → Termux detected: using uv + curated termux-all optional profile...")
            if _is_termux_env(uv_env) and _is_android_python():
                print("  → Termux/Android detected: prebuilding psutil with Linux source path compatibility...")
                _install_psutil_android_compat([uv_bin, "pip"], env=uv_env)
            _install_python_dependencies_with_optional_fallback(
                [uv_bin, "pip"], env=uv_env, group=install_group
            )
        else:
            # Use sys.executable to explicitly call the venv's pip module,
            # avoiding PEP 668 'externally-managed-environment' errors on Debian/Ubuntu.
            # Some environments lose pip inside the venv; bootstrap it back with
            # ensurepip before trying the editable install.
            pip_cmd = [sys.executable, "-m", "pip"]
            try:
                subprocess.run(
                    pip_cmd + ["--version"],
                    cwd=PROJECT_ROOT,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                subprocess.run(
                    [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
            if _is_termux_env():
                install_group = "termux-all"
                print("  → Termux detected: using curated termux-all optional profile...")
            if _is_termux_env() and _is_android_python():
                print("  → Termux/Android detected: prebuilding psutil with Linux source path compatibility...")
                _install_psutil_android_compat(pip_cmd)
            _install_python_dependencies_with_optional_fallback(pip_cmd, group=install_group)

        # Core Python deps installed AND verified (the fallback helper runs
        # _verify_core_dependencies_installed). Clear the interrupted-install
        # breadcrumb now — the remaining steps (lazy refresh, node deps, web
        # UI, desktop rebuild) are non-core and can't brick the venv.
        _clear_update_incomplete_marker()

        # The update process is still the old Python interpreter process. Run
        # one final cache/module refresh immediately before lazy backend
        # refresh, which imports newly-pulled modules that may depend on fresh
        # symbols in hermes_constants or lazy_deps.
        removed = _clear_bytecode_cache(PROJECT_ROOT)
        if removed:
            print(
                f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}"
            )
        _reload_updated_runtime_modules()

        _refresh_active_lazy_features()

        _update_node_dependencies()
        _build_web_ui(PROJECT_ROOT / "web")

        # Rebuild the desktop app if the source tree changed since the last
        # build.  ``hermes desktop --build-only`` uses the content-hash stamp
        # internally, so this is effectively a no-op when nothing changed.
        # Only bother if the user has a desktop app installed (indicated by
        # an existing packaged executable or desktop dist); people who have
        # never run ``hermes desktop`` shouldn't be forced into a full
        # Electron build by ``hermes update``.
        desktop_dir = PROJECT_ROOT / "apps" / "desktop"
        has_desktop_app = _desktop_packaged_executable(desktop_dir) is not None or _desktop_dist_exists(desktop_dir)
        from hermes_constants import find_node_executable

        if (desktop_dir / "package.json").exists() and find_node_executable("npm") and has_desktop_app:
            print("→ Checking if desktop app needs rebuilding...")
            _desktop_build_cmd = [sys.executable, "-m", "hermes_cli.main", "desktop", "--build-only"]
            # Capture the (very loud) Electron/vite build output into
            # update.log instead of streaming it to the terminal. On the rare
            # nonzero exit, retry once after waiting again for the venv — this
            # covers a still-settling rebuild window the first wait didn't fully
            # catch — then surface the captured tail so the failure is
            # debuggable.
            build_result = _run_logged_subprocess(_desktop_build_cmd, cwd=PROJECT_ROOT)
            if build_result.returncode != 0:
                build_result = _run_logged_subprocess(_desktop_build_cmd, cwd=PROJECT_ROOT)
            if build_result.returncode != 0:
                print("  ⚠ Desktop build failed (non-fatal; run `hermes desktop` to retry)")
                tail = "\n".join((build_result.stdout or "").strip().splitlines()[-15:])
                if tail:
                    print(tail)
                from hermes_constants import display_hermes_home as _dhh
                print(f"  Full build log: {_dhh()}/logs/update.log")
            else:
                print("  ✓ Desktop app up to date")

        print()
        print("✓ Code updated!")

        # Seed the model-catalog disk cache from the freshly-pulled checkout.
        # The repo ships the canonical catalog at
        # website/static/api/model-catalog.json, and `git pull` just made it
        # current — so copy it straight over ~/.hermes/cache/model_catalog.json
        # instead of waiting on a network fetch (which can be bot-gated or hit a
        # Portal hiccup). Keeps the model picker's curated/free lists in sync
        # with the version the user just installed. Non-fatal on failure: the
        # normal network refresh still applies on the next picker open.
        try:
            from hermes_cli.model_catalog import seed_cache_from_checkout

            if seed_cache_from_checkout(PROJECT_ROOT):
                print("  ✓ Model catalog cache refreshed from checkout")
        except Exception as e:
            logger.debug("Model catalog seed during update failed: %s", e)

        # Sync bundled skills (copies new, updates changed, respects user deletions)
        try:
            from tools.skills_sync import sync_skills

            print()
            print("→ Syncing bundled skills...")
            result = sync_skills(quiet=True)
            if result["copied"]:
                print(f"  + {len(result['copied'])} new: {', '.join(result['copied'])}")
            if result.get("updated"):
                print(
                    f"  ↑ {len(result['updated'])} updated: {', '.join(result['updated'])}"
                )
            if result.get("user_modified"):
                print(f"  ~ {len(result['user_modified'])} user-modified (kept)")
                print(
                    "    → see them: hermes skills list-modified  "
                    "(diff/reset to resume updates)"
                )
            if result.get("cleaned"):
                print(f"  − {len(result['cleaned'])} removed from manifest")
            if not result["copied"] and not result.get("updated"):
                print("  ✓ Skills are up to date")
        except Exception as e:
            logger.debug("Skills sync during update failed: %s", e)

        # Sync bundled skills to all profiles (including the active one).
        # seed_profile_skills() uses subprocess with an explicit HERMES_HOME so
        # it is not affected by sync_skills()'s module-level HERMES_HOME cache,
        # which means the active profile is reliably synced regardless of whether
        # the caller's HERMES_HOME env var points at the default or a named profile.
        try:
            from hermes_cli.profiles import (
                list_profiles,
                seed_profile_skills,
            )

            all_profiles = list_profiles()
            if all_profiles:
                print()
                print("→ Syncing bundled skills to all profiles...")
                for p in all_profiles:
                    try:
                        r = seed_profile_skills(p.path, quiet=True)
                        if r and r.get("skipped_opt_out"):
                            status = "opted out (--no-skills)"
                        elif r:
                            copied = len(r.get("copied", []))
                            updated = len(r.get("updated", []))
                            modified = len(r.get("user_modified", []))
                            parts = []
                            if copied:
                                parts.append(f"+{copied} new")
                            if updated:
                                parts.append(f"↑{updated} updated")
                            if modified:
                                parts.append(f"~{modified} user-modified")
                            status = ", ".join(parts) if parts else "up to date"
                        else:
                            status = "sync failed"
                        print(f"  {p.name}: {status}")
                    except Exception as pe:
                        print(f"  {p.name}: error ({pe})")
        except Exception:
            pass  # profiles module not available or no profiles

        # Backfill per-profile .env files for profiles created before the
        # .env-seeding fix (#44792). Copies the default install's .env so
        # those profiles keep the credentials they were effectively using.
        try:
            from hermes_cli.profiles import backfill_profile_envs

            backfilled = backfill_profile_envs(quiet=True)
            if backfilled:
                print()
                print(
                    f"→ Seeded .env for {len(backfilled)} profile(s) "
                    f"(copied from default): {', '.join(backfilled)}"
                )
        except Exception:
            pass  # profiles module not available or no profiles

        # Sync Honcho host blocks to all profiles
        try:
            from plugins.memory.honcho.cli import sync_honcho_profiles_quiet

            synced = sync_honcho_profiles_quiet()
            if synced:
                print(f"\n-> Honcho: synced {synced} profile(s)")
        except Exception:
            pass  # honcho plugin not installed or not configured

        # Check for config migrations
        print()
        print("→ Checking configuration for new options...")

        from hermes_cli.config import (
            get_missing_env_vars,
            get_missing_config_fields,
            check_config_version,
            migrate_config,
        )

        missing_env = get_missing_env_vars(required_only=True)
        missing_config = get_missing_config_fields()
        current_ver, latest_ver = check_config_version()

        has_new_options = bool(missing_env or missing_config)
        version_bump_only = (
            not has_new_options and current_ver < latest_ver
        )
        needs_migration = has_new_options or current_ver < latest_ver

        if version_bump_only:
            # Nothing for the user to fill in — only the config format version
            # changed (new defaults already merge in transparently). Asking
            # "configure new options now?" here is misleading: saying yes just
            # bumps the version and looks like a no-op (issue: ScottFive /
            # Tt2021). Apply it silently and say what actually happened.
            print()
            print(
                f"  ℹ Updating config format (v{current_ver} → v{latest_ver})…"
            )
            try:
                migrate_config(interactive=False, quiet=True)
                print("  ✓ Config format updated (no new settings to configure)")
            except Exception as _mig_err:
                print(f"  ⚠️  Config format update failed: {_mig_err}")
                print("     Run 'hermes config migrate' to retry.")
        elif needs_migration:
            print()
            # Show WHAT changed, not just a count, so the user can make an
            # informed yes/no decision (previously the prompt named nothing).
            def _print_items(items, label, key, fallback_key=None):
                if not items:
                    return
                print(f"  {label}:")
                shown = items[:8]
                for it in shown:
                    if isinstance(it, dict):
                        name = it.get(key) or (fallback_key and it.get(fallback_key)) or "?"
                        desc = (it.get("description") or "").strip()
                    else:
                        # Defensive: some callers/mocks pass bare name strings.
                        name = str(it)
                        desc = ""
                    if desc:
                        print(f"      • {name} — {desc}")
                    else:
                        print(f"      • {name}")
                extra = len(items) - len(shown)
                if extra > 0:
                    print(f"      … and {extra} more")

            if missing_env:
                print(
                    f"  ⚠️  {len(missing_env)} new required setting(s) need configuration"
                )
                _print_items(missing_env, "New settings", "name")
            if missing_config:
                print(f"  ℹ️  {len(missing_config)} new config option(s) available")
                _print_items(missing_config, "New options", "key")

            print()
            if assume_yes:
                print(
                    "  ℹ --yes: auto-applying config migration (skipping API-key prompts)."
                )
                response = "y"
            elif gateway_mode:
                response = (
                    _gateway_prompt(
                        "Would you like to configure new options now? [Y/n]", "n"
                    )
                    .strip()
                    .lower()
                )
            elif not (sys.stdin.isatty() and sys.stdout.isatty()):
                print("  ℹ Non-interactive session — applying safe config migrations.")
                response = "auto"
            else:
                try:
                    response = (
                        input("Would you like to configure them now? [Y/n]: ")
                        .strip()
                        .lower()
                    )
                except EOFError:
                    response = "n"

            if response in {"", "y", "yes", "auto"}:
                print()
                # Gateway mode, --yes, and non-interactive update contexts
                # (dashboard / web server actions) cannot prompt for API keys.
                # Still run the non-interactive migration pass before restarting
                # so new default config fields and version bumps are written
                # before the freshly updated gateway validates config at startup.
                interactive_migration = not (
                    gateway_mode or assume_yes or response == "auto"
                )
                results = migrate_config(interactive=interactive_migration, quiet=False)

                if results["env_added"] or results["config_added"]:
                    print()
                    print("✓ Configuration updated!")
                if (gateway_mode or assume_yes or response == "auto") and missing_env:
                    print("  ℹ API keys require manual entry: hermes config migrate")
            else:
                print()
                print("Skipped. Run 'hermes config migrate' later to configure.")
        else:
            print("  ✓ Configuration is up to date")

        # Safety net: config-version migrations have been observed to leave
        # cron/jobs.json valid-but-empty, silently dropping every scheduled
        # job (issue #34600). The desktop scheduler can also overwrite with
        # its own small set, causing partial loss (issue #52144). If the
        # live file now has fewer jobs than the pre-update snapshot, restore
        # it and warn loudly.
        try:
            from hermes_cli.backup import restore_cron_jobs_if_emptied

            cron_restore = restore_cron_jobs_if_emptied(pre_update_snapshot_id)
            if cron_restore:
                print()
                print(
                    "  ⚠️  cron/jobs.json lost jobs during this update — "
                    f"restored {cron_restore['job_count']} job(s) from "
                    f"pre-update snapshot {cron_restore['snapshot_id']}."
                )
        except Exception as exc:
            # Never let the cron safety net break an otherwise-good update.
            logger.debug("Cron jobs auto-restore check failed: %s", exc)

        print()
        print("✓ Update complete!")

        # Curator first-run heads-up. Only prints when curator is enabled AND
        # has never run — i.e. the window where the ticker would otherwise
        # have fired against a fresh skill library. Kept silent on steady
        # state so we don't nag.
        try:
            _print_curator_first_run_notice()
        except Exception as e:
            logger.debug("Curator first-run notice failed: %s", e)

        # Most-recent curator run notice — show-once per run. Surfaces the
        # rename map (`old-name → umbrella`) on the high-attention update
        # surface so users learn about consolidations without having to
        # check `hermes curator status`. Self-stamps after printing so it
        # never repeats for the same run.
        try:
            _print_curator_recent_run_notice()
        except Exception as e:
            logger.debug("Curator recent-run notice failed: %s", e)

        # Repair RHEL-family root installs where /usr/local/bin isn't on PATH
        # for non-login interactive shells.  No-op on every other platform.
        try:
            _ensure_fhs_path_guard()
        except Exception as e:
            logger.debug("FHS PATH guard check failed: %s", e)

        # Refresh the cua-driver binary used by the Computer Use toolset.
        # The upstream installer is gated on supported platforms and on the
        # binary already being on PATH, so this is a no-op for users who
        # don't have it. Tying the refresh to ``hermes update`` gives users a
        # predictable cadence (matches when they pull new agent code) without
        # adding startup latency or a per-launch GitHub API call.
        try:
            refresh_cua_driver = True
            try:
                from hermes_cli.config import load_config

                _update_cfg = (load_config() or {}).get("updates", {})
                if isinstance(_update_cfg, dict):
                    refresh_cua_driver = bool(
                        _update_cfg.get("refresh_cua_driver", True)
                    )
            except Exception as cfg_exc:
                logger.debug("Could not read updates.refresh_cua_driver: %s", cfg_exc)

            if (
                refresh_cua_driver
                and sys.platform in ("darwin", "win32", "linux")
                and shutil.which("cua-driver")
            ):
                from hermes_cli.tools_config import install_cua_driver

                print()
                print("→ Refreshing cua-driver (Computer Use)...")
                install_cua_driver(upgrade=True)
        except Exception as e:
            logger.debug("cua-driver refresh failed: %s", e)

        # Write exit code *before* the gateway restart attempt.
        # When running as ``hermes update --gateway`` (spawned by the gateway's
        # /update command), this process lives inside the gateway's systemd
        # cgroup.  A graceful SIGUSR1 restart keeps the drain loop alive long
        # enough for the exit-code marker to be written below, but the
        # fallback ``systemctl restart`` path (see below) kills everything in
        # the cgroup (KillMode=mixed → SIGKILL to remaining processes),
        # including us and the wrapping bash shell.  The shell never reaches
        # its ``printf $status > .update_exit_code`` epilogue, so the
        # exit-code marker file would never be created.  The new gateway's
        # update watcher would then poll for 30 minutes and send a spurious
        # timeout message.
        #
        # Writing the marker here — after git pull + pip install succeed but
        # before we attempt the restart — ensures the new gateway sees it
        # regardless of how we die.
        if gateway_mode:
            _exit_code_path = get_hermes_home() / ".update_exit_code"
            try:
                _exit_code_path.write_text("0")
            except OSError:
                pass

        # Auto-restart ALL gateways after update.
        # The code update (git pull) is shared across all profiles, so every
        # running gateway needs restarting to pick up the new code.
        try:
            from hermes_cli.gateway import (
                is_macos,
                supports_systemd_services,
                _ensure_user_systemd_env,
                find_gateway_pids,
                find_profile_gateway_processes,
                launch_detached_profile_gateway_restart,
                _get_service_pids,
                _graceful_restart_via_sigusr1,
                _wait_for_gateway_exit,
            )
            import signal as _signal

            def _wait_for_service_active(
                scope_cmd_: list,
                svc_name_: str,
                timeout: float = 10.0,
            ) -> bool:
                """Poll ``systemctl is-active`` until the unit reports active.

                systemd's Stopped -> Started transition after a graceful exit
                (or a hard restart) is not instantaneous; a one-shot check
                races that window and falsely reports the unit as down.
                Poll every 0.5s up to ``timeout`` seconds before giving up.
                """
                deadline = _time.monotonic() + max(timeout, 0.5)
                while True:
                    try:
                        _verify = subprocess.run(
                            scope_cmd_ + ["is-active", svc_name_],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if _verify.stdout.strip() == "active":
                            return True
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        pass
                    if _time.monotonic() >= deadline:
                        return False
                    _time.sleep(0.5)

            def _service_restart_sec(
                scope_cmd_: list,
                svc_name_: str,
                default: float = 0.0,
            ) -> float:
                """Read the unit's ``RestartUSec`` (RestartSec) in seconds.

                After a graceful exit-75, systemd waits ``RestartSec`` before
                respawning the unit.  Callers that poll for ``is-active``
                must use a timeout >= ``RestartSec`` + transition slack, or
                they'll give up *during* the cooldown window and wrongly
                conclude the unit didn't relaunch.
                """
                try:
                    _show = subprocess.run(
                        scope_cmd_
                        + [
                            "show",
                            svc_name_,
                            "--property=RestartUSec",
                            "--value",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    return default
                raw = (_show.stdout or "").strip()
                # systemd emits values like "30s", "100ms", "1min 30s", or
                # "infinity".  Parse conservatively; on any miss return default.
                if not raw or raw == "infinity":
                    return default
                total = 0.0
                matched = False
                for part in raw.split():
                    for _suf, _mult in (
                        ("ms", 0.001),
                        ("us", 0.000001),
                        ("min", 60.0),
                        ("s", 1.0),
                    ):
                        if part.endswith(_suf):
                            try:
                                total += float(part[: -len(_suf)]) * _mult
                                matched = True
                            except ValueError:
                                pass
                            break
                return total if matched else default

            _manage_cmd_cache: dict = {}

            def _resolve_manage_cmd(scope_: str, scope_cmd_: list, svc_name_: str):
                """Resolve the command prefix for manage-units operations.

                Read-only systemctl calls (``is-active``, ``show``,
                ``list-units``) work unprivileged, but manage-units verbs
                (``reset-failed``, ``start``, ``restart``) on a *system*
                service trigger a polkit ``org.freedesktop.systemd1.manage-units``
                authentication prompt when run as a non-root user.  That
                interactive prompt runs inside our captured subprocess with a
                10-15s timeout — the user sees the prompt flash and "exit
                directly" before they can answer, and the resulting
                TimeoutExpired used to be swallowed silently.

                Strategy: if root, plain systemctl.  If not root, try
                non-interactive sudo (``sudo -n``) — first a blanket probe,
                then a targeted ``systemctl reset-failed`` probe so a
                least-privilege sudoers entry scoped to
                ``systemctl ... hermes-gateway*`` also qualifies
                (``reset-failed`` is an idempotent no-op we run before every
                privileged restart anyway).  If neither works, return None —
                the caller must SKIP the restart (without draining the
                gateway first!) and tell the user how to restart manually.
                ``--no-ask-password`` guarantees polkit can never hang a
                captured subprocess on this path.
                """
                if scope_ in _manage_cmd_cache:
                    return _manage_cmd_cache[scope_]
                cmd = scope_cmd_ + ["--no-ask-password"]
                if (
                    scope_ == "system"
                    and hasattr(os, "geteuid")
                    and os.geteuid() != 0  # windows-footgun: ok — systemd path, Linux-only
                ):
                    sudo_cmd = ["sudo", "-n"] + scope_cmd_ + ["--no-ask-password"]
                    sudo_ok = False
                    try:
                        _probe = subprocess.run(
                            ["sudo", "-n", "true"],
                            capture_output=True,
                            timeout=5,
                        )
                        sudo_ok = _probe.returncode == 0
                        if not sudo_ok:
                            # Blanket sudo refused — a targeted sudoers entry
                            # (NOPASSWD for systemctl ... hermes-gateway*)
                            # may still allow the exact commands we need.
                            _probe = subprocess.run(
                                sudo_cmd + ["reset-failed", svc_name_],
                                capture_output=True,
                                timeout=5,
                            )
                            sudo_ok = _probe.returncode == 0
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        sudo_ok = False
                    cmd = sudo_cmd if sudo_ok else None
                _manage_cmd_cache[scope_] = cmd
                return cmd

            # Drain budget for graceful SIGUSR1 restarts.  The gateway drains
            # for up to ``agent.restart_drain_timeout`` (default 60s) before
            # exiting with code 75; we wait slightly longer so the drain
            # completes before we fall back to a hard restart.  On older
            # systemd units without SIGUSR1 wiring this wait just times out
            # and we fall back to ``systemctl restart`` (the old behaviour).
            try:
                from hermes_constants import (
                    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT as _DEFAULT_DRAIN,
                )
            except Exception:
                _DEFAULT_DRAIN = 60.0
            _cfg_drain = None
            try:
                from hermes_cli.config import load_config

                _cfg_agent = load_config().get("agent") or {}
                _cfg_drain = _cfg_agent.get("restart_drain_timeout")
            except Exception:
                pass
            try:
                _drain_budget = (
                    float(_cfg_drain)
                    if _cfg_drain is not None
                    else float(_DEFAULT_DRAIN)
                )
            except (TypeError, ValueError):
                _drain_budget = float(_DEFAULT_DRAIN)
            # Add a 15s margin so the drain loop + final exit finish before
            # we escalate to ``systemctl restart`` / SIGTERM.
            _drain_budget = max(_drain_budget, 30.0) + 15.0

            restarted_services = []
            killed_pids = set()
            relaunched_profiles = []

            # --- Systemd services (Linux) ---
            # Discover all hermes-gateway* units (default + profiles)
            if supports_systemd_services():
                try:
                    _ensure_user_systemd_env()
                except Exception:
                    pass

                for scope, scope_cmd in [
                    ("user", ["systemctl", "--user"]),
                    ("system", ["systemctl"]),
                ]:
                    try:
                        result = subprocess.run(
                            scope_cmd
                            + [
                                "list-units",
                                "hermes-gateway*",
                                "--plain",
                                "--no-legend",
                                "--no-pager",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        for line in result.stdout.strip().splitlines():
                            parts = line.split()
                            if not parts:
                                continue
                            unit = parts[
                                0
                            ]  # e.g. hermes-gateway.service or hermes-gateway-coder.service
                            if not unit.endswith(".service"):
                                continue
                            svc_name = unit.removesuffix(".service")
                            # Check if active
                            check = subprocess.run(
                                scope_cmd + ["is-active", svc_name],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            if check.stdout.strip() != "active":
                                continue

                            # Resolve how we may run manage-units verbs
                            # (reset-failed/start/restart) for this scope.
                            # None ⇒ no non-interactive privilege path; we
                            # must avoid those verbs entirely or polkit will
                            # throw an interactive auth prompt inside our
                            # captured 10-15s subprocess (the user sees it
                            # flash and "exit directly" — reported June 2026).
                            _manage_cmd = _resolve_manage_cmd(
                                scope, scope_cmd, svc_name
                            )

                            # Prefer a graceful SIGUSR1 restart so in-flight
                            # agent runs drain instead of being SIGKILLed.
                            # The gateway's SIGUSR1 handler calls
                            # request_restart(via_service=True) → drain →
                            # exit; systemd's Restart=always respawns the unit.
                            _main_pid = 0
                            try:
                                _show = subprocess.run(
                                    scope_cmd
                                    + [
                                        "show",
                                        svc_name,
                                        "--property=MainPID",
                                        "--value",
                                    ],
                                    capture_output=True,
                                    text=True,
                                    timeout=5,
                                )
                                _main_pid = int((_show.stdout or "").strip() or 0)
                            except (
                                ValueError,
                                subprocess.TimeoutExpired,
                                FileNotFoundError,
                            ):
                                _main_pid = 0

                            _graceful_ok = False
                            if _main_pid > 0:
                                print(
                                    f"  → {svc_name}: draining (up to {int(_drain_budget)}s)..."
                                )
                                _graceful_ok = _graceful_restart_via_sigusr1(
                                    _main_pid,
                                    drain_timeout=_drain_budget,
                                )

                            if _graceful_ok:
                                # Gateway exited after a planned restart.
                                # ``Restart=always`` means systemd WILL respawn
                                # the unit — but only after
                                # ``RestartSec`` (default 60s on our unit
                                # file). That 60s wait is a crash-loop guard,
                                # and is the right default when the gateway
                                # dies unexpectedly. For a voluntary restart
                                # on update, it's dead time the user watches.
                                #
                                # Shortcut it: ``reset-failed`` + ``start``
                                # skips RestartSec entirely (we're manually
                                # initiating the unit, not waiting for
                                # systemd's auto-restart logic). Takes about
                                # as long as the process takes to come up
                                # (~1-3s on a warm box).
                                #
                                # If the unit is already active because
                                # RestartSec elapsed while we were draining,
                                # ``start`` is a no-op and we fall through to
                                # the poll below. Either way we collapse the
                                # 60s+ delay to a ~5s one.
                                #
                                # The shortcut needs manage-units privileges.
                                # Without them (system service, non-root, no
                                # passwordless sudo) skip it — systemd's own
                                # auto-restart still relaunches the unit after
                                # RestartSec, no privileges required.
                                if _manage_cmd is not None:
                                    subprocess.run(
                                        _manage_cmd + ["reset-failed", svc_name],
                                        capture_output=True,
                                        text=True,
                                        timeout=10,
                                    )
                                    subprocess.run(
                                        _manage_cmd + ["start", svc_name],
                                        capture_output=True,
                                        text=True,
                                        timeout=15,
                                    )
                                    # Short poll: the gateway should be up
                                    # within a few seconds now that we
                                    # bypassed RestartSec.
                                    if _wait_for_service_active(
                                        scope_cmd,
                                        svc_name,
                                        timeout=10.0,
                                    ):
                                        restarted_services.append(svc_name)
                                        continue
                                # Passive poll: systemd's auto-restart fires
                                # after RestartSec regardless of privileges.
                                # This is the primary path when _manage_cmd is
                                # None, and the fallback when the explicit
                                # start didn't take.
                                _restart_sec = _service_restart_sec(
                                    scope_cmd,
                                    svc_name,
                                    default=0.0,
                                )
                                _post_drain_timeout = max(
                                    10.0,
                                    _restart_sec + 10.0,
                                )
                                if _manage_cmd is None and _restart_sec > 5.0:
                                    print(
                                        f"  → {svc_name}: waiting for systemd "
                                        f"auto-restart (~{int(_restart_sec)}s; "
                                        "no root for an immediate restart)..."
                                    )
                                if _wait_for_service_active(
                                    scope_cmd,
                                    svc_name,
                                    timeout=_post_drain_timeout,
                                ):
                                    restarted_services.append(svc_name)
                                    continue
                                # Process exited but wasn't respawned (older
                                # unit without Restart=on-failure or
                                # RestartForceExitStatus=75).  Fall through
                                # to systemctl start/restart.
                                print(
                                    f"  ⚠ {svc_name} drained but didn't relaunch — forcing restart"
                                )

                            # Forcing a restart requires manage-units
                            # privileges.  Without a non-interactive path,
                            # running systemctl here would spawn a polkit
                            # auth prompt inside a captured 10-15s subprocess
                            # — it flashes and dies before the user can
                            # answer.  Skip with clear instructions instead.
                            if _manage_cmd is None:
                                print(
                                    f"  ⚠ {svc_name} is a system service and restarting it needs root.\n"
                                    f"    Restart it manually to load the new version:\n"
                                    f"      sudo systemctl restart {svc_name}\n"
                                    f"    To let `hermes update` restart it automatically, allow\n"
                                    f"    passwordless sudo for systemctl, or run updates with sudo."
                                )
                                continue

                            # Fallback: blunt systemctl restart.  This is
                            # what the old code always did; we get here only
                            # when the graceful path failed (unit missing
                            # SIGUSR1 wiring, drain exceeded the budget,
                            # restart-policy mismatch).
                            #
                            # Always `reset-failed` first.  If systemd's own
                            # auto-restart attempts already parked the unit
                            # in a failed state (transient CHDIR / OOM /
                            # filesystem race after our drain + exit-75),
                            # a plain `systemctl restart` can wedge against
                            # the RestartSec backoff and leave the unit
                            # dead.  Clearing the failed state first makes
                            # the restart idempotent.  Mirrors the recovery
                            # path in `hermes gateway restart`
                            # (`systemd_restart()`) as of PR #20949.
                            subprocess.run(
                                _manage_cmd + ["reset-failed", svc_name],
                                capture_output=True,
                                text=True,
                                timeout=10,
                            )
                            restart = subprocess.run(
                                _manage_cmd + ["restart", svc_name],
                                capture_output=True,
                                text=True,
                                timeout=15,
                            )
                            if restart.returncode == 0:
                                # Verify the service actually survived the
                                # restart.  systemctl restart returns 0 even
                                # if the new process crashes immediately.
                                if _wait_for_service_active(
                                    scope_cmd,
                                    svc_name,
                                    timeout=10.0,
                                ):
                                    restarted_services.append(svc_name)
                                else:
                                    # Retry once — transient startup failures
                                    # (stale module cache, import race) often
                                    # resolve on the second attempt.  Again
                                    # clear any failed state first so the
                                    # retry isn't blocked by the previous
                                    # crash.
                                    print(
                                        f"  ⚠ {svc_name} died after restart, retrying..."
                                    )
                                    subprocess.run(
                                        _manage_cmd + ["reset-failed", svc_name],
                                        capture_output=True,
                                        text=True,
                                        timeout=10,
                                    )
                                    subprocess.run(
                                        _manage_cmd + ["restart", svc_name],
                                        capture_output=True,
                                        text=True,
                                        timeout=15,
                                    )
                                    if _wait_for_service_active(
                                        scope_cmd,
                                        svc_name,
                                        timeout=10.0,
                                    ):
                                        restarted_services.append(svc_name)
                                        print(f"  ✓ {svc_name} recovered on retry")
                                    else:
                                        _scope_flag = "--user " if scope == "user" else ""
                                        _sudo_hint = "sudo " if scope == "system" else ""
                                        print(
                                            f"  ✗ {svc_name} failed to stay running after restart.\n"
                                            f"    Check logs: {_sudo_hint}journalctl {_scope_flag}-u {svc_name} --since '2 min ago'\n"
                                            f"    Recover manually:\n"
                                            f"      {_sudo_hint}systemctl {_scope_flag}reset-failed {svc_name}\n"
                                            f"      {_sudo_hint}systemctl {_scope_flag}restart {svc_name}"
                                        )
                            else:
                                print(
                                    f"  ⚠ Failed to restart {svc_name}: {restart.stderr.strip()}"
                                )
                    except FileNotFoundError:
                        pass
                    except subprocess.TimeoutExpired as exc:
                        # Don't swallow this silently — a wedged systemctl
                        # call here used to make the whole restart phase
                        # vanish with no output (June 2026 report).
                        print(
                            f"  ⚠ systemctl timed out during the {scope}-scope "
                            f"gateway restart ({exc.cmd if exc.cmd else 'unknown command'}). "
                            f"Check the gateway with: hermes gateway status"
                        )

            # --- Launchd services (macOS) ---
            if is_macos():
                try:
                    from hermes_cli.gateway import (
                        launchd_restart,
                        get_launchd_label,
                        get_launchd_plist_path,
                    )

                    plist_path = get_launchd_plist_path()
                    if plist_path.exists():
                        check = subprocess.run(
                            ["launchctl", "list", get_launchd_label()],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if check.returncode == 0:
                            try:
                                launchd_restart()
                                restarted_services.append(get_launchd_label())
                            except subprocess.CalledProcessError as e:
                                stderr = (getattr(e, "stderr", "") or "").strip()
                                print(f"  ⚠ Gateway restart failed: {stderr}")
                except (FileNotFoundError, subprocess.TimeoutExpired, ImportError):
                    pass

            # --- Manual (non-service) gateways ---
            # Kill any remaining gateway processes not managed by a service.
            # Exclude PIDs that belong to just-restarted services so we don't
            # immediately kill the process that systemd/launchd just spawned.
            service_pids = _get_service_pids()
            manual_pids = find_gateway_pids(
                exclude_pids=service_pids, all_profiles=True
            )
            profile_processes = {
                proc.pid: proc
                for proc in find_profile_gateway_processes(exclude_pids=service_pids)
                if proc.pid in manual_pids
            }
            for pid, proc in profile_processes.items():
                if not launch_detached_profile_gateway_restart(proc.profile, pid):
                    continue
                # Prefer a graceful SIGUSR1 drain so in-flight agent runs
                # finish before the watcher respawns the gateway.  If the
                # gateway doesn't support SIGUSR1 or doesn't exit within
                # the drain budget, fall back to SIGTERM — the watcher
                # still sees the exit and relaunches either way.
                # Announce the drain first: this wait can hold for the full
                # budget per gateway with no other output, and on surfaces
                # that stream update progress (the desktop updater most of
                # all) the silence reads as a hung update (#44515).
                print(
                    f"  → {proc.profile}: draining gateway PID {pid} "
                    f"(up to {int(_drain_budget)}s)..."
                )
                drained = _graceful_restart_via_sigusr1(
                    pid,
                    drain_timeout=_drain_budget,
                )
                if not drained:
                    try:
                        os.kill(pid, _signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                # Wait for the old process to fully exit before the watcher
                # spawns the new gateway.  Telegram holds the previous
                # getUpdates long-poll session open on its servers for up to
                # ~30s after the client disconnects.  If the new gateway
                # connects before that window expires it receives a 409
                # Conflict, which _handle_polling_conflict() recovers from
                # via back-off retries — but a brief wait here reduces the
                # chance of hitting that path at all, especially on fast
                # machines where the watcher loop restarts in < 1s.
                # We wait up to 5s for the process to exit (the OS-level
                # close, not the Telegram server-side expiry), then let the
                # watcher take over.  The Telegram adapter's retry logic
                # handles any remaining 409s if the server session is still
                # live when the new gateway polls.
                _wait_for_gateway_exit(timeout=5.0, force_after=None)
                killed_pids.add(pid)
                relaunched_profiles.append(proc.profile)

            for pid in manual_pids:
                if pid in profile_processes:
                    continue
                try:
                    os.kill(pid, _signal.SIGTERM)
                    killed_pids.add(pid)
                except (ProcessLookupError, PermissionError):
                    pass

            if restarted_services or killed_pids:
                print()
                for svc in restarted_services:
                    print(f"  ✓ Restarted {svc}")
                if relaunched_profiles:
                    names = ", ".join(relaunched_profiles)
                    print(f"  ✓ Restarting manual gateway profile(s): {names}")
                unmapped_count = len(killed_pids) - len(relaunched_profiles)
                if unmapped_count:
                    print(f"  → Stopped {unmapped_count} manual gateway process(es)")
                    print("    Restart manually: hermes gateway run")
                    if unmapped_count > 1:
                        print(
                            "    (or: hermes -p <profile> gateway run  for each profile)"
                        )

            if not restarted_services and not killed_pids:
                # No gateways were running — nothing to do
                pass

            # --- Post-restart survivor sweep -----------------------------
            # Issue #17648: some gateways ignore SIGTERM (stuck drain,
            # blocked I/O, PID dead but zombie).  The detached profile
            # watchers wait 120s for the old PID to exit — if it never
            # does, no respawn happens and the user keeps hitting
            # ImportError against a stale sys.modules.  Give the
            # graceful paths a brief window to complete, then SIGKILL
            # any remaining pre-update PIDs so the watcher / service
            # manager can relaunch with fresh code.
            try:
                _time.sleep(3.0)
                _service_pids_after = _get_service_pids()
                _surviving = find_gateway_pids(
                    exclude_pids=_service_pids_after,
                    all_profiles=True,
                )
                # Scope to PIDs we already tried to kill during this
                # update (killed_pids).  Anything new is a gateway that
                # started AFTER our restart attempt — respecting user
                # intent, we don't kill those.
                _stuck = [pid for pid in _surviving if pid in killed_pids]
                if _stuck:
                    print()
                    print(
                        f"  ⚠ {len(_stuck)} gateway process(es) ignored SIGTERM — force-killing"
                    )
                    from gateway.status import terminate_pid as _terminate_pid
                    for pid in _stuck:
                        try:
                            # Routes through taskkill /T /F on Windows,
                            # SIGKILL on POSIX — _signal.SIGKILL doesn't
                            # exist on Windows so the old raw os.kill call
                            # used to crash the entire update path.
                            _terminate_pid(pid, force=True)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                    # Give the OS a beat to reap the processes so the
                    # watchers see them exit and respawn.
                    _time.sleep(1.5)
            except Exception as _sweep_exc:
                logger.debug("Post-restart survivor sweep failed: %s", _sweep_exc)

        except Exception as e:
            logger.debug("Gateway restart during update failed: %s", e)

        _resume_windows_gateways_after_update(_windows_gateway_resume)

        # Warn if legacy Hermes gateway unit files are still installed.
        # When both hermes.service (from a pre-rename install) and the
        # current hermes-gateway.service are enabled, they SIGTERM-fight
        # for the same bot token (see PR #11909). Flagging here means
        # every `hermes update` surfaces the issue until the user migrates.
        try:
            from hermes_cli.gateway import (
                has_legacy_hermes_units,
                _find_legacy_hermes_units,
                supports_systemd_services,
            )

            if supports_systemd_services() and has_legacy_hermes_units():
                print()
                print("⚠ Legacy Hermes gateway unit(s) detected:")
                for name, path, is_sys in _find_legacy_hermes_units():
                    scope = "system" if is_sys else "user"
                    print(f"    {path}  ({scope} scope)")
                print()
                print("  These pre-rename units (hermes.service) fight the current")
                print("  hermes-gateway.service for the bot token and cause SIGTERM")
                print("  flap loops. Remove them with:")
                print()
                print("    hermes gateway migrate-legacy")
                print()
                print("  (add `sudo` if any are in system scope)")
        except Exception as e:
            logger.debug("Legacy unit check during update failed: %s", e)

        # Kill stale dashboard processes — the dashboard has no service
        # manager, so leaving it alive after a code update produces a
        # silent frontend/backend mismatch.  We can't auto-restart it
        # (no saved launch args) but we can stop it, and a hint is
        # printed for the user to re-launch.
        _kill_stale_dashboard_processes()

        print()
        print("Tip: You can now select a provider and model:")
        print("  hermes model              # Select provider and model")

    except subprocess.CalledProcessError as e:
        if sys.platform == "win32":
            print(f"⚠ Git update failed: {e}")
            print("→ Falling back to ZIP download...")
            print()
            _update_via_zip(args)
        else:
            print(f"✗ Update failed: {e}")
            sys.exit(1)


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
                        for raw in env_path.read_text().splitlines():
                            line = raw.strip()
                            if not line or line.startswith("#"):
                                continue
                            key = line.split("=", 1)[0].strip()
                            if key == er.name:
                                already = True
                                break
                    except OSError:
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
    """Print ``hermes dashboard`` PIDs and return the count.

    Uses the same detection logic as ``_find_stale_dashboard_pids`` (the
    current process is excluded, but since ``hermes dashboard --status``
    runs in a short-lived CLI process that never matches the pattern,
    the exclusion is irrelevant here).
    """
    pids = _find_stale_dashboard_pids()
    if not pids:
        print("No hermes dashboard processes running.")
        return 0

    print(f"{len(pids)} hermes dashboard process(es) running:")
    for pid in pids:
        # Best-effort: show the full cmdline so users can tell profiles apart.
        cmdline = ""
        try:
            if sys.platform != "win32":
                cmdline_path = f"/proc/{pid}/cmdline"
                if os.path.exists(cmdline_path):
                    with open(cmdline_path, "rb") as f:
                        cmdline = (
                            f.read()
                            .replace(b"\x00", b" ")
                            .decode("utf-8", errors="replace")
                            .strip()
                        )
        except (OSError, ValueError):
            pass
        if cmdline:
            print(f"    PID {pid}: {cmdline}")
        else:
            print(f"    PID {pid}")
    return len(pids)


def _dashboard_listening(host: str, port: int) -> bool:
    """True when something is accepting TCP connections at host:port.

    Any listener counts — even a 401 response proves a dashboard is up.
    Used by the unified profile-launch routing to decide attach-vs-start.
    """
    import socket

    try:
        with socket.create_connection((host or "127.0.0.1", port), timeout=1.5):
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

        cfg = load_config()
        dash = cfg.setdefault("dashboard", {})
        basic = dash.setdefault("basic_auth", {})
        basic["username"] = username
        basic["password_hash"] = password_hash
        # Never persist plaintext: clear any stale plaintext password key.
        basic["password"] = ""
        if not str(basic.get("secret", "") or "").strip():
            basic["secret"] = secret
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


def cmd_dashboard(args):
    """Start the web UI server, or (with --stop/--status) manage running ones."""
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
        _kill_stale_dashboard_processes(reason="requested via --stop")
        # _kill_stale_dashboard_processes prints outcomes itself.  Exit 0 if
        # we killed at least one, 1 if they were all unkillable.
        remaining = _find_stale_dashboard_pids()
        sys.exit(1 if remaining else 0)

    # `serve` is the headless backend: no UI build, no SPA mount, neutral
    # ready sentinel. Resolved once and threaded through the re-exec, the
    # build gate, and start_server.
    _headless_backend = getattr(args, "headless_backend", False)

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
        if args.no_open:
            reexec_argv.append("--no-open")
        if getattr(args, "insecure", False):
            reexec_argv.append("--insecure")
        if getattr(args, "skip_build", False):
            reexec_argv.append("--skip-build")
        env = os.environ.copy()
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
            print(f"✗ --skip-build was passed but no web dist found at: {_dist_root}")
            print("  Pre-build first:  npm install --workspace web && npm run build -w web")
            print("  Or drop --skip-build to build automatically.")
            sys.exit(1)
        print(f"→ Skipping web UI build (--skip-build); using dist at {_dist_root}")

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
        "acp", "auth", "backup", "bundles", "checkpoints", "claw", "completion",
        "computer-use",
        "config", "console", "cron", "curator", "dashboard", "serve", "debug", "doctor",
        "dump", "fallback", "gateway", "hooks", "import", "insights",
        "gui", "desktop", "kanban", "login", "logout", "logs", "lsp", "mcp", "memory", "migrate", "moa",
        "journey", "memory-graph", "learning",
        "model", "pairing", "pets", "plugins", "portal", "postinstall", "profile",
        "project", "proxy",
        "prompt-size",
        "send", "sessions", "setup",
        "skills", "slack", "status", "tools", "uninstall", "update",
        "version", "webhook", "whatsapp", "whatsapp-cloud", "chat", "secrets", "security",
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
    _apply_safe_mode(args)

    _sub_attr, _sub_set = _AGENT_SUBCOMMANDS.get(args.command, (None, None))
    if not (
        args.command in _AGENT_COMMANDS
        or (_sub_attr and getattr(args, _sub_attr, None) in _sub_set)
    ):
        return

    _accept_hooks = bool(getattr(args, "accept_hooks", False))
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
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

        register_from_config(load_config(), accept_hooks=_accept_hooks)
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
        from hermes_cli.oneshot import run_oneshot

        sys.exit(
            run_oneshot(
                args.oneshot,
                model=getattr(args, "model", None),
                provider=getattr(args, "provider", None),
                toolsets=getattr(args, "toolsets", None),
                usage_file=getattr(args, "usage_file", None),
            )
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
    try:
        from hermes_state import SessionDB
        from agent.insights import InsightsEngine

        db = SessionDB()
        engine = InsightsEngine(db)
        report = engine.generate(days=args.days, source=args.source)
        print(engine.format_terminal(report))
        db.close()
    except Exception as e:
        print(f"Error generating insights: {e}")


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


def main():
    """Main entry point for hermes CLI."""
    # Cosmetic: make the process show up as 'hermes' instead of 'python3.11'
    # in ps/top/htop.  Non-fatal — just a nicer UX.
    _set_process_title()

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

    # Lazy import — only pays for itself when this subcommand is actually used.
    from hermes_cli import secrets_cli as _secrets_cli
    from hermes_cli import onepassword_secrets_cli as _op_secrets_cli

    _secrets_cli.register_cli(secrets_bw)
    _op_secrets_cli.register_cli(secrets_op)

    def _dispatch_secrets(args):  # noqa: ANN001
        sub = getattr(args, "secrets_command", None)
        bw_sub = getattr(args, "secrets_bw_command", None)
        op_sub = getattr(args, "secrets_op_command", None)
        if sub in ("bitwarden", "bw") and bw_sub is not None:
            return args.func(args)
        if sub in ("onepassword", "op", "1password") and op_sub is not None:
            return args.func(args)
        secrets_parser.print_help()
        return 0

    secrets_parser.set_defaults(func=_dispatch_secrets)

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
    # postinstall command  (parser built in hermes_cli/subcommands/postinstall.py)
    # =========================================================================
    build_postinstall_parser(subparsers, cmd_postinstall=cmd_postinstall)

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
    # cron command  (parser built in hermes_cli/subcommands/cron.py)
    # =========================================================================
    build_cron_parser(subparsers, cmd_cron=cmd_cron)

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
    # security command — on-demand supply-chain audit
    # =========================================================================
    # security command  (parser built in hermes_cli/subcommands/security.py)
    # =========================================================================
    build_security_parser(subparsers, cmd_security=cmd_security)

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
    # config command  (parser built in hermes_cli/subcommands/config.py)
    # =========================================================================
    build_config_parser(subparsers, cmd_config=cmd_config)

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
            import shutil
            import subprocess
            from hermes_cli.tools_config import _cua_driver_cmd
            # Honor HERMES_CUA_DRIVER_CMD for local-build testing — same
            # resolver `install_cua_driver` and the runtime backend use,
            # so `status` reports what `computer_use` will actually invoke.
            driver_cmd = _cua_driver_cmd()
            path = shutil.which(driver_cmd)
            if path:
                version = ""
                try:
                    from hermes_cli.tools_config import _cua_driver_env
                    version = subprocess.run(
                        [path, "--version"],
                        capture_output=True, text=True, timeout=5,
                        env=_cua_driver_env(),
                    ).stdout.strip()
                except Exception:
                    pass
                if version:
                    print(f"cua-driver: installed at {path} ({version})")
                else:
                    print(f"cua-driver: installed at {path}")
                try:
                    from tools.computer_use.cua_backend import cua_driver_update_check
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

    sessions_export = sessions_subparsers.add_parser(
        "export", help="Export sessions to a JSONL file"
    )
    sessions_export.add_argument(
        "output", help="Output JSONL file path (use - for stdout)"
    )
    sessions_export.add_argument("--source", help="Filter by source")
    sessions_export.add_argument("--session-id", help="Export a specific session")

    sessions_delete = sessions_subparsers.add_parser(
        "delete", help="Delete a specific session"
    )
    sessions_delete.add_argument("session_id", help="Session ID to delete")
    sessions_delete.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation"
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
            help="Only match sessions started within the last AGE "
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

    sessions_subparsers.add_parser("stats", help="Show session store statistics")

    sessions_rename = sessions_subparsers.add_parser(
        "rename", help="Set or change a session's title"
    )
    sessions_rename.add_argument("session_id", help="Session ID to rename")
    sessions_rename.add_argument("title", nargs="+", help="New title for the session")

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

    def _confirm_prompt(prompt: str) -> bool:
        """Prompt for y/N confirmation, safe against non-TTY environments."""
        try:
            return input(prompt).strip().lower() in {"y", "yes"}
        except (EOFError, KeyboardInterrupt):
            return False

    def cmd_sessions(args):
        import json as _json

        action = args.sessions_action

        # 'repair' must run BEFORE opening SessionDB(): a malformed schema is
        # exactly the case where SessionDB() can't open, so it operates on the
        # raw file path instead.
        if action == "repair":
            from hermes_state import (
                DEFAULT_DB_PATH,
                _db_opens_cleanly,
                repair_state_db_schema,
            )

            db_path = DEFAULT_DB_PATH
            if not db_path.exists():
                print(f"No session database at {db_path} (nothing to repair).")
                return
            reason = _db_opens_cleanly(db_path)
            if reason is None:
                print(f"✓ {db_path} opens cleanly — no repair needed.")
                return
            print(f"✗ {db_path} does not open cleanly: {reason}")
            if getattr(args, "check_only", False):
                return
            print("Repairing (a backup copy is made first)…")
            report = repair_state_db_schema(
                db_path, backup=not getattr(args, "no_backup", False)
            )
            if report.get("repaired"):
                if report.get("backup_path"):
                    print(f"  backup: {report['backup_path']}")
                print(f"  strategy: {report.get('strategy')}")
                try:
                    from hermes_state import SessionDB

                    n = SessionDB()._conn.execute(
                        "SELECT COUNT(*) FROM sessions"
                    ).fetchone()[0]
                    print(f"✓ Repaired — {n} sessions recovered.")
                except Exception:
                    print("✓ Repaired.")
            else:
                print(f"✗ Repair failed: {report.get('error')}")
                if report.get("backup_path"):
                    print(f"  A backup is preserved at: {report['backup_path']}")
                print("  Keep state.db and the backup; do not delete them.")
            return

        try:
            from hermes_state import SessionDB

            db = SessionDB()
        except Exception as e:
            print(f"Error: Could not open session database: {e}")
            return

        # Hide third-party tool sessions by default, but honour explicit --source
        _source = getattr(args, "source", None)
        _exclude = None if _source else ["tool"]

        if action == "list":
            sessions = db.list_sessions_rich(
                source=args.source, exclude_sources=_exclude, limit=args.limit
            )
            if not sessions:
                print("No sessions found.")
                return
            has_titles = any(s.get("title") for s in sessions)
            if has_titles:
                print(f"{'Title':<32} {'Preview':<40} {'Last Active':<13} {'ID'}")
                print("─" * 110)
            else:
                print(f"{'Preview':<50} {'Last Active':<13} {'Src':<6} {'ID'}")
                print("─" * 95)
            for s in sessions:
                last_active = _relative_time(s.get("last_active"))
                preview = (
                    s.get("preview", "")[:38]
                    if has_titles
                    else s.get("preview", "")[:48]
                )
                if has_titles:
                    title = (s.get("title") or "—")[:30]
                    sid = s["id"]
                    print(f"{title:<32} {preview:<40} {last_active:<13} {sid}")
                else:
                    sid = s["id"]
                    print(f"{preview:<50} {last_active:<13} {s['source']:<6} {sid}")

        elif action == "export":
            if args.session_id:
                resolved_session_id = db.resolve_session_id(args.session_id)
                if not resolved_session_id:
                    print(f"Session '{args.session_id}' not found.")
                    return
                data = db.export_session(resolved_session_id)
                if not data:
                    print(f"Session '{args.session_id}' not found.")
                    return
                line = _json.dumps(data, ensure_ascii=False) + "\n"
                if args.output == "-":

                    sys.stdout.write(line)
                else:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(line)
                    print(f"Exported 1 session to {args.output}")
            else:
                sessions = db.export_all(source=args.source)
                if args.output == "-":

                    for s in sessions:
                        sys.stdout.write(_json.dumps(s, ensure_ascii=False) + "\n")
                else:
                    with open(args.output, "w", encoding="utf-8") as f:
                        for s in sessions:
                            f.write(_json.dumps(s, ensure_ascii=False) + "\n")
                    print(f"Exported {len(sessions)} sessions to {args.output}")

        elif action == "delete":
            resolved_session_id = db.resolve_session_id(args.session_id)
            if not resolved_session_id:
                print(f"Session '{args.session_id}' not found.")
                return
            if not args.yes:
                if not _confirm_prompt(
                    f"Delete session '{resolved_session_id}' and all its messages? [y/N] "
                ):
                    print("Cancelled.")
                    return
            sessions_dir = get_hermes_home() / "sessions"
            if db.delete_session(resolved_session_id, sessions_dir=sessions_dir):
                print(f"Deleted session '{resolved_session_id}'.")
            else:
                print(f"Session '{args.session_id}' not found.")

        elif action in ("prune", "archive"):
            from hermes_cli.session_filters import (
                build_prune_filters,
                describe_filters,
                format_epoch,
            )

            # Preserve the historical default ONLY for a truly bare
            # `hermes sessions prune`: no time window and no filters at all
            # means "older than 90 days". ANY filter — including --source —
            # suppresses the implicit cutoff, so `prune --source cron`
            # matches ALL cron sessions regardless of age. The preview +
            # confirmation below (count, oldest/newest) is the safety net.
            _non_time_filters = any(
                getattr(args, a, None) is not None
                for a in (
                    "source", "title", "end_reason", "cwd",
                    "min_messages", "max_messages", "model", "provider",
                    "user", "chat_id", "chat_type", "branch",
                    "min_tokens", "max_tokens", "min_cost", "max_cost",
                    "min_tool_calls", "max_tool_calls",
                )
            )
            if (
                action == "prune"
                and args.older_than is None
                and args.newer_than is None
                and args.before is None
                and args.after is None
                and not _non_time_filters
            ):
                args.older_than = "90"

            try:
                filters = build_prune_filters(args)
            except ValueError as e:
                print(f"Error: {e}")
                return

            if action == "archive" and not any(
                v for k, v in filters.items() if k != "older_than_days"
            ):
                print(
                    "Refusing to archive every ended session: pass at least one "
                    "filter (e.g. --newer-than 5h, --source cli, --title codex)."
                )
                return

            # Prune skips archived sessions unless --include-archived;
            # archive only targets not-yet-archived rows (idempotent).
            if action == "prune":
                filters["archived"] = (
                    None if getattr(args, "include_archived", False) else False
                )
            else:
                filters["archived"] = False

            candidates = db.list_prune_candidates(**filters)
            verb = "Delete" if action == "prune" else "Archive"
            if not candidates:
                print(f"No sessions match ({describe_filters(filters)}).")
                return

            # Candidates are ordered oldest-first — surface the age span so
            # the confirmation makes the blast radius obvious.
            _oldest = candidates[0].get("started_at")
            _newest = candidates[-1].get("started_at")
            _span = (
                f"oldest {format_epoch(_oldest)}, newest {format_epoch(_newest)}"
            )

            if args.dry_run or not args.yes:
                shown = candidates if args.dry_run else candidates[:15]
                print(
                    f"{len(candidates)} session(s) match "
                    f"({describe_filters(filters)}; {_span}):"
                )
                for s in shown:
                    title = (s.get("title") or "")[:36]
                    model = (s.get("model") or "-").split("/")[-1][:24]
                    print(
                        f"  {s['id']}  {format_epoch(s['started_at']):<17} "
                        f"{s['source']:<10} {model:<24} "
                        f"{s['message_count']:>4} msgs  {title}"
                    )
                if len(candidates) > len(shown):
                    print(f"  … and {len(candidates) - len(shown)} more")
                if args.dry_run:
                    print(f"Dry run — nothing {'deleted' if action == 'prune' else 'archived'}.")
                    return

            if not args.yes:
                if not _confirm_prompt(
                    f"{verb} these {len(candidates)} session(s) ({_span})? [y/N] "
                ):
                    print("Cancelled.")
                    return

            if action == "prune":
                sessions_dir = get_hermes_home() / "sessions"
                count = db.prune_sessions(sessions_dir=sessions_dir, **filters)
                print(f"Pruned {count} session(s).")
            else:
                count = db.archive_sessions(**filters)
                print(
                    f"Archived {count} session(s). They're hidden from listings "
                    "but fully recoverable (nothing was deleted)."
                )

        elif action == "rename":
            resolved_session_id = db.resolve_session_id(args.session_id)
            if not resolved_session_id:
                print(f"Session '{args.session_id}' not found.")
                return
            title = " ".join(args.title)
            try:
                if db.set_session_title(resolved_session_id, title):
                    print(f"Session '{resolved_session_id}' renamed to: {title}")
                else:
                    print(f"Session '{args.session_id}' not found.")
            except ValueError as e:
                print(f"Error: {e}")

        elif action == "browse":
            limit = getattr(args, "limit", 500) or 500
            source = getattr(args, "source", None)
            _browse_exclude = None if source else ["tool"]
            sessions = db.list_sessions_rich(
                source=source, exclude_sources=_browse_exclude, limit=limit
            )
            db.close()
            if not sessions:
                print("No sessions found.")
                return

            selected_id = _session_browse_picker(sessions)
            if not selected_id:
                print("Cancelled.")
                return

            # Launch hermes --resume <id> by replacing the current process
            print(f"Resuming session: {selected_id}")
            from hermes_cli.relaunch import relaunch

            relaunch(["--resume", selected_id])
            return  # won't reach here after execvp

        elif action == "optimize":
            db_path = db.db_path
            before_mb = (
                os.path.getsize(db_path) / (1024 * 1024)
                if db_path.exists()
                else 0.0
            )
            print("Optimizing session store (FTS merge + VACUUM)…")
            try:
                # vacuum() merges FTS5 segments (optimize_fts) then VACUUMs,
                # and returns the number of indexes it merged.
                n = db.vacuum()
            except Exception as e:
                print(f"Error: optimization failed: {e}")
                db.close()
                return
            after_mb = (
                os.path.getsize(db_path) / (1024 * 1024)
                if db_path.exists()
                else 0.0
            )
            saved = before_mb - after_mb
            print(f"Optimized {n} FTS index(es).")
            print(
                f"Database size: {before_mb:.1f} MB -> {after_mb:.1f} MB "
                f"(reclaimed {saved:.1f} MB)"
            )

        elif action == "stats":
            total = db.session_count()
            msgs = db.message_count()
            print(f"Total sessions: {total}")
            print(f"Total messages: {msgs}")
            for src in ["cli", "telegram", "discord", "whatsapp", "slack"]:
                c = db.session_count(source=src)
                if c > 0:
                    print(f"  {src}: {c} sessions")
            db_path = db.db_path
            if db_path.exists():
                size_mb = os.path.getsize(db_path) / (1024 * 1024)
                print(f"Database size: {size_mb:.1f} MB")

        else:
            sessions_parser.print_help()

        db.close()

    sessions_parser.set_defaults(func=cmd_sessions)

    # =========================================================================
    # insights command  (parser built in hermes_cli/subcommands/insights.py)
    # =========================================================================
    build_insights_parser(subparsers, cmd_insights=cmd_insights)

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

    # Discover Python plugins and register shell hooks once, before any
    # command that can fire lifecycle hooks.  Both are idempotent; gated
    # so introspection/management commands (hermes hooks list, cron
    # list, gateway status, mcp add, ...) don't pay discovery cost or
    # trigger consent prompts for hooks the user is still inspecting.
    _prepare_agent_startup(args)

    # Handle top-level --oneshot / -z: single-shot mode, stdout = final
    # response only, nothing else. Bypasses cli.py entirely.
    if getattr(args, "oneshot", None):
        from hermes_cli.oneshot import run_oneshot

        sys.exit(
            run_oneshot(
                args.oneshot,
                model=getattr(args, "model", None),
                provider=getattr(args, "provider", None),
                toolsets=getattr(args, "toolsets", None),
                usage_file=getattr(args, "usage_file", None),
            )
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

    # Execute the command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
