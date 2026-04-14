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

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

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
    """Pre-parse --profile/-p and set HERMES_HOME before module imports."""
    argv = sys.argv[1:]
    profile_name = None
    consume = 0

    # 1. Check for explicit -p / --profile flag
    for i, arg in enumerate(argv):
        if arg in ("--profile", "-p") and i + 1 < len(argv):
            profile_name = argv[i + 1]
            consume = 2
            break
        elif arg.startswith("--profile="):
            profile_name = arg.split("=", 1)[1]
            consume = 1
            break

    # 2. If no flag, check active_profile in the hermes root
    if profile_name is None:
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
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            # A bug in profiles.py must NEVER prevent hermes from starting
            print(f"Warning: profile override failed ({exc}), using default", file=sys.stderr)
            return
        os.environ["HERMES_HOME"] = hermes_home
        # Strip the flag from argv so argparse doesn't choke
        if consume > 0:
            for i, arg in enumerate(argv):
                if arg in ("--profile", "-p"):
                    start = i + 1  # +1 because argv is sys.argv[1:]
                    sys.argv = sys.argv[:start] + sys.argv[start + consume:]
                    break
                elif arg.startswith("--profile="):
                    start = i + 1
                    sys.argv = sys.argv[:start] + sys.argv[start + 1:]
                    break

_apply_profile_override()

# Load .env from ~/.hermes/.env first, then project root as dev fallback.
# User-managed env files should override stale shell exports on restart.
from hermes_cli.config import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv(project_env=PROJECT_ROOT / '.env')

# Initialize centralized file logging early — all `hermes` subcommands
# (chat, setup, gateway, config, etc.) write to agent.log + errors.log.
try:
    from hermes_logging import setup_logging as _setup_logging
    _setup_logging(mode="cli")
except Exception:
    pass  # best-effort — don't crash the CLI if logging setup fails

# Apply IPv4 preference early, before any HTTP clients are created.
try:
    from hermes_cli.config import load_config as _load_config_early
    from hermes_constants import apply_ipv4_preference as _apply_ipv4
    _early_cfg = _load_config_early()
    _net = _early_cfg.get("network", {})
    if isinstance(_net, dict) and _net.get("force_ipv4"):
        _apply_ipv4(force=True)
    del _early_cfg, _net
except Exception:
    pass  # best-effort — don't crash if config isn't available yet

import logging
import time as _time
from datetime import datetime

from hermes_cli import __version__, __release_date__
from hermes_constants import OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)


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
    provider_env_vars = {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "OPENAI_BASE_URL"}
    for pconfig in PROVIDER_REGISTRY.values():
        if pconfig.auth_type == "api_key":
            provider_env_vars.update(pconfig.api_key_env_vars)
    if any(os.getenv(v) for v in provider_env_vars):
        return True

    # Check .env file for keys
    env_file = get_env_path()
    if env_file.exists():
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
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
            from agent.anthropic_adapter import read_claude_code_credentials, is_claude_code_token_valid
            creds = read_claude_code_credentials()
            if creds and (is_claude_code_token_valid(creds) or creds.get("refreshToken")):
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
                curses.init_pair(1, curses.COLOR_GREEN, -1)   # selected
                curses.init_pair(2, curses.COLOR_YELLOW, -1)  # header
                curses.init_pair(3, curses.COLOR_CYAN, -1)    # search
                curses.init_pair(4, 8, -1)                    # dim

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
                    dim_attr = curses.color_pair(4) if curses.has_colors() else curses.A_DIM
                    stdscr.addnstr(1, 0, col_header, max_x - 1, dim_attr)
                except curses.error:
                    pass

                # Compute visible area
                visible_rows = max_y - 4  # header + col header + blank + footer
                if visible_rows < 1:
                    visible_rows = 1

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
                    if cursor < 0:
                        cursor = 0
                    if cursor < scroll_offset:
                        scroll_offset = cursor
                    elif cursor >= scroll_offset + visible_rows:
                        scroll_offset = cursor - visible_rows + 1

                    for draw_i, i in enumerate(range(
                        scroll_offset,
                        min(len(filtered), scroll_offset + visible_rows)
                    )):
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
                    stdscr.addnstr(footer_y, 0, footer, max_x - 1,
                                   curses.color_pair(4) if curses.has_colors() else curses.A_DIM)
                except curses.error:
                    pass

                stdscr.refresh()
                key = stdscr.getch()

                if key in (curses.KEY_UP, ):
                    if filtered:
                        cursor = (cursor - 1) % len(filtered)
                elif key in (curses.KEY_DOWN, ):
                    if filtered:
                        cursor = (cursor + 1) % len(filtered)
                elif key in (curses.KEY_ENTER, 10, 13):
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
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    if search_text:
                        search_text = search_text[:-1]
                        if search_text:
                            filtered = [s for s in sessions if _match(s, search_text)]
                        else:
                            filtered = list(sessions)
                        cursor = 0
                        scroll_offset = 0
                elif key == ord('q') and not search_text:
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
            if not val or val.lower() in ("q", "quit", "exit"):
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


def _resolve_last_cli_session() -> Optional[str]:
    """Look up the most recent CLI session ID from SQLite. Returns None if unavailable."""
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        sessions = db.search_sessions(source="cli", limit=1)
        db.close()
        if sessions:
            return sessions[0]["id"]
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
    import shutil

    backend = container_info["backend"]
    container_name = container_info["container_name"]
    exec_user = container_info["exec_user"]
    hermes_bin = container_info["hermes_bin"]

    runtime = shutil.which(backend)
    if not runtime:
        print(f"Error: {backend} not found on PATH. Cannot route to container.",
              file=sys.stderr)
        sys.exit(1)

    # Rootful containers (NixOS systemd service) are invisible to unprivileged
    # users — Podman uses per-user namespaces, Docker needs group access.
    # Probe whether the runtime can see the container; if not, try via sudo.
    sudo_path = None
    probe = _probe_container(
        [runtime, "inspect", "--format", "ok", container_name], backend,
    )
    if probe.returncode != 0:
        sudo_path = shutil.which("sudo")
        if sudo_path:
            probe2 = _probe_container(
                [sudo_path, "-n", runtime, "inspect", "--format", "ok", container_name],
                backend, via_sudo=True,
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
                    f'  security.sudo.extraRules = [{{\n'
                    f'    users = [ "{os.getenv("USER", "your-user")}" ];\n'
                    f'    commands = [{{ command = "{runtime}"; options = [ "NOPASSWD" ]; }}];\n'
                    f'  }}];\n'
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
        cmd_prefix + ["exec"]
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
    """
    try:
        from hermes_state import SessionDB
        db = SessionDB()

        # Try as exact session ID first
        session = db.get_session(name_or_id)
        if session:
            db.close()
            return session["id"]

        # Try as title (with auto-latest for lineage)
        session_id = db.resolve_session_by_title(name_or_id)
        db.close()
        return session_id
    except Exception:
        pass
    return None


def cmd_chat(args):
    """Run interactive chat CLI."""
    # Resolve --continue into --resume with the latest CLI session or by name
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
            last_id = _resolve_last_cli_session()
            if last_id:
                args.resume = last_id
            else:
                print("No previous CLI session found to continue.")
                sys.exit(1)

    # Resolve --resume by title if it's not a direct session ID
    resume_val = getattr(args, "resume", None)
    if resume_val:
        resolved = _resolve_session_by_name_or_id(resume_val)
        if resolved:
            args.resume = resolved
        # If resolution fails, keep the original value — _init_agent will
        # report "Session not found" with the original input

    # First-run guard: check if any provider is configured before launching
    if not _has_any_provider_configured():
        print()
        print("It looks like Hermes isn't configured yet -- no API keys or providers found.")
        print()
        print("  Run:  hermes setup")
        print()

        from hermes_cli.setup import is_interactive_stdin, print_noninteractive_setup_guidance

        if not is_interactive_stdin():
            print_noninteractive_setup_guidance(
                "No interactive TTY detected for the first-run setup prompt."
            )
            sys.exit(1)

        try:
            reply = input("Run setup now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = "n"
        if reply in ("", "y", "yes"):
            cmd_setup(args)
            return
        print()
        print("You can run 'hermes setup' at any time to configure.")
        sys.exit(1)

    # Start update check in background (runs while other init happens)
    try:
        from hermes_cli.banner import prefetch_update_check
        prefetch_update_check()
    except Exception:
        pass

    # Sync bundled skills on every CLI launch (fast -- skips unchanged skills)
    try:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)
    except Exception:
        pass

    # --yolo: bypass all dangerous command approvals
    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"

    # --source: tag session source for filtering (e.g. 'tool' for third-party integrations)
    if getattr(args, "source", None):
        os.environ["HERMES_SESSION_SOURCE"] = args.source

    # Import and run the CLI
    from cli import main as cli_main
    
    # Build kwargs from args
    kwargs = {
        "model": args.model,
        "provider": getattr(args, "provider", None),
        "toolsets": args.toolsets,
        "skills": getattr(args, "skills", None),
        "verbose": args.verbose,
        "quiet": getattr(args, "quiet", False),
        "query": args.query,
        "image": getattr(args, "image", None),
        "resume": getattr(args, "resume", None),
        "worktree": getattr(args, "worktree", False),
        "checkpoints": getattr(args, "checkpoints", False),
        "pass_session_id": getattr(args, "pass_session_id", False),
        "max_turns": getattr(args, "max_turns", None),
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
    from hermes_cli.gateway import gateway_command
    gateway_command(args)


def cmd_whatsapp(args):
    """Set up WhatsApp: choose mode, configure, install bridge, pair via QR."""
    _require_tty("whatsapp")
    import subprocess
    from pathlib import Path
    from hermes_cli.config import get_env_value, save_env_value

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
        print("     Requires a second phone number with WhatsApp installed on a device.")
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
        mode_label = "separate bot number" if wa_mode == "bot" else "personal number (self-chat)"
        print(f"\n✓ Mode: {mode_label}")

    # ── Step 2: Enable WhatsApp ──────────────────────────────────────────
    print()
    current = get_env_value("WHATSAPP_ENABLED")
    if current and current.lower() == "true":
        print("✓ WhatsApp is already enabled")
    else:
        save_env_value("WHATSAPP_ENABLED", "true")
        print("✓ WhatsApp enabled")

    # ── Step 3: Allowed users ────────────────────────────────────────────
    current_users = get_env_value("WHATSAPP_ALLOWED_USERS") or ""
    if current_users:
        print(f"✓ Allowed users: {current_users}")
        try:
            response = input("\n  Update allowed users? [y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in ("y", "yes"):
            if wa_mode == "bot":
                phone = input("  Phone numbers that can message the bot (comma-separated): ").strip()
            else:
                phone = input("  Your phone number (e.g. 15551234567): ").strip()
            if phone:
                save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
                print(f"  ✓ Updated to: {phone}")
    else:
        print()
        if wa_mode == "bot":
            print("  Who should be allowed to message the bot?")
            phone = input("  Phone numbers (comma-separated, or * for anyone): ").strip()
        else:
            phone = input("  Your phone number (e.g. 15551234567): ").strip()
        if phone:
            save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
            print(f"  ✓ Allowed users set: {phone}")
        else:
            print("  ⚠ No allowlist — the agent will respond to ALL incoming messages")

    # ── Step 4: Install bridge dependencies ──────────────────────────────
    project_root = Path(__file__).resolve().parents[1]
    bridge_dir = project_root / "scripts" / "whatsapp-bridge"
    bridge_script = bridge_dir / "bridge.js"

    if not bridge_script.exists():
        print(f"\n✗ Bridge script not found at {bridge_script}")
        return

    if not (bridge_dir / "node_modules").exists():
        print("\n→ Installing WhatsApp bridge dependencies...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=str(bridge_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  ✗ npm install failed: {result.stderr}")
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
            response = input("\n  Re-pair? This will clear the existing session. [y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in ("y", "yes"):
            import shutil
            shutil.rmtree(session_dir, ignore_errors=True)
            session_dir.mkdir(parents=True, exist_ok=True)
            print("  ✓ Session cleared")
        else:
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
            ["node", str(bridge_script), "--pair-only", "--session", str(session_dir)],
            cwd=str(bridge_dir),
        )
    except KeyboardInterrupt:
        pass

    # ── Step 7: Post-pairing ─────────────────────────────────────────────
    print()
    if (session_dir / "creds.json").exists():
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


def cmd_setup(args):
    """Interactive setup wizard."""
    from hermes_cli.setup import run_setup_wizard
    run_setup_wizard(args)


def cmd_model(args):
    """Select default model — starts with provider selection, then model picker."""
    _require_tty("model")
    select_provider_and_model(args=args)


def select_provider_and_model(args=None):
    """Core provider selection + model picking logic.

    Shared by ``cmd_model`` (``hermes model``) and the setup wizard
    (``setup_model_provider`` in setup.py).  Handles the full flow:
    provider picker, credential prompting, model selection, and config
    persistence.
    """
    from hermes_cli.auth import (
        resolve_provider, AuthError, format_auth_error,
    )
    from hermes_cli.config import get_compatible_custom_providers, load_config, get_env_value

    config = load_config()
    current_model = config.get("model")
    if isinstance(current_model, dict):
        current_model = current_model.get("default", "")
    current_model = current_model or "(not set)"

    # Read effective provider the same way the CLI does at startup:
    # config.yaml model.provider > env var > auto-detect
    import os
    config_provider = None
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        config_provider = model_cfg.get("provider")

    effective_provider = (
        config_provider
        or os.getenv("HERMES_INFERENCE_PROVIDER")
        or "auto"
    )
    try:
        active = resolve_provider(effective_provider)
    except AuthError as exc:
        warning = format_auth_error(exc)
        print(f"Warning: {warning} Falling back to auto provider detection.")
        try:
            active = resolve_provider("auto")
        except AuthError:
            active = None  # no provider yet; default to first in list

    # Detect custom endpoint
    if active == "openrouter" and get_env_value("OPENAI_BASE_URL"):
        active = "custom"

    from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS

    provider_labels = dict(_PROVIDER_LABELS)  # derive from canonical list
    active_label = provider_labels.get(active, active) if active else "none"

    print()
    print(f"  Current model:    {current_model}")
    print(f"  Active provider:  {active_label}")
    print()

    # Step 1: Provider selection — flat list from CANONICAL_PROVIDERS
    all_providers = [(p.slug, p.tui_desc) for p in CANONICAL_PROVIDERS]

    def _named_custom_provider_map(cfg) -> dict[str, dict[str, str]]:
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
                "api_mode": entry.get("api_mode", ""),
                "provider_key": provider_key,
            }
        return custom_provider_map

    # Add user-defined custom providers from config.yaml
    _custom_provider_map = _named_custom_provider_map(config)  # key → {name, base_url, api_key}
    for key, provider_info in _custom_provider_map.items():
        name = provider_info["name"]
        base_url = provider_info["base_url"]
        short_url = base_url.replace("https://", "").replace("http://", "").rstrip("/")
        saved_model = provider_info.get("model", "")
        model_hint = f" — {saved_model}" if saved_model else ""
        all_providers.append((key, f"{name} ({short_url}){model_hint}"))

    # Build the menu
    ordered = []
    default_idx = 0
    for key, label in all_providers:
        if active and key == active:
            ordered.append((key, f"{label}  ← currently active"))
            default_idx = len(ordered) - 1
        else:
            ordered.append((key, label))

    ordered.append(("custom", "Custom endpoint (enter URL manually)"))
    _has_saved_custom_list = isinstance(config.get("custom_providers"), list) and bool(config.get("custom_providers"))
    if _has_saved_custom_list:
        ordered.append(("remove-custom", "Remove a saved custom provider"))
    ordered.append(("cancel", "Cancel"))

    provider_idx = _prompt_provider_choice(
        [label for _, label in ordered], default=default_idx,
    )
    if provider_idx is None or ordered[provider_idx][0] == "cancel":
        print("No change.")
        return

    selected_provider = ordered[provider_idx][0]

    # Step 2: Provider-specific setup + model selection
    if selected_provider == "openrouter":
        _model_flow_openrouter(config, current_model)
    elif selected_provider == "nous":
        _model_flow_nous(config, current_model, args=args)
    elif selected_provider == "openai-codex":
        _model_flow_openai_codex(config, current_model)
    elif selected_provider == "qwen-oauth":
        _model_flow_qwen_oauth(config, current_model)
    elif selected_provider == "copilot-acp":
        _model_flow_copilot_acp(config, current_model)
    elif selected_provider == "copilot":
        _model_flow_copilot(config, current_model)
    elif selected_provider == "custom":
        _model_flow_custom(config)
    elif selected_provider.startswith("custom:") or selected_provider in _custom_provider_map:
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
    elif selected_provider in ("gemini", "deepseek", "xai", "zai", "kimi-coding-cn", "minimax", "minimax-cn", "kilocode", "opencode-zen", "opencode-go", "ai-gateway", "alibaba", "huggingface", "xiaomi", "arcee"):
        _model_flow_api_key_provider(config, selected_provider, current_model)

    # ── Post-switch cleanup: clear stale OPENAI_BASE_URL ──────────────
    # When the user switches to a named provider (anything except "custom"),
    # a leftover OPENAI_BASE_URL in ~/.hermes/.env can poison auxiliary
    # clients that use provider:auto. Clear it proactively.  (#5161)
    if selected_provider not in ("custom", "cancel", "remove-custom") \
            and not selected_provider.startswith("custom:"):
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
        print(f"Cleared stale OPENAI_BASE_URL from .env (was: {stale_url[:40]}...)"
              if len(stale_url) > 40
              else f"Cleared stale OPENAI_BASE_URL from .env (was: {stale_url})")


def _prompt_provider_choice(choices, *, default=0):
    """Show provider selection menu with curses arrow-key navigation.

    Falls back to a numbered list when curses is unavailable (e.g. piped
    stdin, non-TTY environments).  Returns the selected index, or None
    if the user cancels.
    """
    try:
        from hermes_cli.setup import _curses_prompt_choice
        idx = _curses_prompt_choice("Select provider:", choices, default)
        if idx >= 0:
            print()
            return idx
    except Exception:
        pass

    # Fallback: numbered list
    print("Select provider:")
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


def _model_flow_openrouter(config, current_model=""):
    """OpenRouter provider: ensure API key, then pick model."""
    from hermes_cli.auth import _prompt_model_selection, _save_model_choice, deactivate_provider
    from hermes_cli.config import get_env_value, save_env_value

    api_key = get_env_value("OPENROUTER_API_KEY")
    if not api_key:
        print("No OpenRouter API key configured.")
        print("Get one at: https://openrouter.ai/keys")
        print()
        try:
            import getpass
            key = getpass.getpass("OpenRouter API key (or Enter to cancel): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not key:
            print("Cancelled.")
            return
        save_env_value("OPENROUTER_API_KEY", key)
        print("API key saved.")
        print()

    from hermes_cli.models import model_ids, get_pricing_for_provider
    openrouter_models = model_ids(force_refresh=True)

    # Fetch live pricing (non-blocking — returns empty dict on failure)
    pricing = get_pricing_for_provider("openrouter", force_refresh=True)

    selected = _prompt_model_selection(openrouter_models, current_model=current_model, pricing=pricing)
    if selected:
        _save_model_choice(selected)

        # Update config provider and deactivate any OAuth provider
        from hermes_cli.config import load_config, save_config
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = "openrouter"
        model["base_url"] = OPENROUTER_BASE_URL
        model["api_mode"] = "chat_completions"
        save_config(cfg)
        deactivate_provider()
        print(f"Default model set to: {selected} (via OpenRouter)")
    else:
        print("No change.")


def _model_flow_nous(config, current_model="", args=None):
    """Nous Portal provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_provider_auth_state, _prompt_model_selection, _save_model_choice,
        _update_config_for_provider, resolve_nous_runtime_credentials,
        AuthError, format_auth_error,
        _login_nous, PROVIDER_REGISTRY,
    )
    from hermes_cli.config import get_env_value, save_config, save_env_value
    from hermes_cli.nous_subscription import (
        apply_nous_provider_defaults,
        get_nous_subscription_explainer_lines,
    )
    import argparse

    state = get_provider_auth_state("nous")
    if not state or not state.get("access_token"):
        print("Not logged into Nous Portal. Starting login...")
        print()
        try:
            mock_args = argparse.Namespace(
                portal_url=getattr(args, "portal_url", None),
                inference_url=getattr(args, "inference_url", None),
                client_id=getattr(args, "client_id", None),
                scope=getattr(args, "scope", None),
                no_browser=bool(getattr(args, "no_browser", False)),
                timeout=getattr(args, "timeout", None) or 15.0,
                ca_bundle=getattr(args, "ca_bundle", None),
                insecure=bool(getattr(args, "insecure", False)),
            )
            _login_nous(mock_args, PROVIDER_REGISTRY["nous"])
            print()
            for line in get_nous_subscription_explainer_lines():
                print(line)
        except SystemExit:
            print("Login cancelled or failed.")
            return
        except Exception as exc:
            print(f"Login failed: {exc}")
            return
        # login_nous already handles model selection + config update
        return

    # Already logged in — use curated model list (same as OpenRouter defaults).
    # The live /models endpoint returns hundreds of models; the curated list
    # shows only agentic models users recognize from OpenRouter.
    from hermes_cli.models import (
        _PROVIDER_MODELS, get_pricing_for_provider, filter_nous_free_models,
        check_nous_free_tier, partition_nous_models_by_tier,
    )
    model_ids = _PROVIDER_MODELS.get("nous", [])
    if not model_ids:
        print("No curated models available for Nous Portal.")
        return

    # Verify credentials are still valid (catches expired sessions early)
    try:
        creds = resolve_nous_runtime_credentials(min_key_ttl_seconds=5 * 60)
    except Exception as exc:
        relogin = isinstance(exc, AuthError) and exc.relogin_required
        msg = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
        if relogin:
            print(f"Session expired: {msg}")
            print("Re-authenticating with Nous Portal...\n")
            try:
                mock_args = argparse.Namespace(
                    portal_url=None, inference_url=None, client_id=None,
                    scope=None, no_browser=False, timeout=15.0,
                    ca_bundle=None, insecure=False,
                )
                _login_nous(mock_args, PROVIDER_REGISTRY["nous"])
            except Exception as login_exc:
                print(f"Re-login failed: {login_exc}")
            return
        print(f"Could not verify credentials: {msg}")
        return

    # Fetch live pricing (non-blocking — returns empty dict on failure)
    pricing = get_pricing_for_provider("nous")

    # Check if user is on free tier
    free_tier = check_nous_free_tier()

    # For both tiers: apply the allowlist filter first (removes non-allowlisted
    # free models and allowlist models that aren't actually free).
    # Then for free users: partition remaining models into selectable/unavailable.
    model_ids = filter_nous_free_models(model_ids, pricing)
    unavailable_models: list[str] = []
    if free_tier:
        model_ids, unavailable_models = partition_nous_models_by_tier(model_ids, pricing, free_tier=True)

    if not model_ids and not unavailable_models:
        print("No models available for Nous Portal after filtering.")
        return

    # Resolve portal URL for upgrade links (may differ on staging)
    _nous_portal_url = ""
    try:
        _nous_state = get_provider_auth_state("nous")
        if _nous_state:
            _nous_portal_url = _nous_state.get("portal_base_url", "")
    except Exception:
        pass

    if free_tier and not model_ids:
        print("No free models currently available.")
        if unavailable_models:
            from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL
            _url = (_nous_portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
            print(f"Upgrade at {_url} to access paid models.")
        return

    print(f"Showing {len(model_ids)} curated models — use \"Enter custom model name\" for others.")

    selected = _prompt_model_selection(
        model_ids, current_model=current_model, pricing=pricing,
        unavailable_models=unavailable_models, portal_url=_nous_portal_url,
    )
    if selected:
        _save_model_choice(selected)
        # Reactivate Nous as the provider and update config
        inference_url = creds.get("base_url", "")
        _update_config_for_provider("nous", inference_url)
        current_model_cfg = config.get("model")
        if isinstance(current_model_cfg, dict):
            model_cfg = dict(current_model_cfg)
        elif isinstance(current_model_cfg, str) and current_model_cfg.strip():
            model_cfg = {"default": current_model_cfg.strip()}
        else:
            model_cfg = {}
        model_cfg["provider"] = "nous"
        model_cfg["default"] = selected
        if inference_url and inference_url.strip():
            model_cfg["base_url"] = inference_url.rstrip("/")
        else:
            model_cfg.pop("base_url", None)
        config["model"] = model_cfg
        # Clear any custom endpoint that might conflict
        if get_env_value("OPENAI_BASE_URL"):
            save_env_value("OPENAI_BASE_URL", "")
            save_env_value("OPENAI_API_KEY", "")
        changed_defaults = apply_nous_provider_defaults(config)
        save_config(config)
        print(f"Default model set to: {selected} (via Nous Portal)")
        if "tts" in changed_defaults:
            print("TTS provider set to: OpenAI TTS via your Nous subscription")
        else:
            current_tts = str(config.get("tts", {}).get("provider") or "edge")
            if current_tts.lower() not in {"", "edge"}:
                print(f"Keeping your existing TTS provider: {current_tts}")
        print()
        for line in get_nous_subscription_explainer_lines():
            print(line)
    else:
        print("No change.")


def _model_flow_openai_codex(config, current_model=""):
    """OpenAI Codex provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_codex_auth_status, _prompt_model_selection, _save_model_choice,
        _update_config_for_provider, _login_openai_codex,
        PROVIDER_REGISTRY, DEFAULT_CODEX_BASE_URL,
    )
    from hermes_cli.codex_models import get_codex_model_ids
    import argparse

    status = get_codex_auth_status()
    if not status.get("logged_in"):
        print("Not logged into OpenAI Codex. Starting login...")
        print()
        try:
            mock_args = argparse.Namespace()
            _login_openai_codex(mock_args, PROVIDER_REGISTRY["openai-codex"])
        except SystemExit:
            print("Login cancelled or failed.")
            return
        except Exception as exc:
            print(f"Login failed: {exc}")
            return

    _codex_token = None
    # Prefer credential pool (where `hermes auth` stores device_code tokens),
    # fall back to legacy provider state.
    try:
        _codex_status = get_codex_auth_status()
        if _codex_status.get("logged_in"):
            _codex_token = _codex_status.get("api_key")
    except Exception:
        pass
    if not _codex_token:
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials
            _codex_creds = resolve_codex_runtime_credentials()
            _codex_token = _codex_creds.get("api_key")
        except Exception:
            pass

    codex_models = get_codex_model_ids(access_token=_codex_token)

    selected = _prompt_model_selection(codex_models, current_model=current_model)
    if selected:
        _save_model_choice(selected)
        _update_config_for_provider("openai-codex", DEFAULT_CODEX_BASE_URL)
        print(f"Default model set to: {selected} (via OpenAI Codex)")
    else:
        print("No change.")



_DEFAULT_QWEN_PORTAL_MODELS = [
    "qwen3-coder-plus",
    "qwen3-coder",
]


def _model_flow_qwen_oauth(_config, current_model=""):
    """Qwen OAuth provider: reuse local Qwen CLI login, then pick model."""
    from hermes_cli.auth import (
        get_qwen_auth_status,
        resolve_qwen_runtime_credentials,
        _prompt_model_selection,
        _save_model_choice,
        _update_config_for_provider,
        DEFAULT_QWEN_BASE_URL,
    )
    from hermes_cli.models import fetch_api_models

    status = get_qwen_auth_status()
    if not status.get("logged_in"):
        print("Not logged into Qwen CLI OAuth.")
        print("Run: qwen auth qwen-oauth")
        auth_file = status.get("auth_file")
        if auth_file:
            print(f"Expected credentials file: {auth_file}")
        if status.get("error"):
            print(f"Error: {status.get('error')}")
        return

    # Try live model discovery, fall back to curated list.
    models = None
    try:
        creds = resolve_qwen_runtime_credentials(refresh_if_expiring=True)
        models = fetch_api_models(creds["api_key"], creds["base_url"])
    except Exception:
        pass
    if not models:
        models = list(_DEFAULT_QWEN_PORTAL_MODELS)

    default = current_model or (models[0] if models else "qwen3-coder-plus")
    selected = _prompt_model_selection(models, current_model=default)
    if selected:
        _save_model_choice(selected)
        _update_config_for_provider("qwen-oauth", DEFAULT_QWEN_BASE_URL)
        print(f"Default model set to: {selected} (via Qwen OAuth)")
    else:
        print("No change.")



def _model_flow_custom(config):
    """Custom endpoint: collect URL, API key, and model name.

    Automatically saves the endpoint to ``custom_providers`` in config.yaml
    so it appears in the provider menu on subsequent runs.
    """
    from hermes_cli.auth import _save_model_choice, deactivate_provider
    from hermes_cli.config import get_env_value, load_config, save_config

    current_url = get_env_value("OPENAI_BASE_URL") or ""
    current_key = get_env_value("OPENAI_API_KEY") or ""

    print("Custom OpenAI-compatible endpoint configuration:")
    if current_url:
        print(f"  Current URL: {current_url}")
    if current_key:
        print(f"  Current key: {current_key[:8]}...")
    print()

    try:
        base_url = input(f"API base URL [{current_url or 'e.g. https://api.example.com/v1'}]: ").strip()
        import getpass
        api_key = getpass.getpass(f"API key [{current_key[:8] + '...' if current_key else 'optional'}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    if not base_url and not current_url:
        print("No URL provided. Cancelled.")
        return

    # Validate URL format
    effective_url = base_url or current_url
    if not effective_url.startswith(("http://", "https://")):
        print(f"Invalid URL: {effective_url} (must start with http:// or https://)")
        return

    effective_key = api_key or current_key

    from hermes_cli.models import probe_api_models

    probe = probe_api_models(effective_key, effective_url)
    if probe.get("used_fallback") and probe.get("resolved_base_url"):
        print(
            f"Warning: endpoint verification worked at {probe['resolved_base_url']}/models, "
            f"not the exact URL you entered. Saving the working base URL instead."
        )
        effective_url = probe["resolved_base_url"]
        if base_url:
            base_url = effective_url
    elif probe.get("models") is not None:
        print(
            f"Verified endpoint via {probe.get('probed_url')} "
            f"({len(probe.get('models') or [])} model(s) visible)"
        )
    else:
        print(
            f"Warning: could not verify this endpoint via {probe.get('probed_url')}. "
            f"Hermes will still save it."
        )
        if probe.get("suggested_base_url"):
            suggested = probe["suggested_base_url"]
            if suggested.endswith("/v1"):
                print(f"  If this server expects /v1 in the path, try base URL: {suggested}")
            else:
                print(f"  If /v1 should not be in the base URL, try: {suggested}")

    # Select model — use probe results when available, fall back to manual input
    model_name = ""
    detected_models = probe.get("models") or []
    try:
        if len(detected_models) == 1:
            print(f"  Detected model: {detected_models[0]}")
            confirm = input("  Use this model? [Y/n]: ").strip().lower()
            if confirm in ("", "y", "yes"):
                model_name = detected_models[0]
            else:
                model_name = input("Model name (e.g. gpt-4, llama-3-70b): ").strip()
        elif len(detected_models) > 1:
            print("  Available models:")
            for i, m in enumerate(detected_models, 1):
                print(f"    {i}. {m}")
            pick = input(f"  Select model [1-{len(detected_models)}] or type name: ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(detected_models):
                model_name = detected_models[int(pick) - 1]
            elif pick:
                model_name = pick
        else:
            model_name = input("Model name (e.g. gpt-4, llama-3-70b): ").strip()

        context_length_str = input("Context length in tokens [leave blank for auto-detect]: ").strip()

        # Prompt for a display name — shown in the provider menu on future runs
        default_name = _auto_provider_name(effective_url)
        display_name = input(f"Display name [{default_name}]: ").strip() or default_name
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    context_length = None
    if context_length_str:
        try:
            context_length = int(context_length_str.replace(",", "").replace("k", "000").replace("K", "000"))
            if context_length <= 0:
                context_length = None
        except ValueError:
            print(f"Invalid context length: {context_length_str} — will auto-detect.")
            context_length = None

    if model_name:
        _save_model_choice(model_name)

        # Update config and deactivate any OAuth provider
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = "custom"
        model["base_url"] = effective_url
        if effective_key:
            model["api_key"] = effective_key
        model.pop("api_mode", None)  # let runtime auto-detect from URL
        save_config(cfg)
        deactivate_provider()

        # Sync the caller's config dict so the setup wizard's final
        # save_config(config) preserves our model settings.  Without
        # this, the wizard overwrites model.provider/base_url with
        # the stale values from its own config dict (#4172).
        config["model"] = dict(model)

        print(f"Default model set to: {model_name} (via {effective_url})")
    else:
        if base_url or api_key:
            deactivate_provider()
        # Even without a model name, persist the custom endpoint on the
        # caller's config dict so the setup wizard doesn't lose it.
        _caller_model = config.get("model")
        if not isinstance(_caller_model, dict):
            _caller_model = {"default": _caller_model} if _caller_model else {}
        _caller_model["provider"] = "custom"
        _caller_model["base_url"] = effective_url
        if effective_key:
            _caller_model["api_key"] = effective_key
        _caller_model.pop("api_mode", None)
        config["model"] = _caller_model
        print("Endpoint saved. Use `/model` in chat or `hermes model` to set a model.")

    # Auto-save to custom_providers so it appears in the menu next time
    _save_custom_provider(effective_url, effective_key, model_name or "",
                          context_length=context_length, name=display_name)


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


def _save_custom_provider(base_url, api_key="", model="", context_length=None,
                          name=None):
    """Save a custom endpoint to custom_providers in config.yaml.

    Deduplicates by base_url — if the URL already exists, updates the
    model name and context_length but doesn't add a duplicate entry.
    Uses *name* when provided, otherwise auto-generates from the URL.
    """
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    providers = cfg.get("custom_providers") or []
    if not isinstance(providers, list):
        providers = []

    # Check if this URL is already saved — update model/context_length if so
    for entry in providers:
        if isinstance(entry, dict) and entry.get("base_url", "").rstrip("/") == base_url.rstrip("/"):
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
    if model and context_length:
        entry["models"] = {model: {"context_length": context_length}}

    providers.append(entry)
    cfg["custom_providers"] = providers
    save_config(cfg)
    print(f"  💾 Saved to custom providers as \"{name}\" (edit in config.yaml)")


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
        from simple_term_menu import TerminalMenu
        menu = TerminalMenu(
            [f"  {c}" for c in choices], cursor_index=0,
            menu_cursor="-> ", menu_cursor_style=("fg_red", "bold"),
            menu_highlight_style=("fg_red",),
            cycle_cursor=True, clear_screen=False,
            title="Select provider to remove:",
        )
        idx = menu.show()
        from hermes_cli.curses_ui import flush_stdin
        flush_stdin()
        print()
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
    removed_name = removed.get("name", "unnamed") if isinstance(removed, dict) else str(removed)
    print(f"✅ Removed \"{removed_name}\" from custom providers.")


def _model_flow_named_custom(config, provider_info):
    """Handle a named custom provider from config.yaml custom_providers list.

    Always probes the endpoint's /models API to let the user pick a model.
    If a model was previously saved, it is pre-selected in the menu.
    Falls back to the saved model if probing fails.
    """
    from hermes_cli.auth import _save_model_choice, deactivate_provider
    from hermes_cli.config import load_config, save_config
    from hermes_cli.models import fetch_api_models

    name = provider_info["name"]
    base_url = provider_info["base_url"]
    api_key = provider_info.get("api_key", "")
    key_env = provider_info.get("key_env", "")
    saved_model = provider_info.get("model", "")
    provider_key = (provider_info.get("provider_key") or "").strip()

    print(f"  Provider: {name}")
    print(f"  URL:      {base_url}")
    if saved_model:
        print(f"  Current:  {saved_model}")
    print()

    print("Fetching available models...")
    models = fetch_api_models(api_key, base_url, timeout=8.0)

    if models:
        default_idx = 0
        if saved_model and saved_model in models:
            default_idx = models.index(saved_model)

        print(f"Found {len(models)} model(s):\n")
        try:
            from simple_term_menu import TerminalMenu
            menu_items = [
                f"  {m} (current)" if m == saved_model else f"  {m}"
                for m in models
            ] + ["  Cancel"]
            menu = TerminalMenu(
                menu_items, cursor_index=default_idx,
                menu_cursor="-> ", menu_cursor_style=("fg_green", "bold"),
                menu_highlight_style=("fg_green",),
                cycle_cursor=True, clear_screen=False,
                title=f"Select model from {name}:",
            )
            idx = menu.show()
            from hermes_cli.curses_ui import flush_stdin
            flush_stdin()
            print()
            if idx is None or idx >= len(models):
                print("Cancelled.")
                return
            model_name = models[idx]
        except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
            for i, m in enumerate(models, 1):
                suffix = " (current)" if m == saved_model else ""
                print(f"  {i}. {m}{suffix}")
            print(f"  {len(models) + 1}. Cancel")
            print()
            try:
                val = input(f"Choice [1-{len(models) + 1}]: ").strip()
                if not val:
                    print("Cancelled.")
                    return
                idx = int(val) - 1
                if idx < 0 or idx >= len(models):
                    print("Cancelled.")
                    return
                model_name = models[idx]
            except (ValueError, KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                return
    elif saved_model:
        print("Could not fetch models from endpoint.")
        try:
            model_name = input(f"Model name [{saved_model}]: ").strip() or saved_model
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return
    else:
        print("Could not fetch models from endpoint. Enter model name manually.")
        try:
            model_name = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return
        if not model_name:
            print("No model specified. Cancelled.")
            return

    # Activate and save the model to the custom_providers entry
    _save_model_choice(model_name)

    cfg = load_config()
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        cfg["model"] = model
    if provider_key:
        model["provider"] = provider_key
        model.pop("base_url", None)
        model.pop("api_key", None)
    else:
        model["provider"] = "custom"
        model["base_url"] = base_url
        if api_key:
            model["api_key"] = api_key
    # Apply api_mode from custom_providers entry, or clear stale value
    custom_api_mode = provider_info.get("api_mode", "")
    if custom_api_mode:
        model["api_mode"] = custom_api_mode
    else:
        model.pop("api_mode", None)  # let runtime auto-detect from URL
    save_config(cfg)
    deactivate_provider()

    # Persist the selected model back to whichever schema owns this endpoint.
    if provider_key:
        cfg = load_config()
        providers_cfg = cfg.get("providers")
        if isinstance(providers_cfg, dict):
            provider_entry = providers_cfg.get(provider_key)
            if isinstance(provider_entry, dict):
                provider_entry["default_model"] = model_name
                if api_key and not str(provider_entry.get("api_key", "") or "").strip():
                    provider_entry["api_key"] = api_key
                if key_env and not str(provider_entry.get("key_env", "") or "").strip():
                    provider_entry["key_env"] = key_env
                cfg["providers"] = providers_cfg
                save_config(cfg)
    else:
        # Save model name to the custom_providers entry for next time
        _save_custom_provider(base_url, api_key, model_name)

    print(f"\n✅ Model set to: {model_name}")
    print(f"   Provider: {name} ({base_url})")


# Curated model lists for direct API-key providers — single source in models.py
from hermes_cli.models import _PROVIDER_MODELS


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
    deduped = list(dict.fromkeys(str(effort).strip().lower() for effort in efforts if str(effort).strip()))
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
        from simple_term_menu import TerminalMenu

        choices = [f"  {_label(effort)}" for effort in ordered]
        choices.append(f"  {disable_label}")
        choices.append(f"  {skip_label}")
        menu = TerminalMenu(
            choices,
            cursor_index=default_idx,
            menu_cursor="-> ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("fg_green",),
            cycle_cursor=True,
            clear_screen=False,
            title="Select reasoning effort:",
        )
        idx = menu.show()
        from hermes_cli.curses_ui import flush_stdin
        flush_stdin()
        if idx is None:
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


def _model_flow_copilot(config, current_model=""):
    """GitHub Copilot flow using env vars, gh CLI, or OAuth device code."""
    from hermes_cli.auth import (
        PROVIDER_REGISTRY,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
        resolve_api_key_provider_credentials,
    )
    from hermes_cli.config import save_env_value, load_config, save_config
    from hermes_cli.models import (
        fetch_api_models,
        fetch_github_model_catalog,
        github_model_reasoning_efforts,
        copilot_model_api_mode,
        normalize_copilot_model_id,
    )

    provider_id = "copilot"
    pconfig = PROVIDER_REGISTRY[provider_id]

    creds = resolve_api_key_provider_credentials(provider_id)
    api_key = creds.get("api_key", "")
    source = creds.get("source", "")

    if not api_key:
        print("No GitHub token configured for GitHub Copilot.")
        print()
        print("  Supported token types:")
        print("    → OAuth token (gho_*)          via `copilot login` or device code flow")
        print("    → Fine-grained PAT (github_pat_*)  with Copilot Requests permission")
        print("    → GitHub App token (ghu_*)     via environment variable")
        print("    ✗ Classic PAT (ghp_*)          NOT supported by Copilot API")
        print()
        print("  Options:")
        print("    1. Login with GitHub (OAuth device code flow)")
        print("    2. Enter a token manually")
        print("    3. Cancel")
        print()
        try:
            choice = input("  Choice [1-3]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if choice == "1":
            try:
                from hermes_cli.copilot_auth import copilot_device_code_login
                token = copilot_device_code_login()
                if token:
                    save_env_value("COPILOT_GITHUB_TOKEN", token)
                    print("  Copilot token saved.")
                    print()
                else:
                    print("  Login cancelled or failed.")
                    return
            except Exception as exc:
                print(f"  Login failed: {exc}")
                return
        elif choice == "2":
            try:
                import getpass
                new_key = getpass.getpass("  Token (COPILOT_GITHUB_TOKEN): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not new_key:
                print("  Cancelled.")
                return
            # Validate token type
            try:
                from hermes_cli.copilot_auth import validate_copilot_token
                valid, msg = validate_copilot_token(new_key)
                if not valid:
                    print(f"  ✗ {msg}")
                    return
            except ImportError:
                pass
            save_env_value("COPILOT_GITHUB_TOKEN", new_key)
            print("  Token saved.")
            print()
        else:
            print("  Cancelled.")
            return

        creds = resolve_api_key_provider_credentials(provider_id)
        api_key = creds.get("api_key", "")
        source = creds.get("source", "")
    else:
        if source in ("GITHUB_TOKEN", "GH_TOKEN"):
            print(f"  GitHub token: {api_key[:8]}... ✓ ({source})")
        elif source == "gh auth token":
            print("  GitHub token: ✓ (from `gh auth token`)")
        else:
            print("  GitHub token: ✓")
        print()

    effective_base = pconfig.inference_base_url

    catalog = fetch_github_model_catalog(api_key)
    live_models = [item.get("id", "") for item in catalog if item.get("id")] if catalog else fetch_api_models(api_key, effective_base)
    normalized_current_model = normalize_copilot_model_id(
        current_model,
        catalog=catalog,
        api_key=api_key,
    ) or current_model
    if live_models:
        model_list = [model_id for model_id in live_models if model_id]
        print(f"  Found {len(model_list)} model(s) from GitHub Copilot")
    else:
        model_list = _PROVIDER_MODELS.get(provider_id, [])
        if model_list:
            print("  ⚠ Could not auto-detect models from GitHub Copilot — showing defaults.")
            print('    Use "Enter custom model name" if you do not see your model.')

    if model_list:
        selected = _prompt_model_selection(model_list, current_model=normalized_current_model)
    else:
        try:
            selected = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        selected = normalize_copilot_model_id(
            selected,
            catalog=catalog,
            api_key=api_key,
        ) or selected
        initial_cfg = load_config()
        current_effort = _current_reasoning_effort(initial_cfg)
        reasoning_efforts = github_model_reasoning_efforts(
            selected,
            catalog=catalog,
            api_key=api_key,
        )
        selected_effort = None
        if reasoning_efforts:
            print(f"  {selected} supports reasoning controls.")
            selected_effort = _prompt_reasoning_effort_selection(
                reasoning_efforts, current_effort=current_effort
            )

        _save_model_choice(selected)

        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = provider_id
        model["base_url"] = effective_base
        model["api_mode"] = copilot_model_api_mode(
            selected,
            catalog=catalog,
            api_key=api_key,
        )
        if selected_effort is not None:
            _set_reasoning_effort(cfg, selected_effort)
        save_config(cfg)
        deactivate_provider()

        print(f"Default model set to: {selected} (via {pconfig.name})")
        if reasoning_efforts:
            if selected_effort == "none":
                print("Reasoning disabled for this model.")
            elif selected_effort:
                print(f"Reasoning effort set to: {selected_effort}")
    else:
        print("No change.")


def _model_flow_copilot_acp(config, current_model=""):
    """GitHub Copilot ACP flow using the local Copilot CLI."""
    from hermes_cli.auth import (
        PROVIDER_REGISTRY,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
        get_external_process_provider_status,
        resolve_api_key_provider_credentials,
        resolve_external_process_provider_credentials,
    )
    from hermes_cli.models import (
        fetch_github_model_catalog,
        normalize_copilot_model_id,
    )
    from hermes_cli.config import load_config, save_config

    del config

    provider_id = "copilot-acp"
    pconfig = PROVIDER_REGISTRY[provider_id]

    status = get_external_process_provider_status(provider_id)
    resolved_command = status.get("resolved_command") or status.get("command") or "copilot"
    effective_base = status.get("base_url") or pconfig.inference_base_url

    print("  GitHub Copilot ACP delegates Hermes turns to `copilot --acp`.")
    print("  Hermes currently starts its own ACP subprocess for each request.")
    print("  Hermes uses your selected model as a hint for the Copilot ACP session.")
    print(f"  Command: {resolved_command}")
    print(f"  Backend marker: {effective_base}")
    print()

    try:
        creds = resolve_external_process_provider_credentials(provider_id)
    except Exception as exc:
        print(f"  ⚠ {exc}")
        print("  Set HERMES_COPILOT_ACP_COMMAND or COPILOT_CLI_PATH if Copilot CLI is installed elsewhere.")
        return

    effective_base = creds.get("base_url") or effective_base

    catalog_api_key = ""
    try:
        catalog_creds = resolve_api_key_provider_credentials("copilot")
        catalog_api_key = catalog_creds.get("api_key", "")
    except Exception:
        pass

    catalog = fetch_github_model_catalog(catalog_api_key)
    normalized_current_model = normalize_copilot_model_id(
        current_model,
        catalog=catalog,
        api_key=catalog_api_key,
    ) or current_model

    if catalog:
        model_list = [item.get("id", "") for item in catalog if item.get("id")]
        print(f"  Found {len(model_list)} model(s) from GitHub Copilot")
    else:
        model_list = _PROVIDER_MODELS.get("copilot", [])
        if model_list:
            print("  ⚠ Could not auto-detect models from GitHub Copilot — showing defaults.")
            print('    Use "Enter custom model name" if you do not see your model.')

    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=normalized_current_model,
        )
    else:
        try:
            selected = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if not selected:
        print("No change.")
        return

    selected = normalize_copilot_model_id(
        selected,
        catalog=catalog,
        api_key=catalog_api_key,
    ) or selected
    _save_model_choice(selected)

    cfg = load_config()
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        cfg["model"] = model
    model["provider"] = provider_id
    model["base_url"] = effective_base
    model["api_mode"] = "chat_completions"
    save_config(cfg)
    deactivate_provider()

    print(f"Default model set to: {selected} (via {pconfig.name})")


def _model_flow_kimi(config, current_model=""):
    """Kimi / Moonshot model selection with automatic endpoint routing.

    - sk-kimi-* keys   → api.kimi.com/coding/v1  (Kimi Coding Plan)
    - Other keys        → api.moonshot.ai/v1      (legacy Moonshot)

    No manual base URL prompt — endpoint is determined by key prefix.
    """
    from hermes_cli.auth import (
        PROVIDER_REGISTRY, KIMI_CODE_BASE_URL, _prompt_model_selection,
        _save_model_choice, deactivate_provider,
    )
    from hermes_cli.config import get_env_value, save_env_value, load_config, save_config

    provider_id = "kimi-coding"
    pconfig = PROVIDER_REGISTRY[provider_id]
    key_env = pconfig.api_key_env_vars[0] if pconfig.api_key_env_vars else ""
    base_url_env = pconfig.base_url_env_var or ""

    # Step 1: Check / prompt for API key
    existing_key = ""
    for ev in pconfig.api_key_env_vars:
        existing_key = get_env_value(ev) or os.getenv(ev, "")
        if existing_key:
            break

    if not existing_key:
        print(f"No {pconfig.name} API key configured.")
        if key_env:
            try:
                import getpass
                new_key = getpass.getpass(f"{key_env} (or Enter to cancel): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not new_key:
                print("Cancelled.")
                return
            save_env_value(key_env, new_key)
            existing_key = new_key
            print("API key saved.")
            print()
    else:
        print(f"  {pconfig.name} API key: {existing_key[:8]}... ✓")
        print()

    # Step 2: Auto-detect endpoint from key prefix
    is_coding_plan = existing_key.startswith("sk-kimi-")
    if is_coding_plan:
        effective_base = KIMI_CODE_BASE_URL
        print(f"  Detected Kimi Coding Plan key → {effective_base}")
    else:
        effective_base = pconfig.inference_base_url
        print(f"  Using Moonshot endpoint → {effective_base}")
    # Clear any manual base URL override so auto-detection works at runtime
    if base_url_env and get_env_value(base_url_env):
        save_env_value(base_url_env, "")
    print()

    # Step 3: Model selection — show appropriate models for the endpoint
    if is_coding_plan:
        # Coding Plan models (kimi-for-coding first)
        model_list = [
            "kimi-for-coding",
            "kimi-k2.5",
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
        ]
    else:
        # Legacy Moonshot models (excludes Coding Plan-only models)
        model_list = _PROVIDER_MODELS.get("moonshot", [])

    if model_list:
        selected = _prompt_model_selection(model_list, current_model=current_model)
    else:
        try:
            selected = input("Enter model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        _save_model_choice(selected)

        # Update config with provider and base URL
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = provider_id
        model["base_url"] = effective_base
        model.pop("api_mode", None)  # let runtime auto-detect from URL
        save_config(cfg)
        deactivate_provider()

        endpoint_label = "Kimi Coding" if is_coding_plan else "Moonshot"
        print(f"Default model set to: {selected} (via {endpoint_label})")
    else:
        print("No change.")


def _model_flow_api_key_provider(config, provider_id, current_model=""):
    """Generic flow for API-key providers (z.ai, MiniMax, OpenCode, etc.)."""
    from hermes_cli.auth import (
        PROVIDER_REGISTRY, _prompt_model_selection, _save_model_choice,
        deactivate_provider,
    )
    from hermes_cli.config import get_env_value, save_env_value, load_config, save_config
    from hermes_cli.models import fetch_api_models, opencode_model_api_mode, normalize_opencode_model_id

    pconfig = PROVIDER_REGISTRY[provider_id]
    key_env = pconfig.api_key_env_vars[0] if pconfig.api_key_env_vars else ""
    base_url_env = pconfig.base_url_env_var or ""

    # Check / prompt for API key
    existing_key = ""
    for ev in pconfig.api_key_env_vars:
        existing_key = get_env_value(ev) or os.getenv(ev, "")
        if existing_key:
            break

    if not existing_key:
        print(f"No {pconfig.name} API key configured.")
        if key_env:
            try:
                import getpass
                new_key = getpass.getpass(f"{key_env} (or Enter to cancel): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not new_key:
                print("Cancelled.")
                return
            save_env_value(key_env, new_key)
            print("API key saved.")
            print()
    else:
        print(f"  {pconfig.name} API key: {existing_key[:8]}... ✓")
        print()

    # Optional base URL override
    current_base = ""
    if base_url_env:
        current_base = get_env_value(base_url_env) or os.getenv(base_url_env, "")
    effective_base = current_base or pconfig.inference_base_url

    try:
        override = input(f"Base URL [{effective_base}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        override = ""
    if override and base_url_env:
        if not override.startswith(("http://", "https://")):
            print("  Invalid URL — must start with http:// or https://. Keeping current value.")
        else:
            save_env_value(base_url_env, override)
            effective_base = override

    # Model selection — resolution order:
    #   1. models.dev registry (cached, filtered for agentic/tool-capable models)
    #   2. Curated static fallback list (offline insurance)
    #   3. Live /models endpoint probe (small providers without models.dev data)
    curated = _PROVIDER_MODELS.get(provider_id, [])

    # Try models.dev first — returns tool-capable models, filtered for noise
    mdev_models: list = []
    try:
        from agent.models_dev import list_agentic_models
        mdev_models = list_agentic_models(provider_id)
    except Exception:
        pass

    if mdev_models:
        model_list = mdev_models
        print(f"  Found {len(model_list)} model(s) from models.dev registry")
    elif curated and len(curated) >= 8:
        # Curated list is substantial — use it directly, skip live probe
        model_list = curated
        print(f"  Showing {len(model_list)} curated models — use \"Enter custom model name\" for others.")
    else:
        api_key_for_probe = existing_key or (get_env_value(key_env) if key_env else "")
        live_models = fetch_api_models(api_key_for_probe, effective_base)
        if live_models and len(live_models) >= len(curated):
            model_list = live_models
            print(f"  Found {len(model_list)} model(s) from {pconfig.name} API")
        else:
            model_list = curated
            if model_list:
                print(f"  Showing {len(model_list)} curated models — use \"Enter custom model name\" for others.")
        # else: no defaults either, will fall through to raw input

    if provider_id in {"opencode-zen", "opencode-go"}:
        model_list = [normalize_opencode_model_id(provider_id, mid) for mid in model_list]
        current_model = normalize_opencode_model_id(provider_id, current_model)
        model_list = list(dict.fromkeys(mid for mid in model_list if mid))

    if model_list:
        selected = _prompt_model_selection(model_list, current_model=current_model)
    else:
        try:
            selected = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        if provider_id in {"opencode-zen", "opencode-go"}:
            selected = normalize_opencode_model_id(provider_id, selected)

        _save_model_choice(selected)

        # Update config with provider, base URL, and provider-specific API mode
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = provider_id
        model["base_url"] = effective_base
        if provider_id in {"opencode-zen", "opencode-go"}:
            model["api_mode"] = opencode_model_api_mode(provider_id, selected)
        else:
            model.pop("api_mode", None)
        save_config(cfg)
        deactivate_provider()

        print(f"Default model set to: {selected} (via {pconfig.name})")
    else:
        print("No change.")


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
            is_claude_code_token_valid(creds)
            or bool(creds.get("refreshToken"))
        ):
            use_anthropic_claude_code_credentials(save_fn=save_env_value)
            print("  ✓ Claude Code credentials linked.")
            from hermes_constants import display_hermes_home as _dhh_fn
            print(f"    Hermes will use Claude's credential store directly instead of copying a setup-token into {_dhh_fn()}/.env.")
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
        try:
            import getpass
            manual_token = getpass.getpass("  Paste setup-token (or Enter to cancel): ").strip()
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
        try:
            import getpass
            token = getpass.getpass("  Setup-token (or Enter to cancel): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return False
        if token:
            save_anthropic_oauth_token(token, save_fn=save_env_value)
            print("  ✓ Setup-token saved.")
            return True
        print("  Cancelled — install Claude Code and try again.")
        return False


def _model_flow_anthropic(config, current_model=""):
    """Flow for Anthropic provider — OAuth subscription, API key, or Claude Code creds."""
    from hermes_cli.auth import (
        _prompt_model_selection, _save_model_choice,
        deactivate_provider,
    )
    from hermes_cli.config import (
        save_env_value, load_config, save_config,
        save_anthropic_api_key,
    )
    from hermes_cli.models import _PROVIDER_MODELS

    # Check ALL credential sources
    from hermes_cli.auth import get_anthropic_key
    existing_key = get_anthropic_key()
    cc_available = False
    try:
        from agent.anthropic_adapter import read_claude_code_credentials, is_claude_code_token_valid
        cc_creds = read_claude_code_credentials()
        if cc_creds and is_claude_code_token_valid(cc_creds):
            cc_available = True
    except Exception:
        pass

    has_creds = bool(existing_key) or cc_available
    needs_auth = not has_creds

    if has_creds:
        # Show what we found
        if existing_key:
            print(f"  Anthropic credentials: {existing_key[:12]}... ✓")
        elif cc_available:
            print("  Claude Code credentials: ✓ (auto-detected)")
        print()
        print("    1. Use existing credentials")
        print("    2. Reauthenticate (new OAuth login)")
        print("    3. Cancel")
        print()
        try:
            choice = input("  Choice [1/2/3]: ").strip()
        except (KeyboardInterrupt, EOFError):
            choice = "1"

        if choice == "2":
            needs_auth = True
        elif choice == "3":
            return
        # choice == "1" or default: use existing, proceed to model selection

    if needs_auth:
        # Show auth method choice
        print()
        print("  Choose authentication method:")
        print()
        print("    1. Claude Pro/Max subscription (OAuth login)")
        print("    2. Anthropic API key (pay-per-token)")
        print("    3. Cancel")
        print()
        try:
            choice = input("  Choice [1/2/3]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if choice == "1":
            if not _run_anthropic_oauth_flow(save_env_value):
                return

        elif choice == "2":
            print()
            print("  Get an API key at: https://console.anthropic.com/settings/keys")
            print()
            try:
                import getpass
                api_key = getpass.getpass("  API key (sk-ant-...): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not api_key:
                print("  Cancelled.")
                return
            save_anthropic_api_key(api_key, save_fn=save_env_value)
            print("  ✓ API key saved.")

        else:
            print("  No change.")
            return
    print()

    # Model selection
    model_list = _PROVIDER_MODELS.get("anthropic", [])
    if model_list:
        selected = _prompt_model_selection(model_list, current_model=current_model)
    else:
        try:
            selected = input("Model name (e.g., claude-sonnet-4-20250514): ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        _save_model_choice(selected)

        # Update config with provider — clear base_url since
        # resolve_runtime_provider() always hardcodes Anthropic's URL.
        # Leaving a stale base_url in config can contaminate other
        # providers if the user switches without running 'hermes model'.
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = "anthropic"
        model.pop("base_url", None)
        save_config(cfg)
        deactivate_provider()

        print(f"Default model set to: {selected} (via Anthropic)")
    else:
        print("No change.")


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


def cmd_doctor(args):
    """Check configuration and dependencies."""
    from hermes_cli.doctor import run_doctor
    run_doctor(args)


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


def cmd_version(args):
    """Show version."""
    print(f"Hermes Agent v{__version__} ({__release_date__})")
    print(f"Project: {PROJECT_ROOT}")
    
    # Show Python version
    print(f"Python: {sys.version.split()[0]}")
    
    # Check for key dependencies
    try:
        import openai
        print(f"OpenAI SDK: {openai.__version__}")
    except ImportError:
        print("OpenAI SDK: Not installed")

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


def cmd_uninstall(args):
    """Uninstall Hermes Agent."""
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
            d for d in dirnames
            if d not in ("venv", ".venv", "node_modules", ".git", ".worktrees")
        ]
        if os.path.basename(dirpath) == "__pycache__":
            try:
                import shutil as _shutil
                _shutil.rmtree(dirpath)
                removed += 1
            except OSError:
                pass
            dirnames.clear()  # nothing left to recurse into
    return removed


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
    import time as _time
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


def _build_web_ui(web_dir: Path, *, fatal: bool = False) -> bool:
    """Build the web UI frontend if npm is available.

    Args:
        web_dir: Path to the ``web/`` source directory.
        fatal: If True, print error guidance and return False on failure
               instead of a soft warning (used by ``hermes web``).

    Returns True if the build succeeded or was skipped (no package.json).
    """
    if not (web_dir / "package.json").exists():
        return True
    import shutil
    npm = shutil.which("npm")
    if not npm:
        if fatal:
            print("Web UI frontend not built and npm is not available.")
            print("Install Node.js, then run:  cd web && npm install && npm run build")
        return not fatal
    print("→ Building web UI...")
    r1 = subprocess.run([npm, "install", "--silent"], cwd=web_dir, capture_output=True)
    if r1.returncode != 0:
        print(f"  {'✗' if fatal else '⚠'} Web UI npm install failed"
              + ("" if fatal else " (hermes web will not be available)"))
        if fatal:
            print("  Run manually:  cd web && npm install && npm run build")
        return False
    r2 = subprocess.run([npm, "run", "build"], cwd=web_dir, capture_output=True)
    if r2.returncode != 0:
        print(f"  {'✗' if fatal else '⚠'} Web UI build failed"
              + ("" if fatal else " (hermes web will not be available)"))
        if fatal:
            print("  Run manually:  cd web && npm install && npm run build")
        return False
    print("  ✓ Web UI built")
    return True


def _update_via_zip(args):
    """Update Hermes Agent by downloading a ZIP archive.
    
    Used on Windows when git file I/O is broken (antivirus, NTFS filter 
    drivers causing 'Invalid argument' errors on file creation).
    """
    import shutil
    import tempfile
    import zipfile
    from urllib.request import urlretrieve
    
    branch = "main"
    zip_url = f"https://github.com/NousResearch/hermes-agent/archive/refs/heads/{branch}.zip"
    
    print("→ Downloading latest version...")
    try:
        tmp_dir = tempfile.mkdtemp(prefix="hermes-update-")
        zip_path = os.path.join(tmp_dir, f"hermes-agent-{branch}.zip")
        urlretrieve(zip_url, zip_path)
        
        print("→ Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Validate paths to prevent zip-slip (path traversal)
            tmp_dir_real = os.path.realpath(tmp_dir)
            for member in zf.infolist():
                member_path = os.path.realpath(os.path.join(tmp_dir, member.filename))
                if not member_path.startswith(tmp_dir_real + os.sep) and member_path != tmp_dir_real:
                    raise ValueError(f"Zip-slip detected: {member.filename} escapes extraction directory")
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
        preserve = {'venv', 'node_modules', '.git', '.env'}
        update_count = 0
        for item in os.listdir(extracted):
            if item in preserve:
                continue
            src = os.path.join(extracted, item)
            dst = os.path.join(str(PROJECT_ROOT), item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            update_count += 1
        
        print(f"✓ Updated {update_count} items from ZIP")
        
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"✗ ZIP update failed: {e}")
        sys.exit(1)

    # Clear stale bytecode after ZIP extraction
    removed = _clear_bytecode_cache(PROJECT_ROOT)
    if removed:
        print(f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}")
    
    # Reinstall Python dependencies. Prefer .[all], but if one optional extra
    # breaks on this machine, keep base deps and reinstall the remaining extras
    # individually so update does not silently strip working capabilities.
    print("→ Updating Python dependencies...")
    import subprocess
    uv_bin = shutil.which("uv")
    if uv_bin:
        uv_env = {**os.environ, "VIRTUAL_ENV": str(PROJECT_ROOT / "venv")}
        _install_python_dependencies_with_optional_fallback([uv_bin, "pip"], env=uv_env)
    else:
        # Use sys.executable to explicitly call the venv's pip module,
        # avoiding PEP 668 'externally-managed-environment' errors on Debian/Ubuntu.
        # Some environments lose pip inside the venv; bootstrap it back with
        # ensurepip before trying the editable install.
        pip_cmd = [sys.executable, "-m", "pip"]
        try:
            subprocess.run(pip_cmd + ["--version"], cwd=PROJECT_ROOT, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                cwd=PROJECT_ROOT,
                check=True,
            )
        _install_python_dependencies_with_optional_fallback(pip_cmd)

    # Build web UI frontend (optional — requires npm)
    _build_web_ui(PROJECT_ROOT / "web")

    # Sync skills
    try:
        from tools.skills_sync import sync_skills
        print("→ Syncing bundled skills...")
        result = sync_skills(quiet=True)
        if result["copied"]:
            print(f"  + {len(result['copied'])} new: {', '.join(result['copied'])}")
        if result.get("updated"):
            print(f"  ↑ {len(result['updated'])} updated: {', '.join(result['updated'])}")
        if result.get("user_modified"):
            print(f"  ~ {len(result['user_modified'])} user-modified (kept)")
        if result.get("cleaned"):
            print(f"  − {len(result['cleaned'])} removed from manifest")
        if not result["copied"] and not result.get("updated"):
            print("  ✓ Skills are up to date")
    except Exception:
        pass
    
    print()
    print("✓ Update complete!")


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

    stash_name = datetime.now(timezone.utc).strftime("hermes-update-autostash-%Y%m%d-%H%M%S")
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



def _resolve_stash_selector(git_cmd: list[str], cwd: Path, stash_ref: str) -> Optional[str]:
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



def _print_stash_cleanup_guidance(stash_ref: str, stash_selector: Optional[str] = None) -> None:
    print("  Check `git status` first so you don't accidentally reapply the same change twice.")
    print("  Find the saved entry with: git stash list --format='%gd %H %s'")
    if stash_selector:
        print(f"  Remove it with: git stash drop {stash_selector}")
    else:
        print(f"  Look for commit {stash_ref}, then drop its selector with: git stash drop stash@{{N}}")



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
        print("  Restoring them may reapply local customizations onto the updated codebase.")
        print("  Review the result afterward if Hermes behaves unexpectedly.")
        print("Restore local changes now? [Y/n]")
        if input_fn is not None:
            response = input_fn("Restore local changes now? [Y/n]", "y")
        else:
            response = input().strip().lower()
        if response not in ("", "y", "yes"):
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
        print("⚠ Local changes were restored, but Hermes couldn't find the stash entry to drop.")
        print("  The stash was left in place. You can remove it manually after checking the result.")
        _print_stash_cleanup_guidance(stash_ref)
    else:
        drop = subprocess.run(
            git_cmd + ["stash", "drop", stash_selector],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if drop.returncode != 0:
            print("⚠ Local changes were restored, but Hermes couldn't drop the saved stash entry.")
            if drop.stdout.strip():
                print(drop.stdout.strip())
            if drop.stderr.strip():
                print(drop.stderr.strip())
            print("  The stash was left in place. You can remove it manually after checking the result.")
            _print_stash_cleanup_guidance(stash_ref, stash_selector)

    print("⚠ Local changes were restored on top of the updated codebase.")
    print("  Review `git diff` / `git status` if Hermes behaves unexpectedly.")
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
            response = input("Add official repo as 'upstream' remote? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            response = "n"

        if response in ("", "y", "yes"):
            print("→ Adding upstream remote...")
            if _add_upstream_remote(git_cmd, cwd):
                print("  ✓ Added upstream: https://github.com/NousResearch/hermes-agent.git")
                has_upstream = True
            else:
                print("  ✗ Failed to add upstream remote. Skipping upstream sync.")
                return
        else:
            print("  Skipped. Run 'git remote add upstream https://github.com/NousResearch/hermes-agent.git' to add later.")
            _mark_skip_upstream_prompt()
            return

    # Fetch upstream
    print()
    print("→ Fetching upstream...")
    try:
        subprocess.run(
            git_cmd + ["fetch", "upstream", "--quiet"],
            cwd=cwd,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("  ✗ Failed to fetch upstream. Skipping upstream sync.")
        return

    # Compare origin/main with upstream/main
    origin_ahead = _count_commits_between(git_cmd, cwd, "upstream/main", "origin/main")
    upstream_ahead = _count_commits_between(git_cmd, cwd, "origin/main", "upstream/main")

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
        print("  ✗ Failed to pull from upstream. You may need to resolve conflicts manually.")
        return

    print("  ✓ Updated from upstream")

    # Try to sync fork back to origin
    print("→ Syncing fork...")
    if _sync_fork_with_upstream(git_cmd, cwd):
        print("  ✓ Fork synced with upstream")
    else:
        print("  ℹ Got updates from upstream but couldn't push to fork (no write access?)")
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


def _load_installable_optional_extras() -> list[str]:
    """Return the optional extras referenced by the ``all`` group.

    Only extras that ``[all]`` actually pulls in are retried individually.
    Extras outside ``[all]`` (e.g. ``rl``, ``yc-bench``) are intentionally
    excluded — they have heavy or platform-specific deps that most users
    never installed.
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

    # Parse the [all] group to find which extras it references.
    # Entries look like "hermes-agent[matrix]" or "package-name[extra]".
    all_refs = optional_deps.get("all", [])
    referenced: list[str] = []
    for ref in all_refs:
        if "[" in ref and "]" in ref:
            name = ref.split("[", 1)[1].split("]", 1)[0]
            if name in optional_deps:
                referenced.append(name)

    return referenced



def _install_python_dependencies_with_optional_fallback(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Install base deps plus as many optional extras as the environment supports."""
    try:
        subprocess.run(
            install_cmd_prefix + ["install", "-e", ".[all]", "--quiet"],
            cwd=PROJECT_ROOT,
            check=True,
            env=env,
        )
        return
    except subprocess.CalledProcessError:
        print("  ⚠ Optional extras failed, reinstalling base dependencies and retrying extras individually...")

    subprocess.run(
        install_cmd_prefix + ["install", "-e", ".", "--quiet"],
        cwd=PROJECT_ROOT,
        check=True,
        env=env,
    )

    failed_extras: list[str] = []
    installed_extras: list[str] = []
    for extra in _load_installable_optional_extras():
        try:
            subprocess.run(
                install_cmd_prefix + ["install", "-e", f".[{extra}]", "--quiet"],
                cwd=PROJECT_ROOT,
                check=True,
                env=env,
            )
            installed_extras.append(extra)
        except subprocess.CalledProcessError:
            failed_extras.append(extra)

    if installed_extras:
        print(f"  ✓ Reinstalled optional extras individually: {', '.join(installed_extras)}")
    if failed_extras:
        print(f"  ⚠ Skipped optional extras that still failed: {', '.join(failed_extras)}")


def cmd_update(args):
    """Update Hermes Agent to the latest version."""
    import shutil
    from hermes_cli.config import is_managed, managed_error

    if is_managed():
        managed_error("update Hermes Agent")
        return

    gateway_mode = getattr(args, "gateway", False)
    # In gateway mode, use file-based IPC for prompts instead of stdin
    gw_input_fn = (lambda prompt, default="": _gateway_prompt(prompt, default)) if gateway_mode else None
    
    print("⚕ Updating Hermes Agent...")
    print()
    
    # Try git-based update first, fall back to ZIP download on Windows
    # when git file I/O is broken (antivirus, NTFS filter drivers, etc.)
    use_zip_update = False
    git_dir = PROJECT_ROOT / '.git'
    
    if not git_dir.exists():
        if sys.platform == "win32":
            use_zip_update = True
        else:
            print("✗ Not a git repository. Please reinstall:")
            print("  curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash")
            sys.exit(1)
    
    # On Windows, git can fail with "unable to write loose object file: Invalid argument"
    # due to filesystem atomicity issues. Set the recommended workaround.
    if sys.platform == "win32" and git_dir.exists():
        subprocess.run(
            ["git", "-c", "windows.appendAtomically=false", "config", "windows.appendAtomically", "false"],
            cwd=PROJECT_ROOT, check=False, capture_output=True
        )

    # Build git command once — reused for fork detection and the update itself.
    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]

    # Detect if we're updating from a fork (before any branch logic)
    origin_url = _get_origin_url(git_cmd, PROJECT_ROOT)
    is_fork = _is_fork(origin_url)

    if is_fork:
        print("⚠ Updating from fork:")
        print(f"  {origin_url}")
        print()

    if use_zip_update:
        # ZIP-based update for Windows when git is broken
        _update_via_zip(args)
        return

    # Fetch and pull
    try:

        print("→ Fetching updates...")
        fetch_result = subprocess.run(
            git_cmd + ["fetch", "origin"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if fetch_result.returncode != 0:
            stderr = fetch_result.stderr.strip()
            if "Could not resolve host" in stderr or "unable to access" in stderr:
                print("✗ Network error — cannot reach the remote repository.")
                print(f"  {stderr.splitlines()[0]}" if stderr else "")
            elif "Authentication failed" in stderr or "could not read Username" in stderr:
                print("✗ Authentication failed — check your git credentials or SSH key.")
            else:
                print(f"✗ Failed to fetch updates from origin.")
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

        # Always update against main
        branch = "main"

        # If user is on a non-main branch or detached HEAD, switch to main
        if current_branch != "main":
            label = "detached HEAD" if current_branch == "HEAD" else f"branch '{current_branch}'"
            print(f"  ⚠ Currently on {label} — switching to main for update...")
            # Stash before checkout so uncommitted work isn't lost
            auto_stash_ref = _stash_local_changes_if_needed(git_cmd, PROJECT_ROOT)
            subprocess.run(
                git_cmd + ["checkout", "main"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            auto_stash_ref = _stash_local_changes_if_needed(git_cmd, PROJECT_ROOT)

        prompt_for_restore = auto_stash_ref is not None and (
            gateway_mode or (sys.stdin.isatty() and sys.stdout.isatty())
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
            # Restore stash and switch back to original branch if we moved
            if auto_stash_ref is not None:
                _restore_stashed_changes(
                    git_cmd, PROJECT_ROOT, auto_stash_ref,
                    prompt_user=prompt_for_restore,
                    input_fn=gw_input_fn,
                )
            if current_branch not in ("main", "HEAD"):
                subprocess.run(
                    git_cmd + ["checkout", current_branch],
                    cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
                )
            print("✓ Already up to date!")
            return

        print(f"→ Found {commit_count} new commit(s)")

        print("→ Pulling updates...")
        update_succeeded = False
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
                print("  ⚠ Fast-forward not possible (history diverged), resetting to match remote...")
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
                    print("  Try manually: git fetch origin && git reset --hard origin/main")
                    sys.exit(1)
            update_succeeded = True
        finally:
            if auto_stash_ref is not None:
                # Don't attempt stash restore if the code update itself failed —
                # working tree is in an unknown state.
                if not update_succeeded:
                    print(f"  ℹ️  Local changes preserved in stash (ref: {auto_stash_ref})")
                    print(f"  Restore manually with: git stash apply")
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
            print(f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}")

        # Fork upstream sync logic (only for main branch on forks)
        if is_fork and branch == "main":
            _sync_with_upstream_if_needed(git_cmd, PROJECT_ROOT)
        
        # Reinstall Python dependencies. Prefer .[all], but if one optional extra
        # breaks on this machine, keep base deps and reinstall the remaining extras
        # individually so update does not silently strip working capabilities.
        print("→ Updating Python dependencies...")
        uv_bin = shutil.which("uv")
        if uv_bin:
            uv_env = {**os.environ, "VIRTUAL_ENV": str(PROJECT_ROOT / "venv")}
            _install_python_dependencies_with_optional_fallback([uv_bin, "pip"], env=uv_env)
        else:
            # Use sys.executable to explicitly call the venv's pip module,
            # avoiding PEP 668 'externally-managed-environment' errors on Debian/Ubuntu.
            # Some environments lose pip inside the venv; bootstrap it back with
            # ensurepip before trying the editable install.
            pip_cmd = [sys.executable, "-m", "pip"]
            try:
                subprocess.run(pip_cmd + ["--version"], cwd=PROJECT_ROOT, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                subprocess.run(
                    [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
            _install_python_dependencies_with_optional_fallback(pip_cmd)
        
        # Check for Node.js deps
        if (PROJECT_ROOT / "package.json").exists():
            import shutil
            if shutil.which("npm"):
                print("→ Updating Node.js dependencies...")
                subprocess.run(["npm", "install", "--silent"], cwd=PROJECT_ROOT, check=False)

        # Build web UI frontend (optional — requires npm)
        _build_web_ui(PROJECT_ROOT / "web")

        print()
        print("✓ Code updated!")
        
        # After git pull, source files on disk are newer than cached Python
        # modules in this process.  Reload hermes_constants so that any lazy
        # import executed below (skills sync, gateway restart) sees new
        # attributes like display_hermes_home() added since the last release.
        try:
            import importlib
            import hermes_constants as _hc
            importlib.reload(_hc)
        except Exception:
            pass  # non-fatal — worst case a lazy import fails gracefully
        
        # Sync bundled skills (copies new, updates changed, respects user deletions)
        try:
            from tools.skills_sync import sync_skills
            print()
            print("→ Syncing bundled skills...")
            result = sync_skills(quiet=True)
            if result["copied"]:
                print(f"  + {len(result['copied'])} new: {', '.join(result['copied'])}")
            if result.get("updated"):
                print(f"  ↑ {len(result['updated'])} updated: {', '.join(result['updated'])}")
            if result.get("user_modified"):
                print(f"  ~ {len(result['user_modified'])} user-modified (kept)")
            if result.get("cleaned"):
                print(f"  − {len(result['cleaned'])} removed from manifest")
            if not result["copied"] and not result.get("updated"):
                print("  ✓ Skills are up to date")
        except Exception as e:
            logger.debug("Skills sync during update failed: %s", e)

        # Sync bundled skills to all other profiles
        try:
            from hermes_cli.profiles import list_profiles, get_active_profile_name, seed_profile_skills
            active = get_active_profile_name()
            other_profiles = [p for p in list_profiles() if p.name != active]
            if other_profiles:
                print()
                print("→ Syncing bundled skills to other profiles...")
                for p in other_profiles:
                    try:
                        r = seed_profile_skills(p.path, quiet=True)
                        if r:
                            copied = len(r.get("copied", []))
                            updated = len(r.get("updated", []))
                            modified = len(r.get("user_modified", []))
                            parts = []
                            if copied: parts.append(f"+{copied} new")
                            if updated: parts.append(f"↑{updated} updated")
                            if modified: parts.append(f"~{modified} user-modified")
                            status = ", ".join(parts) if parts else "up to date"
                        else:
                            status = "sync failed"
                        print(f"  {p.name}: {status}")
                    except Exception as pe:
                        print(f"  {p.name}: error ({pe})")
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
            get_missing_env_vars, get_missing_config_fields, 
            check_config_version, migrate_config
        )
        
        missing_env = get_missing_env_vars(required_only=True)
        missing_config = get_missing_config_fields()
        current_ver, latest_ver = check_config_version()
        
        needs_migration = missing_env or missing_config or current_ver < latest_ver
        
        if needs_migration:
            print()
            if missing_env:
                print(f"  ⚠️  {len(missing_env)} new required setting(s) need configuration")
            if missing_config:
                print(f"  ℹ️  {len(missing_config)} new config option(s) available")
            
            print()
            if gateway_mode:
                response = _gateway_prompt(
                    "Would you like to configure new options now? [Y/n]", "n"
                ).strip().lower()
            elif not (sys.stdin.isatty() and sys.stdout.isatty()):
                print("  ℹ Non-interactive session — skipping config migration prompt.")
                print("    Run 'hermes config migrate' later to apply any new config/env options.")
                response = "n"
            else:
                try:
                    response = input("Would you like to configure them now? [Y/n]: ").strip().lower()
                except EOFError:
                    response = "n"
            
            if response in ('', 'y', 'yes'):
                print()
                # In gateway mode, run auto-migrations only (no input() prompts
                # for API keys which would hang the detached process).
                results = migrate_config(interactive=not gateway_mode, quiet=False)
                
                if results["env_added"] or results["config_added"]:
                    print()
                    print("✓ Configuration updated!")
                if gateway_mode and missing_env:
                    print("  ℹ API keys require manual entry: hermes config migrate")
            else:
                print()
                print("Skipped. Run 'hermes config migrate' later to configure.")
        else:
            print("  ✓ Configuration is up to date")
        
        print()
        print("✓ Update complete!")
        
        # Write exit code *before* the gateway restart attempt.
        # When running as ``hermes update --gateway`` (spawned by the gateway's
        # /update command), this process lives inside the gateway's systemd
        # cgroup.  ``systemctl restart hermes-gateway`` kills everything in the
        # cgroup (KillMode=mixed → SIGKILL to remaining processes), including
        # us and the wrapping bash shell.  The shell never reaches its
        # ``printf $status > .update_exit_code`` epilogue, so the exit-code
        # marker file is never created.  The new gateway's update watcher then
        # polls for 30 minutes and sends a spurious timeout message.
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
                is_macos, supports_systemd_services, _ensure_user_systemd_env,
                find_gateway_pids,
                _get_service_pids,
            )
            import signal as _signal

            restarted_services = []
            killed_pids = set()

            # --- Systemd services (Linux) ---
            # Discover all hermes-gateway* units (default + profiles)
            if supports_systemd_services():
                try:
                    _ensure_user_systemd_env()
                except Exception:
                    pass

                for scope, scope_cmd in [("user", ["systemctl", "--user"]), ("system", ["systemctl"])]:
                    try:
                        result = subprocess.run(
                            scope_cmd + ["list-units", "hermes-gateway*", "--plain", "--no-legend", "--no-pager"],
                            capture_output=True, text=True, timeout=10,
                        )
                        for line in result.stdout.strip().splitlines():
                            parts = line.split()
                            if not parts:
                                continue
                            unit = parts[0]  # e.g. hermes-gateway.service or hermes-gateway-coder.service
                            if not unit.endswith(".service"):
                                continue
                            svc_name = unit.removesuffix(".service")
                            # Check if active
                            check = subprocess.run(
                                scope_cmd + ["is-active", svc_name],
                                capture_output=True, text=True, timeout=5,
                            )
                            if check.stdout.strip() == "active":
                                restart = subprocess.run(
                                    scope_cmd + ["restart", svc_name],
                                    capture_output=True, text=True, timeout=15,
                                )
                                if restart.returncode == 0:
                                    restarted_services.append(svc_name)
                                else:
                                    print(f"  ⚠ Failed to restart {svc_name}: {restart.stderr.strip()}")
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        pass

            # --- Launchd services (macOS) ---
            if is_macos():
                try:
                    from hermes_cli.gateway import launchd_restart, get_launchd_label, get_launchd_plist_path
                    plist_path = get_launchd_plist_path()
                    if plist_path.exists():
                        check = subprocess.run(
                            ["launchctl", "list", get_launchd_label()],
                            capture_output=True, text=True, timeout=5,
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
            manual_pids = find_gateway_pids(exclude_pids=service_pids, all_profiles=True)
            for pid in manual_pids:
                try:
                    os.kill(pid, _signal.SIGTERM)
                    killed_pids.add(pid)
                except (ProcessLookupError, PermissionError):
                    pass

            if restarted_services or killed_pids:
                print()
                for svc in restarted_services:
                    print(f"  ✓ Restarted {svc}")
                if killed_pids:
                    print(f"  → Stopped {len(killed_pids)} manual gateway process(es)")
                    print("    Restart manually: hermes gateway run")
                    # Also restart for each profile if needed
                    if len(killed_pids) > 1:
                        print("    (or: hermes -p <profile> gateway run  for each profile)")

            if not restarted_services and not killed_pids:
                # No gateways were running — nothing to do
                pass

        except Exception as e:
            logger.debug("Gateway restart during update failed: %s", e)
        
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
        "chat", "model", "gateway", "setup", "whatsapp", "login", "logout", "auth",
        "status", "cron", "doctor", "config", "pairing", "skills", "tools",
        "mcp", "sessions", "insights", "version", "update", "uninstall",
        "profile", "dashboard",
        "honcho", "claw", "plugins", "acp",
        "webhook", "memory", "dump", "debug", "backup", "import", "completion", "logs",
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
            while i < len(argv) and not argv[i].startswith("-") and argv[i] not in _SUBCOMMANDS:
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
        list_profiles, create_profile, delete_profile, seed_profile_skills,
        set_active_profile, get_active_profile_name,
        check_alias_collision, create_wrapper_script, remove_wrapper_script,
        _is_wrapper_dir_in_path, _get_wrapper_dir,
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
                    print(f"Model:          {p.model}" + (f" ({p.provider})" if p.provider else ""))
                print(f"Gateway:        {'running' if p.gateway_running else 'stopped'}")
                print(f"Skills:         {p.skill_count} installed")
                if p.alias_path:
                    print(f"Alias:          {p.name} → hermes -p {p.name}")
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
        print(f"\n {'Profile':<16} {'Model':<28} {'Gateway':<12} {'Alias'}")
        print(f" {'─' * 15}    {'─' * 27}    {'─' * 11}    {'─' * 12}")

        for p in profiles:
            marker = " ◆" if (p.name == active or (active == "default" and p.is_default)) else "  "
            name = p.name
            model = (p.model or "—")[:26]
            gw = "running" if p.gateway_running else "stopped"
            alias = p.name if p.alias_path else "—"
            if p.is_default:
                alias = "—"
            print(f"{marker}{name:<15} {model:<28} {gw:<12} {alias}")
        print()

    elif action == "use":
        name = args.profile_name
        try:
            set_active_profile(name)
            if name == "default":
                print(f"Switched to: default (~/.hermes)")
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

        try:
            clone_from = getattr(args, "clone_from", None)

            profile_dir = create_profile(
                name=name,
                clone_from=clone_from,
                clone_all=clone_all,
                clone_config=clone,
                no_alias=no_alias,
            )
            print(f"\nProfile '{name}' created at {profile_dir}")

            if clone or clone_all:
                source_label = getattr(args, "clone_from", None) or get_active_profile_name()
                if clone_all:
                    print(f"Full copy from {source_label}.")
                else:
                    print(f"Cloned config, .env, SOUL.md from {source_label}.")

            # Auto-clone Honcho config for the new profile (only with --clone/--clone-all)
            if clone or clone_all:
                try:
                    from plugins.memory.honcho.cli import clone_honcho_for_profile
                    if clone_honcho_for_profile(name):
                        print(f"Honcho config cloned (peer: {name})")
                except Exception:
                    pass  # Honcho plugin not installed or not configured

            # Seed bundled skills (skip if --clone-all already copied them)
            if not clone_all:
                result = seed_profile_skills(profile_dir)
                if result:
                    copied = len(result.get("copied", []))
                    print(f"{copied} bundled skills synced.")
                else:
                    print("⚠ Skills could not be seeded. Run `{} update` to retry.".format(name))

            # Create wrapper alias
            if not no_alias:
                collision = check_alias_collision(name)
                if collision:
                    print(f"\n⚠ Cannot create alias '{name}' — {collision}")
                    print(f"  Choose a custom alias:  hermes profile alias {name} --name <custom>")
                    print(f"  Or access via flag:     hermes -p {name} chat")
                else:
                    wrapper_path = create_wrapper_script(name)
                    if wrapper_path:
                        print(f"Wrapper created: {wrapper_path}")
                        if not _is_wrapper_dir_in_path():
                            print(f"\n⚠ {_get_wrapper_dir()} is not in your PATH.")
                            print(f'  Add to your shell config (~/.bashrc or ~/.zshrc):')
                            print(f'    export PATH="$HOME/.local/bin:$PATH"')

            # Profile dir for display
            try:
                profile_dir_display = "~/" + str(profile_dir.relative_to(Path.home()))
            except ValueError:
                profile_dir_display = str(profile_dir)

            # Next steps
            print(f"\nNext steps:")
            print(f"  {name} setup              Configure API keys and model")
            print(f"  {name} chat               Start chatting")
            print(f"  {name} gateway start      Start the messaging gateway")
            if clone or clone_all:
                print(f"\n  Edit {profile_dir_display}/.env for different API keys")
                print(f"  Edit {profile_dir_display}/SOUL.md for different personality")
            else:
                print(f"\n  ⚠ This profile has no API keys yet. Run '{name} setup' first,")
                print(f"    or it will inherit keys from your shell environment.")
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

    elif action == "show":
        name = args.profile_name
        from hermes_cli.profiles import get_profile_dir, profile_exists, _read_config_model, _check_gateway_running, _count_skills
        if not profile_exists(name):
            print(f"Error: Profile '{name}' does not exist.")
            sys.exit(1)
        profile_dir = get_profile_dir(name)
        model, provider = _read_config_model(profile_dir)
        gw = _check_gateway_running(profile_dir)
        skills = _count_skills(profile_dir)
        wrapper = _get_wrapper_dir() / name

        print(f"\nProfile: {name}")
        print(f"Path:    {profile_dir}")
        if model:
            print(f"Model:   {model}" + (f" ({provider})" if provider else ""))
        print(f"Gateway: {'running' if gw else 'stopped'}")
        print(f"Skills:  {skills}")
        print(f".env:    {'exists' if (profile_dir / '.env').exists() else 'not configured'}")
        print(f"SOUL.md: {'exists' if (profile_dir / 'SOUL.md').exists() else 'not configured'}")
        if wrapper.exists():
            print(f"Alias:   {wrapper}")
        print()

    elif action == "alias":
        name = args.profile_name
        remove = getattr(args, "remove", False)
        custom_name = getattr(args, "alias_name", None)

        from hermes_cli.profiles import profile_exists
        if not profile_exists(name):
            print(f"Error: Profile '{name}' does not exist.")
            sys.exit(1)

        alias_name = custom_name or name

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
            wrapper_path = create_wrapper_script(alias_name)
            if wrapper_path:
                # If custom name, write the profile name into the wrapper
                if custom_name:
                    wrapper_path.write_text(f'#!/bin/sh\nexec hermes -p {name} "$@"\n')
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
            profile_dir = import_profile(args.archive, name=getattr(args, "import_name", None))
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


def cmd_dashboard(args):
    """Start the web UI server."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        print("Web UI dependencies not installed.")
        print("Install them with:  pip install hermes-agent[web]")
        sys.exit(1)

    if not _build_web_ui(PROJECT_ROOT / "web", fatal=True):
        sys.exit(1)

    from hermes_cli.web_server import start_server
    start_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        allow_public=getattr(args, "insecure", False),
    )


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


def main():
    """Main entry point for hermes CLI."""
    parser = argparse.ArgumentParser(
        prog="hermes",
        description="Hermes Agent - AI assistant with tool-calling capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    hermes                        Start interactive chat
    hermes chat -q "Hello"        Single query mode
    hermes -c                     Resume the most recent session
    hermes -c "my project"        Resume a session by name (latest in lineage)
    hermes --resume <session_id>  Resume a specific session by ID
    hermes setup                  Run setup wizard
    hermes logout                 Clear stored authentication
    hermes auth add <provider>    Add a pooled credential
    hermes auth list              List pooled credentials
    hermes auth remove <p> <t>    Remove pooled credential by index, id, or label
    hermes auth reset <provider>  Clear exhaustion status for a provider
    hermes model                  Select default model
    hermes config                 View configuration
    hermes config edit            Edit config in $EDITOR
    hermes config set model gpt-4 Set a config value
    hermes gateway                Run messaging gateway
    hermes -s hermes-agent-dev,github-auth
    hermes -w                     Start in isolated git worktree
    hermes gateway install        Install gateway background service
    hermes sessions list          List past sessions
    hermes sessions browse        Interactive session picker
    hermes sessions rename ID T   Rename/title a session
    hermes logs                   View agent.log (last 50 lines)
    hermes logs -f                Follow agent.log in real time
    hermes logs errors            View errors.log
    hermes logs --since 1h        Lines from the last hour
    hermes debug share             Upload debug report for support
    hermes update                 Update to latest version

For more help on a command:
    hermes <command> --help
"""
    )
    
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="Show version and exit"
    )
    parser.add_argument(
        "--resume", "-r",
        metavar="SESSION",
        default=None,
        help="Resume a previous session by ID or title"
    )
    parser.add_argument(
        "--continue", "-c",
        dest="continue_last",
        nargs="?",
        const=True,
        default=None,
        metavar="SESSION_NAME",
        help="Resume a session by name, or the most recent if no name given"
    )
    parser.add_argument(
        "--worktree", "-w",
        action="store_true",
        default=False,
        help="Run in an isolated git worktree (for parallel agents)"
    )
    parser.add_argument(
        "--skills", "-s",
        action="append",
        default=None,
        help="Preload one or more skills for the session (repeat flag or comma-separate)"
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        default=False,
        help="Bypass all dangerous command approval prompts (use at your own risk)"
    )
    parser.add_argument(
        "--pass-session-id",
        action="store_true",
        default=False,
        help="Include the session ID in the agent's system prompt"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # =========================================================================
    # chat command
    # =========================================================================
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive chat with the agent",
        description="Start an interactive chat session with Hermes Agent"
    )
    chat_parser.add_argument(
        "-q", "--query",
        help="Single query (non-interactive mode)"
    )
    chat_parser.add_argument(
        "--image",
        help="Optional local image path to attach to a single query"
    )
    chat_parser.add_argument(
        "-m", "--model",
        help="Model to use (e.g., anthropic/claude-sonnet-4)"
    )
    chat_parser.add_argument(
        "-t", "--toolsets",
        help="Comma-separated toolsets to enable"
    )
    chat_parser.add_argument(
        "-s", "--skills",
        action="append",
        default=argparse.SUPPRESS,
        help="Preload one or more skills for the session (repeat flag or comma-separate)"
    )
    chat_parser.add_argument(
        "--provider",
        choices=["auto", "openrouter", "nous", "openai-codex", "copilot-acp", "copilot", "anthropic", "gemini", "huggingface", "zai", "kimi-coding", "kimi-coding-cn", "minimax", "minimax-cn", "kilocode", "xiaomi", "arcee"],
        default=None,
        help="Inference provider (default: auto)"
    )
    chat_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    chat_parser.add_argument(
        "-Q", "--quiet",
        action="store_true",
        help="Quiet mode for programmatic use: suppress banner, spinner, and tool previews. Only output the final response and session info."
    )
    chat_parser.add_argument(
        "--resume", "-r",
        metavar="SESSION_ID",
        default=argparse.SUPPRESS,
        help="Resume a previous session by ID (shown on exit)"
    )
    chat_parser.add_argument(
        "--continue", "-c",
        dest="continue_last",
        nargs="?",
        const=True,
        default=argparse.SUPPRESS,
        metavar="SESSION_NAME",
        help="Resume a session by name, or the most recent if no name given"
    )
    chat_parser.add_argument(
        "--worktree", "-w",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Run in an isolated git worktree (for parallel agents on the same repo)"
    )
    chat_parser.add_argument(
        "--checkpoints",
        action="store_true",
        default=False,
        help="Enable filesystem checkpoints before destructive file operations (use /rollback to restore)"
    )
    chat_parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        metavar="N",
        help="Maximum tool-calling iterations per conversation turn (default: 90, or agent.max_turns in config)"
    )
    chat_parser.add_argument(
        "--yolo",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Bypass all dangerous command approval prompts (use at your own risk)"
    )
    chat_parser.add_argument(
        "--pass-session-id",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Include the session ID in the agent's system prompt"
    )
    chat_parser.add_argument(
        "--source",
        default=None,
        help="Session source tag for filtering (default: cli). Use 'tool' for third-party integrations that should not appear in user session lists."
    )
    chat_parser.set_defaults(func=cmd_chat)

    # =========================================================================
    # model command
    # =========================================================================
    model_parser = subparsers.add_parser(
        "model",
        help="Select default model and provider",
        description="Interactively select your inference provider and default model"
    )
    model_parser.add_argument(
        "--portal-url",
        help="Portal base URL for Nous login (default: production portal)"
    )
    model_parser.add_argument(
        "--inference-url",
        help="Inference API base URL for Nous login (default: production inference API)"
    )
    model_parser.add_argument(
        "--client-id",
        default=None,
        help="OAuth client id to use for Nous login (default: hermes-cli)"
    )
    model_parser.add_argument(
        "--scope",
        default=None,
        help="OAuth scope to request for Nous login"
    )
    model_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not attempt to open the browser automatically during Nous login"
    )
    model_parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP request timeout in seconds for Nous login (default: 15)"
    )
    model_parser.add_argument(
        "--ca-bundle",
        help="Path to CA bundle PEM file for Nous TLS verification"
    )
    model_parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for Nous login (testing only)"
    )
    model_parser.set_defaults(func=cmd_model)

    # =========================================================================
    # gateway command
    # =========================================================================
    gateway_parser = subparsers.add_parser(
        "gateway",
        help="Messaging gateway management",
        description="Manage the messaging gateway (Telegram, Discord, WhatsApp)"
    )
    gateway_subparsers = gateway_parser.add_subparsers(dest="gateway_command")
    
    # gateway run (default)
    gateway_run = gateway_subparsers.add_parser("run", help="Run gateway in foreground (recommended for WSL, Docker, Termux)")
    gateway_run.add_argument("-v", "--verbose", action="count", default=0,
                             help="Increase stderr log verbosity (-v=INFO, -vv=DEBUG)")
    gateway_run.add_argument("-q", "--quiet", action="store_true",
                             help="Suppress all stderr log output")
    gateway_run.add_argument("--replace", action="store_true",
                             help="Replace any existing gateway instance (useful for systemd)")
    
    # gateway start
    gateway_start = gateway_subparsers.add_parser("start", help="Start the installed systemd/launchd background service")
    gateway_start.add_argument("--system", action="store_true", help="Target the Linux system-level gateway service")
    
    # gateway stop
    gateway_stop = gateway_subparsers.add_parser("stop", help="Stop gateway service")
    gateway_stop.add_argument("--system", action="store_true", help="Target the Linux system-level gateway service")
    gateway_stop.add_argument("--all", action="store_true", help="Stop ALL gateway processes across all profiles")
    
    # gateway restart
    gateway_restart = gateway_subparsers.add_parser("restart", help="Restart gateway service")
    gateway_restart.add_argument("--system", action="store_true", help="Target the Linux system-level gateway service")
    
    # gateway status
    gateway_status = gateway_subparsers.add_parser("status", help="Show gateway status")
    gateway_status.add_argument("--deep", action="store_true", help="Deep status check")
    gateway_status.add_argument("--system", action="store_true", help="Target the Linux system-level gateway service")
    
    # gateway install
    gateway_install = gateway_subparsers.add_parser("install", help="Install gateway as a systemd/launchd background service")
    gateway_install.add_argument("--force", action="store_true", help="Force reinstall")
    gateway_install.add_argument("--system", action="store_true", help="Install as a Linux system-level service (starts at boot)")
    gateway_install.add_argument("--run-as-user", dest="run_as_user", help="User account the Linux system service should run as")
    
    # gateway uninstall
    gateway_uninstall = gateway_subparsers.add_parser("uninstall", help="Uninstall gateway service")
    gateway_uninstall.add_argument("--system", action="store_true", help="Target the Linux system-level gateway service")

    # gateway setup
    gateway_subparsers.add_parser("setup", help="Configure messaging platforms")

    gateway_parser.set_defaults(func=cmd_gateway)
    
    # =========================================================================
    # setup command
    # =========================================================================
    setup_parser = subparsers.add_parser(
        "setup",
        help="Interactive setup wizard",
        description="Configure Hermes Agent with an interactive wizard. "
                    "Run a specific section: hermes setup model|tts|terminal|gateway|tools|agent"
    )
    setup_parser.add_argument(
        "section",
        nargs="?",
        choices=["model", "tts", "terminal", "gateway", "tools", "agent"],
        default=None,
        help="Run a specific setup section instead of the full wizard"
    )
    setup_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Non-interactive mode (use defaults/env vars)"
    )
    setup_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset configuration to defaults"
    )
    setup_parser.set_defaults(func=cmd_setup)

    # =========================================================================
    # whatsapp command
    # =========================================================================
    whatsapp_parser = subparsers.add_parser(
        "whatsapp",
        help="Set up WhatsApp integration",
        description="Configure WhatsApp and pair via QR code"
    )
    whatsapp_parser.set_defaults(func=cmd_whatsapp)

    # =========================================================================
    # login command
    # =========================================================================
    login_parser = subparsers.add_parser(
        "login",
        help="Authenticate with an inference provider",
        description="Run OAuth device authorization flow for Hermes CLI"
    )
    login_parser.add_argument(
        "--provider",
        choices=["nous", "openai-codex"],
        default=None,
        help="Provider to authenticate with (default: nous)"
    )
    login_parser.add_argument(
        "--portal-url",
        help="Portal base URL (default: production portal)"
    )
    login_parser.add_argument(
        "--inference-url",
        help="Inference API base URL (default: production inference API)"
    )
    login_parser.add_argument(
        "--client-id",
        default=None,
        help="OAuth client id to use (default: hermes-cli)"
    )
    login_parser.add_argument(
        "--scope",
        default=None,
        help="OAuth scope to request"
    )
    login_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not attempt to open the browser automatically"
    )
    login_parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP request timeout in seconds (default: 15)"
    )
    login_parser.add_argument(
        "--ca-bundle",
        help="Path to CA bundle PEM file for TLS verification"
    )
    login_parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification (testing only)"
    )
    login_parser.set_defaults(func=cmd_login)

    # =========================================================================
    # logout command
    # =========================================================================
    logout_parser = subparsers.add_parser(
        "logout",
        help="Clear authentication for an inference provider",
        description="Remove stored credentials and reset provider config"
    )
    logout_parser.add_argument(
        "--provider",
        choices=["nous", "openai-codex"],
        default=None,
        help="Provider to log out from (default: active provider)"
    )
    logout_parser.set_defaults(func=cmd_logout)

    auth_parser = subparsers.add_parser(
        "auth",
        help="Manage pooled provider credentials",
    )
    auth_subparsers = auth_parser.add_subparsers(dest="auth_action")
    auth_add = auth_subparsers.add_parser("add", help="Add a pooled credential")
    auth_add.add_argument("provider", help="Provider id (for example: anthropic, openai-codex, openrouter)")
    auth_add.add_argument("--type", dest="auth_type", choices=["oauth", "api-key", "api_key"], help="Credential type to add")
    auth_add.add_argument("--label", help="Optional display label")
    auth_add.add_argument("--api-key", help="API key value (otherwise prompted securely)")
    auth_add.add_argument("--portal-url", help="Nous portal base URL")
    auth_add.add_argument("--inference-url", help="Nous inference base URL")
    auth_add.add_argument("--client-id", help="OAuth client id")
    auth_add.add_argument("--scope", help="OAuth scope override")
    auth_add.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser for OAuth login")
    auth_add.add_argument("--timeout", type=float, help="OAuth/network timeout in seconds")
    auth_add.add_argument("--insecure", action="store_true", help="Disable TLS verification for OAuth login")
    auth_add.add_argument("--ca-bundle", help="Custom CA bundle for OAuth login")
    auth_list = auth_subparsers.add_parser("list", help="List pooled credentials")
    auth_list.add_argument("provider", nargs="?", help="Optional provider filter")
    auth_remove = auth_subparsers.add_parser("remove", help="Remove a pooled credential by index, id, or label")
    auth_remove.add_argument("provider", help="Provider id")
    auth_remove.add_argument("target", help="Credential index, entry id, or exact label")
    auth_reset = auth_subparsers.add_parser("reset", help="Clear exhaustion status for all credentials for a provider")
    auth_reset.add_argument("provider", help="Provider id")
    auth_parser.set_defaults(func=cmd_auth)

    # =========================================================================
    # status command
    # =========================================================================
    status_parser = subparsers.add_parser(
        "status",
        help="Show status of all components",
        description="Display status of Hermes Agent components"
    )
    status_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all details (redacted for sharing)"
    )
    status_parser.add_argument(
        "--deep",
        action="store_true",
        help="Run deep checks (may take longer)"
    )
    status_parser.set_defaults(func=cmd_status)
    
    # =========================================================================
    # cron command
    # =========================================================================
    cron_parser = subparsers.add_parser(
        "cron",
        help="Cron job management",
        description="Manage scheduled tasks"
    )
    cron_subparsers = cron_parser.add_subparsers(dest="cron_command")
    
    # cron list
    cron_list = cron_subparsers.add_parser("list", help="List scheduled jobs")
    cron_list.add_argument("--all", action="store_true", help="Include disabled jobs")

    # cron create/add
    cron_create = cron_subparsers.add_parser("create", aliases=["add"], help="Create a scheduled job")
    cron_create.add_argument("schedule", help="Schedule like '30m', 'every 2h', or '0 9 * * *'")
    cron_create.add_argument("prompt", nargs="?", help="Optional self-contained prompt or task instruction")
    cron_create.add_argument("--name", help="Optional human-friendly job name")
    cron_create.add_argument("--deliver", help="Delivery target: origin, local, telegram, discord, signal, or platform:chat_id")
    cron_create.add_argument("--repeat", type=int, help="Optional repeat count")
    cron_create.add_argument("--skill", dest="skills", action="append", help="Attach a skill. Repeat to add multiple skills.")
    cron_create.add_argument("--script", help="Path to a Python script whose stdout is injected into the prompt each run")

    # cron edit
    cron_edit = cron_subparsers.add_parser("edit", help="Edit an existing scheduled job")
    cron_edit.add_argument("job_id", help="Job ID to edit")
    cron_edit.add_argument("--schedule", help="New schedule")
    cron_edit.add_argument("--prompt", help="New prompt/task instruction")
    cron_edit.add_argument("--name", help="New job name")
    cron_edit.add_argument("--deliver", help="New delivery target")
    cron_edit.add_argument("--repeat", type=int, help="New repeat count")
    cron_edit.add_argument("--skill", dest="skills", action="append", help="Replace the job's skills with this set. Repeat to attach multiple skills.")
    cron_edit.add_argument("--add-skill", dest="add_skills", action="append", help="Append a skill without replacing the existing list. Repeatable.")
    cron_edit.add_argument("--remove-skill", dest="remove_skills", action="append", help="Remove a specific attached skill. Repeatable.")
    cron_edit.add_argument("--clear-skills", action="store_true", help="Remove all attached skills from the job")
    cron_edit.add_argument("--script", help="Path to a Python script whose stdout is injected into the prompt each run. Pass empty string to clear.")

    # lifecycle actions
    cron_pause = cron_subparsers.add_parser("pause", help="Pause a scheduled job")
    cron_pause.add_argument("job_id", help="Job ID to pause")

    cron_resume = cron_subparsers.add_parser("resume", help="Resume a paused job")
    cron_resume.add_argument("job_id", help="Job ID to resume")

    cron_run = cron_subparsers.add_parser("run", help="Run a job on the next scheduler tick")
    cron_run.add_argument("job_id", help="Job ID to trigger")

    cron_remove = cron_subparsers.add_parser("remove", aliases=["rm", "delete"], help="Remove a scheduled job")
    cron_remove.add_argument("job_id", help="Job ID to remove")

    # cron status
    cron_subparsers.add_parser("status", help="Check if cron scheduler is running")

    # cron tick (mostly for debugging)
    cron_subparsers.add_parser("tick", help="Run due jobs once and exit")

    cron_parser.set_defaults(func=cmd_cron)

    # =========================================================================
    # webhook command
    # =========================================================================
    webhook_parser = subparsers.add_parser(
        "webhook",
        help="Manage dynamic webhook subscriptions",
        description="Create, list, and remove webhook subscriptions for event-driven agent activation",
    )
    webhook_subparsers = webhook_parser.add_subparsers(dest="webhook_action")

    wh_sub = webhook_subparsers.add_parser("subscribe", aliases=["add"], help="Create a webhook subscription")
    wh_sub.add_argument("name", help="Route name (used in URL: /webhooks/<name>)")
    wh_sub.add_argument("--prompt", default="", help="Prompt template with {dot.notation} payload refs")
    wh_sub.add_argument("--events", default="", help="Comma-separated event types to accept")
    wh_sub.add_argument("--description", default="", help="What this subscription does")
    wh_sub.add_argument("--skills", default="", help="Comma-separated skill names to load")
    wh_sub.add_argument("--deliver", default="log", help="Delivery target: log, telegram, discord, slack, etc.")
    wh_sub.add_argument("--deliver-chat-id", default="", help="Target chat ID for cross-platform delivery")
    wh_sub.add_argument("--secret", default="", help="HMAC secret (auto-generated if omitted)")

    webhook_subparsers.add_parser("list", aliases=["ls"], help="List all dynamic subscriptions")

    wh_rm = webhook_subparsers.add_parser("remove", aliases=["rm"], help="Remove a subscription")
    wh_rm.add_argument("name", help="Subscription name to remove")

    wh_test = webhook_subparsers.add_parser("test", help="Send a test POST to a webhook route")
    wh_test.add_argument("name", help="Subscription name to test")
    wh_test.add_argument("--payload", default="", help="JSON payload to send (default: test payload)")

    webhook_parser.set_defaults(func=cmd_webhook)

    # =========================================================================
    # doctor command
    # =========================================================================
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check configuration and dependencies",
        description="Diagnose issues with Hermes Agent setup"
    )
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix issues automatically"
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    # =========================================================================
    # dump command
    # =========================================================================
    dump_parser = subparsers.add_parser(
        "dump",
        help="Dump setup summary for support/debugging",
        description="Output a compact, plain-text summary of your Hermes setup "
                    "that can be copy-pasted into Discord/GitHub for support context"
    )
    dump_parser.add_argument(
        "--show-keys",
        action="store_true",
        help="Show redacted API key prefixes (first/last 4 chars) instead of just set/not set"
    )
    dump_parser.set_defaults(func=cmd_dump)

    # =========================================================================
    # debug command
    # =========================================================================
    debug_parser = subparsers.add_parser(
        "debug",
        help="Debug tools — upload logs and system info for support",
        description="Debug utilities for Hermes Agent. Use 'hermes debug share' to "
                    "upload a debug report (system info + recent logs) to a paste "
                    "service and get a shareable URL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    hermes debug share              Upload debug report and print URL
    hermes debug share --lines 500  Include more log lines
    hermes debug share --expire 30  Keep paste for 30 days
    hermes debug share --local      Print report locally (no upload)
""",
    )
    debug_sub = debug_parser.add_subparsers(dest="debug_command")
    share_parser = debug_sub.add_parser(
        "share",
        help="Upload debug report to a paste service and print a shareable URL",
    )
    share_parser.add_argument(
        "--lines", type=int, default=200,
        help="Number of log lines to include per log file (default: 200)",
    )
    share_parser.add_argument(
        "--expire", type=int, default=7,
        help="Paste expiry in days (default: 7)",
    )
    share_parser.add_argument(
        "--local", action="store_true",
        help="Print the report locally instead of uploading",
    )
    debug_parser.set_defaults(func=cmd_debug)

    # =========================================================================
    # backup command
    # =========================================================================
    backup_parser = subparsers.add_parser(
        "backup",
        help="Back up Hermes home directory to a zip file",
        description="Create a zip archive of your entire Hermes configuration, "
                    "skills, sessions, and data (excludes the hermes-agent codebase). "
                    "Use --quick for a fast snapshot of just critical state files."
    )
    backup_parser.add_argument(
        "-o", "--output",
        help="Output path for the zip file (default: ~/hermes-backup-<timestamp>.zip)"
    )
    backup_parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Quick snapshot: only critical state files (config, state.db, .env, auth, cron)"
    )
    backup_parser.add_argument(
        "-l", "--label",
        help="Label for the snapshot (only used with --quick)"
    )
    backup_parser.set_defaults(func=cmd_backup)

    # =========================================================================
    # import command
    # =========================================================================
    import_parser = subparsers.add_parser(
        "import",
        help="Restore a Hermes backup from a zip file",
        description="Extract a previously created Hermes backup into your "
                    "Hermes home directory, restoring configuration, skills, "
                    "sessions, and data"
    )
    import_parser.add_argument(
        "zipfile",
        help="Path to the backup zip file"
    )
    import_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing files without confirmation"
    )
    import_parser.set_defaults(func=cmd_import)

    # =========================================================================
    # config command
    # =========================================================================
    config_parser = subparsers.add_parser(
        "config",
        help="View and edit configuration",
        description="Manage Hermes Agent configuration"
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    
    # config show (default)
    config_subparsers.add_parser("show", help="Show current configuration")
    
    # config edit
    config_subparsers.add_parser("edit", help="Open config file in editor")
    
    # config set
    config_set = config_subparsers.add_parser("set", help="Set a configuration value")
    config_set.add_argument("key", nargs="?", help="Configuration key (e.g., model, terminal.backend)")
    config_set.add_argument("value", nargs="?", help="Value to set")
    
    # config path
    config_subparsers.add_parser("path", help="Print config file path")
    
    # config env-path
    config_subparsers.add_parser("env-path", help="Print .env file path")
    
    # config check
    config_subparsers.add_parser("check", help="Check for missing/outdated config")
    
    # config migrate
    config_subparsers.add_parser("migrate", help="Update config with new options")
    
    config_parser.set_defaults(func=cmd_config)
    
    # =========================================================================
    # pairing command
    # =========================================================================
    pairing_parser = subparsers.add_parser(
        "pairing",
        help="Manage DM pairing codes for user authorization",
        description="Approve or revoke user access via pairing codes"
    )
    pairing_sub = pairing_parser.add_subparsers(dest="pairing_action")

    pairing_sub.add_parser("list", help="Show pending + approved users")

    pairing_approve_parser = pairing_sub.add_parser("approve", help="Approve a pairing code")
    pairing_approve_parser.add_argument("platform", help="Platform name (telegram, discord, slack, whatsapp)")
    pairing_approve_parser.add_argument("code", help="Pairing code to approve")

    pairing_revoke_parser = pairing_sub.add_parser("revoke", help="Revoke user access")
    pairing_revoke_parser.add_argument("platform", help="Platform name")
    pairing_revoke_parser.add_argument("user_id", help="User ID to revoke")

    pairing_sub.add_parser("clear-pending", help="Clear all pending codes")

    def cmd_pairing(args):
        from hermes_cli.pairing import pairing_command
        pairing_command(args)

    pairing_parser.set_defaults(func=cmd_pairing)

    # =========================================================================
    # skills command
    # =========================================================================
    skills_parser = subparsers.add_parser(
        "skills",
        help="Search, install, configure, and manage skills",
        description="Search, install, inspect, audit, configure, and manage skills from skills.sh, well-known agent skill endpoints, GitHub, ClawHub, and other registries."
    )
    skills_subparsers = skills_parser.add_subparsers(dest="skills_action")

    skills_browse = skills_subparsers.add_parser("browse", help="Browse all available skills (paginated)")
    skills_browse.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    skills_browse.add_argument("--size", type=int, default=20, help="Results per page (default: 20)")
    skills_browse.add_argument("--source", default="all",
                               choices=["all", "official", "skills-sh", "well-known", "github", "clawhub", "lobehub"],
                               help="Filter by source (default: all)")

    skills_search = skills_subparsers.add_parser("search", help="Search skill registries")
    skills_search.add_argument("query", help="Search query")
    skills_search.add_argument("--source", default="all", choices=["all", "official", "skills-sh", "well-known", "github", "clawhub", "lobehub"])
    skills_search.add_argument("--limit", type=int, default=10, help="Max results")

    skills_install = skills_subparsers.add_parser("install", help="Install a skill")
    skills_install.add_argument("identifier", help="Skill identifier (e.g. openai/skills/skill-creator)")
    skills_install.add_argument("--category", default="", help="Category folder to install into")
    skills_install.add_argument("--force", action="store_true", help="Install despite blocked scan verdict")
    skills_install.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt (needed in TUI mode)")

    skills_inspect = skills_subparsers.add_parser("inspect", help="Preview a skill without installing")
    skills_inspect.add_argument("identifier", help="Skill identifier")

    skills_list = skills_subparsers.add_parser("list", help="List installed skills")
    skills_list.add_argument("--source", default="all", choices=["all", "hub", "builtin", "local"])

    skills_check = skills_subparsers.add_parser("check", help="Check installed hub skills for updates")
    skills_check.add_argument("name", nargs="?", help="Specific skill to check (default: all)")

    skills_update = skills_subparsers.add_parser("update", help="Update installed hub skills")
    skills_update.add_argument("name", nargs="?", help="Specific skill to update (default: all outdated skills)")

    skills_audit = skills_subparsers.add_parser("audit", help="Re-scan installed hub skills")
    skills_audit.add_argument("name", nargs="?", help="Specific skill to audit (default: all)")

    skills_uninstall = skills_subparsers.add_parser("uninstall", help="Remove a hub-installed skill")
    skills_uninstall.add_argument("name", help="Skill name to remove")

    skills_publish = skills_subparsers.add_parser("publish", help="Publish a skill to a registry")
    skills_publish.add_argument("skill_path", help="Path to skill directory")
    skills_publish.add_argument("--to", default="github", choices=["github", "clawhub"], help="Target registry")
    skills_publish.add_argument("--repo", default="", help="Target GitHub repo (e.g. openai/skills)")

    skills_snapshot = skills_subparsers.add_parser("snapshot", help="Export/import skill configurations")
    snapshot_subparsers = skills_snapshot.add_subparsers(dest="snapshot_action")
    snap_export = snapshot_subparsers.add_parser("export", help="Export installed skills to a file")
    snap_export.add_argument("output", help="Output JSON file path (use - for stdout)")
    snap_import = snapshot_subparsers.add_parser("import", help="Import and install skills from a file")
    snap_import.add_argument("input", help="Input JSON file path")
    snap_import.add_argument("--force", action="store_true", help="Force install despite caution verdict")

    skills_tap = skills_subparsers.add_parser("tap", help="Manage skill sources")
    tap_subparsers = skills_tap.add_subparsers(dest="tap_action")
    tap_subparsers.add_parser("list", help="List configured taps")
    tap_add = tap_subparsers.add_parser("add", help="Add a GitHub repo as skill source")
    tap_add.add_argument("repo", help="GitHub repo (e.g. owner/repo)")
    tap_rm = tap_subparsers.add_parser("remove", help="Remove a tap")
    tap_rm.add_argument("name", help="Tap name to remove")

    # config sub-action: interactive enable/disable
    skills_subparsers.add_parser("config", help="Interactive skill configuration — enable/disable individual skills")

    def cmd_skills(args):
        # Route 'config' action to skills_config module
        if getattr(args, 'skills_action', None) == 'config':
            _require_tty("skills config")
            from hermes_cli.skills_config import skills_command as skills_config_command
            skills_config_command(args)
        else:
            from hermes_cli.skills_hub import skills_command
            skills_command(args)

    skills_parser.set_defaults(func=cmd_skills)

    # =========================================================================
    # plugins command
    # =========================================================================
    plugins_parser = subparsers.add_parser(
        "plugins",
        help="Manage plugins — install, update, remove, list",
        description="Install plugins from Git repositories, update, remove, or list them.",
    )
    plugins_subparsers = plugins_parser.add_subparsers(dest="plugins_action")

    plugins_install = plugins_subparsers.add_parser(
        "install", help="Install a plugin from a Git URL or owner/repo"
    )
    plugins_install.add_argument(
        "identifier",
        help="Git URL or owner/repo shorthand (e.g. anpicasso/hermes-plugin-chrome-profiles)",
    )
    plugins_install.add_argument(
        "--force", "-f", action="store_true",
        help="Remove existing plugin and reinstall",
    )

    plugins_update = plugins_subparsers.add_parser(
        "update", help="Pull latest changes for an installed plugin"
    )
    plugins_update.add_argument("name", help="Plugin name to update")

    plugins_remove = plugins_subparsers.add_parser(
        "remove", aliases=["rm", "uninstall"], help="Remove an installed plugin"
    )
    plugins_remove.add_argument("name", help="Plugin directory name to remove")

    plugins_subparsers.add_parser("list", aliases=["ls"], help="List installed plugins")

    plugins_enable = plugins_subparsers.add_parser(
        "enable", help="Enable a disabled plugin"
    )
    plugins_enable.add_argument("name", help="Plugin name to enable")

    plugins_disable = plugins_subparsers.add_parser(
        "disable", help="Disable a plugin without removing it"
    )
    plugins_disable.add_argument("name", help="Plugin name to disable")

    def cmd_plugins(args):
        from hermes_cli.plugins_cmd import plugins_command
        plugins_command(args)

    plugins_parser.set_defaults(func=cmd_plugins)

    # =========================================================================
    # Plugin CLI commands — dynamically registered by memory/general plugins.
    # Plugins provide a register_cli(subparser) function that builds their
    # own argparse tree.  No hardcoded plugin commands in main.py.
    # =========================================================================
    try:
        from plugins.memory import discover_plugin_cli_commands
        for cmd_info in discover_plugin_cli_commands():
            plugin_parser = subparsers.add_parser(
                cmd_info["name"],
                help=cmd_info["help"],
                description=cmd_info.get("description", ""),
                formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
            )
            cmd_info["setup_fn"](plugin_parser)
    except Exception as _exc:
        import logging as _log
        _log.getLogger(__name__).debug("Plugin CLI discovery failed: %s", _exc)

    # =========================================================================
    # memory command
    # =========================================================================
    memory_parser = subparsers.add_parser(
        "memory",
        help="Configure external memory provider",
        description=(
            "Set up and manage external memory provider plugins.\n\n"
            "Available providers: honcho, openviking, mem0, hindsight,\n"
            "holographic, retaindb, byterover.\n\n"
            "Only one external provider can be active at a time.\n"
            "Built-in memory (MEMORY.md/USER.md) is always active."
        ),
    )
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    memory_sub.add_parser("setup", help="Interactive provider selection and configuration")
    memory_sub.add_parser("status", help="Show current memory provider config")
    memory_sub.add_parser("off", help="Disable external provider (built-in only)")

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
        else:
            from hermes_cli.memory_setup import memory_command
            memory_command(args)

    memory_parser.set_defaults(func=cmd_memory)

    # =========================================================================
    # tools command
    # =========================================================================
    tools_parser = subparsers.add_parser(
        "tools",
        help="Configure which tools are enabled per platform",
        description=(
            "Enable, disable, or list tools for CLI, Telegram, Discord, etc.\n\n"
            "Built-in toolsets use plain names (e.g. web, memory).\n"
            "MCP tools use server:tool notation (e.g. github:create_issue).\n\n"
            "Run 'hermes tools' with no subcommand for the interactive configuration UI."
        ),
    )
    tools_parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary of enabled tools per platform and exit"
    )
    tools_sub = tools_parser.add_subparsers(dest="tools_action")

    # hermes tools list [--platform cli]
    tools_list_p = tools_sub.add_parser(
        "list",
        help="Show all tools and their enabled/disabled status",
    )
    tools_list_p.add_argument(
        "--platform", default="cli",
        help="Platform to show (default: cli)",
    )

    # hermes tools disable <name...> [--platform cli]
    tools_disable_p = tools_sub.add_parser(
        "disable",
        help="Disable toolsets or MCP tools",
    )
    tools_disable_p.add_argument(
        "names", nargs="+", metavar="NAME",
        help="Toolset name (e.g. web) or MCP tool in server:tool form",
    )
    tools_disable_p.add_argument(
        "--platform", default="cli",
        help="Platform to apply to (default: cli)",
    )

    # hermes tools enable <name...> [--platform cli]
    tools_enable_p = tools_sub.add_parser(
        "enable",
        help="Enable toolsets or MCP tools",
    )
    tools_enable_p.add_argument(
        "names", nargs="+", metavar="NAME",
        help="Toolset name or MCP tool in server:tool form",
    )
    tools_enable_p.add_argument(
        "--platform", default="cli",
        help="Platform to apply to (default: cli)",
    )

    def cmd_tools(args):
        action = getattr(args, "tools_action", None)
        if action in ("list", "disable", "enable"):
            from hermes_cli.tools_config import tools_disable_enable_command
            tools_disable_enable_command(args)
        else:
            _require_tty("tools")
            from hermes_cli.tools_config import tools_command
            tools_command(args)

    tools_parser.set_defaults(func=cmd_tools)
    # =========================================================================
    # mcp command — manage MCP server connections
    # =========================================================================
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Manage MCP servers and run Hermes as an MCP server",
        description=(
            "Manage MCP server connections and run Hermes as an MCP server.\n\n"
            "MCP servers provide additional tools via the Model Context Protocol.\n"
            "Use 'hermes mcp add' to connect to a new server, or\n"
            "'hermes mcp serve' to expose Hermes conversations over MCP."
        ),
    )
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_action")

    mcp_serve_p = mcp_sub.add_parser(
        "serve",
        help="Run Hermes as an MCP server (expose conversations to other agents)",
    )
    mcp_serve_p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging on stderr",
    )

    mcp_add_p = mcp_sub.add_parser("add", help="Add an MCP server (discovery-first install)")
    mcp_add_p.add_argument("name", help="Server name (used as config key)")
    mcp_add_p.add_argument("--url", help="HTTP/SSE endpoint URL")
    mcp_add_p.add_argument("--command", help="Stdio command (e.g. npx)")
    mcp_add_p.add_argument("--args", nargs="*", default=[], help="Arguments for stdio command")
    mcp_add_p.add_argument("--auth", choices=["oauth", "header"], help="Auth method")
    mcp_add_p.add_argument("--preset", help="Known MCP preset name")
    mcp_add_p.add_argument("--env", nargs="*", default=[], help="Environment variables for stdio servers (KEY=VALUE)")

    mcp_rm_p = mcp_sub.add_parser("remove", aliases=["rm"], help="Remove an MCP server")
    mcp_rm_p.add_argument("name", help="Server name to remove")

    mcp_sub.add_parser("list", aliases=["ls"], help="List configured MCP servers")

    mcp_test_p = mcp_sub.add_parser("test", help="Test MCP server connection")
    mcp_test_p.add_argument("name", help="Server name to test")

    mcp_cfg_p = mcp_sub.add_parser("configure", aliases=["config"], help="Toggle tool selection")
    mcp_cfg_p.add_argument("name", help="Server name to configure")

    def cmd_mcp(args):
        from hermes_cli.mcp_config import mcp_command
        mcp_command(args)

    mcp_parser.set_defaults(func=cmd_mcp)

    # =========================================================================
    # sessions command
    # =========================================================================
    sessions_parser = subparsers.add_parser(
        "sessions",
        help="Manage session history (list, rename, export, prune, delete)",
        description="View and manage the SQLite session store"
    )
    sessions_subparsers = sessions_parser.add_subparsers(dest="sessions_action")

    sessions_list = sessions_subparsers.add_parser("list", help="List recent sessions")
    sessions_list.add_argument("--source", help="Filter by source (cli, telegram, discord, etc.)")
    sessions_list.add_argument("--limit", type=int, default=20, help="Max sessions to show")

    sessions_export = sessions_subparsers.add_parser("export", help="Export sessions to a JSONL file")
    sessions_export.add_argument("output", help="Output JSONL file path (use - for stdout)")
    sessions_export.add_argument("--source", help="Filter by source")
    sessions_export.add_argument("--session-id", help="Export a specific session")

    sessions_delete = sessions_subparsers.add_parser("delete", help="Delete a specific session")
    sessions_delete.add_argument("session_id", help="Session ID to delete")
    sessions_delete.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    sessions_prune = sessions_subparsers.add_parser("prune", help="Delete old sessions")
    sessions_prune.add_argument("--older-than", type=int, default=90, help="Delete sessions older than N days (default: 90)")
    sessions_prune.add_argument("--source", help="Only prune sessions from this source")
    sessions_prune.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    sessions_subparsers.add_parser("stats", help="Show session store statistics")

    sessions_rename = sessions_subparsers.add_parser("rename", help="Set or change a session's title")
    sessions_rename.add_argument("session_id", help="Session ID to rename")
    sessions_rename.add_argument("title", nargs="+", help="New title for the session")

    sessions_browse = sessions_subparsers.add_parser(
        "browse",
        help="Interactive session picker — browse, search, and resume sessions",
    )
    sessions_browse.add_argument("--source", help="Filter by source (cli, telegram, discord, etc.)")
    sessions_browse.add_argument("--limit", type=int, default=50, help="Max sessions to load (default: 50)")

    def _confirm_prompt(prompt: str) -> bool:
        """Prompt for y/N confirmation, safe against non-TTY environments."""
        try:
            return input(prompt).strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def cmd_sessions(args):
        import json as _json
        try:
            from hermes_state import SessionDB
            db = SessionDB()
        except Exception as e:
            print(f"Error: Could not open session database: {e}")
            return

        action = args.sessions_action

        # Hide third-party tool sessions by default, but honour explicit --source
        _source = getattr(args, "source", None)
        _exclude = None if _source else ["tool"]

        if action == "list":
            sessions = db.list_sessions_rich(source=args.source, exclude_sources=_exclude, limit=args.limit)
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
                preview = s.get("preview", "")[:38] if has_titles else s.get("preview", "")[:48]
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
                    import sys
                    sys.stdout.write(line)
                else:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(line)
                    print(f"Exported 1 session to {args.output}")
            else:
                sessions = db.export_all(source=args.source)
                if args.output == "-":
                    import sys
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
                if not _confirm_prompt(f"Delete session '{resolved_session_id}' and all its messages? [y/N] "):
                    print("Cancelled.")
                    return
            if db.delete_session(resolved_session_id):
                print(f"Deleted session '{resolved_session_id}'.")
            else:
                print(f"Session '{args.session_id}' not found.")

        elif action == "prune":
            days = args.older_than
            source_msg = f" from '{args.source}'" if args.source else ""
            if not args.yes:
                if not _confirm_prompt(f"Delete all ended sessions older than {days} days{source_msg}? [y/N] "):
                    print("Cancelled.")
                    return
            count = db.prune_sessions(older_than_days=days, source=args.source)
            print(f"Pruned {count} session(s).")

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
            limit = getattr(args, "limit", 50) or 50
            source = getattr(args, "source", None)
            _browse_exclude = None if source else ["tool"]
            sessions = db.list_sessions_rich(source=source, exclude_sources=_browse_exclude, limit=limit)
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
            import shutil
            hermes_bin = shutil.which("hermes")
            if hermes_bin:
                os.execvp(hermes_bin, ["hermes", "--resume", selected_id])
            else:
                # Fallback: re-invoke via python -m
                os.execvp(
                    sys.executable,
                    [sys.executable, "-m", "hermes_cli.main", "--resume", selected_id],
                )
            return  # won't reach here after execvp

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
    # insights command
    # =========================================================================
    insights_parser = subparsers.add_parser(
        "insights",
        help="Show usage insights and analytics",
        description="Analyze session history to show token usage, costs, tool patterns, and activity trends"
    )
    insights_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze (default: 30)")
    insights_parser.add_argument("--source", help="Filter by platform (cli, telegram, discord, etc.)")

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

    insights_parser.set_defaults(func=cmd_insights)

    # =========================================================================
    # claw command (OpenClaw migration)
    # =========================================================================
    claw_parser = subparsers.add_parser(
        "claw",
        help="OpenClaw migration tools",
        description="Migrate settings, memories, skills, and API keys from OpenClaw to Hermes"
    )
    claw_subparsers = claw_parser.add_subparsers(dest="claw_action")

    # claw migrate
    claw_migrate = claw_subparsers.add_parser(
        "migrate",
        help="Migrate from OpenClaw to Hermes",
        description="Import settings, memories, skills, and API keys from an OpenClaw installation. "
                    "Always shows a preview before making changes."
    )
    claw_migrate.add_argument(
        "--source",
        help="Path to OpenClaw directory (default: ~/.openclaw)"
    )
    claw_migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only — stop after showing what would be migrated"
    )
    claw_migrate.add_argument(
        "--preset",
        choices=["user-data", "full"],
        default="full",
        help="Migration preset (default: full). 'user-data' excludes secrets"
    )
    claw_migrate.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip conflicts)"
    )
    claw_migrate.add_argument(
        "--migrate-secrets",
        action="store_true",
        help="Include allowlisted secrets (TELEGRAM_BOT_TOKEN, API keys, etc.)"
    )
    claw_migrate.add_argument(
        "--workspace-target",
        help="Absolute path to copy workspace instructions into"
    )
    claw_migrate.add_argument(
        "--skill-conflict",
        choices=["skip", "overwrite", "rename"],
        default="skip",
        help="How to handle skill name conflicts (default: skip)"
    )
    claw_migrate.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts"
    )

    # claw cleanup
    claw_cleanup = claw_subparsers.add_parser(
        "cleanup",
        aliases=["clean"],
        help="Archive leftover OpenClaw directories after migration",
        description="Scan for and archive leftover OpenClaw directories to prevent state fragmentation"
    )
    claw_cleanup.add_argument(
        "--source",
        help="Path to a specific OpenClaw directory to clean up"
    )
    claw_cleanup.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be archived without making changes"
    )
    claw_cleanup.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts"
    )

    def cmd_claw(args):
        from hermes_cli.claw import claw_command
        claw_command(args)

    claw_parser.set_defaults(func=cmd_claw)

    # =========================================================================
    # version command
    # =========================================================================
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information"
    )
    version_parser.set_defaults(func=cmd_version)
    
    # =========================================================================
    # update command
    # =========================================================================
    update_parser = subparsers.add_parser(
        "update",
        help="Update Hermes Agent to the latest version",
        description="Pull the latest changes from git and reinstall dependencies"
    )
    update_parser.add_argument(
        "--gateway", action="store_true", default=False,
        help="Gateway mode: use file-based IPC for prompts instead of stdin (used internally by /update)"
    )
    update_parser.set_defaults(func=cmd_update)
    
    # =========================================================================
    # uninstall command
    # =========================================================================
    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="Uninstall Hermes Agent",
        description="Remove Hermes Agent from your system. Can keep configs/data for reinstall."
    )
    uninstall_parser.add_argument(
        "--full",
        action="store_true",
        help="Full uninstall - remove everything including configs and data"
    )
    uninstall_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts"
    )
    uninstall_parser.set_defaults(func=cmd_uninstall)

    # =========================================================================
    # acp command
    # =========================================================================
    acp_parser = subparsers.add_parser(
        "acp",
        help="Run Hermes Agent as an ACP (Agent Client Protocol) server",
        description="Start Hermes Agent in ACP mode for editor integration (VS Code, Zed, JetBrains)",
    )

    def cmd_acp(args):
        """Launch Hermes Agent as an ACP server."""
        try:
            from acp_adapter.entry import main as acp_main
            acp_main()
        except ImportError:
            print("ACP dependencies not installed.")
            print("Install them with:  pip install -e '.[acp]'")
            sys.exit(1)

    acp_parser.set_defaults(func=cmd_acp)

    # =========================================================================
    # profile command
    # =========================================================================
    profile_parser = subparsers.add_parser(
        "profile",
        help="Manage profiles — multiple isolated Hermes instances",
    )
    profile_subparsers = profile_parser.add_subparsers(dest="profile_action")

    profile_subparsers.add_parser("list", help="List all profiles")
    profile_use = profile_subparsers.add_parser("use", help="Set sticky default profile")
    profile_use.add_argument("profile_name", help="Profile name (or 'default')")

    profile_create = profile_subparsers.add_parser("create", help="Create a new profile")
    profile_create.add_argument("profile_name", help="Profile name (lowercase, alphanumeric)")
    profile_create.add_argument("--clone", action="store_true",
                                help="Copy config.yaml, .env, SOUL.md from active profile")
    profile_create.add_argument("--clone-all", action="store_true",
                                help="Full copy of active profile (all state)")
    profile_create.add_argument("--clone-from", metavar="SOURCE",
                                help="Source profile to clone from (default: active)")
    profile_create.add_argument("--no-alias", action="store_true",
                                help="Skip wrapper script creation")

    profile_delete = profile_subparsers.add_parser("delete", help="Delete a profile")
    profile_delete.add_argument("profile_name", help="Profile to delete")
    profile_delete.add_argument("-y", "--yes", action="store_true",
                                help="Skip confirmation prompt")

    profile_show = profile_subparsers.add_parser("show", help="Show profile details")
    profile_show.add_argument("profile_name", help="Profile to show")

    profile_alias = profile_subparsers.add_parser("alias", help="Manage wrapper scripts")
    profile_alias.add_argument("profile_name", help="Profile name")
    profile_alias.add_argument("--remove", action="store_true",
                               help="Remove the wrapper script")
    profile_alias.add_argument("--name", dest="alias_name", metavar="NAME",
                               help="Custom alias name (default: profile name)")

    profile_rename = profile_subparsers.add_parser("rename", help="Rename a profile")
    profile_rename.add_argument("old_name", help="Current profile name")
    profile_rename.add_argument("new_name", help="New profile name")

    profile_export = profile_subparsers.add_parser("export", help="Export a profile to archive")
    profile_export.add_argument("profile_name", help="Profile to export")
    profile_export.add_argument("-o", "--output", default=None,
                                help="Output file (default: <name>.tar.gz)")

    profile_import = profile_subparsers.add_parser("import", help="Import a profile from archive")
    profile_import.add_argument("archive", help="Path to .tar.gz archive")
    profile_import.add_argument("--name", dest="import_name", metavar="NAME",
                                help="Profile name (default: inferred from archive)")

    profile_parser.set_defaults(func=cmd_profile)

    # =========================================================================
    # completion command
    # =========================================================================
    completion_parser = subparsers.add_parser(
        "completion",
        help="Print shell completion script (bash, zsh, or fish)",
    )
    completion_parser.add_argument(
        "shell", nargs="?", default="bash", choices=["bash", "zsh", "fish"],
        help="Shell type (default: bash)",
    )
    completion_parser.set_defaults(func=lambda args: cmd_completion(args, parser))

    # =========================================================================
    # dashboard command
    # =========================================================================
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Start the web UI dashboard",
        description="Launch the Hermes Agent web dashboard for managing config, API keys, and sessions",
    )
    dashboard_parser.add_argument("--port", type=int, default=9119, help="Port (default 9119)")
    dashboard_parser.add_argument("--host", default="127.0.0.1", help="Host (default 127.0.0.1)")
    dashboard_parser.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    dashboard_parser.add_argument(
        "--insecure", action="store_true",
        help="Allow binding to non-localhost (DANGEROUS: exposes API keys on the network)",
    )
    dashboard_parser.set_defaults(func=cmd_dashboard)

    # =========================================================================
    # logs command
    # =========================================================================
    logs_parser = subparsers.add_parser(
        "logs",
        help="View and filter Hermes log files",
        description="View, tail, and filter agent.log / errors.log / gateway.log",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    hermes logs                    Show last 50 lines of agent.log
    hermes logs -f                 Follow agent.log in real time
    hermes logs errors             Show last 50 lines of errors.log
    hermes logs gateway -n 100     Show last 100 lines of gateway.log
    hermes logs --level WARNING    Only show WARNING and above
    hermes logs --session abc123   Filter by session ID
    hermes logs --component tools  Only show tool-related lines
    hermes logs --since 1h         Lines from the last hour
    hermes logs --since 30m -f     Follow, starting from 30 min ago
    hermes logs list               List available log files with sizes
""",
    )
    logs_parser.add_argument(
        "log_name", nargs="?", default="agent",
        help="Log to view: agent (default), errors, gateway, or 'list' to show available files",
    )
    logs_parser.add_argument(
        "-n", "--lines", type=int, default=50,
        help="Number of lines to show (default: 50)",
    )
    logs_parser.add_argument(
        "-f", "--follow", action="store_true",
        help="Follow the log in real time (like tail -f)",
    )
    logs_parser.add_argument(
        "--level", metavar="LEVEL",
        help="Minimum log level to show (DEBUG, INFO, WARNING, ERROR)",
    )
    logs_parser.add_argument(
        "--session", metavar="ID",
        help="Filter lines containing this session ID substring",
    )
    logs_parser.add_argument(
        "--since", metavar="TIME",
        help="Show lines since TIME ago (e.g. 1h, 30m, 2d)",
    )
    logs_parser.add_argument(
        "--component", metavar="NAME",
        help="Filter by component: gateway, agent, tools, cli, cron",
    )
    logs_parser.set_defaults(func=cmd_logs)

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
    args = parser.parse_args(_processed_argv)

    # Handle --version flag
    if args.version:
        cmd_version(args)
        return
    
    # Handle top-level --resume / --continue as shortcut to chat
    if (args.resume or args.continue_last) and args.command is None:
        args.command = "chat"
        args.query = None
        args.model = None
        args.provider = None
        args.toolsets = None
        args.verbose = False
        if not hasattr(args, "worktree"):
            args.worktree = False
        cmd_chat(args)
        return
    
    # Default to chat if no command specified
    if args.command is None:
        args.query = None
        args.model = None
        args.provider = None
        args.toolsets = None
        args.verbose = False
        args.resume = None
        args.continue_last = None
        if not hasattr(args, "worktree"):
            args.worktree = False
        cmd_chat(args)
        return
    
    # Execute the command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
