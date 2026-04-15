#!/usr/bin/env python3
"""
Hermes CLI - 메인 진입점.

사용 예:
    hermes                     # 대화형 채팅(기본값)
    hermes chat                # 대화형 채팅
    hermes gateway             # 게이트웨이를 포그라운드에서 실행
    hermes gateway start       # 게이트웨이 서비스를 시작
    hermes gateway stop        # 게이트웨이 서비스를 중지
    hermes setup               # 설정 마법사 실행
    hermes tools               # 활성화할 도구 설정
    hermes model               # 기본 모델 설정
    hermes status              # 시스템/provider 상태 표시
    hermes config              # 현재 설정 표시
    hermes cron                # 크론 작업 관리
    hermes doctor              # 설정과 의존성 점검
    hermes version             # 버전 표시
    hermes update              # 최신 버전으로 업데이트
    hermes uninstall           # Hermes Agent 제거
    hermes acp                 # 에디터 연동용 ACP 서버로 실행
    hermes sessions browse     # 검색 가능한 세션 선택기 실행

    hermes claw migrate --dry-run  # 변경 없이 마이그레이션 미리보기
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


_ARGPARSE_KO_TRANSLATIONS = {
    "usage: ": "사용법: ",
    "options": "옵션",
    "optional arguments": "옵션",
    "positional arguments": "위치 인자",
    "show this help message and exit": "이 도움말을 표시하고 종료",
    "subcommands": "하위 명령어",
}


def _argparse_korean(text: str) -> str:
    return _ARGPARSE_KO_TRANSLATIONS.get(text, text)


argparse._ = _argparse_korean


class KoreanArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._positionals.title = "위치 인자"
        self._optionals.title = "옵션"

    def add_subparsers(self, *args, **kwargs):
        kwargs.setdefault("title", "하위 명령어")
        return super().add_subparsers(*args, **kwargs)

def _require_tty(command_name: str) -> None:
    """Exit with a clear error if stdin is not a terminal.

    Interactive TUI commands (hermes tools, hermes setup, hermes model) use
    curses or input() prompts that spin at 100% CPU when stdin is a pipe.
    This guard prevents accidental non-interactive invocation.
    """
    if not sys.stdin.isatty():
        print(
            f"오류: 'hermes {command_name}' 명령은 대화형 터미널이 필요합니다.\n"
            f"파이프나 비대화형 서브프로세스에서는 실행할 수 없습니다.\n"
            f"터미널에서 직접 실행해 주세요.",
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
        print("세션을 찾지 못했어요.")
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
                        stdscr.addstr(0, 0, "터미널 크기가 너무 작습니다")
                    except curses.error:
                        pass
                    stdscr.refresh()
                    stdscr.getch()
                    return

                # Header line
                if search_text:
                    header = f"  세션 탐색 — 필터: {search_text}█"
                    header_attr = curses.A_BOLD
                    if curses.has_colors():
                        header_attr |= curses.color_pair(3)
                else:
                    header = "  세션 탐색 — ↑↓ 이동  Enter 선택  입력으로 필터  Esc 종료"
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
                col_header = f"   {'제목 / 미리보기':<{name_width}}  {'최근 활동':<10}  {'출처':<5} {'ID'}"
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
                        msg = "  필터와 일치하는 세션이 없습니다."
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
                    footer = f"  {cursor + 1}/{len(filtered)}개 세션"
                    if len(filtered) < len(sessions):
                        footer += f" (전체 {len(sessions)}개 중 필터됨)"
                else:
                    footer = f"  0/{len(sessions)}개 세션"
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
    print("\n  세션 탐색  (번호를 입력하면 이어서 열고, q를 입력하면 취소)\n")
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
            val = input(f"\n  선택 [1-{len(sessions)}]: ").strip()
            if not val or val.lower() in ("q", "quit", "exit"):
                return None
            idx = int(val) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]["id"]
            print(f"  잘못된 선택입니다. 1-{len(sessions)} 또는 q를 입력해 취소하세요.")
        except ValueError:
            print("  잘못된 입력입니다. 숫자 또는 q를 입력해 취소하세요.")
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
                print(f"'{continue_val}'와 일치하는 세션을 찾지 못했어요.")
                print("사용 가능한 세션은 'hermes sessions list'로 확인할 수 있어요.")
                sys.exit(1)
        else:
            # -c with no argument — continue the most recent session
            last_id = _resolve_last_cli_session()
            if last_id:
                args.resume = last_id
            else:
                print("이어서 열 수 있는 이전 CLI 세션이 없어요.")
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
        print("Hermes가 아직 설정되지 않은 것 같아요. API 키나 provider 설정을 찾지 못했어요.")
        print()
        print("  실행: hermes setup")
        print()

        from hermes_cli.setup import is_interactive_stdin, print_noninteractive_setup_guidance

        if not is_interactive_stdin():
            print_noninteractive_setup_guidance(
                "첫 실행 설정 프롬프트에 필요한 대화형 TTY를 찾지 못했어요."
            )
            sys.exit(1)

        try:
            reply = input("지금 setup을 실행할까요? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = "n"
        if reply in ("", "y", "yes"):
            cmd_setup(args)
            return
        print()
        print("설정이 필요하면 언제든 'hermes setup'을 실행하면 돼요.")
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
    print("⚕ WhatsApp 설정")
    print("=" * 50)

    # ── Step 1: Choose mode ──────────────────────────────────────────────
    current_mode = get_env_value("WHATSAPP_MODE") or ""
    if not current_mode:
        print()
        print("Hermes에서 WhatsApp을 어떻게 사용할까요?")
        print()
        print("  1. 별도 봇 번호 사용 (권장)")
        print("     사람들이 봇 번호로 직접 메시지를 보내는 방식이라 가장 깔끔해요.")
        print("     WhatsApp이 설치된 기기와 두 번째 전화번호가 필요해요.")
        print()
        print("  2. 개인 번호 사용 (self-chat)")
        print("     자기 자신에게 메시지를 보내는 방식으로 에이전트와 대화해요.")
        print("     설정은 빠르지만 사용 경험은 덜 직관적일 수 있어요.")
        print()
        try:
            choice = input("  선택 [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n설정을 취소했어요.")
            return

        if choice == "1":
            save_env_value("WHATSAPP_MODE", "bot")
            wa_mode = "bot"
            print("  ✓ 모드: 별도 봇 번호")
            print()
            print("  ┌─────────────────────────────────────────────────┐")
            print("  │  봇용 두 번째 번호 준비 방법:                    │")
            print("  │                                                 │")
            print("  │  가장 쉬운 방법: WhatsApp Business(무료 앱)를   │")
            print("  │  두 번째 번호로 휴대폰에 설치하세요:            │")
            print("  │    • 듀얼 SIM: 두 번째 SIM 슬롯 사용            │")
            print("  │    • Google Voice: 무료 미국 번호(voice.google) │")
            print("  │    • 선불 SIM: 1회 인증용으로 저렴하게 구매      │")
            print("  │                                                 │")
            print("  │  WhatsApp Business는 개인 WhatsApp과 함께       │")
            print("  │  같은 휴대폰에서 사용할 수 있어요.              │")
            print("  └─────────────────────────────────────────────────┘")
        else:
            save_env_value("WHATSAPP_MODE", "self-chat")
            wa_mode = "self-chat"
            print("  ✓ 모드: 개인 번호 (self-chat)")
    else:
        wa_mode = current_mode
        mode_label = "별도 봇 번호" if wa_mode == "bot" else "개인 번호 (self-chat)"
        print(f"\n✓ 모드: {mode_label}")

    # ── Step 2: Enable WhatsApp ──────────────────────────────────────────
    print()
    current = get_env_value("WHATSAPP_ENABLED")
    if current and current.lower() == "true":
        print("✓ WhatsApp이 이미 활성화되어 있어요")
    else:
        save_env_value("WHATSAPP_ENABLED", "true")
        print("✓ WhatsApp을 활성화했어요")

    # ── Step 3: Allowed users ────────────────────────────────────────────
    current_users = get_env_value("WHATSAPP_ALLOWED_USERS") or ""
    if current_users:
        print(f"✓ 허용된 사용자: {current_users}")
        try:
            response = input("\n  허용된 사용자를 수정할까요? [y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in ("y", "yes"):
            if wa_mode == "bot":
                phone = input("  봇에 메시지를 보낼 수 있는 전화번호(쉼표 구분): ").strip()
            else:
                phone = input("  내 전화번호 (예: 15551234567): ").strip()
            if phone:
                save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
                print(f"  ✓ 다음 값으로 업데이트했어요: {phone}")
    else:
        print()
        if wa_mode == "bot":
            print("  누가 봇에게 메시지를 보낼 수 있도록 할까요?")
            phone = input("  전화번호 입력(쉼표 구분, 모두 허용은 *): ").strip()
        else:
            phone = input("  내 전화번호 (예: 15551234567): ").strip()
        if phone:
            save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
            print(f"  ✓ 허용된 사용자를 설정했어요: {phone}")
        else:
            print("  ⚠ allowlist가 없어요 — 에이전트가 모든 수신 메시지에 응답할 수 있어요")

    # ── Step 4: Install bridge dependencies ──────────────────────────────
    project_root = Path(__file__).resolve().parents[1]
    bridge_dir = project_root / "scripts" / "whatsapp-bridge"
    bridge_script = bridge_dir / "bridge.js"

    if not bridge_script.exists():
        print(f"\n✗ bridge 스크립트를 찾지 못했어요: {bridge_script}")
        return

    if not (bridge_dir / "node_modules").exists():
        print("\n→ WhatsApp bridge 의존성을 설치하는 중...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=str(bridge_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  ✗ npm install에 실패했어요: {result.stderr}")
            return
        print("  ✓ 의존성 설치 완료")
    else:
        print("✓ Bridge 의존성이 이미 설치되어 있어요")

    # ── Step 5: Check for existing session ───────────────────────────────
    session_dir = get_hermes_home() / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    if (session_dir / "creds.json").exists():
        print("✓ 기존 WhatsApp 세션을 찾았어요")
        try:
            response = input("\n  다시 페어링할까요? 기존 세션이 초기화돼요. [y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in ("y", "yes"):
            import shutil
            shutil.rmtree(session_dir, ignore_errors=True)
            session_dir.mkdir(parents=True, exist_ok=True)
            print("  ✓ 세션을 초기화했어요")
        else:
            print("\n✓ WhatsApp 설정과 페어링이 완료되어 있어요!")
            print("  gateway 시작 명령: hermes gateway")
            return

    # ── Step 6: QR code pairing ──────────────────────────────────────────
    print()
    print("─" * 50)
    if wa_mode == "bot":
        print("📱 봇 번호가 연결된 휴대폰에서 WhatsApp(또는 WhatsApp Business)을 열고")
        print("   아래 절차로 스캔해 주세요:")
    else:
        print("📱 휴대폰에서 WhatsApp을 열고 아래 절차로 스캔해 주세요:")
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
        print("✓ WhatsApp 페어링이 완료됐어요!")
        print()
        if wa_mode == "bot":
            print("  다음 단계:")
            print("    1. gateway 시작: hermes gateway")
            print("    2. 봇의 WhatsApp 번호로 메시지 보내기")
            print("    3. 에이전트가 자동으로 응답해요")
            print()
            print("  팁: 에이전트 응답은 '⚕ Hermes Agent' 접두사로 표시돼요")
        else:
            print("  다음 단계:")
            print("    1. gateway 시작: hermes gateway")
            print("    2. WhatsApp에서 'Message Yourself' 열기")
            print("    3. 메시지를 입력하면 에이전트가 응답해요")
            print()
            print("  팁: 에이전트 응답은 '⚕ Hermes Agent' 접두사로 표시돼요")
            print("  그래서 내 메시지와 쉽게 구분할 수 있어요.")
        print()
        print("  또는 서비스로 설치: hermes gateway install")
    else:
        print("⚠ 페어링이 완료되지 않았을 수 있어요. 다시 시도하려면 'hermes whatsapp'을 실행해 주세요.")


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
        print(f"경고: {warning} auto provider 감지로 되돌릴게요.")
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
    print(f"  현재 모델:      {current_model}")
    print(f"  활성 provider:  {active_label}")
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

    ordered.append(("custom", "사용자 지정 endpoint (URL 직접 입력)"))
    _has_saved_custom_list = isinstance(config.get("custom_providers"), list) and bool(config.get("custom_providers"))
    if _has_saved_custom_list:
        ordered.append(("remove-custom", "저장된 custom provider 제거"))
    ordered.append(("cancel", "취소"))

    provider_idx = _prompt_provider_choice(
        [label for _, label in ordered], default=default_idx,
    )
    if provider_idx is None or ordered[provider_idx][0] == "cancel":
        print("변경 사항이 없어요.")
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
                "경고: 선택한 저장된 custom provider를 더 이상 사용할 수 없어요. "
                "config.yaml에서 제거되었을 수 있어요. 변경 사항은 없어요."
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
        idx = _curses_prompt_choice("provider 선택:", choices, default)
        if idx >= 0:
            print()
            return idx
    except Exception:
        pass

    # Fallback: numbered list
    print("provider 선택:")
    for i, c in enumerate(choices, 1):
        marker = "→" if i - 1 == default else " "
        print(f"  {marker} {i}. {c}")
    print()
    while True:
        try:
            val = input(f"선택 [1-{len(choices)}] ({default + 1}): ").strip()
            if not val:
                return default
            idx = int(val) - 1
            if 0 <= idx < len(choices):
                return idx
            print(f"1-{len(choices)} 사이의 번호를 입력해 주세요")
        except ValueError:
            print("숫자를 입력해 주세요")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


def _model_flow_openrouter(config, current_model=""):
    """OpenRouter provider: ensure API key, then pick model."""
    from hermes_cli.auth import _prompt_model_selection, _save_model_choice, deactivate_provider
    from hermes_cli.config import get_env_value, save_env_value

    api_key = get_env_value("OPENROUTER_API_KEY")
    if not api_key:
        print("OpenRouter API key가 설정되어 있지 않아요.")
        print("발급 링크: https://openrouter.ai/keys")
        print()
        try:
            import getpass
            key = getpass.getpass("OpenRouter API key 입력(취소하려면 Enter): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not key:
            print("취소했어요.")
            return
        save_env_value("OPENROUTER_API_KEY", key)
        print("API key를 저장했어요.")
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
        print(f"기본 모델을 설정했어요: {selected} (OpenRouter 사용)")
    else:
        print("변경 사항이 없어요.")


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
        print("Nous Portal에 로그인되어 있지 않아요. 로그인을 시작할게요...")
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
            print("로그인을 취소했거나 로그인에 실패했어요.")
            return
        except Exception as exc:
            print(f"로그인에 실패했어요: {exc}")
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
        print("Nous Portal에서 사용할 큐레이션 모델이 없어요.")
        return

    # Verify credentials are still valid (catches expired sessions early)
    try:
        creds = resolve_nous_runtime_credentials(min_key_ttl_seconds=5 * 60)
    except Exception as exc:
        relogin = isinstance(exc, AuthError) and exc.relogin_required
        msg = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
        if relogin:
            print(f"세션이 만료되었어요: {msg}")
            print("Nous Portal에 다시 로그인하는 중...\n")
            try:
                mock_args = argparse.Namespace(
                    portal_url=None, inference_url=None, client_id=None,
                    scope=None, no_browser=False, timeout=15.0,
                    ca_bundle=None, insecure=False,
                )
                _login_nous(mock_args, PROVIDER_REGISTRY["nous"])
            except Exception as login_exc:
                print(f"재로그인에 실패했어요: {login_exc}")
            return
        print(f"자격 증명을 확인하지 못했어요: {msg}")
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
        print("필터링 후 Nous Portal에서 사용할 수 있는 모델이 없어요.")
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
        print("현재 사용할 수 있는 무료 모델이 없어요.")
        if unavailable_models:
            from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL
            _url = (_nous_portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
            print(f"유료 모델을 사용하려면 {_url} 에서 업그레이드해 주세요.")
        return

    print(f"큐레이션 모델 {len(model_ids)}개를 표시합니다. 다른 모델은 \"사용자 지정 모델 이름 입력\"을 이용하세요.")

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
        print(f"기본 모델을 설정했어요: {selected} (Nous Portal 사용)")
        if "tts" in changed_defaults:
            print("TTS provider를 설정했어요: Nous 구독의 OpenAI TTS 사용")
        else:
            current_tts = str(config.get("tts", {}).get("provider") or "edge")
            if current_tts.lower() not in {"", "edge"}:
                print(f"기존 TTS provider 설정을 유지할게요: {current_tts}")
        print()
        for line in get_nous_subscription_explainer_lines():
            print(line)
    else:
        print("변경 사항이 없어요.")


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
        print("OpenAI Codex에 로그인되어 있지 않아요. 로그인을 시작할게요...")
        print()
        try:
            mock_args = argparse.Namespace()
            _login_openai_codex(mock_args, PROVIDER_REGISTRY["openai-codex"])
        except SystemExit:
            print("로그인을 취소했거나 로그인에 실패했어요.")
            return
        except Exception as exc:
            print(f"로그인에 실패했어요: {exc}")
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
        print(f"기본 모델을 설정했어요: {selected} (OpenAI Codex 사용)")
    else:
        print("변경 사항이 없어요.")



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
        print("Qwen CLI OAuth에 로그인되어 있지 않아요.")
        print("실행: qwen auth qwen-oauth")
        auth_file = status.get("auth_file")
        if auth_file:
            print(f"예상 자격 증명 파일 위치: {auth_file}")
        if status.get("error"):
            print(f"오류: {status.get('error')}")
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
        print(f"기본 모델을 설정했어요: {selected} (Qwen OAuth 사용)")
    else:
        print("변경 사항이 없어요.")



def _model_flow_custom(config):
    """Custom endpoint: collect URL, API key, and model name.

    Automatically saves the endpoint to ``custom_providers`` in config.yaml
    so it appears in the provider menu on subsequent runs.
    """
    from hermes_cli.auth import _save_model_choice, deactivate_provider
    from hermes_cli.config import get_env_value, load_config, save_config

    current_url = get_env_value("OPENAI_BASE_URL") or ""
    current_key = get_env_value("OPENAI_API_KEY") or ""

    print("사용자 지정 OpenAI 호환 endpoint 설정:")
    if current_url:
        print(f"  현재 URL: {current_url}")
    if current_key:
        print(f"  현재 키: {current_key[:8]}...")
    print()

    try:
        base_url = input(f"API base URL [{current_url or '예: https://api.example.com/v1'}]: ").strip()
        import getpass
        api_key = getpass.getpass(f"API key [{current_key[:8] + '...' if current_key else '선택 사항'}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n취소했어요.")
        return

    if not base_url and not current_url:
        print("URL이 제공되지 않아 취소했어요.")
        return

    # Validate URL format
    effective_url = base_url or current_url
    if not effective_url.startswith(("http://", "https://")):
        print(f"잘못된 URL이에요: {effective_url} (http:// 또는 https:// 로 시작해야 해요)")
        return

    effective_key = api_key or current_key

    from hermes_cli.models import probe_api_models

    probe = probe_api_models(effective_key, effective_url)
    if probe.get("used_fallback") and probe.get("resolved_base_url"):
        print(
            f"경고: endpoint 검증은 {probe['resolved_base_url']}/models 에서 성공했어요. "
            f"입력한 정확한 URL 대신 동작하는 base URL을 저장할게요."
        )
        effective_url = probe["resolved_base_url"]
        if base_url:
            base_url = effective_url
    elif probe.get("models") is not None:
        print(
            f"{probe.get('probed_url')} 경로로 endpoint 검증을 마쳤어요 "
            f"(보이는 모델 {len(probe.get('models') or [])}개)"
        )
    else:
        print(
            f"경고: {probe.get('probed_url')} 경로로 이 endpoint를 검증하지 못했어요. "
            f"그래도 Hermes에 저장할게요."
        )
        if probe.get("suggested_base_url"):
            suggested = probe["suggested_base_url"]
            if suggested.endswith("/v1"):
                print(f"  이 서버가 경로에 /v1 을 기대한다면 base URL로 다음 값을 시도해 보세요: {suggested}")
            else:
                print(f"  base URL에 /v1 이 없어야 한다면 다음 값을 시도해 보세요: {suggested}")

    # Select model — use probe results when available, fall back to manual input
    model_name = ""
    detected_models = probe.get("models") or []
    try:
        if len(detected_models) == 1:
            print(f"  감지된 모델: {detected_models[0]}")
            confirm = input("  이 모델을 사용할까요? [Y/n]: ").strip().lower()
            if confirm in ("", "y", "yes"):
                model_name = detected_models[0]
            else:
                model_name = input("모델 이름 입력 (예: gpt-4, llama-3-70b): ").strip()
        elif len(detected_models) > 1:
            print("  사용 가능한 모델:")
            for i, m in enumerate(detected_models, 1):
                print(f"    {i}. {m}")
            pick = input(f"  모델 선택 [1-{len(detected_models)}] 또는 이름 직접 입력: ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(detected_models):
                model_name = detected_models[int(pick) - 1]
            elif pick:
                model_name = pick
        else:
            model_name = input("모델 이름 입력 (예: gpt-4, llama-3-70b): ").strip()

        context_length_str = input("컨텍스트 길이(토큰 수) [비워두면 자동 감지]: ").strip()

        # Prompt for a display name — shown in the provider menu on future runs
        default_name = _auto_provider_name(effective_url)
        display_name = input(f"표시 이름 [{default_name}]: ").strip() or default_name
    except (KeyboardInterrupt, EOFError):
        print("\n취소했어요.")
        return

    context_length = None
    if context_length_str:
        try:
            context_length = int(context_length_str.replace(",", "").replace("k", "000").replace("K", "000"))
            if context_length <= 0:
                context_length = None
        except ValueError:
            print(f"잘못된 컨텍스트 길이예요: {context_length_str} — 자동 감지로 진행할게요.")
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

        print(f"기본 모델을 설정했어요: {model_name} ({effective_url} 사용)")
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
        print("Endpoint를 저장했어요. 모델은 채팅에서 `/model` 또는 `hermes model`로 설정할 수 있어요.")

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
    print(f"  💾 custom providers에 \"{name}\" 이름으로 저장했어요 (config.yaml에서 수정 가능)")


def _remove_custom_provider(config):
    """Let the user remove a saved custom provider from config.yaml."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    providers = cfg.get("custom_providers") or []
    if not isinstance(providers, list) or not providers:
        print("설정된 custom provider가 없어요.")
        return

    print("custom provider 제거:\n")

    choices = []
    for entry in providers:
        if isinstance(entry, dict):
            name = entry.get("name", "unnamed")
            url = entry.get("base_url", "")
            short_url = url.replace("https://", "").replace("http://", "").rstrip("/")
            choices.append(f"{name} ({short_url})")
        else:
            choices.append(str(entry))
    choices.append("취소")

    try:
        from simple_term_menu import TerminalMenu
        menu = TerminalMenu(
            [f"  {c}" for c in choices], cursor_index=0,
            menu_cursor="-> ", menu_cursor_style=("fg_red", "bold"),
            menu_highlight_style=("fg_red",),
            cycle_cursor=True, clear_screen=False,
            title="제거할 provider 선택:",
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
            val = input(f"선택 [1-{len(choices)}]: ").strip()
            idx = int(val) - 1 if val else None
        except (ValueError, KeyboardInterrupt, EOFError):
            idx = None

    if idx is None or idx >= len(providers):
        print("변경 사항이 없어요.")
        return

    removed = providers.pop(idx)
    cfg["custom_providers"] = providers
    save_config(cfg)
    removed_name = removed.get("name", "unnamed") if isinstance(removed, dict) else str(removed)
    print(f"✅ custom providers에서 \"{removed_name}\" 항목을 제거했어요.")


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
        print(f"  현재 모델: {saved_model}")
    print()

    print("사용 가능한 모델을 불러오는 중...")
    models = fetch_api_models(api_key, base_url, timeout=8.0)

    if models:
        default_idx = 0
        if saved_model and saved_model in models:
            default_idx = models.index(saved_model)

        print(f"모델 {len(models)}개를 찾았어요:\n")
        try:
            from simple_term_menu import TerminalMenu
            menu_items = [
                f"  {m} (current)" if m == saved_model else f"  {m}"
                for m in models
            ] + ["  취소"]
            menu = TerminalMenu(
                menu_items, cursor_index=default_idx,
                menu_cursor="-> ", menu_cursor_style=("fg_green", "bold"),
                menu_highlight_style=("fg_green",),
                cycle_cursor=True, clear_screen=False,
                title=f"{name}에서 사용할 모델 선택:",
            )
            idx = menu.show()
            from hermes_cli.curses_ui import flush_stdin
            flush_stdin()
            print()
            if idx is None or idx >= len(models):
                print("취소했어요.")
                return
            model_name = models[idx]
        except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
            for i, m in enumerate(models, 1):
                suffix = " (current)" if m == saved_model else ""
                print(f"  {i}. {m}{suffix}")
            print(f"  {len(models) + 1}. 취소")
            print()
            try:
                val = input(f"선택 [1-{len(models) + 1}]: ").strip()
                if not val:
                    print("취소했어요.")
                    return
                idx = int(val) - 1
                if idx < 0 or idx >= len(models):
                    print("취소했어요.")
                    return
                model_name = models[idx]
            except (ValueError, KeyboardInterrupt, EOFError):
                print("\n취소했어요.")
                return
    elif saved_model:
        print("endpoint에서 모델 목록을 가져오지 못했어요.")
        try:
            model_name = input(f"모델 이름 [{saved_model}]: ").strip() or saved_model
        except (KeyboardInterrupt, EOFError):
            print("\n취소했어요.")
            return
    else:
        print("endpoint에서 모델 목록을 가져오지 못했어요. 모델 이름을 직접 입력해 주세요.")
        try:
            model_name = input("모델 이름: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n취소했어요.")
            return
        if not model_name:
            print("모델 이름이 없어 취소했어요.")
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

    print(f"\n✅ 모델을 설정했어요: {model_name}")
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
        print("  지원하는 토큰 종류:")
        print("    → OAuth 토큰 (gho_*)              `copilot login` 또는 device code 흐름으로 발급")
        print("    → Fine-grained PAT (github_pat_*)  Copilot Requests 권한 필요")
        print("    → GitHub App 토큰 (ghu_*)         환경 변수로 제공")
        print("    ✗ Classic PAT (ghp_*)             Copilot API에서 지원하지 않음")
        print()
        print("  선택 사항:")
        print("    1. GitHub로 로그인 (OAuth device code 흐름)")
        print("    2. 토큰 직접 입력")
        print("    3. 취소")
        print()
        try:
            choice = input("  선택 [1-3]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if choice == "1":
            try:
                from hermes_cli.copilot_auth import copilot_device_code_login
                token = copilot_device_code_login()
                if token:
                    save_env_value("COPILOT_GITHUB_TOKEN", token)
                    print("  Copilot 토큰을 저장했어요.")
                    print()
                else:
                    print("  로그인을 취소했거나 로그인에 실패했어요.")
                    return
            except Exception as exc:
                print(f"  로그인에 실패했어요: {exc}")
                return
        elif choice == "2":
            try:
                import getpass
                new_key = getpass.getpass("  Token (COPILOT_GITHUB_TOKEN): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not new_key:
                print("  취소했어요.")
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
            print("  토큰을 저장했어요.")
            print()
        else:
            print("  취소했어요.")
            return

        creds = resolve_api_key_provider_credentials(provider_id)
        api_key = creds.get("api_key", "")
        source = creds.get("source", "")
    else:
        if source in ("GITHUB_TOKEN", "GH_TOKEN"):
            print(f"  GitHub 토큰: {api_key[:8]}... ✓ ({source})")
        elif source == "gh auth token":
            print("  GitHub 토큰: ✓ (`gh auth token`에서 가져옴)")
        else:
            print("  GitHub 토큰: ✓")
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
        print(f"  모델 {len(model_list)}개를 GitHub Copilot에서 찾았어요")
    else:
        model_list = _PROVIDER_MODELS.get(provider_id, [])
        if model_list:
            print("  ⚠ GitHub Copilot에서 모델을 자동 감지하지 못했어요 — 기본 목록을 보여줄게요.")
            print('    원하는 모델이 없으면 "사용자 지정 모델 이름 입력"을 사용하세요.')

    if model_list:
        selected = _prompt_model_selection(model_list, current_model=normalized_current_model)
    else:
        try:
            selected = input("모델 이름: ").strip()
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
            print(f"  {selected} 모델은 reasoning 제어를 지원해요.")
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

        print(f"기본 모델을 설정했어요: {selected} ({pconfig.name} 사용)")
        if reasoning_efforts:
            if selected_effort == "none":
                print("이 모델에서는 reasoning을 비활성화했어요.")
            elif selected_effort:
                print(f"reasoning 강도를 설정했어요: {selected_effort}")
    else:
        print("변경 사항이 없어요.")


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

    print("  GitHub Copilot ACP는 Hermes의 각 요청을 `copilot --acp`로 위임해요.")
    print("  현재 Hermes는 요청마다 자체 ACP 서브프로세스를 시작해요.")
    print("  Hermes는 선택한 모델을 Copilot ACP 세션의 힌트로 전달해요.")
    print(f"  명령어: {resolved_command}")
    print(f"  백엔드 표시값: {effective_base}")
    print()

    try:
        creds = resolve_external_process_provider_credentials(provider_id)
    except Exception as exc:
        print(f"  ⚠ {exc}")
        print("  Copilot CLI가 다른 위치에 설치되어 있다면 HERMES_COPILOT_ACP_COMMAND 또는 COPILOT_CLI_PATH를 설정하세요.")
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
        print(f"  모델 {len(model_list)}개를 GitHub Copilot에서 찾았어요")
    else:
        model_list = _PROVIDER_MODELS.get("copilot", [])
        if model_list:
            print("  ⚠ GitHub Copilot에서 모델을 자동 감지하지 못했어요 — 기본 목록을 보여줄게요.")
            print('    원하는 모델이 없으면 "사용자 지정 모델 이름 입력"을 사용하세요.')

    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=normalized_current_model,
        )
    else:
        try:
            selected = input("모델 이름: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if not selected:
        print("변경 사항이 없어요.")
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

    print(f"기본 모델을 설정했어요: {selected} ({pconfig.name} 사용)")


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
        print(f"{pconfig.name} API key가 설정되지 않았어요.")
        if key_env:
            try:
                import getpass
                new_key = getpass.getpass(f"{key_env} 입력(취소하려면 Enter): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not new_key:
                print("취소했어요.")
                return
            save_env_value(key_env, new_key)
            existing_key = new_key
            print("API key를 저장했어요.")
            print()
    else:
        print(f"  {pconfig.name} API key: {existing_key[:8]}... ✓")
        print()

    # Step 2: Auto-detect endpoint from key prefix
    is_coding_plan = existing_key.startswith("sk-kimi-")
    if is_coding_plan:
        effective_base = KIMI_CODE_BASE_URL
        print(f"  Kimi Coding Plan 키를 감지했어요 → {effective_base}")
    else:
        effective_base = pconfig.inference_base_url
        print(f"  Moonshot endpoint를 사용할게요 → {effective_base}")
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
            selected = input("모델 이름 직접 입력: ").strip()
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
        print(f"기본 모델을 설정했어요: {selected} ({endpoint_label} 사용)")
    else:
        print("변경 사항이 없어요.")


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
        print(f"{pconfig.name} API key가 설정되지 않았어요.")
        if key_env:
            try:
                import getpass
                new_key = getpass.getpass(f"{key_env} 입력(취소하려면 Enter): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not new_key:
                print("취소했어요.")
                return
            save_env_value(key_env, new_key)
            print("API key를 저장했어요.")
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
            print("  잘못된 URL이에요 — http:// 또는 https:// 로 시작해야 해요. 현재 값을 유지할게요.")
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
        print(f"  models.dev 레지스트리에서 모델 {len(model_list)}개를 찾았어요")
    elif curated and len(curated) >= 8:
        # Curated list is substantial — use it directly, skip live probe
        model_list = curated
        print(f"  큐레이션 모델 {len(model_list)}개를 표시합니다 — 다른 모델은 \"사용자 지정 모델 이름 입력\"을 사용하세요.")
    else:
        api_key_for_probe = existing_key or (get_env_value(key_env) if key_env else "")
        live_models = fetch_api_models(api_key_for_probe, effective_base)
        if live_models and len(live_models) >= len(curated):
            model_list = live_models
            print(f"  {pconfig.name} API에서 모델 {len(model_list)}개를 찾았어요")
        else:
            model_list = curated
            if model_list:
                print(f"  큐레이션 모델 {len(model_list)}개를 표시합니다 — 다른 모델은 \"사용자 지정 모델 이름 입력\"을 사용하세요.")
        # else: no defaults either, will fall through to raw input

    if provider_id in {"opencode-zen", "opencode-go"}:
        model_list = [normalize_opencode_model_id(provider_id, mid) for mid in model_list]
        current_model = normalize_opencode_model_id(provider_id, current_model)
        model_list = list(dict.fromkeys(mid for mid in model_list if mid))

    if model_list:
        selected = _prompt_model_selection(model_list, current_model=current_model)
    else:
        try:
            selected = input("모델 이름: ").strip()
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

        print(f"기본 모델을 설정했어요: {selected} ({pconfig.name} 사용)")
    else:
        print("변경 사항이 없어요.")


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
            print("  ✓ Claude Code 자격 증명을 연결했어요.")
            from hermes_constants import display_hermes_home as _dhh_fn
            print(f"    setup-token을 {_dhh_fn()}/.env에 복사하지 않고 Claude의 자격 증명 저장소를 직접 사용할게요.")
            return True
        return False

    try:
        print()
        print("  'claude setup-token'을 실행할게요 — 아래 안내를 따라 진행해 주세요.")
        print("  인증을 위해 브라우저 창이 열릴 거예요.")
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
        print("  위에 setup-token이 표시되었다면 여기에 붙여 넣어 주세요:")
        print()
        try:
            import getpass
            manual_token = getpass.getpass("  setup-token 붙여넣기 (취소하려면 Enter): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return False
        if manual_token:
            save_anthropic_oauth_token(manual_token, save_fn=save_env_value)
            print("  ✓ setup-token을 저장했어요.")
            return True

        print("  ⚠ 저장된 자격 증명을 감지하지 못했어요.")
        return False

    except FileNotFoundError:
        # Claude CLI not installed — guide user through manual setup
        print()
        print("  OAuth 로그인에는 'claude' CLI가 필요해요.")
        print()
        print("  설치 및 인증 방법:")
        print()
        print("    1. Claude Code 설치:  npm install -g @anthropic-ai/claude-code")
        print("    2. 실행:              claude setup-token")
        print("    3. 브라우저에서 인증 프롬프트 진행")
        print("    4. 다시 실행:         hermes model")
        print()
        print("  또는 기존 setup-token을 지금 붙여 넣으세요 (sk-ant-oat-...):")
        print()
        try:
            import getpass
            token = getpass.getpass("  setup-token 입력 (취소하려면 Enter): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return False
        if token:
            save_anthropic_oauth_token(token, save_fn=save_env_value)
            print("  ✓ setup-token을 저장했어요.")
            return True
        print("  취소했어요 — Claude Code를 설치한 뒤 다시 시도해 주세요.")
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
            print(f"  Anthropic 자격 증명: {existing_key[:12]}... ✓")
        elif cc_available:
            print("  Claude Code 자격 증명: ✓ (자동 감지됨)")
        print()
        print("    1. 기존 자격 증명 사용")
        print("    2. 다시 인증하기 (새 OAuth 로그인)")
        print("    3. 취소")
        print()
        try:
            choice = input("  선택 [1/2/3]: ").strip()
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
        print("  인증 방식을 선택해 주세요:")
        print()
        print("    1. Claude Pro/Max 구독 (OAuth 로그인)")
        print("    2. Anthropic API key (사용한 토큰만큼 과금)")
        print("    3. 취소")
        print()
        try:
            choice = input("  선택 [1/2/3]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if choice == "1":
            if not _run_anthropic_oauth_flow(save_env_value):
                return

        elif choice == "2":
            print()
            print("  API key 발급 링크: https://console.anthropic.com/settings/keys")
            print()
            try:
                import getpass
                api_key = getpass.getpass("  API key 입력 (sk-ant-...): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not api_key:
                print("  취소했어요.")
                return
            save_anthropic_api_key(api_key, save_fn=save_env_value)
            print("  ✓ API key를 저장했어요.")

        else:
            print("  변경 사항이 없어요.")
            return
    print()

    # Model selection
    model_list = _PROVIDER_MODELS.get("anthropic", [])
    if model_list:
        selected = _prompt_model_selection(model_list, current_model=current_model)
    else:
        try:
            selected = input("모델 이름 입력 (예: claude-sonnet-4-20250514): ").strip()
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

        print(f"기본 모델을 설정했어요: {selected} (Anthropic 사용)")
    else:
        print("변경 사항이 없어요.")


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
        print("OpenAI SDK: 설치되지 않음")

    # Show update status (synchronous — acceptable since user asked for version info)
    try:
        from hermes_cli.banner import check_for_updates
        from hermes_cli.config import recommended_update_command
        behind = check_for_updates()
        if behind and behind > 0:
            commits_word = "커밋" if behind == 1 else "커밋"
            print(
                f"업데이트 가능: 최신 버전보다 {behind} {commits_word} 뒤처져 있어요 — "
                f"'{recommended_update_command()}' 실행"
            )
        elif behind == 0:
            print("최신 상태예요")
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
            print("Web UI 프런트엔드가 빌드되지 않았고 npm도 사용할 수 없어요.")
            print("Node.js를 설치한 뒤 다음을 실행하세요: cd web && npm install && npm run build")
        return not fatal
    print("→ Web UI를 빌드하는 중...")
    r1 = subprocess.run([npm, "install", "--silent"], cwd=web_dir, capture_output=True)
    if r1.returncode != 0:
        print(f"  {'✗' if fatal else '⚠'} Web UI npm install에 실패했어요"
              + ("" if fatal else " (hermes web을 사용할 수 없어요)"))
        if fatal:
            print("  수동 실행: cd web && npm install && npm run build")
        return False
    r2 = subprocess.run([npm, "run", "build"], cwd=web_dir, capture_output=True)
    if r2.returncode != 0:
        print(f"  {'✗' if fatal else '⚠'} Web UI 빌드에 실패했어요"
              + ("" if fatal else " (hermes web을 사용할 수 없어요)"))
        if fatal:
            print("  수동 실행: cd web && npm install && npm run build")
        return False
    print("  ✓ Web UI 빌드 완료")
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
    
    print("→ 최신 버전을 다운로드하는 중...")
    try:
        tmp_dir = tempfile.mkdtemp(prefix="hermes-update-")
        zip_path = os.path.join(tmp_dir, f"hermes-agent-{branch}.zip")
        urlretrieve(zip_url, zip_path)
        
        print("→ 압축을 푸는 중...")
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
        
        print(f"✓ ZIP에서 {update_count}개 항목을 업데이트했어요")
        
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"✗ ZIP 업데이트에 실패했어요: {e}")
        sys.exit(1)

    # Clear stale bytecode after ZIP extraction
    removed = _clear_bytecode_cache(PROJECT_ROOT)
    if removed:
        print(f"  ✓ 오래된 __pycache__ 디렉터리 {removed}개를 정리했어요")
    
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
                                    # Verify the service actually survived the
                                    # restart.  systemctl restart returns 0 even
                                    # if the new process crashes immediately.
                                    import time as _time
                                    _time.sleep(3)
                                    verify = subprocess.run(
                                        scope_cmd + ["is-active", svc_name],
                                        capture_output=True, text=True, timeout=5,
                                    )
                                    if verify.stdout.strip() == "active":
                                        restarted_services.append(svc_name)
                                    else:
                                        # Retry once — transient startup failures
                                        # (stale module cache, import race) often
                                        # resolve on the second attempt.
                                        print(f"  ⚠ {svc_name} died after restart, retrying...")
                                        retry = subprocess.run(
                                            scope_cmd + ["restart", svc_name],
                                            capture_output=True, text=True, timeout=15,
                                        )
                                        _time.sleep(3)
                                        verify2 = subprocess.run(
                                            scope_cmd + ["is-active", svc_name],
                                            capture_output=True, text=True, timeout=5,
                                        )
                                        if verify2.stdout.strip() == "active":
                                            restarted_services.append(svc_name)
                                            print(f"  ✓ {svc_name} recovered on retry")
                                        else:
                                            print(
                                                f"  ✗ {svc_name} failed to stay running after restart.\n"
                                                f"    Check logs: journalctl --user -u {svc_name} --since '2 min ago'\n"
                                                f"    Restart manually: systemctl {'--user ' if scope == 'user' else ''}restart {svc_name}"
                                            )
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
    parser = KoreanArgumentParser(
        prog="hermes",
        description="Hermes Agent - 도구 호출 기능을 갖춘 AI 어시스턴트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    hermes                        대화형 채팅 시작
    hermes chat -q \"안녕\"          단일 질의 모드
    hermes -c                     가장 최근 세션 이어서 열기
    hermes -c \"my project\"        이름으로 세션 이어서 열기(계보 기준 최신)
    hermes --resume <session_id>  세션 ID로 특정 세션 이어서 열기
    hermes setup                  설정 마법사 실행
    hermes logout                 저장된 인증 정보 지우기
    hermes auth add <provider>    풀 자격 증명 추가
    hermes auth list              풀 자격 증명 목록 표시
    hermes auth remove <p> <t>    인덱스, ID 또는 라벨로 풀 자격 증명 제거
    hermes auth reset <provider>  provider의 소진 상태 초기화
    hermes model                  기본 모델 선택
    hermes config                 현재 설정 표시
    hermes config edit            $EDITOR로 설정 파일 편집
    hermes config set model gpt-4 설정값 저장
    hermes gateway                메시징 게이트웨이 실행
    hermes -s hermes-agent-dev,github-auth
    hermes -w                     격리된 git worktree에서 시작
    hermes gateway install        게이트웨이 백그라운드 서비스 설치
    hermes sessions list          지난 세션 목록 표시
    hermes sessions browse        대화형 세션 선택기 실행
    hermes sessions rename ID T   세션 이름/제목 변경
    hermes logs                   agent.log 보기(최근 50줄)
    hermes logs -f                agent.log 실시간 따라가기
    hermes logs errors            errors.log 보기
    hermes logs --since 1h        최근 1시간 로그만 표시
    hermes debug share            지원용 디버그 리포트 업로드
    hermes update                 최신 버전으로 업데이트

명령어별 자세한 도움말:
    hermes <command> --help
"""
    )
    
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="버전을 표시하고 종료"
    )
    parser.add_argument(
        "--resume", "-r",
        metavar="SESSION",
        default=None,
        help="ID 또는 제목으로 이전 세션 이어서 열기"
    )
    parser.add_argument(
        "--continue", "-c",
        dest="continue_last",
        nargs="?",
        const=True,
        default=None,
        metavar="SESSION_NAME",
        help="이름으로 세션을 이어서 열거나, 이름이 없으면 가장 최근 세션 열기"
    )
    parser.add_argument(
        "--worktree", "-w",
        action="store_true",
        default=False,
        help="격리된 git worktree에서 실행(병렬 에이전트용)"
    )
    parser.add_argument(
        "--skills", "-s",
        action="append",
        default=None,
        help="세션 시작 전에 스킬을 하나 이상 미리 로드(플래그 반복 또는 쉼표 구분)"
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        default=False,
        help="위험 명령 승인 프롬프트를 모두 건너뜀(주의해서 사용)"
    )
    parser.add_argument(
        "--pass-session-id",
        action="store_true",
        default=False,
        help="에이전트 시스템 프롬프트에 세션 ID 포함"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="실행할 명령어")
    
    # =========================================================================
    # chat command
    # =========================================================================
    chat_parser = subparsers.add_parser(
        "chat",
        help="에이전트와 대화형 채팅 실행",
        description="Hermes Agent와의 대화형 채팅 세션 시작"
    )
    chat_parser.add_argument(
        "-q", "--query",
        help="단일 질의 실행(비대화형 모드)"
    )
    chat_parser.add_argument(
        "--image",
        help="단일 질의에 첨부할 로컬 이미지 경로(선택)"
    )
    chat_parser.add_argument(
        "-m", "--model",
        help="사용할 모델(예: anthropic/claude-sonnet-4)"
    )
    chat_parser.add_argument(
        "-t", "--toolsets",
        help="활성화할 toolset 목록(쉼표 구분)"
    )
    chat_parser.add_argument(
        "-s", "--skills",
        action="append",
        default=argparse.SUPPRESS,
        help="세션 시작 전에 스킬을 하나 이상 미리 로드(플래그 반복 또는 쉼표 구분)"
    )
    chat_parser.add_argument(
        "--provider",
        choices=["auto", "openrouter", "nous", "openai-codex", "copilot-acp", "copilot", "anthropic", "gemini", "huggingface", "zai", "kimi-coding", "kimi-coding-cn", "minimax", "minimax-cn", "kilocode", "xiaomi", "arcee"],
        default=None,
        help="추론 provider(기본값: auto)"
    )
    chat_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 출력"
    )
    chat_parser.add_argument(
        "-Q", "--quiet",
        action="store_true",
        help="프로그램용 조용한 모드: 배너, 스피너, 도구 미리보기를 숨기고 최종 응답과 세션 정보만 출력"
    )
    chat_parser.add_argument(
        "--resume", "-r",
        metavar="SESSION_ID",
        default=argparse.SUPPRESS,
        help="이전 세션을 ID로 이어서 열기(종료 시 표시됨)"
    )
    chat_parser.add_argument(
        "--continue", "-c",
        dest="continue_last",
        nargs="?",
        const=True,
        default=argparse.SUPPRESS,
        metavar="SESSION_NAME",
        help="이름으로 세션을 이어서 열고, 이름이 없으면 가장 최근 세션 열기"
    )
    chat_parser.add_argument(
        "--worktree", "-w",
        action="store_true",
        default=argparse.SUPPRESS,
        help="격리된 git worktree에서 실행(같은 repo에서 병렬 에이전트용)"
    )
    chat_parser.add_argument(
        "--checkpoints",
        action="store_true",
        default=False,
        help="파괴적 파일 작업 전에 파일시스템 체크포인트 활성화(/rollback으로 복원 가능)"
    )
    chat_parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        metavar="N",
        help="대화 턴당 최대 도구 호출 반복 횟수(기본값: 90, 또는 config의 agent.max_turns)"
    )
    chat_parser.add_argument(
        "--yolo",
        action="store_true",
        default=argparse.SUPPRESS,
        help="위험 명령 승인 프롬프트를 모두 우회(사용자 책임)"
    )
    chat_parser.add_argument(
        "--pass-session-id",
        action="store_true",
        default=argparse.SUPPRESS,
        help="에이전트 시스템 프롬프트에 세션 ID 포함"
    )
    chat_parser.add_argument(
        "--source",
        default=None,
        help="필터링용 세션 source 태그(기본값: cli). 사용자 세션 목록에 보이면 안 되는 서드파티 연동은 'tool' 사용"
    )
    chat_parser.set_defaults(func=cmd_chat)

    # =========================================================================
    # model command
    # =========================================================================
    model_parser = subparsers.add_parser(
        "model",
        help="기본 모델과 provider 선택",
        description="추론 provider와 기본 모델을 대화형으로 선택"
    )
    model_parser.add_argument(
        "--portal-url",
        help="Nous 로그인용 portal 기본 URL(기본값: 운영 portal)"
    )
    model_parser.add_argument(
        "--inference-url",
        help="Nous 로그인용 inference API 기본 URL(기본값: 운영 inference API)"
    )
    model_parser.add_argument(
        "--client-id",
        default=None,
        help="Nous 로그인에 사용할 OAuth client id(기본값: hermes-cli)"
    )
    model_parser.add_argument(
        "--scope",
        default=None,
        help="Nous 로그인에서 요청할 OAuth scope"
    )
    model_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Nous 로그인 중 브라우저를 자동으로 열지 않음"
    )
    model_parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Nous 로그인용 HTTP 요청 타임아웃(초, 기본값: 15)"
    )
    model_parser.add_argument(
        "--ca-bundle",
        help="Nous TLS 검증용 CA 번들 PEM 파일 경로"
    )
    model_parser.add_argument(
        "--insecure",
        action="store_true",
        help="Nous 로그인용 TLS 검증 비활성화(테스트 전용)"
    )
    model_parser.set_defaults(func=cmd_model)

    # =========================================================================
    # gateway command
    # =========================================================================
    gateway_parser = subparsers.add_parser(
        "gateway",
        help="메시징 게이트웨이 관리",
        description="메시징 게이트웨이(Telegram, Discord, WhatsApp) 관리"
    )
    gateway_subparsers = gateway_parser.add_subparsers(dest="gateway_command")
    
    # gateway run (default)
    gateway_run = gateway_subparsers.add_parser("run", help="게이트웨이를 포그라운드에서 실행(WSL, Docker, Termux 권장)")
    gateway_run.add_argument("-v", "--verbose", action="count", default=0,
                             help="stderr 로그 상세도 증가(-v=INFO, -vv=DEBUG)")
    gateway_run.add_argument("-q", "--quiet", action="store_true",
                             help="stderr 로그 출력을 모두 숨김")
    gateway_run.add_argument("--replace", action="store_true",
                             help="기존 게이트웨이 인스턴스가 있으면 교체(systemd에서 유용)")
    
    # gateway start
    gateway_start = gateway_subparsers.add_parser("start", help="설치된 systemd/launchd 백그라운드 서비스 시작")
    gateway_start.add_argument("--system", action="store_true", help="Linux 시스템 레벨 게이트웨이 서비스 대상")
    
    # gateway stop
    gateway_stop = gateway_subparsers.add_parser("stop", help="게이트웨이 서비스 중지")
    gateway_stop.add_argument("--system", action="store_true", help="Linux 시스템 레벨 게이트웨이 서비스 대상")
    gateway_stop.add_argument("--all", action="store_true", help="모든 프로필의 게이트웨이 프로세스를 전부 중지")
    
    # gateway restart
    gateway_restart = gateway_subparsers.add_parser("restart", help="게이트웨이 서비스 재시작")
    gateway_restart.add_argument("--system", action="store_true", help="Linux 시스템 레벨 게이트웨이 서비스 대상")
    
    # gateway status
    gateway_status = gateway_subparsers.add_parser("status", help="게이트웨이 상태 표시")
    gateway_status.add_argument("--deep", action="store_true", help="심층 상태 점검")
    gateway_status.add_argument("--system", action="store_true", help="Linux 시스템 레벨 게이트웨이 서비스 대상")
    
    # gateway install
    gateway_install = gateway_subparsers.add_parser("install", help="게이트웨이를 systemd/launchd 백그라운드 서비스로 설치")
    gateway_install.add_argument("--force", action="store_true", help="강제로 재설치")
    gateway_install.add_argument("--system", action="store_true", help="Linux 시스템 레벨 서비스로 설치(부팅 시 시작)")
    gateway_install.add_argument("--run-as-user", dest="run_as_user", help="Linux 시스템 서비스가 실행될 사용자 계정")
    
    # gateway uninstall
    gateway_uninstall = gateway_subparsers.add_parser("uninstall", help="게이트웨이 서비스 제거")
    gateway_uninstall.add_argument("--system", action="store_true", help="Linux 시스템 레벨 게이트웨이 서비스 대상")

    # gateway setup
    gateway_subparsers.add_parser("setup", help="메시징 플랫폼 설정")

    gateway_parser.set_defaults(func=cmd_gateway)
    
    # =========================================================================
    # setup command
    # =========================================================================
    setup_parser = subparsers.add_parser(
        "setup",
        help="대화형 설정 마법사",
        description="대화형 마법사로 Hermes Agent를 설정. "
                    "특정 섹션만 실행하려면: hermes setup model|tts|terminal|gateway|tools|agent"
    )
    setup_parser.add_argument(
        "section",
        nargs="?",
        choices=["model", "tts", "terminal", "gateway", "tools", "agent"],
        default=None,
        help="전체 마법사 대신 특정 설정 섹션만 실행"
    )
    setup_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="비대화형 모드(기본값/env var 사용)"
    )
    setup_parser.add_argument(
        "--reset",
        action="store_true",
        help="설정을 기본값으로 초기화"
    )
    setup_parser.set_defaults(func=cmd_setup)

    # =========================================================================
    # whatsapp command
    # =========================================================================
    whatsapp_parser = subparsers.add_parser(
        "whatsapp",
        help="WhatsApp 연동 설정",
        description="WhatsApp을 설정하고 QR 코드로 페어링"
    )
    whatsapp_parser.set_defaults(func=cmd_whatsapp)

    # =========================================================================
    # login command
    # =========================================================================
    login_parser = subparsers.add_parser(
        "login",
        help="추론 provider 인증",
        description="Hermes CLI용 OAuth 디바이스 인증 흐름 실행"
    )
    login_parser.add_argument(
        "--provider",
        choices=["nous", "openai-codex"],
        default=None,
        help="인증할 provider(기본값: nous)"
    )
    login_parser.add_argument(
        "--portal-url",
        help="Portal 기본 URL(기본값: 운영 portal)"
    )
    login_parser.add_argument(
        "--inference-url",
        help="추론 API 기본 URL(기본값: 운영 inference API)"
    )
    login_parser.add_argument(
        "--client-id",
        default=None,
        help="사용할 OAuth client id(기본값: hermes-cli)"
    )
    login_parser.add_argument(
        "--scope",
        default=None,
        help="요청할 OAuth scope"
    )
    login_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="브라우저를 자동으로 열지 않음"
    )
    login_parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP 요청 타임아웃(초, 기본값: 15)"
    )
    login_parser.add_argument(
        "--ca-bundle",
        help="TLS 검증용 CA 번들 PEM 파일 경로"
    )
    login_parser.add_argument(
        "--insecure",
        action="store_true",
        help="TLS 검증 비활성화(테스트 전용)"
    )
    login_parser.set_defaults(func=cmd_login)

    # =========================================================================
    # logout command
    # =========================================================================
    logout_parser = subparsers.add_parser(
        "logout",
        help="추론 provider 인증 해제",
        description="저장된 자격 증명을 제거하고 provider 설정 초기화"
    )
    logout_parser.add_argument(
        "--provider",
        choices=["nous", "openai-codex"],
        default=None,
        help="로그아웃할 provider(기본값: 현재 활성 provider)"
    )
    logout_parser.set_defaults(func=cmd_logout)

    auth_parser = subparsers.add_parser(
        "auth",
        help="provider 풀 자격 증명 관리",
    )
    auth_subparsers = auth_parser.add_subparsers(dest="auth_action")
    auth_add = auth_subparsers.add_parser("add", help="풀 자격 증명 추가")
    auth_add.add_argument("provider", help="Provider id(예: anthropic, openai-codex, openrouter)")
    auth_add.add_argument("--type", dest="auth_type", choices=["oauth", "api-key", "api_key"], help="추가할 자격 증명 유형")
    auth_add.add_argument("--label", help="선택 표시 라벨")
    auth_add.add_argument("--api-key", help="API key 값(없으면 안전하게 프롬프트 표시)")
    auth_add.add_argument("--portal-url", help="Nous portal 기본 URL")
    auth_add.add_argument("--inference-url", help="Nous inference 기본 URL")
    auth_add.add_argument("--client-id", help="OAuth client id")
    auth_add.add_argument("--scope", help="OAuth scope 덮어쓰기")
    auth_add.add_argument("--no-browser", action="store_true", help="OAuth 로그인 때 브라우저를 자동으로 열지 않음")
    auth_add.add_argument("--timeout", type=float, help="OAuth/네트워크 타임아웃(초)")
    auth_add.add_argument("--insecure", action="store_true", help="OAuth 로그인용 TLS 검증 비활성화")
    auth_add.add_argument("--ca-bundle", help="OAuth 로그인용 사용자 지정 CA 번들")
    auth_list = auth_subparsers.add_parser("list", help="풀 자격 증명 목록 표시")
    auth_list.add_argument("provider", nargs="?", help="선택 provider 필터")
    auth_remove = auth_subparsers.add_parser("remove", help="인덱스, id, 라벨로 풀 자격 증명 제거")
    auth_remove.add_argument("provider", help="Provider id")
    auth_remove.add_argument("target", help="자격 증명 인덱스, 항목 id, 또는 정확한 라벨")
    auth_reset = auth_subparsers.add_parser("reset", help="provider의 모든 자격 증명 소진 상태 초기화")
    auth_reset.add_argument("provider", help="Provider id")
    auth_parser.set_defaults(func=cmd_auth)

    # =========================================================================
    # status command
    # =========================================================================
    status_parser = subparsers.add_parser(
        "status",
        help="모든 구성요소 상태 표시",
        description="Hermes Agent 구성요소 상태 표시"
    )
    status_parser.add_argument(
        "--all",
        action="store_true",
        help="모든 세부정보 표시(공유용으로 일부 가림)"
    )
    status_parser.add_argument(
        "--deep",
        action="store_true",
        help="심층 점검 실행(시간이 더 걸릴 수 있음)"
    )
    status_parser.set_defaults(func=cmd_status)
    
    # =========================================================================
    # cron command
    # =========================================================================
    cron_parser = subparsers.add_parser(
        "cron",
        help="크론 작업 관리",
        description="예약 작업 관리"
    )
    cron_subparsers = cron_parser.add_subparsers(dest="cron_command")
    
    # cron list
    cron_list = cron_subparsers.add_parser("list", help="예약 작업 목록 표시")
    cron_list.add_argument("--all", action="store_true", help="비활성화된 작업도 포함")

    # cron create/add
    cron_create = cron_subparsers.add_parser("create", aliases=["add"], help="예약 작업 생성")
    cron_create.add_argument("schedule", help="예: '30m', 'every 2h', '0 9 * * *' 형식의 일정")
    cron_create.add_argument("prompt", nargs="?", help="선택: 독립 실행형 프롬프트 또는 작업 지시")
    cron_create.add_argument("--name", help="선택: 사람이 읽기 쉬운 작업 이름")
    cron_create.add_argument("--deliver", help="전달 대상: origin, local, telegram, discord, signal 또는 platform:chat_id")
    cron_create.add_argument("--repeat", type=int, help="선택: 반복 횟수")
    cron_create.add_argument("--skill", dest="skills", action="append", help="스킬 첨부. 여러 개를 붙이려면 반복 지정")
    cron_create.add_argument("--script", help="실행마다 stdout을 프롬프트에 주입할 Python 스크립트 경로")

    # cron edit
    cron_edit = cron_subparsers.add_parser("edit", help="기존 예약 작업 편집")
    cron_edit.add_argument("job_id", help="편집할 작업 ID")
    cron_edit.add_argument("--schedule", help="새 일정")
    cron_edit.add_argument("--prompt", help="새 프롬프트/작업 지시")
    cron_edit.add_argument("--name", help="새 작업 이름")
    cron_edit.add_argument("--deliver", help="새 전달 대상")
    cron_edit.add_argument("--repeat", type=int, help="새 반복 횟수")
    cron_edit.add_argument("--skill", dest="skills", action="append", help="작업의 스킬 목록을 이 집합으로 교체. 여러 개를 붙이려면 반복 지정")
    cron_edit.add_argument("--add-skill", dest="add_skills", action="append", help="기존 목록을 바꾸지 않고 스킬 추가. 반복 가능")
    cron_edit.add_argument("--remove-skill", dest="remove_skills", action="append", help="첨부된 특정 스킬 제거. 반복 가능")
    cron_edit.add_argument("--clear-skills", action="store_true", help="작업에 붙은 모든 스킬 제거")
    cron_edit.add_argument("--script", help="실행마다 stdout을 프롬프트에 주입할 Python 스크립트 경로. 비우려면 빈 문자열 전달")

    # lifecycle actions
    cron_pause = cron_subparsers.add_parser("pause", help="예약 작업 일시중지")
    cron_pause.add_argument("job_id", help="일시중지할 작업 ID")

    cron_resume = cron_subparsers.add_parser("resume", help="일시중지된 작업 재개")
    cron_resume.add_argument("job_id", help="재개할 작업 ID")

    cron_run = cron_subparsers.add_parser("run", help="다음 스케줄러 tick에서 작업 실행")
    cron_run.add_argument("job_id", help="실행할 작업 ID")

    cron_remove = cron_subparsers.add_parser("remove", aliases=["rm", "delete"], help="예약 작업 제거")
    cron_remove.add_argument("job_id", help="제거할 작업 ID")

    # cron status
    cron_subparsers.add_parser("status", help="크론 스케줄러 실행 상태 확인")

    # cron tick (mostly for debugging)
    cron_subparsers.add_parser("tick", help="실행 시각이 된 작업을 한 번 실행하고 종료")

    cron_parser.set_defaults(func=cmd_cron)

    # =========================================================================
    # webhook command
    # =========================================================================
    webhook_parser = subparsers.add_parser(
        "webhook",
        help="동적 웹훅 구독 관리",
        description="이벤트 기반 에이전트 활성화를 위한 웹훅 구독 생성, 조회, 제거",
    )
    webhook_subparsers = webhook_parser.add_subparsers(dest="webhook_action")

    wh_sub = webhook_subparsers.add_parser("subscribe", aliases=["add"], help="웹훅 구독 생성")
    wh_sub.add_argument("name", help="라우트 이름(URL에서 /webhooks/<name>에 사용)")
    wh_sub.add_argument("--prompt", default="", help="{dot.notation} payload 참조를 포함한 프롬프트 템플릿")
    wh_sub.add_argument("--events", default="", help="허용할 이벤트 유형(쉼표 구분)")
    wh_sub.add_argument("--description", default="", help="이 구독의 용도 설명")
    wh_sub.add_argument("--skills", default="", help="로드할 스킬 이름(쉼표 구분)")
    wh_sub.add_argument("--deliver", default="log", help="전달 대상: log, telegram, discord, slack 등")
    wh_sub.add_argument("--deliver-chat-id", default="", help="크로스플랫폼 전달 대상 chat ID")
    wh_sub.add_argument("--secret", default="", help="HMAC 시크릿(생략 시 자동 생성)")

    webhook_subparsers.add_parser("list", aliases=["ls"], help="동적 구독 전체 목록 표시")

    wh_rm = webhook_subparsers.add_parser("remove", aliases=["rm"], help="구독 제거")
    wh_rm.add_argument("name", help="제거할 구독 이름")

    wh_test = webhook_subparsers.add_parser("test", help="웹훅 라우트로 테스트 POST 전송")
    wh_test.add_argument("name", help="테스트할 구독 이름")
    wh_test.add_argument("--payload", default="", help="전송할 JSON payload(기본값: 테스트 payload)")

    webhook_parser.set_defaults(func=cmd_webhook)

    # =========================================================================
    # doctor command
    # =========================================================================
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="설정과 의존성 점검",
        description="Hermes Agent 설정 문제 진단"
    )
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help="문제를 자동으로 수정 시도"
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    # =========================================================================
    # dump command
    # =========================================================================
    dump_parser = subparsers.add_parser(
        "dump",
        help="지원/디버깅용 설정 요약 출력",
        description="지원 맥락 공유를 위해 Discord/GitHub에 복사해 붙여넣을 수 있는 "
                    "간단한 일반 텍스트 Hermes 설정 요약 출력"
    )
    dump_parser.add_argument(
        "--show-keys",
        action="store_true",
        help="단순 설정 여부 대신 API key 앞/뒤 4글자를 가려서 표시"
    )
    dump_parser.set_defaults(func=cmd_dump)

    # =========================================================================
    # debug command
    # =========================================================================
    debug_parser = subparsers.add_parser(
        "debug",
        help="디버그 도구 — 지원용 로그와 시스템 정보 업로드",
        description="Hermes Agent용 디버그 유틸리티. 'hermes debug share'를 사용해 "
                    "디버그 리포트(시스템 정보 + 최근 로그)를 paste 서비스에 업로드하고 "
                    "공유 가능한 URL을 받을 수 있습니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
예시:
    hermes debug share              디버그 리포트를 업로드하고 URL 출력
    hermes debug share --lines 500  더 많은 로그 줄 포함
    hermes debug share --expire 30  paste를 30일간 유지
    hermes debug share --local      로컬에만 리포트 출력(업로드 안 함)
""",
    )
    debug_sub = debug_parser.add_subparsers(dest="debug_command")
    share_parser = debug_sub.add_parser(
        "share",
        help="디버그 리포트를 paste 서비스에 업로드하고 공유 URL 출력",
    )
    share_parser.add_argument(
        "--lines", type=int, default=200,
        help="로그 파일마다 포함할 줄 수(기본값: 200)",
    )
    share_parser.add_argument(
        "--expire", type=int, default=7,
        help="paste 만료 기간(일, 기본값: 7)",
    )
    share_parser.add_argument(
        "--local", action="store_true",
        help="업로드 대신 로컬에 리포트 출력",
    )
    debug_parser.set_defaults(func=cmd_debug)

    # =========================================================================
    # backup command
    # =========================================================================
    backup_parser = subparsers.add_parser(
        "backup",
        help="Hermes 홈 디렉터리를 zip 파일로 백업",
        description="Hermes 설정, 스킬, 세션, 데이터를 모두 포함한 zip 아카이브 생성 "
                    "(hermes-agent 코드베이스는 제외). "
                    "중요 상태 파일만 빠르게 백업하려면 --quick 사용."
    )
    backup_parser.add_argument(
        "-o", "--output",
        help="zip 파일 출력 경로(기본값: ~/hermes-backup-<timestamp>.zip)"
    )
    backup_parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="빠른 스냅샷: 핵심 상태 파일만 포함(config, state.db, .env, auth, cron)"
    )
    backup_parser.add_argument(
        "-l", "--label",
        help="스냅샷 라벨(--quick 사용 시에만 적용)"
    )
    backup_parser.set_defaults(func=cmd_backup)

    # =========================================================================
    # import command
    # =========================================================================
    import_parser = subparsers.add_parser(
        "import",
        help="zip 파일에서 Hermes 백업 복원",
        description="이전에 만든 Hermes 백업을 Hermes 홈 디렉터리에 풀어 설정, 스킬, 세션, 데이터를 복원"
    )
    import_parser.add_argument(
        "zipfile",
        help="백업 zip 파일 경로"
    )
    import_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="확인 없이 기존 파일 덮어쓰기"
    )
    import_parser.set_defaults(func=cmd_import)

    # =========================================================================
    # config command
    # =========================================================================
    config_parser = subparsers.add_parser(
        "config",
        help="설정 조회 및 편집",
        description="Hermes Agent 설정 관리"
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    
    # config show (default)
    config_subparsers.add_parser("show", help="현재 설정 표시")
    
    # config edit
    config_subparsers.add_parser("edit", help="에디터에서 config 파일 열기")
    
    # config set
    config_set = config_subparsers.add_parser("set", help="설정값 지정")
    config_set.add_argument("key", nargs="?", help="설정 키(예: model, terminal.backend)")
    config_set.add_argument("value", nargs="?", help="설정할 값")
    
    # config path
    config_subparsers.add_parser("path", help="config 파일 경로 출력")
    
    # config env-path
    config_subparsers.add_parser("env-path", help=".env 파일 경로 출력")
    
    # config check
    config_subparsers.add_parser("check", help="누락되었거나 오래된 config 점검")
    
    # config migrate
    config_subparsers.add_parser("migrate", help="새 옵션으로 config 업데이트")
    
    config_parser.set_defaults(func=cmd_config)
    
    # =========================================================================
    # pairing command
    # =========================================================================
    pairing_parser = subparsers.add_parser(
        "pairing",
        help="사용자 인증용 DM 페어링 코드 관리",
        description="페어링 코드로 사용자 접근 승인 또는 철회"
    )
    pairing_sub = pairing_parser.add_subparsers(dest="pairing_action")

    pairing_sub.add_parser("list", help="대기 중 + 승인된 사용자 표시")

    pairing_approve_parser = pairing_sub.add_parser("approve", help="페어링 코드 승인")
    pairing_approve_parser.add_argument("platform", help="플랫폼 이름(telegram, discord, slack, whatsapp)")
    pairing_approve_parser.add_argument("code", help="승인할 페어링 코드")

    pairing_revoke_parser = pairing_sub.add_parser("revoke", help="사용자 접근 권한 철회")
    pairing_revoke_parser.add_argument("platform", help="플랫폼 이름")
    pairing_revoke_parser.add_argument("user_id", help="철회할 사용자 ID")

    pairing_sub.add_parser("clear-pending", help="대기 중인 코드 모두 비우기")

    def cmd_pairing(args):
        from hermes_cli.pairing import pairing_command
        pairing_command(args)

    pairing_parser.set_defaults(func=cmd_pairing)

    # =========================================================================
    # skills command
    # =========================================================================
    skills_parser = subparsers.add_parser(
        "skills",
        help="스킬 검색, 설치, 설정, 관리",
        description="skills.sh, well-known agent skill 엔드포인트, GitHub, ClawHub 등 다양한 레지스트리에서 스킬을 검색, 설치, 점검, 설정, 관리"
    )
    skills_subparsers = skills_parser.add_subparsers(dest="skills_action")

    skills_browse = skills_subparsers.add_parser("browse", help="사용 가능한 스킬 전체 둘러보기(페이지 단위)")
    skills_browse.add_argument("--page", type=int, default=1, help="페이지 번호(기본값: 1)")
    skills_browse.add_argument("--size", type=int, default=20, help="페이지당 결과 수(기본값: 20)")
    skills_browse.add_argument("--source", default="all",
                               choices=["all", "official", "skills-sh", "well-known", "github", "clawhub", "lobehub"],
                               help="source로 필터링(기본값: all)")

    skills_search = skills_subparsers.add_parser("search", help="스킬 레지스트리 검색")
    skills_search.add_argument("query", help="검색어")
    skills_search.add_argument("--source", default="all", choices=["all", "official", "skills-sh", "well-known", "github", "clawhub", "lobehub"])
    skills_search.add_argument("--limit", type=int, default=10, help="최대 결과 수")

    skills_install = skills_subparsers.add_parser("install", help="스킬 설치")
    skills_install.add_argument("identifier", help="스킬 식별자(예: openai/skills/skill-creator)")
    skills_install.add_argument("--category", default="", help="설치할 category 폴더")
    skills_install.add_argument("--force", action="store_true", help="차단 스캔 판정이 있어도 설치")
    skills_install.add_argument("--yes", "-y", action="store_true", help="확인 프롬프트 건너뛰기(TUI 모드에서 필요)")

    skills_inspect = skills_subparsers.add_parser("inspect", help="설치하지 않고 스킬 미리보기")
    skills_inspect.add_argument("identifier", help="스킬 식별자")

    skills_list = skills_subparsers.add_parser("list", help="설치된 스킬 목록 표시")
    skills_list.add_argument("--source", default="all", choices=["all", "hub", "builtin", "local"])

    skills_check = skills_subparsers.add_parser("check", help="설치된 hub 스킬 업데이트 점검")
    skills_check.add_argument("name", nargs="?", help="점검할 특정 스킬(기본값: 전체)")

    skills_update = skills_subparsers.add_parser("update", help="설치된 hub 스킬 업데이트")
    skills_update.add_argument("name", nargs="?", help="업데이트할 특정 스킬(기본값: 오래된 스킬 전체)")

    skills_audit = skills_subparsers.add_parser("audit", help="설치된 hub 스킬 재스캔")
    skills_audit.add_argument("name", nargs="?", help="감사할 특정 스킬(기본값: 전체)")

    skills_uninstall = skills_subparsers.add_parser("uninstall", help="hub 설치 스킬 제거")
    skills_uninstall.add_argument("name", help="제거할 스킬 이름")

    skills_publish = skills_subparsers.add_parser("publish", help="레지스트리에 스킬 게시")
    skills_publish.add_argument("skill_path", help="스킬 디렉터리 경로")
    skills_publish.add_argument("--to", default="github", choices=["github", "clawhub"], help="대상 레지스트리")
    skills_publish.add_argument("--repo", default="", help="대상 GitHub repo(예: openai/skills)")

    skills_snapshot = skills_subparsers.add_parser("snapshot", help="스킬 설정 내보내기/가져오기")
    snapshot_subparsers = skills_snapshot.add_subparsers(dest="snapshot_action")
    snap_export = snapshot_subparsers.add_parser("export", help="설치된 스킬을 파일로 내보내기")
    snap_export.add_argument("output", help="출력 JSON 파일 경로(stdout은 - 사용)")
    snap_import = snapshot_subparsers.add_parser("import", help="파일에서 스킬을 가져와 설치")
    snap_import.add_argument("input", help="입력 JSON 파일 경로")
    snap_import.add_argument("--force", action="store_true", help="주의 판정이 있어도 강제로 설치")

    skills_tap = skills_subparsers.add_parser("tap", help="스킬 source 관리")
    tap_subparsers = skills_tap.add_subparsers(dest="tap_action")
    tap_subparsers.add_parser("list", help="설정된 tap 목록 표시")
    tap_add = tap_subparsers.add_parser("add", help="GitHub repo를 스킬 source로 추가")
    tap_add.add_argument("repo", help="GitHub repo(예: owner/repo)")
    tap_rm = tap_subparsers.add_parser("remove", help="tap 제거")
    tap_rm.add_argument("name", help="제거할 tap 이름")

    # config sub-action: interactive enable/disable
    skills_subparsers.add_parser("config", help="대화형 스킬 설정 — 개별 스킬 활성화/비활성화")

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
        help="플러그인 관리 — 설치, 업데이트, 제거, 목록",
        description="Git 저장소에서 플러그인을 설치하고, 업데이트하거나, 제거하거나, 목록을 확인",
    )
    plugins_subparsers = plugins_parser.add_subparsers(dest="plugins_action")

    plugins_install = plugins_subparsers.add_parser(
        "install", help="Git URL 또는 owner/repo로 플러그인 설치"
    )
    plugins_install.add_argument(
        "identifier",
        help="Git URL 또는 owner/repo 축약형(예: anpicasso/hermes-plugin-chrome-profiles)",
    )
    plugins_install.add_argument(
        "--force", "-f", action="store_true",
        help="기존 플러그인을 제거하고 재설치",
    )

    plugins_update = plugins_subparsers.add_parser(
        "update", help="설치된 플러그인의 최신 변경 사항 가져오기"
    )
    plugins_update.add_argument("name", help="업데이트할 플러그인 이름")

    plugins_remove = plugins_subparsers.add_parser(
        "remove", aliases=["rm", "uninstall"], help="설치된 플러그인 제거"
    )
    plugins_remove.add_argument("name", help="제거할 플러그인 디렉터리 이름")

    plugins_subparsers.add_parser("list", aliases=["ls"], help="설치된 플러그인 목록 표시")

    plugins_enable = plugins_subparsers.add_parser(
        "enable", help="비활성화된 플러그인 활성화"
    )
    plugins_enable.add_argument("name", help="활성화할 플러그인 이름")

    plugins_disable = plugins_subparsers.add_parser(
        "disable", help="플러그인을 제거하지 않고 비활성화"
    )
    plugins_disable.add_argument("name", help="비활성화할 플러그인 이름")

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
        help="외부 memory provider 설정",
        description=(
            "외부 memory provider 플러그인을 설정하고 관리합니다.\n\n"
            "사용 가능한 provider: honcho, openviking, mem0, hindsight,\n"
            "holographic, retaindb, byterover.\n\n"
            "외부 provider는 한 번에 하나만 활성화할 수 있습니다.\n"
            "내장 memory(MEMORY.md/USER.md)는 항상 활성화됩니다."
        ),
    )
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    memory_sub.add_parser("setup", help="대화형 provider 선택 및 설정")
    memory_sub.add_parser("status", help="현재 memory provider 설정 표시")
    memory_sub.add_parser("off", help="외부 provider 비활성화(내장만 사용)")

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
        help="플랫폼별 활성 도구 설정",
        description=(
            "CLI, Telegram, Discord 등 플랫폼별 도구를 활성화, 비활성화, 조회합니다.\n\n"
            "내장 toolset은 일반 이름(예: web, memory)을 사용합니다.\n"
            "MCP 도구는 server:tool 표기(예: github:create_issue)를 사용합니다.\n\n"
            "대화형 설정 UI를 열려면 하위 명령 없이 'hermes tools'를 실행하세요."
        ),
    )
    tools_parser.add_argument(
        "--summary",
        action="store_true",
        help="플랫폼별 활성 도구 요약을 출력하고 종료"
    )
    tools_sub = tools_parser.add_subparsers(dest="tools_action")

    # hermes tools list [--platform cli]
    tools_list_p = tools_sub.add_parser(
        "list",
        help="모든 도구와 활성/비활성 상태 표시",
    )
    tools_list_p.add_argument(
        "--platform", default="cli",
        help="표시할 플랫폼(기본값: cli)",
    )

    # hermes tools disable <name...> [--platform cli]
    tools_disable_p = tools_sub.add_parser(
        "disable",
        help="toolset 또는 MCP 도구 비활성화",
    )
    tools_disable_p.add_argument(
        "names", nargs="+", metavar="NAME",
        help="Toolset 이름(예: web) 또는 server:tool 형식의 MCP 도구",
    )
    tools_disable_p.add_argument(
        "--platform", default="cli",
        help="적용할 플랫폼(기본값: cli)",
    )

    # hermes tools enable <name...> [--platform cli]
    tools_enable_p = tools_sub.add_parser(
        "enable",
        help="toolset 또는 MCP 도구 활성화",
    )
    tools_enable_p.add_argument(
        "names", nargs="+", metavar="NAME",
        help="Toolset 이름 또는 server:tool 형식의 MCP 도구",
    )
    tools_enable_p.add_argument(
        "--platform", default="cli",
        help="적용할 플랫폼(기본값: cli)",
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
        help="MCP 서버 관리 및 Hermes를 MCP 서버로 실행",
        description=(
            "MCP 서버 연결을 관리하고 Hermes를 MCP 서버로 실행합니다.\n\n"
            "MCP 서버는 Model Context Protocol을 통해 추가 도구를 제공합니다.\n"
            "새 서버에 연결하려면 'hermes mcp add'를,\n"
            "Hermes 대화를 MCP로 노출하려면 'hermes mcp serve'를 사용하세요."
        ),
    )
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_action")

    mcp_serve_p = mcp_sub.add_parser(
        "serve",
        help="Hermes를 MCP 서버로 실행(대화를 다른 에이전트에 노출)",
    )
    mcp_serve_p.add_argument(
        "-v", "--verbose", action="store_true",
        help="stderr에 자세한 로그 활성화",
    )

    mcp_add_p = mcp_sub.add_parser("add", help="MCP 서버 추가(discovery-first 설치)")
    mcp_add_p.add_argument("name", help="서버 이름(config 키로 사용)")
    mcp_add_p.add_argument("--url", help="HTTP/SSE 엔드포인트 URL")
    mcp_add_p.add_argument("--command", help="stdio 명령어(예: npx)")
    mcp_add_p.add_argument("--args", nargs="*", default=[], help="stdio 명령어 인자")
    mcp_add_p.add_argument("--auth", choices=["oauth", "header"], help="인증 방식")
    mcp_add_p.add_argument("--preset", help="알려진 MCP preset 이름")
    mcp_add_p.add_argument("--env", nargs="*", default=[], help="stdio 서버용 환경 변수(KEY=VALUE)")

    mcp_rm_p = mcp_sub.add_parser("remove", aliases=["rm"], help="MCP 서버 제거")
    mcp_rm_p.add_argument("name", help="제거할 서버 이름")

    mcp_sub.add_parser("list", aliases=["ls"], help="설정된 MCP 서버 목록 표시")

    mcp_test_p = mcp_sub.add_parser("test", help="MCP 서버 연결 테스트")
    mcp_test_p.add_argument("name", help="테스트할 서버 이름")

    mcp_cfg_p = mcp_sub.add_parser("configure", aliases=["config"], help="도구 선택 토글")
    mcp_cfg_p.add_argument("name", help="설정할 서버 이름")

    def cmd_mcp(args):
        from hermes_cli.mcp_config import mcp_command
        mcp_command(args)

    mcp_parser.set_defaults(func=cmd_mcp)

    # =========================================================================
    # sessions command
    # =========================================================================
    sessions_parser = subparsers.add_parser(
        "sessions",
        help="세션 기록 관리(목록, 이름 변경, 내보내기, 정리, 삭제)",
        description="SQLite 세션 저장소 조회 및 관리"
    )
    sessions_subparsers = sessions_parser.add_subparsers(dest="sessions_action")

    sessions_list = sessions_subparsers.add_parser("list", help="최근 세션 목록 표시")
    sessions_list.add_argument("--source", help="source로 필터링(cli, telegram, discord 등)")
    sessions_list.add_argument("--limit", type=int, default=20, help="표시할 최대 세션 수")

    sessions_export = sessions_subparsers.add_parser("export", help="세션을 JSONL 파일로 내보내기")
    sessions_export.add_argument("output", help="출력 JSONL 파일 경로(stdout은 - 사용)")
    sessions_export.add_argument("--source", help="source로 필터링")
    sessions_export.add_argument("--session-id", help="특정 세션 내보내기")

    sessions_delete = sessions_subparsers.add_parser("delete", help="특정 세션 삭제")
    sessions_delete.add_argument("session_id", help="삭제할 세션 ID")
    sessions_delete.add_argument("--yes", "-y", action="store_true", help="확인 건너뛰기")

    sessions_prune = sessions_subparsers.add_parser("prune", help="오래된 세션 삭제")
    sessions_prune.add_argument("--older-than", type=int, default=90, help="N일보다 오래된 세션 삭제(기본값: 90)")
    sessions_prune.add_argument("--source", help="이 source의 세션만 정리")
    sessions_prune.add_argument("--yes", "-y", action="store_true", help="확인 건너뛰기")

    sessions_subparsers.add_parser("stats", help="세션 저장소 통계 표시")

    sessions_rename = sessions_subparsers.add_parser("rename", help="세션 제목 지정 또는 변경")
    sessions_rename.add_argument("session_id", help="이름을 바꿀 세션 ID")
    sessions_rename.add_argument("title", nargs="+", help="세션의 새 제목")

    sessions_browse = sessions_subparsers.add_parser(
        "browse",
        help="대화형 세션 선택기 — 탐색, 검색, 이어서 열기",
    )
    sessions_browse.add_argument("--source", help="source로 필터링(cli, telegram, discord 등)")
    sessions_browse.add_argument("--limit", type=int, default=50, help="불러올 최대 세션 수(기본값: 50)")

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
                    print("취소했어요.")
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
                    print("취소했어요.")
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
                print("취소했어요.")
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
        help="사용량 인사이트와 분석 표시",
        description="세션 기록을 분석해 토큰 사용량, 비용, 도구 패턴, 활동 추세를 표시"
    )
    insights_parser.add_argument("--days", type=int, default=30, help="분석할 일 수(기본값: 30)")
    insights_parser.add_argument("--source", help="플랫폼으로 필터링(cli, telegram, discord 등)")

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
        help="OpenClaw 마이그레이션 도구",
        description="OpenClaw의 설정, memory, 스킬, API key를 Hermes로 마이그레이션"
    )
    claw_subparsers = claw_parser.add_subparsers(dest="claw_action")

    # claw migrate
    claw_migrate = claw_subparsers.add_parser(
        "migrate",
        help="OpenClaw에서 Hermes로 마이그레이션",
        description="OpenClaw 설치본에서 설정, memory, 스킬, API key를 가져옵니다. "
                    "변경 전에 항상 미리보기를 보여줍니다."
    )
    claw_migrate.add_argument(
        "--source",
        help="OpenClaw 디렉터리 경로(기본값: ~/.openclaw)"
    )
    claw_migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="미리보기만 수행 — 무엇이 마이그레이션될지 보여준 뒤 중지"
    )
    claw_migrate.add_argument(
        "--preset",
        choices=["user-data", "full"],
        default="full",
        help="마이그레이션 preset(기본값: full). 'user-data'는 시크릿 제외"
    )
    claw_migrate.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 파일 덮어쓰기(기본값: 충돌 시 건너뜀)"
    )
    claw_migrate.add_argument(
        "--migrate-secrets",
        action="store_true",
        help="허용 목록 시크릿 포함(TELEGRAM_BOT_TOKEN, API key 등)"
    )
    claw_migrate.add_argument(
        "--workspace-target",
        help="workspace 지침을 복사할 절대 경로"
    )
    claw_migrate.add_argument(
        "--skill-conflict",
        choices=["skip", "overwrite", "rename"],
        default="skip",
        help="스킬 이름 충돌 처리 방식(기본값: skip)"
    )
    claw_migrate.add_argument(
        "--yes", "-y",
        action="store_true",
        help="확인 프롬프트 건너뛰기"
    )

    # claw cleanup
    claw_cleanup = claw_subparsers.add_parser(
        "cleanup",
        aliases=["clean"],
        help="마이그레이션 후 남은 OpenClaw 디렉터리 보관",
        description="상태 분산을 막기 위해 남은 OpenClaw 디렉터리를 스캔하고 보관"
    )
    claw_cleanup.add_argument(
        "--source",
        help="정리할 특정 OpenClaw 디렉터리 경로"
    )
    claw_cleanup.add_argument(
        "--dry-run",
        action="store_true",
        help="변경 없이 무엇이 보관될지 미리보기"
    )
    claw_cleanup.add_argument(
        "--yes", "-y",
        action="store_true",
        help="확인 프롬프트 건너뛰기"
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
        help="버전 정보 표시"
    )
    version_parser.set_defaults(func=cmd_version)
    
    # =========================================================================
    # update command
    # =========================================================================
    update_parser = subparsers.add_parser(
        "update",
        help="Hermes Agent를 최신 버전으로 업데이트",
        description="git에서 최신 변경 사항을 가져오고 의존성을 다시 설치"
    )
    update_parser.add_argument(
        "--gateway", action="store_true", default=False,
        help="게이트웨이 모드: stdin 대신 파일 기반 IPC로 프롬프트를 주고받음(/update 내부 사용)"
    )
    update_parser.set_defaults(func=cmd_update)
    
    # =========================================================================
    # uninstall command
    # =========================================================================
    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="Hermes Agent 제거",
        description="시스템에서 Hermes Agent를 제거합니다. 재설치를 위해 설정/데이터는 유지할 수 있습니다."
    )
    uninstall_parser.add_argument(
        "--full",
        action="store_true",
        help="전체 제거 - 설정과 데이터까지 모두 삭제"
    )
    uninstall_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="확인 프롬프트 건너뛰기"
    )
    uninstall_parser.set_defaults(func=cmd_uninstall)

    # =========================================================================
    # acp command
    # =========================================================================
    acp_parser = subparsers.add_parser(
        "acp",
        help="Hermes Agent를 ACP(Agent Client Protocol) 서버로 실행",
        description="에디터 연동(VS Code, Zed, JetBrains)을 위해 Hermes Agent를 ACP 모드로 시작",
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
        help="프로필 관리 — 여러 격리된 Hermes 인스턴스",
    )
    profile_subparsers = profile_parser.add_subparsers(dest="profile_action")

    profile_subparsers.add_parser("list", help="모든 프로필 목록 표시")
    profile_use = profile_subparsers.add_parser("use", help="고정 기본 프로필 지정")
    profile_use.add_argument("profile_name", help="프로필 이름(또는 'default')")

    profile_create = profile_subparsers.add_parser("create", help="새 프로필 생성")
    profile_create.add_argument("profile_name", help="프로필 이름(소문자, 영숫자)")
    profile_create.add_argument("--clone", action="store_true",
                                help="활성 프로필의 config.yaml, .env, SOUL.md 복사")
    profile_create.add_argument("--clone-all", action="store_true",
                                help="활성 프로필 전체 복사(모든 상태 포함)")
    profile_create.add_argument("--clone-from", metavar="SOURCE",
                                help="복사할 원본 프로필(기본값: 활성 프로필)")
    profile_create.add_argument("--no-alias", action="store_true",
                                help="래퍼 스크립트 생성 건너뛰기")

    profile_delete = profile_subparsers.add_parser("delete", help="프로필 삭제")
    profile_delete.add_argument("profile_name", help="삭제할 프로필")
    profile_delete.add_argument("-y", "--yes", action="store_true",
                                help="확인 프롬프트 건너뛰기")

    profile_show = profile_subparsers.add_parser("show", help="프로필 세부정보 표시")
    profile_show.add_argument("profile_name", help="표시할 프로필")

    profile_alias = profile_subparsers.add_parser("alias", help="래퍼 스크립트 관리")
    profile_alias.add_argument("profile_name", help="프로필 이름")
    profile_alias.add_argument("--remove", action="store_true",
                               help="래퍼 스크립트 제거")
    profile_alias.add_argument("--name", dest="alias_name", metavar="NAME",
                               help="사용자 지정 alias 이름(기본값: 프로필 이름)")

    profile_rename = profile_subparsers.add_parser("rename", help="프로필 이름 변경")
    profile_rename.add_argument("old_name", help="현재 프로필 이름")
    profile_rename.add_argument("new_name", help="새 프로필 이름")

    profile_export = profile_subparsers.add_parser("export", help="프로필을 아카이브로 내보내기")
    profile_export.add_argument("profile_name", help="내보낼 프로필")
    profile_export.add_argument("-o", "--output", default=None,
                                help="출력 파일(기본값: <name>.tar.gz)")

    profile_import = profile_subparsers.add_parser("import", help="아카이브에서 프로필 가져오기")
    profile_import.add_argument("archive", help=".tar.gz 아카이브 경로")
    profile_import.add_argument("--name", dest="import_name", metavar="NAME",
                                help="프로필 이름(기본값: 아카이브에서 추론)")

    profile_parser.set_defaults(func=cmd_profile)

    # =========================================================================
    # completion command
    # =========================================================================
    completion_parser = subparsers.add_parser(
        "completion",
        help="셸 자동완성 스크립트 출력(bash, zsh, fish)",
    )
    completion_parser.add_argument(
        "shell", nargs="?", default="bash", choices=["bash", "zsh", "fish"],
        help="셸 종류(기본값: bash)",
    )
    completion_parser.set_defaults(func=lambda args: cmd_completion(args, parser))

    # =========================================================================
    # dashboard command
    # =========================================================================
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="웹 UI 대시보드 시작",
        description="설정, API key, 세션 관리를 위한 Hermes Agent 웹 대시보드 실행",
    )
    dashboard_parser.add_argument("--port", type=int, default=9119, help="포트(기본값: 9119)")
    dashboard_parser.add_argument("--host", default="127.0.0.1", help="호스트(기본값: 127.0.0.1)")
    dashboard_parser.add_argument("--no-open", action="store_true", help="브라우저를 자동으로 열지 않음")
    dashboard_parser.add_argument(
        "--insecure", action="store_true",
        help="localhost가 아닌 주소 바인딩 허용(위험: 네트워크에 API key 노출 가능)",
    )
    dashboard_parser.set_defaults(func=cmd_dashboard)

    # =========================================================================
    # logs command
    # =========================================================================
    logs_parser = subparsers.add_parser(
        "logs",
        help="Hermes 로그 파일 조회 및 필터링",
        description="agent.log / errors.log / gateway.log 조회, tail, 필터링",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
예시:
    hermes logs                    agent.log 최근 50줄 표시
    hermes logs -f                 agent.log를 실시간으로 따라가기
    hermes logs errors             errors.log 최근 50줄 표시
    hermes logs gateway -n 100     gateway.log 최근 100줄 표시
    hermes logs --level WARNING    WARNING 이상만 표시
    hermes logs --session abc123   세션 ID로 필터링
    hermes logs --component tools  도구 관련 줄만 표시
    hermes logs --since 1h         최근 1시간의 줄만 표시
    hermes logs --since 30m -f     최근 30분부터 시작해 따라가기
    hermes logs list               사용 가능한 로그 파일과 크기 표시
""",
    )
    logs_parser.add_argument(
        "log_name", nargs="?", default="agent",
        help="볼 로그: agent(기본값), errors, gateway 또는 사용 가능한 파일 표시용 'list'",
    )
    logs_parser.add_argument(
        "-n", "--lines", type=int, default=50,
        help="표시할 줄 수(기본값: 50)",
    )
    logs_parser.add_argument(
        "-f", "--follow", action="store_true",
        help="로그를 실시간으로 따라가기(tail -f처럼)",
    )
    logs_parser.add_argument(
        "--level", metavar="LEVEL",
        help="표시할 최소 로그 레벨(DEBUG, INFO, WARNING, ERROR)",
    )
    logs_parser.add_argument(
        "--session", metavar="ID",
        help="이 세션 ID 부분 문자열이 포함된 줄만 필터링",
    )
    logs_parser.add_argument(
        "--since", metavar="TIME",
        help="TIME 전부터의 줄만 표시(예: 1h, 30m, 2d)",
    )
    logs_parser.add_argument(
        "--component", metavar="NAME",
        help="구성요소로 필터링: gateway, agent, tools, cli, cron",
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
