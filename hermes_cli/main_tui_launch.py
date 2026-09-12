"""TUI (ui-tui) launcher: prepared source builds and argv/env assembly.

Split out of ``hermes_cli/main.py``. Names that still live in main (``PROJECT_ROOT``, ...)
are imported lazily inside the functions that use them (avoids an import cycle).
"""

import logging
import contextlib
import json
import os
import shutil
import subprocess
import sys

from pathlib import Path
from typing import Optional

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.main")


def _read_tui_active_session_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return str(data.get("session_id") or "").strip() or None
    except Exception:
        return None


def _print_tui_exit_summary(session_id: Optional[str], active_session_file: Optional[str] = None) -> None:
    """Print a shell-visible epilogue after TUI exits."""
    from hermes_cli.main import _resolve_last_session
    target = (
        _read_tui_active_session_file(active_session_file) or session_id or _resolve_last_session(source="tui")
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
        tokens = {
            k: int(session.get(f"{k}_tokens") or 0)
            for k in ("input", "output", "cache_read", "cache_write", "reasoning")}
    except Exception:
        return
    finally:
        if db is not None:
            db.close()

    print(f"\nResume this session with:\n  hermes --tui --resume {target}")
    if title:
        print(f'  hermes --tui -c "{title}"')
    print(f"\nSession:        {target}")
    if title:
        print(f"Title:          {title}")
    print(f"Messages:       {message_count}")
    print(
        "Tokens:         "
        f"{sum(tokens.values())} (in {tokens['input']}, out {tokens['output']}, "
        f"cache {tokens['cache_read'] + tokens['cache_write']}, reasoning {tokens['reasoning']})"
    )


_TUI_BUILD_INPUT_DIRS = ("src", "packages/hermes-ink/src", "../apps/shared")

_TUI_BUILD_INPUT_FILES = (
    "package.json",
    "package-lock.json",
    "../package.json",
    "../package-lock.json",
    "tsconfig.json",
    "tsconfig.build.json",
    "babel.compiler.config.cjs",
    "../scripts/build/tui.mjs",
    "../scripts/build/frontend-common.mjs",
    "packages/hermes-ink/package.json",
    "packages/hermes-ink/index.js",
    "packages/hermes-ink/text-input.js",
)

_TUI_BUILD_INPUT_SUFFIXES = frozenset({".cjs", ".js", ".jsx", ".json", ".mjs", ".ts", ".tsx"})


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
    """True when ``dist/entry.js`` is missing or older than TUI inputs (Termux cold-start saver);
    ``HERMES_TUI_FORCE_BUILD=1`` forces a rebuild."""
    force = (os.environ.get("HERMES_TUI_FORCE_BUILD") or "").strip().lower()
    if force in {"1", "true", "yes", "on"}:
        return True

    try:
        output_mtime = (root / "dist" / "entry.js").stat().st_mtime
    except OSError:
        return True

    for path in _iter_tui_build_inputs(root):
        try:
            if path.stat().st_mtime > output_mtime:
                return True
        except OSError:
            return True
    return False


def _find_bundled_tui(hermes_cli_dir: Path | None = None) -> Path | None:
    """Find a pre-built TUI entry.js bundled in the wheel."""
    if hermes_cli_dir is None:
        hermes_cli_dir = Path(__file__).parent
    bundled = hermes_cli_dir / "tui_dist" / "entry.js"
    return bundled if bundled.is_file() else None


def _restore_tui_workspace(tui_dir: Path) -> bool:
    """Best-effort ``git restore`` of a missing ``ui-tui/`` (Windows AV/NTFS filters can delete
    tracked files after ``hermes update``); True when the directory exists afterwards.

    On Windows an antivirus / NTFS filter driver can leave tracked ``ui-tui/`` files deleted in the working
    tree after ``hermes update`` (HEAD stays intact; the files just vanish — see issue #49145). Those files
    are tracked, so ``git restore`` puts them back deterministically. Best-effort: returns False (rather
    than raising) when git is unavailable, this isn't a checkout, or the restore leaves the directory still
    missing — the caller then prints the manual-recovery message.
    """
    git = shutil.which("git")
    if not git or not (tui_dir.parent / ".git").exists():
        return False
    try:
        subprocess.run(
            [git, "restore", "--", tui_dir.name], cwd=str(tui_dir.parent), capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=False)
    except OSError:
        return False
    return tui_dir.is_dir()


def _ensure_tui_workspace(tui_dir: Path) -> None:
    """Ensure ``ui-tui/`` exists before it is used as a subprocess cwd (else ``NotADirectoryError``
    / ``WinError 267`` with no usable message): git-restore first, then abort with recovery steps.

    Without this, a missing workspace falls through to ``subprocess.run(..., cwd=<missing ui-tui>)``, which
    crashes with ``NotADirectoryError`` (``WinError 267`` on Windows) instead of a usable message (#49145).
    We first try to self-heal via ``git restore``; only if that can't recover the directory do we abort with
    concrete manual-recovery steps.
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
        file=sys.stderr)
    sys.exit(1)


def _tui_node_bin(bin: str) -> str:
    """Resolve the TUI runtime through PM; an explicit bundled HERMES_NODE wins."""
    if bin == "node":
        env_node = os.environ.get("HERMES_NODE")
        if env_node and os.path.isfile(env_node) and os.access(env_node, os.X_OK):
            return env_node
    from pm import ensure
    path = shutil.which(bin, path=ensure(bin).env["PATH"])
    if not path:
        print(f"{bin} not found — install Node.js to use the TUI.")
        sys.exit(1)
    return path


def _make_tui_argv(tui_dir: Path, tui_dev: bool) -> tuple[list[str], Path]:
    """TUI: --dev → tsx src; else node dist (HERMES_TUI_DIR prebuilt or esbuild)."""

    # Footgun: --dev against a prebuilt bundle that has no source/node_modules.
    ext_dir = os.environ.get("HERMES_TUI_DIR")
    if tui_dev and ext_dir:
        print(
            f"Error: --dev is incompatible with HERMES_TUI_DIR={ext_dir}\n"
            f"The prebuilt TUI has no source code to hot-reload.\n"
            f"Unset HERMES_TUI_DIR (e.g. `unset HERMES_TUI_DIR`) to use --dev from a checkout.",
            file=sys.stderr)
        sys.exit(1)

    # 1. Prebuilt bundle (nix / packaged release / Docker image): just run it.
    # Must run BEFORE _ensure_tui_workspace(): a prebuilt install ships
    # hermes_cli/tui_dist/entry.js but never ui-tui/ (git checkouts only).
    # 1. A prebuilt install (Docker image, Nix build, or prior `npm run build`) ships
    #   hermes_cli/tui_dist/entry.js but never ships ui-tui/ at all (that directory only exists in a git
    #   checkout) — so requiring the workspace to exist first made every prebuilt dashboard Chat tab
    #   connection hard-exit before it ever got a chance to try the bundled entry.js it already has. See
    #   #56665.
    if not tui_dev:
        if ext_dir:
            p = Path(ext_dir)
            if (p / "dist" / "entry.js").is_file():
                return [_tui_node_bin("node"), "--expose-gc", str(p / "dist" / "entry.js")], p

        bundled = _find_bundled_tui()
        if bundled is not None:
            return [_tui_node_bin("node"), "--expose-gc", str(bundled)], bundled.parent

    # About to npm install/build from source, so the workspace must exist.
    if not ext_dir:
        _ensure_tui_workspace(tui_dir)

    if not tui_dev and not _tui_need_rebuild(tui_dir):
        return [_tui_node_bin("node"), "--expose-gc", str(tui_dir / "dist/entry.js")], tui_dir

    from hermes_cli.source_build import build_source_tui, prepare_launch_dependencies, source_build_env

    project_root = tui_dir.parent
    env = source_build_env()
    prepare_launch_dependencies(project_root, env=env)
    if tui_dev:
        # tsx imports @hermes/ink's built exports; the production bundle instead
        # compiles its source directly through scripts/build/tui.mjs.
        npm = shutil.which("npm", path=env["PATH"])
        subprocess.run([npm, "run", "build"], cwd=tui_dir / "packages/hermes-ink", env=env, check=True)
        tsx = tui_dir / "node_modules/.bin/tsx"
        return ([str(tsx), "src/entry.tsx"] if tsx.exists() else [npm, "start"]), tui_dir

    build_source_tui(project_root, env=env)
    node = shutil.which("node", path=env["PATH"])
    return [node, "--expose-gc", str(tui_dir / "dist/entry.js")], tui_dir


def _split_comma_items(items, *, split_non_str: bool = True) -> list[str]:
    """Flatten str / list (comma-separated) input into stripped non-empty parts."""
    raw_items = [items] if isinstance(items, str) else items
    if not isinstance(raw_items, (list, tuple)):
        raw_items = [raw_items]
    normalized: list[str] = []
    for item in raw_items:
        if split_non_str or isinstance(item, str):
            normalized.extend(part.strip() for part in str(item).split(","))
        else:
            normalized.append(str(item).strip())
    return [item for item in normalized if item]


def _normalize_tui_toolsets(toolsets: object) -> list[str]:
    """Normalize argparse/Fire-style toolset input for the TUI subprocess."""
    try:
        from hermes_cli.oneshot import _normalize_toolsets
        return _normalize_toolsets(toolsets) or []
    except (AttributeError, ImportError):
        return _split_comma_items(toolsets, split_non_str=False) if toolsets else []


def _read_cgroup_memory_limit() -> Optional[int]:
    """Container memory limit in bytes, or None if unconstrained (v2 ``memory.max``, then v1).

    V8 is NOT cgroup-aware: a flat 8GB heap grows past a smaller container limit
    and the OOM-killer SIGKILLs Node with no breadcrumb (bare ``stdin EOF``).
    """
    candidates = (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    )
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                raw = f.read().strip()
        except (OSError, ValueError):
            continue
        if raw == "max":
            return None
        if not raw:
            continue  # don't mistake an empty v2 file for "unlimited"
        try:
            limit = int(raw)
        except ValueError:
            continue
        if limit <= 0:
            continue
        if limit >= (1 << 50):  # >= ~1 PB is the v1 "unlimited" sentinel
            return None
        return limit
    return None


def _resolve_tui_heap_mb(default_mb: int = 8192) -> int:
    """V8 ``--max-old-space-size`` (MB) that fits the container: ``default_mb`` when unconstrained,
    else 75% of the cgroup limit (headroom for non-heap RSS + the gateway child), floored at
    1536MB when the container is > 2GB (below that V8 GC-thrashes)."""
    limit = _read_cgroup_memory_limit()
    if not limit:
        return default_mb
    limit_mb = limit // (1024 * 1024)
    sized = int(limit_mb * 0.75)
    if sized >= default_mb:
        return default_mb
    # Below the floor, honor the limit-derived value anyway: a graceful V8 exit
    # beats a silent cgroup kill.
    return max(1536, sized) if limit_mb > 2048 else sized


def _safe_tui_cwd(env: Optional[dict] = None) -> str:
    """Return a stable cwd value for the Node TUI child environment."""
    from hermes_cli.main import PROJECT_ROOT
    try:
        return os.getcwd()
    except FileNotFoundError:
        candidate = ((env or {}).get("PWD") or os.environ.get("PWD") or "").strip()
        if candidate and Path(candidate).is_dir():
            return candidate
        return str(PROJECT_ROOT)


def _apply_tui_python_env(env: dict) -> None:
    """Seed/repair Python-related env vars shared by CLI and dashboard TUI launches."""
    from hermes_cli.main import PROJECT_ROOT
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


def _setup_tui_worktree() -> dict:
    """Create the ``--worktree`` checkout for a TUI launch (prune + async pack maintenance); exits on failure."""
    wt_info = None
    try:
        from cli import _git_repo_root, _maintain_pack_health, _prune_stale_worktrees, _setup_worktree
        repo = _git_repo_root()
        if repo:
            _prune_stale_worktrees(repo)
            # Repack on pack sprawl so `worktree add` never crawls on a
            # multi-agent box; on a thread so it can't block launch.
            import threading as _threading

            _threading.Thread(
                target=_maintain_pack_health, args=(repo,), name="pack-maintenance", daemon=True).start()
        wt_info = _setup_worktree()
    except Exception as exc:
        print(f"✗ Failed to create TUI worktree: {exc}", file=sys.stderr)
    if not wt_info:
        sys.exit(1)
    return wt_info


def _launch_tui(
    resume_session_id: Optional[str] = None, tui_dev: bool = False, model: Optional[str] = None,
    provider: Optional[str] = None, toolsets: object = None, skills: object = None,
    verbose: Optional[bool] = None, quiet: bool = False, query: Optional[str] = None,
    image: Optional[str] = None, worktree: bool = False, checkpoints: bool = False,
    pass_session_id: bool = False, max_turns: Optional[int] = None, accept_hooks: bool = False):
    """Replace current process with the TUI."""
    from hermes_cli.main import PROJECT_ROOT
    tui_dir = PROJECT_ROOT / "ui-tui"

    import tempfile
    # TUI child is a hermes process: propagate the profile-home contract via
    # the single factory; keep secrets (the TUI/agent needs provider creds).
    from tools.environments.local import build_subprocess_env
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
    from hermes_cli.shared_session_attach import configure_tui_attachment
    try:
        configure_tui_attachment(env, resume_session_id)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    try:
        from hermes_cli.config import apply_terminal_config_to_env
        apply_terminal_config_to_env(env=env)
    except Exception:
        logger.debug("Failed to apply terminal config bridge for TUI launch", exc_info=True)
    active_session_fd, active_session_file = tempfile.mkstemp(
        prefix="hermes-tui-active-session-", suffix=".json")
    os.close(active_session_fd)
    env["HERMES_TUI_ACTIVE_SESSION_FILE"] = active_session_file
    env.setdefault("NODE_ENV", "development" if tui_dev else "production")

    wt_info = None
    if worktree:
        wt_info = _setup_tui_worktree()
        env["HERMES_CWD"] = wt_info["path"]
        env["TERMINAL_CWD"] = wt_info["path"]

    _apply_tui_python_env(env)

    skills_value = ""
    if skills:
        skills_value = (
            ",".join(_split_comma_items(skills)) if isinstance(skills, (list, tuple)) else str(skills).strip())
    for key, value in (
        ("HERMES_MODEL", model), ("HERMES_INFERENCE_MODEL", model),
        ("HERMES_TUI_PROVIDER", provider), ("HERMES_INFERENCE_PROVIDER", provider),
        ("HERMES_TUI_TOOLSETS", ",".join(_normalize_tui_toolsets(toolsets))),
        ("HERMES_TUI_SKILLS", skills_value),
        ("HERMES_TUI_QUERY", query), ("HERMES_TUI_IMAGE", image),
        ("HERMES_TUI_CHECKPOINTS", "1" if checkpoints else None),
        ("HERMES_TUI_PASS_SESSION_ID", "1" if pass_session_id else None),
        ("HERMES_TUI_MAX_TURNS", str(max_turns) if max_turns is not None else None),
        ("HERMES_TUI_TOOL_PROGRESS", "verbose" if verbose else "off" if quiet else None),
        ("HERMES_ACCEPT_HOOKS", "1" if accept_hooks else None)):
        if value:
            env[key] = value
    # Generous V8 heap (8GB target; default cap can fatal-OOM on long sessions),
    # sized below the cgroup limit by _resolve_tui_heap_mb() so V8 exits
    # gracefully instead of being reaped silently. Token-level merge respects a
    # user-supplied --max-old-space-size. --expose-gc is NOT added here: Node
    # rejects it in NODE_OPTIONS; _make_tui_argv() passes it as a direct flag.
    _tokens = env.get("NODE_OPTIONS", "").split()
    if not any(t.startswith("--max-old-space-size=") for t in _tokens):
        _tokens.append(f"--max-old-space-size={_resolve_tui_heap_mb()}")
    env["NODE_OPTIONS"] = " ".join(_tokens)
    # HERMES_TUI_RESUME is an internal hand-off to the Ink app. We start from a
    # full os.environ snapshot, so a stale exported value would make a plain
    # `hermes --tui` try to resume a non-existent session; only forward the id
    # argparse resolved for this invocation.
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
        with contextlib.suppress(OSError):
            os.unlink(active_session_file)
        if wt_info:
            with contextlib.suppress(Exception):
                from cli import _cleanup_worktree
                _cleanup_worktree(wt_info)

    # Exit code 42 = TUI requested an update. Relaunch as `hermes update`;
    # preserve_inherited=False keeps --tui and other flags out of the subcommand.
    if code == 42:
        from hermes_cli.relaunch import relaunch
        print("\n⚕ Launching update...\n")
        relaunch(["update"], preserve_inherited=False)

    sys.exit(code)


def _pin_kanban_board_env() -> None:
    """Pin the active kanban board into ``HERMES_KANBAN_BOARD`` so in-process tools and shelled-out
    ``hermes kanban`` calls agree even if a concurrent ``boards switch`` flips the file mid-turn.

    Without this, in-process tools (``kanban_*``) and shelled-out CLI calls (``hermes kanban …``) resolve
    the board on different paths: the env-pin if set, otherwise the global ``<root>/kanban/current`` file. A
    concurrent ``hermes kanban boards switch`` from another session can flip the file mid-turn, so the same
    chat sees its tool calls hit board A while its shell calls hit board B (#20074). Pinning at chat boot
    mirrors what the dispatcher already does for spawned workers.
    """
    if os.environ.get("HERMES_KANBAN_BOARD"):
        return
    with contextlib.suppress(Exception):
        from hermes_cli.kanban_db import get_current_board
        os.environ["HERMES_KANBAN_BOARD"] = get_current_board()


def _sync_bundled_skills_quietly() -> None:
    """Seed ``~/.hermes/skills/`` with the bundled library (idempotent, milliseconds when synced).
    Failures are swallowed: skills are an enhancement, not a hard dependency."""
    with contextlib.suppress(Exception):
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)


def _resolve_use_tui(args) -> bool:
    """Decide whether to launch the TUI: ``--cli`` → classic; ``--tui`` → TUI; no TTY → classic;
    ``HERMES_TUI=1`` → TUI; ``display.interface`` config; default classic.

    The TTY gate is load-bearing: ambient preferences must never hijack a piped
    ``hermes chat -q`` (kanban workers, cron) — the Ink no-TTY bail-out exits 0 and
    the worker dies with a protocol violation. Explicit ``--tui`` still bails out.
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
