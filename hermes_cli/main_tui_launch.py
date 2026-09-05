"""TUI (ui-tui) launcher: node/npm bootstrap, workspace/rebuild checks, argv/env assembly.

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
        data = json.loads(Path(path).read_text(encoding="utf-8"))
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


_NPM_LOCK_RUNTIME_KEYS = frozenset({"ideallyInert", "peer", "dev", "extraneous", "hasInstallScript", "optional"})
"""Lockfile fields npm writes non-deterministically at install time.

``ideallyInert`` marks packages npm skipped (per-platform opt-outs); ``peer`` is
dropped from the hidden ``.package-lock.json`` on dev-deps that are also peers.
``dev`` / ``optional`` / ``extraneous`` / ``hasInstallScript`` are boolean
annotations npm populates differently in the hidden lock (npm >= 10/11), and
may differ even when present in both. None indicate a real declared-vs-installed
skew — the authoritative check is the ``resolved``/``integrity`` pair, which the
intersection comparison in :func:`_tui_need_npm_install` always catches.
"""


def _workspace_root(dir: Path) -> Path:
    """The npm workspace root for *dir*: its parent when *dir* has ``package.json`` but the
    lockfile lives one level up (hoisted node_modules), else *dir* (standalone / prebuilt).
    Shared by the install check, TUI launcher and web build so their cwd can't diverge."""
    if (
        (dir / "package.json").is_file()
        and not (dir / "package-lock.json").is_file()
        and (dir.parent / "package-lock.json").is_file()):
        return dir.parent
    return dir


def _child_workspace_dirs(dir: Path):
    """Sorted ``dir/packages/*`` subdirs that carry a ``package.json``."""
    packages_dir = dir / "packages"
    if not packages_dir.is_dir():
        return
    for child in sorted(packages_dir.iterdir()):
        if child.is_dir() and (child / "package.json").is_file():
            yield child


def _termux_workspace_install_context(
    dir: Path, *, include_child_workspaces: bool = False) -> tuple[Path, tuple[str, ...]]:
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
        for child in _child_workspace_dirs(dir):
            workspace_args.extend(["--workspace", child.relative_to(ws_root).as_posix()])
    workspace_args.append("--include-workspace-root=false")
    return ws_root, tuple(workspace_args)


def _npm_lock_workspace_closure(packages: dict, starts) -> Optional[set]:
    """Package-map keys reachable from the selected workspaces (*starts*: set or str) via npm resolution.

    ``devDependencies`` are followed for each start (npm installs every selected
    workspace's dev toolchain) but not for transitive deps. None when no start is
    in *packages* so callers fall back to the full comparison — which would report
    every OTHER workspace's deps (``apps/desktop``, ``web``) as missing and
    reinstall on every launch. Names resolve by walking up ``node_modules``
    ancestors; ``link: true`` entries are followed to their real package.

    The launch install is scoped with ``npm install --workspace ui-tui`` (see ``_make_tui_argv``), so only
    the ui-tui workspace's dependency closure is written to the hidden ``.package-lock.json``. On Termux it
    additionally selects ui-tui's child ``packages/*`` workspaces, so their devDependencies join the closure
    too. See #66978.
    """
    start_set = {starts} if isinstance(starts, str) else {s for s in starts if s}
    present = [s for s in start_set if s in packages]
    if not present:
        return None

    def resolve(from_key: str, dep: str) -> Optional[str]:
        base = from_key
        while True:
            candidate = f"{base}/node_modules/{dep}" if base else f"node_modules/{dep}"
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
        resolved = entry.get("resolved")
        if entry.get("link") and isinstance(resolved, str) and resolved in packages:
            stack.append(resolved)
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
    """Lock-map keys the launch install scopes to: ui-tui, plus its child ``packages/*`` on Termux
    (each a dev-included closure root). Empty when ui-tui isn't under *ws_root*."""
    from hermes_cli.main import _is_termux_startup_environment
    try:
        keys = {tui_dir.relative_to(ws_root).as_posix()}
    except ValueError:
        return set()
    if _is_termux_startup_environment():
        for child in _child_workspace_dirs(tui_dir):
            try:
                keys.add(child.relative_to(ws_root).as_posix())
            except ValueError:
                continue
    return keys


def _tui_need_npm_install(root: Path) -> bool:
    """True when @hermes/ink is missing or node_modules is behind package-lock.json.

    Prebuilt bundle (``dist/entry.js``, no lockfile): nothing to install. The root
    lock is compared to npm's hidden ``node_modules/.package-lock.json`` by CONTENT
    (git bumps mtimes without changing deps): missing from hidden → reinstall
    unless ``optional``/``peer``/``link`` or outside ``node_modules/``; present in
    both → compare the intersection of non-null fields minus
    ``_NPM_LOCK_RUNTIME_KEYS`` (``resolved``/``integrity`` are always in both).
    Hidden-only entries are ignored; unparseable lockfiles fall back to mtime.
    """
    entry = root / "dist" / "entry.js"
    ws_root = _workspace_root(root)
    lock = ws_root / "package-lock.json"
    if entry.is_file() and not lock.is_file():
        return False

    if not (ws_root / "node_modules" / "@hermes" / "ink" / "package.json").is_file():
        return True
    if not lock.is_file():
        return False
    marker = ws_root / "node_modules" / ".package-lock.json"
    if not marker.is_file():
        return True

    try:
        wanted = json.loads(lock.read_text(encoding="utf-8")).get("packages") or {}
        installed = json.loads(marker.read_text(encoding="utf-8")).get("packages") or {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return lock.stat().st_mtime > marker.stat().st_mtime

    def entries_differ(pkg: dict, installed_pkg: dict) -> bool:
        a = {k: v for k, v in pkg.items() if k not in _NPM_LOCK_RUNTIME_KEYS}
        b = {k: v for k, v in installed_pkg.items() if k not in _NPM_LOCK_RUNTIME_KEYS}
        return any(a[k] is not None and b[k] is not None and a[k] != b[k] for k in a.keys() & b.keys())

    # Shared workspace checkout: the launch install is scoped to ui-tui (+ child
    # packages on Termux), so limit the comparison to that closure. Standalone /
    # own-lockfile layouts do a full install and keep the full comparison.
    # Limit the comparison to the same selected-workspace closure so unrelated workspace deps (apps/desktop,
    # web, …) don't force a reinstall every launch (#66978).
    closure: Optional[set] = None
    if ws_root != root:
        selected = _tui_selected_workspace_keys(root, ws_root)
        if selected:
            closure = _npm_lock_workspace_closure(wanted, selected)

    for name, pkg in wanted.items():
        if not name or (closure is not None and name not in closure) or not isinstance(pkg, dict):
            continue
        if name not in installed:
            # Workspace link entries are never materialized by a partial
            # `npm install --workspace ui-tui`; don't force a reinstall for them.
            # Workspace link entries (`"link": true`, paths outside node_modules/ like `apps/desktop`,
            # `node_modules/web`) are never materialized by a partial `npm install --workspace ui-tui` —
            # they're deliberately skipped (see #38772) and would otherwise force a reinstall on every
            # launch.
            if pkg.get("optional") or pkg.get("peer") or pkg.get("link"):
                continue
            if not name.startswith("node_modules/"):
                continue
            return True
        if isinstance(installed[name], dict) and entries_differ(pkg, installed[name]):
            return True

    return False


_TUI_BUILD_INPUT_DIRS = ("src", "packages/hermes-ink/src")

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


def _ensure_tui_node() -> None:
    """Ensure `node` + `npm` are on PATH: else run node-bootstrap.sh `ensure_node` and prepend
    the resolved node dir to PATH. ``HERMES_SKIP_NODE_BOOTSTRAP=1`` disables auto-install."""
    from hermes_cli.main import PROJECT_ROOT
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
        # Helper logs to stderr; stdout carries `command -v node` — subshell PATH
        # edits don't leak back into Python, so the capture is the bridge.
        result = subprocess.run(
            ["bash", "-c", f'source "{helper}" >&2 && ensure_node >&2 && command -v node'],
            env={**os.environ, "HERMES_HOME": hermes_home},
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    except (OSError, subprocess.SubprocessError):
        return

    parts = os.environ.get("PATH", "").split(os.pathsep)
    resolved = (result.stdout or "").strip()
    extras = [Path(resolved).resolve().parent] if resolved else []
    extras += [Path(hermes_home) / "node" / "bin", Path.home() / ".local" / "bin"]
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


def _npm_lifecycle_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Build a clean environment for the pinned UI toolchain lifecycle."""
    run_env = {**os.environ, **(env or {}), "CI": "1"}
    # esbuild treats this as an executable override. If a shell points it at a
    # different release, the pinned package's postinstall rejects that binary.
    run_env.pop("ESBUILD_BINARY_PATH", None)
    return run_env


def _tui_node_bin(bin: str) -> str:
    """Resolve ``node``/``npm`` for the TUI launch, or exit with a hint. ``HERMES_NODE`` wins for node;
    ``find_node_executable()`` sees the managed ``$HERMES_HOME/node`` tree a bare which() misses."""
    if bin == "node":
        env_node = os.environ.get("HERMES_NODE")
        if env_node and os.path.isfile(env_node) and os.access(env_node, os.X_OK):
            return env_node
    from hermes_constants import find_node_executable
    path = find_node_executable(bin)
    if not path and bin == "node":
        with contextlib.suppress(Exception):
            from hermes_cli.dep_ensure import ensure_dependency
            if ensure_dependency("node"):
                path = find_node_executable("node")
    if not path:
        print(f"{bin} not found — install Node.js to use the TUI.")
        sys.exit(1)
    return path


def _exit_on_npm_failure(result: subprocess.CompletedProcess, message: str, *, sep: str) -> None:
    """Print *message* plus the last 30 lines of npm output and exit 1 on a non-zero rc."""
    if result.returncode == 0:
        return
    combined = f"{result.stdout or ''}{sep}{result.stderr or ''}".strip()
    preview = "\n".join(combined.splitlines()[-30:])
    print(message)
    if preview:
        print(preview)
    sys.exit(1)


def _run_tui_npm_build(npm: str, cwd: Path, failure_message: str) -> None:
    """``npm run build`` in *cwd*; exit with *failure_message* + output tail on failure."""
    result = subprocess.run(
        [npm, "run", "build"], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=_npm_lifecycle_env())
    _exit_on_npm_failure(result, failure_message, sep="")


def _install_tui_dependencies(tui_dir: Path, *, termux_startup: bool) -> None:
    """``npm install`` for the TUI workspace, with one EBADENGINE repair retry. Exits on failure.

    ``--workspace ui-tui`` avoids resolving apps/desktop (Electron + node-pty) and
    is omitted when ui-tui/ has its own lockfile. ``--include=dev``: the build
    toolchain is in devDependencies and an inherited ``NODE_ENV=production`` /
    ``omit=dev`` would silently skip it.
    """
    npm = _tui_node_bin("npm")
    if not os.environ.get("HERMES_QUIET"):
        print("Installing TUI dependencies…")
    npm_cwd = _workspace_root(tui_dir)
    # --workspace ui-tui avoids resolving apps/desktop (Electron + node-pty). See #38772. When ui-tui/ has
    # its own package-lock.json (e.g. curl install), _workspace_root() returns tui_dir itself. Passing
    # --workspace in that case fails because npm cannot find a workspace named "ui-tui" inside ui-tui/. See
    # #42973.
    npm_workspace_args: tuple[str, ...] = () if npm_cwd == tui_dir else ("--workspace", "ui-tui")
    if termux_startup:
        npm_cwd, npm_workspace_args = _termux_workspace_install_context(tui_dir, include_child_workspaces=True)
    npm_install_cmd = [
        npm, "install", *npm_workspace_args,
        "--include=dev", "--silent", "--no-fund", "--no-audit", "--progress=false",
    ]

    def _run_tui_install() -> subprocess.CompletedProcess:
        from hermes_constants import with_hermes_node_path
        # Managed tree first on PATH: if the EBADENGINE repair provisioned a
        # managed Node, npm's shebang/lifecycle scripts must resolve that node.
        return subprocess.run(
            npm_install_cmd, cwd=str(npm_cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            env=_npm_lifecycle_env(with_hermes_node_path()))

    result = _run_tui_install()
    if result.returncode != 0:
        # An npm outside the root `engines.npm` range fails before doing any work;
        # repair once (upgrade a managed npm in place, or provision a managed
        # runtime) and retry rather than dumping EBADENGINE at the user.
        from hermes_cli.npm_engine import maybe_repair_npm_engine
        repaired_npm = maybe_repair_npm_engine(npm, f"{result.stdout or ''}\n{result.stderr or ''}")
        if repaired_npm:
            npm_install_cmd[0] = repaired_npm
            result = _run_tui_install()
    _exit_on_npm_failure(result, "npm install failed.", sep="\n")


def _make_tui_argv(tui_dir: Path, tui_dev: bool) -> tuple[list[str], Path]:
    """TUI: --dev → tsx src; else node dist (HERMES_TUI_DIR prebuilt or esbuild)."""
    from hermes_cli.main import _is_termux_startup_environment
    _ensure_tui_node()

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

    # 2. Normal flow: npm install if needed, esbuild, then node dist/entry.js.
    #    --dev: npm install if needed, then tsx src/entry.tsx.
    termux_startup = _is_termux_startup_environment()
    termux_need_rebuild = termux_startup and not tui_dev and _tui_need_rebuild(tui_dir)
    skip_install_for_fresh_termux_bundle = termux_startup and not tui_dev and not termux_need_rebuild
    did_install = False
    if not skip_install_for_fresh_termux_bundle and _tui_need_npm_install(tui_dir):
        _install_tui_dependencies(tui_dir, termux_startup=termux_startup)
        did_install = True

    if tui_dev:
        # --dev runs src/entry.tsx directly, but @hermes/ink resolves through
        # packages/hermes-ink/dist/entry-exports.js; a stale dist after a pull
        # leaves newer hooks/components missing at runtime. Prebuild it here.
        npm = _tui_node_bin("npm")
        _run_tui_npm_build(npm, tui_dir / "packages" / "hermes-ink", "TUI dev prebuild failed.")
        tsx = tui_dir / "node_modules" / ".bin" / "tsx"
        if tsx.exists():
            return [str(tsx), "src/entry.tsx"], tui_dir
        return [npm, "start"], tui_dir

    # Desktop/dev launches always rebuild; Termux cold starts use the freshness
    # check because esbuild startup is expensive on old mobile CPUs.
    if not termux_startup or did_install or termux_need_rebuild:
        _run_tui_npm_build(_tui_node_bin("npm"), tui_dir, "TUI build failed.")

    return [_tui_node_bin("node"), "--expose-gc", str(tui_dir / "dist" / "entry.js")], tui_dir


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
            with open(path, "r", encoding="utf-8") as f:
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
