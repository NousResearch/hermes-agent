"""Web UI build orchestration — extracted from ``hermes_cli/main.py``.

Mechanical move (main.py decomposition, shard s3 cluster c2): the web-UI
staleness/stamp helpers and the desktop build-stamp helpers. Function bodies
are lifted verbatim; the only mechanical change is that references to helpers
that STAY in ``hermes_cli.main`` (``_resolve_node_runtime_npm``,
``_run_with_idle_timeout``, ``_run_npm_install_deterministic``,
``_workspace_root``, ``_is_termux_startup_environment``,
``_termux_workspace_install_context``, ``_desktop_packaged_executable``) and
to moved-but-test-patched siblings (``_web_ui_build_needed``,
``_write_web_ui_build_stamp``, ``_desktop_stamp_path``,
``_desktop_dist_exists``) are routed through ``_m()`` — a lazy
``hermes_cli.main`` reference — so existing call sites and test monkeypatches
that target ``hermes_cli.main.<name>`` keep working unchanged. ``main.py``
re-imports every moved name from here (``# noqa: E402``) so the call surface
still resolves on ``hermes_cli.main``.

Imports are one-way: ``hermes_cli.main`` imports this module, never the
reverse at import time (``_m()`` resolves lazily at call time, when main.py is
fully loaded, so there is no import cycle).
"""

import hashlib
import json
import logging
import os
import sys
import time as _time
from pathlib import Path

# Bind the adapter's logger by explicit name so log records emitted from these
# functions keep the logger name they had before the move.
logger = logging.getLogger("hermes_cli.main")


def _m():
    """Lazy ``hermes_cli.main`` reference.

    Lets callers keep patching ``hermes_cli.main.<helper>`` (the historical
    test surface) and have those patches reach this code path, and defers the
    import so ``hermes_cli.main`` -> ``hermes_cli.web_ui_build`` stays one-way
    at import time.
    """
    from hermes_cli import main

    return main


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

    if not _m()._web_ui_build_needed(web_dir):
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

    npm = _m()._resolve_node_runtime_npm()
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

    npm_cwd = _m()._workspace_root(web_dir)
    # Scope the install to the web workspace only so that the full workspace
    # graph (including apps/desktop with its Electron + node-pty deps) is never
    # resolved here.  Without --workspace the root package.json's apps/* glob
    # would pull in desktop on every web build. See #38772.
    # When web/ has its own package-lock.json, _workspace_root() returns
    # web_dir itself and --workspace would fail.  See #42973.
    npm_workspace_args: tuple[str, ...] = () if npm_cwd == web_dir else ("--workspace", "web")
    if _m()._is_termux_startup_environment():
        npm_cwd, npm_workspace_args = _m()._termux_workspace_install_context(web_dir)

    def _install_web_deps(*, silent: bool) -> "subprocess.CompletedProcess":
        return _m()._run_npm_install_deterministic(
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
    r2 = _m()._run_with_idle_timeout([npm, "run", "build"], cwd=web_dir, env=build_env)
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
            r2 = _m()._run_with_idle_timeout([npm, "run", "build"], cwd=web_dir, env=build_env)
        if r2.returncode != 0:
            # Retry once after a short delay — covers boot-time races on Windows
            # (antivirus scanning Node.js binaries, npm cache not ready, transient
            # I/O when launched via Scheduled Task at logon). See issue #23817.
            _time.sleep(3)
            r2 = _m()._run_with_idle_timeout([npm, "run", "build"], cwd=web_dir, env=build_env)

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
    _m()._write_web_ui_build_stamp(project_root, web_dir)
    return True


def _desktop_dist_exists(desktop_dir: Path) -> bool:
    """Return True when a local desktop renderer build is present."""
    return (desktop_dir / "dist" / "index.html").exists()


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
        if not _m()._desktop_dist_exists(desktop_dir):
            return True
    else:
        if _m()._desktop_packaged_executable(desktop_dir) is None:
            return True

    stamp_file = _m()._desktop_stamp_path()
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
    stamp_file = _m()._desktop_stamp_path()
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
