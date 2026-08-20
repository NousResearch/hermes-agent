"""Auto-installation of LSP server binaries.

Tries to install missing servers using whatever package manager is
appropriate. All installs go under ``<HERMES_HOME>/lsp/`` so we don't
pollute the user's global toolchain. Standalone binaries and Hermes-owned
Windows npm launchers are staged in ``bin/``; package payloads and npm's
canonical wrappers remain under ``node_modules/``.

Strategies:

- ``auto`` — attempt to install with the best available package
  manager.  This is the default.
- ``manual`` — never install; if a binary is missing, the server is
  silently skipped and the user is told about it via ``hermes lsp
  status``.
- ``off`` — same as ``manual`` for now (kept distinct so we can
  evolve behavior later, e.g. logging differently).

The actual installs happen synchronously the first time a server is
needed and concurrent calls to :func:`try_install` for the same
package are deduplicated via a per-package lock.

Failure modes are non-fatal: every install path is wrapped in
try/except and returns ``None`` on failure.  The tool layer then
falls back to its in-process syntax checker, exactly as if the user
hadn't enabled LSP at all.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli._subprocess_compat import run_windows_batch, windows_hide_flags
from hermes_constants import find_node_executable

logger = logging.getLogger("agent.lsp.install")

# Package-name → install-strategy hint registry.  Each entry is a
# tuple of strategy name + package name + executable name.  When the
# install completes, we look for the executable in
# ``<HERMES_HOME>/lsp/bin/`` first, then on PATH.
#
# Optional fields:
#   - ``extra_pkgs``: list of sibling packages to install alongside
#     ``pkg`` in the same node_modules tree.  Used when an LSP server
#     has a runtime peer dependency that npm doesn't auto-pull (e.g.
#     typescript-language-server needs ``typescript``).
INSTALL_RECIPES: Dict[str, Dict[str, Any]] = {
    # Python
    "pyright": {"strategy": "npm", "pkg": "pyright", "bin": "pyright-langserver"},
    # JS/TS family
    "typescript-language-server": {
        "strategy": "npm",
        "pkg": "typescript-language-server",
        "bin": "typescript-language-server",
        # typescript-language-server requires the `typescript` SDK
        # (tsserver) to be importable from the same node_modules tree;
        # otherwise initialize() fails with "Could not find a valid
        # TypeScript installation".  Install them together.
        # typescript-language-server 5.x loads lib/tsserver.js. TypeScript 7
        # replaced that layout with the native compiler and no longer ships
        # tsserver.js, so an unbounded ``typescript`` install cannot initialize.
        "extra_pkgs": ["typescript@6"],
    },
    "@vue/language-server": {
        "strategy": "npm",
        "pkg": "@vue/language-server",
        "bin": "vue-language-server",
    },
    "svelte-language-server": {
        "strategy": "npm",
        "pkg": "svelte-language-server",
        "bin": "svelteserver",
    },
    "@astrojs/language-server": {
        "strategy": "npm",
        "pkg": "@astrojs/language-server",
        "bin": "astro-ls",
    },
    "yaml-language-server": {
        "strategy": "npm",
        "pkg": "yaml-language-server",
        "bin": "yaml-language-server",
    },
    "bash-language-server": {
        "strategy": "npm",
        "pkg": "bash-language-server",
        "bin": "bash-language-server",
    },
    "intelephense": {"strategy": "npm", "pkg": "intelephense", "bin": "intelephense"},
    "dockerfile-language-server-nodejs": {
        "strategy": "npm",
        "pkg": "dockerfile-language-server-nodejs",
        "bin": "docker-langserver",
    },
    # Go
    "gopls": {"strategy": "go", "pkg": "golang.org/x/tools/gopls@latest", "bin": "gopls"},
    # Rust — too heavy (hundreds of MB to bootstrap).  We do NOT
    # auto-install rust-analyzer; users install via rustup.
    "rust-analyzer": {"strategy": "manual", "pkg": "", "bin": "rust-analyzer"},
    # C/C++ — manual (clangd ships with LLVM, very heavy)
    "clangd": {"strategy": "manual", "pkg": "", "bin": "clangd"},
    # Lua — manual (LuaLS is platform-specific binaries from GitHub
    # releases; complex enough that we punt to the user)
    "lua-language-server": {"strategy": "manual", "pkg": "", "bin": "lua-language-server"},
    # PowerShell — PowerShellEditorServices ships as a GitHub release
    # zip driven by a pwsh bootstrap script, not a single binary.  We
    # require a manual bundle install and probe for the pwsh host so
    # `hermes lsp status` reports the host's presence.
    "powershell": {"strategy": "manual", "pkg": "", "bin": "pwsh"},
}


_install_locks: Dict[str, threading.Lock] = {}
_install_results: Dict[str, Optional[str]] = {}
_install_lock_meta = threading.Lock()
_WINDOWS_WRAPPER_SUFFIXES = (".exe", ".com", ".cmd", ".bat")
_HERMES_NPM_WRAPPER_TAG = ".hermes"
_DOS_HEADER_SIZE = 64
_PE_HEADER_OFFSET_FIELD = 0x3C


def _is_windows() -> bool:
    return os.name == "nt"


def hermes_lsp_bin_dir() -> Path:
    """Return the Hermes-owned bin staging dir for LSP servers."""
    from hermes_constants import get_hermes_home

    p = get_hermes_home() / "lsp" / "bin"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_windows_launchable(path: Path) -> bool:
    """Return whether ``path`` can be passed to Win32 ``CreateProcess``.

    npm installs both an extensionless POSIX shell shim and a native
    ``.cmd`` wrapper. The shell shim is executable to MSYS, but native
    Windows Python cannot spawn it directly. Recognized Windows wrapper
    suffixes are safe; an extensionless file is accepted only when its DOS
    ``MZ`` header points to a valid PE signature.
    """
    if path.suffix.lower() in _WINDOWS_WRAPPER_SUFFIXES:
        return True
    try:
        with path.open("rb") as handle:
            dos_header = handle.read(_DOS_HEADER_SIZE)
            if len(dos_header) < _DOS_HEADER_SIZE or dos_header[:2] != b"MZ":
                return False
            pe_offset = int.from_bytes(
                dos_header[
                    _PE_HEADER_OFFSET_FIELD : _PE_HEADER_OFFSET_FIELD + 4
                ],
                byteorder="little",
            )
            file_size = os.fstat(handle.fileno()).st_size
            if pe_offset < _DOS_HEADER_SIZE or pe_offset + 4 > file_size:
                return False
            handle.seek(pe_offset)
            return handle.read(4) == b"PE\0\0"
    except OSError:
        return False


def _native_binary_candidates(base: Path) -> list[Path]:
    """Return executable candidates, preferring native Windows wrappers."""
    if not _is_windows():
        return [base]
    candidates: list[Path] = []
    existing: set[str] = set()
    for suffix in _WINDOWS_WRAPPER_SUFFIXES:
        candidate = Path(str(base) + suffix)
        key = str(candidate).lower()
        if key not in existing:
            candidates.append(candidate)
            existing.add(key)
    base_key = str(base).lower()
    if base_key not in existing:
        candidates.append(base)
    return candidates


def _existing_binary(
    name: str,
    *,
    include_generated: bool = True,
    include_staged: bool = True,
) -> Optional[str]:
    """Probe Hermes install locations + PATH for a binary named ``name``."""
    bases = [hermes_lsp_bin_dir() / name] if include_staged else []
    if _is_windows():
        # Hermes-generated wrappers avoid npm's own unquoted ``SET dp0=%~dp0``
        # prologue, which breaks when HERMES_HOME contains cmd metacharacters.
        # Prefer them over both the canonical npm shim and stale pre-fix copies.
        generated = hermes_lsp_bin_dir() / f"{name}{_HERMES_NPM_WRAPPER_TAG}"
        # npm .cmd launchers use paths relative to node_modules/.bin. Moving
        # or copying them into lsp/bin breaks those paths, so prefer the
        # canonical npm location only when no generated wrapper is available.
        npm_bin = hermes_lsp_bin_dir().parent / "node_modules" / ".bin" / name
        bases.insert(0, npm_bin)
        if include_generated:
            bases.insert(0, generated)
    for base in bases:
        for staged in _native_binary_candidates(base):
            if (
                staged.exists()
                and os.access(staged, os.X_OK)
                and (not _is_windows() or _is_windows_launchable(staged))
            ):
                return str(staged)
    if _is_windows():
        for suffix in _WINDOWS_WRAPPER_SUFFIXES:
            on_path = shutil.which(f"{name}{suffix}")
            if on_path:
                return on_path
        on_path = shutil.which(name)
        if on_path and _is_windows_launchable(Path(on_path)):
            return on_path
        return None
    on_path = shutil.which(name)
    if on_path:
        return on_path
    return None


def _get_lock(pkg: str) -> threading.Lock:
    with _install_lock_meta:
        lock = _install_locks.get(pkg)
        if lock is None:
            lock = threading.Lock()
            _install_locks[pkg] = lock
        return lock


def try_install(pkg: str, strategy: str = "auto") -> Optional[str]:
    """Try to install ``pkg`` and return the binary path if successful.

    ``strategy`` is ``"auto"``, ``"manual"``, or ``"off"``.  In
    ``manual``/``off`` mode, this function only probes for an
    existing binary and returns ``None`` if not found.

    The install is cached per-package — a second call returns the
    same path (or ``None``) without reinstalling.  Concurrent calls
    are serialized.
    """
    if strategy not in {"auto",}:
        # Only ``auto`` triggers an actual install.  In manual/off,
        # we still check whether the binary already exists.
        recipe = INSTALL_RECIPES.get(pkg, {})
        bin_name = recipe.get("bin", pkg)
        return _existing_binary(bin_name)

    if pkg in _install_results:
        return _install_results[pkg]

    lock = _get_lock(pkg)
    with lock:
        # Double-check after acquiring lock.
        if pkg in _install_results:
            return _install_results[pkg]
        result = _do_install(pkg)
        _install_results[pkg] = result
        return result


def _do_install(pkg: str) -> Optional[str]:
    recipe = INSTALL_RECIPES.get(pkg)
    if recipe is None:
        # Not in our registry — best-effort: just probe PATH.
        return shutil.which(pkg)

    strategy = recipe.get("strategy", "manual")
    bin_name = recipe.get("bin", pkg)

    # Upgrade existing npm installations in place. npm's own .cmd wrapper uses
    # an unquoted ``SET dp0=%~dp0`` and fails when HERMES_HOME contains cmd
    # metacharacters, so merely finding that canonical wrapper is not enough.
    # Regenerate our relative launcher before accepting an existing npm bin.
    if strategy == "npm" and _is_windows():
        generated = _write_windows_node_wrapper(
            hermes_lsp_bin_dir().parent,
            recipe.get("pkg", pkg),
            bin_name,
        )
        if generated is not None:
            return str(generated)
        # A tagged wrapper whose package/script disappeared is stale. Ignore it
        # so auto-install can repair the package instead of returning a dead path.
        existing = _existing_binary(
            bin_name,
            include_generated=False,
            include_staged=False,
        )
    else:
        existing = _existing_binary(bin_name)
    if existing:
        return existing

    if strategy == "manual":
        logger.debug("[install] %s requires manual install (recipe=%s)", pkg, recipe)
        return None

    if strategy == "npm":
        return _install_npm(
            recipe.get("pkg", pkg),
            bin_name,
            extra_pkgs=recipe.get("extra_pkgs") or [],
        )
    if strategy == "go":
        return _install_go(recipe.get("pkg", pkg), bin_name)
    if strategy == "pip":
        return _install_pip(recipe.get("pkg", pkg), bin_name)

    logger.warning("[install] unknown strategy %r for %s", strategy, pkg)
    return None


def _npm_bin_script(staging: Path, pkg: str, bin_name: str) -> Optional[Path]:
    """Resolve an installed npm package's executable JavaScript entry point."""
    package_dir = staging / "node_modules" / Path(*pkg.split("/"))
    package_json = package_dir / "package.json"
    try:
        metadata = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None

    if not isinstance(metadata, dict):
        return None

    bin_spec = metadata.get("bin")
    if isinstance(bin_spec, str):
        entry = bin_spec
    elif isinstance(bin_spec, dict):
        entry = bin_spec.get(bin_name)
    else:
        return None
    if not isinstance(entry, str) or not entry:
        return None

    try:
        script = (package_dir / entry).resolve()
        script.relative_to(package_dir.resolve())
    except (OSError, ValueError):
        return None
    return script if script.is_file() else None


def _batch_static_literal(value: str) -> Optional[str]:
    """Escape static text embedded in an ASCII Windows batch launcher."""
    if any(char in value for char in ('"', "\r", "\n", "\0")):
        return None
    # Percent signs in a batch file must be doubled. Delayed expansion is
    # disabled by the generated launcher, so literal exclamation marks survive.
    return value.replace("%", "%%")


def _write_windows_node_wrapper(
    staging: Path, pkg: str, bin_name: str
) -> Optional[Path]:
    """Create a location-independent wrapper for an installed npm binary.

    npm's Windows shim computes its package root with the unquoted command
    ``SET dp0=%~dp0``. If HERMES_HOME contains ``&`` (or another cmd
    metacharacter), cmd.exe reparses the expanded path and the shim jumps into
    the wrong command. The Hermes wrapper uses a quoted assignment, disables
    delayed expansion, and launches the package's JS entry through node.

    Paths under HERMES_HOME are expressed relative to ``%dp0%``. That keeps the
    wrapper ASCII even when the user's home directory contains Unicode, while
    quoted variable expansion preserves cmd metacharacters in the real path.
    """
    node = find_node_executable("node")
    script = _npm_bin_script(staging, pkg, bin_name)
    if not node or script is None:
        return None

    wrapper = hermes_lsp_bin_dir() / f"{bin_name}{_HERMES_NPM_WRAPPER_TAG}.cmd"
    home = staging.parent
    try:
        node_path = Path(node).resolve()
        node_path.relative_to(home.resolve())
    except (OSError, ValueError):
        node_ref = _batch_static_literal(str(node))
    else:
        node_rel = _batch_static_literal(os.path.relpath(node_path, wrapper.parent))
        node_ref = f"%dp0%{node_rel}" if node_rel is not None else None

    script_rel = _batch_static_literal(os.path.relpath(script, wrapper.parent))
    if node_ref is None or script_rel is None:
        return None
    script_ref = f"%dp0%{script_rel}"
    content = (
        "@echo off\r\n"
        "setlocal DisableDelayedExpansion\r\n"
        'set "dp0=%~dp0"\r\n'
        f'"{node_ref}" "{script_ref}" %*\r\n'
    )
    temp_wrapper = wrapper.with_name(
        f".{wrapper.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        # The dynamic home path lives in %dp0%; only package-relative paths and
        # (for system Node) its resolved executable remain static. Refuse a
        # non-ASCII static path rather than writing a code-page-fragile script.
        content.encode("ascii")
        temp_wrapper.write_text(content, encoding="ascii", newline="")
        os.replace(temp_wrapper, wrapper)
    except (OSError, UnicodeError):
        try:
            temp_wrapper.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return wrapper


def _install_npm(
    pkg: str,
    bin_name: str,
    extra_pkgs: Optional[list] = None,
) -> Optional[str]:
    """Install an npm package into our staging dir.

    Uses ``npm install --prefix`` so the binaries land in
    ``<staging>/node_modules/.bin/<bin_name>``. POSIX launchers are
    symlinked into our stable bin directory. On Windows, Hermes writes a
    location-independent wrapper that invokes the package's JavaScript entry
    through node; npm's canonical wrapper remains untouched as a fallback.

    ``extra_pkgs`` is a list of sibling packages to install in the
    same ``node_modules`` tree.  Used for LSP servers with runtime
    peer deps that npm doesn't auto-pull (typescript-language-server
    needs ``typescript`` next to it; intelephense ships standalone).
    """
    # Managed npm first: $HERMES_HOME/node is not on an arbitrary process's
    # PATH, so a bare which() misses the Node that Hermes installed and
    # reports "npm not on PATH" on a machine that has a perfectly good one.
    npm = find_node_executable("npm")
    if npm is None:
        logger.info("[install] cannot install %s: no usable npm found", pkg)
        return None
    staging = hermes_lsp_bin_dir().parent  # <HERMES_HOME>/lsp/
    install_targets = [pkg] + list(extra_pkgs or [])
    try:
        logger.info(
            "[install] npm install --prefix %s %s",
            staging,
            " ".join(install_targets),
        )
        command = [
            npm,
            "install",
            "--prefix",
            str(staging),
            "--silent",
            "--no-fund",
            "--no-audit",
            *install_targets,
        ]
        if _is_windows() and npm.lower().endswith((".cmd", ".bat")):
            # Batch launchers are reparsed by cmd.exe. Keep metacharacters in
            # HERMES_HOME and package arguments inside quoted placeholders, and
            # own the whole child tree if the bounded install times out.
            proc = run_windows_batch(
                command,
                env=os.environ,
                prefix="HERMES_LSP_INSTALL",
                timeout=300,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
        else:
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=300,
                stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
        if proc.returncode != 0:
            logger.warning(
                "[install] npm install failed for %s: %s", pkg, proc.stderr.strip()[:500]
            )
            return None
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("[install] npm install errored for %s: %s", pkg, e)
        return None

    # Find the bin
    nm_bin = staging / "node_modules" / ".bin" / bin_name
    if _is_windows():
        generated = _write_windows_node_wrapper(staging, pkg, bin_name)
        if generated is not None:
            return str(generated)
    for c in _native_binary_candidates(nm_bin):
        if c.exists() and (not _is_windows() or _is_windows_launchable(c)):
            if _is_windows():
                # npm wrappers are location-dependent: their script path is
                # relative to node_modules/.bin. Execute the native wrapper
                # in place instead of copying a broken shim into lsp/bin.
                return str(c)
            # Symlink into our `lsp/bin/` for stable PATH access.
            link = hermes_lsp_bin_dir() / c.name
            if not link.exists():
                try:
                    link.symlink_to(c)
                except (OSError, NotImplementedError):
                    # Some filesystems do not support symlinks; copy instead.
                    try:
                        shutil.copy2(c, link)
                    except OSError:
                        return str(c)
            return str(link if link.exists() else c)
    logger.warning("[install] npm install for %s succeeded but bin %s not found", pkg, bin_name)
    return None


def _install_go(pkg: str, bin_name: str) -> Optional[str]:
    """Install a Go module to GOBIN=<staging>."""
    go = shutil.which("go")
    if go is None:
        logger.info("[install] cannot install %s: go not on PATH", pkg)
        return None
    staging = hermes_lsp_bin_dir()
    env = dict(os.environ)
    env["GOBIN"] = str(staging)
    try:
        logger.info("[install] go install %s (GOBIN=%s)", pkg, staging)
        proc = subprocess.run(
            [go, "install", pkg],
            check=False,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=600,
            env=env,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        if proc.returncode != 0:
            logger.warning(
                "[install] go install failed for %s: %s", pkg, proc.stderr.strip()[:500]
            )
            return None
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("[install] go install errored for %s: %s", pkg, e)
        return None
    bin_path = staging / bin_name
    if _is_windows():
        bin_path = bin_path.with_suffix(".exe")
    if bin_path.exists():
        return str(bin_path)
    logger.warning("[install] go install for %s succeeded but bin %s not found", pkg, bin_name)
    return None


def _install_pip(pkg: str, bin_name: str) -> Optional[str]:
    """Install a Python package into a hermes-owned target dir.

    We avoid polluting the user's site-packages by using
    ``pip install --target``.  Bins go into
    ``<staging>/python-packages/bin/`` which we symlink into
    ``<staging>/bin``.  Note: this only works for packages that ship a
    console script.
    """
    pip_target = hermes_lsp_bin_dir().parent / "python-packages"
    pip_target.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("[install] pip install --target %s %s", pip_target, pkg)
        from hermes_cli.tools_config import _pip_install

        proc = _pip_install(
            ["--target", str(pip_target), "--quiet", pkg],
            timeout=300,
        )
        if proc.returncode != 0:
            logger.warning(
                "[install] pip install failed for %s: %s", pkg, (proc.stderr or "").strip()[:500]
            )
            return None
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("[install] pip install errored for %s: %s", pkg, e)
        return None
    # Look for the console script.  POSIX wheels generally write to bin/,
    # while native Windows installs use Scripts/.
    script_dirs = [pip_target / "bin"]
    if _is_windows():
        script_dirs.append(pip_target / "Scripts")
    for script_dir in script_dirs:
        for bin_path in _native_binary_candidates(script_dir / bin_name):
            if bin_path.exists() and (
                not _is_windows() or _is_windows_launchable(bin_path)
            ):
                link = hermes_lsp_bin_dir() / bin_path.name
                if not link.exists():
                    try:
                        link.symlink_to(bin_path)
                    except (OSError, NotImplementedError):
                        try:
                            shutil.copy2(bin_path, link)
                        except OSError:
                            return str(bin_path)
                return str(link if link.exists() else bin_path)
    return None


def detect_status(pkg: str) -> str:
    """Return ``installed``, ``missing``, or ``manual-only`` for a package.

    Used by the ``hermes lsp status`` CLI to give users a quick
    overview of what's available without spawning anything.
    """
    recipe = INSTALL_RECIPES.get(pkg)
    bin_name = recipe.get("bin", pkg) if recipe else pkg
    if _existing_binary(bin_name):
        return "installed"
    if recipe and recipe.get("strategy") == "manual":
        return "manual-only"
    return "missing"


__all__ = [
    "INSTALL_RECIPES",
    "try_install",
    "detect_status",
    "hermes_lsp_bin_dir",
]
