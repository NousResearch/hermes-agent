"""Auto-installation of LSP server binaries.

Tries to install missing servers using whatever package manager is
appropriate.  All installs go to a Hermes-owned bin staging dir,
``<HERMES_HOME>/lsp/bin/``, so we don't pollute the user's global
toolchain.

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

import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("agent.lsp.install")

# Package-name → install-strategy hint registry.  Each entry is a
# tuple of strategy name + package name + executable name.  When the
# install completes, we look for the executable in
# ``<HERMES_HOME>/lsp/bin/`` first, then on PATH.
INSTALL_RECIPES: Dict[str, Dict[str, str]] = {
    # Python
    "pyright": {"strategy": "npm", "pkg": "pyright", "bin": "pyright-langserver"},
    # JS/TS family
    "typescript-language-server": {
        "strategy": "npm",
        "pkg": "typescript-language-server",
        "bin": "typescript-language-server",
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
}


_install_locks: Dict[str, threading.Lock] = {}
_install_results: Dict[str, Optional[str]] = {}
_install_lock_meta = threading.Lock()


def hermes_lsp_bin_dir() -> Path:
    """Return the Hermes-owned bin staging dir for LSP servers."""
    home = os.environ.get("HERMES_HOME")
    if home is None:
        home = os.path.join(os.path.expanduser("~"), ".hermes")
    p = Path(home) / "lsp" / "bin"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _existing_binary(name: str) -> Optional[str]:
    """Probe the staging dir + PATH for a binary named ``name``."""
    staged = hermes_lsp_bin_dir() / name
    if staged.exists() and os.access(staged, os.X_OK):
        return str(staged)
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
    if strategy not in ("auto",):
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

    # Check if already present (shutil.which or staging dir)
    existing = _existing_binary(bin_name)
    if existing:
        return existing

    if strategy == "manual":
        logger.debug("[install] %s requires manual install (recipe=%s)", pkg, recipe)
        return None

    if strategy == "npm":
        return _install_npm(recipe.get("pkg", pkg), bin_name)
    if strategy == "go":
        return _install_go(recipe.get("pkg", pkg), bin_name)
    if strategy == "pip":
        return _install_pip(recipe.get("pkg", pkg), bin_name)

    logger.warning("[install] unknown strategy %r for %s", strategy, pkg)
    return None


def _install_npm(pkg: str, bin_name: str) -> Optional[str]:
    """Install an npm package into our staging dir.

    Uses ``npm install --prefix`` so the binaries land in
    ``<staging>/node_modules/.bin/<bin_name>`` and we symlink them up
    one level for direct PATH-style access.
    """
    npm = shutil.which("npm")
    if npm is None:
        logger.info("[install] cannot install %s: npm not on PATH", pkg)
        return None
    staging = hermes_lsp_bin_dir().parent  # <HERMES_HOME>/lsp/
    try:
        logger.info("[install] npm install --prefix %s %s", staging, pkg)
        proc = subprocess.run(
            [npm, "install", "--prefix", str(staging), "--silent", "--no-fund", "--no-audit", pkg],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
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
    if os.name == "nt":
        # On Windows npm sometimes drops `.cmd` shims
        candidates = [nm_bin, nm_bin.with_suffix(".cmd")]
    else:
        candidates = [nm_bin]
    for c in candidates:
        if c.exists():
            # Symlink into our `lsp/bin/` for stable PATH access.
            link = hermes_lsp_bin_dir() / c.name
            if not link.exists():
                try:
                    link.symlink_to(c)
                except (OSError, NotImplementedError):
                    # Symlinks fail on some Windows setups — copy instead.
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
            text=True,
            timeout=600,
            env=env,
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
    if os.name == "nt":
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
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", str(pip_target), "--quiet", pkg],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            logger.warning(
                "[install] pip install failed for %s: %s", pkg, proc.stderr.strip()[:500]
            )
            return None
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("[install] pip install errored for %s: %s", pkg, e)
        return None
    # Look for the script
    bin_path = pip_target / "bin" / bin_name
    if bin_path.exists():
        link = hermes_lsp_bin_dir() / bin_name
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
