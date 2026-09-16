"""Auto-installation of LSP server binaries.

Installs go to a Hermes-owned staging dir, ``<HERMES_HOME>/lsp/bin/``, so the
user's global toolchain stays untouched.  Strategies: ``auto`` (install with
the best available package manager), ``manual`` / ``off`` (probe only; a
missing binary skips the server and ``hermes lsp status`` reports it).
Installs run synchronously the first time a server is needed, serialized
per-package; every failure path returns ``None`` so the tool layer falls
back to its in-process syntax checker.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_constants import find_node_executable

logger = logging.getLogger("agent.lsp.install")


def _recipe(strategy: str, pkg: str, bin_name: str, **extra: Any) -> Dict[str, Any]:
    return {"strategy": strategy, "pkg": pkg, "bin": bin_name, **extra}


def _npm(pkg: str, bin_name: str, **extra: Any) -> Dict[str, Any]:
    return _recipe("npm", pkg, bin_name, **extra)


def _manual(bin_name: str) -> Dict[str, Any]:
    return _recipe("manual", "", bin_name)


# Recipe key → {strategy, pkg, bin[, extra_pkgs]}.  After install we look for
# ``bin`` in ``<HERMES_HOME>/lsp/bin/`` first, then on PATH.  ``extra_pkgs``
# are sibling npm packages a server needs in the same node_modules tree.
INSTALL_RECIPES: Dict[str, Dict[str, Any]] = {
    "pyright": _npm("pyright", "pyright-langserver"),
    # tsserver must be importable from the same node_modules tree or
    # initialize() fails with "Could not find a valid TypeScript installation".
    "typescript-language-server": _npm("typescript-language-server", "typescript-language-server", extra_pkgs=["typescript"]),
    "@vue/language-server": _npm("@vue/language-server", "vue-language-server"),
    "svelte-language-server": _npm("svelte-language-server", "svelteserver"),
    "@astrojs/language-server": _npm("@astrojs/language-server", "astro-ls"),
    "yaml-language-server": _npm("yaml-language-server", "yaml-language-server"),
    "bash-language-server": _npm("bash-language-server", "bash-language-server"),
    "intelephense": _npm("intelephense", "intelephense"),
    "dockerfile-language-server-nodejs": _npm("dockerfile-language-server-nodejs", "docker-langserver"),
    "gopls": _recipe("go", "golang.org/x/tools/gopls@latest", "gopls"),
    # Manual: rust-analyzer (via rustup) and clangd (ships with LLVM) are far too
    # heavy to bootstrap; LuaLS is platform-specific GitHub release binaries.
    "rust-analyzer": _manual("rust-analyzer"),
    "clangd": _manual("clangd"),
    "lua-language-server": _manual("lua-language-server"),
    # PowerShellEditorServices is a release-zip bundle driven by pwsh; we probe
    # the host so `hermes lsp status` reports its presence.
    "powershell": _manual("pwsh"),
}

_install_locks: Dict[str, threading.Lock] = {}
_install_results: Dict[str, Optional[str]] = {}
_install_lock_meta = threading.Lock()
_WINDOWS_WRAPPER_SUFFIXES = (".cmd", ".exe", ".bat")


def _is_windows() -> bool:
    return os.name == "nt"


def hermes_lsp_bin_dir() -> Path:
    """Return the Hermes-owned bin staging dir for LSP servers."""
    from hermes_constants import get_hermes_home

    p = get_hermes_home() / "lsp" / "bin"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _native_binary_candidates(base: Path) -> list[Path]:
    """Return platform-native executable candidates for a staged binary (``base`` plus Windows wrappers)."""
    if not _is_windows():
        return [base]
    cands: Dict[str, Path] = {}
    for c in (base, *(Path(str(base) + s) for s in _WINDOWS_WRAPPER_SUFFIXES)):
        cands.setdefault(str(c).lower(), c)
    return list(cands.values())


def _first_existing(*bases: Path) -> Optional[Path]:
    """First platform-native candidate of any ``base`` that exists on disk."""
    return next((c for base in bases for c in _native_binary_candidates(base) if c.exists()), None)


def _existing_binary(name: str) -> Optional[str]:
    """Probe the staging dir + PATH for a binary named ``name``."""
    for staged in _native_binary_candidates(hermes_lsp_bin_dir() / name):
        if staged.exists() and os.access(staged, os.X_OK):
            return str(staged)
    suffixes = ("", *_WINDOWS_WRAPPER_SUFFIXES) if _is_windows() else ("",)
    return next((p for s in suffixes if (p := shutil.which(f"{name}{s}"))), None)


def try_install(pkg: str, strategy: str = "auto") -> Optional[str]:
    """Try to install ``pkg``; return the binary path or ``None``.

    Only ``"auto"`` installs; ``"manual"``/``"off"`` just probe for an existing
    binary.  Results are cached per package and concurrent calls are serialized.
    """
    if strategy != "auto":
        return _existing_binary(INSTALL_RECIPES.get(pkg, {}).get("bin", pkg))
    if pkg in _install_results:
        return _install_results[pkg]
    with _install_lock_meta:
        lock = _install_locks.setdefault(pkg, threading.Lock())
    with lock:
        if pkg not in _install_results:
            _install_results[pkg] = _do_install(pkg)
        return _install_results[pkg]


def _do_install(pkg: str) -> Optional[str]:
    recipe = INSTALL_RECIPES.get(pkg)
    if recipe is None:
        return shutil.which(pkg)  # not in our registry — best-effort: just probe PATH
    strategy = recipe.get("strategy", "manual")
    bin_name = recipe.get("bin", pkg)
    if existing := _existing_binary(bin_name):
        return existing
    if strategy == "manual":
        logger.debug("[install] %s requires manual install (recipe=%s)", pkg, recipe)
        return None
    installer = _INSTALLERS.get(strategy)
    if installer is None:
        logger.warning("[install] unknown strategy %r for %s", strategy, pkg)
        return None
    return installer(recipe, bin_name)


def _run_installer(tool: str, pkg: str, cmd: list, *, timeout: int, env: Optional[dict] = None) -> bool:
    """Run one install subprocess; log and return False on non-zero exit or error."""
    try:
        proc = subprocess.run(
            cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, env=env, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags(),
        )
        if proc.returncode != 0:
            logger.warning("[install] %s install failed for %s: %s", tool, pkg, proc.stderr.strip()[:500])
            return False
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("[install] %s install errored for %s: %s", tool, pkg, e)
        return False
    return True


def _link_into_bin(target: Path) -> str:
    """Symlink (or copy, where symlinks fail) ``target`` into ``lsp/bin/`` and return the path to use."""
    link = hermes_lsp_bin_dir() / target.name
    if not link.exists():
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            # Symlinks fail on some Windows setups — copy instead.
            try:
                shutil.copy2(target, link)
            except OSError:
                return str(target)
    return str(link if link.exists() else target)


def _install_npm(pkg: str, bin_name: str, extra_pkgs: Optional[list] = None) -> Optional[str]:
    """``npm install --prefix <staging>`` then link ``node_modules/.bin/<bin_name>`` into ``lsp/bin/``."""
    # Managed npm first: $HERMES_HOME/node isn't on an arbitrary process's
    # PATH, so a bare which() would miss the Node that Hermes installed.
    npm = find_node_executable("npm")
    if npm is None:
        logger.info("[install] cannot install %s: no usable npm found", pkg)
        return None
    staging = hermes_lsp_bin_dir().parent  # <HERMES_HOME>/lsp/
    install_targets = [pkg] + list(extra_pkgs or [])
    logger.info("[install] npm install --prefix %s %s", staging, " ".join(install_targets))
    cmd = [npm, "install", "--prefix", str(staging), "--silent", "--no-fund", "--no-audit", *install_targets]
    if not _run_installer("npm", pkg, cmd, timeout=300):
        return None
    found = _first_existing(staging / "node_modules" / ".bin" / bin_name)
    if found is not None:
        return _link_into_bin(found)
    logger.warning("[install] npm install for %s succeeded but bin %s not found", pkg, bin_name)
    return None


def _install_go(pkg: str, bin_name: str) -> Optional[str]:
    """Install a Go module to GOBIN=<staging>."""
    go = shutil.which("go")
    if go is None:
        logger.info("[install] cannot install %s: go not on PATH", pkg)
        return None
    staging = hermes_lsp_bin_dir()
    logger.info("[install] go install %s (GOBIN=%s)", pkg, staging)
    if not _run_installer("go", pkg, [go, "install", pkg], timeout=600, env={**os.environ, "GOBIN": str(staging)}):
        return None
    bin_path = (staging / bin_name).with_suffix(".exe") if _is_windows() else staging / bin_name
    if bin_path.exists():
        return str(bin_path)
    logger.warning("[install] go install for %s succeeded but bin %s not found", pkg, bin_name)
    return None


def _install_pip(pkg: str, bin_name: str) -> Optional[str]:
    """``pip install --target <staging>/python-packages`` then link the console script into ``lsp/bin/``."""
    pip_target = hermes_lsp_bin_dir().parent / "python-packages"
    pip_target.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("[install] pip install --target %s %s", pip_target, pkg)
        from hermes_cli.tools_config import _pip_install

        proc = _pip_install(["--target", str(pip_target), "--quiet", pkg], timeout=300)
        if proc.returncode != 0:
            logger.warning("[install] pip install failed for %s: %s", pkg, (proc.stderr or "").strip()[:500])
            return None
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("[install] pip install errored for %s: %s", pkg, e)
        return None
    # POSIX wheels write console scripts to bin/, native Windows to Scripts/.
    script_dirs = [pip_target / "bin"] + ([pip_target / "Scripts"] if _is_windows() else [])
    found = _first_existing(*(d / bin_name for d in script_dirs))
    return _link_into_bin(found) if found is not None else None


# strategy → installer(recipe, bin_name).  ``manual`` is handled before dispatch.
_INSTALLERS: Dict[str, Callable[[Dict[str, Any], str], Optional[str]]] = {
    "npm": lambda r, b: _install_npm(r["pkg"], b, extra_pkgs=r.get("extra_pkgs") or []),
    "go": lambda r, b: _install_go(r["pkg"], b),
    "pip": lambda r, b: _install_pip(r["pkg"], b),
}


def detect_status(pkg: str) -> str:
    """Return ``installed``, ``missing``, or ``manual-only`` (for ``hermes lsp status``; spawns nothing)."""
    recipe = INSTALL_RECIPES.get(pkg)
    if _existing_binary(recipe.get("bin", pkg) if recipe else pkg):
        return "installed"
    return "manual-only" if recipe and recipe.get("strategy") == "manual" else "missing"


__all__ = ["INSTALL_RECIPES", "try_install", "detect_status", "hermes_lsp_bin_dir"]
