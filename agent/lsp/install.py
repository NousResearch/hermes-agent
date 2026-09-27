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
from hermes_constants import find_node_executable, with_hermes_node_path

logger = logging.getLogger("agent.lsp.install")


def _recipe(strategy: str, pkg: str, bin_name: str, **extra: Any) -> Dict[str, Any]:
    return {"strategy": strategy, "pkg": pkg, "bin": bin_name, **extra}


def _npm(pkg: str, bin_name: str, **extra: Any) -> Dict[str, Any]:
    return _recipe("npm", pkg, bin_name, **extra)


def _manual(bin_name: str) -> Dict[str, Any]:
    return _recipe("manual", "", bin_name)


# TypeScript 7+ is the Go-native port and ships no ``lib/tsserver.js`` /
# ``lib/typescript.js``, so JS-based servers cannot load it as their SDK.
TYPESCRIPT_SDK_PKG = "typescript@6"

# Recipe key → {strategy, pkg, bin[, extra_pkgs]}.  After install we look for
# ``bin`` in ``<HERMES_HOME>/lsp/bin/`` first, then on PATH.  ``extra_pkgs``
# are sibling npm packages a server needs in the same node_modules tree.
INSTALL_RECIPES: Dict[str, Dict[str, Any]] = {
    "pyright": _npm("pyright", "pyright-langserver"),
    # tsserver must be importable from the same node_modules tree or
    # initialize() fails with "Could not find a valid TypeScript installation".
    "typescript-language-server": _npm("typescript-language-server", "typescript-language-server", extra_pkgs=[TYPESCRIPT_SDK_PKG]),
    # 3.x forwards every TypeScript request to a client-hosted tsserver
    # (``tsserver/request`` tunnel) that a generic LSP client does not run, so
    # it never publishes diagnostics; 2.x self-hosts TypeScript from
    # ``initializationOptions.typescript.tsdk`` (see servers._spawn_vue).
    "@vue/language-server": _npm("@vue/language-server@2", "vue-language-server", extra_pkgs=[TYPESCRIPT_SDK_PKG]),
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
    # laravel-lsp ships via composer (`composer global require laravel/lsp`), not npm.
    "laravel-lsp": _manual("laravel-lsp"),
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


def _native_binary_candidates(base: Path, *, is_windows: Optional[bool] = None) -> list[Path]:
    """Return platform-native executable candidates for a staged binary, most runnable first.

    On Windows the ``.cmd``/``.exe``/``.bat`` wrappers come BEFORE the bare name: npm writes a
    POSIX ``#!/bin/sh`` shim under the bare name next to its ``.cmd``, ``os.access(X_OK)`` is
    always true there, and ``CreateProcess`` on the shim fails with WinError 193.  The bare name
    stays as a last resort for genuinely extension-less executables.
    """
    if not (_is_windows() if is_windows is None else is_windows):
        return [base]
    cands: Dict[str, Path] = {}
    for c in (*(Path(str(base) + s) for s in _WINDOWS_WRAPPER_SUFFIXES), base):
        cands.setdefault(str(c).lower(), c)
    return list(cands.values())


def _first_existing(*bases: Path, is_windows: Optional[bool] = None) -> Optional[Path]:
    """First platform-native candidate of any ``base`` that exists on disk."""
    return next((c for base in bases for c in _native_binary_candidates(base, is_windows=is_windows) if c.exists()), None)


def _npm_bin_dir() -> Path:
    """npm's own ``node_modules/.bin`` under the staging tree, where its ``%~dp0``-relative wrappers work."""
    return hermes_lsp_bin_dir().parent / "node_modules" / ".bin"


def _existing_binary(name: str, *, is_windows: Optional[bool] = None) -> Optional[str]:
    """Probe the staging dir (+ npm's bin dir on Windows) then PATH for a binary named ``name``.

    ``is_windows`` overrides the host check so the Windows resolution is testable as data on every lane.
    """
    win = _is_windows() if is_windows is None else is_windows
    bases = [hermes_lsp_bin_dir() / name] + ([_npm_bin_dir() / name] if win else [])
    for staged in (c for base in bases for c in _native_binary_candidates(base, is_windows=win)):
        if staged.exists() and os.access(staged, os.X_OK):
            return str(staged)
    if any(r.get("strategy") == "pip" and r.get("bin") == name for r in INSTALL_RECIPES.values()):
        import pm

        try:
            binary = pm.python_tool(f"lsp-{name}", name)
        except RuntimeError as exc:
            logger.warning("[install] cannot read Python server %s: %s", name, exc)
        else:
            if binary is not None:
                return str(binary)
    suffixes = (*_WINDOWS_WRAPPER_SUFFIXES, "") if win else ("",)
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
            # pnpm reports ERR_PNPM_* on stdout with an empty stderr; log whichever stream carries the reason.
            detail = (proc.stderr.strip() or proc.stdout.strip())[:500]
            logger.warning("[install] %s install failed for %s: %s", tool, pkg, detail)
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


# Node package manager → argv that installs into ``<staging>/node_modules`` (``lsp.package_manager``).
# Every manager keeps the staging-dir semantics: nothing touches the user's project or global tree.
_NODE_PM_ARGV: Dict[str, Callable[[str], list]] = {
    "npm": lambda staging: ["install", "--prefix", staging, "--silent", "--no-fund", "--no-audit"],
    "pnpm": lambda staging: ["add", "--dir", staging],
    # Global ``--cwd`` (before the command) is accepted by both Yarn Classic and Yarn Berry; Berry's
    # default PnP linker writes no ``node_modules/.bin``, so the staging dir needs ``nodeLinker: node-modules``.
    "yarn": lambda staging: ["--cwd", staging, "add"],
}


def _node_package_manager() -> Optional[str]:
    """``lsp.package_manager`` from config (npm default); an unknown value fails closed (``None``)."""
    try:
        from hermes_cli.config import load_config_readonly
        lsp_cfg = load_config_readonly().get("lsp") or {}
    except Exception:  # noqa: BLE001 — installer must not die on a broken config; npm is the historical default
        return "npm"
    pm = str(lsp_cfg.get("package_manager") or "npm").strip().lower() if isinstance(lsp_cfg, dict) else "npm"
    if pm not in _NODE_PM_ARGV:
        # Fail closed: a typo must not silently bypass a pnpm/yarn supply-chain policy by running npm.
        logger.warning("[install] lsp.package_manager=%r is not one of %s; skipping install", pm, sorted(_NODE_PM_ARGV))
        return None
    return pm


def _install_npm(pkg: str, bin_name: str, extra_pkgs: Optional[list] = None) -> Optional[str]:
    """Install with the configured Node package manager into ``<staging>`` and link
    ``node_modules/.bin/<bin_name>`` into ``lsp/bin/``."""
    pm = _node_package_manager()
    if pm is None:
        return None
    # Managed Node first: $HERMES_HOME/node isn't on an arbitrary process's
    # PATH, so a bare which() would miss the Node that Hermes installed.
    pm_bin = find_node_executable(pm)
    if pm_bin is None:
        # Deliberately no silent fallback to npm: a pnpm/yarn choice is usually a supply-chain policy.
        logger.warning("[install] cannot install %s: lsp.package_manager is %r but no usable %s was found "
                       "(install it, or set lsp.package_manager: npm)", pkg, pm, pm)
        return None
    staging = hermes_lsp_bin_dir().parent  # <HERMES_HOME>/lsp/
    install_targets = [pkg] + list(extra_pkgs or [])
    cmd = [pm_bin, *_NODE_PM_ARGV[pm](str(staging)), *install_targets]
    logger.info("[install] %s %s", pm, " ".join(cmd[1:]))
    if not _run_installer(pm, pkg, cmd, timeout=300, env=with_hermes_node_path()):
        return None
    found = _first_existing(staging / "node_modules" / ".bin" / bin_name)
    if found is not None:
        # npm's Windows wrappers resolve their payload via ``%~dp0\..\<pkg>``, so a copy or symlink
        # in ``lsp/bin/`` points at nothing; use them where npm put them (``_existing_binary`` probes there).
        return str(found) if _is_windows() and found.suffix.lower() in (".cmd", ".bat") else _link_into_bin(found)
    logger.warning("[install] %s install for %s succeeded but bin %s not found", pm, pkg, bin_name)
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
    """Provision a Python server in its own PM-managed environment."""
    try:
        import pm

        return str(pm.ensure_python_tool(f"lsp-{bin_name}", [pkg], bin_name, timeout=300))
    except Exception as exc:
        logger.warning("[install] Python server install failed for %s: %s", pkg, exc)
        return None


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
