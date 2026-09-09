"""Guard: Hermes-owned subprocesses must not resolve managed runtimes by bare PATH.

Hermes installs runtimes for itself — ``uv`` at ``$HERMES_HOME/bin/uv``, Node at
``$HERMES_HOME/node``. Neither directory is on the ambient PATH of an arbitrary
process, so ``shutil.which("uv")`` / ``shutil.which("node")`` in Hermes's own
code has two failure modes:

* the managed runtime is invisible, so the caller reports "not installed" or
  degrades to a slower tier on a machine that has exactly what it needed; and
* when a system copy also exists, the one Hermes does not own wins — which is
  how a generated systemd unit or launchd plist can bake a system Node in and
  keep resolving it across reboots.

The fix per call site is one of ``find_node_executable()``,
``iter_hermes_node_dirs()``, ``resolve_uv()``, or ``ensure_uv()``. This test is
the ratchet that stops a new bare lookup from being added back.

Reading source is normally banned (see AGENTS.md). It is the right tool here and
only here: the property under test is "no call site anywhere in the tree spells
it this way", which is a statement about the whole codebase rather than about
one function's behavior, and there is no runtime seam that can observe a lookup
that was never written. Every entry in the allow-list below names a call site
whose behavior is separately covered by a real test.
"""

from __future__ import annotations

import ast
import functools
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Runtimes Hermes provisions into HERMES_HOME and must therefore resolve
# through a managed-aware helper rather than PATH.
_MANAGED_COMMANDS = frozenset({"uv", "node", "npm", "npx"})

# Directories that are not Hermes-owned subprocess code: plugins ship their own
# resolution policy, tests assert against PATH deliberately, and skills/scripts
# run as standalone user-invoked programs.
_EXEMPT_DIRS = (
    "tests",
    "plugins",
    "skills",
    "optional-skills",
    "scripts",
    "website",
    "node_modules",
    ".git",
    ".venv",
    "venv",
    ".worktrees",
)

# Call sites where a bare PATH lookup is the correct answer. Each entry is
# (path, command) -> why. Keep this list short and justified — the default
# answer for a new call site is a managed-aware helper, not a new exemption.
_ALLOWED: dict[tuple[str, str], str] = {
    ("tools/env_probe.py", "uv"): (
        "Reports the environment the MODEL sees in the terminal tool. The model "
        "can only run what is on that subshell's PATH, which local.py populates "
        "with the managed dirs — so PATH is the correct question to ask here."
    ),
    ("hermes_cli/update_cmd_deps.py", "uv"): (
        "Termux fallback: a pkg-installed uv lands on PATH but not in the "
        "managed bin dir, and it is checked only after resolve_uv() misses."
    ),
    ("hermes_cli/update_cmd_deps.py", "npm"): (
        "WSL diagnostic: deliberately inspects what PATH resolves so it can "
        "warn that the only reachable npm is the Windows one."
    ),
    ("tools/lazy_deps.py", "uv"): (
        "Fallback after resolve_uv(), plus the except-branch for the "
        "hermes_cli import guard."
    ),
    ("tools/browser_use_cli.py", "uv"): (
        "install_cli()'s fallback after ensure_uv() misses — a user-installed "
        "uv on PATH is a legitimate last rung before giving up with install "
        "guidance."
    ),
    ("hermes_cli/gateway.py", "node"): (
        "Fallback rung of _append_node_dir_for_service(), after the managed "
        "dirs from iter_hermes_node_dirs() are already appended."
    ),
    ("hermes_cli/main_tui_launch.py", "node"): (
        "_ensure_tui_node()'s idempotence gate: the question really is 'is "
        "node already discoverable on PATH', before bootstrapping one."
    ),
    ("hermes_cli/main_tui_launch.py", "npm"): (
        "Same _ensure_tui_node() gate as node."
    ),
    ("hermes_cli/main_install_repair.py", "npm"): (
        "_resolve_node_runtime_npm()'s WSL re-scan: PATH minus /mnt/* IS the question."
    ),
    ("tools/browser_tool_install.py", "npx"): (
        "agent-browser runs via `npx`, resolved against the extended browser "
        "PATH that _merge_browser_path() already seeds with the managed dirs."
    ),
}


def _iter_which_calls(tree: ast.AST):
    """Yield (command, lineno) for every ``which("<cmd>")`` call in *tree*.

    AST rather than a regex so prose in docstrings and comments that mentions
    ``shutil.which("npm")`` is not mistaken for a call site.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else None
        )
        if name != "which":
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value in _MANAGED_COMMANDS:
            yield first.value, node.lineno


def _source_files() -> list[Path]:
    files: list[Path] = []
    # os.walk instead of Path.rglob: rglob raises FileNotFoundError when a
    # directory vanishes mid-scan — a sibling CI job's sdist extraction
    # (hermes_agent-<version>/) gets created and deleted concurrently, and
    # that TOCTOU failed this guard on runs 33531869442/33455779041-era
    # workspaces. os.walk tolerates vanishing dirs (onerror=None), and
    # pruning exempt/packaging dirs at the top level also skips their
    # subtrees entirely.
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        rel_dir = Path(dirpath).relative_to(REPO_ROOT)
        if rel_dir == Path("."):
            dirnames[:] = [
                d for d in dirnames
                if d not in _EXEMPT_DIRS and not _is_packaging_copy(d)
            ]
        for fname in filenames:
            if fname.endswith(".py"):
                files.append(Path(dirpath) / fname)
    return files


@functools.lru_cache(maxsize=None)
def _is_packaging_copy(top_level: str) -> bool:
    """Whether *top_level* (a dir name under REPO_ROOT) is a build artifact."""
    if top_level in ("build", "dist") or top_level.endswith(".egg-info"):
        return True
    candidate = REPO_ROOT / top_level
    if not candidate.is_dir():
        return False
    return (candidate / "PKG-INFO").exists()


def _findings() -> list[tuple[str, str, int]]:
    """Return (relpath, command, lineno) for every bare managed lookup."""
    found: list[tuple[str, str, int]] = []
    for path in _source_files():
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "which(" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for command, lineno in _iter_which_calls(tree):
            found.append((rel, command, lineno))
    return found


def test_no_unreviewed_bare_managed_runtime_lookups():
    """Every bare which() for a managed runtime is a reviewed exemption."""
    unexpected = [
        (rel, cmd, lineno)
        for rel, cmd, lineno in _findings()
        if (rel, cmd) not in _ALLOWED
    ]

    assert not unexpected, (
        "Bare PATH lookup for a Hermes-managed runtime.\n\n"
        + "\n".join(f"  {rel}:{lineno}  which({cmd!r})" for rel, cmd, lineno in unexpected)
        + "\n\n$HERMES_HOME/bin (uv) and $HERMES_HOME/node are not on an "
        "arbitrary process's PATH, so this resolves a system copy — or nothing "
        "— on an install that has a managed one.\n"
        "Use instead:\n"
        "  uv       -> managed_uv.resolve_uv() (lookup) or ensure_uv() (may install)\n"
        "  node/npm -> hermes_constants.find_node_executable()\n"
        "  PATH env -> hermes_constants.iter_hermes_node_dirs()\n"
        "If PATH really is the right question, add the site to _ALLOWED with a "
        "reason."
    )


def test_allowlist_has_no_stale_entries():
    """A fixed call site must be dropped from the allow-list, not left to rot."""
    live = {(rel, cmd) for rel, cmd, _lineno in _findings()}
    stale = sorted(set(_ALLOWED) - live)

    assert not stale, (
        "Allow-list entries no longer match any source line — the call site was "
        "fixed or moved. Remove them:\n"
        + "\n".join(f"  {rel} ({cmd})" for rel, cmd in stale)
    )


@pytest.mark.parametrize(
    "helper",
    [
        "find_node_executable",
        "find_hermes_node_executable",
        "iter_hermes_node_dirs",
        "with_hermes_node_path",
    ],
)
def test_managed_node_helpers_exist(helper):
    """The alternatives this guard points contributors at must be importable."""
    import hermes_constants

    assert callable(getattr(hermes_constants, helper))


def test_managed_uv_helpers_exist():
    from hermes_cli.managed_uv import ensure_uv, managed_uv_path, resolve_uv

    assert callable(resolve_uv)
    assert callable(ensure_uv)
    assert managed_uv_path().parent.name == "bin"
