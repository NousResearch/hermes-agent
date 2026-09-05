"""Deterministic plugin-tree identity + static hook-registration inventory.

Content-consent (G1): the identity of "what the user consented to" is bound to
the plugin's *artifact*, not to its declared identity (``name`` / ``version`` /
``capabilities`` / ``provides_hooks``). Two checkouts of the same artifact hash
identically, and any drift — a ``git pull``, an out-of-band write, a same-user
writer from another profile — changes the identity and re-enters the consent
path.

* Git checkouts bind consent to the **canonical git tree identity**
  (:func:`git_tree_id`, ``git rev-parse HEAD^{tree}``): one tamper-evident id
  over every tracked entry's bytes + mode + path + type. Nothing git tracks is
  excluded — a tracked ``*.pyc`` or a ``100644 → 100755`` mode flip moves the
  identity even though both are invisible to a byte-only walk.
* Non-git / manual trees fall back to the byte-level whole-tree
  :func:`tree_sha256`, where editor/OS noise exclusions ARE applied (there is no
  index to separate user noise from artifact content).

Determinism is load-bearing (design §4): hashing ``.git`` churn or editor temp
files makes every update look like drift and trains the user to click ``y`` —
the exact fatigue failure the HookPry class exploits. Exclusions in the
whole-tree fallback mirror ``tools.plugin_guard.EXCLUDED_DIRS`` plus ``*.pyc`` /
editor temp so a manual baseline is stable across checkouts.

The registration scanners are static (AST) on purpose: an update-diff review
must show added/removed/changed ``ctx.register_*`` call sites *without importing
plugin code* — importing runs it, which is what the review exists to gate.
"""

from __future__ import annotations

import ast
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple

# Hook/tool/command bindings the update-diff review surfaces as host-surface
# registrations. A diff that adds one of these changes what the plugin binds to
# runtime events — the payload of the HookPry "Temporal Decoupling" attack.
_REGISTRATION_METHODS = (
    "register_hook",
    "register_tool",
    "register_middleware",
    "register_cli_command",
)

# Filenames/suffixes that are editor/OS noise, never plugin content. Kept small
# and obvious; anything else in the tree is content and must be hashed.
_EDITOR_TEMP_SUFFIXES = ("~", ".swp", ".swo", ".swn")
_OS_JUNK_NAMES = {".DS_Store", "Thumbs.db"}

_KIND_FILE = b"F"
_KIND_SYMLINK = b"L"


def _is_excluded(rel_parts: Tuple[str, ...]) -> bool:
    """True when a tree entry is VCS/cache/env noise (mirror of the scanner's walk)."""
    from tools.plugin_guard import EXCLUDED_DIRS

    return any(part in EXCLUDED_DIRS for part in rel_parts)


def _is_noise_file(name: str) -> bool:
    """True for bytecode / editor-temp / OS-junk filenames that must not move the hash."""
    if name.endswith(".pyc"):
        return True
    if name.endswith(_EDITOR_TEMP_SUFFIXES):
        return True
    if name.startswith("#") and name.endswith("#"):  # emacs autosave
        return True
    return name in _OS_JUNK_NAMES


def _tracked_relpaths(plugin_dir: Path, git_exe: Optional[str]) -> Optional[List[str]]:
    """Indexed (tracked) file paths of a git checkout, or ``None`` when unavailable.

    Hashing the tracked set (not the raw directory) keeps the consent baseline
    stable against untracked noise — e.g. the ``*.example`` -> real-name copies
    ``cmd_install`` leaves in the tree — so an unchanged checkout never drifts.
    """
    exe = git_exe or "git"
    try:
        result = subprocess.run(
            [exe, "ls-files", "-z"],
            cwd=str(plugin_dir),
            capture_output=True,
            text=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return [
        p
        for p in result.stdout.decode("utf-8", "surrogateescape").split("\0")
        if p
    ]


def git_tree_id(plugin_dir: Path, git_exe: Optional[str] = None) -> Optional[str]:
    """Canonical git tree identity of *plugin_dir*'s HEAD — ``git rev-parse HEAD^{tree}``.

    The git tree object id is a hash over every tracked entry's path, type
    (blob/tree/commit-link), mode (``100644`` vs ``100755``), and bytes — one
    tamper-evident identity that byte-only walks cannot express (mode flips,
    tracked bytecode). Returns ``None`` when the directory is not a usable git
    checkout (callers fall back to :func:`tree_sha256`).
    """
    exe = git_exe or "git"
    try:
        result = subprocess.run(
            [exe, "rev-parse", "HEAD^{tree}"],
            cwd=str(plugin_dir),
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip().lower()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        return None
    return value


def tree_sha256(plugin_dir: Path, *, tracked_only: bool = False, git_exe: Optional[str] = None) -> str:
    """sha256 over sorted ``(relpath, content)`` pairs of a plugin tree.

    * ``tracked_only=True`` hashes the git index file set of *plugin_dir* — the
      bytes an update can change. **No exclusion applies to tracked entries**:
      ``git ls-files`` has already separated user noise from artifact content,
      so everything it returns is hashed (a tracked ``*.pyc`` is artifact
      content, not noise). Use for git checkouts that need a byte-level digest
      (the consent anchor for git trees is :func:`git_tree_id`; this mode
      detects out-of-band working-tree edits it cannot see).
    * ``tracked_only=False`` hashes the whole directory tree minus exclusions —
      use for non-git plugin trees (manual baselines).
    * Symlinks are hashed as their link-target string (git's own model), so a
      retargeted link drifts without following content outside the tree.

    Deterministic by construction: sorted paths, file bytes only, no timestamps,
    no mode bits, no bytecode. Non-reproducible hashing would turn every update
    into a spurious re-consent prompt (see module docstring).
    """
    entries: List[Tuple[str, bytes]] = []  # (rel_posix, payload)

    relpaths: Optional[List[str]] = _tracked_relpaths(plugin_dir, git_exe) if tracked_only else None
    if relpaths is not None:
        # Git already separated artifact from noise: no _is_excluded / _is_noise_file
        # re-application — a tracked entry is artifact content by definition.
        for rel_posix in sorted(relpaths):
            payload = _entry_payload(plugin_dir, rel_posix)
            if payload is not None:
                entries.append((rel_posix, payload))
    else:
        # Whole-tree walk (non-git tree, or git unavailable): deterministic and
        # mirrors plugin_guard's own exclusion set so hash and scan agree.
        for path in plugin_dir.rglob("*"):
            try:
                rel = path.relative_to(plugin_dir)
            except ValueError:
                continue
            if _is_excluded(rel.parts):
                continue
            if rel.parts and _is_noise_file(rel.parts[-1]):
                continue
            if path.is_dir() and not path.is_symlink():
                continue
            rel_posix = rel.as_posix()
            payload = _entry_payload(plugin_dir, rel_posix)
            if payload is not None:
                entries.append((rel_posix, payload))
        entries.sort(key=lambda item: item[0])

    digest = hashlib.sha256()
    for rel_posix, payload in entries:
        rel_bytes = rel_posix.encode("utf-8", "surrogateescape")
        digest.update(len(rel_bytes).to_bytes(8, "big"))
        digest.update(rel_bytes)
        digest.update(payload)
    return digest.hexdigest()


def _entry_payload(plugin_dir: Path, rel_posix: str) -> Optional[bytes]:
    """Bytes hashed for one tree entry: file content or symlink target string."""
    path = plugin_dir / rel_posix
    try:
        if path.is_symlink():
            return _KIND_SYMLINK + os.readlink(path).encode("utf-8", "surrogateescape")
        if path.is_file():
            return _KIND_FILE + path.read_bytes()
    except OSError:
        return None
    return None


# ── static hook-registration inventory ───────────────────────────────────────

RegistrationCall = Tuple[int, str, str]  # (lineno, method, label)


def _first_string_literal(node: ast.Call) -> Optional[str]:
    """The first string literal at the call site: positional args, then keyword values.

    ``None`` when the binding name is computed (``register_hook(event_name, ...)``),
    in which case the call is still counted but carries no literal label.
    """
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    for kw in node.keywords:
        if kw.arg is not None and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def registered_hook_event_names(source: str) -> Set[str]:
    """String hook event names bound by ``X.register_hook(...)`` calls in *source*.

    Static AST only — never imports the module under review. Recognizes any
    receiver expression (``ctx.register_hook``, ``self.ctx.register_hook``,
    ``foo.bar.register_hook``) and both positional and keyword forms
    (``register_hook('pre_tool_call', cb)`` and
    ``register_hook(event='pre_tool_call', handler=cb)``). Unreadable source
    → empty set (never raises).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "register_hook":
            continue
        name = _first_string_literal(node)
        if name is not None:
            names.add(name)
    return names


def scan_registration_calls(source: str) -> List[RegistrationCall]:
    """``(lineno, method, label)`` for every ``X.register_*`` call site in *source*.

    Static AST only — never imports the module under review. The receiver may
    be ANY expression (``ctx.register_hook``, ``self.ctx.register_hook``,
    ``foo.bar.register_hook`` — attribute receivers are how real plugins bind
    hooks from helper objects). *label* is the first string literal at the
    call site (positional first, then keyword values — so
    ``register_hook(event='on_session_end', handler=h)`` labels as
    ``'on_session_end'``), rendered with quotes; ``''`` when the binding name
    is computed. Enough to tell ``register_hook('pre_tool_call')`` from
    ``register_hook('on_session_start')`` in a review diff.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    calls: Set[RegistrationCall] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _REGISTRATION_METHODS:
            continue
        label = ""
        first_literal = _first_string_literal(node)
        if first_literal is not None:
            label = repr(first_literal)
        calls.add((node.lineno, node.func.attr, label))
    return sorted(calls)


def _call_key(call: RegistrationCall) -> Tuple[str, str]:
    """Registration identity for diffing: a binding is (method, label), not a line."""
    return (call[1], call[2])


def diff_registration_inventories(old_sources: Mapping[str, str], new_sources: Mapping[str, str]) -> List[str]:
    """Human review lines for added/removed ``register_*`` calls across files.

    *old_sources* / *new_sources* map repo-relative ``.py`` paths to their text
    at the old/new revision. Files only in one side are diffed against empty —
    a new file's registrations all read as added.
    """
    lines: List[str] = []
    for rel in sorted(set(old_sources) | set(new_sources)):
        old_calls = scan_registration_calls(old_sources.get(rel, ""))
        new_calls = scan_registration_calls(new_sources.get(rel, ""))
        old_by_key = {_call_key(c): c for c in old_calls}
        new_by_key = {_call_key(c): c for c in new_calls}
        for key in sorted(new_by_key.keys() - old_by_key.keys()):
            lineno, method, label = new_by_key[key]
            lines.append(f"  + {rel}:{lineno} ctx.{method}({label})")
        for key in sorted(old_by_key.keys() - new_by_key.keys()):
            lineno, method, label = old_by_key[key]
            lines.append(f"  - {rel}:{lineno} ctx.{method}({label})")
    return lines


def diff_manifest_hook_declarations(old_manifest: Mapping[str, object], new_manifest: Mapping[str, object]) -> List[str]:
    """Human review lines for added/removed ``provides_hooks`` / ``hooks:`` items."""
    lines: List[str] = []
    for key in ("provides_hooks", "hooks"):
        old_items = _hook_decl_items(old_manifest.get(key))
        new_items = _hook_decl_items(new_manifest.get(key))
        for item in sorted(new_items - old_items):
            lines.append(f"  + {key}: {item}")
        for item in sorted(old_items - new_items):
            lines.append(f"  - {key}: {item}")
    return lines


def _hook_decl_items(value: object) -> Set[str]:
    """Normalize a manifest ``provides_hooks`` / ``hooks`` value to comparable items."""
    if isinstance(value, dict):
        return {f"{k} -> {v}" for k, v in value.items()}
    if isinstance(value, list):
        return {str(v) for v in value if isinstance(v, str)}
    return set()


__all__ = [
    "git_tree_id",
    "tree_sha256",
    "scan_registration_calls",
    "registered_hook_event_names",
    "diff_registration_inventories",
    "diff_manifest_hook_declarations",
]
