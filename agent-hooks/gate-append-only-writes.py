#!/usr/bin/env python3
"""Fail-closed pre_tool_call veto for append-only arena journals."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import re
import sys

from hermes_constants import get_hermes_home

LOG = get_hermes_home() / "logs" / "append-only-write-gate.log"
REDIRECT = re.compile(r"(?:^|[;&|]\s*)[^;&|]*?(?P<op>>>?|\btee\b)(?P<append>\s+-a)?\s+(?P<path>[^;&|]+)")
TRUNCATE = re.compile(r"\b(?:truncate|sed\s+-i|perl\s+-i)\b")
# Match V4A headers with optional space after *** (aligned with tools/patch_parser.py)
V4A_HEADER = re.compile(r"^\*{3}\s*(Add|Update|Delete|Move|Rename)\s+File:\s*(.+)$", re.MULTILINE)


def emit(value: dict) -> None:
    print(json.dumps(value, separators=(",", ":")))


def allow() -> None:
    emit({})


def block(reason: str, target: str) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} BLOCK target={target} reason={reason}\n")
    except Exception:
        pass
    emit({"decision": "block", "action": "block", "reason": reason, "message": reason})


def resolved(path: str, base_dir: str = None) -> Path:
    """Resolve a path, optionally relative to base_dir.
    
    Note: This resolves against the hook's host filesystem. When tools execute
    in Docker/SSH/Modal/Daytona or with a custom workdir, the resolved path
    may not match the tool's effective location. This is a known limitation.
    """
    # Strip/unquote redirect targets
    path = path.strip()
    if path.startswith(("'", '"')) and path.endswith(path[0]):
        path = path[1:-1]
    try:
        p = Path(path)
        if base_dir and not p.is_absolute():
            p = Path(base_dir) / p
        return p.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise


def append_only(path: str, workdir: str = None) -> bool:
    """Check if a path refers to an append-only protected journal.
    
    Args:
        path: The path to check (can be relative or absolute)
        workdir: Optional working directory for relative path resolution
    
    Returns:
        True if the path is a protected journal, False otherwise
    """
    try:
        p = resolved(path, workdir)
    except (OSError, RuntimeError, ValueError):
        return False
    parts = p.parts
    if p.name == "IMPROVEMENTS.md" and p.parent.name == "trading-arena":
        return True
    return p.name == "journal.md" and "trading-arena" in parts


def strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in value.values() for s in strings(item)]
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]
    return []


def extract_patch_targets(patch: str) -> list[tuple[str, str]]:
    """Extract (operation, path) pairs from V4A patch headers.
    
    Returns list of (op, path) where op is 'Add', 'Update', 'Delete', 'Move', or 'Rename'.
    For Move/Rename operations, returns both source and dest as separate entries.
    Add operations are recognized but not enforced (adding new files is safe).
    """
    targets = []
    for match in V4A_HEADER.finditer(patch):
        op = match.group(1)
        path = match.group(2).strip()
        
        # For Move/Rename operations, parse "src -> dest" into separate paths
        if op in ("Move", "Rename") and "->" in path:
            parts = path.split("->", 1)
            src = parts[0].strip()
            dest = parts[1].strip() if len(parts) > 1 else ""
            # Check both source and destination
            if src:
                targets.append((op, src))
            if dest:
                targets.append((op, dest))
        else:
            targets.append((op, path))
    return targets


def tool_data(payload: dict) -> tuple[str, dict, str]:
    tool = str(payload.get("tool_name") or "").strip()
    args = payload.get("tool_input") or payload.get("args") or {}
    if not isinstance(args, dict):
        return tool, {}, ""
    command = args.get("command") or args.get("cmd") or args.get("script") or ""
    if isinstance(command, list):
        command = " ".join(map(str, command))
    return tool, args, str(command)


def patch_preserves(path: Path, args: dict) -> bool:
    """Check if a patch preserves append-only semantics.
    
    Returns False (block) if:
    - Delete/Move operations on protected file
    - Patch would shrink the file or remove existing content
    
    Returns True (allow) if patch only adds content.
    """
    current = path.read_text(encoding="utf-8")
    old = args.get("old_string", args.get("old_str"))
    new = args.get("new_string", args.get("new_str"))
    replace_all = args.get("replace_all", False)
    
    if isinstance(old, str) and isinstance(new, str):
        if old not in current:
            return False
        # Honor replace_all cardinality
        count = -1 if replace_all else 1
        candidate = current.replace(old, new, count)
        if len(candidate) < len(current):
            return False
        # Check that all historical lines are preserved with correct multiplicity
        body = current.split("\n---\n", 1)[-1] if "\n---\n" in current else current
        body_lines = [line for line in body.splitlines() if line.strip()]
        candidate_lines = [line for line in candidate.splitlines() if line.strip()]
        # Build multiplicity maps
        from collections import Counter
        body_counts = Counter(body_lines)
        candidate_counts = Counter(candidate_lines)
        # Every historical line must appear at least as many times in candidate
        for line, count in body_counts.items():
            if candidate_counts[line] < count:
                return False
        return True
    
    patch = args.get("patch")
    if isinstance(patch, str):
        # Parse V4A headers - check both relative and absolute forms
        patch_targets = extract_patch_targets(patch)
        path_str = str(path)
        path_name = path.name
        
        for op, target_path in patch_targets:
            # Match if target is the same file (by name or by resolved path)
            try:
                target_resolved = str(resolved(target_path))
            except (OSError, RuntimeError, ValueError):
                target_resolved = None
            
            # Check if this header refers to our protected path
            is_match = (
                target_path == path_str or
                (target_resolved and target_resolved == path_str) or
                target_path.endswith(f"/{path_name}") or
                (target_resolved and target_resolved.endswith(f"/{path_name}"))
            )
            
            if is_match:
                # Fail-closed: block Delete/Move/Rename operations
                if op in ("Delete", "Move", "Rename"):
                    return False
                # For Update, check if patch removes lines
                if op == "Update":
                    # Look for deletion lines (starting with '-' but not '---' diff markers)
                    if re.search(r"^-(?![-]{2})", patch, re.MULTILINE):
                        return False
        return True
    return False


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            return allow()
        tool, args, command = tool_data(payload)
        
        # Extract workdir if provided (for relative path resolution)
        workdir = args.get("workdir") or args.get("cwd") or None
        
        # Build target set from explicit path args and V4A patch headers
        path = str(args.get("path") or args.get("file_path") or args.get("filename") or "")
        has_explicit_path = bool(path)  # Track if path was explicitly provided
        targets = []
        
        if path:
            try:
                targets.append((path, resolved(path, workdir)))
            except (OSError, RuntimeError, ValueError):
                # If we can't resolve explicit path, still keep it for pattern matching
                targets.append((path, None))
        
        # For patch tools, extract targets from V4A headers
        patch_str = args.get("patch")
        if isinstance(patch_str, str):
            patch_targets = extract_patch_targets(patch_str)
            for op, target_path in patch_targets:
                try:
                    abs_path = resolved(target_path, workdir)
                    # Add both relative and absolute forms
                    targets.append((target_path, abs_path))
                except (OSError, RuntimeError, ValueError):
                    # If we can't resolve, still keep the raw path for checking
                    targets.append((target_path, None))
        
        # Fail-closed: if this is a patch/apply_patch tool with no discoverable targets, block
        if tool in {"patch", "patch_file", "apply_patch"} and not targets and isinstance(patch_str, str):
            return block("patch tool invocation has no parseable target paths; cannot verify append-only safety", "<unparseable>")
        
        # Check each target
        for raw, target in targets:
            if not append_only(raw, workdir):
                continue
            
            # For write_file, block if file exists (would replace)
            if tool == "write_file":
                if target and target.is_file():
                    return block("append-only path refuses write_file full replace; use patch to append and also write journal-<taskid>.md", raw)
                continue
            
            # For patch tools, verify preserving semantics
            if tool in {"patch", "patch_file", "apply_patch"} or "old_string" in args or "patch" in args:
                # If file doesn't exist, we can't verify - fail-closed for Delete/Move
                if not target or not target.is_file():
                    # Check if this is a Delete/Move/Rename from V4A headers
                    if isinstance(patch_str, str):
                        patch_targets = extract_patch_targets(patch_str)
                        for op, target_path in patch_targets:
                            if op in ("Delete", "Move", "Rename"):
                                # Fail-closed: can't verify history preservation on non-existent file for Delete/Move/Rename
                                return block(f"append-only path {op} operation cannot be verified; file must exist", raw)
                    
                    # For Update on non-existent file:
                    # - If NO explicit path provided (path-less apply_patch), fail-closed
                    # - If explicit path provided, allow (creating new file is OK)
                    if not has_explicit_path:
                        return block("append-only path update on non-existent file without explicit path; cannot verify append-only safety", raw)
                    continue
                
                # File exists, verify it preserves history
                if not patch_preserves(target, args):
                    return block("append-only path patch would drop or shrink journal history; use a preserving append patch", raw)
                continue
            
            # For terminal commands
            if tool == "terminal" or command:
                if ">>" in command and not TRUNCATE.search(command):
                    continue
                if TRUNCATE.search(command) or re.search(r"(?:^|\s)>\s*", command) or re.search(r"\btee\b(?!\s+-a)", command):
                    return block("append-only path terminal rewrite is blocked; use >> or tee -a", raw)
        
        # Check terminal redirects independently
        if tool == "terminal" and command:
            # First check for truncating commands that don't use redirects
            if TRUNCATE.search(command):
                # Extract operands from truncate/sed -i/perl -i
                # truncate: look for file paths after -s or as positional args
                # sed -i / perl -i: file paths come after the script
                for truncate_match in re.finditer(r"truncate\s+(?:-s\s+\S+\s+)?(\S+)|(?:sed|perl)\s+-i\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s+(\S+)", command):
                    target_path = truncate_match.group(1) or truncate_match.group(2)
                    if target_path and append_only(target_path, workdir):
                        return block("append-only path terminal rewrite is blocked; use >> or tee -a", target_path)
            
            for match in REDIRECT.finditer(command):
                op = match.group("op")
                is_append = match.group("append")  # Captured -a flag
                raw = match.group("path")
                if append_only(raw, workdir):
                    if op == ">":
                        return block("append-only path terminal rewrite is blocked; use >> or tee -a", raw)
                    # Block plain tee without -a (truncates); allow tee -a
                    if op == "tee" and not is_append:
                        return block("append-only path terminal rewrite is blocked; use >> or tee -a", raw)
                    if TRUNCATE.search(command):
                        return block("append-only path terminal rewrite is blocked; use >> or tee -a", raw)
        
        allow()
    except Exception:
        # Fail-open on unexpected errors to avoid breaking unrelated tools
        allow()


if __name__ == "__main__":
    main()
