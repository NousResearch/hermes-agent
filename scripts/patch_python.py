#!/usr/bin/env python3
"""patch_python.py — Indentation-safe Python patching utility.

Solves the problem: the Hermes `patch` tool and `git apply` both mangle
Python indentation. This tool reads the target file's actual indentation
from surrounding lines and applies changes with correct whitespace.

Usage:
    # Insert code after a line matching a pattern, at the same indent level
    python3 scripts/patch_python.py insert-after tools/approval.py \\
        --match "Phase 2.7:" \\
        --code "if justification_required: return status" \\
        --dedent 0

    # Replace a block between two line patterns
    python3 scripts/patch_python.py replace-between gateway/run.py \\
        --start "# Fallback: plain text" \\
        --end "return" \\
        --file replacement_code.py

    # Add a parameter to a function signature
    python3 scripts/patch_python.py add-param tools/terminal_tool.py \\
        --match "def terminal_tool(" \\
        --param "justification: Optional[str] = None"

    # Verify syntax after patching (always runs by default)
    python3 scripts/patch_python.py insert-after cron/scheduler.py \\
        --match "import sys" \\
        --code "import time"

All operations:
    - Read the target file as raw bytes
    - Detect actual indentation (tabs, 1-space, 4-space, or mixed) from context
    - Apply changes preserving exact whitespace
    - Run ast.parse() validation after every change
    - Abort and restore if syntax is broken
"""

import argparse
import ast
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


def get_indent(line: str) -> str:
    """Extract the leading whitespace from a line."""
    stripped = line.lstrip()
    if not stripped:
        return ""
    return line[: len(line) - len(stripped)]


def detect_indent_style(lines: list[str], around_line: int) -> str:
    """Detect the indentation style used around a given line number.

    Returns the indent string (e.g., '    ' for 4-space, '\\t' for tab,
    ' ' for 1-space) of the line at around_line.
    """
    # Check the target line and a few neighbors
    for offset in [0, -1, -2, 1, 2]:
        idx = around_line + offset
        if 0 <= idx < len(lines):
            indent = get_indent(lines[idx])
            if indent:
                return indent
    return "    "  # Default to 4-space


def validate_syntax(filepath: str) -> tuple[bool, str]:
    """Validate Python syntax. Returns (ok, error_msg)."""
    try:
        with open(filepath) as f:
            ast.parse(f.read())
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def backup_file(filepath: str) -> str:
    """Create a .bak copy. Returns the backup path."""
    bak = filepath + ".bak"
    shutil.copy2(filepath, bak)
    return bak


def restore_backup(filepath: str) -> None:
    """Restore from .bak if it exists."""
    bak = filepath + ".bak"
    if os.path.exists(bak):
        shutil.move(bak, filepath)


def remove_backup(filepath: str) -> None:
    """Remove the .bak file after successful patch."""
    bak = filepath + ".bak"
    if os.path.exists(bak):
        os.unlink(bak)


def cmd_insert_after(args):
    """Insert code after a line matching --match, at the same indent level.

    --dedent N: subtract N indent levels from the matched line's indent
    --indent-override STR: use this exact indent string instead of detecting
    """
    filepath = args.file
    match_pattern = args.match
    code = args.code
    code_file = args.code_file
    dedent = args.dedent or 0
    indent_override = args.indent_override

    if code_file:
        with open(code_file) as f:
            code = f.read()

    with open(filepath) as f:
        lines = f.readlines()

    # Find the matching line
    target_idx = None
    for i, line in enumerate(lines):
        if match_pattern in line:
            target_idx = i
            break

    if target_idx is None:
        print(f"ERROR: Pattern '{match_pattern}' not found in {filepath}")
        return 1

    # Detect indentation
    base_indent = indent_override or get_indent(lines[target_idx])

    # Apply dedent (subtract one indent level per dedent count)
    if dedent > 0:
        # Guess indent unit from context
        if "\t" in base_indent:
            unit = "\t"
        elif len(base_indent) >= 4:
            unit = "    "
        elif len(base_indent) >= 2:
            unit = "  "
        else:
            unit = " "
        for _ in range(dedent):
            if base_indent.endswith(unit):
                base_indent = base_indent[: -len(unit)]

    # Prepare the code lines with proper indentation
    code_lines = []
    for code_line in code.strip("\n").split("\n"):
        # Strip existing indentation from the code and re-apply base indent
        stripped = code_line.lstrip()
        if stripped:  # Non-empty line
            # Preserve relative indentation within the code block
            original_indent_len = len(code_line) - len(stripped)
            # The first line goes at base_indent, subsequent lines maintain
            # their relative offset from the first line
            if not code_lines:  # First line
                code_lines.append(base_indent + stripped + "\n")
                first_indent_len = original_indent_len
            else:
                # Calculate relative offset from first code line
                offset = original_indent_len - first_indent_len
                code_lines.append(base_indent + " " * max(0, offset) + stripped + "\n")
        else:  # Empty line
            code_lines.append("\n")

    # Insert after target line
    backup_file(filepath)
    lines.insert(target_idx + 1, "\n".join(code_lines) if not code_lines else "".join(code_lines))

    with open(filepath, "w") as f:
        f.writelines(lines)

    # Validate
    ok, err = validate_syntax(filepath)
    if ok:
        remove_backup(filepath)
        print(f"✅ Inserted {len(code_lines)} lines after line {target_idx + 1} in {filepath}")
        return 0
    else:
        restore_backup(filepath)
        print(f"❌ Syntax error after insertion: {err}")
        print("   Restored backup — no changes applied")
        return 1


def cmd_replace_between(args):
    """Replace lines between --start and --end patterns with new code."""
    filepath = args.file
    start_pattern = args.start
    end_pattern = args.end
    code = args.code
    code_file = args.code_file

    if code_file:
        with open(code_file) as f:
            code = f.read()

    with open(filepath) as f:
        lines = f.readlines()

    # Find start and end lines
    start_idx = end_idx = None
    for i, line in enumerate(lines):
        if start_pattern in line and start_idx is None:
            start_idx = i
        if end_pattern in line and start_idx is not None and i > start_idx:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        print(f"ERROR: Could not find range '{start_pattern}' ... '{end_pattern}' in {filepath}")
        return 1

    # Detect indent from start line
    base_indent = get_indent(lines[start_idx])

    # Prepare replacement code
    code_lines = []
    for code_line in code.strip("\n").split("\n"):
        stripped = code_line.lstrip()
        if stripped:
            code_lines.append(base_indent + stripped + "\n")
        else:
            code_lines.append("\n")

    # Replace
    backup_file(filepath)
    new_lines = lines[: start_idx] + code_lines + lines[end_idx + 1 :]

    with open(filepath, "w") as f:
        f.writelines(new_lines)

    ok, err = validate_syntax(filepath)
    if ok:
        remove_backup(filepath)
        print(f"✅ Replaced lines {start_idx+1}-{end_idx+1} in {filepath}")
        return 0
    else:
        restore_backup(filepath)
        print(f"❌ Syntax error after replacement: {err}")
        print("   Restored backup — no changes applied")
        return 1


def cmd_add_param(args):
    """Add a parameter to a function matching --match."""
    filepath = args.file
    match_pattern = args.match
    new_param = args.param

    with open(filepath) as f:
        lines = f.readlines()

    # Find the function def
    target_idx = None
    for i, line in enumerate(lines):
        if match_pattern in line and line.strip().startswith("def "):
            target_idx = i
            break

    if target_idx is None:
        print(f"ERROR: Function matching '{match_pattern}' not found in {filepath}")
        return 1

    # Walk forward to find the closing paren of the function signature
    paren_depth = 0
    end_idx = target_idx
    for i in range(target_idx, min(target_idx + 20, len(lines))):
        paren_depth += lines[i].count("(") - lines[i].count(")")
        if paren_depth == 0 and ")" in lines[i]:
            end_idx = i
            break

    # Check if last param line ends with a comma
    last_param_line = lines[end_idx]
    if last_param_line.rstrip().endswith(")"):
        # Single-line function or closing paren — insert before the )
        if last_param_line.rstrip().endswith(",)"):
            # Already has trailing comma before )
            new_line = last_param_line.rstrip()[:-2] + ",\n"
            lines[end_idx] = new_line
            # Insert new param on its own line
            indent = get_indent(last_param_line) or get_indent(lines[target_idx])
            lines.insert(end_idx + 1, indent + new_param + "):\n")
        else:
            # Insert before closing paren
            content = last_param_line.rstrip()
            if content.endswith(")"):
                before_paren = content[:-1]
                if before_paren.endswith(","):
                    # Add param on new line
                    indent = get_indent(lines[target_idx]) + "    "
                    lines[end_idx] = before_paren + "\n"
                    lines.insert(end_idx + 1, indent + new_param + "):\n")
                else:
                    # Add comma + param
                    indent = get_indent(lines[target_idx]) + "    "
                    lines[end_idx] = before_paren + ",\n"
                    lines.insert(end_idx + 1, indent + new_param + "):\n")
    else:
        # Multi-line, closing paren is somewhere
        indent = get_indent(last_param_line) or get_indent(lines[target_idx]) + "    "
        lines.insert(end_idx, indent + new_param + ",\n")

    backup_file(filepath)
    with open(filepath, "w") as f:
        f.writelines(lines)

    ok, err = validate_syntax(filepath)
    if ok:
        remove_backup(filepath)
        print(f"✅ Added parameter '{new_param}' to function at line {target_idx + 1}")
        return 0
    else:
        restore_backup(filepath)
        print(f"❌ Syntax error after adding parameter: {err}")
        print("   Restored backup — no changes applied")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Indentation-safe Python patching utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # insert-after
    p_ins = sub.add_parser("insert-after", help="Insert code after a matching line")
    p_ins.add_argument("file", help="Python file to patch")
    p_ins.add_argument("--match", required=True, help="String to match in the target line")
    p_ins.add_argument("--code", help="Code to insert (use --code-file for multi-line)")
    p_ins.add_argument("--code-file", help="File containing code to insert")
    p_ins.add_argument("--dedent", type=int, default=0, help="Subtract N indent levels")
    p_ins.add_argument("--indent-override", help="Use this exact indent string")

    # replace-between
    p_rep = sub.add_parser("replace-between", help="Replace lines between two patterns")
    p_rep.add_argument("file", help="Python file to patch")
    p_rep.add_argument("--start", required=True, help="Pattern marking start of replacement range")
    p_rep.add_argument("--end", required=True, help="Pattern marking end of replacement range")
    p_rep.add_argument("--code", help="Replacement code")
    p_rep.add_argument("--code-file", help="File containing replacement code")

    # add-param
    p_par = sub.add_parser("add-param", help="Add a parameter to a function")
    p_par.add_argument("file", help="Python file to patch")
    p_par.add_argument("--match", required=True, help="String to match in the function def")
    p_par.add_argument("--param", required=True, help="Parameter to add (e.g., 'foo: str = None')")

    args = parser.parse_args()

    if args.command == "insert-after":
        return cmd_insert_after(args)
    elif args.command == "replace-between":
        return cmd_replace_between(args)
    elif args.command == "add-param":
        return cmd_add_param(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
