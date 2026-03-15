#!/usr/bin/env python3
"""
Execution Integrity Layer — post-tool-call verification.

After critical tool calls complete, this module independently verifies
that the expected world-state change actually occurred before the agent
continues reasoning.

Supported verifications (MVP):
  - terminal / git clone  → target directory exists
  - write_file            → file exists; empty files are flagged as warnings
  - patch                 → modified/created files still exist after the tool claims success

The verifier is deliberately conservative: it only fires for operations it
understands and attaches a structured ``_verification`` block to the tool
result JSON.  Status is one of:

  - ``"verified"``  — world state matches expectations
  - ``"warning"``   — result may be incomplete (e.g. empty file)
  - ``"mismatch"``  — environment state contradicts tool output

On warning or mismatch a top-level ``_warning`` string is injected so the
model cannot ignore the signal.

Integration point:
  model_tools.handle_function_call() calls ``verify_tool_result()`` after
  registry.dispatch() returns.  The cost is one to a few stat/read syscalls
  per tool call — negligible relative to an LLM round-trip.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------

# Valid status values for VerificationResult.status
VERIFIED = "verified"
WARNING = "warning"
MISMATCH = "mismatch"


@dataclass
class VerificationResult:
    """Structured outcome of a post-tool verification check."""
    status: str  # "verified", "warning", or "mismatch"
    tool_name: str
    check: str  # short label, e.g. "dir_exists", "file_written"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "status": self.status,
            "tool": self.tool_name,
            "check": self.check,
        }
        if self.message:
            d["message"] = self.message
        if self.details:
            d["details"] = self.details
        return d


# ---------------------------------------------------------------------------
# Per-tool verification strategies
# ---------------------------------------------------------------------------

# Regex for mkdir
_MKDIR_RE = re.compile(r"\bmkdir\s+(?:-p\s+)?(\S+)", re.IGNORECASE)


def _resolve_path(raw: str) -> str:
    """Expand ~ and resolve to absolute."""
    return str(Path(os.path.expanduser(raw)).resolve())


# git clone flags that consume the next token as their value
_GIT_CLONE_VALUE_FLAGS = {
    "-b", "--branch", "--depth", "--jobs", "-j", "--reference",
    "--reference-if-able", "--origin", "-o", "--upload-pack", "-u",
    "--template", "--config", "-c", "--separate-git-dir", "--filter",
    "--server-option", "--bundle-uri",
}


def _parse_git_clone_positionals(command: str) -> Optional[List[str]]:
    """Extract positional args (repo URL, optional dir) from a git clone command.

    Returns None if the command is not a git clone, otherwise a list of
    1 or 2 positional arguments.
    """
    # Quick check before tokenizing
    if "git" not in command.lower() or "clone" not in command.lower():
        return None

    import shlex
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    # Find "git clone" subsequence
    try:
        git_idx = next(i for i, t in enumerate(tokens) if t.lower() == "git")
    except StopIteration:
        return None
    remaining = tokens[git_idx + 1:]
    if not remaining or remaining[0].lower() != "clone":
        return None
    remaining = remaining[1:]  # skip "clone"

    positionals: List[str] = []
    i = 0
    while i < len(remaining):
        tok = remaining[i]
        if tok.startswith("-"):
            if tok in _GIT_CLONE_VALUE_FLAGS:
                i += 2  # skip flag + its value
            elif "=" in tok:
                i += 1  # --flag=value
            else:
                i += 1  # boolean flag like --bare, --recurse-submodules
        else:
            positionals.append(tok)
            i += 1

    return positionals if positionals else None


def _verify_terminal(args: Dict[str, Any], result_data: Dict[str, Any]) -> Optional[VerificationResult]:
    """Verify terminal tool outcomes.

    Currently checks:
    - git clone  → target directory exists
    - mkdir      → directory exists
    """
    command = args.get("command", "")
    exit_code = result_data.get("exit_code", -1)

    # Only verify commands that claim success
    if exit_code != 0:
        return None

    # --- git clone ---
    positionals = _parse_git_clone_positionals(command)
    if positionals:
        repo_url = positionals[0]
        if len(positionals) >= 2:
            target = _resolve_path(positionals[1])
        else:
            # Infer dir name from repo URL
            repo_name = repo_url.rstrip("/").rsplit("/", 1)[-1]
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]
            workdir = args.get("workdir") or "."
            target = _resolve_path(os.path.join(workdir, repo_name))

        exists = os.path.isdir(target)
        return VerificationResult(
            status=VERIFIED if exists else MISMATCH,
            tool_name="terminal",
            check="git_clone_dir_exists",
            message="" if exists else f"git clone target directory does not exist: {target}",
            details={"expected_dir": target, "exists": exists},
        )

    # --- mkdir ---
    m2 = _MKDIR_RE.search(command)
    if m2:
        target = _resolve_path(m2.group(1))
        exists = os.path.isdir(target)
        return VerificationResult(
            status=VERIFIED if exists else MISMATCH,
            tool_name="terminal",
            check="mkdir_dir_exists",
            message="" if exists else f"mkdir target does not exist: {target}",
            details={"expected_dir": target, "exists": exists},
        )

    return None


def _verify_write_file(args: Dict[str, Any], result_data: Dict[str, Any]) -> Optional[VerificationResult]:
    """Verify write_file: target file exists and has non-zero size."""
    if result_data.get("error"):
        return None

    path = args.get("path", "")
    if not path:
        return None

    resolved = _resolve_path(path)
    exists = os.path.isfile(resolved)

    details: Dict[str, Any] = {"expected_path": resolved, "exists": exists}
    if exists:
        try:
            size = os.path.getsize(resolved)
            details["size_bytes"] = size
            if size == 0:
                return VerificationResult(
                    status=WARNING,
                    tool_name="write_file",
                    check="file_written",
                    message=f"file exists but is empty; result may be incomplete or intentional: {resolved}",
                    details=details,
                )
        except OSError:
            pass

    return VerificationResult(
        status=VERIFIED if exists else MISMATCH,
        tool_name="write_file",
        check="file_written",
        message="" if exists else f"written file does not exist: {resolved}",
        details=details,
    )


def _verify_patch(args: Dict[str, Any], result_data: Dict[str, Any]) -> Optional[VerificationResult]:
    """Verify patch: modified/created files still exist after the tool claims success."""
    if not result_data.get("success"):
        return None

    files_modified = result_data.get("files_modified", [])
    files_created = result_data.get("files_created", [])
    all_files = files_modified + files_created

    if not all_files:
        # replace mode — check path arg
        path = args.get("path", "")
        if path:
            all_files = [path]

    missing: List[str] = []
    for f in all_files:
        resolved = _resolve_path(f)
        if not os.path.exists(resolved):
            missing.append(resolved)

    ok = len(missing) == 0 and len(all_files) > 0
    details: Dict[str, Any] = {"files_checked": [_resolve_path(f) for f in all_files]}
    if missing:
        details["missing"] = missing

    return VerificationResult(
        status=VERIFIED if ok else MISMATCH,
        tool_name="patch",
        check="patched_files_exist",
        message="" if ok else f"patched file(s) missing: {', '.join(missing)}",
        details=details,
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_VERIFIERS = {
    "terminal": _verify_terminal,
    "write_file": _verify_write_file,
    "patch": _verify_patch,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_tool_result(
    tool_name: str,
    tool_args: Dict[str, Any],
    result_json: str,
) -> str:
    """Run post-call verification and attach ``_verification`` to the result.

    Parameters
    ----------
    tool_name : str
        Registered tool name (e.g. ``"terminal"``, ``"write_file"``).
    tool_args : dict
        The arguments originally passed to the tool handler.
    result_json : str
        The JSON string returned by ``registry.dispatch()``.

    Returns
    -------
    str
        The (possibly augmented) JSON string. If no verifier fires or the
        result is not parseable JSON, the original string is returned
        unchanged.
    """
    verifier = _VERIFIERS.get(tool_name)
    if verifier is None:
        return result_json

    try:
        result_data = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return result_json

    try:
        vr = verifier(tool_args, result_data)
    except Exception:
        logger.debug("Verification for %s raised; skipping", tool_name, exc_info=True)
        return result_json

    if vr is None:
        return result_json

    result_data["_verification"] = vr.to_dict()

    if vr.status == WARNING:
        logger.warning("Execution verification WARNING for %s: %s", tool_name, vr.message)
        result_data["_warning"] = (
            "\u26a0\ufe0f VERIFICATION WARNING: Result may be incomplete. "
            "Re-check environment before proceeding."
        )
    elif vr.status == MISMATCH:
        logger.warning("Execution verification MISMATCH for %s: %s", tool_name, vr.message)
        result_data["_warning"] = (
            "\u274c VERIFICATION FAILED: Tool result conflicts with environment state. "
            "Do not assume this step succeeded. Re-check or retry."
        )

    return json.dumps(result_data, ensure_ascii=False)
