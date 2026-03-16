#!/usr/bin/env python3
"""
Execution Integrity Layer — post-tool-call verification.

After critical tool calls complete, this module independently verifies
that the expected world-state change actually occurred before the agent
continues reasoning.

Supported verifications:
  - terminal / git clone   -> target directory exists
  - terminal / git init    -> .git directory exists
  - terminal / mkdir       -> directory exists
  - terminal / cp          -> destination exists
  - terminal / mv          -> destination exists
  - terminal / rm          -> targets no longer exist
  - terminal / touch       -> file exists
  - write_file             -> file exists; empty files flagged only when content was non-empty
  - patch                  -> modified/created files still exist
  - read_file              -> content returned; error with similar_files -> warning
  - browser_navigate       -> success field; bot detection -> warning
  - web_extract            -> results non-empty; per-URL errors -> warning/mismatch

The verifier is deliberately conservative: it only fires for operations it
understands and attaches a structured ``_verification`` block to the tool
result JSON.  Status is one of:

  - ``"verified"``  — world state matches expectations
  - ``"warning"``   — result may be incomplete (e.g. partial extraction)
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
import shlex
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
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_path(raw: str) -> str:
    """Expand ~ and resolve to absolute."""
    return str(Path(os.path.expanduser(raw)).resolve())


def _extract_simple_command(command: str, cmd_name: str) -> Optional[str]:
    """Extract the segment of a compound command that contains *cmd_name*.

    Splits on ``&&``, ``||``, ``;``, and ``|`` and returns the first segment
    that contains the given command name as a whole word.  Returns ``None`` if
    no segment matches.
    """
    segments = re.split(r'\s*(?:&&|\|\||[;|])\s*', command)
    for seg in segments:
        if re.search(rf'\b{re.escape(cmd_name)}\b', seg):
            return seg.strip()
    return None


def _parse_positionals(
    segment: str,
    cmd_name: str,
    value_flags: Optional[set] = None,
) -> Optional[List[str]]:
    """Extract positional arguments from a simple command.

    Tokenises *segment* with ``shlex.split``, locates *cmd_name*, then walks
    the remaining tokens skipping flags (``-…``) and their values.

    Parameters
    ----------
    segment : str
        A single shell command (no ``&&`` / ``||`` / ``;`` / ``|``).
    cmd_name : str
        The command to locate (e.g. ``"cp"``, ``"mv"``).
    value_flags : set, optional
        Flags that consume the next token as their value (e.g. ``{"-t"}``).

    Returns
    -------
    list[str] | None
        The positional arguments, or ``None`` if *cmd_name* was not found.
    """
    try:
        tokens = shlex.split(segment)
    except ValueError:
        tokens = segment.split()

    try:
        cmd_idx = next(i for i, t in enumerate(tokens) if t == cmd_name)
    except StopIteration:
        return None

    remaining = tokens[cmd_idx + 1:]
    value_flags = value_flags or set()
    positionals: List[str] = []
    i = 0
    while i < len(remaining):
        tok = remaining[i]
        if tok.startswith("-"):
            if tok in value_flags:
                i += 2  # skip flag + its value
            elif "=" in tok:
                i += 1  # --flag=value
            else:
                i += 1  # boolean flag
        else:
            positionals.append(tok)
            i += 1

    return positionals if positionals else None


# ---------------------------------------------------------------------------
# Per-tool verification strategies
# ---------------------------------------------------------------------------

# Regex for mkdir
_MKDIR_RE = re.compile(r"\bmkdir\s+(?:-p\s+)?(\S+)", re.IGNORECASE)

# Regex for git init (optional target dir)
_GIT_INIT_RE = re.compile(r"\bgit\s+init\b(?:\s+(\S+))?", re.IGNORECASE)

# git clone flags that consume the next token as their value
_GIT_CLONE_VALUE_FLAGS = {
    "-b", "--branch", "--depth", "--jobs", "-j", "--reference",
    "--reference-if-able", "--origin", "-o", "--upload-pack", "-u",
    "--template", "--config", "-c", "--separate-git-dir", "--filter",
    "--server-option", "--bundle-uri",
}

# touch flags that consume the next token as their value
_TOUCH_VALUE_FLAGS = {"-t", "-r", "-d", "--date", "--reference"}


def _parse_git_clone_positionals(command: str) -> Optional[List[str]]:
    """Extract positional args (repo URL, optional dir) from a git clone command.

    Returns None if the command is not a git clone, otherwise a list of
    1 or 2 positional arguments.
    """
    # Quick check before tokenizing
    if "git" not in command.lower() or "clone" not in command.lower():
        return None

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

    Checks:
    - git clone  -> target directory exists
    - git init   -> .git directory exists
    - mkdir      -> directory exists
    - cp         -> destination exists
    - mv         -> destination exists
    - rm         -> targets no longer exist
    - touch      -> file exists
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

    # --- git init ---
    m_init = _GIT_INIT_RE.search(command)
    if m_init:
        # Skip bare repos — they don't have a .git subdirectory
        if "--bare" in command:
            return None
        target_dir = m_init.group(1) if m_init.group(1) else (args.get("workdir") or ".")
        resolved = _resolve_path(target_dir)
        git_dir = os.path.join(resolved, ".git")
        exists = os.path.isdir(git_dir)
        return VerificationResult(
            status=VERIFIED if exists else MISMATCH,
            tool_name="terminal",
            check="git_init_dir_exists",
            message="" if exists else f"git init: .git directory does not exist in {resolved}",
            details={"expected_dir": resolved, "git_dir_exists": exists},
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

    # --- cp ---
    seg = _extract_simple_command(command, "cp")
    if seg is not None:
        positionals = _parse_positionals(seg, "cp")
        if positionals and len(positionals) >= 2:
            dest = _resolve_path(positionals[-1])
            exists = os.path.exists(dest)
            return VerificationResult(
                status=VERIFIED if exists else MISMATCH,
                tool_name="terminal",
                check="cp_dest_exists",
                message="" if exists else f"cp destination does not exist: {dest}",
                details={"expected_path": dest, "exists": exists},
            )

    # --- mv ---
    seg = _extract_simple_command(command, "mv")
    if seg is not None:
        positionals = _parse_positionals(seg, "mv")
        if positionals and len(positionals) >= 2:
            dest = _resolve_path(positionals[-1])
            exists = os.path.exists(dest)
            return VerificationResult(
                status=VERIFIED if exists else MISMATCH,
                tool_name="terminal",
                check="mv_dest_exists",
                message="" if exists else f"mv destination does not exist: {dest}",
                details={"expected_path": dest, "exists": exists},
            )

    # --- rm ---
    seg = _extract_simple_command(command, "rm")
    if seg is not None:
        positionals = _parse_positionals(seg, "rm")
        if positionals:
            # Skip if any target uses glob patterns — too unreliable
            if any(c in t for t in positionals for c in ("*", "?", "[")):
                return None
            still_exist: List[str] = []
            resolved_targets: List[str] = []
            for t in positionals:
                resolved = _resolve_path(t)
                resolved_targets.append(resolved)
                if os.path.exists(resolved):
                    still_exist.append(resolved)
            ok = len(still_exist) == 0
            details: Dict[str, Any] = {"targets": resolved_targets}
            if still_exist:
                details["still_exist"] = still_exist
            return VerificationResult(
                status=VERIFIED if ok else MISMATCH,
                tool_name="terminal",
                check="rm_targets_removed",
                message="" if ok else f"rm target(s) still exist: {', '.join(still_exist)}",
                details=details,
            )

    # --- touch ---
    seg = _extract_simple_command(command, "touch")
    if seg is not None:
        positionals = _parse_positionals(seg, "touch", value_flags=_TOUCH_VALUE_FLAGS)
        if positionals:
            missing: List[str] = []
            resolved_targets = []
            for t in positionals:
                resolved = _resolve_path(t)
                resolved_targets.append(resolved)
                if not os.path.exists(resolved):
                    missing.append(resolved)
            ok = len(missing) == 0
            details = {"expected_paths": resolved_targets}
            if missing:
                details["missing"] = missing
            return VerificationResult(
                status=VERIFIED if ok else MISMATCH,
                tool_name="terminal",
                check="touch_file_exists",
                message="" if ok else f"touch target(s) do not exist: {', '.join(missing)}",
                details=details,
            )

    return None


def _verify_write_file(args: Dict[str, Any], result_data: Dict[str, Any]) -> Optional[VerificationResult]:
    """Verify write_file: target file exists; empty files flagged only when content was non-empty."""
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
                intended_content = args.get("content", "")
                if intended_content:
                    # Non-empty content was expected but file is empty
                    return VerificationResult(
                        status=WARNING,
                        tool_name="write_file",
                        check="file_written",
                        message=f"file exists but is empty despite non-empty content arg: {resolved}",
                        details=details,
                    )
                # Content was intentionally empty — fall through to VERIFIED
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


def _verify_read_file(args: Dict[str, Any], result_data: Dict[str, Any]) -> Optional[VerificationResult]:
    """Verify read_file: check that content was returned without error."""
    error = result_data.get("error")
    if error:
        similar = result_data.get("similar_files", [])
        details: Dict[str, Any] = {"error": error}
        if similar:
            details["similar_files"] = similar
            return VerificationResult(
                status=WARNING,
                tool_name="read_file",
                check="file_read",
                message=f"read_file error: {error}; similar files available",
                details=details,
            )
        return VerificationResult(
            status=MISMATCH,
            tool_name="read_file",
            check="file_read",
            message=f"read_file error with no alternatives: {error}",
            details=details,
        )

    # Success case: content present
    content = result_data.get("content", "")
    if content:
        return VerificationResult(
            status=VERIFIED,
            tool_name="read_file",
            check="file_read",
            message="",
            details={
                "file_size": result_data.get("file_size", 0),
                "truncated": result_data.get("truncated", False),
            },
        )

    # No error but also no content — genuinely empty file is valid, no opinion
    return None


def _verify_browser_navigate(args: Dict[str, Any], result_data: Dict[str, Any]) -> Optional[VerificationResult]:
    """Verify browser_navigate: check success and bot detection."""
    success = result_data.get("success")

    if success is None:
        return None  # no success field — can't verify

    if not success:
        return VerificationResult(
            status=MISMATCH,
            tool_name="browser_navigate",
            check="navigation_success",
            message=f"Navigation failed: {result_data.get('error', 'unknown error')}",
            details={"url": args.get("url", ""), "error": result_data.get("error")},
        )

    # Success but check for bot detection
    if result_data.get("bot_detection_warning"):
        return VerificationResult(
            status=WARNING,
            tool_name="browser_navigate",
            check="navigation_success",
            message="Navigation succeeded but bot detection was triggered",
            details={
                "url": result_data.get("url", args.get("url", "")),
                "title": result_data.get("title", ""),
                "bot_detection": True,
            },
        )

    return VerificationResult(
        status=VERIFIED,
        tool_name="browser_navigate",
        check="navigation_success",
        message="",
        details={
            "url": result_data.get("url", args.get("url", "")),
            "title": result_data.get("title", ""),
        },
    )


def _verify_web_extract(args: Dict[str, Any], result_data: Dict[str, Any]) -> Optional[VerificationResult]:
    """Verify web_extract: check that results contain content."""
    # Handle top-level error
    if result_data.get("error"):
        return VerificationResult(
            status=MISMATCH,
            tool_name="web_extract",
            check="extraction_results",
            message=f"web_extract failed: {result_data.get('error')}",
            details={"error": result_data.get("error")},
        )

    results = result_data.get("results", [])
    if not results:
        return VerificationResult(
            status=MISMATCH,
            tool_name="web_extract",
            check="extraction_results",
            message="web_extract returned no results",
            details={"result_count": 0},
        )

    # Count successes and failures per URL
    succeeded: List[str] = []
    failed: List[str] = []
    for r in results:
        if r.get("error") or not r.get("content"):
            failed.append(r.get("url", "unknown"))
        else:
            succeeded.append(r.get("url", "unknown"))

    if not succeeded:
        return VerificationResult(
            status=MISMATCH,
            tool_name="web_extract",
            check="extraction_results",
            message=f"All {len(failed)} URL(s) failed to extract content",
            details={"failed_urls": failed, "succeeded": 0, "failed": len(failed)},
        )

    if failed:
        return VerificationResult(
            status=WARNING,
            tool_name="web_extract",
            check="extraction_results",
            message=f"{len(failed)} of {len(results)} URL(s) failed to extract",
            details={"failed_urls": failed, "succeeded": len(succeeded), "failed": len(failed)},
        )

    return VerificationResult(
        status=VERIFIED,
        tool_name="web_extract",
        check="extraction_results",
        message="",
        details={"succeeded": len(succeeded), "urls": succeeded},
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_VERIFIERS = {
    "terminal": _verify_terminal,
    "write_file": _verify_write_file,
    "patch": _verify_patch,
    "read_file": _verify_read_file,
    "browser_navigate": _verify_browser_navigate,
    "web_extract": _verify_web_extract,
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
