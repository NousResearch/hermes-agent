"""Shared engine for the /review command — every surface calls this.

/review spawns an independent, full-privilege background subagent (the same async rail as
``delegate_task(background=true)``) to review whatever the recent conversation presented,
optionally augmented with a bounded uncommitted, merge-base, or single-commit git diff;
its result re-enters the spawning session as a normal async-delegation completion.
Model routing: ``auxiliary.review`` when configured, else the parent agent's credentials,
passed as ``credentials_cfg`` to ``delegate_task`` so native-SDK providers, api_mode
detection and credential pools behave identically to ``delegation.provider`` pins.
Surfaces (CLI/gateway ``/review``, TUI/Desktop) snapshot, call :func:`start_review`, print the note.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# How many recent chat messages (user + assistant turns) the reviewer gets.
DEFAULT_CONTEXT_MESSAGES = 10

# Per-message excerpt cap: generous (a PR summary/diff excerpt is exactly what the
# reviewer needs) but bounded against a pathological turn.
_MESSAGE_CHAR_CAP = 12_000

# Maximum number of characters from a git patch placed in reviewer context. The
# stat is collected separately, and truncation is always disclosed in-band.
GIT_DIFF_CHAR_CAP = 100_000
_GIT_COMMAND_TIMEOUT_SECONDS = 20
_GIT_ERROR_CHAR_CAP = 300
_REVIEW_TARGET_USAGE = (
    "Usage: /review [uncommitted | base <branch> | commit <sha> | review instructions]"
)


@dataclass(frozen=True)
class GitReviewTarget:
    kind: str
    value: str = ""


_REVIEW_GOAL = (
    "Act as an independent senior reviewer. Thoroughly review the work presented in the conversation excerpt "
    "provided in your context: investigate any code, pull request, branch, commit, documentation, design, or other "
    "artifact it references (open the PR, read the diff, run the code or tests where feasible) rather than judging "
    "from the excerpt alone. Produce a full, structured review: what the work does, whether it is correct and "
    "complete, concrete defects or risks found (with file/line references where possible), what was verified vs. "
    "only read, and a clear final verdict with recommended next steps."
)


def _message_text(message: Dict[str, Any]) -> str:
    """Display text of a message; multimodal parts are joined, non-text parts noted."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(part.get("text") or "") if part.get("type") == "text" else f"[{part.get('type', 'attachment')}]"
                 for part in content if isinstance(part, dict)]
        return "\n".join(p for p in parts if p)
    return ""


def snapshot_recent_messages(messages: List[Dict[str, Any]], limit: int = DEFAULT_CONTEXT_MESSAGES) -> List[Dict[str, str]]:
    """Last ``limit`` user/assistant messages with text as {role, text}, oldest first (system, tool and
    pure tool-call stubs excluded)."""
    out: List[Dict[str, str]] = []
    for message in reversed(list(messages or [])):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        text = _message_text(message).strip() if role in ("user", "assistant") else ""
        if not text:
            continue
        if len(text) > _MESSAGE_CHAR_CAP:
            text = text[:_MESSAGE_CHAR_CAP] + "\n[... truncated ...]"
        out.append({"role": role, "text": text})
        if len(out) >= limit:
            break
    out.reverse()
    return out


def parse_review_request(user_prompt: str) -> tuple[Optional[GitReviewTarget], str]:
    """Split a recognized leading git selector from optional review instructions.

    Unknown text remains free-form instructions for backward compatibility.
    """
    text = (user_prompt or "").strip()
    if not text:
        return None, ""
    parts = text.split(None, 2)
    selector = parts[0]
    if selector == "uncommitted":
        return GitReviewTarget(selector), text[len(selector):].strip()
    if selector not in ("base", "commit"):
        return None, text
    if len(parts) < 2:
        raise ValueError(_REVIEW_TARGET_USAGE)
    return GitReviewTarget(selector, parts[1]), parts[2].strip() if len(parts) > 2 else ""


def _git_error(detail: str) -> ValueError:
    cleaned = " ".join((detail or "git command failed").split())
    return ValueError(f"Unable to prepare git review: {cleaned[:_GIT_ERROR_CHAR_CAP]}")


def _run_git(args: List[str], cwd: Path) -> str:
    """Run git without a shell and decode output strictly for reviewer context."""
    command = ["git", "--no-pager", "-c", "core.quotepath=true", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_GIT_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        raise _git_error("git executable was not found") from None
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _git_error(str(exc)) from None
    try:
        stdout = completed.stdout.decode("utf-8", errors="strict")
        stderr = completed.stderr.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _git_error("git produced undecodable output") from None
    if completed.returncode != 0:
        raise _git_error(stderr or stdout)
    return stdout


def _run_git_diff(args: List[str], cwd: Path) -> str:
    """Read at most the documented patch cap, terminating git on overflow."""
    command = ["git", "--no-pager", "-c", "core.quotepath=true", *args]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except FileNotFoundError:
        raise _git_error("git executable was not found") from None
    except OSError as exc:
        raise _git_error(str(exc)) from None

    captured: Dict[str, Any] = {}

    def _read() -> None:
        try:
            captured["text"] = process.stdout.read(GIT_DIFF_CHAR_CAP + 1) if process.stdout else ""
        except UnicodeDecodeError as exc:
            captured["error"] = exc

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    reader.join(_GIT_COMMAND_TIMEOUT_SECONDS)
    if reader.is_alive():
        if process.poll() is None:
            process.kill()
        reader.join()
        process.wait()
        raise _git_error("git diff timed out")
    if captured.get("error") is not None:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise _git_error("git produced undecodable output")

    output = str(captured.get("text") or "")
    if len(output) > GIT_DIFF_CHAR_CAP:
        try:
            process.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return _bounded_diff(output)
    returncode = process.wait()
    if returncode != 0:
        raise _git_error(output)
    return output


def _resolve_commit(ref: str, cwd: Path) -> str:
    resolved = _run_git(
        ["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"], cwd,
    ).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", resolved):
        raise _git_error(f"invalid revision {ref!r}")
    return resolved


def _bounded_diff(diff: str) -> str:
    if len(diff) <= GIT_DIFF_CHAR_CAP:
        return diff
    marker = f"\n[... git diff truncated at {GIT_DIFF_CHAR_CAP} characters ...]"
    return diff[:max(0, GIT_DIFF_CHAR_CAP - len(marker))] + marker


def collect_git_review_context(target: GitReviewTarget, cwd: os.PathLike[str] | str) -> Optional[str]:
    """Return stat + bounded patch for ``target``, or ``None`` for a clean diff."""
    workspace = Path(cwd)
    if target.kind == "uncommitted":
        label = "uncommitted"
        stat_args = ["diff", "--stat", "--no-ext-diff", "HEAD", "--"]
        diff_args = ["diff", "--no-ext-diff", "--no-textconv", "HEAD", "--"]
    elif target.kind == "base":
        target_sha = _resolve_commit(target.value, workspace)
        merge_base = _run_git(["merge-base", "HEAD", target_sha], workspace).strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", merge_base):
            raise _git_error(f"could not find merge base for {target.value!r}")
        label = f"base {target.value} (merge-base {merge_base})"
        stat_args = ["diff", "--stat", "--no-ext-diff", merge_base, "HEAD", "--"]
        diff_args = ["diff", "--no-ext-diff", "--no-textconv", merge_base, "HEAD", "--"]
    elif target.kind == "commit":
        if not re.fullmatch(r"[0-9a-fA-F]{4,64}", target.value):
            raise _git_error(f"invalid commit SHA {target.value!r}")
        commit_sha = _resolve_commit(target.value, workspace)
        label = f"commit {target.value} ({commit_sha})"
        stat_args = ["show", "--stat", "--format=fuller", commit_sha, "--"]
        diff_args = ["show", "--format=fuller", "--patch", "--no-ext-diff", "--no-textconv", commit_sha, "--"]
    else:
        raise _git_error(f"unsupported target {target.kind!r}")

    stat = _run_git_diff(stat_args, workspace).strip()
    diff = _run_git_diff(diff_args, workspace)
    if not diff.strip():
        return None
    return (
        f"Git diff target: {label}\n\n"
        f"Diff stat:\n{stat or '(no stat output)'}\n\n"
        f"Full diff (bounded to {GIT_DIFF_CHAR_CAP} characters):\n{diff}"
    )


def _review_workspace(
    parent_agent, selected_cwd: Optional[os.PathLike[str] | str] = None,
) -> Path:
    hints = getattr(parent_agent, "_subdirectory_hints", None)
    candidates = (
        selected_cwd,
        getattr(hints, "working_dir", None),
        getattr(parent_agent, "terminal_cwd", None),
        getattr(parent_agent, "cwd", None),
        os.getenv("TERMINAL_CWD"),
        os.getcwd(),
    )
    for candidate in candidates:
        if not isinstance(candidate, (str, os.PathLike)):
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_dir():
            return path
    raise _git_error("working directory is unavailable")


def collect_parent_loaded_skills(parent_agent, messages: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    """Skills the parent was operating under: launch-preloaded (marker in ``ephemeral_system_prompt``)
    first, then ``skill_view`` loads from history, deduped, capped at ``limit`` (a reviewer told to load 30
    skills would burn its budget before working)."""
    names: List[str] = []
    prompt = str(getattr(parent_agent, "ephemeral_system_prompt", "") or "")
    candidates = [m.group(1) for m in re.finditer(r'with the "([^"]+)" skill\s+preloaded', prompt)]
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            fn = tool_call.get("function") or {} if isinstance(tool_call, dict) else {}
            if fn.get("name") != "skill_view":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                continue
            # Only whole-skill loads seed the reviewer; a reference-file read is a detail
            # of the parent's task covered by loading the SKILL.md.
            if isinstance(args, dict) and not args.get("file_path"):
                candidates.append(str(args.get("name") or ""))
    for name in candidates:
        cleaned = name.strip()
        if cleaned and cleaned not in names:
            names.append(cleaned)
    return names[:limit]


def build_review_task(
    snapshot: List[Dict[str, str]],
    user_prompt: str = "",
    loaded_skills: Optional[List[str]] = None,
    git_context: str = "",
) -> tuple:
    """Compose a viewer-friendly goal and the complete reviewer briefing."""
    focus = " ".join(user_prompt.split())
    goal = f"Review: {focus}" if focus else "Review recent work"
    if len(goal) > 80:
        goal = goal[:79].rstrip() + "…"
    # The goal is also the live worker label; keep the full instructions in context.
    lines = [
        _REVIEW_GOAL,
        "",
        "You were spawned by the /review command. The following is an excerpt of the most recent conversation "
        "between the user and their primary agent. It is your starting evidence — the work to "
        "review is referenced in it.",
        "",
        "--- Recent conversation (oldest first) ---",
    ]
    for message in snapshot:
        lines += [f"[{'USER' if message['role'] == 'user' else 'PRIMARY AGENT'}]", message["text"], ""]
    lines.append("--- End of conversation excerpt ---")
    if git_context:
        lines += ["", "--- Git changes to review ---", git_context, "--- End of git changes ---"]
    if loaded_skills:
        skill_list = ", ".join(loaded_skills)
        lines += [
            "",
            "The primary agent was operating under these loaded skills: "
            f"{skill_list}. Before reviewing, load each with "
            "skill_view(name=...) and treat their conventions, invariants, "
            "and review standards as binding for your assessment — the work "
            "was produced under them and must be judged against them.",
        ]
    if user_prompt.strip():
        lines += ["", "Additional review instructions from the user:", user_prompt.strip()]
    lines += [
        "",
        "Your review is delivered back into that conversation, addressed to "
        "the primary agent and its user. Be direct and specific; do not "
        "soften findings.",
    ]
    return goal, "\n".join(lines)


def _load_review_credentials_cfg() -> Optional[Dict[str, Any]]:
    """``auxiliary.review`` as a delegation-credentials dict, or None when unconfigured (provider auto/empty
    and no model/base_url) so the reviewer inherits the parent's credentials."""
    try:
        from hermes_cli.config import load_config_readonly
        review = (load_config_readonly().get("auxiliary") or {}).get("review") or {}
    except Exception:
        return None
    if not isinstance(review, dict):
        return None

    cfg = {k: str(review.get(k) or "").strip() for k in ("provider", "model", "base_url", "api_key", "api_mode")}
    if cfg["provider"].lower() == "auto":
        cfg["provider"] = ""
    if not (cfg["provider"] or cfg["model"] or cfg["base_url"]):
        return None
    return cfg


def start_review(
    parent_agent,
    messages: List[Dict[str, Any]],
    user_prompt: str = "",
    *,
    cwd: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    """Dispatch the reviewer subagent; returns the parsed ``delegate_task`` dict (``status: "dispatched"`` +
    ``delegation_id``, or the synchronous result on channels without async completions). Raises ValueError
    when there is nothing to review or the dispatch is rejected/errored."""
    target, review_instructions = parse_review_request(user_prompt)
    if parent_agent is None:
        raise ValueError("No active agent — send a message first.")
    snapshot = snapshot_recent_messages(messages)
    if not snapshot:
        raise ValueError("Nothing to review yet — the conversation is empty.")
    git_context = ""
    if target is not None:
        git_context = collect_git_review_context(target, _review_workspace(parent_agent, cwd)) or ""
        if not git_context:
            raise ValueError("Nothing to review — the selected git diff is clean.")
    goal, context = build_review_task(
        snapshot,
        review_instructions,
        collect_parent_loaded_skills(parent_agent, messages),
        git_context,
    )
    credentials_cfg = _load_review_credentials_cfg()

    from tools.delegate_tool import delegate_task
    raw = delegate_task(goal=goal, context=context, background=True, parent_agent=parent_agent, credentials_cfg=credentials_cfg)
    try:
        result = json.loads(raw)
    except Exception:
        result = None
    if isinstance(result, dict) and result.get("error"):
        raise ValueError(str(result["error"]))
    if not isinstance(result, dict):
        raise ValueError(f"Review dispatch failed: {raw!r}")
    result.setdefault("review_model", (credentials_cfg or {}).get("model") or "")
    return result


def format_dispatch_note(result: Dict[str, Any], user_prompt: str = "") -> str:
    """Human-facing one-liner for a successful dispatch. Shared by surfaces."""
    if result.get("status") == "dispatched":
        return "Review started. Results will return here."
    model = str(result.get("review_model") or "").strip()
    model_note = f" on {model}" if model else ""
    focus_note = f" (focus: {user_prompt.strip()})" if user_prompt.strip() else ""
    # Synchronous fallback (channels that cannot route async completions).
    return (
        f"⚖ Review completed synchronously{model_note}{focus_note} — "
        f"results:\n{json.dumps(result.get('results', result), ensure_ascii=False)[:4000]}"
    )
