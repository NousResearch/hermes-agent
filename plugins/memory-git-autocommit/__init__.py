"""Auto-commit plugin for Hermes skill memory (procedural + shared).

Model (per user decision, post-rollback):
- Each profile writes NEW agent-created skills to its OWN local skills dir
  (default create_dir: ~/.hermes/skills, or <profile>/skills under
  HERMES_HOME for supply/ads/webmaster). skill_manage edits land there.
- ~/.hermes/memory/skills is a curated "proven" shelf. All profiles read it
  (skills.external_dirs) but nothing writes there automatically — promotion
  is a deliberate, manual action (git mv / skill_manage targeting that dir
  explicitly), not an automatic side effect of skill_manage.
- ~/.hermes/memory/context (Obsidian vault) is still common declarative
  memory; write_file/patch there should still auto-commit.

So this hook commits to TWO possible repos depending on what actually
changed, never guessing:
  1. The active profile's local skills dir (HERMES_HOME/skills) — for
     skill_manage calls.
  2. ~/.hermes/memory — for write_file/patch calls whose path falls under it
     (covers ~/.hermes/memory/context and any deliberate ~/.hermes/memory/skills
     promotion done via write_file/patch rather than skill_manage).

Commit messages describe WHAT changed (skill name + action, or file name),
built only from the tool call's own arguments — never a model call, never
prompt/message text, by explicit user decision. The timestamp lives in git's
own commit metadata (author/commit date), not duplicated in the subject
text. If "which prompt caused this change" is ever needed, reconstruct it
after the fact by matching the commit's git timestamp against session
transcripts (~/.hermes/sessions/*.jsonl), not by storing that link up front.
"""

import json
import logging
import os
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger("plugins.memory-git-autocommit")

SHARED_MEMORY_DIR = Path.home() / ".hermes" / "memory"

# Tools whose successful writes may trigger a commit.
WATCHED_TOOLS = {"skill_manage", "write_file", "patch"}

_lock = threading.Lock()


def _hermes_home() -> Path:
    """Active profile's home dir (default profile or HERMES_HOME override)."""
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home).expanduser()
    return Path.home() / ".hermes"


def _local_skills_dir() -> Path:
    return _hermes_home() / "skills"


def _path_under(value, base: Path) -> bool:
    if not isinstance(value, str):
        return False
    try:
        p = Path(value).expanduser().resolve()
        base_r = base.resolve()
        return base_r == p or base_r in p.parents
    except Exception:
        return False


def _result_ok(result) -> bool:
    if not isinstance(result, str):
        return True
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return True
    if isinstance(parsed, dict):
        if parsed.get("success") is False:
            return False
        if parsed.get("error"):
            return False
    return True


def _git(repo_dir: Path, *cmd_args):
    return subprocess.run(
        ["git", "-C", str(repo_dir), *cmd_args],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _has_changes(repo_dir: Path) -> bool:
    r = _git(repo_dir, "status", "--porcelain")
    return bool(r.stdout.strip())


def _describe_change(tool_name: str, args: dict) -> str:
    """Short, factual commit subject.

    Prefers the model-supplied ``change_summary`` (schema field added to
    write_file/patch/skill_manage specifically for this hook — see the LOCAL
    PATCH comments in tools/file_tools.py and tools/skill_manager_tool.py).
    That costs nothing extra: the model already knows why it's making the
    edit when it calls the tool, so filling one more short string field is
    free compared to a separate model call or a network call inside this
    hook (both ruled out deliberately — see module docstring).

    Falls back to an auto-derived label (skill name + action, or file name)
    when change_summary is absent — e.g. older transcripts, or a model that
    didn't fill it in. Still never a model call, never raw prompt text.
    """
    args = args or {}

    if tool_name == "skill_manage":
        ops = args.get("operations") or []
        if isinstance(ops, dict):
            ops = [ops]
        summaries = []
        parts = []
        for op in ops:
            if not isinstance(op, dict):
                continue
            action = op.get("action", "?")
            name = op.get("name", "?")
            parts.append(f"{action} {name}")
            cs = op.get("change_summary")
            if isinstance(cs, str) and cs.strip():
                summaries.append(cs.strip())
        if summaries:
            label = "; ".join(summaries[:3])
            if len(summaries) > 3:
                label += f"; +{len(summaries) - 3} more"
            return f"skill_manage: {label}"
        if parts:
            label = "; ".join(parts[:3])
            if len(parts) > 3:
                label += f"; +{len(parts) - 3} more"
            return f"skill_manage: {label}"
        return "skill_manage: update"

    # write_file / patch
    cs = args.get("change_summary")
    if isinstance(cs, str) and cs.strip():
        return f"{tool_name}: {cs.strip()}"
    path = args.get("path")
    if path:
        return f"{tool_name}: {Path(path).name}"
    return f"{tool_name}: update"


def _commit(repo_dir: Path, tool_name: str, args: dict):
    with _lock:
        if not repo_dir.exists() or not (repo_dir / ".git").exists():
            return
        try:
            if not _has_changes(repo_dir):
                return
            _git(repo_dir, "add", "-A")
            # Subject: what changed, from the tool's own args (no model call,
            # no prompt/message text — explicit user decision). Timestamp goes
            # in the git commit metadata (author/commit date), not duplicated
            # in the subject text.
            msg = _describe_change(tool_name, args)
            r = _git(repo_dir, "commit", "-m", msg)
            if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
                logger.warning("memory-git-autocommit: commit failed in %s: %s", repo_dir, r.stderr.strip())
        except Exception as e:
            logger.warning("memory-git-autocommit: error committing %s: %s", repo_dir, e)


def on_post_tool_call(tool_name=None, args=None, result=None, status=None, **kwargs):
    del kwargs
    if tool_name not in WATCHED_TOOLS:
        return
    if status not in (None, "ok"):
        return
    if not _result_ok(result):
        return

    args = args or {}
    targets = []

    if tool_name == "skill_manage":
        # skill_manage writes new skills to the active profile's local
        # create_dir (default: HERMES_HOME/skills); existing skills are
        # edited wherever they live, including under the shared dir if the
        # agent was explicitly pointed there. Check both candidate repos.
        targets.append(_local_skills_dir())
        targets.append(SHARED_MEMORY_DIR)
    else:
        # write_file / patch: only act if the path is actually under one of
        # our two tracked repos.
        path = args.get("path")
        if _path_under(path, SHARED_MEMORY_DIR):
            targets.append(SHARED_MEMORY_DIR)
        if _path_under(path, _local_skills_dir()):
            targets.append(_local_skills_dir())

    for repo_dir in targets:
        # Run off the hot path; post_tool_call is timeout-bounded.
        threading.Thread(target=_commit, args=(repo_dir, tool_name, args), daemon=True).start()


def register(ctx):
    ctx.register_hook("post_tool_call", on_post_tool_call)
