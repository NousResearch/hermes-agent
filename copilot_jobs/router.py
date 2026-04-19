"""LLM-powered repo router with filesystem discovery.

Discovers repos by scanning $HERMES_WORKSPACE_PATH/repos/ and reads each
repo's README.md. Uses the auxiliary LLM (cheap model) to select the best
matching repository for a given prompt.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from copilot_jobs.models import RepoEntry

logger = logging.getLogger(__name__)


def _get_default_branch(repo_path: Path) -> str:
    """Detect default branch from git remote HEAD. Falls back to 'main'."""
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            capture_output=True, text=True, cwd=repo_path, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().rsplit("/", 1)[-1]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "main"


def _discover_repos(workspace_path: Path = None) -> List[RepoEntry]:
    """Discover repos by scanning the workspace repos/ directory.

    Expects structure: repos/{org}/{repo_name}/
    Reads README.md content for each repo to pass to the LLM.
    """
    if workspace_path is None:
        ws = os.environ.get("HERMES_WORKSPACE_PATH", "")
        if not ws:
            logger.warning("HERMES_WORKSPACE_PATH not set, cannot discover repos")
            return []
        workspace_path = Path(ws)

    repos_dir = workspace_path / "repos"
    if not repos_dir.is_dir():
        logger.warning("Repos directory not found: %s", repos_dir)
        return []

    entries = []
    for org_dir in sorted(repos_dir.iterdir()):
        if not org_dir.is_dir():
            continue
        for repo_dir in sorted(org_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            readme_path = repo_dir / "README.md"
            readme_text = ""
            if readme_path.exists():
                readme_text = readme_path.read_text(errors="replace")

            slug = repo_dir.name
            default_branch = _get_default_branch(repo_dir)

            entries.append(RepoEntry(
                slug=slug,
                path=str(repo_dir),
                readme_summary=readme_text[:2000],
                description="",
                default_branch=default_branch,
            ))

    return entries


def _build_repo_context(entries: List[RepoEntry]) -> str:
    """Build a concise repo catalog for the LLM prompt."""
    lines = []
    for e in entries:
        summary = e.readme_summary.split("\n")[0] if e.readme_summary else "(no README)"
        lines.append(f"- slug: {e.slug}\n  path: {e.path}\n  readme_first_line: {summary}")
    return "\n".join(lines)


def _build_routing_messages(prompt: str, repo_context: str) -> list:
    """Build the chat messages for repo routing."""
    return [
        {
            "role": "system",
            "content": (
                "You are a repo router. Given a user's task prompt and a list of "
                "available repositories, select the single best repository to work in.\n\n"
                "Available repositories:\n"
                f"{repo_context}\n\n"
                "Respond with ONLY a JSON object: {\"slug\": \"<repo-slug>\"}\n"
                "If no repository is a good match, respond: {\"slug\": null}\n"
                "Do not explain. JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def _parse_routing_response(text: str, entries: List[RepoEntry]) -> Optional[RepoEntry]:
    """Parse the LLM's JSON response into a RepoEntry."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Router LLM returned non-JSON: %s", text[:200])
        return None

    slug = data.get("slug")
    if not slug:
        return None

    slug_lower = slug.lower()
    for entry in entries:
        if entry.slug.lower() == slug_lower:
            return entry

    logger.warning("Router LLM returned unknown slug: %s", slug)
    return None


def route_repo(
    prompt: str,
    workspace_path: Path = None,
    *,
    _llm_call=None,
) -> Optional[RepoEntry]:
    """Select the best repo for a prompt using the auxiliary LLM.

    Discovers repos from the filesystem, builds a context summary, and asks
    the cheap model to pick the best match. Falls back to None if the LLM
    is unavailable or returns no match.

    _llm_call is for testing — pass a callable with the same signature as
    agent.auxiliary_client.call_llm to bypass the real provider.
    """
    entries = _discover_repos(workspace_path)
    if not entries:
        return None

    repo_context = _build_repo_context(entries)
    messages = _build_routing_messages(prompt, repo_context)

    try:
        if _llm_call is None:
            from agent.auxiliary_client import call_llm
            _llm_call = call_llm
        response = _llm_call(
            task="repo_routing",
            messages=messages,
            temperature=0.0,
            max_tokens=64,
        )
        text = response.choices[0].message.content or ""
        return _parse_routing_response(text, entries)
    except Exception:
        logger.exception("Router LLM call failed, returning None")
        return None
