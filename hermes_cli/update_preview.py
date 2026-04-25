"""Helpers for summarizing pending Hermes updates before applying them."""

from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
from pathlib import Path
from typing import Optional

from hermes_cli import __version__ as CURRENT_VERSION


_VERSION_RE = re.compile(r'__version__\s*=\s*"([^"]+)"')
_RELEASE_FILE_RE = re.compile(r"RELEASE_v(\d+)\.(\d+)\.(\d+)\.md$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


@dataclass(frozen=True)
class ReleaseSummary:
    version: str
    highlights: list[str]


@dataclass(frozen=True)
class UpdatePreview:
    current_version: str
    target_version: Optional[str]
    commit_count: int
    base_ref: str
    remote_ref: str
    commits: list[str]
    releases: list[ReleaseSummary]


def _git_capture(
    repo_dir: Path,
    git_cmd: list[str],
    args: list[str],
    *,
    timeout: float = 10.0,
) -> Optional[str]:
    try:
        result = subprocess.run(
            git_cmd + args,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _git_ok(
    repo_dir: Path,
    git_cmd: list[str],
    args: list[str],
    *,
    timeout: float = 5.0,
) -> bool:
    try:
        result = subprocess.run(
            git_cmd + args,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return False
    return result.returncode == 0


def resolve_update_base_ref(
    repo_dir: Path,
    git_cmd: Optional[list[str]] = None,
    branch: str = "main",
) -> str:
    """Return the local ref Hermes should compare against origin/main."""
    git_cmd = git_cmd or ["git"]
    current_branch = _git_capture(
        repo_dir, git_cmd, ["rev-parse", "--abbrev-ref", "HEAD"]
    )
    current_branch = (current_branch or "").strip() or "HEAD"
    if current_branch == branch:
        return "HEAD"
    if _git_ok(repo_dir, git_cmd, ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"]):
        return branch
    return "HEAD"


def _extract_version(init_text: str) -> Optional[str]:
    match = _VERSION_RE.search(init_text or "")
    return match.group(1) if match else None


def _clean_markdown(text: str) -> str:
    cleaned = _LINK_RE.sub(r"\1", text or "")
    cleaned = cleaned.replace("**", "").replace("`", "")
    cleaned = " ".join(cleaned.strip().split())
    return cleaned


def _truncate(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _parse_release_highlights(content: str, max_highlights: int) -> list[str]:
    highlights: list[str] = []
    in_highlights = False
    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            if in_highlights:
                break
            in_highlights = "highlight" in line.lower()
            continue
        if in_highlights and line.startswith("- "):
            highlights.append(_truncate(_clean_markdown(line[2:])))
            if len(highlights) >= max_highlights:
                break
    if highlights:
        return highlights

    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            highlights.append(_truncate(_clean_markdown(line[2:])))
            if len(highlights) >= max_highlights:
                break
    return highlights


def _release_sort_key(path: str) -> tuple[int, int, int]:
    match = _RELEASE_FILE_RE.search(path or "")
    if not match:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())


def _release_version_from_path(path: str) -> Optional[str]:
    match = _RELEASE_FILE_RE.search(path or "")
    if not match:
        return None
    return f"v{match.group(1)}.{match.group(2)}.{match.group(3)}"


def _load_release_summaries(
    repo_dir: Path,
    git_cmd: list[str],
    *,
    base_ref: str,
    remote_ref: str,
    max_releases: int,
    max_highlights: int,
) -> list[ReleaseSummary]:
    changed_files = _git_capture(
        repo_dir,
        git_cmd,
        ["diff", "--name-only", f"{base_ref}..{remote_ref}", "--", ":(glob)RELEASE_v*.md"],
    )
    if not changed_files:
        return []

    release_files = sorted(
        [line.strip() for line in changed_files.splitlines() if line.strip()],
        key=_release_sort_key,
        reverse=True,
    )

    summaries: list[ReleaseSummary] = []
    for path in release_files[:max_releases]:
        content = _git_capture(repo_dir, git_cmd, ["show", f"{remote_ref}:{path}"])
        if not content:
            continue
        version = _release_version_from_path(path)
        if not version:
            continue
        highlights = _parse_release_highlights(content, max_highlights)
        if highlights:
            summaries.append(ReleaseSummary(version=version, highlights=highlights))
    return summaries


def get_update_preview(
    repo_dir: Path,
    *,
    git_cmd: Optional[list[str]] = None,
    base_ref: Optional[str] = None,
    remote_ref: str = "origin/main",
    max_commits: int = 6,
    max_releases: int = 2,
    max_release_highlights: int = 2,
) -> Optional[UpdatePreview]:
    """Build a human-friendly summary of pending updates."""
    git_cmd = git_cmd or ["git"]
    base_ref = base_ref or resolve_update_base_ref(repo_dir, git_cmd)

    commit_count_text = _git_capture(
        repo_dir,
        git_cmd,
        ["rev-list", "--count", f"{base_ref}..{remote_ref}"],
    )
    if commit_count_text is None:
        return None

    try:
        commit_count = int((commit_count_text or "").strip())
    except ValueError:
        return None

    commits_text = _git_capture(
        repo_dir,
        git_cmd,
        ["log", "--format=%s", "--no-merges", f"{base_ref}..{remote_ref}"],
    )
    commits = []
    if commits_text:
        for line in commits_text.splitlines():
            cleaned = _clean_markdown(line)
            if cleaned:
                commits.append(_truncate(cleaned, limit=180))
            if len(commits) >= max_commits:
                break

    remote_init = _git_capture(repo_dir, git_cmd, ["show", f"{remote_ref}:hermes_cli/__init__.py"])
    target_version = _extract_version(remote_init or "")

    releases = _load_release_summaries(
        repo_dir,
        git_cmd,
        base_ref=base_ref,
        remote_ref=remote_ref,
        max_releases=max_releases,
        max_highlights=max_release_highlights,
    )

    return UpdatePreview(
        current_version=CURRENT_VERSION,
        target_version=target_version,
        commit_count=commit_count,
        base_ref=base_ref,
        remote_ref=remote_ref,
        commits=commits,
        releases=releases,
    )


def format_update_preview(
    preview: UpdatePreview,
    *,
    confirm_command: Optional[str] = None,
    yes_command: Optional[str] = None,
) -> str:
    """Render an update preview suitable for CLI or gateway output."""
    commit_word = "commit" if preview.commit_count == 1 else "commits"
    lines = [
        "⚕ Hermes update preview",
        "",
        f"{preview.commit_count} {commit_word} ready from {preview.remote_ref}.",
        f"Current version: v{preview.current_version}",
    ]

    if preview.target_version and preview.target_version != preview.current_version:
        lines.append(f"Incoming version: v{preview.target_version}")

    compare_ref = "HEAD" if preview.base_ref == "HEAD" else f"local {preview.base_ref}"
    lines.append(f"Comparing {compare_ref} to {preview.remote_ref}.")

    if preview.commits:
        lines.extend(["", "Key changes:"])
        for commit in preview.commits:
            lines.append(f"- {commit}")
        remaining = preview.commit_count - len(preview.commits)
        if remaining > 0:
            lines.append(f"- ... and {remaining} more")

    if preview.releases:
        lines.extend(["", "Release highlights:"])
        for release in preview.releases:
            lines.append(f"- {release.version}")
            for highlight in release.highlights:
                lines.append(f"  - {highlight}")

    if confirm_command or yes_command:
        lines.append("")
        if confirm_command and yes_command:
            lines.append(f"Proceed with {confirm_command} or skip the preview via {yes_command}.")
        elif confirm_command:
            lines.append(f"Proceed with {confirm_command}.")
        else:
            lines.append(f"Skip the preview via {yes_command}.")

    return "\n".join(lines)
