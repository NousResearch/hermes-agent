"""Resolve promoted source releases, never infer publication from a Git tag."""
from __future__ import annotations

from html.parser import HTMLParser
import json
import logging
import os
import re
import subprocess
import urllib.error
import urllib.request

from hermes_cli.update_channel import is_canary_tag

logger = logging.getLogger(__name__)
_PUBLIC_BASE = "https://hermes-assets.nousresearch.com"
OFFICIAL_REPOSITORY = "NousResearch/hermes-agent"
_GITHUB_ORIGIN = re.compile(
    r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
    r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$", re.IGNORECASE,
)
_STABLE_TAG = re.compile(r"v(?:0|[1-9]\d{0,2})\.\d+\.\d+")
_SHA = re.compile(r"[0-9a-f]{40}")


def source_repository(git_cmd=None, cwd=None) -> str:
    """GitHub forks own their releases; other origins must mirror official tags."""
    if git_cmd is not None:
        result = subprocess.run(
            [*git_cmd, "config", "--get", "remote.origin.url"], cwd=cwd,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        match = _GITHUB_ORIGIN.fullmatch(result.stdout.strip())
        if result.returncode == 0 and match:
            return match[1]
    return OFFICIAL_REPOSITORY


def _read(url: str, *, missing_ok: bool = False) -> str | None:
    request = urllib.request.Request(url, headers={
        "User-Agent": "hermes-update", "Cache-Control": "no-cache",
        "Accept": "application/json, text/html",
    })
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read(2 * 1024 * 1024).decode("utf-8")
    except urllib.error.HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        raise


class _BuildMetadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        if tag == "meta" and fields.get("name") == "hermes-build":
            self.tags.append(fields.get("content"))


def _valid_tag(tag, channel: str) -> bool:
    if not isinstance(tag, str):
        return False
    return bool(_STABLE_TAG.fullmatch(tag)) if channel == "stable" else (
        tag == tag.strip() and is_canary_tag(tag)
    )


def _published(release, channel: str) -> bool:
    return (isinstance(release, dict) and release.get("draft") is False
            and release.get("prerelease") is (channel == "canary")
            and _valid_tag(release.get("tag_name"), channel))


def _json(url: str):
    text = _read(url)
    assert text is not None
    return json.loads(text)


def _published_fallback(channel: str, base: str) -> dict:
    if channel == "stable":
        release = _json(f"{base}/releases/latest")
        if _published(release, channel):
            return release
    else:
        # GitHub lists newest releases first. Bound the scan; failure must
        # never turn into an arbitrary Git-tag update.
        for page in range(1, 11):
            entries = _json(f"{base}/releases?per_page=100&page={page}")
            if not isinstance(entries, list):
                break
            for release in entries:
                if _published(release, channel):
                    return release
            if len(entries) < 100:
                break
    raise ValueError(f"No published {channel} release")


def _release_pointer(channel: str) -> tuple[str | None, str | None]:
    # Stable's completion job writes this before publishing the GitHub draft.
    # Publication is checked separately, so that interval fails closed.
    if channel == "stable":
        text = _read(f"{_PUBLIC_BASE}/releases/stable/release-candidates.json", missing_ok=True)
        if text is not None:
            data = json.loads(text)
            if (not isinstance(data, dict) or not _valid_tag(data.get("tag"), channel)
                    or not isinstance(data.get("commit"), str) or not _SHA.fullmatch(data["commit"])):
                raise ValueError("Invalid stable release pointer")
            return data["tag"], data["commit"]
    text = _read(f"{_PUBLIC_BASE}/releases/{channel}/index.html", missing_ok=True)
    if text is None:
        return None, None
    page = _BuildMetadata()
    page.feed(text)
    if len(page.tags) != 1 or not _valid_tag(page.tags[0], channel):
        raise ValueError(f"Invalid {channel} release pointer")
    return page.tags[0], None


def resolve_source_release(channel: str, git_cmd=None, cwd=None, *, repository=None) -> tuple[str | None, str | None]:
    """Return the published channel's tag and exact commit, or no target on failure.

    Channel pointers outrank GitHub's release listing. A malformed pointer,
    draft, or tag/commit mismatch is not permission to select a different build.
    ``git_cmd`` resolves the selected tag on origin; ZIP callers omit it and
    resolve the same tag through GitHub's commit endpoint.
    """
    if channel not in ("stable", "canary"):
        raise ValueError(f"Not a release channel: {channel}")
    try:
        repository = repository or source_repository(git_cmd, cwd)
        base = f"https://api.github.com/repos/{repository}"
        tag, pinned_sha = (_release_pointer(channel)
                           if repository.lower() == OFFICIAL_REPOSITORY.lower() else (None, None))
        if tag is None:
            release = _published_fallback(channel, base)
            tag = release["tag_name"]
        else:
            release = _json(f"{base}/releases/tags/{tag}")
        if not _published(release, channel) or release["tag_name"] != tag:
            raise ValueError(f"{tag} is not a published {channel} release")
        commit = _json(f"{base}/commits/{tag}")
        sha = commit.get("sha") if isinstance(commit, dict) else None
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ValueError(f"No published commit for release {tag}")
        if git_cmd is not None:
            ref = f"refs/tags/{tag}"
            result = subprocess.run(
                [*git_cmd, "ls-remote", "--tags", "origin", ref, ref + "^{}"],
                cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                check=True, timeout=60, stdin=subprocess.DEVNULL,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
            )
            refs = dict((parts[1], parts[0]) for line in result.stdout.splitlines()
                        if len(parts := line.split()) == 2)
            if refs.get(ref + "^{}", refs.get(ref)) != sha:
                raise ValueError(f"Origin tag {tag} does not match the published release commit")
        if pinned_sha is not None and sha != pinned_sha:
            raise ValueError(f"Release {tag} no longer matches its published commit")
        return tag, sha
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.warning("Could not resolve the %s source release: %s", channel, exc)
        return None, None
