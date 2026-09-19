"""Resolve promoted source releases, never infer publication from a Git tag."""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import json
import logging
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
        from hermes_cli.source_check import source_git_env

        result = subprocess.run(
            [*git_cmd, "config", "--get", "remote.origin.url"], cwd=cwd,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
            stdin=subprocess.DEVNULL, env=source_git_env(),
        )
        match = _GITHUB_ORIGIN.fullmatch(result.stdout.strip())
        if result.returncode == 0 and match:
            return match[1]
    return OFFICIAL_REPOSITORY


@dataclass(frozen=True)
class SourceTarget:
    """A pinned source build or an explicitly declared source-branch delivery."""

    requested_channel: str
    channel: str
    repository: str
    commit: str | None = None
    branch: str | None = None
    version: str | None = None
    build_id: str | None = None

    @property
    def retired(self) -> bool:
        return self.requested_channel != self.channel

    @property
    def label(self) -> str:
        return f"{self.channel} v{self.version} ({self.commit[:12]})" if self.commit else self.channel


def _resolve_channel(name: str, repository: str):
    """The only adapter to the validated requested/terminal records and manifest.

    ChannelReader owns HTTPS, authority and digests. The source adapter below
    admits its retirement constraints before any checkout operation. No legacy
    GitHub fallback is allowed when a record is unavailable.
    """
    from hermes_cli.release_channels import ChannelReader

    return ChannelReader(_PUBLIC_BASE, repository=repository).resolve(name)


def resolve_source_target(channel: str, git_cmd=None, cwd=None, *, repository=None) -> SourceTarget:
    """Resolve every subscription, including default labels, through R2."""
    from hermes_cli.release_channels import validate_name

    validate_name(channel)
    repository = repository or source_repository(git_cmd, cwd)
    resolved = _resolve_channel(channel, repository)
    terminal = resolved.terminal
    if terminal["repository"].lower() != repository.lower():
        raise ValueError("Channel repository does not match this source installation")
    destination = validate_name(terminal["name"])
    if terminal["policy"] == "source-branch":
        if resolved.requested["state"] == "retired":
            raise ValueError("Source retirement requires a published destination commit")
        delivery = terminal["delivery"]
        if delivery["kind"] != "source-branch":
            raise ValueError("Channel has no source-branch delivery")
        return SourceTarget(channel, destination, repository, branch=delivery["branch"])
    if resolved.manifest is None:
        raise ValueError(f"No build published for channel {destination}")
    request = resolved.manifest["request"]

    commit = request["commit"]
    if not isinstance(commit, str) or not _SHA.fullmatch(commit):
        raise ValueError("Channel build has no exact source commit")
    if resolved.requested["state"] == "retired":
        _refuse_retirement_downgrade(request, terminal, git_cmd, cwd)
    return SourceTarget(channel, destination, repository, commit=commit,
                        version=request["sourceVersion"], build_id=request["buildId"])


def _refuse_retirement_downgrade(request: dict, terminal: dict, git_cmd, cwd) -> None:
    """Qualification of preview data is not permission to roll back newer source."""
    from pathlib import Path
    import tomllib

    if cwd is None:
        return
    version_file = Path(cwd) / "pyproject.toml"
    if version_file.exists():
        with version_file.open("rb") as file:
            project = tomllib.load(file).get("project")
        installed_version = project.get("version") if isinstance(project, dict) else None
        if not isinstance(installed_version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", installed_version, re.ASCII):
            raise ValueError("Source retirement cannot verify the installed source version")
        if tuple(map(int, installed_version.split("."))) > tuple(map(int, request["sourceVersion"].split("."))):
            raise ValueError("Source retirement would downgrade a newer source version; select the destination channel explicitly")
    if git_cmd is not None:
        from hermes_cli.source_check import source_git_env

        result = subprocess.run(
            [*git_cmd, "rev-list", "--ancestry-path", f"{request['commit']}..HEAD"], cwd=cwd,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
            stdin=subprocess.DEVNULL, env=source_git_env(),
        )
        # The target need not exist locally before the updater's fetch. When it
        # does, any descendants prove that this pinned build would roll us back.
        if result.returncode == 0 and result.stdout.strip():
            raise ValueError("Source retirement would downgrade a newer source commit; select the destination channel explicitly")
        if terminal["head"]["sequence"] > request["sequence"]:
            # Shallow checkouts may lack the qualified commit, even when HEAD is
            # today's stable build. Read it with the protocol's full digest checks.
            current_manifest = _resolve_channel(terminal["name"], request["repository"]).manifest
            if current_manifest is None:
                raise ValueError("Source retirement cannot verify the current destination build")
            current = current_manifest["request"]
            installed = subprocess.run(
                [*git_cmd, "rev-parse", "HEAD"], cwd=cwd, check=True,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
                stdin=subprocess.DEVNULL, env=source_git_env(),
            ).stdout.strip()
            if installed == current["commit"] and installed != request["commit"]:
                raise ValueError("Source retirement would downgrade the newer destination build; select the destination channel explicitly")


def _read(url: str, *, missing_ok: bool = False) -> str | None:
    request = urllib.request.Request(url, headers={
        "User-Agent": "hermes-update", "Cache-Control": "no-cache",
        "Accept": "application/json, text/html",
    })
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read(2 * 1024 * 1024).decode("utf-8-sig")
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
    """Read historical stable/canary release metadata (not channel discovery).

    Runtime check/apply use ``resolve_source_target`` and never fall back here.
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
            from hermes_cli.source_check import source_git_env

            ref = f"refs/tags/{tag}"
            result = subprocess.run(
                [*git_cmd, "ls-remote", "--tags", "origin", ref, ref + "^{}"],
                cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                check=True, timeout=60, stdin=subprocess.DEVNULL,
                env=source_git_env(),
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
