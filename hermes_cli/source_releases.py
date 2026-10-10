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
from urllib.parse import urlsplit

from hermes_cli.github_api import github_token
from hermes_cli.update_channel import STABLE_TAG_RE, is_canary_tag

logger = logging.getLogger(__name__)
_PUBLIC_BASE = "https://hermes-assets.nousresearch.com"
OFFICIAL_REPOSITORY = "NousResearch/hermes-agent"
_GITHUB_ORIGIN = re.compile(
    r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
    r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$", re.IGNORECASE,
)
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

    ahead: bool = False

    @property
    def label(self) -> str:
        if self.ahead:
            return (f"{self.channel} v{self.version}; this checkout is newer and waits for the next release"
                    " (`hermes update --set-channel main` follows every commit)")
        return f"{self.channel} v{self.version} ({self.commit[:12]})" if self.commit else self.channel


def _resolve_channel(name: str, repository: str):
    """The only adapter to the validated requested/terminal records and manifest.

    ChannelReader owns HTTPS, authority and digests. The source adapter below
    admits its retirement constraints before any checkout operation. No legacy
    GitHub fallback is allowed when a record is unavailable; the one exception
    is an unpublished ``main`` record, which resolves to the main branch.
    """
    from hermes_cli.release_channels import ChannelReader

    return ChannelReader(_PUBLIC_BASE, repository=repository).resolve(name)


def resolve_source_target(channel: str, git_cmd=None, cwd=None, *, repository=None,
                          forward_only: bool = False) -> SourceTarget:
    """Resolve every subscription, including default labels, through R2.

    ``forward_only`` (an unchosen default subscription) pins a checkout that already
    contains the release to its own HEAD, so the update is a no-op instead of a downgrade.
    """
    target = _resolve_source_target(channel, git_cmd, cwd, repository=repository)
    if forward_only and target.commit and git_cmd is not None and cwd is not None:
        head = _head_containing(git_cmd, cwd, target.commit, target.repository)
        if head is not None and head != target.commit:
            from dataclasses import replace

            return replace(target, commit=head, ahead=True)
    return target


def _head_containing(git_cmd, cwd, commit: str, repository: str) -> str | None:
    """HEAD's sha unless HEAD is PROVEN not to contain ``commit``, then None.

    An unchosen default must never move a checkout backward on a guess, so an
    unknown relation (shallow history, GitHub unreachable) keeps HEAD.
    """
    from hermes_cli.source_check import _github_compare, source_git_env

    def run(*args):
        return subprocess.run(
            [*git_cmd, *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=10, stdin=subprocess.DEVNULL, env=source_git_env())

    head = run("rev-parse", "HEAD").stdout.strip()
    if not _SHA.fullmatch(head):
        return None
    ancestry = run("merge-base", "--is-ancestor", commit, head).returncode
    if ancestry == 0:
        return head
    # A shallow boundary makes exit 1 a guess; full history makes it proof.
    if ancestry == 1 and run("rev-parse", "--is-shallow-repository").stdout.strip() == "false":
        return None
    status = (_github_compare(commit, head, repository) or {}).get("status")
    return None if status in ("behind", "diverged") else head


def _resolve_source_target(channel: str, git_cmd=None, cwd=None, *, repository=None) -> SourceTarget:
    from hermes_cli.release_channels import ChannelNotFound, validate_name

    validate_name(channel)
    repository = repository or source_repository(git_cmd, cwd)
    if channel == "stable":
        return _resolve_stable(repository, git_cmd, cwd)
    try:
        resolved = _resolve_channel(channel, repository)
    except ChannelNotFound:
        if channel != "main":
            raise
        # main IS the source branch; its record can only add a retirement.
        # Until one is published, a checkout keeps following the branch via git.
        return SourceTarget(channel, channel, repository, branch="main")
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


def _describe_release_failure(failure: BaseException) -> str:
    """A transport failure names its cause (a rate limit is not "no release")."""
    if isinstance(failure, (urllib.error.URLError, TimeoutError)):
        from hermes_cli.github_api import describe_github_failure

        # The request that failed, not the configuration: a rejected token retries anonymously.
        return describe_github_failure(failure, authenticated=getattr(failure, "hermes_authenticated", False))
    return f"No published stable release could be verified: {failure}"


def _resolve_stable(repository: str, git_cmd=None, cwd=None) -> SourceTarget:
    """Stable IS the repository's latest published GitHub release: the tag a release
    publishes (non-draft, non-prerelease, strict vX.Y.Z), verified against origin.
    No R2 record or pointer is consulted; publishing the release is the only promotion.
    """
    try:
        tag, commit = _resolve_release("stable", git_cmd, cwd, repository=repository, pointer=False)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.warning("Could not resolve the stable source release: %s", exc)
        raise ValueError(_describe_release_failure(exc)) from exc
    return SourceTarget("stable", "stable", repository, commit=commit, version=str(tag).removeprefix("v"))


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


def _github_api_headers(url: str, token: str | None) -> dict[str, str]:
    """Headers for a release-channel HTTP read, authenticated on GitHub only.

    Channel resolution reads api.github.com for release/commit metadata; that
    budget is 60 anonymous requests/hour keyed on the client IP, so a shared
    exit (NAT/VPN/proxy) exhausts it and the 403 surfaces as "No published
    stable release could be verified". Send the configured credential (env →
    home .env → gh CLI, see hermes_cli.github_api) on GitHub API requests only —
    the R2 asset host must never see it.
    """
    headers = {
        "User-Agent": "hermes-update", "Cache-Control": "no-cache",
        "Accept": "application/json, text/html",
    }
    if token and urlsplit(url).hostname == "api.github.com":
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _read(url: str, *, missing_ok: bool = False) -> str | None:
    token = github_token() if urlsplit(url).hostname == "api.github.com" else None
    request = urllib.request.Request(url, headers=_github_api_headers(url, token))
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read(2 * 1024 * 1024).decode("utf-8-sig")
    except urllib.error.HTTPError as exc:
        # A token GitHub rejects drops this request to anonymous rather than
        # failing the channel on a stale credential — the same ladder
        # source_check.py uses.
        if token is not None and exc.code == 401:
            headers = dict(request.headers)
            headers.pop("Authorization", None)
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
                return response.read(2 * 1024 * 1024).decode("utf-8-sig")
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
    return bool(STABLE_TAG_RE.fullmatch(tag)) if channel == "stable" else (
        tag == tag.strip() and is_canary_tag(tag)
    )


def _published(release, channel: str) -> bool:
    return (isinstance(release, dict) and release.get("draft") is False
            and release.get("prerelease") is (channel == "canary")
            and _valid_tag(release.get("tag_name"), channel))


def _json(url: str):
    """GitHub API JSON with the updater's credential ladder (token, gh, anonymous).

    Anonymous api.github.com is 60 requests/hour per address, so an unauthenticated
    stable lookup fails for everyone behind one office, CI or container exit.
    """
    from hermes_cli.source_check import _request

    return json.loads(_request(url))


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


def _resolve_release(channel: str, git_cmd=None, cwd=None, *, repository=None,
                     pointer: bool = True) -> tuple[str, str]:
    """``resolve_source_release`` that raises the failure instead of returning ``(None, None)``."""
    if channel not in ("stable", "canary"):
        raise ValueError(f"Not a release channel: {channel}")
    repository = repository or source_repository(git_cmd, cwd)
    base = f"https://api.github.com/repos/{repository}"
    tag, pinned_sha = (_release_pointer(channel)
                       if pointer and repository.lower() == OFFICIAL_REPOSITORY.lower() else (None, None))
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


def resolve_source_release(channel: str, git_cmd=None, cwd=None, *, repository=None,
                           pointer: bool = True) -> tuple[str | None, str | None]:
    """Read published stable/canary release metadata (not channel discovery).

    The source updater's ``stable`` reads it with ``pointer=False``: GitHub's latest
    published release only. With ``pointer`` the R2 channel pointers outrank GitHub's
    release listing. A malformed pointer,
    draft, or tag/commit mismatch is not permission to select a different build.
    ``git_cmd`` resolves the selected tag on origin; ZIP callers omit it and
    resolve the same tag through GitHub's commit endpoint.
    """
    try:
        return _resolve_release(channel, git_cmd, cwd, repository=repository, pointer=pointer)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.warning("Could not resolve the %s source release: %s", channel, exc)
        return None, None
