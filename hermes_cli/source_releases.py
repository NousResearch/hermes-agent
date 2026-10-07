"""Resolve promoted source releases, never infer publication from a Git tag."""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import http.client
import json
import logging
import re
import subprocess
import urllib.error
import urllib.request

from hermes_cli.update_channel import STABLE_TAG_RE, is_canary_tag
from hermes_cli.release_channels import UPDATER_HEADERS

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
    ancestry_verified: bool = True

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
    GitHub fallback is allowed when a record is unavailable; the one exception
    is an unpublished ``main`` record, which resolves to the main branch.
    """
    from hermes_cli.release_channels import ChannelReader

    return ChannelReader(_PUBLIC_BASE, repository=repository).resolve(name)


def resolve_source_target(channel: str, git_cmd=None, cwd=None, *, repository=None,
                          strict: bool = True) -> SourceTarget:
    """Resolve every subscription, including default labels, through R2.

    ``strict=False`` marks passive checkers: they never fetch, so an
    unprovable retirement ancestry stays permissive and is reported through
    :attr:`SourceTarget.ancestry_verified` instead of refusing the update.
    """
    from hermes_cli.release_channels import ChannelNotFound, validate_name

    validate_name(channel)
    repository = repository or source_repository(git_cmd, cwd)
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
        verified = _retirement_commit_proof(request, terminal, git_cmd, cwd, strict)
    else:
        verified = True
    return SourceTarget(channel, destination, repository, commit=commit,
                        version=request["sourceVersion"], build_id=request["buildId"],
                        ancestry_verified=verified)


def _stamp_commit(cwd) -> str | None:
    """The installed build's commit from its install stamp, or None.

    Packaged trees (the no-Git Windows ZIP path and embedded desktop payloads)
    carry ``install-stamp.json`` naming the exact source commit they were
    built from; the updater's own stamp writer binds it to the checkout.
    """
    from pathlib import Path

    if cwd is None:
        return None
    from pm.paths import install_stamp_path

    try:
        commit = json.loads(install_stamp_path(Path(cwd)).read_text(encoding="utf-8-sig")).get("commit")
    except (OSError, ValueError):
        return None
    return commit if isinstance(commit, str) and _SHA.fullmatch(commit) else None


def _retirement_commit_proof(request: dict, terminal: dict, git_cmd, cwd,
                             strict: bool) -> bool:
    """Prove the installed source is not newer than the qualified retirement build.

    Returns whether the ancestry was proven. Raises for the rollback shapes
    every transport can see: an installed version/commit newer than the
    qualified build, a descendant commit, and — on the strict no-Git apply
    path — an install that carries no evidence of being older than the
    pinned build (no provably older version and no stamp naming it). Passive
    checkers and Git-verified installs instead fail open to the unverified
    answer rather than stranding the retirement (main's posture, with the
    newer-source hole the reviews demonstrated closed).

    ``terminal["head"]`` is a raw record head (see :func:`_head`). The strict
    Git path never needs publication metadata from it; the no-Git stamp check
    additionally admits a stamp naming the pinned target commit.
    """
    from pathlib import Path
    import tomllib
    if cwd is None:
        return True
    proven_older_version = False
    version_file = Path(cwd) / "pyproject.toml"
    if version_file.exists():
        with version_file.open("rb") as file:
            project = tomllib.load(file).get("project")
        installed_version = project.get("version") if isinstance(project, dict) else None
        if not isinstance(installed_version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", installed_version, re.ASCII):
            raise ValueError("Source retirement cannot verify the installed source version")
        if tuple(map(int, installed_version.split("."))) > tuple(map(int, request["sourceVersion"].split("."))):
            raise ValueError("Source retirement would downgrade a newer source version; select the destination channel explicitly")
        proven_older_version = tuple(map(int, installed_version.split("."))) < tuple(map(int, request["sourceVersion"].split(".")))
    if git_cmd is not None:
        return _git_retirement_proof(request, terminal, git_cmd, cwd, strict)
    # No Git: the ZIP updater and the embedded desktop checker. The version
    # floor above is the only ordering evidence this transport has, and it
    # already refused the provably newer installs. Publication heads carry
    # buildId/sequence/manifestKey/sha256, not a commit, so Git-grade ancestry
    # is unavailable here: a stamp naming the pinned target is proven, while
    # any other stamp is unproven rather than proven-newer -- refusing it
    # would strand supported older installs that the permissive posture
    # admits. A stamp proves identity only against the pinned target: a
    # destination-head comparison (when a head publishes a commit field)
    # cannot distinguish "sitting on the destination" from "same version,
    # different build", which is exactly the downgrade the retirement pins.
    stamp = _stamp_commit(cwd)
    if stamp is not None and stamp == request["commit"]:
        return True
    if strict and version_file.exists() and not proven_older_version:
        # The strict apply path cannot treat unverified ordering as
        # authorization when the tree carries build evidence: publication
        # heads carry buildId/sequence/sha256 and no commit, so an equal
        # version with a different (or absent) stamp is exactly the
        # same-version different-build rollback the retirement pins — refuse
        # it with the explicit-destination remedy instead of applying the
        # pinned archive. A tree with no build evidence at all keeps main's
        # permissive ZIP/desktop flow: the version floor and the packaged
        # stamps are the only ordering authorities this transport has, and
        # refusing evidence-less installs would strand the supported
        # updater mode (the tagless-ZIP contract).
        raise ValueError(
            "Source retirement cannot verify that this install is not newer than the "
            f"qualified {terminal['name']} build; select the destination channel explicitly "
            f"(hermes update --channel {terminal['name']})")
    return False


def _git_retirement_proof(request: dict, terminal: dict, git_cmd, cwd,
                          strict: bool) -> bool:
    """Git half of :func:`_retirement_commit_proof`; see there for the contract."""
    from pathlib import Path

    from hermes_cli.source_check import source_git_env

    def run_git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*git_cmd, *args], cwd=cwd,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
            stdin=subprocess.DEVNULL, env=source_git_env(),
        )

    result = run_git("rev-list", "--ancestry-path", f"{request['commit']}..HEAD")
    if result.returncode == 0 and result.stdout.strip():
        raise ValueError("Source retirement would downgrade a newer source commit; select the destination channel explicitly")
    if result.returncode == 0:
        # Git saw the target. On a full history rev-list/merge-base are
        # decisive; on a shallow checkout a graft can DISCONNECT the histories
        # (a depth-1 fetch of the target leaves it parentless), which makes an
        # OLDER HEAD read as a descendant and any divergent pair unprovable.
        # Classify first, refill the grafted history once, then re-judge.
        descendant = bool(result.stdout.strip())
        shallow = run_git("rev-parse", "--is-shallow-repository")
        shallow_p = shallow.returncode == 0 and shallow.stdout.strip() == "true"
        if descendant and not shallow_p:
            raise ValueError("Source retirement would downgrade a newer source commit; select the destination channel explicitly")
        proof = run_git("merge-base", "--is-ancestor", "HEAD", request["commit"])
        if proof.returncode == 0:
            return True
        if not shallow_p:
            raise ValueError("Source retirement cannot verify that the installed source is equal to or an ancestor of the destination; select the destination channel explicitly")
        if not strict:
            # A passive checker never fetches: the graft hides the ancestry
            # (both the descendant and the divergent reading are unreliable),
            # so fail open to the unverified answer instead of stranding the
            # retirement. The apply path re-judges decisively below.
            return False
        # The apply path CAN fetch: judge the fetched objects first, refill
        # the grafted history only when the proof is still inconclusive, then
        # re-judge on real history (all owned by the shared classifier).
        return _classify_strict_ancestry(request, git_cmd, cwd, run_git)
    # The target commit is not locally visible (the depth-1 era never healed,
    # and update_cmd keeps shallow installs shallow until the apply fetch).
    # main's fetch-free refusal still applies first: an install already sitting
    # on the newer destination build is visibly newer without local ancestry.
    # The re-read is conditional: it can only add a refusal when the
    # destination moved past the qualified build (a strictly newer sequence).
    # When it has not, the first read already supplied the pinned target, and
    # the answer must not depend on a second publication fetch succeeding.
    head = terminal.get("head") or {}
    if head.get("sequence") is not None and head["sequence"] > request.get("sequence", 0):
        try:
            current_manifest = _resolve_channel(terminal["name"], request["repository"]).manifest
        except (OSError, ValueError, subprocess.SubprocessError):
            # An unreadable destination must not strand the retirement: the
            # guarded shapes are re-checked from real history after the fetch
            # below, which provably refuses a destination descendant.
            current_manifest = None
        current = None if current_manifest is None else current_manifest["request"]
        installed = run_git("rev-parse", "HEAD").stdout.strip()
        if (current is not None and _SHA.fullmatch(installed)
                and installed == current["commit"] and installed != request["commit"]):
            raise ValueError("Source retirement would downgrade the newer destination build; select the destination channel explicitly")
    if not strict:
        # A passive checker never fetches: stay permissive with main's answer
        # and let the presentation flag the unproven ancestry instead.
        return False
    # The apply path CAN fetch. Pull exactly the pinned commit (cheap even on
    # shallow installs, unlike a full unshallow) and judge on real history
    # through the same bounded shallow-aware classifier as the refilled path
    # above.
    try:
        _strict_git_fetch(git_cmd, cwd, ["fetch", "--no-tags", "origin", request["commit"]], 300)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Source retirement cannot verify that the installed source is not newer; select the destination channel explicitly") from exc
    return _classify_strict_ancestry(request, git_cmd, cwd, run_git)


def _strict_git_fetch(git_cmd, cwd, fetch_args, timeout: int) -> subprocess.CompletedProcess:
    """The one strict-fetch lane: guarded preparation, then the custody fetch.

    Both apply-path fetch branches (the grafted-history refill and the pinned
    -target fetch) share this: the age/liveness-guarded stale-lock cleanup
    (an abandoned ``shallow.lock`` otherwise fails every fetch with exit 128
    before the updater's own later recovery is reachable), then the fetch
    itself through the updater's custody runner (git_argv custody config,
    job-bound child on Windows, the owner-death watchdog's fetch lane) instead
    of a bare ``subprocess.run`` a killed updater could orphan.
    """
    from pathlib import Path

    from hermes_cli.gitlock import clear_stale_git_locks
    from hermes_cli.source_check import source_git_env
    from hermes_cli.update_custody import run_git as custody_git

    clear_stale_git_locks(Path(cwd))
    fetch = custody_git(
        git_cmd, fetch_args, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
        stdin=subprocess.DEVNULL, env=source_git_env(),
    )
    if fetch.returncode != 0:
        raise ValueError("Source retirement cannot verify that the installed source is not newer; select the destination channel explicitly")
    return fetch


def _classify_strict_ancestry(request: dict, git_cmd, cwd, run_git) -> bool:
    """The one post-fetch verdict, shared by both strict fetch branches.

    Judge the fetched objects first: when the ancestry is already provable
    (``rev-list`` succeeds empty and ``HEAD`` is an ancestor of the target),
    the verdict returns without any history refill — a shallow boundary can
    survive the pinned fetch while the ``HEAD -> target`` relation is already
    decided, and unconditionally ``--unshallow``-ing it would download the
    whole blob history behind the boundary (the reason the updater owns
    ``fetch_full_commit_graph`` and its ``blob:none`` conversion). Only an
    inconclusive shallow result enters the guarded refill: ``--unshallow``
    under the same guarded fetch lane, then re-judge on real history; only a
    real-history descendant or divergent pair is refused, so the first attempt
    admits an eligible older install instead of requiring a retry.
    """
    result = run_git("rev-list", "--ancestry-path", f"{request['commit']}..HEAD")
    if result.returncode == 0 and not result.stdout.strip():
        proof = run_git("merge-base", "--is-ancestor", "HEAD", request["commit"])
        if proof.returncode == 0:
            return True
    shallow = run_git("rev-parse", "--is-shallow-repository")
    if shallow.returncode == 0 and shallow.stdout.strip() == "true":
        try:
            _strict_git_fetch(git_cmd, cwd,
                              ["fetch", "--unshallow", "--no-tags", "origin", request["commit"]], 900)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass  # the classification below keeps the refusal honest
    result = run_git("rev-list", "--ancestry-path", f"{request['commit']}..HEAD")
    if result.returncode != 0:
        raise ValueError("Source retirement cannot verify that the installed source is not newer; select the destination channel explicitly")
    if result.stdout.strip():
        raise ValueError("Source retirement would downgrade a newer source commit; select the destination channel explicitly")
    proof = run_git("merge-base", "--is-ancestor", "HEAD", request["commit"])
    if proof.returncode != 0:
        raise ValueError("Source retirement cannot verify that the installed source is equal to or an ancestor of the destination; select the destination channel explicitly")
    return True


def _read(url: str, *, missing_ok: bool = False) -> str | None:
    request = urllib.request.Request(
        url,
        headers={**UPDATER_HEADERS, "Accept": "application/json, text/html"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read(2 * 1024 * 1024).decode("utf-8-sig")
    except urllib.error.HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        raise
    except http.client.HTTPException as exc:
        raise OSError("source release response was incomplete") from exc


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
