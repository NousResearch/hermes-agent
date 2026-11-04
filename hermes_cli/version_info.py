"""Canonical runtime identity for Hermes.

Resolution order:
1. Install stamp (``install-stamp.json``) — written at build time by
   ``scripts/write_install_stamp.py`` for every packager (Docker, Nix, and
   the desktop app). The stamp is authoritative
   for packaged builds.
2. Live git — for unstamped source/dev installs with a ``.git`` directory.
3. Unknown — no stamp and no git. The provenance is unknown.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from hermes_cli.steward import UPDATE_MECHANISMS
from hermes_cli.update_channel import STABLE_TAG_RE


@dataclass(frozen=True)
class VersionInfo:
    base_version: str
    derived_version: str
    distance: int | None
    commit: str | None
    branch: str | None
    source: Literal["build", "commit-build", "ci", "docker", "fallback", "git", "local", "nix", "unknown"]
    dirty: bool = False
    commit_date: int | None = None
    distribution: Literal["docker", "nix", "desktop-app"] | None = None

    @property
    def display_version(self) -> str:
        """``<base>+<distance>`` when a release is known, else the derived identity.
        Keep a tagless checkout's ``git.<sha>`` rather than reducing it to unknown.
        """
        if self.base_version == "unknown":
            return self.derived_version
        return _derived_version(self.base_version, self.distance)


def _derived_version(
    base_version: str,
    distance: int | None,
    dirty: bool = False,
    short_commit: str | None = None,
) -> str:
    if distance and distance > 0:
        suffix = f"{distance}.g{short_commit}" if short_commit else str(distance)
        return f"{base_version}+{suffix}{'.dirty' if dirty and short_commit else ''}"
    if dirty and distance is None:
        return f"{base_version}+?"
    return base_version


_GIT_PROBE_TIMEOUT_SECONDS = 3.0
# One resolution runs on the first outbound request of a process (the nous provider stamps
# ``client=hermes-client-v<version>`` into every request body). A cold or contended checkout must
# not spend nine probes' worth of time there, so the whole walk shares one wall-clock budget.
_RESOLVE_BUDGET_SECONDS = 6.0
# After a truncated resolution, answer "unknown" for a while rather than pay the budget again on
# every request; the next window retries for real.
_RESOLVE_RETRY_SECONDS = 60.0


class _Budget:
    """Wall-clock allowance shared by the probes of ONE resolution."""

    def __init__(self, seconds: float | None) -> None:
        self.deadline = None if seconds is None else time.monotonic() + seconds
        # A probe we refused to start because the budget ran out. The answer is then incomplete,
        # which is not the same as a complete answer that happens to say "unknown".
        self.exhausted = False

    def timeout(self) -> float | None:
        """Per-probe timeout, or None when the budget no longer allows a probe."""
        if self.deadline is None:
            return _GIT_PROBE_TIMEOUT_SECONDS
        remaining = self.deadline - time.monotonic()
        if remaining <= 0.25:
            self.exhausted = True
            return None
        return min(_GIT_PROBE_TIMEOUT_SECONDS, remaining)


def _kill_probe(proc: subprocess.Popen) -> None:
    proc.kill()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _probe(argv: list[str], cwd: Path, timeout: float) -> str | None:
    """Run one read-only probe and return its stripped stdout, or None.

    ``stdout``/``stderr`` go to a temporary FILE, never to pipes. ``subprocess.run(timeout=...)``
    is not a hard bound on Windows: it kills the process it spawned, then re-reads the inherited
    pipes with NO timeout. Git for Windows launches the real git from a wrapper (``cmd/git.exe`` ->
    ``mingw64/bin/git.exe``), so a killed wrapper can leave a grandchild holding those handles and
    the second read never sees EOF — the MainThread parked in ``subprocess.communicate`` inside
    ``_run_git`` for 605 s, aborted by the turn-liveness watchdog while every request paid the
    stall. A file has no reader thread to join: after the deadline this kills what it can and
    reads whatever the probe managed to write.
    """
    with tempfile.TemporaryFile() as sink:
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=sink, stderr=sink,
                cwd=str(cwd), close_fds=True,
            )
        except (OSError, ValueError):
            return None
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_probe(proc)
            return None
        if proc.returncode != 0:
            return None
        sink.seek(0)
        value = sink.read().decode("utf-8", "replace").strip()
        return value or None


def _run_git(repo_dir: Path, *args: str, budget: _Budget | None = None) -> str | None:
    timeout = (budget or _Budget(None)).timeout()
    return None if timeout is None else _probe(["git", *args], repo_dir, timeout)


def _resolve_repo_dir() -> Path | None:
    """Use the executing checkout before a profile's optional clone."""
    repo_dir = Path(__file__).parent.parent.resolve()
    if (repo_dir / ".git").exists():
        return repo_dir
    # The PROCESS home: this is the running code's identity and is cached
    # process-wide, so a profile's context-local override must not pick it.
    from hermes_constants import get_process_hermes_home

    candidate = get_process_hermes_home() / "hermes-agent"
    if (candidate / ".git").exists():
        return candidate
    return None


def _parse_nonnegative(value: str | None) -> int | None:
    try:
        parsed = int(value or "")
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _calver_release_version(repo_dir: Path, budget: _Budget | None = None) -> tuple[str, int] | None:
    """The version the nearest CalVer release shipped, and the commits since it.

    Releases before semver tags existed are tagged ``vYYYY.M.D`` only; the
    version users actually run is in that tag's pyproject. Without this, a
    checkout past such a release would compare as "unknown" against plugins'
    ``requires_hermes``.
    """
    described = _run_git(repo_dir, "describe", "--tags", "--long", "--match", "v2[0-9][0-9][0-9].*", "HEAD", budget=budget)
    if not described:
        return None
    tag, count, _ = described.rsplit("-", 2)
    distance = _parse_nonnegative(count)
    try:
        project = tomllib.loads(_run_git(repo_dir, "show", f"{tag}:pyproject.toml", budget=budget) or "").get("project", {})
    except tomllib.TOMLDecodeError:
        return None
    version = project.get("version")
    if distance is None or not isinstance(version, str) or not STABLE_TAG_RE.fullmatch(f"v{version}"):
        return None
    return version, distance


# --- Install stamp reader ---------------------------------------------------

def _resolve_stamp_file() -> Path | None:
    """The executing tree's stamp (pm.paths.install_stamp_path owns the location)."""
    from pm.paths import install_stamp_path, repo_root

    p = install_stamp_path(repo_root())
    return p if p.is_file() else None


def _stamp_version_info(budget: _Budget | None = None) -> VersionInfo | None:
    """Read provenance from a build-time install stamp."""
    stamp_file = _resolve_stamp_file()
    if stamp_file is None:
        return None
    try:
        raw = stamp_file.read_text(encoding="utf-8-sig")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict) or "commit" not in data:
        return None

    # updateMechanism is required in every stamp. A stamp without it means
    # the writing build lane must be fixed, not tolerated.
    if data.get("updateMechanism") not in UPDATE_MECHANISMS:
        raise RuntimeError(
            f"install-stamp.json at {stamp_file} is missing a valid "
            f"'updateMechanism' (one of {', '.join(UPDATE_MECHANISMS)}). The "
            "build lane that wrote this stamp must pass --update-mechanism to "
            "scripts/write_install_stamp.py (or bake the field directly)."
        )
    # A light artifact ships no Python runtime. Reading a light stamp from
    # a Python process means the artifact was mispackaged.
    if data.get("payload") == "light":
        raise RuntimeError(
            f"install-stamp.json at {stamp_file} marks this artifact as 'light' "
            "(no agent runtime). No Python process can legitimately run from a "
            "light artifact — this build is mispackaged."
        )

    commit = data.get("commit") or None
    if not commit or set(commit) == {"0"}:
        # All-zero placeholder = fallback stamp, not real provenance.
        return None
    stamp_source = str(data.get("source") or "")
    if stamp_source == "git" and (stamp_file.parent / ".git").exists():
        live_commit = _run_git(stamp_file.parent, "rev-parse", "HEAD", budget=budget)
        if live_commit and live_commit != commit:
            return None

    base_version = data.get("baseVersion") or "unknown"
    display_version = data.get("displayVersion") or base_version
    distance = data.get("distance")
    if isinstance(distance, str):
        distance = _parse_nonnegative(distance)

    # ``source`` describes build provenance, while ``distribution`` identifies
    # the package form users installed. Keep both facts intact for support.
    source = (
        cast(Literal["build", "commit-build", "ci", "docker", "fallback", "git", "local", "nix", "unknown"], stamp_source)
        if stamp_source in {"commit-build", "ci", "docker", "fallback", "git", "local", "nix"}
        else "build"
    )
    distribution = data.get("distribution")
    if distribution not in {"docker", "nix", "desktop-app"}:
        distribution = None

    commit_date = data.get("commitDate")
    if not isinstance(commit_date, int):
        commit_date = None

    return VersionInfo(
        base_version,
        display_version,
        distance if isinstance(distance, int) else None,
        commit,
        data.get("branch") or None,
        source,
        bool(data.get("dirty")),
        commit_date,
        distribution,
    )


# --- Git provenance (source/dev installs) -----------------------------------


def _git_version_info(
    repo_dir: Path, *, include_untracked: bool = False, budget: _Budget | None = None,
) -> VersionInfo:
    budget = budget or _Budget(None)
    commit = _run_git(repo_dir, "rev-parse", "HEAD", budget=budget)
    # A detached HEAD has no branch. Leave the field None: every formatter
    # already prints the commit separately and handles a missing branch.
    branch = _run_git(repo_dir, "branch", "--show-current", budget=budget)
    commit_date_raw = _run_git(repo_dir, "log", "-1", "--format=%ct", "HEAD", budget=budget)
    commit_date: int | None = None
    if commit_date_raw and commit_date_raw.isdigit():
        commit_date = int(commit_date_raw)
    # -uno: skip the untracked-file scan. This runs on the startup-banner
    # path, and a full working-tree walk costs real time on large or cold
    # checkouts. Same semantics as write_install_stamp.py.
    # The timeout kills a slow status (a partial clone lazily fetching a tree, a cold or
    # scanned disk), and a status killed while refreshing the index strands .git/index.lock:
    # the update's own merge then dies on "File exists" (#132089). A read-only probe takes
    # no optional lock.
    status_command = ["git", "--no-optional-locks", "status", "--porcelain"]
    if not include_untracked:
        status_command.append("-uno")
    status_timeout = budget.timeout()
    dirty = status_timeout is not None and bool(_probe(status_command, repo_dir, status_timeout))

    tags = _run_git(repo_dir, "tag", "--merged", "HEAD", "--list", "v[0-9]*", budget=budget)
    releases = [
        tag[1:]
        for tag in (tags or "").splitlines()
        if STABLE_TAG_RE.fullmatch(tag)
    ]
    base_version = (
        max(releases, key=lambda value: tuple(int(part) for part in value.split(".")))
        if releases else "unknown"
    )
    distance = _parse_nonnegative(
        _run_git(repo_dir, "rev-list", "--count", f"v{base_version}..HEAD", budget=budget)
    ) if releases else None
    if not releases:
        base_version, distance = _calver_release_version(repo_dir, budget) or ("unknown", None)
    short_commit = _run_git(repo_dir, "rev-parse", "--short=7", "HEAD", budget=budget)
    if base_version == "unknown" and short_commit:
        display_version = f"git.{short_commit}{'.dirty' if dirty else ''}"
    else:
        display_version = _derived_version(base_version, distance, dirty, short_commit)

    return VersionInfo(
        base_version, display_version, distance, commit, branch, "git", dirty, commit_date
    )


# --- Cache + public API -----------------------------------------------------

_cached_version_info: VersionInfo | None = None
# A resolution that ran out of budget is answered but NOT cached; the next window retries.
_retry_after_monotonic: float | None = None

_UNKNOWN_VERSION = VersionInfo("unknown", "unknown", None, None, None, "unknown")


def _reset_version_info_cache() -> None:
    """Test-only cache reset."""
    global _cached_version_info, _retry_after_monotonic
    _cached_version_info = None
    _retry_after_monotonic = None


def _stamp_or_none(budget: _Budget | None = None) -> VersionInfo | None:
    """The install stamp's provenance, or None without a valid stamp.

    A malformed stamp (missing/illegal updateMechanism, mispackaged "light"
    payload) raises RuntimeError — the build lane that wrote it must be
    fixed, but a crashing CLI is the wrong failure mode for every uncached
    caller. Degrade to live git / "unknown" and let the authoring lane's own
    tests surface the malformed stamp.
    """
    try:
        return _stamp_version_info(budget)
    except RuntimeError:
        return None


def get_version_info() -> VersionInfo:
    """Return cached provenance from install stamp, git, or unknown.

    Bounded by construction: every probe is killed at its own deadline and the whole walk shares
    ``_RESOLVE_BUDGET_SECONDS``, so the first outbound request of a process cannot park the turn in
    a subprocess read (the 605 s stall this replaces). A walk that ran out of budget is answered
    but not cached, so a contended disk costs one bounded window instead of a permanently
    degraded identity.
    """
    global _cached_version_info, _retry_after_monotonic
    if _cached_version_info is not None:
        return _cached_version_info
    if _retry_after_monotonic is not None and time.monotonic() < _retry_after_monotonic:
        return _UNKNOWN_VERSION

    budget = _Budget(_RESOLVE_BUDGET_SECONDS)

    # 1. Install stamp (packaged builds: Docker, Nix)
    info = _stamp_or_none(budget)

    # 2. Live git (source/dev installs)
    if info is None:
        repo_dir = _resolve_repo_dir()
        if repo_dir is not None:
            info = _git_version_info(repo_dir, budget=budget)

    # 3. Unknown — no stamp, no git (or no budget left to prove otherwise)
    if info is None:
        info = _UNKNOWN_VERSION

    if budget.exhausted:
        _retry_after_monotonic = time.monotonic() + _RESOLVE_RETRY_SECONDS
        return info
    _cached_version_info = info
    _retry_after_monotonic = None
    return info


def hermes_user_agent(product: str = "hermes-cli") -> str:
    """``<product>/<base version>`` for an outbound request.

    Call it where the request is built, never at module level: a source checkout resolves the
    release from git, so a module-level read made every Hermes start pay that walk whether or not
    it ever sent the request.
    """
    return f"{product}/{get_version_info().base_version}"


def get_code_identity(refresh: bool = False) -> dict:
    """Return the running install's code identity as a flat dict.

    Shape: ``{"sha": full sha | None, "short_sha": str | None, "version":
    base package version | None, "source": str, "commit_date": epoch s | None}`` — what the update
    receipt, runtime inventory, and gateway status stamping consume.
    Backed by :func:`get_version_info` (install stamp first, live git
    second, unknown third), so every consumer shares one resolution
    policy. Returned dicts are fresh copies, never the shared cache.

    ``refresh=True`` drops the per-process cache first — the updater uses
    it to re-read identity after swapping the checkout out from under the
    running process.
    """
    global _cached_version_info
    if refresh:
        _cached_version_info = None
    info = get_version_info()
    return {
        "sha": info.commit,
        "short_sha": info.commit[:8] if info.commit else None,
        "version": info.base_version,
        "source": info.source,
        "commit_date": info.commit_date,
    }


def read_checkout_sha() -> str | None:
    """The running install's commit, re-read now: the stamp's, else ``git rev-parse HEAD``.

    Same answer as ``get_code_identity(refresh=True)["sha"]`` for one git call instead of the
    whole provenance walk (the tag scan alone costs ~0.35 s in a large clone), which the
    pending-restart check on every CLI start was paying just to compare SHAs. Drops the cached
    identity like that refresh does: the checkout may have moved under this process.
    """
    _reset_version_info_cache()
    stamp = _stamp_or_none()
    if stamp is not None:
        return stamp.commit
    repo_dir = _resolve_repo_dir()
    return _run_git(repo_dir, "rev-parse", "HEAD") if repo_dir is not None else None
