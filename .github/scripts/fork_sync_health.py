"""Fork CI's code-health ratchet, aware of upstream release syncs (FORK.md § Fork CI).

The ratchet judges the diff against a base. When the change brings in an upstream release (a sync
PR, or the push to main that merges one), that diff holds every file upstream added or moved, so
upstream's own code reads as new code and blocks. Such a change is judged twice instead, against
the base and against the release, and a finding blocks only if both runs report it:

- new against the base only: upstream's code, arrived unchanged with the release;
- new against the release only: fork delta the base already carried (existing fork debt);
- new against both: the fork delta or the merge resolution made it worse. That is ours.

Any other change gets the plain single run, exactly as before.

Usage (from the repo root): python .github/scripts/fork_sync_health.py --base <rev>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

UPSTREAM = "https://github.com/NousResearch/hermes-agent"
_STABLE = re.compile(r"refs/tags/v\d+\.\d+\.\d+")
Key = tuple[str, str, str]
_GIT_TIMEOUT = 120  # ls-remote is the only network call; the rest are local lookups
_HEALTH_TIMEOUT = 3600  # a sync PR's base-side run measures every upstream file it brings in


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False, timeout=_GIT_TIMEOUT)


def stable_releases(ls_remote: str) -> dict[str, str]:
    """``{tag: commit}`` for upstream's ``vX.Y.Z`` tags, annotated tags at their peeled commit.
    rc, abandoned-rc, canary and channel tags are not releases. Upstream has used both semver and
    CalVer numbers, so the numbers say nothing about which release is newer."""
    commits: dict[str, str] = {}
    for line in ls_remote.splitlines():
        sha, _, ref = line.partition("\t")
        peeled = ref.endswith("^{}")
        ref = ref.removesuffix("^{}")
        if not _STABLE.fullmatch(ref):
            continue
        tag = ref.removeprefix("refs/tags/")
        if peeled or tag not in commits:
            commits[tag] = sha
    return commits


def incoming_release(base: str) -> tuple[str, str] | None:
    """The newest (by commit date) upstream release HEAD contains and ``base`` does not, if any."""
    listing = _git("ls-remote", "--tags", UPSTREAM, "refs/tags/v*")
    if listing.returncode:
        raise SystemExit(f"fork_sync_health: cannot list upstream tags: {listing.stderr.strip()}")
    incoming = [
        (tag, commit) for tag, commit in stable_releases(listing.stdout).items()
        # Only commits already fetched with the fork's history can be in HEAD.
        if not _git("cat-file", "-e", f"{commit}^{{commit}}").returncode
        and not _git("merge-base", "--is-ancestor", commit, "HEAD").returncode
        and _git("merge-base", "--is-ancestor", commit, base).returncode
    ]
    if not incoming:
        return None
    return max(incoming, key=lambda item: int(_git("log", "-1", "--format=%ct", item[1]).stdout))


def _key(finding: dict) -> Key:
    return finding["path"], finding["rule"], finding["scope"]


def _blocks(finding: dict) -> bool:
    return bool(finding["blocking"]) and finding["allowed_reason"] is None


def blocking_in_both(against_base: list[dict], against_release: list[dict]) -> list[dict]:
    """The base run's blocking findings that the release run also reports blocking, matched by
    file, rule and scope (line and cap legitimately differ between the two bases)."""
    release_keys = {_key(f) for f in against_release if _blocks(f)}
    return [f for f in against_base if _blocks(f) and _key(f) in release_keys]


def _findings(base: str) -> list[dict]:
    run = subprocess.run([sys.executable, "-m", "scripts.code_health", "--base", base, "--json"],
                         capture_output=True, text=True, check=False, timeout=_HEALTH_TIMEOUT)
    if run.returncode not in (0, 1):  # 2 is a broken run, not a verdict
        sys.stderr.write(run.stderr)
        raise SystemExit(run.returncode)
    return json.loads(run.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="the revision the change is judged against")
    args = parser.parse_args(argv)
    release = incoming_release(args.base)
    if release is None:
        return subprocess.run([sys.executable, "-m", "scripts.code_health", "--base", args.base],
                              check=False, timeout=_HEALTH_TIMEOUT).returncode
    tag, commit = release
    print(f"This change brings in upstream {tag} ({commit[:12]}): judging it against "
          f"{args.base[:12]} and the release.")
    against_base, against_release = _findings(args.base), _findings(commit)
    ours = blocking_in_both(against_base, against_release)
    print(f"{sum(map(_blocks, against_base))} blocking against the base, "
          f"{sum(map(_blocks, against_release))} against the release, {len(ours)} in both.")
    for f in ours:
        print(f"{f['path']}:{f['line']}  {f['rule']}  {f['scope']}\n    {f['detail']}")
    if ours:
        print("\nThese are new against both sides of the merge. Reproduce: "
              f"python -m scripts.code_health --base {args.base[:12]} (and --base {commit[:12]}).")
    return 1 if ours else 0


if __name__ == "__main__":
    sys.exit(main())
