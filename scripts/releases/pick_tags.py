"""Sample installed release baselines by creation time across version schemes."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, encoding="utf-8", timeout=120)


def pick_tags(repo: Path, count: int, exclude: str = "") -> list[str]:
    if not 1 <= count <= 10:
        raise ValueError("Tag count must be between one and ten")
    # Enumerate origin's tags without fetching any ref: ls-remote lists refs
    # only. The wire protocol cannot match tighter than a glob, so the
    # version-scheme filter stays here.
    raw = _git(repo, "ls-remote", "--refs", "--tags", "origin", "v*")
    tags = [line.split("\t")[1].removeprefix("refs/tags/") for line in raw.splitlines()]
    tags = [tag for tag in tags if re.fullmatch(r"v\d+\.\d+\.\d+(?:\.\d+)?", tag) and tag != exclude]
    if not tags:
        raise ValueError("No released baseline tags remain")
    # Creation dates live inside the tag objects, which ls-remote cannot read:
    # date sort keys resolve against the local object store and fail loudly
    # when an object is absent. Fetch exactly the filtered release tags --
    # never the canary and backup refs a full --tags fetch would pull -- so a
    # checkout with no tags at all still gets the same chronological order.
    _git(repo, "fetch", "--quiet", "--no-tags", "origin", *(f"+refs/tags/{tag}:refs/tags/{tag}" for tag in tags))
    # Order the just-fetched refs by creation date locally: no second remote
    # query to fail or to diverge from what the fetch stored above.
    raw = _git(repo, "for-each-ref", "--sort=creatordate", "--format=%(refname:short)",
               *(f"refs/tags/{tag}" for tag in tags))
    ordered = raw.splitlines()
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[-1]]
    return [ordered[(slot * (len(ordered) - 1) * 2 + count - 1) // ((count - 1) * 2)] for slot in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--exclude-ref", default="")
    args = parser.parse_args()
    print(json.dumps(pick_tags(args.repo, args.count, args.exclude_ref)))


if __name__ == "__main__":
    main()
