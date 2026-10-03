"""Sample installed release baselines by creation time across version schemes."""
import argparse
import json
import re
import subprocess
from pathlib import Path


_STABLE_TAG_RE = re.compile(r"v\d+\.\d+\.\d+(?:\.\d+)?")
_CANARY_TAG_RE = re.compile(r"v\d+\.\d+\.\d+\+canary\.20\d{6}T\d{6}Z")


def _commit_for_tag(repo: Path, tag: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-list", "-1", f"{tag}^{{commit}}"],
        text=True,
        encoding="utf-8",
    ).strip()


def pick_tags(repo: Path, count: int, exclude: str = "") -> list[str]:
    if not 1 <= count <= 10:
        raise ValueError("Tag count must be between one and ten")
    raw = subprocess.check_output(["git", "-C", str(repo), "for-each-ref", "--sort=creatordate",
                                   "--format=%(refname:short)", "refs/tags/v*"], text=True, encoding="utf-8")
    try:
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       text=True, encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        head = ""
    tags = []
    for tag in raw.splitlines():
        if not (_STABLE_TAG_RE.fullmatch(tag) or _CANARY_TAG_RE.fullmatch(tag)):
            continue
        if tag == exclude:
            continue
        try:
            commit = _commit_for_tag(repo, tag)
        except subprocess.CalledProcessError:
            continue
        # A tag pointing at the tested HEAD would make the update leg a no-op.
        if head and commit == head:
            continue
        tags.append(tag)
    if not tags:
        return []
    if len(tags) <= count:
        return tags
    if count == 1:
        return [tags[-1]]
    return [tags[(slot * (len(tags) - 1) * 2 + count - 1) // ((count - 1) * 2)] for slot in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--exclude-ref", default="")
    args = parser.parse_args()
    print(json.dumps(pick_tags(args.repo, args.count, args.exclude_ref)))


if __name__ == "__main__":
    main()
