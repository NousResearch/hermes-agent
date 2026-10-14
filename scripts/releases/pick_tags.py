"""Sample installed release baselines by creation time across version schemes."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def pick_tags(repo: Path, count: int, exclude: str = "") -> list[str]:
    if not 1 <= count <= 10:
        raise ValueError("Tag count must be between one and ten")
    raw = subprocess.check_output(["git", "-C", str(repo), "for-each-ref", "--sort=creatordate",
                                   "--format=%(refname:short)", "refs/tags/v*"], text=True, encoding="utf-8")
    tags = [tag for tag in raw.splitlines() if re.fullmatch(r"v\d+\.\d+\.\d+(?:\.\d+)?", tag) and tag != exclude]
    if not tags:
        raise ValueError("No released baseline tags remain")
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
