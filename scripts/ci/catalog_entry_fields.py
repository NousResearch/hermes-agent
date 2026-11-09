#!/usr/bin/env python3
"""Read the PR-controlled clone source of one catalog entry and print it as one JSON object.

Prints ``{"REPO": ..., "SHA": ..., "SUBDIR": ..., "BAD": ...}``, every value a string. These fields are
PR-controlled, so the entry is rejected here (non-empty ``BAD``) before anything uses them: the
clone must be remote-only and the subdir a plain relative path, or the pinned sha ends up
validating files outside the pinned commit. Run it with ``python3 -I``: cwd is the PR tree, and
isolated mode keeps the PR checkout's own modules off ``sys.path`` so they cannot shadow this
parser's imports. ruamel.yaml comes from installed packages.

    catalog_entry_fields.py <entry.yaml>
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from ruamel.yaml import YAML


def _load(path: str) -> object:
    try:
        with open(path, encoding="utf-8") as fh:  # windows-footgun: ok — Linux CI, same codec as the structural validator
            return YAML(typ="safe").load(fh)
    except Exception:  # health: allow BLE001 -- every read or parse failure is the "unparseable" verdict
        return None


def _rejection(data: object, repo: str, sha: object, subdir_raw: object, subdir: str) -> str:
    if data is None:
        return "entry yaml is empty or unparseable"
    if not isinstance(data, dict):
        return "entry yaml must be a mapping"
    if not re.fullmatch(r"https://\S+", repo):
        # No whitespace either: the value is echoed into the log, where a newline would smuggle
        # workflow commands like ::error::.
        return "repo must be an https:// git URL"
    if not isinstance(sha, str):
        return "sha must be a string"
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        # A ref name would validate the branch tip, not the pin; a leading '-' would become a
        # checkout option.
        return "sha must be exactly 40 lowercase hex characters"
    if not isinstance(subdir_raw, str):
        # Never coerce: a falsy non-string (0, false) would silently mean the clone root while
        # the structural gate rejects the entry.
        return "subdir must be a string"
    if subdir and (not re.fullmatch(r"[A-Za-z0-9._/-]+", subdir)
                   or any(seg in ("", ".", "..") for seg in subdir.split("/"))):
        return "subdir must be a relative path without '.' or '..' segments"
    return ""


def source_fields(data: object) -> dict[str, str]:
    """REPO, SHA, SUBDIR and the BAD verdict for a parsed entry."""
    mapping = data if isinstance(data, dict) else {}
    repo = str(mapping.get("repo", ""))
    sha = mapping.get("sha", "")
    subdir_raw = mapping.get("subdir", "")
    if subdir_raw is None:
        subdir_raw = ""  # `subdir:` with no value means the repo root
    subdir = subdir_raw if isinstance(subdir_raw, str) else ""
    bad = _rejection(data, repo, sha, subdir_raw, subdir)
    return {"REPO": repo, "SHA": str(sha), "SUBDIR": subdir, "BAD": bad}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a catalog entry's clone source.")
    parser.add_argument("entry")
    args = parser.parse_args(argv)
    print(json.dumps(source_fields(_load(args.entry))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
