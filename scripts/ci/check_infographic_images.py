#!/usr/bin/env python3
"""Fail when a PR-infographic image is tracked in git.

Run by the ``Reject committed PR-infographic images`` step of
``.github/workflows/repo-guards.yml``. PR infographics are rendered to an image-provider URL and
embedded in the PR *description*; the binary never belongs in git history. This leaked twice
(#48261, #54564) and ``.gitignore`` cannot stop ``git add -f`` or a differently-spelled
directory (#70552), so this check, not an ignore rule, enforces the policy.

It matches on the IMAGE, not on a directory name: keying it to ``infographic/`` is what let
``infograficos/`` through in #70552. It finds tracked raster images whose path contains an
infographic-ish segment, in any spelling, at any depth. ``docs/assets`` and ``website/``
legitimately hold product imagery referenced from shipped docs pages and never match.

Exit codes: 0 = none tracked, 1 = at least one tracked infographic image.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.review_status import action_required, publish

INFOGRAPHIC_SEGMENT = re.compile(r"(^|/)(infograph|infograf)[^/]*/", re.IGNORECASE)
RASTER_IMAGE = re.compile(r"\.(png|jpe?g|webp|gif)$", re.IGNORECASE)

HOW_TO_FIX = """\
Untrack the image and reference the provider URL from the PR body instead:
```
git rm --cached <path-to-image>
```
Then put it in the PR description:
```
## Infographic

![slug](https://<provider-url>)
```
"""

EXPLANATION = """\
PR infographics are rendered to an image-provider URL and
embedded in the PR DESCRIPTION.  The PR body is the archive —
the binary never enters git history.

This rule has been re-established twice already (#48261,
#54564) and leaked both times, because .gitignore cannot stop
'git add -f' or a differently-spelled directory (#70552).

To fix:
  git rm --cached <path>   # keeps your local copy
  # then embed the provider URL in the PR description"""


def tracked_infographic_images() -> list[str]:
    listing = subprocess.run(("git", "ls-files", "-z"), stdout=subprocess.PIPE, check=False,
                             stdin=subprocess.DEVNULL, timeout=120).stdout.decode("utf-8", "surrogateescape")
    return [path for path in listing.split("\0")
            if INFOGRAPHIC_SEGMENT.search(path) and RASTER_IMAGE.search(path)]


def main() -> int:
    offenders = tracked_infographic_images()
    if not offenders:
        print("::notice::No committed PR-infographic images.")
        publish([])
        return 0
    publish(action_required("committed infographics", "PR infographic committed to the repo",
                            "Infographic images belong in the PR description, never in git.",
                            HOW_TO_FIX, ""))
    print()
    print(f"::error::{len(offenders)} PR-infographic image(s) are tracked in git.")
    print()
    for path in offenders:
        print(f"  {path}")
    print()
    print(EXPLANATION)
    return 1


if __name__ == "__main__":
    sys.exit(main())
