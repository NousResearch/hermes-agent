#!/usr/bin/env python3
"""Create a transfer README for the Signal Room review handoff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def render_transfer_readme(package_dir: Path, checksum_path: Path) -> str:
    checksums = read_json(checksum_path)
    handoff = read_json(package_dir / "handoff_manifest.json")
    package_index = read_json(package_dir / "review_package_index.json")
    bundle = checksums.get("bundle", {})
    bundle_name = Path(str(bundle.get("path", ""))).name
    primary = handoff.get("primary_review_files", {})
    blockers = package_index.get("blockers", [])
    blocker_lines = "\n".join(f"- {blocker}" for blocker in blockers) if blockers else "- none"
    return f"""# Signal Room Review Handoff Transfer

Status: review-only

## Files To Transfer

- `{bundle_name}`
- `{checksum_path.name}`

## Verify Transfer

Expected SHA-256 for `{bundle_name}`:

```text
{bundle.get("sha256", "")}
```

On Linux/macOS, verify with:

```bash
sha256sum {bundle_name}
```

The checksum sidecar also records hashes for {checksums.get("artifact_count", 0)} package artifacts after extraction.

From the Hermes repo, run the full transfer verifier after extraction:

```bash
python scripts/signal_room_transfer_verify.py {checksum_path.name} --package-dir signal-room-review
```

## Open First

After extracting the tarball, open `{primary.get("review_hub", "REVIEW_HUB.html")}`.

Primary files:
- Watchable draft: `{primary.get("watchable_draft", "")}`
- Editorial scorecard: `{primary.get("editorial_scorecard", "")}`
- Pose export intake: `{primary.get("pose_export_intake", "")}`

## Remaining Decisions

{blocker_lines}

Fill `EDITORIAL_SCORECARD.json` after review, then run the editorial scorecard gate before treating the proof as approved.
"""


def write_transfer_readme(package_dir: Path, checksum_path: Path, out: Path) -> dict[str, Any]:
    out.write_text(render_transfer_readme(package_dir, checksum_path))
    return {"passed": True, "output": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--checksums", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = write_transfer_readme(args.package_dir, args.checksums, args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
