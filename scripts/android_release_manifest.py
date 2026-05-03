#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def newest_matching(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} under {path}")
    return matches[0]


def select_release_apk(path: Path) -> Path:
    matches = sorted(path.glob("*.apk"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No APK files found under {path}")

    universal_matches = [item for item in matches if "universal" in item.name.lower()]
    if universal_matches:
        return universal_matches[0]

    if len(matches) == 1:
        return matches[0]

    candidates = ", ".join(item.name for item in matches)
    raise ValueError(
        "Multiple release APKs were produced but no universal APK was found. "
        f"Candidates: {candidates}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename Android release artifacts and emit SHA256 manifests")
    parser.add_argument("--tag", required=True, help="Git tag, for example v2026.4.10")
    parser.add_argument("--apk-dir", default="android/app/build/outputs/apk/release")
    parser.add_argument("--aab-dir", default="android/app/build/outputs/bundle/release")
    parser.add_argument("--output-dir", default="dist/android-release")
    args = parser.parse_args()

    apk_src = select_release_apk(Path(args.apk_dir))
    aab_src = newest_matching(Path(args.aab_dir), "*.aab")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apk_dest = output_dir / f"hermes-agent-android-{args.tag}-universal.apk"
    aab_dest = output_dir / f"hermes-agent-android-{args.tag}.aab"
    shutil.copy2(apk_src, apk_dest)
    shutil.copy2(aab_src, aab_dest)

    for artifact in (apk_dest, aab_dest):
        checksum = sha256sum(artifact)
        (artifact.with_suffix(artifact.suffix + ".sha256")).write_text(f"{checksum}  {artifact.name}\n", encoding="utf-8")
        print(artifact)
        print(artifact.with_suffix(artifact.suffix + ".sha256"))


if __name__ == "__main__":
    main()
