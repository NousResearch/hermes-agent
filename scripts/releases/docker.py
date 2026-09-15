"""Validate staged Docker artifact identities and publish receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys

MANIFEST_SCHEMA = 1
SHA256 = re.compile(r"[a-f0-9]{64}")
GIT_SHA = re.compile(r"[a-f0-9]{40}")
from scripts.releases.semver import STABLE_TAG
ARCHES = ("amd64", "arm64")

class DockerReleaseError(ValueError):
    """Raised when a phase/manifest violates the staged-release contract."""


def require_stable_tag(tag: str) -> str:
    if not isinstance(tag, str) or not STABLE_TAG.fullmatch(tag or ""):
        raise DockerReleaseError(f"Not a stable release tag: {tag!r}")
    return tag


def build_manifest(tag: str, commit: str, digests: dict[str, str], archive_sha256: dict[str, str] | None = None) -> dict:
    """Digest manifest emitted by the test phase (artifact ``docker-test-manifest``)."""
    require_stable_tag(tag)
    if not isinstance(commit, str) or not GIT_SHA.fullmatch(commit):
        raise DockerReleaseError(f"Invalid release commit: {commit!r}")
    if sorted(digests) != sorted(ARCHES):
        raise DockerReleaseError(f"Manifest needs per-arch digests for {ARCHES}, got {sorted(digests)}")
    for arch, digest in digests.items():
        if not SHA256.fullmatch(digest):
            raise DockerReleaseError(f"Invalid digest for {arch}: {digest!r}")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tag": tag,
        "commit": commit,
        "digests": {arch: digests[arch] for arch in ARCHES},
    }
    if archive_sha256 is not None:
        if sorted(archive_sha256) != sorted(ARCHES):
            raise DockerReleaseError(f"Manifest needs per-arch archive hashes for {ARCHES}")
        manifest["archives"] = {arch: archive_sha256[arch] for arch in ARCHES}
    return manifest


def parse_manifest(raw: bytes) -> dict:
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DockerReleaseError(f"Manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise DockerReleaseError("Manifest schema mismatch")
    require_stable_tag(manifest.get("tag", ""))
    if not isinstance(manifest.get("commit"), str) or not GIT_SHA.fullmatch(manifest["commit"]):
        raise DockerReleaseError("Manifest commit is not a full git SHA")
    digests = manifest.get("digests")
    if not isinstance(digests, dict) or sorted(digests) != sorted(ARCHES):
        raise DockerReleaseError(f"Manifest needs per-arch digests for {ARCHES}")
    for arch, digest in digests.items():
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise DockerReleaseError(f"Invalid digest for {arch}")
    archives = manifest.get("archives", {})
    if archives and (not isinstance(archives, dict) or sorted(archives) != sorted(ARCHES)):
        raise DockerReleaseError(f"Manifest archive hashes must cover {ARCHES}")
    if "list-digest" in manifest and not re.fullmatch(r"sha256:[a-f0-9]{64}", manifest["list-digest"]):
        raise DockerReleaseError("Invalid published manifest-list digest")
    return manifest


def verify_manifest(manifest: dict, tag: str, commit: str) -> None:
    """Fail the publish/promote phase unless the manifest matches the release identity."""
    if manifest.get("tag") != tag or manifest.get("commit") != commit:
        raise DockerReleaseError(
            f"Tested manifest identity {manifest.get('tag')}@{manifest.get('commit')} "
            f"does not match release {tag}@{commit}"
        )


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("manifest", help="Emit the tested-image digest manifest JSON")
    p_manifest.add_argument("--tag", required=True)
    p_manifest.add_argument("--commit", required=True)
    p_manifest.add_argument("--digest-amd64", required=True)
    p_manifest.add_argument("--digest-arm64", required=True)
    p_manifest.add_argument("--archive-amd64", default="", help="Optional sha256 file of the amd64 image archive")
    p_manifest.add_argument("--archive-arm64", default="")

    p_verify = sub.add_parser("verify", help="Verify a downloaded manifest against the release identity")
    p_verify.add_argument("--tag", required=True)
    p_verify.add_argument("--commit", required=True)
    p_verify.add_argument("manifest", help="Path to the downloaded manifest JSON")

    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            archive_hashes = {}
            for arch, path in (("amd64", args.archive_amd64), ("arm64", args.archive_arm64)):
                if path:
                    archive_hashes[arch] = sha256_file(path)
            manifest = build_manifest(
                args.tag,
                args.commit,
                {"amd64": args.digest_amd64, "arm64": args.digest_arm64},
                archive_hashes or None,
            )
            print(json.dumps(manifest, indent=2))
        else:
            with open(args.manifest, "rb") as handle:
                manifest = parse_manifest(handle.read())
            verify_manifest(manifest, args.tag, args.commit)
            print(json.dumps(manifest))
    except DockerReleaseError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
