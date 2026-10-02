"""Validate staged Docker artifact identities and publish receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time

MANIFEST_SCHEMA = 1
SHA256 = re.compile(r"[a-f0-9]{64}")
GIT_SHA = re.compile(r"[a-f0-9]{40}")
from hermes_cli.update_channel import STABLE_TAG_RE
from scripts.releases.versioning import parse_attempt_ref
ARCHES = ("amd64", "arm64")
IMAGE = "nousresearch/hermes-agent"

class DockerReleaseError(ValueError):
    """Raised when a phase/manifest violates the staged-release contract."""


def require_stable_tag(tag: str) -> str:
    # The versioned image is tagged by the attempt ref; stable/latest move only
    # at publish. The old v-suffix shape is dead.
    if not isinstance(tag, str) or not (STABLE_TAG_RE.fullmatch(tag) or parse_attempt_ref(tag)):
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


def output(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True, encoding="utf-8").strip().strip('"')


def _inspect(reference: str, run) -> str:
    return run([
        "docker", "buildx", "imagetools", "inspect", reference,
        "--format", "{{json .Manifest.Digest}}",
    ]).strip('"')


def _version_digest(tag: str, suffix: str, run) -> str:
    reference = f"{IMAGE}:{tag}{suffix}"
    try:
        digest = _inspect(reference, run)
    except subprocess.CalledProcessError as exc:
        raise DockerReleaseError(f"Docker versioned tag {reference} is missing") from exc
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise DockerReleaseError(f"Published image manifest digest is invalid: {reference}")
    return digest


def promote_stable(tag: str, digest: str, *, run=output, sleep=time.sleep) -> None:
    """Move each variant's aliases from its immutable versioned registry image."""
    require_stable_tag(tag)
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise DockerReleaseError("Invalid published manifest-list digest")
    if _version_digest(tag, "", run) != digest:
        raise DockerReleaseError("Docker versioned tag differs from the final release receipt")
    # Inspect both before moving either alias; desktop has its own registry digest.
    variants = (("", digest), ("-desktop", _version_digest(tag, "-desktop", run)))
    for suffix, version_digest in variants:
        command = [
            "docker", "buildx", "imagetools", "create", "-t", f"{IMAGE}:stable{suffix}",
            "-t", f"{IMAGE}:latest{suffix}", f"{IMAGE}@{version_digest}",
        ]
        for attempt in range(3):
            try:
                run(command)
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                sleep(20)
        for alias in ("stable", "latest"):
            for attempt in range(3):
                if _inspect(f"{IMAGE}:{alias}{suffix}", run) == version_digest:
                    break
                if attempt < 2:
                    sleep(20)
            else:
                raise DockerReleaseError(f"Docker {alias}{suffix} alias read-back mismatch")


def stable_alias_digest(run=output) -> str | None:
    """The slim ``stable`` alias digest, or None when the alias does not exist yet."""
    try:
        digest = _inspect(f"{IMAGE}:stable", run)
    except subprocess.CalledProcessError:
        return None
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise DockerReleaseError(f"Docker stable alias digest is invalid: {digest!r}")
    return digest


def published_digest(tag: str, run=output) -> str:
    """Read both attempt images; return the slim digest bound to the release receipt.

    Desktop has a separate digest. Its immutable tag must be published before
    the release can finalize; promotion reads it again when moving aliases.
    """
    require_stable_tag(tag)
    digest = _version_digest(tag, "", run)
    _version_digest(tag, "-desktop", run)
    return digest


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
