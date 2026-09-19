"""Advance protected R2 heads from the release transaction's accepted native bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from hermes_cli.release_channels import (
    ChannelError, build_prefix, canonical_json, validate_identity,
)
from scripts.bundles.channel_artifacts import assemble
from scripts.releases import handoff, r2, stable
from scripts.releases.channels import ChannelPublisher, R2ChannelStore

NATIVE_LEGS = ("darwin-arm64", "darwin-x64", "win32-arm64", "win32-x64", "windows-universal")
STABLE_NEEDS = ("admit", "ci", "docker", "acceptance", "candidates", "publication", "promote-docker", "promote-bundles", "windows-packaged", "macos-packaged")
CANARY_NEEDS = ("validate", "build-win32", "build-darwin", "build-linux", "builds-table", "assemble-win32-bundle",
                "smoke-darwin", "smoke-win32", "smoke-win32-universal", "publish-win32-updater", "publish-darwin-updater")


def select_channel(publisher: ChannelPublisher, policy: str) -> str:
    """R2 records select the name; historical labels are first-rollout defaults only."""
    matches = [record for record in publisher.list() if record["policy"] == policy]
    if len(matches) > 1:
        raise ChannelError("Ambiguous protected policy in R2; select one authority before publishing")
    if matches:
        if matches[0]["state"] != "active":
            raise ChannelError("Protected channel is retired")
        return matches[0]["name"]
    return {"stable-release": "stable", "canary-release": "canary"}[policy]


def product_identity(tag: str, run=subprocess.check_output) -> dict:
    """Consume the packager's identity, not a Python copy of its naming rules."""
    env = dict(os.environ, HERMES_DESKTOP_VARIANT="bundled", HERMES_PAYLOAD_TAG=tag)
    for key in ("HERMES_BUILD_COMMIT", "_HERMES_CHANNEL_REQUEST_JSON"):
        env.pop(key, None)
    raw = run(["node", "-e", "console.log(JSON.stringify(require('./apps/desktop/product-identity.cjs')))"],
              env=env, text=True, encoding="utf-8", timeout=30)
    identity = json.loads(raw)
    # This token reserves existing native identity; it does not create a new app.
    identity = {key: value for key, value in identity.items() if key not in {"store", "light", "channel"}}
    identity["token"] = hashlib.sha256(canonical_json(identity)).hexdigest()[:16]
    return validate_identity(identity)


def read_native_receipts(root: Path, tag: str, commit: str) -> dict:
    files = {}
    for name in NATIVE_LEGS:
        receipt = json.loads((root / handoff.receipt_name(name)).read_text(encoding="utf-8-sig"))
        for row in handoff.validate_receipt(receipt, tag, commit, name):
            if row["path"] in files and files[row["path"]] != row:
                raise ChannelError("Native release receipts disagree")
            file = root / row["path"]
            # Only native metadata, distributables and blockmaps are downloaded.
            if file.is_file() and (file.stat().st_size != row["size"] or r2.file_sha256(file) != row["sha256"]):
                raise ChannelError("Native artifact differs from its receipt")
            files[row["path"]] = row
    rows = []
    for platform in ("macos", "windows"):
        for arch in ("arm64", "x64"):
            name = f"metadata-{platform}-{arch}.json"
            if name not in files or not (root / name).is_file():
                raise ChannelError(f"Missing receipt-bound native metadata: {name}")
            row = json.loads((root / name).read_text(encoding="utf-8-sig"))
            if any(row.get(k) != v for k, v in {"platform": platform, "arch": arch, "tag": tag, "commit": commit}.items()):
                raise ChannelError("Native metadata release identity mismatch")
            rows.append(row)
    return {"packages": rows, "files": files}


def match_accepted_packages(manifest: dict, accepted: dict) -> None:
    by_target = {(row["platform"], row["arch"]): row for row in accepted["packages"]}
    if {(row["platform"], row["arch"]) for row in manifest["packages"]} != {
            (platform, arch) for platform in ("darwin", "win32") for arch in ("arm64", "x64")}:
        raise ChannelError("Protected manifest must include every native target")
    for package in manifest["packages"]:
        platform = "macos" if package["platform"] == "darwin" else "windows"
        row = by_target.get((platform, package["arch"]), {})
        signing = "teamId" if platform == "macos" else "publisher"
        if (any(row.get(key) != package.get(key) for key in ("identity", "version", signing))
                or row.get("artifact") != {"url": manifest["request"]["publicBase"] + "/" + package["artifact"]["key"],
                                           "sha256": package["artifact"]["sha256"]}):
            raise ChannelError("Protected package differs from the accepted release")
        if platform == "windows" and row.get("applicationId") != manifest["request"]["identity"]["appNamePascal"]:
            raise ChannelError("Protected application ID differs from the accepted release")
    if manifest.get("receiverProtocol") == 1 and any(
            row.get("receiverProtocol") != 1 for row in accepted["packages"] if row["platform"] in ("windows", "macos")):
        raise ChannelError("Accepted packages do not declare retirement receiver support")


def admit_transaction(policy: str, env: dict, *, run=stable.output) -> tuple[str, str]:
    """A callable CLI is not permission to bypass the existing workflow gate."""
    from scripts.releases.semver import is_valid_version
    from hermes_cli.release_channels import require_commit, validate_repository

    repository = validate_repository(env.get("GITHUB_REPOSITORY"))
    tag = env.get("RELEASE_TAG", "")
    if not tag.startswith("v") or not is_valid_version(tag[1:]):
        raise ChannelError("Invalid protected release tag")
    if (policy == "canary-release") != ("-canary." in tag):
        raise ChannelError("Protected release policy/tag mismatch")
    if env.get("GITHUB_ACTIONS") != "true" or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise ChannelError("Protected heads require the accepted release workflow")
    default = ""
    if policy == "stable-release":
        expected = f"{repository}/.github/workflows/stable-release.yml@refs/tags/{tag}"
        required = STABLE_NEEDS
    elif policy == "canary-release":
        default = run(["gh", "api", f"repos/{repository}", "--jq", ".default_branch"])
        expected = f"{repository}/.github/workflows/desktop-bundled-release.yml@refs/heads/{default}"
        required = CANARY_NEEDS
    else:
        raise ChannelError("Invalid protected release policy")
    if env.get("GITHUB_WORKFLOW_REF") != expected:
        raise ChannelError("Protected publication requires its existing release workflow")
    stable.require_success(json.loads(env.get("RELEASE_NEEDS", "{}")), list(required))
    if policy == "stable-release":
        tag, commit = stable.check_tag(env, run=run)
    else:
        commit = require_commit(env.get("RELEASE_COMMIT"))
        actual = run(["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"])
        remote = dict(line.split()[::-1] for line in run(
            ["git", "ls-remote", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"] ).splitlines())
        if actual != commit or remote.get(f"refs/tags/{tag}^{{}}", remote.get(f"refs/tags/{tag}")) != commit:
            raise ChannelError("Canary release tag moved")
        run(["git", "merge-base", "--is-ancestor", commit, f"origin/{default}"])
    release = json.loads(run(["gh", "release", "view", tag, "--repo", repository,
                              "--json", "tagName,isDraft,isPrerelease"]))
    if (release.get("tagName") != tag or release.get("isDraft") is not False
            or release.get("isPrerelease") is not (policy == "canary-release")):
        raise ChannelError("Protected head requires the published GitHub release transaction")
    return tag, commit


def accepted_stable(publisher: ChannelPublisher, env: dict, tag: str, commit: str) -> dict:
    from hermes_cli.release_channels import decode_json, require_sha256

    digest = require_sha256(env.get("CANDIDATE_MANIFEST_SHA256"))
    key = f"releases/tag/{tag}/release-candidates.json"
    if env.get("CANDIDATE_MANIFEST_URL") != publisher.public_base + "/" + key:
        raise ChannelError("Accepted candidate URL differs from release archive")
    candidate = decode_json(publisher.reader.read_bytes(key, digest))
    stable.validate_candidates(candidate, tag, commit, publisher.public_base)
    accepted = publisher.store.get("releases/stable/release-candidates.json")
    if accepted is None or decode_json(accepted[0]) != candidate:
        raise ChannelError("Stable accepted manifest transaction has not completed")
    if decode_json(publisher.reader.read_bytes("releases/stable/release-candidates.json")) != candidate:
        raise ChannelError("Stable accepted manifest is not publicly visible")
    return candidate


def verify_bootstrap(request: dict, manifest: dict, base: str, repository: str) -> bool:
    """Bootstrap from published release outputs, never caller attestations."""
    from hermes_cli.release_channels import ChannelReader, decode_json
    tag = request.get("releaseTag", "")
    if not tag or manifest.get("request") != request:
        raise ChannelError("Bootstrap requires published release metadata")
    canary = "-canary." in tag
    reader = ChannelReader(base, repository)
    if canary:
        verify_canary_outputs(request, manifest, reader)
    else:
        candidate = decode_json(reader.read_bytes(f"releases/tag/{tag}/release-candidates.json"))
        stable.validate_candidates(candidate, tag, request["commit"], base)
        if decode_json(reader.read_bytes("releases/stable/release-candidates.json")) != candidate:
            raise ChannelError("Bootstrap must use the current accepted stable transaction")
        match_accepted_packages(manifest, candidate)
    identity = product_identity(tag)
    if any(request["identity"][key] != value for key, value in identity.items() if key != "token"):
        raise ChannelError("Bootstrap must retain official native identity")
    release = json.loads(stable.output(["gh", "api", f"repos/{repository}/releases/tags/{tag}"]))
    commit = stable.output(["gh", "api", f"repos/{repository}/commits/{tag}", "--jq", ".sha"])
    if (release.get("draft") is not False or release.get("prerelease") is not canary
            or not release.get("published_at") or commit != request["commit"]):
        raise ChannelError("Bootstrap requires the published release")
    return True


def verify_canary_outputs(request: dict, manifest: dict, reader) -> None:
    """Canary has native promoted feeds, not stable's candidate transaction."""
    import xml.etree.ElementTree as ET

    from scripts.releases.darwin import parse_mac_feed
    tag, base = request["releaseTag"], request["publicBase"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        handoff.fetch(tag, request["commit"], list(NATIVE_LEGS), root,
                      ["metadata-*.json", "*.zip", "*.dmg", "*.blockmap", "*.msixbundle"], public_base=base)
        expected, feeds = assemble(request, read_native_receipts(root, tag, request["commit"]), root,
                                   artifact_prefix=f"releases/tag/{tag}/")
        if manifest != expected:
            raise ChannelError("Canary bootstrap differs from receipt-bound native packages")
        live = parse_mac_feed(reader.read_bytes("releases/darwin/canary/canary-mac.yml").decode())
        pinned = parse_mac_feed(feeds[0].read_text(encoding="utf-8-sig"))
        files = [{**entry, "url": base + entry["url"] if entry["url"].startswith("/releases/") else entry["url"]}
                 for entry in live["files"]]
        if live["version"] != request["version"] or sorted(files, key=lambda x: x["url"]) != sorted(pinned["files"], key=lambda x: x["url"]):
            raise ChannelError("Canary macOS feed has not promoted these packages")
        descriptor = ET.fromstring(reader.read_bytes("releases/win32/canary/canary.appinstaller"))
        bundle = descriptor.find("{*}MainBundle")
        native = next(p for p in manifest["packages"] if p["platform"] == "win32")
        if bundle is None or any(bundle.get(k) != v for k, v in {
                "Name": native["identity"], "Publisher": native["publisher"], "Version": native["version"]}.items()):
            raise ChannelError("Canary Windows feed has not promoted these packages")
        uri = bundle.get("Uri", "")
        if not uri.startswith(base + "/releases/win32/canary/"):
            raise ChannelError("Canary Windows feed archive mismatch")
        r2.download_public_object(base, uri.removeprefix(base + "/"), root / "promoted.msixbundle",
                                  expected_size=native["artifact"]["size"], expected_sha256=native["artifact"]["sha256"])


def publish_release(policy: str, env: dict, root: Path) -> dict:
    from scripts.releases.commit_build import version_at

    tag, commit = admit_transaction(policy, env)
    if env.get("R2_DISPOSABLE_RUN"):
        raise ChannelError("Disposable receiver builds cannot enter production release publication")
    publisher = ChannelPublisher(R2ChannelStore(*r2.credentials()), env["GITHUB_REPOSITORY"],
                                 r2.public_base_url(), authorize=lambda action, record: admit_transaction(policy, env))
    name = select_channel(publisher, policy)
    identity = product_identity(tag)
    current = publisher._read(name)
    if current:
        # Explicit bootstrap may have reserved another opaque token for the same app.
        identity["token"] = current[0]["identity"]["token"]
        if identity != current[0]["identity"]:
            raise ChannelError("Protected R2 identity differs from the existing product")
    accepted = accepted_stable(publisher, env, tag, commit) if policy == "stable-release" else None
    handoff.fetch(tag, commit, list(NATIVE_LEGS), root,
                  ["metadata-*.json", "*.zip", "*.dmg", "*.blockmap", "*.msixbundle"], public_base=publisher.public_base)
    native = read_native_receipts(root, tag, commit)
    windows = next(row for row in native["packages"] if row["platform"] == "windows")

    def release_gate(request: dict) -> bool:
        if admit_transaction(policy, env) != (tag, commit):
            return False
        if policy == "stable-release":
            return accepted_stable(publisher, env, tag, commit) == accepted
        return True

    request = publisher.allocate_protected(name, commit, version_at(None, commit), release_tag=tag,
                                           version=tag[1:], windows_version=windows["version"],
                                           identity=identity, policy=policy, release_gate=release_gate)
    manifest, feeds = assemble(request, native, root, artifact_prefix=f"releases/tag/{tag}/")
    if accepted is not None:
        match_accepted_packages(manifest, accepted)
    prefix = build_prefix(request["buildId"])
    for file in feeds:
        key = prefix + file.relative_to(root).as_posix()
        r2.put(tag=tag, key=key, file=str(file), key_is_full=True, immutable=True)
        if publisher.reader.read_bytes(key) != file.read_bytes():
            raise ChannelError("Immutable protected feed public read-back differs")
    publisher._write(prefix + "build.json", manifest)

    def qualified(pinned: dict, actual: dict) -> bool:
        # Rehash local receipt-bound bytes rather than trusting caller-supplied claims.
        expected, _ = assemble(pinned, read_native_receipts(root, tag, commit), root,
                               artifact_prefix=f"releases/tag/{tag}/")
        if accepted is not None:
            match_accepted_packages(expected, accepted)

        return pinned == request and actual == expected

    publisher.verify_build = qualified
    return publisher.promote_protected(request["buildId"], policy=policy, release_gate=release_gate)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("stable", "canary"), help="Protected release policy, not a channel name")

    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    if args.root:
        result = publish_release(args.role + "-release", dict(os.environ), args.root)
    else:
        with tempfile.TemporaryDirectory() as directory:
            result = publish_release(args.role + "-release", dict(os.environ), Path(directory))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
