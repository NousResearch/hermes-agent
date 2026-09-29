"""Advance protected R2 heads from the release transaction's accepted native bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone

from hermes_cli.release_channels import (
    ChannelError, build_prefix, canonical_json, validate_identity,
)
from scripts.bundles.channel_artifacts import assemble
from scripts.releases import handoff, r2, stable
from scripts.releases.channels import ChannelPublisher, R2ChannelStore
from scripts.releases.versioning import parse_attempt_ref

NATIVE_LEGS = ("darwin-arm64", "darwin-x64", "win32-arm64", "win32-x64", "windows-universal")
CANARY_NEEDS = ("validate", "build-win32-x64", "build-win32-arm64",
                "build-darwin-arm64", "build-darwin-x64",
                "build-linux-x64", "build-linux-arm64", "builds-table",
                "assemble-win32-bundle", "smoke-darwin-arm64", "smoke-darwin-x64",
                "smoke-win32-arm64", "smoke-win32-x64",
                "publish-win32-updater", "publish-darwin-updater")


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


def canary_windows_version(tag: str) -> str:
    from hermes_cli.update_channel import canary_timestamp

    stamp = canary_timestamp(tag)
    if stamp is None:
        raise ChannelError("Invalid canary release identity")
    instant = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return f"{instant.year % 100}.{int(f'{instant.month:02d}{instant.day:02d}')}.{instant.hour}.{int(f'{instant.minute:02d}{instant.second:02d}')}"


def admit_transaction(policy: str, env: dict, *, require_published: bool = False,
                      run=stable.output) -> tuple[str, str]:
    """A callable CLI is not permission to bypass the existing workflow gate."""
    from scripts.releases.semver import is_release_version
    from hermes_cli.release_channels import require_commit, validate_repository
    from hermes_cli.update_channel import is_canary_tag

    repository = validate_repository(env.get("GITHUB_REPOSITORY"))
    tag = env.get("RELEASE_TAG", "")
    if policy == "stable-release":
        # Stable runs on its attempt ref; the final vX.Y.Z tag is a publish receipt.
        if parse_attempt_ref(tag) is None:
            raise ChannelError("Invalid protected release tag")
    else:
        if not tag.startswith("v") or not is_release_version(tag[1:]):
            raise ChannelError("Invalid protected release tag")
        if (policy == "canary-release") != is_canary_tag(tag):
            raise ChannelError("Protected release policy/tag mismatch")
    if env.get("GITHUB_ACTIONS") != "true":
        raise ChannelError("Protected heads require the accepted release workflow")
    default = ""
    if policy == "stable-release":
        default = run(["gh", "api", f"repos/{repository}", "--jq", ".default_branch"])
        claim_tag = env.get("RELEASE_CLAIM_TAG", "")
        expected = {
            f"{repository}/.github/workflows/stable-release.yml@refs/tags/{claim_tag}",
            f"{repository}/.github/workflows/stable-release-publication.yml@refs/heads/{default}",
        }
        if (env.get("GITHUB_EVENT_NAME") not in {"workflow_dispatch", "workflow_run", "schedule"}
                or env.get("GITHUB_WORKFLOW_REF") not in expected):
            raise ChannelError("Protected publication requires its stable release controller")
        try:
            # final_context binds the final vX.Y.Z receipt to the claim; the
            # release tag env carries the attempt ref, so patch the payload tag.
            _verified_tag, verified_commit, _claim = stable.final_context(
                {**env, "RELEASE_TAG": f"v{parse_attempt_ref(tag)[0]}"}, run=run)
        except (ValueError, subprocess.CalledProcessError) as error:
            raise ChannelError(str(error)) from error
        return tag, verified_commit
    elif policy == "canary-release":
        if env.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
            raise ChannelError("Protected heads require the accepted release workflow")
        default = run(["gh", "api", f"repos/{repository}", "--jq", ".default_branch"])
        expected = f"{repository}/.github/workflows/desktop-bundled-release.yml@refs/heads/{default}"
        required = CANARY_NEEDS
    else:
        raise ChannelError("Invalid protected release policy")
    if env.get("GITHUB_WORKFLOW_REF") != expected:
        raise ChannelError("Protected publication requires its existing release workflow")
    stable.require_success(json.loads(env.get("RELEASE_NEEDS", "{}")), list(required))
    commit = require_commit(env.get("RELEASE_COMMIT"))
    expected_object = require_commit(env.get("RELEASE_TAG_OBJECT"))
    try:
        local_object = run(["git", "rev-parse", f"refs/tags/{tag}^{{tag}}"])
    except subprocess.CalledProcessError as error:
        raise ChannelError("Canary release tag must be annotated") from error
    actual = run(["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"])
    remote = dict(line.split()[::-1] for line in run(
        ["git", "ls-remote", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"] ).splitlines())
    if (local_object != expected_object or actual != commit
            or remote.get(f"refs/tags/{tag}") != expected_object
            or remote.get(f"refs/tags/{tag}^{{}}") != commit):
        raise ChannelError("Canary release tag moved")
    run(["git", "merge-base", "--is-ancestor", commit, f"origin/{default}"])
    release = json.loads(run(["gh", "release", "view", tag, "--repo", repository,
                              "--json", "tagName,isDraft,isPrerelease"]))
    if (release.get("tagName") != tag or not isinstance(release.get("isDraft"), bool)
            or release.get("isPrerelease") is not True):
        raise ChannelError("Protected head requires its exact GitHub prerelease transaction")
    if require_published and release["isDraft"]:
        raise ChannelError("Protected head requires the published GitHub release transaction")
    return tag, commit


def accepted_stable(publisher: ChannelPublisher, env: dict, tag: str, commit: str,
                    release_epoch: int, *, skip_tests: bool) -> dict:
    """Read the accepted candidate from the attempt-scoped release archive."""
    from hermes_cli.release_channels import decode_json, require_sha256

    parsed = parse_attempt_ref(tag)
    payload_tag = f"v{parsed[0]}" if parsed else tag
    digest = require_sha256(env.get("CANDIDATE_MANIFEST_SHA256"))
    key = f"releases/tag/{tag}/release-candidates.json"
    if env.get("CANDIDATE_MANIFEST_URL") != publisher.public_base + "/" + key:
        raise ChannelError("Accepted candidate URL differs from release archive")
    candidate = decode_json(publisher.reader.read_bytes(key, digest))
    stable.validate_candidates(candidate, payload_tag, commit, publisher.public_base, release_epoch,
                               archive=tag)
    stable.require_smokes_match_claim(candidate, skip_tests=skip_tests)
    return candidate


def stable_head_version(env: dict) -> str | None:
    """Return the protected stable head's source version, if one exists."""
    publisher = ChannelPublisher(R2ChannelStore(*r2.credentials()), env["GITHUB_REPOSITORY"],
                                 r2.public_base_url(), authorize=lambda _action, _record: None)
    current = publisher._read(select_channel(publisher, "stable-release"))
    if current is None or current[0]["head"] is None:
        return None
    head = current[0]["head"]
    found = publisher.store.get(head["manifestKey"])
    if found is None or hashlib.sha256(found[0]).hexdigest() != head["sha256"]:
        raise ChannelError("Stable protected head manifest is unavailable or changed")
    manifest = json.loads(found[0])
    return manifest["request"]["version"]


def read_archive_bytes(key: str) -> bytes:
    """Read one immutable object from the release archive."""
    store = R2ChannelStore(*r2.credentials())
    found = store.get(key)
    if found is None:
        raise ChannelError(f"Release archive object is unavailable: {key}")
    return found[0]


def advance_stable(env: dict, release: dict, root: Path) -> dict:
    """Advance one published release from its immutable tag-scoped receipts."""
    creds, base, bucket = r2.credentials()
    store = R2ChannelStore(creds, base, bucket)
    public_base = r2.public_base_url()
    key = f"releases/tag/{release['claim_tag']}/release-candidates.json"
    digest = hashlib.sha256(read_archive_bytes(key)).hexdigest()
    if digest != release.get("candidate_manifest_sha256"):
        raise ChannelError("Stable candidate manifest differs from the final release receipt")
    scoped_env = {
        **env,
        "RELEASE_TAG": release["claim_tag"],
        "RELEASE_COMMIT": release["commit"],
        "RELEASE_CLAIM_TAG": release["claim_tag"],
        "RELEASE_CLAIM_OBJECT": release["claim_object"],
        "CANDIDATE_MANIFEST_URL": f"{public_base}/{key}",
        "CANDIDATE_MANIFEST_SHA256": digest,
    }
    return publish_release("stable-release", scoped_env, root)


def promote_stable_feeds(candidate: dict, root: Path, public_base: str) -> None:
    from scripts.bundles.release_artifacts import promote

    promote(candidate, root, public_base)


def verify_bootstrap(request: dict, manifest: dict, base: str, repository: str) -> bool:
    """Bootstrap from published release outputs, never caller attestations."""
    from hermes_cli.release_channels import ChannelReader, decode_json
    tag = request.get("releaseTag", "")
    if not tag or manifest.get("request") != request:
        raise ChannelError("Bootstrap requires published release metadata")
    from hermes_cli.update_channel import is_canary_tag
    canary = is_canary_tag(tag)
    reader = ChannelReader(base, repository)
    payload_tag = tag
    if canary:
        verify_canary_outputs(request, manifest, reader)
    else:
        # The archive prefix is the attempt path the manifest itself names; a
        # releaseTag that is already the attempt ref must agree with it.
        keys = [package["artifact"]["key"] for package in manifest["packages"]]
        archive = os.path.commonprefix(keys)
        parsed = parse_attempt_ref(tag)
        payload_tag = f"v{parsed[0]}" if parsed else tag
        if parsed is not None and archive != f"releases/tag/{tag}/":
            raise ChannelError("Bootstrap archive differs from its attempt ref")
        if not archive.startswith("releases/tag/") or not archive.endswith("/"):
            raise ChannelError("Bootstrap archive key is invalid")
        candidate = decode_json(reader.read_bytes(archive + "release-candidates.json"))
        # The manifest names its own archive ref; the URL prefix it was read
        # from must be the one it claims.
        stable.validate_candidates(candidate, candidate["tag"], request["commit"], base,
                                   archive=archive[len("releases/tag/"):-1])
        if decode_json(reader.read_bytes("releases/stable/release-candidates.json")) != candidate:
            raise ChannelError("Bootstrap must use the current accepted stable transaction")
        match_accepted_packages(manifest, candidate)
    identity = product_identity(payload_tag)
    if any(request["identity"][key] != value for key, value in identity.items() if key != "token"):
        raise ChannelError("Bootstrap must retain official native identity")
    # The GitHub release is resolved by its final payload tag, never the draft.
    release = json.loads(stable.output(["gh", "api", f"repos/{repository}/releases/tags/{payload_tag}"]))
    commit = stable.output(["gh", "api", f"repos/{repository}/commits/{payload_tag}", "--jq", ".sha"])
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
    tag, commit = admit_transaction(policy, env)
    # Stable's tag is the attempt ref: native receipts, product identity and the
    # channel record bind the plain payload tag, while the archive is attempt-scoped.
    parsed = parse_attempt_ref(tag)
    payload_tag = f"v{parsed[0]}" if parsed else tag
    if env.get("R2_DISPOSABLE_RUN"):
        raise ChannelError("Disposable receiver builds cannot enter production release publication")
    publisher = ChannelPublisher(R2ChannelStore(*r2.credentials()), env["GITHUB_REPOSITORY"],
                                 r2.public_base_url(), authorize=lambda action, record:
                                 admit_transaction(policy, env, require_published=True))
    name = select_channel(publisher, policy)
    identity = product_identity(payload_tag)
    current = publisher._read(name)
    if current:
        # Explicit bootstrap may have reserved another opaque token for the same app.
        identity["token"] = current[0]["identity"]["token"]
        if identity != current[0]["identity"]:
            raise ChannelError("Protected R2 identity differs from the existing product")
    final_claim = stable.final_context({**env, "RELEASE_TAG": payload_tag})[2] \
        if policy == "stable-release" else None
    if final_claim is not None and final_claim["skip_bundles"]:
        raise ChannelError("A release that skipped bundles has no native bytes to advance a protected head")
    release_epoch = final_claim["claim_epoch"] if final_claim is not None else None
    accepted = accepted_stable(publisher, env, tag, commit, release_epoch,
                               skip_tests=final_claim["skip_tests"]) \
        if final_claim is not None else None
    # Native handoffs were staged under the attempt ref for stable attempts
    # (identical for canary, where tag == payload_tag).
    handoff.fetch(tag, commit, list(NATIVE_LEGS), root,
                  ["metadata-*.json", "*.zip", "*.dmg", "*.blockmap", "*.msixbundle"], public_base=publisher.public_base)
    native = read_native_receipts(root, tag, commit)
    windows = next(row for row in native["packages"] if row["platform"] == "windows")
    if policy == "canary-release":
        expected_windows = canary_windows_version(tag)
        if (windows["version"] != expected_windows
                or windows.get("executableVersion") != expected_windows):
            raise ChannelError("Canary Windows version differs from its release timestamp")
        stable.output(["gh", "release", "edit", tag, "--repo", env["GITHUB_REPOSITORY"],
                       "--draft=false"])
        if admit_transaction(policy, env, require_published=True) != (tag, commit):
            raise ChannelError("Canary GitHub release publication did not preserve custody")

    def release_gate(request: dict) -> bool:
        if admit_transaction(policy, env, require_published=True) != (tag, commit):
            return False
        if policy == "stable-release":
            if release_epoch is None:
                raise ChannelError("Stable release epoch is unavailable")
            return accepted_stable(publisher, env, tag, commit, release_epoch,
                                   skip_tests=final_claim["skip_tests"]) == accepted
        return True

    source_version = payload_tag[1:].split("+", 1)[0]
    request = publisher.allocate_protected(name, commit, source_version, release_tag=payload_tag,
                                           version=payload_tag[1:], windows_version=windows["version"],
                                           identity=identity, policy=policy, release_gate=release_gate,
                                           archive_ref=tag if parsed else None)
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
    if accepted is not None:
        publisher._write("releases/stable/release-candidates.json", accepted)
        if publisher.reader.read_bytes("releases/stable/release-candidates.json") != canonical_json(accepted):
            raise ChannelError("Stable candidate pointer read-back differs")
        promote_stable_feeds(accepted, root, publisher.public_base)
    result = publisher.promote_protected(request["buildId"], policy=policy, release_gate=release_gate)
    return result


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
