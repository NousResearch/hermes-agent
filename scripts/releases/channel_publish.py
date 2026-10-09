"""Admit channel requests and publish only complete, native-smoked build manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from hermes_cli.release_channels import ChannelReader, build_prefix, canonical_json, decode_json, require_sha256, validate_request
from scripts.releases import commit_build, handoff, r2

NATIVE_LEGS = ("darwin-arm64", "darwin-x64", "win32-arm64", "win32-x64", "windows-universal")
REQUIRED_JOBS = ("validate", "build-darwin-arm64", "build-darwin-x64",
                 "build-win32-arm64", "build-win32-x64", "assemble-win32-bundle",
                 "smoke-darwin-arm64", "smoke-darwin-x64",
                 "smoke-win32-arm64", "smoke-win32-x64")


def read_request(build_id: str, digest: str, public_base: str, repository: str) -> dict:
    require_sha256(digest)
    reader = ChannelReader(public_base, repository=repository)
    body = reader.read_bytes(build_prefix(build_id) + "request.json", digest)
    request = validate_request(decode_json(body), repository=repository, base_url=public_base)
    if request["buildId"] != build_id:
        raise ValueError("Channel request build ID differs from its key")
    return request


def require_success(needs: dict) -> None:
    """Skipped, absent and empty native matrices cannot qualify publication."""
    if not isinstance(needs, dict):
        raise ValueError("Missing workflow results")
    failed = [name for name in REQUIRED_JOBS
              if not isinstance(needs.get(name), dict) or needs[name].get("result") != "success"]
    if failed:
        raise ValueError("Channel publication requires successful jobs: " + ", ".join(failed))


def receiver_request(request: dict) -> bool:
    """An explicit marker is never sufficient without the physical CI scope."""
    if not request.get("receiverCandidate"):
        return False
    from scripts.releases.channel_releases import product_identity
    from scripts.releases.r2_scope import R2Scope, channel_public_base
    scope = R2Scope.configured()
    if (request["receiverCandidate"] is not True or not scope.prefix
            or request["publicBase"] != channel_public_base()
            or request["identity"] != product_identity(request.get("releaseTag", ""))
            or request.get("releaseTag") != "v" + request["version"]
            or request.get("channel") != "stable" or request.get("bundleEnv") != {}):
        raise ValueError("Receiver candidate escaped disposable official identity admission")
    validate_request(request, policy="stable-release")
    return True


def admit(request: dict, env: dict[str, str]) -> dict[str, str]:
    from scripts.releases.job_groups import selects_all

    if (env.get("TAG") or env.get("RELEASE_PHASE")
            or env.get("TERMUX_UPGRADE_FROM_TAG")
            or env.get("UPLOAD_RELEASE", "false") != "false"
            or not selects_all(env.get("JOBS"))):
        raise ValueError("Channel requests cannot select one-off, release or partial job overrides")
    # A one-dispatch disposable run inherits the allocation dispatch's own
    # BUILD_COMMIT/BUNDLE_ENV_JSON inputs. They are not overrides when they are
    # exactly what the digest-pinned request already says; anything else is.
    if env.get("BUILD_COMMIT") not in ("", None, request["commit"]):
        raise ValueError("Channel request commit differs from the dispatch that allocated it")
    if env.get("BUNDLE_ENV_JSON", "") not in ("", "{}"):
        if json.loads(env["BUNDLE_ENV_JSON"]) != request["bundleEnv"]:
            raise ValueError("Channel request bundle env differs from the dispatch that allocated it")
    validate_request(request, repository=env.get("GITHUB_REPOSITORY"))
    admitted = commit_build.admit({**env, "BUILD_COMMIT": request["commit"],
                                   "BUNDLE_ENV_JSON": json.dumps(request["bundleEnv"])})
    if admitted["payload-version"] != request["sourceVersion"]:
        raise ValueError("Channel request source version differs from admitted commit")
    if request.get("controllerCommit", env.get("GITHUB_SHA")) != env.get("GITHUB_SHA"):
        raise ValueError("Channel request controller differs from trusted workflow revision")
    return {"sha": request["commit"], "channel": request["channel"], "payload-version": request["version"],
            "receiver-candidate": str(receiver_request(request)).lower()}


def _metadata(root: Path, request: dict, platform: str, arch: str) -> dict:
    native_platform = {"darwin": "macos", "win32": "windows"}[platform]
    row = json.loads((root / f"metadata-{native_platform}-{arch}.json").read_text(encoding="utf-8-sig"))
    identity_key, version_key = (("appId", "version") if platform == "darwin"
                                 else ("msixAppIdWithOrg", "windowsVersion"))
    if (row.get("request") != request or row.get("commit") != request["commit"]
            or row.get("arch") != arch or row.get("platform") != native_platform
            or row.get("identity") != request["identity"][identity_key]
            or row.get("version") != request[version_key]):
        raise ValueError("Native metadata differs from the admitted channel request")
    if platform == "win32" and row.get("applicationId") != request["identity"]["appNamePascal"]:
        raise ValueError("Windows application ID differs from channel identity")
    return row


def assemble(request: dict, root: Path, *, needs: dict) -> tuple[dict, list[Path]]:
    """Consume downloaded receipts; derive immutable feeds without another build."""
    from scripts.bundles.channel_artifacts import assemble as assemble_native

    require_success(needs)
    validate_request(request)
    prefix = build_prefix(request["buildId"])
    files: dict[str, dict] = {}
    for name in NATIVE_LEGS:
        receipt = json.loads((root / handoff.receipt_name(name)).read_text(encoding="utf-8-sig"))
        listed = handoff.validate_channel_receipt(receipt, request, name)
        product = request["identity"]["artifactNamePascal"]
        if name == "windows-universal":
            expected = {f"{product}-{request['windowsVersion']}-win.msixbundle"}
        else:
            platform, arch = name.split("-")
            native = "macos" if platform == "darwin" else "windows"
            expected = {f"metadata-{native}-{arch}.json"}
            if platform == "darwin":
                expected.update(f"{product}-{request['version']}-mac-{arch}.{suffix}"
                                for suffix in ("zip", "dmg", "zip.blockmap", "dmg.blockmap"))
            else:
                expected.add(f"{product}-{request['version']}-win-{arch}.msix")
        if not expected.issubset({row["path"] for row in listed}):
            raise ValueError(f"Channel handoff {name} omits required native files")
        for row in listed:
            if row["path"] in files and files[row["path"]] != row:
                raise ValueError("Channel handoffs disagree on artifact receipts")
            file = root / row["path"]
            if file.stat().st_size != row["size"] or r2.file_sha256(file) != row["sha256"]:
                raise ValueError("Channel artifact differs from its receipt")
            files[row["path"]] = row
    rows = [_metadata(root, request, platform, arch)
            for platform in ("darwin", "win32") for arch in ("arm64", "x64")]
    return assemble_native(request, {"files": files, "packages": rows}, root, artifact_prefix=prefix)


def publish_receiver(request: dict, root: Path, *, needs: dict, publisher) -> dict:
    from scripts.releases.channel_disposable import require_receiver_scope
    from hermes_cli.release_channels import channel_key, validate_manifest

    require_receiver_scope(publisher)
    if not receiver_request(request) or publisher.request(request["buildId"]) != request:
        raise ValueError("Receiver request admission mismatch")
    current = publisher._read(request["channel"])
    if current is None or current[0].get("testOnly") is not True or current[0]["state"] != "active":
        raise ValueError("Receiver requires a test-only active stable record")
    handoff.fetch_channel_build(request, list(NATIVE_LEGS), root, public_base=request["publicBase"])
    manifest, feeds = assemble(request, root, needs=needs)
    prefix = build_prefix(request["buildId"])
    validate_manifest(manifest, {**current[0], "head": None}, publisher.public_base)
    if manifest.get("receiverProtocol") != 1:
        raise ValueError("Candidate packages have no retirement receiver")
    for file in feeds:
        key = prefix + file.relative_to(root).as_posix()
        r2.put(tag="", key=key, file=str(file), key_is_full=True, immutable=True)
        if publisher.reader.read_bytes(key) != file.read_bytes():
            raise ValueError("Receiver feed readback mismatch")
    publisher._write(prefix + "build.json", manifest)
    head = {"buildId": request["buildId"], "sequence": request["sequence"], "manifestKey": prefix + "build.json",
            "sha256": hashlib.sha256(canonical_json(manifest)).hexdigest()}
    # S becomes the initial disposable destination. T is immutable but deliberately
    # unpromoted until the actual handoff chat asks the lifecycle controller to advance.
    if request["sequence"] == 1 and current[0]["head"] is None:
        publisher._write(channel_key(request["channel"]), {**current[0], "head": head,
                         "revision": current[0]["revision"] + 1}, current[1])
    return {"head": head, "manifest": manifest}


def publish(request: dict, root: Path, *, needs: dict, publisher) -> dict:
    require_success(needs)
    # The protocol publisher rechecks policy/identity/retirement at CAS time.
    if publisher.request(request["buildId"]) != request:
        raise ValueError("Publisher request differs from pinned admission")
    current = publisher._read(request["channel"])
    if current is None:
        raise ValueError("Publication channel not found")
    publisher._preview("promote", current[0])
    handoff.fetch_channel_build(request, list(NATIVE_LEGS), root, public_base=request["publicBase"])
    manifest, feeds = assemble(request, root, needs=needs)
    prefix = build_prefix(request["buildId"])
    for file in feeds:
        key = prefix + file.relative_to(root).as_posix()
        r2.put(tag="", key=key, file=str(file), key_is_full=True, immutable=True)
        r2.download_public_object(request["publicBase"], key, root / "verified" / file.name,
                                  expected_size=file.stat().st_size, expected_sha256=r2.file_sha256(file))
    publisher.store.put(prefix + "build.json", canonical_json(manifest))
    return publisher.promote(request["buildId"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["request", "admit", "publish", "receiver-publish"])
    parser.add_argument("--build-id", default=os.environ.get("CHANNEL_BUILD", ""))
    parser.add_argument("--request-sha256", default=os.environ.get("CHANNEL_REQUEST_SHA256", ""))
    parser.add_argument("--public-base", help="Expected public authority (not an override in disposable CI)")
    parser.add_argument("--disposable-run", help="Controller run ID; propagated to R2 object scope")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    from scripts.releases.r2_scope import R2Scope, channel_public_base, require_run
    if args.disposable_run is not None:
        os.environ["R2_DISPOSABLE_RUN"] = require_run(args.disposable_run)
    args.public_base = channel_public_base(args.public_base)
    request = read_request(args.build_id, args.request_sha256, args.public_base, args.repository)
    if args.command == "admit":
        values = admit(request, dict(os.environ))
        values["public-root"] = os.environ["CLOUDFLARE_R2_PUBLIC_URL"]
        values["public-base"] = args.public_base
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write("".join(f"{key}={value}\n" for key, value in values.items()))
    elif args.command in {"publish", "receiver-publish"}:
        from scripts.releases.channels import ChannelPublisher, R2ChannelStore
        needs = json.loads(os.environ.get("RELEASE_NEEDS", "{}"))
        require_success(needs)
        # Repeat maintainer/default-controller admission on promotion, including reruns.
        admit(request, dict(os.environ))
        def authorize(action: str, record: dict) -> None:
            if action != "promote" or record["policy"] != "preview":
                raise ValueError("This workflow only promotes preview channel builds")
            admit(request, dict(os.environ))

        def qualified(pinned: dict, manifest: dict) -> bool:
            # Re-evaluate exact downloaded receipt bytes on every CAS attempt.
            expected, _feeds = assemble(pinned, root, needs=needs)
            return pinned == request and manifest == expected

        publisher = ChannelPublisher(R2ChannelStore(*r2.credentials()), args.repository, args.public_base,
                                     authorize=authorize, verify_build=qualified)
        if args.command.startswith("receiver-"):
            if args.root is None:
                parser.error("receiver publication requires --root")
            result = publish_receiver(request, args.root, needs=needs, publisher=publisher)
        elif args.root:
            root = args.root
            result = publish(request, args.root, needs=needs, publisher=publisher)
        else:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = publish(request, root, needs=needs, publisher=publisher)
        if args.command == "publish" and not R2Scope.configured().prefix:
            commit_build.publish_receipt(
                "channel", dict(os.environ), version=request["version"],
                commit=request["commit"], details={
                    "buildId": request["buildId"],
                    "channel": request["channel"],
                    "requestSha256": args.request_sha256,
                },
            )
        print(json.dumps(result, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
