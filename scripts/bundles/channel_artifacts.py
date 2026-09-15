"""One native manifest/feed writer for admitted previews and protected releases."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path


from hermes_cli.release_channels import ChannelError, build_prefix, canonical_json, validate_manifest
from scripts.bundles.release_artifacts import single, validate_windows_bundle, write_appinstaller


def assemble(request: dict, native: dict, root: Path, *, artifact_prefix: str) -> tuple[dict, list[Path]]:
    files = native["files"]
    rows = native["packages"]
    if sorted((row["platform"], row["arch"]) for row in rows) != [
            (platform, arch) for platform in ("macos", "windows") for arch in ("arm64", "x64")]:
        raise ChannelError("Native metadata must cover both platforms and architectures")
    prefix = build_prefix(request["buildId"])
    base = request["publicBase"]
    windows = [row for row in rows if row["platform"] == "windows"]
    if windows[0]["applicationId"] != request["identity"]["appNamePascal"]:
        raise ChannelError("Windows application identity mismatch")
    bundle_name = single(name for name in files if name.endswith(".msixbundle") and not name.startswith("Store-"))
    validate_windows_bundle(root / bundle_name, windows)
    packages, mac_files = [], []
    for row in rows:
        mac = row["platform"] == "macos"
        filename = row["filename"] if mac else bundle_name
        receipt = files.get(filename)
        if receipt is None or not (root / filename).is_file():
            raise ChannelError("Missing receipt-bound native artifact")
        signing = "teamId" if mac else "publisher"
        packages.append({"platform": "darwin" if mac else "win32", "arch": row["arch"], "variant": "bundled",
                         "identity": row["identity"], "version": row["version"], signing: row.get(signing),
                         "artifact": {"key": artifact_prefix + filename, "sha256": receipt["sha256"], "size": receipt["size"]},
                         "feed": {"key": prefix + ("darwin/stable-mac.yml" if mac else "win32/stable.appinstaller"),
                                  "channel": "stable"}})
        if mac:
            for name in (filename, filename.removesuffix(".zip") + ".dmg"):
                if any(item not in files or not (root / item).is_file() for item in (name, name + ".blockmap")):
                    raise ChannelError("Missing receipt-bound macOS package or blockmap")
                with (root / name).open("rb") as stream:
                    digest = base64.b64encode(hashlib.file_digest(stream, "sha512").digest()).decode("ascii")
                mac_files.append({"url": f"{base}/{artifact_prefix}{name}", "sha512": digest, "size": files[name]["size"]})
    manifest = {"schema": 1, "request": request, "packages": packages}
    if all(row.get("receiverProtocol") == 1 for row in rows):
        manifest["receiverProtocol"] = 1
    tag = request.get("releaseTag", "")
    record = {"name": request["channel"], "repository": request["repository"], "identity": request["identity"],
              "policy": ("canary-release" if "-canary." in tag else "stable-release") if tag else "preview", "head": None}
    validate_manifest(manifest, record, base)
    descriptor = root / "win32" / "stable.appinstaller"
    write_appinstaller(descriptor, identity=windows[0]["identity"], publisher=windows[0]["publisher"],
                       version=windows[0]["version"], self_uri=f"{base}/{prefix}win32/stable.appinstaller",
                       artifact_uri=f"{base}/{artifact_prefix}{bundle_name}", update_policy="pinned")
    feed = root / "darwin" / "stable-mac.yml"
    feed.parent.mkdir(parents=True, exist_ok=True)
    first = single(row for row in mac_files if row["url"].endswith("-arm64.zip"))
    # A neutral feed name avoids interpreting custom channel names as prereleases.
    feed.write_bytes(canonical_json({"version": request["version"], "files": mac_files,
                                    "path": first["url"], "sha512": first["sha512"]}))
    return manifest, [feed, descriptor]
