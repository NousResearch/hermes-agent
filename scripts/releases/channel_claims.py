"""A channel claim's one channel request: allocated at admission, then built and published.

A channel build is a Stable Release run on ``channel.<stamp>-vX.Y.Z`` (release.py --channel NAME
--build-commit REV). Admission allocates its request in the channel CAS (creating the channel with
the claim's branding if it is new); the request's release tag is the claim, so its packages carry
the claim's version and stamped Windows quad and live in the claim's tag archive. A write-once
pointer in that archive, ``releases/tag/<claim>/channel-request.json``, makes the allocation
idempotent across reruns, and every later job binds itself to it (require_admitted_request).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from hermes_cli.release_channels import (ChannelError, ChannelReader, build_prefix, canonical_json, decode_json,
                                         require_sha256)


def pointer_key(tag: str) -> str:
    return f"releases/tag/{tag}/channel-request.json"


def _publisher(env: dict, actions: frozenset[str] = frozenset({"create", "allocate"})):
    from scripts.releases import r2
    from scripts.releases.channels import ChannelPublisher, R2ChannelStore
    from scripts.releases.r2_scope import channel_public_base

    def authorize(action: str, record: dict) -> None:
        # Admission creates a preview channel and allocates in it; publish-channel only promotes.
        # Neither ever touches a protected (stable or canary) channel.
        if action not in actions or record["policy"] != "preview":
            raise ValueError(f"Channel claims may not {action} on a {record['policy']} channel")

    return ChannelPublisher(R2ChannelStore(*r2.credentials()), env["GITHUB_REPOSITORY"],
                            channel_public_base(), authorize=authorize)


def admit_request(admitted: dict, env: dict) -> dict[str, str]:
    """Allocate (or find again) the admitted channel claim's request; the admit job's outputs."""
    publisher = _publisher(env)
    publisher.create(admitted["channel"], admitted["branding"])
    before = publisher._read(admitted["channel"])
    request = publisher.allocate_claim(admitted["channel"], admitted["claim_tag"], admitted["commit"],
                                       admitted["version"].split("+", 1)[0], admitted["bundle_env"],
                                       env.get("GITHUB_SHA"))
    probe_cas(publisher, admitted["channel"], before, request)
    return {"channel-build": request["buildId"],
            "channel-request-sha256": hashlib.sha256(canonical_json(request)).hexdigest()}


def probe_cas(publisher, name: str, before: tuple[dict, str] | None, request: dict) -> None:
    """The CAS/CDN probe, on the real channel: when this admission advanced the record, R2 must
    refuse the old record under its old ETag, and the public origin must already show the
    allocation. A rerun that found its allocation (nothing advanced) has nothing stale to replay."""
    from hermes_cli.release_channels import channel_key
    from scripts.releases.channels import ChannelConflict

    after = publisher._read(name)
    if before is None or after is None:
        raise ChannelError("Channel record vanished around allocation")
    if after[1] == before[1]:
        return
    try:
        publisher.store.put(channel_key(name), canonical_json(before[0]), before[1])
    except ChannelConflict:
        pass
    else:
        raise ChannelError("R2 accepted a stale conditional channel write")
    if publisher.reader.resolve(name).terminal["nextSequence"] <= request["sequence"]:
        raise ChannelError("The public channel record does not show this allocation")


def read_admitted_request(tag: str, public_base: str, repository: str) -> tuple[dict, str]:
    """The request the claim's pointer names, read back from the public origin and validated."""
    reader = ChannelReader(public_base, repository=repository)
    pointer = decode_json(reader.read_bytes(pointer_key(tag)))
    if not isinstance(pointer, dict) or set(pointer) != {"schema", "buildId", "sha256"} or pointer["schema"] != 1:
        raise ValueError("Invalid channel claim request pointer")
    from scripts.releases.channel_publish import read_request

    digest = require_sha256(pointer["sha256"])
    return read_request(pointer["buildId"], digest, public_base, repository), digest


def require_admitted_request(tag: str, claim: dict, env: dict) -> dict:
    """Refuse a leg that would build anything but exactly this claim's admitted request."""
    from scripts.releases.r2_scope import channel_public_base

    request, digest = read_admitted_request(tag, channel_public_base(), env["GITHUB_REPOSITORY"])
    if (env.get("CHANNEL_BUILD"), env.get("CHANNEL_REQUEST_SHA256")) != (request["buildId"], digest):
        raise ValueError("Channel request differs from the one admitted for this claim")
    expected = {"releaseTag": tag, "channel": claim["channel"], "commit": claim["commit"],
                "bundleEnv": claim.get("bundle_env") or {}}
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("Channel request does not match its claim")
    return request


def _write_once(publisher, key: str, body: bytes, put) -> None:
    """Write an immutable object, or accept the identical one an earlier attempt wrote."""
    found = publisher.store.get(key)
    if found is not None:
        if found[0] != body:
            raise ChannelError(f"{key} already holds different bytes")
        return
    put()
    if publisher.reader.read_bytes(key) != body:
        raise ChannelError(f"{key} public read-back differs")


def publish_claim(env: dict, root: Path) -> dict:
    """Advance the claim's channel to its build, from the accepted candidate's receipts.

    The packages already sit in the claim archive; this writes the build's feeds and manifest
    under releases/channel-builds/<id>/ and CASes the preview channel head. Never a protected head.
    """
    from scripts.bundles.channel_artifacts import assemble
    from scripts.releases import channel_releases, handoff, r2, stable

    admitted = stable.check_claim(env)
    if admitted["policy"] != "channel":
        raise ChannelError("publish-channel serves channel claims only")
    tag, commit = admitted["claim_tag"], admitted["commit"]
    request = require_admitted_request(tag, admitted, env)
    publisher = _publisher(env, frozenset({"promote"}))
    accepted = stable.read_admitted_candidate(tag, commit, publisher.public_base,
                                              env.get("CANDIDATE_MANIFEST_SHA256", ""), archive=tag)
    legs = channel_releases.native_legs(accepted)
    # The accepted candidate's receipt sha256s bind every package: only metadata and the per-arch
    # macOS feed legs are downloaded.
    handoff.fetch(tag, commit, list(legs), root, ["metadata-*.json", "*-mac.yml"], public_base=publisher.public_base)
    native = channel_releases.read_native_receipts(root, tag, commit, legs)
    windows = next(row for row in native["packages"] if row["platform"] == "windows")
    if windows["version"] != request["windowsVersion"] or windows.get("executableVersion") != request["windowsVersion"]:
        raise ChannelError("Channel Windows version differs from its claim's timestamp")
    manifest, feeds = assemble(request, native, root, artifact_prefix=f"releases/tag/{tag}/",
                               targets=channel_releases._leg_targets(legs), local_bytes=False)
    channel_releases.match_accepted_packages(manifest, accepted)
    prefix = build_prefix(request["buildId"])
    for file in feeds:
        key = prefix + file.relative_to(root).as_posix()
        _write_once(publisher, key, file.read_bytes(),
                    lambda: r2.put(tag=tag, key=key, file=str(file), key_is_full=True, immutable=True))
    body = canonical_json(manifest)
    _write_once(publisher, prefix + "build.json", body, lambda: publisher._write(prefix + "build.json", manifest))
    publisher.verify_build = lambda pinned, actual: pinned == request and actual == manifest
    return publisher.promote(request["buildId"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish",))
    parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as directory:
        result = publish_claim(dict(os.environ), Path(directory))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
