"""Privileged allocation: probe CAS/CDN and allocate the channel this run builds.

The workflow's allocation step runs here for BOTH the disposable test path
(``disposable_channel`` input, scoped to a ``ci-disposable/`` namespace) and the
production preview path (``channel`` input, unscoped production namespace). The
local ``release.py --channel`` command no longer touches R2: it only dispatches
this workflow, and this privileged step creates the channel and mints the
immutable build request that the build legs consume via job outputs.
"""  # noqa: E501
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from hermes_cli.release_channels import canonical_json, channel_key, validate_name
from scripts.releases import commit_build, r2
from scripts.releases.channels import ChannelConflict, ChannelPublisher, R2ChannelStore
from scripts.releases.job_groups import selects_all
from scripts.releases.r2_scope import R2Scope, channel_public_base, require_run


def probe(publisher: ChannelPublisher, commit: str, version: str, controller: str) -> None:
    """Exercise real create, CAS, stale-writer refusal, list and CDN readback."""
    scope = publisher.store.scope
    if not scope.prefix or publisher.public_base != channel_public_base():
        raise ValueError("CAS/CDN probes require the configured disposable namespace")
    name = "ci-cas-probe"
    publisher.create(name)
    current = publisher._read(name)
    if current is None:
        raise ValueError("Probe channel disappeared")
    old, etag = current
    first = publisher.allocate(name, commit, version, controller_commit=controller)
    second = publisher.allocate(name, commit, version, controller_commit=controller)
    try:
        publisher.store.put(channel_key(name), canonical_json(old), etag)
    except ChannelConflict:
        pass
    else:
        raise ValueError("R2 accepted a stale conditional channel write")
    if (second["sequence"] <= first["sequence"]
            or publisher.request(second["buildId"]) != second
            or name not in {record["name"] for record in publisher.list()}):
        raise ValueError("Disposable channel CAS/list probe failed")
    visible = publisher.reader.resolve(name).terminal
    if visible["nextSequence"] != second["sequence"] + 1 or visible["head"] is not None:
        raise ValueError("Disposable channel public readback differs from allocation")


def require_receiver_scope(publisher) -> None:
    scope = getattr(publisher.store, "scope", None)
    if not isinstance(scope, R2Scope) or not scope.prefix or not publisher.public_base.endswith("/" + scope.prefix.rstrip("/")):
        raise ValueError("Receiver candidates require physically disposable storage")


def allocate_receivers(publisher, commit: str, source_version: str, controller: str) -> dict:
    """Reserve official-identity test packages, not accepted or published releases."""
    from hermes_cli.release_channels import build_prefix, validate_request
    from scripts.releases.channel_releases import product_identity

    require_receiver_scope(publisher)
    if publisher._read("stable") is not None:
        raise ValueError("Receiver allocation requires a fresh disposable stable record")
    identity = product_identity("v0.0.1")
    requests = {}
    for sequence, slot in enumerate((("S", "T")), 1):
        request = {"schema": 1, "channel": "stable", "repository": publisher.repository,
                   "sequence": sequence, "commit": commit, "sourceVersion": source_version,
                   "controllerCommit": controller, "version": f"0.0.{sequence}",
                   "windowsVersion": f"0.0.{sequence}.0", "releaseTag": f"v0.0.{sequence}",
                   "receiverCandidate": True, "identity": identity, "bundleEnv": {},
                   "publicBase": publisher.public_base}
        request["buildId"] = hashlib.sha256(canonical_json(request)).hexdigest()[:32]
        validate_request(request, policy="stable-release")
        requests[slot] = request
    record = {"schema": 1, "name": "stable", "repository": publisher.repository, "policy": "stable-release",
              "testOnly": True, "state": "active", "revision": 1, "nextSequence": 3, "head": None,
              "identity": identity}
    publisher._write(channel_key("stable"), record)
    for request in requests.values():
        publisher._write(build_prefix(request["buildId"]) + "request.json", request)
    return requests


def allocate(env: dict[str, str]) -> dict:
    disposable_name = env.get("DISPOSABLE_CHANNEL", "")
    channel_name = env.get("CHANNEL", "")
    if disposable_name and channel_name:
        raise ValueError("Choose disposable_channel or channel, not both")
    disposable = bool(disposable_name)
    name = validate_name(disposable_name or channel_name)
    if disposable:
        if name in {"ci-cas-probe", "stable", "canary", "main"}:
            raise ValueError("Use a non-protected disposable preview name")
        if env.get("CHANNEL_BUILD") or env.get("CHANNEL_REQUEST_SHA256"):
            raise ValueError("Allocation cannot reuse a build request or select a storage scope")
        if not selects_all(env.get("JOBS")):
            raise ValueError("Disposable allocation cannot select partial job groups")
        # One dispatch now allocates AND builds, so the namespace must survive a
        # "re-run failed jobs": lease by the run id alone (no attempt suffix), and
        # never accept a caller-supplied path, public URL, or allocation namespace.
        run = require_run(env.get("GITHUB_RUN_ID", ""))
    admitted = commit_build.admit(env)
    controller = commit_build.require_commit(env.get("GITHUB_SHA", ""))
    if disposable:
        os.environ["R2_DISPOSABLE_RUN"] = run
    scope = R2Scope.configured()
    base = channel_public_base()
    repository = env["GITHUB_REPOSITORY"]

    def authorize(action: str, record: dict) -> None:
        if action not in {"create", "allocate"} or record["policy"] != "preview":
            raise ValueError("Controller only creates and allocates previews")
        commit_build.admit(env)

    publisher = ChannelPublisher(R2ChannelStore(*r2.credentials(), scope=scope), repository,
                                 base, authorize=authorize)
    if disposable:
        probe(publisher, admitted["sha"], admitted["payload-version"], controller)
    publisher.create(name)
    from scripts.releases.bundle_env import decode
    request = publisher.allocate(name, admitted["sha"], admitted["payload-version"],
                                  decode(env.get("BUNDLE_ENV_JSON", "")), controller)
    requests = {"A": request}
    if env.get("DISPOSABLE_RECEIVERS") == "true":
        requests["B"] = publisher.allocate(name, admitted["sha"], admitted["payload-version"],
                                           decode(env.get("BUNDLE_ENV_JSON", "")), controller)
        requests.update(allocate_receivers(publisher, admitted["sha"], admitted["payload-version"], controller))
    digest = hashlib.sha256(canonical_json(request)).hexdigest()
    return {"request": request, "requestSha256": digest, "disposableRun": run if disposable else "",
            "storagePrefix": scope.prefix, "requests": requests}


def main() -> None:
    result = allocate(dict(os.environ))
    outputs = {
        "channel_build": result["request"]["buildId"],
        "channel_request_sha256": result["requestSha256"],
        "public_base": channel_public_base(),
    }
    if result["disposableRun"]:
        outputs["disposable_run"] = result["disposableRun"]
        outputs["storage_prefix"] = result["storagePrefix"]
    # Job outputs feed the build legs of THIS run (one dispatch). The summary is
    # informational; the request is consumed through job outputs.
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write("".join(f"{key}={value}\n" for key, value in outputs.items()))
    if result["disposableRun"]:
        title = "## Disposable channel allocation\n\n"
        note = ("CAS, stale-writer rejection, listing and public readback passed. "
                "No native build or production record was published.")
        prefix = f"Storage prefix: `{result['storagePrefix']}`"
    else:
        title = "## Channel allocation\n\n"
        note = "Created the channel and allocated this build's immutable request in the production namespace."
        prefix = f"Channel: `{result['request']['channel']}` · build `{result['request']['buildId']}`"
    summary = (title + note + "\n\n" + prefix + "\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
        stream.write(summary)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()