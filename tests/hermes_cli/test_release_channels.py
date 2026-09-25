"""Public channel resolution is strict, pinned, and independent of name registries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import importlib.util

import pytest

# The HTTP fixture is shared without making tests/ an importable package.
_spec = importlib.util.spec_from_file_location("channel_http_fixture", Path(__file__).parents[1] / "scripts/test_release_channels.py")
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
object_server, publisher = _fixture.object_server, _fixture.publisher


def seed_manifest(pub, request, objects):
    from hermes_cli.release_channels import build_prefix, canonical_json
    key = build_prefix(request["buildId"]) + "build.json"
    manifest = {"schema": 1, "receiverProtocol": 1, "request": request, "packages": [{"platform": "darwin", "arch": "arm64", "variant": "bundled",
        "artifact": {"key": build_prefix(request["buildId"]) + "darwin/package.zip", "sha256": "f" * 64, "size": 123},
        "version": request["version"], "identity": request["identity"]["appId"], "teamId": "ABCDEFGHIJ",
        "feed": {"key": build_prefix(request["buildId"]) + "darwin/stable-mac.yml", "channel": "stable"}}]}
    body = canonical_json(manifest)
    objects[key] = body
    head = {"buildId": request["buildId"], "sequence": request["sequence"], "manifestKey": key, "sha256": hashlib.sha256(body).hexdigest()}
    record_key = f"releases/channels/{request['channel']}.json"
    record = json.loads(objects[record_key])
    record["head"] = head
    objects[record_key] = canonical_json(record)
    return manifest, record


def test_resolve_verifies_exact_manifest_and_preserves_retirement_constraints():
    from hermes_cli.release_channels import ChannelReader, ChannelError, canonical_json
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url)
        for name in ("old-preview", "next-preview"):
            pub.create(name)
        request = pub.allocate("next-preview", "a" * 40, "1.2.3")
        request.update(version="1.2.3", windowsVersion="1.2.3.0", releaseTag="v1.2.3")
        manifest, record = seed_manifest(pub, request, objects)
        record["policy"] = "stable-release"
        objects["releases/channels/next-preview.json"] = canonical_json(record)
        reader = ChannelReader(url + "/bucket", repository="example/hermes-agent")
        assert reader.resolve("next-preview").manifest == manifest
        old = pub._read("old-preview")[0]
        old.update(state="retired", destination="next-preview", minimumVersion="1.0.0",
                   destinationHead=record["head"], receiverProtocol=1,
                   receiver={"kind": "discontinued"}, lastHead=None)
        objects["releases/channels/old-preview.json"] = canonical_json(old)
        resolved = reader.resolve("old-preview")
        assert resolved.requested == old and resolved.terminal == record
        assert resolved.requested["minimumVersion"] == "1.0.0"
        # Retirement pins the qualified destination, not an unqualified later head.
        newer = dict(request, buildId="f" * 32, sequence=2, sourceVersion="1.3.0", version="1.3.0", windowsVersion="1.3.0.0", releaseTag="v1.3.0")
        record["nextSequence"] = 3
        objects["releases/channels/next-preview.json"] = canonical_json(record)
        seed_manifest(pub, newer, objects)
        after_advance = reader.resolve("old-preview")
        assert after_advance.manifest == manifest
        assert after_advance.terminal["head"]["buildId"] == newer["buildId"]
        objects[record["head"]["manifestKey"]] += b" "
        with pytest.raises(ChannelError, match="SHA256"):
            reader.resolve("old-preview")


@pytest.mark.parametrize("name", ["X", "../x", "two--parts", "a/b", "a%2fb", "a_1", "con", "nul", "x-", "x" * 33, " a", "a\n"])
def test_names_are_validated_without_normalizing(name):
    from hermes_cli.release_channels import ChannelError, validate_name
    with pytest.raises(ChannelError):
        validate_name(name)


def test_reader_rejects_cycles_identity_substitution_and_cross_authority():
    from hermes_cli.release_channels import ChannelReader, ChannelError, canonical_json
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url)
        pub.create("alpha")
        request = pub.allocate("alpha", "a" * 40, "1.2.3")
        manifest, record = seed_manifest(pub, request, objects)
        reader = ChannelReader(url + "/bucket", repository="example/hermes-agent")
        substituted = deepcopy(manifest)
        substituted["request"]["identity"]["appId"] = "ai.other.identity"
        raw = canonical_json(substituted)
        objects[record["head"]["manifestKey"]] = raw
        record["head"]["sha256"] = hashlib.sha256(raw).hexdigest()
        objects["releases/channels/alpha.json"] = canonical_json(record)
        with pytest.raises(ChannelError, match="identity"):
            reader.resolve("alpha")
        record["repository"] = "other/hermes-agent"
        objects["releases/channels/alpha.json"] = canonical_json(record)
        with pytest.raises(ChannelError, match="authority"):
            reader.resolve("alpha")
        record["repository"] = "example/hermes-agent"
        record.update(state="retired", destination="alpha", minimumVersion="1.0.0", lastHead=record["head"],
                      destinationHead=record["head"], receiverProtocol=1,
                      receiver={"kind": "discontinued"})
        objects["releases/channels/alpha.json"] = canonical_json(record)
        with pytest.raises(ChannelError, match="cycle"):
            reader.resolve("alpha")


def test_legacy_bootstrap_uses_real_archive_keys_and_source_main_has_no_bundle():
    from hermes_cli.release_channels import canonical_json, ChannelError
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url, verify_build=lambda request, manifest: True)
        preview = pub.create("temporary")
        request = pub.allocate("temporary", "a" * 40, "2.0.0")
        manifest, _ = seed_manifest(pub, request, objects)
        for name, policy, tag in [("stable", "stable-release", "v2.0.0"), ("canary", "canary-release", "v2.1.0+canary.20260913T000100Z")]:
            legacy = deepcopy(manifest)
            legacy["request"].update(channel=name, buildId=("a" if name == "stable" else "b") * 32,
                                     releaseTag=tag, version=tag[1:], windowsVersion="2.0.0.0" if name == "stable" else "2.1.0.10")
            legacy["packages"][0].update(version=tag[1:])
            legacy["packages"][0]["artifact"]["key"] = f"releases/tag/{tag}/actual.zip"
            legacy["packages"][0]["feed"]["key"] = f"releases/tag/{tag}/stable-mac.yml"
            record = dict(preview, name=name, policy=policy, nextSequence=2,
                          head={"buildId": legacy["request"]["buildId"], "sequence": 1,
                                "manifestKey": "releases/channel-builds/" + legacy["request"]["buildId"] + "/build.json",
                                "sha256": hashlib.sha256(canonical_json(legacy)).hexdigest()})
            pub.bootstrap(record, legacy, publish=True)
            assert pub.reader.resolve(name).manifest == legacy
            assert pub.request(legacy["request"]["buildId"]) == legacy["request"]
            from hermes_cli.release_channels import validate_request
            invalid_versions = ["2.1.0.65536", "65536.1.0.0", "2.1.0.-1", "2.1.0.1.0", "2.1.0.x"]
            if policy == "stable-release":
                invalid_versions.append("2.0.0.10")
            for windows_version in invalid_versions:
                with pytest.raises(ChannelError, match="Windows version"):
                    validate_request(dict(legacy["request"], windowsVersion=windows_version), policy=policy)
            with pytest.raises(ChannelError, match="Protected"):
                pub.promote(legacy["request"]["buildId"])
            with pytest.raises(ChannelError, match="Protected"):
                pub.allocate(name, "a" * 40, "2.0.0")


@pytest.mark.parametrize("sequence", [1, 65535, 65536, 2**32 - 1])
def test_sequence_versions_remain_monotonic_at_native_rollover(sequence):
    from hermes_cli.release_channels import package_versions
    version, windows = package_versions(sequence)
    assert version == f"0.0.{sequence}"
    quad = tuple(map(int, windows.split(".")))
    assert quad[1] * 65536 + quad[2] == sequence and max(quad) <= 65535
    if sequence > 1:
        assert quad > tuple(map(int, package_versions(sequence - 1)[1].split(".")))


@pytest.mark.parametrize("sequence", [0, -1, True, 2**32, "1"])
def test_sequence_exhaustion_never_wraps(sequence):
    from hermes_cli.release_channels import package_versions, ChannelError
    with pytest.raises(ChannelError):
        package_versions(sequence)


def test_malformed_record_and_unqualified_retirement_never_resolve():
    from hermes_cli.release_channels import ChannelError, canonical_json
    with object_server() as (url, objects, headers, requests, faults):
        pub = publisher(url)
        pub.create("preview")
        record = pub._read("preview")[0]
        malformed = dict(record)
        del malformed["head"]
        objects["releases/channels/preview.json"] = canonical_json(malformed)
        with pytest.raises(ChannelError):
            pub.reader.resolve("preview")
        objects["releases/channels/preview.json"] = canonical_json(record)
        pub.create("destination")
        request = pub.allocate("destination", "a" * 40, "1.0.0")
        request.update(version="1.0.0", windowsVersion="1.0.0.0", releaseTag="v1.0.0")
        _, target = seed_manifest(pub, request, objects)
        target["policy"] = "stable-release"
        objects["releases/channels/destination.json"] = canonical_json(target)
        record.update(state="retired", destination="destination", minimumVersion="2.0.0", lastHead=None,
                      destinationHead=target["head"], receiverProtocol=1,
                      receiver={"kind": "discontinued"})
        objects["releases/channels/preview.json"] = canonical_json(record)
        with pytest.raises(ChannelError, match="minimum version"):
            pub.reader.resolve("preview")


@pytest.mark.parametrize("field,value", [("policy", []), ("state", {}), ("revision", True), ("head", []), ("identity", None)])
def test_malformed_wire_types_are_channel_errors(field, value):
    from hermes_cli.release_channels import validate_record, ChannelError
    from scripts.releases.channels import preview_identity
    record = {"schema": 1, "name": "arbitrary", "repository": "example/hermes-agent", "policy": "preview",
              "state": "active", "revision": 1, "nextSequence": 1, "head": None,
              "identity": preview_identity("arbitrary", "a" * 16)}
    record[field] = value
    with pytest.raises(ChannelError):
        validate_record(record)


def stable_request(version="1.2.3", archive_ref=None):
    from scripts.releases.channels import preview_identity
    request = {"schema": 1, "buildId": "a" * 32, "channel": "stable", "sequence": 1,
               "repository": "example/hermes-agent", "commit": "b" * 40,
               "sourceVersion": version, "version": version, "windowsVersion": version + ".0",
               "releaseTag": "v" + version, "identity": preview_identity("stable", "a" * 16),
               "bundleEnv": {}, "publicBase": "https://releases.example"}
    if archive_ref is not None:
        request["archiveRef"] = archive_ref
    return request


def stable_manifest(request, archive_prefix):
    from hermes_cli.release_channels import build_prefix
    manifest = {"schema": 1, "receiverProtocol": 1, "request": request, "packages": [
        {"platform": "darwin", "arch": "arm64", "variant": "bundled",
         "identity": request["identity"]["appId"], "version": request["version"], "teamId": "ABCDEFGHIJ",
         "artifact": {"key": archive_prefix + "Hermes.dmg", "sha256": "d" * 64, "size": 100},
         "feed": {"key": archive_prefix + "stable-mac.yml", "channel": "stable"}}]}
    record = {"schema": 1, "name": request["channel"], "repository": request["repository"],
              "policy": "stable-release", "state": "active", "revision": 1, "nextSequence": 2,
              "identity": request["identity"],
              "head": {"buildId": request["buildId"], "sequence": request["sequence"],
                       "manifestKey": build_prefix(request["buildId"]) + "build.json", "sha256": "a" * 64}}
    return manifest, record


def test_archive_ref_names_the_protected_archive_prefix():
    from hermes_cli.release_channels import validate_manifest, validate_request
    request = stable_request(archive_ref="rc.2-v1.2.3")
    validate_request(request, policy="stable-release")
    manifest, record = stable_manifest(request, "releases/tag/rc.2-v1.2.3/")
    assert validate_manifest(manifest, record, request["publicBase"]) == manifest


def test_stable_manifest_without_archive_ref_fails_closed_outside_the_tag_prefix():
    from hermes_cli.release_channels import validate_manifest, ChannelError
    request = stable_request()
    manifest, record = stable_manifest(request, "releases/tag/rc.2-v1.2.3/")
    with pytest.raises(ChannelError, match="namespace"):
        validate_manifest(manifest, record, request["publicBase"])


def test_archive_ref_equal_to_the_release_tag_keeps_the_tag_prefix():
    from hermes_cli.release_channels import validate_manifest, validate_request
    request = stable_request(archive_ref="v1.2.3")
    validate_request(request, policy="stable-release")
    manifest, record = stable_manifest(request, "releases/tag/v1.2.3/")
    assert validate_manifest(manifest, record, request["publicBase"]) == manifest


def test_archive_ref_must_name_the_release_version():
    from hermes_cli.release_channels import validate_request, ChannelError
    with pytest.raises(ChannelError, match="(?i)archive ref"):
        validate_request(stable_request(archive_ref="rc.2-v1.2.4"), policy="stable-release")


@pytest.mark.parametrize("ref", ["rc.1-v0.21.5", "rc.12-v1.0.0"])
def test_attempt_ref_shapes_are_attempt_refs(ref):
    from scripts.releases.versioning import parse_attempt_ref
    assert parse_attempt_ref(ref) is not None


@pytest.mark.parametrize("ref", ["v0.21.5-rc", "v0.21.5-rc.1", "rc.01-v0.21.5", "rc.0-v0.21.5",
                                 "rc.1-v2026.9.21", "v0.21.5", "abandoned-rc.1-v0.21.5"])
def test_non_attempt_ref_shapes_are_rejected_as_archive_refs(ref):
    from hermes_cli.release_channels import validate_request, ChannelError
    with pytest.raises(ChannelError, match="(?i)archive ref"):
        validate_request(stable_request(archive_ref=ref), policy="stable-release")
