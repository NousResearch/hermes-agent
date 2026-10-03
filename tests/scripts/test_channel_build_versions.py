"""A channel request is a complete, immutable input, not a commit flavor."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.bundles.desktop_prepare import BuildRequest
from tests.scripts.test_desktop_preparation import _project


def channel_request(commit: str, sequence: int = 65536) -> dict:
    token = "ab12cd34ef56ab78"
    return {
        "schema": 1, "buildId": "a" * 32, "channel": "unregistered-preview",
        "sequence": sequence, "repository": "fixture/project", "commit": commit,
        "sourceVersion": "1.2.3", "version": f"0.0.{sequence}",
        "windowsVersion": f"0.{sequence // 65536}.{sequence % 65536}.0",
        "identity": {
            "token": token, "displayName": "Hermes unregistered-preview",
            "appId": f"com.nousresearch.hermes-channel-{token}",
            "appNamePascal": f"HermesChannel{token}", "artifactNamePascal": "HermesBundled",
            "cliName": "hermes-unregistered-preview", "windowsExecutableName": "hermes-unregistered-preview",
            "msixAppIdWithOrg": f"NousResearch.HermesChannel{token}",
        },
        "bundleEnv": {"HERMES_GUEST_ONBOARDING": "1", "HERMES_HOME": None},
        "publicBase": "https://builds.example.test",
    }


def test_preparation_preserves_exact_channel_inputs_without_source_version_ordering(tmp_path):
    source, commit = _project(tmp_path)
    admitted = channel_request(commit)
    request = BuildRequest.create(source, tag=None, commit=commit, variant="bundled",
                                  work=tmp_path / "work", cache=tmp_path / "cache",
                                  bundle_env={}, channel_request=admitted)
    assert request.version == admitted["version"] != admitted["sourceVersion"]
    assert request.tag is None
    assert request.bundle_env == admitted["bundleEnv"]
    restored = BuildRequest.from_data(json.loads(json.dumps(request.data())))
    assert restored.channel_request == admitted
    admitted["identity"]["cliName"] = "changed"
    assert restored.channel_request == request.channel_request
    assert request.channel_request["identity"]["cliName"] != "changed"

    for changes in ({"sourceVersion": "0.9.0"}, {"commit": "b" * 40}, {"version": "1.2.3"},
                    {"sequence": 2**32}, {"identity": {**channel_request(commit)["identity"], "cliName": "../bad"}}):
        invalid = {**channel_request(commit), **changes}
        with pytest.raises(ValueError):
            BuildRequest.create(source, tag=None, commit=commit, variant="bundled",
                                work=tmp_path / "work", cache=tmp_path / "cache",
                                bundle_env={}, channel_request=invalid)
    for variant in ("light", "store"):
        with pytest.raises(ValueError, match="bundled"):
            BuildRequest.create(source, tag=None, commit=commit, variant=variant,
                                work=tmp_path / "work", cache=tmp_path / "cache",
                                bundle_env={}, channel_request=channel_request(commit))
    for args in (["-c.extraMetadata.version=9.9.9"], ["--config.extraMetadata.shortVersionWindows=1.0.0.0"]):
        from scripts.bundles.desktop_inputs import validate_builder_identity
        with pytest.raises(ValueError, match="channel"):
            validate_builder_identity(request, args)
    assert not (tmp_path / "work").exists()


def test_prepare_cli_accepts_request_path_but_refuses_mismatched_checkout_before_provisioning(tmp_path):
    source, commit = _project(tmp_path)
    request = channel_request(commit)
    request["commit"] = "b" * 40
    path = tmp_path / "channel.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(repo / "scripts/bundles/desktop.py"),
                             "--prepare-only", "--repo", str(source), "--channel-request", str(path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "unrecognized arguments" not in result.stderr
    assert "checkout" in result.stderr
    assert not (source / ".build").exists()


def test_channel_environment_and_stamp_bind_the_request_not_ambient_oneoff_identity(tmp_path, monkeypatch):
    from scripts.bundles.desktop_inputs import identity_environment
    from scripts import write_install_stamp

    source, commit = _project(tmp_path)
    admitted = channel_request(commit)
    request = BuildRequest.create(source, tag=None, commit=None, variant="bundled",
                                  work=tmp_path / "work", cache=tmp_path / "cache",
                                  bundle_env={}, channel_request=admitted)
    inherited = {"HERMES_BUILD_COMMIT": "b" * 40, "HERMES_PAYLOAD_TAG": "v9.9.9",
                 "GITHUB_SHA": "b" * 40, "GITHUB_REF_NAME": "workflow-branch", "BUILD_NUMBER": "999"}
    env = identity_environment(request, "bundled", inherited)
    assert json.loads(env["_HERMES_CHANNEL_REQUEST_JSON"]) == admitted
    assert env["HERMES_PAYLOAD_VERSION"] == admitted["version"]
    assert env["GITHUB_SHA"] == commit
    assert not {"HERMES_BUILD_COMMIT", "HERMES_PAYLOAD_TAG", "GITHUB_REF_NAME", "BUILD_NUMBER"} & env.keys()
    assert inherited["HERMES_BUILD_COMMIT"] == "b" * 40
    changed = request.data()
    changed["channel_request"]["bundleEnv"] = changed["bundle_env"] = {}
    assert BuildRequest.from_data(changed).identity_digest() != request.identity_digest()
    from scripts.bundles.desktop_prepare import PreparedDesktop
    prepared = PreparedDesktop(request, Path(sys.executable), Path(sys.executable), Path(sys.executable),
                               tmp_path / "native", tmp_path / "packager", None, {}, "fixture",
                               request.identity_digest())
    assert request.channel_request is not None
    request.channel_request["buildId"] = "c" * 32
    with pytest.raises(ValueError, match="identity changed"):
        prepared.validate()
    monkeypatch.setattr(write_install_stamp, "_REPO_ROOT", source)
    monkeypatch.setenv("HERMES_DESKTOP_VARIANT", "bundled")
    # Python receives the request explicitly, not a new runtime setting.
    stamp = write_install_stamp.build_stamp(update_mechanism="electron-updater", channel_request=admitted)
    assert stamp["source"] == "channel-build"
    assert stamp["dirty"] is False
    assert stamp["channelBuild"] == admitted
    assert stamp["tag"] is None and stamp["branch"] is None
    assert stamp["baseVersion"] == admitted["sourceVersion"]
    assert stamp["updateMechanism"] == "electron-updater"
    admitted["bundleEnv"].clear()
    assert stamp["channelBuild"]["bundleEnv"]
