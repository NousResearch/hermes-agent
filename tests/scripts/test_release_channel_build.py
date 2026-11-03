"""release.py --channel NAME --build-commit REV claims a channel build; Stable Release builds it."""
import json
import sys

import pytest

SHA = "a" * 40
TAG = "channel.20261028T120000Z-v1.2.3"


def test_a_channel_claim_names_its_channel_and_branding_and_admits_as_one():
    from scripts.releases import commit_build, lanes, stable

    selected = lanes.select(policy="channel")
    raw = commit_build.claim_message(version="1.2.3", commit=SHA, lanes=selected, require_green=False,
                                     bundle_env={"HERMES_PREVIEW_FULL_CONNECTORS": "1"},
                                     channel="custom-branch", branding="stable")
    claim = json.loads(raw)
    assert claim["kind"] == "channel" and claim["channel"] == "custom-branch" and claim["branding"] == "stable"
    admitted = stable.stamped_claim(raw, tag=TAG, commit=SHA)
    assert admitted["channel"] == "custom-branch" and admitted["requireGreen"] is False
    # A commit claim carries no channel, and a channel claim takes no unknown branding or bad name.
    assert "channel" not in json.loads(commit_build.claim_message(
        version="1.2.3", commit=SHA, lanes=lanes.select(policy="commit"), require_green=True, bundle_env={}))
    with pytest.raises(ValueError, match="branding"):
        commit_build.claim_message(version="1.2.3", commit=SHA, lanes=selected, require_green=True,
                                   bundle_env={}, channel="custom-branch", branding="loud")
    with pytest.raises(ValueError):
        commit_build.claim_message(version="1.2.3", commit=SHA, lanes=selected, require_green=True,
                                   bundle_env={}, channel="bad/name")


def test_channel_lanes_refuse_what_a_channel_build_cannot_run():
    from scripts.releases import lanes

    assert "install-e2e" in lanes.select(policy="channel")
    for refused in ("upgrade", "windows-store"):
        assert refused not in lanes.select(policy="channel")


def test_dispatch_runs_stable_release_on_a_commit_or_channel_claim_only():
    from scripts.releases import commit_build

    command = commit_build.dispatch_command(TAG, "example/hermes-agent", "main")
    assert command[:3] == ["gh", "workflow", "run"] and command[-1] == f"tag={TAG}"
    assert commit_build.dispatch_command("commit.20261028T120000Z-v1.2.3", "example/hermes-agent", "main")
    for other in ("canary.20261028T120000Z-v1.2.3", "rc.1-v1.2.3", "v1.2.3"):
        with pytest.raises(ValueError, match="claim tag"):
            commit_build.dispatch_command(other, "example/hermes-agent", "main")


def test_channel_specific_data_dirs_bake_a_per_channel_suffix():
    from argparse import Namespace

    from scripts.releases.commit_build import channel_bundle_env

    args = Namespace(bundle_env=["HERMES_PREVIEW_FULL_CONNECTORS=1"], bundle_unset=[], channel="custom-branch",
                     channel_specific_data_dirs=True)
    assert channel_bundle_env(args) == {"HERMES_PREVIEW_FULL_CONNECTORS": "1", "HERMES_DATA_DIR_SUFFIX": "-channel-build-custom-branch",
                                        "HERMES_HOME": None, "HERMES_DESKTOP_USER_DATA_DIR": None,
                                        "HERMES_SHARED_AUTH_DIR": None}


def test_channel_administration_talks_to_production_records(monkeypatch):
    from scripts.releases import channel_build, r2

    monkeypatch.setenv("CLOUDFLARE_R2_PUBLIC_URL", "https://archive.example")
    monkeypatch.setattr(r2, "credentials", lambda: ({"access_key_id": "inert", "secret_key": "inert"},
                                                    "https://r2.example", "bucket"))
    publisher = channel_build.configured_publisher("ethernet8023/hermes-agent")
    assert publisher.reader.base_url == "https://archive.example"
    assert not hasattr(publisher.store, "scope")


def test_release_parser_routes_channel_builds_through_the_claim_flow(monkeypatch):
    from scripts import release
    from scripts.releases import channel_build, commit_build

    calls = []
    monkeypatch.setattr(commit_build, "cmd_build_commit",
                        lambda args: calls.append(("claim", args.build_commit, args.channel, args.only)))
    monkeypatch.setattr(channel_build, "cmd_channel", lambda args: calls.append(("admin", args.channels)))

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["release.py", *argv])
        release.main()

    run("--build-commit", "main")
    run("--channel", "custom-branch", "--build-commit", "main", "--only", "bundles", "--no-require-green")
    run("--channel", "custom-branch", "--branding", "stable", "--build-commit", "main")
    run("--channels")
    assert calls == [("claim", "main", None, None), ("claim", "main", "custom-branch", ["bundles"]),
                     ("claim", "main", "custom-branch", None), ("admin", True)]
    for argv in (("--channel", "bad/name", "--build-commit", "main"),
                 ("--channels", "--bundle-env", "X=y"),
                 ("--branding", "stable", "--build-commit", "main"),
                 ("--channel", "custom-branch")):
        with pytest.raises(SystemExit) as refused:
            run(*argv)
        assert refused.value.code == 2
    assert len(calls) == 4
