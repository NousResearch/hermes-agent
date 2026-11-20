"""Stable and canary share alias publication, with policy checks in their callers."""
from collections import Counter
import subprocess

import pytest

from scripts.releases import docker


DIGEST = "sha256:" + "d" * 64


def test_alias_publication_retries_push_and_each_alias_readback():
    creates, inspections, sleeps = [], Counter(), []

    def run(argv):
        if argv[3] == "create":
            creates.append(argv)
            if len(creates) == 1:
                raise subprocess.CalledProcessError(1, argv)
            return ""
        assert argv[3] == "inspect"
        inspections[argv[4]] += 1
        return DIGEST if inspections[argv[4]] == 2 else "sha256:" + "e" * 64

    docker._publish_aliases(DIGEST, ("stable", "latest"), run=run, sleep=sleeps.append)
    command = ["docker", "buildx", "imagetools", "create",
               "-t", f"{docker.IMAGE}:stable", "-t", f"{docker.IMAGE}:latest", f"{docker.IMAGE}@{DIGEST}"]
    assert creates == [command, command]
    assert inspections == {f"{docker.IMAGE}:stable": 2, f"{docker.IMAGE}:latest": 2}
    assert sleeps == [20, 20, 20]


@pytest.mark.parametrize("policy", ["stable", "canary"])
@pytest.mark.parametrize("failure", ["push", "readback", "inspect"])
def test_both_policies_preserve_retry_exhaustion_and_fail_closed(policy, failure):
    tag = "rc.1-v1.2.3" if policy == "stable" else "canary.20261028T120000Z-v1.2.3"
    calls, sleeps = [], []

    def run(argv):
        calls.append(argv)
        if argv[3] == "create":
            if failure == "push":
                raise subprocess.CalledProcessError(1, argv)
            return ""
        if argv[4] in (f"{docker.IMAGE}:{tag}", f"{docker.IMAGE}:{tag}-desktop"):
            return DIGEST
        if failure == "inspect":
            raise subprocess.CalledProcessError(1, argv)
        return "sha256:" + "e" * 64

    error = docker.DockerReleaseError if failure == "readback" else subprocess.CalledProcessError
    with pytest.raises(error) as raised:
        if policy == "stable":
            docker.promote_stable(tag, DIGEST, "v1.2.3", run=run, sleep=sleeps.append)
        else:
            docker.promote_canary(tag, run=run, sleep=sleeps.append)
    assert [cmd[4] for cmd in calls[:2]] == [f"{docker.IMAGE}:{tag}", f"{docker.IMAGE}:{tag}-desktop"]
    creates = [cmd for cmd in calls if cmd[3] == "create"]
    assert len(creates) == (3 if failure == "push" else 1)
    assert sleeps == ([] if failure == "inspect" else [20, 20])
    if failure == "readback":
        alias = "v1.2.3" if policy == "stable" else "canary"
        assert str(raised.value) == f"Docker {alias} alias read-back mismatch"
        assert len([cmd for cmd in calls[2:] if cmd[3] == "inspect"]) == 3
    if policy == "canary":
        assert all(cmd[4:] == ["-t", f"{docker.IMAGE}:canary", f"{docker.IMAGE}@{DIGEST}"] for cmd in creates)


def test_promote_canary_refuses_rollback_behind_live_head_before_touching_docker(monkeypatch):
    from scripts.releases import versioning

    older = "canary.20261027T064100Z-v1.2.3"
    newer = "canary.20261028T064100Z-v1.2.3"
    docker_calls = []

    def fail_if_called(argv):
        docker_calls.append(argv)
        raise AssertionError(f"docker must not run for a stale canary: {argv}")

    monkeypatch.setattr(
        versioning,
        "published_channel_identity",
        lambda repo, channel, *, base_url=None: ("1.2.3+canary.20261028T064100Z", "a" * 40),
    )
    with pytest.raises(docker.DockerReleaseError, match="roll Docker canary aliases back"):
        docker.promote_canary(older, run=fail_if_called, repository="o/r", public_base="https://r2.example")
    assert docker_calls == []

    # Same-stamp retry of the current live canary is allowed, and a newer canary over a
    # legacy live head is also allowed.
    for candidate, live_version in (
        (newer, "1.2.3+canary.20261028T064100Z"),
        ("canary.20261120T064100Z-v0.23.2", "0.23.2+canary.20261119T064637Z"),
    ):
        seen = []

        def run(argv):
            seen.append(argv)
            return "" if argv[3] == "create" else DIGEST

        monkeypatch.setattr(
            versioning,
            "published_channel_identity",
            lambda repo, channel, *, base_url=None, v=live_version: (v, "b" * 40),
        )
        docker.promote_canary(candidate, run=run, sleep=lambda _s: None, repository="o/r",
                              public_base="https://r2.example")
        creates = [cmd for cmd in seen if cmd[3] == "create"]
        assert len(creates) == 2

    # Unreadable live R2 head fails closed without touching Docker Hub.
    from hermes_cli.release_channels import ChannelError

    def broken(_repo, _channel, *, base_url=None):
        raise ChannelError("R2 unreachable")

    monkeypatch.setattr(versioning, "published_channel_identity", broken)
    with pytest.raises(docker.DockerReleaseError, match="live canary head"):
        docker.promote_canary(newer, run=fail_if_called, repository="o/r", public_base="https://r2.example")
    assert docker_calls == []


def test_promote_canary_cli_wires_repository_and_public_base(monkeypatch):
    received = []
    monkeypatch.setattr(
        docker,
        "promote_canary",
        lambda tag, **kw: received.append((tag, kw)),
    )
    assert docker.main([
        "promote-canary", "--tag", "canary.20261028T064100Z-v1.2.3",
        "--repository", "o/r", "--public-base", "https://r2.example",
    ]) == 0
    assert received == [
        ("canary.20261028T064100Z-v1.2.3", {"repository": "o/r", "public_base": "https://r2.example"}),
    ]
    with pytest.raises(SystemExit):
        docker.main(["promote-canary", "--tag", "canary.20261028T064100Z-v1.2.3"])
