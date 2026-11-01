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
