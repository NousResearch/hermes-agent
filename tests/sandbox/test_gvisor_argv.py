"""gVisor provider passes Docker runtime flag."""

import sys

import pytest

from sandbox.providers.gvisor import GVisorSandboxProvider


@pytest.mark.skipif(sys.platform == "win32", reason="docker argv POSIX-oriented")
def test_gvisor_argv_contains_runtime():
    prov = GVisorSandboxProvider(
        {"image": "alpine:3.19", "gvisor": {"docker_runtime": "runsc"}, "profiles": {"default": {}}}
    )
    argv = prov.build_popen_argv(
        workdir="/tmp/gv",
        inner_cmd=["python", "x.py"],
        child_env={},
    )
    idx = argv.index("--runtime")
    assert argv[idx + 1] == "runsc"
