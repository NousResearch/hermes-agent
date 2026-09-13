"""Installer SSH must use the same identity it provisions in the guest."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

PLUGIN = Path(__file__).resolve().parents[2] / "plugins/hermes-realms"
load = runpy.run_path(str(PLUGIN / "realms/_binding.py"))["load_runtime"]


@pytest.mark.linux_only
def test_guest_connections_pin_provisioned_key_and_disable_ambient_ssh(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    (home / ".ssh").mkdir()
    key = home / ".ssh/id_ed25519"
    key.touch()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, source in {
        "systemctl": "#!/bin/sh\nexit 0\n",
        "ssh": f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n",
    }.items():
        executable = binaries / name
        executable.write_text(source)
        executable.chmod(0o700)
    result = subprocess.run([str(PLUGIN / "realms/vendor/omarchy-vm"), "ssh", "true"],
                            env={"PATH": str(binaries) + ":/usr/bin:/bin", "HOME": str(home), "USER": "fixture"},
                            capture_output=True, text=True, check=True, timeout=10)
    installer = json.loads(result.stdout)
    manager = load("vm_manager").VmManager(tmp_path / "profile")
    runtime = manager.ssh_argv({"ssh_port": 2300})[1:]
    for argv in (installer, runtime):
        assert argv[argv.index("-F") + 1] == "/dev/null"
        assert argv[argv.index("-i") + 1] == str(key)
        assert {"IdentitiesOnly=yes", "ForwardAgent=no", "ForwardX11=no", "ForwardX11Trusted=no"} <= set(argv)
