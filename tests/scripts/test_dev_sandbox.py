"""Both development entrypoints keep their data identities and cleanup contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("entry,prefix", [("dev-sandbox.sh", "HermesSandbox-"), ("dev-minimal-sandbox.sh", "HermesMinimalSandbox-")])
def test_ephemeral_sandbox_preserves_command_and_removes_state(tmp_path, entry, prefix):
    root = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True, stdin=subprocess.DEVNULL, timeout=10)
    output = tmp_path / "observed.json"
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "keys = ['HERMES_HOME', 'HERMES_DESKTOP_USER_DATA_DIR', 'HERMES_DESKTOP_APP_NAME']\n"
        "Path(sys.argv[1]).write_text(json.dumps({'env': {k: os.environ[k] for k in keys}, 'args': sys.argv[2:]}), encoding='utf-8')\n"
        "sys.exit(7)\n", encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("HERMES_DEV_SANDBOX_")}
    result = subprocess.run(["bash", str(root / "scripts" / entry), "--", sys.executable,
                             str(probe), str(output), "argument with spaces"], cwd=checkout,
                            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
    assert result.returncode == 7, result.stderr
    observed = json.loads(output.read_text(encoding="utf-8-sig"))
    home = Path(observed["env"]["HERMES_HOME"])
    assert observed["args"] == ["argument with spaces"]
    assert observed["env"]["HERMES_DESKTOP_APP_NAME"].startswith(prefix)
    assert Path(observed["env"]["HERMES_DESKTOP_USER_DATA_DIR"]).parent == home.parent
    assert not home.parent.exists()


@pytest.mark.platforms("posix")
def test_persistent_identities_seed_once_and_honor_overrides(tmp_path):
    root = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True, stdin=subprocess.DEVNULL, timeout=10)
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "marker").write_text("first", encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("HERMES_DEV_SANDBOX_")}
    for entry, directory in [("dev-sandbox.sh", ".hermes-sandbox"), ("dev-minimal-sandbox.sh", ".hermes-minimal-sandbox")]:
        command = ["bash", str(root / "scripts" / entry), "--persistent", "--from", str(seed), "--", "true"]
        subprocess.run(command, cwd=checkout, env=env, check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=20)
        marker = checkout / directory / "hermes-home" / "marker"
        assert marker.read_text(encoding="utf-8-sig") == "first"
        marker.write_text("retained", encoding="utf-8")
        subprocess.run(command, cwd=checkout, env=env, check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=20)
        assert marker.read_text(encoding="utf-8-sig") == "retained"
    env.update(HERMES_DEV_SANDBOX_DIR=".custom", HERMES_DEV_SANDBOX_NAME="CustomSandbox")
    result = subprocess.run(["bash", str(root / "scripts/dev-minimal-sandbox.sh"), "--persistent", "--", "env"],
                            cwd=checkout, env=env, check=True, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
    assert f"HERMES_HOME={checkout / '.custom/hermes-home'}" in result.stdout
    assert "HERMES_DESKTOP_APP_NAME=CustomSandbox" in result.stdout
    assert (checkout / ".custom/user-data").is_dir()
