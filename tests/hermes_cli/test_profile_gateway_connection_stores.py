"""The machine's saved-gateways store never travels with a profile.

Contracts: ``profile create --clone-all`` leaves ``gateway-connections.json``,
``gateway-connection-tokens.json`` and their lock/corrupt sidecars behind (they are root state every
profile reads at the root); ``profile export`` ships none of them, and any ``{encoding, value}`` secret
envelope that reaches an export in another JSON file is masked. Driven through the real CLI.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

SECRET = "SECRET-SESSION-TOKEN-123"
REPO = Path(__file__).resolve().parents[2]


def _hermes(home: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_")}
    env.update(HOME=str(home), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", *args], env=env, cwd=home,
                          capture_output=True, text=True, timeout=120, check=False)


def _seed_root(home: Path) -> Path:
    root = home / ".hermes"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text("model: test\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_")}
    env.update(HOME=str(home), HERMES_HOME=str(root), PYTHONPATH=str(REPO))
    script = ("from hermes_agent.hermes_cli import gateway_connections as g\n"
              f"g.save_connection({{'kind': 'remote', 'label': 'Box', 'url': 'http://10.0.0.5:9119', 'token': '{SECRET}'}})\n"
              "c, _ = g.save_connection({'kind': 'remote', 'label': 'Gw', 'url': 'https://gw.example', 'authMode': 'oauth'})\n"
              "g.write_tokens(c['id'], {'accessToken': 'AT-123456789012345', 'refreshToken': 'RT-123456789012345'})\n")
    subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=60)
    (root / "gateway-connections.json.corrupt-1700000000000").write_text(
        json.dumps({"connections": [{"token": {"encoding": "plain", "value": SECRET}}]}), encoding="utf-8")
    (root / "gateway-connections.json.lock").mkdir()
    return root


def test_clone_all_and_export_leave_the_saved_gateways_store_and_its_secrets_behind(tmp_path):
    home = tmp_path / "home"
    root = _seed_root(home)
    assert SECRET in (root / "gateway-connections.json").read_text()

    created = _hermes(home, "profile", "create", "cl", "--clone-all", "--no-alias")
    assert created.returncode == 0, created.stderr + created.stdout
    clone = root / "profiles" / "cl"
    assert clone.is_dir()
    assert not [p.name for p in clone.iterdir() if p.name.startswith("gateway-connection")]

    # A store copied in by hand (an older clone) still never ships, and neither does an envelope.
    for name in ("gateway-connections.json", "gateway-connection-tokens.json",
                 "gateway-connections.json.corrupt-1700000000000"):
        (clone / name).write_bytes((root / name).read_bytes())
    (clone / "notes.json").write_text(json.dumps({"saved": {"encoding": "plain", "value": SECRET}}))
    out = tmp_path / "cl.tar.gz"
    exported = _hermes(home, "profile", "export", "cl", "-o", str(out))
    assert exported.returncode == 0, exported.stderr + exported.stdout
    with tarfile.open(out) as archive:
        names = archive.getnames()
        blob = b"".join(archive.extractfile(m).read() for m in archive.getmembers() if m.isfile())
    assert not [n for n in names if "gateway-connection" in n], names
    assert "cl/notes.json" in names
    assert SECRET.encode() not in blob
