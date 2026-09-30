"""Standalone relay routing must not bootstrap the gateway or reload secrets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import hermes_yaml as yaml


@pytest.mark.parametrize("case", ["native", "url-only", "disabled", "managed", "managed-only", "scoped"])
def test_cold_fronted_platforms_is_read_only(tmp_path, case):
    home = tmp_path / "primary"
    home.mkdir()
    scoped = tmp_path / "secondary"
    scoped.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    config = {"platforms": {"relay": {"enabled": True}}}
    if case == "disabled":
        config["platforms"]["relay"]["enabled"] = False
    if case not in {"native", "url-only", "managed-only"}:
        (home / "config.yaml").write_text(yaml.safe_dump(config))
    if case in {"managed", "managed-only"}:
        (managed / "config.yaml").write_text("platforms:\n  relay:\n    enabled: false\n")
    if case == "scoped":
        (scoped / "config.yaml").write_text("platforms:\n  relay:\n    enabled: false\n")
    for root in (home, scoped):
        (root / ".env").write_text("GATEWAY_RELAY_SECRET=dotenv-only-test-secret\n")
    # Allowlist only: never inherit live credentials, profile selectors, or
    # pytest's imported modules. HOME and cwd are temporary as well as HERMES_HOME.
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR") if k in os.environ}
    env.update({"HOME": str(tmp_path), "USERPROFILE": str(tmp_path),
                "HERMES_HOME": str(home), "HERMES_MANAGED_DIR": str(managed),
                "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"})
    if case != "native":
        env.update({"GATEWAY_RELAY_URL": "wss://connector.example/relay",
                    "GATEWAY_RELAY_PLATFORMS": "slack",
                    "GATEWAY_RELAY_SECRET": "inherited-test-secret"})
    code = textwrap.dedent("""
        import json, os, sys
        from hermes_constants import set_hermes_home_override
        if sys.argv[1] == "scoped":
            set_hermes_home_override(sys.argv[2])
        before = dict(os.environ)
        assert "gateway.run" not in sys.modules
        from gateway.relay import relay_fronted_platforms
        fronted = relay_fronted_platforms()
        changed = {k: [before.get(k), os.environ.get(k)]
                   for k in before.keys() | os.environ.keys()
                   if before.get(k) != os.environ.get(k)}
        print(json.dumps({"fronted": sorted(fronted),
                          "bootstrapped": "gateway.run" in sys.modules,
                          "changed_env": changed}))
    """)
    child = subprocess.run([sys.executable, "-c", code, case, str(scoped)],
                           cwd=tmp_path, env=env, text=True, capture_output=True, timeout=45)
    assert child.returncode == 0, child.stderr
    observed = json.loads(child.stdout.splitlines()[-1])
    assert observed == {"fronted": ["slack"] if case == "url-only" else [],
                        "bootstrapped": False, "changed_env": {}}
