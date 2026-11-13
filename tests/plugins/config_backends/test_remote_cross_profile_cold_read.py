"""A cold Remote Config read of profile B from inside profile A's bound scope completes.

In Cloud mode the plane bearer is the Nous token from the profile's ``auth.json``. The credential
reader used to select that file by moving the HERMES_HOME override to B while B's first fetch held
B's fetch lock. With a scope bound, that override move rebinds the WHOLE scope to B (the legacy
setter cannot split home from secrets and terminal policy), and building B's terminal policy reads
B's config: the same fetch, waiting on its own lock forever. Real imports in a fresh interpreter,
as boot runs; the in-process suites stub the credential and cannot see it.
"""
from __future__ import annotations

import json
import os
import sys

from .stub_plane import INSTANCE, NOUS_TOKEN, StubPlane
from .test_remote_nous_cold_boot import REPO, _auth_json, _run, _value

CROSS_PROFILE_READ = """
import json, os
from pathlib import Path
home = Path(os.environ["HERMES_HOME"])
from hermes_agent.agent.runtime_scope import bind_profile_scope
from hermes_cli.config_backend import get_config_backend
with bind_profile_scope(home / "profiles" / "a", hydrate_secrets=False):
    get_config_backend()._state(home / "profiles" / "b")  # B's first read, while A is bound
from hermes_scope import current_scope
assert current_scope() is None
print("RESULT=" + json.dumps("ok"))
"""


def test_cold_read_of_another_profile_inside_a_bound_scope_completes(tmp_path):
    home = tmp_path / ".hermes"
    for name in ("a", "b"):
        (home / "profiles" / name).mkdir(parents=True)
    with StubPlane() as plane:
        for directory in (home, home / "profiles" / "a", home / "profiles" / "b"):
            (directory / "auth.json").write_text(json.dumps(_auth_json(plane, expired=False)))
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "TZ", "TMPDIR", "PYTHONHASHSEED"}}
        env.update(HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1",
                   HERMES_CONFIG_BACKEND="remote", HERMES_CONFIG_REMOTE_URL=plane.url,
                   HERMES_CONFIG_INSTANCE_ID=INSTANCE)
        assert not any(k.startswith("GATEWAY_RELAY_IDP_") for k in env)  # Cloud: the Nous credential
        assert _value(_run([sys.executable, "-c", CROSS_PROFILE_READ], env)) == "ok"
        gets = [r for r in plane.requests if r["method"] == "GET"]
        assert {r["profile"] for r in gets} >= {"b"} and {r["auth"] for r in gets} == {f"Bearer {NOUS_TOKEN}"}
