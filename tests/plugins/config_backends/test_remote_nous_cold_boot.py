"""A Cloud agent's first remote config read, in a cold process, with the Nous credential.

In Cloud mode (no ``GATEWAY_RELAY_IDP_*``) the plane bearer is the Nous Portal access token from the
profile's ``auth.json``. Resolving it imports ``hermes_cli.auth`` inside the profile's first fetch;
that import discovers model providers, and provider discovery reads config (the gate on
pip-installed provider plugins). That read used to wait on the fetch's own lock forever, so a Cloud
agent never booted (stage2's ``refresh_dependencies`` hung). These tests run real imports in a
fresh interpreter, as the boot does; the in-process suites stub the credential and cannot see it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from .stub_plane import INSTANCE, NOUS_TOKEN, StubPlane

REPO = Path(__file__).resolve().parents[3]
HARD_TIMEOUT_S = 60  # a healthy run takes about a second; the deadlock never ends

# What docker/stage2-hook.sh runs before the gateway starts: refresh_dependencies() first reads the
# plugin selection through pm.workspace (the read that hung), then the CLI reads a value.
STAGE2_SNIPPET = """
import json, os, sys
from pathlib import Path
from pm.recovery import refresh_dependencies  # stage2-hook.sh's import
assert "hermes_cli.auth" not in sys.modules and "hermes_cli.config" not in sys.modules  # cold
from pm.workspace import enabled_member_dirs
enabled_member_dirs()  # refresh_dependencies -> venv_is_current -> expected_stamp reads this first
import providers
if os.environ.get("PROBE_PROVIDER"):  # a $HERMES_HOME provider plugin loads once config is loaded
    assert providers.get_provider_profile("probe-provider") is not None
from hermes_cli.config import load_config
print("RESULT=" + json.dumps(load_config()["display"]["skin"]))
"""


def _iso(offset_s: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() + offset_s))


def _auth_json(plane: StubPlane, *, expired: bool) -> dict:
    nous = {"access_token": "expired-token" if expired else NOUS_TOKEN, "refresh_token": "nous-refresh-1",
            "client_id": "hermes-cli-vps", "portal_base_url": plane.url,
            "expires_at": _iso(-3600 if expired else 3600)}
    return {"version": 1, "active_provider": "nous", "providers": {"nous": nous}}


def _run(args, env):
    try:
        return subprocess.run(args, cwd=REPO, env=env, capture_output=True, text=True, timeout=HARD_TIMEOUT_S,
                              stdin=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{args[1:]} did not finish in {HARD_TIMEOUT_S}s: the first remote read deadlocked.\n"
                    f"{(exc.stderr or b'')[-3000:]!r}")


def _value(proc) -> str:
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT=")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(lines[-1][len("RESULT="):])


PROBE_INIT = (
    "from providers import register_provider\n"
    "from providers.base import ProviderProfile\n"
    "register_provider(ProviderProfile(name='probe-provider', base_url='https://probe.test/v1'))\n")


def _install_probe_provider(home: Path, layout: str) -> None:
    """A $HERMES_HOME provider plugin: importing one reads ``plugins.isolation`` from config."""
    plugin = home / "plugins" / ("model-providers/probe" if layout == "model-providers" else "probe")
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: probe\nkind: model-provider\nversion: 0.0.1\n")
    (plugin / "__init__.py").write_text(PROBE_INIT)


@pytest.mark.parametrize("expired, layout", [(False, None), (True, None), (False, "model-providers"), (False, "flat")],
                         ids=["valid-token", "token-refreshed-at-boot", "home-provider-plugin", "flat-provider-plugin"])
def test_cloud_agent_first_remote_read_completes_cold(tmp_path, expired, layout):
    home = tmp_path / "home"
    home.mkdir()
    if layout:
        _install_probe_provider(home, layout)
    with StubPlane() as plane:
        plane.upper = {"display": {"skin": "ares"}}
        (home / "auth.json").write_text(json.dumps(_auth_json(plane, expired=expired)))
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "TZ", "TMPDIR", "PYTHONHASHSEED"}}
        env.update(HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1",
                   HERMES_CONFIG_BACKEND="remote", HERMES_CONFIG_REMOTE_URL=plane.url,
                   HERMES_CONFIG_INSTANCE_ID=INSTANCE, PROBE_PROVIDER="1" if layout else "")
        assert not any(k.startswith("GATEWAY_RELAY_IDP_") for k in env)  # Cloud: the Nous credential

        assert _value(_run([sys.executable, "-c", STAGE2_SNIPPET], env)) == "ares"
        cli = _run([sys.executable, "-m", "hermes_cli.main", "config", "get", "display.skin"], env)
        assert cli.returncode == 0, cli.stderr[-3000:]
        assert cli.stdout.strip().splitlines()[-1] == "ares"

        gets = [r for r in plane.requests if r["method"] == "GET"]
        assert len(gets) >= 2 and {r["auth"] for r in gets} == {f"Bearer {NOUS_TOKEN}"}
        assert plane.nous_refreshes == (1 if expired else 0)
        assert not (home / "config.yaml").exists()


AUTH_IMPORT_READS_NO_CONFIG = """
import sys
import hermes_cli.config_backend as cb
reads = []
def refuse(*a, **k):
    reads.append(1)
    raise cb.ConfigBackendUnavailable("this test's backend refuses every read")
cb._route = refuse  # every config read, through any backend, now raises
import hermes_cli.auth
from hermes_cli.auth import resolve_nous_access_token
assert resolve_nous_access_token() == sys.argv[1]
assert reads == [], reads
print("RESULT=" + '"ok"')
"""


def test_importing_auth_and_resolving_the_nous_token_read_no_config(tmp_path):
    """The Cloud credential path (import hermes_cli.auth, resolve the Nous token) must not need the
    config it is used to fetch: under a backend whose every read raises, both succeed cold."""
    home = tmp_path / "home"
    home.mkdir()
    _install_probe_provider(home, "model-providers")  # a $HERMES_HOME provider plugin is not read either
    with StubPlane() as plane:
        (home / "auth.json").write_text(json.dumps(_auth_json(plane, expired=False)))
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "TZ", "TMPDIR", "PYTHONHASHSEED"}}
        env.update(HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
        assert _value(_run([sys.executable, "-c", AUTH_IMPORT_READS_NO_CONFIG, NOUS_TOKEN], env)) == "ok"


CONCURRENT_AUTH_IMPORT = """
import json, os, sys, threading
from pathlib import Path
from plugins.config_backends.remote import client
from plugins.config_backends.remote import get_remote_backend
home = Path(os.environ["HERMES_HOME"])
in_fetch, imported = threading.Event(), threading.Event()
real_request = client.request
waited = {}
def request(method, *args, **kwargs):  # thread A, holding the profile's fetch lock
    in_fetch.set()
    waited["b_done"] = imported.wait(20)
    return real_request(method, *args, **kwargs)
client.request = request
def b():
    in_fetch.wait(20)
    import hermes_cli.auth  # cold, while A holds the fetch lock
    imported.set()
tb = threading.Thread(target=b, daemon=True)
tb.start()
get_remote_backend().version(home)  # thread A: the profile's first read
tb.join(20)
assert waited.get("b_done") and not tb.is_alive(), "the cold auth import waited for the fetch in flight"
from hermes_cli.config import load_config
print("RESULT=" + json.dumps(load_config()["display"]["skin"]))
"""


def test_cold_auth_import_on_another_thread_never_waits_for_the_fetch(tmp_path):
    """Thread A holds a profile's fetch lock; thread B cold-imports hermes_cli.auth. The import
    must not read config (it would wait for A's fetch while A, resolving its own credential, can
    need the import lock B holds)."""
    from .stub_plane import remote_env

    home = tmp_path / "home"
    home.mkdir()
    with StubPlane() as plane:
        plane.upper = {"display": {"skin": "ares"}}
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "TZ", "TMPDIR", "PYTHONHASHSEED"}}
        env.update(remote_env(plane), HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(REPO),
                   PYTHONDONTWRITEBYTECODE="1")
        assert _value(_run([sys.executable, "-c", CONCURRENT_AUTH_IMPORT], env)) == "ares"

