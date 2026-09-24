"""Per-profile secret scope on ONE multi-profile ``hermes serve`` host.

Class: multiplex secret isolation. Users hit it as "my dashboard plugin says 'Balance unavailable'
only when I have more than one profile" (#120310) or, worse, "profile B's MCP server received
profile A's API key". Under multi-profile hosting ``get_secret`` fails closed with no scope bound,
so every request path must bind the REQUESTING profile's scope — never borrow the launch profile's
``os.environ`` and never leave the read unscoped.

Harness: a fake HOME with two profiles (``default`` = launch home, ``beta`` under ``profiles/``);
each ``.env`` carries a distinct random canary under the same variable NAME plus the tenancy
harness canaries (provider key, API-server key, marker). A user dashboard plugin whose API route
reads that variable through ``agent.secret_scope.get_secret`` is enabled via ``plugins.enabled``,
and each profile configures a stdio MCP server that dumps the environment it was spawned with and
is handed exactly one secret (``env: {X: ${SECRET}}``). One real ``hermes serve`` hosts both;
the Desktop's per-request profile selector is ``?profile=<name>`` (HTTP) / ``session.create
{profile}`` (``/api/ws``).

Scenarios (one boot):
- control: the fixed sibling path ``GET /api/mcp/servers?profile=X`` resolves each profile's own
  ``${SECRET}`` reference (proves the host is multiplexing and that scoped paths see the right
  profile, so the plugin scenario is not vacuous);
- ``plugin_route`` (KNOWN #120310): ``/api/plugins/<id>/secret`` returns the requesting profile's
  value, A -> B -> A, never the other's and never ``UnscopedSecretError``;
- MCP subprocess env: each profile's server env carries its own configured secret and none of the
  other profile's canaries, nor the launch profile's provider key (never configured via ``env:``).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.tenancy import _helpers as T

from . import _scope as S
from ._helpers import BoundaryBreach, known_param, poll

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups (serve teardown)")

PROFILES = ("default", "beta")

KNOWN: dict[str, str] = {
    "plugin_route": "#120310 /api/plugins/<id>/ routes run with no profile secret scope under multiplex",
}


class Host:
    def __init__(self, root: Path, tenants: dict[str, T.Tenant], secrets: dict[str, str],
                 backend: T.ServeBackend) -> None:
        self.root, self.tenants, self.secrets, self.backend = root, tenants, secrets, backend

    def get(self, path: str, profile: str | None) -> tuple[int, Any]:
        query = f"?profile={profile}" if profile else ""
        req = urllib.request.Request(f"http://127.0.0.1:{self.backend.port}{path}{query}",
                                     headers={"Authorization": f"Bearer {self.backend.token}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode(errors="replace")

    def dump(self, name: str) -> Path:
        return self.root / f"mcp-env-{name}.json"

    def foreign(self, owner: str) -> dict[str, str]:
        """Every secret-bearing canary that does NOT belong to ``owner``."""
        out = {f"{u}:{k}": v for u, t in self.tenants.items() if u != owner for k, v in t.canaries().items()}
        out.update({f"{u}:{S.SECRET_ENV}": v for u, v in self.secrets.items() if u != owner})
        return out


@pytest.fixture(scope="module")
def host(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("scope")
    home = root / "home"
    tenants = T.make_tenants(root, PROFILES)
    home.mkdir(parents=True, exist_ok=True)
    T.assert_profiles_root_under(root, home)  # HOME-anchored profile root: prove it is sandboxed first
    server = S.write_env_dump_server(root / "envdump_server.py")
    # the suffix names the owner, so even the 4-char tail a redacted display keeps is unambiguous
    secrets = {name: f"scopesecret-{t.tag}-{name[:4]}" for name, t in tenants.items()}
    for name, t in tenants.items():
        assert t.srv is not None
        t.srv.start()
        # the plugin lives in the launch home; every profile enables it (a fix may gate per profile)
        cfg: dict[str, Any] = {"mcp_servers": S.mcp_config(server, root / f"mcp-env-{name}.json"),
                               "plugins": {"enabled": [S.PLUGIN_ID]}}
        T.write_tenant_home(t, extra_config=cfg, extra_env={S.SECRET_ENV: secrets[name]})
    S.install_plugin(tenants["default"].home)
    backend = T.ServeBackend(home, root / "serve.log")
    try:
        h = Host(root, tenants, secrets, backend)
        # one Desktop session per profile: each spawns that profile's MCP servers
        for name in PROFILES:
            backend.ok("session.create", {} if name == "default" else {"profile": name})
        yield h
    finally:
        backend.close()
        for t in tenants.values():
            assert t.srv is not None
            t.srv.stop()
        assert not T.pid_alive(backend.proc.pid), f"serve pid {backend.proc.pid} survived teardown"


def _selector(name: str) -> str | None:
    return None if name == "default" else name  # the launch profile is what an unqualified request means


def test_scoped_sibling_route_resolves_each_profiles_secret(host: Host) -> None:
    """Control: the MCP list route (fixed for #113746) runs in the requested profile's scope, so each
    profile's ``${SECRET}`` reference resolves to its OWN value on this same multiplexed host."""
    for name in ("default", "beta", "default"):
        status, body = host.get("/api/mcp/servers", _selector(name))
        assert status == 200, f"/api/mcp/servers for {name}: HTTP {status} {body}"
        [entry] = [s for s in body["servers"] if s["name"] == S.MCP_SERVER_NAME]
        shown = entry["env"][S.MCP_ENV_KEY]
        own = host.secrets[name]
        if not (shown == own or shown.endswith(own[-4:])):
            others = [u for u, v in host.secrets.items() if u != name and shown.endswith(v[-4:])]
            raise BoundaryBreach(f"/api/mcp/servers?profile={name} resolved ${{{S.SECRET_ENV}}} to {shown!r} "
                                 f"(own ends {own[-4:]!r}; matches {others or 'nobody'})")


@pytest.mark.parametrize("scenario", [known_param("plugin_route", KNOWN)])
def test_plugin_route_runs_in_requesting_profile_scope(host: Host, scenario: str) -> None:
    replies: list[str] = []
    for name in ("default", "beta", "default"):
        status, body = host.get(f"/api/plugins/{S.PLUGIN_ID}/secret", _selector(name))
        assert status == 200, f"plugin route not mounted/served for {name}: HTTP {status} {body}"
        if not body.get("ok"):
            replies.append(f"{name}: get_secret raised {body.get('error')}")
            continue
        value = body.get("value")
        if value != host.secrets[name]:
            owner = next((u for u, v in host.secrets.items() if v == value), "<nobody>")
            replies.append(f"{name}: got {owner}'s value {value!r}")
    if replies:
        raise BoundaryBreach(f"{scenario}: plugin API route did not run in the requesting profile's "
                             "secret scope:\n  " + "\n  ".join(replies))


def test_mcp_subprocess_env_holds_only_own_profile_secrets(host: Host) -> None:
    envs = {name: poll(lambda n=name: host.dump(n).is_file() and json.loads(host.dump(n).read_text()), 90,
                       f"{name}'s MCP server to dump its env") for name in PROFILES}
    problems: list[str] = []
    for name, env in envs.items():
        blob = json.dumps(env)
        if env.get(S.MCP_ENV_KEY) != host.secrets[name]:
            problems.append(f"{name}: configured {S.MCP_ENV_KEY} is {env.get(S.MCP_ENV_KEY)!r}, "
                            f"not its own {S.SECRET_ENV}")
        problems += [f"{name}: MCP server env carries {label}" for label, v in host.foreign(name).items()
                     if v in blob]
        # the launch profile's provider key is never handed to any MCP server (no env: entry names it)
        launch_key = host.tenants["default"].provider_key
        if launch_key in blob or T.PROVIDER_KEY_ENV in env:
            problems.append(f"{name}: MCP server env carries the launch profile's provider key")
    if problems:
        raise BoundaryBreach("MCP subprocess env crossed a profile boundary:\n  " + "\n  ".join(problems))
