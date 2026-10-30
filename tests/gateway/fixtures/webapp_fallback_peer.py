"""Start the gateway's listener the way the runtime does and report what native clients and a browser get.

Run as ``python webapp_fallback_peer.py '<spec json>'`` with an isolated ``HERMES_HOME``. The spec:

- ``webapp``: the ``webapp:`` block written to config.yaml, or null for none;
- ``dashboard_public_url``: ``dashboard.public_url``, or null;
- ``basic_auth``: register the bundled password sign-in provider (a throwaway account);
- ``fail_after_mount``: the listener's startup fails once, after the dashboard mount is up, as when
  another process starts listening on the port first;
- ``hold_port``: another socket already listens on the Webapp's port;
- ``credential_unusable``: a file stands where the browser credential's folder belongs, so the
  credential can be neither read nor written;
- ``rival_listens_during_mount``: another socket bound the port with SO_REUSEADDR before the gateway
  and starts listening while the gateway mounts the dashboard (Linux only);
- ``connect_during_mount``: a client connects to the Webapp's port while the dashboard mounts;
- ``built``: whether the Webapp bundle has its ``index.html``;
- ``update_fails``: before the gateway starts, ``hermes update``'s Webapp step runs for this home and its
  build fails; once the gateway serves, ``hermes webapp``'s build succeeds;
- ``probe_host``: a Host header to send (a public hostname), or null;
- ``credential``: the stored browser credential this gateway finds: ``"folder-is-a-file"`` (a plain
  file where the ``webapp/`` folder goes), ``"token-is-a-folder"`` (a folder where
  ``webapp/session-token`` goes), ``"not-utf8"`` (the token saved as UTF-16), or null for a
  readable token an earlier gateway stored.

Writes ``receipt.json`` into ``HERMES_HOME`` (the gateway redirects stdout). No inference: the model points
at a closed port and no turn is submitted. Tests start it with :func:`run`.
"""
import asyncio
import errno
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

SEEDED_TOKEN = "seeded-browser-credential-from-an-earlier-0"  # 43 URL-safe characters
INDEX = "<!doctype html><html><body>WEBAPP_BUNDLE</body></html>"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run(tmp_path, spec) -> dict:
    """Run this peer for *spec* in a fresh home under *tmp_path*; returns its receipt."""
    from tests.gateway.fixtures.local_recovery_probe import child_env

    root = Path(__file__).resolve().parents[3]
    home, user = tmp_path / "state", tmp_path / "user"
    for path in (home, user):
        path.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED="1")
    result = subprocess.run([sys.executable, __file__, json.dumps(spec)], cwd=root, env=env,
                            capture_output=True, text=True, timeout=120, check=False)
    receipt = home / "receipt.json"
    assert result.returncode == 0, result.stdout + result.stderr + (receipt.read_text() if receipt.exists() else "")
    return json.loads(receipt.read_text())


def _config(spec) -> str:
    document = {"gateway": {"multiplex_profiles": False},
                "model": {"provider": "custom", "default": "no-inference", "base_url": "http://127.0.0.1:1/v1"},
                "auxiliary": {"title_generation": {"enabled": False}}}
    dashboard = {}
    if spec.get("dashboard_public_url"):
        dashboard["public_url"] = spec["dashboard_public_url"]
    if spec.get("basic_auth"):
        dashboard["basic_auth"] = {"username": "fixture-user", "password": "throwaway-fixture-password"}
    if dashboard:
        document["dashboard"] = dashboard
    if spec.get("webapp") is not None:
        document["webapp"] = spec["webapp"]
    return json.dumps(document, indent=2)  # JSON is YAML


def _seed_credential(home, kind):
    """The browser credential an earlier gateway stored, or (*kind*) one this gateway cannot use."""
    folder = home / "webapp"
    if kind == "folder-is-a-file":
        folder.write_text("not a folder\n", encoding="utf-8")
        return
    # Seeded user-only, as the gateway writes it: a stored token other users can read is replaced.
    folder.mkdir(mode=0o700, exist_ok=True)
    token = folder / "session-token"
    if kind == "token-is-a-folder":
        token.mkdir()
    elif kind == "not-utf8":
        token.write_bytes((SEEDED_TOKEN + "\n").encode("utf-16"))
    else:
        token.write_text(SEEDED_TOKEN + "\n", encoding="utf-8")
        token.chmod(0o600)


def _stored_token(home):
    """The browser credential stored in *home*, or None when it cannot be read as text."""
    try:
        return (home / "webapp" / "session-token").read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None


def _fail_startup_once():
    """The listener's first startup raises the error a lost listen() race gives."""
    import uvicorn

    real = uvicorn.Server.startup
    calls = []

    async def startup(self, sockets=None):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.EADDRINUSE, "Address already in use")
        return await real(self, sockets=sockets)

    uvicorn.Server.startup = startup


def _fail_webapp_update(home, checkout):
    """``hermes update``'s Webapp step for an install whose only home is *home*, with a build that
    fails as a broken npm install would."""
    import hermes_cli.webapp
    from hermes_cli import source_build
    from pm import plugins_state

    def fail(_project_root, **_options):
        raise hermes_cli.webapp.WebappBuildError("Browser-hosted Desktop build failed: npm ci exited 1")

    plugins_state.dependency_homes = lambda: [home]
    real, hermes_cli.webapp.prepare_webapp_renderer = hermes_cli.webapp.prepare_webapp_renderer, fail
    try:
        source_build._rebuild_served_webapp(checkout, lambda _stage: None)
    except hermes_cli.webapp.WebappBuildError:
        pass
    else:
        raise AssertionError("the update's failing Webapp build did not fail")
    finally:
        hermes_cli.webapp.prepare_webapp_renderer = real


def _rebuild_webapp(checkout, dist):
    """``hermes webapp``'s build, with the compiler stood in by one that writes a new bundle."""
    import hermes_cli.webapp
    from hermes_cli import source_build

    def compile_bundle(_root, *, env, explicit):
        (dist / "index.html").write_text(INDEX.replace("WEBAPP_BUNDLE", "WEBAPP_BUNDLE rebuilt"), encoding="utf-8")

    source_build.source_product_current = lambda *_args: False
    source_build.source_build_env = lambda *, explicit=False: {}
    source_build.build_source_webapp = compile_bundle
    hermes_cli.webapp.prepare_webapp_renderer(checkout, explicit=True)


async def _native_session_list(runner, origin) -> bool:
    from websockets.asyncio.client import connect

    from gateway.runtime_bootstrap import TicketStore
    from tests.gateway.fixtures.local_recovery_probe import rpc

    authority = runner.session_authority
    runner.session_ticket_store = TicketStore("webapp-fallback", frozenset({authority.profile_id}))
    ticket = runner.session_ticket_store.mint(
        profile_id=authority.profile_id, subject="fixture-same-user", purpose="interactive")
    url = origin.replace("http:", "ws:") + "/api/ws"
    async with connect(url, subprotocols=["hermes-gateway-v1", f"hermes-gateway-ticket.{ticket}"]) as ws:
        return "result" in await rpc(ws, "session.list")


def _mounted_state(home):
    web = sys.modules.get("hermes_cli.web_server")
    if web is None:
        return {"mounted": False}
    state = web.app.state
    stored = _stored_token(home)
    return {
        "mounted": True,
        "surface": getattr(state, "ui_surface", "<absent>"),
        "web_dist_set": hasattr(state, "web_dist"),
        "auth_required": bool(getattr(state, "auth_required", False)),
        "trusted_public_hosts": sorted(getattr(state, "trusted_public_hosts", frozenset()) or ()),
        "session_token_is_browser_credential": bool(stored) and web._SESSION_TOKEN == stored,
    }


def _redirect_uri(host_header: str):
    """The sign-in callback the listener's dashboard routes hand an identity provider."""
    from starlette.requests import Request

    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.routes import _redirect_uri

    scope = {"type": "http", "method": "GET", "path": "/auth/login", "raw_path": b"/auth/login",
             "query_string": b"", "headers": [(b"host", host_header.encode())], "scheme": "http",
             "server": (host_header.split(":")[0], 80), "root_path": "", "app": web_server.app,
             "router": web_server.app.router}
    return _redirect_uri(Request(scope))


async def probe(spec):
    import httpx

    from gateway.control_socket import GatewayControlServer
    from gateway.run import GatewayRunner
    from gateway.run_api import stop_gateway_api
    from gateway.run_runtime import start_gateway_runtime_api
    from gateway.session_authority import initialize_session_authority
    import hermes_cli.webapp

    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(_config(spec), encoding="utf-8")
    # A file where the folder belongs (``credential_unusable``) is the ``folder-is-a-file`` case.
    _seed_credential(home, "folder-is-a-file" if spec.get("credential_unusable") else spec.get("credential"))
    dist = home.parent / "dist-webapp"
    dist.mkdir(exist_ok=True)
    if spec.get("built", True):
        (dist / "index.html").write_text(INDEX, encoding="utf-8")
    hermes_cli.webapp.webapp_dist_dir = lambda _root: dist  # the bundle, without a build
    checkout = home.parent / "checkout"
    if spec.get("update_fails"):
        _fail_webapp_update(home, checkout)
    if spec.get("fail_after_mount"):
        _fail_startup_once()
    rival = None
    port = (spec.get("webapp") or {}).get("port")
    if spec.get("hold_port"):
        rival = socket.socket()
        rival.bind(("127.0.0.1", port))
        rival.listen()
    if spec.get("rival_listens_during_mount") or spec.get("connect_during_mount"):
        # Another process that bound the port with SO_REUSEADDR and starts listening while the
        # gateway mounts the dashboard (Linux lets both bind until one listens); or a client that
        # connects then.
        from gateway.run_api import GatewayDashboardMount

        if spec.get("rival_listens_during_mount"):
            rival = socket.socket()
            rival.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            rival.bind(("127.0.0.1", port))
        real_ensure = GatewayDashboardMount.ensure

        async def ensure(self):
            if rival is not None and "rival_listen" not in report:
                try:
                    rival.listen()
                    report["rival_listen"] = "listening"
                except OSError as exc:
                    report["rival_listen"] = errno.errorcode.get(exc.errno, str(exc.errno))
            if spec.get("connect_during_mount") and "connect_during_mount" not in report:
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=5).close()
                    report["connect_during_mount"] = "connected"
                except OSError as exc:
                    report["connect_during_mount"] = errno.errorcode.get(exc.errno, str(exc))
            return await real_ensure(self)

        GatewayDashboardMount.ensure = ensure

    runner = GatewayRunner()
    runner.session_control_server = GatewayControlServer(home)  # verbs only; not listening
    runner.session_runtime_descriptor = {"runtime_protocol": 1, "state": "ready", "capabilities": [],
                                         "served_profiles": []}
    await initialize_session_authority(runner, profile_id=str(home), instance_id="webapp-fallback")

    def ask(verb):
        raw = json.dumps({"protocol": 1, "verb": verb, "id": 1, "params": {}}).encode()
        answer = json.loads(runner.session_control_server.handle_request_line(raw, "fixture-same-user"))
        return answer.get("result") if answer.get("ok") else {"refused": answer.get("error")}

    report = {"start_error": None}
    try:
        await start_gateway_runtime_api(runner)
    except BaseException as exc:  # what would take the whole gateway down
        report["start_error"] = f"{type(exc).__name__}: {exc}"
        report["start_traceback"] = traceback.format_exc()
        return report
    api = runner.session_api
    origin = api.api_origin
    report.update(api_origin=origin, status=ask("webapp-status"), launch=ask("webapp-launch"))
    try:
        async with httpx.AsyncClient(base_url=origin, timeout=20) as client:
            loopback = await client.get("/api/status")  # mounts the dashboard on the private listener
            report["loopback_status_code"] = loopback.status_code
            index = await client.get("/")
            report["index"] = {"status": index.status_code, "body": index.text[:300]}
            if spec.get("probe_host"):
                foreign = await client.get("/api/status", headers={"Host": spec["probe_host"]})
                report["probe_host_status_code"] = foreign.status_code
                report["redirect_uri_for_probe_host"] = _redirect_uri(spec["probe_host"])
            report["state"] = _mounted_state(home)
            report["native_session_list"] = await _native_session_list(runner, origin)
            if not spec.get("built", True):
                (dist / "index.html").write_text(INDEX, encoding="utf-8")
                report["status_once_built"] = ask("webapp-status")
                built = await client.get("/")
                report["index_once_built"] = {"status": built.status_code,
                                              "is_webapp": "WEBAPP_BUNDLE" in built.text}
            elif spec.get("update_fails"):
                _rebuild_webapp(checkout, dist)
                report["status_once_built"] = ask("webapp-status")
                built = await client.get("/")
                report["index_once_built"] = {"status": built.status_code,
                                              "is_webapp": "WEBAPP_BUNDLE rebuilt" in built.text}
        report["token_rotated"] = _stored_token(home) != SEEDED_TOKEN
    finally:
        await stop_gateway_api(api)
        if rival is not None:
            rival.close()
    return report


def main():
    receipt = Path(os.environ["HERMES_HOME"]) / "receipt.json"
    try:
        receipt.write_text(json.dumps(asyncio.run(probe(json.loads(sys.argv[1])))))
    except BaseException:
        receipt.write_text(json.dumps({"error": traceback.format_exc()}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
