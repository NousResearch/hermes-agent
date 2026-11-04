"""Explicit browser reauthorization preserves existing state until a verified grant."""

import argparse
import json
from pathlib import Path
import stat

import pytest

from hermes_cli import mcp_config as cli
from hermes_constants import get_hermes_home
from tools.mcp_oauth import HermesTokenStorage
from tools.mcp_oauth_manager import get_manager


SERVER = {"url": "https://mcp.example.test/mcp", "auth": "oauth"}


def seed(home, *, access="old-access", client="old-client"):
    directory = home / "mcp-tokens"
    directory.mkdir(parents=True, exist_ok=True)
    values = {
        "service.json": json.dumps({"access_token": access, "refresh_token": "fake-refresh", "token_type": "Bearer"}).encode(),
        "service.client.json": json.dumps({"client_id": client}).encode(),
        "service.meta.json": json.dumps({"issuer": "https://auth.example.test", "token_endpoint": "https://auth.example.test/token"}).encode(),
        "service.cimd-off": b"",
    }
    for name, raw in values.items():
        path = directory / name
        path.write_bytes(raw)
        path.chmod(0o600)
    return values


def state(home):
    return {p.name: p.read_bytes() for p in (home / "mcp-tokens").iterdir() if not p.name.endswith(".refresh.lock")}


@pytest.mark.parametrize("outcome", ["timeout", "cancel", "no-token", "partial-client"])
def test_failed_browser_login_preserves_original_files_and_provider(monkeypatch, capsys, outcome):
    home = get_hermes_home()
    original = seed(home)
    paths = [home / "mcp-tokens" / name for name in original]
    versions = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in paths}
    manager = get_manager()
    entry = object()
    manager._entries[manager._key("service")] = entry
    candidate_home = []

    def probe(name, config, **kwargs):
        candidate_home.append(get_hermes_home())
        directory = get_hermes_home() / "mcp-tokens"
        directory.mkdir(exist_ok=True)
        (directory / "service.client.json").write_text('{"client_id":"partial-client"}')
        if outcome == "cancel":
            raise KeyboardInterrupt
        if outcome in {"timeout", "partial-client"}:
            raise TimeoutError("Synthetic callback timeout")
        return [("public-tool", "Works without authentication")]

    monkeypatch.setattr(cli, "_probe_single_server", probe)
    if outcome == "cancel":
        with pytest.raises(KeyboardInterrupt):
            cli._reauth_oauth_server("service", SERVER)
    else:
        assert cli._reauth_oauth_server("service", SERVER) is False
    assert state(home) == original
    assert {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in paths} == versions
    assert manager._entries[manager._key("service")] is entry
    assert get_hermes_home() == home
    assert candidate_home and candidate_home[0] != home and not candidate_home[0].exists()
    output = capsys.readouterr().out
    assert "Authenticated" not in output
    assert "old-access" not in output and "fake-refresh" not in output


@pytest.mark.parametrize("outcome", ["success", "peer-rotation", "publish-error", "persistent-full", "rollback-error"])
def test_publish_requires_verified_grant_and_unchanged_live_state(monkeypatch, outcome):
    home = get_hermes_home()
    original = seed(home)
    peer = {}
    manager = get_manager()
    cached = object()
    manager._entries[manager._key("service")] = cached
    candidate_home = []

    def probe(name, config, **kwargs):
        candidate_home.append(get_hermes_home())
        candidate = HermesTokenStorage(name)
        assert candidate._tokens_path().exists() is False
        assert candidate._client_info_path().exists() is False
        assert candidate._meta_path().read_bytes() == original["service.meta.json"]
        assert state(home) == original
        assert stat.S_IMODE(get_hermes_home().stat().st_mode) == 0o700
        seed(get_hermes_home(), access="fresh-access", client="fresh-client")
        if outcome == "peer-rotation":
            peer.update(seed(home, access="peer-access", client="peer-client"))
        return [("read", "Read a record")]

    monkeypatch.setattr(cli, "_probe_single_server", probe)
    if outcome in {"publish-error", "persistent-full", "rollback-error"}:
        import utils
        from tools import mcp_oauth_login

        real_write = utils.atomic_write_bytes
        real_replace = mcp_oauth_login.os.replace
        failed = False

        def fail_once(path, content, **kwargs):
            nonlocal failed
            if outcome == "publish-error" and ".previous" in str(path) and not failed:
                failed = True
                raise OSError("Synthetic disk write failure")
            if outcome in {"persistent-full", "rollback-error"} and failed:
                raise OSError("Synthetic persistent ENOSPC")
            return real_write(path, content, **kwargs)

        def failing_replace(source, target):
            nonlocal failed
            if (Path(source).name == "service.json.next"
                    or (outcome == "rollback-error" and Path(source).name == "service.client.json.rollback")):
                failed = True
                raise OSError("Synthetic persistent I/O failure")
            return real_replace(source, target)

        monkeypatch.setattr(utils, "atomic_write_bytes", fail_once)
        monkeypatch.setattr(mcp_oauth_login.os, "replace", failing_replace)
    assert cli._reauth_oauth_server("service", SERVER) is (outcome == "success")
    assert get_hermes_home() == home
    assert not candidate_home[0].exists()
    if outcome == "success":
        assert json.loads((home / "mcp-tokens/service.json").read_bytes())["access_token"] == "fresh-access"
        assert json.loads((home / "mcp-tokens/service.client.json").read_bytes())["client_id"] == "fresh-client"
        assert manager._key("service") not in manager._entries
    elif outcome != "rollback-error":
        assert state(home) == (peer if outcome == "peer-rotation" else original)
        assert manager._entries[manager._key("service")] is cached
    else:
        # A failed rename does not prevent restoring the other state members.
        assert (home / "mcp-tokens/service.json").read_bytes() == original["service.json"]
        assert (home / "mcp-tokens/service.meta.json").read_bytes() == original["service.meta.json"]
        recovery = list((home / "mcp-tokens").glob(".oauth-login-*"))
        assert len(recovery) == 1
        assert (recovery[0] / "service.client.json.previous").read_bytes() == original["service.client.json"]
        assert stat.S_IMODE(recovery[0].stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in (home / "mcp-tokens").iterdir() if p.is_file())


@pytest.mark.parametrize("action", ["login", "reauth"])
@pytest.mark.parametrize("known,success,expected", [(True, True, 0), (True, False, 1), (False, False, 3)])
def test_cli_dispatch_propagates_authentication_result(monkeypatch, action, known, success, expected):
    from hermes_cli.main import cmd_mcp

    monkeypatch.setattr(cli, "_get_mcp_servers", lambda: {"service": SERVER} if known else {})
    monkeypatch.setattr(cli, "_reauth_oauth_server", lambda *args, **kwargs: success)
    args = argparse.Namespace(mcp_action=action, name="service", all=False, flow=None)
    assert cmd_mcp(args) == expected


def test_reauth_all_reports_partial_failure_with_nonzero_exit(monkeypatch):
    monkeypatch.setattr(cli, "_get_mcp_servers", lambda: {"a": SERVER, "b": SERVER})
    monkeypatch.setattr(cli, "_reauth_oauth_server", lambda name, config: name == "a")
    assert cli.cmd_mcp_reauth(argparse.Namespace(name=None, all=True)) == 1


def test_real_probe_loop_keeps_profile_scope_and_auth_state_isolated(monkeypatch):
    from tools import mcp_tool_discovery

    home = get_hermes_home()
    original = seed(home)

    class Server:
        _tools = []

        async def shutdown(self):
            assert get_hermes_home() != home

    async def connect(name, config):
        candidate_home = get_hermes_home()
        assert candidate_home != home
        assert state(home) == original
        assert HermesTokenStorage(name)._tokens_path().exists() is False
        seed(candidate_home, access="loop-access", client="loop-client")
        return Server()

    monkeypatch.setattr(mcp_tool_discovery, "_connect_server", connect)
    assert cli._reauth_oauth_server("service", SERVER) is True
    assert get_hermes_home() == home
    assert json.loads((home / "mcp-tokens/service.json").read_bytes())["access_token"] == "loop-access"


def test_browser_probe_sends_original_profile_identity(monkeypatch, tmp_path):
    import httpx2
    from hermes_cli.profiles import get_active_profile_name
    from tools import mcp_tool_discovery
    from tools.mcp_tool_transport import _connect_inputs

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes/profiles/registered-profile"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert get_active_profile_name() == "registered-profile"
    seed(home)
    requests = []

    class Server:
        _tools = []

        async def shutdown(self):
            pass

    async def connect(name, config):
        assert get_active_profile_name() == "custom"
        (url, headers), _ = _connect_inputs(name, config)
        transport = httpx2.MockTransport(lambda request: requests.append(request) or httpx2.Response(200))
        async with httpx2.AsyncClient(transport=transport) as client:
            await client.get(url, headers=headers)
        seed(get_hermes_home(), access="profile-access")
        return Server()

    monkeypatch.setattr(mcp_tool_discovery, "_connect_server", connect)
    config = {**SERVER, "identity_header": {"name": "X-Profile", "value_from": "profile"}}
    assert cli._reauth_oauth_server("service", config) is True
    assert [request.headers["X-Profile"] for request in requests] == ["registered-profile"]
