"""Own gateway proof takes precedence over explicit, same-tenant host service."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway import host_rendezvous
from gateway.host_attach import HostGateway
from hermes_cli import gateway as gw, gateway_deferred_restart as cli, gateway_runtime_discovery as discovery


def identity(home, **changes):
    return {"protocol": 1, "kind": "hermes-gateway", "pid": 42, "hermes_home": str(home),
            "supervisor": "external", "state": "ready", "served_profiles": ["default", "coder"], **changes}


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / "tenant"
    home = root / "profiles" / "coder"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("gateway: {}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gw, "_current_profile_name", lambda: "coder")
    host = HostGateway(42, root, ("default", "coder"))
    monkeypatch.setattr(gw, "host_multiplexer_serving", Mock(return_value=host))
    monkeypatch.setattr(host_rendezvous, "read_record", lambda role: SimpleNamespace(home=str(root), pid=42))
    return home, root


def test_own_identity_wins(homes, monkeypatch):
    home, _root = homes
    monkeypatch.setattr(discovery, "query_identify", Mock(return_value=identity(home)))
    target, info, satellite = cli.resolve_target()
    assert target == home.resolve() and info["pid"] == 42 and satellite is False
    gw.host_multiplexer_serving.assert_not_called()


def satellite_probe(home, root):
    def query(target, **kwargs):
        if target == home:
            raise FileNotFoundError
        assert target == root
        return identity(root)
    return query


def test_confirmed_satellite(homes, monkeypatch):
    home, root = homes
    query = Mock(side_effect=satellite_probe(home, root))
    monkeypatch.setattr(discovery, "query_identify", query)
    target, _, satellite = cli.resolve_target()
    assert target == root and satellite
    assert [c.args[0] for c in query.call_args_list] == [home, root]


@pytest.mark.parametrize("failure", [TimeoutError, PermissionError, ValueError])
def test_unreachable_identity_never_tries_host(homes, monkeypatch, failure):
    monkeypatch.setattr(discovery, "query_identify", Mock(side_effect=failure))
    with pytest.raises((OSError, ValueError)):
        cli.resolve_target()
    gw.host_multiplexer_serving.assert_not_called()


@pytest.mark.parametrize("artifact", ["gateway.pid", "gateway.lock", "gateway.sock", "gateway.sock.path"])
def test_stale_or_malformed_own_records_are_not_absence(homes, monkeypatch, artifact):
    home, _ = homes
    (home / artifact).write_text("malformed")
    monkeypatch.setattr(discovery, "query_identify", Mock(side_effect=FileNotFoundError))
    with pytest.raises(ValueError):
        cli.resolve_target()
    gw.host_multiplexer_serving.assert_not_called()


@pytest.mark.parametrize("state", ["starting", "inaccessible"])
def test_unconfirmed_owner_does_not_route_to_host(homes, monkeypatch, state):
    monkeypatch.setattr(discovery, "query_identify", Mock(side_effect=FileNotFoundError))
    monkeypatch.setattr(discovery, "missing_owner_state", lambda home: state)
    with pytest.raises(ValueError):
        cli.resolve_target()
    gw.host_multiplexer_serving.assert_not_called()


@pytest.mark.parametrize("host_case", ["none", "unknown", "unserved", "cross_tenant", "legacy", "record_mismatch"])
def test_ambiguous_host_refused(homes, monkeypatch, host_case):
    home, root = homes
    monkeypatch.setattr(discovery, "query_identify", satellite_probe(home, root))
    hosts = {"none": None, "unknown": HostGateway(42, root, (), served_known=False),
             "unserved": HostGateway(42, root, ("default",)),
             "cross_tenant": HostGateway(42, root.parent / "foreign", ("coder",)),
             "legacy": HostGateway(42, root, ("coder",)),
             "record_mismatch": HostGateway(99, root, ("coder",))}
    monkeypatch.setattr(gw, "host_multiplexer_serving", lambda profile: hosts[host_case])
    if host_case == "legacy":
        monkeypatch.setattr(host_rendezvous, "read_record", lambda role: SimpleNamespace(home="", pid=42))
    if host_case == "cross_tenant":
        monkeypatch.setattr(host_rendezvous, "read_record", lambda role: SimpleNamespace(home=str(root.parent / "foreign"), pid=42))
    with pytest.raises(ValueError):
        cli.resolve_target()


def test_standalone_exclusion(homes, monkeypatch):
    home, root = homes
    (home / "config.yaml").write_text("gateway:\n  standalone: true\n")
    monkeypatch.setattr(discovery, "query_identify", satellite_probe(home, root))
    with pytest.raises(ValueError, match="standalone"):
        cli.resolve_target()
    gw.host_multiplexer_serving.assert_not_called()


@pytest.mark.parametrize("changes", [{"hermes_home": "/other"}, {"pid": None}, {"pid": True}, {"state": "starting"}])
def test_identity_mismatch_refused(homes, monkeypatch, changes):
    home, _ = homes
    monkeypatch.setattr(discovery, "query_identify", lambda *a, **kw: identity(home, **changes))
    with pytest.raises(ValueError):
        cli.resolve_target()


def test_canonical_identity(homes, monkeypatch, tmp_path):
    home, _ = homes
    alias = tmp_path / "alias"
    alias.symlink_to(home, target_is_directory=True)
    monkeypatch.setattr(discovery, "query_identify", lambda *a, **kw: identity(alias))
    assert cli.resolve_target()[0] == home.resolve()


@pytest.mark.parametrize("pid", [42, 99])
def test_held_reservation_requires_identified_host_owner(homes, monkeypatch, pid):
    from gateway import status
    home, root = homes
    monkeypatch.setattr(discovery, "query_identify", satellite_probe(home, root))
    monkeypatch.setattr(discovery, "missing_owner_state", lambda home: "starting")
    monkeypatch.setattr(status, "get_running_pid_identity_strict", lambda path: (pid, 123.0))
    if pid == 42:
        assert cli.resolve_target()[0] == root
    else:
        with pytest.raises(ValueError, match="ambiguous"):
            cli.resolve_target()


@pytest.mark.parametrize("case", ["host_unreachable", "service_changed", "resolver_failed"])
def test_host_resolution_failure(homes, monkeypatch, case):
    home, root = homes
    def query(target, **kwargs):
        if target == home:
            raise FileNotFoundError
        if case == "host_unreachable":
            raise TimeoutError
        return identity(root, served_profiles=["default"])
    monkeypatch.setattr(discovery, "query_identify", query)
    if case == "resolver_failed":
        monkeypatch.setattr(gw, "host_multiplexer_serving", Mock(side_effect=RuntimeError))
    with pytest.raises((OSError, ValueError, RuntimeError)):
        cli.resolve_target()
