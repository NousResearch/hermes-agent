"""Host topology: the DEFAULT profile is a served profile, not a special owner.

``multiplexer_liveness_for_profile`` returned None for the default home by construction, so the
one profile the host gateway always serves could never be *reported* as served — the ladder had to
fall back to PID files that a multiplexed default may not own (a launch-service gateway, a
re-exec'd Desktop backend).
"""

import pytest


@pytest.fixture
def host_gateway(tmp_path, monkeypatch):
    from gateway import host_rendezvous as hr

    root = tmp_path / "hermes"
    (root / "profiles" / "coder").mkdir(parents=True)
    locks = tmp_path / "locks"
    locks.mkdir()
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(locks))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    hr.publish_record(hr.ROLE_GATEWAY, profiles=("default", "coder"))
    return root


def test_default_home_is_reported_as_served_by_the_host_gateway(host_gateway):
    import os

    from gateway import status

    for home in (host_gateway, host_gateway / "profiles" / "coder"):
        resolved = status.multiplexer_liveness_for_profile(home)
        assert resolved is not None, f"{home} is served by the host gateway but reported unserved"
        assert resolved[0] == os.getpid()


def test_unserved_profile_is_not_claimed_by_the_host_gateway(host_gateway, monkeypatch):
    from gateway import status

    monkeypatch.setattr("hermes_cli.gateway.named_profile_served_by_running_multiplexer", lambda *a: False)
    (host_gateway / "profiles" / "other").mkdir()
    assert status.multiplexer_liveness_for_profile(host_gateway / "profiles" / "other") is None


def test_unprovable_record_is_a_candidate_not_the_host_gateway(host_gateway, monkeypatch):
    """A leftover record whose (pid, createTime) cannot be POSITIVELY matched must never be
    reported as the live host gateway — otherwise it is a permanent "running" lie."""
    from gateway import host_topology

    monkeypatch.setattr("gateway.host_rendezvous.liveness_is_proven", lambda record: False)
    monkeypatch.setattr("hermes_cli.gateway_multiplex_served.live_default_gateway_pid", lambda: None)
    assert host_topology.host_gateway_topology() is None


def test_record_without_createtime_is_never_the_host_gateway(host_gateway, monkeypatch):
    """``createTime: null`` is the UNPROVABLE case, not the always-live one.

    ``_same_incarnation`` treats a missing create_time as "matches", so liveness_is_proven() says
    True for whatever process happens to own that PID today — a record pointing at an unrelated
    `sleep 60` made every surface report a live host gateway."""
    import json

    from gateway import host_rendezvous as hr
    from gateway import host_topology

    monkeypatch.setattr("hermes_cli.gateway_multiplex_served.live_default_gateway_pid", lambda: None)
    path = hr.record_path(hr.ROLE_GATEWAY)
    record = json.loads(path.read_text())
    record["createTime"] = None
    path.write_text(json.dumps(record))

    parsed = hr.read_record(hr.ROLE_GATEWAY)
    assert parsed is not None and parsed.create_time is None  # the record itself still parses
    assert host_topology.host_gateway_topology() is None


def test_topology_ignores_another_tenants_record_and_reads_the_launch_homes_state(tmp_path, monkeypatch):
    """#121352 reporting half: a record from ANOTHER Hermes root is not this tenant's host gateway
    (doctor / cron status / the dashboard ladder), and a NAMED-hosted multiplexer's platforms are read
    from the home that launched it, not from a stale standalone record at the default root."""
    import json
    import os

    from gateway import host_rendezvous as hr
    from gateway import host_topology, status

    root_a, root_b = tmp_path / "hermes-a", tmp_path / "hermes-b"
    argus = root_b / "profiles" / "argus"
    argus.mkdir(parents=True)
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv("HERMES_HOME", str(root_b))
    monkeypatch.setattr("hermes_constants._default_hermes_root_memo", None)

    hr.publish_record(hr.ROLE_GATEWAY, profiles=("default", "coder"), home=str(root_a))
    assert host_topology.host_gateway_topology() is None

    hr.publish_record(hr.ROLE_GATEWAY, profiles=("default", "argus"), home=str(argus))
    (argus / "gateway_state.json").write_text(json.dumps(
        {"pid": os.getpid(), "gateway_state": "running", "platforms": {"telegram": {"state": "connected"}}}),
        encoding="utf-8")
    (root_b / "gateway_state.json").write_text(json.dumps(
        {"pid": 1, "gateway_state": "stopped", "platforms": {"discord": {"state": "disconnected"}}}),
        encoding="utf-8")
    topology = host_topology.host_gateway_topology()
    assert topology is not None and topology.home == argus
    resolved = status.multiplexer_liveness_for_profile(root_b)
    assert resolved is not None and resolved[0] == os.getpid()
    assert set(resolved[1]["platforms"]) == {"telegram"}


def test_named_standalone_record_falls_through_to_served_record(host_gateway, monkeypatch):
    """A named standalone owner must not be reported as the host multiplexer."""
    import os

    from gateway import host_rendezvous as hr
    from gateway import host_topology

    hr.publish_record(hr.ROLE_GATEWAY, profiles=("coder",))
    monkeypatch.setattr(
        "hermes_cli.gateway_multiplex_served.live_default_gateway_pid", lambda: os.getpid()
    )
    monkeypatch.setattr(
        "hermes_cli.gateway_multiplex_served.recorded_served_profiles",
        lambda: ["coder"],
    )

    topology = host_topology.host_gateway_topology()
    assert topology is not None
    assert topology.source == "served_record"
    assert topology.profiles == ("default", "coder")
