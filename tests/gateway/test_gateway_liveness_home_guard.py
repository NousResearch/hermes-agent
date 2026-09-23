"""A profile directory copied out of another HERMES_HOME must never read as running.

Copying a profile directory wholesale (sandbox injection, restore-from-backup, cloning) carries
that home's ``gateway_state.json`` along, and a copy with NO identity files at all still keeps
the profile *name*. The liveness ladder must neither believe a record stamped with another
home nor borrow the process home's record / the multiplexer's roster for it.
"""

import json

import pytest

from gateway import status
import hermes_constants

_LIVE_PID = 4242


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(hermes_constants.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None, raising=False)
    return root


def _live_record(hermes_home):
    return {
        "pid": _LIVE_PID, "kind": "hermes-gateway", "gateway_state": "running", "start_time": 1000,
        "argv": ["python", "-m", "hermes_cli.main", "-p", "eagle", "gateway", "run"],
        "hermes_home": str(hermes_home),
    }


def _alive(monkeypatch, cmdline="python -m hermes_cli.main -p eagle gateway run"):
    monkeypatch.setattr(status, "_pid_exists", lambda pid: pid == _LIVE_PID)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 1000)
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: cmdline)


def test_a_runtime_record_stamped_for_another_home_is_not_this_homes_gateway(fake_root, tmp_path, monkeypatch):
    """Rung 3: ``gateway_state.json`` copied along with the directory names the ORIGINAL home."""
    _alive(monkeypatch)
    copied = tmp_path / "sandbox-root" / "profiles" / "eagle"
    copied.mkdir(parents=True)
    foreign = _live_record(fake_root / "profiles" / "eagle")
    (copied / "gateway_state.json").write_text(json.dumps(foreign), encoding="utf-8")
    monkeypatch.setattr(status, "multiplexer_liveness_for_profile", lambda *a, **k: None)

    assert status.get_runtime_status_running_pid(foreign, expected_home=copied) is None
    assert status.resolve_gateway_liveness(profile_dir=copied, use_cache=False).running is False
    # Control: the same record asked about from the home it names is believed.
    assert status.get_runtime_status_running_pid(foreign, expected_home=fake_root / "profiles" / "eagle") == _LIVE_PID


def test_a_copied_dir_with_no_identity_files_borrows_neither_the_process_record_nor_the_roster(
    fake_root, tmp_path, monkeypatch
):
    """The process home's ``gateway_state.json`` names a live ``-p eagle`` gateway and the multiplexer
    roster lists ``eagle``; a same-named directory under another root is served by neither."""
    _alive(monkeypatch)
    (fake_root / "gateway_state.json").write_text(json.dumps(_live_record(fake_root)), encoding="utf-8")
    copied = tmp_path / "sandbox-root" / "profiles" / "eagle"
    copied.mkdir(parents=True)
    from gateway import host_topology
    monkeypatch.setattr(
        host_topology, "host_gateway_topology",
        lambda: host_topology.HostGatewayTopology(pid=_LIVE_PID, profiles=("default", "eagle"), source="test"),
    )

    assert status.resolve_gateway_liveness(profile_dir=copied, use_cache=False).running is False
    assert status.multiplexer_liveness_for_profile(copied) is None
    # Control: the pooled ``<root>/profiles/eagle`` IS the served home.
    pooled = fake_root / "profiles" / "eagle"
    pooled.mkdir(parents=True)
    assert status.multiplexer_liveness_for_profile(pooled) is not None
