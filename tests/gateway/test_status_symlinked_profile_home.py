"""A profile home that is a symlink: its gateway runs under the resolved home (#160648)."""

import pytest

from gateway import status


@pytest.fixture
def homes(tmp_path, monkeypatch):
    real = tmp_path / "estate" / "orchestrator" / "state"
    real.mkdir(parents=True)
    link = tmp_path / "root" / "profiles" / "orchestrator"
    link.parent.mkdir(parents=True)
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 1000)
    return link, real.resolve()


def _record(home=None):
    record = {"pid": 139, "gateway_state": "running", "kind": "hermes-gateway",
              "argv": ["hermes", "gateway", "run"], "start_time": 1000}
    return record if home is None else {**record, "hermes_home": str(home)}


def test_gateway_launched_under_the_resolved_home_is_found_through_the_symlink(homes, monkeypatch):
    link, real = homes
    # The unit carries HERMES_HOME in its environment (bare argv) or spells the resolved path.
    for cmdline in ("hermes gateway run --replace", f"HERMES_HOME={real} hermes gateway run"):
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid, c=cmdline: c)
        assert status.get_runtime_status_running_pid(_record(real), expected_home=link) == 139, cmdline
        assert status.get_runtime_status_running_pid(_record(real), expected_home=real) == 139, cmdline


def test_symlinked_home_does_not_borrow_another_gateway(homes, monkeypatch):
    link, real = homes
    for cmdline in ("hermes -p coder gateway run", "HERMES_HOME=/opt/other hermes gateway run"):
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid, c=cmdline: c)
        assert status.get_runtime_status_running_pid(_record(real), expected_home=link) is None, cmdline
    # A record that does not name its home proves nothing: the PID may have been recycled onto the
    # bare default gateway, which must not count as this profile's.
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "hermes gateway run --replace")
    assert status.get_runtime_status_running_pid(_record(), expected_home=link) is None
