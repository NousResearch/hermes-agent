"""The Windows update pause owns only workers from the effective Hermes home."""

from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gateway_cli
from hermes_cli import update_cmd_windows


pytestmark = pytest.mark.platforms("windows")


def _discovery(monkeypatch, *, home, workers, profiles=(), services=()):
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_cli, "find_gateway_pids", lambda **_kw: list(workers))
    monkeypatch.setattr(gateway_cli, "find_profile_gateway_processes", lambda **_kw: list(profiles))
    monkeypatch.setattr(gateway_cli, "find_windows_gateway_services", lambda **_kw: list(services))


def test_foreign_gateway_is_excluded_before_any_pause(monkeypatch, tmp_path):
    home = tmp_path / "isolated"
    foreign = tmp_path / "live"
    home.mkdir()
    foreign.mkdir()
    _discovery(monkeypatch, home=home, workers=[202])
    monkeypatch.setattr(update_cmd_windows, "_gateway_process_home", lambda _pid: foreign)
    monkeypatch.setattr(update_cmd_windows, "_windows_cold_start_plan", lambda: None)
    monkeypatch.setattr(update_cmd_windows, "_record_attested_cold_start_profiles", lambda *_a: None)
    monkeypatch.setattr(
        update_cmd_windows, "_request_socket_pauses",
        lambda *_a: pytest.fail("a foreign gateway reached the pause"),
    )

    assert update_cmd_windows._discover_windows_gateways() == ({}, [], set(), [])
    assert update_cmd_windows._pause_windows_gateways_for_update() is None


def test_custom_checkout_keeps_its_own_profile_and_excludes_foreign(monkeypatch, tmp_path):
    home = tmp_path / "isolated"
    foreign = tmp_path / "live"
    home.mkdir()
    foreign.mkdir()
    own = SimpleNamespace(pid=101, path=home, profile="default")
    _discovery(monkeypatch, home=home, workers=[101, 202], profiles=[own])
    monkeypatch.setattr(
        update_cmd_windows, "_gateway_process_home",
        lambda pid: home if pid == 101 else foreign,
    )

    assert update_cmd_windows._discover_windows_gateways() == ({101: own}, [], set(), [101])


def test_unmapped_worker_in_target_home_aborts_before_pause(monkeypatch, tmp_path):
    home = tmp_path / "isolated"
    home.mkdir()
    _discovery(monkeypatch, home=home, workers=[101])
    monkeypatch.setattr(update_cmd_windows, "_gateway_process_home", lambda _pid: home)

    with pytest.raises(RuntimeError, match="without a verified profile or service owner"):
        update_cmd_windows._discover_windows_gateways()


def test_pid_file_cannot_claim_a_foreign_worker(monkeypatch, tmp_path):
    home = tmp_path / "isolated"
    foreign = tmp_path / "live"
    home.mkdir()
    foreign.mkdir()
    claimed = SimpleNamespace(pid=101, path=home, profile="default")
    _discovery(monkeypatch, home=home, workers=[101], profiles=[claimed])
    monkeypatch.setattr(update_cmd_windows, "_gateway_process_home", lambda _pid: foreign)

    with pytest.raises(RuntimeError, match="disagrees with its profile PID file"):
        update_cmd_windows._discover_windows_gateways()


def test_service_worker_must_match_its_profile_home(monkeypatch, tmp_path):
    home = tmp_path / "isolated"
    foreign = tmp_path / "live"
    home.mkdir()
    foreign.mkdir()
    service = SimpleNamespace(gateway_pid=101, profile="default")
    _discovery(monkeypatch, home=home, workers=[101], services=[service])
    monkeypatch.setattr(update_cmd_windows, "_gateway_process_home", lambda _pid: home)

    assert update_cmd_windows._discover_windows_gateways() == ({}, [service], {101}, [101])
    monkeypatch.setattr(update_cmd_windows, "_gateway_process_home", lambda _pid: foreign)
    with pytest.raises(RuntimeError, match="disagrees with its Windows service profile"):
        update_cmd_windows._discover_windows_gateways()


def test_unreadable_process_home_aborts_discovery(monkeypatch, tmp_path):
    home = tmp_path / "isolated"
    home.mkdir()
    _discovery(monkeypatch, home=home, workers=[101])
    monkeypatch.setattr(
        update_cmd_windows, "_gateway_process_home",
        lambda _pid: (_ for _ in ()).throw(OSError("access denied")),
    )

    with pytest.raises(RuntimeError, match="Could not establish Windows gateway home"):
        update_cmd_windows._discover_windows_gateways()
