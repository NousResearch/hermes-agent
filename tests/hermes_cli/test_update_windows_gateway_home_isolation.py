"""A checkout outside the effective Hermes home must not pause gateways."""

import pytest

from hermes_cli import main as cli_main
from hermes_cli import update_cmd_windows


@pytest.mark.windows_only
def test_explicit_home_outside_checkout_skips_host_wide_gateway_discovery(monkeypatch, tmp_path):
    home = tmp_path / "isolated-home"
    checkout = tmp_path / "isolated-checkout"
    home.mkdir()
    checkout.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", checkout)

    def must_not_discover():
        pytest.fail("host-wide gateway discovery crossed the isolated home boundary")

    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", must_not_discover)
    with pytest.raises(RuntimeError, match="outside the managed Hermes checkout"):
        update_cmd_windows._pause_windows_gateways_for_update()


@pytest.mark.windows_only
def test_default_home_outside_checkout_skips_host_wide_gateway_discovery(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path / "isolated-checkout")

    def must_not_discover():
        pytest.fail("host-wide gateway discovery crossed the default home boundary")

    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", must_not_discover)
    with pytest.raises(RuntimeError, match="outside the managed Hermes checkout"):
        update_cmd_windows._pause_windows_gateways_for_update()


@pytest.mark.windows_only
def test_unverified_discovery_result_cannot_reach_gateway_pause(monkeypatch, tmp_path):
    home = tmp_path / "managed-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", home / "hermes-agent")
    monkeypatch.setattr(
        update_cmd_windows, "_discover_windows_gateways",
        lambda: ({}, [], set(), [202]),
    )
    with pytest.raises(RuntimeError, match="unverified process owner"):
        update_cmd_windows._pause_windows_gateways_for_update()


@pytest.mark.windows_only
def test_home_switch_a_to_b_to_a_rechecks_checkout_before_discovery(monkeypatch, tmp_path):
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", home_b / "hermes-agent")
    discovered = []

    def record_discovery():
        discovered.append(True)
        raise LookupError("matched home reached discovery")

    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", record_discovery)

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    with pytest.raises(RuntimeError, match="outside the managed Hermes checkout"):
        update_cmd_windows._pause_windows_gateways_for_update()

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    with pytest.raises(LookupError, match="matched home reached discovery"):
        update_cmd_windows._pause_windows_gateways_for_update()

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    with pytest.raises(RuntimeError, match="outside the managed Hermes checkout"):
        update_cmd_windows._pause_windows_gateways_for_update()

    assert discovered == [True]
