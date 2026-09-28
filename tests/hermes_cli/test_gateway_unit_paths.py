"""User systemd unit location and identity: the account home owns the unit dir (#98699) and a bare
``hermes-gateway.service`` pinning THIS home is this home's service (#109476)."""

import pytest

import hermes_cli.gateway as gateway

pytestmark = pytest.mark.platforms("linux")


def test_user_unit_dir_follows_the_account_home_not_a_profile_pinned_process_home(tmp_path, monkeypatch):
    # ``hermes -p x gateway install`` launched from a process whose HOME profile isolation already
    # pointed at the ACTIVE profile's ``{HERMES_HOME}/home``; the unit must land where
    # ``systemctl --user`` looks — under the account home — not under that profile dir.
    account_home = tmp_path / "account"
    active_root = tmp_path / "active-profile-root"
    process_home = active_root / "home"
    process_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(active_root))
    monkeypatch.setenv("HOME", str(process_home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(account_home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    unit_path = gateway.get_systemd_unit_path(system=False)

    assert unit_path.parent == account_home / ".config" / "systemd" / "user"
    assert not unit_path.is_relative_to(process_home)

