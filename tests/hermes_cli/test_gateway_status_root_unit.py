"""A system unit without a User= line must not crash read-only status/refresh checks.

Older ``install --system --run-as-user root`` wrote a unit with no ``User=`` directive (systemd
then runs the service as root by default). ``systemd_unit_is_current`` used to rebuild the
expected unit with ``run_as_user=None``, which resolves the *current* user and hits the
install-time root guard — crashing ``hermes gateway status`` on root deployments (#138226).
"""

from unittest.mock import Mock

import hermes_cli.gateway as gateway_cli

ROOT_GUARD_MESSAGE = (
    "Refusing to install the gateway system service as root; "
    "pass --run-as-user root to override (e.g. in LXC containers)"
)


def _install_userless_root_unit(tmp_path, monkeypatch) -> Mock:
    """Write a User=-less system unit and stub identity like a root host: only an explicit
    ``run_as_user`` satisfies the guard, ``None`` raises exactly as on a root deployment."""
    hermes_home = tmp_path / "root" / ".hermes"
    hermes_home.mkdir(parents=True)
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text(
        "[Unit]\n"
        "Description=Hermes Gateway\n"
        "\n"
        "[Service]\n"
        'Environment="HERMES_HOME=/root/.hermes"\n'
        "ExecStart=/usr/local/bin/hermes-gateway run\n"
        "Restart=always\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", "/root/.hermes")
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)

    def identity_like_root_host(run_as_user=None):
        if not run_as_user:
            raise ValueError(ROOT_GUARD_MESSAGE)
        return ("root", "root", "/root", 0)

    identity = Mock(side_effect=identity_like_root_host)
    monkeypatch.setattr(gateway_cli, "_system_service_identity", identity)
    return identity


def test_userless_system_unit_does_not_crash_the_drift_check(tmp_path, monkeypatch):
    identity = _install_userless_root_unit(tmp_path, monkeypatch)

    # Read-only check must decline to judge drift rather than raise the install guard.
    assert gateway_cli.systemd_unit_is_current(system=True) is True
    identity.assert_not_called()


def test_refresh_leaves_a_userless_system_unit_untouched(tmp_path, monkeypatch):
    identity = _install_userless_root_unit(tmp_path, monkeypatch)
    unit_path = gateway_cli.get_systemd_unit_path(system=True)
    before = unit_path.read_text(encoding="utf-8")
    rewrite = Mock()
    monkeypatch.setattr(gateway_cli, "generate_systemd_unit", rewrite)
    monkeypatch.setattr(gateway_cli, "_prepare_service_launcher", rewrite)
    monkeypatch.setattr(gateway_cli, "_run_systemctl", lambda *a, **k: None)

    assert gateway_cli.refresh_systemd_unit_if_needed(system=True) is False
    assert unit_path.read_text(encoding="utf-8") == before
    identity.assert_not_called()


def test_unit_with_user_line_still_gets_compared(tmp_path, monkeypatch):
    """The skip only applies to User=-less units; a normal unit keeps the drift check."""
    hermes_home = tmp_path / "alice" / ".hermes"
    hermes_home.mkdir(parents=True)
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text(
        "[Unit]\n"
        "Description=Hermes Gateway\n"
        "\n"
        "[Service]\n"
        "User=alice\n"
        f'Environment="HERMES_HOME={hermes_home}"\n'
        "ExecStart=/usr/local/bin/hermes-gateway run\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
    monkeypatch.setattr(
        gateway_cli, "_system_service_identity",
        lambda run_as_user=None: ("alice", "alice", str(tmp_path / "alice"), 1001),
    )
    monkeypatch.setattr(gateway_cli, "_build_user_local_paths", lambda home, existing: [])
    monkeypatch.setattr(gateway_cli.shutil, "which", lambda cmd: None)

    # Hand-built unit differs from the generated one, so drift is still reported.
    assert gateway_cli.systemd_unit_is_current(system=True) is False
