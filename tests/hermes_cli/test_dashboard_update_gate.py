"""Which dashboards may run ``hermes update`` themselves."""

import pytest

import hermes_cli.config as _cfg_mod
import hermes_constants
from hermes_cli import web_server_files


@pytest.mark.parametrize(
    ("opt_data_root", "in_container", "install_method", "managed"),
    [
        # Hosted image: /opt/data inside a container stays managed, even for git.
        (True, True, "git", True),
        (True, True, "docker", True),
        # Native install that reuses /opt/data (e.g. a VM migrated from Docker).
        (True, False, "git", False),
        # Plain host install.
        (False, False, "git", False),
        # Containers: only a bind-mounted git checkout updates itself.
        (False, True, "git", False),
        (False, True, "pip", True),
        (False, True, "docker", True),
    ],
)
def test_update_gate(monkeypatch, opt_data_root, in_container, install_method, managed):
    monkeypatch.setattr(web_server_files, "_default_hermes_root_is_opt_data", lambda: opt_data_root)
    monkeypatch.setattr(hermes_constants, "is_container", lambda: in_container)
    monkeypatch.setattr(_cfg_mod, "detect_install_method", lambda _root: install_method)

    assert web_server_files._dashboard_local_update_managed_externally() is managed


@pytest.mark.parametrize(("opt_data_root", "managed"), [(True, True), (False, False)])
def test_update_gate_keeps_hosted_posture_when_container_probe_fails(monkeypatch, opt_data_root, managed):
    def _probe_fails():
        raise OSError("probe failed")

    monkeypatch.setattr(web_server_files, "_default_hermes_root_is_opt_data", lambda: opt_data_root)
    monkeypatch.setattr(hermes_constants, "is_container", _probe_fails)

    assert web_server_files._dashboard_local_update_managed_externally() is managed
