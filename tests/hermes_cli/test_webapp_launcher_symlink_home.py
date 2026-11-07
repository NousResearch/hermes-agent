"""Regression coverage for symlinked Hermes homes in the Webapp control client (#149810)."""

import pytest

from hermes_cli import gateway_client
from hermes_cli import webapp_launcher as launcher


@pytest.mark.platforms("posix")
def test_control_resolves_a_symlinked_home_before_socket_validation(tmp_path, monkeypatch):
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    seen = []

    def query(home, request):
        seen.append((home, request))
        return b'{"ok": true}\n'

    monkeypatch.setattr(gateway_client, "private_control_query", query)

    assert launcher._control(linked_home, "status") == {"ok": True}
    assert seen == [(
        real_home.resolve(),
        b'{"protocol": 1, "id": 1, "verb": "status", "params": {}}\n',
    )]
