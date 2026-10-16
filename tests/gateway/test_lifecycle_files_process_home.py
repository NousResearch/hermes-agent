"""The gateway's process-level lifecycle files stay in the launch home under a profile override.

A multiplexed gateway runs each served profile's work inside that profile's home override
(``set_hermes_home_override``). The lifecycle sentinel (``gateway.lifecycle.json``) and the
loop heartbeat are identities of the gateway PROCESS: the next boot and the supervisor read them
from the launch home. Both modules used to fall back to ``get_hermes_home()`` when
``HERMES_HOME`` is unset (a default gateway started in the foreground), and that resolver follows
the override, so a write made from a served profile's context landed in ``profiles/<name>/``.
Same class as the delivery-ledger fix (``test_delivery_ledger_process_home.py``).
"""

from __future__ import annotations

import pytest

import hermes_constants
from gateway import lifecycle_ledger, shutdown_watchdog


@pytest.fixture(params=["hermes-home-env", "platform-default"])
def launch_home(request, tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "profiles" / "research").mkdir(parents=True)
    if request.param == "hermes-home-env":
        monkeypatch.setenv("HERMES_HOME", str(root))
    else:
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: root)
    return root


def _in_profile_context(root, write):
    token = hermes_constants.set_hermes_home_override(str(root / "profiles" / "research"))
    try:
        return write()
    finally:
        hermes_constants.reset_hermes_home_override(token)


@pytest.mark.parametrize("write, relative", [
    (lambda: lifecycle_ledger.record_startup(), ("state", "gateway.lifecycle.json")),
    (lambda: shutdown_watchdog.write_loop_heartbeat(), shutdown_watchdog._HEARTBEAT_RELATIVE),
], ids=["lifecycle-sentinel", "loop-heartbeat"])
def test_process_lifecycle_file_lands_in_the_launch_home(launch_home, write, relative):
    _in_profile_context(launch_home, write)

    assert launch_home.joinpath(*relative).is_file()
    assert not (launch_home / "profiles" / "research").joinpath(*relative).exists()
