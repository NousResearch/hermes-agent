"""Embedded-daemon stop gate for externally-supervised setups (issue #82944).

Without a check, a config change makes the plugin call ``client._manager.stop()``
on any healthy daemon it finds on the port — including one an external
supervisor (e.g. systemd) started and owns. Neither ``is_running()`` nor
``stop()`` in the upstream manager can tell the two cases apart, so the plugin
has to gate the call itself using an explicit ``externally_managed`` config flag.
"""

import importlib

hindsight = importlib.import_module("plugins.memory.hindsight")
_daemon_stop_allowed = hindsight._daemon_stop_allowed


def test_stop_allowed_by_default():
    assert _daemon_stop_allowed({}) is True


def test_stop_allowed_when_not_externally_managed():
    assert _daemon_stop_allowed({"externally_managed": False}) is True


def test_stop_blocked_when_externally_managed():
    assert _daemon_stop_allowed({"externally_managed": True}) is False
