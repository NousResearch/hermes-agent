"""discover_plugins() must not load another profile's plugins from a REQUEST scope (#106608).

A web request scoped to another profile (the dashboard's ``?profile=<name>`` handling) enters the
context-local ``HERMES_HOME`` override marked request-scoped. Plugin registration has process-global
side effects — dashboard-auth providers are upserted by name into a process-global registry — so
loading that profile's manager would replace the provider validating the running dashboard's
sessions and 401 every live cookie.

The override ALONE must not suppress the load: the cron external worker and a multiplexed gateway
turn each scope the profile the process is acting AS, and both need that profile's plugin tools
(``agent/agent_init.py::_load_tools`` calls this under those scopes).
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_constants import (
    is_request_scoped_hermes_home,
    request_scoped_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


def test_discover_plugins_skips_load_under_request_scope(tmp_path, monkeypatch):
    from hermes_cli import plugins as P

    other = tmp_path / "profiles" / "worker"
    other.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with request_scoped_hermes_home(other):
        with patch.object(P, "get_plugin_manager") as mgr, \
             patch.object(P, "_join_background_discovery") as join:
            P.discover_plugins()

    # The manager for the overridden home is never loaded (the #106608 trigger), but the
    # process-global startup discovery is still joined — skipping that would let a scoped
    # request race in-flight process-global registration.
    mgr.assert_not_called()
    join.assert_called_once()


def test_discover_plugins_loads_under_override_without_the_request_marker(tmp_path, monkeypatch):
    """Both non-request override shapes keep loading their profile's plugins.

    * cron external worker — ``cron/scheduler.py`` sets the override to the worker's OWN home
      (its ``HERMES_HOME`` env), and nothing has discovered plugins before that point.
    * multiplexed gateway turn — ``gateway/run.py::_profile_runtime_scope`` scopes one turn to a
      secondary profile's home, and ``agent/agent_init.py::_load_tools`` discovers that profile's
      plugins for the turn's tool snapshot.

    Both reach ``discover_plugins()`` through ``_load_tools``, so a request-only skip is what keeps
    their plugin-provided tools.
    """
    from hermes_cli import plugins as P

    default_home = tmp_path / "default"
    worker_home = tmp_path / "profiles" / "worker"
    for home in (default_home, worker_home):
        home.mkdir(parents=True)

    for env_home, override_home in ((worker_home, worker_home), (default_home, worker_home)):
        monkeypatch.setenv("HERMES_HOME", str(env_home))
        token = set_hermes_home_override(str(override_home))
        try:
            with patch.object(P, "get_plugin_manager") as mgr, \
                 patch.object(P, "_join_background_discovery") as join:
                P.discover_plugins()
        finally:
            reset_hermes_home_override(token)

        join.assert_called_once()
        mgr.assert_called_once()
        mgr.return_value.discover_and_load.assert_called_once()


def test_discover_plugins_loads_without_override(tmp_path, monkeypatch):
    from hermes_cli import plugins as P

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with patch.object(P, "get_plugin_manager") as mgr, \
         patch.object(P, "_join_background_discovery") as join:
        P.discover_plugins()

    join.assert_called_once()
    mgr.assert_called_once()
    mgr.return_value.discover_and_load.assert_called_once()


def test_request_scope_marker_lives_only_for_the_block(tmp_path):
    assert is_request_scoped_hermes_home() is False
    with request_scoped_hermes_home(tmp_path / "profiles" / "worker"):
        assert is_request_scoped_hermes_home() is True
    assert is_request_scoped_hermes_home() is False
