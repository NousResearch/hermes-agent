"""Watchdog shutdown releases its children and preserves the booting profile's policy."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("enabled", [True, False])
def test_watchdog_reads_the_booting_profiles_restart_policy(tmp_path, monkeypatch, enabled):
    from hermes_cli.local_runtime import supervisor
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "secondary"
    profile.mkdir(parents=True)
    (home / "config.yaml").write_text(
        f"local_runtime:\n  enabled: {str(not enabled).lower()}\n", encoding="utf-8")
    (profile / "config.yaml").write_text(
        f"local_runtime:\n  enabled: {str(enabled).lower()}\n", encoding="utf-8")
    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path / "runtimes")
    monkeypatch.setattr(supervisor, "_port_in_use", lambda _: False)
    monkeypatch.setattr(supervisor, "_MAX_RESTARTS", 1)
    monkeypatch.setattr(supervisor.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    def run_watchdog(scope):
        sup = supervisor.LlamaServerSupervisor(tmp_path / "llama-server", tmp_path, port=59998)
        spawns = []

        def spawn():
            spawns.append(True)
            sup.proc = SimpleNamespace(pid=100 + len(spawns), poll=lambda: 1)

        monkeypatch.setattr(sup, "_spawn", spawn)
        monkeypatch.setattr(sup, "_wait_health", lambda _: None)
        monkeypatch.setattr(sup, "_reap_orphaned_children", lambda: None)
        monkeypatch.setattr(sup._stop_event, "wait", lambda _: False)
        token = set_hermes_home_override(scope)
        try:
            sup.start()
        finally:
            reset_hermes_home_override(token)
        sup._watchdog.join(timeout=5)
        assert not sup._watchdog.is_alive()
        return len(spawns)

    # A → B → A in one process also catches config-cache or thread-scope leakage on return.
    observed = [run_watchdog(scope) for scope in (home, profile, home)]
    assert observed == ([1, 2, 1] if enabled else [2, 1, 2])


def test_exhausted_restart_budget_releases_owned_children(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path / "runtimes")
    sup = supervisor.LlamaServerSupervisor(tmp_path / "llama-server", tmp_path, port=59998)
    sup.proc = SimpleNamespace(pid=100, poll=lambda: 1)
    sup._restarts = supervisor._MAX_RESTARTS
    closed = []
    sup._job = SimpleNamespace(close=lambda: closed.append(True))
    spawns = []
    monkeypatch.setattr(sup, "_spawn", lambda: spawns.append(True))

    sup._watch()

    assert closed == [True]
    assert sup._job is None
    assert spawns == []
