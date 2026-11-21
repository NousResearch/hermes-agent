import logging
import threading


def test_retired_admitted_failure_is_logged_without_touching_new_generation(monkeypatch, caplog):
    from agent import plugin_stream_hooks as psh
    from hermes_cli import plugins

    psh.shutdown_plugin_observer_dispatcher()
    started, release, logged = threading.Event(), threading.Event(), threading.Event()
    old, replacement = plugins.PluginManager(), plugins.PluginManager()
    old._discovered = replacement._discovered = True
    reports = []

    def fail(**kwargs):
        started.set()
        assert release.wait(5)
        raise ValueError("secret-exception")

    old._hooks["memory_prefetch"] = [fail]
    old._report_hook_failure = lambda *args: reports.append("old")
    replacement._report_hook_failure = lambda *args: reports.append("replacement")
    current = [old]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: current[0])
    original_warning = psh.logger.warning

    def warning(*args, **kwargs):
        original_warning(*args, **kwargs)
        logged.set()

    monkeypatch.setattr(psh.logger, "warning", warning)
    caplog.set_level(logging.WARNING, logger=psh.logger.name)
    try:
        assert psh.enqueue_plugin_observer_hook("memory_prefetch", event_id="secret-payload")
        assert started.wait(5)
        old.unload()
        current[0] = replacement
        release.set()
        assert logged.wait(5)
        assert reports == []
        assert not old._hook_failures_reported
        assert not replacement._hook_failures_reported
        assert [record.getMessage() for record in caplog.records] == [
            "retired plugin observer callback failed; continuing dispatcher"
        ]
        assert all(not record.exc_info for record in caplog.records)
    finally:
        release.set()
        psh.shutdown_plugin_observer_dispatcher(timeout=5)
