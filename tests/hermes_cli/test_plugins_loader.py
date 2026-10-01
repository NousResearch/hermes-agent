from hermes_cli import plugins_loader
import threading


def test_plugin_load_marker_propagates_into_deadline_worker(monkeypatch):
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 1.0)

    class Context:
        def _abandon_load(self):
            raise AssertionError("unexpected timeout")

    observed = []

    def load():
        observed.append(plugins_loader.in_plugin_load_worker())

    plugins_loader.run_with_load_deadline("test", Context(), load)

    assert observed == [True]


def test_nested_plugin_load_runs_inline_on_deadline_worker(monkeypatch):
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 1.0)

    class Context:
        def _abandon_load(self):
            raise AssertionError("unexpected timeout")

    observed_threads = []

    def nested_load():
        observed_threads.append(threading.current_thread())

    def outer_load():
        observed_threads.append(threading.current_thread())
        plugins_loader.run_with_load_deadline("nested", Context(), nested_load)

    plugins_loader.run_with_load_deadline("outer", Context(), outer_load)

    assert len(observed_threads) == 2
    assert observed_threads[0] is observed_threads[1]
