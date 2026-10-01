from hermes_cli import plugins_loader


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
