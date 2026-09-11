"""The public PM surface never installs into the caller's Python process."""
import importlib


def test_public_mutations_delegate_but_environment_reads_stay_local(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    import pm
    client = importlib.import_module("pm.client")
    engine = importlib.import_module("pm.ensure")
    calls = []
    monkeypatch.setattr(client, "_request", lambda operation, arguments, **kw: calls.append(operation))
    monkeypatch.setattr(engine, "ensure", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("inline install")))
    monkeypatch.setattr(engine, "sync_venv", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("inline sync")))
    monkeypatch.setattr(client, "is_runtime", lambda: False, raising=False)
    pm.sync_venv(["all"], explicit=True, plugin_dirs=lambda: (_ for _ in ()).throw(AssertionError("inline sync")))
    pm.ensure("node", explicit=True)
    assert calls == ["sync_venv", "ensure"]
    assert pm.env_for is engine.env_for
