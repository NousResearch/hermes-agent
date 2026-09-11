"""Admission includes profile changes committed before its install lock is acquired."""
from contextlib import contextmanager
import importlib
from types import SimpleNamespace


def test_admission_reads_other_profiles_after_taking_lock(tmp_path, monkeypatch):
    from hermes_cli import plugins_admission as admission
    from hermes_cli import runtime_state
    import pm.paths as paths

    ensure = importlib.import_module("pm.ensure")
    # Exercise the engine lock ordering, not cross-process transport.
    monkeypatch.setattr("pm.client.sync_venv", ensure.sync_venv)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(ensure, "_facts", lambda: {})
    sibling = tmp_path / "sibling-plugin"
    state = {"members": [], "inside": False}
    real_lock = runtime_state.runtime_lock

    @contextmanager
    def after_competing_publication(project):
        with real_lock(project):
            state["members"] = [sibling]
            state["inside"] = True
            try:
                yield
            finally:
                state["inside"] = False

    def members(*args, **kwargs):
        return list(state["members"])

    def apply(extras, *, plugin_dirs):
        assert state["inside"]
        assert plugin_dirs == [sibling]
        return {}

    monkeypatch.setattr(runtime_state, "runtime_lock", after_competing_publication)
    monkeypatch.setattr(admission, "candidate_member_dirs", members)
    monkeypatch.setattr(ensure, "get_package", lambda _: SimpleNamespace(
        expected_stamp=lambda *args, **kwargs: "new", apply=apply,
    ))
    admission.admit_plugin_set_change({"requested"}, set())
