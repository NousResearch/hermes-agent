"""Regression for #123798: aliases of one store must reuse its PM generation."""
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("stamp", [None, {}, [], "malformed", "unreadable"])
def test_home_store_alias_reuses_selected_runtime(tmp_path, monkeypatch, request, stamp):
    from pm import paths, runtime

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (project / "uv.lock").write_text("version = 1\n")
    stamp_path = project / "install-stamp.json"
    if stamp is not None:
        stamp_path.write_text(stamp if isinstance(stamp, str) else json.dumps(stamp))
    if stamp == "unreadable":
        read_text = Path.read_text

        def read(path, *args, **kwargs):
            if path == stamp_path:
                raise PermissionError("fixture stamp cannot be read")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(paths, "repo_root", lambda: project)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    home = tmp_path / "owner"
    store = home / "tools"
    store.mkdir(parents=True)
    alias = tmp_path / "alternate"
    alias.mkdir()
    try:
        (alias / "tools").symlink_to(store, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    (store / "python").touch()
    selected = tmp_path / "pm-runtime"
    generation = selected / "generations" / "existing"
    generation.mkdir(parents=True)
    (generation / "pm-runtime.json").write_text("{}")
    (generation / ".lease-managed").touch()

    def release():
        held = runtime._HELD.pop(generation, None)
        if held is not None:
            held()

    request.addfinalizer(release)
    record = {"inputs": runtime._inputs(project, store.resolve() / "python"),
              "generation": "generations/existing"}
    (selected / "selected.json").write_text(json.dumps(record))
    before = (selected / "selected.json").read_bytes()
    validated = []
    # Dependency validation is outside this path-identity contract; staging is
    # forbidden, so no interpreter/dependency installation can mask a mismatch.
    monkeypatch.setattr(runtime, "_validate", lambda python, env: validated.append(python) or "")
    monkeypatch.setattr("pm.runtime_stage.stage_runtime", lambda *a, **kw: pytest.fail("rebuilt current runtime"))

    for active in (home, alias, home):
        monkeypatch.setenv("HERMES_HOME", str(active))
        result = runtime.prepare_runtime(
            store / "uv", paths.store_root() / "python", selected,
            project=project, bootstrap=False,
        )
        assert result == runtime._python(generation)
        assert paths.store_root() == store.resolve()
        assert (selected / "selected.json").read_bytes() == before
    assert validated == [runtime._python(generation)] * 3
    from hermes_cli.runtime_state import leases_held

    assert leases_held(generation)
    assert len(list((generation / ".leases").iterdir())) == 1
    release()
    assert not leases_held(generation)
    assert list((selected / "generations").iterdir()) == [generation]


@pytest.mark.parametrize("source", ["override", "stamp", "payload", "missing-default"])
def test_store_resolution_keeps_caller_precedence(tmp_path, monkeypatch, source):
    from pm import environments

    project = tmp_path / "payload" / "project"
    project.mkdir(parents=True)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    expected = home / "tools"
    if source != "missing-default":
        expected = project.parent / "chosen"
        expected.mkdir()
        alias = project.parent / "alias"
        try:
            alias.symlink_to(expected, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks unavailable: {exc}")
        if source == "override":
            monkeypatch.setenv("HERMES_RUNTIME_DIR", str(alias))
            (project.parent / "manifest.json").write_text("invalid, overridden")
        elif source == "stamp":
            (project / "install-stamp.json").write_text(json.dumps({"runtimeDir": str(alias)}))
        else:
            (project.parent / "manifest.json").write_text(json.dumps({"repo": "project", "store": "alias"}))
    assert environments.store_root(project) == expected.resolve()
    if source == "missing-default":
        assert not expected.exists(), "resolving a missing store must not create it"
