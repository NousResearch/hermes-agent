"""Catalog pins and custom Git updates publish only a validated dependency set."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.pm.test_plugin_survival_contract import admission_env  # noqa: F401


def _commit(repo, message):
    env = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
           "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
    subprocess.run(["git", "add", "--all"], cwd=repo, env=env, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=repo, env=env, check=True, capture_output=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _version(repo, version, *, broken=False, minimum="", libraries=("shared-library",)):
    from hermes_cli.plugins_manifest import SUPPORTED_MANIFEST_VERSION

    (repo / "plugin.yaml").write_text(
        f"name: transactional\nversion: {version}\nmanifest_version: {SUPPORTED_MANIFEST_VERSION}\n"
        f"requires_hermes: '{minimum}'\n", encoding="utf-8")
    (repo / "__init__.py").write_text(f"VERSION = {version!r}\ndef register(ctx):\n    pass\n", encoding="utf-8")
    (repo / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    deps = '["impossible-plugin-dep==1", "impossible-plugin-dep==2"]' if broken else '[]'
    for name in libraries:
        library = repo.parent / name
        library.mkdir(exist_ok=True)
        (library / "pyproject.toml").write_text(
            f'[project]\nname="{name}"\nversion="1"\nrequires-python=">=3.14"\n'
            '[tool.uv]\npackage=false\n', encoding="utf-8")
    if not broken:
        deps = json.dumps(list(libraries))
    sources = "".join(f'{name}={{path="../{name}"}}\n' for name in libraries)
    (repo / "pyproject.toml").write_text(
        f'[project]\nname="transactional"\nversion="{version}"\nrequires-python=">=3.14"\n'
        f'dependencies={deps}\n[tool.uv]\npackage=false\n'
        f'[tool.uv.sources]\n{sources}', encoding="utf-8")
    return _commit(repo, version)


@pytest.fixture
def installed(admission_env, monkeypatch, request):
    from hermes_cli import plugin_catalog, plugins_cmd, plugins_cmd_catalog

    root, home = admission_env
    repo = root / "plugin-origin"
    repo.mkdir()
    subprocess.run(["git", "init", "-qb", "main"], cwd=repo, check=True, capture_output=True)
    first = _version(repo, "1.0.0")
    catalog = request.param == "catalog"
    state = {"sha": first}

    def entry():
        return plugin_catalog.PluginCatalogEntry(
            name="transactional", repo=repo.as_uri(), sha=state["sha"], description="fixture", maintainer="fixture")

    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **kwargs: None)
    monkeypatch.setattr(plugin_catalog, "load_catalog", lambda catalog_dir=None: [entry()])
    monkeypatch.setattr(plugin_catalog, "load_removed_list", lambda catalog_dir=None: [])
    monkeypatch.setattr(plugins_cmd, "_scan_on_install_enabled", lambda: False)
    if catalog:
        target, _, _ = plugins_cmd_catalog.install_catalog_entry(entry(), force=False)
    else:
        target, _, _ = plugins_cmd._install_plugin_core(repo.as_uri(), force=False)
    import shutil

    shutil.copytree(repo.parent / "shared-library", target.parent / "shared-library")
    plugins_cmd._set_plugin_enabled("transactional", enable=True)
    return root, home, repo, target, state


def _head(repo) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


@pytest.mark.parametrize("installed", ["catalog", "custom"], indirect=True)
def test_unattended_update_refuses_newly_declared_python_dependencies(installed):
    """Install gates Python deps behind a prompt; an update that ADDS a dependency must not
    slip them into the shared environment without one. With nobody to ask (dashboard, gateway
    ``plugins.auto_apply``) the update is refused and nothing — code, env, metadata — moves."""
    import shutil
    from hermes_cli import plugins_cmd
    from pm.environments import selected_venv

    root, home, repo, target, state = installed
    old_head, old_venv = _head(target), selected_venv(root / "core")
    state["sha"] = _version(repo, "2.0.0", libraries=("shared-library", "second-library"))
    shutil.copytree(repo.parent / "second-library", target.parent / "second-library", dirs_exist_ok=True)

    result = plugins_cmd.dashboard_update_user_plugin("transactional")

    assert result["ok"] is False, result
    assert "second-library" in result["error"]
    assert _head(target) == old_head
    assert selected_venv(root / "core") == old_venv


@pytest.mark.parametrize("installed", ["custom"], indirect=True)
@pytest.mark.parametrize("answer, published", [("y", True), ("n", False)])
def test_interactive_update_asks_before_installing_new_python_dependencies(
        installed, monkeypatch, answer, published):
    """A terminal user is asked the same question install asks; yes publishes code AND the
    dependency together, no leaves both untouched."""
    import shutil
    from types import SimpleNamespace
    from hermes_cli import plugins_cmd
    from hermes_cli.plugins_transaction import update_plugin
    from pm.environments import selected_venv

    root, home, repo, target, state = installed
    old_head, old_venv = _head(target), selected_venv(root / "core")
    state["sha"] = _version(repo, "2.0.0", libraries=("shared-library", "second-library"))
    shutil.copytree(repo.parent / "second-library", target.parent / "second-library", dirs_exist_ok=True)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("sys.stdout", SimpleNamespace(isatty=lambda: True, write=lambda *a: None, flush=lambda: None))
    asked: list = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or answer)

    if published:
        update_plugin(target, interactive=True)
    else:
        with pytest.raises(plugins_cmd.PluginOperationError, match="declined"):
            update_plugin(target, interactive=True)

    assert asked, "no dependency consent prompt was shown"
    assert (_head(target) == state["sha"]) is published
    assert (selected_venv(root / "core") != old_venv) is published
    assert (_head(target) != old_head) is published


@pytest.mark.parametrize("installed", ["catalog", "custom"], indirect=True)
def test_update_rebuilds_an_accepted_node_sidecar_when_its_manifest_moves(installed, monkeypatch):
    """Publication swaps the whole tree, so a node_modules the user accepted at install would
    come back stale (custom pull copies it) or missing (catalog re-clone). When package.json
    changes, the sidecar is rebuilt in the staged copy and published with the code."""
    from hermes_cli import plugins_cmd
    from pm import workspace

    root, home, repo, target, state = installed
    (repo / "package.json").write_text('{"name": "transactional", "dependencies": {}}', encoding="utf-8")
    _commit(repo, "add node sidecar")
    # the user accepted the sidecar at install time
    (target / "node_modules").mkdir()
    (target / "node_modules" / "stale").write_text("v1", encoding="utf-8")
    (repo / "package.json").write_text('{"name": "transactional", "dependencies": {"left-pad": "*"}}', encoding="utf-8")
    state["sha"] = _commit(repo, "bump node deps")
    rebuilt: list = []

    def fake_npm(plugin_dir, *, explicit=False):
        rebuilt.append(plugin_dir)
        (plugin_dir / "node_modules").mkdir(exist_ok=True)
        (plugin_dir / "node_modules" / "fresh").write_text("v2", encoding="utf-8")
        return None

    monkeypatch.setattr(workspace, "install_node_sidecar", fake_npm)
    result = plugins_cmd.dashboard_update_user_plugin("transactional")

    assert result["ok"] is True, result
    assert len(rebuilt) == 1 and rebuilt[0] != target, "npm must run in the staged copy, not the live tree"
    assert (target / "node_modules" / "fresh").is_file()
    assert _head(target) == state["sha"]


@pytest.mark.parametrize("installed", ["catalog", "custom"], indirect=True)
@pytest.mark.parametrize("failure", ["dependencies", "version", "publication", "manifest"])
def test_failed_update_keeps_code_metadata_config_and_environment(installed, monkeypatch, failure):
    from hermes_cli import plugins_cmd
    from pm.environments import selected_venv
    from pm import paths
    from pm.lock import Facts

    root, home, repo, target, state = installed
    selected = selected_venv(root / "core")
    watched = [target / "plugin.yaml", target / "__init__.py", target / "pyproject.toml",
               home / "config.yaml", home / "plugins/.install-metadata.json", paths.runtime_facts_path()]

    before = {path: path.read_bytes() for path in watched}
    old_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    state["sha"] = _version(repo, "2.0.0", broken=failure == "dependencies",
                             minimum=">=999.0.0" if failure == "version" else "")
    if failure == "version":
        # A checkout without a vX.Y.Z tag (CI's depth-1 clone) runs an unparseable version,
        # which the gate deliberately treats as permissive; pin a real one so it can refuse.
        from hermes_cli import plugins_manifest
        monkeypatch.setattr(plugins_manifest, "running_hermes_version", lambda: "1.0.0")
    if failure == "manifest":
        (repo / "plugin.yaml").write_text("name: [broken", encoding="utf-8")
        state["sha"] = _commit(repo, "invalid manifest")
    if failure == "publication":
        def deny(*args, **kwargs):
            raise OSError("fixture refuses environment publication")
        monkeypatch.setattr(Facts, "record_state", deny)
    result = plugins_cmd.dashboard_update_user_plugin("transactional")
    assert result["ok"] is False, result
    assert result.get("error")
    assert selected_venv(root / "core") == selected
    assert {path: path.read_bytes() for path in watched} == before
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip() == old_head


@pytest.mark.parametrize("installed", ["catalog", "custom"], indirect=True)
def test_successful_update_publishes_matching_code_and_durable_workspace(installed):
    import tomllib
    from hermes_cli import plugins_cmd
    from pm.environments import selected_venv
    from pm import paths
    from pm.lock import Facts
    from pm.packages import Venv

    root, home, repo, target, state = installed
    before = selected_venv(root / "core")
    config = (home / "config.yaml").read_bytes()
    state["sha"] = _version(repo, "2.0.0")
    result = plugins_cmd.dashboard_update_user_plugin("transactional")
    assert result["ok"] is True, result
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip() == state["sha"]
    assert 'VERSION = \'2.0.0\'' in (target / "__init__.py").read_text(encoding="utf-8")
    assert (home / "config.yaml").read_bytes() == config
    fact = Facts(paths.runtime_facts_path()).get("venv")
    assert selected_venv(root / "core") != before
    assert fact["stamp"] == Venv().expected_stamp(fact["extras"])
    workspace = Path(fact["resolved_lock"]).parent
    document = tomllib.loads((workspace / "pyproject.toml").read_text(encoding="utf-8"))
    members = document["tool"]["uv"]["workspace"]["members"]
    assert members
    for member in members:
        path = (workspace / member).resolve()
        assert path.is_relative_to(workspace)
        assert (path / "pyproject.toml").is_file()
    record = json.loads((home / "plugins/.install-metadata.json").read_text(encoding="utf-8"))["transactional"]
    assert record["revision"] == state["sha"]

    # A packaged member's source changes even when its dependency metadata does not.
    previous_stamp = fact["stamp"]
    (repo / "__init__.py").write_text("VERSION = 'code-only'\ndef register(ctx):\n    pass\n", encoding="utf-8")
    state["sha"] = _commit(repo, "code-only update")
    result = plugins_cmd.dashboard_update_user_plugin("transactional")
    assert result["ok"], result
    record = json.loads((home / "plugins/.install-metadata.json").read_text(encoding="utf-8"))["transactional"]
    assert Facts(paths.runtime_facts_path()).get("venv")["stamp"] != previous_stamp

    from hermes_cli.plugins_updates import run_checks

    def no_network(*args):
        raise AssertionError("catalog pin must not consult a custom update source")

    if (record.get("catalog") or {}).get("name"):
        state["sha"] = _version(repo, "3.0.0")
        check = next(row for row in run_checks(home / "plugins", include_pip=False,
                                              fetch=no_network, ls_remote=no_network)
                     if row.name == "transactional")
        assert check.klass == "catalog" and check.update_available is True
        assert check.current == record["revision"] and check.latest == state["sha"]
