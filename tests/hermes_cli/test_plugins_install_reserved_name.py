"""``hermes plugins install`` never installs a plugin AS ``plugins/model-providers``.

Provider discovery imports every package inside ``$HERMES_HOME/plugins/model-providers/`` by
placement alone, so a plugin installed under that folder name would load its packages as providers
while install, ``plugins list`` and ``plugins disable`` all call it not enabled. The installer refuses
that name (any letter case) before anything is written; operator-placed providers in that folder keep
loading exactly as before.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_cli import plugins_cmd
from tests.pm.test_plugin_survival_contract import _uv_available, admission_env

pytestmark = pytest.mark.skipif(not _uv_available(), reason="uv not on PATH")

_PROVIDER_SOURCE = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(name="{name}", aliases=(), base_url="acp://{name}",
                                      auth_type="external_process"))
    """
)


def _clear_provider_caches():
    import providers

    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False
    for mod in list(sys.modules):
        if mod.startswith(("plugins.model_providers", "_hermes_user_provider")):
            del sys.modules[mod]


@pytest.fixture
def world(admission_env):  # health: allow F811 -- pytest injects the imported admission_env fixture by parameter name
    _root, home = admission_env
    _clear_provider_caches()
    yield home
    _clear_provider_caches()


def _git_repo(path: Path, manifest_name: str, marker: Path) -> Path:
    """A local git plugin whose nested ``telemetry`` package writes *marker* when imported."""
    (path / "telemetry").mkdir(parents=True)
    (path / "plugin.yaml").write_text(f"name: {manifest_name}\nversion: 0.1.0\n", encoding="utf-8")
    (path / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (path / "telemetry" / "__init__.py").write_text(f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}}
    for args in (("init", "-q"), ("add", "."),
                 ("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "fixture")):
        subprocess.run(["git", *args], cwd=path, env=env, check=True, capture_output=True, timeout=30)
    return path


def _tree_digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir"
            for p in sorted(root.rglob("*"))}


def _discover_providers():
    import providers

    _clear_provider_caches()
    return {p.name for p in providers.list_providers()}


@pytest.mark.parametrize("manifest_name", ["model-providers", "Model-Providers"])
@pytest.mark.parametrize("flags", [
    {"enable": False}, {"enable": True}, {"enable": None}, {"enable": False, "force": True},
    {"enable": True, "force": True},
])
def test_install_refuses_the_provider_plugins_folder_name(world, tmp_path, capsys, manifest_name, flags):
    marker = tmp_path / "imported-marker"
    repo = _git_repo(tmp_path / "src", manifest_name, marker)
    config_before = (world / "config.yaml").read_bytes()

    with pytest.raises(SystemExit) as exc:
        plugins_cmd.cmd_install(repo.as_uri(), yes_deps=True, allow_removed=True, **flags)

    assert exc.value.code == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "model-providers is the folder Hermes reserves for provider plugins" in out
    assert "cannot be used as a plugin's own name" in out
    assert "Change the name in the plugin's manifest" in out
    assert list((world / "plugins").iterdir()) == []
    assert (world / "config.yaml").read_bytes() == config_before
    _discover_providers()
    assert not marker.exists()


def test_forced_install_leaves_hand_placed_provider_folder_byte_for_byte(world, tmp_path):
    folder = world / "plugins" / "model-providers"
    (folder / "handprov").mkdir(parents=True)
    (folder / "handprov" / "__init__.py").write_text(_PROVIDER_SOURCE.format(name="handprov"), encoding="utf-8")
    (folder / "handprov" / "plugin.yaml").write_text("name: handprov\nkind: model-provider\n", encoding="utf-8")
    before = _tree_digest(folder)
    marker = tmp_path / "imported-marker"
    repo = _git_repo(tmp_path / "src", "model-providers", marker)

    with pytest.raises(SystemExit):
        plugins_cmd.cmd_install(repo.as_uri(), force=True, enable=False, yes_deps=True, allow_removed=True)

    assert _tree_digest(folder) == before
    assert "handprov" in _discover_providers()
    assert not marker.exists()


def test_dashboard_install_reports_the_same_refusal(world, tmp_path):
    marker = tmp_path / "imported-marker"
    repo = _git_repo(tmp_path / "src", "model-providers", marker)

    result = plugins_cmd.dashboard_install_plugin(repo.as_uri(), force=True, enable=True)

    assert result["ok"] is False
    assert "reserves for provider plugins" in result["error"]
    assert list((world / "plugins").iterdir()) == []


@pytest.mark.parametrize("manifest_name", ["probe-plugin", "my-model-providers"])
def test_other_names_install_as_before(world, tmp_path, capsys, manifest_name):
    marker = tmp_path / "imported-marker"
    repo = _git_repo(tmp_path / "src", manifest_name, marker)

    plugins_cmd.cmd_install(repo.as_uri(), enable=False, yes_deps=True, allow_removed=True)

    out = " ".join(capsys.readouterr().out.split())
    assert (world / "plugins" / manifest_name / "telemetry" / "__init__.py").is_file()
    assert (f"Plugin installed but not enabled. Run `hermes plugins enable {manifest_name}` to activate."
            in out)
    _discover_providers()
    assert not marker.exists()


def test_hand_placed_provider_under_the_folder_is_still_discovered(world):
    provider = world / "plugins" / "model-providers" / "handprov"
    provider.mkdir(parents=True)
    (provider / "__init__.py").write_text(_PROVIDER_SOURCE.format(name="handprov"), encoding="utf-8")

    assert "handprov" in _discover_providers()


def test_update_of_an_existing_install_under_that_name_still_re_clones(world, tmp_path):
    """An install that already sits under that name from an earlier Hermes keeps updating as before."""
    import shutil

    from hermes_cli.plugins_cmd_update import _reclone_plugin_update

    marker = tmp_path / "imported-marker"
    repo = _git_repo(tmp_path / "src" / "outer", "model-providers", marker)
    (repo.parent / "unused").mkdir()
    source = f"{repo.parent.as_uri()}#outer"
    subprocess.run(["git", "init", "-q"], cwd=repo.parent, check=True, capture_output=True)
    shutil.rmtree(repo / ".git")
    subprocess.run(["git", "add", "."], cwd=repo.parent, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "f"],
                   cwd=repo.parent, check=True, capture_output=True)
    target = world / "plugins" / "model-providers"
    shutil.copytree(repo, target)
    plugins_cmd._write_install_metadata({"model-providers": {"source": source, "revision": "0" * 40,
                                                             "pinned": False}})

    output = _reclone_plugin_update(target, source, "0" * 40, allow_removed=True)

    assert output.startswith(f"Re-installed from {source}")
    assert (target / "telemetry" / "__init__.py").is_file()
