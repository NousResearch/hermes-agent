"""A Nix package's environment is admitted against, never rebuilt (pm.nix_admission)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from pm import paths
from pm.package import InstallError
from pm.plugin_inputs import Selection, StagedUpdate

from tests.pm._fixtures import (
    client as client,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
    isolated_python as isolated_python,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
)

# Spawns children with a home it builds itself; the parent's must stay real.
pytestmark = pytest.mark.real_machine_home

ABSENT = "hermes-nix-absent-dist>=1"


def _nix_package(tmp_path, monkeypatch, isolated_python) -> Path:
    """The Nix layout: the executing tree carries no stamp, lockfile or project file of its own
    that PM may use; the wrapper's HERMES_INSTALL_ROOT names the directory holding the stamp,
    which names the package's PM runtime."""
    site = subprocess.run([str(isolated_python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                          capture_output=True, text=True, check=True).stdout.strip()
    runtime = tmp_path / "pm-runtime"
    runtime.mkdir()
    (runtime / "pm-runtime.json").write_text(json.dumps({"python": str(isolated_python), "sitePackages": site}))
    share = tmp_path / "share" / "hermes-agent"
    share.mkdir(parents=True)
    (share / "install-stamp.json").write_text(json.dumps(
        {"distribution": "nix", "baseVersion": "1.2.3", "pmRuntime": str(runtime)}))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(share))
    executing = Path(paths.__file__).resolve().parent.parent
    assert paths.repo_root() == executing and not (executing / "install-stamp.json").exists()
    return executing


def _home(tmp_path, plugins: dict[str, list[str]], enabled=()):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    for name, requirements in plugins.items():
        directory = home / "plugins" / name
        directory.mkdir(parents=True)
        (directory / "plugin.yaml").write_text(
            f"name: {name}\n" + (f"python_dependencies: {json.dumps(requirements)}\n" if requirements else ""))
    (home / "config.yaml").write_text(f"# keep my comment\nplugins:\n  enabled: {json.dumps(list(enabled))}\n")
    return home


def _select(client, home, enabled, disabled=()):
    client.sync_venv(explicit=True, plugins=Selection({
        "home": str(home), "enabled": list(enabled), "disabled": list(disabled), "extra_dirs": [],
    }))


def _enabled(home):
    from utils import fast_safe_load

    return fast_safe_load((home / "config.yaml").read_text())["plugins"]["enabled"]


def _built_nothing(repo):
    from pm.environments import install_state_dir

    state = install_state_dir(repo)
    return not any((state / name).exists() for name in ("facts.json", "environments", "publication.json")) \
        and not paths.facts_path().exists()


def test_a_selection_is_published_without_building(client, tmp_path, monkeypatch, isolated_python):
    repo = _nix_package(tmp_path, monkeypatch, isolated_python)
    home = _home(tmp_path, {"plain": []})
    _select(client, home, ["plain"])
    assert _enabled(home) == ["plain"]
    assert "# keep my comment" in (home / "config.yaml").read_text()
    assert _built_nothing(repo)


def test_a_plugin_whose_requirements_are_already_importable_is_admitted(client, tmp_path, monkeypatch, isolated_python):
    repo = _nix_package(tmp_path, monkeypatch, isolated_python)
    home = _home(tmp_path, {"satisfied": ["packaging>=20"]})
    _select(client, home, ["satisfied"])
    assert _enabled(home) == ["satisfied"]
    assert _built_nothing(repo)


def test_a_plugin_is_refused_by_the_package_it_lacks(client, tmp_path, monkeypatch, isolated_python):
    repo = _nix_package(tmp_path, monkeypatch, isolated_python)
    home = _home(tmp_path, {"needy": [ABSENT, "packaging>=99999"]})
    before = (home / "config.yaml").read_bytes()
    with pytest.raises(InstallError) as caught:
        _select(client, home, ["needy"])
    assert f"{ABSENT} (plugin 'needy')" in caught.value.cause
    assert "packaging>=99999 (found " in caught.value.cause
    assert "extraPythonPackages" in caught.value.remedy
    assert (home / "config.yaml").read_bytes() == before
    assert _built_nothing(repo)


def test_an_unsatisfied_plugin_enabled_earlier_does_not_block_other_changes(client, tmp_path, monkeypatch,
                                                                             isolated_python):
    """Enabled by hand before admission worked: it stays, and neither it nor its neighbours are stuck."""
    _nix_package(tmp_path, monkeypatch, isolated_python)
    home = _home(tmp_path, {"broken": [ABSENT], "plain": []}, enabled=["broken"])
    _select(client, home, ["broken", "plain"])
    assert _enabled(home) == ["broken", "plain"]
    _select(client, home, ["plain"], disabled=["broken"])
    assert _enabled(home) == ["plain"]
    with pytest.raises(InstallError, match="plugin 'broken'"):
        _select(client, home, ["broken", "plain"])


@pytest.mark.parametrize("active,requirements,published", [
    (True, [ABSENT], False),
    (True, ["packaging>=20"], True),
    (False, [ABSENT], True),  # a disabled plugin's code brings nothing into the environment
])
def test_a_plugin_update_follows_the_same_rule(client, tmp_path, monkeypatch, isolated_python,
                                               active, requirements, published):
    from pm.store import tree_digest

    repo = _nix_package(tmp_path, monkeypatch, isolated_python)
    home = _home(tmp_path, {"example": []}, enabled=["example"] if active else [])
    target = home / "plugins" / "example"
    (target / "code.py").write_text("old code")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "plugin.yaml").write_text(f"name: example\npython_dependencies: {json.dumps(requirements)}\n")
    (staged / "code.py").write_text("new code")
    update = StagedUpdate({"staged": str(staged), "target": str(target), "target_digest": tree_digest(target),
                           "old_metadata": {}, "new_metadata": {"example": {"revision": "new"}}})
    if published:
        client.sync_venv(explicit=True, plugins=update)
    else:
        with pytest.raises(InstallError, match="plugin 'example'"):
            client.sync_venv(explicit=True, plugins=update)
    assert (target / "code.py").read_text() == ("new code" if published else "old code")
    assert not list(target.parent.glob(".previous-*"))
    assert _built_nothing(repo)


def test_extras_are_checked_through_the_worker(client, tmp_path, monkeypatch, isolated_python):
    _nix_package(tmp_path, monkeypatch, isolated_python)
    (tmp_path / "home").mkdir()
    with pytest.raises(InstallError, match="extra 'no-such-extra' \\(not built into this package\\)"):
        client.sync_venv(["no-such-extra"], explicit=True)
    assert client.venv_is_current() is True
    assert client.venv_is_current(extras=["no-such-extra"]) is False
    with pytest.raises(InstallError, match="part of its Nix package"):
        client.sync_venv(repair=True)


def test_every_pm_child_receives_the_applications_import_path(monkeypatch):
    """`hermes pm install --extra X` runs PM's CLI, not the worker: both must be able to check."""
    import sys
    from pm import nix_admission
    from pm.runtime import runtime_environment

    monkeypatch.delenv(nix_admission.APPLICATION_PATH_ENV, raising=False)
    assert runtime_environment()[nix_admission.APPLICATION_PATH_ENV].split(os.pathsep) == [p for p in sys.path if p]
    # PM's own interpreter passes the application's path on instead of replacing it with its own.
    monkeypatch.setenv(nix_admission.APPLICATION_PATH_ENV, "/application/site-packages")
    monkeypatch.setattr("pm.runtime.is_runtime", lambda: True)
    assert runtime_environment()[nix_admission.APPLICATION_PATH_ENV] == "/application/site-packages"


def _distribution(site: Path, name: str, version: str, lines=()):
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("\n".join(
        ["Metadata-Version: 2.3", f"Name: {name}", f"Version: {version}", *lines]) + "\n")


def test_missing_reports_extras_against_the_installed_core(tmp_path, monkeypatch):
    from pm import nix_admission

    site = tmp_path / "site"
    _distribution(site, "hermes-agent", "1.2.3", [
        "Provides-Extra: web", "Provides-Extra: voice", "Provides-Extra: all",
        "Requires-Dist: present-dist>=1; extra == 'web'",
        "Requires-Dist: absent-dist>=2; extra == 'voice'",
        "Requires-Dist: other-os-dist; sys_platform == 'no-such-os' and extra == 'web'",
        "Requires-Dist: hermes-agent[web]; extra == 'all'",
        "Requires-Dist: hermes-agent[voice]; extra == 'all'",
    ])
    _distribution(site, "present-dist", "1.5")
    _distribution(site, "odd-dist", "1.0-nix-unstable")
    monkeypatch.setenv(nix_admission.APPLICATION_PATH_ENV, str(site))
    assert nix_admission.missing(["web"]) == []
    assert nix_admission.missing(["voice"]) == ["absent-dist>=2 (extra 'voice')"]
    assert nix_admission.missing(["all"]) == ["absent-dist>=2 (extra 'all')"]
    assert nix_admission.missing(["unknown"]) == ["extra 'unknown' (not built into this package)"]
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text('name: plugin\npython_dependencies: ["odd-dist>=1", "present-dist>=2"]\n')
    # A version no specifier can be compared with is present, not provably wrong.
    assert nix_admission.missing((), [plugin]) == ["present-dist>=2 (found 1.5) (plugin 'plugin')"]


def test_a_member_that_excludes_the_running_python_is_reported(tmp_path, monkeypatch):
    from pm import nix_admission

    monkeypatch.setenv(nix_admission.APPLICATION_PATH_ENV, str(tmp_path))
    plugin = tmp_path / "strict"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: strict\n")
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "strict"\nversion = "1"\nrequires-python = ">=3.99"\ndependencies = ["packaging"]\n')
    assert any("plugin 'strict' requires Python >=3.99" in entry for entry in nix_admission.missing((), [plugin]))


def test_nothing_is_admitted_unchecked_when_the_applications_path_is_unknown(tmp_path, monkeypatch):
    from pm import nix_admission

    monkeypatch.delenv(nix_admission.APPLICATION_PATH_ENV, raising=False)
    plugin = tmp_path / "needy"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(f'name: needy\npython_dependencies: ["{ABSENT}"]\n')
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "plugin.yaml").write_text("name: plain\n")
    assert nix_admission.missing((), [plain]) == []
    assert nix_admission.missing(["voice"], [plugin, plain]) == [
        "extra 'voice' (cannot be checked: the application's import path is unknown)",
        "plugin 'needy' (cannot be checked: the application's import path is unknown)",
    ]


def test_only_the_executing_nix_tree_is_a_nix_package(tmp_path, monkeypatch):
    """The package's stamp answers for the executing tree, never for a checkout it was asked to sync."""
    from pm import nix_admission

    share = tmp_path / "share"
    share.mkdir()
    (share / "install-stamp.json").write_text(json.dumps({"distribution": "nix"}))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(share))
    assert nix_admission.is_nix_package() is True
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: checkout)
    assert nix_admission.is_nix_package() is False
    (checkout / "install-stamp.json").write_text(json.dumps({"distribution": "docker"}))
    assert nix_admission.is_nix_package() is False
