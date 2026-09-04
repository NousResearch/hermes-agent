"""Heal dangling setuptools editable finder mappings (#97819)."""

from __future__ import annotations

import json
from pathlib import Path

import hermes_editable_heal as heal
from hermes_cli import _early_recovery as er


def _write_finder(site: Path, mapping: dict, namespaces: dict | None = None) -> Path:
    site.mkdir(parents=True, exist_ok=True)
    path = site / "__editable___hermes_agent_0_20_6_finder.py"
    ns = namespaces if namespaces is not None else {}
    path.write_text(
        "from __future__ import annotations\n"
        f"MAPPING: dict[str, str] = {mapping!r}\n"
        f"NAMESPACES: dict[str, list[str]] = {ns!r}\n",
        encoding="utf-8",
    )
    return path


def _write_direct_url(site: Path, repo: Path) -> None:
    info = site / "hermes_agent-0.20.6.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "direct_url.json").write_text(
        json.dumps({"url": repo.resolve().as_uri(), "dir_info": {"editable": True}}),
        encoding="utf-8",
    )


def test_retarget_missing_and_site_packages_paths(tmp_path):
    repo = tmp_path / "hermes-agent"
    (repo / "hermes_cli").mkdir(parents=True)
    (repo / "hermes_cli" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "hermes_bootstrap.py").write_text("", encoding="utf-8")
    site = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
    deleted = site / "hermes_cli"
    mapping = {
        "hermes_cli": str(deleted),
        "hermes_bootstrap": str(site / "hermes_bootstrap"),
    }
    new, changed = heal.retarget_mapping(mapping, repo, site)
    assert changed
    assert new["hermes_cli"] == str(repo / "hermes_cli")
    assert new["hermes_bootstrap"] == str(repo / "hermes_bootstrap")


def test_heal_finder_rewrites_file_and_installs_hook(tmp_path):
    repo = tmp_path / "hermes-agent"
    (repo / "hermes_cli").mkdir(parents=True)
    (repo / "hermes_cli" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='hermes-agent'\n", encoding="utf-8")
    site = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
    finder = _write_finder(
        site,
        {"hermes_cli": str(site / "hermes_cli")},
        {"hermes_cli.data": [str(site / "hermes_cli" / "data")]},
    )
    (repo / "hermes_cli" / "data").mkdir()
    _write_direct_url(site, repo)

    assert heal.heal(project_root=repo, site_packages=site)

    text = finder.read_text(encoding="utf-8")
    assert str(repo / "hermes_cli") in text
    assert str(site / "hermes_cli") not in text
    assert (site / heal.HOOK_PTH).is_file()
    assert (site / heal.HOOK_MODULE).is_file()
    assert (site / heal.HOOK_PTH).read_text(encoding="utf-8") == heal._PTH_LINE
    hook = (site / heal.HOOK_MODULE).read_text(encoding="utf-8")
    assert hook == heal._THIN_HOOK
    assert "try:\n    heal()" not in hook


def test_resolve_project_root_from_direct_url(tmp_path):
    repo = tmp_path / "src"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    site = tmp_path / "site-packages"
    _write_direct_url(site, repo)
    assert heal.resolve_project_root(hint=site, site_packages=site) == str(
        repo.resolve()
    )


def test_project_root_hint_that_is_site_packages_uses_direct_url(tmp_path):
    from hermes_cli import _startup_fast

    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    site = tmp_path / "site-packages"
    site.mkdir()
    _write_direct_url(site, repo)
    assert heal.resolve_project_root(hint=site) == str(repo.resolve())
    # In-tree hermes_cli still resolves to this checkout.
    assert (Path(_startup_fast.project_root_str()) / "pyproject.toml").is_file()


def test_exhausted_early_recovery_still_heals_finder(tmp_path, monkeypatch):
    repo = tmp_path / "hermes-agent"
    (repo / "hermes_cli").mkdir(parents=True)
    (repo / "hermes_cli" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='hermes-agent'\n", encoding="utf-8")
    venv_site = repo / "venv" / "lib" / "python3.11" / "site-packages"
    finder = _write_finder(venv_site, {"hermes_cli": str(venv_site / "hermes_cli")})
    _write_direct_url(venv_site, repo)
    marker = repo / ".update-incomplete"
    marker.write_text(
        json.dumps({"attempts": er._EARLY_CORE_INSTALL_MAX_ATTEMPTS}),
        encoding="utf-8",
    )

    from hermes_cli import _install_repair as ir

    monkeypatch.setattr(
        ir,
        "run_core_install",
        lambda _r: (_ for _ in ()).throw(
            AssertionError("install must NOT run past the attempts ceiling")
        ),
    )

    er.recover_if_needed(project_root=repo, argv=[])

    text = finder.read_text(encoding="utf-8")
    assert str(repo / "hermes_cli") in text
    assert marker.exists()


def test_startup_hook_is_idempotent_across_repeated_heal(tmp_path):
    repo = tmp_path / "hermes-agent"
    (repo / "hermes_cli").mkdir(parents=True)
    (repo / "hermes_cli" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='hermes-agent'\n", encoding="utf-8")
    site = repo / "venv" / "lib" / "python3.11" / "site-packages"
    _write_finder(site, {"hermes_cli": str(site / "hermes_cli")})
    _write_direct_url(site, repo)

    assert heal.heal(project_root=repo, site_packages=site)
    hook_path = site / heal.HOOK_MODULE
    first = hook_path.read_text(encoding="utf-8")
    assert first == heal._THIN_HOOK

    # Same as the sidecar running heal() again on the next interpreter start.
    for _ in range(5):
        heal.heal(project_root=repo, site_packages=site)
    assert hook_path.read_text(encoding="utf-8") == first
    assert (site / "hermes_editable_heal.py").read_text(
        encoding="utf-8"
    ) == Path(heal.__file__).read_text(encoding="utf-8")


def test_pth_hook_heals_before_console_script_import(tmp_path):
    """The dead-launcher path: import hermes_cli with only site-packages on path."""
    import os
    import subprocess
    import sys

    repo = tmp_path / "hermes-agent"
    pkg = repo / "hermes_cli"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("marker = 'from-repo'\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='hermes-agent'\n", encoding="utf-8")
    site = tmp_path / "site-packages"
    finder = _write_finder(site, {"hermes_cli": str(site / "hermes_cli")})
    finder.write_text(
        finder.read_text(encoding="utf-8")
        + "\nimport sys\n"
        + "class _F:\n"
        + "    @classmethod\n"
        + "    def find_spec(cls, fullname, path=None, target=None):\n"
        + "        if fullname not in MAPPING:\n"
        + "            return None\n"
        + "        from importlib.util import spec_from_file_location\n"
        + "        from pathlib import Path\n"
        + "        loc = Path(MAPPING[fullname])\n"
        + "        init = loc / '__init__.py'\n"
        + "        if init.is_file():\n"
        + "            return spec_from_file_location(fullname, init)\n"
        + "        return None\n"
        + "def install():\n"
        + "    if not any(f is _F for f in sys.meta_path):\n"
        + "        sys.meta_path.append(_F)\n",
        encoding="utf-8",
    )
    (site / "__editable__.hermes_agent-0.20.6.pth").write_text(
        "import __editable___hermes_agent_0_20_6_finder; "
        "__editable___hermes_agent_0_20_6_finder.install()\n",
        encoding="utf-8",
    )
    _write_direct_url(site, repo)
    heal.install_startup_hook(site)
    assert str(site / "hermes_cli") in finder.read_text(encoding="utf-8")

    cwd = tmp_path / "not-the-repo"
    cwd.mkdir()
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import site\n"
        f"site.addsitedir({str(site)!r})\n"
        "import hermes_cli\n"
        "print(hermes_cli.marker)\n"
        "print(hermes_cli.__file__)\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "from-repo" in result.stdout
    assert "hermes-agent" in result.stdout.replace("\\", "/")
