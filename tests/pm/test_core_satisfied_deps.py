"""Part A: skip PM venv rebuild when Hermes core's base environment already satisfies a plugin."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from pm.plugin_declarations import (
    PythonDeclaration,
    WorkspaceMembershipAssessment,
    assess_workspace_membership,
    needs_workspace_member,
    read_python_declaration,
    unsatisfied_requirements,
)
from pm import workspace as ws
from tests.pm import _fixtures


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_plugin(
    plugin_dir: Path,
    *,
    name: str = "probe",
    pyproject: str | None = None,
    manifest: str | None = None,
) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        manifest if manifest is not None else f"name: {name}\n",
        encoding="utf-8",
    )
    if pyproject is not None:
        (plugin_dir / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return plugin_dir


def _make_dist_info(site_dir: Path, name: str, version: str) -> None:
    dist_info = site_dir / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )


def _locked_core(tmp_path: Path, deps: list[str], wheels: Path) -> Path:
    core = tmp_path / "core"
    core.mkdir(parents=True, exist_ok=True)
    (core / "pyproject.toml").write_text(
        '[project]\nname = "fake-core"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        f"dependencies = {json.dumps(deps)}\n"
        "[tool.uv]\npackage = false\nno-index = true\n"
        f"find-links = [{json.dumps(wheels.as_posix())}]\n",
        encoding="utf-8",
    )
    subprocess.run(
        [shutil.which("uv"), "lock", "--python", sys.executable],
        cwd=core,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return core


# ── Pure declaration satisfaction against real Hermes core uv.lock ───────────


@pytest.mark.parametrize("deps", [
    ["cryptography>=50,<51"],
    ["httpx>=0.27,<1"],
    ["httpx>=0.27,<1", "PyJWT[crypto]>=2.8", "cryptography>=50,<51"],
    ["fastapi>=0.104,<1", "pydantic>=2.0,<3"],
    ["requests>=2.30,<3"],
], ids=["cryptography", "httpx", "google-shape", "fastapi-pydantic", "requests"])
def test_core_satisfied_pure_plugin_skips_workspace_membership(tmp_path, deps):
    plugin = _write_plugin(
        tmp_path / "satisfied",
        pyproject=(
            '[project]\nname = "satisfied"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            f"dependencies = {json.dumps(deps)}\n"
        ),
    )
    decl = read_python_declaration(plugin)
    assert decl.installs_packages is True
    assert unsatisfied_requirements(decl, project_root=REPO_ROOT) == ()
    assert needs_workspace_member(decl, project_root=REPO_ROOT) is False


def test_legacy_manifest_core_satisfied_plugin_skips_workspace_membership(tmp_path):
    plugin = _write_plugin(
        tmp_path / "legacy-satisfied",
        manifest='name: legacy-satisfied\npython_dependencies: ["cryptography>=50,<51", "httpx>=0.27,<1"]\n',
    )
    decl = read_python_declaration(plugin)
    assert unsatisfied_requirements(decl, project_root=REPO_ROOT) == ()
    assert needs_workspace_member(decl, project_root=REPO_ROOT) is False


def test_missing_package_requires_workspace_member(tmp_path):
    plugin = _write_plugin(
        tmp_path / "langfuse-plug",
        pyproject=(
            '[project]\nname = "langfuse-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["cryptography>=50,<51", "langfuse==4.15.2"]\n'
        ),
    )
    decl = read_python_declaration(plugin)
    assert unsatisfied_requirements(decl, project_root=REPO_ROOT) == ("langfuse==4.15.2",)
    assert needs_workspace_member(decl, project_root=REPO_ROOT) is True


def test_version_conflict_with_core_requires_workspace_member(tmp_path):
    plugin = _write_plugin(
        tmp_path / "conflict-plug",
        pyproject=(
            '[project]\nname = "conflict-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["cryptography>=99"]\n'
        ),
    )
    decl = read_python_declaration(plugin)
    assert unsatisfied_requirements(decl, project_root=REPO_ROOT) == ("cryptography>=99",)
    assert needs_workspace_member(decl, project_root=REPO_ROOT) is True


def test_build_system_with_empty_dependencies_still_requires_workspace_member(tmp_path):
    plugin = _write_plugin(
        tmp_path / "buildable",
        pyproject=(
            '[project]\nname = "buildable"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = []\n'
            '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n'
        ),
    )
    decl = read_python_declaration(plugin)
    assert decl.pyproject_installs is True
    assert needs_workspace_member(decl, project_root=REPO_ROOT) is True


def test_tool_uv_package_true_with_core_satisfied_deps_still_requires_workspace_member(tmp_path):
    plugin = _write_plugin(
        tmp_path / "uv-pkg",
        pyproject=(
            '[project]\nname = "uv-pkg"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["cryptography>=50,<51"]\n'
            "[tool.uv]\npackage = true\n"
        ),
    )
    decl = read_python_declaration(plugin)
    assert decl.pyproject_installs is True
    assert needs_workspace_member(decl, project_root=REPO_ROOT) is True


def test_vcs_or_url_source_requires_workspace_member(tmp_path):
    # uv workspace VCS/URL/path override via [tool.uv.sources] (direct dep)
    plugin = _write_plugin(
        tmp_path / "vcs-plug",
        pyproject=(
            '[project]\nname = "vcs-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["cryptography>=50,<51"]\n'
            '[tool.uv.sources]\ncryptography = { git = "https://github.com/pyca/cryptography.git" }\n'
        ),
    )
    decl = read_python_declaration(plugin)
    assert unsatisfied_requirements(decl, project_root=REPO_ROOT) == ("cryptography>=50,<51",)
    assert needs_workspace_member(decl, project_root=REPO_ROOT) is True

    # Direct PEP 508 URL requirement paired with a core-satisfied dep via a real pyproject.toml
    # must NOT be bypassed by applicable_requirements() URL stripping.
    mixed_url_plugin = _write_plugin(
        tmp_path / "mixed-url-plug",
        pyproject=(
            '[project]\nname = "mixed-url-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["httpx>=0.27,<1", "custom @ https://example.invalid/p.whl"]\n'
        ),
    )
    mixed_decl = read_python_declaration(mixed_url_plugin)
    assert unsatisfied_requirements(mixed_decl, project_root=REPO_ROOT) == (
        "custom @ https://example.invalid/p.whl",
    )
    assert needs_workspace_member(mixed_decl, project_root=REPO_ROOT) is True

    # Direct PEP 508 URL requirement alone via a real pyproject.toml.
    url_only_plugin = _write_plugin(
        tmp_path / "url-only-plug",
        pyproject=(
            '[project]\nname = "url-only-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["cryptography @ https://example.invalid/cryptography-50.0.2-py3-none-any.whl"]\n'
        ),
    )
    url_only_decl = read_python_declaration(url_only_plugin)
    assert unsatisfied_requirements(url_only_decl, project_root=REPO_ROOT) == (
        "cryptography @ https://example.invalid/cryptography-50.0.2-py3-none-any.whl",
    )
    assert needs_workspace_member(url_only_decl, project_root=REPO_ROOT) is True

    # Marker-gated direct URL only applies on the matching target platform.
    gated_url_plugin = _write_plugin(
        tmp_path / "gated-url-plug",
        pyproject=(
            '[project]\nname = "gated-url-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["httpx>=0.27,<1", "winpkg @ https://example.invalid/w.whl ; sys_platform == \'win32\'"]\n'
        ),
    )
    gated_decl = read_python_declaration(gated_url_plugin)
    linux_env = {
        "implementation_name": "cpython", "implementation_version": "3.14.9",
        "os_name": "posix", "platform_machine": "x86_64", "platform_python_implementation": "CPython",
        "platform_release": "6.8.0", "platform_system": "Linux", "platform_version": "",
        "python_full_version": "3.14.9", "python_version": "3.14", "sys_platform": "linux",
    }
    win_env = {
        **linux_env, "os_name": "nt", "platform_machine": "AMD64",
        "platform_release": "11", "platform_system": "Windows", "sys_platform": "win32",
    }
    assert needs_workspace_member(gated_decl, project_root=REPO_ROOT, target_env=linux_env) is False
    assert needs_workspace_member(gated_decl, project_root=REPO_ROOT, target_env=win_env) is True


def test_transitive_tool_uv_sources_override_requires_workspace_member(tmp_path):
    """Any non-empty ``[tool.uv.sources]`` (even overriding a transitive like ``socksio`` under
    ``httpx[socks]``) must conservatively require a workspace member."""
    plugin = _write_plugin(
        tmp_path / "transitive-uv-source",
        pyproject=(
            '[project]\nname = "transitive-uv-source"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["httpx[socks]>=0.27,<1"]\n'
            '[tool.uv.sources]\nsocksio = { git = "https://example.invalid/socksio.git" }\n'
        ),
    )
    decl = read_python_declaration(plugin)
    assert unsatisfied_requirements(decl, project_root=REPO_ROOT) == ("httpx[socks]>=0.27,<1",)
    assert needs_workspace_member(decl, project_root=REPO_ROOT) is True


def test_unverified_extra_is_conservatively_unsatisfied(tmp_path):
    # httpx[http2] is NOT installed by core (core only installs httpx[socks])
    plugin = _write_plugin(
        tmp_path / "extra-plug",
        pyproject=(
            '[project]\nname = "extra-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["httpx[http2]>=0.27,<1"]\n'
        ),
    )
    decl = read_python_declaration(plugin)
    assert unsatisfied_requirements(decl, project_root=REPO_ROOT) == ("httpx[http2]>=0.27,<1",)
    assert needs_workspace_member(decl, project_root=REPO_ROOT) is True

    # Against bare base_site_packages (no uv.lock extra provenance), any extra is conservatively unsatisfied.
    site_dir = tmp_path / "site-packages"
    _make_dist_info(site_dir, "pkg", "1.0.0")
    extra_site_plug = _write_plugin(
        tmp_path / "site-extra",
        manifest='name: site-extra\npython_dependencies: ["pkg[foo]>=1"]\n',
    )
    site_decl = read_python_declaration(extra_site_plug)
    assert unsatisfied_requirements(site_decl, base_site_packages=site_dir) == ("pkg[foo]>=1",)
    assert needs_workspace_member(site_decl, base_site_packages=site_dir) is True


def test_incompatible_requires_python_never_skips_workspace_membership(tmp_path):
    plugin = _write_plugin(
        tmp_path / "old-py",
        pyproject=(
            '[project]\nname = "old-py"\nversion = "1.0.0"\nrequires-python = "<3.12"\n'
            'dependencies = ["cryptography>=50,<51"]\n'
        ),
    )
    decl = read_python_declaration(plugin)
    assert needs_workspace_member(decl, project_root=REPO_ROOT) is True


def test_missing_or_unreadable_base_environment_fails_conservatively(tmp_path):
    plugin = _write_plugin(
        tmp_path / "plug",
        manifest='name: plug\npython_dependencies: ["cryptography>=50,<51"]\n',
    )
    decl = read_python_declaration(plugin)
    missing_site = tmp_path / "does-not-exist"
    assert unsatisfied_requirements(decl, base_site_packages=missing_site) == ("cryptography>=50,<51",)
    assert needs_workspace_member(decl, base_site_packages=missing_site) is True

    empty_root = tmp_path / "no-lock-root"
    empty_root.mkdir()
    assert unsatisfied_requirements(decl, project_root=empty_root) == ("cryptography>=50,<51",)
    assert needs_workspace_member(decl, project_root=empty_root) is True


def test_sealed_payload_intersects_core_lock_with_payload_site_packages(tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "corepkg", "1.2.0")
    _fixtures._wheel(wheels, "extrapkg", "2.0.0")
    payload = tmp_path / "payload"
    core = payload / "hermes"
    core.mkdir(parents=True)
    (core / "pyproject.toml").write_text(
        '[project]\nname = "fake-core"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["corepkg==1.2.0"]\n'
        '[project.optional-dependencies]\nopt = ["extrapkg==2.0.0"]\n'
        "[tool.uv]\npackage = false\nno-index = true\n"
        f"find-links = [{json.dumps(wheels.as_posix())}]\n",
        encoding="utf-8",
    )
    subprocess.run(
        [shutil.which("uv"), "lock", "--python", sys.executable],
        cwd=core,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    pvenv = payload / "venv"
    (pvenv / "pyvenv.cfg").parent.mkdir(parents=True)
    (pvenv / "pyvenv.cfg").write_text(
        f"version = {sys.version_info.major}.{sys.version_info.minor}.0\n",
        encoding="utf-8",
    )
    from pm.environments import site_packages

    psite = site_packages(pvenv)
    _make_dist_info(psite, "corepkg", "1.2.0")
    _make_dist_info(psite, "extrapkg", "2.0.0")
    (payload / "manifest.json").write_text(
        json.dumps({"repo": "hermes", "venv": "venv", "store": "tools", "runtime": {"commands": {"hermes": "bin/hermes"}}}),
        encoding="utf-8",
    )

    core_plug = _write_plugin(tmp_path / "core-plug", manifest='name: core-plug\npython_dependencies: ["corepkg>=1,<2"]\n')
    extra_plug = _write_plugin(tmp_path / "extra-plug", manifest='name: extra-plug\npython_dependencies: ["extrapkg>=2,<3"]\n')

    # corepkg is in BOTH uv.lock base closure AND payload_venv -> skipped
    assert needs_workspace_member(read_python_declaration(core_plug), project_root=core) is False
    # extrapkg is ONLY in a core optional extra (not base closure), even though present in payload_venv -> NOT skipped
    assert needs_workspace_member(read_python_declaration(extra_plug), project_root=core) is True

    # If payload_venv's site-packages loses corepkg, fail conservatively -> NOT skipped
    shutil.rmtree(psite / "corepkg-1.2.0.dist-info")
    assert needs_workspace_member(read_python_declaration(core_plug), project_root=core) is True


# ── End-to-end PM admission + rebuild skipping + isolation from other plugins ──


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_enabling_core_satisfied_plugin_skips_venv_rebuild_and_never_piggybacks_on_other_plugins(
    tmp_path, monkeypatch
):
    """Enabling a plugin whose deps are in core's base lock skips Venv.apply (zero rebuild, zero uv).
    Enabling a plugin whose dep is ONLY installed by another plugin still joins the workspace and
    triggers a rebuild so removing the first plugin can never break the second."""
    import importlib
    from hermes_cli import plugins_admission as admission
    from pm.environments import runtime_facts_path, selected_venv
    from pm.install import sync_venv, venv_is_current
    from pm.lock import Facts
    from pm.packages import Venv
    from pm.receipt import latest
    import pm.paths

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "corelib", "50.0.2")
    _fixtures._wheel(wheels, "plugonly", "1.0.0")
    core = _locked_core(tmp_path, ["corelib==50.0.2"], wheels)

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")

    ensure = importlib.import_module("pm.install")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setattr(pm.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr("pm.client.sync_venv", lambda *a, project_root=None, **kw: ensure.sync_venv(*a, **kw))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (Path(shutil.which("uv")), Path(sys.executable)))

    # Initial core-only sync.
    sync_venv(explicit=True)
    base_gen = selected_venv(core)
    base_fact = Facts(runtime_facts_path(core), strict=True).get("venv")

    # 1. Core-satisfied plugin -> MUST NOT rebuild venv (zero Venv.apply calls).
    satisfied = _write_plugin(
        home / "plugins" / "mobile-connector",
        name="mobile-connector",
        pyproject=(
            '[project]\nname = "mobile-connector"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["corelib>=50,<51"]\n'
        ),
    )
    apply_calls = []
    real_apply = Venv.apply

    def spy_apply(self, *args, **kwargs):
        apply_calls.append((args, kwargs))
        return real_apply(self, *args, **kwargs)

    monkeypatch.setattr(Venv, "apply", spy_apply)

    admission.admit_plugin_set_change({"mobile-connector"}, set(), active_plugins_dir=home / "plugins")

    assert apply_calls == [], "core-satisfied plugin must not trigger Venv.apply / venv rebuild"
    assert selected_venv(core) == base_gen
    assert Facts(runtime_facts_path(core), strict=True).get("venv") == base_fact
    assert yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["plugins"]["enabled"] == [
        "mobile-connector"
    ]
    assert latest()["venv_rebuild"] == {"ok": False, "reason": "already in sync"}
    assert ws.enabled_member_dirs() == []
    assert venv_is_current(project_root=core) is True
    assert satisfied.is_dir()

    # 2. Plugin A introduces `plugonly==1.0.0` (not in core) -> MUST join workspace & rebuild.
    plug_a = _write_plugin(
        home / "plugins" / "plug-a",
        name="plug-a",
        pyproject=(
            '[project]\nname = "plug-a"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["plugonly==1.0.0"]\n'
            f'[tool.uv]\nno-index = true\nfind-links = [{json.dumps(wheels.as_posix())}]\n'
        ),
    )
    admission.admit_plugin_set_change({"mobile-connector", "plug-a"}, set(), active_plugins_dir=home / "plugins")
    assert len(apply_calls) == 1
    gen_with_a = selected_venv(core)
    assert gen_with_a != base_gen
    # Once plug-a forces a shared solve, mobile-connector joins too so its corelib>=50,<51 pin stays enforced.
    assert ws.enabled_member_dirs() == [satisfied, plug_a]

    # 3. Plugin B ALSO needs `plugonly>=1,<2` (currently installed in gen_with_a ONLY because of plug-a!).
    #    Plugin B MUST STILL be a workspace member so removing plug-a never breaks plug-b.
    plug_b = _write_plugin(
        home / "plugins" / "plug-b",
        name="plug-b",
        pyproject=(
            '[project]\nname = "plug-b"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["plugonly>=1,<2"]\n'
        ),
    )
    assert needs_workspace_member(read_python_declaration(plug_b), project_root=core) is True
    admission.admit_plugin_set_change(
        {"mobile-connector", "plug-a", "plug-b"}, set(), active_plugins_dir=home / "plugins"
    )
    assert len(apply_calls) == 2
    assert ws.enabled_member_dirs() == [satisfied, plug_a, plug_b]

    # 4. Core upgrades and bumps corelib to 51.0.0 -> mobile-connector's >=50,<51 is now UNSATISFIED!
    _fixtures._wheel(wheels, "corelib", "51.0.0")
    _locked_core(tmp_path, ["corelib==51.0.0"], wheels)
    assert needs_workspace_member(read_python_declaration(satisfied), project_root=core) is True
    assert satisfied in ws.enabled_member_dirs()


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_incompatible_requires_python_with_core_satisfied_dep_is_still_refused_at_admission(
    tmp_path, monkeypatch
):
    """Skipping rebuild for core-satisfied deps must never bypass ``requires-python`` refusal."""
    import importlib
    from hermes_cli import plugins_admission as admission
    import pm.paths

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "corelib", "50.0.2")
    core = _locked_core(tmp_path, ["corelib==50.0.2"], wheels)

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")

    ensure = importlib.import_module("pm.install")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setattr(pm.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr("pm.client.sync_venv", lambda *a, project_root=None, **kw: ensure.sync_venv(*a, **kw))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (Path(shutil.which("uv")), Path(sys.executable)))

    _write_plugin(
        home / "plugins" / "too-old-py",
        name="too-old-py",
        pyproject=(
            '[project]\nname = "too-old-py"\nversion = "1.0.0"\n'
            f'requires-python = "<{sys.version_info[0]}.{sys.version_info[1]}"\n'
            'dependencies = ["corelib>=50,<51"]\n'
        ),
    )

    with pytest.raises(admission.AdmissionRefused, match="Python"):
        admission.admit_plugin_set_change({"too-old-py"}, set(), active_plugins_dir=home / "plugins")
    assert yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["plugins"]["enabled"] == []


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_sealed_bundle_first_launch_default_plugin_skips_uv_and_cache_seed(tmp_path, monkeypatch):
    """On a sealed bundle with baked store facts and payload_venv, enabling a core-satisfied plugin
    must skip Venv.apply and never touch uv_cache_dir() (zero uv-cache seed copy)."""
    import importlib
    from hermes_cli import plugins_admission as admission
    from pm.environments import site_packages
    from pm.lock import Facts
    from pm.packages import Venv
    from pm.receipt import latest
    import pm.packages
    import pm.paths

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "corelib", "50.0.2")

    payload = tmp_path / "payload"
    core = payload / "hermes"
    core.mkdir(parents=True)
    (core / "pyproject.toml").write_text(
        '[project]\nname = "fake-core"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["corelib==50.0.2"]\n'
        "[tool.uv]\npackage = false\nno-index = true\n"
        f"find-links = [{json.dumps(wheels.as_posix())}]\n",
        encoding="utf-8",
    )
    subprocess.run(
        [shutil.which("uv"), "lock", "--python", sys.executable],
        cwd=core,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    pvenv = payload / "venv"
    pvenv.mkdir(parents=True)
    (pvenv / "pyvenv.cfg").write_text(
        f"version = {sys.version_info.major}.{sys.version_info.minor}.0\n",
        encoding="utf-8",
    )
    _make_dist_info(site_packages(pvenv), "corelib", "50.0.2")
    store = payload / "tools"
    store.mkdir()
    (payload / "manifest.json").write_text(
        json.dumps({"repo": "hermes", "venv": "venv", "store": "tools", "runtime": {"commands": {"hermes": "bin/hermes"}}}),
        encoding="utf-8",
    )

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")

    ensure = importlib.import_module("pm.install")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(pm.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr("pm.client.sync_venv", lambda *a, project_root=None, **kw: ensure.sync_venv(*a, **kw))

    # Bake store facts the way scripts/bundles/native.py does at build time.
    Facts(store / "facts.json").record_state("venv", Venv(core).expected_stamp([], plugin_dirs=[]), [])

    _write_plugin(
        home / "plugins" / "mobile-connector",
        name="mobile-connector",
        pyproject=(
            '[project]\nname = "mobile-connector"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["corelib>=50,<51"]\n'
        ),
    )

    monkeypatch.setattr(Venv, "apply", lambda *a, **kw: pytest.fail("Venv.apply must not run"))
    monkeypatch.setattr(pm.packages, "uv_cache_dir", lambda: pytest.fail("uv_cache_dir must not run"))

    admission.admit_plugin_set_change({"mobile-connector"}, set(), active_plugins_dir=home / "plugins")

    assert yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["plugins"]["enabled"] == [
        "mobile-connector"
    ]
    assert latest()["venv_rebuild"] == {"ok": False, "reason": "already in sync"}


# ── Consent, dashboard summary, and unattended update parity ─────────────────


def test_consent_and_update_skip_prompt_for_core_satisfied_deps_while_summary_stays_accurate(tmp_path, monkeypatch):
    from hermes_cli import plugins_cmd_install, plugins_transaction

    plugin = _write_plugin(
        tmp_path / "sat-plug",
        name="sat-plug",
        pyproject=(
            '[project]\nname = "sat-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["cryptography>=50,<51", "httpx>=0.27,<1"]\n'
        ),
    )
    monkeypatch.setattr(plugins_cmd_install.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(plugins_cmd_install.sys.stdout, "isatty", lambda: False)
    lines: list[str] = []
    console = SimpleNamespace(print=lambda *a, **kw: lines.extend(str(x) for x in a))

    # Non-interactive consent passes immediately because nothing will be installed.
    consented, reason = plugins_cmd_install._install_plugin_python_deps(
        {"name": "sat-plug"}, plugin, console
    )
    assert (consented, reason) == (True, None)
    assert not any("declares Python dependencies" in line for line in lines)

    # Dashboard summary still reports the plugin's declared dependencies accurately.
    warnings: list[str] = []
    assert plugins_cmd_install._python_dependency_summary(plugin, warnings) == [
        "cryptography>=50,<51",
        "httpx>=0.27,<1",
    ]
    assert warnings == []

    # Unattended update adding a core-satisfied dep succeeds without refusal.
    before = _write_plugin(tmp_path / "before", name="sat-plug")
    monkeypatch.setattr(ws, "enabled_plugin_dirs", lambda **kw: [before])
    plugins_transaction._refresh_declared_dependencies(
        before, plugin, {"name": "sat-plug"}, interactive=False
    )

    # Unattended update adding an unsatisfied dep STILL refuses.
    unsat = _write_plugin(
        tmp_path / "unsat",
        name="sat-plug",
        pyproject=(
            '[project]\nname = "sat-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["langfuse==4.15.2"]\n'
        ),
    )
    from hermes_cli import plugins_cmd

    with pytest.raises(plugins_cmd.PluginOperationError, match="langfuse==4.15.2"):
        plugins_transaction._refresh_declared_dependencies(
            before, unsat, {"name": "sat-plug"}, interactive=False
        )


# ── Pre-push review regression tests (Findings #1, #2, #3) ───────────────────


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_generation_workspace_never_shadows_canonical_core_root(tmp_path, monkeypatch):
    """Finding #1: running from ``environments/<gen>/workspace`` whose ``uv.lock`` includes a
    plugin package (``langfuse==4.15.2``) must read the canonical core root, never the generation
    workspace's lock."""
    import pm.paths
    from pm.environments import install_key

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "corelib", "50.0.2")
    _fixtures._wheel(wheels, "langfuse", "4.15.2")

    real_core = _locked_core(tmp_path / "real-checkout", ["corelib==50.0.2"], wheels)

    state = tmp_path / "home" / "installs" / install_key(real_core)
    gen_ws = state / "environments" / "gen01" / "workspace"
    gen_ws.mkdir(parents=True)
    (gen_ws / "pyproject.toml").write_text(
        '[project]\nname = "fake-core"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["corelib==50.0.2", "langfuse==4.15.2"]\n'
        "[tool.uv]\npackage = false\nno-index = true\n"
        f"find-links = [{json.dumps(wheels.as_posix())}]\n",
        encoding="utf-8",
    )
    subprocess.run(
        [shutil.which("uv"), "lock", "--python", sys.executable],
        cwd=gen_ws,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    langfuse_plug = _write_plugin(
        tmp_path / "langfuse-plug",
        name="langfuse-plug",
        pyproject=(
            '[project]\nname = "langfuse-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["langfuse==4.15.2"]\n'
        ),
    )
    core_plug = _write_plugin(
        tmp_path / "core-plug",
        name="core-plug",
        pyproject=(
            '[project]\nname = "core-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["corelib>=50,<51"]\n'
        ),
    )
    langfuse_decl = read_python_declaration(langfuse_plug)
    core_decl = read_python_declaration(core_plug)

    monkeypatch.setattr(pm.paths, "repo_root", lambda: gen_ws)

    # 1a. HERMES_INSTALL_ROOT points at the canonical core root.
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(real_core))
    assert needs_workspace_member(langfuse_decl) is True
    assert needs_workspace_member(core_decl) is False

    # 1b. No HERMES_INSTALL_ROOT, but state/inputs/.project-root names the canonical core root.
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    (state / "inputs").mkdir(parents=True, exist_ok=True)
    (state / "inputs" / ".project-root").write_text(str(real_core.resolve()), encoding="utf-8")
    assert needs_workspace_member(langfuse_decl) is True
    assert needs_workspace_member(core_decl) is False

    # 1c. Neither HERMES_INSTALL_ROOT nor .project-root: fail conservatively, never trust gen_ws/uv.lock.
    (state / "inputs" / ".project-root").unlink()
    assert needs_workspace_member(langfuse_decl) is True
    assert needs_workspace_member(core_decl) is True


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_core_satisfied_plugin_constraints_survive_when_another_plugin_forces_shared_solve(
    tmp_path, monkeypatch
):
    """Finding #2: plugin A (core-satisfied ``idna==3.20``) alone produces ``members == []``,
    but when plugin B (unsatisfied ``idna==3.18``) forces a shared solve, BOTH A and B stay in
    ``members`` so uv catches the conflict instead of downgrading ``idna`` and breaking A."""
    import importlib
    from hermes_cli import plugins_admission as admission
    from pm.install import sync_venv
    from pm.publication import candidate_members
    import pm.paths

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "idna", "3.20")
    _fixtures._wheel(wheels, "idna", "3.18")
    core = _locked_core(tmp_path, ["idna>=3.18,<4"], wheels)

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: [plug-a]\n", encoding="utf-8")

    plug_a = _write_plugin(
        home / "plugins" / "plug-a",
        name="plug-a",
        pyproject=(
            '[project]\nname = "plug-a"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["idna==3.20"]\n'
        ),
    )
    plug_b = _write_plugin(
        home / "plugins" / "plug-b",
        name="plug-b",
        pyproject=(
            '[project]\nname = "plug-b"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["idna==3.18"]\n'
        ),
    )

    ensure = importlib.import_module("pm.install")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    monkeypatch.setattr(pm.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr("pm.client.sync_venv", lambda *a, project_root=None, **kw: ensure.sync_venv(*a, **kw))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (Path(shutil.which("uv")), Path(sys.executable)))

    # Plugin A alone (core locked idna==3.20): zero members -> zero rebuild.
    assert needs_workspace_member(read_python_declaration(plug_a), project_root=core) is False
    assert needs_workspace_member(read_python_declaration(plug_b), project_root=core) is True
    assert ws.enabled_member_dirs() == []
    assert candidate_members() == []

    # Plugin A + Plugin B: B forces a shared solve -> BOTH A and B MUST be in members!
    (home / "config.yaml").write_text("plugins:\n  enabled: [plug-a, plug-b]\n", encoding="utf-8")
    assert ws.enabled_member_dirs() == [plug_a, plug_b]
    assert candidate_members() == [plug_a, plug_b]

    # End-to-end: sync core with plug-a enabled, then attempt to admit plug-b -> uv MUST detect
    # the idna==3.20 vs idna==3.18 conflict rather than silently downgrading idna to 3.18!
    (home / "config.yaml").write_text("plugins:\n  enabled: [plug-a]\n", encoding="utf-8")
    sync_venv(explicit=True)
    with pytest.raises(admission.DependencyConflict):
        admission.admit_plugin_set_change({"plug-a", "plug-b"}, set(), active_plugins_dir=home / "plugins")


def test_admission_evaluates_markers_against_target_interpreter_not_worker_python(tmp_path, monkeypatch):
    """Finding #3: worker running Python 3.14.6 admitting a plugin for target managed Python
    3.14.9 with ``fixturedep; python_full_version >= '3.14.8'`` MUST include the plugin."""
    import packaging.markers
    import pm.paths
    from pm.publication import candidate_members

    home = tmp_path / "home"
    home.mkdir()
    plugin = _write_plugin(
        home / "plugins" / "marker-plug",
        name="marker-plug",
        pyproject=(
            '[project]\nname = "marker-plug"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["fixturedep>=1; python_full_version >= \'3.14.8\'"]\n'
        ),
    )
    (home / "config.yaml").write_text("plugins:\n  enabled: [marker-plug]\n", encoding="utf-8")

    worker_env = {
        "implementation_name": "cpython", "implementation_version": "3.14.6",
        "os_name": "posix", "platform_machine": "x86_64", "platform_python_implementation": "CPython",
        "platform_release": "6.8.0", "platform_system": "Linux", "platform_version": "",
        "python_full_version": "3.14.6", "python_version": "3.14", "sys_platform": "linux",
    }
    target_env = {
        **worker_env,
        "implementation_version": "3.14.9",
        "python_full_version": "3.14.9",
    }
    fake_target_py = tmp_path / "managed" / "python3"

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    monkeypatch.setattr(pm.paths, "repo_root", lambda: REPO_ROOT)
    monkeypatch.setattr(ws.paths, "repo_root", lambda: REPO_ROOT)
    monkeypatch.setattr(packaging.markers, "default_environment", lambda: dict(worker_env))

    # Case A: managed target Python 3.14.9 is installed and probed via _toolchain(realize=False).
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (tmp_path / "uv", fake_target_py))
    monkeypatch.setattr(ws, "_target_environment", lambda py: dict(target_env) if py == fake_target_py else dict(worker_env))
    assert candidate_members() == [plugin]
    assert ws.enabled_member_dirs() == [plugin]

    # Case B: target interpreter cannot be determined (_toolchain returns None) -> conservatively
    # treat marker-sensitive requirement as needing admission rather than dropping it on 3.14.6.
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: None)
    assert candidate_members() == [plugin]
    assert ws.enabled_member_dirs() == [plugin]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_eviction_retry_keeps_core_satisfied_enabled_plugin_constraints(tmp_path, monkeypatch):
    """Greptile P1: when enabled plugin A is core-satisfied (``idna==3.20``) and disabled
    plugin B (``idna==3.18``) is due for retry, eviction retry MUST include A's constraints
    in the shared solve so uv catches the conflict instead of downgrading ``idna``."""
    import importlib
    import pm.paths
    from pm.environments import selected_venv, site_packages
    from pm.install import sync_venv
    from pm.packages import Venv

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "idna", "3.20")
    _fixtures._wheel(wheels, "idna", "3.18")
    core = _locked_core(tmp_path, ["idna>=3.18,<4"], wheels)

    home = tmp_path / "home"
    home.mkdir()
    plug_a = _write_plugin(
        home / "plugins" / "plug-a",
        name="plug-a",
        pyproject=(
            '[project]\nname = "plug-a"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["idna==3.20"]\n'
        ),
    )
    _write_plugin(
        home / "plugins" / "plug-b",
        name="plug-b",
        pyproject=(
            '[project]\nname = "plug-b"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["idna==3.18"]\n'
        ),
    )

    ensure = importlib.import_module("pm.install")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    monkeypatch.setattr(pm.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (Path(shutil.which("uv")), Path(sys.executable)))

    # Initial sync with plug-a enabled (core-satisfied -> idna==3.20 installed).
    (home / "config.yaml").write_text("plugins:\n  enabled: [plug-a]\n", encoding="utf-8")
    sync_venv(explicit=True)
    base_gen = selected_venv(core)
    assert any(p.name.startswith("idna-3.20") for p in site_packages(base_gen).iterdir())

    # Record plug-b as previously evicted and due for retry.
    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled: [plug-a]\n"
        "  disabled: [plug-b]\n"
        "  disabled_by_update:\n"
        "    plug-b:\n"
        "      enabled: true\n"
        "      reason: earlier conflict\n",
        encoding="utf-8",
    )

    sync_venv(explicit=True, evict_incompatible_plugins=True)

    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["plugins"]
    assert cfg["enabled"] == ["plug-a"]
    assert cfg["disabled"] == ["plug-b"]
    assert "plug-b" in cfg["disabled_by_update"]
    assert selected_venv(core) == base_gen
    assert any(p.name.startswith("idna-3.20") for p in site_packages(selected_venv(core)).iterdir())

    # A follow-up update with nothing changed must NOT retry plug-b again or rebuild.
    builds: list[list[str]] = []
    real_apply = Venv.apply

    def spy_apply(self, extras, *, plugin_dirs=None, **kwargs):
        builds.append([Path(p).name for p in plugin_dirs or ()])
        return real_apply(self, extras, plugin_dirs=plugin_dirs, **kwargs)

    monkeypatch.setattr(Venv, "apply", spy_apply)
    sync_venv(explicit=True, evict_incompatible_plugins=True)
    assert builds == []

    # Once plug-a relaxes its constraint, the fingerprint changes -> plug-b retries and rejoins!
    (plug_a / "pyproject.toml").write_text(
        '[project]\nname = "plug-a"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["idna>=3.18,<4"]\n',
        encoding="utf-8",
    )
    sync_venv(explicit=True, evict_incompatible_plugins=True)
    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["plugins"]
    assert cfg["enabled"] == ["plug-a", "plug-b"]
    assert not cfg.get("disabled")
    assert "disabled_by_update" not in cfg


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_eviction_retry_keeps_statically_rejoining_core_satisfied_plugin_constraints(tmp_path, monkeypatch):
    """A core-satisfied plugin C rejoining without a trial must still contribute its constraints
    when a sibling retried plugin B forces a workspace solve; when only C rejoins, no rebuild runs."""
    import importlib
    import pm.paths
    from pm.environments import selected_venv
    from pm.install import sync_venv
    from pm.packages import Venv

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "idna", "3.20")
    _fixtures._wheel(wheels, "idna", "3.18")
    core = _locked_core(tmp_path, ["idna>=3.18,<4"], wheels)

    home = tmp_path / "home"
    home.mkdir()
    _write_plugin(
        home / "plugins" / "plug-c",
        name="plug-c",
        pyproject=(
            '[project]\nname = "plug-c"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["idna==3.20"]\n'
        ),
    )
    _write_plugin(
        home / "plugins" / "plug-b",
        name="plug-b",
        pyproject=(
            '[project]\nname = "plug-b"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["idna==3.18"]\n'
        ),
    )

    ensure = importlib.import_module("pm.install")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    monkeypatch.setattr(pm.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (Path(shutil.which("uv")), Path(sys.executable)))

    (home / "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    sync_venv(explicit=True)
    base_gen = selected_venv(core)

    # Case 1: only core-satisfied plug-c rejoins -> zero Venv.apply calls.
    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled: []\n"
        "  disabled: [plug-c]\n"
        "  disabled_by_update:\n"
        "    plug-c:\n"
        "      enabled: true\n"
        "      reason: earlier misfit\n",
        encoding="utf-8",
    )
    real_apply = Venv.apply
    rejoin_only_calls: list[list[str]] = []

    def spy_apply(self, extras, *, plugin_dirs=None, **kwargs):
        rejoin_only_calls.append([Path(p).name for p in plugin_dirs or ()])
        return real_apply(self, extras, plugin_dirs=plugin_dirs, **kwargs)

    monkeypatch.setattr(Venv, "apply", spy_apply)
    sync_venv(explicit=True, evict_incompatible_plugins=True)
    assert rejoin_only_calls == []
    assert selected_venv(core) == base_gen
    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["plugins"]
    assert cfg["enabled"] == ["plug-c"]

    # Case 2: BOTH core-satisfied plug-c (rejoins statically) and conflicting plug-b (due for retry).
    monkeypatch.setattr(Venv, "apply", real_apply)
    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled: []\n"
        "  disabled: [plug-c, plug-b]\n"
        "  disabled_by_update:\n"
        "    plug-c:\n"
        "      enabled: true\n"
        "      reason: earlier misfit\n"
        "    plug-b:\n"
        "      enabled: true\n"
        "      reason: earlier conflict\n",
        encoding="utf-8",
    )
    sync_venv(explicit=True, evict_incompatible_plugins=True)
    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["plugins"]
    assert cfg["enabled"] == ["plug-c"]
    assert cfg["disabled"] == ["plug-b"]
    assert selected_venv(core) == base_gen


def test_assess_workspace_membership_single_source_of_truth(tmp_path):
    """Greptile P2: assess_workspace_membership returns structured reasons and agrees with
    needs_workspace_member across satisfied, build-system, uv-sources, requires-python, and
    unsatisfied-dep cases."""
    ok_plugin = _write_plugin(
        tmp_path / "ok",
        pyproject=(
            '[project]\nname = "ok"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["cryptography>=50,<51"]\n'
        ),
    )
    ok_decl = read_python_declaration(ok_plugin)
    ok_res = assess_workspace_membership(ok_decl, project_root=REPO_ROOT)
    assert isinstance(ok_res, WorkspaceMembershipAssessment)
    assert ok_res == WorkspaceMembershipAssessment(
        needs_member=False,
        reasons=(),
        unsatisfied_requirements=(),
        pyproject_installs=False,
        python_error=None,
    )
    assert needs_workspace_member(ok_decl, project_root=REPO_ROOT) is ok_res.needs_member

    multi_plugin = _write_plugin(
        tmp_path / "multi",
        pyproject=(
            '[project]\nname = "multi"\nversion = "1.0.0"\nrequires-python = "<3.12"\n'
            'dependencies = ["langfuse==4.15.2"]\n'
            '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n'
            '[tool.uv.sources]\nlangfuse = { git = "https://example.invalid/langfuse.git" }\n'
        ),
    )
    multi_decl = read_python_declaration(multi_plugin)
    multi_res = assess_workspace_membership(multi_decl, project_root=REPO_ROOT)
    assert multi_res.needs_member is True
    assert multi_res.pyproject_installs is True
    assert multi_res.unsatisfied_requirements == ("langfuse==4.15.2",)
    assert multi_res.python_error is not None and "<3.12" in multi_res.python_error
    assert any("[build-system]" in r for r in multi_res.reasons)
    assert any("[tool.uv.sources]" in r for r in multi_res.reasons)
    assert any("langfuse==4.15.2" in r for r in multi_res.reasons)
    assert needs_workspace_member(multi_decl, project_root=REPO_ROOT) is multi_res.needs_member


