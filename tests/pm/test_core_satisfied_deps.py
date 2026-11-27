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
    core.mkdir(exist_ok=True)
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
    # uv workspace VCS/URL/path override via [tool.uv.sources]
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

    # Direct PEP 508 URL requirement passed into a declaration object is never treated as satisfied.
    class _UrlDecl(PythonDeclaration):
        @property
        def install_requirements(self) -> tuple[str, ...]:
            return ("cryptography @ https://example.invalid/cryptography-50.0.2-py3-none-any.whl",)

    url_decl = _UrlDecl(files=(), pyproject=None, requirements=(), manifest={})
    assert unsatisfied_requirements(url_decl, project_root=REPO_ROOT) == url_decl.install_requirements
    assert needs_workspace_member(url_decl, project_root=REPO_ROOT) is True


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
    assert ws.enabled_member_dirs() == [plug_a]

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
    assert ws.enabled_member_dirs() == [plug_a, plug_b]

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
