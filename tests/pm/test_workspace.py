"""pm.workspace: the generated uv-workspace root for plugin deps.

The workspace root is a pm-GENERATED project (never the committed
pyproject.toml — sealed installs are read-only and member lists are
machine-specific). Its pyproject = core's pyproject verbatim +
``[tool.uv.workspace] members`` pointing at each snapshotted plugin.
``uv lock`` unions core + plugin deps into ONE lock; conflict = loud refusal.
"""

from __future__ import annotations

import os
import subprocess
import shutil
import sys
from pathlib import Path

import pytest

import pm.workspace as ws
from pm.environment import managed_environment
from tests.pm.test_environment_build import locked_project




@pytest.fixture
def layout(locked_project, tmp_path, monkeypatch):
    core, uv, _ = locked_project
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace('[tool.uv.workspace]\nmembers=["member"]\n', ""))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (uv, Path(sys.executable)))
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    return tmp_path, core, core / "member", tmp_path / "store"




def test_preparation_refuses_existing_workspace_without_mutating_it(layout):
    from pm.package import InstallError

    tmp, core, plug_a, _ = layout
    root = tmp / "workspace"
    environment = managed_environment(tmp / "env")
    kwargs = dict(root=root, source=core, seed_lock=None,
                  environment=environment)
    ws.lock_and_sync([plug_a], [], **kwargs)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    (plug_a / "pyproject.toml").write_text('changed after publication')
    with pytest.raises(InstallError, match="fresh"):
        ws.lock_and_sync([], [], **kwargs)
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_missing_explicit_seed_cannot_silently_resolve_new_versions(layout):
    tmp, core, plug_a, _ = layout
    with pytest.raises(FileNotFoundError):
        ws.lock_and_sync([plug_a], [], root=tmp / "workspace", source=core,
                         seed_lock=tmp / "missing.lock", environment=managed_environment(tmp / "env"))
    assert not (tmp / "env").exists()


def test_plugin_workspace_trusts_core_lock_instead_of_rechecking_release_age(layout, locked_project):  # health: allow F811 -- pytest injects the imported locked_project fixture by parameter name
    """Release age is checked when Hermes locks, never on the user's machine (#135589).

    The generated root carries no ``exclude-newer`` (a mirror without upload times makes
    uv reject every file it cannot date) and adds no constraints of its own: core's ranges
    still bound what a plugin may move, and plugin-only packages stay free (#120076).
    """
    import tomllib

    tmp, core, plug_a, _ = layout
    _, uv, env = locked_project
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace("[tool.uv]\n", '[tool.uv]\nexclude-newer="3 days"\n', 1)
                        + '[tool.uv.exclude-newer-package]\nBase_Dep = false\n')
    subprocess.run([str(uv), "lock", "--python", sys.executable], cwd=core, env=env, check=True,
                   capture_output=True)
    core_lock = tomllib.loads((core / "uv.lock").read_text())["package"]
    core_packages = {p["name"]: p["version"] for p in core_lock if "registry" in p.get("source", {})}
    assert {"base-dep", "chosen-dep"} <= set(core_packages) and "member-dep" not in core_packages

    root = tmp / "workspace"
    ws.lock_and_sync([plug_a], [], root=root, source=core, seed_lock=core / "uv.lock",
                     environment=managed_environment(tmp / "env"))

    policy = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]
    assert "exclude-newer" not in policy and "exclude-newer-package" not in policy
    assert "constraint-dependencies" not in policy
    locked = {p["name"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    assert "member-dep" in locked


def test_plugin_workspace_solves_only_for_the_installing_interpreter(layout):
    """A platform the install never runs on must not fail its solve.

    uv fails the whole lock when one fork lacks wheels (onnxruntime and torch ship none for
    Android), so a desktop update evicted mnemosyne over Android (#143254, #143305, #144003).
    Core's own environment bounds still apply on top.
    """
    import tomllib

    from packaging.markers import Marker, default_environment

    tmp, core, plug_a, _ = layout
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace("[tool.uv]\n", "[tool.uv]\nenvironments=[\"python_version >= '3.11'\"]\n", 1))
    root = tmp / "workspace"
    ws.lock_and_sync([plug_a], [], root=root, source=core, seed_lock=None,
                     environment=managed_environment(tmp / "env"))

    [solved] = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]["environments"]
    host = default_environment()
    platforms = [p for p in ("android", "win32", "linux") if p != host["sys_platform"]]
    machine = next(m for m in ("riscv64", "s390x") if m != host["platform_machine"])
    for environment, expected in ((host, True), *(({**host, "sys_platform": p}, False) for p in platforms),
                                  ({**host, "platform_machine": machine}, False),
                                  ({**host, "python_version": "3.10"}, False)):
        assert Marker(solved).evaluate(environment) is expected, environment






# --- classified failures + staging surface (FINAL-RUNTIME-CONTRACT) ---


def test_plugin_local_groups_do_not_constrain_runtime_workspace(layout):
    """Local developer tools must not participate in the host's runtime lock."""
    import tomllib

    tmp, core, plugin, _ = layout
    metadata = plugin / "pyproject.toml"
    metadata.write_text(metadata.read_text() +
        '\n[project.optional-dependencies]\nfeature=["chosen-dep==1.0"]\n'
        '[dependency-groups]\ndev=["base-dep==2.0"]\nlint=[{include-group="dev"}]\n'
        '[tool.uv.dependency-groups]\ndev={requires-python=">=3.11"}\n')
    metadata.write_text(metadata.read_text().replace('[tool.uv]\n',
        '[tool.uv]\ndefault-groups=["dev"]\ndev-dependencies=["other-dep==2.0"]\n'
        'conflicts=[[{group="dev"},{extra="feature"}]]\n', 1))
    original = metadata.read_bytes()
    root = tmp / "runtime-workspace"
    environment = managed_environment(tmp / "runtime-env")
    ws.lock_and_sync([plugin], ["feature"], root=root, source=core, seed_lock=None,
                     environment=environment)
    snapshot = next((root / "plugin-sources").iterdir())
    projected = tomllib.loads((snapshot / "pyproject.toml").read_text())
    assert "dependency-groups" not in projected
    assert projected["project"]["dependencies"] == ["member-dep==1.0"]
    assert projected["project"]["optional-dependencies"] == {"feature": ["chosen-dep==1.0"]}
    assert not {"default-groups", "dev-dependencies", "dependency-groups", "conflicts"} & projected["tool"]["uv"].keys()
    assert metadata.read_bytes() == original
    locked = tomllib.loads((root / "uv.lock").read_text())["package"]
    assert any(p["name"] == "base-dep" and p["version"] == "1.0" for p in locked)
    assert any(p["name"] == "member-dep" for p in locked)

    # A buildable member keeps its original package identity and runtime metadata too.
    metadata.write_text(metadata.read_text() +
        '\n[build-system]\nrequires=[]\nbuild-backend="local_backend"\n')
    buildable_original = metadata.read_bytes()
    buildable = ws._workspace_member(plugin, tmp / "buildable", identity=plugin)
    built = tomllib.loads((buildable / "pyproject.toml").read_text())
    assert built["project"]["name"] == "plugin-member"
    assert "dependency-groups" not in built
    assert built["project"]["dependencies"] == ["member-dep==1.0"]
    assert metadata.read_bytes() == buildable_original

    # Genuine published runtime conflicts still fail loudly; this is not a resolver bypass.
    metadata.write_text(metadata.read_text().replace('dependencies=["member-dep==1.0"]',
                                                     'dependencies=["base-dep==2.0"]'))
    with pytest.raises(ws.ResolutionConflict):
        ws.lock_and_sync([plugin], [], root=tmp / "conflicting-runtime", source=core,
                         seed_lock=None, environment=managed_environment(tmp / "conflicting-env"))


def test_unselected_plugin_extras_do_not_constrain_runtime_workspace(layout):
    """A plugin's test extra conflicting with a core pin must not fail the lock (and so evict
    the plugin on update): sync installs only the extras it names, so only those may reach it."""
    import tomllib

    tmp, core, plugin, _ = layout
    metadata = plugin / "pyproject.toml"
    metadata.write_text(metadata.read_text() +
        '\n[project.optional-dependencies]\ndev=["base-dep==2.0"]\nfeature=["chosen-dep==1.0"]\n')

    def solve(name, extras):
        root = tmp / name
        ws.lock_and_sync([plugin], extras, root=root, source=core, seed_lock=None,
                         environment=managed_environment(tmp / f"{name}-env"))
        snapshot = tomllib.loads(next((root / "plugin-sources").iterdir()).joinpath("pyproject.toml").read_text())
        locked = {p["name"]: p["version"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
        return snapshot["project"].get("optional-dependencies"), locked

    optional, locked = solve("unselected", [])
    assert optional is None
    assert locked["base-dep"] == "1.0" and "member-dep" in locked

    optional, locked = solve("selected", ["feature"])
    assert optional == {"feature": ["chosen-dep==1.0"]}
    assert locked["base-dep"] == "1.0"


def test_plugin_extras_its_own_requirements_name_stay_in_the_snapshot(layout):
    """``plugin[runtime]`` in the plugin's own dependencies installs that extra on every sync."""
    import tomllib

    tmp, _, plugin, _ = layout
    metadata = plugin / "pyproject.toml"
    metadata.write_text(metadata.read_text().replace(
        'dependencies=["member-dep==1.0"]', 'dependencies=["member-dep==1.0", "Plugin_Member[Runtime]"]') +
        '\n[project.optional-dependencies]\nruntime=["plugin-member[extra-io]"]\nextra_io=["other-dep==1.0"]\n'
        'dev=["base-dep==2.0"]\n[build-system]\nrequires=[]\nbuild-backend="local_backend"\n')
    member = ws._workspace_member(plugin, tmp / "buildable", identity=plugin)
    optional = tomllib.loads((member / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert optional == {"runtime": ["plugin-member[extra-io]"], "extra_io": ["other-dep==1.0"]}


def test_plugin_extras_named_only_for_another_platform_do_not_constrain_the_lock(layout):
    """``plugin[windows]; sys_platform == 'win32'`` installs nothing on Linux, so a Windows-only
    pin inside that extra must not fail the Linux lock (and evict the plugin on update)."""
    import tomllib

    from packaging.markers import default_environment

    tmp, core, plugin, _ = layout
    other = next(p for p in ("win32", "linux") if p != default_environment()["sys_platform"])
    metadata = plugin / "pyproject.toml"
    metadata.write_text(metadata.read_text() +
        '\n[project.optional-dependencies]\n'
        f'runtime=["plugin-member[other]; sys_platform == {other!r}", "chosen-dep==1.0"]\n'
        'other=["base-dep==2.0"]\n')
    root = tmp / "workspace"
    ws.lock_and_sync([plugin], ["runtime"], root=root, source=core, seed_lock=None,
                     environment=managed_environment(tmp / "env"))
    snapshot = tomllib.loads(next((root / "plugin-sources").iterdir()).joinpath("pyproject.toml").read_text())
    # Still declared, so a build backend validating the reference accepts it, but empty.
    assert snapshot["project"]["optional-dependencies"]["other"] == []
    locked = {p["name"]: p["version"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    assert locked["base-dep"] == "1.0" and "chosen-dep" in locked


def test_base_self_reference_gated_on_a_selected_extra_keeps_its_extra(layout):
    """``plugin[feature]; extra == 'runtime'`` in the base dependencies installs ``feature``
    whenever ``runtime`` is installed, so its pins must stay in the snapshot."""
    import tomllib

    from packaging.markers import default_environment

    tmp, _, plugin, _ = layout
    metadata = plugin / "pyproject.toml"
    metadata.write_text(metadata.read_text().replace(
        'dependencies=["member-dep==1.0"]',
        'dependencies=["member-dep==1.0", "plugin-member[feature]; extra == \'runtime\'", '
        '"plugin-member[deep]; extra == \'feature\'"]') +
        '\n[project.optional-dependencies]\nruntime=[]\n'
        'feature=["chosen-dep==1.0"]\ndeep=["other-dep==1.0"]\ndev=["base-dep==2.0"]\n'
        '[build-system]\nrequires=[]\nbuild-backend="local_backend"\n')
    member = ws._workspace_member(plugin, tmp / "buildable", identity=plugin, extras=["runtime"],
                                  target=dict(default_environment()))
    optional = tomllib.loads((member / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    # ``deep`` is gated on ``feature``, which only the first reference brings in.
    assert optional == {"runtime": [], "feature": ["chosen-dep==1.0"], "deep": ["other-dep==1.0"]}


def test_classify_network_failure_stays_generic():
    from pm.workspace import ResolutionConflict, classify_uv_failure

    err = classify_uv_failure("lock", 1, "error: Failed to fetch https://pypi.org (timed out)")
    assert not isinstance(err, ResolutionConflict)


def test_core_only_generation_carries_no_release_age_cutoff(layout):
    """A core-only install (hermes update) is a frozen sync: the cutoff there only gates the build
    tools uv fetches, and a mirror without upload times rejects every one of them (#135589)."""
    tmp, core, _, _ = layout
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace("[tool.uv]\n", '[tool.uv]\nexclude-newer = "3 days"\n', 1))
    root = tmp / "workspace"
    ws._generate_pyproject([], root, source=core, python=Path(sys.executable))
    generated = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "exclude-newer" not in generated and "[tool.uv]" in generated


def test_undated_index_failure_names_the_index_not_a_conflict():
    """uv's "has no publish time" is the index omitting upload times (#135589), not a conflict."""
    from pm.workspace import ResolutionConflict, classify_uv_failure

    seeded = ("  x No solution found when resolving dependencies:\n"
              "  Because resvg-py==0.4.0 has no publish time and hermes-agent depends on resvg-py==0.4.0, "
              "we can conclude that hermes-agent's requirements are unsatisfiable.")
    cold = ("warning: cffi-2.0.0.tar.gz is missing an upload date, but user provided: 2026-09-25T00:00:00Z\n"
            "  x No solution found when resolving dependencies:\n"
            "  Because there are no versions of cffi and you require cffi>=2.0.0, we can conclude that "
            "your requirements are unsatisfiable.\n"
            "hint: `cffi` was filtered by `exclude-newer` to only include packages uploaded before 2026-09-25")
    for output in (seeded, cold):
        err = classify_uv_failure("lock", 1, output)
        assert not isinstance(err, ResolutionConflict)
        assert "upload times" in err.remedy and "pypi.org" in err.remedy
    # A real conflict on a partly undated mirror still reads as a conflict.
    conflict = cold.splitlines()[0] + "\n  x No solution found: Because a==1 depends on b<2 and c==1 depends on b>=2"
    assert isinstance(classify_uv_failure("lock", 1, conflict), ResolutionConflict)


def test_sync_failure_is_never_a_conflict(layout, monkeypatch):
    from pm.package import InstallError

    tmp, core, _, _ = layout
    environment = managed_environment(tmp / "candidate")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs:
                        subprocess.CompletedProcess(cmd, 1, "", "Failed to download wheel"))
    with pytest.raises(InstallError) as excinfo:
        ws.lock_and_sync([], [], root=tmp / "workspace", source=core,
                         seed_lock=None, environment=environment, frozen=True)
    assert not isinstance(excinfo.value, ws.ResolutionConflict)


def test_staging_root_and_env_are_honored_without_live_mutation(layout, monkeypatch):
    tmp, core, _, _ = layout
    staging = tmp / "staging-ws"
    monkeypatch.setenv("PM_WORKSPACE_TEST_SENTINEL", "live")
    environment = managed_environment(tmp / "staging-venv", env={
        "PATH": "/staged/bin", "PM_WORKSPACE_TEST_SENTINEL": "staged",
    })
    # The prepared environment is authoritative; workspace never discovers tools.
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: pytest.fail("PATH discovery"))
    seen = []
    def run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", run)
    ws.lock_and_sync([], [], root=staging, source=core, seed_lock=None, environment=environment)
    assert [cmd[1] for cmd, _ in seen] == ["lock", "sync"]
    for cmd, kwargs in seen:
        assert Path(cmd[0]) == environment.uv
        assert Path(kwargs["cwd"]) == staging
        assert kwargs["env"]["PM_WORKSPACE_TEST_SENTINEL"] == "staged"
        assert kwargs["env"]["UV_CACHE_DIR"] == str(environment.cache)
        assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(environment.destination)
        assert kwargs["env"]["UV_PYTHON"] == str(environment.python)
    assert os.environ["PM_WORKSPACE_TEST_SENTINEL"] == "live"
