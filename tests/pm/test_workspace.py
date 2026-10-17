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
    lock_doc = tomllib.loads((root / "uv.lock").read_text())
    options = lock_doc.get("options") or {}
    assert "exclude-newer" not in options and "exclude-newer-span" not in options
    locked = {p["name"] for p in lock_doc["package"]}
    assert "member-dep" in locked






# --- classified failures + staging surface (FINAL-RUNTIME-CONTRACT) ---


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
    ws._generate_pyproject([], root, source=core)
    generated = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "exclude-newer" not in generated and "[tool.uv]" in generated


def test_without_release_age_lock_strips_options_and_empty_section():
    """Lock-level exclude-newer* must not survive onto a user-side generation (#138131)."""
    raw = (
        'version = 1\n'
        '[options]\n'
        'exclude-newer = "0001-01-01T00:00:00Z" # relative span companion\n'
        'exclude-newer-span = "P3D"\n'
        '\n'
        '[manifest]\n'
        'overrides = []\n'
    )
    cleaned = ws._without_release_age_lock(raw)
    assert "exclude-newer" not in cleaned
    assert "[options]" not in cleaned
    assert "[manifest]" in cleaned


def test_seeded_lock_drops_release_age_options(layout, monkeypatch):
    """hermes update seeds core's uv.lock; that copy must not keep exclude-newer-span (#138131)."""
    import tomllib

    tmp, core, _, _ = layout
    seed = core / "uv.lock"
    assert seed.is_file()
    body = seed.read_text(encoding="utf-8")
    if "[options]" in body:
        seeded = body.replace(
            "[options]\n",
            '[options]\nexclude-newer = "0001-01-01T00:00:00Z"\nexclude-newer-span = "P3D"\n',
            1,
        )
    else:
        seeded = body.replace("version = 1\n", 'version = 1\n\n[options]\nexclude-newer-span = "P3D"\n\n', 1)
    seed.write_text(seeded, encoding="utf-8")
    assert "exclude-newer-span" in seed.read_text(encoding="utf-8")

    class _Env:
        def sync(self, root, extras, frozen=False):
            return None

    root = tmp / "workspace"
    ws.lock_and_sync([], [], root=root, source=core, seed_lock=seed, environment=_Env(), frozen=True)
    options = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8")).get("options") or {}
    assert "exclude-newer" not in options and "exclude-newer-span" not in options


def test_replayed_lock_drops_release_age_options(layout, monkeypatch):
    """Repair replay must strip a recorded lock's cutoff the same way (#138131)."""
    import tomllib

    tmp, core, _, _ = layout
    replay = tmp / "recorded"
    replay.mkdir()
    (replay / "pyproject.toml").write_text(
        '[project]\nname="r"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=[]\n[tool.uv]\npackage=false\nexclude-newer = "3 days"\n',
        encoding="utf-8",
    )
    (replay / "uv.lock").write_text(
        'version = 1\n\n[options]\nexclude-newer-span = "P3D"\n\n[manifest]\noverrides = []\n',
        encoding="utf-8",
    )

    class _Env:
        def sync(self, root, extras, frozen=False):
            self.frozen = frozen

    env = _Env()
    root = tmp / "workspace"
    ws.lock_and_sync([], [], root=root, source=core, seed_lock=None, environment=env, replay=replay)
    assert env.frozen is True
    policy = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"]
    assert "exclude-newer" not in policy
    options = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8")).get("options") or {}
    assert "exclude-newer" not in options and "exclude-newer-span" not in options


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
