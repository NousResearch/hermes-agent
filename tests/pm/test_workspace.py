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
from tests.pm.test_environment_build import locked_project  # noqa: F401




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


def _seed_rebase_root(root, names=("member-proof",), *, buildable=True):
    import tomli_w

    members = []
    for index, name in enumerate(names):
        relative = f"plugin-sources/new-{index}"
        member = root / relative
        member.mkdir(parents=True)
        metadata = {"project": {"name": name, "version": "2"}}
        metadata["build-system" if buildable else "tool"] = (
            {"requires": [], "build-backend": "local_backend"} if buildable else
            {"uv": {"package": False}})
        (member / "pyproject.toml").write_text(tomli_w.dumps(metadata), encoding="utf-8")
        members.append(relative)
    (root / "pyproject.toml").write_text(tomli_w.dumps(
        {"tool": {"uv": {"workspace": {"members": members}}}}), encoding="utf-8")
    return members


def _editable_seed(path="plugin-sources/old", *, name="member-proof"):
    import tomli_w

    return tomli_w.dumps({"version": 1, "options": {"exclude-newer": "2025-01-01T00:00:00Z"},
                         "manifest": {"members": ["core", name]}, "package": [
                             {"name": "core", "version": "1", "source": {"editable": "."}},
                             {"name": name, "version": "1", "source": {"editable": path}},
                             {"name": "registry-pin", "version": "1.2",
                              "source": {"registry": "https://example.invalid/simple"},
                              "wheels": [{"url": "https://example.invalid/pin.whl",
                                          "hash": "sha256:unchanged"}]}]}).encode()


def _assert_seed_rebase_references(tmp_path, _monkeypatch, _kind):
    import copy
    import tomllib
    import tomli_w

    _seed_rebase_root(tmp_path, ("Member_Proof",))
    original = tomllib.loads(_editable_seed().decode())
    old = "plugin-sources/old"
    original["package"][0]["dependencies"] = [
        {"name": "member-proof", "version": "1", "source": {"editable": old}}]
    original["package"][0]["metadata"] = {"requires-dist": [
        {"name": "member-proof", "editable": old, "marker": "python_version >= '3.11'"}]}
    expected = copy.deepcopy(original)
    expected["package"][1]["source"]["editable"] = "plugin-sources/new-0"
    expected["package"][0]["dependencies"][0]["source"]["editable"] = "plugin-sources/new-0"
    expected["package"][0]["metadata"]["requires-dist"][0]["editable"] = "plugin-sources/new-0"
    seed = tomli_w.dumps(original).encode()
    actual = ws._rebase_seed_lock(seed, tmp_path)
    assert tomllib.loads(actual.decode()) == expected
    assert seed == tomli_w.dumps(original).encode()


def _assert_seed_rebase_dynamic(tmp_path, _monkeypatch, _kind):
    import tomllib

    _seed_rebase_root(tmp_path, ("member-proof", "unrelated-dynamic"))
    dynamic = tmp_path / "plugin-sources/new-1/pyproject.toml"
    dynamic.write_text('[project]\ndynamic=["name"]\n[build-system]\nrequires=[]\n'
                       'build-backend="local_backend"\n', encoding="utf-8")
    result = tomllib.loads(ws._rebase_seed_lock(_editable_seed(), tmp_path).decode())
    assert result["package"][1]["source"] == {"editable": "plugin-sources/new-0"}
    assert dynamic.read_text().startswith('[project]\ndynamic=["name"]')


def _assert_seed_rebase_urls(tmp_path, _monkeypatch, _kind):
    import copy
    import tomllib
    import tomli_w

    _seed_rebase_root(tmp_path)
    original = tomllib.loads(_editable_seed().decode())
    original["package"][2]["source"]["registry"] = "https://index.invalid/plugin-sources/old"
    original["package"][2]["wheels"][0]["url"] = "https://index.invalid/plugin-sources/old/pkg.whl"
    original["package"].append({"name": "git-pin", "version": "1",
                                "source": {"git": "https://example.invalid/plugin-sources/old#abcdef"}})
    expected = copy.deepcopy(original)
    expected["package"][1]["source"] = {"editable": "plugin-sources/new-0"}
    result = ws._rebase_seed_lock(tomli_w.dumps(original).encode(), tmp_path)
    assert tomllib.loads(result.decode()) == expected


def _assert_frozen_seed(tmp_path, monkeypatch, _kind):
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text('[project]\nname="core"\nversion="1"\n'
                                       '[tool.uv]\npackage=false\n', encoding="utf-8")
    seed = core / "uv.lock"
    raw = b'# exact frozen formatting\r\nversion = 1\r\n'
    seed.write_bytes(raw)
    monkeypatch.setattr(ws, "_rebase_seed_lock", lambda *a: pytest.fail("rebased frozen seed"))
    observed = []

    class NeverUV:
        def sync(self, root, **options):
            observed.append((root, options, (root / "uv.lock").read_bytes()))
    root = tmp_path / "candidate"
    ws.lock_and_sync([], [], root=root, source=core, seed_lock=seed,
                     environment=NeverUV(), frozen=True)
    assert observed == [(root, {"extras": [], "frozen": True}, raw)]
    assert seed.read_bytes() == raw


def _assert_seed_byte_identity(tmp_path, _monkeypatch, kind):
    _seed_rebase_root(tmp_path, buildable=kind != "virtual")
    seed = b"# preserve lock formatting\n" + _editable_seed(
        "plugin-sources/new-0" if kind == "same" else "plugin-sources/old",
        name="unrelated-removed" if kind == "removed" else "member-proof")
    assert ws._rebase_seed_lock(seed, tmp_path) == seed


def _assert_seed_refusal(tmp_path, monkeypatch, kind):
    import tomllib
    import tomli_w

    names = {"current-duplicate": ("member-proof", "MEMBER_Proof"),
             "inverse-collision": ("member-proof", "other-proof"),
             "inverse-noop": ("member-proof", "other-proof")}.get(kind, ("member-proof",))
    _seed_rebase_root(tmp_path, names)
    if kind == "inverse-noop":
        (tmp_path / "plugin-sources/new-0").rename(tmp_path / "plugin-sources/old")
        metadata = tmp_path / "pyproject.toml"
        generated = tomllib.loads(metadata.read_text())
        generated["tool"]["uv"]["workspace"]["members"][0] = "plugin-sources/old"
        metadata.write_text(tomli_w.dumps(generated), encoding="utf-8")
    document = tomllib.loads(_editable_seed().decode())
    ambiguous = "duplicate" in kind or kind.startswith("inverse-")
    reason = "ambiguous" if ambiguous else "unsafe" if kind == "unsafe" else (
        "escapes" if kind == "redirected" else "unsupported")
    changes = {
        "old-duplicate": lambda: document["package"].append({
            "name": "member-proof", "version": "1",
            "source": {"editable": "plugin-sources/other-old"}}),
        "inverse-collision": lambda: document["package"].append({
            "name": "other-proof", "version": "1",
            "source": {"editable": "plugin-sources/old"}}),
        "inverse-noop": lambda: document["package"].append({
            "name": "other-proof", "version": "1",
            "source": {"editable": "plugin-sources/old"}}),
        "unsafe": lambda: document["package"][1]["source"].update(
            editable="plugin-sources/../escape"),
        "unsupported": lambda: document["package"][1].update(unexpected="plugin-sources/old"),
    }
    if kind in changes:
        changes[kind]()
    if kind == "redirected":
        target = tmp_path / "plugin-sources/new-0/pyproject.toml"
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    seed = tomli_w.dumps(document).encode()
    source = tmp_path / "source"
    source.mkdir()
    (source / "uv.lock").write_bytes(seed)
    root = tmp_path / "candidate"
    def generate(*args, **kwargs):
        shutil.copytree(tmp_path / "plugin-sources", root / "plugin-sources")
        shutil.copy2(tmp_path / "pyproject.toml", root / "pyproject.toml")
    monkeypatch.setattr(ws, "_generate_pyproject", generate)
    if kind == "redirected":
        target = root / "plugin-sources/new-0/pyproject.toml"
    environment = type("NeverUV", (), {"sync": lambda *a, **k: pytest.fail("uv was invoked")})()
    with pytest.raises(ws.InstallError, match=reason):
        ws.lock_and_sync([], [], root=root, source=source, seed_lock=source / "uv.lock",
                         environment=environment)
    assert (source / "uv.lock").read_bytes() == seed
    assert not (root / "uv.lock").exists()


@pytest.mark.parametrize("check,kind", [
    pytest.param(_assert_seed_rebase_references, None, id="source-references"),
    pytest.param(_assert_seed_rebase_dynamic, None, id="dynamic-metadata"),
    pytest.param(_assert_seed_rebase_urls, None, id="external-urls"),
    pytest.param(_assert_frozen_seed, None, id="frozen-copy"),
    *[pytest.param(_assert_seed_byte_identity, kind, id=kind)
      for kind in ("same", "removed", "virtual")],
    *[pytest.param(_assert_seed_refusal, kind, id=kind)
      for kind in ("current-duplicate", "old-duplicate", "inverse-collision", "inverse-noop", "unsafe",
                   "unsupported", "redirected")],
])
def test_editable_seed_member_contract(tmp_path, monkeypatch, check, kind):
    check(tmp_path, monkeypatch, kind)


def test_core_quarantine_covers_core_packages_and_not_plugin_ones(layout, locked_project):
    """Regression for #120076: Hermes's 14-day cutoff must not filter a plugin's own deps.

    A global ``exclude-newer`` in the generated root made a catalog pin floored on a fresh
    release unresolvable. The cutoff now travels per package: every registry package in
    core's lock keeps it (or core's explicit exemption), plugin-only packages follow the
    plugin's policy, and the rewritten root still locks and syncs.
    """
    import tomllib

    tmp, core, plug_a, _ = layout
    _, uv, env = locked_project
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace("[tool.uv]\n", '[tool.uv]\nexclude-newer="14 days"\n', 1)
                        + '[tool.uv.exclude-newer-package]\nBase_Dep = false\n')
    subprocess.run([str(uv), "lock", "--python", sys.executable], cwd=core, env=env, check=True,
                   capture_output=True)
    core_packages = {p["name"] for p in tomllib.loads((core / "uv.lock").read_text())["package"]
                     if "registry" in p.get("source", {})}
    assert {"base-dep", "chosen-dep"} <= core_packages and "member-dep" not in core_packages

    root = tmp / "workspace"
    ws.lock_and_sync([plug_a], [], root=root, source=core, seed_lock=core / "uv.lock",
                     environment=managed_environment(tmp / "env"))

    policy = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]
    assert "exclude-newer" not in policy
    per_package = policy["exclude-newer-package"]
    assert per_package == {name: False if name == "base-dep" else "14 days" for name in core_packages}
    locked = {p["name"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    assert "member-dep" in locked and "member-dep" not in per_package






# --- classified failures + staging surface (FINAL-RUNTIME-CONTRACT) ---


def test_classify_network_failure_stays_generic():
    from pm.workspace import ResolutionConflict, classify_uv_failure

    err = classify_uv_failure("lock", 1, "error: Failed to fetch https://pypi.org (timed out)")
    assert not isinstance(err, ResolutionConflict)


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
