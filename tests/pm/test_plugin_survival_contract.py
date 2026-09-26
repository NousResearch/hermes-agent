"""Plugin survival + upgrade contracts (mnemosyne-oss/mnemosyne#859).

Hermes-side contracts an external memory-provider wrapper can hold us to,
exercised through PUBLIC paths (no source inspection, no network — uv
resolves offline local path-source fixtures):

1. Sidecar isolation. A wrapper plugin exposes NO dependency surface at
   the scanned plugin root (no pyproject.toml, no legacy dep keys); a
   pyproject belonging to a nested or external sidecar dir must NOT
   join the pm workspace union, because pm scans only plugin roots.
2. Conflict admission. Through the PUBLIC admission authority
   (hermes_cli.plugins_admission.admit_plugin_set_change — the one path
   `hermes plugins enable/install` use), a candidate union with no
   valid solution is REFUSED before anything is published: the
   candidate stays unenabled (so the loader never imports it), the
   plugin trees and every home's config.yaml survive untouched, the
   refusal message carries the plugin identity + the resolver's
   reason, and a machine-readable pm receipt records the failure.
   The retry path — re-admitting only the resolvable candidate —
   commits through the same public function.
3. Active-home propagation. The active CONTEXT home
   (hermes_constants.set_hermes_home_override) is what wrapper/sidecar
   subprocess launches must inherit through build_subprocess_env.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import hermes_yaml as yaml

import pm.workspace as ws
from pm.plugin_inputs import Members


def _write_enabled(home: Path, enabled: list, provider: str | None = None) -> None:
    home.mkdir(parents=True, exist_ok=True)
    cfg: dict = {"plugins": {"enabled": enabled}}
    if provider:
        cfg["memory"] = {"provider": provider}
    with (home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def _uv_available() -> bool:
    return shutil.which("uv") is not None


# 1. Sidecar with no root dependency surface never joins the union

def test_sidecar_no_root_pyproject_excludes_nested_and_external(tmp_path, monkeypatch):
    """The mnemosyne-wrapper shape: plugin root has ONLY plugin.yaml +
    marker; its runtime lives in sidecar dirs with their own pyprojects.
    Neither the nested subdir pyproject nor the external one may join
    the workspace union."""
    home = tmp_path / "home"
    _write_enabled(home, ["mnemosyne-wrapper"])
    plugins_dir = home / "plugins"

    wrapper = plugins_dir / "mnemosyne-wrapper"
    wrapper.mkdir(parents=True)
    (wrapper / "plugin.yaml").write_text("name: mnemosyne-wrapper\n", encoding="utf-8")
    (wrapper / "mnemosyne-wrapper.json").write_text('{"wrapper": true}\n', encoding="utf-8")
    # a nested pyproject INSIDE the plugin dir (below the scanned root)
    nested = wrapper / "runtime"
    nested.mkdir()
    (nested / "pyproject.toml").write_text(
        '[project]\nname = "mnemosyne-runtime"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    # an EXTERNAL sidecar next to the plugin (the wrapper's own venv project)
    external = plugins_dir / ".mnemosyne-sidecar"
    external.mkdir()
    (external / "pyproject.toml").write_text(
        '[project]\nname = "mnemosyne-sidecar"\nversion = "1.0.0"\n', encoding="utf-8"
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert ws._is_member_candidate(wrapper) is False, (
        "a wrapper root without pyproject/dep keys must never be a member candidate"
    )

    members = ws.enabled_member_dirs()
    member_names = [p.name for p in members]
    assert "mnemosyne-wrapper" not in member_names
    assert "mnemosyne-sidecar" not in member_names, (
        "an external sidecar pyproject must not join the union — pm scans "
        "only plugin roots, and the wrapper owns its runtime"
    )
    assert all("runtime" not in str(p) for p in members)


# 2. Conflict through the PUBLIC admission path: refused, preserved, retry

def _local_conflict_members(home: Path) -> tuple[Path, Path, Path, Path]:
    """plug-a and plug-b both need a local project named sharedlib, but
    map it to DIFFERENT path sources (v1 vs v2) — a union with NO valid
    solution, resolvable by uv fully OFFLINE."""
    shared1 = home / "sidecars" / "sharedlib-v1"
    shared2 = home / "sidecars" / "sharedlib-v2"
    for path, version in ((shared1, "1.0.0"), (shared2, "2.0.0")):
        path.mkdir(parents=True)
        (path / "pyproject.toml").write_text(
            "[project]\nname = \"sharedlib\"\n"
            f'version = "{version}"\nrequires-python = ">=3.11"\n',
            encoding="utf-8",
        )
    plugins_dir = home / "plugins"
    members = []
    for name, pin, shared in (("plug-a", "1.0.0", shared1), ("plug-b", "2.0.0", shared2)):
        plug = plugins_dir / name
        plug.mkdir(parents=True)
        (plug / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
        (plug / "pyproject.toml").write_text(
            "[project]\n"
            f'name = "{name}"\nversion = "0.1.0"\n'
            'requires-python = ">=3.11"\n'
            f'dependencies = ["sharedlib=={pin}"]\n'
            "\n[tool.uv.sources]\n"
            f'sharedlib = {{ path = "{shared.as_posix()}" }}\n',
            encoding="utf-8",
        )
        members.append(plug)
    return members[0], members[1], shared1, shared2


@pytest.fixture
def admission_env(tmp_path, monkeypatch):
    """Fake core repo + temp HERMES_HOME so the REAL pm.install.sync_venv
    transaction (lock, receipts, config publication) runs entirely under
    tmp — the production path, temp homes."""
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        "[project]\n"
        'name = "fake-core"\nversion = "0.1.0"\n'
        'requires-python = ">=3.11"\ndependencies = []\n',
        encoding="utf-8",
    )
    # the venv package's stamp digest + lock seed read core/uv.lock —
    # produce a real one (no deps: uv lock resolves offline)
    subprocess.run(
        [shutil.which("uv"), "lock"], cwd=core, check=True,
        capture_output=True, text=True, timeout=120,
    )
    home = tmp_path / "home"
    _write_enabled(home, [])  # nothing enabled yet — the candidate must move it

    import importlib

    ensure = importlib.import_module("pm.install")
    import pm.paths

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pm.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    # Exercise the real dependency transaction in-process so the local uv
    # fixture owns provisioning; worker transport is covered separately. The
    # facade's project_root selects a foreign checkout for the worker; this
    # fixture's repo_root already IS the core under test.
    def in_process_sync(*args, project_root=None, **kwargs):
        assert project_root is None or Path(project_root).resolve() == core.resolve()
        return ensure.sync_venv(*args, **kwargs)

    monkeypatch.setattr("pm.client.sync_venv", in_process_sync)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(shutil.which("uv")), Path(sys.executable)))
    return tmp_path, home


def _latest_receipt(home: Path) -> dict:
    from pm.receipt import latest
    receipt = latest()
    assert receipt is not None, f"no pm receipt written for {home}"
    return receipt


@pytest.mark.skipif(not _uv_available(), reason="uv not on PATH")
def test_conflicting_candidate_refused_unenabled_and_unimported(admission_env):
    """Public admission of an unsatisfiable union: AdmissionRefused with
    plugin identity + resolver reason; the candidate is NOT published to
    config (so the plugin loader never imports it); plugin trees and
    configs survive; the receipt records the failure."""
    from hermes_cli import plugins_admission as admission

    tmp_path, home = admission_env
    plug_a, plug_b, *_ = _local_conflict_members(home)
    _write_enabled(home, [], provider="plug-a")
    admission.admit_plugin_set_change(set(), set(), active_plugins_dir=home / "plugins")
    from pm.environments import selected_venv
    working = selected_venv(tmp_path / "core")
    config_before = (home / "config.yaml").read_bytes()
    tree_before = {p: sorted(str(f) for f in p.rglob("*")) for p in (plug_a, plug_b)}
    wrapper = home / "plugins/mnemosyne-wrapper"
    wrapper.mkdir()
    marker = wrapper / "mnemosyne-wrapper.json"
    marker.write_bytes(b'{"wrapper":true}\n')
    sidecar = tmp_path / "external-sidecar"
    subprocess.run([shutil.which("uv"), "venv", "--python", sys.executable, str(sidecar)], check=True, capture_output=True, timeout=60)
    sidecar_python = sidecar / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    before = marker.read_bytes()

    with pytest.raises(admission.AdmissionRefused) as excinfo:
        admission.admit_plugin_set_change(
            {"plug-a", "plug-b"}, set(), active_plugins_dir=home / "plugins"
        )
    message = str(excinfo.value)
    assert "plug-b" in message or "plug-a" in message, (
        f"refusal must carry the plugin identity, got: {message}"
    )
    assert "sharedlib" in message.lower() or "conflict" in message.lower(), (
        f"refusal must carry the resolver's reason, got: {message}"
    )

    # unenabled → unimported: the candidate never reached the enabled list
    with (home / "config.yaml").open(encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    assert cfg["plugins"]["enabled"] == [], (
        "a refused candidate must stay unenabled — config published a set that never resolved"
    )
    assert (home / "config.yaml").read_bytes() == config_before
    assert cfg["memory"]["provider"] == "plug-a"
    assert selected_venv(tmp_path / "core") == working

    # no plugin tree was deleted or mutated by the failed resolution
    for plug, listing in tree_before.items():
        assert plug.is_dir(), f"failed resolution deleted plugin tree {plug}"
        assert sorted(str(f) for f in plug.rglob("*")) == listing

    # the machine-readable receipt records the failed sync
    receipt = _latest_receipt(home)
    assert receipt.get("outcome") == "failed"
    flattened = json.dumps(receipt)
    assert "plug-b" in flattened or "sharedlib" in flattened, (
        "receipt must carry the conflict identity/reason"
    )
    # retry: drop the conflicting candidate, keep the good one
    admission.admit_plugin_set_change(
        {"plug-a"}, set(), active_plugins_dir=home / "plugins"
    )

    with (home / "config.yaml").open(encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    assert cfg["plugins"]["enabled"] == ["plug-a"]
    assert "plug-b" not in cfg["plugins"]["enabled"], (
        "the conflicting candidate must remain unenabled after the retry"
    )
    assert marker.read_bytes() == before
    child = subprocess.run([str(sidecar_python), "-c", "import sys; print(sys.prefix)"], check=True, capture_output=True, text=True, timeout=30)
    assert Path(child.stdout.strip()) == sidecar
    from pm.environments import selected_venv
    selected = selected_venv(tmp_path / "core")
    assert selected.is_dir() and selected != sidecar
    # A declared version range remains a member across the next managed rebuild.
    project = plug_a / "pyproject.toml"
    project.write_text(project.read_text(encoding="utf-8").replace("sharedlib==1.0.0", "sharedlib>=1,<2"), encoding="utf-8")
    admission.admit_plugin_set_change({"plug-a"}, set(), active_plugins_dir=home / "plugins")
    assert selected_venv(tmp_path / "core") != selected
    assert marker.read_bytes() == before
    subprocess.run([str(sidecar_python), "-c", "import sys; assert sys.prefix != sys.base_prefix"], check=True, timeout=30)


@pytest.mark.skipif(not _uv_available(), reason="uv not on PATH")
def test_malformed_secondary_cannot_evict_recorded_member(admission_env, monkeypatch, caplog):
    """A→B→A: passive inspection survives bad config; A's recorded graph does not shrink."""
    from pm.environments import runtime_facts_path, selected_venv
    from pm.install import sync_venv, venv_is_current
    from pm.lock import Facts

    tmp_path, home_a = admission_env
    core = tmp_path / "core"
    profile = home_a / "profiles" / "work"
    profile.mkdir(parents=True)
    _write_enabled(profile, ["profile-dep"])
    member = profile / "plugins" / "profile-dep"
    member.mkdir(parents=True)
    (member / "pyproject.toml").write_text(
        '[project]\nname="profile-dep"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=[]\n[tool.uv]\npackage=false\n', encoding="utf-8",
    )
    sync_venv(explicit=True)
    recorded = Facts(runtime_facts_path(core), strict=True).get("venv")
    selected = selected_venv(core)
    assert recorded["stamp"] and selected.is_dir()
    assert "profile-dep" in (Path(recorded["resolved_lock"]).parent / "pyproject.toml").read_text()

    bad = profile / "config.yaml"
    bad.write_text("plugins: [broken]\n", encoding="utf-8")
    candidate = home_a / "plugins" / "new-dep"
    candidate.mkdir(parents=True)
    (candidate / "pyproject.toml").write_text(
        '[project]\nname="new-dep"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=[]\n[tool.uv]\npackage=false\n', encoding="utf-8",
    )
    _write_enabled(home_a, ["new-dep"])
    assert venv_is_current(project_root=core) is False
    assert str(bad) in caplog.text
    with pytest.raises(ValueError, match="config.yaml"):
        sync_venv(explicit=True)
    assert Facts(runtime_facts_path(core), strict=True).get("venv") == recorded
    assert selected_venv(core) == selected
    # Even a precomputed member list cannot bypass a newly broken profile.
    with pytest.raises(ValueError, match="config.yaml"):
        sync_venv(explicit=True, plugins=Members([]))
    assert Facts(runtime_facts_path(core), strict=True).get("venv") == recorded

    home_b = tmp_path / "home-b"
    _write_enabled(home_b, [])
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    sync_venv(explicit=True)
    assert selected_venv(core).is_dir()
    assert Facts(runtime_facts_path(core), strict=True).get("venv")["stamp"] != recorded["stamp"]

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    with pytest.raises(ValueError, match="config.yaml"):
        sync_venv(explicit=True)
    assert selected_venv(core) == selected
    _write_enabled(profile, ["profile-dep"])
    sync_venv(explicit=True)
    restored = Facts(runtime_facts_path(core), strict=True).get("venv")
    assert selected_venv(core).is_dir()
    assert restored["stamp"] != recorded["stamp"]
    assert "profile-dep" in (Path(restored["resolved_lock"]).parent / "pyproject.toml").read_text()


@pytest.mark.skipif(not _uv_available(), reason="uv not on PATH")
def test_update_sync_disables_plugin_excluded_by_requires_python(admission_env):
    """An update never fails because of a plugin: a member whose requires-python excludes the
    interpreter core now runs on is disabled in every home that enables it (plugins.enabled in
    one, memory.provider in another), the rest build, and the next boot sees a current venv."""
    from pm.environments import runtime_facts_path, selected_venv
    from pm.install import sync_venv, venv_is_current
    from pm.lock import Facts
    from pm.package import InstallError

    tmp_path, home = admission_env
    core = tmp_path / "core"
    for name, requires in (("fits", ">=3.11"), ("too-old", f"<{sys.version_info[0]}.{sys.version_info[1]}")):
        plugin = home / "plugins" / name
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(
            f'[project]\nname="{name}"\nversion="1"\nrequires-python="{requires}"\n'
            'dependencies=[]\n[tool.uv]\npackage=false\n', encoding="utf-8",
        )
    (home / "config.yaml").write_text("# operator note\nplugins:\n  enabled: [fits, too-old]\n", encoding="utf-8")
    profile = home / "profiles" / "work"
    _write_enabled(profile, [], provider="too-old")
    (profile / "plugins" / "too-old").mkdir(parents=True)
    (profile / "plugins" / "too-old" / "pyproject.toml").write_bytes(
        (home / "plugins" / "too-old" / "pyproject.toml").read_bytes().replace(b'name="too-old"', b'name="too-old-work"'))

    with pytest.raises(InstallError, match="Python requirement"):
        sync_venv(explicit=True)  # admission semantics stay: an ordinary sync refuses

    sync_venv(explicit=True, evict_incompatible_plugins=True)

    workspace = Path(Facts(runtime_facts_path(core), strict=True).get("venv")["resolved_lock"]).parent
    assert selected_venv(core).is_dir()
    assert "fits" in (workspace / "pyproject.toml").read_text()
    assert "too-old" not in (workspace / "pyproject.toml").read_text()
    text = (home / "config.yaml").read_text(encoding="utf-8")
    assert "# operator note" in text
    assert "too-old" in yaml.safe_load(text)["plugins"]["disabled"]
    assert yaml.safe_load((profile / "config.yaml").read_text(encoding="utf-8"))["memory"]["provider"] == ""
    assert "too-old" in json.dumps(_latest_receipt(home).get("warnings"))
    assert venv_is_current(project_root=core) is True


@pytest.mark.skipif(not _uv_available(), reason="uv not on PATH")
def test_update_sync_disables_later_plugin_of_unresolvable_union(admission_env):
    """Core moved under two admitted plugins that no longer co-resolve: the update keeps the
    first in config order, disables the one that breaks the build, and completes."""
    from pm.install import sync_venv, venv_is_current

    tmp_path, home = admission_env
    _local_conflict_members(home)
    _write_enabled(home, ["plug-a", "plug-b"])

    sync_venv(explicit=True, evict_incompatible_plugins=True)

    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["plugins"]["disabled"] == ["plug-b"]
    assert venv_is_current(project_root=tmp_path / "core") is True


@pytest.mark.skipif(not _uv_available(), reason="uv not on PATH")
def test_update_sync_survives_unreadable_secondary_profile(admission_env):
    """A secondary profile's broken config.yaml cannot fail an update: its plugins sit out
    (reported), the rest build, and the next boot sees a current venv."""
    from pm.environments import runtime_facts_path
    from pm.install import sync_venv, venv_is_current
    from pm.lock import Facts

    tmp_path, home = admission_env
    core = tmp_path / "core"
    member = home / "plugins" / "primary-dep"
    member.mkdir(parents=True)
    (member / "pyproject.toml").write_text(
        '[project]\nname="primary-dep"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=[]\n[tool.uv]\npackage=false\n', encoding="utf-8",
    )
    _write_enabled(home, ["primary-dep"])
    broken = home / "profiles" / "work" / "config.yaml"
    broken.parent.mkdir(parents=True)
    broken.write_text("plugins: [broken]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config.yaml"):
        sync_venv(explicit=True)  # an ordinary sync still refuses to shrink the graph

    sync_venv(explicit=True, evict_incompatible_plugins=True)

    workspace = Path(Facts(runtime_facts_path(core), strict=True).get("venv")["resolved_lock"]).parent
    assert "primary-dep" in (workspace / "pyproject.toml").read_text()
    assert broken.read_text(encoding="utf-8") == "plugins: [broken]\n"
    assert str(broken.parent) in json.dumps(_latest_receipt(home).get("warnings"))
    assert venv_is_current(project_root=core) is True


@pytest.mark.skipif(not _uv_available(), reason="uv not on PATH")
def test_plugin_our_version_rejects_sits_out_without_being_disabled(admission_env, monkeypatch):
    """requires_hermes is judged against our version identity, which can lag (an untagged
    source checkout reads as an older release). Such a plugin sits out: config untouched,
    boot's currency check neither raises nor loops, and it rejoins once the verdict flips."""
    from pm.environments import runtime_facts_path
    from pm.install import sync_venv, venv_is_current
    from pm.lock import Facts

    import hermes_cli.plugins_manifest as plugins_manifest

    tmp_path, home = admission_env
    core = tmp_path / "core"
    # A tagless checkout (CI's) has no parseable version, which makes requires_hermes permissive.
    monkeypatch.setattr(plugins_manifest, "running_hermes_version", lambda: "1.0.0")
    for name in ("fits", "needs-newer"):
        plugin = home / "plugins" / name
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(
            f'[project]\nname="{name}"\nversion="1"\nrequires-python=">=3.11"\n'
            'dependencies=[]\n[tool.uv]\npackage=false\n', encoding="utf-8",
        )
    manifest = home / "plugins" / "needs-newer" / "plugin.yaml"
    manifest.write_text("name: needs-newer\nrequires_hermes: '>=999'\n", encoding="utf-8")
    _write_enabled(home, ["fits", "needs-newer"])
    before = (home / "config.yaml").read_bytes()

    assert venv_is_current(project_root=core) is False  # boot probes it before any sync
    sync_venv(explicit=True, evict_incompatible_plugins=True)

    assert (home / "config.yaml").read_bytes() == before
    assert "Left plugin 'needs-newer'" in json.dumps(_latest_receipt(home).get("warnings"))
    workspace = Path(Facts(runtime_facts_path(core), strict=True).get("venv")["resolved_lock"]).parent
    assert "needs-newer" not in (workspace / "pyproject.toml").read_text()
    assert venv_is_current(project_root=core) is True

    manifest.write_text("name: needs-newer\nrequires_hermes: '>=0'\n", encoding="utf-8")
    assert venv_is_current(project_root=core) is False
    sync_venv(explicit=True, evict_incompatible_plugins=True)
    workspace = Path(Facts(runtime_facts_path(core), strict=True).get("venv")["resolved_lock"]).parent
    assert "needs-newer" in (workspace / "pyproject.toml").read_text()


@pytest.mark.skipif(not _uv_available(), reason="uv not on PATH")
def test_update_sync_retries_a_fetch_failure_once_before_disabling(admission_env, monkeypatch):
    """Build evidence disables a plugin at once. A fetch failure could be the moment, so it
    gets one more try first. Either way the recorded state is the one boot expects."""
    from pm.install import sync_venv, venv_is_current
    from pm.packages import Venv

    trials: list[str] = []
    real_apply = Venv.apply

    def counting_apply(self, extras, *, plugin_dirs=None, **kwargs):
        if plugin_dirs:
            trials.append(Path(plugin_dirs[-1]).name)
        return real_apply(self, extras, plugin_dirs=plugin_dirs, **kwargs)

    monkeypatch.setattr(Venv, "apply", counting_apply)

    tmp_path, home = admission_env
    monkeypatch.setenv("UV_HTTP_RETRIES", "0")
    broken = tmp_path / "broken-lib"
    broken.mkdir()
    (broken / "pyproject.toml").write_text(
        '[project]\nname="broken-lib"\ndynamic=["version"]\n'
        '[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n', encoding="utf-8")
    (broken / "backend.py").write_text(
        "def get_requires_for_build_wheel(config=None): return []\n"
        "def prepare_metadata_for_build_wheel(directory, config=None): raise SystemExit('no build')\n"
        "def build_wheel(directory, config=None, metadata=None): raise SystemExit('no build')\n", encoding="utf-8")
    # Plugin requirements are index-only; local and remote inputs come through tool.uv.sources.
    for name, dependency, source in (("wont-build", "broken-lib", f'{{ path = "{broken.as_posix()}" }}'),
                                     ("offline-dep", "gone", '{ url = "https://127.0.0.1:9/gone-1.0-py3-none-any.whl" }')):
        plugin = home / "plugins" / name
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(
            f'[project]\nname="{name}"\nversion="1"\nrequires-python=">=3.11"\n'
            f'dependencies=["{dependency}"]\n[tool.uv]\npackage=false\n'
            f'[tool.uv.sources]\n{dependency} = {source}\n', encoding="utf-8",
        )
    _write_enabled(home, ["wont-build", "offline-dep"])

    sync_venv(explicit=True, evict_incompatible_plugins=True)

    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["plugins"]["disabled"] == ["wont-build", "offline-dep"]
    # After the full selection's own attempt: build evidence once, the fetch failure twice.
    assert trials[1:] == ["wont-build", "offline-dep", "offline-dep"]
    assert "could not be prepared, twice" in json.dumps(_latest_receipt(home).get("warnings"))
    assert venv_is_current(project_root=tmp_path / "core") is True


def test_active_context_home_exported_to_wrapper_subprocess(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.environments import local as local_env

    # ``build_subprocess_env`` also prepends the resolved console-script dir to PATH; the
    # resolver stats the interpreter's own bin dir, which for a default install (checkout
    # inside the home) is Hermes-owned state the home guard refuses. This test is about
    # HERMES_HOME propagation, so pin the PATH decoration out of the way.
    monkeypatch.setattr(local_env, "_resolve_hermes_bin_dir", lambda: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))
    active = tmp_path / "custom-root/profiles/worker"
    active.mkdir(parents=True)
    token = set_hermes_home_override(active)
    try:
        child_env = local_env.build_subprocess_env()
        child = subprocess.run(
            [sys.executable, "-c", "import os; print(os.environ['HERMES_HOME'], end='')"],
            env=child_env, capture_output=True, text=True, check=True, timeout=60,
        )
        assert child.stdout == str(active)
    finally:
        reset_hermes_home_override(token)
