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

import pm.plugins_state as pstate
import pm.workspace as ws


def _write_enabled(home: Path, enabled: list, provider: str | None = None) -> None:
    home.mkdir(parents=True, exist_ok=True)
    cfg: dict = {"plugins": {"enabled": enabled}}
    if provider:
        cfg["memory"] = {"provider": provider}
    with (home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def _uv_available() -> bool:
    return shutil.which("uv") is not None


# ---------------------------------------------------------------------------
# 1. Sidecar with no root dependency surface never joins the union
# ---------------------------------------------------------------------------

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

    monkeypatch.setenv("HERMES_HOME", str(home))

    assert ws._is_member_candidate(wrapper) is False, (
        "a wrapper root without pyproject/dep keys must never be a member candidate"
    )
    scan = ws.scan_plugin(wrapper)
    assert scan["pyproject"] is False and scan["legacy_deps"] is False

    members = ws.enabled_member_dirs()
    member_names = [p.name for p in members]
    assert "mnemosyne-wrapper" not in member_names
    assert "mnemosyne-sidecar" not in member_names, (
        "an external sidecar pyproject must not join the union — pm scans "
        "only plugin roots, and the wrapper owns its runtime"
    )
    assert all("runtime" not in str(p) for p in members)


# ---------------------------------------------------------------------------
# 2. Conflict through the PUBLIC admission path: refused, preserved, retry
# ---------------------------------------------------------------------------

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
    """Fake core repo + temp HERMES_HOME so the REAL pm.ensure.sync_venv
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

    ensure = importlib.import_module("pm.ensure")
    import pm.paths

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pm.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    # Exercise the real dependency transaction in-process so the local uv
    # fixture owns provisioning; worker transport is covered separately.
    monkeypatch.setattr("pm.client.sync_venv", ensure.sync_venv)
    from pm.packages import uv_env
    monkeypatch.setattr(ensure, "uv", lambda **kwargs: (
        shutil.which("uv"), {**uv_env(kwargs.get("base_env")), "UV_PYTHON": sys.executable},
    ))
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
    from hermes_cli.runtime_paths import selected_venv
    working = selected_venv(tmp_path / "core")
    config_before = (home / "config.yaml").read_bytes()
    tree_before = {p: sorted(str(f) for f in p.rglob("*")) for p in (plug_a, plug_b)}

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


@pytest.mark.skipif(not _uv_available(), reason="uv not on PATH")
def test_retry_after_conflict_enables_resolvable_candidate(admission_env):
    """The retry path: re-admitting ONLY the resolvable candidate through
    the same public admission commits config + environment together; the
    conflicting plugin stays unenabled (never imported)."""
    from hermes_cli import plugins_admission as admission

    tmp_path, home = admission_env
    plug_a, plug_b, *_ = _local_conflict_members(home)

    wrapper = home / "plugins/mnemosyne-wrapper"
    wrapper.mkdir()
    marker = wrapper / "mnemosyne-wrapper.json"
    marker.write_bytes(b'{"wrapper":true}\n')
    sidecar = tmp_path / "external-sidecar"
    subprocess.run([shutil.which("uv"), "venv", "--python", sys.executable, str(sidecar)], check=True, capture_output=True, timeout=60)
    sidecar_python = sidecar / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    before = marker.read_bytes()
    with pytest.raises(admission.AdmissionRefused):
        admission.admit_plugin_set_change(
            {"plug-a", "plug-b"}, set(), active_plugins_dir=home / "plugins"
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
    from hermes_cli.runtime_paths import selected_venv
    selected = selected_venv(tmp_path / "core")
    assert selected.is_dir() and selected != sidecar
    # A declared version range remains a member across the next managed rebuild.
    project = plug_a / "pyproject.toml"
    project.write_text(project.read_text(encoding="utf-8").replace("sharedlib==1.0.0", "sharedlib>=1,<2"), encoding="utf-8")
    admission.admit_plugin_set_change({"plug-a"}, set(), active_plugins_dir=home / "plugins")
    assert selected_venv(tmp_path / "core") != selected
    assert marker.read_bytes() == before
    subprocess.run([str(sidecar_python), "-c", "import sys; assert sys.prefix != sys.base_prefix"], check=True, timeout=30)


# ---------------------------------------------------------------------------
# 4. Active context home propagates to wrapper subprocess launches
# ---------------------------------------------------------------------------

def test_active_context_home_exported_to_wrapper_subprocess(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.environments.local import build_subprocess_env

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))
    active = tmp_path / "custom-root/profiles/worker"
    active.mkdir(parents=True)
    token = set_hermes_home_override(active)
    try:
        child_env = build_subprocess_env()
        child = subprocess.run(
            [sys.executable, "-c", "import os; print(os.environ['HERMES_HOME'], end='')"],
            env=child_env, capture_output=True, text=True, check=True, timeout=60,
        )
        assert child.stdout == str(active)
    finally:
        reset_hermes_home_override(token)
