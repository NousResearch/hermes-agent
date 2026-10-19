"""Behavior contracts for the config.d drop-in override layer.

Drop-ins are the user-owned overlay BETWEEN config.yaml and the managed scope:
``DEFAULT_CONFIG < config.yaml < config.d/*.yaml (lexicographic) < managed scope``. They
exist so shared config can be symlinked per profile with small per-profile overrides —
which only works if every loader applies them, cache invalidation sees every edit class,
and no write-back path (raw reads, save_config) ever persists a drop-in value INTO
config.yaml, where it would sit shadowed by the drop-in forever.

Four independent loaders must agree (``hermes_cli/AGENTS.md`` § Config system):
``hermes_cli.config.load_config``, ``cli.load_cli_config``, ``gateway.config_loader
.read_yaml_layers``, ``hermes_cli.config_effective.load_user_config_effective``.

Real imports against temp HERMES_HOMEs — no mocking the thing under test.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

import pytest

from hermes_cli import config as config_mod
from hermes_cli import config_d


# ── Fixtures + helpers ───────────────────────────────────────────────────────

@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp HERMES_HOME with every config cache cleared before AND after the test."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    _clear_caches()
    yield h
    _clear_caches()


def _clear_caches() -> None:
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    config_d.invalidate_config_d_cache()


def _drop_in_dir(home: Path) -> Path:
    d = home / "config.d"
    d.mkdir(exist_ok=True)
    return d


def _rewrite(path: Path, content: str) -> None:
    """Rewrite *path* so its file_signature necessarily changes.

    The kernel's coarse timestamp clock can leave mtime/size/ctime of a fast in-place rewrite
    equal to the cached signature — the same hazard tests/hermes_cli/test_config_cache_signature.py
    guards. Without this, the invalidation tests would flake on the cache MISSING an edit.

    Waits on mtime, not ctime: on Windows st_ctime_ns is the file's CREATION time and never
    advances on an in-place rewrite (see utils.file_signature), so a ctime wait would spin
    forever there. mtime is settable via os.utime on every platform, and the loop is bounded
    so a stalled filesystem clock fails the test instead of hanging the run.
    """
    before = path.stat()
    path.write_text(content, encoding="utf-8")
    deadline = time.monotonic() + 10.0
    while path.stat().st_mtime_ns <= before.st_mtime_ns:
        if time.monotonic() > deadline:
            raise AssertionError(f"filesystem clock did not advance for {path}")
        os.utime(path)


def _replace_pinning_mtime(path: Path, content: str) -> None:
    """Replace *path* pinning mtime+size (``cp -p``-style); ctime/inode still move.

    Same helper pattern as tests/hermes_cli/test_config_cache_signature.py: on POSIX ctime is
    the only differing stat field, so the signature must catch the change through it. The
    mtime wait is bounded for the same Windows-clock reason as _rewrite.
    """
    before = path.stat()
    other = path.with_name("other.yaml")
    other.write_text(content, encoding="utf-8")
    deadline = time.monotonic() + 10.0
    while other.stat().st_mtime_ns <= before.st_mtime_ns:
        if time.monotonic() > deadline:
            raise AssertionError(f"filesystem clock did not advance for {other}")
        os.utime(other)
    shutil.copy2(other, path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


    # ── Module unit: file selection ──────────────────────────────────────────────

def test_file_selection_accepts_yaml_yml_and_uppercase(tmp_path):
    d = tmp_path / "config.d"
    d.mkdir()
    (d / "10-a.yaml").write_text("a: 1\n", encoding="utf-8")
    (d / "20-b.yml").write_text("b: 2\n", encoding="utf-8")
    (d / "30-c.YAML").write_text("c: 3\n", encoding="utf-8")  # Windows case
    assert [p.name for p in config_d.config_d_files(d)] == ["10-a.yaml", "20-b.yml", "30-c.YAML"]


def test_file_selection_skips_dotfiles_non_yaml_and_directories(tmp_path):
    d = tmp_path / "config.d"
    d.mkdir()
    (d / "10-a.yaml").write_text("a: 1\n", encoding="utf-8")
    (d / ".50-x.yaml").write_text("x: 1\n", encoding="utf-8")          # editor backup
    (d / ".50-x.yaml.swp").write_text("x: 1\n", encoding="utf-8")      # vim swap
    (d / "README.md").write_text("not yaml\n", encoding="utf-8")
    (d / "50-x.yaml").mkdir()                                          # a DIRECTORY named like one
    assert [p.name for p in config_d.config_d_files(d)] == ["10-a.yaml"]


def test_absent_dir_yields_empty_overlay(tmp_path):
    assert config_d.config_d_files(tmp_path / "config.d") == []
    assert config_d.load_config_d(config_path=tmp_path / "config.yaml") == {}


# ── Module unit: merge order + symlinks + fail-open ──────────────────────────

def test_lexicographic_order_later_wins_and_siblings_survive(tmp_path):
    d = tmp_path / "config.d"
    d.mkdir()
    (d / "10-a.yaml").write_text(
        "display:\n  skin: from_a\n  show_reasoning: true\n", encoding="utf-8")
    (d / "90-z.yaml").write_text("display:\n  skin: from_z\n", encoding="utf-8")
    merged = config_d.load_config_d(config_path=tmp_path / "config.yaml")
    assert merged["display"]["skin"] == "from_z"          # later filename wins
    assert merged["display"]["show_reasoning"] is True    # sibling leaf kept


@pytest.mark.platforms("linux", "macos")
def test_symlinked_drop_in_file_is_followed(tmp_path):
    """The core use case: one shared drop-in symlinked from ~/agents/ into each profile."""
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "base.yaml").write_text("display:\n  skin: shared_skin\n", encoding="utf-8")
    d = tmp_path / "config.d"
    d.mkdir()
    (d / "10-shared.yaml").symlink_to(shared / "base.yaml")
    merged = config_d.load_config_d(config_path=tmp_path / "config.yaml")
    assert merged["display"]["skin"] == "shared_skin"


@pytest.mark.platforms("linux", "macos")
def test_symlinked_config_d_dir_works(tmp_path):
    real = tmp_path / "real-config-d"
    real.mkdir()
    (real / "10-a.yaml").write_text("display:\n  skin: via_dir_symlink\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.d").symlink_to(real)
    merged = config_d.load_config_d(config_path=home / "config.yaml")
    assert merged["display"]["skin"] == "via_dir_symlink"


@pytest.mark.platforms("linux", "macos")
def test_dangling_symlink_warned_and_skipped(tmp_path, caplog):
    d = tmp_path / "config.d"
    d.mkdir()
    (d / "10-a.yaml").write_text("display:\n  skin: ok_skin\n", encoding="utf-8")
    (d / "50-gone.yaml").symlink_to(tmp_path / "missing-target.yaml")
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config_d"):
        merged = config_d.load_config_d(config_path=tmp_path / "config.yaml")
    assert merged["display"]["skin"] == "ok_skin"  # sibling still loads
    assert "dangling" in caplog.text and "50-gone.yaml" in caplog.text


def test_broken_yaml_drop_in_warned_and_skipped(tmp_path, caplog):
    d = tmp_path / "config.d"
    d.mkdir()
    (d / "10-a.yaml").write_text("display:\n  skin: ok_skin\n", encoding="utf-8")
    (d / "20-broken.yaml").write_text("display: [unclosed\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config_d"):
        merged = config_d.load_config_d(config_path=tmp_path / "config.yaml")
    assert merged["display"]["skin"] == "ok_skin"  # fail-open: the good drop-in still applies
    assert "20-broken.yaml" in caplog.text


def test_empty_drop_in_file_is_an_empty_layer(tmp_path, caplog):
    d = tmp_path / "config.d"
    d.mkdir()
    (d / "10-empty.yaml").write_text("", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config_d"):
        merged = config_d.load_config_d(config_path=tmp_path / "config.yaml")
    assert merged == {}
    assert not caplog.records  # an empty file is valid, not an error


def test_top_level_list_drop_in_warned_and_skipped(tmp_path, caplog):
    d = tmp_path / "config.d"
    d.mkdir()
    (d / "10-list.yaml").write_text("- a\n- b\n", encoding="utf-8")
    (d / "90-ok.yaml").write_text("display:\n  skin: ok_skin\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config_d"):
        merged = config_d.load_config_d(config_path=tmp_path / "config.yaml")
    assert merged["display"]["skin"] == "ok_skin"
    assert "10-list.yaml" in caplog.text and "mapping" in caplog.text


def test_signature_detects_edit_add_remove_and_rename(tmp_path):
    d = tmp_path / "config.d"
    d.mkdir()
    (d / "10-a.yaml").write_text("a: 1\n", encoding="utf-8")
    sig1 = config_d.config_d_signature(d)
    _rewrite(d / "10-a.yaml", "a: 2\n")
    sig2 = config_d.config_d_signature(d)
    assert sig2 != sig1  # in-place edit
    (d / "20-b.yaml").write_text("b: 1\n", encoding="utf-8")
    sig3 = config_d.config_d_signature(d)
    assert sig3 != sig2  # added file (tuple length)
    (d / "20-b.yaml").unlink()
    assert config_d.config_d_signature(d) == sig2  # removed file
    (d / "10-a.yaml").rename(d / "10-c.yaml")  # identical content, new name
    assert config_d.config_d_signature(d) != sig2  # rename is a name-set change


def test_key_sources_attributes_each_leaf_to_its_file(home):
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text(
        "display:\n  skin: from_a\nagent:\n  max_turns: 7\n", encoding="utf-8")
    (d / "90-z.yaml").write_text("display:\n  skin: from_z\n", encoding="utf-8")
    sources = config_d.config_d_key_sources()
    assert sources["display.skin"] == ["10-a.yaml", "90-z.yaml"]
    assert sources["agent.max_turns"] == ["10-a.yaml"]


# ── Loader integration: hermes_cli.config.load_config ────────────────────────

def test_load_config_applies_drop_ins_preserving_siblings(home):
    (home / "config.yaml").write_text(
        "display:\n  skin: user_skin\n  show_reasoning: true\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    cfg = config_mod.load_config()
    assert cfg["display"]["skin"] == "dropin_skin"      # drop-in wins at the leaf
    assert cfg["display"]["show_reasoning"] is True     # config.yaml sibling preserved
    assert config_mod.load_config_readonly()["display"]["skin"] == "dropin_skin"


def test_managed_scope_beats_drop_in(home, tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("display:\n  skin: managed_skin\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    try:
        (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
        d = _drop_in_dir(home)
        (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
        assert config_mod.load_config()["display"]["skin"] == "managed_skin"
    finally:
        managed_scope.invalidate_managed_cache()


def test_config_d_only_profile_no_config_yaml(home):
    """A profile with NO config.yaml at all still loads its drop-ins — and caches them."""
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    cfg = config_mod.load_config_readonly()
    assert cfg["display"]["skin"] == "dropin_skin"
    # The guard fix in _load_config_cache_sig: a config.d-only profile still caches, so
    # the second read is the identical object rather than a fresh directory scan.
    assert config_mod.load_config_readonly() is cfg


def test_drop_in_legacy_model_string_matches_the_file_form(home, tmp_path, monkeypatch):
    """A drop-in's legacy ``model: x`` string form flows through the same merge +
    canonicalize chokepoint as config.yaml's: load_config() returns exactly what the
    same string in the file itself would produce (the runtime reader resolves both)."""
    from hermes_cli.config import split_model_config_default

    # Case 1: the bare string in a drop-in, over a dict-form config.yaml.
    (home / "config.yaml").write_text("model:\n  default: file/model\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("model: legacy/drop-in-model\n", encoding="utf-8")
    _clear_caches()
    from_drop_in = config_mod.load_config()["model"]

    # Case 2: the SAME bare string in config.yaml itself (the parity reference).
    file_home = tmp_path / "file-home"
    file_home.mkdir()
    (file_home / "config.yaml").write_text("model: legacy/drop-in-model\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(file_home))
    _clear_caches()
    from_file = config_mod.load_config()["model"]

    assert from_drop_in == from_file == "legacy/drop-in-model"
    assert split_model_config_default(from_drop_in) == ("legacy/drop-in-model", "")


def test_drop_in_root_max_turns_is_promoted(home):
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("max_turns: 7\n", encoding="utf-8")
    assert config_mod.load_config()["agent"]["max_turns"] == 7


def test_drop_in_root_max_turns_promotes_alongside_config_yaml(home):
    """With a config.yaml present, a drop-in's legacy root max_turns still promotes
    (the promotion runs on the merged user layer, before the DEFAULT_CONFIG merge)."""
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("max_turns: 7\n", encoding="utf-8")
    cfg = config_mod.load_config()
    assert cfg["agent"]["max_turns"] == 7
    assert cfg["display"]["skin"] == "user_skin"  # config.yaml sibling still merged


def test_drop_in_moa_presets_are_authoritative(home):
    """config.yaml explicitly deleted every MoA preset (an empty map is authoritative);
    a drop-in's presets map must replace it without resurrecting schema defaults."""
    (home / "config.yaml").write_text("moa:\n  presets: {}\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text(
        "moa:\n"
        "  presets:\n"
        "    mine:\n"
        "      reference_models: []\n"
        "      aggregator:\n"
        "        provider: openrouter\n"
        "        model: test/aggregator\n"
        "      enabled: true\n",
        encoding="utf-8")
    presets = config_mod.load_config()["moa"]["presets"]
    assert set(presets) == {"mine"}  # the drop-in's map, not defaults + drop-in


def test_env_ref_in_drop_in_is_expanded_by_loader(home, monkeypatch):
    """Expansion is the loader's job: the overlay stays raw, load_config expands."""
    monkeypatch.setenv("CONFIG_D_SKIN", "env_skin")
    (home / "config.yaml").write_text("display:\n  show_reasoning: true\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: ${CONFIG_D_SKIN}\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "env_skin"


# ── Loader integration: cache invalidation ───────────────────────────────────

def test_cache_invalidation_edit_drop_in(home):
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_v1\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "dropin_v1"
    _rewrite(d / "10-a.yaml", "display:\n  skin: dropin_v2\n")
    # No cache clearing: the signature must see the edit.
    assert config_mod.load_config()["display"]["skin"] == "dropin_v2"


def test_cache_invalidation_add_drop_in(home):
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "user_skin"
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "dropin_skin"


def test_cache_invalidation_remove_drop_in(home):
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "dropin_skin"
    (d / "10-a.yaml").unlink()
    assert config_mod.load_config()["display"]["skin"] == "user_skin"


def test_cache_invalidation_rename_with_identical_content(home):
    """A rename that changes merge ORDER (not content) must still invalidate: the
    signature is keyed on the file-name set, not just the bytes."""
    d = _drop_in_dir(home)
    (d / "50-a.yaml").write_text("display:\n  skin: from_a\n", encoding="utf-8")
    (d / "50-b.yaml").write_text("display:\n  skin: from_b\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "from_b"  # 50-b sorts last
    (d / "50-a.yaml").rename(d / "60-c.yaml")  # identical content, new name
    # 60-c now sorts AFTER 50-b, so the renamed file wins.
    assert config_mod.load_config()["display"]["skin"] == "from_a"


@pytest.mark.platforms("posix")
def test_cache_invalidation_mtime_pinned_replacement(home):
    """A cp -p-style replacement that pins mtime AND size must still invalidate."""
    d = _drop_in_dir(home)
    drop = d / "10-a.yaml"
    drop.write_text("display:\n  skin: v1_skin\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "v1_skin"
    _replace_pinning_mtime(drop, "display:\n  skin: v2_skin\n")  # same byte length
    assert config_mod.load_config()["display"]["skin"] == "v2_skin"


# ── Loader integration: cli.load_cli_config ──────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_cli_config(code: str, env_extra: dict[str, str], tmp_path: Path) -> dict:
    """Run *code* in a subprocess; results come back via a JSON file.

    Subprocess (not in-process monkeypatching), like tests/hermes_cli/test_config_loader_e2e.py:
    cli.py binds ``_hermes_home = get_hermes_home()`` at import time, and ``import cli`` runs
    hermes_bootstrap's dependency activation — whose sealed-payload probe reads the REAL
    home's manifest.json when the checkout sits inside ~/.hermes, which the pytest home-io
    guard refuses (the pre-existing failure mode of test_managed_scope_cli_config.py there).
    HERMES_HOME set BEFORE the import is also exactly how production loads.
    """
    import subprocess
    import sys

    out_file = tmp_path / "cli_config_result.json"
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)  # managed_scope._under_pytest() must be False (production)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["E2E_OUT_FILE"] = str(out_file)
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=180, check=False)
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    assert out_file.exists(), f"no result file:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(out_file.read_text(encoding="utf-8"))


_CLI_CONFIG_PROBE = """
import json, os
from pathlib import Path
import cli
cfg = cli.load_cli_config()
Path(os.environ["E2E_OUT_FILE"]).write_text(json.dumps({
    "skin": (cfg.get("display") or {}).get("skin"),
    "show_reasoning": (cfg.get("display") or {}).get("show_reasoning"),
}), encoding="utf-8")
"""


def _empty_managed_dir(tmp_path: Path) -> Path:
    """An empty HERMES_MANAGED_DIR: the subprocess has no _under_pytest guard, so this
    pins the managed layer to nothing regardless of what /etc/hermes holds on the host."""
    managed = tmp_path / "managed-empty"
    managed.mkdir(exist_ok=True)
    return managed


def test_cli_config_applies_drop_ins_preserving_siblings(home, tmp_path):
    (home / "config.yaml").write_text(
        "display:\n  skin: user_skin\n  show_reasoning: true\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    out = _run_cli_config(
        _CLI_CONFIG_PROBE,
        {"HERMES_MANAGED_DIR": str(_empty_managed_dir(tmp_path))},
        tmp_path)
    assert out["skin"] == "dropin_skin"      # reaches CLI_CONFIG (the TUI's source)
    assert out["show_reasoning"] is True     # user sibling preserved


def test_cli_config_drop_ins_skipped_under_ignore_user_config(home, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    out = _run_cli_config(
        _CLI_CONFIG_PROBE,
        {"HERMES_IGNORE_USER_CONFIG": "1",
         "HERMES_MANAGED_DIR": str(_empty_managed_dir(tmp_path))},
        tmp_path)
    # The switch means "no user config" — drop-ins are user config.
    assert out["skin"] not in ("dropin_skin", "user_skin")


# ── Loader integration: gateway read_yaml_layers ─────────────────────────────

def test_gateway_read_yaml_layers_applies_drop_ins(home):
    (home / "config.yaml").write_text(
        "display:\n  skin: user_skin\n  show_reasoning: true\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    from gateway.config_loader import read_yaml_layers

    cfg = read_yaml_layers(home)
    assert cfg["display"]["skin"] == "dropin_skin"
    assert cfg["display"]["show_reasoning"] is True


def test_gateway_read_yaml_layers_managed_wins(home, tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("display:\n  skin: managed_skin\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    try:
        (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
        d = _drop_in_dir(home)
        (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
        from gateway.config_loader import read_yaml_layers

        assert read_yaml_layers(home)["display"]["skin"] == "managed_skin"
    finally:
        managed_scope.invalidate_managed_cache()


def test_gateway_read_yaml_layers_dir_from_passed_home(home, tmp_path):
    """The gateway loader probes foreign profiles: the drop-in dir must derive from the
    PASSED home, never the active one."""
    other = tmp_path / "other-home"
    other.mkdir()
    (other / "config.yaml").write_text("display:\n  skin: other_user_skin\n", encoding="utf-8")
    d = other / "config.d"
    d.mkdir()
    (d / "10-a.yaml").write_text("display:\n  skin: other_dropin_skin\n", encoding="utf-8")
    from gateway.config_loader import read_yaml_layers

    # The ACTIVE home (home) has no drop-ins at all.
    assert read_yaml_layers(other)["display"]["skin"] == "other_dropin_skin"


# ── Loader integration: config_effective.load_user_config_effective ──────────

def test_load_user_config_effective_applies_drop_ins_without_defaults(home):
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    from hermes_cli.config_effective import load_user_config_effective

    cfg = load_user_config_effective(home / "config.yaml")
    assert cfg["display"]["skin"] == "dropin_skin"
    # No DEFAULT_CONFIG merge: a defaults-only section is absent, not backfilled.
    assert "compression" not in cfg


def test_effective_tracks_drop_in_env_ref(home, monkeypatch):
    """The overlay lands BEFORE the env snapshot, so a drop-in's ${VAR} ref invalidates
    the effective cache when the var changes value."""
    monkeypatch.setenv("CONFIG_D_EFFECTIVE_TOKEN", "first")
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("custom_token: ${CONFIG_D_EFFECTIVE_TOKEN}\n", encoding="utf-8")
    from hermes_cli.config_effective import load_user_config_effective

    assert load_user_config_effective(home / "config.yaml")["custom_token"] == "first"
    monkeypatch.setenv("CONFIG_D_EFFECTIVE_TOKEN", "second")
    assert load_user_config_effective(home / "config.yaml")["custom_token"] == "second"


# ── WRITE-BACK PURITY (the critical regression) ──────────────────────────────

def test_raw_reads_exclude_drop_ins(home):
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text(
        "display:\n  skin: dropin_skin\ncustom_drop_in_key: from_drop_in\n", encoding="utf-8")
    raw = config_mod.read_raw_config()
    assert "custom_drop_in_key" not in raw
    assert raw["display"]["skin"] == "user_skin"  # file-only view
    raw2 = config_mod.read_user_config_raw()
    assert "custom_drop_in_key" not in raw2
    assert raw2["display"]["skin"] == "user_skin"
    # The loader still applies them — the purity is on the raw/write-back side only.
    assert config_mod.load_config()["display"]["skin"] == "dropin_skin"


def test_save_config_refuses_when_a_drop_in_cannot_be_parsed(home):
    """An unreadable drop-in's keys are unknown, so the save guard cannot tell which leaves it
    shadows. Guessing would either freeze that drop-in's values into config.yaml or silently
    drop the caller's edit, so the save is refused and config.yaml is left untouched."""
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "50-broken.yaml").write_text("display: [broken yaml\n", encoding="utf-8")
    # The broken drop-in is skipped at load time (fail-open), so config still loads.
    assert config_mod.load_config()["display"]["skin"] == "user_skin"
    before = (home / "config.yaml").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="could not be read"):
        config_mod.save_config(config_mod.load_config())
    assert (home / "config.yaml").read_text(encoding="utf-8") == before
    # Repairing the drop-in restores normal saves.
    (d / "50-broken.yaml").write_text("display:\n  skin: fixed\n", encoding="utf-8")
    config_d.invalidate_config_d_cache()
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod.save_config(config_mod.load_config())
    text = (home / "config.yaml").read_text(encoding="utf-8")
    assert "broken" not in text  # the repaired drop-in's value is still not persisted
    assert "fixed" not in text


def test_absent_config_d_dir_stays_silent(home, caplog):
    """A profile with no config.d is the common case: an empty overlay and no log noise."""
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "dropin_skin"
    (d / "10-a.yaml").unlink()
    d.rmdir()
    with caplog.at_level(logging.ERROR, logger="hermes_cli.config_d"):
        assert config_d.load_config_d() == {}
    assert "could not be listed" not in caplog.text


@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root lists a 0o000 dir")
def test_unreadable_config_d_dir_is_reported_not_silent(home, caplog):
    """A config.d dir that exists but cannot be listed must not silently drop every override.

    The distinction that matters: an ABSENT dir returns ``[]`` quietly (a normal profile), while a
    PRESENT-but-unlistable one applies NO override at all — so it has to say so, or the user sees a
    drop-in that is simply not working. Uses a real directory mode, not a patched ``iterdir``.
    """
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "dropin_skin"
    os.chmod(d, 0o000)
    try:
        with caplog.at_level(logging.ERROR, logger="hermes_cli.config_d"):
            assert config_d.load_config_d() == {}  # not listed => no override applied
        assert "could not be listed" in caplog.text
        assert str(d) in caplog.text  # the warning names the directory to fix
    finally:
        os.chmod(d, 0o755)  # pytest's tmp_path cleanup needs it back


def test_save_config_never_persists_drop_in_keys(home):
    """load_config -> mutate -> save_config must not freeze drop-in values into
    config.yaml, where they would sit shadowed by the drop-in forever."""
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text(
        "display:\n  skin: dropin_skin\nagent:\n  max_turns: 42\n", encoding="utf-8")
    cfg = config_mod.load_config()
    assert cfg["display"]["skin"] == "dropin_skin"
    cfg["custom_new_key"] = 1
    config_mod.save_config(cfg)
    text = (home / "config.yaml").read_text(encoding="utf-8")
    assert "dropin_skin" not in text       # drop-in value not persisted
    assert "max_turns: 42" not in text     # drop-in leaf stripped from the bulk save
    assert "custom_new_key" in text        # the actual edit landed
    # The drop-in still wins at load time after the save.
    assert config_mod.load_config()["display"]["skin"] == "dropin_skin"


def test_set_config_value_on_drop_in_overridden_key(home, capsys):
    """config set is non-blocking: it writes config.yaml and prints the override notice."""
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    config_mod.set_config_value("display.skin", "cli_skin")
    out = capsys.readouterr().out
    assert "✓ Set display.skin = cli_skin" in out
    assert "config.d" in out and "10-a.yaml" in out  # override notice naming the drop-in
    assert "cli_skin" in (home / "config.yaml").read_text(encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "dropin_skin"  # drop-in still wins


# ── Recovery + terminal bridge ───────────────────────────────────────────────

def test_broken_config_yaml_serves_last_known_good_with_drop_ins(home):
    """A config.yaml broken mid-edit must not silently discard valid drop-ins: the
    recovered last-known-good user layer is served WITH the current drop-ins applied."""
    (home / "config.yaml").write_text(
        "display:\n  skin: user_skin\nagent:\n  max_turns: 11\n", encoding="utf-8")
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("display:\n  skin: dropin_skin\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "dropin_skin"  # primes backups
    (home / "config.yaml").write_text("display: [broken yaml\n", encoding="utf-8")
    # Drop the in-process caches so recovery comes from backups/config/*.good.*.
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    cfg = config_mod.load_config()
    assert cfg["display"]["skin"] == "dropin_skin"  # drop-ins applied on the recovered layer
    assert cfg["agent"]["max_turns"] == 11          # LKG user values recovered


def test_in_process_last_known_good_reapplies_a_drop_in_edited_while_broken(home):
    """The LKG arm a live gateway actually takes must not freeze the overlay it last expanded.

    This is the other half of the recovery contract, and the arm the backup test above steps
    around by clearing ``_LAST_EXPANDED_CONFIG_BY_PATH``: here the in-process LKG survives, so the
    served config is the one expanded BEFORE the drop-in edit. A drop-in is an independent valid
    file, so editing it while config.yaml is broken must still take effect.
    """
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
    d = _drop_in_dir(home)
    drop = d / "10-a.yaml"
    drop.write_text("display:\n  skin: dropin_v1\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "dropin_v1"  # primes the in-process LKG
    (home / "config.yaml").write_text("display: [broken yaml\n", encoding="utf-8")
    _rewrite(drop, "display:\n  skin: dropin_v2\n")  # the user fixes the drop-in, not config.yaml
    # Only the load cache goes: _LAST_EXPANDED_CONFIG_BY_PATH deliberately survives.
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_d.invalidate_config_d_cache()
    assert config_mod.load_config()["display"]["skin"] == "dropin_v2"


def test_in_process_last_known_good_keeps_admin_pins_over_reapplied_drop_in(
        home, tmp_path, monkeypatch):
    """Re-applying the overlay on the in-process LKG must stay UNDER the managed layer.

    The in-process LKG is an ALREADY-merged config, so a naive re-overlay would let a drop-in
    shadow an administrator-pinned value — inverting the documented order.
    """
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("display:\n  skin: managed_skin\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    try:
        (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")
        d = _drop_in_dir(home)
        drop = d / "10-a.yaml"
        drop.write_text("display:\n  skin: dropin_v1\n", encoding="utf-8")
        assert config_mod.load_config()["display"]["skin"] == "managed_skin"
        (home / "config.yaml").write_text("display: [broken yaml\n", encoding="utf-8")
        _rewrite(drop, "display:\n  skin: dropin_v2\n")
        config_mod._LOAD_CONFIG_CACHE.clear()
        config_d.invalidate_config_d_cache()
        assert config_mod.load_config()["display"]["skin"] == "managed_skin"
    finally:
        managed_scope.invalidate_managed_cache()


def test_drop_in_terminal_section_counts_as_file_sourced(home):
    """Same contract as cli_config_load's _file_has_terminal_config: only a file-sourced
    terminal section may overwrite .env-set TERMINAL_* vars — and a drop-in IS a file."""
    (home / "config.yaml").write_text("display:\n  skin: user_skin\n", encoding="utf-8")  # no terminal section
    d = _drop_in_dir(home)
    (d / "10-a.yaml").write_text("terminal:\n  backend: docker\n", encoding="utf-8")
    env = {"TERMINAL_ENV": "local"}  # a stale .env-set value
    config_mod.apply_terminal_config_to_env(env=env)
    assert env["TERMINAL_ENV"] == "docker"  # the drop-in counts as file-sourced config
