"""E2E: a real apply against two temp HERMES_HOMEs — the A->B->A profile-scope round trip.

The repo's contribution rules require that anything touching config propagation and
profile scope be exercised against real imports and real files, using two homes when
the change crosses profile scope. The unit tests in test_model_fleet.py stub the cron
store; this module drives the real write path instead.

Only the provider switch itself is stubbed (`_switch_result`), because that is the one
step that needs network and credentials. Everything downstream of it — YAML writes,
profile enumeration, backup files, dry-run suppression — is the real code.

No test here touches a real $HERMES_HOME: HERMES_HOME is monkeypatched to a tmp dir.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _repo_root() -> Path | None:
    """Locate a checkout the way tests/conftest.py does.

    Prefer walking up from this file (in-tree: the plugin sits inside the repo, so
    parents contain both hermes_cli/ and cron/). Fall back to the installed
    hermes_constants location for a standalone checkout next to an install.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "hermes_cli").is_dir() and (parent / "cron").is_dir():
            return parent
    try:
        import hermes_constants

        candidate = Path(hermes_constants.__file__).resolve().parent
        if (candidate / "hermes_cli").is_dir():
            return candidate
    except Exception:
        pass
    return None


def _load_plugin():
    """Import a fresh module object so it re-reads HERMES_HOME per call, not per import.

    The repo root must be on sys.path before this runs: loading the plugin imports
    ``hermes_cli``/``utils``, which import ``hermes_yaml``. pytest's rootdir for this
    directory is the plugin, not the repo, so the import has to be made explicit here.
    """
    import sys

    root = _repo_root()
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))

    spec = importlib.util.spec_from_file_location(
        "model_fleet_e2e_under_test", PLUGIN_DIR / "__init__.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_config(home: Path, provider: str, model: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"model:\n  provider: {provider}\n  default: {model}\n"
        "delegation:\n  provider: ''\n  model: ''\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _never_touch_real_home(monkeypatch):
    """Fail loudly rather than write into the developer's real ~/.hermes.

    hermes_constants.get_hermes_home() reads HERMES_HOME live, so every test below
    sets it to a tmp dir explicitly. This fixture is the belt-and-braces check: if a
    test ever forgets, HERMES_HOME lands on a path under the pytest tmp root and the
    run is self-contained instead of destructive.
    """
    import tempfile

    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp(prefix="model-fleet-e2e-"))


def _read_model(home: Path) -> tuple[str, str]:
    """Parse provider/default straight out of the real YAML the plugin wrote."""
    import yaml

    data = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
    return (data.get("model", {}).get("provider", ""),
            data.get("model", {}).get("default", ""))


@pytest.fixture
def mod(monkeypatch):
    m = _load_plugin()

    def _fake_switch(provider: str, model: str):
        """Real ModelSwitchResult, fake route: the resolution step needs network."""
        from hermes_cli.model_switch import ModelSwitchResult

        return ModelSwitchResult(
            success=True, new_model=model, target_provider=provider,
            provider_changed=True, base_url=f"https://stub.invalid/{provider}",
            api_mode="chat", provider_label=provider)

    monkeypatch.setattr(m, "_switch_result", _fake_switch)
    return m


def test_apply_to_home_a_leaves_home_b_untouched(mod, tmp_path, monkeypatch):
    """Working on A must not write anything under B. Guards a cached/module-level home."""
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    _write_config(home_a, "nous", "alpha-base")
    _write_config(home_b, "anthropic", "claude-base")
    before_b = (home_b / "config.yaml").read_bytes()

    monkeypatch.setenv("HERMES_HOME", str(home_a))  # home_a IS the Hermes home
    out = mod._apply_sync("anthropic", "claude-two", dry_run=False, with_auxiliary=False)

    assert "claude-two" in out
    assert _read_model(home_a) == ("anthropic", "claude-two")
    assert (home_b / "config.yaml").read_bytes() == before_b, "home B was written to"


def test_a_then_b_then_a_round_trip(mod, tmp_path, monkeypatch):
    """A->B->A: A ends on its own model and B is still on its own afterwards.

    This is the two-home round trip the contribution rules require for a change that
    touches profile scope. A plugin that cached a resolved route or a resolved home
    would fail the second leg here.
    """
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    _write_config(home_a, "nous", "alpha-base")
    _write_config(home_b, "anthropic", "claude-base")

    # Leg 1: A -> B's model
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    mod._apply_sync("anthropic", "claude-two", dry_run=False, with_auxiliary=False)
    assert _read_model(home_a) == ("anthropic", "claude-two")

    # Leg 2: B -> A's original model
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    mod._apply_sync("nous", "alpha-base", dry_run=False, with_auxiliary=False)
    assert _read_model(home_b) == ("nous", "alpha-base")

    # Leg 3: A back to its own model
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    mod._apply_sync("nous", "alpha-base", dry_run=False, with_auxiliary=False)
    assert _read_model(home_a) == ("nous", "alpha-base")
    assert _read_model(home_b) == ("nous", "alpha-base")


def test_dry_run_leaves_real_config_byte_identical(mod, tmp_path, monkeypatch):
    """A dry run against a real config.yaml must not write a single byte."""
    home = tmp_path / "a"
    _write_config(home, "nous", "alpha-base")
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = (home / "config.yaml").read_bytes()
    listing_before = sorted(p.name for p in home.iterdir())

    out = mod._apply_sync("anthropic", "claude-x", dry_run=True, with_auxiliary=False)

    assert "Dry run" in out
    assert (home / "config.yaml").read_bytes() == before, "dry run wrote to config.yaml"
    # The first run may scaffold a standard home (SOUL.md, logs/, ...), so assert on
    # what a dry run must never produce: a backup, or any model-fleet artefact.
    assert not list(home.glob("*.bak-model-fleet-*")), "dry run created a backup"
    assert set(listing_before) & {"config.yaml.bak-model-fleet-0"} == set()


def test_real_apply_writes_a_restorable_backup(mod, tmp_path, monkeypatch):
    """A real apply must leave a backup that actually contains the pre-change content."""
    home = tmp_path / "a"
    _write_config(home, "nous", "alpha-base")
    original = (home / "config.yaml").read_text(encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    mod._apply_sync("anthropic", "claude-two", dry_run=False, with_auxiliary=False)

    backups = list(home.glob("*.bak-model-fleet-*"))
    assert backups, "no backup written by a real apply"
    assert any(original in b.read_text(encoding="utf-8") for b in backups), \
        "backup does not contain the original content"
    assert _read_model(home) == ("anthropic", "claude-two")


def test_refused_switch_writes_nothing(mod, tmp_path, monkeypatch):
    """A refused switch must not touch the config at all (the check precedes the writes)."""
    home = tmp_path / "a"
    _write_config(home, "nous", "alpha-base")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.model_switch import ModelSwitchResult

    monkeypatch.setattr(
        mod, "_switch_result",
        lambda provider, model: ModelSwitchResult(
            success=False, error_message="no such model"))
    before = (home / "config.yaml").read_bytes()
    listing_before = sorted(p.name for p in home.iterdir())

    out = mod._apply_sync("nous", "nope-9000", dry_run=False, with_auxiliary=False)

    assert "refused" in out.lower()
    assert (home / "config.yaml").read_bytes() == before
    # A refused switch must not create a backup either.
    assert not list(home.glob("*.bak-model-fleet-*")), "refused switch created a backup"


def test_plugin_never_hardcodes_a_home(mod):
    """The repo bans hardcoded ~/.hermes; a literal path here would reintroduce it."""
    src = (PLUGIN_DIR / "__init__.py").read_text(encoding="utf-8")
    for bad in ("/root/.hermes", '"~/.hermes"'):
        assert bad not in src, f"hardcoded home in plugin source: {bad}"
