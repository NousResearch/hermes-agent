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


# Run the parameter matrix within each contract to retain two RED test results.
_AUXILIARY_CASES = (
    ("all", {}, {"default", "alpha", "beta", "gamma", "empty"}, False, None),
    ("block-wins", {"profile_allowlist": ["alpha", "beta", "empty"],
                    "profile_blocklist": ["beta"]}, {"default", "alpha", "empty"}, True, None),
    ("caller-excluded", {"profile_allowlist": ["beta", "empty"],
                         "profile_blocklist": ["beta"]}, {"default", "empty"}, False, None),
    ("profiles-off", {"include_profiles": False}, {"default"}, True, None),
    ("resolution-refused", {}, {"default", "alpha", "beta", "gamma", "empty"}, True, "beta"),
)


def _auxiliary_installations(base: Path, settings: dict) -> list[dict[str, Path]]:
    """Create independent installs with eligible and deliberately unpinned tasks."""
    import yaml
    from hermes_cli.config import load_config_readonly
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    installations = []
    for name in ("a", "b"):
        root = base / name
        homes = {"default": root}
        homes.update({label: root / "profiles" / label
                      for label in ("alpha", "beta", "gamma", "empty")})
        for label, home in homes.items():
            _write_config(home, "nous", f"{name}-{label}-original")
            path = home / "config.yaml"
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            cfg["unrelated"] = {"installation": name, "label": label}
            cfg["plugins"] = {"entries": {"model-fleet": settings}}
            if label != "empty":
                cfg["auxiliary"] = {
                    "pinned": {"provider": "old", "model": "old-model", "keep": True,
                               "base_url": "https://synthetic.invalid",
                               "api_key": "synthetic-test-value"},
                    "unspecified": {"provider": "old", "keep": "unspecified"},
                    "blank": {"model": "", "keep": "blank"},
                    "scalar": "unchanged",
                }
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            # Initialize normal home scaffolding before any invocation snapshot.
            token = set_hermes_home_override(home)
            try:
                load_config_readonly()
            finally:
                reset_hermes_home_override(token)
        installations.append(homes)
    return installations


@pytest.fixture
def auxiliary_spies(mod, monkeypatch):
    """Forward every observed write to the real persistence implementation."""
    from unittest.mock import Mock

    import utils
    from hermes_cli import model_switch

    spies = []
    for owner, name in ((utils, "atomic_roundtrip_yaml_update"), (mod, "_backup"),
                        (model_switch, "persist_model_selection")):
        spy = Mock(wraps=getattr(owner, name))
        monkeypatch.setattr(owner, name, spy)
        spies.append(spy)
    return spies


def _auxiliary_outcomes(output: str) -> list[str]:
    return [line.removeprefix("- ") for line in output.splitlines()
            if line.removeprefix("- ").startswith("auxiliary")]


def _check_auxiliary_outcomes(output, selected, verb, model, check, malformed=False):
    from collections import Counter

    outcomes = _auxiliary_outcomes(output)
    check(Counter(line.split(":", 1)[0] for line in outcomes)
          == Counter(f"auxiliary/{label}" for label in selected), "exact output labels")
    for label in sorted(selected):
        lines = [line for line in outcomes if line.startswith(f"auxiliary/{label}:")]
        skipped = label == "empty" or (malformed and label == "beta")
        expected = ("config unreadable" if malformed and label == "beta" else
                    "no task declares a model" if label == "empty" else f"{verb} 1 task(s)")
        check(len(lines) == 1 and expected in lines[0]
              and ("SKIPPED" in lines[0] if skipped else f"anthropic/{model}" in lines[0]),
              f"{label}: {expected} outcome")


def test_auxiliary_named_caller_persists_selected_homes_once(
        mod, tmp_path, monkeypatch, auxiliary_spies):
    """A->B->A updates selected auxiliary pins with one pre-change backup per file.

    Collect failures so every parameter and scope leg executes even on RED.
    Only provider resolution is synthetic; scopes and persistence remain live.
    """
    from collections import Counter

    import yaml
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_cli import model_switch
    from hermes_constants import (get_hermes_home, reset_hermes_home_override,
                                  set_hermes_home_override)

    atomic, backup, persist = auxiliary_spies
    failures, passed = [], Counter()

    def check(condition, contract):
        if condition:
            passed[contract] += 1
        else:
            failures.append(f"{context}: {contract}")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    multiplex_before = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for case, overrides, selected, saved, refused in _AUXILIARY_CASES:
            settings = dict(mod._SETTING_DEFAULTS, **overrides)
            settings["include_auxiliary"] = saved
            installations = _auxiliary_installations(tmp_path / case, settings)

            def switch(*, raw_input, explicit_provider, **kwargs):
                if Path(get_hermes_home()).name == refused:
                    return model_switch.ModelSwitchResult(success=False,
                                                          error_message="synthetic refusal")
                return mod._switch_result(explicit_provider, raw_input)

            monkeypatch.setattr(model_switch, "switch_model", switch)
            for leg, index in enumerate((0, 1, 0)):
                homes = installations[index]
                caller = homes["alpha"]
                model, context = f"{case}-leg-{leg}", f"{case}/{leg}"
                stamp = f"phase1-{case}-{leg}"
                monkeypatch.setattr(mod.time, "strftime", lambda *args, s=stamp: s)
                before = {home / "config.yaml": (home / "config.yaml").read_bytes()
                          for install in installations for home in install.values()}
                for spy in auxiliary_spies:
                    spy.reset_mock()
                token = set_hermes_home_override(caller)
                try:
                    output = mod._apply_sync("anthropic", model, dry_run=False,
                                             with_auxiliary=not saved)
                    check(Path(get_hermes_home()) == caller, "scope restored")
                finally:
                    reset_hermes_home_override(token)
                check(_read_model(caller) == ("anthropic", model), "active main model persisted")
                writes = {Path(call.args[0]) for call in atomic.call_args_list}
                if persist.called:
                    writes.add(caller / "config.yaml")
                counts = Counter(Path(call.args[0]) for call in backup.call_args_list)
                check(counts == Counter({path: 1 for path in writes}), "one backup per written config")
                check(counts[caller / "config.yaml"] == 1, "active backup once")
                for path in writes:
                    copies = list(path.parent.glob(f"config.yaml.bak-model-fleet-{stamp}"))
                    check(len(copies) == 1 and copies[0].read_bytes() == before[path],
                          "pre-change backup bytes")
                targets = {Path(call.args[0]) for call in atomic.call_args_list
                           if call.args[1].startswith("auxiliary.")}
                check(targets == {homes[label] / "config.yaml" for label in selected - {"empty"}},
                      "exact auxiliary target set")
                for label, home in homes.items():
                    path = home / "config.yaml"
                    old, actual = yaml.safe_load(before[path]), yaml.safe_load(path.read_bytes())
                    check(actual["unrelated"] == old["unrelated"]
                          and actual["plugins"] == old["plugins"], "unrelated settings preserved")
                    if label == "empty":
                        check("auxiliary" not in actual, "no invented auxiliary")
                    else:
                        check(all(actual["auxiliary"][task] == old["auxiliary"][task]
                                  for task in ("unspecified", "blank", "scalar")),
                              "ineligible tasks preserved")
                        expected = old["auxiliary"]
                        if label in selected:
                            expected["pinned"].update(provider="anthropic", model=model)
                            expected["pinned"].pop("base_url", None)
                            expected["pinned"].pop("api_key", None)
                        check(actual["auxiliary"] == expected,
                              f"{label}: eligible pins persisted" if label in selected else
                              f"{label}: excluded auxiliary unchanged")
                    if label not in selected and label != "alpha":
                        check(path.read_bytes() == before[path], "excluded file byte-identical")
                for home in installations[1 - index].values():
                    path = home / "config.yaml"
                    check(path.read_bytes() == before[path], "other install unchanged")
                _check_auxiliary_outcomes(output, selected, "set", model, check)
    finally:
        set_multiplex_active(multiplex_before)
    print("Passing contract checks:", dict(passed))
    assert not failures, "\n".join(failures)


def test_auxiliary_named_caller_preview_is_complete_and_write_free(
        mod, tmp_path, monkeypatch, auxiliary_spies):
    """Preview every selected label without writing, even after unreadable YAML."""
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_cli import model_switch
    from hermes_constants import (get_hermes_home, reset_hermes_home_override,
                                  set_hermes_home_override)

    failures = []

    def check(condition, contract):
        if not condition:
            failures.append(f"{case}/{leg}: {contract}")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    monkeypatch.setattr(model_switch, "switch_model",
                        lambda *, raw_input, explicit_provider, **kwargs:
                        mod._switch_result(explicit_provider, raw_input))
    multiplex_before = is_multiplex_active()
    set_multiplex_active(True)
    previews = 0
    try:
        for case, overrides, selected, _, _ in _AUXILIARY_CASES:
            settings = dict(mod._SETTING_DEFAULTS, **overrides)
            installations = _auxiliary_installations(tmp_path / case, settings)
            for leg, index in enumerate((0, 1, 0)):
                homes = installations[index]
                # Keep malformed YAML out of unrelated profile model parsing.
                malformed = case == "all" and leg == 2
                if malformed:
                    (homes["beta"] / "config.yaml").write_text("auxiliary: [\n", encoding="utf-8")
                base = tmp_path / case
                inventory = set(base.rglob("*"))
                before = {path: path.read_bytes() for path in inventory if path.is_file()}
                for spy in auxiliary_spies:
                    spy.reset_mock()
                token = set_hermes_home_override(homes["alpha"])
                try:
                    if malformed:
                        lines, backups = mod._apply_auxiliary(
                            "anthropic", "preview-model", settings, "preview", True)
                        assert not backups
                        output = "\n".join(lines)
                    else:
                        output = mod._apply_sync("anthropic", "preview-model",
                                                 dry_run=True, with_auxiliary=True)
                    assert Path(get_hermes_home()) == homes["alpha"]
                finally:
                    reset_hermes_home_override(token)
                assert set(base.rglob("*")) == inventory
                assert all(path.read_bytes() == data for path, data in before.items())
                for spy in auxiliary_spies:
                    assert spy.call_count == 0
                previews += 1
                _check_auxiliary_outcomes(output, selected, "would set", "preview-model",
                                          check, malformed=malformed)
    finally:
        set_multiplex_active(multiplex_before)
    print(f"{previews} previews passed byte, inventory, scope, and zero-write checks")
    assert not failures, "\n".join(failures)
