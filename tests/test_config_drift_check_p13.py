"""P13 isolation proof for scripts/config-drift-check.py.

Verifies config-drift-check.py:
- reads HERMES_HOME-derived config.yaml (not a hard-coded absolute path)
- silent + exit 0 when enabled_skills exactly matches the expected set
- reports drift labels + exit 1 when one missing + one extra
- prints "[DRIFT] Check failed:" + exit 2 on malformed YAML
- never mutates the config file (hash fixture before/after)
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "config-drift-check.py"

# The expected set must mirror the script's own `expected` literal. We assert
# a contract: with the exact set, exit 0 and empty stdout. We do NOT freeze the
# literal contents here (the script's expected set is intentionally editable);
# we import it to reuse the current value.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("_drift_probe", SCRIPT)
_probe = importlib.util.module_from_spec(_spec)
# Stop exec at import — the script runs code at module top level, so we exec it
# in a sandboxed env where the open() raises FileNotFoundError, then read the
# `expected` constant. Simpler: parse the literal via a temp HERMES_HOME.
# Instead, exec the module with a bogus config path so the try/except catches.
_orig_open = open
def _bump(*a, **k):
    raise FileNotFoundError("sandbox")
import builtins
builtins.open = _bump
try:
    _spec.loader.exec_module(_probe)
except SystemExit:
    pass
finally:
    builtins.open = _orig_open
EXPECTED_SET = _probe.expected


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(hermes_home: Path):
    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    # Strip a possibly-inherited HERMES_KANBAN_DB so no leak.
    env.pop("HERMES_KANBAN_DB", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


def _write_config(hermes_home: Path, skills: list[str]) -> Path:
    cfg = hermes_home / "config.yaml"
    import yaml
    # Step-6: the drift checker now validates the fleet-policy surfaces in
    # addition to the skills allow-list, so the minimal fixture carries the
    # policy-compliant base (the isolation assertions below are unchanged).
    cfg.write_text(yaml.safe_dump({**_policy_compliant_root_config(),
                                   "skills": {"enabled_skills": skills}}))
    return cfg


def test_exact_expected_set_is_silent_exit_0(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    cfg = _write_config(home, sorted(EXPECTED_SET))
    before = _sha(cfg)
    r = _run(home)
    assert r.returncode == 0, r.stderr
    assert r.stdout == ""
    assert _sha(cfg) == before, "config mutated"


def test_one_missing_one_extra_reports_drift_exit_1(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    # remove one expected, add one unknown
    skills = sorted(EXPECTED_SET - {"arxiv"} | {"bogus-extra-skill"})
    cfg = _write_config(home, skills)
    before = _sha(cfg)
    r = _run(home)
    assert r.returncode == 1, r.stderr
    assert "[DRIFT]" in r.stdout
    assert "arxiv" in r.stdout, "missing skill not named"
    assert "bogus-extra-skill" in r.stdout, "extra skill not named"
    assert _sha(cfg) == before


def test_malformed_yaml_reports_check_failed_exit_2(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    cfg = home / "config.yaml"
    cfg.write_text("skills: {enabled_skills: [unterminated\n")
    before = _sha(cfg)
    r = _run(home)
    assert r.returncode == 2, r.stderr
    assert "[DRIFT] Check failed" in r.stdout
    assert _sha(cfg) == before


def test_resolves_via_hermes_home_env_not_hardcoded_path(tmp_path):
    """If the script still hard-codes /home/kensei/.hermes/config.yaml it will
    either FileNotFoundError (no such file under the temp) or read the real
    file. Either way the exact-match case fails. This is the RED guard."""
    home = tmp_path / "hermes"
    home.mkdir()
    _write_config(home, sorted(EXPECTED_SET))
    r = _run(home)
    # Must succeed against the temp HERMES_HOME, proving env-resolution.
    assert r.returncode == 0, (
        f"script did not honour HERMES_HOME (hard-coded path?). "
        f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"
    )


# ===========================================================================
# STEP 6 (fleet-policy extension) — appended; does not alter the four P13
# isolation tests above in any way.
#
# Contract under test (scripts/config-drift-check.py):
#   rc 0 = silent healthy, rc 1 = drift (concise deterministic "[DRIFT] ..."
#   lines, no timestamps), rc 2 = execution/parse error.
#
# Approved surfaces ONLY — never providers, model, base_url,
# fallback_providers, fallback_reasoning, auxiliary, custom_providers,
# credential pools, MCP servers, or secret values.
# ===========================================================================

import copy  # noqa: E402

import yaml as _yaml6  # noqa: E402


# ── expected policy matrix (step-6 approved surfaces) ──────────────────────
# The policy values live in this test block only — a policy change is a diff
# to this section and to scripts/config-drift-check.py together.

SCHEMA_VERSION = 39  # must stay in sync with KenseiAgent DEFAULT_CONFIG

# profiles requiring the FULL policy surface: curator safety, schema version,
# (budgets / personality only where listed separately below).
_FULL_POLICY_PROFILES = [
    "misa-misa", "remii", "wesker", "gojo", "octacon", "ceecee",
    "denji", "light", "quan", "dezzy", "kensei-review", "sirvir",
]

# budgets only.
_BUDGET_PROFILE_MAX_TURNS_NONE = {
    "octacon", "remii", "wesker", "quan", "dezzy", "denji", "light",
    "kensei-review", "orchestrator",
}
_BUDGET_PROFILE_DELEGATION = {
    "dezzy", "gojo", "kensei-review", "light", "octacon", "remii", "wesker",
}

# display.personality.
_PERSONALITY = {
    "mrhermagi": "minimal", "wesker": "minimal", "gojo": "minimal",
    "octacon": "minimal", "remii": "minimal", "light": "minimal",
    "kensei-review": "minimal", "sirvir": "kawaii",
}

# profiles that must have curator.enabled explicitly FALSE.
_CURATOR_DISABLED_PROFILES = {"mrhermagi"}

# profiles exempt from the schema-version check.
# NOTE: intentionally empty for the step-6 policy version; deferred migrations
# are flagged on purpose (task R3). Profiles absent on disk (sirvir on some
# rigs) are skipped by the checker; a test fixtures-only home controls which
# profiles exist.
_SCHEMA_ALL_PROFILES = (
    _FULL_POLICY_PROFILES + ["mrhermagi", "orchestrator"]
)

# provider-ish surfaces that must NEVER be inspected/reported (documentation
# of the exclusion contract; proven by test_provider_surface_mutations_*):
# model, base_url, fallback_providers, fallback_reasoning, auxiliary,
# custom_providers, api_key, credential_pool, provider, mcp_servers.


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_yaml6.safe_dump(data, sort_keys=False))


def _policy_compliant_root_config() -> dict:
    """Root-config fields that satisfy the step-6 policy (reused by both the
    legacy `_write_config` fixture and the full fleet fixture)."""
    return {
        "_config_version": SCHEMA_VERSION,
        "curator": {
            "enabled": True,
            "interval_hours": 168,  # non-checked surface, free value
            "stale_after_days": 45,
            "archive_after_days": 120,
            "consolidate": False,
            "prune_builtins": False,
            "backup": {"enabled": True, "keep": 5},
        },
        "agent": {"max_turns": "none"},
        "delegation": {"max_iterations": 250, "max_concurrent_children": 10},
        "platform_toolsets": {
            "cli": [
                "guard", "terminal", "file", "skills", "delegation",
                "memory", "session_search", "cronjob", "todo",
            ]
        },
    }


def _make_policy_compliant_home(base: Path):
    """Build a temp HERMES_HOME whose root + all scoped profiles satisfy the
    full step-6 policy. Re-entrant within one test (covers only `base/hermes`
    already existing when a test builds multiple homes). Returns
    (home, root_cfg_dict)."""
    home = base / "hermes"
    if home.exists():
        home = base / f"hermes-{len(list(base.iterdir()))}"
    home.mkdir(parents=True, exist_ok=False)

    root = {
        **_policy_compliant_root_config(),
        "skills": {"enabled_skills": sorted(EXPECTED_SET)},
        # provider-ish noise that must be ignored entirely
        "model": {"default": "evil-provider/model-x"},
        "providers": {"openrouter": {"api_key": "sk-not-a-real-key"}},
        "fallback_providers": [{"provider": "groq", "model": "m"}],
        "auxiliary": {"curator": {"provider": "openrouter"}},
    }
    _write_yaml(home / "config.yaml", root)

    for name in _SCHEMA_ALL_PROFILES:
        prof = {
            "_config_version": SCHEMA_VERSION,
            "curator": {
                "enabled": name not in _CURATOR_DISABLED_PROFILES,
                "interval_hours": 168,
                "stale_after_days": 45,
                "archive_after_days": 120,
                "consolidate": False,
                "prune_builtins": False,
                "backup": {"enabled": True},
            },
            "display": {"personality": _PERSONALITY.get(name)},
            "skills": {"enabled_skills": []},
        }
        if name in _BUDGET_PROFILE_MAX_TURNS_NONE:
            prof["agent"] = {"max_turns": "none"}
        elif name in {"misa-misa", "ceecee", "gojo", "mrhermagi", "sirvir"}:
            # profiles not in the max_turns-none set get a different free value
            prof["agent"] = {"max_turns": 90}
        if name in _BUDGET_PROFILE_DELEGATION:
            prof["delegation"] = {
                "max_iterations": 250, "max_concurrent_children": 10}
        prof["model"] = {"default": f"evil/{name}-model"}
        prof["auxiliary"] = {"curator": {"provider": "openrouter"}}
        _write_yaml(home / "profiles" / name / "config.yaml", prof)

    for name in _PERSONALITY:
        soul = home / "profiles" / name / "SOUL.md"
        soul.parent.mkdir(parents=True, exist_ok=True)
        soul.write_text(f"# {name}\nsoul body for policy testing.\n")
    return home, root


# ── R1: absent root enabled_skills must NOT pre-empt other checks ──────────


def _run_step6(home: Path):
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env.pop("HERMES_KANBAN_DB", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


def test_absent_root_enabled_skills_does_not_skip_other_checks(tmp_path):
    """Bug-class: `enabled_skills: None` used to sys.exit(0) BEFORE the other
    checks ran (fail-unsafe). A deliberate schema_version drift downstream of
    the skills block must still be reported."""
    home = tmp_path / "hermes"
    home.mkdir()
    _write_yaml(home / "config.yaml", {
        "skills": {},                      # no enabled_skills key at all
        "_config_version": 1,              # deliberate drift below the block
        "curator": {"enabled": True, "stale_after_days": 45,
                    "archive_after_days": 120, "consolidate": False,
                    "prune_builtins": False, "backup": {"enabled": True}},
    })
    r = _run_step6(home)
    assert r.returncode == 1, (
        f"absent enabled_skills short-circuited the other checks "
        f"(rc={r.returncode}) stdout={r.stdout!r}"
    )
    assert "[DRIFT]" in r.stdout
    assert "schema:" in r.stdout


def test_absent_root_skills_healthy_home_stays_silent(tmp_path):
    """No enabled_skills + everything else fine → silent rc 0 (the
    implicit-discovery semantics are preserved)."""
    home = tmp_path / "hermes"
    home.mkdir()
    _write_yaml(home / "config.yaml", {
        "_config_version": SCHEMA_VERSION,
        "curator": {"enabled": True, "stale_after_days": 45,
                    "archive_after_days": 120, "consolidate": False,
                    "prune_builtins": False, "backup": {"enabled": True}},
        "agent": {"max_turns": "none"},
        "delegation": {"max_iterations": 250, "max_concurrent_children": 10},
        "platform_toolsets": {"cli": ["guard", "terminal", "file", "skills",
                                      "delegation", "memory",
                                      "session_search", "cronjob", "todo"]},
    })
    r = _run_step6(home)
    assert r.returncode == 0, r.stdout
    assert r.stdout == ""


# ── R2: schema version derived from code, root + all profiles matched ─────


def test_schema_version_matches_kenseiagent_default_config():
    """The expected schema version must be DERIVED from the KenseiAgent code
    (hermes_cli.config_defaults.DEFAULT_CONFIG), not duplicated as a
    drift-script literal."""
    spec = importlib.util.spec_from_file_location(
        "_drift_schema_probe", SCRIPT)
    assert spec and spec.loader
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    expected_version = DEFAULT_CONFIG.get("_config_version")
    assert isinstance(expected_version, int)
    # The script's expectation equals the code's value.
    assert probe.EXPECTED_SCHEMA_VERSION == expected_version


def test_root_schema_version_drift_detected(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    _write_yaml(home / "config.yaml", {"_config_version": 30})
    r = _run_step6(home)
    assert r.returncode == 1
    assert "_config_version" in r.stdout
    assert "39" in r.stdout, "expected version must be named in the drift line"


# ── R2b: deployed-cron copy (no checkout above it) still derives v39 ──────


def test_deployed_hermes_home_copy_derives_schema_version(tmp_path, monkeypatch):
    """The live deployment copies this script to HERMES_HOME/scripts/ and the
    daily cron runs it from there with cwd = scripts dir. parents[1] is then
    ~/.hermes (no hermes_cli above it). The derivation must fall through to
    HERMES_AGENT_ROOT / the canonical checkout instead of rc=2-ing every run.
    Contract: fail-closed stays — with NO resolvable checkout AND no cached
    import, the script must still report execution error (rc=2), not silence.
    """
    # Sanity guard: the probe must never report a fabricated fallback literal.
    import hermes_cli.config_defaults as cfg_defaults

    assert not hasattr(cfg_defaults, "_KENSEI_FALLBACK_SCHEMA_VERSION")
    # Copy the script into a bare HERMES_HOME/scripts (deployed layout) and
    # give it a minimal healthy root config so the only open question is the
    # schema derivation.
    bare_home = tmp_path / "bare-hermes-home"
    (bare_home / "scripts").mkdir(parents=True)
    deployed = bare_home / "scripts" / "config-drift-check.py"
    deployed.write_text(SCRIPT.read_text())
    home, _root = _make_policy_compliant_home(tmp_path)
    # Move the policy-compliant home's scripts copy into place: the deployed
    # script lives under HERMES_HOME/scripts (away from any checkout).
    (home / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, home / "scripts" / "config-drift-check.py")
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["HERMES_AGENT_ROOT"] = str(REPO_ROOT)
    env.pop("PYTHONPATH", None)
    r = subprocess.run(
        [sys.executable, str(deployed)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )
    assert r.returncode == 0, (
        f"deployed copy failed to derive schema version (rc={r.returncode}): "
        f"{r.stdout} {r.stderr}"
    )
    assert r.stdout == "", f"healthy home must stay silent: {r.stdout!r}"


def test_deployed_copy_fails_closed_without_any_checkout(tmp_path):
    """No resolvable checkout at all → rc=2 execution error (never silent)."""
    bare_home = tmp_path / "orphan-hermes-home"
    (bare_home / "scripts").mkdir(parents=True)
    deployed = bare_home / "scripts" / "config-drift-check.py"
    deployed.write_text(SCRIPT.read_text())
    env = dict(os.environ)
    env["HERMES_HOME"] = str(bare_home)
    env.pop("HERMES_AGENT_ROOT", None)
    env.pop("PYTHONPATH", None)
    r = subprocess.run(
        [sys.executable, str(deployed)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )
    assert r.returncode == 2, f"must fail closed, got rc={r.returncode}"
    assert "Check failed" in r.stdout


def test_profile_schema_version_drift_detected(tmp_path):
    """R2's intentional flag: profiles whose config.yaml has NOT yet been
    migrated to the current schema version are reported."""
    home, _root = _make_policy_compliant_home(tmp_path)
    stale_cfg = home / "profiles" / "remii" / "config.yaml"
    data = _yaml6.safe_load(stale_cfg.read_text())
    data["_config_version"] = 23
    _write_yaml(stale_cfg, data)
    r = _run_step6(home)
    assert r.returncode == 1, "deferred profile migration was not flagged"
    assert "schema:" in r.stdout
    assert "v23=1" in r.stdout


def test_profile_schema_version_missing_detected(tmp_path):
    home, _root = _make_policy_compliant_home(tmp_path)
    missing_cfg = home / "profiles" / "wesker" / "config.yaml"
    data = _yaml6.safe_load(missing_cfg.read_text())
    del data["_config_version"]
    _write_yaml(missing_cfg, data)
    r = _run_step6(home)
    assert r.returncode == 1
    assert "schema:" in r.stdout
    assert "missing=1" in r.stdout


def test_schema_drift_is_one_aggregated_line(tmp_path):
    """A fleet-wide version lag must not emit one #ops line per profile."""
    home, _root = _make_policy_compliant_home(tmp_path)
    for name, version in (("remii", 23), ("wesker", 33)):
        cfg = home / "profiles" / name / "config.yaml"
        data = _yaml6.safe_load(cfg.read_text())
        data["_config_version"] = version
        _write_yaml(cfg, data)
    missing_cfg = home / "profiles" / "gojo" / "config.yaml"
    missing = _yaml6.safe_load(missing_cfg.read_text())
    del missing["_config_version"]
    _write_yaml(missing_cfg, missing)

    r = _run_step6(home)
    schema_lines = [line for line in r.stdout.splitlines()
                    if line.startswith("[DRIFT] schema:")]
    assert r.returncode == 1
    assert len(schema_lines) == 1, r.stdout
    assert "3 config(s)" in schema_lines[0]
    assert "missing=1" in schema_lines[0]
    assert "v23=1" in schema_lines[0]
    assert "v33=1" in schema_lines[0]


def test_missing_required_profile_config_is_drift(tmp_path):
    home, _root = _make_policy_compliant_home(tmp_path)
    (home / "profiles" / "remii" / "config.yaml").unlink()
    r = _run_step6(home)
    assert r.returncode == 1
    assert "required profile config missing: remii" in r.stdout


def test_healthy_fleet_fixture_is_silent_rc0(tmp_path):
    home, _root = _make_policy_compliant_home(tmp_path)
    r = _run_step6(home)
    assert r.returncode == 0, (
        f"policy-compliant fixture flagged: rc={r.returncode}\n"
        f"stdout={r.stdout}\nstderr={r.stderr}"
    )
    assert r.stdout == ""


# ── curator safety drift (root) ────────────────────────────────────────────


@pytest.mark.parametrize("surface,new_value", [
    ("enabled", False),
    ("stale_after_days", 30),
    ("archive_after_days", 90),
    ("consolidate", True),
    ("prune_builtins", True),
])
def test_root_curator_safety_drift_detected(tmp_path, surface, new_value):
    home, root = _make_policy_compliant_home(tmp_path)
    root["curator"][surface] = new_value
    _write_yaml(home / "config.yaml", root)
    r = _run_step6(home)
    assert r.returncode == 1
    assert f"curator.{surface}" in r.stdout


def test_root_curator_backup_disabled_detected(tmp_path):
    home, root = _make_policy_compliant_home(tmp_path)
    root["curator"]["backup"] = {"enabled": False}
    _write_yaml(home / "config.yaml", root)
    r = _run_step6(home)
    assert r.returncode == 1
    assert "curator.backup.enabled" in r.stdout


# ── curator safety drift (profiles) ────────────────────────────────────────


@pytest.mark.parametrize("profile", sorted(_FULL_POLICY_PROFILES))
def test_profile_curator_stale_after_days_drift(tmp_path, profile):
    home, _root = _make_policy_compliant_home(tmp_path)
    cfg = home / "profiles" / profile / "config.yaml"
    data = _yaml6.safe_load(cfg.read_text())
    data["curator"]["stale_after_days"] = 30
    _write_yaml(cfg, data)
    r = _run_step6(home)
    assert r.returncode == 1, f"{profile} curator drift not caught"
    assert f"profiles/{profile}" in r.stdout
    assert "stale_after_days" in r.stdout


def test_mrhermagi_curator_enabled_false_required(tmp_path):
    home, _root = _make_policy_compliant_home(tmp_path)
    cfg = home / "profiles" / "mrhermagi" / "config.yaml"
    data = _yaml6.safe_load(cfg.read_text())
    data["curator"] = {"enabled": True, "stale_after_days": 45,
                       "archive_after_days": 120, "consolidate": False,
                       "prune_builtins": False, "backup": {"enabled": True}}
    _write_yaml(cfg, data)
    r = _run_step6(home)
    assert r.returncode == 1, "mrhermagi curator.enabled=true not flagged"
    assert "profiles/mrhermagi" in r.stdout


# ── budgets ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path,value", [
    (("agent", "max_turns"), 90),
    (("delegation", "max_iterations"), 90),
    (("delegation", "max_concurrent_children"), 3),
])
def test_root_budgets_are_enforced(tmp_path, path, value):
    home, root = _make_policy_compliant_home(tmp_path)
    root.setdefault(path[0], {})[path[1]] = value
    _write_yaml(home / "config.yaml", root)
    r = _run_step6(home)
    assert r.returncode == 1
    assert ".".join(path) in r.stdout


@pytest.mark.parametrize("profile", sorted(_BUDGET_PROFILE_MAX_TURNS_NONE))
def test_budget_max_turns_must_be_none(tmp_path, profile):
    home, _root = _make_policy_compliant_home(tmp_path)
    cfg = home / "profiles" / profile / "config.yaml"
    data = _yaml6.safe_load(cfg.read_text())
    data.setdefault("agent", {})["max_turns"] = 30
    _write_yaml(cfg, data)
    r = _run_step6(home)
    assert r.returncode == 1, f"{profile} max_turns-infinity revoked, not caught"
    assert "agent.max_turns" in r.stdout


@pytest.mark.parametrize("keys", [
    ("max_iterations", 30), ("max_concurrent_children", 3),
])
@pytest.mark.parametrize("profile", sorted(_BUDGET_PROFILE_DELEGATION))
def test_budget_delegation_caps(tmp_path, profile, keys):
    key, val = keys
    home, _root = _make_policy_compliant_home(tmp_path)
    cfg = home / "profiles" / profile / "config.yaml"
    data = _yaml6.safe_load(cfg.read_text())
    data.setdefault("delegation", {})[key] = val
    _write_yaml(cfg, data)
    r = _run_step6(home)
    assert r.returncode == 1, f"{profile}.{key} cap revoked, not caught"
    assert f"delegation.{key}" in r.stdout


def test_orchestrator_max_turns_must_be_none(tmp_path):
    """orchestrator is budget-only (not in the curator-scope list)."""
    home, _root = _make_policy_compliant_home(tmp_path)
    cfg = home / "profiles" / "orchestrator" / "config.yaml"
    data = _yaml6.safe_load(cfg.read_text())
    data["agent"] = {"max_turns": 5}
    _write_yaml(cfg, data)
    r = _run_step6(home)
    assert r.returncode == 1
    assert "profiles/orchestrator" in r.stdout


# ── display.personality + SOUL.md presence ─────────────────────────────────


@pytest.mark.parametrize("personality", ["minimal", "kawaii"])
def test_personality_drift_detected(tmp_path, personality):
    """Flip every personality away from its approved value; one at a time."""
    for name, approved in _PERSONALITY.items():
        if approved == personality:
            continue
        home, _root = _make_policy_compliant_home(tmp_path)
        cfg = home / "profiles" / name / "config.yaml"
        data = _yaml6.safe_load(cfg.read_text())
        data.setdefault("display", {})["personality"] = (
            "kawaii" if approved == "minimal" else "minimal")
        _write_yaml(cfg, data)
        r = _run_step6(home)
        assert r.returncode == 1, f"{name} personality drift not caught"
        assert "display.personality" in r.stdout


def test_personality_missing_detected(tmp_path):
    home, _root = _make_policy_compliant_home(tmp_path)
    cfg = home / "profiles" / "wesker" / "config.yaml"
    data = _yaml6.safe_load(cfg.read_text())
    data.setdefault("display", {})["personality"] = None
    _write_yaml(cfg, data)
    r = _run_step6(home)
    assert r.returncode == 1
    assert "display.personality" in r.stdout


@pytest.mark.parametrize("name", sorted(_PERSONALITY))
def test_soul_md_missing_detected(tmp_path, name):
    home, _root = _make_policy_compliant_home(tmp_path)
    (home / "profiles" / name / "SOUL.md").unlink()
    r = _run_step6(home)
    assert r.returncode == 1, f"{name} missing SOUL.md not caught"
    assert "SOUL.md" in r.stdout


def test_soul_md_empty_detected(tmp_path):
    home, _root = _make_policy_compliant_home(tmp_path)
    (home / "profiles" / "octacon" / "SOUL.md").write_text("")
    r = _run_step6(home)
    assert r.returncode == 1
    assert "SOUL.md" in r.stdout


# ── root platform_toolsets.cli ─────────────────────────────────────────────


@pytest.mark.parametrize("removed", ["guard", "session_search", "cronjob"])
def test_root_cli_toolset_missing_required_tool(tmp_path, removed):
    home, root = _make_policy_compliant_home(tmp_path)
    root["platform_toolsets"]["cli"] = [
        t for t in root["platform_toolsets"]["cli"] if t != removed]
    _write_yaml(home / "config.yaml", root)
    r = _run_step6(home)
    assert r.returncode == 1
    assert removed in r.stdout


# ── provider/fallback/auxiliary exclusion (fail-safe hardening) ────────────


@pytest.mark.parametrize("mutation", [
    {"model": {"default": "totally-different/model"}},
    {"providers": {"openrouter": {"api_key": "sk-another-fake"}},
     "custom_providers": {"x": {"base_url": "https://evil.example"}},
     "fallback_providers": [{"provider": "together", "model": "y"},
                            {"provider": "deepseek", "model": "z"}],
     "fallback_reasoning": {"model.default": "high"},
     "auxiliary": {"curator": {"provider": "deepseek",
                               "model": "deepseek-chat"}},
    },
])
def test_provider_surface_mutations_stay_healthy(tmp_path, mutation):
    """Arbitrary changes under provider-ish surfaces must NEVER produce drift."""
    home, root = _make_policy_compliant_home(tmp_path)
    root.update(copy.deepcopy(mutation))
    _write_yaml(home / "config.yaml", root)
    r = _run_step6(home)
    assert r.returncode == 0, (
        f"provider/fallback/aux mutation detected as drift: {r.stdout!r}"
    )
    assert r.stdout == ""


@pytest.mark.parametrize("mutation", [
    {"model": {"default": "other/model"}},
    {"auxiliary": {"curator": {"provider": "groq"}}},
    {"fallback_providers": []},
])
def test_profile_provider_mutations_stay_healthy(tmp_path, mutation):
    home, _root = _make_policy_compliant_home(tmp_path)
    for name in ("remii", "wesker", "octacon", "gojo"):
        cfg = home / "profiles" / name / "config.yaml"
        data = _yaml6.safe_load(cfg.read_text())
        data.update(copy.deepcopy(mutation))
        _write_yaml(cfg, data)
    r = _run_step6(home)
    assert r.returncode == 0, (
        f"profile provider mutation detected as drift: {r.stdout!r}"
    )


def test_secret_values_never_appear_in_output(tmp_path):
    """A deliberately secret-looking value sitting on an unchecked surface
    must not leak into drift output even when real drift exists elsewhere."""
    home, root = _make_policy_compliant_home(tmp_path)
    root["providers"] = {"openrouter": {"api_key": "sk-SUPER-SECRET-1234"}}
    root["auxiliary"] = {"curator": {
        "api_key": "sk-AUX-SECRET-9876",
        "provider": "openrouter",
        "model": "gpt-9",
        "base_url": "https://internal.example/v1",
    }}
    root["curator"]["stale_after_days"] = 999  # force real drift
    _write_yaml(home / "config.yaml", root)
    r = _run_step6(home)
    assert r.returncode == 1
    for secret in ("sk-SUPER-SECRET-1234", "sk-AUX-SECRET-9876",
                   "internal.example", "gpt-9"):
        assert secret not in r.stdout and secret not in r.stderr, (
            f"{secret} leaked into drift output"
        )
    assert "curator.stale_after_days" in r.stdout


def test_drift_output_has_no_timestamps(tmp_path):
    home, root = _make_policy_compliant_home(tmp_path)
    root["curator"]["consolidate"] = True
    cfg = home / "profiles" / "remii" / "config.yaml"
    data = _yaml6.safe_load(cfg.read_text())
    data["curator"]["archive_after_days"] = 90
    _write_yaml(home / "config.yaml", root)
    _write_yaml(cfg, data)
    r = _run_step6(home)
    assert r.returncode == 1
    import re as _re
    assert not _re.search(r"\d{4}-\d{2}-\d{2}", r.stdout), r.stdout


def test_malformed_profile_config_is_execution_error_rc2(tmp_path):
    """A profile config that cannot be parsed is an *execute* failure, not
    silent drift — rc 2 with the [DRIFT] Check failed: banner."""
    home, _root = _make_policy_compliant_home(tmp_path)
    (home / "profiles" / "gojo" / "config.yaml").write_text(
        "curator: {enabled: [unterminated\n")
    r = _run_step6(home)
    assert r.returncode == 2, r.stdout
    assert "[DRIFT] Check failed" in r.stdout


def test_allowed_cli_extra_tools_do_not_drift(tmp_path):
    """platform_toolsets.cli may contain MORE than the required nine; the
    checker pins the required set, not exact equality (fleet adds tts, web,
    vision, etc. legitimately)."""
    home, root = _make_policy_compliant_home(tmp_path)
    root["platform_toolsets"]["cli"].append("web_search_extra")
    _write_yaml(home / "config.yaml", root)
    r = _run_step6(home)
    assert r.returncode == 0, r.stdout


# ── standby utility profile skip (Sahil-approved 2026-09-01) ──────────────


def test_standby_profile_model_guard_and_schema_skipped(tmp_path):
    """A standby utility profile (kind=utility, lifecycle=standby, no
    gateway, no model routing by design) must NOT produce drift when its
    config lacks a schema version or a model-routing baseline entry."""
    home, _root = _make_policy_compliant_home(tmp_path)
    prof_dir = home / "profiles" / "work"
    prof_dir.mkdir(parents=True, exist_ok=True)
    # No _config_version and no baseline entry: both would be drift for a
    # routed profile. For a standby profile this must stay silent.
    _write_yaml(prof_dir / "config.yaml", {
        "model": {"default": "utility/no-routing"},
        "skills": {"enabled_skills": []},
    })
    r = _run_step6(home)
    assert r.returncode == 0, r.stdout
    assert r.stdout == ""


def test_non_standby_profile_without_baseline_is_still_drift(tmp_path):
    """The standby skip must be an explicit allow-list: a NON-standby
    profile absent from the approved baseline is still drift."""
    home, _root = _make_policy_compliant_home(tmp_path)
    # Build a minimal approved baseline so the guard is active (not fail-open)
    import json as _json
    gov = home / "governance"
    gov.mkdir(parents=True, exist_ok=True)
    (gov / "model-routing-baseline.json").write_text(_json.dumps({"sig": {}}))
    prof_dir = home / "profiles" / "routedextra"
    prof_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(prof_dir / "config.yaml", {
        "_config_version": SCHEMA_VERSION,
        "model": {"default": "evil/extra-model"},
        "skills": {"enabled_skills": []},
    })
    r = _run_step6(home)
    assert r.returncode == 1, r.stdout
    assert "routedextra" in r.stdout
