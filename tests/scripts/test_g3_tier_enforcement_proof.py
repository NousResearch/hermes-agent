#!/usr/bin/env python3
"""G3 regression tests for scripts/tier-enforcement-proof.py.

The proof script must DERIVE profile tiers from on-disk config.yaml files
and validate Tier-3 identity/count against the governing registry — not
from a hard-coded source list that can go stale (the G3 defect: the script
hard-coded the 9 real Tier-3 profiles as Tier-2 and falsely reported
Tier-3=0).

These tests use temporary fixture roots (profiles dir + registry fixture)
injected via the script's function parameters. No test reads or mutates
the real ~/.hermes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "tier-enforcement-proof.py"


# ── helpers ─────────────────────────────────────────────────────────────────


def _load_script():
    """Import the proof script module.

    Paths are injected via function parameters (profiles_dir /
    registry_path) on each call — no environment-variable seam.
    """
    spec = importlib.util.spec_from_file_location(
        "tier_enforcement_proof_under_test", str(SCRIPT)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_profile(profiles_root: Path, name: str, tier: int) -> None:
    """Create a minimal profile directory + config.yaml with a tier field."""
    d = profiles_root / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"tier": tier, "name": name, "model": {"default": "test/m"}}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))


def _make_registry(
    tmp_path: Path,
    tier1: list[str],
    tier2: list[str],
    tier3: list[str],
) -> Path:
    """Write a minimal registry markdown fixture the script can parse.

    Format mirrors the real registry: markdown tables under tier headers.
    The parser extracts profile names from the first column of each tier
    section.
    """
    lines = ["# Profile Tier Registry", ""]

    lines.append(f"## Tier 1 — Active Gateway Leads ({len(tier1)})")
    lines.append("")
    lines.append("| Profile | Lead domain | Systemd unit |")
    lines.append("|---|---|---|")
    for name in tier1:
        lines.append(f"| {name} | domain | unit |")
    lines.append("")

    lines.append(f"## Tier 2 — Standing Delegation Targets ({len(tier2)})")
    lines.append("")
    lines.append("| Profile | Sub |")
    lines.append("|---|---|")
    for name in tier2:
        lines.append(f"| {name} | sub |")
    lines.append("")

    lines.append(f"## Tier 3 — Dormant / Specialized ({len(tier3)})")
    lines.append("")
    lines.append("| Profile | Reason |")
    lines.append("|---|---|")
    for name in tier3:
        lines.append(f"| {name} | G3: keep disabled |")
    lines.append("")

    reg = tmp_path / "registry.md"
    reg.write_text("\n".join(lines))
    return reg


# ── fixtures ────────────────────────────────────────────────────────────────


# Deliberately arbitrary fixture profiles. These names and counts must never
# encode or act as a change detector for the deployed fleet.
T1_NAMES = ["alpha", "beta"]
T2_NAMES = ["gamma", "delta"]
T3_NAMES = ["epsilon"]


@pytest.fixture
def full_fixture(tmp_path):
    """Build an arbitrary small fixture with all three tier classes."""
    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir()

    for n in T1_NAMES:
        _make_profile(profiles_root, n, 1)

    for n in T2_NAMES:
        _make_profile(profiles_root, n, 2)

    for n in T3_NAMES:
        _make_profile(profiles_root, n, 3)

    reg = _make_registry(tmp_path, T1_NAMES, T2_NAMES, T3_NAMES)
    mod = _load_script()
    return mod, profiles_root, reg


# ── tests ───────────────────────────────────────────────────────────────────


class TestDerivesTiersFromConfig:
    """G3 core: tiers must come from config.yaml, not a hard-coded dict."""

    def test_tier3_detected_from_config(self, full_fixture):
        """9 Tier-3 profiles with tier:3 in config.yaml must be detected."""
        mod, profiles_root, _ = full_fixture
        tiers = mod.derive_profile_tiers(profiles_root)
        t3 = {n for n, t in tiers.items() if t == 3}
        assert t3 == set(T3_NAMES)

    def test_tier3_count_matches_fixture(self, full_fixture):
        """Tier-3 count derives from the fixture, never a deployed count."""
        mod, profiles_root, _ = full_fixture
        tiers = mod.derive_profile_tiers(profiles_root)
        t3_count = sum(1 for t in tiers.values() if t == 3)
        assert t3_count == len(T3_NAMES)

    def test_hardcoded_zero_tier3_cannot_occur(self, full_fixture):
        """The bug: a hard-coded roster reporting Tier-3=0 must never happen.

        With fixture profiles having tier:3 in config, derive_profile_tiers() must
        return non-zero Tier-3 count. If the script still used a stale
        CANONICAL dict with no Tier-3 entries, this would fail.
        """
        mod, profiles_root, _ = full_fixture
        tiers = mod.derive_profile_tiers(profiles_root)
        assert any(t == 3 for t in tiers.values()), (
            "Tier-3=0 is the G3 defect — tiers must be derived from config"
        )

    def test_tier1_count(self, full_fixture):
        """Fixture Tier-1 identities derive from the temporary tree."""
        mod, profiles_root, _ = full_fixture
        tiers = mod.derive_profile_tiers(profiles_root)
        t1 = {n for n, t in tiers.items() if t == 1}
        assert t1 == set(T1_NAMES)

    def test_tier2_count(self, full_fixture):
        """All fixture Tier-2 profiles detected."""
        mod, profiles_root, _ = full_fixture
        tiers = mod.derive_profile_tiers(profiles_root)
        t2 = {n for n, t in tiers.items() if t == 2}
        assert t2 == set(T2_NAMES)


class TestRegistryValidation:
    """Tier-3 identity must be validated against the governing registry."""

    def test_matching_registry_passes(self, full_fixture):
        """When config tiers match registry tiers exactly, proof passes."""
        mod, profiles_root, reg = full_fixture
        result = mod.run_proof(profiles_root, reg)
        assert result.exit_code == 0
        assert f"Tier 3: {len(T3_NAMES)}" in result.stdout

    def test_registry_mismatch_fails(self, tmp_path):
        """Registry lists a Tier-3 profile not in config → clear failure."""
        profiles_root = tmp_path / "profiles"
        profiles_root.mkdir()
        for n in T1_NAMES:
            if n != "kensei":
                _make_profile(profiles_root, n, 1)
            else:
                (profiles_root / n).mkdir(exist_ok=True)
        for n in T2_NAMES:
            _make_profile(profiles_root, n, 2)
        for n in T3_NAMES:
            _make_profile(profiles_root, n, 3)

        # Registry claims an extra Tier-3 not in config
        bad_t3 = T3_NAMES + ["nonexistent-profile"]
        reg = _make_registry(tmp_path, T1_NAMES, T2_NAMES, bad_t3)
        mod = _load_script()

        result = mod.run_proof(profiles_root, reg)
        assert result.exit_code != 0
        assert "mismatch" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_config_tier3_not_in_registry_fails(self, tmp_path):
        """Config has a Tier-3 profile not in registry → clear failure."""
        profiles_root = tmp_path / "profiles"
        profiles_root.mkdir()
        for n in T1_NAMES:
            if n != "kensei":
                _make_profile(profiles_root, n, 1)
            else:
                (profiles_root / n).mkdir(exist_ok=True)
        for n in T2_NAMES:
            _make_profile(profiles_root, n, 2)
        for n in T3_NAMES:
            _make_profile(profiles_root, n, 3)
        # Add a profile with tier:3 in config but NOT in registry
        _make_profile(profiles_root, "rogue-tier3", 3)

        reg = _make_registry(tmp_path, T1_NAMES, T2_NAMES, T3_NAMES)
        mod = _load_script()

        result = mod.run_proof(profiles_root, reg)
        assert result.exit_code != 0

    def test_tier3_identity_matches_registry(self, full_fixture):
        """The set of Tier-3 profiles from config must exactly equal the
        Tier-3 set in the registry."""
        mod, profiles_root, reg = full_fixture
        config_tiers = mod.derive_profile_tiers(profiles_root)
        config_t3 = {n for n, t in config_tiers.items() if t == 3}
        registry_t3 = mod.parse_registry_t3(reg)
        assert config_t3 == registry_t3


class TestSafetyPolicyPreserved:
    """Tier-3 must remain non-delegatable; script only proves/enforces."""

    def test_tier3_marked_blocked(self, full_fixture):
        """Tier-3 profiles must be reported as BLOCKED in output."""
        mod, profiles_root, reg = full_fixture
        result = mod.run_proof(profiles_root, reg)
        assert result.exit_code == 0
        # Each Tier-3 profile should be listed as BLOCKED
        for name in T3_NAMES:
            assert name in result.stdout
        assert "BLOCKED" in result.stdout

    def test_script_does_not_modify_profiles(self, full_fixture):
        """Script must not create/activate/modify any profile config."""
        mod, profiles_root, reg = full_fixture
        # Snapshot all config.yaml contents
        snapshots = {}
        for cfg in profiles_root.rglob("config.yaml"):
            snapshots[cfg] = cfg.read_text()

        mod.run_proof(profiles_root, reg)

        for cfg, original in snapshots.items():
            assert cfg.read_text() == original, f"{cfg} was modified"


class TestFunctionParameterInjection:
    """Script must accept temporary roots via function parameters.

    Direct parameter fixture coverage for temporary profile-root and
    registry inputs — no environment-variable seam used.
    """

    def test_profiles_dir_param_injection(self, tmp_path):
        """derive_profile_tiers accepts profiles_dir parameter."""
        profiles_root = tmp_path / "profiles"
        profiles_root.mkdir()
        _make_profile(profiles_root, "test-t1", 1)
        mod = _load_script()
        tiers = mod.derive_profile_tiers(profiles_root)
        assert "test-t1" in tiers
        assert tiers["test-t1"] == 1

    def test_registry_path_param_injection(self, tmp_path):
        """parse_registry accepts registry_path parameter."""
        reg = _make_registry(tmp_path, ["test-t1"], ["test-t2"], ["test-t3"])
        mod = _load_script()
        reg_tiers = mod.parse_registry(reg)
        assert "test-t1" in reg_tiers["1"]
        assert "test-t2" in reg_tiers["2"]
        assert "test-t3" in reg_tiers["3"]

    def test_run_proof_param_injection(self, tmp_path):
        """run_proof accepts profiles_dir and registry_path parameters."""
        profiles_root = tmp_path / "profiles"
        profiles_root.mkdir()
        _make_profile(profiles_root, "test-t1", 1)
        reg = _make_registry(tmp_path, ["test-t1"], [], [])
        mod = _load_script()
        result = mod.run_proof(profiles_root, reg)
        assert result.exit_code == 0

    def test_default_paths_are_real_deployed_paths(self):
        """Module-level PROFILES_DIR and REGISTRY point at real ~/.hermes."""
        import os
        mod = _load_script()
        assert str(mod.PROFILES_DIR) == os.path.expanduser("~/.hermes/profiles")
        assert str(mod.REGISTRY) == os.path.expanduser("~/.hermes/governance/profile-tier-registry.md")

    def test_empty_profiles_dir_param(self, tmp_path):
        """derive_profile_tiers with empty dir returns empty dict."""
        empty = tmp_path / "empty"
        empty.mkdir()
        mod = _load_script()
        assert mod.derive_profile_tiers(empty) == {}

    def test_nonexistent_profiles_dir_param(self, tmp_path):
        """derive_profile_tiers with missing dir returns empty dict."""
        mod = _load_script()
        assert mod.derive_profile_tiers(tmp_path / "nonexistent") == {}

    def test_nonexistent_registry_param(self, tmp_path):
        """parse_registry with missing file returns empty tiers."""
        mod = _load_script()
        result = mod.parse_registry(tmp_path / "nonexistent.md")
        assert result == {"1": set(), "2": set(), "3": set()}


# ── Live-evidence separation ────────────────────────────────────────────────
#
# The REAL deployed fleet proof (T1=14, T2=39, T3=9) is a read-only EVIDENCE
# command, NOT a unit-test assertion:
#
#     python scripts/tier-enforcement-proof.py
#
# It must never be asserted inside pytest — the fleet count drifts and the
# counts are environment-specific. A prior P08 agent added
# `TestLiveRegistryCountDrift` that hard-coded 39/9 and read live paths; it was
# rejected as an invalid change-detector. The behavioural coverage below is the
# permanent replacement: it uses ARBITRARY SMALL fixture tiers (e.g. 2/2/1) and
# proves the parser + report detect registry-vs-disk mismatch and pass when the
# sets match. No live ~/, no fixed production fleet count, no T1=14/T2=39/T3=9.


# Arbitrary small fixture tiers — deliberately NOT the real fleet sizes.
_SMALL_T1 = ["alpha", "beta"]
_SMALL_T2 = ["gamma", "delta"]
_SMALL_T3 = ["epsilon"]


@pytest.fixture
def small_fixture(tmp_path):
    """Minimal 2/2/1 fixture with disk + registry in agreement."""
    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir()
    for n in _SMALL_T1:
        _make_profile(profiles_root, n, 1)
    for n in _SMALL_T2:
        _make_profile(profiles_root, n, 2)
    for n in _SMALL_T3:
        _make_profile(profiles_root, n, 3)
    reg = _make_registry(tmp_path, _SMALL_T1, _SMALL_T2, _SMALL_T3)
    mod = _load_script()
    return mod, profiles_root, reg


class TestTierConsistencyFixtureMismatch:
    """Fixture-driven replacement for the rejected live-count drift test.

    Proves the parser + report detect registry-vs-disk mismatch under
    arbitrary small fixtures, and pass when the sets match. Never asserts the
    real fleet sizes (14/39/9).
    """

    def test_small_fixture_matching_sets_pass(self, small_fixture):
        """Disk tiers equal registry tiers → proof passes (exit 0)."""
        mod, profiles_root, reg = small_fixture
        result = mod.run_proof(profiles_root, reg)
        assert result.exit_code == 0
        # Reports the ARBITRARY small fixture counts, not live 14/39/9.
        assert "Tier 1: 2" in result.stdout
        assert "Tier 2: 2" in result.stdout
        assert "Tier 3: 1" in result.stdout
        assert "TIER ENFORCEMENT PROVEN" in result.stdout

    def test_parser_returns_exact_small_counts(self, tmp_path):
        """Parser returns ONLY what the fixture file contains — no live leak."""
        reg = _make_registry(tmp_path, _SMALL_T1, _SMALL_T2, _SMALL_T3)
        mod = _load_script()
        parsed = mod.parse_registry(reg)
        assert parsed["1"] == set(_SMALL_T1)
        assert parsed["2"] == set(_SMALL_T2)
        assert parsed["3"] == set(_SMALL_T3)

    def test_registry_extra_profile_detected(self, tmp_path):
        """Registry lists a Tier-2 profile absent on disk → mismatch error."""
        profiles_root = tmp_path / "profiles"
        profiles_root.mkdir()
        for n in _SMALL_T1:
            _make_profile(profiles_root, n, 1)
        for n in _SMALL_T2:
            _make_profile(profiles_root, n, 2)
        for n in _SMALL_T3:
            _make_profile(profiles_root, n, 3)

        # Registry claims an extra Tier-2 not present on disk.
        bad_t2 = _SMALL_T2 + ["zeta-unregistered"]
        reg = _make_registry(tmp_path, _SMALL_T1, bad_t2, _SMALL_T3)
        mod = _load_script()

        result = mod.run_proof(profiles_root, reg)
        assert result.exit_code != 0
        assert "mismatch" in result.stdout.lower()
        assert "zeta-unregistered" in result.stdout

    def test_disk_extra_profile_detected(self, tmp_path):
        """Disk has a Tier-2 profile absent from registry → mismatch error."""
        profiles_root = tmp_path / "profiles"
        profiles_root.mkdir()
        for n in _SMALL_T1:
            _make_profile(profiles_root, n, 1)
        for n in _SMALL_T2:
            _make_profile(profiles_root, n, 2)
        for n in _SMALL_T3:
            _make_profile(profiles_root, n, 3)
        # Extra profile on disk with tier:2 but not in registry.
        _make_profile(profiles_root, "omega-orphan", 2)

        reg = _make_registry(tmp_path, _SMALL_T1, _SMALL_T2, _SMALL_T3)
        mod = _load_script()

        result = mod.run_proof(profiles_root, reg)
        assert result.exit_code != 0
        assert "mismatch" in result.stdout.lower()
        assert "omega-orphan" in result.stdout
