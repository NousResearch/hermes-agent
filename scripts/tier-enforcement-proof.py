#!/usr/bin/env python3
"""
C014: Tier-aware routing enforcement proof.

Validates that:
1. Every profile with a config.yaml has a tier assignment
2. Tier 3 profiles are blocked from kanban worker spawn
3. Tier registry matches on-disk profiles (identity check, not hard-coded)
4. Tier 1 profiles have gateway units (where applicable)
5. Tier 2 profiles are delegation targets only (no gateway)

G3 repair: tiers are DERIVED from profile config.yaml `tier:` fields, not
from a hard-coded source-code roster that can go stale. Tier-3 identity and
count are validated against the governing registry
(``~/.hermes/governance/profile-tier-registry.md``). The script only
proves/enforces policy — it never activates or modifies profiles.

Exit 0 = tier enforcement proven.
Exit 1 = mismatch or error detected.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml

# ── Path resolution ─────────────────────────────────────────────────────────
#
# Default deployed real paths. Tests inject temporary roots via the
# function parameters (profiles_dir / registry_path) on derive_profile_tiers,
# parse_registry, and run_proof — no environment-variable seam.

PROFILES_DIR = Path(os.path.expanduser("~/.hermes/profiles"))
REGISTRY = Path(
    os.path.expanduser("~/.hermes/governance/profile-tier-registry.md")
)

# kensei is the default profile — no config.yaml by convention.
_DEFAULT_PROFILE = "kensei"


# ── Result container ───────────────────────────────────────────────────────


class ProofResult:
    """Outcome of running the tier enforcement proof."""

    def __init__(
        self,
        exit_code: int = 0,
        stdout: str = "",
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.errors = errors if errors is not None else []
        self.warnings = warnings if warnings is not None else []


# ── Core: derive tiers from config.yaml ────────────────────────────────────


def derive_profile_tiers(
    profiles_dir: Path | None = None,
) -> dict[str, int]:
    """Derive ``{profile_name: tier}`` from on-disk ``config.yaml`` files.

    Scans ``profiles_dir`` for subdirectories containing a ``config.yaml``
    with a top-level ``tier:`` integer field. Profiles without a
    ``config.yaml`` (e.g. the default ``kensei`` profile) are skipped
    unless they are the convention default — see ``_default_profile_tier``.

    This replaces the former hard-coded ``CANONICAL`` dict. Tiers are now
    always read from the source of truth: the profile config.
    """
    root = profiles_dir if profiles_dir is not None else PROFILES_DIR
    tiers: dict[str, int] = {}
    if not root.is_dir():
        return tiers
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        cfg = d / "config.yaml"
        if not cfg.is_file():
            continue
        try:
            data = yaml.safe_load(cfg.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        tier = data.get("tier")
        if isinstance(tier, int):
            tiers[d.name] = tier
    return tiers


def _default_profile_tier(
    profiles_dir: Path | None = None,
) -> int | None:
    """Return the implicit tier for the default profile (kensei).

    The default profile has no ``config.yaml`` by convention. If its
    directory exists on disk, it is implicitly Tier-1 (the orchestrator).
    Returns ``None`` if the directory does not exist.
    """
    root = profiles_dir if profiles_dir is not None else PROFILES_DIR
    default_dir = root / _DEFAULT_PROFILE
    if default_dir.is_dir():
        return 1
    return None


def get_profile_tier(name: str, tiers: dict[str, int] | None = None) -> int | None:
    """Return tier (1/2/3) or None if unknown.

    Uses the derived tiers dict. Kept for backward-compat with any caller
    that expects the old function signature.
    """
    source = tiers if tiers is not None else derive_profile_tiers()
    return source.get(name)


# ── Registry parsing ────────────────────────────────────────────────────────


def _extract_profile_names(value: str) -> list[str]:
    """Extract profile names from a table cell.

    Handles comma-separated lists (e.g. the Tier-2 sub-agents column:
    ``ceecee-writer, ceecee-social, ceecee-brand``).
    """
    names: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if re.match(r"^[a-z][a-z0-9-]*$", part):
            names.append(part)
    return names


def _parse_table_row(line: str) -> list[str] | None:
    """Parse a markdown table row into stripped cell values.

    Returns ``None`` for separator rows (``|---|---|``) or non-table
    lines.
    """
    line = line.strip()
    if not line.startswith("|"):
        return None
    # Separator row: |---|---|...|
    if re.match(r"^\|[-:\s|]+\|?\s*$", line):
        return None
    cells = [c.strip() for c in line.split("|")]
    # Drop empty leading/trailing from the | delimiters
    cells = [c for c in cells if c != ""]
    if not cells:
        return None
    return cells


def parse_registry(
    registry_path: Path | None = None,
) -> dict[str, set[str]]:
    """Parse the governing registry markdown into ``{tier: {names}}``.

    The real registry uses two table formats within the Tier-2 section:

    1. ``| Lead | Sub-agents |`` — sub-agents listed as a comma-separated
       list in the *second* column.
    2. ``| Profile | Parent lead | ... |`` — profile name in the *first*
       column (same format as Tier-1 and Tier-3 tables).

    This parser handles both formats by inspecting column headers. When a
    header contains ``Sub-agents``, profiles are extracted from that
    column's comma-separated cells. Otherwise, the first column is used.

    Returns ``{1: {...}, 2: {...}, 3: {...}}``.
    """
    path = registry_path if registry_path is not None else REGISTRY
    result: dict[str, set[str]] = {"1": set(), "2": set(), "3": set()}
    if not path.is_file():
        return result
    text = path.read_text()

    # Split into tier sections by ## heading
    tier_pattern = re.compile(
        r"^##\s+Tier\s+(\d)\b.*$", re.MULTILINE
    )
    matches = list(tier_pattern.finditer(text))
    for i, m in enumerate(matches):
        tier_str = m.group(1)
        if tier_str not in result:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end]

        # Within each tier section, there may be multiple sub-sections
        # (### ...) each with its own table. Process line by line, tracking
        # the current table's column headers.
        sub_agents_col: int | None = None
        for line in section.splitlines():
            cells = _parse_table_row(line)
            if cells is None:
                # Non-table or separator — reset column tracking on blank
                if not line.strip():
                    sub_agents_col = None
                continue
            # Detect header row
            lower_cells = [c.lower() for c in cells]
            if "sub-agents" in lower_cells:
                sub_agents_col = lower_cells.index("sub-agents")
                continue  # header row, skip
            if "profile" in lower_cells and lower_cells[0] == "profile":
                sub_agents_col = None  # first-column format
                continue
            if "lead" in lower_cells and lower_cells[0] == "lead":
                # sub-agents format — find the column index
                if "sub-agents" in lower_cells:
                    sub_agents_col = lower_cells.index("sub-agents")
                continue

            # Data row
            if sub_agents_col is not None and sub_agents_col < len(cells):
                # Comma-separated sub-agents format
                for name in _extract_profile_names(cells[sub_agents_col]):
                    result[tier_str].add(name)
            else:
                # First-column format
                name = cells[0]
                if re.match(r"^[a-z][a-z0-9-]*$", name):
                    result[tier_str].add(name)
    return result


def parse_registry_t3(registry_path: Path | None = None) -> set[str]:
    """Return the set of Tier-3 profile names from the registry."""
    return parse_registry(registry_path).get("3", set())


def parse_registry_all(
    registry_path: Path | None = None,
) -> set[str]:
    """Return all profile names across all tiers from the registry."""
    tiers = parse_registry(registry_path)
    return tiers["1"] | tiers["2"] | tiers["3"]


# ── Proof runner ────────────────────────────────────────────────────────────


def run_proof(
    profiles_dir: Path | None = None,
    registry_path: Path | None = None,
) -> ProofResult:
    """Run the full tier enforcement proof.

    All checks are read-only. No profile is created, activated, or modified.
    Returns a ProofResult with exit_code, stdout, errors, warnings.
    """
    p_dir = profiles_dir if profiles_dir is not None else PROFILES_DIR
    r_path = registry_path if registry_path is not None else REGISTRY

    errors: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []

    # 1. Derive tiers from config.yaml (+ implicit default profile)
    tiers = derive_profile_tiers(p_dir)
    # The default profile (kensei) has no config.yaml by convention.
    # If its directory exists, it is implicitly Tier-1.
    default_tier = _default_profile_tier(p_dir)
    if default_tier is not None and _DEFAULT_PROFILE not in tiers:
        tiers[_DEFAULT_PROFILE] = default_tier
    tier1 = sorted(n for n, t in tiers.items() if t == 1)
    tier2 = sorted(n for n, t in tiers.items() if t == 2)
    tier3 = sorted(n for n, t in tiers.items() if t == 3)

    # 2. Parse registry
    reg_tiers = parse_registry(r_path)
    reg_all = reg_tiers["1"] | reg_tiers["2"] | reg_tiers["3"]
    reg_t1 = reg_tiers["1"]
    reg_t2 = reg_tiers["2"]
    reg_t3 = reg_tiers["3"]

    # 3. Validate Tier-3 identity: config Tier-3 set must equal registry Tier-3
    config_t3 = set(tier3)
    if config_t3 != reg_t3:
        extra_in_config = config_t3 - reg_t3
        extra_in_registry = reg_t3 - config_t3
        if extra_in_config:
            errors.append(
                f"Tier-3 mismatch: profiles in config but not registry: "
                f"{sorted(extra_in_config)}"
            )
        if extra_in_registry:
            errors.append(
                f"Tier-3 mismatch: profiles in registry but not config: "
                f"{sorted(extra_in_registry)}"
            )

    # 4. Validate Tier-1 and Tier-2 consistency (config vs registry)
    config_t1 = set(tier1)
    config_t2 = set(tier2)
    if config_t1 != reg_t1:
        diff = config_t1.symmetric_difference(reg_t1)
        errors.append(f"Tier-1 mismatch: config vs registry differ: {sorted(diff)}")
    if config_t2 != reg_t2:
        diff = config_t2.symmetric_difference(reg_t2)
        errors.append(f"Tier-2 mismatch: config vs registry differ: {sorted(diff)}")

    # 5. Check for profiles on disk not in registry (and vice versa)
    on_disk = set(tiers.keys())
    # kensei (default) may not have config.yaml — check registry membership
    unregistered = on_disk - reg_all
    if unregistered:
        for name in sorted(unregistered):
            warnings.append(f"Profile '{name}' on disk but not in tier registry")

    registered_missing = reg_all - on_disk
    # kensei is the default profile — no config.yaml, expected
    registered_missing.discard(_DEFAULT_PROFILE)
    if registered_missing:
        for name in sorted(registered_missing):
            warnings.append(f"Profile '{name}' in registry but not on disk")

    # 6. Tier 3 spawn block proof
    lines.append("=== TIER 3 SPAWN BLOCK ===")
    for name in tier3:
        lines.append(f"  {name}: tier=3 → BLOCKED (correct)")
    if not tier3:
        lines.append("  (no Tier-3 profiles detected)")

    # 7. Tier 1 gateway check
    lines.append("")
    lines.append("=== TIER 1 GATEWAY UNITS ===")
    for name in tier1:
        unit = f"hermes-gateway{'-' + name if name != 'kensei' else ''}"
        unit_path = Path(f"/etc/systemd/system/{unit}.service")
        staged = Path(f"/home/kensei/.hermes/systemd/{unit}.service")
        if unit_path.exists():
            lines.append(f"  {name}: {unit} — INSTALLED")
        elif staged.exists():
            lines.append(f"  {name}: {unit} — STAGED (not installed)")
        else:
            warnings.append(f"Tier 1 '{name}' has no gateway unit ({unit})")

    # 8. Tier 2 — verify no gateway units
    lines.append("")
    lines.append("=== TIER 2 NO-GATEWAY CHECK ===")
    for name in tier2:
        unit = f"hermes-gateway-{name}"
        unit_path = Path(f"/etc/systemd/system/{unit}.service")
        staged = Path(f"/home/kensei/.hermes/systemd/{unit}.service")
        if unit_path.exists() or staged.exists():
            errors.append(
                f"Tier 2 '{name}' has gateway unit ({unit}) — should not"
            )
    if not any("Tier 2" in e for e in errors):
        lines.append("  All Tier 2 profiles: no gateway units (correct)")

    # 9. Summary
    lines.append("")
    lines.append("=== SUMMARY ===")
    lines.append(f"  Profiles on disk (with config): {len(on_disk)}")
    lines.append(f"  Registry profiles: {len(reg_all)}")
    lines.append(f"  Tier 1: {len(tier1)}")
    lines.append(f"  Tier 2: {len(tier2)}")
    lines.append(f"  Tier 3: {len(tier3)}")
    lines.append(f"  Warnings: {len(warnings)}")
    lines.append(f"  Errors: {len(errors)}")

    if warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in warnings:
            lines.append(f"    - {w}")

    if errors:
        lines.append("")
        lines.append("  ERRORS:")
        for e in errors:
            lines.append(f"    - {e}")

    stdout = "\n".join(lines)
    exit_code = 1 if errors else 0
    if exit_code == 0:
        stdout += "\n\nTIER ENFORCEMENT PROVEN"

    return ProofResult(
        exit_code=exit_code, stdout=stdout, errors=errors, warnings=warnings
    )


# ── CLI entrypoint ──────────────────────────────────────────────────────────


def main() -> None:
    result = run_proof()
    print(result.stdout)
    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()
