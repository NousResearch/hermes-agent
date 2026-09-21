#!/usr/bin/env python3
"""Skill-drift guard — assert the shared-library model stays intact.

Fails (and prints findings) if any profile re-accumulates bundled skills that
duplicate the shared base library, or if a profile lost its external_dirs link.
Designed to be run by `denji-skill-audit` (or standalone) so the 2,462-copy
duplication that was eliminated on 2026-06-03 cannot silently return.

Exit 0 = clean. Exit 1 = drift found (stdout lists it, Evidence-Gate style).
"""
from __future__ import annotations
import sys
from pathlib import Path

BASE = Path('/home/kensei/.hermes/skills')
PROF = Path('/home/kensei/.hermes/profiles')
BASE_STR = str(BASE)
EXCLUDE = ('.hub', '.restore-backups', '_archived')


def base_names() -> set[str]:
    return {md.parent.name for md in BASE.rglob('SKILL.md')
            if not any(x in md.parts for x in EXCLUDE)}


def main() -> int:
    import yaml
    bn = base_names()
    dup_findings: list[str] = []
    link_findings: list[str] = []

    for pdir in sorted(PROF.iterdir()):
        sk = pdir / 'skills'
        cfg = pdir / 'config.yaml'
        # Only profiles that have a local skills dir are in scope.
        if not sk.exists():
            continue
        # local skills that duplicate base = drift
        local = {md.parent.name for md in sk.rglob('SKILL.md')
                 if not any(x in md.parts for x in EXCLUDE)}
        dups = sorted(local & bn)
        if dups:
            dup_findings.append(f"{pdir.name}: {len(dups)} bundled dup(s) -> {', '.join(dups[:8])}{'...' if len(dups) > 8 else ''}")
        # external_dirs must point at base
        ext = []
        if cfg.exists():
            try:
                c = yaml.safe_load(cfg.read_text()) or {}
                ext = ((c.get('skills') or {}).get('external_dirs')) or []
            except Exception as e:  # noqa: BLE001
                link_findings.append(f"{pdir.name}: config unreadable ({e})")
                continue
        if BASE_STR not in [str(x) for x in ext]:
            link_findings.append(f"{pdir.name}: external_dirs missing base link")

    # H1 guard: profiles that declare always_skills but resolve NO
    # external_dirs cannot load those skills — fail as drift.
    for pdir in sorted(PROF.iterdir()):
        if any(x in pdir.parts for x in EXCLUDE):
            continue
        cfg = pdir / 'config.yaml'
        if not cfg.exists():
            continue
        try:
            c = yaml.safe_load(cfg.read_text()) or {}
            skills_block = c.get('skills') or {}
            always = skills_block.get('always_skills') or []
            ext = skills_block.get('external_dirs') or []
        except Exception:
            continue
        if always and not ext:
            link_findings.append(
                f"{pdir.name}: declares always_skills ({', '.join(always[:5])}{'...' if len(always) > 5 else ''}) "
                f"but has no external_dirs — skills will not load"
            )

    if not dup_findings and not link_findings:
        # Zero-signal: contract says stay silent.
        return 0

    print("⚠️ Skill-library drift detected")
    if dup_findings:
        print("\nProfiles re-accumulating bundled skills (should resolve from base):")
        for f in dup_findings:
            print(f"  • {f}")
        print("  Evidence: profile skills/ contains skill names also present in", BASE_STR)
    if link_findings:
        print("\nProfiles not linked to the shared base library:")
        for f in link_findings:
            print(f"  • {f}")
        print("  Evidence: skills.external_dirs in the profile config.yaml has no", BASE_STR, "entry")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
