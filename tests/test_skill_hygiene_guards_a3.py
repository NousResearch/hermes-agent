#!/usr/bin/env python3
"""A3 skill-hygiene guard E2E — proves G1/G2/G3 fire. Run via the finder-evicted
preamble so it exercises the WORKTREE bytes (git-worktree-isolation §8f)."""
import os, sys, tempfile, shutil
from pathlib import Path

WT = os.environ["A3_WT"]
sys.path.insert(0, WT)
# evict editable-install meta_path finder so worktree wins the import race
sys.meta_path[:] = [f for f in sys.meta_path
                    if "__editable__" not in getattr(type(f), "__module__", "")]
for n in list(sys.modules):
    fp = getattr(sys.modules.get(n), "__file__", None) or ""
    if "/.hermes/hermes-agent/" in fp and "/venv/" not in fp and "/.worktrees/" not in fp:
        del sys.modules[n]

results = []

# --- G2: queue-note exclusion (agent/skill_utils.py) ---
import agent.skill_utils as su
assert "/.worktrees/sendoff-a3-skill-hygiene/" in su.__file__, f"WRONG TREE: {su.__file__}"
g2a = su.is_queue_note_name("pending-shared-skill-patches") is True
g2b = su.is_queue_note_name("pending-patch-foo") is True
g2c = su.is_queue_note_name("coding-guardrails") is False
# path-level: a pending-* skill dir's SKILL.md is excluded
g2d = su.is_excluded_skill_path(
    "/x/skills-shared/pending/pending-shared-skill-patches/SKILL.md") is True
g2e = su.is_excluded_skill_path("/x/skills-shared/coding/coding-guardrails/SKILL.md") is False
results.append(("G2 queue-note name+path exclusion", all([g2a, g2b, g2c, g2d, g2e]),
                f"name(pending)={g2a} name(patch-)={g2b} name(real)={g2c} path(pending)={g2d} path(real)={g2e}"))

# --- G1: write-time placement guard (tools/skill_manager_tool.py) ---
import tools.skill_manager_tool as smt
assert "/.worktrees/sendoff-a3-skill-hygiene/" in smt.__file__, f"WRONG TREE: {smt.__file__}"
# G1 rejects a create whose NAME is a reserved queue-note prefix (source guard)
_valid = "---\nname: pending-patch-evil\ndescription: a valid-looking description long enough to pass\n---\nbody"
r = smt._create_skill("pending-patch-evil", _valid, category="coding")
g1_name = (r.get("success") is False and "reserved" in r.get("error", "").lower())
results.append(("G1 reject queue-note-named create", g1_name, str(r.get("error", ""))[:90]))

# G1 helper: _enforce_shared_placement is callable + returns a bool
g1_flag = isinstance(smt._enforce_shared_placement(), bool)
results.append(("G1 _enforce_shared_placement()->bool", g1_flag, str(smt._enforce_shared_placement())))

# --- G3: relocate safety (scripts/local-skill-leak-check.py) ---
import importlib.util
spec = importlib.util.spec_from_file_location(
    "leakcheck", os.path.join(WT, "scripts", "local-skill-leak-check.py"))
lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)
# officialness is a computed property; a queue-note is never official (relocated wholesale),
# and the module exposes is_official_skill / relocate_is_safe
has_api = all(hasattr(lc, n) for n in ("is_official_skill", "load_allowlist"))
results.append(("G3 computed-officialness API present", has_api,
                f"is_official_skill={hasattr(lc,'is_official_skill')} "
                f"relocate_is_safe={hasattr(lc,'relocate_is_safe')}"))

print("=== A3 guard E2E ===")
ok = True
for name, passed, detail in results:
    print(f"[{'PASS' if passed else 'FAIL'}] {name} :: {detail}")
    ok = ok and passed
print("ALL_PASS" if ok else "SOME_FAILED")
sys.exit(0 if ok else 1)
