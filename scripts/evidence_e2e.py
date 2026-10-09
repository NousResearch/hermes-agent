"""End-to-end proof of the product claim, through the real tool stack.

The unit tests prove the merge arithmetic. They do NOT prove the thing the
project exists for:

    a skill records whether it worked, and that record SURVIVES later edits.

This exercises the real path — registry dispatch -> write gate -> guarded write
-> disk — against a temp HERMES_HOME, then applies an unrelated body edit and
checks the evidence block is still there and still correct.

Run:
    cd /tmp/rebase-test
    HERMES_HOME=$(mktemp -d) python scripts/evidence_e2e.py

It refuses to run against a real ~/.hermes unless SEL_E2E_ALLOW=1.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Refuse to touch real skills. TMPDIR is redirected under ~/.hermes/cache/scratch
# on this host, so a naive "is HERMES_HOME inside ~/.hermes" test would fire even
# for a throwaway dir. Check the paths that actually hold skills instead.
_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).resolve()
_real_skills = (Path.home() / ".hermes" / "skills").resolve()
_writes_skills = (_home / "skills").resolve()
if (_writes_skills == _real_skills and os.environ.get("SEL_E2E_ALLOW") != "1"):
    sys.exit(f"refusing to write real skills in {_writes_skills}; set SEL_E2E_ALLOW=1 to override")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import skill_manager_tool as smt  # noqa: E402
from tools import write_approval  # noqa: E402

PASSED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(label)
        print(f"  [OK]   {label}")
    else:
        print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))
        raise SystemExit(f"E2E FAILED at: {label}")


def call(**kwargs) -> dict | str:
    """Call skill_manage through the real handler and parse its JSON result.

    Returns the raw value when it is not JSON: a staged approval is the literal
    string "staged", not a JSON object, and parsing it would be a harness bug
    masquerading as a product failure.
    """
    raw = smt.skill_manage(**kwargs)
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main() -> int:
    from agent.skill_utils import get_all_skills_dirs

    skills_root = Path(os.environ["HERMES_HOME"]) / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)

    # The real dispatch entry, not the bare function.
    entry = smt.registry.get_entry("skill_manage")
    check("skill_manage is registered in the skills toolset",
          entry is not None and entry.toolset == "skills")

    skill_dir = skills_root / "e2e-skill"
    skill_dir.mkdir(exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: e2e-skill\n"
        "description: Use when proving evidence survives unrelated edits. Additive counters only.\n"
        "evidence:\n"
        "  success_count: 11\n"
        "  fail_count: 1\n"
        "  steps: []\n"
        "  evolution: []\n"
        "---\n"
        "\n"
        "# E2E Skill\n"
        "\n"
        "ORIGINAL_BODY_MARKER\n",
        encoding="utf-8")

    # 1. An evidence update through the real registered handler.
    result = call(action="patch", name="e2e-skill", evidence_merge={
        "success_count": 1, "fail_count": 0,
        "steps": [{"name": "measure", "ok": 12, "fail": 0}],
        "evolution": [{"from": 0, "to": 1, "date": "2026-09-28", "reason": "e2e proof"}],
    })
    check("evidence_merge succeeds through the registered tool", result.get("success") is True,
          json.dumps(result)[:200])

    text = skill_md.read_text(encoding="utf-8")
    check("counter accumulated 11 -> 12 (not overwritten to 1)", "success_count: 12" in text)
    check("fail_count preserved at 1 (not reset)", "fail_count: 1" in text)
    check("step recorded", "name: measure" in text)
    check("evolution advanced version to 1", "version: 1" in text)

    # 2. THE ACTUAL CLAIM: an unrelated body edit must not destroy the evidence.
    patched = call(action="patch", name="e2e-skill",
                   old_string="ORIGINAL_BODY_MARKER", new_string="EDITED_BODY_MARKER")
    check("unrelated body patch succeeds", patched.get("success") is True,
          json.dumps(patched)[:200])

    after = skill_md.read_text(encoding="utf-8")
    check("body edit took effect", "EDITED_BODY_MARKER" in after)
    check("EVIDENCE SURVIVED the unrelated edit", "success_count: 12" in after)
    check("step tally survived", "name: measure" in after)
    check("version survived", "version: 1" in after)

    # 3. A second evidence run accumulates on top of the first — 11+1+1, not 11+1=reset.
    second = call(action="patch", name="e2e-skill", evidence_merge={"success_count": 1})
    check("second evidence run succeeds", second.get("success") is True)
    check("counter accumulated 12 -> 13 (recency did not win)", "success_count: 13" in
          skill_md.read_text(encoding="utf-8"))

    # 4. The approval gate is not bypassed: a staged candidate must be the merged bytes.
    captured: dict = {}
    orig_gate = smt._run_write_gate  # noqa: SLF001 — restored before each real replay

    class _WA:
        @staticmethod
        def skill_pending_diff(record):
            return write_approval.skill_pending_diff(record)

        @staticmethod
        def skill_gist(*a, **k):
            return "gist"

    def _fake_gate(build):
        captured["payload"], _ = build(_WA)
        return "staged"

    smt._run_write_gate = _fake_gate  # noqa: SLF001 — capture what the reviewer would see
    staged = call(action="patch", name="e2e-skill", evidence_merge={"success_count": 1})
    check("evidence_merge stages for approval (gate not bypassed)", staged == "staged",
          repr(staged)[:200])
    em = captured["payload"]["evidence_merge"]
    check("staged payload carries the merged candidate", "success_count: 14" in em["_candidate_content"])
    check("staged payload binds the source digest for staleness", bool(em.get("_source_digest")))

    # 5. The REAL replay path (what /skills approve calls) must apply the frozen candidate.
    smt._run_write_gate = orig_gate  # restore: the replay bypasses the gate, not replaces it
    ok = smt.apply_skill_pending(captured["payload"])  # noqa: SLF001 — the approve entry point
    check("approved replay writes the frozen candidate", '"success": true' in ok, ok[:160])
    check("replay landed the accumulated count", "success_count: 14" in
          skill_md.read_text(encoding="utf-8"))

    # 6. A replay against a skill that moved underneath is refused, not silently applied.
    #    Order matters: STAGE against the current source first, then move the source, then replay.
    #    Staging after the edit would legitimately re-digest the new content and not be stale.
    fresh: dict = {}

    def _recapture(build):
        fresh["payload"], _ = build(_WA)
        return "staged"

    smt._run_write_gate = _recapture
    call(action="patch", name="e2e-skill", evidence_merge={"success_count": 1})
    smt._run_write_gate = orig_gate
    check("re-staged payload still carries its candidate",
          "_candidate_content" in fresh["payload"]["evidence_merge"])
    # Now the source moves AFTER approval — this is the lost-update we must refuse.
    approved = skill_md.read_text(encoding="utf-8")
    skill_md.write_text(approved.replace("success_count: 14", "success_count: 99"), encoding="utf-8")
    stale_replay = smt.apply_skill_pending(fresh["payload"])  # noqa: SLF001
    check("stale replay is refused", '"success": false' in stale_replay, stale_replay[:200])
    check("refusal names staleness", "stale" in stale_replay.lower())
    check("the newer on-disk count was NOT clobbered",
          "success_count: 99" in skill_md.read_text(encoding="utf-8"))

    # 7. Concurrency: two threads updating the same skill must ADD (11+1+1 = 13), not overwrite.
    import threading

    skill_md.write_text(
        "---\nname: e2e-skill\ndescription: race\nevidence:\n  success_count: 11\n"
        "  fail_count: 0\n  steps: []\n  evolution: []\n---\n\n# Race\n", encoding="utf-8")
    smt._run_write_gate = orig_gate  # gate off => the write path runs
    errs: list = []

    def bump():
        try:
            smt.skill_manage(action="patch", name="e2e-skill", evidence_merge={"success_count": 1})
        except Exception as exc:  # noqa: BLE001
            errs.append(repr(exc))

    ts = [threading.Thread(target=bump) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    check("concurrent writers do not deadlock", not any(t.is_alive() for t in ts))
    check("concurrent writers raise nothing", not errs, str(errs)[:200])
    check("11 + 1 + 1 = 13 — no lost update (real stack)",
          "success_count: 13" in skill_md.read_text(encoding="utf-8"),
          str([l for l in skill_md.read_text(encoding="utf-8").splitlines() if "success_count" in l]))

    print()
    print(f"E2E PASS — {len(PASSED)} checks against the real tool stack")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
