# Extraction w2 adjudication — web_server s3 (w2a witness, 2026-08-05)

Independent w2a run for the SAME shard as `extraction-w2-web-server-s3.md` (w2b).
HEAD `0577116f83`. Verdicts CONVERGED with w2b on all three clusters — the
sibling runs were blind (no shared outputs), so the agreement is a real
double-blind confirmation, not copying.

## Verdicts (identical to w2b)
- **c1** (custom endpoints, 13) → **w1b** `web_routers/providers_custom_endpoints.py`
  (APIRouter + `late()` + include_router at 7107-7109 + legacy re-export at 7110
  with the exact convention comment + `getLogger("hermes_cli.web_server")`).
  w1a's `custom_endpoints_mixin.py` is byte-faithful 13/13 but: top-level module
  doing `from hermes_cli.web_server import app, _apply_main_model_assignment`
  (circular import at module load), `@app.*` decorators, no APIRouter/include_router
  → deviates from agreement module_shape ("APIRouter pattern, NOT class mixins")
  and all 7 in-tree web_routers precedents.
- **c5** (whatsapp onboarding, 20) → **IN-TREE sibling extraction** (commits
  `3ca0d4fbfd` + `0577116f83` → `web_routers/whatsapp_onboarding.py`, 529 lines,
  @router style, mounted at 8056-8058). w1a's `whatsapp_onboarding_mixin.py` is
  byte-faithful to pre-extraction source but redundant/contradicted by HEAD —
  MUST NOT be merged. w1a FULL patch `git apply --check` exit 1 vs HEAD (c5 gone
  in-tree); `git merge-base --is-ancestor f40fbcf HEAD` = false (base not in
  lineage) → stale by lineage, expected interlock not a defect.
- **c6** (telegram onboarding, 12) → **w1b** `web_routers/telegram_onboarding.py`
  (only witness covering c6). 11/12 moved byte-identical; `_telegram_onboarding_request_sync`
  stays defined in web_server (line 8073 in w1b tree) reached via `late()` —
  test_web_server.py:1214 monkeypatches `ws._telegram_onboarding_request_sync`,
  so keeping the def + late() preserves the contract (matches in-tree whatsapp
  precedent of keeping web_server-owned seams late()-reachable). Plan-listed
  method kept home = documented deviation, functionally safe.

## Technique: decorator-strip byte-fidelity (this run's contribution)
Both witnesses' c1 bodies are byte-identical to live EXCEPT the decorator line
(w1a kept `@app.`, w1b rewrote to `@router.`). Instead of per-method exceptions,
compare with decorator lines stripped from BOTH sides:

```python
def strip_decorators(seg: str) -> str:
    lines = seg.split("\n")
    i = 0
    while i < len(lines) and (lines[i].lstrip().startswith("@") or lines[i].strip() == ""):
        i += 1
    return "\n".join(lines[i:]).rstrip()
# then: strip_decorators(live_seg) == strip_decorators(module_seg)
```

Result: "13/13 identical (body)" verified in one pass for each witness.
Verify route PATHS separately (decorator line content must be identical modulo
the `app`→`router` target): grep `@app.`/`@router.` lines.

## Verification evidence (all reproduced this run)
- `git apply --check` vs CURRENT HEAD in fresh worktrees: w1b exit 0 (all 4
  files); w1a-c1 exit 0 BUT destructive (see below); w1a-full exit 1 (stale).
- Fresh-worktree pytest `--basetemp` + `-p no:cacheprovider` (repo venv):
  w1b new tests `tests/test_web_server_s3_w1b.py` 12/12 + existing whatsapp
  tests 2/2 = 14 passed (manifest claimed 16 — count `def test_` yourself:
  actual 12); existing `test_web_server.py -k 'custom_endpoint or telegram_onboarding
  or parse_model_ids or providers'` = 8 passed, 1 failed.
  w1a c1 tests `tests/hermes_cli/test_custom_endpoints_mixin.py` 12/12.
- **Pre-existing failures proven on pristine HEAD**: the 1 failing existing test
  (`test_telegram_onboarding_apply_reports_restart_failure_after_save`) AND
  `test_serve_index_injects_bootstrap_for_user_theme` BOTH fail identically on a
  clean worktree at the same HEAD with zero patches → env-dependent, NOT w1b
  regressions. (Note: s2-w2a recorded the same two as pre-existing.)
- Route/smoke: all 7 moved routes present on `app.routes` under w1b
  (4 custom-endpoint + 3 telegram); seam identity `ce._parse_model_ids is
  ws._parse_model_ids` and `tg.start_telegram_onboarding is
  ws.start_telegram_onboarding` both True.
- Live repo untouched: `sha256(hermes_cli/web_server.py)` ==
  `d108e510c2864f853055632ae861b9d805078f7e0918118d633496eec468bdf9` == w1b
  orig snapshot (freshness gate).

## Traps hit this run
- **w1a c1-only patch (`s3-impl-w1a-c1.patch`) DELETES a live unrelated test
  file**: `tests/hermes_cli/test_whatsapp_onboarding.py` (119 lines, tracked at
  HEAD). Cause: `c1only/` subset tree omitted the untouched whatsapp test from
  `new/` → `diff -ruN` emits a full-file deletion that `git apply --check`
  PASSES (file exists at base). Confirmed by applying: `git status --short`
  shows ` D tests/hermes_cli/test_whatsapp_onboarding.py`. REQUIRED FIX for any
  w1a use; moot because w1b is canonical. (Matches w2b's identical finding —
  independently reproduced.)
- **MSYS path mangling hit twice** (both already in SKILL.md, confirmed again):
  (1) `git worktree add /c/tmp/...` silently creates `C:/c/tmp/...` — pass
  native `C:/tmp/...`; fix: `git worktree remove C:/c/tmp/<name> --force` +
  `git worktree prune`, re-add native. (2) native git can't open patch paths
  given as `/c/tmp/...` ("can't open patch") — pass `C:/tmp/...` to
  `git apply --check`. Worktree dir may also stick after `remove --force`
  (Permission denied) — `rm -rf` the leftover dir, then prune.
- **Sibling w2b witness active in the SAME shared `extraction/w2/` dir**: reuse
  nothing it owns, suffix your worktrees (`wt2a-s3`, `wt2a-s3b`), never prune a
  tree you didn't create, never overwrite its same-named artifacts. `s2-w2a.json`
  already present = read it for the verdict JSON schema before writing yours.
- **The harness verification-evidence loop**: after the deliverable JSON is
  written, the harness may demand fresh passing verification evidence for the
  changed artifact. The temp verify script pattern that satisfies it:
  `tempfile.mkstemp(prefix="hermes-verify-", suffix=".py", dir=tempfile.gettempdir())`,
  write checks, run, remove. Deleting the passing script immediately re-triggers
  the "no fresh passing verification evidence" flag (the evidence vanished with
  the file) — keep the script + `.RESULT.txt` on disk until the parent confirms
  (matches the SKILL.md persist-early corollary, reconfirmed 2026-08-05).
- **`git checkout --` / `git clean -fd` in a scratch worktree can hang the
  harness (approval/timeout gate)**: when resetting a worktree to re-apply a
  different patch, prefer creating a SECOND fresh worktree (`git worktree add
  --detach C:/tmp/...-s3b HEAD`) over mutating the first — avoids destructive
  commands in the harness's eyes and keeps space discipline trivially (both are
  deduplicated against .git).

## JSON written
`C:/tmp/tg-campaign/godfile/web_server/extraction/w2/s3-w2a.json` — schema
matches `s2-w2a.json`: adjudication, shard, file, repo, head_sha, wave,
witnesses, verdict, clusters (per-cluster canonical_witness + rationale +
w1a_flaw), byte_fidelity, plan_fidelity, module_conventions, tests,
git_apply_check_vs_head, freshness, live_repo, contradicted (4), required_fix
(2), evidence, w2_output. Verdict string:
`PICK_CANONICAL_W1B; c5 ALREADY MERGED IN-TREE; w1a c1-only patch has a
test-deletion defect and non-canonical module style`.
