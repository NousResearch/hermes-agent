# Extraction w2 adjudication — web_server s3 (w2b, 2026-08-05)

Wave-2 witness recipe for web_server shard s3 at HEAD `0577116f83`. This is the
CONTESTED-cluster + in-tree-interlock case: c1 (custom endpoints, 13 methods)
was extracted by BOTH w1 witnesses with different conventions; c5 (whatsapp,
20 methods) was already extracted in-tree by a sibling lane; c6 (telegram
onboarding, 12 methods) was covered by only w1b.

## Coverage map (determine FIRST)
- **c1** — covered by w1a AND w1b → head-to-head on conventions (bodies byte-identical both).
- **c5** — covered by w1a only, but ALREADY COMMITTED in-tree (commits `3ca0d4fbfd` +
  `0577116f83` → `web_routers/whatsapp_onboarding.py`). Canonical = the in-tree
  extraction; w1a's duplicate mixin is discarded. w1a's FULL patch does not apply
  to HEAD (`git apply --check` exit 1) — that's expected interlock, not a defect.
- **c6** — covered by w1b only → canonical = w1b, minus documented deviations.
- Both witnesses ran reduced scope (top-2 highest-agreement move clusters), consistent
  with the campaign brief; c2/c3/c4/c7/c8/c9/c10 were not covered by either — leave for later waves.

## Verdicts recorded
- c1 → **w1b** (`web_routers/providers_custom_endpoints.py`): APIRouter + `late()`
  seam, `include_router` at web_server:7109, legacy re-exports, logger
  `"hermes_cli.web_server"` — matches the merged in-tree precedent. w1a's
  `hermes_cli/custom_endpoints_mixin.py` (top-level module, direct `app` import,
  `@app.*` decorators) is byte-faithful 13/13 but fails the standalone-import probe
  (see SKILL.md pitfall) and its c1-only patch deletes a live unrelated test file.
- c5 → **in-tree sibling extraction** (not either witness's patch).
- c6 → **w1b** (`web_routers/telegram_onboarding.py`): 11/12 moved byte-identical;
  `_telegram_onboarding_request_sync` intentionally kept in web_server and reached
  via `late()` (test_web_server.py:1214 monkeypatches `web_server._telegram_onboarding_request_sync`
  — verified present; plan-fidelity deviation, functionally safe). REQUIRED FIX:
  module-level `_TELEGRAM_USER_ID_RE = late_attr("_TELEGRAM_USER_ID_RE")` breaks
  standalone import — use `LateState` or call-site lookup.

## Evidence numbers (reproducible)
- Byte fidelity: 13/13 (w1a c1), 13/13 (w1b c1), 11/11 (w1b c6 moved bodies) —
  AST source-segment vs `git show HEAD:hermes_cli/web_server.py`.
- `git apply --check -p1`: w1a-c1 exit 0 vs HEAD-index AND worktree; w1a-full exit 1
  (stale); w1b exit 0 both. `git merge-base --is-ancestor f40fbcf HEAD` = false
  (w1a's stated base not in HEAD lineage).
- Fresh-worktree pytest `--basetemp`: w1b new file 12/12 (manifest claimed 16 —
  count `def test_` yourself), combined 20 passed, full `test_web_server.py`
  131 passed / 4 skipped / 2 failed, `test_web_server_profile_unification.py`
  15/1 — ALL failures reproduced identically on a pristine-HEAD worktree
  (`test_telegram_onboarding_apply_reports_restart_failure_after_save`,
  `test_serve_index_injects_bootstrap_for_user_theme`, `..._restarts_target`).
- Standalone-import probes: w1a mixin → `ImportError: cannot import name
  '_api_key_display' from partially initialized module`; w1b telegram → ImportError
  via module-level `late_attr`; w1b providers → OK; in-tree whatsapp/git routers → OK (baseline).

## Traps specific to this shard
1. **`diff -ruN` subset-tree deletion artifact**: w1a's `c1only/` tree omitted
   `tests/hermes_cli/test_whatsapp_onboarding.py` from `new/` while `orig/` had it →
   the c1-only patch emits a full-file deletion that `git apply --check` accepts.
   Always list the patch's file set and eyeball whole-file `@@ -1,N +0,0 @@` hunks.
2. **c1only orig web_server.py vs HEAD content**: verify the subset tree's `orig/`
   matches HEAD before trusting `--check`; the manifest's "generated against new HEAD
   content" claim was accurate here (apply exit 0), but the deletion defect made the
   patch unusable regardless.
3. **In-tree sibling commits shift line numbers**: w1b's c6 line numbers (~8436-8780)
   are ~460 lower than the plan's (8896-9259) because the build tree already contained
   the whatsapp extraction. Freshness ≠ stale: check the patch applies to HEAD, not
   that line numbers match the plan.
4. **`_CREDENTIAL_PROBES` stays in web_server**: used by the staying
   `validate_provider_credential` (c2, singleton, excluded by reduced scope) — neither
   witness moved it; correct.
5. **Baseline failures must be proven on pristine HEAD**: three tests fail in the w1b
   worktree; all three fail identically on a clean worktree at the same HEAD with no
   patch. Run the failing set on the clean worktree BEFORE writing "pre-existing" in
   the verdict.

## JSON shape written
`s3-w2b.json` keys (matches w2a precedent `s2-w2a.json`): adjudication, shard, file,
repo, head_sha, wave, witnesses, verdict, clusters (per-cluster canonical_witness +
rationale + byte_fidelity), byte_fidelity, plan_fidelity, module_conventions, tests,
git_apply_check_vs_head (both HEAD-index and worktree results), freshness, live_repo,
contradicted (4 items), required_fix (3 items), evidence, w2_output.
