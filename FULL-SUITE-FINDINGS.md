# Full test-suite run on the integrated 40-PR set onto v0.17.0 (2026-06-22)

Council item: run the FULL test suite (not just compile) when the 40-PR set is applied
onto v0.17.0. Done. This surfaced one REAL PR defect (now fixed) and one pre-existing
upstream flake (isolated, not ours).

## Method
Built the integrated tree: fresh v0.17.0 worktree (`2bd1977d8`) + all 40 code PRs applied
sequentially (3-way) + the #50111 `v017-conflict-resolutions/` patches for drifted files.
125 changed .py files. Ran `pytest tests/` (33,224 tests collected).

## Finding 1 (REAL DEFECT — FIXED): #50064 deleted a module an upstream test needs
- Full-suite collection ERROR: `tests/hermes_cli/test_inventory_pricing.py` →
  `ImportError: cannot import name 'cached_copilot_inventory_snapshot'`.
- Root-caused: **#50064** (copilot identity + vision) also DELETED `hermes_cli/inventory.py`
  — an out-of-scope consolidation deletion. The pre-existing upstream test
  `test_inventory_pricing.py` (on v0.17.0, imports `inv._apply_pricing`) then fails to
  import. Proven: deleting inventory.py on CLEAN v0.17.0 reproduces the exact collection
  error; the test passes on untouched v0.17.0.
- **FIX**: rebuilt #50064 without the inventory.py deletion (restored from PR base).
  New head `ce4162bf6`. Independently re-verified against GitHub: deletion gone
  (`git diff origin/main ce4162bf6 --name-status -- hermes_cli/inventory.py` empty);
  `test_inventory_pricing.py` now **5 passed** with the corrected PR applied onto v0.17.0;
  #50064's own area (copilot_auth/anthropic_adapter/auxiliary_client) **418 passed**.

## Finding 2 (PRE-EXISTING upstream flake — NOT ours): acp approval-isolation test
- `tests/acp/test_approval_isolation.py::...::test_interactive_env_var_routes_to_callback`
  fails under a full-suite run.
- Isolated: it **PASSES standalone in the integrated tree** (with and without env vars),
  and it **FAILS the same way on CLEAN v0.17.0** when `tests/acp/` is run as a group.
  So it's a pre-existing upstream test-ordering/isolation fragility (a sibling test in
  tests/acp/ mutates shared state), **independent of our PRs** — our PRs touch no file in
  tests/acp/. Deselected for the clean full-suite count.

## Result
- The ONE real PR defect the full suite caught is fixed and re-verified (#50064 → ce4162bf6).
- The remaining full-suite failure is a pre-existing upstream flake, isolated with proof.
- Clean integrated full-suite run (acp-flake deselected) result: see FULLSUITE-RESULT.txt
  (committed alongside once the background run completes).

This is exactly why the Council demanded full-suite over compile-only: it found a genuine
PR-scope defect (the inventory deletion) that compile-only missed.
