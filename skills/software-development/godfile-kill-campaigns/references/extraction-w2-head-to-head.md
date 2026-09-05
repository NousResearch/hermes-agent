# Extraction w2 adjudication — head-to-head (overlapping coverage) recipe (cli.py shard s4, witness w2b)

Blind cross-check when BOTH wave-1 witnesses covered the SAME clusters. Session-tested 2026-08-05 on cli.py s4 (w1a + w1b both did c8 modal-prompts + c6 voice; identical 30 top-level + 10 nested method sets). Complement to `extraction-w2-adjudication.md` (the disjoint case, cli s1).

## When this applies
- w1 MANIFESTs list the same `clusters_covered` and the same moved method names (compare method SETS, not just cluster names).
- Then the w2 job is a direct patch-vs-patch comparison: same content, different craftsmanship. Every fidelity dimension becomes a tie-breaker.

## Read order (same as disjoint, plus one)
1. `shard-plan.json` → `shards.s4` (clusters, classification, proposed_module, slice_order).
2. Both sandboxes: `MANIFEST.json`, patch, `orig/` + `new/` trees.
3. Live godfile (READ ONLY) + brief.
4. **The merged precedent mixins** (`hermes_cli/cli_commands_mixin.py`, `cli_billing_mixin.py`) and their extraction commits — conventions are adjudicated against what ALREADY shipped, not opinion:
   ```bash
   git log --all --diff-filter=A -- hermes_cli/cli_commands_mixin.py   # find extraction commit
   git show <commit> -- cli.py                                          # how seams were closed
   ```
   Confirmed conventions from precedent: one blank line between mixin methods; NO in-class `# ====` section comments inside mixin classes (they either never existed or traveled with the cluster); lazy `from cli import ...` per-method lines as first body statement.

## Decisive criteria in order (each one actually split the s4 witnesses)
1. **Blank-line separator fidelity** — w1a stripped the trailing blank after EVERY moved method → its mixin has methods jammed together, zero blank lines between (PEP8 violation; precedent mixins have exactly one). w1b preserved → byte-identical to live. Detection:
   ```python
   prev = None
   for i, line in enumerate(lines):
       if re.match(r'^    def ', line):
           if prev is not None and lines[prev+1:i][-1].strip() != "":
               print("NO BLANK before next method")
           prev = i
   ```
   Note: the LAST method of each mixin file legitimately has no trailing blank (EOF) — not a defect.
2. **In-class section comment placement** — w1a left `# ==== Voice mode methods ====` orphaned in new cli.py (now mislabeling the NEXT staying block, the wake-word section) plus a double blank at the seam; w1b moved the header into the mixin with the cluster and left a clean single-blank seam. Check the seam:
   ```
   grep -n "Voice mode methods" new/cli.py          # orphaned = present
   grep -n "Voice mode methods" new/hermes_cli/*.py # traveled = present in mixin
   ```
3. **MRO order between the NEW mixins is irrelevant** — w1a: `(..., CLIVoiceMixin, CLIModalPromptsMixin)`, w1b: `(..., CLIModalPromptsMixin, CLIVoiceMixin)`. Verify no method-name collisions across ALL mixins (existing + both new) by set intersection; empty = order is cosmetic, keep the canonical witness's.
4. **Patch base freshness** — w1a's `orig/cli.py` was byte-identical to CURRENT live (zero-offset apply); w1b's was the pre-drift f40fbcf409 base (applies with a benign 4-line offset, documented in its MANIFEST). Both apply cleanly (`git apply --check` rc=0) — the offset is NOT a defect when hunks are region-scoped (contrast: whole-file hunks die on drift, see main SKILL.md freshness pitfall). Zero-offset is nicer but both are shippable; freshness alone did not decide s4.

## Scoped byte-fidelity compare (the script that worked)
Naive name-keyed block extraction produced FALSE DIFFERS twice before the fix:
- **Parent tracking must treat `class` as a boundary**: a module-level `def` before the class became the "parent" of every class method (all 30 methods reported "no live block with parent=None").
- **Duplicate nested names across parents**: `_panel_box_width` exists 3× in live (parents `_get_slash_confirm_display_fragments`, `_get_approval_display_fragments`, `run`) with DIFFERENT signatures (56/86 vs 46/76 vs 46/76) — name-only lookup compares the wrong copies.
Fixed extractor: regex `^(\s*)(?:def|class)\s+(\w+)\s*[\(:]`, track (start, indent, name, parent) with parent = nearest enclosing def/class of strictly smaller indent; blocks keyed `(name, parent)`; compare after normalizing `^\s*from cli import .*$` lines (the one sanctioned per-method insertion). Result: w1b 28/30 byte-identical + 2 EOF artifacts; w1a 0/30 identical.

## Verification runs (independent, both witnesses)
- `git apply --check` from the LIVE REPO ROOT (from the sandbox dir it fails "cli.py: No such file or directory"); use native `C:/...` paths for the patch argument (git rejects MSYS `/c/`).
- `py_compile` all 10 changed/new .py files with repo venv python — note `cfile=os.devnull` fails on Windows ("nul is a non-regular file"); use a tempdir cfile.
- pytest with a dedicated `--basetemp` (default basetemp collides with parallel pytest runs → PermissionError WinError 5).
  - w1a: its validate-tree conftest (sys.path.insert live repo + `hermes_cli.__path__.insert(0, validate/hermes_cli)`) → 41 passed.
  - w1b: kept worktree `wt-s4-w1b` (f40fbcf409 + patch applied) → 42 new + 48 existing-net (approval_ui + extension_hooks + voice_cli_integration) = 90 passed, matches its MANIFEST.
- **Count defs to verify MANIFEST counts**: `^    def ` = top-level, indented = nested. w1b MANIFEST claimed "41 moved (30 top-level + 11 nested)" but the real count is 40 (30 + 10). Doc-only miscount — catch by counting, never trust the MANIFEST number verbatim.

## Verdict JSON
- Schema: match the sibling outputs already shipped in `extraction/w2/` (`run-godfile-extraction-w2-witness.v1` — same keys as the disjoint recipe).
- `overlap_note` must state the head-to-head condition explicitly (same clusters, same method sets, "a true head-to-head comparison").
- Per-witness `patch_verdicts`: byte_fidelity PASS/PARTIAL (state exactly which methods differ and why: separator blanks / EOF artifact / next-section comment), plan_fidelity, module_conventions, regression_tests (with independent re-run counts), git_apply_check, freshness.
- `canonical`: per-cluster map — s4: c8→w1b, c6→w1b (the byte-faithful witness), NOT w1a despite its zero-offset freshness advantage.
- Self-verify (throwaway script, `hermes-verify-` prefix, tempfile under %TEMP%, auto-clean): JSON parses; schema keys match sibling; canonical values ∈ {w1a, w1b}; re-run `git apply --check` on both patches (factual anchor); live head/blob match the recorded anchors; `git status --short` shows ONLY the pre-existing mods — normalize BOTH sides to bare filenames (`l.split(None, 1)[1]`) or the two-column status prefix causes a false "live repo touched" failure.
- Merge instructions: apply the canonical patch as-is; document why the other witness's patch was rejected (concrete fidelity/convention deltas, not "worse"); note MRO-order equivalence and test-file placement choice.

## Cleanup
`rm -rf` any scratch basetemp dirs; leave both w1 sandboxes and the w1b validation worktree intact (they are the parent's evidence).
