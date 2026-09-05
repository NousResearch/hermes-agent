# cli.py kill closeout (2026-08-05) — merger schema shapes, union assembly, PR-script derivation traps

The cli kill (eighth god, 5 PRs #79706 + #79708-#79711) closed the campaign's hard-won
merger-parser gaps and added two union-assembly patterns. Everything here extends the
pitfalls in the umbrella SKILL.md; this file carries the exact shapes and recipes.

## Verdict schema shapes the merger parser now handles (six total)

Proven shapes, in the order the parser branched on them:

1. `patch_verdicts: [{patch, verdict}, ...]` — LIST of dicts (older shape).
   ```python
   pw = "w1a" if "w1a" in pv.get("patch", "") else "w1b"
   v = pv.get("verdict")
   ```
2. Per-cluster `canonical` / `winner` / `source` fields (s5 style).
3. Top-level `verdict` (string) + `canonical` / `canonical_patch` / `recommendation`.
4. `adjudication` dicts (w2b) and `canonical_merged_patch` keys (w2a).
5. **`patch_verdicts` as a DICT keyed by witness** (cli s3-w2a):
   ```json
   {"patch_verdicts": {"w1a": {"verdict": "VERIFIED_CONTENT_CONTRADICTED_WIRING_C12",
                               "clusters_covered": ["c11","c12"], ...},
                       "w1b": {...}}}
   ```
   Iterating this as a list yields STRING keys → `AttributeError: 'str' object has no
   attribute 'get'` at `pv.get("patch")`. Fix:
   ```python
   if isinstance(pvs, dict):
       for pw_key, pv in pvs.items():
           pw = "w1a" if "w1a" in str(pw_key) else "w1b"
           v = pv.get("verdict") if isinstance(pv, dict) else pv
   elif isinstance(pvs, list):
       # skip non-dict entries
   ```
6. **NO `patch_verdicts` at all — `overall_verdict` + `cluster_adjudications`** (cli s3-w2b),
   and w3 `verdict` as a DICT (cli s3-w3a/w3b):
   ```json
   {"verdict": {"canonicals": "PASS — all 4 canonical mixin modules ... byte-fidelity 27/27, shipped tests 44/44",
                "c12_repair": "VERIFIED — PEP 562 module __getattr__ bridge removed; in-method lazy 'from cli import' imports present ..."}}
   ```
   Fix: scan `overall_verdict`/`cluster_adjudications` per-cluster fields for witness names,
   fall back on the overall text; for a dict verdict join its values and scan for
   PASS/VERIFIED + witness names:
   ```python
   v_text = " ".join(str(x) for x in v_raw.values())
   if "PASS" in v_text or "VERIFIED" in v_text: ...  # then look for "w1a"/"w1b" in v_text
   ```

## The 422 uncommitted-worktree trap (cli, 2026-08-05)

The merger's `apply_and_verify` applies the canonical patch to each `wt-<region>` worktree
but does NOT commit. `git push fork <branch>` succeeds (branch exists from
`git worktree add -b`), yet the branch's commit is still `main` → PR creation 422s:
`"No commits between NousResearch:main and andrexibiza:gfg/cli-extract-s3-w1a"`.

Symptom pattern seen: s1 (manually committed union) got PR #79706; s2-s5 (never committed)
all 422'd. Fix: commit EVERY worktree before PR creation:
```bash
git -C <wt> add -A && git -C <wt> commit -m "refactor(cli): extract shard s<N> ..." --no-verify
git ls-remote --heads fork | grep cli-extract   # verify each branch sha != main's sha
```
Treat any 422 after a derivation run as a symptom of either this (uncommitted worktree)
or the wrong-mapping symptom (see below), never a transient.

## Surgical-rebase union: patch rewrites only the godfile, modules copy separately (cli s1)

The w2 witness produced `s1-w1a-surgical-rebase.patch` — `git apply --numstat` shows ONE
file (`cli.py`, 26+/783−) because w1 had already extracted the modules and the surgical
patch only re-slices the godfile at HEAD. Applying it alone then running the shipped tests
fails with `ModuleNotFoundError` at the godfile's re-export block (cli.py:57 —
`from hermes_cli.cli_content_scrub import (...)`).

Fix — after applying both the w1b patch (as-is) and the surgical-rebase w1a patch, copy the
w1a witness's modules + tests from its `new/` tree:
```bash
cp <w1>/s1-w1a/new/hermes_cli/cli_content_scrub.py <w1>/s1-w1a/new/hermes_cli/cli_display_formatters.py <wt>/hermes_cli/
cp <w1>/s1-w1a/new/tests/hermes_cli/test_cli_content_scrub.py <w1>/s1-w1a/new/tests/hermes_cli/test_cli_display_formatters.py <wt>/tests/hermes_cli/
```
Then run the FULL shipped set (attachments + worktree + content_scrub + display_formatters):
64 passed. The ModuleNotFoundError signature distinguishes this from a real extraction
defect — the modules exist in `new/`; the surgical patch just didn't carry them.

## PR-script derivation sequence that bit (tui, then caught on cli)

1. `sed` the previous god's script: W dir, MAINFILE path, BRANCH_PREFIX.
2. REWRITE MIXINS + TITLES dicts (sed does NOT touch them).
3. REWRITE the branch-winner mapping line from the merger's manifest, never the previous
   god's pattern (tui: s1/s2→w1b, s3/s5→w1a, s4→w1b — the sed'd kanban mapping was s1/s2/s5→w1a).
4. `grep -n 'w1a" if region'` BEFORE firing.
5. After push, verify `git ls-remote --heads fork | grep <prefix>` and PR head shas.
6. On 422: check uncommitted worktrees first (above), then the mapping.

## Godfile real paths differ per god (EX_MAINFILE)

- adapter: `plugins/platforms/telegram/adapter.py`
- base: `gateway/platforms/base.py`
- tui_server: `tui_gateway/server.py` (NOT `hermes_cli/tui_server.py`)
- cli: `cli.py` at repo ROOT (NOT `hermes_cli/cli.py`)
A merger printing `py_compile=failed` on EVERY region = wrong EX_MAINFILE, not broken
patches — re-verify against the real path before believing it.
