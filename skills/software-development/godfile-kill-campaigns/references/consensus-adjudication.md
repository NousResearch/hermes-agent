# Region-analysis consensus adjudication (blind passes → R*-CONSENSUS.md)

Recipe for adjudicating blind region-analysis passes (`R*-analysis.md`, pass A + pass B) into a
per-region consensus doc (`R*-CONSENSUS.md`) that fixes the first-cluster mixin spec. Proven on
slack adapter.py R2 (2026-08-05, epic #78647 / target #78638). READ-ONLY adjudication — the ONLY
file written is the consensus doc.

## Trigger
A godfile region's two blind witness analyses have (nominally) landed and an adjudicator must
produce the consensus verdict the extractors will implement.

## Step 0 — inventory on disk FIRST (do not trust the brief)
- `ls` the region dir. Witness passes may be MISSING: throttled stub waves silently drop pass B
  (slack R2: R1-B/R3-B/R4-B/R5-B present, **no R2-B**). Record it in the comparison table as
  MISSING and adjudicate on pass A + independent verification at pin — do not block on it.

## Step 1 — read pass A fully
Load the whole file. Extract its load-bearing claims into a checklist before verifying: member
ranges, class seam, base-hook census, call-site counts, entanglement claims (test imports/patches),
straddler ownership, open-PR overlap list, golden sha.

## Step 2 — independent verification battery at pin
Worktree detached at the pin (`git rev-parse HEAD` == golden sha, `git status --short` clean).
Every claim gets re-derived by script — **witness prose counts are unreliable, count yourself**
(pass A claimed "17 `self.send()` sites in base.py", exact regex gives 13, and its own cited list
even included a docstring line; claimed "52 bundle entries", bundle file has 54). Same rule as
byte-count-yourself in wave-3: a wrong count in the witness does not overturn the verdict, but the
CORRECTED number goes in the consensus §Corrections.

1. **Class seam**: `sed -n '<classline>p'` the godfile — confirm the single base and that the
   mixin-first line is a real change, not already present.
2. **AST member ranges**: parse the godfile; find the class BY NAME —
   `[n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '<Adapter>']`. Never
   `tree.body[0]` (module docstring is an `Expr`; helper classes like `_ThreadContextCache` appear
   before the adapter and crash `ast.unparse` on bases). Walk class-body FunctionDef/AsyncFunctionDef
   for `(lineno, end_lineno)`; assert the cluster window is contiguous (3023−2439+1 == 585L) with
   zero interleaved members.
3. **Per-body property scan**: for each candidate member, scan its line slice for slack-sdk names
   (`slack_sdk`, `slack_bolt`, `AsyncApp`, `AsyncWebClient`, ...), `super()`, `global`. Zero-everything
   → seam-free cluster, no lazy-import seam needed, mixin-first requirement comes purely from
   hook-shadowing (contrast C2's 2 `super()` calls which force mixin-first for MRO target reasons).
4. **Base-hook census**: grep base.py for `def <hookname>` — the 6-of-7 hook override claim must
   resolve to real def lines; a member NOT in base.py (e.g. `send_or_update_status`) confirms
   duck-typed dispatch (`getattr(adapter, name, None)` in run.py).
5. **Exact call-site counts**: `re.findall(r'self\.send\s*\(', src)` — NOT `grep self.send` (that
   catches `send_typing`/`send_multiple_images`/docstring mentions). Verify the framework-hook
   invocation claim with these numbers.
6. **Bundle cross-check**: count the region bundle's method entries by script; every entry must
   match the AST inventory by name+line. Nested defs inside a straddler (15 in `connect`, 3 in
   `format_message`) count as entries — explain the total (15 nested + 36 class methods + 3 nested).
7. **MRO back-refs**: grep the move window for `self.<name>` of helpers that STAY in the adapter
   (slash-context, client caches, thread-ts resolvers) — proves no helper needs to move; MRO is the
   seam. Also confirm in-file `self.send(` callers OUTSIDE the window (R3 media paths) exist — the
   live seam proof.
8. **Test entanglement**: repo-wide greps — any test importing the member names `from
   ...adapter`? (only `SlackAdapter` class imports are fine — still valid post-extraction); any
   dotted-path patch `slack.adapter.<member>`? False hit to know: `slack_adapter.send.assert_
   awaited_once()` is a LOCAL mock var in a discord test, not a patch of the real path.
9. **Open-PR gate (the deciding coordination factor)**: live `gh pr view <n> --json
   state,baseRefName,title` + `gh pr diff <n> --name-only` + hunk parse. Map hunks by
   **function-context header** (the `@@ ... @@ def <name>` line), NEVER raw hunk numbers (PR base
   coords differ from pin coords). Confirm the witness's overlap list member-by-member.
   - **MSYS /tmp trap**: if Windows Python must read the PR diff, redirect to `C:/tmp/...` — a
     git-bash `/tmp/x.diff` is invisible to Windows python (FileNotFoundError). `/c/tmp/...` works.
10. **Witness pin divergence is cosmetic when the godfile blob matches (main.py R2, 2026-08-05)**:
    pass A pinned `9a9cf6ae83`, pass B pinned `169758d42f`, and neither equaled the campaign
    convention `5c5f1a6b76` — yet `git rev-parse <each-sha>:<godfile>` returned the IDENTICAL blob
    (`e6612636da…`, 12,599 lines) for all three, so every witness line number stayed valid. Resolve
    pin conflicts by blob identity, never commit equality; record the campaign-convention pin + the
    shared blob in the consensus header and park the divergence in §Corrections as cosmetic. Only a
    blob MISMATCH forces re-derivation of line numbers.
   - **Sweep ALL open PRs touching the godfile, not just the witnesses' lists (main.py R4,
     2026-08-05)**: `gh pr list --repo <r> --state open --json number --limit 100`, then per PR
     `gh pr diff <n> --name-only | grep <godfile>`. A MIS-TITLED sibling shard PR can be silently
     extracting YOUR region: #79661 was titled "kanban shard s4" but its branch
     (`gfg/main-extract-s4-w1a`) and file list were main.py and its hunks sat inside R4.
     Cross-check title vs branch vs file list vs hunk context — a mismatch is the sed-derived-PR
     title-inheritance bug seen from the adjudicator side; flag it for the Credit-Ledger.
   - **Def-level overlap: parse the PR diff, don't guess.** Extract each new module's moved defs
     (`^\+def (\w+)` in that file's diff section) and confirm which defs stay (`-def <name>`
     absent). Map hunks to OLD-line ranges and intersect with the region window (main.py R4:
     OLD 7640–7930 + 7983–8223 → recovery + quarantine clusters = pass-B's C4 value bundle,
     already in flight). Verify base lineage: `git merge-base --is-ancestor <pr-base> <pin>` —
     ancestor ⇒ the extraction is from a real superset; coordinate/don't duplicate.
   - **Verdict impact**: an in-flight extraction of a cluster can DISQUALIFY a witness's value
     pick as first slice (C4+C5+C6 update_install ~890L lost to the 59L C7 node-runtime leaf on
     interlock hygiene, not merit) and re-gates the surviving remainder ("land #79661 first").
     Adjudicate first-slice candidates on entanglement / seam / shims / coverage /
     dependency-root; a dependency-root leaf wins ties when the value bundle collides with
     in-flight PRs.

## Step 3 — write R*-CONSENSUS.md (fixed structure)
1. Header: epic/target, region, pin sha (RE-VERIFIED this session), adjudication mode.
2. **Witness comparison table**: | Witness | Status | First-cluster pick | Mixin | Class seam | Evidence quality | — pass B row
   says MISSING on disk; independent-verification row says CONFIRMS pass A with 14/14 (or N/M) claims.
3. **Independent verification table**: every load-bearing pass-A claim, the check performed, ✅/⚠️ result.
4. **CONSENSUS-VERDICT** block: cluster, mixin module/class, EXTRACTABLE-NOW + conditions (open-PR
   interlock), mixin-FIRST mandatory statement with the hook census, no-re-export statement, BLOCKED list.
5. **§Corrections**: witness count errors, non-verdict-affecting (e.g. 13-not-17 call sites,
   54-not-52 bundle entries). A corrections section is a feature, not a failure — it is exactly what
   the independent verification is for.
6. **First-cluster spec table**: mixin file, class, members (verbatim ranges), move window, golden
   sha, class seam line, new imports, re-export block (NONE if verified), deletions.
7. **Execution order**: pre-flight re-pin; Option A (interlock with open PR → ship C1) vs Option B
   (no interlock → ship zero-collision cluster C3 first); second wave; follow-on; NEVER-slice list.
8. **Seam/test plan**: telegram `a is b` identity asserts per member, isinstance probe, existing
   test surface (flagship file line count, per-cluster dedicated files, cross-platform generic
   subset), mock-bootstrap note (e2e conftest installs SDK mocks — mixin rides same load path).
9. **Evidence anchors**: every line number verified at pin in the detached worktree; READ-ONLY
   statement.

## Pitfalls (each cost real time this session)
- **Witness prose numbers are NOT evidence**: re-derive every count (call sites, bundle entries,
  test counts) at pin by script before writing the consensus. Document the delta in §Corrections.
- **`tree.body[0]` is not the adapter**: find the class by name or you crash on the docstring
  `Expr` / helper classes and end up inventorying the wrong class.
- **`grep self.send` ≠ `self.send(` sites**: include the paren in the regex or you count
  `send_typing`/`send_multiple_images`/docstring prose.
- **Consensus is authoritative over the task brief**: if the brief's parenthetical says one thing
  and the hook census says mixin-first, the census wins (same rule as the discord blind-rereview pass).
- **A missing pass B is a documented condition, not a blocker**: adjudicate on pass A + independent
  verification and say so explicitly in the comparison table.
