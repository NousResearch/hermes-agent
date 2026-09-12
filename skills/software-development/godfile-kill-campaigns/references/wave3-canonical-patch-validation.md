# Wave-3 canonical-patch validation (extraction stage 8, validator witness)

Validated recipe from main.py shard s2, witness w3b (2026-08-05): empirically
validate the CANONICAL patch (chosen in wave 2) in a fresh deduplicated
worktree, then write a verdict JSON + 3-line receipt and exit 0. Never edit
the live repo.

## Context you must establish first
1. Read BOTH wave-2 verdicts for the shard — they may DISAGREE on canonical
   (s2: w2a→w1a, w2b→w1b). The task brief names the canonical; your job is to
   confirm it empirically, not to re-adjudicate from scratch.
2. Read the canonical patch + its sandbox `new/` tree + the brief
   (EXTRACT ALWAYS + SPACE DISCIPLINE).
3. Pin the live state: `git rev-parse HEAD`, `git status --porcelain`,
   `wc -l hermes_cli/main.py`, sha256 of the godfile. The live file may have
   drifted from the plan pin (s2: +3 lines at 6876, ELECTRON_DISABLE_SANDBOX)
   — record it, verify the delta is outside the s2 region AND outside every
   patch hunk before trusting apply results.

## Validation protocol (all validated)
```
git worktree add --detach C:/tmp/.../extraction/w3/wt-sN-w3b <HEAD-sha>
cd <wt> && git apply --check -v -p1 <patch>   # capture $? immediately, no pipe
git apply -p1 <patch>                          # then git status --porcelain
```
- Use native `C:/...` paths for the patch argument — MSYS mangles `/c/...`
  into `C:\c\...` for native git/python (loud error) or silently for
  `git worktree` paths (silent wrong-location checkout).
- Worktree at CURRENT HEAD is valid when the live worktree's godfile is
  clean (`git status` shows only unrelated pre-existing mods); the +N-line
  drift since the plan pin is expected and harmless if outside all hunks.

## Checks, in order
1. **Apply**: `--check` exit 0, then `git apply` exit 0; expected status =
   exactly `M <godfile>` + the new files untracked.
2. **Patch reproduces sandbox new/ tree**: `sha256sum`/`cmp` the applied
   files vs sandbox `new/`. New modules + shipped test must be byte-identical.
   The godfile may differ by EXACTLY the live repo's own drift — run a full
   `diff -u` and confirm the only hunks are the documented unrelated
   insertions, then count them.
3. **py_compile** every changed .py with the repo venv python.
4. **Byte-fidelity of moved units** (strip-then-exact-equal):
   - AST-extract each moved unit's source from LIVE main.py and from the
     applied module (handle BOTH `ast.Assign` and `ast.AnnAssign` —
     `_AUX_TASKS: list[...] = [...]` is AnnAssign).
   - Strip sanctioned lazy-import insertions from the module copy with
     whole-line regexes, then assert exact equality:
     ```python
     LAZY_1LINE = re.compile(r"^[ \t]*from hermes_cli\.main import [A-Za-z_][A-Za-z0-9_]*\n", re.MULTILINE)
     LAZY_BLOCK = re.compile(r"^[ \t]*from hermes_cli\.main import \([^)]*\)\n", re.MULTILINE | re.DOTALL)
     ```
     Replacing with `\n` instead of `""` leaves indentation-only lines that
     break equality; the parenthesized multi-line block needs DOTALL.
   - Also assert: 0 moved defs remain DEFINED in applied main.py; both
     re-export `from hermes_cli.<module> import` blocks present; runtime
     identity (`main.cmd_model is picker.cmd_model`).
5. **Tests in applied state** (from worktree root, `PYTHONPATH=<worktree>`,
   `-p no:cacheprovider`, scratch `--basetemp="C:/..."`):
   - shipped regression test file (must pass);
   - the DECISIVE existing tests (see below);
   - adjacent suite (the files each w2 verdict ran) for zero-regression.
   Verify module resolution first: `python -c "import hermes_cli.main; print(main.__file__)"`
   must print the worktree path, not the live repo's.

## Decisive-test adjudication of conflicting w2 verdicts
When w2 witnesses picked different canonicals, the conflict usually hinges on
a monkeypatch seam. s2: `tests/hermes_cli/test_gmi_provider.py:296`
monkeypatches `hermes_cli.main._model_flow_api_key_provider` then calls
`select_provider_and_model()`. w1a's moved code lazy-imports the flows inside
the function body (`from hermes_cli.main import (...)`) → resolves the
monkeypatched attr at call time → test passes. w1b binds the flows at module
level from `model_setup_flows` → monkeypatch bypassed → real interactive flow
runs → test fails. Protocol:
- Run the decisive test file(s) in the applied worktree (w1a: 22/22 for
  test_gmi_provider + test_aux_config).
- Read each w2 verdict's suite MANIFEST: w2b's "81/81 adjacent" omitted
  test_gmi_provider.py entirely, which is why it missed w1b's break. A
  verdict whose suite omits the decisive test is weaker evidence.
- State both verdicts' suite composition in your JSON.

## Shared-dir collisions (parallel witnesses)
`extraction/w3/` is shared by all shard witnesses running concurrently.
- Shard-suffix every artifact you create: `check_fidelity_s2_w3b.py` (the
  generic `check_fidelity_w3b.py` was silently overwritten by the s1-w3b
  witness mid-run).
- If a script you own suddenly errors with a path referencing another
  shard's `wt-s*` worktree, or its output names methods you never wrote —
  re-read the file; a sibling overwrote it. Recreate under a unique name.
- Never delete a same-named file you didn't create; clean only your own
  worktrees and basetemps.
- After ANY fix to your artifacts, re-run a self-verify script against the
  final on-disk state (catches stale copied claims, e.g. junk-line counts).

## Verdict JSON + receipt
Schema `godfile-extraction-w3-witness.v1`: witness/wave/phase/shard/file/repo,
live_pin (head, line count, sha256, drift note), canonical_patch, resolved
wave-2 conflict, per-check validation results with evidence, verdict,
caveats_for_ship, verification_artifacts, notes, receipt (exactly 3 lines,
last line ends `exit 0`). Validate the JSON parses before finishing.

## Cleanup (space discipline)
`git worktree remove --force <wt>` from the live repo + `git worktree prune`;
remove your pytest basetemps; confirm the live repo's `git status` is
unchanged from before (same pre-existing mods only). Do NOT prune worktrees
you didn't create.
