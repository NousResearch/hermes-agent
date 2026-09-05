# Wave-3 combined-canonical validation (two disjoint patches applied sequentially)

Validated recipe from tui_server shard s3, witness w3b (2026-08-05). Use when
wave-2 verdicts AGREE on a COMBINED canonical: both w1 patches are accepted,
each covering DISJOINT clusters (here: w1a c14+c22 → message_projection.py +
pet_payload_mixin.py; w1b c5+c6 → agent_build_config_mixin.py +
preview_restart_mixin.py). The w3 validator applies BOTH patches to one fresh
worktree and verifies the combined result — not a single-patch run.

## Protocol (all executed, all green)

1. **Pin live state**: `git rev-parse HEAD`, `git status --porcelain` (only
   unrelated pre-existing mods; the godfile itself must be clean), `wc -l` +
   sha256 + `git hash-object` of the godfile. Live HEAD may be ahead of the
   plan pin (tui_server: HEAD 0577116f83 vs plan pin f40fbcf409, +3 lines at
   274-279 inside `_LONG_HANDLERS` — outside every hunk).
2. **Orig freshness gate**: `sha256sum` BOTH witnesses' `orig/<godfile>` and
   compare against `git show <plan_pin>:<godfile> | sha256sum` — both must be
   byte-identical to the same baseline blob (both were e4bd9009..., 13905
   lines).
3. **Fresh worktree**: `git worktree add --detach C:/.../extraction/w3/wt-s3-w3b <HEAD-sha>`
   (native C:/ path). Verify baseline: worktree godfile sha256 == live blob,
   line count == live. Same-named `wt-s3-w3b` worktrees exist under OTHER
   campaigns (kanban_db, main) — confirm the full unique path before and after.
4. **Apply w1a then w1b**:
   - `git apply --check -v -p1 <w1a.patch>` → exit 0, offsets +3 (upstream drift).
   - `git apply -p1 <w1a.patch>` → exit 0. Status: `M <godfile>` + w1a's
     untracked modules/tests.
   - `git apply --check -v -p1 <w1b.patch>` ON THE ALREADY-PATCHED TREE →
     exit 0 with a NEGATIVE tail offset (`Hunk #2 succeeded at 13007 (offset
     -669 lines)`) — that is w1a's deletions shifting the tail context; it is
     the EXPECTED signature of clean sequential composition, not a warning.
   - `git apply -p1 <w1b.patch>` → exit 0. Final status: exactly `M <godfile>`
     + all untracked modules + tests from both patches (6 files here).
   - **Line math**: live − del1 − del2 + add1 + add2 == final `wc -l`
     (13908 − 672 − 211 + 45 + 4 = 13074 ✓; del/add counts from the two
     patches' own stats).
5. **Patch reproduces sandbox new/ trees**:
   - `cmp` each applied new module + test vs its sandbox `new/` file → 6/6
     byte-identical.
   - `git diff --no-index --stat <applied godfile> <w1a-new godfile>` must show
     EXACTLY three drift classes: (a) the upstream head-drift lines, (b) the
     OTHER patch's removed span (present in w1a-new, deleted in applied), (c)
     the OTHER patch's tail additions (present in applied, absent in w1a-new).
     Mirror for w1b. Count hunks (3 vs w1a-new, 4 vs w1b-new) and eyeball each
     `@@` context line to confirm it is one of those three classes. Anything
     else = real fidelity finding.
6. **py_compile** all 7 changed .py with the REPO VENV PYTHON the witnesses
   used — see interpreter-matching pitfall below.
7. **Byte-fidelity of moved units**: AST raw-slice compare of every moved
   unit vs pristine `git show HEAD:<godfile>` (not the worktree file), strip
   sanctioned lazy-import lines, assert 0 moved defs remain DEFINED in the
   applied (combined) godfile, both seam blocks present (w1a tail re-export
   `from .message_projection import` / `from .pet_payload_mixin import`; w1b
   `register(` loops). 43/43 units verbatim here.
8. **Runtime identity probe** (cwd = worktree, `sys.path.insert(0, '.')`):
   - w1a seam: `server.<name> is module.<name>` for every re-exported name
     (8/8 here, incl. module state like `_pet_payload_cache`,
     `_PET_REFERENCE_MIME_EXT`).
   - w1b seam (register/FunctionType rebinding): `server.<name>.__code__ is
     mixin.<name>.__code__` AND `server.<name>.__globals__ is vars(server)` —
     the rebound function is a NEW object, so `is` identity does NOT hold;
     code-object identity + globals-namespace check is the correct probe
     (4/4 here).
   - `server._methods['pet.info']` registered (HandlerRegistry seam) 1/1.
9. **Tests** (repo 3.11 venv python, `PYTHONPATH=<worktree>`,
   `-p no:cacheprovider`, scratch `--basetemp`): both shipped files (29+22=51)
   + the adjacent files each w2 verdict ran (78/78 here) + the full
   pre-existing suite `tests/test_tui_gateway_server.py` for zero-regression
   parity (517 passed == w2's recorded 517=517=517 baseline parity).
10. **Reverse-apply**: `git apply --check -R -p1 <w1b.patch>` FIRST (it was
    applied last), then `-R <w1a.patch>` — both exit 0.
11. **JSON + receipt**: `godfile-extraction-w3-witness.v1`, record BOTH w2
    verdicts' suite composition (their manifests, not summaries), expected
    drift deltas, interpreter used, combined line math. 3-line receipt, last
    line ends `exit 0`.
12. **Cleanup**: `git worktree remove --force` + `git worktree prune`, remove
    your own basetemps, confirm live repo status unchanged. Windows dir-stuck
    fallback below.

## Pitfalls from this run

- **Interpreter matching via pyc artifacts + dual-venv**: sandbox `new/`
  trees carry `__pycache__/*.cpython-311-pytest-9.1.1.pyc` — the pyc filename
  IS the witness's interpreter (3.11). This repo has BOTH `.venv` (3.12,
  environment-broken: missing certifi → `module 'hermes_cli' has no attribute
  'model_switch'` in test_make_agent_provider.py) and `venv/` (3.11, working).
  The 3.12 failures reproduced IDENTICALLY against the LIVE repo with the same
  interpreter → env-driven, not a regression; 78/78 with the 3.11 venv.
- **try/except-embedded module-state names**: `_PET_REFERENCE_MAX_BYTES` is
  assigned inside a module-level `try/except`, so it is NOT a top-level
  Assign/AnnAssign node — a `tree.body`-only finder returns None → false
  "NOT FOUND". Fix: `ast.walk` for Assign/AnnAssign targets anywhere, then
  compare the ENCLOSING top-level statement (the whole try block) raw slice
  live-vs-module. Same shape applies to `if`-wrapped constants.
- **Per-witness leftover-def scoping in post-worktree ad-hoc re-verify**:
  after the worktree is gone, re-running the fidelity core against the sandbox
  `new/` trees must scope the no-leftover-defs check PER WITNESS — each
  sandbox `new/server.py` is only ONE patch's server and legitimately still
  defines the OTHER patch's moved names. An all-names check yields 43 false
  failures (each witness's server "defines" the other's 21/22 names). The
  all-names check is only valid on the actually-combined applied tree (done
  in step 7). Scope: w1a's new/server.py → w1a's own moved names + its own
  seam marker; mirror for w1b.
- **Windows worktree dir stuck after `git worktree remove --force`**: the
  remove de-registers (git-hygiene-critical part — `git worktree list` clean)
  but the physical dir can refuse deletion ("Permission denied" / "Device or
  resource busy") when any process holds a handle. Do NOT kill python
  processes — most are sibling witnesses' shared-venv work. `rm -rf` may keep
  failing; `cmd //c rd /s /q` mangles the path in MSYS; PowerShell
  `Remove-Item -LiteralPath 'C:\native\path' -Recurse -Force` succeeds.
- **Heredoc driver trailing parse error**: a `python - <<'PYEOF'` driver that
  ran the temp verification script can print "unexpected EOF while looking
  for matching `)`" from bash AFTER the python output completed — the python
  evidence is complete; confirm with a one-liner (`python -c` json.load +
  py_compile) rather than re-running the whole driver.
