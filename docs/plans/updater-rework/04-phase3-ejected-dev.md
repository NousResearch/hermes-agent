# Phase 3 — Ejected / Dev Mode: in-repo launcher, `dev sync`, worktree updates

> **For Hermes:** subagent-driven-development, task-by-task.
> Read `docs/updater-world.md` §2.5, §2.5.1, §2.5.2, §2.9 first.

**Goal:** A git checkout is a first-class activation target: it carries its
own launcher (`bin/hermes`), gets provisioned by one verb (`hermes dev
sync`), launches app surfaces only from fresh builds, and updates via
worktrees instead of autostash.

**Definition of done:** `scripts/e2e/test-ejected-worktrees.sh` passes:
two worktrees, independent venvs, symlink switching between them and a
managed slot, and a worktree-style update on a dirty tree that never
touches the user's changes.

---

## Task 3.0: Ship the launcher in-repo

**Files:**
- Create: `bin/hermes` (committed binary? NO — see below), `bin/README.md`
- Modify: `.gitignore`, `scripts/release/build-bundle.sh`

**Decision (already made in design, §2.5.1):** the launcher binary itself
is NOT committed to git (platform-specific, binary-in-repo pain). Instead:
- `bin/hermes` in a CHECKOUT is a tiny committed shell/cmd polyglot stub
  that (a) uses a prebuilt launcher at `.hermes-launcher/hermes` if `dev
  sync` has installed one, else (b) falls back to `exec .venv/bin/python -m
  hermes_cli.main "$@"` with inline env hygiene (unset PYTHONPATH/
  PYTHONHOME) — so a bare clone works with zero downloads.
- `hermes dev sync` (task 3.2) downloads the release launcher binary into
  `.hermes-launcher/` (gitignored) for full native behavior.
- In BUNDLES, `bin/hermes` is the real native binary (phase 1 already does
  this).

**Step 1:** write the stub (POSIX sh + `.cmd` sibling for Windows), commit.
Keep it under 30 lines; its ONLY jobs are env hygiene + venv exec +
"run dev sync" error text when `.venv` is missing.

**Step 2 (verify):** from this worktree: `./bin/hermes --dev --version`
works against `.venv` (plain `--version` refuses per §2.5.1a — cwd is
inside the checkout); delete `.venv` → clear error message, exit 3.

**Step 3:** Commit: `feat(dev): in-repo launcher stub`.

## Task 3.1: `hermes dev` subcommand skeleton

**Files:**
- Create: `hermes_cli/subcommands/dev.py`
- Modify: `hermes_cli/main.py` (wire parser)
- Test: `tests/hermes_cli/test_dev_cmd.py`

**Step 1 (failing test):** `hermes dev` exists with verbs `sync`, `status`,
`gc`; refuses (exit 2, clear message) when the tree kind is a slot
("managed install — dev commands operate on source checkouts").

**Step 2-4:** red → implement skeleton → green.

**Step 5:** Commit: `feat(dev): hermes dev subcommand skeleton`.

## Task 3.2: `dev sync` — the single provision verb (§2.9)

**Files:**
- Modify: `hermes_cli/subcommands/dev.py`
- Create: `hermes_cli/dev_sync.py` (logic, testable without argparse)
- Test: `tests/hermes_cli/test_dev_sync.py`

**Step 1 — inventory what it replaces (read these before writing code):**
- venv: `install.sh setup_venv()` + `install_deps()` tiering
- node deps: `_update_node_dependencies()` (`hermes_cli/main.py:8129`)
- web build: `_build_web_ui()` (`hermes_cli/main.py:4871`)
- desktop build + stamp: `_desktop_build_needed()` /
  `_write_desktop_build_stamp()` (`hermes_cli/main.py:5002-5124`)

**Step 2 (failing tests, stamp logic first):** generalize the desktop
content-hash stamp into `dev_sync.py::ArtifactStamp` — inputs: a set of
source globs; output: needs_build bool + write_stamp(). TDD against tmpdir
fixtures (source change → needs build; no change → no-op; missing dist →
needs build). Port the hashing from `_desktop_build_needed` — do not
reinvent it.

**Step 3:** implement `dev_sync.run(tree_root, *, watch=False, only=None)`:
1. venv: `uv venv .venv` (managed uv via `ensure_uv()`) if missing;
   `uv sync --extra all --locked` (fall back to `uv pip install -e .[all]`
   when lockfile is stale — reuse
   `_install_python_dependencies_with_optional_fallback` by extraction,
   not duplication);
2. install the release launcher into `.hermes-launcher/` (best-effort;
   stub fallback keeps working);
3. node deps: root + `ui-tui` + `web` workspaces (mirror
   `_update_node_dependencies` argv exactly, including
   `--workspaces=false` then `--workspace` pairs);
4. builds, each gated by its ArtifactStamp: tui dist, web dist, and
   desktop pack ONLY if `--desktop` or a previous desktop build exists
   (same has-desktop heuristic as `cmd_update`);
5. apply the feature ledger if present (phase 5; soft-import, skip if
   module absent);
6. delete the venv's `.launcher-ok` stamp whenever step 1 touched the
   venv (the launcher's self-check cache — task 1.2 keys it on
   pyvenv.cfg + uv.lock, but dev sync deleting it after any venv
   mutation is the belt-and-suspenders half of that contract);
7. print a summary table of built/skipped.

**Step 4:** green: `scripts/run_tests.sh tests/hermes_cli/test_dev_sync.py -q`.

**Step 5:** Commit: `feat(dev): dev sync single provision verb`.

## Task 3.3: Launch-time staleness refusal for app surfaces (§2.9)

**Files:**
- Modify: `hermes_cli/main.py` — `cmd_desktop`, `cmd_dashboard`/`serve`,
  TUI launch path
- Test: `tests/hermes_cli/test_surface_staleness.py`

**Step 1 (failing test):** in a checkout with a stale stamp,
`hermes desktop` (no flags) exits 4 with:

```
desktop build is behind the source tree (last built <rel-time>).
  run: hermes dev sync            # rebuild what changed
  or:  hermes desktop --build     # build now and launch
```

`--build` preserves today's build-then-launch. In a SLOT, no staleness
check at all (bundle artifacts are always current by construction).

**Step 2:** implement using ArtifactStamp from 3.2. Apply the same check to
the TUI (`hermes --tui`) and web (`hermes dashboard`) launch paths.

**Step 3:** green. **Step 4:** Commit:
`feat(dev): app surfaces refuse stale builds instead of surprise-building`.

## Task 3.4: Worktree-based ejected update (§2.5.2)

**Files:**
- Create: `hermes_cli/dev_update.py`
- Modify: `hermes_cli/main.py` — `_cmd_update_impl` gains an early branch:
  tree kind == checkout AND worktrees viable → route to `dev_update`;
  `--in-place` flag forces the legacy flow
- Test: `tests/hermes_cli/test_dev_update_worktree.py`

**Step 1 (failing tests, against fixture git repos):**
- clean tree, target newer → fast-forward in place (no worktree needed —
  don't create ceremony where the old flow was already safe);
- dirty tree → returns the 3-option choice (switch/merge/cancel);
  `choose="switch"` creates `.worktrees/<target>` via `git worktree add`,
  runs a (mocked) `dev sync`, re-points the PATH symlink to the new
  worktree's `bin/hermes`, and asserts the ORIGINAL tree's
  `git status --porcelain` is BYTE-IDENTICAL before/after;
- `choose="merge"` runs fetch + merge, stops on conflict exactly like git
  (no stash, no auto-resolution), exit code mirrors git's;
- worktree creation failure (e.g. exotic fs) → falls back to legacy
  autostash flow with a warning.

**Step 2:** implement. Naming: `.worktrees/v<tag>` for tags,
`.worktrees/main-<shortsha>` for branch tracking (mirror the convention
this very worktree uses). `hermes dev gc` lists version-worktrees and
removes merged/inactive ones with keep-N=2 (never the active symlink
target — check by resolving the PATH symlink).

**Step 3:** green:
`scripts/run_tests.sh tests/hermes_cli/test_dev_update_worktree.py -q`.

**Step 4:** Commit: `feat(dev): worktree-based updates for modified checkouts`.

## Task 3.4b: The cwd guard (`--dev` / `--global`, §2.5.1a)

**Files:**
- Modify: `apps/hermes-launcher/src/main.rs` (guard before exec),
  `bin/hermes` stub (same check in ~5 lines of sh)
- Test: launcher unit tests + a case in the phase E2E

**Step 1 (failing tests, pure logic):** `cwd_guard(launcher_tree, cwd,
argv) -> Run | ReExec(path) | Refuse(msg)`. STRICT rule: cwd inside any
hermes-agent checkout ⇒ a flag is required, no exceptions:

- no enclosing hermes-agent checkout above cwd → `Run` (flags accepted
  as no-ops so scripts can pin intent unconditionally);
- inside a checkout, no flag → `Refuse` with the §2.5.1a message
  (exit 2) — even when the invoked launcher IS that checkout's own;
- inside a checkout + `--dev`: launcher's own tree == the cwd checkout
  → `Run` (strip the flag); otherwise → `ReExec(<cwd
  checkout>/bin/hermes)` **retaining `--dev`** — the re-exec'd launcher
  wakes up inside the same checkout and must resolve to `Run` on its
  own pass, not refuse or re-exec forever. Assert exactly one hop in
  the tests (the retained flag is what terminates the recursion);
- inside a checkout + `--global` → `Run` with the invoked launcher
  (strip the flag);
- both flags → `Refuse` (contradictory).

Enclosing-checkout detection: walk up from cwd to the first dir with
`pyproject.toml` containing `name = "hermes-agent"` (string probe is
fine — no toml parser in the launcher); stop at filesystem root. A
worktree's `.git` FILE bounds the tree like a `.git` dir. Detection of
"which tree does this launcher belong to" is task 1.1's
`resolve_tree_root` — reuse it.

**Step 2:** wire into the launcher main before venv resolution (the
refusal must fire in milliseconds, before any Python starts) and mirror
the same check in the `bin/hermes` stub for bare clones that haven't run
`dev sync` yet.

**Step 3:** green. Guard cases in the E2E (task 3.7): from inside
checkout B, plain `hermes` refuses regardless of where the symlink
points — `--dev` runs B (one hop, even when invoked via B's own
`./bin/hermes`), `--global` runs the symlink target; from `$HOME`,
plain `hermes` runs without flags.

**Step 4:** Commit: `feat(dev): cwd guard — explicit --dev/--global
inside checkouts`.

## Task 3.5: `hermes eject`

**Files:**
- Create: `hermes_cli/subcommands/eject.py`
- Test: `tests/hermes_cli/test_eject.py`

**Step 1 (failing test):** from a SLOT install, `hermes eject
[--dir PATH]` (default `$HERMES_HOME/source`): clones the repo at the
slot's `git_sha` (from the slot manifest), runs `dev sync`, re-points the
PATH symlink to the checkout's `bin/hermes`, prints the ejected-contract
caveats (§2.5), and records `.pre-eject-target` for undo symmetry with
adopt. From a checkout: exits 0 with "already ejected" + status.

**Step 2-3:** implement → green.

**Step 4:** Commit: `feat(dev): hermes eject`.

## Task 3.6: run_tests.sh venv fallback retirement

**Files:**
- Modify: `scripts/run_tests.sh`

**Step 1:** Deprecate the third probe (`$HOME/.hermes/hermes-agent/venv`):
keep it working but print a one-line warning pointing at
`hermes dev sync` (per §2.5.1 the shared-venv fallback is skew-prone).
Removal is a phase-5 sunset item, not now.

**Step 2 (verify):** `scripts/run_tests.sh tests/hermes_cli/test_dev_cmd.py -q`
from a worktree with its own `.venv` → no warning; hide `.venv` → warning
text appears, tests still run.

**Step 3:** Commit: `chore(tests): deprecate shared-venv fallback`.

## Task 3.7: E2E gate — worktrees + switching

**Files:**
- Create: `scripts/e2e/test-ejected-worktrees.sh`

**Contract:**

```
1. temp clone of this repo; ./bin/hermes --dev --version works after
   `hermes dev sync` (stub path AND native-launcher path; --dev because
   the invocation cwd is inside the checkout).
2. Dirty the tree (append a comment to run_agent.py). Run
   hermes update, choose "switch" → .worktrees/<target> created +
   synced; symlink now points there; original tree's dirty diff
   byte-identical; `hermes --version` reports the new tree.
3. Point the symlink back at the original tree → old version again;
   dirty change still present.
4. If a slot exists from phase 1's fixture: symlink to slot launcher →
   managed version runs. All three targets coexist.
5. cwd-guard cases (task 3.4b, strict): from inside worktree B, plain
   `hermes` refuses (exit 2) no matter where the symlink points — even
   via B's own `./bin/hermes`; `hermes --dev` runs B (assert one hop);
   `hermes --global` runs the symlink target; from $HOME plain `hermes`
   runs with no flag.
6. hermes dev gc --keep 1 → old version-worktree removed; active one
   survives.
```

**Verification:** exits 0 locally; CI linux job.

**Commit:** `test(e2e): ejected worktree lifecycle gate` — **phase 3
complete.**

## Pitfalls

- `git worktree` inside a worktree: `git worktree add` from a linked
  worktree works but paths land relative to the MAIN checkout's
  `.worktrees` only if you pass an absolute path — always pass absolute.
- The stub launcher must not swallow exit codes — `exec` preserves them on
  POSIX; on Windows `.cmd`, use `call` + `exit /b %ERRORLEVEL%`.
- Do NOT delete the autostash machinery in this phase — it is the §2.5.2
  fallback AND still serves `--in-place`. Deletion decisions live in
  phase 5.
- ArtifactStamp source-globs must include lockfiles (`package-lock.json`,
  `uv.lock`) — a dep bump with no source change still needs a rebuild.
