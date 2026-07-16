# Phase 2 — Compat Fence + Legacy Adoption Funnel

> **For Hermes:** subagent-driven-development, task-by-task.
> Read `docs/updater-world.md` §2.13 first — this phase IS §2.13.

**Goal:** (a) freeze and CI-enforce the contract that keeps `main` a valid
update target for every historical updater (hop 1), (b) ship the adoption
detector + `adopt` flow that migrates legacy installs to slots (hops 2-3).

**Definition of done:** `scripts/e2e/test-adoption.sh` passes: a REAL legacy
install created from an old release tag updates itself to current main via
its own `hermes update`, then adopts to slots on next launch, with rollback
by one symlink re-point.

**⚠ Task 2.1 should land as early as possible — even during phase 0/1 —
because every day without the fence is a day a refactor can silently break
hop 1.**

---

## Task 2.1: Enumerate the frozen contract

**Objective:** Find every symbol old updaters touch post-pull; write them
down as an explicit registry.

**Files:**
- Create: `hermes_cli/updater_compat.py`

**Step 1 — archaeology (do not skip, do not guess):** For each entry below,
confirm by reading the code; the design doc's list (§2.13) is the starting
set, NOT the finished one. Method: `git log --follow -p` over
`_cmd_update_impl` back ~18 months, noting every `from X import Y` executed
AFTER the `git pull` line, plus every subprocess the old code launches.
Known set to verify:

- `hermes_cli.managed_uv.ensure_uv` (dual-shape `_UvResult` contract)
- `hermes_cli.managed_uv.update_managed_uv`
- `hermes_cli.managed_uv.rebuild_venv` (tombstone — Appendix B)
- `hermes_cli.main._install_python_dependencies_with_optional_fallback`
- `tools.skills_sync.sync_skills`
- `tools.lazy_deps.active_features` / `refresh_active_features`
- `hermes_cli.profiles.list_profiles` / `seed_profile_skills` /
  `backfill_profile_envs`
- `hermes_cli.model_catalog.seed_cache_from_checkout`
- `hermes_cli.config.get_missing_env_vars` / `get_missing_config_fields` /
  `check_config_version` / `migrate_config`
- `hermes_constants` attributes read post-reload:
  `find_node_executable`, `with_hermes_node_path`, `display_hermes_home`
- CLI surface old updaters shell out to: `hermes desktop --build-only`,
  `hermes update --gateway` argv shape, `pyproject.toml` editable-install
  with `[all]` extra, `constraints-termux.txt` existing.

**Step 2:** Write the registry:

```python
"""Symbols frozen for legacy-updater compatibility. See docs/updater-world.md §2.13.

Every entry here is imported/called by SOME historical `hermes update`
after it has pulled current code. Changing a signature or deleting an
entry bricks that population's next update. Guarded by
tests/test_updater_compat_fence.py. Sunset: see docs/plans/updater-rework/
06-phase5-ledger-and-sunset.md.
"""
FROZEN_CALLABLES: dict[str, str] = {
    # "module:qualname" -> frozen signature string (inspect.signature format)
    "hermes_cli.managed_uv:ensure_uv": "()",
    "hermes_cli.managed_uv:rebuild_venv": "(uv_bin: str, venv_dir: pathlib.Path, python_version: str = '3.11') -> bool",
    # ... every verified entry from step 1
}
FROZEN_CLI_SURFACES: list[list[str]] = [
    ["update", "--gateway"],
    ["update", "--yes", "--gateway", "--force", "--branch"],
    ["desktop", "--build-only"],
]
```

**Step 3:** Commit: `feat(compat): frozen legacy-updater contract registry`.

## Task 2.2: The CI fence — TDD

**Files:**
- Create: `tests/test_updater_compat_fence.py`

**Step 1 (the test IS the deliverable):** for every `FROZEN_CALLABLES`
entry: import the module, resolve the qualname, assert it exists and
`str(inspect.signature(fn))` equals the frozen string. For
`FROZEN_CLI_SURFACES`: build the argparse parser
(`hermes_cli.main.build_parser()` or equivalent — find the actual
constructor with `search_files 'add_parser' hermes_cli/`) and assert
`parse_args` accepts each argv without SystemExit. This is a *behavior
contract* test, explicitly exempt from the change-detector rule — say so in
the module docstring with a link to §2.13.

**Step 2:** `scripts/run_tests.sh tests/test_updater_compat_fence.py -q` →
pass on current main (if anything fails HERE, a compat break already
shipped — escalate to the maintainer immediately, do not "fix the test").

> **Known limitation (state it in the module docstring):** the fence
> freezes signatures *as they exist on current main*. If a frozen symbol
> already drifted between some historical release and today, the fence
> enshrines the drifted shape and the fence alone won't notice. The fence
> is necessary-not-sufficient: it stops FUTURE drift. The authority on
> whether hop 1 actually works for a given vintage is task 2.8's E2E
> (step 3), which runs a real old release against current main — when the
> E2E and the fence disagree, the E2E wins and the fence gets corrected.

**Step 3:** Add a dedicated CI job name (`updater-compat-fence`) so a
failure is legible in PR checks, not buried.

**Step 4:** Commit: `test(compat): CI fence for legacy-updater contract`.

## Task 2.3: Legacy-layout detection — TDD

**Files:**
- Create: `hermes_cli/adoption.py`
- Test: `tests/hermes_cli/test_adoption_detect.py`

**Step 1 (failing tests):** `detect_legacy_install(project_root, hermes_home)
-> LegacyInfo | None` returns None when: running from a slot (root under
`versions/`), docker/nixos/homebrew (`detect_install_method()` result),
pip installs. Returns `LegacyInfo(pristine: bool, reasons: list[str])`
for git checkouts, where pristine == clean tree AND official origin
(reuse `canonicalGitHubRemote` logic — port the check from
`apps/desktop/electron/update-remote.ts` into python, TDD it) AND on
main/known branch AND no local commits ahead of origin. Fixture-based
tests: temp git repos for each cohort in §2.13's table.

**Step 2-4:** red → implement → green.

**Step 5:** Commit: `feat(adoption): legacy install detection + cohorts`.

## Task 2.4: Adoption offer at launch (hop 2)

**Files:**
- Modify: `hermes_cli/main.py` — top of `main()`, BEFORE heavy imports
- Modify: `hermes_cli/config.py` — `DEFAULT_CONFIG["updates"]["adopt"] =
  "prompt"` (`auto|prompt|never`); NO config version bump needed (new key
  in existing section deep-merges)
- Test: `tests/hermes_cli/test_adoption_offer.py`

**Step 1 (failing test):** with a legacy fixture + `adopt: prompt` + a TTY,
the offer text is printed once per N days (snooze stamp in
`$HERMES_HOME/state/adoption-snooze`); with `never`, silent; with `auto` +
pristine + non-interactive, it invokes the adopt callable. The offer NEVER
raises — any internal error logs and continues to normal startup
(crash-proof detector, §2.13).

**Step 2:** Implementation notes: the detector must run before anything
that could trip on a stale venv; guard the whole call in
`try/except Exception`. Interactive prompt copy:

```
⚕ Hermes can switch this install to managed releases (faster, atomic,
  rollbackable updates — no local building). Your current checkout is
  kept untouched as a fallback.
    hermes adopt         # switch now
    hermes adopt --help  # details
  (configure: updates.adopt = auto|prompt|never)
```

**Step 3-4:** red → green →
`scripts/run_tests.sh tests/hermes_cli/test_adoption_offer.py -q`.

**Step 5:** Commit: `feat(adoption): launch-time offer (hop 2)`.

## Task 2.5: `hermes adopt` — fetch updater, exec, exit (hop 3, python side)

**Files:**
- Create: `hermes_cli/subcommands/adopt.py`
- Test: `tests/hermes_cli/test_adopt_cmd.py`

**Step 1 (failing test):** `cmd_adopt` (a) refuses for
docker/nix/brew/pip with the existing recommended-command messages,
(b) for dirty/fork cohorts prints the eject-vs-adopt choice and requires
explicit `--yes-dirty` to proceed, (c) downloads the platform
`hermes-updater` to `$HERMES_HOME/bin/` (sha256-verified against the
release's published checksums; the EXPECTED hash list ships in the python
package, updated by the release workflow), (d) `os.execv`s it with
`["hermes-updater", "adopt", "--from-checkout", PROJECT_ROOT]` — python
NEVER returns (assert via a monkeypatched execv). `cmd_adopt` accepts
`--source <url>` (https:// or file://) and forwards it to the updater's
existing `--source` flag — this is how tests and the E2E gate inject the
fixture release server. No `HERMES_UPDATER_SOURCE` env var: ground rule
3 (no new `HERMES_*` env vars for non-secret config) applies to test
plumbing too, and the updater already speaks `--source`.

**Step 2-4:** red → implement → green.

**Step 5:** Commit: `feat(adoption): hermes adopt hands off and exits`.

## Task 2.6: `adopt` verb in the updater (hop 3, rust side)

**Files:**
- Modify: `apps/hermes-launcher/src/apply.rs` (+ `src/adopt.rs`)

**Step 1:** `adopt --from-checkout <path>`:
1. read the checkout's version/sha (for choosing the matching or newest
   bundle);
2. normal apply pipeline: download → verify → stage → preflight → flip;
3. re-point the PATH symlink: find the existing `hermes` link
   (`get_command_link_dir()` equivalents: `~/.local/bin`, `/usr/local/bin`,
   Termux prefix — port the resolution) and re-target it to
   `$HERMES_HOME/bin/hermes` (the stable launcher — it resolves
   `current.txt` per task 1.4); keep a `.pre-adopt-target` note file
   recording the old target for one-command undo;
4. DO NOT touch the checkout (assert in E2E: tree hash identical before/
   after);
5. migrate data-dir state: nothing to move (config/skills/sessions already
   live in `$HERMES_HOME`), but seed `state/features.json` from a venv
   probe of the OLD checkout's venv (phase 5 task 5.1 ships the ledger;
   if phase 5 hasn't landed, emit the probe result to
   `state/features.pending.json` for it to consume);
6. `adopt --undo`: re-point the symlink at `.pre-adopt-target`, print
   confirmation.

**Step 2:** Commit: `feat(updater): adopt verb (hop 3)`.

## Task 2.7: Gateway + desktop adoption arms

**Files:**
- Modify: `gateway/slash_commands.py` — `/update` gains detection: when
  `detect_legacy_install()` returns pristine, append one line to the
  update-complete notification: "This install can switch to managed
  releases — run `hermes adopt` or reply `/update adopt`." `/update adopt`
  spawns the detached adopt exactly like the existing detached update
  (reuse the setsid/helper machinery at `gateway/slash_commands.py:4566+`).
- Modify: `apps/desktop/src/store/updates.ts` + overlay — when backend
  `session.info` reports legacy+pristine (add the field to the existing
  runtime-info payload), show the adopt offer in the updates overlay.
  Desktop APPLY path is phase 4; here it's offer + copyable command only.

**Verification:** `scripts/run_tests.sh tests/gateway/test_update_command.py -q`
still green; new tests for the adopt arm beside it.

**Commit:** `feat(adoption): gateway and desktop offer surfaces`.

## Task 2.8: E2E gate — the full funnel

**Files:**
- Create: `scripts/e2e/test-adoption.sh`

**Contract (this is the phase's reason to exist — build it carefully):**

```
FIXTURE: pick a REAL past release tag OLD_TAG (maintainer supplies one
known-good, e.g. 6 months old). The script takes OLD_TAG as a parameter
so CI can run it as a matrix — start with one tag, and grow the matrix
toward the oldest still-in-the-wild vintage (support-channel signal):
one tag proves one cohort, but the population is arbitrary-age updaters,
and ancient updaters touch symbols newer ones don't.

1. Legacy install: git clone at OLD_TAG into temp HERMES_HOME/hermes-agent,
   create venv the way that era's install.sh did (run THAT TAG's
   scripts/install.sh --skip-setup --skip-browser against the temp home).
2. Point its origin at the CURRENT repo (file:// remote of this checkout)
   so "origin/main" is today's code.
3. Run the OLD tree's own updater: venv/bin/hermes update --yes
   → must exit 0 (retry once permitted, mirroring Tauri behavior).
   THIS LINE IS THE HOP-1 PROOF. If it fails, the compat fence has a hole:
   identify the symbol from the traceback, add it to updater_compat, fix.
4. Next launch: venv/bin/hermes --version with updates.adopt=prompt
   → adoption offer text appears exactly once.
5. venv/bin/hermes adopt --yes --source file://$BUNDLE_FIXTURE
   (the --source flag from task 2.5 — no env var; ground rule 3)
   → versions/<v>/ + current exist; PATH symlink re-pointed;
     checkout tree-hash unchanged; `hermes --version` == bundle version.
6. hermes-updater adopt --undo → symlink back; old venv hermes still runs.
```

**Verification:** exits 0 locally; add to CI as a nightly job (it is slow —
full legacy install — so nightly, not per-PR).

**Commit:** `test(e2e): legacy adoption funnel gate` — **phase 2 complete.**

## Pitfalls

- Step 3 of the E2E is the single most valuable test in this whole project.
  When it breaks, the fix is ALWAYS "widen updater_compat", never "patch
  the old tag" (you cannot — it's already on user machines).
- The adoption detector runs on EVERY launch forever (until sunset) — it
  must be <5ms in the common no-op case. Stat `versions/` + one marker
  before doing anything else.
- Do not bump `_config_version` for the new `updates.adopt` key (deep-merge
  handles new keys; see AGENTS.md "Adding Configuration").
- `/update adopt` must respect the same platform allow-list as `/update`
  (`_UPDATE_ALLOWED_PLATFORMS` + plugin `allow_update_command`).
