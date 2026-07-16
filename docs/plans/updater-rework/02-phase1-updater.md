# Phase 1 — The `hermes-updater` / launcher binary, slots, and atomic flip

> **For Hermes:** subagent-driven-development, task-by-task.
> Read `docs/updater-world.md` §2.2, §2.3, §2.3.1, §2.5.1 first.

**Goal:** A small Rust binary that is BOTH the `hermes` launcher and the
updater; `install/apply/rollback/status/adopt` verbs; versioned slots with
atomic `current` flip. New installs use this path; existing installs are
untouched (that's phase 2).

**Tech:** Rust (grow from `apps/bootstrap-installer/src-tauri` — reuse its
process-wait/marker/streamed-output code, but the new crate is a plain CLI
binary, NO Tauri/GUI deps). Crate location: `apps/hermes-launcher/`.

**Definition of done:** `scripts/e2e/test-slot-lifecycle.sh` passes:
fresh install → run → apply update to v2 → rollback to v1 → all against a
local file:// release server, on a temp `$HERMES_HOME`.

---

## Task 1.0: Crate skeleton

**Files:**
- Create: `apps/hermes-launcher/Cargo.toml`, `src/main.rs`, `src/cli.rs`

**Step 1:** `cargo new --bin` layout with subcommand dispatch (clap,
derive). Binary name is `hermes`; when invoked as `hermes-updater`
(argv[0] sniff, busybox-style) the updater verbs are the default namespace.
Verbs (stubs returning `todo!` for now): `launch` (default), `install`,
`apply`, `rollback`, `status`, `adopt`, `self-restage`.

**Step 2:** `cargo build && cargo test` in `apps/hermes-launcher/` → green.

**Step 3:** Commit: `feat(launcher): crate skeleton with verb dispatch`.

## Task 1.1: Tree-root self-location + env setup (§2.5.1a) — TDD

**Files:**
- Modify: `apps/hermes-launcher/src/tree.rs` (create)
- Test: `apps/hermes-launcher/src/tree.rs` `#[cfg(test)]`

**Step 1 (failing tests first):** `resolve_tree_root(exe_path)` — walks up
from the binary's real path (symlinks resolved) to the first dir containing
one of: `current.txt` (managed root — resolve the active version via
`resolve_current`, task 1.4, and recurse into `versions/<v>/`),
`manifest.json` (slot), or `pyproject.toml` + `.git` (checkout).
Returns `TreeKind::Slot | TreeKind::Checkout` + root path (the managed
root always resolves to a Slot — the launcher never runs "from" the data
dir itself). Test with tmpdir
fixtures for: managed root (stable launcher beside `current.txt`), slot
layout, checkout layout, worktree layout (`.git` FILE
containing `gitdir:` — not a dir!), and neither (error).

**Step 2:** run tests → fail. **Step 3:** implement. **Step 4:** pass.

**Step 5:** `build_child_env(tree) -> Vec<(String,String)>` — TDD the exact
contract from §2.5.1:
- PATH: prepend `<tree>/runtime/tools` + `<tree>/runtime/node/bin` +
  `<tree>/runtime/python/bin` (slot — `runtime/tools` carries bundled
  native CLIs like `rg` so existing `shutil.which` call sites resolve
  the pinned copies, task 0.3) or `$HERMES_HOME/node/bin` +
  `$HERMES_HOME/bin` (checkout);
- set `VIRTUAL_ENV=<tree venv>`, `UV_PYTHON=<tree venv python>`,
  `UV_NO_CONFIG=1`;
- REMOVE `PYTHONPATH`, `PYTHONHOME`.
Venv path: `runtime/venv` for slots, `.venv` (fallback `venv`) for
checkouts — matching what `scripts/run_tests.sh` probes.

**Step 6:** Commit: `feat(launcher): tree resolution + env contract`.

## Task 1.2: `launch` verb

**Files:**
- Modify: `apps/hermes-launcher/src/launch.rs` (create)

**Step 1:** implement: resolve tree → build env → `exec` (unix
`execvp`-style via `std::os::unix::process::CommandExt::exec`; Windows:
spawn + wait + mirror exit code) of
`<venv python> -m hermes_cli.main <args...>`.

**Step 2: launcher self-check (§2.5.1).** Before exec, verify the venv
python exists and `-c "import hermes_cli"` exits 0. Cache the success in
a `.launcher-ok` stamp beside the venv so the probe runs once, not
per-invocation — but do NOT key it on the venv directory's mtime: a
dir's mtime does not change when files deep inside `site-packages`
change (`dev sync`, a lazy-feature install, a partial wipe), so an
mtime key happily reports "healthy" over a venv that was just churned.
Key the stamp on content that actually moves when the venv does:
sha256 of (`pyvenv.cfg` bytes + `uv.lock` bytes + interpreter path).
Slots are immutable so the stamp is effectively permanent there;
checkouts re-probe exactly when the lockfile or venv config changed.
`dev sync` (phase 3) also deletes the stamp after mutating the venv —
belt and suspenders. On failure print EXACTLY:

```
hermes: this tree's virtualenv is missing or broken.
  tree: <root>
  fix:  hermes dev sync        (source checkout)
        hermes-updater apply   (managed install)
```

and exit 3 — FAST, so a supervisor respawn loop spins on a cheap process.

**Step 3 (manual verify):** build; copy binary into this checkout as
`./target/debug/hermes`; run `./target/debug/hermes --version` from the
worktree → execs the venv and prints the python CLI's version.

**Step 4:** Commit: `feat(launcher): launch verb with venv self-check`.

## Task 1.3: Release fetching + verification

**Files:**
- Create: `apps/hermes-launcher/src/release.rs`

**Step 1 (TDD):** `ReleaseSource` supporting `https://` (GitHub Releases
API) and `file://` (E2E fixtures). Functions: `latest(channel)`,
`get(version)` → resolves the platform's
`hermes-<v>-<platform>.tar.zst` + `manifest.json` + `.minisig` URLs.

**Step 2 (TDD):** `verify(bundle_dir)` — minisign-verify manifest with the
embedded pubkey (compile-time `include_str!` of `keys/hermes-release.pub`),
then sha256 every file against `manifest.files`. Tests: tampered file →
error; tampered manifest → error; clean → ok. Generate a throwaway keypair
in tests; the pubkey is injectable for testability (`#[cfg(test)]`
constructor).

**Step 3:** Commit: `feat(launcher): release fetch + signature verification`.

## Task 1.4: `install` + `apply` — staging and the flip (§2.2)

**Files:**
- Create: `apps/hermes-launcher/src/slots.rs`, `src/apply.rs`

**Step 1 (TDD `slots.rs`):** pure slot management against an injectable
root: `stage(version)` → `versions/<v>.staging/`; `commit_staging` →
fsync + rename to `versions/<v>`; `flip(version)`; `previous` link
update; `gc(keep_n)` — never GC the targets of `current`/`previous`;
`cleanup_stale_staging()`.

**The flip (get this right — it is THE atomic commit point of the whole
design):** ONE mechanism on every platform. `current.txt` at the slots
root holds the active version string; the flip is write
`current.txt.new` + fsync + rename over `current.txt`. File
rename-over-existing is atomic everywhere (POSIX `rename()`, Windows
`MoveFileExW(MOVEFILE_REPLACE_EXISTING)`), so there is no per-platform
commit logic to diverge. Everything load-bearing resolves through one
function — `resolve_current(root) -> version` — which reads
`current.txt`. `previous.txt` is the same format; rollback is rewriting
`current.txt` from it.

Deliberately NOT a symlink/junction commit: a *directory* symlink can be
atomically renamed over on POSIX but a junction cannot on Windows
(`MOVEFILE_REPLACE_EXISTING` can't replace a directory entry;
delete-then-rename has a crash window). Rather than two commit
mechanisms with different failure modes, both platforms use the file.
`flip()` MAY refresh a `current` convenience symlink for humans and
shell tools after the commit, best-effort, where symlinks are free
(POSIX); its absence or staleness must never matter — enforce that by
making `resolve_current` the only reader in the codebase.

**Step 2 (apply flow):** download → verify → stage → **preflight** →
flip → restage self (task 1.6) → restart services (task 1.7). Preflight =
run the STAGED slot's `bin/hermes doctor --preflight` (task 1.5) and
require exit 0. On any failure before the flip: delete staging, report,
exit nonzero — `current` untouched.

**Step 3:** `rollback` verb: rewrite `current.txt` from `previous.txt`
(and swap). `status` verb: print current/previous versions, staged
leftovers, channel, `--json` flag for machine consumption.

**Step 4:** update-in-progress marker: hold
`$HERMES_HOME/.hermes-update-in-progress` (same pid+timestamp format as
`apps/desktop/electron/update-marker.ts` — keep byte-compat, the desktop
already parses it) for the flip+restart critical section ONLY.

**Step 5:** Commit: `feat(updater): install/apply/rollback with atomic flip`.

## Task 1.5: `hermes doctor --preflight` (python side)

**Files:**
- Create: `hermes_cli/subcommands/doctor_preflight.py`
- Modify: `hermes_cli/main.py` (wire subcommand arg only)
- Test: `tests/hermes_cli/test_doctor_preflight.py`

**Step 1 (failing test):** `preflight()` returns `(ok: bool, report: dict)`;
checks: core imports (`run_agent`, `model_tools`, `gateway.run`,
`hermes_cli.main` — import them for real in a subprocess), config parses
(`load_config()` doesn't raise), config version migratable
(`check_config_version()` current ≤ latest), **and artifact roots
resolve** (task 0.2b): `get_artifact_root()` succeeds and each accessor
(`bundled_skills_dir()`, `web_dist_dir()`, `tui_dist_dir()`) points at
an existing, non-empty directory — skipping any the manifest flags
absent (e.g. `"desktop": false`). This is what makes preflight catch
the non-editable-install gap: imports alone go green while every
repo-root-relative asset lookup is broken. Test against the isolated
`HERMES_HOME` fixture; assert a broken venv path is reported not raised.

**Step 2-4:** red → implement → green:
`scripts/run_tests.sh tests/hermes_cli/test_doctor_preflight.py -q`.

**Step 5:** Commit: `feat(cli): doctor --preflight for slot activation gate`.
(Also tighten `scripts/e2e/test-bundle-boot.sh` to use it — see phase 0.)

## Task 1.6: Self-restage + bootstrap hop (§2.3.1)

**Files:**
- Create: `apps/hermes-launcher/src/selfupdate.rs`

**Step 1 (TDD, pure logic):** `needs_hop(my_version, manifest.min_updater_version)`
semver compare. `hop(bundle, args)`: extract `bin/hermes` from the verified
bundle to a temp path, re-exec with original argv + `--hopped`; if
`--hopped` is already present, refuse (return error → the OLD updater
reports failure). Test the flag plumbing.

> **Contract note:** the hop hardcodes `bin/hermes` inside the bundle —
> that path is now part of the updater↔bundle contract at the same tier
> as `min_updater_version` and root-level `manifest.json`: an old staged
> updater must be able to find the new binary in ANY future bundle it is
> asked to hop into. A layout reshuffle that moves `bin/hermes` requires
> a `min_updater_version` bump AND keeping a compat copy at the old path
> for one contract window. Record this in `scripts/release/README.md`'s
> contract section (task 0.0).

**Step 2:** `self-restage`: POSIX = write `bin/.hermes-updater.new` +
rename over `bin/hermes-updater`; Windows = rename running exe to
`hermes-updater.old.exe`, move new into place, best-effort delete `.old`
(and sweep `*.old.exe` at every updater start). Restage failure after a
successful flip is a WARNING, not an error (§2.3.1 failure containment).

**Step 3:** Wire into apply: hop check happens AFTER verify, BEFORE stage.
Restage happens AFTER flip.

**Step 4:** Commit: `feat(updater): self-update via bootstrap hop + restage`.

## Task 1.7: Service restart hooks (§2.4)

**Files:**
- Create: `apps/hermes-launcher/src/services.rs`
- Modify: `gateway/run.py` (add SIGUSR2 → drain-then-exit-75 if not already
  wired; CHECK FIRST: `search_files 'SIGUSR1' gateway/` — a restart signal
  path already exists, extend it rather than adding a parallel one)

**Step 1:** After a flip, the updater signals running gateways: read pids
from the existing gateway status file (`gateway/status.py` — reuse its
discovery, don't invent a new pidfile), send the drain signal, and let the
supervisor (systemd/s6/desktop) respawn on exit 75. If no supervisor is
detected (status says "foreground terminal"), just print "restart your
gateway to pick up <v>".

**Step 2:** Notification plumbing: when invoked by gateway `/update`
(`--notify-file <path>`), write the same `.update_exit_code` /
`.update_output.txt` files `gateway/run.py`'s watcher already polls —
byte-compatible, zero gateway changes needed for phase 1.

**Step 3:** Commit: `feat(updater): gateway drain + restart after flip`.

## Task 1.8: Thin `install.sh` path for new installs

**Objective:** New installs can opt into the bundle world; default remains
legacy until phase 2 sign-off.

**Files:**
- Modify: `scripts/install.sh` (additive flag only)

**Step 1:** Add `--bundle` flag: skip clone/venv/deps entirely; download
the platform `hermes-updater` from the release, verify (sha256 published in
the release), run `hermes-updater install --channel stable`, symlink
`$(get_command_link_dir)/hermes -> $HERMES_HOME/bin/hermes` (the stable
launcher, which resolves `current.txt`).

**Step 2:** Manual verify on a scratch user/container:
`bash scripts/install.sh --bundle --skip-setup` → `hermes --version` works,
`versions/<v>/` + `current` exist, no venv build ran.

**Step 3:** Commit: `feat(install): --bundle fast path via hermes-updater`.

## Task 1.9: E2E gate — slot lifecycle

**Files:**
- Create: `scripts/e2e/test-slot-lifecycle.sh`

**Contract:** temp `HERMES_HOME`; local `file://` release dir with two
bundle versions (built by `scripts/release/build-bundle.sh`, second one
with a bumped version string):

```
1. hermes-updater install --source file://$FIXTURE --channel stable
   → versions/v1, current.txt says v1;  bin: hermes --version == v1
2. Start `hermes serve` (background), record pid.
3. hermes-updater apply --source file://$FIXTURE (v2 available)
   → current.txt says v2, previous.txt says v1; the OLD serve process is
     still alive and still running v1 code (assert via its /api status
     endpoint) until restart — proving no in-place mutation.
4. Restart serve → reports v2.
5. hermes-updater rollback → current.txt says v1; hermes --version == v1.
6. Corrupt one file in a staged v3 → apply FAILS pre-flip; current.txt
   still says v1. (tamper test)
7. kill -9 the updater mid-download (SIGKILL during a large fixture)
   → re-run apply succeeds; `.staging` was cleaned. (interrupt test)
```

**Verification:** script exits 0 locally AND is added to CI (linux-x64).

**Commit:** `test(e2e): slot lifecycle gate` — **phase 1 complete.**

## Pitfalls

- **The flip is the `current.txt` file replace on every platform** — do
  not introduce a symlink/junction commit path anywhere, even where it
  would work (task 1.4). One mechanism, one `resolve_current` reader.
  Still test on a real Windows runner — WSL will lie to you.
- **Do not add Tauri deps** to the launcher crate; it must build in seconds
  and produce a <5MB static-ish binary (musl target for linux).
- The marker file format must stay byte-compatible with
  `update-marker.ts::readLiveUpdateMarker` — the desktop app parses it TODAY.
- exec on POSIX must preserve argv[0] semantics; `hermes --tui` etc. are
  pass-through args, do not parse python-side args in the launcher.
