# The Hermes Updater World: Inventory & Redesign Proposal

> Status: design document, no code changes. Written against `ethie/updater-rework`
> (main @ 29b8cacfa, July 2026).
>
> Part 1 is an archaeology dig through every install/update path that exists
> today. Part 2 proposes a redesigned world built around a **pre-bundled
> release artifact** for normal users and an explicit **"ejected" source mode**
> for people who modify the code.

---

# Part 1 — Inventory: how it all works today

## 1.1 The cast of characters

There are **five** distinct installer/updater surfaces, plus three
"distribution methods" that opt out of all of them:

| Surface | Entry point | Platforms | What it owns |
|---|---|---|---|
| Bootstrap shell installer | `scripts/install.sh` (~3,100 lines) | Linux, macOS, Termux | clone repo, uv, Python, Node, venv, deps, PATH shim, config seeding, optional desktop build |
| Bootstrap PowerShell installer | `scripts/install.ps1` (~3,300 lines) | Windows | same, plus PortableGit, portable Node zip, longpath handling, ZIP fallbacks |
| CLI self-update | `hermes update` → `cmd_update` in `hermes_cli/main.py` (~1,000 lines + ~15 helpers) | all | git fetch/pull/reset, venv dep sync, node deps, web UI build, desktop rebuild, skills sync, config migration |
| Gateway `/update` | `gateway/slash_commands.py` (~line 4505) | messaging platforms | detached `hermes update --gateway` with file-based IPC + restart notification markers |
| Desktop self-update | `apps/desktop/electron/main.ts` (`checkUpdates`/`applyUpdates`) + Tauri `apps/bootstrap-installer` | Win/mac/Linux | update detection UI, handoff to `hermes-setup --update` (Windows) or in-app `hermes update` + rebuild + relaunch (POSIX) |

Opt-out distributions detected by `detect_install_method()`
(`hermes_cli/config.py:386`): **docker** (told to `docker pull`), **nixos** /
**homebrew** ("managed mode", update blocked with a pointer at the package
manager), **pip / uv tool / pipx** (deprecated; `_cmd_update_pip` shells out to
the right upgrader). The install method is recorded as a **code-scoped stamp**
`<install tree>/.install_method` — deliberately not in `$HERMES_HOME`, because
a container and a host install can share one data dir (#34397).

## 1.2 The install layout (what "an install" is)

A managed install today is a **git checkout in editable-install mode**:

```
$HERMES_HOME/                      # data: config.yaml, .env, skills/, sessions/, logs/
├── bin/uv                         # Hermes-owned uv (managed_uv.py — single canonical path)
├── node/                          # Hermes-managed Node 22 LTS tarball extract (POSIX)
│   └── etc/npmrc                  # prefix redirect so `npm i -g` lands on PATH
├── hermes-setup[.exe]             # staged Tauri updater binary (self-copied by installer)
├── .hermes-update-in-progress     # mutual-exclusion marker (pid + started_at)
├── .update_pending.json / .update_output.txt / .update_exit_code   # gateway /update IPC
└── hermes-agent/                  # INSTALL_DIR — full git clone (shallow, single-branch)
    ├── venv/                      # uv-created Python 3.11 venv, `pip install -e .[all]`
    ├── .install_method            # "git"
    ├── node_modules/              # root workspace deps (browser tools)
    ├── ui-tui/, web/              # workspaces installed during update
    └── apps/desktop/release/<plat>-unpacked/   # locally-built Electron app
```

Variants: root-on-Linux uses FHS (`/usr/local/lib/hermes-agent` +
`/usr/local/bin/hermes`, uv Python under `/usr/local/share/uv`); Termux keeps
everything in `$HERMES_HOME` and uses pkg-provided python/node + pip instead
of uv. The `hermes` command is a **generated bash/cmd shim** that unsets
`PYTHONPATH`/`PYTHONHOME` and execs `venv/bin/hermes` (a symlink was a footgun
— #21454, the shim used to overwrite the venv entrypoint through the link).

**Key property: every user machine is a build machine.** There is no binary
release of the agent. Every install compiles its own venv from source (with a
`uv sync --locked` hash-verified tier, then progressively less-safe PyPI
fallback tiers), builds the TUI, builds the web dashboard, and — if the user
has the desktop app — runs a full local Electron/Vite build via
`electron-builder --dir`.

## 1.3 Native dependency bundling today

Three different strategies for three deps, each with its own drift:

- **uv** — the cleanest story. `hermes_cli/managed_uv.py` declares ONE path
  (`$HERMES_HOME/bin/uv`), `install.sh`/`install.ps1` install to the same
  place via the astral installer's `UV_UNMANAGED_INSTALL`/`UV_INSTALL_DIR`,
  and `hermes update` runs `uv self update` each run. But even here the
  update-boundary problem left scar tissue: `_UvResult` is a `str` subclass
  whose `__iter__` yields a 2-tuple so that *old in-memory code* calling
  `uv_bin, fresh = ensure_uv()` against *freshly-pulled* module code doesn't
  crash — and it must NOT be returned on Windows because
  `subprocess.list2cmdline` iterates argv strings char-by-char (#39780/#39820).
  A whole polymorphic-return hack exists because the updater runs old code
  against new code in one process.
- **Node** — installed to `$HERMES_HOME/node/` by the installers (tarball from
  nodejs.org on POSIX, portable zip on Windows), with symlinks into the command
  link dir and an npm `prefix` redirect so `npm install -g` binaries land on
  PATH and survive Node upgrades. Runtime resolution is
  `hermes_constants.find_node_executable()`: prefer the managed tree, attempt
  self-heal if broken, and **refuse to fall back to system npm when a managed
  tree exists** (so a broken managed Node can't silently become "use whatever's
  on PATH"). Version floor logic (`node_satisfies_build`, Vite 8 needs
  `^20.19 || >=22.12`) is duplicated between install.sh, install.ps1, and the
  desktop build path.
- **Python** — delegated to uv (`uv python install 3.11`), except Termux (pkg
  python + patched psutil sdist) and root FHS (world-readable
  `UV_PYTHON_INSTALL_DIR` — #21457). `UV_PYTHON` must be re-pinned in every
  stage process because stages don't share env (an inherited `UV_PYTHON=3.14`
  used to silently rebuild the venv on the wrong Python and die in maturin).

Plus the second ring: Playwright Chromium (with its own apt-release-newer-
than-Playwright-knows hang detection, #35166, and Snap-Chromium override
stripping), ripgrep, ffmpeg, PortableGit on Windows, build-essential on
Debian. Each with per-distro prompts, sudo negotiation, and manual-install
hints.

## 1.4 `hermes update` — anatomy of the CLI apply path

`_cmd_update_impl` is the heart. Roughly in order:

1. **Preflight**: deprecation warning for pip/brew; hard block for managed
   (nix/brew) and docker; `--check` short-circuits to a fetch-and-count.
2. **Windows process guards** (two!): `_detect_concurrent_hermes_instances`
   (another hermes.exe holding the venv shim → exit 2) and
   `_detect_venv_python_processes` (anything running the venv interpreter
   keeps `.pyd` files locked; mutating the venv under it half-bricks the
   install). Bypasses: `--force` skips the first, `--force-venv` the second.
   The gateway is paused/resumed around the whole update
   (`_pause_windows_gateways_for_update`).
3. **Hangup protection**: SIGHUP ignored, stdout/stderr wrapped in
   `_UpdateOutputStream` that mirrors to `~/.hermes/logs/update.log` and
   swallows `BrokenPipeError` so an SSH disconnect can't kill a half-applied
   update.
4. **Git dance**: discard npm lockfile churn → detect fork (with upstream
   sync) → scoped single-branch fetch (the repo has thousands of
   auto-generated branches; a bare fetch stalls for minutes) → autostash
   local changes → checkout target branch → `pull --ff-only`, falling back to
   `reset --hard origin/<branch>` on divergence → **post-pull syntax guard**
   (py_compile critical files; roll back to the pre-pull SHA if a bad commit
   snuck through CI — the #28452 "orphan conflict markers bricked everyone for
   7 minutes" incident) → stash restore/discard per
   `updates.non_interactive_local_changes`.
5. **Venv sync**: write `.update-incomplete` breadcrumb → `uv self update` →
   `_install_python_dependencies_with_optional_fallback` (the tiered
   `.[all]` → minus-broken → core-only ladder again, this time at update
   time) → verify core imports → clear breadcrumb. Even the "already up to
   date" path probes venv health and repairs, because a previously-interrupted
   sync leaves a current checkout with a broken venv ("Already up to date!"
   would gaslight the user).
6. **JS builds**: `_update_node_dependencies` (root + ui-tui + web workspaces,
   desktop deliberately excluded), `_build_web_ui`, then a conditional
   `hermes desktop --build-only` **only if** a packaged desktop app already
   exists (content-hash stamp makes it a no-op when nothing changed; retried
   once; output captured to update.log).
7. **Data syncs**: bytecode cache purge (stale `.pyc` → ImportError on gateway
   restart), model-catalog cache seed from checkout, `importlib.reload` of
   `hermes_constants` (!), bundled-skills sync to every profile, profile .env
   backfill, honcho profile sync, config migration
   (silent version-bump vs. interactive new-options flow).

The recurring theme: **the updater is a long-lived process running pre-update
code that mutates the code underneath itself**, and half of the machinery
(marker files, `_UvResult`, module reloads, retry-once in the Tauri wrapper,
`gateway/code_skew.py` refusing risky lazy imports after a hot pull) exists
purely to survive that self-surgery.

## 1.5 Gateway `/update` and restarts

- `/update` from a messaging platform spawns `hermes update --gateway`
  **detached** (`setsid` on POSIX; an inline Python watcher helper on
  Windows) so it survives the gateway restart the update itself triggers.
  IPC is file-based: `.update_pending.json` (who to notify),
  `.update_output.txt` (streamed to the chat), `.update_exit_code`, plus a
  prompt/answer file pair so interactive questions (stash restore, config
  migration) can be forwarded to the messenger.
- Restart contract: exit code **75** (`GATEWAY_SERVICE_RESTART_EXIT_CODE`,
  EX_TEMPFAIL) asks the supervisor to restart after a graceful drain; exit
  **78** (EX_CONFIG) means permanent config failure — s6 translates it to
  "stop restarting" (#51228). On boot the gateway checks the pending-update
  markers and either notifies "update done" or schedules a watcher until the
  detached updater finishes.
- `gateway/code_skew.py` snapshots the checkout revision at boot; risky
  paths (e.g. `/model`) compare and refuse with "restart the gateway" instead
  of crashing on a stale-`sys.modules` lazy import.

## 1.6 Desktop app: bootstrap, update, rebuild

Three intertwined mechanisms:

**First-launch bootstrap** (`electron/bootstrap-runner.ts`): the desktop app
itself drives `install.ps1`/`install.sh` **stage-by-stage** using the
`--manifest` / `--stage <name> --json` protocol (stages: prerequisites,
repository, venv, python-deps, node-deps, path, config, setup, gateway,
[desktop], complete). Stage scripts are resolved pinned-commit-first from
GitHub with local fallbacks. Stages needing user input are skipped
(`-NonInteractive`) and replaced by the in-app onboarding. Each stage is a
separate process — which is why env pinning (UV_PYTHON, PATH for managed Node)
has to be re-derived inside every stage function.

**Update detection** (`checkUpdates` in main.ts): git-based, against
`resolveUpdateRoot()`. Refuses on non-git checkouts ("desktop self-update only
runs against a source install"). Passive checks against an SSH official remote
are rewritten to an HTTPS `ls-remote` so a FIDO2-backed SSH key doesn't get a
mystery hardware-touch prompt (`update-remote.ts`). Shallow clones skip
`rev-list --count` (walking thousands of remote commits, #51922) and degrade
to SHA compare. The renderer store (`store/updates.ts`) adds time-based toast
snoozing (~100 commits/day made per-SHA dismissal useless) and tracks
`client` vs `backend` (remote gateway) targets separately, plus a
`REQUIRED_BACKEND_CONTRACT` skew warning.

**Apply**:
- **Windows**: hand off to the staged Tauri binary
  (`$HERMES_HOME/hermes-setup.exe`, self-copied at install). The app writes
  the update marker with the spawned updater's PID *before* quitting (the
  Tauri window takes seconds to boot; without the pre-write a relaunched
  desktop spawns a backend that re-locks `.pyd` files and bricks the venv
  sync — `update-marker.ts` docstring). The Rust side
  (`bootstrap-installer/src-tauri/src/update.rs`) then: waits for install
  locks to free (bounded, then force-kills stragglers) → `hermes update
  --yes --gateway --force --branch <pin>` **retried once** (the
  update-boundary crash class: first run executes old in-memory code against
  new on-disk modules) → `hermes desktop --build-only` retried once → launch.
- **POSIX in-app** (`applyUpdatesPosixInApp`): no staged updater; run
  `hermes update --yes` + `desktop --build-only` as children, passing
  `HERMES_DESKTOP_CHILD_PID` so the updater's stale-backend reaper spares the
  app's own backend (#37532). Then the honesty ladder (`update-relaunch.ts`):
  relaunch in place only if the running binary IS under the rebuilt
  `release/<plat>-unpacked` **and** the chrome-sandbox helper passes a
  root+setuid preflight (or the user opted out via
  `ELECTRON_DISABLE_SANDBOX`/`--no-sandbox`); otherwise `guiSkew` (AppImage/
  .deb/.rpm shell was NOT replaced — telling the user "new version next
  launch" would be a lie, #37541) or `manual` (don't quit into a dead app).
  Relaunch itself is a detached self-deleting bash watcher that waits for the
  parent PID, SIGKILLs after 30s, restores cwd/env/filtered-argv, and execs.

**Backend spawn skew**: the desktop spawns `hermes serve`, falling back to
`dashboard --no-open` when the resolved runtime predates `serve` — detected by
**regexing the installed `dashboard.py` source** (`sourceDeclaresServe`).
That's the level of contortion the "everything is a source checkout of
arbitrary age" world forces.

## 1.7 Failure modes the current design has accumulated (a partial list)

Every one of these is a real, fixed incident with a guard now permanently in
the codebase:

1. Update-boundary module skew (old code in memory, new code on disk):
   `_UvResult`, retry-once in Tauri + rebuild, `importlib.reload`,
   `code_skew.py`, bytecode purge.
2. Windows file locking: two process guards, gateway pause/resume, marker
   files, lock polling + straggler kill, `--force`/`--force-venv`, ZIP
   fallback when git I/O is broken by AV, `windows.appendAtomically=false`.
3. Dirty-tree management: npm lockfile churn discard (in three places),
   autostash + restore prompt + `non_interactive_local_changes`, unmerged-
   index reset (#4735), no-initial-commit checkout quarantine (#40998).
4. Bad code shipped: post-pull syntax guard + auto-rollback (#28452).
5. Interrupted updates: `.update-incomplete` breadcrumb + launch-time
   recovery, venv-health probe on "already up to date", broken-clone
   move-aside.
6. Environment leakage: PYTHONPATH/PYTHONHOME unsetting (installer, shim, uv
   env), UV_NO_CONFIG (#21269), UV_PYTHON pinning per stage, npm prefix
   redirect.
7. Download flakiness: Electron mirror fallback ladder + wall-clock timeout
   watchdog (#39219, #47266), Node tar.xz→tar.gz fallback, uv two-stage
   download, Playwright unrecognized-distro handling (#35166).
8. GUI/backend honesty: guiSkew terminal state, sandbox preflight,
   `REQUIRED_BACKEND_CONTRACT`, `sourceDeclaresServe` source-sniffing.

None of these guards are wrong individually — they're all correct responses
to real breakage. Collectively they are the cost of the architecture: **N
mutable source installs × M platforms × in-place self-surgery, all building
locally**. The complexity is superlinear because every guard has to exist in
every updater surface (install.sh, install.ps1, cmd_update, Tauri, Electron
in-app), and they drift.

---

# Part 2 — A better updater world

## 2.0 Design goals

1. **One artifact, not one build per machine.** Normal users should download
   bytes that CI already built and tested — never run pip resolution, npm
   install, or electron-builder locally.
2. **Two explicit modes with an explicit switch between them**:
   - **Managed (un-ejected)**: pre-bundled release from GitHub Releases.
     Atomic, versioned, rollbackable, no git, no compilers.
   - **Ejected (source)**: a real development checkout. Full power, explicit
     caveats, and the updater degrades gracefully into "you own this now"
     with helpful tooling instead of pretending it's still managed.
3. **Atomic switch, not in-place mutation.** New version installs *next to*
   the old one; atomically replacing a `current.txt` indirection file is
   the commit point — the same mechanism on every platform (§2.2). The
   running process never modifies the code it is executing.
4. **One dependency manifest** consumed by every surface, instead of version
   floors and URLs copy-pasted across install.sh / install.ps1 / main.ts /
   Rust.
5. **The updater is a tiny, separately-shipped program** — not 1,000 lines
   inside the thing being updated.

## 2.1 The release artifact

CI builds, per platform/arch (`linux-x64`, `linux-arm64`, `darwin-arm64`,
`darwin-x64`, `win-x64`), a **self-contained bundle** published on the
NousResearch GitHub Releases page:

```
hermes-<version>-<platform>.tar.zst          # or .zip on Windows
├── manifest.json          # version, git sha, per-file hashes, min-updater version
├── runtime/
│   ├── python/            # uv-managed relocatable CPython (uv already ships these!)
│   ├── venv/              # fully resolved site-packages, built in CI from uv.lock
│   └── node/              # Node LTS runtime (only what browser tools need at runtime)
├── app/                   # the hermes-agent source tree (no .git), pyc-precompiled
├── ui/
│   ├── tui/dist/          # pre-built Ink bundle
│   └── web/dist/          # pre-built dashboard SPA
├── desktop/               # pre-built electron app (asar + unpacked), signed on mac/win
└── bin/hermes             # launcher shim (static, sets env, execs runtime/venv python)
```

Key points:

- **The venv is built in CI** from `uv.lock` with `--locked` — the
  hash-verified Tier 0 becomes the *only* tier for managed users. The
  multi-tier PyPI fallback ladder, the broken-extras list, the Termux psutil
  shim… all become CI problems solved once per release instead of per-user
  install-time gambles. uv's relocatable python + `uv venv --relocatable`
  makes the venv path-independent enough to unpack anywhere.
- **Node ships inside the bundle** for runtime use (agent-browser, etc.).
  npm/electron-builder are *not needed on user machines at all* in managed
  mode — the desktop app and TUI arrive pre-built.
- **Optional heavy extras** (playwright chromium, whisper models) stay
  download-on-demand, but driven by the manifest (pinned URLs + hashes) via
  `hermes deps ensure <name>`, replacing the per-installer bespoke logic.
- Release cadence: this repo lands ~100 commits/day, so tagged releases
  (say, nightly `hermes-nightly` + weekly promoted `stable`) replace
  "track main by SHA". The desktop update pill counts releases behind, not
  commits behind — which also fixes the toast-spam problem the snooze
  timers currently paper over.

## 2.2 On-disk layout: versioned slots + atomic flip

```
$HERMES_HOME/
├── versions/
│   ├── 1.42.0/            # unpacked bundle (immutable after verify)
│   └── 1.43.0/
├── current.txt                       # THE commit point, all platforms: one
│                                     #   line, the active version string,
│                                     #   replaced by atomic rename (atomic
│                                     #   for files everywhere — a dir
│                                     #   symlink/junction rename is not
│                                     #   atomic on Windows, so we don't
│                                     #   build on it anywhere)
├── previous.txt                      # instant rollback target, same format
├── current -> versions/1.43.0        # convenience symlink for humans and
│                                     #   shell tools; refreshed best-effort
│                                     #   AFTER the commit, never read by
│                                     #   launcher/updater code
├── bin/hermes             # stable launcher: reads current.txt, execs
│                          #   versions/<v>/bin/hermes (same binary; §2.5.1)
└── (config.yaml, .env, skills/, sessions/… unchanged — data dir stays as-is)
```

The update algorithm becomes boring, which is the point:

1. `hermes-updater` downloads `hermes-<v>-<plat>.tar.zst` + verifies
   hashes/signature (sigstore or minisign key baked into the updater).
2. Unpacks into `versions/<v>.staging/`, fsyncs, renames to `versions/<v>`.
3. Runs the new version's self-check **from the new tree**
   (`versions/<v>/bin/hermes doctor --preflight`): imports core modules,
   validates config compat, checks data-migration needs. This replaces the
   post-pull syntax guard — and it tests the *actual artifact*, not just
   py_compile of a hardcoded critical-file list.
4. Commits: atomically replaces `current.txt` with the new version string
   (write `current.txt.new`, rename over — atomic for files on every
   platform). This is the single commit point. The convenience `current`
   symlink is refreshed best-effort afterwards; nothing load-bearing
   reads it.
5. Restarts services (below). Old version stays in `versions/` — `hermes
   update --rollback` rewrites `current.txt` from `previous.txt`, replacing
   today's git-reflog-archaeology recovery instructions. Keep last N=2.

What this deletes from the current world:

- **The entire update-boundary class.** Running processes keep executing the
  old tree (the launcher resolves `current.txt` to a concrete
  `versions/<v>` path before exec, so a mid-run flip doesn't swap modules
  under anyone). `_UvResult`, retry-once, importlib
  reload, bytecode purge, code_skew: all unnecessary. New code only ever runs
  in a fresh process.
- **The Windows lock war, mostly.** Nothing ever writes into the tree a
  running process has mapped; the new version unpacks into a fresh directory.
  The only lock-sensitive moment is the `current.txt` replace (atomic file
  rename — open `.pyd`s in the *old* slot don't block it because nothing
  renames over them) and the stable launcher itself,
  which is never rewritten during update. `--force`, `--force-venv`,
  gateway pause/resume, venv-holder detection: gone. Old versions are
  garbage-collected on a *later* run when no process has them open.
- **Dirty-tree management.** There is no working tree. Lockfile churn,
  autostash, unmerged-index recovery: gone (moved to ejected mode where
  they belong — see 2.5).
- **Interrupted-update recovery.** An interrupted download/unpack leaves a
  `.staging` dir that gets deleted and retried; `current.txt` never named
  it. The breadcrumb-marker + launch-time-repair machinery goes away.

## 2.3 The updater as a separate tiny program

`hermes-updater` is a small static binary (Rust — grow it out of the existing
`apps/bootstrap-installer` Tauri core, which already has the process-wait,
marker, and streamed-stage machinery) shipped both **inside the bundle** and
staged at `$HERMES_HOME/bin/hermes-updater`.

**Hard invariant: no Hermes Python process is alive while the updater
mutates anything.** Today `cmd_update` runs pre-pull code in memory against
freshly-pulled modules on disk, and every first-time lazy import in the
post-pull phase crosses that boundary — which is why tombstones like
`rebuild_venv` and `_UvResult` (Appendix B) can never be deleted. In the new
world the CLI/gateway/desktop *request* an update, fully exit, and the
updater does everything; the tombstone class becomes deletable.

Everything else calls it:

- `hermes update` (CLI) → execs `hermes-updater apply [--channel stable]`,
  streams its progress. The Python CLI keeps only the *decision* layer
  (which channel, managed-mode refusals, docker/nix messages).
- Gateway `/update` → spawns `hermes-updater apply --notify-file …`
  detached. The file-IPC protocol survives but shrinks: no interactive
  prompts remain in managed mode (no stash, no dep tiers — migrations are
  handled by the new version at first boot).
- Desktop → invokes the same binary with `--report json` and renders
  progress. Windows/mac/Linux all use the *same* flow now: quit, updater
  flips, updater relaunches. The macOS bundle-swap and Linux
  unpacked-relaunch special cases collapse into "the desktop app is in the
  bundle; the flip replaced it; relaunch `current`'s binary". The guiSkew
  state disappears for managed installs because the GUI is versioned with
  everything else. (AppImage/.deb/.rpm users are ejected-adjacent: they keep
  a package-manager-owned shell and the updater only updates the backend —
  the ONE remaining place a skew message is honest and needed.)
- First install = the updater with no `versions/` yet. `install.sh` shrinks
  to: detect platform → download `hermes-updater` → `hermes-updater install`.
  The 3,100-line install.sh and 3,300-line install.ps1 reduce to
  bootstrap-fetch + PATH setup; the stage-manifest protocol the desktop
  bootstrap uses is served by the updater natively instead of by bash/pwsh
  emitting JSON frames.

### 2.3.1 Who updates the updater?

The trick is that there are TWO copies with different lifecycles, and only
one of them needs ceremony:

- **The in-bundle copy** (`versions/<v>/bin/hermes` — remember launcher and
  updater are the same crate) is immutable content inside a slot. It updates
  by the normal flip, like every other file. Zero special handling.
- **The staged copy** (`$HERMES_HOME/bin/hermes-updater`, plus the PATH
  launcher symlink target) is what runs *before* any particular version is
  current — it's the bootstrap. This is the only self-replacement problem.

The flow, per update:

1. **Version check first.** The *old* staged updater downloads the bundle
   and reads `manifest.json`'s `min_updater_version`. If it's too old to
   apply this bundle correctly (layout change, new migration step), it does
   the **bootstrap hop**: extract just the new updater binary from the
   already-verified bundle into a temp path, re-exec into it with the same
   argv (+ `--hopped` so a broken new updater can't loop), and the new
   updater performs the apply. The old binary never mutates anything it
   doesn't understand.
2. **Apply as usual** (unpack → preflight → flip).
3. **Restage after the flip.** The (possibly hopped) updater replaces the
   staged copy from the freshly-current slot:
   - **POSIX**: write to `bin/.hermes-updater.new`, `rename()` over the old
     path. Atomic; a running old instance keeps executing its unlinked inode
     happily. Done.
   - **Windows**: a running exe can't be deleted or overwritten, but it CAN
     be renamed. So: rename the running `hermes-updater.exe` →
     `hermes-updater.old.exe`, move the new one into the real name, and
     delete `*.old.exe` best-effort now + on the next run (the same
     two-step the Tauri installer's deferred-`.exe`-replacement handling
     already dances around today, but as a designed step instead of a
     WinError-32 fallback).
4. **Failure containment.** Restaging happens *after* the flip commits, so
   a failed restage leaves a working install with an old-but-functional
   staged updater — which will simply hop again next time. The hop is the
   safety net that makes restaging non-critical.

Invariants that make this safe:

- `min_updater_version` in every manifest = an explicit compatibility
  contract between bundle layout and updater, replacing today's implicit
  "hope the old in-memory code understands the new tree" (the
  `sourceDeclaresServe` / `_UvResult` class of problem).
- **The paths the hop itself depends on are frozen at the same tier**:
  `manifest.json` at the bundle root and the updater/launcher binary at
  `bin/hermes` must exist in every future bundle an old staged updater
  could be asked to hop into. Moving either requires a
  `min_updater_version` bump plus a compat copy at the old path for one
  contract window — otherwise the hop can't find the binary it needs to
  hop *to*.
- The hop re-execs a **verified** binary (hash-checked as part of the
  bundle signature) — the old updater's signature check is the root of
  trust; a new updater is never run before its bundle verifies.
- The updater is intentionally tiny and boring precisely so this is rare:
  most releases change the *bundle*, not the updater contract, and
  `min_updater_version` stays put for months.
- One-shot hop guard (`--hopped`): if the new updater immediately dies, the
  old one reports failure instead of re-hopping — no infinite exec loop.

(Same pattern as rustup/deno/gh: the tool that performs replacement is the
newest one available, reached by at most one exec hop, and the staged copy
is repaired opportunistically after every successful update.)

## 2.4 Service/gateway restarts in the new world

Keep the good bones — exit-code contract 75/78 and drain — but make the
sequencing explicit and updater-driven:

1. Updater flips `current`.
2. Updater signals the gateway: `hermes gateway drain --then-exit-75` (or
   SIGUSR2). Gateway finishes in-flight turns within
   `restart_drain_timeout`, then exits 75; the supervisor (systemd/s6/
   launchd/desktop app) restarts it — the fresh process resolves `current`
   and comes up on the new version.
3. Notification markers stay (`.update_pending.json` → "back on v1.43.0"),
   but the gateway no longer needs the "is the detached updater still
   running?" watcher loop — the updater owns the whole lifecycle and pokes
   the notify file itself.
4. Desktop backend: same. The app's supervised `hermes serve` child exits 75
   on drain, the app respawns it from `current`. The desktop GUI itself
   relaunches via the updater when its own files changed (the manifest says
   whether `desktop/` differs between versions — backend-only releases don't
   even need a GUI restart, which today is impossible to know).
5. The update-in-progress marker (`.hermes-update-in-progress`) survives as
   the mutual-exclusion primitive between the updater and app launches — it's
   a good design (pid + age + self-heal), it just guards a much shorter
   critical section now (download happens before the marker; only flip +
   restart are inside it).

## 2.5 Ejected mode

`hermes eject` sets up a source checkout and points your PATH `hermes`
symlink at its in-repo launcher (the mechanism is §2.5.1 — going back to
managed is re-pointing the symlink at `$HERMES_HOME/bin/hermes`;
the ejected tree is kept):

```
$HERMES_HOME/                          # data dir, unchanged
~/src/hermes-agent/                    # (or wherever) — a plain git clone
├── bin/hermes                         # in-repo native launcher (§2.5.1)
├── .venv/                             # per-checkout venv, managed uv + python
└── ...
~/.local/bin/hermes -> ~/src/hermes-agent/bin/hermes
```

Semantics:

- **Detection is where the launcher lives**, not fragile method-sniffing:
  at `$HERMES_HOME/bin/hermes` (beside `versions/` + `current.txt`) =
  managed, inside a `.git` tree = ejected
  (see §2.5.1). `detect_install_method()` keeps docker/nix/brew but the
  git-vs-pip guessing collapses.
- **Ejected updates are today's `hermes update`, honestly scoped** — but see
  §2.5.2: on a tree with local changes, the *default* answer becomes a
  worktree, not a stash.
- **Caveats are surfaced, not discovered**: on eject, print (and record) the
  contract — you build locally (need Node/npm, build tools), syntax guard
  can only rollback git state not your venv, desktop rebuilds are on you,
  update-boundary bugs can require running update twice, CI-untested
  combinations are possible. The desktop app's update pill switches to
  "source install: N commits behind main" with the apply button running the
  source flow.
- **The bundled runtimes remain available to ejected mode.** The checkout's
  venv is still created with the managed uv against the managed python —
  ejecting means "I edit the Python/TS source", not "I manage toolchains".
  A `hermes eject --full` tier could opt out of even that, for people on
  nix-style self-managed toolchains.
- Today's implicit third state — "cloned the repo yourself and ran
  install.sh" (the current dev setup, and what *this* worktree is) — is just
  ejected mode without the ceremony; `hermes doctor` should recognize and
  stamp it.

### 2.5.1 Ejected mode ≡ the dev clone: the launcher IS the activation

No registry, no `source use` verb. Two ideas compose instead:

**(a) `hermes` is a native launcher that owns the environment.** `bin/hermes`
in every tree — managed bundle or git checkout — is the same small static
binary (same Rust crate as `hermes-updater`). On start it:

1. self-locates its tree root (`readlink /proc/self/exe` / argv[0], walk up),
2. sets up the environment *for that tree*: prepend the managed
   Node/uv/python bin dirs to PATH, export `VIRTUAL_ENV`/`UV_PYTHON` pinned
   to the tree's venv, unset `PYTHONPATH`/`PYTHONHOME`, `UV_NO_CONFIG=1`,
3. execs the tree's `venv/bin/python -m hermes_cli.main`.

This ONE binary replaces the generated bash/cmd shim, and it deletes the
env-hygiene guard family at the root: the PYTHONPATH module-shadowing
unsets (#21454-adjacent), the per-stage `UV_PYTHON` re-pinning, the
`with_hermes_node_path` PATH surgery, and the npm-prefix gymnastics all
live in one audited place instead of being re-derived in install.sh,
install.ps1, main.ts, and every bootstrap stage subprocess.

**(b) Activation = which launcher your PATH symlink resolves to.**

```
# managed (default):
~/.local/bin/hermes -> $HERMES_HOME/bin/hermes     # stable launcher; reads
                                                   #   current.txt → active slot

# ejected / dev — indistinguishable from any git-cloned tool:
~/.local/bin/hermes -> ~/src/hermes-agent/bin/hermes
# or a worktree:
~/.local/bin/hermes -> ~/src/hermes-agent/.worktrees/ethie-updater-rework/bin/hermes
```

That's the whole mechanism. "Eject" is: clone the repo, provision it
(`hermes dev sync` inside the checkout: per-tree `.venv` via the managed
uv + managed Python, node deps, builds), and point your symlink at it.
Going back to managed is pointing the symlink back at
`$HERMES_HOME/bin/hermes`. `hermes eject` can automate exactly
those steps for the casual case, but there is no special state to be in —
`which hermes` + one readlink IS the mode detection.

Consequences:

- **Worktrees are free.** Every worktree carries its own `bin/hermes`
  launcher (it's in the repo) and its own `.venv` (uv's hardlinked cache
  makes N venvs cheap). Run a worktree in place by invoking its launcher
  directly — `./bin/hermes`, no global state touched — or point the PATH
  symlink at it when you want the *services* (gateway, desktop backend,
  cron) running that branch. This retires `run_tests.sh`'s third fallback
  (sharing `~/.hermes/hermes-agent/venv` across worktrees), which is
  update-boundary skew wearing a dev hat.
- **Services follow the launcher.** The gateway unit / desktop backend
  spawn `hermes` off PATH (or an explicitly configured launcher path), so
  drain + exit-75 + respawn picks up whatever the symlink targets — the
  same restart contract covers a managed flip and a dev branch switch.
- **The launcher self-checks instead of a flip-time gate.** There's no
  `source use` moment to preflight, so the launcher does it: venv missing
  or core imports broken → print "run `hermes dev sync`" and exit nonzero
  *fast*, so a mispointed symlink can't put the gateway supervisor into an
  ImportError respawn loop (restart_loop_guard's job gets easier, not
  harder).
- **Mode detection collapses.** managed = launcher lives at
  `$HERMES_HOME/bin/hermes` beside a `versions/` + `current.txt`;
  ejected = anywhere else with a `.git`. That
  replaces `.install_method` stamp archaeology for the git/pip cases, and
  `hermes update` scopes itself accordingly: managed defers to
  `hermes-updater`; in a checkout it runs today's git flow against that
  checkout alone (and can decline politely on a dirty dev tree).
- **Windows**: same binary, `hermes.exe`; the activation "symlink" is the
  launcher copied/hardlinked into the link dir — same resolution rule,
  and the stable-name-on-PATH property is what the current shim already
  provides. (Activation is a rare, user-driven action, so copy semantics
  are fine here — unlike the update flip, which is why the flip commits
  through `current.txt` rather than any link, §2.2.)

### 2.5.1a The cwd guard: inside a checkout, always say which hermes

Activation-by-symlink leaves one honest ambiguity: your PATH `hermes`
resolves to one tree (a managed slot, or checkout A) while your cwd is
inside another (checkout B you're hacking on). You type `hermes` and get
code from somewhere else — worst on `hermes update`, where "update the
thing I'm standing in" and "update the install I'm running" silently
diverge.

Rule: **inside any hermes-agent checkout, plain `hermes` refuses — you
always state which one you mean.**

- The launcher walks up from cwd for the nearest enclosing hermes-agent
  checkout (a `pyproject.toml` with `name = "hermes-agent"`; a worktree's
  `.git` *file* marks the tree boundary the same as a `.git` dir).
- No enclosing checkout → run normally. The flag tax exists only where
  the ambiguity exists.
- Inside a checkout → refuse fast (before any imports), even when the
  invoked launcher IS that checkout's own. Consistency over cleverness:
  "cwd inside a checkout ⇒ flag required" is one rule with no cases to
  reason about, and it means muscle-memory `hermes update` can never hit
  the wrong tree — today's #1 way to shred a dev setup. The two ways out:

```
hermes: you are inside a hermes-agent checkout (~/src/hermes-agent/.worktrees/foo).
say which hermes you mean:
  hermes --dev       run THIS checkout's ./bin/hermes
  hermes --global    run the installed hermes (managed or PATH target)
```

- `--dev` re-execs the cwd checkout's `bin/hermes` (which self-checks its
  venv per §2.5.1 and says "run `hermes dev sync`" if unprovisioned).
  `--global` proceeds with the invoked launcher. Both flags are also
  honored (as no-ops or re-exec respectively) outside checkouts, so
  scripts can pin intent unconditionally. Non-interactive callers
  (gateway supervisor, cron, desktop backend spawn) run with cwd outside
  any checkout, so services never trip the guard; anything scripted that
  does run inside a checkout states its intent and becomes
  self-documenting. Shell-alias `hermes --dev` if you live in a worktree
  all day.

The guard makes the symlink model safe for people with several worktrees:
inside a checkout, which tree executes is always explicit, never
inferred.

### 2.5.2 Ejected updates: worktree instead of stash

Today, `hermes update` on a tree with local changes runs the scariest code
in the whole updater: autostash → pull/reset → reapply, with the
unmerged-index recovery (#4735), the interrupted-stash quarantine, the
"restore local changes now? [Y/n]" prompt, and the
`non_interactive_local_changes: discard` footgun. All of it exists because
the update and the user's changes are forced to share ONE working tree.

With per-checkout launchers + venvs (§2.5.1), they don't have to. Ejected
`hermes update` on a modified tree offers:

```
Your checkout has local changes (3 files modified, branch ari/pty-fix).

  [1] Switch to unmodified Hermes v1.44.0  (default)
      → git worktree add .worktrees/v1.44.0 v1.44.0
      → provision it (hermes dev sync), repoint your `hermes` symlink at it
      → your branch stays EXACTLY as it is — nothing stashed, nothing touched
  [2] Merge v1.44.0 into ari/pty-fix
      → fetch + merge (or rebase) in place; you resolve conflicts like any repo
  [3] Cancel
```

Option 1 semantics:

- `git worktree add` from the same clone: near-free (shared object store),
  and the user's branch remains checked out in the original tree with all
  its uncommitted changes untouched — the failure mode where the stash
  machinery eats someone's work simply cannot occur.
- The new worktree is named for the target (`.worktrees/v1.44.0` for a
  release tag, or `.worktrees/main-<shortsha>` when tracking main),
  provisioned with `hermes dev sync`, and the PATH symlink flips to its
  launcher. Services drain/respawn per §2.4 onto the new tree.
- Going back to your branch = repoint the symlink at your original tree's
  launcher. Diffing your work against the fresh version = `git diff` across
  worktrees of the same repo. Old version-worktrees are listed and GC'd via
  `hermes dev gc` (with the same keep-N policy as managed slots).
- This makes ejected updates *converge* with the managed model: a version
  switch is a provision-next-to + atomic-activate, never an in-place
  mutation. The only in-place path left is option 2, which is an ordinary
  git merge/rebase that the user asked for by name — no hidden stash, no
  auto-reapply, no prompt about restoring anything.

Option 2 keeps today's flow but honestly: fetch, then `merge` (or
`--rebase`) into the user's branch, stop on conflicts like git always does.
The syntax-guard rollback still applies (roll back the merge commit), and
`hermes dev sync` re-provisions after the merge lands.

The autostash machinery survives only as a fallback for repos where
worktrees are unavailable (bare-ish setups, exotic filesystems), and for
`--in-place` die-hards.

Why a mode split beats today's one-path-fits-all: nearly every guard in §1.7
exists because the updater cannot tell "pristine managed install" from
"user has local commits, a dirty tree, a fork remote, and a half-broken
venv". Splitting the population lets the managed path delete the guards and
the ejected path keep them with honest UX.

## 2.6 One dependency manifest

`runtime-deps.json` at repo root, consumed by CI (to build bundles), the
updater (to fetch on-demand extras), and `hermes doctor`:

```json
{
  "python": { "version": "3.11.x", "source": "uv" },
  "node":   { "version": "22.x", "floor_reason": "vite8 needs ^20.19||>=22.12" },
  "uv":     { "channel": "latest-stable" },
  "chromium": { "source": "playwright", "on_demand": true },
  "ffmpeg": { "on_demand": true, "feature": "tts-voice" },
  "ripgrep": { "bundled": true }
}
```

This kills the duplicated version floors (`node_satisfies_build` in bash ≡
`Test-NodeVersionOk` in pwsh ≡ the Vite comment in main.ts), the scattered
download URL construction, and gives `hermes doctor` a single table to
validate an ejected environment against.

Native deps split into two explicit tiers, and the manifest is where a
dep declares which one it's in:

- **`"bundled": true`** (ripgrep): small, static-friendly, version-
  sensitive CLIs ship inside the bundle at `runtime/tools/`, which the
  launcher prepends to PATH — existing `shutil.which("rg")` call sites
  (dep_ensure, /files completion) resolve the pinned copy with no code
  changes, and a slot flip updates them atomically like everything else.
- **`"on_demand": true`** (ffmpeg, Playwright Chromium): system-level
  deps that are too big or too OS-entangled to bundle stay installed
  where they live today — system package manager or vendor caches
  *outside* the install tree — so slot flips and fresh worktrees never
  lose them. The `features.json` ledger (§2.10) carries the *intent* to
  have them; `dep_ensure.py` remains the lazy install path. Its
  installation backend is today install.sh's package-manager machinery —
  which is exactly why the phase-5 install.sh shrink has a dedicated
  extract-first precondition in the sunset checklist.

## 2.7 Migration path (incremental, not big-bang)

1. **Phase 0 — CI bundles exist.** Add the release pipeline; publish bundles
   next to the existing flow. Nothing consumes them yet. (Prereq: make the
   venv relocatable — audit for absolute-path baking; uv does most of this.)
2. **Phase 1 — updater binary.** Extract the Tauri installer's Rust core
   into `hermes-updater` with `install/apply/rollback/status --report json`.
   New installs get versioned-slot layout; `install.sh` becomes the thin
   fetcher but keeps the legacy path behind `--source` (= eject at install
   time).
3. **Phase 2 — migrate existing installs.** `hermes update` on a legacy
   git-managed install offers: "switch to managed releases (recommended) or
   stay on source (eject)". Pristine trees (no local commits, clean status,
   official origin) default to managed; anything dirty defaults to eject.
   This is the moment the two populations self-select.
4. **Phase 3 — desktop unification.** Desktop update/bootstrap flows call the
   updater; delete `applyUpdatesPosixInApp`, the relaunch-watcher script
   generation, the marker pre-write dance (updater owns the marker), and the
   Tauri `run_update` orchestration (it *becomes* the updater).
5. **Phase 4 — deletion.** Once managed-mode telemetry (opt-in) shows the
   legacy path is <n%, strip the guards from the managed path and leave them
   ejected-only. `cmd_update` shrinks from ~1,000 lines to a dispatcher.

## 2.8 What deliberately does NOT change

- `$HERMES_HOME` as the data dir, config.yaml/.env split, skills sync
  semantics, profiles — untouched. Data migrations run at first boot of a
  new version (guarded by `doctor --preflight`), which they effectively
  already do via config-version merge.
- Docker/NixOS/Homebrew stay opt-out distributions; nix actually gets
  *easier* since a hermetic pre-built bundle is much closer to what a nix
  derivation wants than "editable pip install at activation time".
- Termux: bundles are feasible (linux-arm64 + bionic quirks) but realistically
  stay source-mode/ejected initially — it already has its own pip path.
- The gateway drain/restart exit-code contract, the update notification
  marker UX, and the toast/snooze renderer logic all survive with smaller
  inputs.

## 2.9 App surfaces in ejected mode: `hermes desktop` / `hermes web` / TUI

In managed mode these are trivially solved — `ui/tui/dist`, `ui/web/dist`,
and `desktop/` arrive pre-built in the bundle, and the commands just launch
them. The interesting question is the git-checkout world, where today the
builds are smeared across `hermes update` (ui-tui + web workspaces),
`hermes desktop --build-only` (content-hash stamp), install.sh
`--include-desktop`, and the desktop app's own rebuild-during-update — four
callers, one implicit contract.

**Rule: launching never builds implicitly; `dev sync` owns all builds.**

- `hermes dev sync` becomes the single provision verb for a checkout:
  venv (uv sync), node deps, and the JS surfaces — extending the
  content-hash stamp logic that `_desktop_build_needed` already proves out
  to *every* artifact (tui dist, web dist, desktop unpacked), so it's
  incremental and a no-op when nothing changed. `dev sync --watch` can wrap
  the existing `npm run dev` flows for active hacking.
- `hermes desktop` / `hermes web` / `hermes --tui` in a checkout **check
  the stamp and refuse-with-instructions when stale or missing**
  ("built artifacts are behind the source tree — run `hermes dev sync`"),
  with an explicit `--build` opt-in for the old behavior. Today's
  launch-time surprise Electron build (minutes of vite output because you
  ran `hermes desktop` after a pull) disappears.
- Stale-vs-source detection is cheap and already half-built: the content
  hash stamp says whether dist matches the tree. The TUI/web/desktop each
  gain the same three-line check instead of desktop-only.
- **Skew guards become launcher-level.** The desktop GUI↔backend contract
  (`REQUIRED_BACKEND_CONTRACT`) and the `sourceDeclaresServe` sniffing
  exist because surfaces of different ages meet at runtime. In a checkout,
  all surfaces come from ONE tree and one `dev sync`, so version skew
  within a checkout is impossible by construction; the contract check
  remains only for genuinely remote backends (desktop app → remote
  gateway), which is the one place it's honest.
- The desktop app run from a checkout identifies itself as a dev shell
  (execPath under `release/<plat>-unpacked` of a git tree — the
  `resolveUnpackedRelease` logic already computes this) and routes its
  in-app "update" affordance to §2.5.2's worktree/merge flow instead of the
  managed apply.

## 2.10 Optional features (`lazy_deps`) across updates

Found while inventorying: `tools/lazy_deps.py::active_features()` decides
"which features has the user activated" by **probing the venv** (is at
least one of the feature's packages importable/installed). That works today
only because the venv is a single long-lived mutable object. It breaks in
both new worlds:

- **Managed slots**: every version ships a fresh CI-built venv →
  every lazy feature silently vanishes on every update.
- **Per-checkout venvs**: a new worktree's venv starts clean → features
  don't follow the user to the tree they just activated.

Fix: **activation is user intent; intent lives in the data dir, not the
venv.** Record it:

```
$HERMES_HOME/state/features.json
{ "matrix": {"activated_at": "...", "via": "gateway-setup"},
  "faster-whisper": {"activated_at": "...", "via": "lazy-install"} }
```

- `lazy_deps.ensure(feature)` writes the record on first successful
  install (a one-time migration seeds it from the current venv-probe on
  the first run that sees no state file — existing users lose nothing).
  Deactivation (`hermes features disable <x>`) removes the record; probing
  remains only as the "is it satisfied *right now*" check, never as the
  source of truth for intent.
- **Managed**: after a flip, first boot of the new version (or the updater
  itself, post-flip pre-restart) runs `lazy_deps.apply(state)` — install
  each recorded feature into the new slot's venv under the new pins. This
  *replaces* `_refresh_active_lazy_features` rather than adding to it: the
  CVE-pin-bump problem it solves ("user keeps stale version forever") is
  handled for free, because every update reinstalls active features at
  current pins into a fresh venv.
- **Ejected**: `hermes dev sync` applies the same record to the checkout's
  venv. Activate matrix once, and every worktree you sync afterwards has
  it — matching the intuition that features belong to *you*, not to the
  tree you happened to be standing in.
- Feature payloads that are venv-external (Playwright's Chromium, whisper
  model weights) already live in caches outside the install tree; the
  record just makes their *registration* survive too. Bundle-level
  `runtime-deps.json` (§2.6) and `features.json` are complementary: one
  says what a feature needs, the other says whether this user wants it.
- Same pattern already exists in miniature: `.no-bundled-skills` is a
  data-dir marker that install/update honor across versions. This
  generalizes it from one negative flag to a real feature ledger.

## 2.11 What happens to the partial-install recovery machinery

Today's codebase carries a whole recovery subsystem because the mutable
tree can be caught mid-surgery. The redesign doesn't "improve" most of it —
it removes the *state* the recovery exists to repair. Mechanism by
mechanism:

| Mechanism today | Why it exists | Fate |
|---|---|---|
| `.update-incomplete` marker + `_recover_from_interrupted_install()` on launch (+ its `.lock`) | venv is mutated in place; a kill mid-`pip install` strands it between versions | **Deleted (managed).** The venv arrives complete inside the bundle; an interrupted download/unpack leaves only a `.staging` dir that never became `current`. Cleanup = delete staging, retry. |
| `_venv_core_imports_healthy()` probe + repair inside "already up to date" | a current checkout does NOT imply a healthy venv | **Demoted to diagnosis.** `doctor --preflight` runs the same import probe, but *before the flip* against the staged slot — it gates activation instead of repairing damage after the fact. A managed slot that later goes unhealthy (disk corruption) is answered by re-flip to `previous` or re-download, not in-place repair. |
| Post-pull syntax guard + rollback to pre-pull SHA (#28452) | bad commit can land on main between CI and the user's pull | **Replaced by CI + preflight.** Bundles are built and smoke-tested in CI, so the class "user pulls unparseable source" can't ship; `doctor --preflight` on the staged slot is the residual belt-and-suspenders, and "rollback" is the `previous` link, always. Survives as-is in **ejected** option-2 merges (roll back the merge commit). |
| Interrupted-clone quarantine (`.broken-<ts>` move-aside, #40998), unmerged-index reset (#4735), autostash restore prompts | shared mutable working tree | **Managed: no working tree, gone. Ejected: mostly dissolved by §2.5.2** (worktree-not-stash); survives only in the `--in-place`/no-worktree fallback path. |
| Windows exe quarantine dance (`_quarantine_running_hermes_exe`, `PendingFileRenameOperations` reboot-deferred renames, `.old.*` sweep at startup) | pip must overwrite a running `hermes.exe` shim | **Shrinks to one designed step.** The only running binary that ever needs replacing is the tiny updater/launcher itself, handled by the §2.3.1 rename two-step. The venv's exe shims live inside slots and are never overwritten. The startup `.old.*` sweep survives in miniature (GC of `hermes-updater.old.exe`). |
| Retry-once loops (Tauri update + rebuild, desktop rebuild retry) | first run executes old in-memory code against new on-disk modules; second run loads clean | **Deleted.** No process ever runs old code against new modules (§2.3 invariant). |
| Update-in-progress marker (`.hermes-update-in-progress`) | desktop relaunch mid-update spawns a backend that re-locks the venv | **Kept, smaller.** Still the mutual-exclusion primitive, but its critical section shrinks from "entire multi-minute update" to "flip + service restart" (seconds), so the 20-minute staleness ceiling and pid-liveness self-heal almost never engage. |
| Gateway `.update_pending.json` / output / exit-code files | detached updater must survive the gateway restart and report back | **Kept.** This is notification plumbing, not damage recovery — it works the same, with fewer intermediate states to report. |
| `restart_loop_guard`, `code_skew.py` | respawn loops on broken code; stale `sys.modules` after hot pull | Loop guard **kept** (it also covers config errors); code_skew **deleted for managed** (the tree under a running process never changes) and moot for ejected (dev sync + explicit restart). |

The honest residue: recovery code doesn't hit zero. What remains is
(a) *staging* hygiene — delete `.staging`, GC old slots/worktrees,
(b) the updater's own §2.3.1 two-step, and (c) the ejected in-place
fallback, which keeps today's machinery but is explicitly opt-in. The
difference in kind: today's recovery repairs a **broken current
install**; the remaining recovery only ever discards **not-yet-activated
staging** — the user's working install is never the patient.

## 2.12 Docker

Docker is already the closest thing to the target architecture: the
published image is an immutable, CI-built, pre-resolved tree at
`/opt/hermes` (no `.git`, baked `docker` install-method stamp), updated by
replacing the whole artifact (`docker pull`) with data bind-mounted from
outside (`~/.hermes:/opt/data`). The redesign mostly means **the image
becomes a thin wrapper around the same bundle everyone else gets**:

- **Image = bundle + entrypoint.** The Dockerfile unpacks
  `hermes-<version>-linux-<arch>.tar.zst` instead of running its own
  pip/npm build — one build pipeline feeds releases AND images, so the
  image can't drift from the release (today they're two separate builds of
  the same commit). The in-container launcher is the same §2.5.1 binary;
  `versions/`/`current` degenerate to a single baked slot.
- **`hermes update` in-container stays a redirect**, same as today:
  `format_docker_update_message()` pointing at `docker pull` (+ tag pinning
  and config-persistence notes). The updater refuses politely; the *image
  tag* is the version selector. No slots inside the container — that's the
  orchestrator's job (pull, recreate).
- **The `/update` gateway command** on a Docker gateway should reply with
  the pull instructions rather than attempting anything — which is today's
  behavior, kept.
- **Shared-`$HERMES_HOME` coexistence gets cleaner.** The code-scoped
  `.install_method` stamp exists because a containerized gateway and a host
  install can share one data dir (§1.1). In the new world the host side
  doesn't consult a stamp at all — mode is where the launcher lives
  (§2.5.1) — so the "container stamped `docker` into my shared home and
  bricked host updates" class (#34397) loses its remaining legacy-stamp
  edge cases too.
- **`features.json` (§2.10) works unchanged**: it lives in the bind-mounted
  data dir, so a recreated container re-applies the user's lazy features on
  first boot exactly like a flipped slot would. (Container-local venv
  installs don't survive recreation today either — the ledger makes that
  loss self-healing instead of silent.)
- **Ejected-in-Docker** (bind-mounting a source checkout into a container)
  is just the dev-shell case: run the checkout's launcher inside the
  container; `dev sync` provisions a venv scoped to the container's
  platform. Worth a doc page, not special machinery.

Net: Docker needs almost nothing new — it's the existence proof that the
immutable-artifact model already works for this codebase. The redesign
brings the *other* install methods up to the property Docker always had,
and unifies the build pipeline so the image is derived from, not parallel
to, the release.

## 2.13 The migration chicken-and-egg: getting every old version through

The hard constraint: we cannot push code. Every existing install reaches
the new world by running its *own old updater* — which pulls new code onto
disk and then keeps executing **old in-memory post-update code against
it**. Any migration design that asks the old updater to *perform* the
migration inherits every update-boundary bug we're trying to eliminate,
against a population of arbitrary-age updaters we can't enumerate.

So the design rule is:

> **Old code never performs the migration. Old code's only job is to
> complete one more ordinary old-world update. The migration executes
> exclusively in a fresh process running current code.**

### The funnel

Every legacy install, no matter how ancient, follows the same three hops —
one funnel, not N per-version migration paths:

```
old vN ──(its own git-flow update, unchanged)──▶ current main
        ──(next launch: fresh process, new code)──▶ adoption offer
        ──(hermes-updater adopt: verified binary, own process)──▶ new world
```

**Hop 1 — any old version → current main, via the flow it already has.**
This works only if main remains a *valid old-world update target* for the
whole migration window. That means a frozen compatibility contract:

- Every symbol any historical updater touches post-pull is enumerated and
  pinned: `_install_python_dependencies_with_optional_fallback`'s calling
  convention, `ensure_uv()`'s dual-shape return (`_UvResult`),
  `rebuild_venv` (Appendix B), `tools.skills_sync.sync_skills`,
  `hermes_constants` attributes lazily read after the reload,
  `hermes_cli.profiles.seed_profile_skills`, the config-migration
  functions, `pyproject.toml` remaining editable-installable with an
  `[all]` extra. Collect them into an explicit `updater_compat` registry
  **with a CI test that fails if any pinned symbol changes signature or
  disappears** — turning today's implicit, incident-discovered contract
  (#39780 et al.) into an enforced one. The tombstones finally get a
  fence around them instead of scar-tissue comments.
- The retry-once behavior already deployed in the Tauri/desktop flows
  covers the crash-on-boundary stragglers: even if a very old updater
  trips once, the second run executes current code and lands.

**Hop 2 — the adoption trigger runs in a fresh process only.** After hop 1
finishes, the *next* `hermes` invocation is new code in a new process (CLI
launch, gateway supervisor respawn after exit-75, desktop backend
respawn — every cohort naturally produces one). That code path detects the
legacy layout (venv-in-checkout, no `versions/`, launcher is the old bash
shim) and stages adoption. Critically, the detection/offer code ships on
main *now-ish* and reaches everyone through hop 1 — it's the payload the
old updaters unknowingly deliver.

**Hop 3 — adoption is performed by the updater binary, not by Python.**
The Python side does the minimum possible: fetch the platform
`hermes-updater` (signature-verified against a key shipped in the new
code), exec it with `adopt`, and **fully exit** — honoring the §2.3
invariant from the very first new-world operation. `adopt` then:

1. downloads + verifies the bundle matching the checkout's release,
2. creates `versions/<v>/`, runs `doctor --preflight` against it,
3. points the PATH `hermes` symlink at `$HERMES_HOME/bin/hermes`
   and commits `current.txt`,
4. **leaves the old checkout completely untouched** — it is the rollback
   artifact (worst case: repoint the symlink back and you are exactly
   where you were) and, for users who later want it, the ready-made
   ejected tree,
5. restages itself (§2.3.1), restarts services (§2.4).

Adoption failing at ANY step leaves a fully working old-world install:
nothing about it is destructive until the PATH-symlink re-point, and even
that is reversible with one `ln -sf` back at the old target (recorded in
`.pre-adopt-target`).

### Per-cohort behavior

| Cohort | Trigger | Policy |
|---|---|---|
| Pristine managed install (clean tree, official origin, on main/known branch, no local commits) | next launch after hop 1 | offer with default **yes**; desktop/gateway non-interactive installs may auto-adopt after a config-gated grace period (`updates.adopt: auto\|prompt\|never`) |
| Dirty tree / local commits / fork remote | next launch | never auto: offer **eject** (keep exactly what they have, formalized per §2.5) alongside adopt; this is the self-selection moment from §2.7 phase 2 |
| Gateway-only headless boxes | first supervisor respawn after hop 1 | message to home channel with the adopt command; `/update` gains an `adopt` arm |
| Desktop (Windows staged Tauri) | old `hermes-setup.exe --update` still works through hop 1 (its two child commands are part of the frozen contract); the *new* backend then drives adoption, and adopt replaces the staged binary | prompt in the update overlay |
| Docker / NixOS / Homebrew / pip | no adoption — they keep their own channels (§2.12); pip keeps its deprecation warning |

### The long tail and the sunset

- **The compat contract on main is time-boxed but generous** (12+ months):
  old updaters fetch `origin/main` by name, so there is no way to redirect
  them — main itself must stay hop-1-viable until the legacy population is
  negligible. New installs go straight to the new world from day one, so
  the population only shrinks.
- **Measuring the tail**: the adoption offer (and its decline) can be
  counted only via opt-in diagnostics per repo policy — otherwise sunset
  timing falls back to support-channel signal.
- **After sunset**, the frozen symbols and the giant `_cmd_update_impl`
  are deleted in one commit. An unmigrated install that updates after that
  will crash mid-update **once**, with hop 1 having already placed new
  code on disk — so its next launch still lands on the adoption path. Ugly
  but self-healing; and it's the fate only of installs that ignored a year
  of prompts. (This is also why the adoption *detector* should live in the
  most crash-proof spot possible: top of `main()`, before anything that
  could trip on a stale venv.)

The meta-point: this migration is the update-boundary problem *one final
time*, solved the same way the steady state solves it — old code's last
act is to deliver bytes and exit; new code, in a new process, does
everything else. We pay the tombstone tax once more, fence it with CI so
it can't silently grow, and then the fence and the tax both get deleted
together.

---

## Appendix A — file map of today's updater world

| Concern | Files |
|---|---|
| Bootstrap install | `scripts/install.sh`, `scripts/install.ps1`, `scripts/install.cmd` |
| CLI update | `hermes_cli/main.py` (`cmd_update`, `_cmd_update_impl`, `_cmd_update_pip`, `_update_via_zip`, `_update_node_dependencies`, `_build_web_ui`, `_desktop_build_needed`, ~15 more helpers), `hermes_cli/subcommands/update.py` |
| Install-method detection | `hermes_cli/config.py` (`detect_install_method`, `is_managed`, `is_uv_tool_install`, `recommended_update_command`, unsupported-method warnings) |
| Managed native deps | `hermes_cli/managed_uv.py` (uv), `hermes_constants.py` (`find_node_executable`, `iter_hermes_node_dirs`, `heal_hermes_managed_node`, `with_hermes_node_path`) |
| Gateway update/restart | `gateway/slash_commands.py` (`/update`), `gateway/run.py` (update watcher, notification, exit codes), `gateway/restart.py`, `gateway/code_skew.py`, `gateway/restart_loop_guard.py`, `gateway/drain_control.py` |
| Desktop bootstrap | `apps/desktop/electron/bootstrap-runner.ts`, `bootstrap-platform.ts` |
| Desktop update | `apps/desktop/electron/main.ts` (`checkUpdates`, `applyUpdates`, `applyUpdatesPosixInApp`, `resolveUpdaterBinary`), `update-remote.ts`, `update-relaunch.ts`, `update-marker.ts`, `update-rebuild.ts`, `update-count.ts`, `backend-command.ts` (serve/dashboard fallback) |
| Desktop update UI | `apps/desktop/src/store/updates.ts`, `src/app/updates-overlay.tsx`, `src/lib/update-copy.ts` |
| Tauri installer/updater | `apps/bootstrap-installer/src-tauri/src/{update,bootstrap,install_script,paths,powershell}.rs` |

## Appendix B — the `rebuild_venv` tombstone (update-boundary, reverse direction)

`hermes_cli/managed_uv.py:253` ends with what looks like a dead stub:

```python
def rebuild_venv(uv_bin: str, venv_dir: Path, python_version: str = "3.11") -> bool:
    True # dont remove me. ask ethernet
```

It has no callers on current main — but it is NOT dead code. It's a
**tombstone symbol** for the update-boundary problem, in the opposite
direction from `_UvResult`:

- `_UvResult` protects **old call sites against a new module** whose return
  contract changed (2-tuple vs single value).
- `rebuild_venv` protects **old call sites against a new module that removed
  a symbol**. `cmd_update` used to call `rebuild_venv` in its post-pull
  phase. The updater runs *pre-pull* code in memory, pulls new code onto
  disk, then continues — and any *first-time lazy import* during that
  post-pull phase resolves against the freshly-pulled files. If a release
  deleted `rebuild_venv`, every install parked on a version that still
  called it would crash its first update with `AttributeError` — after the
  pull, mid-surgery. So the symbol must stay parked forever (or until the
  parked population is provably gone).

This is the sharpest single illustration of §1.7 item 1 and the core
argument for §2.2/§2.3: **a public API surface between updater and updatee
that can never shrink** is what you get when the process being updated
orchestrates its own update. The fix is not more tombstones — it's that
Hermes fully exits and a separately-versioned external updater owns the
entire mutation, so no old code ever imports new modules. In the proposed
world, `rebuild_venv` (and `_UvResult`) are deletable the day the managed
population migrates to versioned slots.
