# Phase 0 — CI Release Bundles

> **For Hermes:** Use subagent-driven-development to implement task-by-task.
> Read `docs/updater-world.md` §2.1, §2.6 first.

**Goal:** CI produces, for every release tag, a self-contained per-platform
bundle that boots Hermes with zero local building. Nothing consumes bundles
yet — this phase is pure additive pipeline work and can land without risk.

**Tech:** GitHub Actions, `uv` (python-build-standalone runtimes,
`uv sync --locked`), `zstd`, existing npm build scripts, `minisign` (or
sigstore-cosign — decide with maintainer in Task 0.0) for signing.

**Definition of done:** `scripts/e2e/test-bundle-boot.sh` passes on
linux-x64: download (file://) → unpack → `bin/hermes --version` and
`bin/hermes doctor --preflight` succeed on a machine with **no** system
python/node/git, verified inside a `debian:stable-slim` container.

---

## Task 0.0: Decisions checkpoint (maintainer sign-off, no code)

**Objective:** Pin the decisions the rest of the phase builds on.

Confirm with the maintainer and record the answers at the top of
`scripts/release/README.md` (create it):

1. Signing scheme: minisign key pair vs sigstore. (Spec default: minisign —
   one static pubkey embedded in the updater, no OIDC dependency.)
2. Channels: `nightly` (every main merge? daily cron?) and `stable`
   (manually promoted tag). Spec default: daily nightly + manual stable.
3. Platform matrix for v1: `linux-x64`, `linux-arm64`, `darwin-arm64`,
   `win-x64`. (`darwin-x64` deferred unless CI capacity allows.)
4. Bundle versioning: calver `YYYY.MM.DD[.N]` for nightlies, semver for
   stable. Spec default: yes.

**Verification:** README committed; maintainer ack in PR review.

## Task 0.1: `runtime-deps.json` manifest (§2.6)

**Objective:** Single source of truth for runtime dependency versions.

**Files:**
- Create: `runtime-deps.json` (repo root)
- Test: `tests/test_runtime_deps_manifest.py`

**Step 1:** Create the manifest. Derive the CURRENT values from code — do
not invent them. Sources: `PYTHON_VERSION`/`NODE_VERSION` in
`scripts/install.sh:59-60`, the Vite floor comment in
`scripts/install.sh:781-786`.

```json
{
  "schema": 1,
  "python": { "version": "3.11", "source": "uv" },
  "node": { "version": "22", "floor": "^20.19 || >=22.12", "floor_reason": "vite8 util.styleText" },
  "uv": { "channel": "latest-stable" },
  "chromium": { "source": "playwright", "on_demand": true },
  "ffmpeg": { "on_demand": true },
  "ripgrep": { "bundled": true }
}
```

**Step 2:** Write the invariant test (behavior, not snapshot!): the manifest
parses, `schema == 1`, python/node entries exist and their `version` fields
are non-empty strings matching `^\d+(\.\d+)?$`. Do NOT assert exact versions
(change-detector).

**Step 3:** `scripts/run_tests.sh tests/test_runtime_deps_manifest.py -q` →
expected: pass.

**Step 4:** Commit: `feat(release): add runtime-deps.json manifest`.

## Task 0.2: Relocatable venv audit

**Objective:** Prove (or fix) that a CI-built venv works after being moved
to a different absolute path — the property slots depend on.

**Files:**
- Create: `scripts/release/check-relocatable.sh`

**Step 1:** Write the check script. Two traps make a naive version
false-pass, and both MUST be handled or this check proves nothing:

1. **`uv sync` installs the project EDITABLE by default.** An editable
   install's `.pth` points back at the source checkout — which still
   exists after you move the venv — so imports succeed via the *source
   tree*, not the venv. Pass `--no-editable`.
2. **The source tree must be unreachable during the probe.** Even a
   non-editable install can false-pass if cwd is the repo root (`''` on
   `sys.path` resolves top-level modules like `run_agent` from the
   checkout). Copy the tree, build from the copy, then DELETE the copy
   before probing, and probe from a neutral cwd.

```bash
#!/usr/bin/env bash
set -euo pipefail
# Build a venv from a throwaway COPY of the source at path A, delete the
# copy, move the venv to path B, verify core imports work from a neutral
# cwd. Proves the venv carries everything and bakes no absolute paths.
REPO_ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d) && DST=$(mktemp -d)
git -C "$REPO_ROOT" archive HEAD | tar -x -C "$WORK"      # clean copy, no .git
cp "$REPO_ROOT/uv.lock" "$WORK/uv.lock" 2>/dev/null || true
UV=${UV:-uv}
"$UV" venv --python 3.11 --relocatable "$WORK/venv"
VIRTUAL_ENV="$WORK/venv" "$UV" sync --extra all --locked --no-editable \
  --project "$WORK" --python "$WORK/venv/bin/python"
mv "$WORK/venv" "$DST/venv"
rm -rf "$WORK"                                             # source is GONE
cd /
"$DST/venv/bin/python" -c "import hermes_cli, run_agent, model_tools; print('RELOCATABLE_OK')"
```

**Step 2:** Run it: `bash scripts/release/check-relocatable.sh`
Expected: prints `RELOCATABLE_OK`. If the import fails with a path under
the (deleted) work dir, that IS the finding — an editable/.pth/shebang
leak; see step 3.

**Step 3 (only if step 2 fails):** The failures will be shebang paths or
baked absolute paths in `.pth`/entry-point files. Fix by (a) using
`--relocatable`, (b) replacing the editable install with a regular
(non-`-e`) install for bundle builds — editable installs bake the source
path. Record findings in `scripts/release/README.md`. **Do not proceed to
0.3 until this passes.**

**Step 4:** Commit: `feat(release): relocatable venv check script`.

## Task 0.2b: Slot artifact-root resolver (the non-editable gap)

**Objective:** Close the gap that relocatability alone does NOT cover:
today's world is `pip install -e .`, so `hermes_cli.__file__` lives *in
the checkout* and every repo-root-relative lookup works for free —
bundled `skills/` sync, the web dashboard's `web_dist`, ui-tui dist
resolution, `PROJECT_ROOT` detection, `constraints-termux.txt`. In a slot
the venv is a **regular** install: code resolves from `site-packages`,
while the artifacts live at `<slot>/app/`, `<slot>/ui/tui/dist`,
`<slot>/ui/web/dist`. Without an owner, `hermes --tui`, dashboard SPA
serving, and skills sync are silently broken in every bundle — and a
naive import-only preflight goes green anyway.

**Files:**
- Modify: `hermes_constants.py` (single resolver, beside `get_hermes_home()`)
- Test: `tests/test_artifact_roots.py`

**Step 1 — inventory (do not guess):** find every repo-root-relative
lookup: `search_files '__file__' hermes_cli/ tools/ gateway/` plus
`search_files 'parent.parent' --file_glob '*.py'`, and trace how
`PROJECT_ROOT` in `hermes_cli/main.py`, `_build_web_ui`'s dist consumer,
the TUI entry resolution, and `tools/skills_sync.py` locate their assets.
Record the list in this doc's PR description — phase 1's preflight (task
1.5) probes each of them.

**Step 2 (failing tests first):** one function, one rule:

```python
def get_artifact_root() -> Path:
    """Root for repo-shipped assets (skills/, ui/, app source).

    Slot: the bundle root (walk up from __file__ to the dir containing
    manifest.json — site-packages lives under <root>/runtime/venv/).
    Checkout: the repo root (dir containing pyproject.toml + .git),
    which is what today's editable-install world already resolves to.
    """
```

plus thin, named accessors used by the call sites from step 1
(`bundled_skills_dir()`, `web_dist_dir()`, `tui_dist_dir()`) so the
layout difference (`skills/` vs `app/skills/`, `web/web_dist` vs
`ui/web/dist`) is encoded in ONE module, not at N call sites. TDD with
tmpdir fixtures of both layouts.

**Step 3:** migrate the call sites from step 1's inventory onto the
accessors. Behavior in a checkout must be byte-identical (the accessors
return exactly the paths those sites compute today — assert this in the
tests against a checkout-layout fixture).

**Step 4:** `scripts/run_tests.sh tests/test_artifact_roots.py -q` → pass.

**Step 5:** Commit: `feat(release): artifact-root resolver for slot layout`.

> **Contract note:** `manifest.json` at the slot root is what makes slot
> detection work — task 0.3's layout and task 1.1's `resolve_tree_root`
> both key on it. It is part of the updater↔bundle contract (§2.3.1).

## Task 0.3: Bundle build script

**Objective:** One script that assembles the full bundle layout from §2.1
on the current machine.

**Files:**
- Create: `scripts/release/build-bundle.sh`

**Layout produced** (must match §2.1 exactly):

```
dist/bundle/
├── manifest.json        # written by task 0.4
├── runtime/python/      # uv-managed CPython (uv python install --install-dir)
├── runtime/venv/        # uv sync --locked, NON-editable install of the repo
├── runtime/node/        # node LTS tarball extract (reuse install.sh's URL logic)
├── runtime/tools/       # bundled native CLIs from runtime-deps.json entries
│                        #   marked "bundled": ripgrep's `rg` (static musl build
│                        #   on linux, official archives elsewhere). The launcher
│                        #   prepends this dir to PATH before exec so existing
│                        #   shutil.which("rg") call sites resolve it with zero
│                        #   code changes; a system rg earlier on PATH still
│                        #   wins nothing — slot dir is prepended, giving the
│                        #   pinned version priority. ffmpeg/chromium are NOT
│                        #   here (on_demand system-level deps, §2.6).
├── app/                 # git archive of the source tree (no .git), precompiled .pyc
├── ui/tui/dist/         # npm run build in ui-tui
├── ui/web/dist/         # web build (_build_web_ui equivalent: npm run build in web/)
├── desktop/             # npm run pack output (release/<plat>-unpacked)
└── bin/hermes           # placeholder shell shim until phase 1's binary exists:
                         #   #!/bin/sh
                         #   exec "$(dirname "$0")/../runtime/venv/bin/hermes" "$@"
                         # NOTE: the venv entrypoint's shebang must survive
                         # relocation — that's exactly what task 0.2's check
                         # proves (uv --relocatable rewrites shebangs to the
                         # `#!/bin/sh`-exec trampoline form). If 0.2 didn't
                         # pass, this shim breaks on unpack; safer variant:
                         #   exec ".../runtime/venv/bin/python" -m hermes_cli.main "$@"
                         # which sidesteps entrypoint shebangs entirely.
```

Key implementation notes for the implementer:
- `app/` via `git archive HEAD | tar -x -C dist/bundle/app` — never copy the
  working tree (dirty-tree leakage).
- The venv installs `app/` as a **regular** package (`uv pip install
  dist/bundle/app` with `--locked` constraints), NOT `-e .`.
- Node: reuse the resolution logic from `install.sh:830-943` (latest tarball
  from `nodejs.org/dist/latest-v22.x/`) but pin the resolved version into the
  manifest for reproducibility.
- Desktop build: `cd apps/desktop && npm run pack` (same as
  `install_desktop()`), copy `release/<plat>-unpacked` into `desktop/`.
- Everything is best-effort EXCEPT runtime/ + app/: a bundle without desktop/
  is valid (flag it in the manifest as `"desktop": false`).

**Verification:**
```bash
bash scripts/release/build-bundle.sh --out dist/bundle
dist/bundle/bin/hermes --version        # prints version, exit 0
```

**Commit:** `feat(release): bundle build script`.

## Task 0.4: manifest.json + hashing + signing

**Objective:** Every bundle carries integrity + compat metadata.

**Files:**
- Create: `scripts/release/write-manifest.py`
- Test: `tests/release/test_bundle_manifest.py`

**manifest.json schema:**

```json
{
  "schema": 1,
  "version": "2026.07.14",
  "channel": "nightly",
  "git_sha": "<40-hex>",
  "platform": "linux-x64",
  "min_updater_version": "0.1.0",
  "desktop": true,
  "files": { "runtime/venv/bin/python": "sha256:...", "...": "..." }
}
```

`files` covers every regular file in the bundle. The signature is a
minisign signature over `manifest.json` itself, shipped as
`manifest.json.minisig` — verify-manifest-then-verify-files gives whole-
bundle integrity with one signature.

**Test (invariants):** round-trip write→verify on a tiny fixture tree;
verification fails when any file is modified; fails when manifest is
modified. Use a throwaway minisign key generated in the test tmpdir.

**Verification:** `scripts/run_tests.sh tests/release/ -q` → pass.

**Commit:** `feat(release): bundle manifest + signing`.

## Task 0.5: GitHub Actions release workflow

**Objective:** Automate 0.3+0.4 per platform, upload to GitHub Releases.

**Files:**
- Create: `.github/workflows/release-bundles.yml`

**Requirements:**
- Trigger: tag push `v*` (stable) + `workflow_dispatch` + daily cron
  (nightly). Nightly uploads to a rolling `hermes-nightly` prerelease tag.
- Matrix: from Task 0.0's platform list. linux-arm64 via
  `ubuntu-24.04-arm` runner; mac on `macos-14`; windows on `windows-2022`
  (adapt build-bundle.sh into a thin `build-bundle.ps1` wrapper ONLY where
  bash isn't viable — prefer bash via git-bash on Windows).
- **Pin every action to a commit SHA with a version comment** (repo
  dependency-pinning policy — see AGENTS.md).
- Signing key from repo secrets (`MINISIGN_SECRET_KEY`).
- Smoke test IN the workflow before upload: unpack into a clean dir and run
  `bin/hermes --version` + `python -c "import run_agent"` from the venv.

**Verification:** `workflow_dispatch` run on a branch produces downloadable
artifacts for at least linux-x64; smoke step green.

**Commit:** `ci(release): bundle build + publish workflow`.

## Task 0.6: E2E gate

**Objective:** The phase-closing proof.

**Files:**
- Create: `scripts/e2e/test-bundle-boot.sh`

**Script contract:** takes a bundle path; runs inside
`debian:stable-slim` (docker) with NO python/node/git installed:

```bash
docker run --rm -v "$BUNDLE_DIR":/b:ro debian:stable-slim /bin/sh -c '
  set -e
  /b/bin/hermes --version
  HERMES_HOME=/tmp/hh /b/bin/hermes doctor --preflight 2>/dev/null || \
  HERMES_HOME=/tmp/hh /b/runtime/venv/bin/python -c "import hermes_cli.main, run_agent, model_tools, gateway.run; print(\"PREFLIGHT_OK\")"
'
```

(The `doctor --preflight` subcommand ships in phase 1; until then the
python-import fallback line is the gate. Leave both lines — the script
tightens automatically when phase 1 lands.)

**Verification:** run against a Task-0.5 artifact → `PREFLIGHT_OK`, exit 0.

**Commit:** `test(e2e): bundle boot gate` — **phase 0 complete.**

## Pitfalls for this phase

- `uv sync --extra all` vs `--all-extras`: use `--extra all` (curated set).
  See the long comment at `scripts/install.sh:1474-1483` for why.
- Precompiling `.pyc`: use `python -m compileall -j0 app/` with
  `--invalidation-mode unchecked-hash` so timestamps don't matter in an
  immutable tree.
- macOS: the desktop `.app` must be signed/notarized for distribution —
  coordinate with maintainer; an unsigned bundle is acceptable for the
  phase-0 gate (mark `"desktop_signed": false` in the manifest).
- Do NOT touch install.sh/install.ps1/cmd_update in this phase. Additive
  only.
