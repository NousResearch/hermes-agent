# HERMES-ORCH-000 — `apps/` Ownership Classification

Date: 2026-07-11

## Goal

Classify every untracked file under `C:\Users\fallo\AppData\Local\hermes\hermes-agent\apps` before controlled activation of the verified Hermes candidate. This was a read-only audit of the dirty live repository. No live file, branch, gateway process, profile, model, credential, or configuration was changed.

## Plain-language summary

All 694 untracked files are generated JavaScript emitted in place by the desktop TypeScript build. They are not copies of tracked JavaScript, unique source work, or unresolved unknown files.

Every file:

- ends in `.js`;
- has a same-stem tracked `.ts` or `.tsx` source file;
- was modified during the same one-minute window (`2026-07-09 20:37` local time);
- is absent at that exact `.js` path from live `main`, all known local/remote refs, the candidate, and current upstream `main`;
- is outside the build-output ignore rules because TypeScript emitted it beside source under `apps/*/src`;
- would be left untouched by the candidate fast-forward because the candidate changes no `apps/` path.

The set is a coherent compiler-output tree: the candidate contains 696 tracked TypeScript/TSX files in the relevant `apps/desktop/src` and `apps/shared/src` roots; 694 emit corresponding `.js` files, while the two declaration inputs (`global.d.ts` and `vite-env.d.ts`) correctly emit no JavaScript.

## Live repository state

- Repository: `C:\Users\fallo\AppData\Local\hermes\hermes-agent`
- Branch: `main`
- HEAD: `540f90190f50f9518bf36632a724e0e58877a10b`
- Dirty state: 694 untracked entries, all under `apps/`
- Tracked modifications/deletions: 0
- Gateway remained running from the installed live venv; it was not stopped or restarted.

Stable enumeration command:

```text
git -C C:\Users\fallo\AppData\Local\hermes\hermes-agent \
  ls-files --others --exclude-standard -z -- apps/
```

The NUL-delimited result was decoded, sorted lexically, and verified to contain exactly 694 paths.

## Candidate state

- Branch: `feature/hermes-orch-000-candidate`
- Candidate worktree: `C:\Users\fallo\AppData\Local\hermes\worktrees\hermes-orch-000-candidate`
- Starting candidate HEAD for classification: `980553ea303feb6d904234a58fbfca96e17c1382`
- Starting candidate worktree: clean
- Merge base with live `main`: `540f90190f50f9518bf36632a724e0e58877a10b`
- Candidate relative to live HEAD: 13 commits ahead
- Candidate paths changed under `apps/`: 0

## Exact untracked count

```text
694 files
4,527,011 total bytes
694 .js files
146 generated test files (*.test.js / *.spec.js)
```

Source-counterpart breakdown:

```text
393 generated from tracked .ts files
301 generated from tracked .tsx files
694 generated outputs with a tracked same-stem source
0 generated outputs without a tracked same-stem source
```

Completeness check:

```text
696 tracked .ts/.tsx inputs in apps/desktop/src + apps/shared/src
694 corresponding .js outputs present
2 declaration-only inputs without output:
  apps/desktop/src/global.d.ts
  apps/desktop/src/vite-env.d.ts
```

## Directory breakdown

```text
apps/desktop/src: 691
apps/shared/src:     3
```

Desktop source subtrees:

```text
apps/desktop/src/app:        283
apps/desktop/src/components: 185
apps/desktop/src/lib:        101
apps/desktop/src/store:       82
apps/desktop/src/i18n:        14
apps/desktop/src/themes:      14
apps/desktop/src/hooks:        6
other direct/types paths:      6
```

Shared outputs:

```text
apps/shared/src/index.js
apps/shared/src/json-rpc-gateway.js
apps/shared/src/websocket-url.js
```

## Classification totals

| Category | Count | Meaning |
|---|---:|---|
| A | 0 | Exact tracked copies |
| B | 694 | Generated/build artifacts |
| C | 0 | Unique local work |
| D | 0 | Unknown |

Each file is assigned to exactly one category in both manifests.

## Category A — exact tracked copies

Count: 0.

For every file, its Git blob identity was computed with `git hash-object --no-filters`. Same-path and same-blob comparisons were performed against 22 unique commits represented by all known local/remote refs plus the explicit live and candidate commits. No exact same-path or alternate-path tracked blob match was found.

The currently advertised upstream `main` was also inspected without updating the live repository:

```text
git ls-remote origin refs/heads/main
5ecc07986f46463ca3096679b03a46402eb19cee
```

The recursive GitHub tree for that commit was complete (`truncated=false`, 7,112 entries). All 694 `.js` paths were absent. Therefore none is an exact tracked copy in current upstream either.

## Category B — generated/build artifacts

Count: 694.

### Build-rule evidence

`apps/desktop/package.json` defines:

```text
scripts.build = node scripts/assert-root-install.mjs &&
                node scripts/write-build-stamp.mjs &&
                tsc -b && vite build &&
                node scripts/bundle-electron-main.mjs &&
                node scripts/stage-native-deps.mjs
```

`apps/desktop/tsconfig.json`:

- includes `src` and `../shared/src`;
- enables `jsx: react-jsx`;
- sets no `noEmit` option;
- sets no `outDir` option.

Consequently, `tsc -b` emits `.js` beside the included `.ts`/`.tsx` sources.

A representative pair confirms the transform:

```text
tracked source: apps/desktop/src/app/chat/composer/attachments.tsx
untracked emit: apps/desktop/src/app/chat/composer/attachments.js
```

The emitted file removes TypeScript types and compiles JSX to `react/jsx-runtime` calls while preserving the implementation. This is compiler output, not independently authored JavaScript.

### Timestamp evidence

All 694 outputs have modification timestamps within:

```text
minimum: 2026-07-09 20:37:38.172630 local
maximum: 2026-07-09 20:37:39.134843 local
```

Related build artifacts immediately follow:

```text
apps/desktop/tsconfig.tsbuildinfo: 20:37:39.141349
apps/desktop/dist directory:       20:37:42.323366
apps/desktop/release directory:    20:37:44.805597
```

The tracked desktop `package.json` was updated at `20:36:03.721565`, and Git reflog records `main` moving to `origin/main` at `20:36:04`. The sequence is consistent with a managed update followed by a desktop TypeScript/Vite/package build. The exact command-launch log was not found, so the report labels the launcher attribution as a high-confidence reconstruction rather than a direct log quotation.

### Ignore-rule evidence

The repository intentionally ignores canonical output/dependency paths:

```text
.gitignore:45  node_modules/
.gitignore:69  apps/desktop/build/
.gitignore:70  apps/desktop/dist/
.gitignore:71  apps/desktop/release/
.gitignore:72  apps/desktop/*.tsbuildinfo
```

The 694 in-place `src/**/*.js` compiler emissions do not match those rules. They therefore appear as untracked despite being reproducible build artifacts. Classification is based on the compiler/build relationship and complete source mapping, not extension alone.

## Category C — unique local work

Count: 0.

No file requires preservation as unique authored source. The intentional source, tests, package boundaries, and project structure are the tracked TypeScript/TSX counterparts. Generated tests are compiler emissions of tracked `.test.ts`/`.test.tsx` inputs.

## Category D — unknown

Count: 0.

No file remains uncertain. Every output has a tracked source counterpart and participates in the complete TypeScript emission pattern.

## Branch and commit comparisons

Compared by Git object inspection without checkout:

- live `main` / `540f90190f50f9518bf36632a724e0e58877a10b`;
- local `origin/main` ref;
- `feature/hermes-obs-001`;
- `feature/hermes-orch-000-candidate` / `980553ea303feb6d904234a58fbfca96e17c1382`;
- all other known local and remote refs (22 unique commit trees total);
- current remote `main` advertised at `5ecc07986f46463ca3096679b03a46402eb19cee`, inspected through the GitHub tree API without fetching or changing refs.

Results:

```text
same exact path + same blob: 0
same blob at another tracked apps path: 0
exact .js path absent from candidate: 694
same-stem tracked .ts/.tsx source in candidate: 694
candidate changes under apps/: 0
```

`apps/` is a normal tracked product tree in Hermes history. Representative history includes desktop/profile source introduced by earlier merged work and subsequent tracked source changes. What is anomalous is only the in-place JavaScript emission, not the existence of the desktop/shared packages.

## Hash manifests

Created in the clean candidate worktree:

- `docs/evidence/HERMES-ORCH-000-APPS-UNTRACKED-MANIFEST.json`
- `docs/evidence/HERMES-ORCH-000-APPS-UNTRACKED-MANIFEST.csv`

For every file they record:

- stable relative path;
- byte size;
- SHA-256;
- UTC modified timestamp;
- type;
- exclusive classification and reason;
- exact tracked-blob comparison results;
- same-stem tracked source path, commit, and blob;
- ignore/build evidence;
- candidate relationship;
- activation outcome;
- preservation recommendation.

No file contents or credentials are included.

## Candidate collision analysis

| Outcome | Count |
|---|---:|
| Overwritten by candidate | 0 |
| Checkout/fast-forward refused due to same-path tracked file | 0 |
| Left untracked and untouched | 694 |
| Combined/ambiguous state | 0 |

Evidence:

- every exact `.js` path is absent from candidate;
- `git diff --name-only live..candidate -- apps/` returns zero paths;
- the candidate modifies only OBS ledger/evidence files outside `apps/`.

A fast-forward to the candidate would not overwrite these files and would not create an untracked-file collision. It would leave all 694 artifacts in place, so the live working tree would remain dirty. This is operationally undesirable even though it is not a Git collision.

## Likely origin of the `apps/` tree

High-confidence conclusion: an in-place TypeScript build generated the files during the post-update desktop build on 2026-07-09.

Evidence chain:

1. Reflog records `main` reset to `origin/main` at 20:36:04.
2. All 694 `.js` files were created/modified at 20:37:38–20:37:39.
3. `tsconfig.tsbuildinfo` follows at 20:37:39.
4. `dist` follows at 20:37:42.
5. `release` follows at 20:37:44.
6. The package build command runs `tsc -b`, then Vite, then Electron staging.
7. All 694 outputs map one-to-one to the non-declaration tracked TypeScript inputs.

This is consistent with a desktop build invoked after the update, not a partial source extraction, manual copy, or unique local implementation.

## Unique local work findings

None. The tree is internally complete as generated output, but it is not the canonical source tree. Canonical ownership remains with the tracked `.ts`/`.tsx` files and package/build configuration.

## Unknown files and reasons

None.

## Preservation recommendation

Category B disposition for all 694 files:

1. Preserve a timestamped archive outside the repository if an audit/rollback artifact is desired.
2. Remove the in-place generated `.js` files only after explicit user authorization.
3. Do not add them to Git.
4. Regenerate them from tracked TypeScript with the established desktop build only when required.
5. Consider a future, separately reviewed ignore/build fix so `tsc -b` does not dirty managed source checkouts; that is outside HERMES-ORCH-000 classification scope.

No archive, removal, move, rename, stage, stash, or commit of live `apps/` files occurred in this task.

## Safe activation options

### Recommended

- Create a timestamped archive outside the live repository.
- Verify archive count/hashes against the committed manifests.
- With explicit authorization, remove only the 694 manifest-listed generated `.js` files.
- Confirm the live repository is clean.
- Resume the controlled activation preflight.

### Technically possible but not preferred

Because the candidate has zero same-path collisions, a fast-forward could leave all 694 files untouched. This would violate the activation milestone's clean-worktree precondition and preserve ambiguous generated runtime state, so controlled activation should not resume this way without the user explicitly changing that precondition.

## Unsafe actions

Do not:

- use `git clean`, reset, restore, stash, or broad recursive deletion;
- delete `apps/` or any tracked `.ts`/`.tsx` source;
- add generated `.js` files to Git;
- overwrite the current live auth/config/profile state;
- activate while assuming a dirty tree is harmless merely because no candidate path collides.

Any approved cleanup must be manifest-scoped to the exact 694 paths and preceded by preservation/verification.

## What was verified

- Exactly 694 untracked files were enumerated and sorted.
- Every file was hashed with SHA-256 and sized/timestamped.
- Every file received exactly one classification.
- All known Git refs/commits and current upstream were compared without checkout.
- All 694 exact candidate paths are absent.
- All 694 have tracked TypeScript/TSX counterparts.
- Build, compiler, ignore, timestamp, and completeness evidence converge on generated output.
- Candidate activation would leave all 694 untouched and would not overwrite/refuse on these paths.
- The live gateway remained running and the live repository was not modified.

## What failed

No classification failed. No file remains unknown.

One evidence limitation remains: no direct update/build log line naming the exact process invocation at 20:37 was found. The origin conclusion is nevertheless high confidence because the complete compiler mapping, build configuration, and timestamp sequence independently identify the output mechanism.

## Current exact state

- Live repository: still `main` at `540f90190f50f9518bf36632a724e0e58877a10b` with the same 694 untracked generated `.js` files.
- Live gateway: remained running and untouched.
- Candidate branch: `feature/hermes-orch-000-candidate`.
- Candidate activation: not performed.
- Live files removed/moved/renamed/staged/stashed: none.

## Remaining blocker

The technical ownership question is resolved. The remaining blocker is authorization for the preservation and manifest-scoped removal disposition required to restore a clean live worktree.

## Next actionable step

Authorize a separate preservation/cleanup milestone that archives the exact 694 manifest-listed files outside the repository, verifies the archive against the manifest, removes only those generated outputs, confirms a clean live tree, and then resumes controlled activation. Do not begin HERMES-ORCH-001.
