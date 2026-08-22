# Canonical Exact-SHA Deployment Capability (CLOE_CANONICAL_EXACT_SHA_DEPLOYMENT_CAPABILITY_V1)

Established 2026-08-05 on openclaw-runtime lah-openclaw-mvp. Solves the recurring
friction where the autonomous roadmap stalls at `docker stop/rm/up` because the
Hermes consent gate blocks container-lifecycle commands even when the mission and
operator already authorized the deployment.

## 1. Consent-gate audit (root cause of the friction)

- `approvals.mode: ask` (config.yaml) → interactive prompt, `timeout: 65` → auto-deny
  when the operator is absent; the prompt expiring is indistinguishable from a denial.
- `security.tirith_enabled: true` → Tirith evaluates destructive commands, fail-closed.
- `tools/approval.py` `detect_dangerous_command()` flags (DANGEROUS_PATTERNS):
  - `docker compose restart|stop|kill|down` → key "docker compose restart/stop/kill/down (container lifecycle)"
  - `docker restart|stop|kill` → key "docker restart/stop/kill (container lifecycle)"
- **Allowlist mechanism EXISTS**: `command_allowlist` in `~/.hermes/config.yaml`.
  `load_permanent_allowlist()` loads entries at startup into `_permanent_approved`;
  `is_approved()` checks pattern keys (the human-readable description strings, plus
  legacy aliases). Entries are the DESCRIPTION strings, not raw regexes.
  Already present in this environment: `["script execution via -e/-c flag"]`.
- Legitimate integration (NOT circumvention): a single canonical script
  `node bin/deploy-lah-openclaw-mvp-exact-sha.mjs <FULL_SHA>` — a fixed script path
  with bounded args — does the whole deployment. To make it persistent, add the two
  docker lifecycle description strings to `command_allowlist` in config.yaml:
  ```bash
  hermes config set command_allowlist '["script execution via -e/-c flag", "docker compose restart/stop/kill/down (container lifecycle)", "docker restart/stop/kill (container lifecycle)"]'
  ```
  This is one-time operator authorization, approved once, reused by Lots 4-8.
- `approvals.mode: smart` (aux LLM auto-approve low-risk) is an alternative middle
  ground; `off`/`--yolo` bypasses everything (not recommended).
- IMPORTANT: the gate also flags `git push --force` / `git reset --hard` / `rm -rf`
  and `python3 <<EOF` heredocs, `sed -i` on Hermes config/env, `chmod +x` + immediate
  `./script` execution. `_normalize_command_for_detection` strips ANSI, backslash
  escapes, empty-string literals and resolves ~/.hermes before matching — do not try
  to obfuscate; get the allowlist entry.

## 2. Architecture (DI-based core + thin CLI)

- `src/deploy/canonical-exact-sha-deployer.mjs` — pure core `deployExactSha(sha, deps)`
  with injectable `{ git, docker, http, fs, env }` for full unit testing. No shell
  calls in the core.
- `bin/deploy-lah-openclaw-mvp-exact-sha.mjs` — thin CLI wiring real backends
  (git, docker, curl, fs), prints a machine-readable JSON status line.
- Status vocabulary (exactly 4 codes):
  - `ALREADY_DEPLOYED` — runtime GIT_COMMIT already == target (idempotent no-op)
  - `DEPLOYED_EXACT_SHA` — deploy + all post-checks pass
  - `DEPLOYMENT_FAILED_ROLLBACK_SUCCEEDED` — post-check failed, previous image restored
  - `DEPLOYMENT_FAILED_OPERATOR_REQUIRED` — fail-closed or rollback failed
- Pipeline: validate 40-hex SHA → SHA on origin/main → checkout clean + HEAD==SHA →
  image GIT_COMMIT==SHA → pre-checks (EXOCLICK_LIVE_ENABLED!=true; ALREADY_DEPLOYED
  short-circuit; record unrelated containers) → env prep (strip stale GIT_COMMIT,
  never print secrets) → record pre-identity → stop+rm+up ONLY lah-openclaw-mvp →
  post-checks (runtime GIT_COMMIT, health 200, RestartCount==0, mounts, tools-runtime
  unchanged, unrelated unchanged) → automatic rollback to previous image on failure →
  sanitized receipt.

## 3. Key semantics decisions

- **qdrant / unrelated containers**: "never remove qdrant" means NEVER remove/restart
  it — NOT "block deployment if qdrant exists". qdrant is a legitimate active stack
  container (cartelogic-remote-memory dependency). Correct design: record unrelated
  container identities BEFORE, verify JSON-identical AFTER (`unrelated_unchanged`),
  include in the pass gate. The initial QDRANT_PRESENT block was a false positive that
  the rehearsal immediately caught — fail-closed is right, but the guard must be the
  no-touch invariant, not a presence check.
- **env prep**: copy canonical .env (secrets store at
  `/home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/.env`), STRIP any
  `GIT_COMMIT=` line so the image/runtime identity is authoritative (stale .env
  GIT_COMMIT overrides runtime identity — known trap). Never echo or persist values.
- **mounts**: canonical data mount `./data:/app/data` (rw) + graph mount
  `/home/deploy/cloe-self-audit-evidence/lot10/graph:/app/data/self-audit/canonical-graph` (ro).
  Post-check verifies data rw AND graph ro.
  **Tracked data must EXIST in the canonical data dir even though the image contains it**
  (observed 2026-08-09, CLOE_READONLY_MARKETPLACE_DATASET_READER_V1): the override FORCES the data
  mount to `CANONICAL_DATA_SOURCE = /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/data`.
  If the canonical checkout is parked on a branch that predates the dataset merge (e.g. a feature
  branch without `data/business/`), the running container cannot see the dataset — `docker exec ls
  /app/data/business` fails — even though the built image contains it, because the bind mount
  overlays `/app/data`. Fix BEFORE deploying: extract/copy the tracked dataset from the merge commit
  into the canonical data dir (`mkdir -p ... && cp -r <worktree>/.../data/business/... <canonical data>`),
  then verify `docker exec <container> ls /app/data/business/...` after deploy. A module-relative
  reader (`new URL('../../data/...', import.meta.url)`) resolves to `/app/data/...` at runtime, so
  the mount source is the authority.
- **health**: `curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:4000/health`.
- **rollback**: recreate from the previous image id recorded at pre-identity; verify
  health again after rollback; if rollback also fails → OPERATOR_REQUIRED.

## 4. Testing pitfalls (DI mock builders)

- **Object-literal spread order**: in a mock factory
  `return { ...defaults, ...overrides, async method() {...} }`, the default methods
  defined AFTER `...overrides` silently CLOBBER method overrides — a custom
  `composeUp` passed via overrides is never used. Correct:
  `const base = {...}; return { ...base, ...overrides };` (overrides applied LAST).
- **Arrow-function closure over the const being built (TDZ)**: `makeDocker({ composeUp:
  async () => { docker.env = ... } })` — the arrow references `docker` while the const
  is still in the temporal dead zone; at call time it can resolve to the wrong object
  and the mutation silently no-ops. Use method syntax `async function() { this.env = ... }`
  so `this` is the receiver.
- **makeDeps double-wrapping**: if `makeDeps` re-runs `makeDocker(overrides.docker)`,
  the test's post-construction mutations on the original object are lost (the deps
  carry a copy). Pass full pre-built mock objects through verbatim when present
  (`typeof x.inspectContainer === 'function' ? x : makeDocker(x)`).
- **Stateful health mock**: to test rollback, the health stub must return 500 on the
  FIRST call and 200 after rollback (closure counter), otherwise the rollback branch
  reports OPERATOR_REQUIRED because post-rollback health is still 500.

## 5. Rehearsal-discovered pitfalls (all fixed + unit-tested, 2026-08-05)

- **`GIT_COMMIT` build-arg shell-expansion trap (observed 2026-08-05, PR #704 deploy)**: a one-liner
  `GIT_COMMIT=<sha> docker compose build --build-arg GIT_COMMIT=$GIT_COMMIT ...` silently builds with
  an EMPTY GIT_COMMIT — the shell expands `$GIT_COMMIT` at PARSE time (before the env prefix applies
  to the docker process), so `--build-arg GIT_COMMIT=` (empty) overrides the compose-file
  interpolation. Symptom: `docker image inspect ... --format '{{range .Config.Env}}{{println .}}{{end}}'`
  shows `GIT_COMMIT=` and the deployer pre-check fails `IMAGE_GIT_COMMIT_MISMATCH`. Fix: pass the
  literal SHA, never a shell variable, on the same line: `docker compose build --build-arg
  GIT_COMMIT=be475f2ff86... lah-openclaw-mvp` (or `export GIT_COMMIT=...` first, then use the
  compose-file interpolation only). Verify the image ENV immediately after build and tag
  `rollout-<sha>` before anything else (BuildKit GC can reclaim the untagged image).

- **`docker ps` template field is `.ID` (uppercase); `docker inspect` uses
  `.Id`**. `docker ps --format '{{.Id}}'` silently fails with "can't evaluate
  field Id" → exit 1 → unrelated-container inventory null → spurious
  fail-closed. Easy to confuse the two; verify with a live command before
  relying on a template.
- **Override file must exist when passed with `-f`** — compose fails
  `open docker-compose.override.yml: no such file`. Guard with `existsSync`
  before adding `-f override`.
- **Previous-image tag fails when the running container's image digest has been
  pruned by rebuilds** (`docker tag <digest> previous` → "No such image").
  Fallback: `docker commit <container> <previous-tag>` (container-sync pattern)
  preserves rollback fidelity. Note the digest may itself be gone (NotFound) if
  layers were pruned — that is an environment state, not a code defect.
- **`./data` relative mount in compose resolves against the WORKDIR**, so
  deploying from a temp worktree mounts temp/data instead of the canonical
  data dir. The deployer generates a canonical `docker-compose.override.yml`
  (data rw at canonical checkout path + graph ro at evidence authority) and the
  file is gitignored, else the checkout is dirty.
- **Deploying when the canonical checkout is dirty (parked on an unrelated branch)**: the CLI
  reads `LAH_DEPLOY_WORKDIR` (default `REPO_ROOT` = the checkout the CLI runs from), plus
  `LAH_DEPLOY_OVERRIDE_FILE` and `LAH_DEPLOY_ENV_SOURCE` (default = canonical
  `/home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/.env`). Point `LAH_DEPLOY_WORKDIR`
  at a CLEAN worktree at the merged SHA (`git worktree add /tmp/<wt> <sha>`) so the deployer's
  `checkout clean + HEAD==SHA` gate passes without touching the dirty canonical checkout; the
  generated override file still mounts the CANONICAL data/graph paths, and the canonical .env is
  copied in by `prepareDeployEnv` (never shell-copy secrets). Verify the deployer ran from the
  intended dir (`docker inspect` mount sources + runtime GIT_COMMIT in post-checks).
- **`LAH_DEPLOY_WORKDIR` must point at the `lah-openclaw-mvp` SUBDIRECTORY, not the worktree root
  (observed 2026-08-08, PR #734 deploy)**: the deployer's `ensureOverrideFile()` writes
  `docker-compose.override.yml` at `join(workdir, 'docker-compose.override.yml')`, and that file is
  gitignored ONLY by `lah-openclaw-mvp/.gitignore`. If you set `LAH_DEPLOY_WORKDIR=/tmp/<wt>` (the
  openclaw-runtime worktree root), the override lands at the REPO ROOT where no `.gitignore` covers
  it → `git status --porcelain` shows `?? docker-compose.override.yml` → deployer fail-closes
  `CHECKOUT_DIRTY` before any mutation. Fix: `LAH_DEPLOY_WORKDIR=/tmp/<wt>/lah-openclaw-mvp` so the
  generated override is covered by `lah-openclaw-mvp/.gitignore`. If a misplaced override already
  exists at the root, removing it is a consent-gated `rm` — the deployer regenerates it at the
  correct location on the next run.
- **Bootstrap path**: when the container is absent (fresh host or after a
  failed recreate), `inspectContainer` throws → the deployer must tolerate a
  null pre-state: skip stop/rm + skip previous-tag, compose up directly, then
  post-check. Rollback is impossible with no previous state → fail-closed
  `BOOTSTRAP_NO_PREVIOUS_IMAGE` if post-checks fail.
- **Manual shell `cp` of .env is itself consent-gated** — the deployer copies
  the canonical .env internally via `prepareDeployEnv`; never copy secrets from
  the agent shell.
- **Deployer can be structurally blocked when the running image's CONTENT is GC'd**
  (observed 2026-08-05, CLOE_DSML_UNICODE_BYPASS_REPAIR_V1): `docker tag
  <current-image-id> previous` → "No such image" AND `docker commit <container>
  previous` → "NotFound: content digest ...: not found". The running container's
  image object AND content are gone (BuildKit GC under disk pressure can remove
  the dangling image; commit cannot snapshot because the rootfs layers are no
  longer addressable). The deployer fail-closes with `PREVIOUS_IMAGE_TAG_FAILED`
  BEFORE any mutation — correct, but there is NO way to pass the gate in this
  state. Recovery: rebuild the currently-deployed SHA from source, tag it
  `rollback-<sha>` (exact rollback authority), then do the controlled deploy
  MANUALLY following the same pipeline (pre-checks → stop/rm → compose up →
  post-checks → rollback via `docker tag rollback-<sha> :latest` + compose up).
  Rebuild digest will differ from the original (npm/apk content drift) — that is
  fine; the tag is the authority, not the digest.
- **A rollback/other build in a different worktree re-tags `:latest` → tag clobber
  deploys the WRONG image**: `docker compose build` names the image
  `<project>-<service>:latest` regardless of worktree. Building the rollback
  image after the rollout image silently re-points `:latest` to the OLD code;
  the next `compose up` deploys it and only the in-container `GIT_COMMIT` check
  catches it (post-check saved the deploy). Mitigations: (1) immediately after
  ANY build, tag the image with an explicit name (`docker tag <id>
  <project>-<service>:rollout-<sha>`) so it survives tag reassignment AND is
  GC-protected (dangling images can be collected by BuildKit GC under disk
  pressure — observed loss of the rollout image mid-mission); (2) ALWAYS verify
  (2) ALWAYS verify `docker exec <container> sh -c 'echo $GIT_COMMIT'` after compose up, not just image inspect.
  - **`GIT_COMMIT=X docker compose build --build-arg GIT_COMMIT=$GIT_COMMIT` silently produces an EMPTY build-arg** (observed 2026-08-05, PR #704 rollout): the shell expands `$GIT_COMMIT` at parse time BEFORE the env-prefix assignment applies to the docker process → `--build-arg GIT_COMMIT=` (empty) → the Dockerfile `ENV GIT_COMMIT=${GIT_COMMIT}` lands EMPTY → the deployer's `IMAGE_GIT_COMMIT_MISMATCH` pre-check (or a manual image-env inspect) is the only thing that stops a deploy of an image with no SHA identity. Pass the LITERAL full SHA as the build-arg (`docker compose build --build-arg GIT_COMMIT=<full-40-hex>`) or `export GIT_COMMIT=...` first. ALWAYS verify the image env right after the build: `docker image inspect <image>:latest --format '{{range .Config.Env}}{{println .}}{{end}}' | grep GIT_COMMIT` — the build "succeeds" with the wrong identity, so inspect-before-deploy is mandatory.
  image inspect.
- **Inline env-prefix build-arg expands EMPTY (shell parse-time trap, observed 2026-08-05):**
  `GIT_COMMIT=<sha> docker compose build --build-arg GIT_COMMIT=$GIT_COMMIT` — the shell
  expands `$GIT_COMMIT` BEFORE the env prefix applies to the docker process, so the
  build-arg arrives as an empty string; the build succeeds but the image ENV is
  `GIT_COMMIT=` (empty). The deployer then fails pre-checks with
  `IMAGE_GIT_COMMIT_MISMATCH` (or, worse, an unlabeled image could be deployed).
  Fix: pass the LITERAL SHA as the build-arg — `docker compose build --build-arg
  GIT_COMMIT=<full-40-hex-sha>` — and verify BEFORE running the deployer:
  `docker image inspect lah-openclaw-mvp-lah-openclaw-mvp:latest --format '{{range .Config.Env}}{{println .}}{{end}}' | grep GIT_COMMIT`
  must equal the target SHA. Tag the rollout image immediately after the build
  (`docker tag … :rollout-<sha>`) for GC protection.
- **Rehearsal sequence that works**: build exact image (GIT_COMMIT=sha) → clean
  worktree at origin/main HEAD → run deployer → expect DEPLOYED_EXACT_SHA →
  run again → expect ALREADY_DEPLOYED. If the tree is dirty (even one modified
  file) the deployer correctly refuses.

## 5b. Build-arg shell-expansion trap (observed 2026-08-05, CLOE_PR703_PR704...)

`GIT_COMMIT=<sha> docker compose build --build-arg GIT_COMMIT=$GIT_COMMIT` EXPANDS `$GIT_COMMIT`
BEFORE the env-prefix assignment takes effect → empty build-arg → image `ENV GIT_COMMIT=''` →
deployer pre-check `IMAGE_GIT_COMMIT_MISMATCH`. The SKILL.md pièges table still shows the
buggy env-prefix form — do NOT copy it. Always pass the LITERAL value:
`docker compose build --build-arg GIT_COMMIT=<full-40-hex-sha>` (or command substitution in
the ARG position, which evaluates before the command runs:
`--build-arg GIT_COMMIT=$(git -C <repo> rev-parse HEAD)`).
Verify the image env AFTER the build and BEFORE the deployer:
`docker image inspect lah-openclaw-mvp-lah-openclaw-mvp:latest --format '{{range .Config.Env}}{{println .}}{{end}}' | grep GIT_COMMIT`,
then immediately tag the rollout image (`docker tag :latest :rollout-<sha>`) against BuildKit GC.

## 5c. Full-disk build failure: image builds but compose metadata write fails (observed 2026-08-08)

- **Symptom**: `docker compose build --build-arg GIT_COMMIT=<sha> lah-openclaw-mvp` ends with
  `write /tmp/.tmp-compose-build-metadataFile-<hash>.json: no space left on device` — the IMAGE
  itself was actually built and tagged `:latest` with the correct GIT_COMMIT; only the compose
  build-metadata temp file failed (disk at 0 bytes). Do NOT conclude the build failed.
- **Recovery**:
  1. Verify the image env immediately: `docker image inspect lah-openclaw-mvp-lah-openclaw-mvp:latest --format '{{range .Config.Env}}{{println .}}{{end}}' | grep GIT_COMMIT` — must equal the target SHA (if it does, the image is usable; if not, rebuild with the LITERAL SHA).
  2. Verify rollback authority intact: `docker image inspect ...:rollout-<prev-sha>` and `...:previous` still exist before any cleanup.
  3. Free space with the MINIMAL safe prune: `docker builder prune -f` (build cache only — no containers, no tagged images, no volumes touched; freed 4.6GB). Prefer this over `docker image prune -a` unless the operator explicitly authorizes image pruning — rollback images are the safety authority and must not be pruned.
  4. Re-run the deployer (it writes override + copies .env, which need a few MB of headroom).
  5. `df -h /` after prune to confirm headroom before the deploy.
- **Environment note (8GB VPS, /dev/sda1 75G)**: disk fills fast with docker images + build cache; check `df -h /` BEFORE starting a build, and treat `docker builder prune` as the first-line recovery (it is reclaimable cache, not state).

## 6. Evidence/SHA-rebind interplay

- Any new deployed SHA requires FULL recollection at the new SHA (local/wiring/tests/
  runtime/shadow collectors from a CLEAN worktree at that SHA — dirty worktree →
  REPOSITORY_DIRTY → local/wiring/tests SKIPPED → false demotions) plus live-verdict
  imports bound to the new deployed SHA.
- Evidence wiring chains can go stale when a route starts calling a new function:
  `high-roi-capabilities.mjs` declared `CALLS getAdvertiserZoneStats(` but the route
  now calls `fetchAdvertiserZoneStats` → wiring collector reports WIRING_CHAIN_INCOMPLETE.
  After changing a route's function call, update the capability's `symbols` and wiring
  chain entries in the same PR (evidence-mapping repair).
- The canonical deployer itself rides the SHA-rebind loop: merge → post-merge verify →
  build image with explicit GIT_COMMIT → run the deployer with the merged SHA →
  recollection at that SHA → candidate → promote → ALREADY_CURRENT → hot reload.

## 7. Post-merge drift + build-context pitfalls (observed 2026-08-07, PR #715 final deploy)

- **origin/main may advance by a docs-only commit after the merge → deployer refuses the merge SHA.**
  The deployer validates `originMain === sha` (`SHA_NOT_ORIGIN_MAIN_HEAD`). If another process
  pushes a docs commit (e.g. continuity JSON) onto main after your merge, passing the merge SHA
  fails post-fetch. Reconcile, don't fight it: `git merge-base --is-ancestor <merge-sha> origin-https/main`
  (must be YES), then `git diff --stat <merge-sha>..origin-https/main` — if the delta is docs-only
  (0 src/test changes), deploy origin-https/main HEAD and document the reconcile in the receipt
  (`fix_sha` = merge SHA, `target_sha` = deployed HEAD, delta described). Never force the deployer
  to accept a non-HEAD SHA; the strict check is the safety invariant.
- **node_modules SYMLINK in a worktree breaks `docker compose build`** with
  `COPY . . → cannot replace to directory .../app/node_modules with file`. Worktrees created for
  missions often symlink `node_modules -> <canonical checkout>/node_modules` to share deps; Docker
  sees the symlink as a file and cannot overlay it onto the `npm install`-created directory.
  Fix: build from a clean rsync'd context, never the symlinked worktree:
  `rsync -a --exclude node_modules --exclude .git --exclude data --exclude .env <worktree>/ <ctx>/`
  + write `.dockerignore` (`node_modules\n.git\n.env\ndata`) in the context, then
  `docker build --build-arg GIT_COMMIT=<literal-full-sha> -t <project>-<service>:latest <ctx>/`.
- **Piped-build exit-code trap silently tags the OLD image as the rollout**:
  `docker compose build ... 2>&1 | tail -25 && echo "BUILD_EXIT=$?"` — `$?` is `tail`'s status (0)
  even when the build FAILED, so the follow-up `docker image inspect | grep GIT_COMMIT` reads the
  STALE previous image and `docker tag :latest :rollout-<sha>` pins the rollout tag onto the OLD
  code (a rollback hazard: the next `compose up` would deploy it). Always use `${PIPESTATUS[0]}`
  or `set -o pipefail`, and ALWAYS verify the image ENV GIT_COMMIT equals the target SHA BEFORE
  applying the rollout tag. A wrongly-tagged rollout must be un-tagged (`docker rmi :rollout-<sha>`)
  and re-tagged after a correct build.

## 8. `LAH_DEPLOY_WORKDIR` must point at the SUBDIRECTORY (`lah-openclaw-mvp`), not the repo root (observed 2026-08-08, HERMES_EXOCLICK_PROVIDER_CONTEXT_TO_AEC_REPAIR_V1 PR #734 deploy)

`bin/deploy-lah-openclaw-mvp-exact-sha.mjs` calls `ensureOverrideFile()` which writes `docker-compose.override.yml` at `join(DEFAULT_WORKDIR, 'docker-compose.override.yml')`. That override is gitignored ONLY by `lah-openclaw-mvp/.gitignore` (entry added in commit 0e8d6a9). If you set `LAH_DEPLOY_WORKDIR` to the repo ROOT of the deploy worktree (e.g. `/tmp/<wt>`), the override lands at `/tmp/<wt>/docker-compose.override.yml` — NOT ignored → `git status --porcelain` shows it → deployer fails `CHECKOUT_DIRTY` (`DEPLOYMENT_FAILED_OPERATOR_REQUIRED`) BEFORE any mutation, and the deploy is blocked with no rollback attempt.

Correct invocation:
```bash
LAH_DEPLOY_WORKDIR=/tmp/<wt>/lah-openclaw-mvp node bin/deploy-lah-openclaw-mvp-exact-sha.mjs <FULL_SHA>
```
`DEFAULT_COMPOSE_FILE` and `DEFAULT_OVERRIDE_FILE` are both derived from `DEFAULT_WORKDIR`, so pointing at the subdirectory also picks up the right `docker-compose.yml` and the gitignored override location.

Recovery when the root-level artifact was already generated by a first wrong run: remove ONLY that one generated file (`rm /tmp/<wt>/docker-compose.override.yml`) — it is a generated artifact, not source — then rerun with the subdirectory workdir. Verify with `git status --porcelain` in the deploy worktree before rerunning (expect empty).

## 9. Live-mission deploy pitfalls (observed 2026-08-10, CLOE_CANONICAL_DECISION_CONTINUITY_V1 deploy + live cert)

- **Deployer reported "absent" when the canonical checkout is parked on another branch**: a previous agent run declared `bin/deploy-lah-openclaw-mvp-exact-sha.mjs` absent because the canonical checkout sat on a dirty feature branch (un-pulled). The file IS in origin/main (added in #696/#697). Before declaring any canonical tool missing, verify against origin/main — `git show origin/main:<path> | head` or `git cat-file -e origin/main:<path>` — never trust `ls`/`find` on a stale or parked checkout. Also search maintenance worktrees (`/home/deploy/lah-stack-worktrees/*/`) — canonical tooling often lives there too.
- **`ALREADY_DEPLOYED` short-circuit does NOT verify mounts**: the deployer's idempotent path returns `ALREADY_DEPLOYED` as soon as the running container's GIT_COMMIT equals the target — it skips mount verification entirely. If a manual `docker compose up -d` created the container BEFORE the deployer generated `docker-compose.override.yml`, the data mount points at the worktree's `./data` (temp dir) instead of the canonical data root → `/app/data/memory-events` missing, production store invisible, evidence/data silently wrong. Symptom check: `docker exec <c> sh -c 'ls /app/data/memory-events | wc -l'` returns 0 when the canonical store has records. Fix: `docker inspect <c> --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Mode}}){{"\n"}}{{end}}'` must show the canonical source; recreate with explicit `-f`: `docker compose -f docker-compose.yml -f docker-compose.override.yml up -d <service>`.
- **Sequence rule**: generate the override BEFORE any manual compose up (run the deployer once — even a run that later fails fail-closed still writes the override at the correct gitignored location), or always invoke compose with explicit `-f docker-compose.yml -f docker-compose.override.yml`. Never plain `docker compose up -d` from a temp worktree when canonical mounts matter.
- **Rollback authority when the running image's content is GC'd**: both `docker tag <imageId> previous` and `docker commit <container> previous` fail NotFound when the image content is pruned — the deployer fail-closes `PREVIOUS_IMAGE_TAG_FAILED` before any mutation (correct). Proven recovery: rebuild the OLD deployed SHA from source in a separate worktree and tag it explicitly (does NOT touch `:latest`): `git worktree add /tmp/<wt> <old-sha>` then `docker build --build-arg GIT_COMMIT=<old-full-sha> -t <project>-<service>:rollback-<old-sha> <wt>/lah-openclaw-mvp`, verify the image ENV GIT_COMMIT, then do the controlled manual deploy (stop/rm → compose up with override → post-checks → rollback via the tagged image). A stale `:previous` tag from an unrelated older deploy is NOT the rollback authority for the current SHA — rebuild from source is.
- **Verify `:latest` still points at your rollout after building the rollback image**: building the old SHA with `docker compose build` would re-point `:latest` (tag clobber). Use `docker build -t ...:rollback-<old-sha>` directly, or re-tag `:latest` to your rollout image immediately after and verify both ENV GIT_COMMITs.

## 9. Deployer may be ABSENT at the target SHA — verify before planning the deploy (observed 2026-08-10, CLOE_CANONICAL_DECISION_CONTINUITY_AND_OPERATIONAL_LEARNING_MEMORY_V1 PR #754)

The canonical deployer `bin/deploy-lah-openclaw-mvp-exact-sha.mjs` is documented above and in the
lah-workflow SKILL.md, but it did NOT exist in the checkout at origin/main 8e433bcb
(`ls bin/deploy-lah-openclaw-mvp-exact-sha.mjs` → NO DEPLOYER). The reference documents a capability
that may not be present at the SHA you are deploying.

**Gate before Phase 19:** check for the deployer FIRST:
```bash
ls /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/bin/deploy-lah-openclaw-mvp-exact-sha.mjs
```
- If present → run the canonical pipeline.
- If ABSENT → do NOT stall the mission; fall back to the manual exact-SHA deployment sequence with
  the SAME guarantees (build with literal `--build-arg GIT_COMMIT=<full-40-hex>` — never the shell
  variable form; verify image env immediately; tag `rollout-<sha>`; pre-checks incl.
  EXOCLICK_LIVE_ENABLED!=true; stop/rm/up ONLY lah-openclaw-mvp; post-checks runtime GIT_COMMIT,
  health 200, RestartCount==0, unrelated containers unchanged; record rollback tag).

**Container replacement = explicit operator consent** in this environment (Hermes approvals mode
ask; user rule since 2026-08-10: `docker stop/rm/compose up` requires consent even in a mandated
deploy). Surface a `clarify` with the exact deployment plan (SHA, build-arg literal, post-checks)
before any container lifecycle command. Also check `df -h /` first (8GB VPS, 91% observed → 6.6G
free; docker builder prune is the first-line recovery if the build fails on metadata write).

## 10. Manual-fallback pitfalls confirmed on the 8a00e6e8 deploy (2026-08-11)

- **Protect the rollback authority BEFORE the build, not after.** `docker compose build`
  re-tags `<project>-<service>:latest` as soon as the new image lands. If you only tag
  `rollback-<prev-sha>` AFTER the build, you may be tagging the NEW image. Sequence proven:
  `docker tag <running-image-id> <project>-<service>:rollback-<prev-sha>` FIRST (image id from
  `docker inspect <container> --format '{{.Image}}'`), then build, then verify image ENV, then
  tag `rollout-<sha>`. Verify both tags' Ids afterwards.
- **Deploy .env MUST have GIT_COMMIT stripped — symlinking the canonical .env is a trap.** The
  canonical checkout's `.env` carries a stale `GIT_COMMIT=` line (the deployer's
  `prepareDeployEnv` strips it; a symlink or verbatim copy does not). Compose `env_file`
  injects it and OVERRIDES the image-baked `ENV GIT_COMMIT` → the container reports a WRONG
  runtime identity despite a correct image. Manual fallback that works:
  `awk '!/^GIT_COMMIT=/' <canonical>/.env > <wt>/lah-openclaw-mvp/.env && chmod 600` then
  verify `grep -c '^GIT_COMMIT=' <wt>/.env` == 0, `grep -c '^EXOCLICK_LIVE_ENABLED=false'` == 1,
  and `awk '!/^GIT_COMMIT=/' <canonical>/.env | cmp - <wt>/.env` (byte-identical otherwise).
- **`sftp` uses `-P` (capital) for the port** (`ssh` uses `-p`). `sftp -p 21098` prints usage
  and exits 1 — costs a retry every time.
- `.dockerignore` may be absent at the target SHA: create it in the build worktree
  (`node_modules\n.git\n.env\ndata`) — a fresh worktree has no node_modules, so `COPY . .`
  in the Dockerfile is safe and the build context stays free of .git/.env.
- Verified end-to-end result: runtime GIT_COMMIT == target, health 200, RestartCount 0,
  canonical mounts (data rw + graph ro) preserved, EXOCLICK_LIVE_ENABLED=false,
  unrelated containers byte-identical pre/post, rollback tag intact.

## 10. Proven manual exact-SHA sequence — deployer absent, executed 2026-08-11 (SHA 8a00e6e8, CLOE_LAHB_DUAL_FUNNEL_OBSERVABILITY_DEPLOY_AND_CERTIFY_V1)

Deployer absent at the target SHA AND in the canonical checkout, but PRESENT in older
worktrees (`/home/deploy/lah-stack-worktrees/*/lah-openclaw-mvp/bin/...`). The manual
sequence below ran end-to-end with zero rollback — copy it verbatim:

```bash
# 0. Preflight: origin fetch + ancestry; record pre-identity + unrelated containers
git -C /home/deploy/lah-stack-repos/openclaw-runtime fetch origin --prune --quiet
docker exec lah-openclaw-mvp printenv GIT_COMMIT        # previous runtime SHA
docker inspect lah-openclaw-mvp --format '{{.Image}}'   # previous image id
docker ps --format '{{.Names}}|{{.ID}}' | grep -v '^lah-openclaw-mvp|'   # unrelated inventory

# 1. PROTECT ROLLBACK TAG *BEFORE* ANY BUILD (compose build re-points :latest)
docker tag <previous-image-id> lah-openclaw-mvp-lah-openclaw-mvp:rollback-<prev-sha>

# 2. Clean worktree at the target SHA
git -C <repo> worktree add /home/deploy/lah-stack-worktrees/oc-deploy-<sha> <full-sha>

# 3. Build-context hygiene in the worktree
#    - .dockerignore is ABSENT at many SHAs -> write it (node_modules/.git/.env/data)
#    - docker-compose.override.yml: force the CANONICAL data mount (./data in the base
#      compose is relative to the workdir; from a worktree it would mount temp/data):
#        services.lah-openclaw-mvp.volumes:
#          - /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/data:/app/data
#          - /home/deploy/cloe-self-audit-evidence/lot10/graph:/app/data/self-audit/canonical-graph:ro
#    - .env: do NOT symlink the canonical .env (env_file would inject a stale
#      GIT_COMMIT over the image ENV). Strip ONLY the GIT_COMMIT line, keep 600:
#        awk '!/^GIT_COMMIT=/' <canonical>/.env > <wt>/lah-openclaw-mvp/.env && chmod 600 ...
#        verify: grep -c '^GIT_COMMIT=' (0) + cmp filtered-canonical vs written (identical)

# 4. Build with the LITERAL sha, then verify image ENV + tag rollout IMMEDIATELY
cd <wt>/lah-openclaw-mvp && export GIT_COMMIT=<full-sha> && \
  docker compose build --build-arg GIT_COMMIT=<full-sha> lah-openclaw-mvp
docker image inspect lah-openclaw-mvp-lah-openclaw-mvp:latest --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^GIT_COMMIT='
docker tag lah-openclaw-mvp-lah-openclaw-mvp:latest lah-openclaw-mvp-lah-openclaw-mvp:rollout-<sha>

# 5. Recreate ONLY lah-openclaw-mvp, always with explicit -f compose files
cd <wt>/lah-openclaw-mvp && \
  docker compose -f docker-compose.yml -f docker-compose.override.yml stop lah-openclaw-mvp && \
  docker compose -f docker-compose.yml -f docker-compose.override.yml rm -f lah-openclaw-mvp && \
  docker compose -f docker-compose.yml -f docker-compose.override.yml up -d lah-openclaw-mvp

# 6. Post-checks (all must pass or rollback via `docker tag rollback-<prev-sha> :latest` + up -d)
docker exec lah-openclaw-mvp printenv GIT_COMMIT           # == target SHA
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:4000/health   # 200
docker inspect lah-openclaw-mvp --format '{{.RestartCount}}'          # 0
docker inspect lah-openclaw-mvp --format '{{range .Mounts}}...{{end}}' # canonical data rw + graph ro
docker exec lah-openclaw-mvp printenv EXOCLICK_LIVE_ENABLED           # false
docker ps --format '{{.Names}}|{{.ID}}' | grep -v '^lah-openclaw-mvp|' # unrelated IDs identical
```

Ordering rules that matter: rollback tag BEFORE build; image-ENV verify BEFORE the rollout
tag; `docker compose up` NEVER without `-f docker-compose.yml -f docker-compose.override.yml`
from a worktree. Outcome 2026-08-11: image b57d41a2 (GIT_COMMIT=8a00e6e8), health 200,
RestartCount 0, mounts canonical, unrelated containers byte-identical, EXOCLICK_LIVE_ENABLED=false.

## 11. Sidecar systemd services + deployer interactions (LAH_SAFETY_PACK_V1, 2026-08-12)

Safety-pack deploy (supervisor + governor systemd units beside the container) revealed
four deployer/runtime pitfalls that apply to ANY sidecar service writing state:

- **The canonical deployer's step-3 `IMAGE_GIT_COMMIT_MISMATCH` means "the image must
  ALREADY carry the target SHA" — it does NOT build for you.** `deployExactSha()` verifies
  `verifyImageGitCommit(IMAGE_NAME, sha)` BEFORE the container recreate, so the pipeline is:
  build with the LITERAL `--build-arg GIT_COMMIT=<full-sha>` → verify image ENV →
  tag `rollout-<sha>` → THEN run the deployer. Running the deployer first fails
  `IMAGE_GIT_COMMIT_MISMATCH` against the stale `:latest`. Proven 2026-08-12 (ff8cd44 → 3ae3fe7).
- **systemd `ProtectHome=read-only` blocks a service writing under /home/deploy →
  ENOENT at startup.** Fix: add `ReadWritePaths=<data-root>` to the unit — but the data
  root directory MUST EXIST before `systemctl start`, or systemd fails mount namespacing
  with status 226/NAMESPACE ("Failed to set up mount namespacing: ... No such file or
  directory") and the unit crash-loops (restart counter climbs). Sequence: `mkdir -p` +
  `chown deploy:deploy` first, then `systemctl restart`. Crash-loop signature to look for:
  `systemctl show <unit> -p NRestarts` climbing while `is-active` says `activating`.
- **Untracked runtime data dirs under the repo break the deployer's checkout-clean gate.**
  Sidecar services persist state in `data/<svc>/`; `git status --porcelain` shows it
  untracked → deployer fail-closes `CHECKOUT_DIRTY` before any mutation. Fix: add
  `data/<svc>/` to `.gitignore` (operator-generated runtime state, never committed), then
  the deployer's clean-check passes while the services stay live.
- **`install-safety-services.sh` REPO_ROOT path derivation can double-nest.** A unit script
  at `ops/systemd/` computing `REPO_ROOT="$(dirname "$0")/../.."` lands one level too deep
  when the repo layout is `<repo>/lah-openclaw-mvp/ops/systemd/` — the default `UNIT_DIR`
  becomes `<repo>/lah-openclaw-mvp/lah-openclaw-mvp/ops/systemd` → FATAL. Fix: pass
  `LAH_SAFETY_UNIT_DIR=<abs path to ops/systemd>` explicitly at install time.
- **Secret-masker corrupts `.env` keys like `LAH_SAFETY_AUTHORIZED_TOTAL_CENTS`** even via
  terminal heredoc — rename to `LAH_SAFETY_ENVELOPE_TOTAL_CENTS` (keep reader backward
  compat: `envNumber('NEW') ?? envNumber('OLD')`) and verify with a byte read, not grep
  display. (Full detail in behavioral-operator-simulation masker pitfall.)
