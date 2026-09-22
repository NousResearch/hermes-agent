# Fork patch inventory (ForgeGuard fork)

The fork-patch re-verification checklist for [step 4 of the sync runbook](./sync-policy.md#4-resolve-conflicts-then-re-verify-the-fork-patch-checklist).
After merging an upstream tag, **explicitly re-verify every item below** is still
present and correct on the merged branch — a clean auto-merge can silently keep
the wrong side, and these are exactly the places most likely to matter to the
fork's own CI/release behaviour.

This inventory is deliberately specific: concise enough to maintain, detailed
enough to catch silent merge damage. Keep it current as the fork's patch set
evolves.

## CI / workflow guards

- [ ] **`contributor-check` upstream-only guard** — since the v2026.8.16 sync
      upstream ships the check as an *unguarded* reusable workflow
      (`.github/workflows/contributor-check.yml`, `workflow_call`-only); the
      fork guard lives at the **`ci.yaml` call site** (upstream renamed
      `ci.yml` → `ci.yaml` in 0.21.0; git rename detection carried the fork
      guard across at the v2026.8.31 sync — verify only `ci.yaml` exists): the
      `contributor-check` job's `if:` includes
      `github.repository == 'NousResearch/hermes-agent'`.
      Verify no other workflow calls `contributor-check.yml` without a guard.
- [ ] **`docker-lint.yml` direct linter invocation** — the forgeguard-ai org's
      actions policy only allows GitHub-authored / org-owned /
      Marketplace-verified actions, and `ci.yml` calls this file as a reusable
      workflow, so a disallowed action here fails the ENTIRE CI run at startup
      (nothing ran on the fork until this was fixed, 2026-08-12). The fork
      runs hadolint (digest-pinned 2.12.0 container) and shellcheck (apt) as
      plain commands with behaviour held equivalent to upstream's
      `hadolint/hadolint-action` + `ludeeus/action-shellcheck` (same
      `.hadolint.yaml`, same severity threshold, extension + shebang file
      discovery). An upstream sync touching this file conflicts here; keep the
      direct invocations and port upstream's substantive changes (versions,
      configs, new lint targets) into them.
- [ ] **`osv-scanner.yml` direct-actions scan job** — the forgeguard-ai org
      policy only allows reusable workflows FROM forgeguard-ai-owned repos
      (external reusable workflows fail the whole CI run at startup with
      "all reusable workflows must be from a repository owned by
      forgeguard-ai"), while external ACTIONS from verified creators are
      allowed. Upstream's `scan` job calls
      `google/osv-scanner-action/.github/workflows/osv-scanner-reusable.yml`;
      the fork's `scan` job invokes the `osv-scanner-action` +
      `osv-reporter-action` ACTIONS directly instead, producing the same
      `osv-results.sarif` under the "OSV Scanner SARIF file" artifact the
      `emit-status` wrapper expects. If upstream changes its reusable-workflow
      inputs (lockfile list, artifact names), port the substance into the
      direct steps — never restore the external reusable-workflow call. The
      same rule applies to ANY new upstream `uses: <owner>/<repo>/.github/...`
      job-level call: audit for it on every sync (caught live 2026-08-16,
      first PR CI run of the v2026.8.16 sync).
- [ ] **`build-runtime-images.yml`** exists at
      `.github/workflows/build-runtime-images.yml`, still matrixes over
      `target: [runtime, cli]`, and pushes both `runtime-*` and `cli-*` tags to
      `ghcr.io/forgeguard-ai/hermes-agent`. It must have **no `push:` trigger** —
      `release-on-merge.yml` is the single merge-time builder (a push trigger
      here re-introduces double builds on qualifying merges).
- [ ] **`build-desktop-client.yml`** exists at
      `.github/workflows/build-desktop-client.yml` with the Linux, macOS and Windows
      jobs, and **no `push:` trigger** (same single-builder rule as above).
- [ ] **`artifact-retention.yml`** exists at
      `.github/workflows/artifact-retention.yml` with its helper
      `.github/scripts/prune_artifacts.py`. Fork-only; upstream has no
      equivalent, so a sync will not carry it — confirm both files survived.
      It must stay **`workflow_dispatch`-only with `dry_run` defaulting to
      `true`**, must keep passing dispatch inputs through `env:` rather than
      `${{ }}` inside `run:` (the job holds contents/packages/actions write),
      and the script must keep its two guards: `PACKAGES` naming only
      `hermes-agent`, and skipping every untagged GHCR version (the
      `buildcache-*` registry cache lives there). If `build-runtime-images.yml`
      ever changes its tag families, update `TARGETS`/`ALWAYS_KEEP_TAGS` to
      match or the prune will delete live tags.
- [ ] **`branch-cleanup.yml`** exists at `.github/workflows/branch-cleanup.yml`
      with its helper `.github/scripts/prune_branches.py`. Fork-only, so a sync
      will not carry it. It must stay **`workflow_dispatch`-only with `dry_run`
      defaulting to `true`**, must keep `keep` defaulting to `dev` (the
      upstream-sync staging branch — deleting it would break
      [`sync-policy.md`](./sync-policy.md)), and must keep refusing to delete a
      branch that has an open PR or unmerged commits unless explicitly forced.
- [ ] **`release-on-merge.yml`** exists at
      `.github/workflows/release-on-merge.yml`, calls `build-runtime-images.yml`
      with `version:` (not the retired `extra_tag`), still carries its
      release gating (`no-release` label + release-relevant-paths check in
      `compute-version`), and still names releases after the `pyproject.toml`
      product semver (`v<hermes-version>`, `-forgeguard.<n>` suffix only on
      re-cuts of an already-released version; scheme since Hermes 0.19.0) —
      `FORK_UPSTREAM_BASE` feeds only the release-notes "Upstream release"
      line, not the tag.
- [ ] ~~**`upload_to_pypi.yml` guards (all three jobs)**~~ — **retired at the
      v2026.8.16 sync.** Upstream retired the brew + pip/PyPI wheel channels in
      0.20.0 and deleted the workflow; the fork accepted the deletion. (Its
      `sign`-job lesson — an explicit `if:` bypasses `needs:`
      success-skip-propagation, so every job needs its own guard — lives on in
      the entries below.)
- [ ] **Upstream-only guards on the tag/schedule-triggered workflows** that would
      otherwise fire for real (or just burn scheduled runs) on the fork:
      `.github/workflows/deploy-site.yml` (`deploy-vercel` job),
      `.github/workflows/skills-index.yml` (BOTH the `build-index` job — guarded
      on `schedule` events so the twice-daily cron doesn't run on the fork — and
      the `trigger-deploy` job), and
      `.github/workflows/skills-index-freshness.yml` (its check job — the
      every-4h cron is upstream-infrastructure-only). Pattern:
      `if: github.repository == 'NousResearch/hermes-agent'` (combined with the
      job's own other conditions via `&&`). **Audit every job in a multi-job
      workflow file individually** — a guard on the first job in a dependency
      chain does not automatically protect a downstream job that has its own
      explicit `if:`. As of the v2026.7.20 sync the guarded set also includes:
      `.github/workflows/js-autofix.yml` (BOTH jobs, `generate-patch` and
      `apply-patch` — the autofix bot pushes with an upstream `AUTOFIX_BOT_PAT`
      and auto-squash-merges its PRs, neither of which exists nor is wanted on
      the fork; `apply-patch` has its own explicit `if:` so it needs its own
      guard) and `.github/workflows/osv-scanner.yml` (BOTH the `scan` job and
      its `emit-status` wrapper job, guarded only for `schedule` events —
      `github.event_name != 'schedule' || github.repository == 'NousResearch/hermes-agent'` —
      so the `workflow_call` from `ci.yml` and manual dispatch still run on the
      fork while the weekly cron stays upstream-only). As of the v2026.8.16
      sync the guarded set also includes: `.github/workflows/install-e2e.yml`
      (ALL THREE jobs — `pick-releases`, `update`, `installer`; the twice-daily
      cron and the version-tag trigger exercise upstream install/update
      channels, and the fork's own release tags match its `v*` tag patterns;
      `update`/`installer` carry explicit `if:` conditions and so need their
      own guards) and `.github/workflows/publish-e2e-evidence.yml` (`publish`
      job — publishes E2E evidence via the upstream-owned
      `GH_IMAGE_SESSION_TOKEN` PAT). `ci-review-comment.yml` and
      `label-rerun.yml` are `GITHUB_TOKEN`-only and deliberately left
      UNguarded — they enhance fork PRs. `docker.yml` and
      `deploy-site.yml`'s `deploy-docs` job carry upstream-side repository
      guards of their own; verify they remain guarded but do not re-add fork
      copies.
- [ ] ~~**`ci.yml` PR concurrency**~~ — **retired-superseded at the v2026.8.31
      sync.** Upstream's `ci.yaml` ships its own `concurrency:` group with
      `cancel-in-progress: ${{ github.event_name == 'pull_request' }}` —
      the same semantics the fork carried. Verify `ci.yaml` still has it.
- [ ] **Larger-runner labels mapped to standard runners** (fork v0.21.0) — upstream
      0.21.0 moved its heavy CI jobs onto PAID larger runners provisioned in
      NousResearch's org: `ubuntu-latest-96-core` (`tests.yml`),
      `ubuntu-latest-32-core` (`js-tests.yml`, `rust-tests.yml`,
      `e2e-desktop.yml`, `nix.yml`, `docker.yml`),
      `ubuntu-latest-32-arm-core` (`docker.yml`) and `windows-latest-32-core`
      (`tests-os.yml` matrix). Those labels do not resolve on the fork, so the
      jobs sit **queued forever** — no error, no failure, just a PR that never
      goes green (caught live on the v2026.8.31 sync PR: four jobs queued 50+
      minutes while every standard-runner job passed). The fork maps each one
      to the standard GitHub-hosted runner, and adapts the settings upstream
      tuned for that hardware:
      `tests.yml` `HERMES_TEST_WORKERS` tracks the runner's core count (4, per
      upstream's own "one worker per core wins" measurement) instead of 96, and
      `js-tests.yml` runs `run-workspace-checks.mjs --concurrency 1` because the
      script's default (one check per core) starves the checks on a 4-core box.
      Timeouts raised to match the slower hardware (tests/js 60, rust 45).
      **On every sync: re-check every `runs-on:` and matrix `runner:` for a
      `*-<n>-core` label an upstream merge re-introduced** — this failure mode is
      silent, so a queued job is the only symptom.
- [ ] **Trigger-stripped upstream-infra workflows** (fork v0.21.0) — the fork
      removes the event triggers whose jobs are entirely upstream-guarded, so
      the Actions tab stops accumulating skipped runs (523 skipped scheduled
      runs by 2026-09-03). Each file keeps `workflow_dispatch` (and
      `workflow_call` where callers exist) so nothing reusable breaks:
      `skills-index-freshness.yml` (dropped `schedule`), `skills-index.yml`
      (dropped `schedule` + `push`), `install-e2e.yml` (dropped `schedule` +
      `push` — its `v*` tag patterns match fork release tags),
      `osv-scanner.yml` (dropped `schedule` only; the `workflow_call` from
      `ci.yaml` is the fork's real scan), `docker.yml` (dropped
      `pull_request`/`push`/`release`; added `workflow_dispatch` — the fork
      publishes GHCR images via `build-runtime-images.yml`, not Docker Hub),
      `js-autofix.yml` (dropped `push`), `publish-e2e-evidence.yml` (dropped
      `workflow_run`; added `workflow_dispatch`), `deploy-site.yml` (dropped
      `release` + `push`), `nix.yml` (dropped `pull_request` + `push` — its
      cachix/nix-community actions are org-policy-blocked and startup-fail
      every run; the fork does not distribute via nix). On every sync:
      re-strip any of these that an
      upstream merge re-grows, and triage NEW upstream workflows for the same
      treatment (a new unguarded workflow with `schedule`/`push`/`release`
      triggers runs or skips on the fork forever). The job-level
      `NousResearch/hermes-agent` guards stay as defense in depth.
- [ ] **`workflow_call` upload/push gating** — in both
      `build-desktop-client.yml`'s "Upload Linux/macOS/Windows installers" steps and
      `build-runtime-images.yml`'s "Push image to GHCR" step, the `if:` must gate
      on `inputs.upload` / `inputs.push` directly. **Do not** reintroduce
      `github.event_name == 'workflow_call'` — `github.event_name` inside a
      reusable workflow is always the *caller's* triggering event (e.g.
      `pull_request` for `release-on-merge.yml`), never literally
      `"workflow_call"`. This exact regression silently skipped every installer
      upload and image push for two releases before it was caught (2026-07-02) —
      the jobs report "success" either way, so this only surfaces by checking the
      *individual step* conclusions, not the job conclusion.

## Docker / build structure

- [ ] **Dockerfile multi-target structure intact** — the fork's `Dockerfile`
      declares `base` / `toolchain` / `venv-runtime` / `runtime` stages, with
      **`runtime` as the LAST stage** (a target-less `docker build .` — compose,
      tests/docker fixture — must keep producing the full supervised image),
      compilers confined to `toolchain`, the
      `com.forgeguard.hermes.prebaked=1` label on the published target, and the
      `HEALTHCHECK` + `docker/healthcheck.sh` still present. If upstream
      restructures its single-stage Dockerfile, re-apply their substantive
      change inside the matching stage rather than reverting the split.
      **The `venv-cli` / `cli` stages and `docker/cli/` were removed in
      v0.20.7** (see the cli-image retirement below); do not resurrect them from
      an upstream diff.
- [ ] **`cli-*` image retirement** (fork v0.20.7) — the `cli` Dockerfile stage,
      its `venv-cli` venv, `docker/cli/{hermes-shim,profile}.sh`, and the `cli`
      leg of `build-runtime-images.yml`'s `target:` matrix are all GONE. Agent
      Command deploys `runtime-*` only, and a broken cli build blocked the whole
      release (v0.20.7's first attempt). Verify the matrix is still
      `target: [runtime]` after a sync, and that no workflow references
      `cli-latest`. Docs: `docs/site/deployment/distrobox-cli.md` carries the
      retirement banner.

## Desktop app

- [ ] ~~**`apps/desktop/vite.config.ts`** test scope fix~~ — **retired at the
      v2026.7.20 sync.** Upstream's `apps/desktop/vitest.config.ts` now defines
      separate `react-ui` (`src/**/*.test.{ts,tsx}`) and `electron-native`
      projects, absorbing the fork's scoping fix, and the electron `.test.cjs`
      node:test suites were ts-ified into vitest suites. `vite.config.ts` is
      upstream-identical again; verify it has NOT re-grown a fork `test:` block.
- [ ] **`apps/desktop/package.json` `"version"`** tracks the Hermes product
      version (`pyproject.toml`), not upstream's stale desktop version — bump it
      (and the `"apps/desktop"` entry in the root `package-lock.json`) on every
      sync that changes the product version.
- [ ] **`apps/desktop/src/app/settings/gateway-settings.test.tsx`** carries the
      fork's `localDesc` wording ("… Works offline.", no "This is the default")
      in its two label assertions — the fork is client-mode-first, so upstream's
      wording (re-)appearing here means an auto-merge silently restored
      upstream's string; re-adapt the assertions, not the fork's i18n.
- [ ] **`apps/desktop/package.json`** has a top-level `"homepage"` field
      (`https://github.com/forgeguard-ai/hermes-agent#readme`) — required by
      electron-builder's Linux `deb` target; its absence fails `dist:linux` with
      `Please specify project homepage`.
- [ ] **`apps/desktop/electron/managed-ssh-update.test.ts`** resolves its
      stand-in hermes binary (`TRUE_BINARY`: `/bin/true` else `/usr/bin/true`)
      instead of hard-coding `/bin/true`. macOS ships `true` at `/usr/bin/true`
      ONLY, so upstream's literal makes the POSIX launcher test exit 127 on the
      fork's macOS desktop build (which upstream never runs — its only macOS
      lane is Python). Blocked the v0.21.0 release, 2026-09-04. If an upstream
      sync restores the literal, re-apply the resolver; better, check whether
      upstream took the fix.

## Carried runtime patches (verify by running their test files)

These fork fixes live inside upstream-owned Python/TS files and are the most
likely to be silently dropped by a clean auto-merge. Each has a test file that
IS the verification hook — run it on the merged branch instead of eyeballing:

- [ ] **`hermes_cli/auth.py` — "no-key" deploy sentinel is a placeholder**
      (fork v0.19.2). `"no-key"` in the placeholder-secret set;
      `"no-key-required"` deliberately stays out. Verified NOT upstream as of
      v2026.8.16 (the fork test fails on a clean upstream checkout).
      Test: `tests/hermes_cli/test_auth_usable_secret.py`.
- [ ] **custom:`<name>` slug handling** (fork v0.19.3):
      `hermes_cli/providers.py::canonicalize_provider_slug`, the
      `custom:custom` bare-bucket collapse in `hermes_cli/web_server.py`'s
      `_normalize_main_model_assignment` (upstream preserves *named* unresolved
      slugs verbatim — keep both behaviors), `resolve_provider`'s bare-slug
      resolution in `auth.py`, and the `/api/providers/validate`
      custom-endpoint key probe. `canonicalize_provider_slug` does not exist
      upstream (the fork test file cannot even import there).
      Test: `tests/hermes_cli/test_custom_provider_slug_canonicalization.py`.
- [ ] **mem0 embedder-bearer scoping** (fork v0.19.1):
      `plugins/memory/mem0/_backend.py` (scoped Ollama-client Authorization
      patch), `_oss_providers.py`, `_setup.py`. Verified NOT upstream as of
      v2026.8.16. Tests: `tests/plugins/memory/test_mem0_backend.py` +
      `test_mem0_setup.py`.
- [ ] **Desktop connection cluster runtime patches** — reshaped by upstream
      0.21.0's connections-registry rework (v2026.8.31 sync). Still fork-owned:
      TLS bypass threading (the `hostAllowsInvalidCertificate` per-host
      fallback + statusCode-carrying errors inside `fetchJson`/
      `fetchPublicJson`, `installCertificateBypass()`, the
      `allowInvalidCertificate` field on `DesktopRemoteRoute` in
      `apps/desktop/electron/desktop-remote-route.ts`, the 8th
      `allowInvalidCertificate` arg of `buildRemoteConnection` — headers is
      9th; upstream calls pass 8 args, so every new upstream call site must be
      re-threaded on sync — plus `fetchConnectionStatus`'s 5th arg and the
      flag in `createPrimaryRemoteConnection`), the 401/403 native-token drop
      in `mintGatewayWsTicket`, and the first-run choice IPC + gates in
      `apps/desktop/electron/main.ts` + the first-run gate wrapper
      (`initGatewayBoot` + `enterFirstRunChoice` + the
      `FIRST_RUN_CHOICE_REQUIRED` safety net) in
      `apps/desktop/src/app/gateway/hooks/use-gateway-boot.ts`.
      **Retired-superseded at v2026.8.31**: the fork's hook-level
      `gateway.ping` heartbeat and bounded initial-connect retries — upstream's
      shared `JsonRpcGateway` now owns the heartbeat (interval + inbound
      deadline, sends `gateway.ping`) and the hook re-attempts whole boots
      with jittered backoff. The fork's server-side
      `@method("gateway.ping")` in `tui_gateway/server.py` stays: upstream's
      ws fast-path (`tui_gateway/ws.py`) answers before dispatch, and the
      dispatch method keeps non-ws transports answerable.
      Tests: `apps/desktop/src/app/gateway/hooks/use-gateway-boot.test.tsx`
      (fork cases), `apps/desktop/electron/first-run-choice.test.ts`,
      `apps/desktop/electron/connection-config.test.ts`,
      `tests/tui_gateway/test_gateway_ping.py`.
      **Registry threading (fork, 2026-09-22):** the TLS opt-in now also rides
      the v2 registry — `allowInvalidCertificate` on `RegistryConnection` /
      `ConnectionInput` in `connection-registry.ts`, set in
      `normalizeConnectionInput`, `mergeConnectionInput` (inherit),
      `connectionDialFieldsChanged`, `normalizeRegistry`,
      `migrateV1ToRegistry` and `reconcileAppliedGlobalConnection` (stored only
      when true, like `headers`); `hostAllowsInvalidCertificate` walks the
      registry too; `installCertificateBypass`'s verify closure is the named
      `gatewayCertificateVerifyProc`, also installed on every per-connection
      cookie jar created in `getOauthSessionForUrl`; the `connections:test`
      handler skips its Node WebSocket leg under the opt-in (as the v1 test
      does); the Connections editor (`src/app/settings/connections-registry.tsx`)
      has the toggle and sends the field; `src/global.d.ts` declares it. Before
      this, a gateway saved in Settings → Connections could never carry the
      opt-in: every registry normalization dropped it, and the registry dial
      and `connections:test` read `false` through a cast. Tests: the
      `allowInvalidCertificate (ForgeGuard fork)` block at the end of
      `apps/desktop/electron/connection-registry.test.ts`, and the two
      self-signed cases in `src/app/settings/connections-registry.test.tsx`.
      Also fork-only: `electron/gateway-probe-errors.ts` (+ test) — the
      onboarding/Settings probe explains TLS alert 112 (`unrecognized_name`,
      a reverse proxy with no site for the hostname) instead of echoing the
      TLS library, since it otherwise reads like a certificate problem
      directly above the self-signed toggle that cannot fix it.
- [ ] **Desktop per-connection cookie moves** (fork, 2026-09-22) — upstream's
      #92183 decides a cookie-flow gateway's jar from the registry's current
      shape (`electron/oauth-partition.ts`, untouched), so a gateway changes
      jars when its status changes and its session stays behind: Settings →
      Connections signs in BEFORE Save (unregistered URL → legacy jar; saved
      non-primary entry → its own empty jar → "not signed in" on switch), and
      a primary flip swaps two gateways' jars. Fork-only modules:
      `electron/oauth-partition-moves.ts` (pure planner: before/after registry
      diff, URLs still registered after the mutation only, legacy-source
      moves first) and `electron/oauth-cookie-move.ts` (moves
      `hermes_session*` cookies against the gateway's own origin; host-only
      cookies re-set without a domain, which `__Host-` requires; skips a
      destination that already holds a live session). Wiring in `main.ts`:
      `migrateOauthCookiesForRegistryChange` + `pendingOauthPartitionMoves`
      beside `getOauthSessionForUrl`; the await at the top of
      `warmOauthCookieStore`'s block; calls in `saveRegistryConnection`, the
      `connections:set-primary` handler, and inside the `apply` callback of
      the `connection-config:apply` handler (after the writes, before the
      re-home dials, reversed if activation throws). Proven against real
      Chromium (Electron 40.10.6) across a relaunch, 2026-09-22. **Retire**
      when upstream moves or re-keys the session on sign-in-before-save and
      on primary changes. Tests: `electron/oauth-partition-moves.test.ts`,
      `electron/oauth-cookie-move.test.ts`, and the fork contract case at the
      end of `electron/oauth-partition.test.ts`.
- [ ] **Compressor output reservation** (fork v0.20.3) —
      `agent/output_reservation.py` (fork-only module: the reservation the
      transport will actually send — user tier or provider-profile default),
      seeded into `agent/agent_init.py` (`_compressor_max_tokens =
      _resolve_agent_output_reservation(agent)`, composed with upstream's
      native-Gemini fallback since v2026.8.31) and threaded through
      `agent/agent_runtime_helpers.py`'s `update_model` call via
      `reservation_kwargs` (inside upstream's try/`_restore_snapshot` wrapper).
      Upstream fixed only the native-Gemini slice (#57275) and calls the
      generic gap #63839 open — take upstream's version if #63839 ever lands.
      Tests: `tests/agent/test_output_reservation.py`.
- [ ] **OpenAI streaming-TTS hardening** (fork v0.20.7) — four sites in
      upstream-owned files: `tools/tts_streaming.py::OpenAIStreamer.stream`
      (endpoint is `tts.openai.base_url` or `DEFAULT_OPENAI_BASE_URL`, never
      `OPENAI_BASE_URL`; `speed`/`lang_code` parity with
      `_generate_openai_tts`), `hermes_cli/doctor_live.py::_audio_provider_probe`
      (`openai_cfg=` — probes `<tts.openai.base_url>/models` with
      `tts.openai.api_key` or the `_openai_audio_key()` seam),
      `hermes_cli/web_server.py::speak_stream_ws` (`{"type":"error"}` instead of
      `end` when the producer raised), and
      `apps/desktop/src/lib/voice-playback.ts::openSpeechStream` (`error` frame
      → `'fallback'` before audio / drain-then-`'done'` after). Verified NOT
      upstream as of v2026.8.16.2. Tests: `tests/tools/test_tts_streaming.py`
      (`test_openai_streamer_*`), `tests/hermes_cli/test_doctor_live.py`
      (`test_tts_openai_probe_*`),
      `tests/hermes_cli/test_web_server_speak_stream.py`
      (`test_provider_failure_*`), `apps/desktop/src/lib/voice-playback.test.ts`.
- [ ] **TTS chunking modes** (fork v0.20.8) —
      `tts.streaming.chunking` (`punctuation` | `paragraphs` | `none`)
      across upstream-owned files:
      `tools/tts_streaming.py` (`SentenceChunker` `mode=` +
      `PARAGRAPH_BOUNDARY_RE` + `resolve_chunking_mode`),
      `hermes_cli/web_server.py::speak_stream_ws` (`_resolve` returns the
      profile-scoped mode, `_produce` builds the chunker with it, `none`
      skips the idle flush), `hermes_cli/config_defaults.py` (default
      `tts.streaming.chunking: punctuation` block), `tools/tts_tool.py` +
      `gateway/streaming_tts_consumer.py` (mode threading), and the desktop
      settings surface (`apps/desktop/src/app/settings/constants.ts` — voice
      key, enum, `TTS_CHUNKING_LABELS`, field copy; `config-settings.tsx` —
      `voiceFieldVisible` early return + option labels). Tests:
      `tests/tools/test_tts_streaming.py` (`TestSentenceChunkerModes`,
      `TestResolveChunkingMode`),
      `tests/hermes_cli/test_web_server_speak_stream.py`
      (`test_paragraphs_chunking_*`, `test_none_chunking_*`),
      `apps/desktop/src/app/settings/voice-field-visible.test.ts`.
- [ ] **Linux app icon set** (fork v0.20.7) — `build.linux.icon` in
      `apps/desktop/package.json` points at `assets/icons/` (new, fork-only:
      16→512 PNGs) instead of inheriting the top-level single-PNG
      `assets/icon`. electron-builder returns a one-entry icon set for a single
      PNG, so the 1024x1024 master installed to
      `hicolor/1024x1024/apps/Hermes.png`, which no icon theme indexes. Keep
      this key if upstream rewrites the `linux` block. Tests:
      `apps/desktop/scripts/linux-icons.test.mjs`.
- [ ] **Security dependency pins ahead of upstream** (fork v0.20.8; re-verified
      v2026.8.31) — the fork pins `electron` **40.10.6**
      (`apps/desktop/package.json` devDependency *and* `build.electronVersion`,
      which two guards require to match:
      `apps/desktop/electron/desktop-electron-pin.test.ts` and the exact-version
      `allowScripts` key in the root `package.json`, guarded by
      `tests-js/allow-scripts-sync.test.ts`). Upstream still pins ^40.10.2, which
      carries CVE-2026-70606. 40.10.6 also drops `extract-zip` for
      `@electron-internal/extract-zip`, removing unpatched CVE-2026-56876 from the
      tree — so regenerate the lock with a real resolve
      (`npm install electron@<v> --package-lock-only -w apps/desktop`), never by
      editing lock entries by hand: a hand-edited entry keeps the OLD dependency
      list and `npm ci` then builds a tree missing `@electron-internal/extract-zip`,
      which only fails on a cold cache. **Workspace staleness trap (hit
      2026-09-03):** `npm install --package-lock-only` can silently keep the
      lock's stale embedded `apps/desktop` manifest (and even revert the disk
      manifest to upstream's `^40.10.2`) — after any lock regen, verify BOTH the
      disk manifest and the lock's `packages["apps/desktop"].devDependencies`
      say the exact fork pin and `node_modules/electron` resolved to it.
      Also ahead of upstream: `h2` **4.4.1** in `uv.lock` (CVE-2026-71554;
      upstream ships 4.3.0 — re-apply with `uv lock --upgrade-package h2`);
      relock with the CI-pinned uv (0.9.28), since a newer uv rewrites the
      lock's options format. **`nanoid` retired-superseded at v2026.8.31**:
      upstream moved the hoisted copy to 6.0.0 and the remaining nested 3.x
      copies resolve to the patched 3.3.18. Take upstream's versions when they
      pass these CVEs.
- [ ] ~~**Desktop voice mic re-arm** (`use-voice-conversation.ts`)~~ —
      **retired-superseded at the v2026.8.16 sync.** Upstream's live-speech
      rewrite (`settleAfterSpeech` → `pendingStartRef` → loop-effect
      `startListening`) fixes the dead-mic class structurally and absorbed the
      fork's no-early-return in the tool-only branch. The fork keeps its
      regression scenario as coverage upstream does not have:
      `apps/desktop/src/app/chat/composer/hooks/use-voice-conversation.rearm.test.tsx`
      (seams adapted to the new hook; verify it stays green).

## Docs / instructions

- [ ] **`README.md`** still surfaces the ForgeGuard fork identity below the
      title/badges block (the compact fork alert linking to `docs/site/` and the
      fork pages).
- [ ] **`docs/site/`** and **`docs/maintainers/`** still contain the ForgeGuard
      documentation overlay: the user-facing pages under `docs/site/` and this
      runbook (plus [sync policy](./sync-policy.md) and
      [conflict resolution](./conflict-resolution.md)),
      [release process](../release/release-process.md),
      [review](../development/review.md), and
      [graphify-refresh](../development/graphify-refresh.md) under
      `docs/maintainers/`.
- [ ] **`AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`** still have
      the "ForgeGuard Fork — Additions Below This Line" section / pointer content
      intact (`AGENTS.md` is the source of truth; the other two are thin pointers
      to it), and their runbook links point at
      `docs/maintainers/upstream-sync/sync-policy.md`.

## Marker

- [ ] **`FORK_UPSTREAM_BASE`** — not a merge-conflict risk (you rewrite it in
      step 6 of the [sync policy](./sync-policy.md#6-update-fork_upstream_base)),
      but don't forget it.

## Related

- [Sync policy](./sync-policy.md)
- [Conflict resolution](./conflict-resolution.md)
- [Release process](../release/release-process.md)
