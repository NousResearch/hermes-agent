# Plan 006 — Upstream sync, version bump, and GHCR release

Status: implemented on `upgrade/upstream-20261110`
Base: fork `main` @ `ce476f23b1` (tag `fork-tip-20261111`)
Target: `upstream/main` @ `fb0c056dc1` (2026-11-10)

## 1. Where the fork stood

| Fact | Value |
|---|---|
| Merge base | `f97608f178` = tag `v2026.9.24` (v0.21.5), 2026-09-24 |
| Fork-unique commits | 72 (66 non-merge) |
| Upstream-unique commits | **25,659** |
| Fork-unique files | 87 (+14,210 / −86) |
| Newest stable upstream tag | `v2026.9.24` — *already the merge base* |

There is **no stable release tag past v2026.9.24**. `rc.35-v0.21.5` is itself 918
commits behind `upstream/main`, and upstream publishes `latest` continuously. So
"upgrade to the next stable release" is not an available option: `upstream/main`
is the only forward line, and the fork has to pin a commit rather than a tag.

## 2. Do NOT cherry-pick upstream into the fork

The instinct is to replay the 72 fork commits onto upstream. That is the wrong
direction for this fork and would be strictly worse:

- 87 files, but only **7 conflicts** appear when upstream is merged *into* the
  fork. Almost every fork fix lands in a file upstream never touched
  (`docker/*`, `optional-mcps/firecrawl/`, 25 skills, `.github/*`).
- A 72-commit replay re-resolves every one of those conflicts by hand, and each
  rebased commit loses its original blob identity, making later `git cherry`
  audits unreliable.
- The fork's patches are *additive* (new routes, new files, guards). Upstream's
  changes are *structural* (module moves). Additive patches survive a merge;
  they must be re-authored after a rebase.

**Direction used: `git merge upstream/main` into the fork.** `git cherry` then
gives a clean per-commit audit (`-` = already upstream, `+` = genuinely
fork-only), and the fork tip stays reachable at `fork-tip-20261111`.

### Cherry-pick is still the tool for *specific* fixes

Three fork fixes had to move to where upstream relocated their code. That is a
surgical re-application of one commit, not a bulk replay:

| Fork commit | Upstream moved it to | Action |
|---|---|---|
| `ce476f23b1` bare-custom `model.base_url` fallback | `hermes_cli/model_assignment.py` | re-applied by hand (function split across files) |
| `4bbc12c48e` `PROFILE_SCOPED_PREFIXES` | `web/src/lib/api-request-scope.ts` | re-applied by hand (list relocated) |
| all `weixin` patches | `plugins/platforms/weixin/adapter.py` | **auto-merged** — no action needed |

## 3. What upstream restructured (the actual upgrade work)

1. **weixin became a plugin.** `gateway/platforms/weixin.py` is deleted; the
   adapter, env schema and the `hermes gateway setup` flow live in
   `plugins/platforms/weixin/`. Consequences:
   - `hermes_cli/gateway_setup_wizard.py` loses the fork's `_setup_weixin`;
     upstream's plugin `interactive_setup()` owns that flow.
   - `gateway/platforms/weixin_qr_session.py` (fork-only file) imported the
     deleted module in 4 places → re-pointed at `plugins.platforms.weixin.adapter`.
   - the fork's canonical-domain normalisation was auto-carried into the plugin.
2. **Docker build moved from `uv sync` to `pm.build_env`.** The `uv_source`
   stage is gone, so the fork's `uv pip install piper-tts==1.6.0` would fail with
   `uv: not found`. Voice lanes now ride the single resolve as
   `--extra voice --extra edge-tts --extra piper`; piper moves 1.6.0 → 1.8.0.
3. **`tools/lazy_deps.py` retired** (777 → 20 lines). `pm.ensure_import` writing
   PM generations under `$HERMES_HOME/installs` replaced it; the fork's
   VIRTUAL_ENV fix (`41a934af2b`) has no target left.
4. **Dashboard main-model pick extracted** to `hermes_cli/model_assignment.py`.
5. **`--insecure` is inert** (June 2026 fix): a non-loopback bind always requires
   an auth provider, which `dashboard.basic_auth` supplies.

## 4. Version strategy

`pyproject.toml` is `version = "0.0.0"` upstream **by design**. The runtime
version is not read from it; it comes from `/opt/hermes/install-stamp.json`
(`baseVersion`), written by `scripts/write_install_stamp.py` at image-build time
and read by `hermes_cli/__init__.py.__version__` → `hermes_cli/steward.py`.

Consequences, and this is why the fork's GHCR images reported nothing useful:

- The fork's `fork-ghcr-publish.yml` never ran `write_install_stamp.py`, so every
  fork image shipped the all-zero fallback stamp: `hermes --version` → `0.0.0`,
  and `/etc/hermes/image-provenance.json` → `revision: null`.
- It also passed `HERMES_GIT_SHA` as a build-arg, which **no Dockerfile on either
  side declares** (it was a fork-local Dockerfile hack that upstream's rewrite
  deleted). The arg was silently ignored.

**Do not bump `pyproject.toml`.** A hand-edited version diverges from the stamp
and makes `hermes doctor` report a source tree that does not match its image.
Instead the workflow now writes a real stamp, so the image reports the actual
commit. Tag the fork release by date (`fork-20261110`) rather than inventing an
upstream-looking number.

## 5. Is a GHCR rebuild required? — **Yes**

`ghcr.io/jzkk720/hermes-agent:latest` was built from `ce476f23b1` on
2026-10-05. That image is the *old* tree: it has `gateway/platforms/weixin.py`,
`uv sync`, and `tools/lazy_deps.py` with the allowlist. The merged tree is not
compatible with it, so **the compose stack cannot pick up this upgrade by
pulling**; the image must be rebuilt and pushed.

Two further reasons the rebuild is mandatory rather than optional:

1. `docker-compose.*` mount `docker/hermes-config.yaml` and run the image's own
   entrypoint; nothing is bind-mounted over the Python source, so the image *is*
   the code.
2. Voice packages moved from a startup `pip install` into the image build. An
   image built before this change has no `piper`/`edge-tts`, and the new compose
   no longer installs them.

Rebuild path — **CI only, never a local build.** The compose lanes resolve the
image from `HERMES_FORK_IMAGE` and declare no `build:` key, so the workflow is the
single producer. Its push trigger has proven unreliable on this fork, so dispatch:

```bash
gh workflow run "Fork Docker Build and Publish to GHCR" \
  --repo JZKK720/hermes-agent --ref main
```

The run's step summary prints the immutable pin for that build:

```
ghcr.io/jzkk720/hermes-agent:sha-<short>        # immutable tag
ghcr.io/jzkk720/hermes-agent@sha256:<digest>    # strongest pin
```

Deploy that exact build (no rebuild, no `latest` drift):

```bash
HERMES_FORK_IMAGE=ghcr.io/jzkk720/hermes-agent@sha256 <digest> \
  docker compose -f docker-compose.upstream.yml up -d --force-recreate --remove-orphans
```

### Published pin for this sync

| | |
|---|---|
| workflow run | `43066031645` (success, 10m04s) |
| commit | `c633bd9e367a858dca3c41d9f9bcb7dc4b86e56e` |
| immutable tag | `ghcr.io/jzkk720/hermes-agent:sha-c633bd9` |
| digest | `ghcr.io/jzkk720/hermes-agent@sha256:b7d0d0ba05945df6b6df84ebd5c7a94b6c60683a86cbd22669d83347b50516d4` |

The first dispatch (`43062782083`) failed at `docker.build_agent` — see phase 7.

## 6. Phases and gates

| # | Phase | Gate |
|---|---|---|
| 1 | Protect fork tip (`fork-tip-20261111`) | tag exists, `git rev-parse` matches |
| 2 | Merge `upstream/main` in an isolated worktree | only known conflicts; no markers left |
| 3 | Re-home moved patches (weixin plugin, model_assignment, api-request-scope) | imports resolve; AST shows handlers in the right class |
| 4 | Converge deployment files (compose argv, dead env vars, `--insecure`) | `docker compose config --quiet` = 0 on all three |
| 5 | Static checks | `compileall` = 0; `check_lazy_deps_imports.py` = 0; `buildx --check` clean |
| 6 | Test | fork test files run; failures compared against the **fork-tip baseline**, never against zero |
| 7 | Build image | **CI only** (`fork-ghcr-publish.yml`); the workflow must succeed |
| 8 | Rebuild GHCR + redeploy | pin `sha-<short>` or the digest; `docker compose pull` gets it |

Phase 5's `buildx --check` is necessary but **not sufficient**: it validates
syntax and base-image resolution, not whether a `COPY --from` source path is
actually produced by its stage. The first CI run failed at
`python -m docker.build_agent` with
`FileNotFoundError: required agent input missing: /opt/hermes/hermes_cli/web_dist`
because resolving the Dockerfile conflict dropped upstream's three product COPY
lines. The real gate for a Dockerfile is therefore: **diff the merged file against
`upstream/main` and confirm every remaining hunk is intended** — that diff is what
proved the fix (3 intended hunks, nothing else).

Phase 6 matters most: `tests/gateway/test_api_server.py` has **17 failures on the
fork tip before any of this work**. Comparing the merged run against that
baseline (17 vs 17, with 4 more passing) is the only way to avoid either
"we fixed nothing" or "we broke everything".

## 7. Rollback

```bash
# code
git reset --hard fork-tip-20261111

# image (once the previous digest is known)
HERMES_FORK_IMAGE=ghcr.io/jzkk720/hermes-agent:sha-<prev> \
  docker compose -f docker-compose.upstream.yml up -d --force-recreate
```

`data/.env`, `data/config.yaml`, `data/sessions/`, `data/memories/` and the
`hermes-agent_postgres_data` volume are untouched by any step above.

## 8. Known carry-over risks

- **piper 1.6.0 → 1.8.0.** Upstream's `[piper]` pin carries a platform marker the
  old bake had no equivalent for. Voice output should be re-smoke-tested.
- **`--extra voice`** pulls `faster-whisper` (needs `ctranslate2`/`onnxruntime`).
  It is gated off win32-arm64 and darwin-x64; linux/amd64 is supported, which is
  the only platform this image builds for.
- **Weixin CLI wizard** is now the plugin's, not the fork's. If a fork-specific
  prompt is missed, it has to be re-added inside `plugins/platforms/weixin/`.
