# KarinAI patch log

This fork tracks upstream NousResearch Hermes Agent while carrying KarinAI-specific integration work.

Upstream remote:

- https://github.com/NousResearch/hermes-agent

Local remotes expected in this clone:

- origin: git@github.com:Bambak-org/karinai-agent.git
- upstream: https://github.com/NousResearch/hermes-agent.git

## Patch policy

- Keep KarinAI-specific code isolated under `karinai/`, runtime packaging/config directories, or clearly named docs whenever possible.
- If a core Hermes file must change, document the reason here before or in the same commit.
- Mark each core patch as either upstreamable, temporary, or permanent product-specific behavior.
- Prefer small commits that are easy to rebase or merge across upstream updates.

## Current KarinAI-specific changes

Initial fork setup and managed-runtime scaffolding:

- Added this patch log.
- Added `docs/karinai-runtime-notes.md`.
- Added `docs/karinai-runtime-contract.md`.
- Added `docs/karinai-prompt-branding.md`.
- Added `karinai/README.md`.
- Added `karinai/runtime/` managed runtime config, prompt rendering, startup, and tool-policy helpers.
- Added `karinai/prompts/system.base.md.j2`.
- Added `karinai/config/managed-runtime.env.example` and `karinai/config/tool-policy.beta.yaml`.
- Added `karinai/scripts/` prompt rendering and branding audit helpers.
- Added `karinai/docker/managed-env.sh` for Docker/s6 managed-mode env mapping.
- Added `tests/karinai/test_managed_runtime.py`.
- Added `tests/karinai/test_managed_container_startup.py`.
- Added `plugins/image_gen/karinai-image-gateway/` for managed image-generation gateway calls.

## Core upstream file patches

### `agent/system_prompt.py`

Status: permanent product-specific behavior unless upstream later grows a generic branding/prompt-template hook.

Reason: KarinAI managed mode must present product-facing identity and policy from KarinAI templates, not the upstream default “You are Hermes Agent…” identity or user-editable `SOUL.md`. The patch is gated by `KARINAI_MANAGED_RUNTIME`; normal upstream Hermes behavior is unchanged when the env flag is absent.

Behavior:

- In managed mode, render the KarinAI system prompt from `karinai/prompts/system.base.md.j2`.
- Do not let user-editable `SOUL.md` override managed product policy.
- Do not inject upstream Hermes help/profile guidance into the product-facing managed prompt.

### `gateway/platforms/api_server.py`

Status: temporary/product-specific bridge until a generic managed-runtime/platform-policy hook exists upstream.

Reason: KarinAI runtime-manager must control the beta tool policy for private `/v1/runs` containers instead of trusting user-editable platform tool config. The patch is gated by `KARINAI_MANAGED_RUNTIME`; normal API server toolset resolution is unchanged otherwise.

Behavior:

- In managed mode, use `karinai.runtime.managed_agent_toolsets()` for `AIAgent.enabled_toolsets` and `AIAgent.disabled_toolsets`.
- Outside managed mode, keep the existing `platform_toolsets.api_server`/`hermes-api-server` resolution.

### `tools/image_generation_tool.py`

Status: temporary/product-specific bridge until upstream has a generic managed media-gateway hook.

Reason: KarinAI managed containers must route `image_generate` through the trusted backend image gateway when runtime-manager provides `KARINAI_IMAGE_GATEWAY_URL`, even if a persisted upstream `image_gen.provider` value is stale. Raw image-provider credentials must stay outside the user container.

Behavior:

- In managed mode with `KARINAI_IMAGE_GATEWAY_URL`, force the effective image provider to `karinai-image-gateway`; if the URL is absent, fail image generation closed instead of honoring stale upstream `image_gen` config.
- Outside managed mode, keep normal upstream image-generation provider selection.

### `agent/chat_completion_helpers.py`

Status: temporary/product-specific bridge until upstream has a generic way for managed custom providers to declare Responses backend variants.

Reason: KarinAI managed containers call the trusted KarinAI model gateway, not `chatgpt.com` directly. When that gateway is backed by OpenAI Codex, the agent still needs Codex Responses request-shaping behavior while keeping Codex credentials outside the user container.

Behavior:

- In managed mode, if `KARINAI_MODEL_GATEWAY_API_MODE=codex_responses` and `KARINAI_MODEL_GATEWAY_BACKEND_PROVIDER=openai-codex`, classify the custom gateway as a Codex backend for Responses payload construction.
- Outside that exact managed gateway configuration, keep upstream provider/base-url detection unchanged.

### `docker/stage2-hook.sh`

Status: permanent product-specific Docker bootstrap behavior unless upstream later exposes a generic managed-container bootstrap hook.

Reason: KarinAI managed containers use runtime-manager-provided state/workspace paths instead of the generic `/opt/data` Docker home. The s6 cont-init hook must apply those paths before it creates, owns, seeds, and migrates `$HERMES_HOME`; otherwise the gateway process and bootstrapping step would disagree about runtime state.

Behavior:

- When `KARINAI_MANAGED_RUNTIME` is truthy, source `karinai/docker/managed-env.sh` before `$HERMES_HOME` setup.
- Map `KARINAI_RUNTIME_STATE_DIR` to `HERMES_HOME` and `HOME` for all s6-supervised processes.
- Map `KARINAI_WORKSPACE_DIR` to `TERMINAL_CWD` and `HERMES_WRITE_SAFE_ROOT`.
- Disable the dashboard in managed beta containers through `HERMES_DASHBOARD=false`.
- Render the `karinai-model-gateway` provider config after upstream Docker config migration when `KARINAI_MODEL_GATEWAY_URL` is configured.
- Create the managed workspace and runtime home directories with hermes-user ownership where possible.

### `docker/main-wrapper.sh`

Status: permanent product-specific Docker command routing unless upstream later supports product-managed default command hooks.

Reason: runtime-manager should be able to start a KarinAI agent container with the normal image entrypoint and managed env contract, without passing a fragile long command. Normal upstream Docker behavior must remain unchanged outside managed mode.

Behavior:

- Source `karinai/docker/managed-env.sh` before command routing.
- With no Docker CMD and `KARINAI_MANAGED_RUNTIME=true`, run `python -m karinai.runtime.start_managed` instead of the default `hermes` CLI.
- Add an explicit `karinai-managed-runtime` command alias for orchestrators/tests.
- Preserve existing generic Hermes routing when managed mode is disabled.

### `Dockerfile`

Status: product-specific documentation-only patch.

Reason: the image still uses the upstream s6 entrypoint, but its command-routing comments need to document the managed KarinAI branch so runtime-manager operators know the supported startup path.

Behavior:

- Documents the `KARINAI_MANAGED_RUNTIME=true` no-args startup path.
- Documents the explicit `karinai-managed-runtime` command alias.

### `hermes_cli/config.py`

Status: managed-mode gate is permanent product-specific behavior; the non-fatal seeding write is upstreamable hardening.

Reason: upstream `_ensure_default_soul_md` (new in v2026.7.1) upgrades a legacy-template `SOUL.md` in place during `ensure_hermes_home()`. Managed KarinAI state dirs carry a read-only `SOUL.md` seeded by `docker/stage2-hook.sh`, so the in-place write raised `PermissionError` and crashed agent creation on every `/v1/runs` request (caught by the stage deploy smoke on the v2026.7.1 sync).

Behavior:

- In managed mode (`KARINAI_MANAGED_RUNTIME` truthy), skip SOUL.md seeding entirely — the managed system prompt takes its identity from `karinai/prompts/`, not `SOUL.md`.
- Outside managed mode, an unwritable `SOUL.md` degrades to a warning instead of failing agent startup (seeding is best-effort).
- Tests: `tests/karinai/test_soul_md_seeding.py`.
- Known pre-existing gap (unchanged by this patch, product follow-up): managed prompts replace the identity slot but still include context files, so a stage2-seeded `SOUL.md` can surface under "# Project Context" in managed containers. Consider `skip_context_files` or dropping the stage2 SOUL.md seed for managed images.

### `pyproject.toml`

Status: product-specific packaging patch.

Reason: the managed runtime must work from installed/editable environments, not only from a source checkout used by tests. Product prompt/config/docker helper files also need to be included in package metadata.

Behavior:

- Adds the `karinai` package namespace to setuptools package discovery.
- Ships KarinAI prompt templates, config examples, and Docker helper shell scripts as package data.
- Adds `karinai-agent-managed` as a console-script entrypoint for non-Docker or explicit startup paths.

## Upstream sync checklist

Automated by `karinai/scripts/sync_upstream.sh` (first validated on the
v2026.7.1 sync, PR #15). Policy decisions baked into it:

- **Sync to upstream release tags** (`vYYYY.M.D`, cut roughly biweekly), not the
  tip of `upstream/main` — reproducible target, matches upstream's own release
  gate, and names the PR (`sync: upstream v2026.7.1`).
- **Merge, not rebase** — `main` is published and PR-reviewed; explicit merge
  commits keep sync points auditable. Never squash-merge a sync PR: squashing
  flattens the upstream commits into one patch and destroys the shared history
  the NEXT sync's merge-base depends on.
- **Cadence: every upstream tag.** Conflict cost grows faster than linearly
  with the gap; a 2-week/1,949-commit gap cost 3 conflicted files + one
  renamed-API adaptation, thanks to the `karinai/`-isolation patch policy above.

Cycle:

1. `karinai/scripts/sync_upstream.sh start [<tag>]` — fetches upstream tags,
   creates worktree + `sync/upstream-<tag>` branch off `origin/main`, merges.
2. Resolve conflicts using this file as the map of intentional divergence: a
   conflict hunk that isn't part of a documented KarinAI patch → upstream wins.
   Commit the merge.
3. `karinai/scripts/sync_upstream.sh gates` — `tests/karinai/`, branding audit,
   patched-area tests per-file (CI-style isolation), import smoke.
4. Push, open the PR, let CI run. **Merging is user-gated** (merge commit, not
   squash). If `main` moves while the PR is open, merge `origin/main` back into
   the sync branch (GitHub can't build a merge ref for a conflicted PR, so CI
   silently won't start until you do).
5. After merge: stage deploy + Tier-2 smoke journeys before calling the sync
   landed.
6. Update this file only if KarinAI patches changed shape during resolution.
