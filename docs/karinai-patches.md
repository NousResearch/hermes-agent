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

Initial fork setup and planning docs only:

- Added this patch log.
- Added `docs/karinai-runtime-notes.md`.
- Added `docs/karinai-runtime-contract.md`.
- Added `docs/karinai-prompt-branding.md`.
- Added `karinai/README.md`.

No upstream Hermes core files have been changed yet.

## Upstream sync checklist

1. Confirm the tree is clean or commit/stash local KarinAI work first.
2. `git fetch upstream main`
3. Review upstream changes before merging into the KarinAI branch.
4. Prefer `git merge --no-edit upstream/main` on the public product branch so sync points stay explicit.
5. Resolve conflicts in the smallest possible patches.
6. Run targeted tests around API server `/v1/runs`, Docker/runtime startup, tool policy, cron/scheduler behavior, prompt rendering, and any touched files.
7. Run KarinAI product tests under `tests/karinai/` once they exist.
8. Update this file only if KarinAI patches change.
