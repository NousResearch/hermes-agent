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

Initial fork setup only:

- Added this patch log.
- Added `docs/karinai-runtime-notes.md`.
- Added `karinai/README.md`.

No upstream Hermes core files have been changed yet.

## Upstream sync checklist

1. `git fetch upstream`
2. Review upstream changes before merging into the KarinAI branch.
3. Merge or rebase intentionally, resolving conflicts in the smallest possible patches.
4. Run the upstream Hermes test suite relevant to touched areas.
5. Update this file if KarinAI patches change.
