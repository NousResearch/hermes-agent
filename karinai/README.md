# KarinAI integration layer

This directory is reserved for KarinAI-specific runtime integration on top of upstream Hermes Agent.

Use this area for product-owned adapters, managed runtime entrypoints, config/templates, prompt rendering, packaging helpers, backend-mediated tool bridges, and tests that should not be mixed into upstream Hermes core unless absolutely necessary.

Current design docs:

- `docs/karinai-runtime-notes.md`
- `docs/karinai-runtime-contract.md`
- `docs/karinai-prompt-branding.md`
- `docs/karinai-patches.md`

Implementation shape:

```text
karinai/
  runtime/      # managed runtime startup/config rendering
  prompts/      # product-facing prompt templates
  config/       # beta tool policy and managed runtime examples
  docker/       # Docker/s6 bootstrap helpers for managed containers
  scripts/      # prompt/branding audit and rendering helpers
  tools/        # backend-mediated capabilities such as schedule intent (future)
```

Product-facing runtime identity should be KarinAI agent. Upstream Hermes should remain the engine/base and be referenced where technically useful, but KarinAI prompt/branding should be template/config driven rather than hardcoded directly into upstream files.

If a change must touch upstream core files, document it in `docs/karinai-patches.md` with the reason and whether it is upstreamable, temporary, or permanent KarinAI behavior.


## Container image release

The KarinAI managed agent runtime image is built from the upstream Hermes `Dockerfile` by `.github/workflows/karinai-image.yml` and published as:

```text
ghcr.io/bambak-org/karinai-agent:<commit-sha>
```

Staging should pin this commit-SHA tag or a digest in `karinai-infra` rather than using mutable tags.
