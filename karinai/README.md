# KarinAI integration layer

This directory is reserved for KarinAI-specific runtime integration on top of upstream Hermes Agent.

Use this area for product-specific adapters, runtime entrypoints, config templates, packaging helpers, and tests that should not be mixed into upstream Hermes core unless absolutely necessary.

If a change must touch upstream core files, document it in `docs/karinai-patches.md` with the reason and whether it is upstreamable or permanent KarinAI behavior.
