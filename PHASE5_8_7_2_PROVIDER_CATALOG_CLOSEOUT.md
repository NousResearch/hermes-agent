# Phase 5.8.7.2 — Provider discovery and catalogue closeout

Base: `e034e727d9` (5.8.7.1 inventory).
Branch: `refactor/phase5-8-runtime-consumers`.

## Ownership cut

- **Anthropic:** The registered provider plugin owns the `/v1/models` protocol:
  guarded HTTP, `limit=1000`, cursor traversal, deduplication, repeated-cursor
  termination, API-key and OAuth headers, and the 400 long-context beta retry.
  The remaining application `_fetch_anthropic_models` wrapper resolves account,
  endpoint, and pooled credentials and then invokes the provider's fetch method.
  It does not duplicate pagination. The public `fetch_models` hook remains intact.
- **DeepInfra source:** The registered provider profile owns its single guarded
  `/models?filter=true&sort_by=hermes` source and response decoding.
- **DeepInfra normalized cache:** `models.catalog_deepinfra` owns one positive
  and negative catalogue cache for every surface. Its cache scope includes
  active profile, effective endpoint, and a credential fingerprint; no raw
  secrets appear in cache keys. A failed fetch and an authoritative empty
  catalogue remain distinct. Negative failures expire after 60 seconds.
  Tag filtering handles chat/image/video/audio/embed, explicit surface tags,
  untagged legacy chat fallback, and unserved stubs.
- **Application inputs:** `application_deepinfra_catalog` supplies
  profile-scoped credentials and non-secret endpoint declarations to the
  canonical cache; it does not own an alternate cache or catalogue policy.
- **Consumers:** DeepInfra chat/vision, image and video discovery, chat pricing,
  TTS, STT and their configuration now consume the same catalogue observation.
  Media-specific presentation and generation remain in their respective
  plugins. Existing caller-specific model overrides and explicit endpoint
  precedence are preserved.

The old DeepInfra cache, tagged filtering and Anthropic URL/cursor helpers
have been removed from `hermes_cli.models`. Internal plugin consumers do not
import those former CLI semantic surfaces. No forwarding cache was retained.

## Verification

- Combined Anthropic, DeepInfra, plugin, audio, pricing, cache isolation and
  multiplexed-profile regression gate: **98 passed**.
- New standalone canonical-cache checks: **5 passed**.
- Focused Ruff checks on migrated production and tests: **passed**.
- Targeted Python compilation and unstaged Git whitespace/diff checks:
  **passed**.
- Scoped source search: no remaining internal imports of the removed CLI
  DeepInfra tagged helper or Anthropic pagination constant.
- OAuth header/retry-specific regression gate: **2 passed**.

No CBM/whole-repository scan or full inherited 5.8 baseline is claimed by
this closeout. Existing Phase 6 credential acquisition is intentionally not
migrated.

Next: **5.8.7.3** — Copilot/OpenRouter verification and Nous recommendation
ownership.
