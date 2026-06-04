# Session note — live-action The Boys batch blocked by Hermes image backend auth/provider mismatch

Use this when a neon sticker-bomb batch fails before any real image output, especially if the prompts are sound but `image_generate` or direct CLIProxyAPI calls fail immediately.

## Durable lesson

This session's failure was **not a prompt/style issue**. It was a Hermes image-backend configuration mismatch:

- `model.api_key` successfully authenticated against `http://127.0.0.1:8317/v1/models`
- `image_gen.cliproxyapi-gptimage2.api_key` returned `401 Unauthorized`
- Hermes `image_generate` also surfaced `CLIProxyAPI image generation failed (502): unknown provider for model gpt-5.5`

## What to do next time

1. If every image in the batch fails immediately, do **not** keep rewriting prompts first.
2. Verify whether the failure is workflow/config, not art direction:
   - compare `model.api_key` vs `image_gen.cliproxyapi-gptimage2.api_key`
   - if the chat key works but the image key 401s, classify it as an image backend auth mismatch
3. If the provider reports `unknown provider for model gpt-5.5` while the main config uses another live chat model (for example `gpt-5.4`), classify it as a provider/model-routing mismatch inside the image backend path.
4. Report the batch as blocked by Hermes image backend config rather than claiming the prompt failed.
5. Preserve the manifest / prepared prompts so the images can be regenerated immediately after config repair.

## Why this belongs here

For neon-image batches, a clean distinction between **prompt failure** and **backend failure** prevents wasted rerolls and bad QC decisions. The right recovery is backend repair first, then rerun the same prepared batch.

## Session artifact

Prepared manifest path from this blocked run:
`/Users/nick/.hermes/profiles/jea/state/theboys_liveaction_neon_20260521_2003_2up.json`
