# Session 2026-05 — Onmyoji batch retry and manifest preservation

## Context

Nick asked to generate four Onmyoji / 阴阳师 characters in the existing glossy neon sticker-bomb poster style. The batch used direct CLIProxyAPI `/v1/responses` image generation with a sidecar manifest:

- `neon_onmyoji_batch_20260509_123216.json`
- Characters: 酒吞童子 / Shuten Doji, 玉藻前 / Tamamo no Mae, 茨木童子 / Ibaraki Doji, 妖刀姬 / Yoto Hime

## What happened

1. First run generated index 1 successfully, then failed indices 2-4 with HTTP 429 quota/cooldown:
   - `usage_limit_reached`
   - `model_cooldown`
   - `resets_at` and `resets_in_seconds` were provided in the error payload.
2. User later asked to continue generating the unfinished 3 images.
3. A retry script was created, but it reused the same batch manifest path and called `save_manifest(manifest)` before loading prior completed entries. That briefly overwrote the known-good index 1 status/path.
4. After retrying with a fallback dialogue model (`gpt-5.4`) while still using `gpt-image-2` as the image tool model:
   - indices 3 and 4 succeeded;
   - indices 1 and 2 initially hit `Tool choice 'image_generation' not found in 'tools' parameter` during one retry attempt;
   - index 2 was retried separately and succeeded.
5. The manifest had to be manually repaired to mark all four items `done` and restore known paths.

## Reusable lessons

### Preserve manifest before writing

When retrying a partially completed batch, load the existing manifest **before** writing a fresh pending manifest to the same path. Otherwise a retry script can erase successful item paths/status.

Safe retry sequence:

1. Read existing manifest if present.
2. Build `previous_done = {index: item for item in old.items if item.status == 'done'}`.
3. Create new manifest object.
4. Merge previous done items into the new object.
5. Save.
6. Generate only items not marked done.

Do **not** call `save_manifest(new_pending_manifest)` before step 1.

### Retry only failed indices

For a command like “继续生成未完成的3张”, do not regenerate successful siblings. Keep the original index mapping because Nick may later say `发布1234` or `发布24`.

### Treat 429 reset fields as actionable

If CLIProxyAPI returns 429 with `resets_at` / `resets_in_seconds`, use `date` / Python datetime to calculate whether it is worth retrying now. Report cooldown only if no alternative is being attempted.

### Model fallback caveat

Switching the dialogue/model field from `gpt-5.5` to a fallback such as `gpt-5.4` can sometimes help route around cooldowns, while keeping the actual image tool model as `gpt-image-2`. However, provider routing may intermittently return:

```text
Tool choice 'image_generation' not found in 'tools' parameter
```

Treat that as transient/provider-routing/tool-schema incompatibility. Retry only the affected failed index, preferably after the reset time or with the previously working model. Do not let it overwrite completed manifest entries.

## Verification pattern

After generation completes:

- Read the manifest and ensure every requested index has `status: done` and a local `path`.
- Run visual inspection on newly generated images.
- Patch/repair the manifest immediately if a retry attempt overwrote prior successful entries.
- Final response should list the stable manifest path and include media attachments in the same numeric order.
