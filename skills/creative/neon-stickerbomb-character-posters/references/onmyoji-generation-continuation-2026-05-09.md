# Onmyoji neon batch continuation — 2026-05-09

## Scenario

Nick asked for four Onmyoji characters in the established neon sticker-bomb poster style. The batch was interrupted by provider limits after one successful image and later continued.

Characters and composition rotation:

1. 酒吞童子 / Shuten Doji — foreground sake gourd + depth
2. 玉藻前 / Tamamo no Mae — side-profile split + circular fox-tail ring
3. 茨木童子 / Ibaraki Doji — low-angle fashion diagonal with demonic arm
4. 妖刀姬 / Yoto Hime — Dutch-angle cursed-blade slash ring

Manifest:

- `/Users/nick/.hermes/profiles/jea/state/neon_onmyoji_batch_20260509_123216.json`

## Lessons

1. **Write and preserve the manifest early.**
   - The first run completed item 1, then items 2-4 failed with HTTP 429 usage/cooldown.
   - Keeping a manifest with prompts and paths made later continuation and `发布1234` mapping recoverable.

2. **Retry only failed/pending items.**
   - Do not regenerate successful siblings.
   - Patch the retry script to skip known `status: done` items and update only failed/pending indices.
   - After a retry accidentally overwrites item statuses, restore known successful paths from prior logs before proceeding.

3. **Provider/model workaround.**
   - A retry using a different chat model route (`gpt-5.4` instead of `gpt-5.5`) generated items 3 and 4 while some `tool_choice` errors occurred for earlier items.
   - When `Tool choice 'image_generation' not found in 'tools' parameter` appears for only some attempts, treat it as per-request/provider routing instability. Preserve successful results and retry only the remaining target item.

4. **Visual verification.**
   - Use vision verification on completed images to confirm: vertical poster format, subject cues, neon sticker-bomb style, and obvious issues.
   - Expected minor issues: pseudo-text, dense clutter, imperfect small labels. These do not necessarily block use if subject/style are strong.

## Successful final paths

- 酒吞童子 — `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260509_124142_shuten_doji_sake_gourd.png`
- 玉藻前 — `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260509_134636_tamamo_no_mae_fox_ring.png`
- 茨木童子 — `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260509_133933_ibaraki_doji_demonic_arm.png`
- 妖刀姬 — `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260509_134243_yoto_hime_cursed_blade.png`
