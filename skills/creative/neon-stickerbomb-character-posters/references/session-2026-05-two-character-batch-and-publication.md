# 2026-05 two-character neon batch + publication notes

## Context

Nick requested a small named-character neon batch: `哈利奎恩, cammy white`, then selected `发布12`.

## Generation lessons

- Use the exact named characters, not a generic known-female set.
- For two-character batches, still rotate composition strongly:
  - Harley Quinn: foreground prop + depth, oversized mallet, red/blue split-light chaos poster.
  - Cammy White: Dutch-angle action, red gauntlet foreground, twin-braid diagonal motion.
- Keep adult human characters covered/non-explicit even when their canon designs are often sexualized. Use cyber-streetwear/tactical layers, leggings, jackets, straps, patches, gloves and prop labels instead of exposed-skin emphasis.
- `image_generate` via CLIProxyAPI can time out at 300s for one item while another parallel item succeeds. Do not abandon the batch; retry only the failed item with the same prompt and then save the manifest with successful paths.

## Manifest pattern

Saved manifest:

```text
~/.hermes/profiles/jea/state/neon_named_female_batch_20260507_2017.json
```

Required fields remained: `batch_id`, `skill`, `folder`, `model`, and item-level `index`, `title`, `semantic`, `slug`, `prompt`, `status`, `path`.

## Publication result

Selected publication `发布12` mapped to this latest manifest and imported both images to Eagle `neon` with `image2skill` tags, then rebuilt image2skill.

Eagle IDs:

- Harley Quinn — `MOVGWQ3OAQ3R8`
- Cammy White — `MOVGWQNJLJSYR`

The image2skill rebuild generated short sanitized detail pages instead of full semantic filenames:

- `harley-quinn-foreground-aq3r8.html`
- `cammy-white-dutch-ljsyr.html`

Future verification should discover generated pages by title/ID prefix when the rebuild's title sanitizer/slugger produces shortened pages, rather than assuming the full semantic filename page exists.

## Verification checklist from this session

- Verify Eagle item IDs with `/api/item/info?id=<ID>` because Eagle folder/keyword indexes may lag.
- Confirm `image2skill` and `prompt-saved` tags are present.
- Rebuild with `python3 /Users/nick/.hermes/profiles/jea/outputs/image2skill-redesign/rebuild_image2skill.py`.
- Extract inline JS for `node --check` while skipping JSON-LD scripts.
- Search homepage/detail pages for leak terms: `/Users/`, `.hermes`, `Eagle`, `localhost`, `discord://`, `openai_codex`, `cliproxyapi`, `prompt-unavailable`.
