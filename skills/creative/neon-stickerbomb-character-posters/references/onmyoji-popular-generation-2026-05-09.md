# Onmyoji popular-character neon generation — 2026-05-09

## Batch

User requested: `生成阴阳师的4个热门角色`.

Chosen characters and composition rotation:

1. Kagura / 神乐 — foreground paper umbrella + depth; wholesome young onmyoji heroine, red-white palette, talismans/shikigami.
2. Ootengu / 大天狗 — circular wing-storm motion ring; black wings, fan, tengu masks, wind/storm labels.
3. Enma / 阎魔 — side profile split + verdict scroll; underworld judge queen, red judgment seals, skull/legal motifs.
4. Susabi / 荒 — diagonal floating star orbit; celestial oracle, star maps, astrolabe, constellation/talisman motifs.

Manifest:

`/Users/nick/.hermes/profiles/jea/state/neon_onmyoji_popular_batch_20260509_140421.json`

Successful paths:

- `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260509_141959_kagura_paper_umbrella.png`
- `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260509_140838_ootengu_black_wings.png`
- `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260509_141159_enma_underworld_verdict.png`
- `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260509_141626_susabi_star_orbit.png`

## Provider/tool quirks

- Using CLIProxyAPI `/v1/responses` with `model=gpt-5.4` and explicit `tool_choice: {"type":"image_generation"}` sometimes failed immediately:

```text
HTTP 400: Tool choice 'image_generation' not found in 'tools' parameter.
```

- Removing `tool_choice` allowed some items to succeed, but one retry returned `No image result found` for Kagura.
- Switching only the dialogue/model field back to `gpt-5.5` while keeping `gpt-image-2` as the image tool succeeded for Kagura.

## Manifest preservation pitfall

The first retry script rebuilt a fresh pending manifest and only generated indices 1 and 4. This temporarily overwrote successful index 2/3 paths from the original run. After retry, repair the manifest from known output logs/paths before reporting or publishing.

Safer future pattern:

1. Read the existing manifest first.
2. Preserve all `status: done` entries.
3. Retry only `failed`/`pending` target indices.
4. Save the merged manifest after each retry result.
5. If a retry script accidentally overwrites done entries, restore from process logs before continuing.

## Visual QA notes

- All four images matched vertical neon sticker-bomb anime poster style with strong subject cues.
- Small text/pseudo-Japanese remained AI-like; acceptable for this style unless Nick asks for exact typography.
- Susabi generated `NickZag` imperfectly as something like `NickZano`; if exact creator text matters for the batch, regenerate that single item with a stronger physical-label instruction.