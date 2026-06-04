# Dragon Ball female neon batches + Rednote mirror delivery — 2026-05-19

## What happened
Nick asked to publish the just-generated Launch / Android 21 / Pan / Marron batch plus Kale / Kefla to Xiaohongshu, then continue generating four more Dragon Ball female characters.

## Durable workflow lessons

1. **Selection can span multiple recent manifests.**
   - The publish request named characters, not only numeric indices.
   - Resolve each named character back to its manifest/path before sending.
   - Preserve the new follow-up batch as its own manifest; do not overwrite previous batch manifests.

2. **Use the Rednote mirror target explicitly.**
   - A bare `send_message(target="discord")` can fail or choose the wrong/home Discord session.
   - For Xiaohongshu/Rednote mirror publication, call `send_message(action="list")` if needed and then send to `discord:#rednote`.
   - Send each selected image as its own message with short Chinese mood copy and one `MEDIA:/absolute/path` attachment.

3. **Continue generation in parallel with publication when independent.**
   - The selected images were already generated and local paths were known, so publishing them and generating the next batch could proceed in parallel.
   - If any publication send fails, do not discard successful generation; record the publication status in the manifest or final reply.

4. **Dragon Ball female follow-up roster used here.**
   - Prior female batch: Launch, Android 21, Pan, Marron.
   - Next partial batch published: Kale, Kefla.
   - Follow-up generated batch: Fasha / Seripa, Towa, Vados, Heles.

## Copy style used for Rednote mirror
Keep captions short and character-centered. Avoid mentioning the neon/stickerbomb style explicitly. Example patterns:

- `一秒切换气场的双面女孩，糖果色外壳下面是随时点燃的火药味。`
- `融合后的战斗笑容太张扬，绿色能量像把整张海报点燃。`

## Manifest convention
For the follow-up batch, manifest name pattern:

`neon_dragonball_female_<character-slugs>_YYYYMMDD_manifest.json`

Include `published_to_rednote` when a generation batch also completes publication of earlier selections in the same turn.
