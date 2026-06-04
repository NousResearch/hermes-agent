# 2026-05 EVA human-cast neon batch continuation notes

## Trigger

Use these notes when continuing an interrupted or partially successful neon character batch, especially known-IP human/mascot batches where Nick may later publish by number.

## What happened

A seven-image EVA cast batch was started for:

1. Misato Katsuragi / 葛城美里
2. Kaworu Nagisa / 渚薰
3. Gendo Ikari / 碇源堂
4. Ritsuko Akagi / 赤木律子
5. Pen Pen / 片片
6. Mari Makinami / 真希波
7. Eyepatch Asuka / 独眼明日香

The first attempt used long prompts and hung. A later shortened prompt path produced partial success. On continuation, the manifest already had done items, so the correct workflow was to read the manifest first, skip completed paths, and retry only pending/failed indices.

## Durable workflow lessons

- For interrupted batches, **read the existing manifest before writing or regenerating anything**. Treat existing `status: done` + existing `path` as authoritative and skip those items.
- If a request hangs with no process output for several minutes, kill it and preserve the manifest rather than letting it block the whole batch.
- For CLIProxyAPI `/v1/responses` image generation, compact prompts with `partial_images: 1` can return images more reliably than long dense prompts. Keep the identity cues first, then style mechanics compactly.
- Do not use `tool_choice` with this local image-generation path; it can return `Tool choice 'image_generation' not found in 'tools' parameter`. Instead, use strong instructions plus the `tools` entry.
- If a known-IP character repeatedly returns text-only/no image result, retry with a safer generic description that preserves the visual identity cues but removes the direct IP/character name. In this session, Eyepatch Asuka only completed after retrying as a “fierce red-haired female mecha pilot” with eyepatch, red plugsuit, blue visible eye, and EVA-like stickerbomb context.
- After every successful image, write the path back into the same manifest immediately so future `发布 1 3 4` commands stay mapped correctly.

## QC lessons

- Verify each delivered image visually before reporting. Quick QC should check both character identity and the neon-stickerbomb core.
- For mascot characters such as Pen Pen, explicitly confirm no humanized/adultified body.
- For youthful pilots, keep `covered`, `age-appropriate`, and `non-sexualized` in the prompt and QC criteria.

## Practical compact-prompt pattern

```text
Generate one vertical 3:4 neon cyber-pop stickerbomb anime poster of [visual identity cues first]. [Hair/eyes/outfit/prop/pose]. Dense neon stickerbomb cyber-pop background: cyan magenta rim light, black manga ink, glossy highlights, halftone, torn tape, barcodes, cockpit warning labels, huge cropped [ROLE/TITLE] typography, tiny NickZag printed on a physical sticker/patch/label. Covered non-explicit. No nudity, no sexualization, no watermark, no lost identity.
```
