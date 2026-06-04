# 2026-05 EVA human-cast neon batch + Rednote publication notes

## Trigger

Nick asked for a seven-character Evangelion human/mascot batch in the established neon sticker-bomb style, then asked to continue generation and finally publish everything to Xiaohongshu/Rednote.

Characters:

1. 葛城美里 / Misato Katsuragi
2. 渚薰 / Kaworu Nagisa
3. 碇源堂 / Gendo Ikari
4. 赤木律子 / Ritsuko Akagi
5. 片片 / Pen Pen
6. 真希波 / Mari Makinami
7. 独眼明日香 / Eyepatch Asuka

## Durable generation lessons

### 1. Preserve the manifest first

For long multi-image batches, the sidecar manifest must remain the source of truth. Before retrying, read the existing manifest and skip any `status: done` item whose path exists. Do not overwrite a partial manifest with a fresh pending list.

### 2. CLIProxyAPI image_generation payload pitfall

On this session, `tool_choice` with `image_generation` returned:

```text
Tool choice 'image_generation' not found in 'tools' parameter.
```

The more reliable local Responses API payload was:

- `model`: `gpt-5.5` for the dialogue/request wrapper
- `tools`: one `image_generation` tool with `model: gpt-image-2`
- include `partial_images: 1`
- omit `tool_choice`
- `size`: `1024x1536`
- `quality`: `medium`
- `output_format`: `png`
- `background`: `opaque`

Without `tool_choice`, some requests may complete with only text messages and no image. The practical recovery is not to add `tool_choice`; instead compact and simplify the prompt, then retry only that index.

### 3. Compact prompts beat long prompts when the gateway hangs

The first long, rich prompts hung for several minutes. Shorter prompts with essential identity cues + dense neon style block succeeded. For difficult IP batches, prefer a compact 500–900 word/character prompt that includes:

- exact subject identity and 4–6 visual anchors
- one distinct composition cue
- `dense neon cyber-pop sticker-bomb`, `thick black manga ink`, `glossy highlights`, `cyan/magenta rim light`, `halftone`, `torn tape`, `barcodes`, `warning labels`, `huge cropped typography`
- non-sexualized/covered constraints for pilots/humans
- in-scene `NickZag` as sticker/patch/label, not watermark

Do not keep adding prose to force compliance; if the image tool stalls, shorten.

### 4. Retry failed characters individually

A batch may produce a mix of done, failed, text-only, and hung items. Resume with a script that skips done items and retries only failed/pending indices. For stubborn single characters, use progressively shorter prompts and preserve the intended index in the manifest.

### 5. If a named IP prompt repeatedly returns text-only, retain visual cues but reduce explicit franchise dependence

For the eyepatch Asuka item, retries with the explicit name produced text-only outputs. A final retry succeeded after reducing the phrasing to a visual descriptor:

```text
fierce red-haired female mecha pilot, one black eyepatch, one visible bright blue eye, full red futuristic pilot bodysuit, diagonal red spear handle, dense neon stickerbomb background
```

Keep enough visible anchors for recognition, but avoid making the whole prompt depend on the IP/name when the gateway refuses to call the image tool.

## Publication lesson

When Nick says `全部发布到小红书` after a generated numbered manifest, interpret it as:

- publish every `status: done` manifest item to the default Xiaohongshu/Rednote target
- one character per separate message
- attach the image as native `MEDIA:/absolute/path`
- include Xiaohongshu-style Chinese copy, not technical prompt text
- do not include image2skill links unless requested

Default target used here: `discord:1502172364987830393`.

## Worked copy pattern

Use the `ip-xiaohongshu-character-copy` skill. For each post:

- title line: short temperament title + character name + emoji
- 4–5 short paragraph blocks
- story/temperament centered, not prompt-centered
- no repeated `画面里/这张图`
- close with concise hashtags: IP name, character name, AI绘画, 角色海报, theme tags
