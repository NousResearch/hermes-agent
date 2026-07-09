---
name: seedance-script-designer
description: Convert scripts, storyboard text, or video concepts into Seedance 2.0 production packages with story breakdowns, 12-column keyframe tables, reference planning, and video prompts.
---

# Seedance Script Designer

## Workflow

Use this skill in Chinese by default unless the user asks for another language.

1. If the user has not specified a visual style, ask one concise clarification question before producing assets.
   - Offer options such as: 电影写实、动漫二次元、插画风、吉卜力式温暖手绘感、新海诚式清透光影感、日漫风、欧美动漫风、赛博朋克、国风水墨、黏土定格、像素风。
   - If the user names a living artist or studio style, preserve the intent by describing visual traits instead of saying to imitate the artist directly.
2. Read the script and extract: characters, locations, props, emotional beats, dialogue, and scene transitions.
   - **If the user provides only a vague idea** (not a script): restructure it into a story premise, protagonist, desire, opposition, stakes, and 3-5 beats first. Continue into the requested output mode with clear assumptions unless visual style is missing.
   - **If the user provides an existing video to continue or extend**: use extension/first-last-frame continuity. Preserve the previous video's character identity, costume, lighting, rhythm, and spatial direction. State which @video is the base and describe only the new segment.
   - **If the input is music-driven or has no dialogue**: leave `台词` and `旁白` columns empty. Focus on music rhythm, `状态/音效`, motion beat points, and camera movement. Map cuts and accents to the beat.
   - **If the user provides an old storyboard table to normalize**: identify column count and structure, then reformat it to the 12-column professional table without changing the content.
3. Read `references/story-structure-method.md` and identify dramatic action before designing shots.
4. Build a consistent design bible before shot prompts: character identity, costume, color palette, age, body type, silhouette, texture, and recurring visual motifs.
   - Before storyboard work, isolate four production layers: character identity, environment, prop/asset, and emotional/cinematic language.
   - For stories spanning time, create age-specific character versions with fixed facial anchors, hair, wardrobe palette, and signature props.
5. Read `references/shot-breakdown-method.md` before creating the storyboard table or Seedance prompts.
6. Produce the output using the structure in `references/output-schema.md`.
7. Keep prompts Seedance-ready: concrete subject, action, camera, lighting, style, composition, motion, mood, duration, aspect ratio if known, continuity notes, and @reference intent.
8. When creating or saving production files, follow the `00剧本` project storage rules below before final delivery.

## Input Handling

Handle non-standard inputs without forcing the user to rewrite them as a complete screenplay:

- Vague idea instead of script: turn it into a concise premise, protagonist, desire, opposition, stakes, style assumption, and 3-5 story beats before making the storyboard package. Ask only for visual style if it is missing and blocks useful output.
- Existing video continuation or sequel: treat it as a video-extension / first-last-frame continuity task. Preserve the previous video's identity, wardrobe, lighting, camera direction, rhythm, and final-frame spatial logic; then design the next beat.
- Pure music or no-dialogue segment: leave `台词` and `旁白` empty unless the user supplies vocals. Use `状态/音效`, music tempo, beat points, camera motion, and visual action as the primary structure.
- Existing partial table or draft: normalize it into the current 12-column standard, preserve user-authored content where possible, and only fill missing columns that are needed for production.

## Required Output

Choose the smallest useful output mode from `references/quick-modes.md`. Do not force all sections for narrow requests. For a full production package, include these sections when enough script detail exists:

- 风格确认: summarize chosen style and translate named-style requests into neutral visual traits when needed.
- 剧作结构拆解: protagonist desire, conflict, value shift, turning point, beats, crisis/climax/resolution when present.
- 角色设计三视图: front, side, back prompts for each major character, plus costume/material/color notes.
- 元素分层资产清单: character identity layer, environment layer, prop/asset layer, and emotional/cinematic layer.
- 角色情绪神态镜头: expression and emotional close-up prompts for each important emotional state.
- 场景图: location/environment prompts with time of day, lighting, weather, composition, and continuity details.
- 道具资产图: props with shape, material, scale, function, wear, and relation to characters.
- 分镜镜头表: shot number, scene, character, action, camera language, duration, Seedance 2.0 prompt, negative prompt if useful.
- 台词与音乐对应: line/dialogue, emotional intention, music style, instrumentation, tempo, sound design, cue timing.
- 音频同步表: for video-generation handoff, separate music, sound effects, dialogue, and narration by exact time range, volume, and notes.
- 素材引用规划: image/video/audio references, how each @asset is used, and whether the shot should use all-round reference, first/last frame, extension, or edit mode.
- 连贯性提示: reusable constraints for character consistency, wardrobe, palette, lens, and world rules.

## Prompt Rules

- Write prompts as production instructions, not vague adjectives.
- Keep character identity stable across all prompts by repeating the same core descriptors.
- Use camera terms when helpful: wide shot, medium shot, close-up, over-the-shoulder, tracking shot, dolly in, handheld, crane shot, low angle, shallow depth of field.
- Include movement for video prompts: character motion, environment motion, camera motion, timing, and transition.
- Avoid overloaded prompts. If a shot has multiple ideas, split it into multiple shots.
- For music, map every major dialogue or beat to mood, tempo, instrumentation, and cue point. Do not generate lyrics unless the user asks.
- If the script is long, process it scene by scene and tell the user what range was covered.
- Prefer the S-A-C-S-C prompt shape: Subject, Action, Camera, Style, Constraints.
- Split compound actions into multiple shots unless the user explicitly asks for one continuous take.
- When the user provides or implies reference materials, describe each as an @asset role instead of re-describing visual details that should come from the asset.
- For dialogue shots, specify lip-sync, language/accent if present, emotional delivery, and background sound.
- For music-driven shots, specify beat points, visual cuts, motion accents, and whether the reference video/audio controls rhythm.
- Before visual prompt writing, make sure each scene changes a value, exposes conflict, or advances the character's pursuit. If a scene has no dramatic action, mark it as decorative and recommend compressing or merging it.
- When the user asks for a detailed scene storyboard image, create or provide an annotated storyboard version using the required keyframe annotation format in `references/storyboard-annotation-rules.md`.

## Cinematic Storyboard Integration Rules

Use these rules when the user asks for cinematic storyboards, 3x3 storyboard images, full-script storyboard packages, or high-consistency AI video planning:

- Treat the 3x3 storyboard as a continuity matrix, not just nine pretty frames.
- A 15-second 3x3 group should maintain narrative flow across all 9 frames with exact timestamps and visible cause-effect progression.
- Enforce visual variance inside each 3x3 group: include establishing/wide shots, medium/action shots, and close-up/extreme close-up emotion or prop shots.
- Every 3x3 group needs identity anchors: repeated face structure, costume, signature prop, lighting logic, and environment grammar from frame 01 to frame 09.
- Generate or specify character sheets, environment concepts, and prop assets before detailed storyboard images when the script has recurring characters, places, or objects.
- For full-script storyboard image generation, prefer the key-segment-first strategy: generate the opening, one major turning-point segment, and the climax segment first so the user can validate visual direction before producing every remaining segment.
- For long scripts, add a segment-function plan before the detailed keyframe table. Label each 15-second group as 建立, 推进, 转折, 高潮, 余韵, 过场, or other useful dramatic function.
- Translate emotions into observable details: eye focus, jaw tension, breath, hand pressure, posture shift, clothing movement, camera pressure, and lighting change.
- For final storyboard images, default to clean cinematic panels: 3x3 grid, strict black panel borders, 2K minimum width, no dialogue bubbles, no subtitles, no large labels inside the visual frames.
- Keep text, dialogue, timing, and metadata in the Markdown keyframe table. Add in-frame annotations only when the user explicitly asks for annotated storyboard images.
- During revisions, prefer frame-level correction for angle, expression, continuity, or emotional intensity instead of redesigning the whole sequence.
- For staged collaboration, surface confirmation checkpoints after script analysis, the segment-function plan and 12-column keyframe table, core assets, key-segment storyboard images, and final video prompt handoff. If the user asks for a complete one-pass deliverable, proceed with clear assumptions instead of blocking at every checkpoint.
- For batch production, label work as 批次01*验证段, 批次02*补齐段, 批次03\_修订段, or another clear batch ID so generated storyboard images, tables, and revisions stay traceable.
- For long packages, default to Markdown output unless the user asks for CSV/Excel-style tables or JSON. Mention that the 12-column table is suitable for conversion to those structured formats.

## Seedance Video Prompt Copy-Block Rules

Use this format when the user asks for video prompts intended to be copied into 即梦/Seedance:

- Default to one fenced `text` code block per 15-second storyboard segment, not one block per keyframe.
- Keep each segment code block under 2000 Chinese characters when possible, especially for 即梦 direct copy/paste.
- Start the block with the reference-image instruction when a 3x3 keyframe storyboard image exists: `@图1作为九宫格关键帧分镜参考。不要生成九宫格拼贴，不要保留黑色分镜边框。把参考图里的9个画面，按顺序展开成一段15秒连续电影级视频。`
- Include `语言规则` once per block: all spoken dialogue must use the requested spoken language, no subtitles/text/speech bubbles/screen text, Chinese is production guidance only, lip-sync follows the spoken-language dialogue.
- Include `角色连续` once per block with concise identity anchors for only the characters appearing in that segment.
- For each of the 9 timed beats, use this readable field layout instead of Markdown tables: `序号`, `时间`, `镜头`, `运镜`, `动作`, `情绪/细节`, `台词`, `旁白`, `状态/音效`.
- Do not include the `关键帧` field in direct-copy video prompt blocks. Keep priority markers such as `⭐视觉`, `⭐台词`, and `⭐转折` in the 12-column production table only.
- For dialogue and narration fields, use paired Chinese reference and actual spoken-language text, for example `台词：中文：别低头。；英文：Aelira says: "Do not look down."` and `旁白：中文：无；英文：无`.
- Keep Chinese reference meaning concise; avoid long bilingual duplication beyond the `台词` and `旁白` fields.
- End each segment block with one shared `负面约束（避免）`, not one negative prompt per keyframe.
- Use this shared negative constraint unless the user gives a stricter one: `视频不要出现任何字幕、标题、台词文字，角色不一致，五官漂移，额外手指，肢体畸形，重复人物，低清晰度，过曝，脏乱背景，镜头切换混乱，服装低俗，五官模糊，动作僵硬，画面廉价感，水印，logo`.

## Project Storage Rules

Use these rules whenever creating, copying, saving, or reporting storyboard Markdown files, 3x3 keyframe storyboard images, character assets, scene assets, prop assets, or other production outputs:

- Treat `00剧本\<剧本项目>\` as the default production root for script-specific work. Do not use workspace-root `image\` or historical image folders as the default destination unless the user explicitly asks.
- A script project folder usually has an indexed name such as `00剧本\01初吻\` or `00剧本\02灰烬恋人_\`. If the user names an existing script folder, use it exactly.
- Before starting a new script/project batch, ask the user what `00剧本\<剧本项目>` folder name to use unless the user already provided an exact folder or file path.
- Put generated storyboard text/production-package Markdown directly under the script project root, for example `00剧本\02灰烬恋人_\灰烬恋人_3分钟Seedance分镜包.md`.
- Put storyboard and keyframe images under `00剧本\<剧本项目>\image\<剧本名>\分镜图关键帧\`.
- Put reusable production assets under `00剧本\<剧本项目>\image\<剧本名>\资产\`.
- If a project uses both the indexed folder name and a clean script title, preserve both levels. Example: `00剧本\02灰烬恋人_\image\灰烬恋人\分镜图关键帧\`.
- Keep original generated-image staging files intact unless the user explicitly asks otherwise. The deliverable copy must be placed in the project folder with a meaningful filename.
- Never leave random generated IDs as final filenames. Name files by sequence, time range, scene/title, asset type, and version.
- Recommended storyboard filename pattern: `01_00-00-00-15_灰烬山路_无标注九宫格.png`, `02_00-15-00-30_黑铁王冠与灰烬契约_标注九宫格.png`.
- Recommended asset filename pattern: `角色_艾莉娅_三视图_v01.png`, `场景_灰烬神庙_内殿_v01.png`, `道具_黑铁王冠_v01.png`.
- In final responses, report the `00剧本\<剧本项目>\...` project-local paths first. Mention temporary generated-image staging paths only when the user asks where the raw generated files are.

## Professional Keyframe Table Rules

Use the professional 12-column keyframe table by default for storyboard keyframe tables, especially 3x3 Seedance production annotations:

`序号 | 关键帧 | 时间 | 镜头 | 运镜 | 转场 | 动作 | 情绪/细节 | 台词 | 旁白 | 状态/音效 | 英文`

- `关键帧` marks priority frames. Use typed markers when useful: `⭐情绪`, `⭐动作`, `⭐台词`, `⭐旁白`, `⭐视觉`, `⭐转折`. Mark about 2-3 frames per 9-frame group unless the user requests otherwise.
- `镜头` records shot size or composition, such as 人群近景, 黎明宽景, 王冠特写.
- `运镜` records camera movement, such as 固定机位, 缓慢推进, 拉远, 摇臂上升, 跟随, 环绕, 手持晃动.
- `转场` means the transition entering the current keyframe from the previous keyframe. Use 淡入 or 黑屏切入 for frame 01 when useful; use 切, 叠化, 划像, 无缝衔接, 淡出, or 切至下一段 for later frames.
- `动作` must stay visual and behavioral.
- `情绪/细节` captures performance, expression, breathing, hand tension, eye focus, costume movement, and other acting details.
- `台词` is only for character speech, formatted as `[角色]说：...`.
- `旁白` is only for narration or voice-over.
- `状态/音效` is only for non-speech information, including low bells, ash falling, no cheering, breathing, female vocalizing, wooden wheel sounds, crowd murmurs, silence, music cues, ambience, and sound effects.
- `英文` contains the English version of any dialogue or narration. For sound or ambience, use `[SFX: ...]`, `[Ambience: ...]`, or `[N/A]`.
- Do not merge dialogue, narration, and sound into one prefixed cell. Keep the three audio-language channels separated for TTS, voice-over, and sound design.
- Audio sync rows may overlap in time. Music can run across 0-15s while dialogue, narration, and sound effects occupy shorter overlapping ranges.
- For 3x3 storyboard images, the 12-column table is the source of truth. Do not add per-frame `参考资产` by default; the user normally uses one generated 3x3 keyframe storyboard image as the reference. If asset references are needed, keep them in a separate asset plan or video prompt, not in the keyframe table.
- For storyboard images, visible annotations may use a compact subset: `序号 / 时间 / 镜头 / 动作 / 台词或旁白或状态`.
- Before exporting or packaging large tables, validate the Markdown table with `scripts/validate-keyframe-table.mjs` and use the example files in `examples/` as formatting references.

Example row:

`07 | ⭐台词 | 9.5-11.2s | 人群近景 | 缓慢推进 | 切 | 小女孩抬头 | 眼神好奇，嘴唇微张，眉头微皱 | 小女孩说：妈妈，王后会疼吗？ |  | 人群低声议论 | Mother. Will the Queen be in pain? [SFX: crowd murmurs]`

## Missing Information

Ask only for information that blocks useful output. Style is mandatory. Other details are optional and should be assumed conservatively:

- Aspect ratio: default to 16:9 unless the user says short video, then use 9:16.
- Duration: estimate shot durations based on dialogue length and emotional weight.
- Language: preserve the user's script language.
- Rating: keep visual content suitable for general production unless the script clearly requires otherwise.

## Reference

For the exact output skeleton and reusable wording, read `references/output-schema.md`.

For quick output modes and per-mode delivery rules (fast storyboard, audio-only, video prompt, asset planning, revision, full package), read `references/quick-modes.md`.

For story structure, scene value shifts, beats, conflict, turning points, crisis, climax, and resolution checks, read `references/story-structure-method.md`.

For Seedance 2.0-specific shot splitting, @reference usage, first/last frame, extension, edit, and music beat rules, read `references/shot-breakdown-method.md`.

For annotated storyboard images and keyframe labels, read `references/storyboard-annotation-rules.md`.

For asset naming and manifests, use `assets/README.md` and `assets/asset-manifest-template.md`.

For exporting 12-column keyframe tables to CSV or JSON, use `scripts/convert-keyframe-table.mjs`.

For validating 12-column keyframe tables, use `scripts/validate-keyframe-table.mjs`.

For checking whether `dist/seedance-script-designer.skill` is stale before distribution, use `scripts/check-dist-freshness.mjs` and `references/release-checklist.md`.

For concrete reference outputs, read `examples/minimal-12col-keyframes.md`, `examples/asset-manifest-example.md`, and `examples/audio-sync-example.md`.
