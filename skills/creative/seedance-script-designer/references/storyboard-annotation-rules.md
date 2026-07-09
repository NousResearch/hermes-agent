# Storyboard Annotation Rules

Use these rules whenever the user asks for detailed scene storyboard images, keyframe storyboard sheets, or annotated storyboard panels.

## Professional Keyframe Table

Use this 12-column table as the default production source of truth for every storyboard panel/keyframe:

| 序号 | 关键帧 | 时间 | 镜头 | 运镜 | 转场 | 动作 | 情绪/细节 | 台词 | 旁白 | 状态/音效 | 英文 |
|---|---|---|---|---|---|---|---|---|---|---|---|

Use detailed descriptions, not compressed notes.

Example row style:

`07 | ⭐台词 | 9.5-11.2s | 人群近景 | 缓慢推进 | 切 | 小女孩抬头 | 眼神好奇，嘴唇微张，眉头微皱 | 小女孩说：妈妈，王后会疼吗？ |  | 人群低声议论 | Mother. Will the Queen be in pain? [SFX: crowd murmurs]`

## Dialogue, Narration, and State Labels

Do not mix dialogue, narration, sound, mood, and environmental notes in one cell.

- `台词` is only for character speech. Use `[角色]说：...`.
- `旁白` is only for narration or voice-over.
- `状态/音效` is only for non-speech scene/audio information.
- `状态/音效` includes low bells, ash falling, no cheering, breathing, female vocalizing, wooden wheel sounds, crowd murmurs, silence, music cues, ambience, and sound effects.
- When a character speaks, provide both Chinese and English unless the user explicitly says not to translate.
- For narration, provide a full English voice-over translation.
- For sound or ambience, write English as `[SFX: ...]`, `[Ambience: ...]`, or `[N/A]`.
- If one row has both speech and sound, keep Chinese separated across `台词` and `状态/音效`, then combine the English cell if needed, such as `Mother. Will the Queen be in pain? [SFX: crowd murmurs]`.
- Keep `动作` limited to visible behavior and physical action. Put speech, narration, sound, music, and ambience only in their dedicated columns.

## Camera, Transition, and Performance Fields

- `关键帧` uses typed priority markers for later image or video generation. Use `⭐情绪` for emotional turns, `⭐动作` for key actions, `⭐台词` for important dialogue, `⭐旁白` for important voice-over, `⭐视觉` for visual highlight moments, and `⭐转折` for narrative turns. A normal 9-frame group should usually have 2-3 starred frames.
- `镜头` records shot size or composition, such as 人群近景, 黎明宽景, 王冠特写.
- `运镜` records camera movement, such as 固定机位, 缓慢推进, 拉远, 摇臂上升, 摇臂下降, 左摇, 右摇, 跟随, 环绕, 手持晃动.
- `转场` means the transition entering the current keyframe from the previous keyframe.
- For frame 01, use `淡入`, `黑屏切入`, or `切` when useful.
- For middle frames, prefer `切`, `叠化`, `划像`, or `无缝衔接`.
- For frame 09, use `切至下一段`, `淡出`, or `黑屏` when useful.
- `情绪/细节` captures expression, emotion, breathing, hand tension, eye focus, costume movement, and other performance details.
- Do not include a per-frame `参考资产` column in the 3x3 keyframe table by default. If the user needs @ references, keep them in a separate asset plan or video prompt.

## Cinematic Storyboard Image Standard

When creating final storyboard images or image-generation prompts:

- Default to a clean 3x3 cinematic storyboard grid.
- Use strict black panel borders, visually equivalent to 4px borders on a 2K-wide canvas.
- Use 2K width minimum for clarity; use 4K or larger when the image contains multiple 3x3 groups or dense visual detail.
- Do not place dialogue bubbles, subtitles, long captions, or large labels inside the visual frames by default.
- Keep all timing, dialogue, narration, state, English, camera, and transition metadata in the Markdown keyframe table.
- Each panel must map to one keyframe row and one dominant visual idea.
- The grid should show visual continuity across all 9 frames: stable character identity, matching wardrobe, repeated signature props, consistent lighting direction, and coherent location grammar.
- Include visual variance inside the grid: at least one establishing/wide shot, multiple medium/action shots, and multiple close-up or extreme close-up emotion/prop shots.
- If a user asks for a contact sheet covering many 3x3 groups, keep each 3x3 block visually separated with clean gutters and preserve readable panel borders.

## Clean 3x3 Storyboard Prompt Template

Use this template when generating a clean storyboard image prompt from a completed 12-column table:

```text
Create a clean cinematic 3x3 storyboard contact sheet, 2K minimum width, strict black panel borders, no dialogue bubbles, no subtitles, no large in-frame labels. Each panel maps to one keyframe row in reading order from 01 to 09. Preserve the same character identity, costume, signature props, lighting direction, location grammar, and color palette across all panels. Show clear visual variance across wide, medium, close-up, and prop/emotion shots. Keep all timing, dialogue, narration, status/sound, English text, camera movement, and transition notes outside the image in the Markdown keyframe table.
```

## Image Annotation Standard

When adding annotations onto a storyboard image:

- Keep the original unannotated image intact.
- Create a new annotated copy.
- Add each annotation inside its corresponding panel.
- Use a readable semi-transparent dark annotation box, usually near the bottom of the panel.
- Keep the 12-column table as the editable source of truth.
- Use compact visible annotations on the image only when the user explicitly asks for annotated storyboard images: `序号 / 时间 / 镜头 / 动作 / 台词或旁白或状态`.
- Include `运镜`, `转场`, and English in the image only when the canvas has enough space.
- Keep annotations aligned panel-by-panel in reading order: left to right, top to bottom.
- If the text is long, increase output resolution or use a larger canvas instead of deleting detail.
- Reinforce uniform panel borders after annotation if needed.
- Do not let labels hide the main face/action unless there is no alternative; prefer lower third placement.

## Default 3x3 Storyboard Timing

For a 15-second 3x3 storyboard, use this default timing unless the user provides another duration:

This is the Level 2 detailed keyframe timeline. For Level 1 macro rhythm planning before the 3x3 table, see `shot-breakdown-method.md` Section 6.

| 序号 | 时间 |
|---|---|
| 01 | 0.0-1.5s |
| 02 | 1.5-3.0s |
| 03 | 3.0-4.8s |
| 04 | 4.8-6.2s |
| 05 | 6.2-7.8s |
| 06 | 7.8-9.5s |
| 07 | 9.5-11.2s |
| 08 | 11.2-12.7s |
| 09 | 12.7-15.0s |

Adjust timing by dialogue length and action complexity when needed.

## Delivery Standard

When delivering an annotated storyboard:

- Provide the annotated image.
- Provide the file path.
- Include or summarize the keyframe table if the user may need to edit the text later.

When delivering any generated storyboard or keyframe image:

- Save or copy the project-local deliverable under `00剧本\<剧本项目>\image\<剧本名>\分镜图关键帧\` unless the user provides another exact folder.
- If the project folder does not exist yet, ask the user which `00剧本\<剧本项目>` folder to create before starting a new batch.
- Name each image by sequence, time range, scene title, and annotation status, not by the default generated file ID.
- Use names such as `01_00-00-00-15_灰烬山路_无标注九宫格.png` or `01_00-00-00-15_灰烬山路_标注九宫格.png`.
- Keep original generated-image staging files intact and report the named `00剧本` project-local paths in the final response.
