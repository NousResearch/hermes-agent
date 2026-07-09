# Seedance 2.0 Shot Breakdown Method

Use these rules when converting a script into production prompts for Seedance 2.0.

## 1. Director Mindset

Treat the prompt writer as a director, not a visual describer.

- Use reference assets to anchor identity, object details, composition, motion, and audio.
- Do not over-describe colors, shapes, logos, or facial details when an @image already defines them.
- Tell Seedance what each asset is for: subject, first frame, last frame, action reference, camera reference, style reference, scene reference, voice reference, sound effect, or music rhythm.
- Keep each shot controllable. One shot should usually contain one core action, one camera idea, and one emotional beat.

## 2. Element Isolation Framework

Before storyboard or image prompt generation, decompose the script into four independent production layers:

- Character Identity Layer: fixed face structure, age version, hair, body type, costume, palette, signature props, and identity anchors. For time-spanning stories, define separate age-specific versions while preserving recognizable facial anchors.
- Environmental Layer: locations, atmosphere, lighting logic, spatial structure, weather, architecture, color grammar, and recurring background motifs.
- Prop/Asset Layer: story-critical objects, materials, scale, wear, macro details, symbolic function, and which shots reveal or transform them.
- Emotional/Cinematic Layer: abstract emotions translated into visible micro-expression, posture, breath, hand tension, camera pressure, lens distance, and lighting change.

Use these layers as continuity constraints before writing the 12-column keyframe table or Seedance prompts.

## 3. Seedance 2.0 Prompt Formula

Use S-A-C-S-C for every shot prompt:

1. Subject: who or what is anchored, preferably with @image references.
2. Action: physical movement, expression change, prop interaction, dialogue, or transformation.
3. Camera: shot size, angle, lens feeling, camera path, speed, and transition.
4. Style: visual style, light, texture, color, mood, genre.
5. Constraints: identity lock, no deformation, no extra text, no logo warping, no camera shake unless intended, no clothing drift.

Compact prompt template:

`主体: @图X 的角色/物体。动作: ...。运镜: ...。风格: ...。音频: ...。约束: ...。`

## 4. Multi-Modal @Reference Planning

Seedance 2.0 can combine text, images, video, and audio. Plan references before writing final prompts.

- Images: use for character identity, costume, product shape, scene layout, first frame, last frame, style board, prop asset.
- Videos: use for motion, camera path, transition rhythm, action choreography, facial expression pattern, special effects.
- Audio: use for dialogue lip-sync, narration voice, music tempo, sound effects, beat cuts.
- Keep within the practical planning limit from the manuals: up to 12 mixed files, commonly up to 9 images, 3 videos, and 3 audio files.
- If too many references exist, choose the ones that affect identity, camera, or rhythm most. Mention omitted assets as optional.

Reference map format:

| 资产 | 用途 | 绑定对象 | 使用方式 |
|---|---|---|---|
| @图1 | 角色身份 | 女主 | 锁定五官、发型、服装 |
| @视频1 | 运镜参考 | 镜头 03-05 | 参考推拉摇移和切换节奏 |
| @音频1 | 台词/音乐 | 镜头 02 | 口型同步或音乐卡点 |

## 5. Shot Splitting Rules

Split the script by control risk, not only by paragraph.

- New location: new shot or scene.
- New emotional state: new close-up or reaction shot.
- New physical action: new shot if it contains a separate start, impact, or result.
- New speaker: usually new shot or over-the-shoulder/reaction shot.
- New prop reveal: use insert shot or close-up.
- New camera movement: separate shot unless it is a one-take design.
- Complex sentence with "然后/接着/突然/切到/最后": split at those connectors.

Avoid compound action overload. Instead of one prompt saying "runs in, jumps, turns, drinks, speaks", split into:

1. entering/running shot;
2. jump or impact shot;
3. turn/reaction shot;
4. drink/prop close-up;
5. dialogue close-up.

## 6. Time-Coded Storyboard Pattern

Two timing levels exist. Use them at different stages:

**Level 1: Macro segment rhythm** — use when planning a 4-15 second clip without a full 3x3 table. Divide by narrative function:

- 0-3s: setup, entry, first line, establishing motion.
- 3-6s: action or dialogue escalation.
- 6-9s: reveal, cutaway, or emotional shift.
- 9-12s: payoff, reaction, product/prop highlight.
- 12-15s: title, slogan, final pose, transition, or audio hit.

**Level 2: 3x3 keyframe timeline** — use when filling the 12-column professional table for a full 15-second 3x3 storyboard group. Default timing per frame:

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

Level 1 is for narrative rhythm decisions. Level 2 is the production source of truth for image and video generation. Always use Level 2 when filling the 12-column table. Adjust frame timing by dialogue length and action complexity when needed.

For full-script work, assign a segment function before filling the 12-column keyframe table:

| 段落功能 | Use when |
|---|---|
| 建立 | introducing world, location, mood, protagonist, or rule |
| 推进 | advancing pursuit, conflict, relationship, or investigation |
| 转折 | changing value, revealing new stakes, or forcing a choice |
| 高潮 | resolving the central pressure through decisive action |
| 余韵 | showing aftermath, emotional echo, title, or final image |
| 过场 | moving between locations or compressing connective material |
| 信息揭示 | exposing truth, rules, evidence, or hidden history |
| 关系变化 | changing trust, intimacy, loyalty, or power balance |

## 7. 3x3 Continuity Matrix

For a 15-second 3x3 storyboard group:

- Treat the 9 panels as one continuity matrix, not isolated still images.
- Keep exact timing across the sequence and make each frame advance cause, reaction, reveal, emotional shift, or transition.
- Include visual variance: establishing/wide shots, medium/action shots, and close-up/extreme close-up emotion or prop shots.
- Use the 3-3-3 ratio as the default: about 3 wide/establishing frames, 3 medium/action frames, and 3 close-up or extreme close-up frames. Adjust to 4-3-2 for environment/action emphasis or 2-3-4 for emotion-heavy scenes.
- Use frame 01 to establish space or pressure, frames 02-04 to introduce action, frames 05-07 to reveal or turn emotion, and frames 08-09 to pay off or transition.
- Preserve identity anchors from frame 01 to frame 09: face, costume, prop, lighting direction, environment design, and palette.
- Avoid duplicate compositions unless repetition is the point. If two frames feel visually identical, change shot size, angle, camera movement, action, or emotional detail.

## 8. Professional Keyframe Table to Prompt Mapping

For detailed 3x3 keyframe work, use the 12-column professional table as the structured production source:

`序号 | 关键帧 | 时间 | 镜头 | 运镜 | 转场 | 动作 | 情绪/细节 | 台词 | 旁白 | 状态/音效 | 英文`

Map the table into Seedance prompts this way:

- `关键帧` -> priority marker for generation, review, and video prompt detail. Mark about 2-3 frames per 9-frame group with typed markers: `⭐情绪`, `⭐动作`, `⭐台词`, `⭐旁白`, `⭐视觉`, `⭐转折`.
- `镜头` + `运镜` -> camera instruction, including shot size, angle, movement path, and speed.
- `转场` -> edit instruction. Treat it as the transition entering the current keyframe from the previous keyframe.
- `动作` + `情绪/细节` -> character performance instruction.
- `台词` -> lip-sync dialogue instruction, with speaker, language, delivery, and facial expression.
- `旁白` -> voice-over instruction, not lip-sync unless the narrator is visibly speaking.
- `状态/音效` -> ambience, music cue, Foley, crowd sound, weather, or impact sound instruction.
- `英文` -> bilingual delivery reference. Use direct English for dialogue and narration; use `[SFX: ...]`, `[Ambience: ...]`, or `[N/A]` for sound-only rows.

Do not add a per-frame `参考资产` column to the 3x3 keyframe table by default. For the user's normal workflow, the generated 3x3 keyframe storyboard image is used as a single visual reference. If additional references are needed, put them in a separate asset plan or the final video prompt.

Default camera movement vocabulary:

- 固定机位
- 缓慢推进 / 快速推进
- 拉远
- 摇臂上升 / 摇臂下降
- 左摇 / 右摇
- 跟随
- 环绕
- 手持晃动

Default transition vocabulary:

- 切
- 淡入 / 淡出
- 黑屏切入 / 黑屏
- 叠化
- 划像
- 无缝衔接
- 切至下一段

Keep the professional table as the editable source of truth. When rendering text onto a storyboard image, use a compact subset if the full table would make the panel unreadable.

## 9. Key-Segment-First Generation Strategy

When a full script contains many 15-second groups, avoid generating every storyboard image before the visual direction is proven.

- First generate or draft the opening establishment segment, one major turning-point segment, and the climax segment.
- Use these three segments to validate character identity, scene grammar, palette, composition density, and emotional intensity.
- After user confirmation or a clear visual direction, generate the remaining groups in batches.
- If the user explicitly asks for a complete one-pass deliverable, still keep the key segments clearly marked for review.
- Use batch IDs in filenames, tables, and revision notes when possible: `批次01_验证段`, `批次02_补齐段`, `批次03_修订段`.

## 10. Video Prompt Package and Audio Sync

For video-generation handoff, convert each 15-second 3x3 group into three 5-second video instructions by default unless the story beat clearly requires another split:

- Shot 1: frames 01-03, usually setup, entry, and establishing motion.
- Shot 2: frames 04-06, usually action, escalation, or reveal.
- Shot 3: frames 07-09, usually payoff, reaction, title, or transition.

Each video instruction should include:

- based frames;
- duration;
- subject and optional reference assets when the user supplies them;
- camera movement and transition;
- action and emotional detail;
- dialogue, narration, ambience, and sound effects;
- continuity and negative constraints.

Also produce an audio sync table when needed. Audio rows may overlap because real edits layer music, dialogue, ambience, and sound effects:

| 时间 | 类型 | 内容 | 音量 | 备注 |
|---|---|---|---|---|
| 0.0-15.0s | 音乐 | 背景音乐或主题氛围 | 60-70% | 可贯穿整段 |
| 4.8-6.2s | 台词 | 角色台词 | 80-90% | 需压过音乐 |
| 5.0-5.4s | 音效 | 动作或环境音效 | 40-60% | 可与台词/音乐叠加 |

Use types `音乐`, `音效`, `台词`, and `旁白`. Keep dialogue and narration readable over music.

## 11. First/Last Frame and Extension

Use first/last frame mode when the user needs a controlled transition between two images.

- State which @image is first frame and which @image is last frame.
- Describe the transformation path between them, not just the endpoints.
- For before/after, product reveal, outfit change, portal, weather change, or emotional transformation, explicitly name the intermediate motion.

Use extension mode when continuing an existing video.

- State "将 @视频X 延长 Ns".
- The selected duration should be the new extra segment, not the whole original video.
- Preserve the previous video's lighting, motion direction, character identity, rhythm, and last-frame spatial logic.
- Add the next beat as a natural continuation: camera keeps moving, character completes an unfinished action, sound continues or resolves.

## 12. One-Take Continuity

Use one-take prompts only when the script benefits from spatial continuity.

- Say "全程不要切镜头" or "一镜到底" when required.
- Describe camera path as a chain: start point -> movement -> subject reveal -> environment transition -> ending frame.
- Use occluders, pans, zooms, passing objects, darkness, water, curtains, doors, or body movement for seamless transitions.
- Keep identity and wardrobe constraints repeated because one-take prompts are long.
- If the one-take becomes too dense, split into multiple shots and preserve continuity through matching action or first/last frames.

## 13. Dialogue, Voice, and Music

For dialogue:

- Assign each line to a speaker, shot, facial expression, body action, and lip-sync requirement.
- Include language, accent, whisper/shout/singing style if present.
- Use reaction shots between lines when emotion changes.

For music:

- Map each dialogue or action beat to music mood, tempo, instrumentation, and cue timing.
- For music-card shots, specify cuts and motion accents on the beat.
- If a reference audio/video controls rhythm, write "动作、切镜和转场卡点参考 @音频X/@视频X".
- Add sound effects for physical impacts, footsteps, cloth movement, door locks, crowd noise, weather, machinery, and UI sounds.

## 14. Camera Vocabulary

Combine camera type with speed and purpose:

- Slow dolly-in: emotional pressure, product premium feel, revelation.
- Fast pan: surprise, transition, chase, comedic turn.
- Tracking shot: walking, running, spatial continuity.
- Low angle: heroism, threat, scale.
- Over-the-shoulder: dialogue relation.
- Macro close-up: product detail, hands, key, logo, texture.
- Handheld with slight breathing: documentary or UGC realism.
- Hitchcock zoom: fear, shock, disorientation.
- Orbit shot: glamour, fight choreography, product wraparound.
- First-person POV: immersion, chase, discovery.

## 15. Iteration Triage

When the user requests changes, locate the problem layer before rewriting:

| 问题类型 | 回溯层级 | 影响范围 |
|---|---|---|
| 剧本理解错误 | 剧作结构/场景节拍 | 可能影响全部输出 |
| 时间分配不合理 | 12列关键帧表 | 后续分镜图和视频提示词 |
| 台词/旁白/音效错误 | 12列关键帧表/音频同步表 | 音频和口型同步 |
| 角色/场景不符 | 元素分层资产/素材规划 | 分镜图和视频提示词 |
| 分镜图视觉问题 | 3x3视觉板 | 后续图生视频提示词 |
| 视频提示词不清晰 | 视频指令包 | 只影响视频生成 |
| 局部细节优化 | 对应帧/对应段 | 只做局部修订 |

Prefer local frame or segment repair over full regeneration when identity and structure are already correct.

## 16. Prompt Quality Checks

Before finalizing each Seedance prompt, check:

- Does it identify the subject clearly, and use reference assets only when the user supplies or asks for them?
- Is there only one main action or a deliberate continuous-take chain?
- Is the camera language concrete enough?
- Are audio, dialogue, music, and sound effects aligned with the action?
- Are continuity constraints repeated for identity, costume, scene, and props?
- Are negative constraints included where artifacts are likely?
