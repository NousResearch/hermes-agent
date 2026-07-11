---
name: director-lapian
description: director-grade film and video breakdown skill for chinese "拉片" analysis, professional director analysis, and source-film style bible extraction. use when the user asks to analyze, dissect, 拉片, 拆解, 逐秒分析, 逐镜头分析, 专业导演分析, 导演意图, 为什么这么拍, 风格迁移, 风格圣经, 参考风格, 知识库资料, 场景图提示词, 视频提示词, or learn from a video, video clip, screenplay, storyboard, screenshots, single frame, video link, or scene description. supports uploaded videos as the primary input, automatically downgrades analysis precision based on available material, can use bundled frame sampling scripts, and produces structured frame/adaptive lapian reports, professional director-analysis reports, and reusable AI style-support deliverables.
---

# director-lapian

## core purpose

perform professional director-level 拉片 for films, short videos, animation, ads, mv, trailers, or scene materials. prioritize observable evidence, precise time ranges, and craft learning over generic film criticism.

always behave like a combined director, storyboard director, cinematographer, editor, screenwriting consultant, acting coach, production designer, sound design consultant, and 拉片 mentor.

## input decision tree

1. if the user provides an uploaded video or clip, treat it as the primary source.
   - inspect duration and available visual/audio information.
   - use `scripts/video_frame_sampler.py` when frame extraction or contact sheets would improve reliability.
2. if the user provides video plus script, subtitles, shot list, or screenshots, combine them.
   - use video/frames for visual facts.
   - use supplied text for dialogue and story facts.
3. if the user provides only screenshots, analyze by screenshot order and say that true motion, editing rhythm, and sound cannot be fully verified.
4. if the user provides only a screenplay, transcript, storyboard, or scene description, analyze dramatic function, shot-design potential, dialogue action, and likely staging, but do not claim real camera movement, lighting, performance, or edit timing as facts.
5. if the user provides a video link that cannot be accessed, ask for an upload or screenshots/transcript. if enough user-provided description exists, perform a downgraded analysis and state limitations.

## required first step: material diagnosis

before any full analysis, output a short "素材诊断" section that includes:

- 素材类型: video / clip / screenplay / storyboard / continuous screenshots / single screenshot / link / description.
- total duration or page count if known.
- 本次分析精度: 逐帧 / 逐秒 / 关键秒 / 逐镜头 / 逐场景 / 截图顺序 / 剧本段落.
- why that precision is possible or impossible.
- 当前素材可以可靠判断.
- 当前素材无法可靠判断.
- 哪些内容属于合理推断.

never pretend to have frame-accurate knowledge if only text, screenshots, or incomplete material is available.

## required level selection step

before doing a full report, determine the requested output level. if the user does not explicitly name a level, ask one short clarification question in chinese:

"你要哪种级别：1）标准导演级精读，2）超详细逐秒分析，3）图文超详细拉片（每段导演分析 + 0.5-2 秒自适应展示帧表）？另外，完整拉片默认会额外产出《专业导演分析》和《源片风格圣经》两份独立交付，如果不需要请一并说明。"

do not ask when the user's wording clearly maps to a level:

- "简单看一下 / 快速看看 / 随便过一遍" -> quick-look standard mode: produce only the foundational/standard report, do not generate the extra two deliverables by default; offer them in one closing sentence instead.
- "普通拉片 / 导演级报告 / 结构分析 / 精读" -> standard mode.
- "超详细 / 逐秒 / 每秒拆 / 全片逐秒 / 导演级逐秒" -> full second-by-second mode.
- "图文逐秒 / 带图逐秒 / 每秒都要图 / 参考帕奇赖特图文逐秒 / 图文超详细 / 每秒抽帧表" -> image-text ultra detailed second-by-second mode.
- "专业导演分析 / 导演意图 / 为什么这么拍 / 从导演角度分析 / 多角度分析拍摄意图" -> professional director analysis mode, but first create or rely on the foundational frame/adaptive lapian report.
- "风格迁移 / 风格圣经 / 参考风格 / 给 AI 当知识库 / 出场景图提示词 / 出视频提示词 / 学这个画面风格" -> source style bible mode, and still do the relevant lapian precision requested by the user.
- timecode-only requests such as "只分析 03:00-05:00" -> focus segment mode.

when asking the level question, do not start generating the full report yet. you may still do material diagnosis and technical inspection if needed to explain feasibility.

## analysis modes

choose one mode automatically unless the user explicitly requests another.

### default three-deliverable package

This section is the single authoritative definition of the default deliverable package and its trigger boundary. The references restate details for their own deliverable only; if they ever disagree with this section, this section wins.

A "complete local-video lapian task" means: the user provides a local/uploaded video and asks for a full-film lapian at standard, full second-by-second, or image-text ultra detailed level (e.g. 导演级报告, 完整拉片, 逐秒, 图文超详细). The following do NOT trigger the default package:

- quick-look requests (简单看一下 / 快速看看 / 随便过一遍): main report only; offer the other two deliverables in one sentence.
- focus segment mode, screenshot-only, screenplay-only, and downgraded mode.
- the user explicitly opting out of one or both extra deliverables.

If the user asks only for the 源片风格圣经 or only for the 专业导演分析, still build or locate the frame evidence layer first (sampling, manifests, key seconds). A full foundational report file is not mandatory in that case, but the deliverable must cite the frame evidence paths it relies on and state that no separate foundational report was produced.

For every complete local-video lapian task, treat the deliverable package as three separate files unless the user explicitly asks for fewer:

1. `帧/自适应详细拉片报告`: mandatory foundation. It contains evidence frames, timecodes, scene facts, selected adaptive image rows, subtitles/sound notes, visible changes, and basic director commentary.
2. `专业导演分析`: independent interpretation report. It is written after the foundation report and explains why scenes are directed this way, what the director intends, how audience psychology is manipulated, and what methods creators can learn. Read `references/professional_director_analysis.md`.
3. `源片风格圣经`: independent style-transfer knowledge base for another AI. It extracts scene style cards, element cards, and image/video prompt templates. Read `references/style_transfer_bible.md`.

Do not merge all three into one Word file by default. The first report is the evidence base; the professional director analysis is the craft/intent reading; the style bible is the migration asset.

### default professional director analysis deliverable

For every complete local-video lapian task, generate an additional independent "专业导演分析" deliverable by default after the foundational frame/adaptive lapian report, unless the user explicitly says not to.

Default professional director analysis requirements:
- output an independent Markdown draft and an independent embedded-image DOCX when Word delivery tools are available.
- reference the foundational frame/adaptive lapian report as its evidence base.
- do not repeat the full adaptive frame table; use 3-8 representative images per major scene and write dense analysis around them.
- each major scene must include: `这一幕讲了什么事`, `核心节拍`, `高光分镜`, `导演意图层`, `观众心理层`, `视听执行层`, `主题表达层`, `与前后幕关系`, `可学习方法`, and representative frames.
- write from multiple angles: narrative function, directing intention, audience psychology, camera/staging/composition, light/color/art, performance, sound/editing, commercial rhythm, theme strategy, and transferable craft.
- use the user's preferred case style from `万物生 副本.docx`: explain why a choice works, not only what appears on screen.
- every representative frame's `为什么选这一帧` entry must be a unique single-frame deep reading, not a reusable caption. It should explicitly cover `情节`, `构图`, `导演意图`, and `含义`. If a row could be pasted onto another frame unchanged, rewrite it.

### default source style bible deliverable

For every complete local-video lapian task, generate an additional independent "源片风格圣经" deliverable by default, unless the user explicitly says not to. This applies to standard mode, full second-by-second mode, and image-text ultra detailed mode. Do not generate it by default for focus segment mode, screenshot-only mode, screenplay-only mode, or downgraded mode unless the user asks for style migration, style bible, AI knowledge base, image prompts, or video prompts.

The style bible is not a normal analysis appendix. It is a reusable source-film style library for another AI to learn the video's visual style and generate similar scene images or video prompts later. Read and follow `references/style_transfer_bible.md` when producing it.

Default style bible requirements:
- output an independent Markdown draft and an independent embedded-image DOCX when Word delivery tools are available.
- keep it separate from the main lapian report; do not bury it at the end of a long second-by-second report.
- target a medium-length Word document, roughly 20-50 pages.
- organize images by film/story segment, not by cold material taxonomy. Each segment should usually select 6-10 key images, unless the source is very short.
- make images function as a visual asset index for other AI systems.
- make source-plot function and visual style strongly bound: explain what narrative/emotional job each style asset performs.
- include scene-level style asset cards and prioritized element-level asset cards.
- include Chinese-first variable prompt templates for image generation and video generation. Add English only when the user requests a specific international model or bilingual prompts.
- do not add a "style replication test" section by default.
- copyright/similarity avoidance is optional, not mandatory; include it only when the user asks for safer adaptation of famous IP or brand-specific sources.

### standard mode

use for 3-15 minute videos when the user does not explicitly demand full second-by-second output.

output:
- complete structure timeline.
- scene-by-scene analysis.
- key shot table.
- key frame table.
- 1-3 "特别局部拉片" high-value passages.
- craft lessons.

### full second-by-second mode

use only when the user explicitly asks for 全片逐秒, 每秒拆, 逐秒详细分析, or equivalent.

requirements:
- split long videos into parts rather than compressing everything into one unreadable answer.
- for 10-minute videos, recommend part ranges such as 00:00-02:00, 02:00-04:00, 04:00-06:00, 06:00-08:00, 08:00-end.
- still start with complete structure timeline before the first detailed section.
- use frame sampling at 1-second interval if possible.
- minimum scene/subscene duration: 2 seconds. 1-second frame sampling is for evidence, not for creating 1-second analysis fragments. when adjacent seconds share the same action, staging, emotional state, or dramatic function, merge them into one paragraph/segment and list the report-display frames underneath it. isolated 1-second changes should be attached to the nearest preceding or following segment unless they are a true hard cut/title card/black frame/major reveal.

### evidence layer vs adaptive report display layer

default to a two-layer workflow:
- evidence layer: keep 1-second frame sampling and full frame manifests for audit, re-checking, and finding exact key seconds.
- report display layer: never assume every sampled second must be printed. Select an adaptive display interval from 0.5 to 2 seconds according to content density.
- normal segments: use about 2-second display cadence, plus first/last frames and important changes.
- dense segments: use 1-second display, or 0.5-second sampling/display for short focus passages only when movement, comedy timing, subtitle density, facial change, action, or a hard visual transformation genuinely needs it.
- sparse/repeated layer: for static holds, repeated walking, credits, unchanging title cards, or visually stable mood passages, show only representative start/middle/end frames and explain the continuity in prose.
- selection rule: judge by content, not by clock habit. A director-grade report may use 0.5s, 1s, 2s, or sparse representative frames in different passages of the same video.

do not confuse "逐秒证据" with "逐秒展示". a report can be director-grade even when it does not print every evidence frame, as long as the adaptive selection policy and preserved key seconds are documented.

### image-text ultra detailed second-by-second mode

use when the user asks for 图文逐秒, 带图逐秒, 每秒都要图, 图文超详细, or references a prior report like `帕奇赖特_赛博朋克短片_图文逐秒拉片报告.md`.

this is the most detailed default deliverable for users who want a reusable shot-study artifact, not just prose analysis.

requirements:
- output a standalone markdown report.
- do not overwrite existing reports; create a clearly named new file such as `片名_图文逐秒超详细拉片报告_v01.md` (follow the `_v01` versioned naming from `references/delivery_audio_workflow.md`) unless the user explicitly asks to overwrite.
- start with 素材与精度说明 and 全片结构速览.
- before any large frame table, write case-grade scene analysis blocks. Each major scene must first answer `这一幕讲了什么事`, `核心节拍`, `高光分镜`, `画面调度/构图/视角`, `光影色彩与美术`, `角色情绪与表演`, `与上一幕/下一幕的关系`, and `主题与导演意图`. Use the frame table as evidence, not as a substitute for director judgment.
- include a section named `图文变化表` or `图文自适应变化表`. state that the evidence layer can remain 1fps while the report layer uses adaptive 0.5-2s display plus key seconds. Only call the table `图文逐秒变化表` when the user explicitly asked every sampled second to appear in the deliverable.
- split the full video into meaningful 2-20 second segments. segments can be longer only when the image, action, staging, emotional state, and narrative function are genuinely continuous.
- each segment must include:
  - `### start-end 段落标题`
  - `**段落：**` major story section name.
  - `**导演分析：**` a dense paragraph explaining narrative function, camera/staging/composition, sound/subtitle/UI evidence, emotional change, theme/craft value, and why the segment exists. It must name the director's tactic, e.g. contrast, misdirection, scale pressure, foreground blocking, sound-image counterpoint, meta-gag, value reversal, or visualized metaphor.
  - a report-frame table with columns exactly: `时间 | 对应抽帧 | 节点记录 | 字幕/声音`.
- each displayed row must include the selected frame image link, not only the timecode. example:
  `| 00:05 | ![00:05](<片名_逐秒抽帧/frames/frame_000006.jpg>) | ... | ... |`
- do not write the same sentence for every second. for repeated action, vary the node record by focusing on composition, posture, gaze, blocking, movement, UI/subtitle, sound cue, or how this second carries the segment's function.
- still avoid false precision: if subtitles, sound, or exact edit points are uncertain, say "无完整逐字字幕可确认；以画面动作和声场推进" or equivalent.
- after the 图文逐秒变化表, add a compact director recap:
  - 最关键结构设计.
  - 最值得逐帧复看的段落.
  - 摄影和构图核心规律.
  - 可迁移创作方法.
  - 最终总结.
- for videos around 10 minutes, a complete image-text report may contain hundreds of table rows. write it to a markdown file instead of trying to fit the whole artifact into chat.
- if generating mechanically from a frame manifest, prefer `scripts/select_lapian_report_frames.py` to create a report-frame selection plan. verify markdown image references are at least the selected report-frame count, not necessarily the full evidence-frame count, unless the user explicitly asked for every sampled second to be printed.

### focus segment mode

use when the user provides timecodes or asks to analyze a specific scene.

requirements:
- analyze only the requested segment deeply.
- include relation to the surrounding story if known.
- use 0.5-1 second key-frame sampling for dense action, comedy, or visual transformation moments when possible.

### downgraded mode

use for scripts, screenshots, storyboards, links that cannot be accessed, or partial descriptions.

requirements:
- clearly label what cannot be judged.
- do not invent actual lens movement, edit points, music, performance, or lighting beyond the material.
- use "possible shot design" or "reasonable inference" language when proposing director solutions.

## evidence rules

use evidence levels for important claims:

- e1 直接证据: visible image, audible line, subtitle, user transcript, explicit action, or directly observable edit.
- e2 合理推断: supported by multiple e1 facts, but not explicitly stated by the material.
- x 无法判断: insufficient material.

apply evidence labels to key judgments, contested claims, and limitations. do not mark every sentence if it harms readability.

## output skeleton

use a fixed skeleton plus case-style high-value 拉片. for detailed templates, read `references/report_structure.md` when needed.

minimum order:

1. 素材诊断
2. 素材总览
3. 全片 / 全段结构时间轴
4. 逐幕 / 逐场景深度拉片
5. 关键镜头 / 关键秒 / 关键帧表
6. 镜头语言与摄影设计
7. 画面调度、构图与视角
8. 光影、色彩与美术
9. 声音、音乐与剪辑节奏
10. 表演与人物情绪曲线
11. 场景价值翻转
12. 对白与潜台词
13. 主题与导演策略
14. 可学习创作方法
15. 最终总结

for each major scene, answer:
- what happens?
- who wants what?
- what obstacle appears?
- what changes by the end?
- what new information does the audience get?
- what are the key shots or frames?
- how do camera, blocking, composition, light, color, art, sound, editing, and performance serve story?
- what can creators learn from this passage?

case-grade report standard:
- do not let the report become "many images plus generic captions". First explain the scene's dramatic engine, then use selected frames as proof.
- every major scene needs at least one "why this works" paragraph. Explain how the director manipulates audience attention, emotion, expectation, or interpretation.
- identify high-value craft moments, such as a background gag, sudden medium switch, table/door/window framing, prop invasion, fourth-wall gesture, value reversal, color shift, or sound-image mismatch. Explain why they are not random decoration.
- include scene-to-scene relationship: contrast, escalation, payoff, misdirection, reset, or hook.
- include one creator-facing takeaway per major scene. It should be a reusable method, not a vague praise sentence.

## style requirements

write in chinese unless the user asks otherwise.

avoid generic praise such as "电影感很强", "高级", "氛围好" unless immediately supported by craft evidence.

prefer concrete director language:
- "这个镜头为什么存在"
- "这个道具承担了什么叙事功能"
- "剪辑点落在动作、眼神、声音还是信息上"
- "前景、中景、后景是否同时讲不同信息"
- "这个段落的价值从什么翻转到什么"
- "观众的视线如何被引导、阻断或欺骗"

when naming a director tactic or comparing a shot to a master's signature technique, read `references/craft_pattern_library.md` and follow its rules: match the card's core recognition features before applying a label, phrase master comparisons as "与X同源/近似" (E2) rather than claiming homage, and always state what problem the tactic solves in this video, not just its name.

include commercial/creator analysis when relevant:
- opening hook
- completion-rate logic
- fan service as staging, gaze, or rhythm design
- visual shareability
- scene as reusable directing method

for sexualized or fan-service imagery, analyze camera gaze, blocking, consent/framing, comedic obstruction, audience psychology, and commercial function. do not write voyeuristic or explicit description.

## frame sampling workflow

when analyzing uploaded video and precision matters, prefer the full evidence preparation script first. it creates the standard archive folders, records ffprobe metadata, samples frames, extracts 16k mono audio when available, writes audio levels, and creates `manifest.json`.

```bash
python scripts/prepare_lapian_evidence.py path/to/video.mp4 --out-root 08拉片输出 --project-name 片名 --task-name 导演级拉片 --interval 1
```

ASR is intentionally optional because it can download models or take a long time:

```bash
python scripts/prepare_lapian_evidence.py path/to/video.mp4 --project-name 片名 --interval 1 --asr-model base
```

use the lower-level frame sampler only when you need a custom frame-only extraction:

```bash
python scripts/video_frame_sampler.py path/to/video.mp4 --out-dir /tmp/lapian_frames --interval 2 --contact-sheet
```

for full second-by-second mode:

```bash
python scripts/video_frame_sampler.py path/to/video.mp4 --out-dir /tmp/lapian_frames --interval 1 --contact-sheet
```

for a dense segment:

```bash
python scripts/video_frame_sampler.py path/to/video.mp4 --out-dir /tmp/lapian_frames --start 00:04:00 --duration 00:02:00 --interval 0.5 --contact-sheet
```

shot boundary detection runs by default inside `prepare_lapian_evidence.py` and writes `01_逐秒抽帧/片名_镜头切分.json` (cut list, shot table, average/median shot length). to run it alone:

```bash
python scripts/detect_shot_cuts.py path/to/video.mp4 --out output/01_逐秒抽帧/片名_镜头切分.json --threshold 0.3
```

when a detected cut list exists, shot tables and editing-rhythm analysis should be based on it and labeled "基于场景检测的镜头切分". detected hard cuts count as strong evidence (E1-adjacent), but keep the caveat: dissolves, slow transitions, and fade-to-black can be missed; fast motion, flicker, and strong light effects can create false cuts. spot-check suspicious cuts against sampled frames before building conclusions on them. use shot count and average/median shot length from the stats block for editing-rhythm claims instead of estimating.

use extracted frames to support visual observations. only when no cut list exists, fall back to grouping from frames and call it "粗略镜头切分" rather than professional edl.

when making an image+text second-by-second report, group frames into analysis segments with a minimum duration of 2 seconds by default. do not write duplicated analysis for every sampled second; write one segment analysis, then attach the corresponding report-frame/time rows.

for image-text ultra detailed reports, prefer the bundled frame sampler at 1-second interval and contact sheets:

```bash
python scripts/video_frame_sampler.py path/to/video.mp4 --out-dir output/片名_逐秒抽帧 --interval 1 --contact-sheet
```

then generate a report-display frame plan. this keeps the 1fps evidence archive while making the report readable:

```bash
python scripts/select_lapian_report_frames.py --manifest output/片名_逐秒抽帧/frames_manifest.json --subtitle-json output/02_音频分析/片名_asr_from_vtt_subtitles.json --levels-csv output/02_音频分析/片名_audio_levels.csv --report-interval 2 --out output/03_MD报告/片名_报告展示帧清单.json --csv-out output/03_MD报告/片名_报告展示帧清单.csv
```

the subtitle and levels inputs are optional; the plan still works without them. for short videos (about 1 minute or less), lower `--top-audio-peaks` to 3-5, otherwise the default of 20 marks almost every second as a key second and the sparse display collapses.

pass the selection plan to QA when the report intentionally displays fewer images than the 1fps evidence archive. the selection plan applies to the main lapian report only; the professional director analysis and the style bible are judged by their own image rules:

```bash
python scripts/qa_lapian_delivery.py --project-dir output/片名_导演级拉片 --selection-plan output/03_MD报告/片名_报告展示帧清单.json --strict
```

if `python` is unavailable on Windows, try `py` before changing approach. if Pillow or another dependency is missing, diagnose the exact missing component; do not recursively delete environments or node_modules.

## resume, handoff, and finalization workflow

when the user points to an existing output folder or asks to "continue", first scan the project state instead of restarting from scratch:

```bash
python scripts/lapian_delivery_status.py --project-dir 08拉片输出/YYYYMMDD/片名_导演级拉片
```

use the `done` and `next_steps` fields as the working checklist. preserve completed artifacts unless they are stale because the Markdown report changed after DOCX/PDF/QA generation. if the Markdown report is edited after conversion, regenerate DOCX, PDF, QA JSON, and manifest.

for final package handoff, use a stable QA output path so the latest audit is easy to find. in `--project-dir` mode the QA script audits every Markdown report under `03_` and every DOCX under `04_`, including the per-deliverable checks (professional frame-reason deep reading, style bible structure and image assets); a single run covers the whole three-deliverable package:

```bash
python scripts/qa_lapian_delivery.py --project-dir 08拉片输出/YYYYMMDD/片名_导演级拉片 --overwrite-out --strict
```

after QA passes, update `manifest.json` with the final report, DOCX, PDF, QA, frame, and ASR paths:

```bash
python scripts/lapian_finalize_manifest.py --project-dir 08拉片输出/YYYYMMDD/片名_导演级拉片
```

keep versioned QA backups only when deliberately preserving an earlier audit. the final handoff should always have one stable `06_QA审计/lapian_delivery_qa.json` or a clearly named stable QA JSON.

before final delivery, run the bundled audit script when report files, frames, audio, DOCX, or a project archive exist:

```bash
python scripts/qa_lapian_delivery.py --project-dir 08拉片输出/YYYYMMDD/片名_导演级拉片 --overwrite-out --strict
```

if auditing a single markdown report:

```bash
python scripts/qa_lapian_delivery.py --report path/to/report.md --frames-dir path/to/frame_archive --strict
```

## audio, delivery, and final audit addendum

when the user asks for audio, music, sound effects, dialogue, voice, subtitles, Feishu upload, Word conversion, embedded images, or a complete deliverable package, read `references/delivery_audio_workflow.md` and follow it.

additional rules:
- if the user names an output folder, put the report, frames, audio analysis, and converted deliverables under that folder or a clearly named child folder.
- if the user does not name an output folder, use the dated archive pattern `08拉片输出/YYYYMMDD/片名_任务名/` and keep frames, audio, reports, Word/PDF deliverables, and QA records in separate child folders.
- for complete local-video lapian tasks, first create or preserve the foundational frame/adaptive detailed lapian report; the professional director analysis and source style bible must be based on that evidence layer.
- for complete local-video lapian tasks, also read `references/professional_director_analysis.md` and create the independent professional director analysis deliverable by default, unless the user explicitly opts out.
- for complete local-video lapian tasks, also read `references/style_transfer_bible.md` and create the independent source style bible deliverable by default, unless the user explicitly opts out.
- for Feishu upload, prefer a DOCX report with images embedded. Markdown with local image paths is useful for editing but is not a reliable self-contained Feishu upload artifact.
- when converting a Markdown image report to DOCX, prefer `scripts/md_image_report_to_docx.py` if it fits the report structure.
- when audio is requested, include audio evidence in the main report, not only in side files.
- when writing full image reports, keep 1fps evidence on disk when useful, but default the visible report to adaptive 0.5-2s display plus key seconds. only print every sampled second when the user explicitly asks for every second/every frame in the deliverable.
- before final delivery, verify image reference count, missing image count, first/last second coverage, audio artifact existence when requested, and UTF-8 readability.

## mandatory critique pass before delivery

for every complete report deliverable (foundational lapian report, professional director analysis, style bible), do a second full pass over the draft before QA, DOCX conversion, or final reply. this is not the quality checklist below; it is an adversarial re-read in a different stance: a harsh senior reviewer hunting for weaknesses, not the author admiring the draft.

hunt specifically for:

1. 过度解读: interpretation claims (E2) written as if they were visible facts (E1); cut the claim or downgrade and label it.
2. 无证据运镜: camera-movement, edit-point, or sound-design claims with no continuous-frame, cut-list, or audio evidence behind them; rewrite using the conservative phrasings from the evidence rules.
3. 反证遗漏: at least once per major scene, ask "哪一帧/哪一秒和我的解读矛盾？" if a contradicting frame exists, address it instead of ignoring it.
4. 套话与复读: generic praise, template sentences, and row notes that repeat the same structure; rewrite with the specific craft observation that earns the sentence.
5. 注意力平均化: if every scene got the same word count and the same enthusiasm, the report is summarizing, not judging; expand the one or two passages that genuinely deserve depth and compress the routine ones.

after the critique pass, apply the fixes directly. if the pass found nothing to fix, that is suspicious for any report longer than a few pages; re-read the densest scene once more before accepting it. do not show the critique notes in the deliverable; only the improved text ships.

## quality self-check before answering

internally verify:

- did you diagnose material type and precision first?
- did you build a complete timeline before detailed commentary?
- did each scene state what happened, what changed, and why it matters?
- did you distinguish e1/e2/x where needed?
- did you avoid invented lens, sound, edit, or performance details?
- if a shot-cut list exists, did shot tables and editing-rhythm claims use it instead of guessing, with suspicious cuts spot-checked against frames?
- did you run the mandatory critique pass and apply its fixes before QA/DOCX/final reply?
- did you include key frames or key seconds when visual material exists?
- did you explain craft function rather than merely naming techniques?
- did you include creator-transferable methods?
- if the user requested audio, did you analyze music, sound effects, voice/dialogue, silence, and sound-image relation with evidence limits?
- if the user requested Feishu/Word delivery, did you create or recommend a self-contained DOCX with embedded images and verify embedded media/drawing references where possible?
- if the user requested 图文超详细 mode, did you document whether the report uses full 1fps display or an adaptive 0.5-2s display plan, and did QA verify image references against the right expected count?
- for complete local-video lapian, did you produce the foundational frame/adaptive detailed report first, or explicitly identify an existing one as the evidence base?
- for complete local-video lapian, did you produce or explicitly explain the absence of the independent professional director analysis Word?
- does the professional director analysis explain why scenes are directed this way from director intention, audience psychology, visual execution, theme strategy, and transferable craft angles?
- in the professional director analysis, did every representative frame answer why this exact frame was selected with concrete `情节 / 构图 / 导演意图 / 含义`, with no repeated generic reason?
- for complete local-video lapian, did you produce or explicitly explain the absence of the independent source style bible Word?
- does the source style bible contain segment-based image asset indexes, scene-level asset cards, prioritized element-level asset cards, Chinese variable image prompts, and Chinese variable video prompts?

only show the checklist if the user asks for it.
