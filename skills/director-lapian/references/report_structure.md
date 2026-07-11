# director-lapian report structure reference

use this reference when a full structured report is needed.

## 素材总览 table

| 项目 | 内容 |
|---|---|
| 素材类型 | 视频 / 片段 / 剧本 / 分镜 / 截图 / 描述 / 链接 |
| 总时长 / 篇幅 |  |
| 本次分析精度 | 逐帧 / 逐秒 / 关键秒 / 逐镜头 / 逐场景 / 截图顺序 |
| 主要人物 |  |
| 疑似主角 | 写依据，不明确则说核心视角 |
| 主要场景 |  |
| 核心事件 |  |
| 整体类型 | 神话 / 废土 / 都市 / 悬疑 / 动作 / 情感 / 喜剧 / 纪录 / 广告 / mv / 其他 |
| 整体情绪基调 |  |
| 主要视听风格 |  |
| 主要叙事结构 | 三幕式 / 多线并行 / 元叙事 / 反转式 / 情绪流 / 其他 |
| 当前可可靠判断 |  |
| 当前无法判断 |  |

## complete timeline table

| 幕 / 段落 | 时间范围 | 这一段讲了什么事 | 核心节拍 | 主角状态变化 | 世界状态变化 | 叙事功能 | 情绪变化 |
|---|---:|---|---|---|---|---|---|

requirements:
- every scene must have a concrete time range.
- write what physically happens, not only "建立世界观" or "制造悬念".
- describe character state or world state change.
- describe emotional change.

## narrative layer table

use this table when the material mixes reality, memory, dream/consciousness, platform cards, setting cards, behind-the-scenes fragments, UI interfaces, or meta-narrative layers.

| 时间范围 | 叙事层级 | 画面证据 | 声音 / 字幕证据 | 该层级的叙事功能 | 观众获得的信息 | 证据等级 |
|---|---|---|---|---|---|---|

common layer names:
- 现实线
- 意识 / 梦境 / 心象线
- 回忆线
- 平台 / UI / 系统信息层
- 设定卡 / 情报卡
- 花絮 / 制作展示层
- 旁白评论层

do not force a layer label when the evidence is weak. mark it as "疑似" or evidence level X when necessary.

## image-text ultra detailed adaptive-frame report

use this template when the user asks for 图文逐秒, 带图逐秒, 每秒都要图, 图文超详细, or references a previous image-text second-by-second report.

the artifact should be a standalone markdown file, usually named:

```text
片名_图文逐秒超详细拉片报告_v01.md
```

do not overwrite an existing report unless explicitly requested.

### required opening

```markdown
# 《片名》图文逐秒超详细拉片报告

生成者：啪趴虾

版本说明：本报告用于核对画面、字幕、动作、调度和叙事功能。它不是普通精读摘要，而是“段落导演分析 + 0.5-2 秒自适应展示帧表”的图文拉片底稿。

## 素材与精度说明

- 使用视频：
- 视频时长：
- 证据层抽帧：通常 1 秒 1 张代表帧；高密度局部可用 0.5 秒；全部证据帧留档供复核。
- 报告展示层：按内容密度自适应选择 0.5-2 秒展示帧；常规段落约 2 秒一图，关键变化回到 1 秒或 0.5 秒，静态/重复段落只放代表帧。
- 这里的“图文超详细”不是 24fps 全帧报告，也不等于每个镜头每秒都必须写入 Word。
- 声音/字幕栏主要依据画面内硬字幕、UI、可听对白方向和段落声场判断；未做专业音轨分离或完整 ASR。
- 证据标记：E1 / E2 / X。
```

### full structure quick view

```markdown
## 全片结构速览

| 段落 | 时间 | 叙事功能 | 情绪变化 |
|---|---:|---|---|
|  |  |  |  |
```

### image-text adaptive-frame table section

```markdown
## 图文自适应变化表

阅读方式：先读每一幕的导演级场景拆解，再看该幕内的自适应展示帧。展示间隔由内容密度决定，通常在 0.5-2 秒之间变化；这样同一场景不机械复制同一句，也不把 Word 撑成逐秒全量帧库。

### 00:00-00:39 序章：远古废土与神话解构

#### 这一幕讲了什么事
用 3-6 句写清楚：发生了什么、谁的处境被建立、世界规则是什么、段落结束时观众多知道了什么。不要只写“建立世界观”；必须写出画面中具体发生的事件。

#### 核心节拍
- 00:00-00:10：第一个事件/信息点，说明叙事功能和情绪效果。
- 00:10-00:25：第二个事件/信息点，说明观众获得的新信息。
- 00:25-00:39：第三个事件/信息点，说明段落怎样收束或转向。

#### 高光分镜
- `第一个分镜：时间范围 + 景别/运镜 + 画面事件 + 情绪效果 + 叙事功能。`
- `第二个分镜：...`
- 高光分镜要写“为什么这一镜值得停下来学”，例如反差设计、尺度压迫、前景遮挡、背景彩蛋、跨媒介切换、声音反讽、画面内视线控制。

#### 画面调度、构图与视角
写清楚导演如何安排主体、前景、中景、后景、视线、遮挡、框中框、低角度/俯视/主观视角，以及这些安排如何改变权力关系或观众注意力。

#### 光影、色彩与美术
写清楚主要光源、冷暖关系、主色如何服务情绪；道具、服装、场景材质如何承载主题。不要只写“电影感”“高级”，要说它具体高级在哪里。

#### 角色情绪与表演
写角色的外在动作、表面情绪、深层情绪和微表情证据：眼神、停顿、吞咽、握拳、低头、闪避、失语、爆发。

#### 与上一幕 / 下一幕的关系
说明这一幕是反差、升级、铺垫、误导、回收、情绪缓冲，还是下一幕的钩子。

#### 主题与导演意图
写表层主题、深层主题，以及导演把主题视听化的方式。必须有一段“为什么这样有效”的判断。

### 00:00-00:06 段落标题

**段落：** 大段落名

**导演分析：** 这一组连续画面承担的是「叙事功能」。画面上主要表现为：……；镜头和调度上重点看……；声音/字幕/UI 上主要依靠……。这个小段的核心不是单秒孤立信息，而是把观众从上一节拍推到下一节拍。创作者可学习点：……

| 时间 | 对应抽帧 | 节点记录 | 字幕/声音 |
|---:|---|---|---|
| 00:00 | ![00:00](<片名_逐秒抽帧/frames/frame_000001.jpg>) | 段落起点：…… | …… |
| 00:02 | ![00:02](<片名_逐秒抽帧/frames/frame_000003.jpg>) | 常规 2 秒展示帧：…… | …… |
| 00:03 | ![00:03](<片名_逐秒抽帧/frames/frame_000004.jpg>) | 关键变化补帧：…… | …… |
```

### case-grade director-analysis rules

- A good report is not only "what is on screen"; it explains the director's tactic.
- Every major scene should include at least one paragraph shaped like: `表层策略：...；深层策略：...` or `导演这样做不是闲笔，因为...`.
- When there is comedy, horror, erotic tension, spectacle, or emotional reversal, explain how the timing works: what makes the viewer relax, expect, misunderstand, laugh, fear, or re-interpret.
- If there is a background action or prop intrusion, decide whether it is an information cue, a mood valve, a fourth-wall gesture, a visual metaphor, or a transition device.
- If a shot feels visually impressive, translate it into craft language: scale contrast, lens pressure, god's-eye view, foreground invasion, silhouette, color coding, sound bridge, action cut, reaction cut, or value reversal.
- End each major scene with `可学习方法：...`, phrased as a reusable technique for new scripts.

### adaptive display-row writing rules

- do not print every sampled second by default.
- every selected report-display frame must have one image row.
- only when the user explicitly says "每一秒都要放图 / 逐秒全部图 / every sampled second" should every 1fps evidence frame become a report row.
- do not repeat the same note for every second.
- vary the row note by focusing on:
  - composition / blocking.
  - posture / gaze / gesture.
  - foreground / midground / background information.
  - UI / subtitle / sound cue.
  - camera pressure or movement impression.
  - how the second supports the segment's narrative function.
- if no exact subtitle is available, write:
  `无完整逐字字幕可确认；以画面动作、环境声、音乐/能量/观众反应推进。`
- if a subtitle or UI is directly visible, quote or summarize it briefly, without inventing missing text.
- if exact shot boundaries are not verified, call any shot grouping "粗略镜头切分".

### required final recap for image-text reports

```markdown
## 逐秒报告后的导演复盘

### 1. 全片最关键的结构设计

### 2. 最值得逐帧复看的段落

### 3. 摄影和构图核心规律

### 4. 可迁移创作方法

## 最终总结
```

### verification for image-text reports

before final response:

- verify the markdown file exists and is UTF-8 readable.
- verify image reference count equals the selected report-display frame count. Only require full sampled frame count when the user explicitly asked every sampled second to be printed.
- verify the final referenced frame exists.
- report the path, segment count, image row count, and any limitations.

## professional director analysis report

Use `references/professional_director_analysis.md` for the full template and writing standard.

The professional director analysis is a separate director-intent deliverable, not a replacement for the `帧/自适应详细拉片报告`. For complete local-video lapian tasks, produce it after the foundational report unless the user opts out.

Minimum content:
- 素材与分析依据, explicitly pointing to the foundational frame/adaptive lapian report.
- 全片导演策略总览.
- scene-by-scene professional analysis.
- for each major scene: `这一幕讲了什么事`, `核心节拍`, `高光分镜`, `导演意图层`, `观众心理层`, `视听执行层`, `主题表达层`, `与前后幕关系`, `可学习方法`, and representative frames.
- every `为什么选这一帧` row must be a single-frame interpretation with `情节 / 构图 / 导演意图 / 含义`; do not use a shared generic reason across all frames.
- director-facing recap: core strategy, strongest craft moments, audience psychology, transferable methods, and what to preserve/replace when migrating to a new script.

Default delivery:
- `片名_专业导演分析_v01.md`.
- `片名_专业导演分析_嵌图版_v01.docx` when Word embedding is available.

## source style bible report

Use `references/style_transfer_bible.md` for the full template and QA rules.

The source style bible is a separate AI knowledge-base deliverable, not a section appended to the main lapian report. For complete local-video lapian tasks, produce it by default unless the user opts out.

Minimum content:
- 风格定位总述.
- 风格 DNA 总表.
- 影片段落式图片资产索引, with 6-10 key images per major segment when possible.
- 场景级风格资产卡.
- 分优先级的元素级风格资产卡.
- 中文图像生成变量模板.
- 中文视频生成变量模板.
- 风格保持与跑偏警报.

Default delivery:
- `片名_源片风格圣经_v01.md`.
- `片名_源片风格圣经_嵌图版_v01.docx` when Word embedding is available.

## scene analysis block

for each major scene:

```markdown
## 【幕名】：【scene title】（start - end）

### 这一幕讲了什么事
3-6 sentences. include event, character goal, obstacle, end-state change, and new audience information.

### 核心节拍表
| 节拍 | 时间码 | 发生了什么 | 观众获得的信息 | 情绪效果 | 叙事功能 | 证据等级 |
|---|---:|---|---|---|---|---|

### 关键秒 / 逐秒拆解表
| 时间点 | 画面发生什么 | 景别 | 镜头运动 | 构图 / 视角 | 动作变化 | 情绪变化 | 声音 / 台词 / 音乐 | 剪辑点 | 叙事功能 |
|---:|---|---|---|---|---|---|---|---|---|

### 逐镜头拆解表
| 镜头编号 | 时间范围 | 镜头时长 | 画面事件 | 景别 | 机位 / 角度 | 镜头运动 | 构图 | 焦点 / 景深 | 光影色彩 | 声音 | 剪辑方式 | 情绪效果 | 叙事功能 | 专业点评 |
|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|
```

if true shot boundaries are not verified, label the table "粗略镜头拆解".

## key frame table

| 关键帧 | 时间码 | 画面内容 | 主体 | 前景 / 中景 / 后景 | 构图方式 | 色彩与光影 | 人物状态 | 画面信息量 | 视觉记忆点 | 这一帧为什么重要 |
|---|---:|---|---|---|---|---|---|---|---|---|

prioritize:
- opening frame.
- first character close-up.
- first world-building shot.
- first emotional turn.
- first crisis.
- first visual spectacle.
- end frame of each major scene.
- most shareable cover frame.
- best poster frame.

## craft modules

### 画面调度、构图与视角
analyze center, edge, foreground, midground, background, power, gaze, framing, scale, crowd blocking, background gags, frame-within-frame, subjective view, god's-eye view, surveillance view, fourth-wall awareness.

### 摄影、景别与镜头运动
explain functions of wide, full, medium, close-up, extreme close-up. for movement, state why it moves, who it follows, what information changes, and whether it creates pressure, speed, intimacy, comedy, or spectacle.

### 光影、色彩与美术
include key light direction when visible, contrast, silhouette, dominant colors, cold/warm contrast, safety/danger color coding, setting material, props, costume function, mechanical/religious/modern/classical symbols, visual motifs.

### 声音、音乐与剪辑
include environment sound, effects, music, voiceover, dialogue, silence, sound bridge, sound-image counterpoint, action cut, reaction cut, information cut, rhythm changes. if sound cannot be judged, state it.

### 表演与情绪曲线
| 阶段 | 时间范围 | 外在事件 | 主角外在动作 | 表面情绪 | 深层情绪 | 情绪转折原因 | 证据 |
|---|---:|---|---|---|---|---|---|

### 场景价值翻转
| 幕 / 场景 | 时间范围 | 价值起点 | 价值终点 | 转折点 | 谁付出代价 | 视觉记忆点 |
|---|---:|---|---|---|---|---|

common values: safe to dangerous, real to surreal, ordinary to mythic, control to loss of control, ignorance to knowledge, human to object/part/resource, order to chaos, serious to absurd, false to higher truth.

### 对白与潜台词
| 时间码 | 角色 | 台词 / 概述 | 表面意思 | 实际行动 | 潜台词 | 对关系 / 剧情的影响 | 证据等级 |
|---:|---|---|---|---|---|---|---|

actual dialogue actions include threat, seduction, plea, mockery, test, concealment, misdirection, farewell, judgment, delay, control, help-seeking, self-protection, bargaining, power display, topic shift, information release.

### 主题与导演策略
include:
- 表层主题
- 深层主题
- 主题递进
- 主题的视听化方式
- 主题的角色化方式
- 主题的道具化方式
- 主题的空间化方式

strategy table:

| 策略 | 具体做法 | 出现段落 | 产生效果 | 可学习点 |
|---|---|---|---|---|

### 可学习创作方法
| 维度 | 可学习技巧 | 具体出现在哪里 | 为什么有效 | 可迁移到什么场景 |
|---|---|---|---|---|

include writing, directing, cinematography, editing, acting, art direction, sound, action design, comedy design when relevant.

## 视听与交付补充模板

当报告包含音频分析、飞书上传或 Word 嵌图交付时，补充读取 `references/delivery_audio_workflow.md`，并在报告中加入以下模块。

### 音频证据概览

| 项目 | 结果 | 证据文件 / 说明 |
|---|---|---|
| 是否有音轨 |  |  |
| 音频时长 |  |  |
| ASR / 字幕来源 |  |  |
| 音乐总体策略 |  |  |
| 主要音效类型 |  |  |
| 人声 / 对白状态 |  |  |
| 听辨限制 |  |  |

### 分段声音设计表

| 时间范围 | 音乐 | 音效 | 人声 / 对白 | 环境声 | 静默 / 留白 | 声画关系 | 叙事功能 | 证据等级 |
|---|---|---|---|---|---|---|---|---|

### 对白与潜台词表

| 时间码 | 角色 | 台词 / ASR 概述 | 表面意思 | 实际行动 | 潜台词 | 对关系 / 剧情的影响 | 证据等级 |
|---:|---|---|---|---|---|---|---|

### 交付审计表

| 检查项 | 结果 |
|---|---|
| 报告 Markdown 路径 |  |
| 抽帧目录 |  |
| 音频分析目录 |  |
| Markdown 图片引用数量 |  |
| 缺失图片数量 |  |
| 第一秒 / 最后一秒覆盖 |  |
| DOCX 嵌图版路径 |  |
| DOCX 内媒体 / 绘图引用数量 |  |
| PDF 或预览抽查 |  |
| 当前限制 |  |

## final summary format

end full reports with:

1. 这个片段 / 影片最重要的叙事作用是：
2. 最精彩的一幕是：
3. 最关键的时间点是：
4. 主角最重要的情绪变化是：
5. 最突出的视听特点是：
6. 真正表达的主题是：
7. 最值得学习的创作方法是：
8. 最适合反复拉片学习的镜头是：
9. 最能体现导演能力的设计是：
10. 如果只学习一个方法，最该学习的是：

then add one complete professional evaluation sentence:

"这是一个通过【核心手法】完成【叙事 / 情绪 / 主题效果】的片段，其创作价值在于【专业判断】。"
