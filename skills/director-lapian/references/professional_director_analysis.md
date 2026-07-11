# professional director analysis reference

Use this reference when producing the independent `专业导演分析` deliverable.

This document is based on the user's preferred case style exemplified by `万物生 副本.docx`: the value is not more screenshots, but multi-angle explanation of why the scene is directed this way.

## relationship to the three deliverables

The authoritative trigger boundary for the default three-deliverable package (including which requests do NOT trigger it) is the "default three-deliverable package" section in `SKILL.md`. This file only details the professional analysis deliverable itself.

For complete local-video lapian tasks, deliver three separate files unless the user opts out:

1. `帧/自适应详细拉片报告`: the foundation. It preserves evidence, timecodes, selected frames, subtitles/sound notes, visible changes, and scene facts.
2. `专业导演分析`: the director-level interpretation built on the foundation report. It explains intent, strategy, audience psychology, craft design, and transferable methods.
3. `源片风格圣经`: the style-transfer knowledge base for another AI. It converts the source film style into scene cards, element cards, and image/video prompt templates.

Do not merge these three into one Word file by default. The professional director analysis may reference selected frames, but it should not repeat the whole adaptive frame table.

## purpose

The professional director analysis answers:

- Why does this scene exist?
- Why did the director shoot it this way?
- What does this shot make the audience expect, feel, misunderstand, fear, laugh at, or re-interpret?
- How do camera, staging, composition, light, color, art, sound, editing, and performance work together?
- What is the visible evidence?
- What method can be learned and migrated into a new project?

## default delivery

- Markdown: `片名_专业导演分析_v01.md`
- DOCX with embedded selected images when Word delivery is available: `片名_专业导演分析_嵌图版_v01.docx`

Default image density:
- use 3-8 representative images per major scene.
- use more only for very important set pieces, mixed-media transitions, action beats, comedy timing, expression turns, or visual transformations.
- do not embed the full 1fps evidence archive.

## required opening

```markdown
# 《片名》专业导演分析 v01

生成者：啪趴虾

## 阅读说明

本文件基于《片名_帧/自适应详细拉片报告》生成。基础拉片报告负责时间码、证据帧和画面变化；本文负责从专业导演角度解释“为什么这样拍、意图是什么、观众被怎样调度、创作者能学走什么”。
```

## core structure

```markdown
## 01 素材与分析依据

| 项目 | 内容 |
|---|---|
| 源视频 |  |
| 基础拉片报告 |  |
| 抽帧证据 |  |
| ASR/字幕证据 |  |
| 听辨/画面限制 |  |

## 02 全片导演策略总览

### 一句话判断
用一句话概括全片最核心的导演策略。

### 全片策略表
| 策略 | 具体做法 | 作用于观众 | 服务的叙事/主题 | 代表段落 |
|---|---|---|---|---|
```

## major scene block

Every major scene must use this structure:

```markdown
## 【第一幕】场景标题（00:00-00:39）

### 这一幕讲了什么事
3-6 句。写清楚事件、角色目标、阻力、结尾状态变化、观众获得的新信息。不要只写“建立世界观”或“制造情绪”。

### 核心节拍
| 节拍 | 时间 | 发生了什么 | 观众获得的信息 | 情绪效果 | 叙事功能 |
|---|---:|---|---|---|---|

### 高光分镜
第一个分镜：时间范围 + 景别/运镜 + 画面事件 + 情绪效果 + 叙事功能。

第二个分镜：...

### 导演意图层：为什么要这么拍
解释导演为什么选择这种进入方式、视角、节奏、信息释放顺序、反差或压迫方式。必须回答“如果换一种更普通的拍法，会损失什么”。

### 观众心理层：导演如何操控观众
解释观众在这一幕中的心理变化：被吸引、放松、误判、期待、被打断、被压迫、被震惊、重新理解。重点写 timing 和 expectation。

### 视听执行层：怎么做到的
- 镜头/景别/运镜：
- 调度/构图/视角：
- 光影/色彩/美术：
- 声音/音乐/剪辑：
- 表演/表情/身体动作：

### 主题表达层：这一幕如何把主题拍出来
写表层主题、深层主题，以及主题如何被道具、空间、身体、光线、声音或动作视觉化。

### 与前后幕关系
说明这一幕和上一幕/下一幕的关系：反差、升级、误导、回收、铺垫、缓冲、钩子、价值翻转。

### 可学习方法
用创作者能直接迁移的方式写 1-3 条。例如：
- 先用喜剧让观众卸下防备，再用惊悚反杀。
- 把抽象命运变成可见的物理阻挡。
- 用前景物体暴力侵入镜头，切断观众窥视欲。

### 代表帧
| 时间 | 帧图 | 为什么选这一帧 |
|---:|---|---|
```

### representative-frame deep-reading rule

The `为什么选这一帧` column is mandatory craft analysis, not an image caption. Every representative frame must explain the specific reason this exact image was selected.

Each row must include four concrete parts:

```text
情节：这一秒具体发生了什么。
构图：主体、前景/中景/后景、遮挡、光线、色彩、道具或空间关系如何组织。
导演意图：导演为什么选择这种拍法，而不是更普通的交代镜头。
含义：这一帧在全剧人物命运、情绪推进、主题表达或风格迁移中承担什么。
```

四个标签词（情节/构图/导演意图/含义）必须完整出现，标签后建议使用全角冒号"："；QA 同时接受半角冒号，但缺少标签词的行会被判为不合格。不要在理由单元格内使用 `|` 字符，它会破坏表格解析。

Do not use generic reusable reasons such as:

- "作为本幕导演策略的视觉证据。"
- "它能显示空间、道具、表情、色彩或类型转换中的关键一步。"
- "这一帧很重要，体现了导演意图。"

Good example:

```markdown
| 00:12 | ![](../01_逐秒抽帧/frames/frame_000013.jpg) | 情节：男孩站在强烈窗光前，面对家庭命令。构图：逆光把身体切成剪影，桌椅和墙面把他围住。导演意图：不拍大人脸，而拍男孩被光审问，突出弱者被制度照射。含义：他不是在普通争吵里，而是在一个早已判好的家庭法庭里。 |
```

Before delivery, self-check:

- the number of `情节：`, `构图：`, `导演意图：`, and `含义：` entries should match the representative-frame row count.
- exact duplicate reasons should be zero or exceptional.
- repeated template phrases must be zero.

## mandatory analysis angles

Use these angles when relevant. Do not force all of them into every scene, but a complete professional report should cover most across the whole video.

| 角度 | 必须回答的问题 |
|---|---|
| 叙事功能 | 这一幕把故事从什么状态推到什么状态？ |
| 导演意图 | 导演为什么用这种方式呈现，而不是直白说明？ |
| 观众心理 | 观众被怎样引导、欺骗、放松、压迫或震惊？ |
| 镜头策略 | 景别、运镜、角度、焦点怎样改变信息和情绪？ |
| 调度构图 | 人、物、空间、前景、后景怎样构成权力关系？ |
| 光影色彩 | 主色和光源如何编码安全、危险、欲望、神性或荒诞？ |
| 美术道具 | 哪个道具不是装饰，而是在承担叙事/主题？ |
| 表演情绪 | 角色表面情绪和深层情绪是否相反？证据是什么？ |
| 声音剪辑 | 声音、沉默、音乐、声画错位或剪辑节奏怎样制造效果？ |
| 商业节奏 | 是否有钩子、福利、笑点、奇观、反转、悬念保留？ |
| 主题策略 | 主题如何被角色化、道具化、空间化、动作化？ |
| 可迁移方法 | 这个拍法能迁移到什么类型的新剧本？ |

## writing standard

- Write like a senior director/film analyst, not a captioning tool.
- Avoid generic praise such as "很有电影感". Translate praise into craft: scale contrast, foreground invasion, god's-eye view, frame-within-frame, sound-image counterpoint, value reversal, color coding, mixed media, or rhythm trap.
- When a tactic matches a known master signature, consult `references/craft_pattern_library.md` for the correct name, function, and comparison phrasing; do not invent technique names or claim homage without E1 evidence.
- Whenever saying "高级", explain what problem it solves.
- Use evidence labels for uncertain claims:
  - E1: directly visible/audible evidence.
  - E2: reasonable inference from multiple E1 facts.
  - X: unknown or unverifiable.
- Prefer `表层策略 / 深层策略` paragraphs for important scenes.
- If a background action, prop intrusion, or strange visual detail appears, decide whether it is a gag, a pressure valve, a metaphor, a transition device, or a fourth-wall move.
- End the report with a director-facing recap, not only a plot summary.

## final recap

```markdown
## 最终导演复盘

### 1. 全片最核心的导演策略

### 2. 最值得学习的高光段落

### 3. 最有效的观众心理操控

### 4. 最可迁移的视听方法

### 5. 如果迁移到新剧本，应该保留什么、替换什么
```
