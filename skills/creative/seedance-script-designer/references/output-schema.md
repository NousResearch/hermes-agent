# Seedance 2.0 Script Design Output Schema

Use this schema as the response skeleton when converting a script into a production package.

**Do not force all 18 sections for every request.** Read `references/quick-modes.md` to determine which sections are required for the current mode. Only produce a full package when the user explicitly asks for a complete deliverable.

For concrete formatting examples, use `examples/minimal-12col-keyframes.md`, `examples/asset-manifest-example.md`, and `examples/audio-sync-example.md`. Validate finished 12-column tables with `scripts/validate-keyframe-table.mjs` before CSV/JSON conversion or packaging.

## Output Mode Selection

Do not duplicate the mode table here. Use `references/quick-modes.md` as the single source of truth for output mode selection. If the user's request is narrow, output only the relevant sections and say which mode was used. If the user asks for "完整", "全套", "生产包", or a long-script end-to-end deliverable, use the full package.

## 1. 风格确认

- 选择风格:
- 视觉关键词:
- 色彩/光影:
- 画面比例:
- Seedance 模式: 全能参考 / 首尾帧 / 视频延长 / 局部编辑 / 纯文本生成
- 连贯性总原则:

## 2. 剧作结构拆解

- 故事/场景前提:
- 主角:
- 外在欲望:
- 内在需求:
- 对抗力量:
- 利害关系:
- 核心价值变化:
- 激励事件:
- 危机/选择:
- 高潮/结果:
- 可压缩或需要补强的段落:

## 3. 场景节拍表

| 场景 | 节拍 | 角色目标 | 阻力 | 动作/反应 | 转折点 | 价值变化 | 镜头建议 |
|---|---|---|---|---|---|---|---|

## 4. 分段节奏与批次表

Use this before the 12-column keyframe table for long scripts or full storyboard packages.

| 段落 | 时间 | 段落功能 | 叙事任务 | 节奏 | 关键帧策略 | 交付批次 |
|---|---|---|---|---|---|---|

Segment function examples: `建立`, `推进`, `转折`, `高潮`, `余韵`, `过场`, `信息揭示`, `关系变化`. Batch examples: `批次01_验证段`, `批次02_补齐段`, `批次03_修订段`.

## 5. 素材引用规划

| 资产 | 类型 | 用途 | 绑定对象/镜头 | 使用方式 |
|---|---|---|---|---|

## 6. 元素分层资产清单

Use this section before storyboard generation when the script includes recurring characters, locations, props, or emotional motifs.

### 角色身份层

| 角色 | 年龄/阶段 | 五官锚点 | 发型/体态 | 服装/色彩 | 标志道具 | 连续性约束 |
|---|---|---|---|---|---|---|

### 环境层

| 场景 | 氛围基调 | 空间结构 | 光影逻辑 | 色彩语法 | 可重复视觉母题 |
|---|---|---|---|---|---|

### 道具资产层

| 道具 | 剧情功能 | 材质/尺寸 | 使用痕迹 | 特写细节 | 绑定角色/场景 |
|---|---|---|---|---|---|

### 情绪/电影语言层

| 情绪/节拍 | 可见表演细节 | 镜头压力 | 光影变化 | 声音/音乐提示 |
|---|---|---|---|---|

## 7. 角色设计三视图

For each major character:

### 角色名

- 角色定位:
- 核心外观:
- 服装与材质:
- 色彩方案:
- 正面图提示词:
- 侧面图提示词:
- 背面图提示词:
- 统一性约束:

## 8. 角色情绪神态镜头

For each important emotion:

- 角色:
- 情绪:
- 表情细节:
- 肢体语言:
- 镜头:
- Seedance 2.0 提示词:

## 9. 场景图

For each location:

- 场景名:
- 剧情功能:
- 时间/天气:
- 空间布局:
- 光影:
- 色彩:
- Seedance 2.0 场景提示词:
- 可复用连续性提示:

## 10. 道具资产图

For each prop:

- 道具名:
- 剧情作用:
- 形状/材质/尺寸:
- 使用痕迹:
- 资产图提示词:
- 与角色/场景的关系:

## 11. 分镜镜头表

| 镜头 | 时间 | 场景 | 角色 | 剧情节拍 | 戏剧功能 | 画面内容 | 镜头语言 | Seedance 模式 | @参考 | Seedance 2.0 提示词 | 备注 |
|---|---:|---|---|---|---|---|---|---|---|---|---|

## 12. 台词与音乐对应

| 镜头 | 台词/剧情节拍 | 说话人 | 情绪意图 | 口型/音色 | 音乐风格 | 乐器/音色 | 速度 | 音效 | 入点/出点 |
|---|---|---|---|---|---|---|---|---|---|

## 12A. 对白优化与角色声音表

Include this section when the script relies on dialogue, ritual speech, market calls, legal/procedural speech, songs, voice-over, confession, or strong character voice. Use `references/dialogue-voice-method.md`.

### 角色声音圣经

| 说话人 | 表层声音 | 潜台词/语言行动 | 专属词库 | 句子节奏 | 禁止习惯 |
|---|---|---|---|---|---|

### 关键台词升级表

| 段落/镜头 | 原功能 | 当前台词 | 升级台词 | 台词行动 | 优化理由 |
|---|---|---|---|---|---|

### AI 视频对白规则

```text
对白规则：所有对白都必须短、准、有行动。不要解释世界观，不要把人物内心直接讲明。每个说话人要有不同词库和节奏；关键台词前允许停顿、呼吸和反应。画面中不要出现任何字幕、标题、台词文字或气泡；台词只通过口型、声音和表演呈现。
```

## 13. 音频同步表

Use this as an independent handoff table when the output will feed video generation, editing, TTS, voice-over, or sound design.

| 时间 | 类型 | 内容 | 音量 | 备注 |
|---|---|---|---|---|

Types: `音乐`, `音效`, `台词`, `旁白`. Rows may overlap in time. Music may run across a full segment, while dialogue, narration, ambience, and one-shot sound effects occupy shorter overlapping ranges. Keep dialogue/narration clear above music; a useful default is background music around 60-70%, dialogue/narration around 80-90%, and sound effects around 40-60%, adjusted by scene intensity.

## 14. 分镜图关键帧标注表

Use this table whenever the user asks for a detailed storyboard image or annotated storyboard sheet.

Default to the professional 12-column production table:

| 序号 | 关键帧 | 时间 | 镜头 | 运镜 | 转场 | 动作 | 情绪/细节 | 台词 | 旁白 | 状态/音效 | 英文 |
|---|---|---|---|---|---|---|---|---|---|---|---|

Use `关键帧` to mark priority frames with typed markers: `⭐情绪`, `⭐动作`, `⭐台词`, `⭐旁白`, `⭐视觉`, `⭐转折`. Use `台词` only for character speech, `旁白` only for voice-over, and `状态/音效` only for ambience, music, and sound effects. Do not include a `参考资产` column in the keyframe table by default; keep asset references in Section 5 or in video prompts only when the user asks for them. For visible annotations directly on a storyboard image, a compact subset may be used when space is limited.

## 15. 3x3 连续性审查

Use this checklist for each 15-second 3x3 storyboard group:

| 检查项 | 结果 |
|---|---|
| 9格是否按时间推进，不是孤立美图 |  |
| 是否包含广角/中景/特写的视觉变化 |  |
| 角色五官、服装、道具是否连续 |  |
| 场景光影、空间方向、色彩语法是否连续 |  |
| 每格是否只有一个主动作和一个情绪节拍 |  |
| 文字信息是否保留在表格，不默认压到画面里 |  |

## 16. 关键段落优先生成策略

Use this when the user asks for many storyboard images or a full-script storyboard package:

| 批次 | 段落 | 目的 | 用户确认点 |
|---|---|---|---|
| 批次01_验证段 | 开场建立段 | 验证世界观、色彩、角色基调 | 视觉方向是否正确 |
| 批次01_验证段 | 关键转折段 | 验证情绪和戏剧冲突 | 表演和镜头是否有效 |
| 批次01_验证段 | 高潮段 | 验证视觉强度和风格上限 | 视觉高光是否成立 |
| 批次02_补齐段 | 其余段落 | 在已确认风格下批量扩展 | 只做局部修订 |
| 批次03_修订段 | 反馈问题段落 | 局部修正角度、表情、连续性或提示词 | 确认修改是否命中问题 |

## 17. 全局负面提示词

List only if useful:

- 角色不一致
- 面部变形
- 多余手指
- 服装变化
- 文字乱码
- logo/watermark
- camera jitter

## 18. 连贯性提示

- 角色一致性:
- 场景一致性:
- 服装一致性:
- 光影一致性:
- 运动与镜头一致性:
- 音乐/音效一致性:
