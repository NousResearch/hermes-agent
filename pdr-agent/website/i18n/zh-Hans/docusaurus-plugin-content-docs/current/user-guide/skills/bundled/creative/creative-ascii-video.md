---
title: "Ascii Video — ASCII 视频：将视频/音频转换为彩色 ASCII MP4/GIF"
sidebar_label: "Ascii Video"
description: "ASCII 视频：将视频/音频转换为彩色 ASCII MP4/GIF"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Ascii Video

ASCII 视频：将视频/音频转换为彩色 ASCII MP4/GIF。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/ascii-video` |
| 平台 | linux, macos, windows |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# ASCII 视频生产流水线

## 使用时机

当用户请求以下内容时使用：ASCII 视频、文字艺术视频、终端风格视频、字符艺术动画、复古文字可视化、ASCII 音频可视化器、将视频转换为 ASCII 艺术、矩阵风格特效，或任何动态 ASCII 输出。

## 内容概述

用于 ASCII 艺术视频的生产流水线——支持任意格式。将视频/音频/图像/生成式输入转换为彩色 ASCII 字符视频输出（MP4、GIF、图像序列）。涵盖：视频转 ASCII、音频响应式音乐可视化器、生成式 ASCII 艺术动画、视频+音频混合响应、文字/歌词叠加、实时终端渲染。

## 创作标准

这是视觉艺术。ASCII 字符是媒介；电影是标准。

**在写下任何一行代码之前**，先阐明创作概念。氛围是什么？这讲述了怎样的视觉故事？是什么让这个项目与其他所有 ASCII 视频不同？用户的 prompt（提示词）只是起点——以创作野心去诠释它，而非字面转录。

**首次渲染即达到卓越水准，不可妥协。** 输出必须在无需修改的情况下具有视觉冲击力。如果看起来平庸、单调，或像"AI 生成的 ASCII 艺术"，那就是错的——在交付前重新思考创作概念。

**超越参考词汇表。** 参考资料中的特效目录、shader（着色器）预设和调色板库只是起点词汇。每个项目都应组合、修改并发明新的模式。目录是颜料——你来作画。

**主动发挥创造力。** 当项目需要时，扩展 skill 的词汇表。如果参考资料无法满足创作愿景，就自己构建。至少加入一个用户没有要求但会欣赏的视觉时刻——一个过渡、一个特效、一个提升整体作品的色彩选择。

**整体美学优先于技术正确性。** 视频中的所有场景必须通过统一的视觉语言相互关联——共同的色温、相关的字符调色板、一致的运动词汇。一个技术上正确但每个场景随机使用不同特效的视频，在美学上是失败的。

**密集、分层、深思熟虑。** 每一帧都应值得细看。绝不使用纯黑背景。始终使用多网格构图。始终保持逐场景变化。始终使用有意为之的色彩。

## 模式

| 模式 | 输入 | 输出 | 参考 |
|------|-------|--------|-----------|
| **视频转 ASCII** | 视频文件 | 源素材的 ASCII 重现 | `references/inputs.md` § Video Sampling |
| **音频响应式** | 音频文件 | 由音频特征驱动的生成式视觉效果 | `references/inputs.md` § Audio Analysis |
| **生成式** | 无（或种子参数） | 程序化 ASCII 动画 | `references/effects.md` |
| **混合式** | 视频 + 音频 | 带音频响应叠加层的 ASCII 视频 | 两个输入参考 |
| **歌词/文字** | 音频 + 文字/SRT | 带视觉特效的定时文字 | `references/inputs.md` § Text/Lyrics |
| **TTS 旁白** | 文字引用 + TTS API | 带打字文字效果的旁白证言/引用视频 | `references/inputs.md` § TTS Integration |

## 技术栈

每个项目使用单一自包含 Python 脚本。无需 GPU。

| 层级 | 工具 | 用途 |
|-------|------|---------|
| 核心 | Python 3.10+, NumPy | 数学运算、数组操作、向量化特效 |
| 信号 | SciPy | FFT、峰值检测（音频模式） |
| 图像 | Pillow (PIL) | 字体光栅化、帧解码、图像 I/O |
| 视频 I/O | ffmpeg (CLI) | 解码输入、编码输出、混合音频 |
| 并行 | concurrent.futures | N 个 worker 用于批量/片段渲染 |
| TTS | ElevenLabs API（可选） | 生成旁白片段 |
| 可选 | OpenCV | 视频帧采样、边缘检测 |

## 流水线架构

每种模式遵循相同的 6 阶段流水线：

```
INPUT → ANALYZE → SCENE_FN → TONEMAP → SHADE → ENCODE
```

1. **INPUT** — 加载/解码源素材（视频帧、音频采样、图像，或无输入）
2. **ANALYZE** — 提取逐帧特征（音频频段、视频亮度/边缘、运动向量）
3. **SCENE_FN** — 场景函数渲染到像素画布（`uint8 H,W,3`）。通过 `_render_vf()` + 像素混合模式组合多个字符网格。参见 `references/composition.md`
4. **TONEMAP** — 基于百分位数的自适应亮度归一化。参见 `references/composition.md` § Adaptive Tonemap
5. **SHADE** — 通过 `ShaderChain` + `FeedbackBuffer` 进行后处理。参见 `references/shaders.md`
6. **ENCODE** — 将原始 RGB 帧通过管道传输至 ffmpeg 进行 H.264/GIF 编码

## 创作方向

### 美学维度

| 维度 | 选项 | 参考 |
|-----------|---------|-----------|
| **字符调色板** | 密度渐变、块状元素、符号、文字（片假名、希腊字母、符文、盲文）、项目专属 | `architecture.md` § Palettes |
| **色彩策略** | HSV、OKLAB/OKLCH、离散 RGB 调色板、自动生成和声、单色、色温 | `architecture.md` § Color System |
| **背景纹理** | 正弦场、fBM 噪声、域扭曲、voronoi、反应扩散、元胞自动机、视频 | `effects.md` |
| **主要特效** | 环形、螺旋、隧道、漩涡、波浪、干涉、极光、火焰、SDF、奇异吸引子 | `effects.md` |
| **粒子** | 火花、雪花、雨滴、气泡、符文、轨道、群集 boid、流场跟随者、轨迹 | `effects.md` § Particles |
| **Shader 风格** | 复古 CRT、简洁现代、故障艺术、电影感、梦幻、工业、迷幻 | `shaders.md` |
| **网格密度** | xs(8px) 到 xxl(40px)，每层可混合使用 | `architecture.md` § Grid System |
| **坐标空间** | 笛卡尔、极坐标、平铺、旋转、鱼眼、Möbius、域扭曲 | `effects.md` § Transforms |
| **Feedback** | 缩放隧道、彩虹轨迹、幽灵回声、旋转曼陀罗、色彩演化 | `composition.md` § Feedback |
| **遮罩** | 圆形、环形、渐变、文字模板、动态虹膜/擦除/溶解 | `composition.md` § Masking |
| **过渡** | 交叉淡化、擦除、溶解、故障切换、虹膜、基于遮罩的揭示 | `shaders.md` § Transitions |

### 逐段变化

绝不对整个视频使用相同配置。对每个段落/场景：
- **不同的背景特效**（或组合 2-3 种）
- **不同的字符调色板**（匹配氛围）
- **不同的色彩策略**（或至少使用不同色调）
- **变化 shader 强度**（高潮时更多泛光，安静时更多颗粒感）
- **不同的粒子类型**（如果粒子处于激活状态）

### 项目专属创新

每个项目至少发明以下之一：
- 匹配主题的自定义字符调色板
- 自定义背景特效（组合/修改现有构建块）
- 自定义色彩调色板（匹配品牌/氛围的离散 RGB 集合）
- 自定义粒子字符集
- 新颖的场景过渡或视觉时刻

不要只从目录中挑选。目录是词汇——你来写诗。

## 工作流程

### 第一步：创作愿景

在任何代码之前，阐明创作概念：

- **氛围/气氛**：观众应该感受到什么？充满活力、冥想感、混沌、优雅、不祥？
- **视觉故事**：在整个时长内发生了什么？积累张力？转变？消解？
- **色彩世界**：暖色/冷色？单色？霓虹？大地色调？主色调是什么？
- **字符质感**：密集数据？稀疏星点？有机点阵？几何块状？
- **与众不同之处**：是什么让这个项目独一无二？
- **情感弧线**：场景如何推进？以能量开场，积累至高潮，然后解决？

将用户的 prompt 映射到美学选择。"轻松 lo-fi 可视化器"与"故障赛博朋克数据流"在各方面都要求截然不同的处理。

### 第二步：技术设计

- **模式** — 上述 6 种模式中的哪一种
- **分辨率** — 横屏 1920x1080（默认）、竖屏 1080x1920、方形 1080x1080 @ 24fps
- **硬件检测** — 自动检测核心数/内存，设置质量配置文件。参见 `references/optimization.md`
- **段落** — 将时间戳映射到场景函数，每个场景有其自己的特效/调色板/色彩/shader 配置
- **输出格式** — MP4（默认）、GIF（640x360 @ 15fps）、PNG 序列

### 第三步：构建脚本

单一 Python 文件。组件（含参考）：

1. **硬件检测 + 质量配置文件** — `references/optimization.md`
2. **输入加载器** — 依模式而定；`references/inputs.md`
3. **特征分析器** — 音频 FFT、视频亮度，或合成
4. **网格 + 渲染器** — 带位图缓存的多密度网格；`references/architecture.md`
5. **字符调色板** — 每个项目多个；`references/architecture.md` § Palettes
6. **色彩系统** — HSV + 离散 RGB + 和声生成；`references/architecture.md` § Color
7. **场景函数** — 每个返回 `canvas (uint8 H,W,3)`；`references/scenes.md`
8. **Tonemap** — 自适应亮度归一化；`references/composition.md`
9. **Shader 流水线** — `ShaderChain` + `FeedbackBuffer`；`references/shaders.md`
10. **场景表 + 调度器** — 时间 → 场景函数 + 配置；`references/scenes.md`
11. **并行编码器** — N worker 片段渲染，使用 ffmpeg 管道
12. **Main** — 编排完整流水线

### 第四步：质量验证

- **先测试帧**：在完整渲染前，在关键时间戳渲染单帧
- **亮度检查**：所有 ASCII 内容的 `canvas.mean() > 8`。如果偏暗，降低 gamma
- **视觉连贯性**：所有场景是否感觉属于同一个视频？
- **创作愿景检查**：输出是否与第一步的概念相符？如果看起来平庸，请返回重做

## 关键实现注意事项

### 亮度——使用 `tonemap()`，而非线性乘数

这是第一大视觉问题。黑色背景上的 ASCII 本质上偏暗。**绝不使用 `canvas * N` 乘数**——它们会截断高光。使用自适应 tonemap：

```python
def tonemap(canvas, gamma=0.75):
    f = canvas.astype(np.float32)
    lo, hi = np.percentile(f[::4, ::4], [1, 99.5])
    if hi - lo < 10: hi = lo + 10
    f = np.clip((f - lo) / (hi - lo), 0, 1) ** gamma
    return (f * 255).astype(np.uint8)
```

流水线：`scene_fn() → tonemap() → FeedbackBuffer → ShaderChain → ffmpeg`

逐场景 gamma：默认 0.75，日晒效果 0.55，色调分离 0.50，明亮场景 0.85。暗层使用 `screen` 混合（而非 `overlay`）。

### 字体单元高度

macOS Pillow：`textbbox()` 返回错误高度。使用 `font.getmetrics()`：`cell_height = ascent + descent`。参见 `references/troubleshooting.md`。

### ffmpeg 管道死锁

长时间运行的 ffmpeg 绝不使用 `stderr=subprocess.PIPE`——缓冲区在 64KB 时填满并死锁。重定向到文件。参见 `references/troubleshooting.md`。

### 字体兼容性

并非所有 Unicode 字符都能在所有字体中渲染。在初始化时验证调色板——渲染每个字符，检查是否有空白输出。参见 `references/troubleshooting.md`。

### 逐片段架构

对于分段视频（引用、场景、章节），将每段渲染为独立的片段文件，以支持并行渲染和选择性重渲染。参见 `references/scenes.md`。

## 性能目标

| 组件 | 预算 |
|-----------|--------|
| 特征提取 | 1-5ms |
| 特效函数 | 2-15ms |
| 字符渲染 | 80-150ms（瓶颈） |
| Shader 流水线 | 5-25ms |
| **总计** | ~100-200ms/帧 |

## 参考资料

| 文件 | 内容 |
|------|----------|
| `references/architecture.md` | 网格系统、分辨率预设、字体选择、字符调色板（20+）、色彩系统（HSV + OKLAB + 离散 RGB + 和声生成）、`_render_vf()` 辅助函数、GridLayer 类 |
| `references/composition.md` | 像素混合模式（20 种）、`blend_canvas()`、多网格构图、自适应 `tonemap()`、`FeedbackBuffer`、`PixelBlendStack`、遮罩/模板系统 |
| `references/effects.md` | 特效构建块：值场生成器、色调场、噪声/fBM/域扭曲、voronoi、反应扩散、元胞自动机、SDF、奇异吸引子、粒子系统、坐标变换、时间连贯性 |
| `references/shaders.md` | `ShaderChain`、`_apply_shader_step()` 调度、38 种 shader 目录、音频响应式缩放、过渡、色调预设、输出格式编码、终端渲染 |
| `references/scenes.md` | 场景协议、`Renderer` 类、`SCENES` 表、`render_clip()`、节拍同步剪切、并行渲染、设计模式（层级结构、方向弧线、视觉隐喻、构图技法）、各复杂度级别的完整场景示例、场景设计检查清单 |
| `references/inputs.md` | 音频分析（FFT、频段、节拍）、视频采样、图像转换、文字/歌词、TTS 集成（ElevenLabs、声音分配、音频混合） |
| `references/optimization.md` | 硬件检测、质量配置文件、向量化模式、并行渲染、内存管理、性能预算 |
| `references/troubleshooting.md` | NumPy 广播陷阱、混合模式陷阱、多进程/pickling、亮度诊断、ffmpeg 问题、字体问题、常见错误 |

---

## 创意发散（仅在用户请求实验性/创意性/独特输出时使用）

如果用户要求创意性、实验性、令人惊喜或非常规的输出，选择最适合的策略，并在生成代码**之前**推理其步骤。

- **强制关联** — 当用户想要跨领域灵感时（"让它看起来有机感"、"工业美学"）
- **概念融合** — 当用户命名两个要组合的事物时（"海洋遇见音乐"、"太空 + 书法"）
- **斜向策略** — 当用户完全开放时（"给我惊喜"、"我从未见过的东西"）

### 强制关联
1. 选择一个与视觉目标无关的领域（天气系统、微生物学、建筑、流体动力学、纺织编织）
2. 列出其核心视觉/结构元素（侵蚀 → 逐渐揭示；有丝分裂 → 分裂复制；编织 → 交错图案）
3. 将这些元素映射到 ASCII 字符和动画模式
4. 综合——"侵蚀"或"结晶"在字符网格中看起来是什么样的？

### 概念融合
1. 命名两个不同的视觉/概念空间（例如，海浪 + 乐谱）
2. 映射对应关系（波峰 = 高音，波谷 = 休止，浪花 = 断奏）
3. 选择性融合——保留最有趣的映射，舍弃牵强的
4. 发展只存在于融合中的涌现属性

### 斜向策略
1. 抽取一张："将错误视为隐藏的意图" / "使用一个旧想法" / "你最亲密的朋友会怎么做？" / "强调缺陷" / "颠倒过来" / "只取一部分，而非全部" / "反转"
2. 将该指令对照当前 ASCII 动画挑战进行诠释
3. 在编写代码之前，将这一横向洞见应用于视觉设计