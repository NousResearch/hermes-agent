---
name: ai-music-generator
description: Use when generating AI music/BGM for video projects, MV creation, or soundtrack composition. Handles style selection, prompt generation, loudness standards, and quality control for music generation platforms like Suno, Seedance Audio, and Udio.
version: v1.0
created: 2026-04-07
updated: 2026-04-07
source: bitwize-music-studio (CC0 license)
---

# AI Music Generator

Generate original background music and soundtracks for video projects using AI music generation platforms.

## Core Workflow

```
视频情绪分析 → 风格匹配 → AI提示词生成 → 音乐生成 → QC检查 → 交付
```

## Step 1: 视频情绪分析

分析视频项目的情绪基调：

| 情绪类型 | 描述 | 推荐风格 |
|----------|------|----------|
| 史诗/震撼 | 大场面、冲击力、高潮迭起 | cinematic-epic, electronic-hybrid |
| 温暖/治愈 | 美好生活、情感共鸣、正能量 | lo-fi, acoustic, ambient-soft |
| 神秘/悬疑 | 未知、紧张、等待揭晓 | ambient-dark, cinematic-tension |
| 酷炫/潮流 | 时尚、科技、年轻态度 | electronic, synthwave, urban |
| 优雅/高端 | 奢侈品、品质生活、精致 | orchestral, cinematic-chamber |
| 安静/留白 | 极简、禅意、呼吸感 | ambient, neo-classical |
| 复古/怀旧 | 回忆、时光流逝、经典 | lo-fi, city-pop, analog |
| 激烈/动感 | 运动、节奏、能量释放 | electronic-upbeat, hip-hop, edm |

## Step 2: 音乐风格参数库

### 2.1 Cinematic（电影感）

最最适合 MV 制作的风格。

| 子风格 | BPM | 乐器 | 情绪关键词 | Suno 提示词模板 |
|--------|-----|------|-----------|----------------|
| **Epic/Trailer** | 80-120 | 管弦乐+铜管+合唱 | epic, heroic, dramatic, cinematic build | `epic orchestral, dramatic build, cinematic, trailer music, brass fanfares, choir, 90-second arc` |
| **Orchestral** | 60-90 | 弦乐+木管+钢琴 | emotional, intimate, melancholic | `orchestral, string section, piano, film score, emotional, Hans Zimmer style` |
| **Electronic Hybrid** | 90-140 | 电子+管弦+合成器 | modern, tension, pulse | `electronic hybrid, synth orchestration, modern film score, Hans Zimmer, tense atmosphere` |
| **Ambient Score** | 40-80 | 合成器+钢琴+弦乐 | atmospheric, introspective, space | `ambient score, atmospheric, minimal, film texture, subtle tension` |
| **Action** | 120-160 | 打击乐+铜管+电子 | driving, energetic, momentum | `action score, driving percussion, propulsive, energetic, epic drums` |

**Suno 提示词示例**：
```
cinematic, orchestral, epic trailer music, dramatic crescendo,
brass fanfares, string section, choir, 90-second arc build,
triumphant, heroic, Hans Zimmer style
```

### 2.2 Ambient（氛围音乐）

适合品牌广告开场、产品特写的空灵留白。

| 子风格 | BPM | 乐器 | 情绪关键词 | Suno 提示词模板 |
|--------|-----|------|-----------|----------------|
| **Dark Ambient** | 30-70 | 合成器+ drone + 工业 | mysterious, eerie, unsettling | `dark ambient, ominous, unsettling, drone, industrial textures, horror atmosphere` |
| **Space Ambient** | 40-80 | 电子合成器+氛围 | cosmic, expansive, sci-fi | `space ambient, cosmic, ethereal, synth pads, Hans Zimmer, interstellar` |
| **Neo-Classical** | 50-90 | 钢琴+弦乐+合成器 | contemplative, modern, emotional | `neo-classical, piano, strings, Max Richter style, emotional, minimalist` |
| **Nature Ambient** | 0-60 | 自然声+合成器+水声 | calm, natural, meditative | `nature ambient, field recordings, water, wind, meditative, peaceful` |

**Suno 提示词示例**：
```
ambient, atmospheric, textural, meditative, reverb-heavy,
minimalist, ethereal, contemplative, slowly evolving soundscape,
instrumental, 80s ambient style
```

### 2.3 Electronic（电子音乐）

适合潮流/科技/年轻向内容。

| 子风格 | BPM | 乐器 | 情绪关键词 | Suno 提示词模板 |
|--------|-----|------|-----------|----------------|
| **Synthwave** | 80-118 | 模拟合成器+808 | retro, neon, 80s, nostalgic | `synthwave, retrowave, 80s, analog synths, neon atmosphere, Kavinsky, Perturbator style` |
| **Downtempo** | 70-100 | 电子+环境音+打击乐 | relaxed, smooth, sophisticated | `downtempo, chillout, smooth beats, relaxed, Air style, trip-hop` |
| **Lo-Fi** | 70-90 | 采样+钢琴+黑胶噪音 | nostalgic, cozy, study, warm | `lo-fi hip hop, jazz samples, vinyl crackle, Nujabes style, chill beats, study music` |
| **Industrial** | 110-140 | 合成器+机械节奏+ distortion | aggressive, dark, intense | `industrial, EBM, aggressive, mechanical, Nine Inch Nails, dark synth` |

**Suno 提示词示例**：
```
synthwave, retrowave, 80s aesthetic, neon, analog synths,
gated reverb drums, nostalgic, cinematic, Perturbator style
```

### 2.4 Acoustic（原声音乐）

适合人文/生活方式/真诚表达。

| 子风格 | BPM | 乐器 | 情绪关键词 | Suno 提示词模板 |
|--------|-----|------|-----------|----------------|
| **Folk** | 80-120 | 原声吉他+人声+小提琴 | warm, storytelling, organic | `folk, acoustic guitar, singer-songwriter, warm, storytelling, intimate` |
| **Acoustic Pop** | 100-130 | 吉他+钢琴+弦乐 | uplifting, positive, fresh | `acoustic pop, uplifting, fresh, guitar, piano, positive energy` |
| **Jazz** | 60-120 | 钢琴+贝斯+鼓+铜管 | sophisticated, cool, mature | `jazz, piano trio, cool jazz, sophisticated, Blue Note style, Miles Davis` |
| **Classical** | 40-100 | 古典乐器+室内乐 | elegant, refined, timeless | `classical, chamber music, elegant, string quartet, piano` |

**Suno 提示词示例**：
```
lo-fi hip hop, chillhop, jazz hop, vinyl crackle, dusty samples,
Rhodes piano, Nujabes style, boom bap drums, nostalgic, mellow
```

### 2.5 Urban（城市音乐）

适合街头/潮流/年轻文化内容。

| 子风格 | BPM | 乐器 | 情绪关键词 | Suno 提示词模板 |
|--------|-----|------|-----------|----------------|
| **Hip-Hop** | 70-100 | 808+采样+说唱 | street, urban, confident | `hip hop, boom bap, 90s hip hop, sample-based, J Dilla style` |
| **Trap** | 130-170 | 808+合成器+ hi-hats | aggressive, modern, energy | `trap, 808, modern hip hop, heavy bass, 2010s style` |
| **R&B** | 60-100 | 合成器+人声+鼓机 | smooth, sensual, modern | `R&B, smooth, modern production, 808 drums, silky vocals` |

## Step 3: AI 提示词生成

### 3.1 Suno 提示词结构

```
[风格标签], [情绪描述], [乐器配置], [制作特点], [参考艺术家], [时长/结构]
```

**示例 - 品牌广告 BGM**：
```
ambient, atmospheric, corporate, peaceful, piano, strings,
reverb-heavy, minimal, uplifting, hopeful, Max Richter style,
no drums, 60-second loop
```

**示例 - MV 主题曲**：
```
cinematic, epic orchestral, dramatic build, brass fanfares,
string section, Hans Zimmer style, emotional crescendo,
triumphant, 90-second arc, trailer music
```

**示例 - 抖音 BGM**：
```
upbeat electronic, energetic, synth pop, driving beat,
catchy, modern, chart-topping, 120 BPM, drop ready
```

### 3.2 Seedance Audio 提示词

```
风格: [中文风格描述]
节奏: [BPM]
情绪: [情绪关键词]
乐器: [主要乐器]
特殊要求: [渐入/渐出/循环等]
```

## Step 4: 流媒体响度标准

### 4.1 平台标准

| 平台 | 目标 LUFS | True Peak | 备注 |
|------|-----------|-----------|------|
| **抖音** | -14 LUFS | -1.0 dBTP | 平台会自动归一化 |
| **B站** | -14 LUFS | -1.0 dBTP | 与抖音一致 |
| **微博** | -14 LUFS | -1.0 dBTP | 短视频标准 |
| **YouTube** | -13 to -15 LUFS | -1.0 dBTP | 自动归一化到 -14 |
| **Spotify** | -14 LUFS | -1.0 dBTP | 流媒体标准 |

### 4.2 内容类型标准

| 内容类型 | 目标 LUFS | 动态范围 | 说明 |
|----------|-----------|----------|------|
| **品牌广告** | -16 to -18 LUFS | 高 | 需要更大动态，保留情感起伏 |
| **短视频 BGM** | -14 LUFS | 中 | 平台标准，保证响度一致 |
| **MV/电影配乐** | -14 to -16 LUFS | 高 | 保持艺术完整性 |
| **播客/旁白** | -16 LUFS | 低 | 人声为主，音乐为辅 |

### 4.3 Loudness 概念

```
LUFS = Loudness Units Full Scale
- 测量感知响度，不是峰值
- 平台归一化后，-14 LUFS 的内容听感一致
- 过响会被压低，损失动态
- 过轻会被提升，可能引入噪声
```

## Step 5: 质量控制检查清单

### 5.1 生成前检查

- [ ] 确认视频时长（5s/15s/30s/60s/3min）
- [ ] 确认情绪基调（参考 Step 1）
- [ ] 确认目标平台（抖音/B站/微博）
- [ ] 确认是否需要循环（广告通常需要循环）

### 5.2 生成后检查

- [ ] **时长检查**：音乐时长是否匹配视频？（允许 ±2s）
- [ ] **响度检查**：使用 LUFS meter 测量是否为 -14 LUFS ± 0.5
- [ ] **峰值检查**：True Peak 是否 < -1.0 dBTP？（防止削波）
- [ ] **循环点检查**：如果需要循环，起始和结束是否衔接自然？
- [ ] **淡入淡出**：开头和结尾是否有适当淡入淡出？（0.5-1s）
- [ ] **人声污染检查**：BGM 是否有人声？是否干扰旁白？

### 5.3 问题诊断

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 听起来"闷" | 低频过多 | 减少 Sub Bass (20-60 Hz) |
| 听起来"薄" | 缺少中低频 | 增加 Piano/Guitar 中频 |
| 人声被掩盖 | BGM 太响 | 降低 BGM 响度至 -18 LUFS |
| 循环不自然 | 起始/结束不匹配 | 重新生成或手动编辑 |
| 动态被压扁 | 过度压缩 | 重新母带处理 |

## Step 6: 交付格式

### 6.1 标准交付

```
文件名: [项目名]_[风格]_[日期].wav
格式: WAV (24-bit, 48kHz)
响度: -14 LUFS
峰值: < -1.0 dBTP
时长: [X]s
```

### 6.2 分轨交付（如需要）

```
STEMS/
├── drums.wav
├── bass.wav
├── piano.wav
├── strings.wav
└── master.wav (完整混音)
```

## 白梦客美学系统适配

### 高定线（高端品牌广告）

适合风格：
- cinematic-orchestral
- cinematic-ambient
- neo-classical

提示词关键词：
```
minimalist, elegant, restrained, natural light audio,
long takes silence, understated, premium,
black and white aesthetic sound, Chanel audio
```

### 通用线（故事/情绪短片）

适合风格：
- cinematic-epic
- ambient-dark
- lo-fi
- synthwave

提示词关键词：
```
emotional, narrative, Wang Kar-wai style atmosphere,
Miyazaki emotional resonance, contemplative,
earth tones audio, neon noir
```

## 参考来源

本技能整合自 [bitwize-music-studio](https://github.com/bitwize-music-studio/claude-ai-music-skills)（CC0 Public Domain License）。

---

## 版本历史

- v1.0 (2026-04-07): 初始创建，整合 bitwize-music-studio 音乐风格库
