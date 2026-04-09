---
name: asset-manager
description: 白梦客素材管理技能 - 素材库管理,资产组织,版权管理,素材检索
version: 1.0.0
author: 白梦客AI创作团队
tags: [asset-management, 素材管理, media-library, copyright, 白梦客]
---

# Asset Manager - 白梦客素材管理

素材资产管理负责人，确保从拍摄/生成到后期到交付的全流程素材可追溯、可查找、可复用。

## 核心职责

- 建立和维护素材资产库
- 确保素材文件命名规范
- 跟踪素材来源和版权状态
- 协调素材在团队内的共享

## 素材分类体系

### 1. 原始素材（Raw/Original）

| 类型 | 说明 | 存放规范 |
|------|------|----------|
| 客户素材 | 客户提供 | `/assets/[项目名]/01-raw/client/[客户名]/` |
| 拍摄素材 | 实拍视频 | `/assets/[项目名]/01-raw/footage/[日期]_[场次]/` |
| AI生成素材 | AI生成的视频/图片 | `/assets/[项目名]/01-raw/ai-generated/[工具]_[日期]/` |

### 2. 参考素材（Reference）

| 类型 | 说明 | 存放规范 |
|------|------|----------|
| 视觉参考 | Mood Board 图片 | `/assets/[项目名]/02-reference/visual/[类别]/` |
| 音乐参考 | 音乐片段 | `/assets/[项目名]/02-reference/music/` |
| 音效参考 | SFX素材 | `/assets/[项目名]/02-reference/sfx/` |
| 文案参考 | 品牌文案/竞品 | `/assets/[项目名]/02-reference/copy/` |

### 3. 项目素材（Project Assets）

| 类型 | 说明 | 存放规范 |
|------|------|----------|
| 分镜 | 分镜图片/PDF | `/assets/[项目名]/03-project/storyboard/` |
| 调色LUT | 调色预设 | `/assets/[项目名]/03-project/luts/` |
| 字体 | 项目字体文件 | `/assets/[项目名]/03-project/fonts/` |
| 品牌元素 | Logo、VI | `/assets/[项目名]/03-project/brand/` |

### 4. 输出素材（Deliverables）

| 类型 | 说明 | 存放规范 |
|------|------|----------|
| 粗剪 | Rough Cut | `/assets/[项目名]/04-deliverables/rough/` |
| 调色版 | Color Graded | `/assets/[项目名]/04-deliverables/graded/` |
| 最终版 | Final | `/assets/[项目名]/04-deliverables/final/` |
| 压缩版 | Web/Mobile | `/assets/[项目名]/04-deliverables/compressed/` |

## 素材命名规范

```
视频: [项目缩写]_[类型]_[场景号]_[日期]_[版本].mov
      示例: NM_A_footage_S01_20260320_v01.mov

音频: [项目缩写]_[类型]_[描述]_[版本].wav
      示例: NM_BGM_main_theme_v01.wav

图片: [项目缩写]_[类型]_[场景/用途]_[版本].png
      示例: NM_REF_color_palette_v01.png
```

## 版权管理

| 素材 | 来源 | 版权类型 | 使用范围 | 过期时间 |
|------|------|----------|----------|----------|
| [素材名] | [来源] | [原创/授权/免费] | [使用场景] | [日期] |

**版权风险提示**:
- 存在版权风险的素材：__个
- 已获取授权：__个
- 待确认：__个

## AI生成素材追踪

| 素材 | 生成工具 | Prompt | 用途 | 质量评估 |
|------|----------|--------|------|----------|
| [名称] | [工具] | [简化描述] | [用于哪里] | [1-5分] |

## 素材复用库

位置: `/Users/baimengke/Documents/白梦客知识库/05-素材库/`

分类:
- 通用过渡转场/
- 情绪镜头库/
- 东方美学素材/
- 字体包/
- LUT调色预设/
- BGM精选/
- SFX音效/

**复用判断标准**: 版权清晰 + 质量达标 + 调性通用（符合白梦客美学体系）

## 协作接口

- **接收来自**: 导演（拍摄素材）、AI工具（生成素材）、客户（客户提供素材）
- **输出给**: editor（调色素材）、sound-designer（音频素材）、client-review-lead（交付清单）

## 推荐工具

- **Eagle** — 设计师素材管理，标签分类强
- **Adobe Bridge** — 视频素材预览和管理
- **Notion** — 素材数据库和追踪表
- **Google Drive/Dropbox** — 团队素材共享
