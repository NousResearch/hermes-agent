---
name: competitive-research
description: "Use when researching competitors, analyzing 小红书/Xiaohongshu content, comparing products, or producing structured market/social-media reports in Chinese or English."
version: 1.1.0
author: yimgao
license: MIT
metadata:
  hermes:
    tags: [research, competitive-analysis, market-intelligence, xiaohongshu, social-media, chinese, bilingual, content-analysis, web-research]
    related_skills: [writing-plans, spike]
---

# Competitive Research Assistant / 竞品调研助手

> **Research competitors, analyze 小红书 trends, compare products — in Chinese or English.**
> **竞品调研、小红书趋势分析、产品对比 — 支持中英文。**

---

## Overview / 概述

This skill helps you research competitors, analyze 小红书 (Xiaohongshu / RED) content, and produce structured reports — all within Hermes Agent. Whether you need a Western market deep-dive or Chinese social-media trend analysis, the research methodology is the same: search, browse, extract, compile.

本技能帮助你做竞品调研、小红书内容分析、产品对比，输出结构化报告。无论是西方市场调研还是中文社交媒体趋势分析，方法论是一致的：搜索 → 浏览 → 提取 → 编译。

**What it can do / 它能做什么：**

| English | 中文 |
|---------|------|
| Research a single competitor in depth | 深度调研单个竞品 |
| Compare 2-5 competitors side-by-side | 对比 2-5 个竞品 |
| Analyze market positioning & SWOT | 市场定位 + SWOT 分析 |
| **Research 小红书 trends & content** | **调研小红书趋势与内容** |
| Analyze competitors' 小红书 strategies | 分析竞品的小红书策略 |
| Find popular 小红书 topics & formats | 发现小红书热门话题 & 笔记形式 |
| Monitor competitors / trends via cron | 通过 cron 定时监控 |
| Track pricing changes over time | 追踪价格变化 |

---

## When to Use / 什么时候使用

| English | 中文 |
|---------|------|
| Before entering a new market | 进入新市场前 |
| Before building a new feature | 开发新功能前 |
| When researching 小红书 content strategy | 调研小红书内容策略时 |
| For weekly competitor monitoring | 每周竞品监控 |
| For pitch decks / investor materials | 做路演/投资材料时 |
| When you need bilingual analysis | 需要双语分析时 |

---

## 核心工作流 / Core Workflow

### 🇨🇳 小红书内容研究 (Chinese Social Media Research)

```
用户：帮我分析小红书上『AI 笔记工具』这个赛道的热门内容
Hermes 应该：
  1. 搜索：小红书 AI笔记工具 热门笔记 2026
           小红书 AI笔记 app 推荐
           小红书 笔记工具 评测 对比
  2. 搜索：知乎 AI笔记工具 评测
           百度 AI笔记工具 哪个好用
  3. 分析热门笔记的特点：
     - 标题风格（emoji使用、关键词）
     - 封面图风格
     - 点赞/收藏/评论量
     - 笔记结构（图文、视频、合集）
  4. 识别关键竞争者在小红书的布局
  5. 输出中文报告
```

```
用户：Research popular 小红书 content formats for skincare products
Hermes should:
  1. Search: 小红书 护肤 爆款笔记 格式
             小红书 美妆 热门内容 分析
  2. Search: 小红书 护肤品类 种草笔记
  3. Analyze content patterns: titles, covers, hashtags, formats
  4. Identify top KOL/KOC accounts in skincare
  5. Output bilingual or English report
```

### 🇨🇳 竞品小红书策略分析

```
用户：分析 Notion 和 FlowUs 在小红书上的内容策略差异
Hermes 应该：
  1. 搜索：小红书 Notion 笔记 教程
           小红书 FlowUs 笔记 使用
  2. 分析每个品牌的内容：
     - 发布的笔记数量和频率
     - 内容类型（教程、评测、对比）
     - 互动情况
     - 受众反馈
  3. 对比总结两个品牌的小红书策略差异
  4. 输出中文报告
```

### 🇺🇸 🇬🇧 Single Competitor Deep Dive (English)

```
User: Research Confluence as a documentation tool
Hermes should:
  1. Web search: "Confluence documentation features pricing 2026"
  2. Web search: "Confluence reviews pros cons"
  3. Browse the Confluence pricing page
  4. Browse 1-2 review sites (G2, Capterra)
  5. Compile report covering: Features, Pricing, Target Users, Pros/Cons, Alternatives
```

### 🌐 Competitive Comparison

```
User: Compare Notion, Obsidian, and Craft for personal knowledge management
Hermes should:
  1. Research each product individually (features, pricing, reviews)
  2. Create a comparison table
  3. Add pros/cons for each
  4. Summarize best fits
  5. Save report
```

### 📊 Market Landscape + SWOT

```
User: Analyze the AI coding assistant market — GitHub Copilot, Cursor, Codex
Hermes should:
  1. Research each competitor
  2. Market positioning analysis
  3. SWOT for each
  4. Overall market trends summary
  5. Save report
```

---

## Research Methodology / 研究方法

### Step 1: Define Scope / 明确范围

Before searching, clarify / 搜索前先明确：

- **What / 调研什么？** (product, company, category, 小红书话题)
- **Dimensions / 分析维度？** (features, pricing, reviews, 热门内容, 互动数据)
- **Language / 语言？** (English, 中文, or bilingual / 双语)
- **Scope / 范围？** (1 deep dive, 3-5 comparison, landscape)
- **Output / 输出格式？** (brief summary, detailed report, comparison table)

### Step 2: Multi-Source Strategy / 多源策略

**For English/Global research:**

| Source Type | Examples | What to extract |
|-------------|----------|-----------------|
| Official website | product.site, docs.product.site | Features, pricing, use cases |
| Review platforms | G2, Capterra, ProductHunt | Ratings, pros/cons, reviews |
| Tech media | TechCrunch, The Verge, blogs | News, funding, market position |
| Community | Reddit, Hacker News, Discord | Real user sentiment |
| Comparisons | "X vs Y" blog posts | Side-by-side feature lists |

**For 中文/中国 research:**

| 来源类型 | 示例 | 提取内容 |
|---------|------|---------|
| 小红书 | 小红书搜索 | 热门笔记、话题趋势、用户互动 |
| 知乎 | zhihu.com 搜索 | 深度评测、用户讨论 |
| 百度 | 百度搜索 | 产品信息、用户评价 |
| 抖音/微信 | 相关内容 | 短视频内容趋势 |
| 科技媒体 | 36氪、虎嗅、爱范儿 | 行业新闻、融资信息 |
| 电商平台 | 淘宝、京东 | 产品销量、用户评价 |

### 🌟 小红书特有分析方法

#### 内容维度分析

```
分析一篇小红书笔记时，提取以下信息：

📌 基础信息
- 标题：关键词、emoji使用、句式
- 封面：风格（纯文字/产品图/对比图）
- 标签：#话题标签
- 形式：图文 / 视频 / 合集

📊 互动数据
- 点赞数
- 收藏数
- 评论数
- 分享数
- 评论内容（用户关注点、常见问题）

🎯 内容策略分析
- 笔记类型：种草 / 教程 / 对比 / 合集 / vlog
- 目标受众判断
- 卖点提取
- 转化路径（评论区引导、主页链接等）
```

#### 竞品在小红书的布局分析

```
分析一个品牌在小红书的存在：

1. 官方账号运营：
   - 粉丝数、笔记数
   - 内容风格和频率
   - 互动率

2. KOL/KOC 合作：
   - 合作的博主类型和量级
   - 合作内容形式
   - 用户反馈

3. 话题和标签策略：
   - 品牌相关话题热度
   - 使用的话题标签
   - 内容分发策略
```

### Step 3: Data Extraction / 数据提取

For each source, extract / 对每个来源，提取：

| English | 中文 |
|---------|------|
| Features / 产品功能 | 功能列表、独特卖点 |
| Pricing / 价格 | 免费版、付费版、企业版 |
| Target audience / 目标用户 | 个人 / 团队 / 企业 |
| Strengths / 优势 | 用户最称赞的 |
| Weaknesses / 不足 | 常见投诉 |
| Market position / 市场定位 | 领导者 / 挑战者 / 细分 |
| **小红书热度 / 小红书 popularity** | 笔记量、互动量、话题热度 |

### Step 4: Report Compilation / 报告编译

Structure the output following the language of the user's query / 根据用户查询语言选择报告语言：

**English report template:**

```markdown
# Competitive Research Report: [Category]

**Date:** YYYY-MM-DD
**Scope:** [description]

---

## Executive Summary
[2-3 paragraph overview]

## Competitor Overview

| Competitor | Key Differentiator | Price | Target |
|------------|--------------------|-------|--------|
| Product A | ... | $X/mo | ... |

## Detailed Analysis

### Product A
**Features:** ...
**Pricing:** ...
**Strengths:** ...
**Weaknesses:** ...

## Comparison Table

...

## Recommendations

...

## Sources
...
```

**中文报告模板：**

```markdown
# 竞品调研报告：[类别]

**日期：** YYYY-MM-DD
**范围：** [描述]

---

## 执行摘要
[2-3 段概述]

## 竞品概览

| 竞品 | 核心差异化 | 价格 | 目标用户 |
|------|-----------|------|---------|
| 产品A | ... | ... | ... |

## 深度分析

### 产品A
**功能：** ...
**价格：** ...
**优势：** ...
**不足：** ...

## 对比表格

...

## 建议方案

...

## 来源
...
```

**小红书内容分析报告模板：**

```markdown
# 小红书内容分析报告：[话题/品类]

**日期：** YYYY-MM-DD

---

## 热门内容概览

| 笔记标题 | 类型 | 点赞 | 收藏 | 评论 | 博主类型 |
|---------|------|------|------|------|---------|
| ... | 图文 | 2.3w | 1.1w | 456 | KOL |

## 内容趋势分析

### 热门标题模式
- 模式1：...
- 模式2：...

### 常见内容形式
- 图文教程：...
- 对比评测：...
- 合集推荐：...

### 热门话题标签
- #标签1（xxx篇笔记）
- #标签2（xxx篇笔记）

## 竞品在小红书的布局

### 品牌A
- 官方账号：xx粉丝，xx笔记
- 内容风格：...
- 合作KOL：...

### 品牌B
- ...

## 关键发现与建议

1. ...
2. ...

## 来源
- [小红书链接](url)
- [知乎链接](url)
```

### Step 5: Save / 保存

Save reports to / 保存到：

```
# English reports
~/research/competitive/<category>/YYYY-MM-DD-<topic>.md

# 中文报告
~/research/红书研究/<品类>/YYYY-MM-DD-<话题>.md
~/research/竞品调研/<品类>/YYYY-MM-DD-<主题>.md
```

---

## 示例 Prompt / Example Prompts

### 🇨🇳 小红书内容研究
```
帮我分析小红书上『AI写作工具』的热门内容趋势。我需要知道：最火的笔记标题是什么、什么类型的笔记互动最高、大家都在讨论哪些功能点、有哪些博主在这个领域有影响力。用中文输出报告。
```

### 🇨🇳 小红书竞品分析
```
分析 Notion 和 我来(wolai) 在小红书上的内容策略差异。对比：笔记数量和频率、内容类型、互动情况、用户反馈。输出中文报告。
```

### 🇨🇳 小红书话题调研
```
研究小红书『效率工具』话题下，2026年最新的热门笔记趋势。提取前10篇高赞笔记的标题、封面风格、内容结构，总结爆款公式。
```

### 🌐 English quick comparison
```
Compare Linear and Jira for project management. Features table, pricing, and which is better for a 10-person startup.
```

### 🌐 English deep dive
```
Deep dive on Mintlify as a documentation platform. Features, pricing tiers, reviews, user complaints, main alternatives.
```

### 🌐 Bilingual report
```
Research MCP (Model Context Protocol) server ecosystem. What MCP servers exist? Compare 5 popular ones. Output the report in both English and Chinese (side by side or alternating sections).
```

### 🔄 Cron monitoring
```
Set up a weekly cron job to monitor new competitor news about Windsurf and Cursor IDE. Also check 小红书 for any trending posts about them.
```

---

## Common Pitfalls / 常见问题

| # | English | 中文 |
|---|---------|------|
| 1 | **Relying on one source.** Always cross-reference. | **只依赖单一来源。** 一定要交叉验证。 |
| 2 | **Outdated pricing info.** Visit official pricing page. | **价格信息过时。** 以官网定价页为准。 |
| 3 | **Confusing claims with reality.** Check reviews and screenshots. | **把宣传当事实。** 看用户真实评价。 |
| 4 | **Ignoring context.** Always note team size, use case, budget. | **忽略上下文。** 注明团队规模、使用场景。 |
| 5 | **Features only.** Pricing, support, ecosystem matter too. | **只看功能。** 价格、服务、生态同样重要。 |
| 6 | **Forgetting to save.** Write the report to a file. | **忘记保存。** 一定要把报告存到文件里。 |
| 7 | **小红书搜索局限。** 搜索内容有限，需要结合多个关键词反复搜索。 | **小红书 search limitations.** Multiple keyword searches needed. |
| 8 | **忽略语言适配。** 英文报告不要生硬翻译成中文，反之亦然。 | **Language mismatch.** Adapt tone per language, don't machine-translate. |

---

## Verification Checklist / 验证清单

- [ ] Gathered data from 3+ sources per competitor / 每个竞品至少 3 个信息源
- [ ] For 小红书 research: searched multiple keywords / 小红书研究：多个关键词搜索
- [ ] Pricing verified on official page / 价格已在官网核实
- [ ] At least one user review source / 至少一个用户评价来源
- [ ] Report language matches user's query / 报告语言与用户查询一致
- [ ] Report saved to file / 报告已保存到文件
- [ ] Sources listed / 来源已列出
