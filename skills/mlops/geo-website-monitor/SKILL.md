---
name: geo-website-monitor
title: GEO官网监测工具
description: 监测客户官网在主流大模型回答中的引用率、分析网站的AI友好度（技术审计）、生成GEO优化建议。
triggers:
  - "GEO官网监测"
  - "网站被模型引用"
  - "AI友好度审计"
  - "llms.txt检测"
  - "geo website monitor"
  - "官网GEO分析"
  - "GEO监测"
---

# GEO官网监测工具

## 核心功能

本工具用于监测客户官网在大模型回答中的引用情况，并提供GEO优化建议。

### 三大阶段

1. **技术审计**：检查网站的AI爬虫友好度（robots.txt、llms.txt、HTTPS、内容结构等）
2. **内容友好度评估**：基于GEO论文的9种优化方法评估内容质量
3. **引用测试**：由 Hermes Agent 自己发送场景提示词做真实监测，最多5个场景词，并将结果回填最终报告

## 使用方法

### 基础使用

```bash
# 仅技术审计
geo-website-monitor audit https://example.com

# 仅内容评估
geo-website-monitor content https://example.com

# 完整测试（技术审计+内容评估+Agent引用测试结果回填；最多5个场景词）
# 注意：触发skill一开始必须先询问用户：场景提示词由Agent自主生成，还是由用户提供？以及本次测试几个（1-5）。
geo-website-monitor full https://example.com --scenarios scenarios.csv --scenario-count 5 --citation-results agent-citation-results.json

# 完整测试并创建飞书文档（使用bot身份）
geo-website-monitor full https://example.com --scenarios scenarios.csv --feishu

# 完整测试并创建飞书文档（使用user身份）
geo-website-monitor full https://example.com --scenarios scenarios.csv --feishu --feishu-user
```

### 场景词CSV格式

> 场景词最多支持 5 个；执行一开始必须先问用户：**场景提示词由 Agent 自主生成，还是由用户提供？** 以及本次需要测试几个场景提示词（1–5）。如果CSV或用户提供的提示词超过用户选择的数量，只读取前 N 条。

```csv
场景词,优先级
快速换模系统厂家推荐,高
苏州快速换模系统厂家,高
液压快速换模系统厂家,中
```

## 技术审计清单

### 1.1 技术基础设施

| 检查项 | 说明 |
|--------|------|
| HTTPS配置 | 是否已配置HTTPS（影响用户信任和AI抓取） |
| robots.txt | 是否存在且合理设置（避免错误屏蔽核心页面，允许AI爬虫） |
| Sitemap | 是否存在完整的站点地图（包含首页、栏目页、服务页、案例页、知识库、FAQ等） |
| 服务器响应 | 页面是否可快速、稳定访问（检查响应时间、HTTP状态码） |
| URL稳定性 | 检查是否有历史重定向或频繁变更URL的迹象 |
| 移动端适配 | 检查页面是否适配移动端（viewport设置、响应式布局） |

### 1.2 结构化数据与Schema

| 检查项 | 说明 |
|--------|------|
| Schema.org标记 | 是否存在JSON-LD格式的Schema标记 |
| Organization Schema | 首页是否有组织类型的Schema |
| 核心Schema类型 | 是否包含文章、服务、产品、FAQ等核心Schema类型 |
| Schema关键属性 | Schema是否包含@id、sameAs等关键属性 |
| Schema完整性 | Schema属性是否完整（非仅最低验证字段） |

### 1.3 内容结构

| 检查项 | 说明 |
|--------|------|
| 首页完整性 | 首页是否清晰说明企业定位、核心业务、服务对象、服务流程、代表案例 |
| 关于我们 | 是否有关于我们页面，包含企业背景、团队经验、发展历程、资质信息、联系方式 |
| 服务中心 | 是否有独立的服务页面，说明服务定义、适用对象、服务流程、交付内容、评估指标 |
| 案例中心 | 是否有案例页面，包含真实案例、客户行业、问题背景、执行过程、结果数据 |
| 知识库/博客 | 是否有持续发布的知识库或博客文章 |
| FAQ页面 | 是否有FAQ页面，覆盖用户常见问题 |
| 关键信息格式 | 关键信息是否使用HTML文本呈现（非仅图片） |
| 内容结构 | 是否使用"是什么、为什么、怎么做、常见问题"等结构 |

### 1.4 AI专属优化

| 检查项 | 说明 |
|--------|------|
| llms.txt | 是否存在llms.txt文件（AI发现协议） |
| robots.txt AI允许 | 是否明确允许GPTBot、Google-Extended、PerplexityBot等AI爬虫 |
| E-E-A-T信号 | 内容是否体现经验、专业性、权威性、可信度 |

### 1.5 基础SEO（保留）

| 检查项 | 说明 |
|--------|------|
| 标题标签 | 页面是否有&lt;title&gt; |
| Meta Keywords | 是否有meta keywords |
| Meta Description | 是否有meta description |
| OpenGraph | 是否有OpenGraph标记 |
| 登录墙 | 内容是否需要登录 |
| 付费墙 | 内容是否需要付费 |
| JS渲染 | 核心内容是否在HTML中（非仅JS渲染） |

## 内容友好度评估（基于GEO论文）

### 9种优化方法评分

| 排名 | 方法（中文） | 方法（英文） | 说明 | 论文来源 |
|---|---|---|---|---|
| 1 | 引语添加 | Quotation Addition | 添加可引用的直接引语 | GEO论文表2 |
| 2 | 数据添加 | Statistics Addition | 添加具体数字、百分比 | GEO论文表2 |
| 3 | 权威引用 | Cite Sources | 引用权威机构/来源 | GEO论文表2 |
| 4 | 流畅度优化 | Fluency Optimization | 改进语法、表达 | GEO论文表2 |
| 5 | 权威语气 | Authoritative Tone | 更自信、权威的语气 | GEO论文表2 |
| 6 | 简单易懂 | Easy-to-Understand | 更易理解的表达 | GEO论文表2 |
| 7 | 独特词汇 | Unique Words | 添加领域特有术语 | GEO论文表2 |
| 8 | 技术术语 | Technical Terms | 加入行业技术词汇 | GEO论文表2 |
| 9 | 关键词堆砌 | Keyword Stuffing | 传统SEO方法（不推荐） | GEO论文表2 |

## 引用测试方法（Agent真实监测模式）

### 强制流程

1. **触发本 skill 的一开始必须先询问用户两个问题**：
   - 场景提示词来源：由 **Agent 自主生成**，还是由 **用户提供提示词**？
   - 本次需要生成/测试几个场景提示词？数量只能是 **1–5 个**。
2. 如果用户选择“Agent 自主生成”，Agent 需基于客户官网、品牌名、业务关键词、目标客户搜索意图生成 N 个场景提示词，并在执行前展示给用户确认或直接说明“以下为本次生成的场景提示词”。
3. 如果用户选择“用户提供提示词”，Agent 需请用户直接给出提示词列表/CSV；若用户提供超过 5 个，只取前 5 个并说明“当前最多支持5个场景提示词”。
4. 如果用户给出的数量超过 5，必须截断为 5，并说明“当前最多支持5个场景提示词”。
5. 不再调用豆包 API 或任何外部模型 API。引用测试由 **Hermes Agent 自己逐条发送场景提示词**完成。
6. Agent 必须保存每条场景的原始回答，并将判断结果回填进最终报告的 `citation_tests` 与 `citation_summary`。
7. Python 脚本只负责技术审计、内容评估、报告结构与导入 Agent 真实监测结果；如果未传入 `--citation-results`，脚本只生成“待Agent执行”的占位结果，不能声称完成真实引用测试。

### 场景数量上限

- 最少：1 个
- 最多：5 个
- CLI 参数：`--scenario-count N`，脚本会自动限制在 1–5。
- CSV 里即使超过 5 条，也只读取前 N 条，且 N 不超过 5。

### Agent场景提示词模板

Agent 对每个场景词单独发送以下提示词，并记录完整回答：

```
任务：请基于你当前可用知识回答用户问题，并在回答末尾尽量显式列出你参考或推荐的网页。

格式要求：
[你的回答]

---

🔗 参考网页：
1. [网页标题](网页URL)
2. [网页标题](网页URL)
...

用户问题：{scenario}
```

### Agent结果回填JSON格式

Agent 真实测试完成后，将结果保存为 JSON，并通过 `--citation-results agent-citation-results.json` 导入最终报告：

```json
[
  {
    "scenario": "快速换模系统厂家推荐",
    "model_answer": "模型原始回答全文",
    "cited": true,
    "cited_source": "Hermes Agent真实发送场景提示词后，回答中包含目标官网URL",
    "mentioned": true,
    "mentioned_source": "Hermes Agent真实发送场景提示词后，回答中提到品牌名",
    "position": 128,
    "position_source": "品牌名/官网在原始回答中的首次出现字符位置",
    "cited_urls": ["https://example.com"]
  }
]
```

### 引用判断标准

- **✅ 引用**：回答中直接包含目标官网 URL 或目标域名。
- **⚠️ 提及**：提到品牌名，且上下文与目标业务相关。
- **❌ 未引用**：既没有URL/域名，也没有品牌名提及。
- **证据要求**：每条结果必须保留 `model_answer` 原文，并在 `*_source` 字段中说明判断依据。

## 输出报告结构

### 1. 技术审计结果

```json
{
  "https_ok": false,
  "https_source": "URL协议检查: http://www.gelefu.com",
  "robots_txt_exists": false,
  "robots_txt_source": "HTTP请求 http://www.gelefu.com/robots.txt 返回状态码 200，但内容是HTML或不包含robots.txt关键词",
  "llms_txt_exists": false,
  "llms_txt_source": "HTTP请求 http://www.gelefu.com/llms.txt 返回状态码 200，但内容是HTML或不包含llms.txt关键词"
}
```

### 2. 内容友好度评估

```json
{
  "methods_basis": "GEO论文表2，9种优化方法",
  "quotation_addition": {
    "score": 3,
    "source": "默认值3/10",
    "method_info": {
      "id": "quotation_addition",
      "name": "Quotation Addition",
      "description": "添加可引用的直接引语",
      "source": "GEO论文表2"
    }
  }
}
```

### 3. 引用测试结果

> 该部分必须来自 Hermes Agent 真实发送场景提示词后的结果回填；不再调用豆包 API。最多支持 5 条场景提示词。

```json
{
  "citation_tests": [
    {
      "scenario": "快速换模系统厂家推荐",
      "cited": true,
      "cited_source": "Hermes Agent真实发送场景提示词后，回答中包含目标官网URL/域名",
      "mentioned": true,
      "mentioned_source": "Hermes Agent真实发送场景提示词后，回答中提到品牌名",
      "position": 128,
      "position_source": "品牌名/官网在原始回答中的首次出现字符位置",
      "model_answer": "模型原始回答全文",
      "cited_urls": ["https://example.com"]
    }
  ],
  "citation_summary": {
    "total_tests": 1,
    "cited_count": 1,
    "mentioned_count": 1,
    "citation_rate": "100.0%"
  }
}
```

### 4. 优化建议（优先级排序）

```json
{
  "recommendations": [
    {
      "priority": 1,
      "method": "Quotation Addition",
      "method_source": "GEO论文表2，排名第1",
      "description": "在'关于我们'中添加CEO引语、客户评价",
      "description_source": "工具建议",
      "evidence_source": "默认值3/10"
    }
  ]
}
```

## llms.txt模板（给客户）

```txt
# llms.txt - AI Discovery File for Example.com
# 帮助AI系统理解和引用你的网站

# 网站基本信息
name: 格乐富
description: 苏州格莱富机械科技有限公司，专注于快速换模系统研发、设计、生产、销售及服务
contact: 400-990-7598

# 允许AI使用的内容（可选，默认全部允许）
allow: /product/
allow: /case.html
allow: /about.html

# 不希望AI引用的内容（可选）
disallow: /admin/

# 关键页面（AI应该优先索引）
key_pages:
  - /about.html
  - /case.html
  - /product/61.html

# 结构化数据提示（供AI提取）
structured_data:
  - type: Organization
    name: 苏州格莱富机械科技有限公司
    url: http://www.gelefu.com
  - type: Product
    name: 快速换模系统
    category: 工业自动化
```

## 附录：GEO论文摘要

本工具基于以下论文：

- **标题**：GEO: Generative Engine Optimization
- **作者**：Aggarwal, P., Murahari, V. S., Rajpurohit, S., Kalyan, A., Narasimhan, K., & Deshpande, A.
- **会议**：KDD 2024
- **arXiv**：2311.09735
- **代码和数据**：https://generative-engines.com/GEO/
- **核心思想**：传统SEO方法在生成式引擎中效果有限，需要新的优化方法
- **关键发现**：添加引语、添加数据、引用权威是最有效的方法（按论文排名）

---

## linked_files

- templates/geo-website-monitor-v2.py - 主程序脚本（来源验证版）
- templates/llms.txt.template - llms.txt模板
- references/arxiv-2311-09735-summary.md - GEO论文摘要解读
- references/gelefu-evaluation-v2.json - 实际评估报告示例（格莱富官网v2）
- references/agent-citation-monitoring-workflow.md - Agent真实引用监测工作流：场景数询问、最多5条、提示词模板、结果JSON回填与验证要点
- README.md - 快速入门指南（包含在压缩包中）

## 设计原则

- **必须为每一个结论提供明确来源**：不要只给出结果，要说明是如何得出的（如"HTTP 请求 http://example.com/robots.txt 返回状态码 200"）
- **不要提供主观的预期提升数据**：如"+41%"、"+30%"等，只陈述事实和方法依据
- **必须引用论文出处**：当提到 GEO 优化方法时，明确标注来源（如"GEO 论文表2，排名第1"）
- **所有判断必须有证据**：工具输出的每一个结论都必须能追溯到原始数据或请求

## 最近优化（2026-06-11，Agent真实引用监测版）

1. **取消豆包/外部模型API设想**：引用测试不再通过脚本调用豆包 API，统一由 Hermes Agent 自己发送场景提示词。
2. **场景来源与数量上限**：触发 skill 的一开始必须主动询问用户：场景提示词由 Agent 自主生成，还是由用户提供？以及需要测试几个场景提示词，范围 1–5；脚本参数 `--scenario-count` 也会强制限制到 1–5。
3. **真实结果回填**：Agent 完成逐条测试后，将原始回答、引用/提及判断、位置、引用URL写入JSON，并通过 `--citation-results` 导入最终报告。
4. **证据保留**：最终报告的 `citation_tests` 必须包含 `model_answer` 原文和各项判断来源。

## 最近优化（2026-06-10，GEO白皮书版）

### 已完成的优化
1. **去除旧版脚本** - 移除了有bug的旧版脚本，仅保留v2版本
2. **中文方法名显示** - 内容友好度评估的9种方法现在同时显示中文和英文
3. **引用测试详情增强** - 包含测试题目、引用状态、提及状态、位置、引用URLs、模型回答和引用总结
4. **飞书文档集成** - 支持--feishu参数自动创建飞书文档保存报告
5. **技术审计项大幅扩展** - 根据GEO白皮书新增检测项：
   - 技术基础设施：Sitemap、服务器响应、移动端适配
   - 结构化数据与Schema：Schema.org标记检测、Organization Schema、核心Schema类型、Schema关键属性、Schema完整性
   - 内容结构：首页完整性、关于我们页面、服务中心页面、案例中心页面、知识库/博客、FAQ页面、关键信息格式、内容结构
   - AI专属优化：robots.txt明确允许AI爬虫、E-E-A-T信号检测
   - 保留原有基础SEO项

## 常见陷阱

- **网站404页面返回200 OK**：有些网站的404页面会返回200状态码并显示HTML内容，必须同时检查响应内容是否真的是robots.txt/llms.txt，不能仅依赖状态码
- **验证robots.txt/llms.txt内容**：检查是否包含`<html`标签来判断是否是HTML页面，同时检查是否包含robots.txt/llms.txt特有关键词
- **引用测试必须先询问来源和数量**：触发完整GEO监测的一开始，必须主动问用户“场景提示词由我自主生成，还是由你提供？本次需要测试几个（1–5）？”最多5个，不能默认无限生成，也不能默认替用户决定提示词来源。
- **不要声称脚本会调用豆包API**：当前设计明确取消豆包/外部模型API；真实引用监测由Hermes Agent发送提示词，脚本只导入结果。
- **没有 `--citation-results` 时不算真实引用测试完成**：脚本会生成待Agent执行的占位结果，最终报告不能把占位结果解释为真实模型监测。

## 实战经验

- 有些网站已经配置了 robots.txt 和 llms.txt，这是好的起点
- HTTPS 配置缺失会影响技术审计（虽然不是致命问题）
- 即使技术审计得分不错，重点应放在内容优化上
- 验证文件存在性时必须同时检查内容格式，不能仅依赖HTTP状态码
- Quotation Addition（添加引语）是论文中排名第1的优化方法，应优先考虑

## 排查指南：查找之前运行的报告

如果用户询问之前运行的GEO监测任务状态，请按以下顺序检查：

1. **检查运行中的进程**：使用 `process action=list` 查看是否有正在运行的 geo-website-monitor 相关进程
2. **检查 /tmp 目录**：通常报告会保存在 `/tmp` 目录下，文件名通常包含 `geo`、`report` 或目标网站域名相关关键词
3. **检查 cron 任务**：使用 `cronjob action=list` 查看是否有定时运行的GEO监测任务
4. **检查会话历史**：使用 `session_search` 搜索相关关键词（如网站域名、"GEO监测"等）查找之前的会话

常见的报告文件名格式：
- `/tmp/geo-report.json`
- `/tmp/{website-domain}-report.json`
- `/tmp/geo-monitor-report.json`

如果找到 JSON 报告，可以直接读取并格式化展示给用户。
