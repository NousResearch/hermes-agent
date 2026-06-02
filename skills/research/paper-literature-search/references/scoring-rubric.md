# 文献检索综合分（研究员默认权重）

面向「给课题组快速扫必读论文」，不是系统综述。在 **Semantic Scholar 检索候选集内** 做归一化后加权。

## 信号与权重

| 信号 | 符号 | 权重 | 说明 |
|------|------|------|------|
| 检索相关性 | `rel` | **0.35** | S2 `relevance`（0–1）；无则按标题/摘要词命中估算 |
| 被引强度 | `cite` | **0.30** | `log1p(citationCount)`，批内 min-max 归一化 |
| 高影响引用 | `infl` | **0.15** | `log1p(influentialCitationCount)`，批内归一化 |
| 时效 | `rec` | **0.15** | 发表年份；ML/NLP 默认 2020+ 满分衰减 |
| 可获取全文 | `oa` | **0.05** | 有 `openAccessPdf` 或 arXiv id |

**综合分：** `score = 0.35·rel + 0.30·cite + 0.15·infl + 0.15·rec + 0.05·oa`（0–100 展示为 `round(score*100)`）

## 时效分（默认 profile: `ml`）

| 年份 | rec |
|------|-----|
| 当前年 | 1.0 |
| 去年 | 0.92 |
| 2 年前 | 0.78 |
| 3 年前 | 0.55 |
| 更早 | 0.35 |

`profile=survey` 时 2015+ 衰减更慢（经典综述可进 Top）。

## 过滤（硬规则）

- 去掉无标题、重复 arXiv id（保留高分那条）
- 默认 `min_citations=0`；用户说「经典」「必读」→ `min_citations=50`
- 用户说「最新」「SOTA」→ `year_floor=当前年-2`，`rec` 权重 +0.05、`cite` -0.05（脚本 `--boost-recency`）

## 数据源

1. **Semantic Scholar Graph API**（主）：`paper/search`，字段含引用/影响力/相关性  
2. **arXiv API**（补）：S2 不足 `limit` 时用 arXiv 关键词补候选，引用记 0、靠 rel+rec 撑榜

## 输出

- Top **8**（默认，可用 `--top 12`）
- 飞书 IM：分 3 条进度 + 1 条结果表（见 `feishu-delivery.md`）
- **不**创建飞书云文档、**不**走 `paper_doc_registry`
