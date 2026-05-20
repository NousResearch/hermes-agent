# 飞书投递（无在线文档）

与 `kanban-paper-nexus` 相同：**`kanban-feishu-live` + `lark-cli`**，board=`paper-search`。

## 消息节奏（4 条 IM）

| 顺序 | event | 内容 |
|------|-------|------|
| 1 | `pipeline_started` | 检索式、数据源、预计条数 |
| 2 | `stage_done` T0 | 候选 N 篇（S2+arXiv） |
| 3 | `stage_done` T1 | 排序完成，展示权重说明一行 |
| 4 | `pipeline_done` | Top-K 列表（标题、年份、引用、分、链接） |

禁止上传 `docs +create`；长摘要不进 IM（每条 ≤2 行）。

## 脚本一键（E2E / 编排收尾）

```bash
python3 skills/research/paper-literature-search/scripts/paper_search_pipeline.py \
  "多模态大模型" --chat-id "$HERMES_SESSION_CHAT_ID" --top 8
```

编排器也可分步：`paper_search_rank.py` → `paper_search_feishu_deliver.py`。
