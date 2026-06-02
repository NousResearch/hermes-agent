# 飞书投递（默认无在线文档）

与 `kanban-paper-nexus` 相同：**`kanban-feishu-live` + `lark-cli`**，board=`paper-search`。

## 消息节奏（4 条 IM）

| 顺序 | event | 内容 |
|------|-------|------|
| 1 | `pipeline_started` | 检索式、数据源、预计条数 |
| 2 | `stage_done` T0 | 候选 N 篇（S2+arXiv） |
| 3 | `stage_done` T1 | 排序完成，展示权重说明一行 |
| 4 | `pipeline_done` | Top-K 双语快览（前 5 篇双语导读 + 剩余简表） |

禁止默认上传 `docs +create`；长摘要不进 IM。每篇最多 4 行：中文题意、英文原题、中英文一句话导读、链接/venue。

## 脚本一键（E2E / 编排收尾）

```bash
python3 skills/research/paper-literature-search/scripts/paper_search_pipeline.py \
  "多模态大模型" --chat-id "$HERMES_SESSION_CHAT_ID" --top 8
```

编排器也可分步：`paper_search_rank.py` → `paper_search_feishu_deliver.py`。
