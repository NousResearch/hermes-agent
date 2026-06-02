# Paper Nexus — Worker Playbook

Board: `paper-nexus`. Titles: `[paper] <arxiv_id> …`.

**Read first:** `paper-reading-framework.md` (CEL + 实验五问 + 写作原则).  
**跨会话：** `memory-os.md`（每阶段 `store_memory_markdown`）。  
**飞书实时：** `feishu-live-updates.md`（每阶段 `paper_feishu_stage_notify.py`，不改 Hermes 核心）。

## Scripts & registry

```bash
python3 skills/research/kanban-paper-nexus/scripts/paper_nexus_metadata.py <id>
python3 skills/research/kanban-paper-nexus/scripts/paper_doc_registry.py resolve <id>
python3 skills/research/kanban-paper-nexus/scripts/paper_feishu_doc_sync.py <id>
```

## Handoff schema (`$HERMES_KANBAN_WORKSPACE/handoff.json`)

```json
{
  "paper_id": "2402.03300v3",
  "canonical_id": "2402.03300",
  "title_zh": "DeepSeekMath 开放数学推理模型",
  "stage": "T1",
  "thesis_one_liner": "用公开数学语料续训 7B 提升 MATH 推理。",
  "arxiv_abs": "https://arxiv.org/abs/2402.03300",
  "arxiv_pdf": "https://arxiv.org/pdf/2402.03300",
  "pdf_sections_read": ["abstract", "fig1", "§3", "Table.2"],
  "claims": [
    {
      "id": "C1",
      "claim_zh": "…",
      "evidence": "Table.1 MATH 51.7%",
      "strength": "strong",
      "limit_zh": "…"
    }
  ],
  "experiment_audit": {},
  "feishu_doc_url": "https://my.feishu.cn/docx/...",
  "feishu_doc_action": "create",
  "open_source_refs": []
}
```

Downstream: `kanban_show` 读父任务 handoff；**禁止**编造父任务未出现的数字。

---

## T0 — 论点与阅读地图

- `paper_nexus_metadata.py`
- 产出 `thesis_one_liner`（≤40 汉字）与 **`title_zh`**（论文中文名，供飞书在线文档名）
- `reading_map`：5 步扫读顺序 + 计划读的 §/Fig
- `deep` 模式：`web_extract` PDF 前 3–5 页 + 目录；记入 `pdf_sections_read`

**完成标准：** 读者不看 PDF 也能知道「这篇想证明什么、去哪找证据」。

**Memory：** `paper_memory_search_query.py` → `search_memory`（仅代号+标题，`limit=3`）；完成后 `store_memory_markdown` stage=T0。  
**Feishu：** `notify --event stage_done --stage T0`（摘要=thesis_one_liner）。

---

## T1 — 主张-证据链（CEL）

- 填 CEL 表 ≥3 行（`deep` ≥5 行）
- 每行 Evidence 必须可定位（§3.1 / Fig.2 / Table.1）
- 至少 1 行 strength 为 `medium` 或 `weak`

**完成标准：** 无「显著提升」类空话；局限列诚实。

**Memory：** stage=T1，`importance_score`≥0.75，CEL 全文写入 body。

---

## T2 — 方法与复现要点

- 数据从哪来、模型结构、训练/推理流程（符号表可选）
- **复现清单：** 代码 URL、权重、GPU 量级、关键超参 — 没有的写「未提供」

可与 T3 并行（均只依赖 T0）。

---

## T3 — 对标与开源地图

- 2–3 篇最接近 arXiv 工作（标题 + id + 差异一句）
- 开源表：PaperQA / GROBID / 领域 repo — 写清**替代哪一步**

---

## T4 — 实验审计与局限

- 按 framework **五问** 填表（baseline / 指标 / 方差 / 消融 / 复现）
- 汇总 3–5 条局限 bullet 进 `experiment_audit`
- 只用 T1/T2 已出现过的指标；新数字需注明出处

---

## T5 — 飞书精读文档

1. 读 `lark-shared`、`lark-doc`、`feishu-doc-bilingual-template.md`
2. 合并 T0–T4 handoff，**替换** skeleton 中【待填】（可先 `build_bilingual_doc_md.py` 出骨架再手改）
3. 最终写飞书文档时，默认走 `paper_feishu_doc_sync.py --handoff handoff.json`；不要直接手搓 `lark-cli docs +create` 整篇正文，否则容易退化成“摘要版”而漏掉公式 / 边界 / Delta 节。
3. `paper_feishu_doc_sync.py --handoff handoff.json` — 在线文档名 `[canonical_id] title_zh`；遵守 create/append
4. 核心总结 5 条必须中文可读；English mirror

**禁止：** 跨论文 doc；handoff 无依据的新 SOTA 数字。

**Memory：** stage=T5，含 `feishu_doc_url` + 中文核心总结 5 条。

---

## T6 — QA

- 逐项 `qa-rubric.md`；输出 `qa_pass` + `recommendation`（精读/引用/不建议复现）
- FAIL → block 并 @ 对应 stage
- PASS → `store_memory_markdown` stage=T6（`qa_pass` + `recommendation_zh`）

---

## Open-source index

| Project | URL | Use |
|---------|-----|-----|
| PaperQA | https://github.com/Future-House/paper-qa | 章节 QA |
| arxiv skill | `skills/research/arxiv` | 元数据 |
| GROBID | https://github.com/kermitt2/grobid | PDF TEI |
| Unstructured | https://github.com/Unstructured-IO/unstructured | 分块 |
| Semantic Scholar API | https://api.semanticscholar.org | 引用图（可选） |

---

## Tools

- Worker 用 `kanban_*`，不用 `hermes kanban` CLI（容器内无 CLI）
- 每阶段完成前：`paper_feishu_stage_notify.py notify`（短 IM + DAG 行）；T5 勿把 doc 全文贴进 IM
- Gateway Notifier 仅兜底（`paper_feishu_subscribe.py`）；不替代阶段 notify
