# T6 QA 门禁量表

`kanban-qa` 在 `kanban_complete` 前逐项打勾；任一项 FAIL → `kanban_block(reason="qa-fail: ...")` 或补 comment 后退回 T5。

## A. 流水线完整性（本卡为 T6）

- [ ] **T0–T5**（非本卡）均 `done` 或已注明 `blocked` 原因
- [ ] 每阶段 `handoff.json` 存在且 `canonical_id` 一致
- [ ] `feishu_doc_url` 与 `paper_doc_registry.json` 一致

## B. 读论文质量

- [ ] T0 含 `thesis_one_liner`（≤40 字中文）+ `reading_map`
- [ ] T1 CEL 表 ≥3 行（`deep` 模式 ≥5 行），每行 Evidence 含 §/Fig/Table
- [ ] 至少 1 条主张为 `medium` 或 `weak`（全 `strong` → 退回 T1）
- [ ] T4 `experiment_audit` 覆盖五问中 ≥3 项，且与 T1 数字无冲突
- [ ] 若论文含目标函数 / 损失 / 奖励 / 策略式，T5 文档必须有“核心公式”节，且每条公式带符号释义
- [ ] T5 文档必须有“运行时策略细节”与“软硬件 Delta 对照表”

## C. 文档质量（`docs +fetch` 抽检）

- [ ] 中英双语：核心总结 + 参考方向 + CEL 表头 均有 ZH + EN
- [ ] “边界分析”明确区分：今天仍成立的假设 / 今天已失效的假设 / 可替换的新范式
- [ ] 无空 `##` 节；**无残留【待填】**
- [ ] 飞书云文档在线名称为 `[canonical_id] title_zh`（中文，非英文标题）
- [ ] 数字/指标均能在 T1/T4 handoff 中找到出处
- [ ] arXiv / PDF 链接有效

## D. 文档策略

- [ ] 本篇论文独立 doc（registry canonical_id 唯一）
- [ ] 同论文重跑为 append，非第二篇 doc

## E. Memory OS

- [ ] `search_memory` 在流水线开始前已执行；query **仅** canonical_id + 论文名（见 `paper_memory_search_query.py`，禁止全文 query）

## F. 飞书实时（feishu-live-updates）

- [ ] 编排已 `paper_feishu_stage_notify.py init` + `pipeline_started`
- [ ] T0–T6 每阶段完成均有 `notify stage_done`（或 comment 注明 `[feishu-notify-skipped]` 原因）
- [ ] T6 有 `pipeline_done`；IM 无 PDF/CEL 全文
- [ ] `workflow_id: paper-nexus:<canonical_id>` 至少 1 条记忆含 `feishu_doc_url`
- [ ] T1 记忆含 CEL 或 `claims`；T6 或 T5 含 `recommendation_zh` / `qa_pass`
- [ ] 未将 PDF 全文写入 Memory（仅摘要/CEL/链接）

## 输出格式

```json
{
  "qa_pass": true,
  "checks_failed": [],
  "doc_url": "https://my.feishu.cn/docx/...",
  "recommendation": "team-deep-read | methods-only | cite-with-caution | skip-reproduce",
  "recommendation_zh": "可供组内精读 / 仅引用方法节 / 可引用但慎复现 / 不建议复现"
}
```
