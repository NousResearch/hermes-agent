# Sprint 02 阶段总结

## 1. 本阶段目标

建立 Review Gate 质量门禁，让 Hermes 在交付结果前自动进行结构性审查，阻断不完整的输出，允许标注风险的降级交付。

## 2. 本阶段完成内容

- ✅ 新增 `agent/review_templates.py`：3 套硬编码审查模板（品牌/架构/通用），13 条检查
- ✅ 新增 `agent/review_gate.py`：ReviewGate 类 + rule-based 自动检查 + 阻断/降级逻辑 + 评分引擎
- ✅ 修改 `run_agent.py`：3 处集成点（import、ReviewGate 初始化、check 调用 + 阻断处理）
- ✅ 手动测试验证：评分逻辑、阻断规则、降级交付、revision 耗尽
- ✅ 语法验证：中文引号转义修复

## 3. 当前功能

```
Agent 执行完成
  → ReviewGate.check(task_card, result_summary)
    → Rule checks (自动):
      - has_task_card ✓
      - has_compiled_intent ✓
      - has_result_summary ✓
      - success_criteria_addressed ✓
    → LLM checks (模板定义):
      - brand_strategy: 5 条
      - architecture_review: 5 条
      - universal: 4 条
    → 评分 + 阻断判断
  → 阻断? → status=reviewing, result.review_blocked=True
  → 降级? → result.review_risks=[...]
  → 通过? → 正常交付
  → EventLog → _persist_session → return
```

## 4. 当前架构

```
hermes-agent/
  agent/
    review_gate.py         ← 新增：ReviewGate + ReviewResult + CheckResult
    review_templates.py    ← 新增：3 套硬编码模板
  run_agent.py             ← 修改：Review Gate 集成
```

## 5. 关键实现细节

- **Rule checks**：4 个确定性检查，不依赖 LLM
- **LLM checks**：9 个语义检查模板，pending 状态不计入评分
- **评分**：rule checks 全部计入 + LLM checks 已评估的计入 = (passed / evaluated) * 100
- **阻断**：任一 rule 失败 → 阻断；score < 70 且未 revision → 阻断
- **降级**：revision_count >= 1 且仍未通过 → review_exhausted + 降级交付
- **TaskCard 集成**：review_result 写入 TaskCard JSON，review_blocked/risks 注入 result dict

## 6. 新增/修改文件

| 文件 | 改动类型 | 行数（约） |
|------|---------|-----------|
| `agent/review_templates.py` | **新增** | 90 |
| `agent/review_gate.py` | **新增** | 230 |
| `run_agent.py` | **修改** | +35 |

## 7. 已知问题

1. LLM 检查尚未自动执行（模板定义，pending 状态）
2. success_criteria_addressed 使用关键词 heuristic
3. 子 Agent 任务不经过 Review Gate
4. 品牌/架构模板仅有 LLM 检查，自动审查时全部 pending

## 8. 下一阶段 (Sprint 3) 建议

**目标**：Lightweight Memory（结构化记忆系统）

**关键任务**：
1. 重构 `tools/memory_tool.py`：新增 type/scope/confidence/source/last_verified_at 字段
2. YAML frontmatter 正式读写，向后兼容旧格式
3. 记忆注入策略：按 scope 过滤（global 最多 20 条、project 最多 30 条、session 最多 20 条）
4. Review Gate 接入 Memory：`matches_user_preferences` 和 `matches_project_context` 检查

**风险**：
- 旧格式记忆迁移需自动添加默认元数据
- scope 过滤逻辑需要正确的项目上下文检测
- Review Gate 接入 Memory 后 LLM 检查可部分自动化

## 9. 验收结果

- [x] Rule-based 检查不通过 → 阻断交付
- [x] 阻断后 status 变为 reviewing
- [x] quality_score < 70 且 revision_count < 1 → 阻断
- [x] revision_count >= 1 → review_exhausted + 降级交付
- [x] 正常任务（全部 rule pass）→ score 100，不阻断
- [x] ReviewResult 写入 TaskCard.review_result
- [x] result dict 包含 review_blocked 和 review_risks

## 10. Git 提交说明

```bash
git add .
git commit -m "feat: add Review Gate and static review templates (Sprint 2)"
```
