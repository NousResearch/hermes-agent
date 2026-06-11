# Agent 真实引用监测工作流（2026-06-11）

## 背景

本次优化将 `geo-website-monitor` 的引用测试从“脚本/外部 API 设想”改为 **Hermes Agent 亲自发送场景提示词并回填报告**。明确取消豆包 API 或任何外部模型 API 调用。

## 强制流程

1. 触发完整 GEO 官网监测的一开始，先问用户两个问题：
   - 场景提示词由 Agent 自主生成，还是由用户提供？
   - 本次需要测试几个场景提示词？范围 1–5。
2. 如果用户选择 Agent 自主生成，Agent 根据客户官网、品牌名、业务关键词、目标客户搜索意图生成 N 个场景词，并在执行前展示或说明本次使用的场景词。
3. 如果用户选择用户提供提示词，Agent 等待用户给出提示词列表/CSV；超过 5 个时只取前 5 个。
4. 如果用户要求超过 5 个，截断为 5，并说明当前最多支持 5 个。
5. Agent 对每个场景词单独发送提示词，保存模型原始回答。
6. Agent 判断回答中是否：
   - 包含目标官网 URL 或域名（`cited`）
   - 提及品牌名且上下文相关（`mentioned`）
   - 出现位置（`position`，首次出现字符位置；没有则 null）
6. Agent 将结果保存为 JSON，通过脚本参数 `--citation-results` 导入最终报告。

## 场景提示词模板

```text
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

## 回填 JSON 格式

```json
[
  {
    "scenario": "快速换模系统厂家推荐",
    "model_answer": "模型原始回答全文",
    "cited": true,
    "cited_source": "Hermes Agent真实发送场景提示词后，回答中包含目标官网URL/域名",
    "mentioned": true,
    "mentioned_source": "Hermes Agent真实发送场景提示词后，回答中提到品牌名",
    "position": 128,
    "position_source": "品牌名/官网在原始回答中的首次出现字符位置",
    "cited_urls": ["https://example.com"]
  }
]
```

## 脚本调用

```bash
python templates/geo-website-monitor-v2.py full https://example.com \
  --brand Example \
  --scenarios scenarios.csv \
  --scenario-count 5 \
  --citation-results agent-citation-results.json \
  --output report.json
```

## 验证要点

- `citation_tests` 数量不超过 5。
- `citation_summary.total_tests` 等于导入条数。
- 每条结果保留 `model_answer`。
- 第 6 条及以后不得出现在最终报告。
- 没有 `--citation-results` 时，只能视为待 Agent 执行/占位，不能声称真实引用测试已完成。
