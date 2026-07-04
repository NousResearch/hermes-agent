---
name: contract_review_skill
description: 面向企业合同文本的结构化条款识别、风险审查、修改建议和中文审核报告生成 Skill。
version: 0.1.0
metadata:
  hermes:
    tags: [contract, legal, risk-review, enterprise, chinese]
    related_files:
      - contract_review_prompt.md
      - contract_review_skill.py
      - schema/input_schema.json
      - schema/output_schema.json
---

# contract_review_skill

## 触发场景

当用户上传、粘贴或要求审核合同、协议、订单、采购文件、销售合同、服务合同、租赁合同、劳动合同、保密协议、NDA 或类似法律商务文本时使用本 Skill。

典型请求：

- 帮我审核这份合同。
- 识别合同条款和风险。
- 输出一份法务/业务可读的合同审核报告。
- 从甲方/乙方视角审查付款、违约、验收、争议解决等条款。
- 对照公司模板或内部规则检查合同缺失项。

## 能力边界

- 本 Skill 用于合同初审、业务辅助和风险提示，不构成正式法律意见。
- 不承诺合同一定合法、一定有效或一定无风险。
- 不编造合同中不存在的条款、事实、金额、日期、主体或法律依据。
- 无法判断或未在合同中出现的内容必须标记为“未在合同中明确约定”。
- 每条风险必须引用合同原文；缺失条款的 `original_text` 使用“未在合同中明确约定”。
- 修改建议必须具体、可执行，`suggested_revision` 尽量提供可直接替换到合同中的中文条款文本。
- 重大合同、复杂交易、涉外合同、监管高风险合同应提示由专业法务或律师复核。

## 标准输入

使用如下 JSON 结构：

```json
{
  "contract_text": "合同全文文本",
  "contract_type": "采购合同/销售合同/服务合同/租赁合同/劳动合同/保密协议/其他",
  "review_perspective": "甲方/乙方/中立",
  "review_depth": "standard/deep",
  "industry": "可选，行业信息",
  "company_policy": "可选，企业内部审核规则",
  "reference_template": "可选，标准合同模板文本",
  "extra_requirements": "可选，用户额外审核要求"
}
```

`schema/input_schema.json` 是标准输入契约。`examples/input_example.json` 可作为 Workflow 或 API 调用样例。

## 标准输出

必须输出结构化 JSON，并包含：

- `summary`
- `extracted_fields`
- `key_clauses`
- `risks`
- `missing_clauses`
- `optimization_suggestions`
- `suggested_action_items`
- `final_report_markdown`

`final_report_markdown` 必须是可直接展示给业务、法务、销售、采购人员的中文审核报告，包含：

1. 合同概览
2. 总体风险结论
3. 关键条款摘要
4. 高风险问题
5. 中低风险问题
6. 缺失条款
7. 修改建议
8. 建议下一步动作
9. 免责声明：本审核结果仅作为合同初审和业务辅助参考，不构成正式法律意见，重大合同应由专业法务或律师复核。

`schema/output_schema.json` 是标准输出契约。`examples/output_example.json` 是示例输出。

## 审核流程

按以下顺序执行：

1. 输入校验：确认 `contract_text` 非空，识别合同类型、审核视角和审核深度。
2. 合同基础信息提取：合同名称、编号、类型、主体、日期、金额、付款、期限、地点、标的、联系人、附件。
3. 合同类型识别：若用户未明确类型，基于标题和正文关键词推断。
4. 关键条款切分：识别主体、标的、价款付款、交付、验收、发票税务、权利义务、违约、赔偿、保密、知识产权、数据安全、不可抗力、变更、解除终止、争议解决、通知送达、附件。
5. 条款完整性检查：对照合同类型的必备条款输出缺失项。
6. 风险规则匹配：识别主体、金额、履约、违约、法务、商务、知识产权、保密、数据安全、税务、缺失条款和表述风险。
7. 高风险问题识别：优先列出可能影响签署、回款、履约、追责或争议解决的风险。
8. 修改建议生成：输出可执行建议和建议替换条款。
9. 风险等级汇总：按最高风险输出总体等级。
10. 输出结构化 JSON。
11. 生成 Markdown 审核报告。

## 风险等级

- `high`：高风险，可能导致重大法律、财务、履约或回款风险，建议必须修改。
- `medium`：中风险，存在争议、执行难度或责任不清，建议修改或补充。
- `low`：低风险，表述可优化，不影响合同核心履行。
- `info`：提示信息，仅作为业务关注点或优化建议。

每条风险必须包含：

- `risk_id`
- `risk_title`
- `risk_level`
- `risk_type`
- `related_clause`
- `original_text`
- `risk_description`
- `business_impact`
- `suggestion`
- `suggested_revision`

## 合同类型审核重点

### 采购合同

重点审核供应商主体资质、交付时间、验收标准、质量保证、付款条件、违约责任、售后服务、发票税率。

### 销售合同

重点审核客户付款能力、回款周期、付款节点、验收条件、逾期付款责任、所有权转移、交付证明、争议解决。

### 服务合同

重点审核服务范围、服务标准、SLA、人员投入、交付物、验收方式、变更机制、知识产权归属。

### 保密协议

重点审核保密信息范围、保密期限、例外情形、违约责任、信息返还或销毁、关联方披露、举证责任。

### 租赁合同

重点审核租赁物描述、租期、租金、押金、维修责任、提前解除、转租限制、违约责任。

## 可复用入口

本 Skill 同时提供轻量 Python 入口：

```python
from contract_review_skill import review_contract

result = review_contract({
    "contract_text": "...",
    "contract_type": "服务合同",
    "review_perspective": "甲方",
    "review_depth": "standard"
})
```

也可以用 stdin/stdout 方式运行：

```bash
python skills/legal/contract_review_skill/contract_review_skill.py < skills/legal/contract_review_skill/examples/input_example.json
```

Python 入口提供确定性的基础解析和规则审查，适合自动化测试、Workflow 编排和后续工具集成；深度法律语义判断应结合本 Skill 的提示词、RAG、模板库和法务复核。

## 扩展点

保留以下扩展方向：

- 企业合同模板库比对
- 法务知识库 RAG 检索
- 行业审核规则配置
- 不同客户的审核规则自定义
- 多语言合同审核
- Word/PDF 合同解析接入
- 合同红线修订
- 历史合同风险统计
- 审批流系统集成
- CRM/ERP/OA 系统集成
