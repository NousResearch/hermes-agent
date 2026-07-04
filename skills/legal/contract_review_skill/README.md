# contract_review_skill

企业合同审核 Skill，用于对合同文本进行结构化解析、关键条款识别、基础风险审查、修改建议生成、风险等级评估和中文审核报告输出。

## 文件结构

```text
contract_review_skill/
├── SKILL.md
├── README.md
├── contract_review_prompt.md
├── contract_review_skill.py
├── schema/
│   ├── input_schema.json
│   └── output_schema.json
└── examples/
    ├── input_example.json
    └── output_example.json
```

## 能力范围

- 合同基础信息提取：合同名称、编号、类型、主体、日期、金额、付款、期限、地点、标的、联系人、附件。
- 关键条款识别：主体、标的、价款付款、交付、验收、发票税务、权利义务、违约、赔偿、保密、知识产权、数据安全、不可抗力、变更、解除终止、争议解决、通知送达、附件。
- 风险识别：主体、金额、履约、违约、法务、商务、知识产权、保密、数据安全、税务、缺失条款和表述风险。
- 输出结构化 JSON 和可直接展示给业务、法务、销售、采购人员的中文 Markdown 审核报告。

## Python 调用

```python
from pathlib import Path
import importlib.util

skill_path = Path("skills/legal/contract_review_skill/contract_review_skill.py")
spec = importlib.util.spec_from_file_location("contract_review_skill", skill_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

result = module.review_contract({
    "contract_text": "软件开发服务合同\n甲方：...\n乙方：...",
    "contract_type": "服务合同",
    "review_perspective": "甲方",
    "review_depth": "standard"
})
```

也可以通过 stdin/stdout 调用：

```bash
python skills/legal/contract_review_skill/contract_review_skill.py \
  < skills/legal/contract_review_skill/examples/input_example.json
```

## Hermes Skill 使用

在 Hermes 运行时同步 bundled skills 后，可通过自然语言触发：

```text
请使用 contract_review_skill 审核以下服务合同，并从甲方视角输出结构化 JSON 和中文审核报告：...
```

同步命令：

```bash
source venv/bin/activate
python tools/skills_sync.py
```

同步后 Skill 会出现在 `~/.hermes/skills/legal/contract_review_skill/`。

## 输出约定

输出必须包含：

- `summary`
- `extracted_fields`
- `key_clauses`
- `risks`
- `missing_clauses`
- `optimization_suggestions`
- `suggested_action_items`
- `final_report_markdown`

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

## 扩展方向

当前 Python 入口是轻量、确定性的规则基线，适合自动化测试和 Workflow 编排。后续可以扩展：

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

## 免责声明

本 Skill 输出仅作为合同初审和业务辅助参考，不构成正式法律意见，重大合同应由专业法务或律师复核。
