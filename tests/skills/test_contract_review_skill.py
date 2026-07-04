from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = PROJECT_ROOT / "skills" / "legal" / "contract_review_skill"
SCRIPT_PATH = SKILL_DIR / "contract_review_skill.py"


def load_module():
    spec = importlib.util.spec_from_file_location("contract_review_skill_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_review_contract_normal_service_contract_outputs_structured_report():
    mod = load_module()
    contract_text = """
软件开发服务合同
合同编号：SVC-2026-009
甲方：上海星河科技有限公司 统一社会信用代码：91310000MA1K000001
乙方：杭州云启软件有限公司 统一社会信用代码：91330100MA2K000002
一、服务内容：乙方为甲方开发客户管理系统并提交源代码、部署文档和用户手册。
二、合同金额：人民币300000元（大写：叁拾万元整），含6%增值税。
三、付款方式：合同签署后5个工作日内支付30%；系统验收合格并收到合规增值税专用发票后10个工作日内支付60%；质保期满后支付10%。
四、交付与验收：乙方应于2026年9月30日前交付系统。甲方应在收到交付物后10个工作日内按照附件一验收标准验收。
五、发票与税务：乙方应开具6%增值税专用发票。
六、双方权利义务：甲方应配合提供需求资料，乙方应按约定完成开发职责。
七、知识产权：本项目定制开发成果及源代码归甲方所有。
八、数据安全与隐私：乙方处理客户数据时应采取必要安全措施，未经甲方书面同意不得向第三方披露。
九、保密：双方对合作中获悉的保密信息承担保密义务，保密期限为合同终止后3年。
十、违约责任与赔偿：乙方逾期交付的，每逾期一日按合同总金额的0.05%支付违约金；造成损失的应赔偿。
十一、不可抗力：受不可抗力影响的一方应在3日内书面通知对方并提供证明。
十二、合同变更：任何变更均应经双方书面确认。
十三、解除与终止：一方严重违约且在收到书面通知后10日内未改正的，守约方有权解除合同。
十四、争议解决：因本合同产生的争议，双方应友好协商；协商不成的，提交甲方所在地有管辖权的人民法院解决。
十五、通知送达：双方通讯地址以本合同首页载明地址为准，变更应书面通知。
十六、附件：附件一《验收标准》与本合同具有同等效力。
签署日期：2026年7月1日
生效日期：2026年7月1日
终止日期：2026年12月31日
""".strip()

    result = mod.review_contract({
        "contract_text": contract_text,
        "contract_type": "服务合同",
        "review_perspective": "甲方",
        "review_depth": "standard",
    })

    assert result["summary"]["contract_name"] == "软件开发服务合同"
    assert result["summary"]["contract_type"] == "服务合同"
    assert result["summary"]["overall_risk_level"] in {"info", "low"}
    assert "上海星河科技有限公司" in result["summary"]["parties"]
    assert any(clause["title"] == "验收条款" and clause["present"] for clause in result["key_clauses"])
    assert "## 9. 免责声明" in result["final_report_markdown"]
    assert all(
        {
            "risk_id",
            "risk_title",
            "risk_level",
            "risk_type",
            "related_clause",
            "original_text",
            "risk_description",
            "business_impact",
            "suggestion",
            "suggested_revision",
        }.issubset(risk)
        for risk in result["risks"]
    )


def test_review_contract_detects_missing_key_clauses():
    mod = load_module()
    result = mod.review_contract({
        "contract_text": "采购合同\n甲方：上海星河科技有限公司\n乙方：杭州云启贸易有限公司\n标的：甲方向乙方采购办公电脑50台。\n合同金额：人民币100000元。",
        "contract_type": "采购合同",
        "review_perspective": "甲方",
        "review_depth": "standard",
    })

    missing_titles = {item["clause_type"] for item in result["missing_clauses"]}
    risk_titles = {risk["risk_title"] for risk in result["risks"]}

    assert "验收条款" in missing_titles
    assert "争议解决条款" in missing_titles
    assert "验收标准缺失" in risk_titles
    assert "争议解决条款缺失" in risk_titles
    assert result["summary"]["overall_risk_level"] == "high"


def test_review_contract_detects_amount_and_payment_risk():
    mod = load_module()
    result = mod.review_contract({
        "contract_text": "销售合同\n甲方：北京客户有限公司\n乙方：上海销售有限公司\n合同金额：人民币500000元。\n付款方式：具体付款时间和付款比例由双方另行协商。\n交付：乙方于2026年8月31日前交付产品。\n验收：甲方在5个工作日内验收。\n违约责任：违约方赔偿守约方损失。\n解除与终止：严重违约可解除合同。\n争议解决：提交乙方所在地有管辖权的人民法院解决。",
        "contract_type": "销售合同",
        "review_perspective": "乙方",
        "review_depth": "standard",
    })

    amount_risks = [risk for risk in result["risks"] if risk["risk_type"] == "金额风险"]

    assert any(risk["risk_title"] == "付款条件存在不确定表述" for risk in amount_risks)
    assert any("另行协商" in risk["original_text"] for risk in amount_risks)


def test_review_contract_detects_missing_dispute_resolution():
    mod = load_module()
    result = mod.review_contract({
        "contract_text": "租赁合同\n甲方：出租方公司\n乙方：承租方公司\n租赁物：办公场地。\n租期：2026年1月1日至2026年12月31日。\n租金：每月人民币10000元，按月支付。\n违约责任：逾期支付租金的，乙方应承担违约责任。\n解除与终止：任一方严重违约的，守约方可以解除合同。",
        "contract_type": "租赁合同",
        "review_perspective": "中立",
        "review_depth": "standard",
    })

    assert any(risk["risk_title"] == "争议解决条款缺失" for risk in result["risks"])
    assert any(risk["related_clause"] == "争议解决条款" for risk in result["risks"])


def test_review_contract_rejects_empty_or_invalid_input():
    mod = load_module()

    with pytest.raises(ValueError, match="contract_text"):
        mod.review_contract({"contract_text": ""})

    with pytest.raises(ValueError, match="input_data"):
        mod.review_contract(None)


def test_contract_review_schema_and_examples_are_valid_json():
    for path in [
        SKILL_DIR / "schema" / "input_schema.json",
        SKILL_DIR / "schema" / "output_schema.json",
        SKILL_DIR / "examples" / "input_example.json",
        SKILL_DIR / "examples" / "output_example.json",
    ]:
        with path.open(encoding="utf-8") as f:
            assert json.load(f)
