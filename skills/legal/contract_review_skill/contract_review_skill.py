from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


SKILL_VERSION = "0.1.0"

SUPPORTED_CONTRACT_TYPES = {
    "采购合同",
    "销售合同",
    "服务合同",
    "租赁合同",
    "劳动合同",
    "保密协议",
    "其他",
}

SUPPORTED_PERSPECTIVES = {"甲方", "乙方", "中立"}
SUPPORTED_DEPTHS = {"standard", "deep"}

RISK_LEVEL_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class ContractReviewInput:
    contract_text: str
    contract_type: str = "其他"
    review_perspective: str = "中立"
    review_depth: str = "standard"
    industry: Optional[str] = None
    company_policy: Optional[str] = None
    reference_template: Optional[str] = None
    extra_requirements: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContractReviewInput":
        if not isinstance(data, dict):
            raise ValueError("input_data must be a dict")
        contract_text = data.get("contract_text")
        if not isinstance(contract_text, str) or not contract_text.strip():
            raise ValueError("contract_text is required and must not be empty")

        contract_type = str(data.get("contract_type") or "其他").strip()
        if contract_type not in SUPPORTED_CONTRACT_TYPES:
            contract_type = "其他"

        review_perspective = str(data.get("review_perspective") or "中立").strip()
        if review_perspective not in SUPPORTED_PERSPECTIVES:
            review_perspective = "中立"

        review_depth = str(data.get("review_depth") or "standard").strip().lower()
        if review_depth not in SUPPORTED_DEPTHS:
            raise ValueError("review_depth must be 'standard' or 'deep'")

        return cls(
            contract_text=contract_text,
            contract_type=contract_type,
            review_perspective=review_perspective,
            review_depth=review_depth,
            industry=_optional_str(data.get("industry")),
            company_policy=_optional_str(data.get("company_policy")),
            reference_template=_optional_str(data.get("reference_template")),
            extra_requirements=_optional_str(data.get("extra_requirements")),
        )


@dataclass
class ClauseMatch:
    clause_type: str
    title: str
    present: bool
    original_text: str
    confidence: str = "medium"
    normalized_summary: str = ""


@dataclass
class RiskFinding:
    risk_id: str
    risk_title: str
    risk_level: str
    risk_type: str
    related_clause: str
    original_text: str
    risk_description: str
    business_impact: str
    suggestion: str
    suggested_revision: str


@dataclass
class MissingClause:
    clause_type: str
    importance: str
    reason: str
    suggested_clause: str


@dataclass
class ContractReviewResult:
    summary: Dict[str, Any]
    extracted_fields: Dict[str, Any]
    key_clauses: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    missing_clauses: List[Dict[str, Any]]
    optimization_suggestions: List[Dict[str, Any]]
    suggested_action_items: List[str]
    final_report_markdown: str
    review_metadata: Dict[str, Any] = field(default_factory=dict)


CLAUSE_DEFINITIONS = [
    ("contract_subject", "合同主体条款", ["甲方", "乙方", "丙方", "主体", "委托方", "受托方", "买方", "卖方", "出租方", "承租方"]),
    ("subject_matter", "标的条款", ["标的", "采购内容", "销售产品", "服务内容", "租赁物", "项目内容", "工作内容"]),
    ("price_payment", "价款与付款条款", ["合同金额", "价款", "费用", "付款", "支付", "结算", "预付款", "尾款"]),
    ("delivery", "交付条款", ["交付", "交货", "交付时间", "交付地点", "交付物", "交货期"]),
    ("acceptance", "验收条款", ["验收", "验收标准", "验收方式", "确认合格", "验收报告"]),
    ("invoice_tax", "发票与税务条款", ["发票", "税率", "增值税", "开票", "税费", "专票", "普票"]),
    ("rights_obligations", "双方权利义务条款", ["权利义务", "双方义务", "甲方义务", "乙方义务", "职责"]),
    ("breach", "违约责任条款", ["违约", "违约责任", "违约金", "逾期", "赔偿"]),
    ("indemnity", "赔偿责任条款", ["赔偿", "损失", "补偿", "损害赔偿"]),
    ("confidentiality", "保密条款", ["保密", "保密信息", "商业秘密", "保密期限"]),
    ("intellectual_property", "知识产权条款", ["知识产权", "著作权", "专利", "商标", "成果归属", "源代码", "使用权"]),
    ("data_security", "数据安全与隐私条款", ["数据安全", "个人信息", "隐私", "客户数据", "数据处理", "网络安全"]),
    ("force_majeure", "不可抗力条款", ["不可抗力", "不可预见", "不可避免", "不可克服"]),
    ("change", "合同变更条款", ["合同变更", "变更", "补充协议", "书面确认"]),
    ("termination", "合同解除与终止条款", ["解除", "终止", "提前终止", "合同解除", "合同终止"]),
    ("dispute_resolution", "争议解决条款", ["争议解决", "管辖", "仲裁", "法院", "诉讼", "协商解决"]),
    ("jurisdiction", "管辖法院或仲裁机构", ["管辖法院", "仲裁委员会", "仲裁机构", "人民法院", "管辖地"]),
    ("notice", "通知送达条款", ["通知", "送达", "通讯地址", "电子邮件", "书面通知"]),
    ("attachments", "附件条款", ["附件", "补充协议", "技术规格书", "报价单", "订单"]),
]

REQUIRED_BY_TYPE = {
    "采购合同": ["合同主体条款", "标的条款", "价款与付款条款", "交付条款", "验收条款", "发票与税务条款", "违约责任条款", "合同解除与终止条款", "争议解决条款"],
    "销售合同": ["合同主体条款", "标的条款", "价款与付款条款", "交付条款", "验收条款", "发票与税务条款", "违约责任条款", "合同解除与终止条款", "争议解决条款"],
    "服务合同": ["合同主体条款", "标的条款", "价款与付款条款", "交付条款", "验收条款", "双方权利义务条款", "知识产权条款", "违约责任条款", "合同解除与终止条款", "争议解决条款"],
    "租赁合同": ["合同主体条款", "标的条款", "价款与付款条款", "违约责任条款", "合同解除与终止条款", "争议解决条款"],
    "劳动合同": ["合同主体条款", "标的条款", "价款与付款条款", "双方权利义务条款", "合同解除与终止条款", "争议解决条款"],
    "保密协议": ["合同主体条款", "保密条款", "违约责任条款", "合同解除与终止条款", "争议解决条款"],
    "其他": ["合同主体条款", "标的条款", "价款与付款条款", "违约责任条款", "合同解除与终止条款", "争议解决条款"],
}

TYPE_FOCUS = {
    "采购合同": ["供应商主体资质", "交付时间", "验收标准", "质量保证", "付款条件", "违约责任", "售后服务", "发票税率"],
    "销售合同": ["客户付款能力", "回款周期", "付款节点", "验收条件", "逾期付款责任", "所有权转移", "交付证明", "争议解决"],
    "服务合同": ["服务范围", "服务标准", "SLA", "人员投入", "交付物", "验收方式", "变更机制", "知识产权归属"],
    "保密协议": ["保密信息范围", "保密期限", "例外情形", "违约责任", "信息返还或销毁", "关联方披露", "举证责任"],
    "租赁合同": ["租赁物描述", "租期", "租金", "押金", "维修责任", "提前解除", "转租限制", "违约责任"],
}

MISSING_CLAUSE_SUGGESTIONS = {
    "验收条款": "建议补充：甲方应在收到交付物后【】个工作日内完成验收。验收标准为【列明客观标准/技术指标/文档清单】；如甲方逾期未提出书面异议，视为验收通过。",
    "争议解决条款": "建议补充：因本合同引起或与本合同有关的争议，双方应先友好协商；协商不成的，任一方可向【有管辖权的人民法院/约定仲裁委员会】申请解决。",
    "管辖法院或仲裁机构": "建议补充：双方同意由【合同签署地/被告住所地/甲方所在地】有管辖权的人民法院管辖，或提交【】仲裁委员会仲裁。",
    "不可抗力条款": "建议补充：因不可抗力导致不能履行的，受影响方应在【】日内书面通知对方并提供证明，双方可根据影响程度部分或全部免除责任。",
    "合同解除与终止条款": "建议补充：发生严重违约、资质丧失、逾期超过【】日、无法继续履行等情形时，守约方有权书面通知解除合同，并要求违约方承担责任。",
    "违约责任条款": "建议补充：任一方违反本合同约定造成对方损失的，应赔偿实际损失；逾期履行的，每逾期一日按未履行金额的【】%支付违约金。",
    "发票与税务条款": "建议补充：收款方应在收到付款前/后【】个工作日内开具合法有效的【增值税专用/普通】发票，税率为【】%，因发票不合规导致的损失由责任方承担。",
    "知识产权条款": "建议补充：项目成果及相关知识产权归属【甲方/乙方/双方约定】；未经权利方书面同意，另一方不得超出本合同目的复制、许可、转让或商业使用。",
    "保密条款": "建议补充：双方应对在合作中获悉的商业秘密、技术资料、客户信息及其他非公开信息承担保密义务，保密期限为合同终止后【】年。",
    "数据安全与隐私条款": "建议补充：涉及个人信息、客户数据或商业秘密时，接收方应采取不低于行业合理标准的安全措施，未经授权不得处理、披露或转委托处理相关数据。",
}


def review_contract(input_data: Dict[str, Any] | ContractReviewInput) -> Dict[str, Any]:
    """Review a contract with deterministic extraction and baseline risk rules.

    The Hermes Skill prompt can call an LLM for deeper legal/business reasoning.
    This function provides a local, dependency-light baseline that is suitable
    for unit tests, workflow wiring, and first-pass triage.
    """
    review_input = (
        input_data if isinstance(input_data, ContractReviewInput) else ContractReviewInput.from_dict(input_data)
    )
    text = _normalize_text(review_input.contract_text)
    contract_type = review_input.contract_type
    if contract_type == "其他":
        contract_type = _infer_contract_type(text)

    extracted_fields = _extract_fields(text, contract_type)
    key_clauses = _extract_key_clauses(text)
    key_clause_dicts = [asdict(clause) for clause in key_clauses]
    missing_clauses = _find_missing_clauses(contract_type, key_clauses)
    risks = _detect_risks(review_input, text, contract_type, extracted_fields, key_clauses, missing_clauses)
    overall_risk_level = _overall_risk_level(risks)
    optimization_suggestions = _build_optimization_suggestions(
        review_input,
        contract_type,
        key_clauses,
        missing_clauses,
        risks,
    )
    suggested_action_items = _build_action_items(overall_risk_level, risks, missing_clauses)
    summary = _build_summary(extracted_fields, contract_type, overall_risk_level)
    final_report_markdown = _build_report_markdown(
        review_input=review_input,
        summary=summary,
        key_clauses=key_clause_dicts,
        risks=[asdict(risk) for risk in risks],
        missing_clauses=[asdict(item) for item in missing_clauses],
        optimization_suggestions=optimization_suggestions,
        suggested_action_items=suggested_action_items,
    )

    result = ContractReviewResult(
        summary=summary,
        extracted_fields=extracted_fields,
        key_clauses=key_clause_dicts,
        risks=[asdict(risk) for risk in risks],
        missing_clauses=[asdict(item) for item in missing_clauses],
        optimization_suggestions=optimization_suggestions,
        suggested_action_items=suggested_action_items,
        final_report_markdown=final_report_markdown,
        review_metadata={
            "skill_name": "contract_review_skill",
            "skill_version": SKILL_VERSION,
            "review_perspective": review_input.review_perspective,
            "review_depth": review_input.review_depth,
            "industry": review_input.industry or "",
            "extension_points": [
                "enterprise_template_comparison",
                "legal_rag_retrieval",
                "industry_rule_config",
                "customer_policy_rules",
                "multilingual_review",
                "word_pdf_parsing",
                "redline_revision",
                "risk_statistics",
                "approval_workflow",
                "crm_erp_oa_integration",
            ],
        },
    )
    return asdict(result)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _infer_contract_type(text: str) -> str:
    title = _extract_contract_name(text)
    haystack = f"{title}\n{text[:1000]}"
    if any(keyword in haystack for keyword in ("采购", "供应商", "采购订单")):
        return "采购合同"
    if any(keyword in haystack for keyword in ("销售", "买卖", "客户购买")):
        return "销售合同"
    if any(keyword in haystack for keyword in ("服务", "委托", "SLA", "运维")):
        return "服务合同"
    if any(keyword in haystack for keyword in ("租赁", "出租", "承租")):
        return "租赁合同"
    if any(keyword in haystack for keyword in ("劳动", "员工", "用人单位")):
        return "劳动合同"
    if any(keyword in haystack for keyword in ("保密协议", "保密合同", "NDA")):
        return "保密协议"
    return "其他"


def _extract_fields(text: str, contract_type: str) -> Dict[str, Any]:
    party_a = _extract_party(text, "甲方")
    party_b = _extract_party(text, "乙方")
    other_parties = _extract_other_parties(text)
    amount = _first_match(
        text,
        [
            r"(?:合同总价|合同金额|合同价款|总价|服务费|租金)[为：:\s]*([^。\n；;]{1,80})",
            r"((?:人民币|RMB|USD|EUR|CNY|¥|\$)\s*[0-9一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟万亿,，.]+[^。\n；;]{0,30})",
        ],
    )
    payment_texts = _find_sentences(text, ["付款", "支付", "结算", "预付款", "尾款", "回款"])
    subject_text = _first_sentence(text, ["标的", "采购内容", "销售产品", "服务内容", "租赁物", "项目内容", "工作内容"])

    return {
        "contract_name": _extract_contract_name(text),
        "contract_number": _first_match(text, [r"(?:合同编号|协议编号|编号)[：:\s]*([A-Za-z0-9_\-（）()第号]+)"]) or "未在合同中明确约定",
        "contract_type": contract_type,
        "party_a": party_a,
        "party_b": party_b,
        "other_parties": other_parties,
        "parties": [p for p in [party_a, party_b, *other_parties] if p and p != "未在合同中明确约定"],
        "party_a_credit_code": _extract_credit_code_near(text, "甲方"),
        "party_b_credit_code": _extract_credit_code_near(text, "乙方"),
        "signing_date": _extract_date_near(text, ["签署日期", "签订日期", "签约日期"]),
        "effective_date": _extract_date_near(text, ["生效日期", "生效"]),
        "end_date": _extract_date_near(text, ["终止日期", "截止日期", "有效期至", "租期至"]),
        "amount": amount or "未在合同中明确约定",
        "currency": _detect_currency(text, amount or ""),
        "payment_method": _summarize_sentences(payment_texts, fallback="未在合同中明确约定"),
        "payment_nodes": payment_texts[:5],
        "performance_period": _first_match(text, [r"(?:履约期限|服务期限|交付期限|租期)[为：:\s]*([^。\n；;]{1,80})"]) or "未在合同中明确约定",
        "performance_location": _first_match(text, [r"(?:履约地点|交付地点|服务地点|租赁物所在地)[为：:\s]*([^。\n；;]{1,80})"]) or "未在合同中明确约定",
        "subject_matter": subject_text or "未在合同中明确约定",
        "contacts": _extract_contacts(text),
        "attachments": _find_sentences(text, ["附件", "补充协议", "技术规格书", "报价单"])[:5],
    }


def _extract_contract_name(text: str) -> str:
    for line in text.splitlines()[:12]:
        cleaned = line.strip(" #　\t")
        if 4 <= len(cleaned) <= 80 and any(token in cleaned for token in ("合同", "协议")):
            return cleaned
    match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9（）()《》]{2,40}(?:合同|协议))", text[:500])
    return match.group(1) if match else "未在合同中明确约定"


def _extract_party(text: str, label: str) -> str:
    patterns = [
        rf"{label}(?:（[^）]*）|\([^)]*\))?[：:\s]*([^\n；;，,]+)",
        rf"{label}名称[：:\s]*([^\n；;，,]+)",
    ]
    value = _first_match(text, patterns)
    if not value:
        return "未在合同中明确约定"
    value = re.sub(r"(统一社会信用代码|地址|联系人|法定代表人).*$", "", value).strip()
    return value[:80] or "未在合同中明确约定"


def _extract_other_parties(text: str) -> List[str]:
    parties = []
    for label in ("丙方", "丁方"):
        value = _extract_party(text, label)
        if value != "未在合同中明确约定":
            parties.append(value)
    return parties


def _extract_credit_code_near(text: str, label: str) -> str:
    pattern = rf"{label}[\s\S]{{0,120}}?(?:统一社会信用代码|信用代码)[：:\s]*([A-Z0-9]{{15,18}})"
    match = re.search(pattern, text)
    return match.group(1) if match else "未在合同中明确约定"


def _extract_date_near(text: str, labels: Iterable[str]) -> str:
    date_pattern = r"([0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日|[0-9]{4}[-/.][0-9]{1,2}[-/.][0-9]{1,2})"
    for label in labels:
        match = re.search(rf"{re.escape(label)}[为：:\s自起]*(?:之日)?[\s\S]{{0,20}}?{date_pattern}", text)
        if match:
            return match.group(1)
    return "未在合同中明确约定"


def _detect_currency(text: str, amount: str) -> str:
    haystack = f"{amount}\n{text[:1000]}"
    if any(token in haystack for token in ("美元", "USD", "$")):
        return "USD"
    if any(token in haystack for token in ("欧元", "EUR")):
        return "EUR"
    if any(token in haystack for token in ("人民币", "CNY", "RMB", "¥", "元")):
        return "CNY"
    return "未在合同中明确约定"


def _extract_contacts(text: str) -> List[Dict[str, str]]:
    contacts = []
    phone_matches = re.findall(r"(?:电话|手机|联系方式)[：:\s]*([0-9+\-\s]{7,20})", text)
    email_matches = re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)
    for phone in phone_matches[:5]:
        contacts.append({"type": "phone", "value": phone.strip()})
    for email in email_matches[:5]:
        contacts.append({"type": "email", "value": email.strip()})
    return contacts


def _extract_key_clauses(text: str) -> List[ClauseMatch]:
    clauses = []
    for clause_key, title, keywords in CLAUSE_DEFINITIONS:
        matches = _find_sentences(text, keywords)
        original_text = _summarize_sentences(matches[:2], fallback="未在合同中明确约定")
        clauses.append(
            ClauseMatch(
                clause_type=clause_key,
                title=title,
                present=bool(matches),
                original_text=original_text,
                confidence="high" if matches else "low",
                normalized_summary=_normalize_clause_summary(title, original_text, bool(matches)),
            )
        )
    return clauses


def _normalize_clause_summary(title: str, original_text: str, present: bool) -> str:
    if not present:
        return f"{title}未在合同中明确约定。"
    return f"已识别到{title}：{original_text[:120]}"


def _find_missing_clauses(contract_type: str, clauses: List[ClauseMatch]) -> List[MissingClause]:
    present_by_title = {clause.title: clause.present for clause in clauses}
    missing = []
    required = REQUIRED_BY_TYPE.get(contract_type, REQUIRED_BY_TYPE["其他"])
    for title in required:
        if not present_by_title.get(title):
            importance = "high" if title in {"争议解决条款", "违约责任条款", "验收条款"} else "medium"
            missing.append(
                MissingClause(
                    clause_type=title,
                    importance=importance,
                    reason=f"{contract_type}通常需要明确{title}，当前文本未识别到可执行约定。",
                    suggested_clause=MISSING_CLAUSE_SUGGESTIONS.get(
                        title,
                        f"建议补充{title}，明确适用条件、责任主体、操作流程和违约后果。",
                    ),
                )
            )
    if "不可抗力条款" not in present_by_title or not present_by_title.get("不可抗力条款"):
        missing.append(
            MissingClause(
                clause_type="不可抗力条款",
                importance="medium",
                reason="不可抗力条款有助于处理极端事件下的履约障碍和责任免除边界。",
                suggested_clause=MISSING_CLAUSE_SUGGESTIONS["不可抗力条款"],
            )
        )
    return missing


def _detect_risks(
    review_input: ContractReviewInput,
    text: str,
    contract_type: str,
    extracted_fields: Dict[str, Any],
    clauses: List[ClauseMatch],
    missing_clauses: List[MissingClause],
) -> List[RiskFinding]:
    risks: List[RiskFinding] = []

    def add(
        title: str,
        level: str,
        risk_type: str,
        related_clause: str,
        original_text: str,
        description: str,
        impact: str,
        suggestion: str,
        revision: str,
    ) -> None:
        risks.append(
            RiskFinding(
                risk_id=f"CR-{len(risks) + 1:03d}",
                risk_title=title,
                risk_level=level,
                risk_type=risk_type,
                related_clause=related_clause,
                original_text=original_text or "未在合同中明确约定",
                risk_description=description,
                business_impact=impact,
                suggestion=suggestion,
                suggested_revision=revision,
            )
        )

    if extracted_fields["party_a"] == "未在合同中明确约定" or extracted_fields["party_b"] == "未在合同中明确约定":
        add(
            "合同主体信息不完整",
            "high",
            "主体风险",
            "合同主体条款",
            _clause_text(clauses, "合同主体条款"),
            "合同未完整列明甲乙双方名称，无法稳定确认权利义务主体。",
            "可能影响签署效力判断、开票收款、诉讼或仲裁主体确认。",
            "补充完整主体名称、统一社会信用代码、注册地址、法定代表人或授权代表。",
            "甲方：【完整公司名称】，统一社会信用代码：【】，地址：【】，法定代表人/授权代表：【】；乙方：【完整公司名称】，统一社会信用代码：【】，地址：【】，法定代表人/授权代表：【】。",
        )
    elif (
        extracted_fields["party_a_credit_code"] == "未在合同中明确约定"
        or extracted_fields["party_b_credit_code"] == "未在合同中明确约定"
    ):
        add(
            "主体识别信息不足",
            "medium",
            "主体风险",
            "合同主体条款",
            _clause_text(clauses, "合同主体条款"),
            "合同已列明主体名称，但未完整识别到统一社会信用代码。",
            "同名主体、分支机构或关联公司较多时，可能影响主体识别和后续追责。",
            "补充统一社会信用代码、注册地址和授权签署人信息。",
            "双方主体信息应补充为：公司名称：【】；统一社会信用代码：【】；注册地址：【】；授权签署人：【】。",
        )

    amount_text = extracted_fields.get("amount", "")
    payment_text = extracted_fields.get("payment_method", "")
    if amount_text == "未在合同中明确约定":
        add(
            "合同金额未明确",
            "high",
            "金额风险",
            "价款与付款条款",
            _clause_text(clauses, "价款与付款条款"),
            "合同未明确总价、计费标准或租金/服务费计算方式。",
            "可能导致付款金额、预算审批、开票金额和争议金额无法确定。",
            "明确总价、币种、含税口径、计费标准和价格调整条件。",
            "本合同总金额为人民币【】元（大写：【】），该金额【含/不含】增值税，除本合同另有约定外，乙方不得另行收取其他费用。",
        )
    elif payment_text == "未在合同中明确约定":
        add(
            "付款安排缺失",
            "medium",
            "金额风险",
            "价款与付款条款",
            amount_text,
            "合同虽约定金额，但未明确付款节点、付款条件或付款期限。",
            "可能造成付款审批、回款计划和违约认定困难。",
            "按里程碑、交付验收或固定日期明确付款比例和付款条件。",
            "甲方应在【验收合格/收到合规发票】后【】个工作日内支付【】%；剩余【】%在【】后支付。",
        )

    if _contains_uncertain_payment(payment_text):
        add(
            "付款条件存在不确定表述",
            "medium",
            "金额风险",
            "价款与付款条款",
            payment_text,
            "付款条款包含“另行协商、视情况、适时”等不确定表述，缺少可执行节点。",
            "容易造成回款延迟、付款审批被退回或双方对付款义务产生争议。",
            "将不确定表述替换为明确的日期、比例、条件和单据要求。",
            "甲方应在收到乙方提交的合规发票及验收确认单后【】个工作日内支付对应款项，付款比例和金额以本合同附件【】为准。",
        )

    if _has_high_prepayment(text):
        add(
            "预付款比例偏高",
            "medium",
            "商务风险",
            "价款与付款条款",
            _first_sentence(text, ["预付款"]) or payment_text,
            "合同约定较高比例预付款，但未同步识别到充分的交付、担保或退款保障。",
            "一旦对方履约能力不足，可能增加资金占用和追偿难度。",
            "降低预付款比例，或增加履约保证金、银行保函、分阶段交付验收和退款条件。",
            "预付款比例调整为【】%；乙方未按期交付或验收不合格的，应在甲方通知后【】日内退还相应预付款并承担违约责任。",
        )

    if _missing(clauses, "交付条款") and contract_type in {"采购合同", "销售合同", "服务合同"}:
        add(
            "交付安排不清晰",
            "medium",
            "履约风险",
            "交付条款",
            "未在合同中明确约定",
            "合同未明确交付时间、地点、方式或交付物清单。",
            "可能导致履约边界、延期责任和验收起算时间不清。",
            "补充交付清单、交付标准、交付时间和交付证明。",
            "乙方应于【】前在【】向甲方交付【交付物清单】，并提交【交付证明/签收单/系统记录】作为交付完成依据。",
        )

    if _missing(clauses, "验收条款") and contract_type in {"采购合同", "销售合同", "服务合同"}:
        add(
            "验收标准缺失",
            "high",
            "履约风险",
            "验收条款",
            "未在合同中明确约定",
            "合同未明确验收标准、验收期限和异议处理机制。",
            "可能导致付款条件、质量责任和项目完成标准无法落地。",
            "补充客观验收标准、验收流程、整改期限和逾期视为验收规则。",
            MISSING_CLAUSE_SUGGESTIONS["验收条款"],
        )

    if _missing(clauses, "违约责任条款"):
        add(
            "违约责任缺失",
            "high",
            "违约风险",
            "违约责任条款",
            "未在合同中明确约定",
            "合同未明确逾期交付、逾期付款、质量不合格、保密违约等责任后果。",
            "违约发生时缺少可直接适用的责任基础，追偿和谈判成本较高。",
            "按主要义务分别约定违约金、赔偿范围、整改期限和解除权。",
            MISSING_CLAUSE_SUGGESTIONS["违约责任条款"],
        )

    if _missing(clauses, "争议解决条款"):
        add(
            "争议解决条款缺失",
            "high",
            "法务风险",
            "争议解决条款",
            "未在合同中明确约定",
            "合同未约定争议解决方式、管辖法院或仲裁机构。",
            "争议发生后可能增加管辖争议、维权周期和成本。",
            "补充明确的协商、诉讼或仲裁路径，避免同时约定法院和仲裁导致冲突。",
            MISSING_CLAUSE_SUGGESTIONS["争议解决条款"],
        )
    elif _missing(clauses, "管辖法院或仲裁机构"):
        add(
            "管辖机构不明确",
            "medium",
            "法务风险",
            "管辖法院或仲裁机构",
            _clause_text(clauses, "争议解决条款"),
            "合同提到争议解决，但未明确具体管辖法院或仲裁机构。",
            "可能导致维权地点和程序不确定。",
            "明确选择诉讼或仲裁，并写清具体法院连接点或仲裁委员会名称。",
            MISSING_CLAUSE_SUGGESTIONS["管辖法院或仲裁机构"],
        )

    if _missing(clauses, "合同解除与终止条款"):
        add(
            "解除与终止条件不清",
            "medium",
            "法务风险",
            "合同解除与终止条款",
            "未在合同中明确约定",
            "合同未明确提前解除、严重违约解除、终止后的结算和资料返还。",
            "业务无法退出低质量或高风险合作，终止后责任边界不清。",
            "补充解除触发条件、通知方式、结算规则和终止后义务。",
            MISSING_CLAUSE_SUGGESTIONS["合同解除与终止条款"],
        )

    if contract_type == "服务合同" and _missing(clauses, "知识产权条款"):
        add(
            "服务成果知识产权归属不明确",
            "medium",
            "知识产权风险",
            "知识产权条款",
            "未在合同中明确约定",
            "服务合同未明确交付成果、源代码、文档或创作内容的权利归属和使用范围。",
            "可能影响成果复用、二次开发、商业发布和侵权责任承担。",
            "明确背景知识产权、项目成果归属、授权范围、第三方素材责任和侵权处理。",
            MISSING_CLAUSE_SUGGESTIONS["知识产权条款"],
        )

    if contract_type == "保密协议" and _missing(clauses, "保密条款"):
        add(
            "保密义务核心条款缺失",
            "high",
            "保密风险",
            "保密条款",
            "未在合同中明确约定",
            "保密协议未明确保密信息范围、保密期限、例外情形和违约后果。",
            "商业秘密或敏感资料泄露后，举证和追责难度较高。",
            "补充保密范围、例外、接触人员控制、返还销毁和违约责任。",
            MISSING_CLAUSE_SUGGESTIONS["保密条款"],
        )

    if _mentions_sensitive_data(text) and _missing(clauses, "数据安全与隐私条款"):
        add(
            "涉及数据但缺少保护约定",
            "high",
            "数据安全风险",
            "数据安全与隐私条款",
            _first_sentence(text, ["数据", "个人信息", "客户信息", "商业秘密"]) or "未在合同中明确约定",
            "合同内容涉及数据、个人信息、客户信息或商业秘密，但未明确数据保护义务。",
            "可能引发客户投诉、监管合规风险、数据泄露赔偿或业务声誉损失。",
            "补充数据处理目的、范围、安全措施、转委托限制、泄露通知和返还删除机制。",
            MISSING_CLAUSE_SUGGESTIONS["数据安全与隐私条款"],
        )

    if contract_type in {"采购合同", "销售合同", "服务合同"} and _missing(clauses, "发票与税务条款"):
        add(
            "发票与税务安排不明确",
            "medium",
            "税务风险",
            "发票与税务条款",
            "未在合同中明确约定",
            "合同未明确发票类型、税率、开票时间或发票不合规责任。",
            "可能影响付款审批、税务抵扣和财务入账。",
            "补充发票类型、税率、开票节点、收票信息和不合规处理。",
            MISSING_CLAUSE_SUGGESTIONS["发票与税务条款"],
        )

    if _missing(clauses, "不可抗力条款"):
        add(
            "不可抗力处理机制缺失",
            "medium",
            "缺失条款风险",
            "不可抗力条款",
            "未在合同中明确约定",
            "合同未约定不可抗力的通知、证明、减损和责任免除机制。",
            "极端事件发生时，双方对延期、解除和责任免除可能产生争议。",
            "补充不可抗力定义、通知期限、证明材料、减损义务和后续处理。",
            MISSING_CLAUSE_SUGGESTIONS["不可抗力条款"],
        )

    ambiguous = _find_ambiguous_expressions(text, deep=review_input.review_depth == "deep")
    for expression_text in ambiguous[:5]:
        add(
            "存在不可执行或边界不清表述",
            "low",
            "表述风险",
            "相关表述",
            expression_text,
            "合同中存在“另行协商、及时、合理、相关费用”等表述，缺少明确判断标准。",
            "履行中容易形成理解差异，影响验收、付款、追责或内部审批。",
            "将模糊表达替换为明确期限、金额、比例、标准、责任主体和操作流程。",
            "将该表述修改为：【责任主体】应在【明确期限】内按照【明确标准】完成【具体义务】；未完成的，应承担【具体责任】。",
        )

    if review_input.company_policy:
        add(
            "需结合企业内部审核规则复核",
            "info",
            "商务风险",
            "企业内部规则",
            review_input.company_policy[:180],
            "用户提供了企业内部审核规则，本地规则引擎只能做文本提示，未执行完整制度比对。",
            "如合同偏离内部红线，可能影响审批通过或带来授权合规风险。",
            "将 company_policy 接入规则引擎或 RAG，并逐条输出命中情况。",
            "根据企业制度补充或调整条款：【引用内部规则编号/红线项】。",
        )

    if review_input.reference_template:
        add(
            "建议进行标准模板差异比对",
            "info",
            "缺失条款风险",
            "标准模板比对",
            "用户已提供 reference_template",
            "用户提供了标准合同模板，当前基础版本仅保留扩展入口，未输出逐条差异。",
            "模板偏离项可能包含业务审批红线或法务标准条款缺失。",
            "后续接入模板比对模块，输出缺失、弱化、冲突和新增条款清单。",
            "以企业标准模板为基准补齐缺失条款，并对偏离条款标注审批理由。",
        )

    return risks


def _build_summary(extracted_fields: Dict[str, Any], contract_type: str, overall_risk_level: str) -> Dict[str, Any]:
    conclusion_by_level = {
        "high": "识别到高风险问题，建议修改后再进入签署或审批流程。",
        "medium": "识别到中等风险问题，建议补充关键条款并复核后继续推进。",
        "low": "主要风险较低，建议优化表述后继续推进。",
        "info": "未识别到明确风险，仍建议结合业务背景和法务意见复核。",
    }
    return {
        "contract_name": extracted_fields.get("contract_name", "未在合同中明确约定"),
        "contract_type": contract_type,
        "parties": extracted_fields.get("parties", []),
        "amount": extracted_fields.get("amount", "未在合同中明确约定"),
        "currency": extracted_fields.get("currency", "未在合同中明确约定"),
        "effective_date": extracted_fields.get("effective_date", "未在合同中明确约定"),
        "end_date": extracted_fields.get("end_date", "未在合同中明确约定"),
        "overall_risk_level": overall_risk_level,
        "review_conclusion": conclusion_by_level[overall_risk_level],
    }


def _build_optimization_suggestions(
    review_input: ContractReviewInput,
    contract_type: str,
    clauses: List[ClauseMatch],
    missing_clauses: List[MissingClause],
    risks: List[RiskFinding],
) -> List[Dict[str, str]]:
    suggestions: List[Dict[str, str]] = []
    focus_items = TYPE_FOCUS.get(contract_type, [])
    if focus_items:
        suggestions.append(
            {
                "title": f"{contract_type}专项审核重点",
                "description": "建议围绕以下事项逐项复核：" + "、".join(focus_items) + "。",
            }
        )
    if missing_clauses:
        suggestions.append(
            {
                "title": "补齐关键缺失条款",
                "description": "优先补齐：" + "、".join(item.clause_type for item in missing_clauses[:6]) + "。",
            }
        )
    if any(risk.risk_level == "high" for risk in risks):
        suggestions.append(
            {
                "title": "高风险问题签署前闭环",
                "description": "高风险问题应形成修订稿、业务确认记录和法务复核意见后再签署。",
            }
        )
    if review_input.extra_requirements:
        suggestions.append(
            {
                "title": "用户额外要求",
                "description": f"请在人工复核时额外关注：{review_input.extra_requirements}",
            }
        )
    if not suggestions:
        suggestions.append(
            {
                "title": "表述一致性优化",
                "description": "建议统一合同主体简称、金额币种、日期格式和附件编号，降低执行歧义。",
            }
        )
    return suggestions


def _build_action_items(overall_risk_level: str, risks: List[RiskFinding], missing_clauses: List[MissingClause]) -> List[str]:
    actions = []
    high_count = sum(1 for risk in risks if risk.risk_level == "high")
    medium_count = sum(1 for risk in risks if risk.risk_level == "medium")
    if high_count:
        actions.append(f"优先处理 {high_count} 项高风险问题，修改完成前不建议直接签署。")
    if medium_count:
        actions.append(f"安排业务负责人和法务复核 {medium_count} 项中风险问题。")
    if missing_clauses:
        actions.append("根据缺失条款清单补充合同文本，并确认是否需要客户/供应商重新确认。")
    if overall_risk_level in {"low", "info"}:
        actions.append("在签署前完成主体资质、授权签署人、附件版本和用印流程核对。")
    actions.append("重大合同或非常规条款应提交专业法务或外部律师复核。")
    return actions


def _build_report_markdown(
    review_input: ContractReviewInput,
    summary: Dict[str, Any],
    key_clauses: List[Dict[str, Any]],
    risks: List[Dict[str, Any]],
    missing_clauses: List[Dict[str, Any]],
    optimization_suggestions: List[Dict[str, Any]],
    suggested_action_items: List[str],
) -> str:
    high_risks = [risk for risk in risks if risk["risk_level"] == "high"]
    other_risks = [risk for risk in risks if risk["risk_level"] != "high"]
    present_clauses = [clause for clause in key_clauses if clause["present"]]

    lines = [
        "# 合同审核报告",
        "",
        "## 1. 合同概览",
        f"- 合同名称：{summary['contract_name']}",
        f"- 合同类型：{summary['contract_type']}",
        f"- 审核视角：{review_input.review_perspective}",
        f"- 合同主体：{_join_or_unknown(summary.get('parties', []))}",
        f"- 合同金额：{summary['amount']}",
        f"- 币种：{summary['currency']}",
        f"- 生效日期：{summary['effective_date']}",
        f"- 终止日期：{summary['end_date']}",
        "",
        "## 2. 总体风险结论",
        f"- 总体风险等级：{summary['overall_risk_level']}",
        f"- 审核结论：{summary['review_conclusion']}",
        "",
        "## 3. 关键条款摘要",
    ]
    if present_clauses:
        for clause in present_clauses[:12]:
            lines.append(f"- {clause['title']}：{clause['original_text'][:160]}")
    else:
        lines.append("- 未识别到完整关键条款。")

    lines.extend(["", "## 4. 高风险问题"])
    if high_risks:
        for risk in high_risks:
            lines.extend(_risk_markdown_lines(risk))
    else:
        lines.append("- 未识别到高风险问题。")

    lines.extend(["", "## 5. 中低风险问题"])
    if other_risks:
        for risk in other_risks:
            lines.extend(_risk_markdown_lines(risk))
    else:
        lines.append("- 未识别到中低风险问题。")

    lines.extend(["", "## 6. 缺失条款"])
    if missing_clauses:
        for item in missing_clauses:
            lines.append(f"- {item['clause_type']}（{item['importance']}）：{item['reason']}")
    else:
        lines.append("- 未识别到必备条款缺失。")

    lines.extend(["", "## 7. 修改建议"])
    if optimization_suggestions:
        for suggestion in optimization_suggestions:
            lines.append(f"- {suggestion['title']}：{suggestion['description']}")
    else:
        lines.append("- 建议统一合同术语、金额、日期、附件编号和签署信息。")

    lines.extend(["", "## 8. 建议下一步动作"])
    for action in suggested_action_items:
        lines.append(f"- {action}")

    lines.extend(
        [
            "",
            "## 9. 免责声明",
            "本审核结果仅作为合同初审和业务辅助参考，不构成正式法律意见，重大合同应由专业法务或律师复核。",
        ]
    )
    return "\n".join(lines)


def _risk_markdown_lines(risk: Dict[str, Any]) -> List[str]:
    return [
        f"- [{risk['risk_id']}] {risk['risk_title']}（{risk['risk_level']} / {risk['risk_type']}）",
        f"  - 相关条款：{risk['related_clause']}",
        f"  - 原文依据：{risk['original_text'][:180]}",
        f"  - 影响：{risk['business_impact']}",
        f"  - 建议：{risk['suggestion']}",
    ]


def _overall_risk_level(risks: List[RiskFinding]) -> str:
    if not risks:
        return "info"
    return max((risk.risk_level for risk in risks), key=lambda level: RISK_LEVEL_ORDER[level])


def _missing(clauses: List[ClauseMatch], title: str) -> bool:
    for clause in clauses:
        if clause.title == title:
            return not clause.present
    return True


def _clause_text(clauses: List[ClauseMatch], title: str) -> str:
    for clause in clauses:
        if clause.title == title:
            return clause.original_text
    return "未在合同中明确约定"


def _first_match(text: str, patterns: Iterable[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" ：:，,；;\n\t")
    return ""


def _split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[。；;!?！？])|\n+", text)
    sentences = []
    for chunk in chunks:
        cleaned = chunk.strip(" \t\n")
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _find_sentences(text: str, keywords: Iterable[str]) -> List[str]:
    return [
        sentence
        for sentence in _split_sentences(text)
        if any(keyword.lower() in sentence.lower() for keyword in keywords)
    ]


def _first_sentence(text: str, keywords: Iterable[str]) -> str:
    matches = _find_sentences(text, keywords)
    return matches[0] if matches else ""


def _summarize_sentences(sentences: List[str], fallback: str) -> str:
    if not sentences:
        return fallback
    summary = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
    return summary[:500]


def _contains_uncertain_payment(payment_text: str) -> bool:
    if not payment_text or payment_text == "未在合同中明确约定":
        return False
    return any(token in payment_text for token in ("另行协商", "视情况", "适时", "待定", "原则上", "尽快"))


def _has_high_prepayment(text: str) -> bool:
    for sentence in _find_sentences(text, ["预付款"]):
        for raw in re.findall(r"([0-9]{1,3})\s*%", sentence):
            if int(raw) >= 50:
                return True
        if any(token in sentence for token in ("五成", "六成", "七成", "八成", "九成", "全款")):
            return True
    return False


def _mentions_sensitive_data(text: str) -> bool:
    return any(token in text for token in ("个人信息", "客户数据", "客户信息", "敏感数据", "商业秘密", "数据处理"))


def _find_ambiguous_expressions(text: str, deep: bool) -> List[str]:
    keywords = ["另行协商", "适时", "及时", "尽快", "合理", "相关费用", "视情况", "原则上", "待定"]
    if deep:
        keywords.extend(["等", "必要时", "一般", "适当", "尽量"])
    results = []
    for sentence in _split_sentences(text):
        if any(keyword in sentence for keyword in keywords):
            results.append(sentence)
    return results


def _join_or_unknown(values: List[str]) -> str:
    return "、".join(values) if values else "未在合同中明确约定"


if __name__ == "__main__":
    import json
    import sys

    payload = json.load(sys.stdin)
    json.dump(review_contract(payload), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
