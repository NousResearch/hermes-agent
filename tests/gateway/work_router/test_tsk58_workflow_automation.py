from gateway.work_router.lifecycle import (
    WorkflowAssessment, WorkflowNextAction, OwnerWorkflowResult,
    assessment_from_owner_result, parse_owner_workflow_response,
    lifecycle_route_action, route_owner_workflow_response,
    should_activate_owner_lifecycle, decide_next_action, decide_pm_review_outcome,
    parse_pm_review_response, pm_review_ingress_action,
)


def test_owner_intermediate_output_does_not_trigger_result_report():
    assessment = WorkflowAssessment(
        internal_work_remaining=True,
        advisor_required=False,
        customer_information_required=False,
        commercial_decision_required=False,
        pm_approved=False,
        sinclair_approved=False,
    )

    result = decide_next_action(assessment)

    assert result.action is WorkflowNextAction.CONTINUE_AUTOMATION
    assert result.publish_result_report is False
    assert result.approval_required is False


def test_advisor_work_continues_without_sinclair_approval():
    assessment = WorkflowAssessment(
        internal_work_remaining=True,
        advisor_required=True,
        customer_information_required=False,
        commercial_decision_required=False,
        pm_approved=False,
        sinclair_approved=False,
    )

    result = decide_next_action(assessment)

    assert result.action is WorkflowNextAction.CALL_ADVISOR
    assert result.publish_result_report is False
    assert result.approval_required is False


def test_customer_questions_are_prepared_before_decision_gate():
    assessment = WorkflowAssessment(
        internal_work_remaining=False,
        advisor_required=False,
        customer_information_required=True,
        commercial_decision_required=False,
        customer_questions_ready=False,
        pm_approved=False,
        sinclair_approved=False,
    )

    result = decide_next_action(assessment)

    assert result.action is WorkflowNextAction.PREPARE_CUSTOMER_QUESTIONS
    assert result.publish_result_report is False
    assert result.approval_required is False


def test_real_decision_gate_publishes_decision_required_only_after_pm_review():
    assessment = WorkflowAssessment(
        internal_work_remaining=False,
        advisor_required=False,
        customer_information_required=False,
        commercial_decision_required=True,
        customer_questions_ready=True,
        pm_approved=True,
        sinclair_approved=False,
    )

    result = decide_next_action(assessment)

    assert result.action is WorkflowNextAction.WAIT_SINCLAIR_DECISION
    assert result.publish_result_report is True
    assert result.result_report_kind == "decision_required"
    assert result.approval_required is True


def test_sinclair_approval_allows_final_result():
    assessment = WorkflowAssessment(
        internal_work_remaining=False,
        advisor_required=False,
        customer_information_required=False,
        commercial_decision_required=False,
        customer_questions_ready=True,
        pm_approved=True,
        sinclair_approved=True,
    )

    result = decide_next_action(assessment)

    assert result.action is WorkflowNextAction.FINALIZE
    assert result.publish_result_report is True
    assert result.result_report_kind == "final_result"
    assert result.approval_required is False


def test_intermediate_owner_output_is_not_publishable():
    assessment = WorkflowAssessment(
        internal_work_remaining=True,
        pm_approved=False,
        sinclair_approved=False,
    )

    result = decide_next_action(assessment)

    assert result.allows_outbox_publication is False


def test_decision_gate_is_publishable_after_pm_review():
    assessment = WorkflowAssessment(
        commercial_decision_required=True,
        customer_questions_ready=True,
        pm_approved=True,
        sinclair_approved=False,
    )

    result = decide_next_action(assessment)

    assert result.allows_outbox_publication is True
    assert result.result_report_kind == "decision_required"


def test_final_result_is_publishable_after_final_approval():
    assessment = WorkflowAssessment(
        customer_questions_ready=True,
        pm_approved=True,
        sinclair_approved=True,
    )

    result = decide_next_action(assessment)

    assert result.allows_outbox_publication is True
    assert result.result_report_kind == "final_result"


def test_intermediate_work_has_no_result_report_response_kind():
    result = decide_next_action(
        WorkflowAssessment(
            internal_work_remaining=True,
            pm_approved=False,
        )
    )

    assert result.result_response_kind is None


def test_decision_gate_uses_explicit_result_response_kind():
    result = decide_next_action(
        WorkflowAssessment(
            commercial_decision_required=True,
            customer_questions_ready=True,
            pm_approved=True,
            sinclair_approved=False,
        )
    )

    assert result.result_response_kind == "decision_required"


def test_final_result_uses_explicit_result_response_kind():
    result = decide_next_action(
        WorkflowAssessment(
            pm_approved=True,
            sinclair_approved=True,
        )
    )

    assert result.result_response_kind == "final_result"


def test_owner_result_converts_internal_followups_to_automation():
    result = OwnerWorkflowResult(
        confirmed_information=("기존 고객",),
        internal_followups=("기존 계약 조회",),
        customer_questions=(),
        advisor_required=False,
        commercial_decision_required=False,
        customer_ready=False,
    )

    assessment = assessment_from_owner_result(result)

    assert assessment.internal_work_remaining is True
    assert decide_next_action(assessment).action is WorkflowNextAction.CONTINUE_AUTOMATION


def test_owner_result_converts_advisor_requirement():
    result = OwnerWorkflowResult(
        confirmed_information=(),
        internal_followups=(),
        customer_questions=(),
        advisor_required=True,
        commercial_decision_required=False,
        customer_ready=False,
    )

    assessment = assessment_from_owner_result(result)

    assert assessment.advisor_required is True
    assert decide_next_action(assessment).action is WorkflowNextAction.CALL_ADVISOR


def test_customer_questions_are_not_treated_as_internal_completion():
    result = OwnerWorkflowResult(
        confirmed_information=("내부 확인 완료",),
        internal_followups=(),
        customer_questions=("추가 도입 대상이 1,000대가 맞습니까?",),
        advisor_required=False,
        commercial_decision_required=False,
        customer_ready=True,
    )

    assessment = assessment_from_owner_result(result)

    assert assessment.customer_information_required is True
    assert assessment.customer_questions_ready is True


def test_commercial_decision_requires_explicit_flag():
    result = OwnerWorkflowResult(
        confirmed_information=("견적 기초자료 확인 완료",),
        internal_followups=(),
        customer_questions=(),
        advisor_required=False,
        commercial_decision_required=True,
        customer_ready=True,
    )

    assessment = assessment_from_owner_result(result)

    assert assessment.commercial_decision_required is True


def test_owner_control_block_is_parsed_and_removed_from_visible_text():
    raw = """고객 전달용 견적 준비를 계속 진행하겠습니다.

<HERMES_WORKFLOW>
{
  "confirmed_information": ["기존 고객"],
  "internal_followups": ["기존 계약 조회"],
  "customer_questions": [],
  "advisor_required": false,
  "commercial_decision_required": false,
  "customer_ready": false
}
</HERMES_WORKFLOW>"""

    parsed = parse_owner_workflow_response(raw)

    assert parsed.visible_text == "고객 전달용 견적 준비를 계속 진행하겠습니다."
    assert parsed.workflow.internal_followups == ("기존 계약 조회",)
    assert parsed.workflow.confirmed_information == ("기존 고객",)


def test_owner_control_block_never_leaks_to_visible_text():
    raw = """최종 검토안을 준비했습니다.

<HERMES_WORKFLOW>
{
  "confirmed_information": ["검토 완료"],
  "internal_followups": [],
  "customer_questions": ["추가 도입 대상이 1,000대가 맞습니까?"],
  "advisor_required": false,
  "commercial_decision_required": false,
  "customer_ready": true
}
</HERMES_WORKFLOW>"""

    parsed = parse_owner_workflow_response(raw)

    assert "HERMES_WORKFLOW" not in parsed.visible_text
    assert "customer_questions" not in parsed.visible_text
    assert parsed.workflow.customer_ready is True


def test_missing_control_block_is_backward_compatible():
    raw = "기존 일반 Staff 응답입니다."

    parsed = parse_owner_workflow_response(raw)

    assert parsed.visible_text == raw
    assert parsed.workflow is None


def test_lifecycle_continue_maps_to_owner_continuation():
    decision = decide_next_action(
        WorkflowAssessment(internal_work_remaining=True)
    )
    route = lifecycle_route_action(decision, owner_profile="Hans")

    assert route.kind == "dispatch"
    assert route.target_profile == "Hans"
    assert route.response_kind == "workflow_continue"


def test_lifecycle_advisor_maps_to_existing_advisor_dispatch():
    decision = decide_next_action(
        WorkflowAssessment(advisor_required=True)
    )
    route = lifecycle_route_action(
        decision,
        owner_profile="Hans",
        advisor_profile="Tesla",
    )

    assert route.kind == "dispatch"
    assert route.target_profile == "Tesla"
    assert route.response_kind == "advisor_request"


def test_lifecycle_pm_review_maps_to_demian():
    decision = decide_next_action(
        WorkflowAssessment(
            customer_information_required=True,
            customer_questions_ready=True,
            pm_approved=False,
        )
    )
    route = lifecycle_route_action(decision, owner_profile="Hans")

    assert route.kind == "dispatch"
    assert route.target_profile == "Demian"
    assert route.response_kind == "pm_review"


def test_lifecycle_decision_gate_is_not_owner_dispatch():
    decision = decide_next_action(
        WorkflowAssessment(
            commercial_decision_required=True,
            pm_approved=True,
            sinclair_approved=False,
        )
    )
    route = lifecycle_route_action(decision, owner_profile="Hans")

    assert route.kind == "result_report"
    assert route.response_kind == "decision_required"


def test_lifecycle_final_result_maps_to_final_report():
    decision = decide_next_action(
        WorkflowAssessment(
            pm_approved=True,
            sinclair_approved=True,
        )
    )
    route = lifecycle_route_action(decision, owner_profile="Hans")

    assert route.kind == "result_report"
    assert route.response_kind == "final_result"


def test_plain_staff_response_does_not_activate_workflow_lifecycle():
    parsed = parse_owner_workflow_response("기존 일반 Hans 응답입니다.")

    assert parsed.workflow is None


def test_structured_owner_response_activates_workflow_lifecycle():
    raw = """내부 계약정보를 추가 확인하겠습니다.

<HERMES_WORKFLOW>
{
  "confirmed_information": ["기존 고객"],
  "internal_followups": ["계약 및 라이선스 조회"],
  "customer_questions": [],
  "advisor_required": false,
  "commercial_decision_required": false,
  "customer_ready": false
}
</HERMES_WORKFLOW>"""

    result = route_owner_workflow_response(
        raw,
        owner_profile="Hans",
    )

    assert result.visible_text == "내부 계약정보를 추가 확인하겠습니다."
    assert result.lifecycle_active is True
    assert result.route.kind == "dispatch"
    assert result.route.target_profile == "Hans"
    assert result.route.response_kind == "workflow_continue"


def test_plain_response_keeps_legacy_router_path():
    result = route_owner_workflow_response(
        "기존 일반 응답",
        owner_profile="Hans",
    )

    assert result.visible_text == "기존 일반 응답"
    assert result.lifecycle_active is False
    assert result.route is None


def test_lifecycle_activation_requires_current_owner():
    assert should_activate_owner_lifecycle(
        author_profile="Hans",
        owner_profile="Hans",
        text="<HERMES_WORKFLOW>{}</HERMES_WORKFLOW>",
    ) is True

    assert should_activate_owner_lifecycle(
        author_profile="Tesla",
        owner_profile="Hans",
        text="<HERMES_WORKFLOW>{}</HERMES_WORKFLOW>",
    ) is False


def test_lifecycle_activation_requires_control_block():
    assert should_activate_owner_lifecycle(
        author_profile="Hans",
        owner_profile="Hans",
        text="일반 Hans 업무 답변",
    ) is False


def test_human_message_never_activates_owner_lifecycle():
    assert should_activate_owner_lifecycle(
        author_profile=None,
        owner_profile="Hans",
        text="<HERMES_WORKFLOW>{}</HERMES_WORKFLOW>",
    ) is False


def test_pm_review_rework_returns_to_owner():
    result = decide_pm_review_outcome(
        pm_approved=False,
        rework_required=True,
        decision_required=False,
    )

    assert result.next_state == "OWNER_REWORK"
    assert result.target_profile == "OWNER"
    assert result.publish_result_report is False


def test_pm_review_decision_required_waits_for_sinclair():
    result = decide_pm_review_outcome(
        pm_approved=True,
        rework_required=False,
        decision_required=True,
    )

    assert result.next_state == "WAIT_SINCLAIR_DECISION"
    assert result.target_profile == "Sinclair"
    assert result.approval_required is True
    assert result.publish_result_report is True


def test_pm_review_approved_without_decision_finishes():
    result = decide_pm_review_outcome(
        pm_approved=True,
        rework_required=False,
        decision_required=False,
    )

    assert result.next_state == "FINAL_RESULT"
    assert result.target_profile is None
    assert result.approval_required is False
    assert result.publish_result_report is True


def test_pm_review_route_targets_demian():
    decision = decide_next_action(
        WorkflowAssessment(
            customer_information_required=True,
            customer_questions_ready=True,
            pm_approved=False,
        )
    )

    route = lifecycle_route_action(
        decision,
        owner_profile="Hans",
    )

    assert decision.action is WorkflowNextAction.PM_REVIEW
    assert route.kind == "dispatch"
    assert route.target_profile == "Demian"
    assert route.response_kind == "pm_review"


def test_pm_review_is_not_sinclair_result_publication():
    decision = decide_next_action(
        WorkflowAssessment(
            customer_information_required=True,
            customer_questions_ready=True,
            pm_approved=False,
        )
    )

    assert decision.publish_result_report is False
    assert decision.approval_required is False
    assert decision.result_response_kind is None


def test_pm_response_rework_contract():
    raw = """검토 결과 고객 환경 기준을 추가 확인해서 다시 작성해주세요.
<HERMES_PM_REVIEW>
{
  "pm_approved": false,
  "rework_required": true,
  "decision_required": false
}
</HERMES_PM_REVIEW>"""

    parsed = parse_pm_review_response(raw)

    assert "HERMES_PM_REVIEW" not in parsed.visible_text
    assert parsed.pm_approved is False
    assert parsed.rework_required is True
    assert parsed.decision_required is False


def test_pm_response_decision_gate_contract():
    raw = """기술 검토는 완료되었습니다. 가격 정책은 Sinclair 결정이 필요합니다.
<HERMES_PM_REVIEW>
{
  "pm_approved": true,
  "rework_required": false,
  "decision_required": true
}
</HERMES_PM_REVIEW>"""

    parsed = parse_pm_review_response(raw)
    outcome = decide_pm_review_outcome(
        pm_approved=parsed.pm_approved,
        rework_required=parsed.rework_required,
        decision_required=parsed.decision_required,
    )

    assert outcome.next_state == "WAIT_SINCLAIR_DECISION"
    assert outcome.approval_required is True


def test_pm_response_final_contract():
    raw = """검토 완료. 추가 의사결정 없이 최종 결과로 보고 가능합니다.
<HERMES_PM_REVIEW>
{
  "pm_approved": true,
  "rework_required": false,
  "decision_required": false
}
</HERMES_PM_REVIEW>"""

    parsed = parse_pm_review_response(raw)
    outcome = decide_pm_review_outcome(
        pm_approved=parsed.pm_approved,
        rework_required=parsed.rework_required,
        decision_required=parsed.decision_required,
    )

    assert outcome.next_state == "FINAL_RESULT"
    assert outcome.publish_result_report is True


def test_pm_control_block_never_leaks():
    raw = """최종 검토 결과입니다.
<HERMES_PM_REVIEW>
{"pm_approved":true,"rework_required":false,"decision_required":false}
</HERMES_PM_REVIEW>"""

    parsed = parse_pm_review_response(raw)

    assert "<HERMES_PM_REVIEW>" not in parsed.visible_text
    assert "pm_approved" not in parsed.visible_text


def test_pm_ingress_rework_routes_back_to_original_owner():
    parsed = parse_pm_review_response(
        """고객 환경 기준을 보완해서 다시 작성해주세요.
<HERMES_PM_REVIEW>
{
  "pm_approved": false,
  "rework_required": true,
  "decision_required": false
}
</HERMES_PM_REVIEW>"""
    )

    result = pm_review_ingress_action(
        parsed,
        owner_profile="Hans",
        review_generation=2,
    )

    assert result.next_state == "OWNER_REWORK"
    assert result.target_profile == "Hans"
    assert result.response_kind == "owner_rework"
    assert result.review_generation == 3
    assert result.approval_required is False
    assert result.publish_result_report is False


def test_pm_ingress_decision_gate_does_not_dispatch_fake_sinclair_profile():
    parsed = parse_pm_review_response(
        """가격 조건은 최종 결정이 필요합니다.
<HERMES_PM_REVIEW>
{
  "pm_approved": true,
  "rework_required": false,
  "decision_required": true
}
</HERMES_PM_REVIEW>"""
    )

    result = pm_review_ingress_action(
        parsed,
        owner_profile="Hans",
        review_generation=1,
    )

    assert result.next_state == "WAIT_SINCLAIR_DECISION"
    assert result.target_profile is None
    assert result.response_kind == "decision_required"
    assert result.review_generation == 1
    assert result.approval_required is True
    assert result.publish_result_report is True


def test_pm_ingress_final_result_uses_result_publication_path():
    parsed = parse_pm_review_response(
        """PM 검토 완료.
<HERMES_PM_REVIEW>
{
  "pm_approved": true,
  "rework_required": false,
  "decision_required": false
}
</HERMES_PM_REVIEW>"""
    )

    result = pm_review_ingress_action(
        parsed,
        owner_profile="Hans",
        review_generation=1,
    )

    assert result.next_state == "FINAL_RESULT"
    assert result.target_profile is None
    assert result.response_kind == "final_result"
    assert result.approval_required is False
    assert result.publish_result_report is True
