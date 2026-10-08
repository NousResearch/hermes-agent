"""Native action/phase requirements, separated from wait projection and receipts."""

def blocked_action(conn, t, decision, selected, health):
    from hermes_cli.kanban_decision import _latest_event, _json
    task_id = str(t["id"])
    event = _latest_event(conn, task_id, ("blocked", "block_loop_detected"))
    payload = _json(event[2]) if event else {}
    if health["state"] == "INFRASTRUCTURE_FAULT" and payload.get("kind") in (None, "transient"):
        resume = payload.get("resume_status") or payload.get("source_status") or "ready"
        decision.update(workflow_stage="REVIEW" if resume == "review" else "READY",
                        reason="Semantic action remains valid; launch environment is unhealthy",
                        owner=t.get("assignee"), next_action=selected.get("action") or
                        ("Run independent review" if resume == "review" else "Run the authorised action"),
                        resume_condition="environment fingerprint changes",
                        worker_executable_now=False, dispatchable=False)
    else:
        reason = payload.get("reason")
        resume = payload.get("resume_condition") or payload.get("resume_status")
        resolver = payload.get("resolver")
        incomplete_blocker = not (reason and resume and resolver)
        if incomplete_blocker:
            decision["invariant_violations"].append("BLOCKED lacks blocker/resolver/resume_condition")
        kind = payload.get("kind") or t.get("block_kind")
        if not resolver:
            resolver = {"needs_input": "operator", "capability": "operator",
                        "transient": "dispatcher"}.get(kind)
        decision.update(workflow_stage="BLOCKED", block_kind=kind,
                        reason=reason or "Blocker classification is incomplete",
                        owner=resolver or t.get("assignee"),
                        next_action=payload.get("blocked_action") or selected.get("action") or
                        "Resolve the exact recorded fault",
                        resume_condition=resume)
        if incomplete_blocker:
            decision.update(workflow_stage="WORKFLOW_FAULT", owner="control plane",
                            reason="Recorded block lacks a complete condition/resolver/release contract: " + str(reason or "no condition recorded"),
                            next_action="Reconcile the exact blocker or retained completion verdict",
                            resume_condition="control plane records a complete gate or valid transition")


def review_action(conn, t, decision, selected, eligible):
    from hermes_cli.kanban_decision import _latest_event, _json
    task_id = str(t["id"])
    handoff = _latest_event(conn, task_id, ("operator_review_handoff",))
    requested = _latest_event(conn, task_id, ("review_requested",))
    if handoff is not None and (requested is None or handoff["id"] > requested["id"]):
        decision.update(workflow_stage="WORKFLOW_FAULT",
                        reason="Reviewer returned to the control plane without a native verdict transition",
                        owner="control plane",
                        next_action="Bind the retained verdict; close approved work or select concrete corrections",
                        resume_condition="retained review verdict is reconciled natively")
    elif eligible:
        decision.update(workflow_stage="REVIEW", dispatchable=True,
                        worker_executable_now=True,
                        reason="A substantive independent review remains unresolved",
                        owner=t.get("assignee") or "reviewer",
                        next_action=selected.get("action") or
                        "Accept, reject, or request concrete changes",
                        current_next_action_type="review")
    else:
        decision.update(workflow_stage="WORKFLOW_FAULT",
                        reason="Review is disabled without a substantive decision or STOP",
                        owner="control plane", next_action="Reconcile retained review authority and eligibility",
                        resume_condition="native review action or exact gate is recorded")
        decision["invariant_violations"].append(
            "REVIEW advertised runnable but claim rejects")


def triage_action(conn, t, decision, selected, remaining, evidence_identities):
    from hermes_cli.kanban_decision import _latest_event, _json
    task_id = str(t["id"])
    children = conn.execute("SELECT 1 FROM task_links WHERE parent_id=? LIMIT 1", (task_id,)).fetchone()
    unknown = not selected.get("action")
    requirement = remaining[0] if remaining else None
    requirement_data = requirement if isinstance(requirement, dict) else {}
    requirement_identity = requirement_data.get("evidence_identity") or requirement_data.get("requirement_id")
    evidence_already_accepts = bool(
        requirement_identity and any(
            item.get("identity") == requirement_identity for item in evidence_identities
        )
    )
    # Intake contracts (control_plane_pre_promotion, #t_9c2c5733) carry the scope as a
    # plain string equal to selected_next_action.action. That is a bounded, already
    # authoritative scope: the aux specifier/decomposer rewrites title/body without
    # changing the requirement, so auto-triage is safe for exactly this shape.
    deterministic_scope = bool(
        not unknown
        and isinstance(requirement, str)
        and selected.get("type") == "worker_action"
        and str(requirement).strip() == str(selected.get("action") or "").strip()
    )
    safely_decomposable = bool(
        requirement_data.get("worker_executable") is True
        or requirement_data.get("decomposable") is True
        or deterministic_scope
    )
    decision.update(workflow_stage="TRIAGE", reason="The next required action is unknown" if unknown else "Triage controller assesses bounded scope",
                    owner=t.get("assignee") or "triage controller",
                    next_action="Derive one bounded unsatisfied requirement")
    decision["auto_decompose_allowed"] = bool(
        safely_decomposable and not evidence_already_accepts
        and not requirement_data.get("owner_task_id") and not children
    )

def apply_action_decision(conn, t, decision, contract, status, eligible, selected, health, accepted, remaining, remaining_declared, evidence_identities, *, preparation_satisfied=False):
    from hermes_cli.kanban_completion_evidence import validate_contract
    from hermes_cli.kanban_readiness import preparation_reason
    from hermes_cli.kanban_decision import _latest_event, _json
    task_id = str(t["id"])
    explicitly_unqualified = contract.get("qualified_for_dispatch") is False
    guarded_contract = (
        str(t.get("idempotency_key") or "").startswith(("codex-goalpack-", "workflow-programme-"))
        or conn.execute(
            "SELECT 1 FROM task_events WHERE task_id=? AND kind='completion_requirements' LIMIT 1",
            (task_id,),
        ).fetchone() is not None
    )
    try:
        validate_contract(contract)
        contract_valid = True
    except ValueError:
        # Preserve legacy intake until it enters the explicit contract
        # lifecycle. New guarded work fails closed and self-prepares.
        contract_valid = not guarded_contract
    prep = preparation_reason(conn, task_id)
    if preparation_satisfied:
        explicitly_unqualified = False
        contract_valid = True
        prep = ""
        if status != "review" and not selected.get("action"):
            selected = {"action": "The authoritative action selected by the preparation resolver", "type": "counterfactual_preparation"}
    if contract_valid and remaining_declared and not remaining and accepted:
        decision.update(
            workflow_stage="DONE", dispatchable=False,
            reason="All required actions have accepted evidence and no action remains",
            owner=None, next_action=None, current_next_action=None,
            current_next_action_type=None,
        )
    elif explicitly_unqualified:
        decision.update(workflow_stage="PREPARE",
                        reason="Original scope and acceptance await qualification",
                        owner=t.get("assignee") or "preparation controller",
                        next_action="Qualify the authoritative scope, workspace, and acceptance",
                        current_next_action_type="preparation",
                        worker_executable_now=eligible)
    elif not contract_valid:
        deterministic = bool(str(t.get("title") or "").strip() and str(t.get("body") or "").strip())
        decision.update(workflow_stage="PREPARE" if deterministic else "TRIAGE",
                        reason="Completion contract must be materialised from authoritative scope"
                        if deterministic else "Authoritative scope is genuinely ambiguous",
                        owner="control plane" if deterministic else "operator",
                        next_action="Materialise deterministic scope and acceptance metadata"
                        if deterministic else "Clarify the bounded scope and acceptance",
                        current_next_action_type="deterministic_preparation" if deterministic else "scope_decision",
                        worker_executable_now=False)
    elif prep:
        decision.update(workflow_stage="PREPARE", reason=prep,
                        owner=t.get("assignee") or "preparation controller",
                        next_action=prep.removeprefix("Preparation required: ").strip(),
                        current_next_action_type="preparation",
                        worker_executable_now=eligible)
        preparation_event = _latest_event(conn, task_id, ("workspace_preparation_failed", "workspace_prepared"))
        if preparation_event and preparation_event[1] == "workspace_preparation_failed":
            failure = _json(preparation_event[2])
            decision.update(workflow_stage="WORKFLOW_FAULT", owner="control plane",
                            reason="Native workspace preparation failed: " + str(failure.get("reason")),
                            next_action="Repair the recorded workspace provisioning fault", worker_executable_now=False,
                            resume_condition=failure.get("resume_condition"))
    elif status == "blocked":
        blocked_action(conn, t, decision, selected, health)
    elif status == "review":
        review_action(conn, t, decision, selected, eligible)
    elif status in ("ready", "todo") and eligible and selected.get("action"):
        action = selected["action"]
        decision.update(workflow_stage="READY", dispatchable=True,
                        worker_executable_now=True, reason="All current action prerequisites are satisfied",
                        owner=t.get("assignee"), next_action=action,
                        current_next_action=action,
                        current_next_action_type=selected.get("type") or "worker_action")
    elif status in ("ready", "todo") and eligible:
        decision.update(
            workflow_stage="PREPARE",
            reason="No explicit next action is selected; dispatch would manufacture work",
            owner="control plane",
            next_action="Select one unsatisfied worker-executable requirement",
            current_next_action_type="deterministic_preparation",
            worker_executable_now=False,
        )
    elif status == "triage":
        triage_action(conn, t, decision, selected, remaining, evidence_identities)

    else:
        decision.update(workflow_stage="WORKFLOW_FAULT",
                        reason="Dispatch is disabled without a recorded substantive hold",
                        owner="control plane", next_action="Reconcile eligibility against retained authority and next-action evidence",
                        resume_condition="control plane records a justified action or exact gate")
        decision["invariant_violations"].append("Disabled eligibility has no substantive hold")
