"""Receipt-bound owner escalation for production team command approvals.

This is a mechanical safety boundary, not a semantic router.  GPT authors the
Canonical task plan and exact command.  When that command reaches the existing
dangerous-action gate from an authenticated Discord team lane, this
module binds only immutable IDs and hashes, appends a pending handoff, and uses
the existing privileged Discord route-back executor to notify the owner.

No raw command, description, task prose, or credential is persisted or sent.
No owner reply is interpreted here: the existing exact ``/approve <id>`` and
``/deny <id>`` control protocol remains the only one-shot response path, while
plan capabilities remain model-authored and owner-ID bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Mapping


ESCALATION_SCHEMA = "muncho-owner-command-approval-escalation.v1"
ESCALATION_CONFIG_KEYS = frozenset(
    {
        "enabled",
        "owner_user_id",
        "owner_guild_id",
        "owner_channel_id",
        "owner_target_type",
    }
)
OWNER_TARGET_TYPE = "guild_channel"

_SNOWFLAKE = re.compile(r"^[0-9]{17,20}$")
_APPROVAL_ID = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CASE_ID = re.compile(r"^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_PLAN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}$")
_EVENT_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


class OwnerApprovalEscalationError(RuntimeError):
    """Stable, non-secret escalation failure."""

    def __init__(self, code: str) -> None:
        normalized = str(code or "owner_approval_escalation_failed").strip()
        if re.fullmatch(r"[a-z0-9_]{1,96}", normalized) is None:
            normalized = "owner_approval_escalation_failed"
        self.code = normalized
        super().__init__(normalized)


@dataclass(frozen=True)
class OwnerApprovalEscalationReceipt:
    approval_id: str
    case_id: str
    plan_id: str
    plan_revision: int
    command_sha256: str
    handoff_event_id: str
    route_back_receipt: Mapping[str, Any]


def _mapping(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OwnerApprovalEscalationError(code)
    return dict(value)


def _snowflake(value: Any, code: str) -> str:
    normalized = str(value or "").strip()
    if _SNOWFLAKE.fullmatch(normalized) is None:
        raise OwnerApprovalEscalationError(code)
    return normalized


def _decode_receipt(value: Any, code: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise OwnerApprovalEscalationError(code) from exc
    return _mapping(value, code)


def _default_workspace_resolver(**kwargs: Any) -> Mapping[str, Any]:
    from gateway.canonical_brain_task_workspace import (
        BOUNDARY_RESTART_RESUME,
        prepare_task_workspace_resume,
    )

    return prepare_task_workspace_resume(
        thread_id=kwargs["thread_id"],
        session_key=kwargs["session_key"],
        todo_store=kwargs.get("todo_store"),
        hydrate_local_state=False,
        boundary=BOUNDARY_RESTART_RESUME,
    )


def _default_active_plan_revision(*, case_id: str, plan_id: str) -> int | None:
    from tools.canonical_brain_tool import canonical_active_plan_revision

    return canonical_active_plan_revision(case_id=case_id, plan_id=plan_id)


def _default_event_appender(**kwargs: Any) -> str:
    from tools.canonical_brain_tool import canonical_event_append_tool

    return canonical_event_append_tool(**kwargs)


def _default_route_back_executor(**kwargs: Any) -> str:
    from tools.canonical_brain_tool import route_back_execute_tool

    return route_back_execute_tool(**kwargs)


def _default_binding_preparer(**kwargs: Any) -> bool:
    from tools.approval import prepare_gateway_owner_escalation_binding

    return prepare_gateway_owner_escalation_binding(**kwargs)


def _default_binding_activator(*, session_key: str, approval_id: str) -> bool:
    from tools.approval import activate_gateway_owner_escalation_binding

    return activate_gateway_owner_escalation_binding(session_key, approval_id)


def _default_binding_clearer(*, session_key: str, approval_id: str) -> None:
    from tools.approval import clear_gateway_owner_escalation_binding

    clear_gateway_owner_escalation_binding(session_key, approval_id)


def _escalation_config(config: Mapping[str, Any]) -> dict[str, str]:
    approvals = _mapping(config.get("approvals"), "owner_escalation_config_missing")
    raw = _mapping(
        approvals.get("gateway_owner_escalation"),
        "owner_escalation_config_missing",
    )
    if set(raw) != ESCALATION_CONFIG_KEYS or raw.get("enabled") is not True:
        raise OwnerApprovalEscalationError("owner_escalation_config_not_exact")
    owner_user_id = _snowflake(raw.get("owner_user_id"), "owner_user_id_invalid")
    authorized_ids = {
        str(value or "").strip()
        for value in approvals.get("gateway_authorized_user_ids") or []
    }
    if owner_user_id not in authorized_ids:
        raise OwnerApprovalEscalationError("owner_user_not_approval_authority")
    owner_guild_id = _snowflake(raw.get("owner_guild_id"), "owner_guild_id_invalid")
    owner_channel_id = _snowflake(
        raw.get("owner_channel_id"),
        "owner_channel_id_invalid",
    )
    if raw.get("owner_target_type") != OWNER_TARGET_TYPE:
        raise OwnerApprovalEscalationError("owner_target_not_guild_channel")
    return {
        "owner_user_id": owner_user_id,
        "owner_guild_id": owner_guild_id,
        "owner_channel_id": owner_channel_id,
        "owner_target_type": OWNER_TARGET_TYPE,
    }


def owner_escalation_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return only the literal config gate; perform no source/intent logic."""

    if not isinstance(config, Mapping):
        return False
    approvals = config.get("approvals")
    if not isinstance(approvals, Mapping):
        return False
    value = approvals.get("gateway_owner_escalation")
    return isinstance(value, Mapping) and value.get("enabled") is True


def _source_binding(source: Any, *, owner_guild_id: str) -> dict[str, str]:
    platform = getattr(source, "platform", None)
    platform = str(getattr(platform, "value", platform) or "")
    if platform != "discord":
        raise OwnerApprovalEscalationError("owner_escalation_source_not_discord")
    if getattr(source, "delivered_via_upstream_relay", False) is not True:
        raise OwnerApprovalEscalationError("owner_escalation_source_not_authenticated")
    chat_type = str(getattr(source, "chat_type", "") or "")
    if chat_type not in {"channel", "thread"}:
        raise OwnerApprovalEscalationError("owner_escalation_source_not_guild_lane")

    guild_id = _snowflake(
        getattr(source, "scope_id", None) or getattr(source, "guild_id", None),
        "owner_escalation_source_guild_invalid",
    )
    if guild_id != owner_guild_id:
        raise OwnerApprovalEscalationError("owner_escalation_cross_guild_forbidden")
    channel_id = _snowflake(
        getattr(source, "chat_id", None),
        "owner_escalation_source_channel_invalid",
    )
    raw_thread_id = str(getattr(source, "thread_id", None) or "").strip()
    if chat_type == "thread":
        thread_id = _snowflake(
            raw_thread_id or channel_id,
            "owner_escalation_source_thread_invalid",
        )
    else:
        if raw_thread_id:
            raise OwnerApprovalEscalationError(
                "owner_escalation_source_shape_invalid"
            )
        thread_id = ""
    parent_channel_id = str(getattr(source, "parent_chat_id", None) or "").strip()
    if parent_channel_id:
        parent_channel_id = _snowflake(
            parent_channel_id,
            "owner_escalation_source_parent_invalid",
        )
    message_id = _snowflake(
        getattr(source, "message_id", None),
        "owner_escalation_source_message_invalid",
    )
    user_id = _snowflake(
        getattr(source, "user_id", None),
        "owner_escalation_source_user_invalid",
    )
    lane_id = thread_id or channel_id
    return {
        "platform": "discord",
        "guild_id": guild_id,
        "channel_id": channel_id,
        "thread_id": thread_id,
        "parent_channel_id": parent_channel_id,
        "lane_id": lane_id,
        "message_id": message_id,
        "user_id": user_id,
    }


def _workspace_binding(
    workspace: Mapping[str, Any],
    *,
    active_plan_revision: Callable[..., int | None],
) -> dict[str, Any]:
    if workspace.get("status") != "exact":
        raise OwnerApprovalEscalationError("canonical_active_plan_required")
    case_id = str(workspace.get("case_id") or "").strip()
    plan_id = str(workspace.get("plan_id") or "").strip()
    plan_revision = workspace.get("plan_revision")
    plan_event_id = str(workspace.get("plan_event_id") or "").strip().lower()
    if _CASE_ID.fullmatch(case_id) is None:
        raise OwnerApprovalEscalationError("canonical_case_binding_invalid")
    if _PLAN_ID.fullmatch(plan_id) is None:
        raise OwnerApprovalEscalationError("canonical_plan_binding_invalid")
    if (
        not isinstance(plan_revision, int)
        or isinstance(plan_revision, bool)
        or not 1 <= plan_revision <= 999_999_999
    ):
        raise OwnerApprovalEscalationError("canonical_plan_revision_invalid")
    if _EVENT_ID.fullmatch(plan_event_id) is None:
        raise OwnerApprovalEscalationError("canonical_plan_event_invalid")
    if active_plan_revision(case_id=case_id, plan_id=plan_id) != plan_revision:
        raise OwnerApprovalEscalationError("canonical_active_plan_changed")
    return {
        "case_id": case_id,
        "plan_id": plan_id,
        "plan_revision": plan_revision,
        "plan_event_id": plan_event_id,
    }


def _handoff_committed(
    raw: Any,
    *,
    case_id: str,
) -> tuple[dict[str, Any], str]:
    receipt = _decode_receipt(raw, "canonical_handoff_receipt_invalid")
    event_id = str(receipt.get("event_id") or "").strip().lower()
    if (
        receipt.get("success") is not True
        or _EVENT_ID.fullmatch(event_id) is None
        or str(receipt.get("case_id") or case_id) != case_id
    ):
        raise OwnerApprovalEscalationError("canonical_handoff_not_committed")
    return receipt, event_id


def _route_back_sent(raw: Any) -> tuple[dict[str, Any], Mapping[str, Any]]:
    receipt = _decode_receipt(raw, "owner_route_back_receipt_invalid")
    terminal = str(receipt.get("terminal_event_type") or "")
    canonical = receipt.get("route_back_record")
    canonical = canonical if isinstance(canonical, Mapping) else {}
    canonical_sent = terminal == "route_back.sent" or (
        canonical.get("success") is True and canonical.get("outcome") == "sent"
    )
    if receipt.get("success") is not True or not canonical_sent:
        raise OwnerApprovalEscalationError("owner_route_back_not_sent")
    edge = receipt.get("receipt") or receipt.get("edge_receipt") or {}
    if not isinstance(edge, Mapping) or not edge:
        raise OwnerApprovalEscalationError("owner_route_back_receipt_missing")
    return receipt, dict(edge)


def _clear_prepared_binding(
    clearer: Callable[..., None],
    *,
    session_key: str,
    approval_id: str,
) -> None:
    """Best-effort revoke a non-active binding while preserving root failure."""

    try:
        clearer(session_key=session_key, approval_id=approval_id)
    except Exception:
        # A prepared binding is never accepted by the response resolver, so a
        # cleanup failure cannot create authority.  Preserve the route-back
        # blocker instead of replacing it with cleanup exception text.
        pass


def owner_escalation_model_message(code: str) -> str:
    """Return bounded guidance for an exact mechanical escalation failure."""

    if code in {
        "canonical_active_plan_required",
        "canonical_case_binding_invalid",
        "canonical_plan_binding_invalid",
        "canonical_plan_revision_invalid",
        "canonical_plan_event_invalid",
        "canonical_active_plan_changed",
    }:
        return (
            "BLOCKED: Owner escalation requires one exact active Canonical Task "
            "Workspace bound to this Discord guild lane. Record or refresh a "
            "model-authored case/plan/revision without secrets, then retry the "
            "same intended command once. Do not use another mutation path."
        )
    return (
        f"BLOCKED: Owner approval escalation did not obtain a verified "
        f"guild-channel delivery receipt ({code}). Do not execute or bypass the protected "
        "action. Continue safe read-only work and report this exact blocker."
    )


def escalate_production_team_approval(
    *,
    config: Mapping[str, Any],
    source: Any,
    session_key: str,
    approval_data: Mapping[str, Any],
    todo_store: Any = None,
    workspace_resolver: Callable[..., Mapping[str, Any]] = (
        _default_workspace_resolver
    ),
    active_plan_revision: Callable[..., int | None] = (
        _default_active_plan_revision
    ),
    event_appender: Callable[..., Any] = _default_event_appender,
    route_back_executor: Callable[..., Any] = _default_route_back_executor,
    binding_preparer: Callable[..., bool] = _default_binding_preparer,
    binding_activator: Callable[..., bool] = _default_binding_activator,
    binding_clearer: Callable[..., None] = _default_binding_clearer,
) -> OwnerApprovalEscalationReceipt:
    """Persist and deliver one exact owner approval escalation.

    All semantic inputs (case, plan and command) already exist before this
    function runs.  The function validates and receipts those inputs only.
    """

    if not str(session_key or "").strip():
        raise OwnerApprovalEscalationError("owner_escalation_session_missing")
    cfg = _escalation_config(_mapping(config, "owner_escalation_config_missing"))
    source_ref = _source_binding(source, owner_guild_id=cfg["owner_guild_id"])
    approval = _mapping(approval_data, "owner_escalation_approval_invalid")
    approval_id = str(approval.get("approval_id") or "").strip().lower()
    command_sha256 = str(approval.get("command_sha256") or "").strip().lower()
    if _APPROVAL_ID.fullmatch(approval_id) is None:
        raise OwnerApprovalEscalationError("owner_escalation_approval_id_invalid")
    if _SHA256.fullmatch(command_sha256) is None:
        raise OwnerApprovalEscalationError("owner_escalation_command_digest_invalid")

    try:
        workspace = workspace_resolver(
            thread_id=source_ref["lane_id"],
            session_key=str(session_key),
            todo_store=todo_store,
        )
        plan = _workspace_binding(
            _mapping(workspace, "canonical_workspace_receipt_invalid"),
            active_plan_revision=active_plan_revision,
        )
    except OwnerApprovalEscalationError:
        raise
    except Exception as exc:
        raise OwnerApprovalEscalationError(
            "canonical_workspace_resolution_failed"
        ) from exc

    exact_refs: dict[str, Any] = {
        "schema": ESCALATION_SCHEMA,
        "platform": "discord",
        "approval_id": approval_id,
        "guild_id": source_ref["guild_id"],
        "channel_id": source_ref["channel_id"],
        "message_id": source_ref["message_id"],
        "user_id": source_ref["user_id"],
        "case_id": plan["case_id"],
        "plan_id": plan["plan_id"],
        "plan_revision": plan["plan_revision"],
        "plan_event_id": plan["plan_event_id"],
        "command_sha256": command_sha256,
    }
    if source_ref["thread_id"]:
        exact_refs["thread_id"] = source_ref["thread_id"]
    if source_ref["parent_channel_id"]:
        exact_refs["parent_channel_id"] = source_ref["parent_channel_id"]

    try:
        binding_prepared = binding_preparer(
            session_key=str(session_key),
            approval_id=approval_id,
            owner_user_id=cfg["owner_user_id"],
            owner_guild_id=cfg["owner_guild_id"],
            source_lane_id=source_ref["lane_id"],
            case_id=plan["case_id"],
            plan_id=plan["plan_id"],
            plan_revision=plan["plan_revision"],
            command_sha256=command_sha256,
        )
    except Exception as exc:
        _clear_prepared_binding(
            binding_clearer,
            session_key=str(session_key),
            approval_id=approval_id,
        )
        raise OwnerApprovalEscalationError(
            "owner_approval_binding_prepare_failed"
        ) from exc
    if binding_prepared is not True:
        _clear_prepared_binding(
            binding_clearer,
            session_key=str(session_key),
            approval_id=approval_id,
        )
        raise OwnerApprovalEscalationError("owner_approval_binding_not_prepared")

    handoff_key = f"owner-approval-handoff:{approval_id}:{command_sha256[:32]}"
    try:
        handoff_raw = event_appender(
            event_type="handoff.waiting",
            case_id=plan["case_id"],
            summary="Dangerous action is waiting for exact owner approval",
            source_refs=exact_refs,
            actors={
                "actor": {"type": "runtime", "id": "gateway-approval-boundary"},
                "subject": {"type": "plan", "id": plan["plan_id"]},
            },
            payload={
                "handoff": {
                    **exact_refs,
                    "state": "pending_owner_approval",
                    "owner_user_id": cfg["owner_user_id"],
                    "owner_guild_id": cfg["owner_guild_id"],
                    "owner_channel_id": cfg["owner_channel_id"],
                    "owner_target_type": cfg["owner_target_type"],
                },
                "next_action": {
                    "kind": "exact_owner_approval",
                    "approval_id": approval_id,
                    "source_lane_id": source_ref["lane_id"],
                },
                "outbound": False,
            },
            safety={
                "contains_secret": False,
                "contains_payment_credential": False,
                "business_mutation": False,
                "outbound": False,
            },
            idempotency_key=handoff_key,
        )
    except Exception as exc:
        _clear_prepared_binding(
            binding_clearer,
            session_key=str(session_key),
            approval_id=approval_id,
        )
        raise OwnerApprovalEscalationError("canonical_handoff_append_failed") from exc
    try:
        _, handoff_event_id = _handoff_committed(
            handoff_raw,
            case_id=plan["case_id"],
        )
    except OwnerApprovalEscalationError:
        _clear_prepared_binding(
            binding_clearer,
            session_key=str(session_key),
            approval_id=approval_id,
        )
        raise

    jump_url = (
        "https://discord.com/channels/"
        f"{source_ref['guild_id']}/{source_ref['lane_id']}/"
        f"{source_ref['message_id']}"
    )
    escalation_message = "\n".join(
        [
            f"<@{cfg['owner_user_id']}> ⚠️ Owner approval is required.",
            f"approval_id: `{approval_id}`",
            f"source_user_id: `{source_ref['user_id']}`",
            f"source_guild_id: `{source_ref['guild_id']}`",
            f"source_lane_id: `{source_ref['lane_id']}`",
            f"source_message_id: `{source_ref['message_id']}`",
            f"case_id: `{plan['case_id']}`",
            f"plan_id: `{plan['plan_id']}`",
            f"plan_revision: `{plan['plan_revision']}`",
            f"command_sha256: `{command_sha256}`",
            f"Open the exact source: {jump_url}",
            (
                "After review, respond in that source lane with "
                f"`/approve {approval_id}` or `/deny {approval_id} <reason>`."
            ),
            (
                "The command and task prose are intentionally not copied here; "
                "no free-form reply is converted into approval authority."
            ),
        ]
    )
    route_key = f"owner-approval-escalation:{approval_id}:{command_sha256[:32]}"
    try:
        route_raw = route_back_executor(
            case_id=plan["case_id"],
            target_ref={
                "id": cfg["owner_user_id"],
                "channel_id": cfg["owner_channel_id"],
                "guild_id": cfg["owner_guild_id"],
                "target_type": cfg["owner_target_type"],
            },
            message=escalation_message,
            message_summary="Exact owner command-approval escalation",
            source_refs={**exact_refs, "handoff_event_id": handoff_event_id},
            idempotency_key=route_key,
        )
        _, route_receipt = _route_back_sent(route_raw)
    except OwnerApprovalEscalationError:
        _clear_prepared_binding(
            binding_clearer,
            session_key=str(session_key),
            approval_id=approval_id,
        )
        raise
    except Exception as exc:
        _clear_prepared_binding(
            binding_clearer,
            session_key=str(session_key),
            approval_id=approval_id,
        )
        raise OwnerApprovalEscalationError("owner_route_back_failed") from exc
    try:
        binding_active = binding_activator(
            session_key=str(session_key),
            approval_id=approval_id,
        )
    except Exception as exc:
        raise OwnerApprovalEscalationError(
            "owner_approval_binding_activation_failed"
        ) from exc
    if binding_active is not True:
        raise OwnerApprovalEscalationError("owner_approval_binding_not_active")
    return OwnerApprovalEscalationReceipt(
        approval_id=approval_id,
        case_id=plan["case_id"],
        plan_id=plan["plan_id"],
        plan_revision=plan["plan_revision"],
        command_sha256=command_sha256,
        handoff_event_id=handoff_event_id,
        route_back_receipt=route_receipt,
    )


__all__ = [
    "ESCALATION_SCHEMA",
    "OwnerApprovalEscalationError",
    "OwnerApprovalEscalationReceipt",
    "escalate_production_team_approval",
    "owner_escalation_enabled",
    "owner_escalation_model_message",
]
