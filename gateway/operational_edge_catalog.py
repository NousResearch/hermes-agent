"""Code-owned catalog for Cloud Muncho's credential-scoped operations edge.

Every row is an exact mechanical operation.  The model chooses the row; no
prose, output, or argument value influences dispatch.  Dynamic values are
validated only against the declared type/bounds and are appended to argv with
``shell=False`` semantics by :mod:`gateway.operational_edge_service`.
"""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from gateway.operational_edge_protocol import OperationalAccess


CATALOG_SCHEMA = "muncho-operational-edge-catalog.v1"
ASSET_ROOT_RELATIVE = Path("ops/muncho/runtime/operational-assets")
HERMES_HOME = Path("/opt/adventico-ai-platform/hermes-home")
CANONICAL_BRAIN = Path("/opt/adventico-ai-platform/canonical-brain")

MAX_TEXT_CHARS = 12_000
MAX_SQL_CHARS = 32_000

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,239}$")
_DATE = re.compile(r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]$")
_SHA = re.compile(r"^[0-9a-fA-F]{8,40}$")
_EMAIL_LIST = re.compile(r"^[^\x00\r\n]{1,1600}$")


# Static documentation for GPT.  Runtime dispatch never reads these strings;
# it still selects solely by the exact ``operation_id`` supplied by the model.
OPERATION_PURPOSES: Mapping[str, str] = MappingProxyType(
    {
        "skyvision.email.status": "Check SkyVision mail service readiness.",
        "skyvision.email.smoke": "Run a bounded read-only SkyVision mailbox probe.",
        "skyvision.email.search": "Search one SkyVision mailbox with explicit filters.",
        "skyvision.email.read": "Read one exact SkyVision message by mailbox UID.",
        "skyvision.email.send_plan": "Build a non-sending SkyVision email plan.",
        "skyvision.email.send_request_approval": "Request approval for one exact SkyVision email send.",
        "skyvision.email.send_execute": "Execute one previously approved SkyVision email send.",
        "adventico.email.status": "Check Adventico mail service readiness.",
        "adventico.email.smoke": "Run a bounded read-only Adventico mailbox probe.",
        "adventico.email.search": "Search the Adventico mailbox with explicit filters.",
        "adventico.email.read": "Read one exact Adventico message by mailbox UID.",
        "bitrix.crm.status": "Check SkyVision Bitrix CRM connectivity.",
        "bitrix.crm.smoke": "Run a bounded read-only Bitrix CRM probe.",
        "bitrix.crm.profile": "Read the authenticated Bitrix CRM profile.",
        "bitrix.crm.status_list": "List Bitrix CRM statuses, optionally for one entity.",
        "bitrix.crm.fields": "List fields for one Bitrix CRM entity type.",
        "bitrix.crm.entity_get": "Read one exact Bitrix CRM entity by type and ID.",
        "bitrix.crm.deal_search": "Search Bitrix deals using explicit model-authored filters.",
        "bitrix.crm.contact_search": "Search Bitrix contacts using explicit model-authored filters.",
        "bitrix.crm.lead_search": "Search Bitrix leads using explicit model-authored filters.",
        "bitrix.crm.timeline_list": "Read recent timeline comments for one Bitrix entity.",
        "bitrix.crm.lead_add": "Create one owner-approved Bitrix CRM lead.",
        "bitrix.crm.timeline_add": "Add one owner-approved Bitrix timeline comment.",
        "bitrix.voucher.status": "Check the Bitrix voucher helper readiness.",
        "bitrix.voucher.find": "Find Bitrix vouchers by exact serial.",
        "bitrix.voucher.get_deal": "Read the Bitrix deal for one exact deal ID.",
        "bitrix.voucher.plan_extension": "Plan a voucher validity extension without applying it.",
        "bitrix.voucher.request_extension": "Request approval for one exact voucher extension.",
        "bitrix.voucher.execute_extension": "Execute one previously approved voucher extension.",
        "skyvision.db.probe": "Run the fixed read-only SkyVision database readiness query.",
        "skyvision.db.query": "Run one bounded read-only query against an allowed SkyVision database.",
        "skyvision.panel.status": "Check SkyVision panel read-only helper readiness.",
        "skyvision.panel.check_voucher_user": "Check the panel user linked to one voucher.",
        "skyvision.panel.invoice_lookup": "Look up an invoice by invoice ID or order ID.",
        "skyvision.gitlab.status": "Check SkyVision GitLab read-only access.",
        "skyvision.gitlab.projects": "List bounded projects in the configured SkyVision GitLab group.",
        "skyvision.gitlab.project": "Read metadata for one exact SkyVision GitLab project.",
        "skyvision.gitlab.branches": "List bounded branches for one SkyVision GitLab project.",
        "skyvision.gitlab.mrs": "List bounded merge requests for one SkyVision GitLab project.",
        "skyvision.gitlab.pipelines": "List bounded pipelines for one SkyVision GitLab project.",
        "skyvision.gitlab.tree": "List a bounded repository tree at an explicit path and ref.",
        "skyvision.gitlab.file_stat": "Read metadata for one exact repository file and ref.",
        "skyvision.deploy.status": "Check the SkyVision production deploy helper status.",
        "skyvision.deploy.preflight": "Run read-only preflight for one exact SkyVision revision.",
        "skyvision.deploy.request_approval": "Request approval for one exact SkyVision production deploy.",
        "skyvision.deploy.execute": "Execute one previously approved SkyVision production deploy.",
        "infra.contabo.observe": "Read the bounded Contabo infrastructure inventory.",
        "infra.alwyzon.observe": "Read bounded Alwyzon Phoenix host status.",
        "infra.watchtower.fast": "Run the fixed fast infrastructure observation job.",
        "infra.watchtower.full": "Run the fixed full infrastructure observation job.",
        "infra.watchtower.hourly": "Run the fixed hourly infrastructure observation job.",
        "infra.watchtower.digest": "Run the fixed infrastructure digest job.",
        "cron.canonical.heartbeat": "Run the fixed Canonical Brain heartbeat writer job.",
        "cron.canonical.projections": "Refresh Canonical Brain projections incrementally.",
        "cron.canonical.routeback_violation": "Check for route-back receipt violations.",
        "cron.canonical.git_parity": "Check Cloud and local Git revision parity.",
        "cron.canonical.knowledge_voucher": "Check the voucher knowledge artifact sync state.",
        "cron.canonical.knowledge_all": "Check all Canonical Brain knowledge artifact sync states.",
        "cron.canonical.routeback_watchdog": "Run the fixed stale route-back state watchdog.",
        "cron.canonical.persistence_watchdog": "Run the fixed operational persistence watchdog.",
        "cron.skyvision.heart_weekly": "Run the fixed weekly SkyVision From the Heart count.",
        "cron.github.refs": "Collect the fixed bounded GitHub reference set.",
    }
)

WEBSITE_RELEASE_CONTRACT_BLOCKER = "website_release_contract_unavailable"
WEBSITE_RELEASE_CONTRACT_REQUIREMENT = (
    "Requires a trusted external SkyVision website receipt proving exact "
    "Node, npm, lockfile/build command, build/PM2 executable parity, a 5-10% "
    "canary, a 2-4 hour metrics soak, and tested rollback."
)


class OperationalCatalogError(ValueError):
    pass


class ArgumentKind(StrEnum):
    TEXT = "text"
    IDENTIFIER = "identifier"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    DATE = "date"
    SHA = "sha"
    IP = "ip"
    EMAIL_LIST = "email_list"
    SQL = "sql"


@dataclass(frozen=True)
class ArgumentSpec:
    name: str
    flag: str
    kind: ArgumentKind = ArgumentKind.TEXT
    required: bool = False
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    maximum_chars: int = MAX_TEXT_CHARS
    repeat: bool = False


@dataclass(frozen=True)
class CredentialBinding:
    name: str
    source_path: Path
    target_path: Path


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    source_root: str
    source_relative: Path
    packaged_relative: Path


@dataclass(frozen=True)
class OperationSpec:
    operation_id: str
    purpose: str
    domain: str
    access: OperationalAccess
    asset_id: str
    argv_prefix: tuple[str, ...]
    arguments: tuple[ArgumentSpec, ...] = ()
    timeout_seconds: int = 60
    probe_operation_id: str = ""
    expected_json: bool = True
    cron_source_job_id: str = ""
    runner_module: str = ""
    requires_any_of: tuple[tuple[str, ...], ...] = ()
    available: bool = True
    blocker_code: str = ""
    availability_requirement: str = ""


def _arg(
    name: str,
    *,
    flag: str | None = None,
    kind: ArgumentKind = ArgumentKind.TEXT,
    required: bool = False,
    choices: Sequence[str] = (),
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_chars: int = MAX_TEXT_CHARS,
    repeat: bool = False,
) -> ArgumentSpec:
    return ArgumentSpec(
        name=name,
        flag=flag or "--" + name.replace("_", "-"),
        kind=kind,
        required=required,
        choices=tuple(choices),
        minimum=minimum,
        maximum=maximum,
        maximum_chars=maximum_chars,
        repeat=repeat,
    )


def _asset(name: str, *, root: str = "hermes", relative: str | None = None) -> AssetSpec:
    source = Path(relative or f"scripts/{name}")
    return AssetSpec(
        asset_id=name,
        source_root=root,
        source_relative=source,
        packaged_relative=ASSET_ROOT_RELATIVE / name,
    )


ASSETS: tuple[AssetSpec, ...] = (
    _asset("skyvision_email_ops.py"),
    _asset("adventico_email_ops.py"),
    _asset("bitrix_skyvision_crm.py"),
    _asset("bitrix_skyvision_crm_report.py"),
    _asset("bitrix_voucher_ops.py"),
    _asset("skyvision_callcenter_api_tools.py"),
    _asset("skyvision_db_readonly.py", relative="bin/skyvision_db_readonly"),
    _asset("skyvision_panel_ops_readonly.py"),
    _asset("skyvision_gitlab_prod_deploy.py"),
    _asset("skyvision_gitlab_readonly.py", root="release", relative="ops/muncho/runtime/skyvision_gitlab_readonly.py"),
    _asset("skyvision_whm_csf_exact_ip.py"),
    _asset("devops_watchtower_phase1.py"),
    _asset("contabo_observer.py"),
    _asset("alwyzon_phoenix_observer.py"),
    _asset("cloud_heartbeat_writer_no_enforcement_gate.py", root="canonical", relative="bin/cloud_heartbeat_writer_no_enforcement_gate.py"),
    _asset("canonical_brain_projections_v1_1_refresh.py", root="canonical", relative="bin/canonical_brain_projections_v1_1_refresh.py"),
    _asset("handoff_route_back_violation_alert_gate.py", root="canonical", relative="bin/handoff_route_back_violation_alert_gate.py"),
    _asset("cloud_local_git_drift_monitor.py", root="canonical", relative="scripts/cloud_local_git_drift_monitor.py"),
    _asset("knowledge_artifact_sync_status.py", root="canonical", relative="bin/knowledge_artifact_sync_status.py"),
    _asset("free_hermes_route_back_state_watchdog.py", root="canonical", relative="bin/free_hermes_route_back_state_watchdog.py"),
    _asset("free_hermes_operational_persistence_watchdog.py", root="canonical", relative="bin/free_hermes_operational_persistence_watchdog.py"),
    _asset("skyvision_from_heart_weekly_count.py"),
    _asset("gh-hermes", relative="bin/gh-hermes"),
    _asset("contabo-api", relative="bin/contabo-api"),
    _asset("ssh-alwyzon-phoenix", relative="bin/ssh-alwyzon-phoenix"),
    _asset("muncho_step_up_verify", relative="bin/muncho_step_up_verify"),
    _asset("muncho_dangerous_action_guard", relative="bin/muncho_dangerous_action_guard"),
)


SKYVISION_EMAIL_ACCOUNTS = (
    "orders",
    "reservations",
    "office",
    "info",
    "support",
    "orders@skyvision.bg",
    "reservations@skyvision.bg",
    "office@skyvision.bg",
    "info@skyvision.bg",
    "support@skyvision.bg",
)

EMAIL_SEND_ARGS = (
    _arg("account", required=True, kind=ArgumentKind.CHOICE, choices=SKYVISION_EMAIL_ACCOUNTS),
    _arg("to", required=True, kind=ArgumentKind.EMAIL_LIST),
    _arg("cc", kind=ArgumentKind.EMAIL_LIST),
    _arg("bcc", kind=ArgumentKind.EMAIL_LIST),
    _arg("subject", required=True, maximum_chars=998),
    _arg("body", required=True, maximum_chars=12_000),
    _arg("in_reply_to", maximum_chars=998),
    _arg("references", maximum_chars=2000),
    _arg("case_id", required=True, kind=ArgumentKind.IDENTIFIER),
    _arg("requester", required=True, maximum_chars=240),
    _arg("reason", required=True, maximum_chars=2000),
)


def _op(
    operation_id: str,
    domain: str,
    asset_id: str,
    *prefix: str,
    access: OperationalAccess = OperationalAccess.READ,
    arguments: Sequence[ArgumentSpec] = (),
    timeout: int = 60,
    probe: str = "",
    cron: str = "",
    runner: str = "",
    requires_any_of: Sequence[Sequence[str]] = (),
    available: bool = True,
    blocker_code: str = "",
    availability_requirement: str = "",
) -> OperationSpec:
    return OperationSpec(
        operation_id=operation_id,
        purpose=OPERATION_PURPOSES.get(operation_id, ""),
        domain=domain,
        access=access,
        asset_id=asset_id,
        argv_prefix=tuple(prefix),
        arguments=tuple(arguments),
        timeout_seconds=timeout,
        probe_operation_id=probe,
        cron_source_job_id=cron,
        runner_module=runner,
        requires_any_of=tuple(tuple(group) for group in requires_any_of),
        available=available,
        blocker_code=blocker_code,
        availability_requirement=availability_requirement,
    )


OPERATIONS: tuple[OperationSpec, ...] = (
    # Operational mail (all content fields are model-authored; no classifier).
    _op("skyvision.email.status", "skyvision_email", "skyvision_email_ops.py", "status"),
    _op("skyvision.email.smoke", "skyvision_email", "skyvision_email_ops.py", "smoke-readonly", arguments=(_arg("account", kind=ArgumentKind.CHOICE, choices=SKYVISION_EMAIL_ACCOUNTS),)),
    _op("skyvision.email.search", "skyvision_email", "skyvision_email_ops.py", arguments=(
        _arg("account", required=True, kind=ArgumentKind.CHOICE, choices=SKYVISION_EMAIL_ACCOUNTS),
        _arg("since", kind=ArgumentKind.DATE), _arg("from", flag="--from", maximum_chars=320),
        _arg("to", maximum_chars=320), _arg("subject", maximum_chars=998),
        _arg("text", maximum_chars=2000), _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=25),
    ), timeout=90, probe="skyvision.email.status", runner="ops.muncho.runtime.skyvision_email_utf8_search"),
    _op("skyvision.email.read", "skyvision_email", "skyvision_email_ops.py", "read", arguments=(
        _arg("account", required=True, kind=ArgumentKind.CHOICE, choices=SKYVISION_EMAIL_ACCOUNTS),
        _arg("uid", required=True, maximum_chars=200), _arg("max_chars", kind=ArgumentKind.INTEGER, minimum=200, maximum=4000),
    ), probe="skyvision.email.status"),
    _op("skyvision.email.send_plan", "skyvision_email", "skyvision_email_ops.py", "send", arguments=EMAIL_SEND_ARGS, probe="skyvision.email.status"),
    _op("skyvision.email.send_request_approval", "skyvision_email", "skyvision_email_ops.py", "request-send-approval", arguments=(*EMAIL_SEND_ARGS, _arg("requester_discord_user_id", required=True, kind=ArgumentKind.IDENTIFIER)), access=OperationalAccess.MUTATION, timeout=90, probe="skyvision.email.status"),
    _op("skyvision.email.send_execute", "skyvision_email", "skyvision_email_ops.py", "execute-send", arguments=(*EMAIL_SEND_ARGS, _arg("request_id", required=True, kind=ArgumentKind.IDENTIFIER)), access=OperationalAccess.MUTATION, timeout=90, probe="skyvision.email.status"),
    _op("adventico.email.status", "adventico_email", "adventico_email_ops.py", "status"),
    _op("adventico.email.smoke", "adventico_email", "adventico_email_ops.py", "smoke-readonly", arguments=(_arg("account", kind=ArgumentKind.CHOICE, choices=("info", "info@adventico.com")),)),
    _op("adventico.email.search", "adventico_email", "adventico_email_ops.py", "search", arguments=(
        _arg("account", kind=ArgumentKind.CHOICE, choices=("info", "info@adventico.com")), _arg("folder", maximum_chars=240),
        _arg("since", kind=ArgumentKind.DATE), _arg("from_addr", flag="--from-addr", maximum_chars=320), _arg("to_addr", flag="--to-addr", maximum_chars=320),
        _arg("subject", maximum_chars=998), _arg("text", maximum_chars=2000), _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=25),
    ), timeout=90, probe="adventico.email.status"),
    _op("adventico.email.read", "adventico_email", "adventico_email_ops.py", "read", arguments=(
        _arg("account", kind=ArgumentKind.CHOICE, choices=("info", "info@adventico.com")), _arg("folder", maximum_chars=240),
        _arg("uid", required=True, maximum_chars=200), _arg("max_chars", kind=ArgumentKind.INTEGER, minimum=200, maximum=4000),
    ), probe="adventico.email.status"),

    # Bitrix read/report and helper-enforced owner mutations.
    _op("bitrix.crm.status", "bitrix", "bitrix_skyvision_crm.py", "status"),
    _op("bitrix.crm.smoke", "bitrix", "bitrix_skyvision_crm.py", "smoke"),
    _op("bitrix.crm.profile", "bitrix", "bitrix_skyvision_crm.py", "profile"),
    _op("bitrix.crm.status_list", "bitrix", "bitrix_skyvision_crm.py", "status-list", arguments=(_arg("entity_id", maximum_chars=80),)),
    _op("bitrix.crm.fields", "bitrix", "bitrix_skyvision_crm.py", "fields", arguments=(_arg("entity", required=True, kind=ArgumentKind.CHOICE, choices=("lead", "deal", "contact", "company")),)),
    _op("bitrix.crm.entity_get", "bitrix", "bitrix_skyvision_crm.py", "entity-get", arguments=(
        _arg("entity", required=True, kind=ArgumentKind.CHOICE, choices=("lead", "deal", "contact", "company")), _arg("id", required=True, kind=ArgumentKind.IDENTIFIER),
        _arg("mode", kind=ArgumentKind.CHOICE, choices=("team", "owner")), _arg("requester", maximum_chars=240),
    )),
    _op("bitrix.crm.deal_search", "bitrix", "bitrix_skyvision_crm.py", "deal-search", arguments=(
        _arg("deal_id", kind=ArgumentKind.IDENTIFIER), _arg("voucher_serial", maximum_chars=80), _arg("title_contains", maximum_chars=160),
        _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=20), _arg("mode", kind=ArgumentKind.CHOICE, choices=("team", "owner")), _arg("requester", maximum_chars=240),
    )),
    _op("bitrix.crm.contact_search", "bitrix", "bitrix_skyvision_crm.py", "contact-search", arguments=(
        _arg("email", maximum_chars=320), _arg("phone", maximum_chars=80), _arg("name", maximum_chars=160),
        _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=20), _arg("mode", kind=ArgumentKind.CHOICE, choices=("team", "owner")), _arg("requester", maximum_chars=240),
    )),
    _op("bitrix.crm.lead_search", "bitrix", "bitrix_skyvision_crm.py", "lead-search", arguments=(
        _arg("email", maximum_chars=320), _arg("phone", maximum_chars=80), _arg("name", maximum_chars=160), _arg("title_contains", maximum_chars=160),
        _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=20), _arg("mode", kind=ArgumentKind.CHOICE, choices=("team", "owner")), _arg("requester", maximum_chars=240),
    )),
    _op("bitrix.crm.timeline_list", "bitrix", "bitrix_skyvision_crm.py", "timeline-comment-list", arguments=(
        _arg("entity_type", required=True, maximum_chars=80), _arg("entity_id", required=True, kind=ArgumentKind.IDENTIFIER),
        _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=20), _arg("mode", kind=ArgumentKind.CHOICE, choices=("team", "owner")), _arg("requester", maximum_chars=240),
    )),
    _op("bitrix.crm.lead_add", "bitrix", "bitrix_skyvision_crm.py", "lead-add", arguments=(
        _arg("title", required=True, maximum_chars=160), _arg("name", maximum_chars=160), _arg("last_name", maximum_chars=160),
        _arg("company_title", maximum_chars=160), _arg("phone", maximum_chars=80), _arg("email", maximum_chars=320),
        _arg("comments", maximum_chars=4000), _arg("source_description", maximum_chars=240), _arg("assigned_by_id", kind=ArgumentKind.IDENTIFIER),
        _arg("requester", required=True, maximum_chars=240), _arg("reason", required=True, maximum_chars=2000), _arg("execute", kind=ArgumentKind.BOOLEAN),
    ), access=OperationalAccess.MUTATION, probe="bitrix.crm.status"),
    _op("bitrix.crm.timeline_add", "bitrix", "bitrix_skyvision_crm.py", "timeline-comment-add", arguments=(
        _arg("entity_type", required=True, maximum_chars=80), _arg("entity_id", required=True, kind=ArgumentKind.IDENTIFIER),
        _arg("comment", required=True, maximum_chars=4000), _arg("requester", required=True, maximum_chars=240),
        _arg("reason", required=True, maximum_chars=2000), _arg("execute", kind=ArgumentKind.BOOLEAN),
    ), access=OperationalAccess.MUTATION, probe="bitrix.crm.status"),
    _op("bitrix.voucher.status", "bitrix", "bitrix_voucher_ops.py", "status"),
    _op("bitrix.voucher.find", "bitrix", "bitrix_voucher_ops.py", "find-by-serial", arguments=(_arg("serial", required=True, maximum_chars=80), _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=20))),
    _op("bitrix.voucher.get_deal", "bitrix", "bitrix_voucher_ops.py", "get-deal", arguments=(_arg("deal_id", required=True, kind=ArgumentKind.IDENTIFIER),)),
    _op("bitrix.voucher.plan_extension", "bitrix", "bitrix_voucher_ops.py", "plan-extension", arguments=(
        _arg("serial", maximum_chars=80), _arg("deal_id", kind=ArgumentKind.IDENTIFIER), _arg("slot", kind=ArgumentKind.INTEGER, minimum=1, maximum=8),
        _arg("new_valid_until", required=True, kind=ArgumentKind.DATE), _arg("case_id", required=True, kind=ArgumentKind.IDENTIFIER),
        _arg("requester", required=True, maximum_chars=240), _arg("reason", required=True, maximum_chars=2000),
        _arg("set_stage_payment", kind=ArgumentKind.BOOLEAN), _arg("no_auto_stage_from_expired", kind=ArgumentKind.BOOLEAN),
    )),
    _op("bitrix.voucher.request_extension", "bitrix", "bitrix_voucher_ops.py", "request-extension-approval", arguments=(
        _arg("serial", maximum_chars=80), _arg("deal_id", kind=ArgumentKind.IDENTIFIER), _arg("slot", kind=ArgumentKind.INTEGER, minimum=1, maximum=8),
        _arg("new_valid_until", required=True, kind=ArgumentKind.DATE), _arg("case_id", required=True, kind=ArgumentKind.IDENTIFIER),
        _arg("requester", required=True, maximum_chars=240), _arg("reason", required=True, maximum_chars=2000),
        _arg("set_stage_payment", kind=ArgumentKind.BOOLEAN), _arg("no_auto_stage_from_expired", kind=ArgumentKind.BOOLEAN),
        _arg("requester_discord_user_id", required=True, kind=ArgumentKind.IDENTIFIER),
    ), access=OperationalAccess.MUTATION, probe="bitrix.voucher.status"),
    _op("bitrix.voucher.execute_extension", "bitrix", "bitrix_voucher_ops.py", "execute-extension", arguments=(
        _arg("serial", maximum_chars=80), _arg("deal_id", kind=ArgumentKind.IDENTIFIER), _arg("slot", kind=ArgumentKind.INTEGER, minimum=1, maximum=8),
        _arg("new_valid_until", required=True, kind=ArgumentKind.DATE), _arg("case_id", required=True, kind=ArgumentKind.IDENTIFIER),
        _arg("requester", required=True, maximum_chars=240), _arg("reason", required=True, maximum_chars=2000),
        _arg("set_stage_payment", kind=ArgumentKind.BOOLEAN), _arg("no_auto_stage_from_expired", kind=ArgumentKind.BOOLEAN),
        _arg("request_id", required=True, kind=ArgumentKind.IDENTIFIER),
    ), access=OperationalAccess.MUTATION, probe="bitrix.voucher.status"),

    # Database and panel read boundaries.
    _op("skyvision.db.probe", "skyvision_db", "skyvision_db_readonly.py", "--db", "skyvisio_fp", "--query", "SELECT 1 AS operational_edge_ready", "--case-id", "case:operational-edge-readiness", "--requester", "operational-edge-readiness", "--purpose", "Credential-scoped read-only operational edge readiness", "--max-rows", "1", "--timeout-seconds", "10", "--sensitivity", "normal", timeout=30),
    _op("skyvision.db.query", "skyvision_db", "skyvision_db_readonly.py", arguments=(
        _arg("db", required=True, kind=ArgumentKind.CHOICE, choices=("skyvisio_fp", "skyvisio_laravel", "skyvisio_wp64")), _arg("query", required=True, kind=ArgumentKind.SQL, maximum_chars=MAX_SQL_CHARS),
        _arg("case_id", required=True, kind=ArgumentKind.IDENTIFIER), _arg("requester", required=True, maximum_chars=240),
        _arg("requester_id", kind=ArgumentKind.IDENTIFIER), _arg("purpose", required=True, maximum_chars=2000),
        _arg("max_rows", kind=ArgumentKind.INTEGER, minimum=1, maximum=1000), _arg("timeout_seconds", kind=ArgumentKind.INTEGER, minimum=1, maximum=30),
        _arg("sensitivity", kind=ArgumentKind.CHOICE, choices=("auto", "normal", "sensitive")), _arg("step_up_scope", maximum_chars=120),
        _arg("step_up_request_id", kind=ArgumentKind.IDENTIFIER),
    ), timeout=60, probe="skyvision.db.probe"),
    _op("skyvision.panel.status", "skyvision_panel", "skyvision_panel_ops_readonly.py", "status"),
    _op("skyvision.panel.check_voucher_user", "skyvision_panel", "skyvision_panel_ops_readonly.py", "check-voucher-user", arguments=(
        _arg("voucher", required=True, maximum_chars=80), _arg("case_id", required=True, kind=ArgumentKind.IDENTIFIER), _arg("requester", required=True, maximum_chars=240),
    )),
    _op("skyvision.panel.invoice_lookup", "skyvision_panel", "skyvision_panel_ops_readonly.py", "invoice-lookup", arguments=(
        _arg("invoice_id", kind=ArgumentKind.IDENTIFIER), _arg("order_id", kind=ArgumentKind.IDENTIFIER), _arg("case_id", required=True, kind=ArgumentKind.IDENTIFIER),
        _arg("requester", required=True, maximum_chars=240), _arg("requester_id", kind=ArgumentKind.IDENTIFIER), _arg("step_up_request_id", kind=ArgumentKind.IDENTIFIER),
    ), requires_any_of=(("invoice_id", "order_id"),)),

    # GitLab read/deploy and infrastructure observers.
    _op("skyvision.gitlab.status", "skyvision_gitlab", "skyvision_gitlab_readonly.py", "status"),
    _op("skyvision.gitlab.projects", "skyvision_gitlab", "skyvision_gitlab_readonly.py", "group-projects", arguments=(_arg("group", maximum_chars=240), _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=100))),
    _op("skyvision.gitlab.project", "skyvision_gitlab", "skyvision_gitlab_readonly.py", "project", arguments=(_arg("project", required=True, maximum_chars=240),)),
    _op("skyvision.gitlab.branches", "skyvision_gitlab", "skyvision_gitlab_readonly.py", "branches", arguments=(_arg("project", required=True, maximum_chars=240), _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=100))),
    _op("skyvision.gitlab.mrs", "skyvision_gitlab", "skyvision_gitlab_readonly.py", "mrs", arguments=(_arg("project", required=True, maximum_chars=240), _arg("state", kind=ArgumentKind.CHOICE, choices=("opened", "closed", "merged", "all")), _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=100))),
    _op("skyvision.gitlab.pipelines", "skyvision_gitlab", "skyvision_gitlab_readonly.py", "pipelines", arguments=(_arg("project", required=True, maximum_chars=240), _arg("ref", maximum_chars=240), _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=100))),
    _op("skyvision.gitlab.tree", "skyvision_gitlab", "skyvision_gitlab_readonly.py", "tree", arguments=(_arg("project", required=True, maximum_chars=240), _arg("path", maximum_chars=1000), _arg("ref", maximum_chars=240), _arg("limit", kind=ArgumentKind.INTEGER, minimum=1, maximum=100))),
    _op("skyvision.gitlab.file_stat", "skyvision_gitlab", "skyvision_gitlab_readonly.py", "file-stat", arguments=(_arg("project", required=True, maximum_chars=240), _arg("path", required=True, maximum_chars=1000), _arg("ref", maximum_chars=240))),
    _op("skyvision.deploy.status", "skyvision_gitlab", "skyvision_gitlab_prod_deploy.py", "status"),
    _op("skyvision.deploy.preflight", "skyvision_gitlab", "skyvision_gitlab_prod_deploy.py", "preflight", arguments=(_arg("sha", required=True, kind=ArgumentKind.SHA),), timeout=120),
    _op("skyvision.deploy.request_approval", "skyvision_gitlab", "skyvision_gitlab_prod_deploy.py", "request-prod-deploy-approval", arguments=(
        _arg("sha", required=True, kind=ArgumentKind.SHA), _arg("case_id", required=True, kind=ArgumentKind.IDENTIFIER),
        _arg("requester_discord_user_id", required=True, kind=ArgumentKind.IDENTIFIER), _arg("reason", required=True, maximum_chars=2000),
    ), access=OperationalAccess.MUTATION, timeout=120, probe="skyvision.deploy.status",
        available=False, blocker_code=WEBSITE_RELEASE_CONTRACT_BLOCKER,
        availability_requirement=WEBSITE_RELEASE_CONTRACT_REQUIREMENT),
    _op("skyvision.deploy.execute", "skyvision_gitlab", "skyvision_gitlab_prod_deploy.py", "execute-prod-deploy", arguments=(
        _arg("sha", required=True, kind=ArgumentKind.SHA), _arg("case_id", required=True, kind=ArgumentKind.IDENTIFIER),
        _arg("request_id", required=True, kind=ArgumentKind.IDENTIFIER), _arg("reason", required=True, maximum_chars=2000),
        _arg("monitor", kind=ArgumentKind.BOOLEAN), _arg("monitor_seconds", kind=ArgumentKind.INTEGER, minimum=60, maximum=3600),
    ), access=OperationalAccess.MUTATION, timeout=900, probe="skyvision.deploy.status",
        available=False, blocker_code=WEBSITE_RELEASE_CONTRACT_BLOCKER,
        availability_requirement=WEBSITE_RELEASE_CONTRACT_REQUIREMENT),
    _op("infra.contabo.observe", "infrastructure", "contabo_observer.py", "instances", timeout=120),
    _op("infra.alwyzon.observe", "infrastructure", "alwyzon_phoenix_observer.py", "status", timeout=120),
    _op("infra.watchtower.fast", "infrastructure", "devops_watchtower_phase1.py", "--mode", "fast", "--no-dispatch", timeout=240, probe="infra.contabo.observe", cron="7e4a90bdeff0"),
    _op("infra.watchtower.full", "infrastructure", "devops_watchtower_phase1.py", "--mode", "infra", "--no-dispatch", timeout=420, probe="infra.contabo.observe", cron="27f7f59fa0ca"),
    _op("infra.watchtower.hourly", "infrastructure", "devops_watchtower_phase1.py", "--mode", "hourly", "--no-dispatch", timeout=420, probe="infra.contabo.observe", cron="6faf380f3512"),
    _op("infra.watchtower.digest", "infrastructure", "devops_watchtower_phase1.py", "--mode", "digest", "--no-dispatch", timeout=420, probe="infra.contabo.observe", cron="90ac99d45130"),

    # Exact scheduled mechanical jobs. Unknown job IDs cannot reach these rows.
    _op("cron.canonical.heartbeat", "canonical", "cloud_heartbeat_writer_no_enforcement_gate.py", access=OperationalAccess.MECHANICAL, timeout=120, probe="cron.canonical.knowledge_all", cron="8d09136f7da5"),
    _op("cron.canonical.projections", "canonical", "canonical_brain_projections_v1_1_refresh.py", "--mode", "incremental", "--active-write", access=OperationalAccess.MECHANICAL, timeout=420, probe="cron.canonical.knowledge_all", cron="81ed8a3ea0d9"),
    _op("cron.canonical.routeback_violation", "canonical", "handoff_route_back_violation_alert_gate.py", access=OperationalAccess.MECHANICAL, timeout=120, probe="cron.canonical.knowledge_all", cron="58344347b373"),
    _op("cron.canonical.git_parity", "canonical", "cloud_local_git_drift_monitor.py", "--summary-only", "--report-always", access=OperationalAccess.MECHANICAL, timeout=180, probe="cron.canonical.knowledge_all", cron="a05143c24275"),
    _op("cron.canonical.knowledge_voucher", "canonical", "knowledge_artifact_sync_status.py", "--artifact", "skyvision-voucher-terms-invoicing", timeout=120, probe="cron.canonical.knowledge_all", cron="a77d64526f9a"),
    _op("cron.canonical.knowledge_all", "canonical", "knowledge_artifact_sync_status.py", timeout=120, cron="0d446b5df20c"),
    _op("cron.canonical.routeback_watchdog", "canonical", "free_hermes_route_back_state_watchdog.py", "--stale-minutes", "30", access=OperationalAccess.MECHANICAL, timeout=120, probe="cron.canonical.knowledge_all", cron="54976db7a384"),
    _op("cron.canonical.persistence_watchdog", "canonical", "free_hermes_operational_persistence_watchdog.py", access=OperationalAccess.MECHANICAL, timeout=120, probe="cron.canonical.knowledge_all", cron="d84d45a86b80"),
    _op("cron.skyvision.heart_weekly", "skyvision_db", "skyvision_from_heart_weekly_count.py", timeout=420, probe="skyvision.db.probe", cron="dee523e6f47b"),
    _op("cron.github.refs", "github", "gh-hermes", timeout=180, cron="06ef64d72891", runner="ops.muncho.runtime.github_refs_collector"),
)


CREDENTIALS_BY_DOMAIN: Mapping[str, tuple[CredentialBinding, ...]] = MappingProxyType(
    {
        "skyvision_email": (
            CredentialBinding("skyvision-email-env", HERMES_HOME / "secrets/skyvision_email_ops.env", HERMES_HOME / "secrets/skyvision_email_ops.env"),
        ),
        "adventico_email": (
            CredentialBinding("adventico-email-env", HERMES_HOME / "secrets/adventico_email_ops.env", HERMES_HOME / "secrets/adventico_email_ops.env"),
        ),
        "bitrix": (
            CredentialBinding("bitrix-webhook-url", HERMES_HOME / "secrets/bitrix_skyvision_crm_webhook.url", HERMES_HOME / "secrets/bitrix_skyvision_crm_webhook.url"),
        ),
        "skyvision_db": (
            CredentialBinding("skyvision-db-fp", HERMES_HOME / "secrets/skyvision_db_readonly_fp.env", HERMES_HOME / "secrets/skyvision_db_readonly_fp.env"),
            CredentialBinding("skyvision-db-laravel", HERMES_HOME / "secrets/skyvision_db_readonly_laravel.env", HERMES_HOME / "secrets/skyvision_db_readonly_laravel.env"),
            CredentialBinding("skyvision-db-wp64", HERMES_HOME / "secrets/skyvision_db_readonly_wp64.env", HERMES_HOME / "secrets/skyvision_db_readonly_wp64.env"),
            CredentialBinding("alwyzon-ssh-key", HERMES_HOME / "secrets/alwyzon_phoenix_ed25519", HERMES_HOME / "secrets/alwyzon_phoenix_ed25519"),
            CredentialBinding("alwyzon-ssh-env", HERMES_HOME / "secrets/alwyzon_phoenix_ssh.env", HERMES_HOME / "secrets/alwyzon_phoenix_ssh.env"),
        ),
        "skyvision_panel": (
            CredentialBinding("skyvision-db-fp", HERMES_HOME / "secrets/skyvision_db_readonly_fp.env", HERMES_HOME / "secrets/skyvision_db_readonly_fp.env"),
            CredentialBinding("alwyzon-ssh-key", HERMES_HOME / "secrets/alwyzon_phoenix_ed25519", HERMES_HOME / "secrets/alwyzon_phoenix_ed25519"),
            CredentialBinding("alwyzon-ssh-env", HERMES_HOME / "secrets/alwyzon_phoenix_ssh.env", HERMES_HOME / "secrets/alwyzon_phoenix_ssh.env"),
        ),
        "skyvision_gitlab": (
            CredentialBinding("skyvision-gitlab-group", HERMES_HOME / "secrets/skyvision_gitlab_group_ops.env", HERMES_HOME / "secrets/skyvision_gitlab_group_ops.env"),
            CredentialBinding("skyvision-gitlab-project", HERMES_HOME / "secrets/skyvision_gitlab_frontend_project_bot.env", HERMES_HOME / "secrets/skyvision_gitlab_frontend_project_bot.env"),
            CredentialBinding("skyvision-gitlab-group-home", HERMES_HOME / "secrets/skyvision_gitlab_group_ops.env", Path("/opt/adventico-ai-platform/.hermes/secrets/skyvision_gitlab_group_ops.env")),
            CredentialBinding("skyvision-gitlab-project-home", HERMES_HOME / "secrets/skyvision_gitlab_frontend_project_bot.env", Path("/opt/adventico-ai-platform/.hermes/secrets/skyvision_gitlab_frontend_project_bot.env")),
        ),
        "infrastructure": (
            CredentialBinding("contabo-api", HERMES_HOME / "secrets/contabo_api.env", HERMES_HOME / "secrets/contabo_api.env"),
            CredentialBinding("alwyzon-ssh-key", HERMES_HOME / "secrets/alwyzon_phoenix_ed25519", HERMES_HOME / "secrets/alwyzon_phoenix_ed25519"),
            CredentialBinding("alwyzon-ssh-env", HERMES_HOME / "secrets/alwyzon_phoenix_ssh.env", HERMES_HOME / "secrets/alwyzon_phoenix_ssh.env"),
            CredentialBinding("github-ops", HERMES_HOME / "secrets/github_ops.env", HERMES_HOME / "secrets/github_ops.env"),
        ),
        "github": (
            CredentialBinding("github-ops", HERMES_HOME / "secrets/github_ops.env", HERMES_HOME / "secrets/github_ops.env"),
        ),
        "canonical": (),
    }
)


def asset_catalog() -> Mapping[str, AssetSpec]:
    result = {item.asset_id: item for item in ASSETS}
    if len(result) != len(ASSETS):
        raise OperationalCatalogError("operational asset catalog contains duplicates")
    return MappingProxyType(result)


def operation_catalog() -> Mapping[str, OperationSpec]:
    assets = asset_catalog()
    result = {item.operation_id: item for item in OPERATIONS}
    if len(result) != len(OPERATIONS):
        raise OperationalCatalogError("operational catalog contains duplicates")
    if set(OPERATION_PURPOSES) != set(result):
        raise OperationalCatalogError(
            "operational purpose catalog must exactly cover every operation"
        )
    cron_ids: set[str] = set()
    for item in OPERATIONS:
        if item.asset_id not in assets or item.domain not in CREDENTIALS_BY_DOMAIN:
            raise OperationalCatalogError("operational catalog references an unknown asset/domain")
        if (
            item.purpose != OPERATION_PURPOSES[item.operation_id]
            or item.purpose != item.purpose.strip()
            or not 8 <= len(item.purpose) <= 240
        ):
            raise OperationalCatalogError("operational purpose text is invalid")
        if item.available:
            if item.blocker_code or item.availability_requirement:
                raise OperationalCatalogError(
                    "available operation contains an availability blocker"
                )
        elif (
            _ID.fullmatch(item.blocker_code) is None
            or item.availability_requirement != item.availability_requirement.strip()
            or not 8 <= len(item.availability_requirement) <= 500
        ):
            raise OperationalCatalogError(
                "unavailable operation is missing its static blocker contract"
            )
        if item.cron_source_job_id:
            if item.cron_source_job_id in cron_ids:
                raise OperationalCatalogError("operational cron mapping contains duplicates")
            cron_ids.add(item.cron_source_job_id)
        names = [argument.name for argument in item.arguments]
        flags = [argument.flag for argument in item.arguments]
        if len(names) != len(set(names)) or len(flags) != len(set(flags)):
            raise OperationalCatalogError("operational argument catalog contains duplicates")
        for group in item.requires_any_of:
            if (
                len(group) < 2
                or len(group) != len(set(group))
                or any(name not in names for name in group)
            ):
                raise OperationalCatalogError(
                    "operational any-of argument contract is invalid"
                )
    for item in OPERATIONS:
        if item.probe_operation_id:
            probe = result.get(item.probe_operation_id)
            if probe is None or probe.domain != item.domain or probe.access is not OperationalAccess.READ:
                raise OperationalCatalogError("operational probe binding is invalid")
    return MappingProxyType(result)


def required_cron_operations() -> Mapping[str, str]:
    result = {
        item.cron_source_job_id: item.operation_id
        for item in OPERATIONS
        if item.cron_source_job_id
    }
    return MappingProxyType(dict(sorted(result.items())))


def _scalar(argument: ArgumentSpec, value: Any) -> str | bool:
    if argument.kind is ArgumentKind.BOOLEAN:
        if type(value) is not bool:
            raise OperationalCatalogError(f"argument {argument.name} must be boolean")
        return value
    if argument.kind is ArgumentKind.INTEGER:
        if type(value) is not int:
            raise OperationalCatalogError(f"argument {argument.name} must be integer")
        number = float(value)
        rendered = str(value)
    elif argument.kind is ArgumentKind.NUMBER:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OperationalCatalogError(f"argument {argument.name} must be numeric")
        number = float(value)
        rendered = format(number, ".15g")
    else:
        if not isinstance(value, str) or not value or "\x00" in value or len(value) > argument.maximum_chars:
            raise OperationalCatalogError(f"argument {argument.name} is invalid")
        rendered = value
        number = None
    if number is not None and (
        argument.minimum is not None and number < argument.minimum
        or argument.maximum is not None and number > argument.maximum
    ):
        raise OperationalCatalogError(f"argument {argument.name} is out of range")
    if argument.kind is ArgumentKind.CHOICE and rendered not in argument.choices:
        raise OperationalCatalogError(f"argument {argument.name} is not an allowed choice")
    if argument.kind is ArgumentKind.IDENTIFIER and _ID.fullmatch(rendered) is None:
        raise OperationalCatalogError(f"argument {argument.name} is not an identifier")
    if argument.kind is ArgumentKind.DATE and _DATE.fullmatch(rendered) is None:
        raise OperationalCatalogError(f"argument {argument.name} is not a date")
    if argument.kind is ArgumentKind.SHA and _SHA.fullmatch(rendered) is None:
        raise OperationalCatalogError(f"argument {argument.name} is not a revision")
    if argument.kind is ArgumentKind.IP:
        try:
            ipaddress.ip_address(rendered)
        except ValueError as exc:
            raise OperationalCatalogError(f"argument {argument.name} is not an IP address") from exc
    if argument.kind is ArgumentKind.EMAIL_LIST and _EMAIL_LIST.fullmatch(rendered) is None:
        raise OperationalCatalogError(f"argument {argument.name} is not a bounded recipient list")
    if argument.kind is ArgumentKind.SQL and len(rendered) > MAX_SQL_CHARS:
        raise OperationalCatalogError(f"argument {argument.name} is too large")
    return rendered


def build_operation_argv(spec: OperationSpec, arguments: Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(arguments, Mapping) or any(not isinstance(key, str) for key in arguments):
        raise OperationalCatalogError("operation arguments must be an object")
    declared = {item.name: item for item in spec.arguments}
    if set(arguments) - set(declared):
        raise OperationalCatalogError("operation contains undeclared arguments")
    if any(item.required and item.name not in arguments for item in spec.arguments):
        raise OperationalCatalogError("operation is missing a required argument")
    for group in spec.requires_any_of:
        if not any(name in arguments and arguments[name] is not None for name in group):
            raise OperationalCatalogError(
                "operation requires at least one of: " + ", ".join(group)
            )
    result = list(spec.argv_prefix)
    for item in spec.arguments:
        if item.name not in arguments or arguments[item.name] is None:
            continue
        raw = arguments[item.name]
        values = raw if item.repeat else [raw]
        if item.repeat and (not isinstance(raw, list) or not 1 <= len(raw) <= 64):
            raise OperationalCatalogError(f"argument {item.name} must be a bounded list")
        for value in values:
            rendered = _scalar(item, value)
            if item.kind is ArgumentKind.BOOLEAN:
                if rendered is True:
                    result.append(item.flag)
            else:
                result.extend((item.flag, str(rendered)))
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 128 * 1024:
        raise OperationalCatalogError("operation argv is too large")
    return tuple(result)


def catalog_public_contract() -> Mapping[str, Any]:
    operations = []
    for item in OPERATIONS:
        operations.append(
            {
                "operation_id": item.operation_id,
                "purpose": item.purpose,
                "domain": item.domain,
                "access": item.access.value,
                "available": item.available,
                "blocker_code": item.blocker_code or None,
                "availability_requirement": item.availability_requirement or None,
                "asset_id": item.asset_id,
                "argv_prefix": list(item.argv_prefix),
                "arguments": [
                    {
                        "name": argument.name,
                        "flag": argument.flag,
                        "kind": argument.kind.value,
                        "required": argument.required,
                        "choices": list(argument.choices),
                        "minimum": argument.minimum,
                        "maximum": argument.maximum,
                        "maximum_chars": argument.maximum_chars,
                        "repeat": argument.repeat,
                    }
                    for argument in item.arguments
                ],
                "requires_any_of": [list(group) for group in item.requires_any_of],
                "timeout_seconds": item.timeout_seconds,
                "probe_operation_id": item.probe_operation_id,
                "expected_json": item.expected_json,
                "cron_source_job_id": item.cron_source_job_id,
                "runner_module": item.runner_module,
            }
        )
    return {
        "schema": CATALOG_SCHEMA,
        "operations": operations,
        "semantic_routing": False,
        "unknown_operation_fails_closed": True,
    }


__all__ = [
    "ASSETS",
    "ASSET_ROOT_RELATIVE",
    "CREDENTIALS_BY_DOMAIN",
    "CredentialBinding",
    "OPERATION_PURPOSES",
    "OperationSpec",
    "OperationalCatalogError",
    "WEBSITE_RELEASE_CONTRACT_BLOCKER",
    "WEBSITE_RELEASE_CONTRACT_REQUIREMENT",
    "asset_catalog",
    "build_operation_argv",
    "catalog_public_contract",
    "operation_catalog",
    "required_cron_operations",
]
