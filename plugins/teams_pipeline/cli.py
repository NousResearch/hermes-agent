"""CLI commands for the Teams meeting pipeline plugin."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hermes_constants import display_hermes_home
from gateway.config import Platform, load_gateway_config
from plugins.teams_pipeline.meetings import (
    enrich_meeting_with_call_record,
    fetch_preferred_transcript_text,
    list_recording_artifacts,
    resolve_meeting_reference,
)
from plugins.teams_pipeline.models import GraphSubscription
from plugins.teams_pipeline.pipeline import TeamsMeetingPipeline
from plugins.teams_pipeline.store import TeamsPipelineStore, resolve_teams_pipeline_store_path
from plugins.teams_pipeline.subscriptions import (
    build_graph_client,
    maintain_graph_subscriptions,
    sync_graph_subscription_record,
)
from tools.microsoft_graph_auth import MicrosoftGraphConfigError, MicrosoftGraphTokenProvider


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="teams_pipeline_action")

    list_p = subs.add_parser("list", aliases=["ls"], help="List recent Teams pipeline jobs")
    list_p.add_argument("--limit", type=int, default=20)
    list_p.add_argument("--status", default="")
    list_p.add_argument("--store-path", default="")

    show_p = subs.add_parser("show", help="Show a stored Teams pipeline job")
    show_p.add_argument("job_id")
    show_p.add_argument("--store-path", default="")

    run_p = subs.add_parser("run", aliases=["replay"], help="Replay a stored Teams pipeline job")
    run_p.add_argument("job_id")
    run_p.add_argument("--store-path", default="")

    fetch_p = subs.add_parser("fetch", aliases=["test"], help="Dry-run meeting artifact resolution")
    fetch_p.add_argument("--meeting-id", default="")
    fetch_p.add_argument("--join-web-url", default="")
    fetch_p.add_argument("--tenant-id", default="")
    fetch_p.add_argument("--call-record-id", default="")

    subs_p = subs.add_parser("subscriptions", aliases=["subs"], help="List Graph subscriptions")
    subs_p.add_argument("--store-path", default="")

    sub_p = subs.add_parser("subscribe", help="Create a Microsoft Graph subscription")
    sub_p.add_argument("--resource", required=True)
    sub_p.add_argument("--notification-url", required=True)
    sub_p.add_argument("--change-type", default="")
    sub_p.add_argument("--expiration", default="")
    sub_p.add_argument("--client-state", default="")
    sub_p.add_argument("--lifecycle-notification-url", default="")
    sub_p.add_argument("--latest-supported-tls-version", default="v1_2")
    sub_p.add_argument("--store-path", default="")

    renew_p = subs.add_parser("renew-subscription", help="Renew a Microsoft Graph subscription")
    renew_p.add_argument("subscription_id")
    renew_p.add_argument("--expiration", required=True)
    renew_p.add_argument("--store-path", default="")

    delete_p = subs.add_parser("delete-subscription", help="Delete a Microsoft Graph subscription")
    delete_p.add_argument("subscription_id")
    delete_p.add_argument("--store-path", default="")

    maintain_p = subs.add_parser("maintain-subscriptions", help="Renew near-expiry managed subscriptions")
    maintain_p.add_argument("--renew-within-hours", type=int, default=24)
    maintain_p.add_argument("--extend-hours", type=int, default=24)
    maintain_p.add_argument("--dry-run", action="store_true")
    maintain_p.add_argument("--store-path", default="")
    maintain_p.add_argument("--client-state", default="")

    token_p = subs.add_parser("token-health", aliases=["token"], help="Inspect Graph token health")
    token_p.add_argument("--force-refresh", action="store_true")

    validate_p = subs.add_parser("validate", help="Validate Teams pipeline configuration snapshot")
    validate_p.add_argument("--store-path", default="")

    subparser.set_defaults(func=teams_pipeline_command)


def teams_pipeline_command(args: argparse.Namespace) -> int:
    action = getattr(args, "teams_pipeline_action", None)
    if not action:
        print(
            "Usage: hermes teams-pipeline "
            "{list|show|run|fetch|subscriptions|subscribe|renew-subscription|delete-subscription|maintain-subscriptions|token-health|validate}"
        )
        return 2

    try:
        if action in ("list", "ls"):
            _cmd_list(args)
        elif action == "show":
            _cmd_show(args)
        elif action in ("run", "replay"):
            _cmd_run(args)
        elif action in ("fetch", "test"):
            _cmd_fetch(args)
        elif action in ("subscriptions", "subs"):
            _cmd_subscriptions(args)
        elif action == "subscribe":
            _cmd_subscribe(args)
        elif action == "renew-subscription":
            _cmd_renew_subscription(args)
        elif action == "delete-subscription":
            _cmd_delete_subscription(args)
        elif action == "maintain-subscriptions":
            _cmd_maintain_subscriptions(args)
        elif action in ("token-health", "token"):
            _cmd_token_health(args)
        elif action == "validate":
            _cmd_validate(args)
        else:
            print(f"Unknown teams-pipeline action: {action}")
            return 2
        return 0
    except MicrosoftGraphConfigError:
        print(_graph_setup_hint())
        return 1


def _run_async(coro):
    return asyncio.run(coro)


def _store_path(path_arg: str | None) -> Path:
    return resolve_teams_pipeline_store_path(path_arg)


def _graph_setup_hint() -> str:
    return f"""
  Microsoft Graph is not configured. Add these to {display_hermes_home()}/.env:

    MSGRAPH_TENANT_ID=...
    MSGRAPH_CLIENT_ID=...
    MSGRAPH_CLIENT_SECRET=...

  Then restart the gateway or rerun this command.
"""


def _iso_utc_timestamp(hours_from_now: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")


def _default_change_type_for_resource(resource: str) -> str:
    normalized = str(resource or "").strip().lower()
    if normalized.startswith("communications/onlinemeetings/getalltranscripts"):
        return "created"
    if normalized.startswith("communications/onlinemeetings/getallrecordings"):
        return "created"
    if normalized.startswith("communications/callrecords"):
        return "created"
    return "updated"


def _compact_job(job: dict) -> dict:
    payload = dict(job)
    summary = dict(payload.get("summary_payload") or {})
    transcript = summary.pop("transcript_text", None)
    if transcript:
        summary["transcript_preview"] = str(transcript)[:240]
    payload["summary_payload"] = summary or None
    return payload


def _sync_subscription_record(
    store: TeamsPipelineStore,
    subscription_payload: dict[str, Any],
    *,
    status: str = "active",
    renewed: bool = False,
) -> dict[str, Any]:
    normalized = GraphSubscription.from_dict(subscription_payload).to_dict()
    normalized["status"] = status
    if renewed:
        normalized["latest_renewal_at"] = _iso_utc_timestamp(0)
    return store.upsert_subscription(normalized["subscription_id"], normalized)


def _validate_configuration_snapshot(store: TeamsPipelineStore) -> dict[str, Any]:
    env = os.environ
    issues: list[str] = []
    warnings: list[str] = []
    gateway_config = load_gateway_config()
    webhook_config = gateway_config.platforms.get(Platform.MSGRAPH_WEBHOOK)
    teams_config = gateway_config.platforms.get(Platform("teams"))

    graph = {
        "tenant_id": bool(env.get("MSGRAPH_TENANT_ID")),
        "client_id": bool(env.get("MSGRAPH_CLIENT_ID")),
        "client_secret": bool(env.get("MSGRAPH_CLIENT_SECRET")),
    }
    webhook_enabled = bool(webhook_config and webhook_config.enabled)
    teams_enabled = bool(teams_config and teams_config.enabled)
    teams_extra = dict((teams_config.extra or {}) if teams_config else {})
    teams_mode = str(teams_extra.get("delivery_mode") or "").strip() or None

    if not all(graph.values()):
        issues.append("Microsoft Graph app-only credentials are incomplete.")
    if not webhook_enabled:
        issues.append("MSGRAPH_WEBHOOK_ENABLED is not enabled.")
    if not teams_enabled:
        warnings.append("Teams outbound delivery is disabled.")
    elif teams_mode == "incoming_webhook":
        if not teams_extra.get("incoming_webhook_url"):
            issues.append("TEAMS_INCOMING_WEBHOOK_URL is required for incoming_webhook mode.")
    elif teams_mode == "graph":
        missing: list[str] = []
        has_graph_delivery_token = bool(
            (teams_config.token if teams_config else "") or teams_extra.get("access_token")
        )
        has_graph_app_credentials = all(graph.values())
        if not has_graph_delivery_token and not has_graph_app_credentials:
            missing.append(
                "TEAMS_GRAPH_ACCESS_TOKEN or complete MSGRAPH_* app credentials"
            )
        if not teams_extra.get("team_id"):
            missing.append("TEAMS_TEAM_ID")
        channel_id = teams_extra.get("channel_id") or teams_extra.get("chat_id")
        if not channel_id and not (teams_config and teams_config.home_channel):
            missing.append("TEAMS_CHANNEL_ID")
        for key in missing:
            issues.append(f"{key} is required for graph delivery mode.")
    else:
        warnings.append("TEAMS_DELIVERY_MODE is not set.")

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "graph_config": graph,
        "webhook_enabled": webhook_enabled,
        "teams_enabled": teams_enabled,
        "teams_delivery_mode": teams_mode,
        "store_path": str(store.path),
        "store_stats": store.stats(),
    }


def _cmd_list(args) -> None:
    store = TeamsPipelineStore(_store_path(getattr(args, "store_path", None)))
    jobs = list(store.list_jobs().values())
    status = str(getattr(args, "status", "") or "").strip().lower()
    if status:
        jobs = [job for job in jobs if str(job.get("status") or "").lower() == status]
    jobs.sort(key=lambda item: str((item or {}).get("updated_at") or ""), reverse=True)
    limit = max(1, min(int(getattr(args, "limit", 20) or 20), 100))
    jobs = jobs[:limit]

    if not jobs:
        print("No Teams meeting pipeline jobs found.")
        return

    print(f"\n{len(jobs)} Teams pipeline job(s):\n")
    for job in jobs:
        meeting_id = ((job.get("meeting_ref") or {}).get("meeting_id") or "unknown")
        print(f"  ◆ {job.get('job_id')}")
        print(f"    status: {job.get('status')}")
        print(f"    meeting: {meeting_id}")
        if job.get("selected_artifact_strategy"):
            print(f"    strategy: {job.get('selected_artifact_strategy')}")
        if job.get("updated_at"):
            print(f"    updated: {job.get('updated_at')}")
        if job.get("error_info"):
            print(f"    error: {job.get('error_info')}")
        print()


def _cmd_show(args) -> None:
    job_id = str(getattr(args, "job_id", "") or "").strip()
    if not job_id:
        print("job_id is required")
        return
    store = TeamsPipelineStore(_store_path(getattr(args, "store_path", None)))
    job = store.get_job(job_id)
    if not job:
        print(f"Unknown job: {job_id}")
        return
    print(json.dumps(_compact_job(job), indent=2, sort_keys=True))


def _cmd_run(args) -> None:
    job_id = str(getattr(args, "job_id", "") or "").strip()
    if not job_id:
        print("job_id is required")
        return
    store = TeamsPipelineStore(_store_path(getattr(args, "store_path", None)))
    pipeline = TeamsMeetingPipeline(graph_client=build_graph_client(), store=store, config={})
    result = _run_async(pipeline.run_job(job_id))
    print(json.dumps(_compact_job(result.to_dict()), indent=2, sort_keys=True))


def _cmd_fetch(args) -> None:
    meeting_id = str(getattr(args, "meeting_id", "") or "").strip() or None
    join_web_url = str(getattr(args, "join_web_url", "") or "").strip() or None
    tenant_id = str(getattr(args, "tenant_id", "") or "").strip() or None
    call_record_id = str(getattr(args, "call_record_id", "") or "").strip() or None
    if not meeting_id and not join_web_url:
        print("meeting_id or join_web_url is required")
        return

    client = build_graph_client()
    meeting_ref = _run_async(
        resolve_meeting_reference(
            client,
            meeting_id=meeting_id,
            join_web_url=join_web_url,
            tenant_id=tenant_id,
        )
    )
    transcript_artifact, transcript_text = _run_async(fetch_preferred_transcript_text(client, meeting_ref))
    recordings = _run_async(list_recording_artifacts(client, meeting_ref))
    call_record = _run_async(
        enrich_meeting_with_call_record(client, meeting_ref, call_record_id=call_record_id)
    )
    print(
        json.dumps(
            {
                "meeting_ref": meeting_ref.to_dict(),
                "transcript_available": bool(transcript_artifact and transcript_text),
                "transcript_artifact": transcript_artifact.to_dict() if transcript_artifact else None,
                "transcript_preview": (transcript_text or "")[:240] or None,
                "recording_count": len(recordings),
                "recordings": [recording.to_dict() for recording in recordings[:5]],
                "call_record": call_record.to_dict() if call_record else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _cmd_subscriptions(args) -> None:
    store = TeamsPipelineStore(_store_path(getattr(args, "store_path", None)))
    client = build_graph_client()
    subscriptions = _run_async(client.collect_paginated("/subscriptions"))
    for sub in subscriptions:
        try:
            _sync_subscription_record(store, sub, status="active")
        except Exception:
            continue
    if not subscriptions:
        print("No Microsoft Graph subscriptions found.")
        return

    print(f"\n{len(subscriptions)} Microsoft Graph subscription(s):\n")
    for sub in subscriptions:
        print(f"  ◆ {sub.get('id') or 'unknown'}")
        print(f"    resource: {sub.get('resource') or 'unknown'}")
        print(f"    changeType: {sub.get('changeType') or 'unknown'}")
        if sub.get("expirationDateTime"):
            print(f"    expires: {sub.get('expirationDateTime')}")
        if sub.get("notificationUrl"):
            print(f"    notify: {sub.get('notificationUrl')}")
        print()


def _cmd_subscribe(args) -> None:
    store = TeamsPipelineStore(_store_path(getattr(args, "store_path", None)))
    resource = str(getattr(args, "resource", "") or "").strip()
    notification_url = str(getattr(args, "notification_url", "") or "").strip()
    change_type = str(getattr(args, "change_type", "") or "").strip() or _default_change_type_for_resource(resource)
    expiration = str(getattr(args, "expiration", "") or "").strip() or _iso_utc_timestamp(1)
    client_state = str(getattr(args, "client_state", "") or "").strip()
    lifecycle_url = str(getattr(args, "lifecycle_notification_url", "") or "").strip()
    tls_version = str(getattr(args, "latest_supported_tls_version", "") or "").strip() or "v1_2"

    payload = {
        "changeType": change_type,
        "notificationUrl": notification_url,
        "resource": resource,
        "expirationDateTime": expiration,
        "latestSupportedTlsVersion": tls_version,
    }
    if client_state:
        payload["clientState"] = client_state
    if lifecycle_url:
        payload["lifecycleNotificationUrl"] = lifecycle_url

    result = _run_async(build_graph_client().post_json("/subscriptions", json_body=payload))
    _sync_subscription_record(store, result, status="active")
    print(json.dumps(result, indent=2, sort_keys=True))


def _cmd_renew_subscription(args) -> None:
    subscription_id = str(getattr(args, "subscription_id", "") or "").strip()
    expiration = str(getattr(args, "expiration", "") or "").strip()
    if not subscription_id or not expiration:
        print("subscription_id and --expiration are required")
        return

    store = TeamsPipelineStore(_store_path(getattr(args, "store_path", None)))
    result = _run_async(
        build_graph_client().patch_json(
            f"/subscriptions/{subscription_id}",
            json_body={"expirationDateTime": expiration},
        )
    )
    merged = {"id": subscription_id, **(result or {}), "expirationDateTime": expiration}
    _sync_subscription_record(store, merged, status="active", renewed=True)
    print(json.dumps(merged, indent=2, sort_keys=True))


def _cmd_delete_subscription(args) -> None:
    subscription_id = str(getattr(args, "subscription_id", "") or "").strip()
    if not subscription_id:
        print("subscription_id is required")
        return
    store = TeamsPipelineStore(_store_path(getattr(args, "store_path", None)))
    result = _run_async(build_graph_client().delete(f"/subscriptions/{subscription_id}"))
    store.delete_subscription(subscription_id)
    print(json.dumps({"subscription_id": subscription_id, "result": result}, indent=2, sort_keys=True))


def _cmd_maintain_subscriptions(args) -> None:
    store = TeamsPipelineStore(_store_path(getattr(args, "store_path", None)))
    result = _run_async(
        maintain_graph_subscriptions(
            client=build_graph_client(),
            store=store,
            renew_within_hours=int(getattr(args, "renew_within_hours", 24) or 24),
            extend_hours=int(getattr(args, "extend_hours", 24) or 24),
            dry_run=bool(getattr(args, "dry_run", False)),
            client_state=str(getattr(args, "client_state", "") or "").strip() or None,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _cmd_token_health(args) -> None:
    provider = MicrosoftGraphTokenProvider.from_env()
    health = provider.inspect_token_health()
    payload = dict(health)
    if getattr(args, "force_refresh", False):
        try:
            token = _run_async(provider.get_access_token(force_refresh=True))
            payload["last_refresh_succeeded"] = True
            payload["access_token_length"] = len(token or "")
        except Exception as exc:
            payload["last_refresh_succeeded"] = False
            payload["refresh_error"] = str(exc)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_validate(args) -> None:
    store = TeamsPipelineStore(_store_path(getattr(args, "store_path", None)))
    snapshot = _validate_configuration_snapshot(store)
    print(json.dumps(snapshot, indent=2, sort_keys=True))
