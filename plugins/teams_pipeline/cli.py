"""CLI commands for the Teams meeting pipeline plugin."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Callable

from hermes_constants import display_hermes_home
from gateway.config import Platform, load_gateway_config
from plugins.teams_pipeline.meetings import (
    enrich_meeting_with_call_record, fetch_preferred_transcript_text, list_recording_artifacts, resolve_meeting_reference)
from plugins.teams_pipeline.pipeline import TeamsMeetingPipeline
from plugins.teams_pipeline.store import TeamsPipelineStore, resolve_teams_pipeline_store_path
from plugins.teams_pipeline.subscriptions import (
    build_graph_client, maintain_graph_subscriptions, sync_graph_subscription_record, utc_timestamp)
from tools.microsoft_graph_auth import MicrosoftGraphConfigError, MicrosoftGraphTokenProvider


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="teams_pipeline_action")
    for name, aliases, help_text, options, _handler in _SUBCOMMANDS:
        parser = subs.add_parser(name, aliases=aliases, help=help_text)
        for flag, kwargs in options:
            parser.add_argument(flag, **kwargs)
    subparser.set_defaults(func=teams_pipeline_command)


def teams_pipeline_command(args: argparse.Namespace) -> int:
    action = getattr(args, "teams_pipeline_action", None)
    if not action:
        print(f"Usage: hermes teams-pipeline {{{'|'.join(spec[0] for spec in _SUBCOMMANDS)}}}")
        return 2
    handler = _ACTIONS.get(action)
    if handler is None:
        print(f"Unknown teams-pipeline action: {action}")
        return 2
    required, missing_message = _REQUIRED_ARGS.get(handler, ((), ""))
    if not all(_text(args, name) for name in required):
        print(missing_message)
        return 0
    try:
        handler(args)
        return 0
    except MicrosoftGraphConfigError:
        print(_graph_setup_hint())
        return 1


def _text(args: argparse.Namespace, name: str) -> str:
    """Stripped string form of an optional CLI arg ('' when absent/None)."""
    return str(getattr(args, name, "") or "").strip()


def _int_arg(args: argparse.Namespace, name: str, default: int) -> int:
    return int(getattr(args, name, default) or default)


def _open_store(args: argparse.Namespace) -> TeamsPipelineStore:
    return TeamsPipelineStore(resolve_teams_pipeline_store_path(getattr(args, "store_path", None)))


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _print_records(noun: str, empty_message: str, records: list[tuple[Any, list[tuple[str, Any, bool]]]]) -> None:
    """Print bulleted records; fields are (label, value, required) — optional fields skip falsy values."""
    if not records:
        print(empty_message)
        return
    print(f"\n{len(records)} {noun}:\n")
    for title, fields in records:
        print(f"  ◆ {title}")
        for label, value, required in fields:
            if required or value:
                print(f"    {label}: {value}")
        print()


def _graph_setup_hint() -> str:
    return f"""
  Microsoft Graph is not configured. Add these to {display_hermes_home()}/.env:

    MSGRAPH_TENANT_ID=...
    MSGRAPH_CLIENT_ID=...
    MSGRAPH_CLIENT_SECRET=...

  Then restart the gateway or rerun this command.
"""


# Graph only emits "created" for the artifact/callRecord collection resources; meetings use "updated".
_CREATED_RESOURCE_PREFIXES = (
    "communications/onlinemeetings/getalltranscripts", "communications/onlinemeetings/getallrecordings", "communications/callrecords")


def _default_change_type_for_resource(resource: str) -> str:
    return "created" if str(resource or "").strip().lower().startswith(_CREATED_RESOURCE_PREFIXES) else "updated"


def _compact_job(job: dict) -> dict:
    payload = dict(job)
    summary = dict(payload.get("summary_payload") or {})
    if transcript := summary.pop("transcript_text", None):
        summary["transcript_preview"] = str(transcript)[:240]
    payload["summary_payload"] = summary or None
    return payload


def _cmd_validate(args) -> None:
    store = _open_store(args)
    issues: list[str] = []
    warnings: list[str] = []
    gateway_config = load_gateway_config()
    webhook_config = gateway_config.platforms.get(Platform.MSGRAPH_WEBHOOK)
    teams_config = gateway_config.platforms.get(Platform("teams"))
    graph = {key: bool(os.environ.get(f"MSGRAPH_{key.upper()}")) for key in ("tenant_id", "client_id", "client_secret")}
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
        # Graph delivery can authenticate with either a dedicated delivery token or the app-only creds.
        if not (teams_config.token or teams_extra.get("access_token")) and not all(graph.values()):
            issues.append("TEAMS_GRAPH_ACCESS_TOKEN or complete MSGRAPH_* app credentials is required for graph delivery mode.")
        if not teams_extra.get("team_id"):
            issues.append("TEAMS_TEAM_ID is required for graph delivery mode.")
        if not (teams_extra.get("channel_id") or teams_extra.get("chat_id") or teams_config.home_channel):
            issues.append("TEAMS_CHANNEL_ID is required for graph delivery mode.")
    else:
        warnings.append("TEAMS_DELIVERY_MODE is not set.")
    _print_json({
        "ok": not issues, "issues": issues, "warnings": warnings, "graph_config": graph,
        "webhook_enabled": webhook_enabled, "teams_enabled": teams_enabled, "teams_delivery_mode": teams_mode,
        "store_path": str(store.path), "store_stats": store.stats()})


def _cmd_list(args) -> None:
    jobs = list(_open_store(args).list_jobs().values())
    if status := _text(args, "status").lower():
        jobs = [job for job in jobs if str(job.get("status") or "").lower() == status]
    jobs.sort(key=lambda item: str((item or {}).get("updated_at") or ""), reverse=True)
    jobs = jobs[: max(1, min(_int_arg(args, "limit", 20), 100))]
    _print_records("Teams pipeline job(s)", "No Teams meeting pipeline jobs found.", [
        (job.get("job_id"), [
            ("status", job.get("status"), True),
            ("meeting", (job.get("meeting_ref") or {}).get("meeting_id") or "unknown", True),
            ("strategy", job.get("selected_artifact_strategy"), False),
            ("updated", job.get("updated_at"), False),
            ("error", job.get("error_info"), False)])
        for job in jobs])


def _cmd_show(args) -> None:
    job_id = _text(args, "job_id")
    job = _open_store(args).get_job(job_id)
    if not job:
        print(f"Unknown job: {job_id}")
        return
    _print_json(_compact_job(job))


def _cmd_run(args) -> None:
    pipeline = TeamsMeetingPipeline(graph_client=build_graph_client(), store=_open_store(args), config={})
    _print_json(_compact_job(asyncio.run(pipeline.run_job(_text(args, "job_id"))).to_dict()))


def _cmd_fetch(args) -> None:
    meeting_id = _text(args, "meeting_id") or None
    join_web_url = _text(args, "join_web_url") or None
    if not meeting_id and not join_web_url:
        print("meeting_id or join_web_url is required")
        return
    client = build_graph_client()
    meeting_ref = asyncio.run(resolve_meeting_reference(
        client, meeting_id=meeting_id, join_web_url=join_web_url,
        tenant_id=_text(args, "tenant_id") or None, organizer_user_id=_text(args, "organizer_user_id") or None))
    transcript_artifact, transcript_text = asyncio.run(fetch_preferred_transcript_text(client, meeting_ref))
    recordings = asyncio.run(list_recording_artifacts(client, meeting_ref))
    call_record = asyncio.run(enrich_meeting_with_call_record(client, meeting_ref, call_record_id=_text(args, "call_record_id") or None))
    _print_json({
        "meeting_ref": meeting_ref.to_dict(),
        "transcript_available": bool(transcript_artifact and transcript_text),
        "transcript_artifact": transcript_artifact.to_dict() if transcript_artifact else None,
        "transcript_preview": (transcript_text or "")[:240] or None,
        "recording_count": len(recordings),
        "recordings": [recording.to_dict() for recording in recordings[:5]],
        "call_record": call_record.to_dict() if call_record else None})


def _cmd_subscriptions(args) -> None:
    store = _open_store(args)
    subscriptions = asyncio.run(build_graph_client().collect_paginated("/subscriptions"))
    for sub in subscriptions:
        try:
            sync_graph_subscription_record(store, sub, status="active")
        except Exception:
            continue
    _print_records("Microsoft Graph subscription(s)", "No Microsoft Graph subscriptions found.", [
        (sub.get("id") or "unknown", [
            ("resource", sub.get("resource") or "unknown", True),
            ("changeType", sub.get("changeType") or "unknown", True),
            ("expires", sub.get("expirationDateTime"), False),
            ("notify", sub.get("notificationUrl"), False)])
        for sub in subscriptions])


def _cmd_subscribe(args) -> None:
    store = _open_store(args)
    resource = _text(args, "resource")
    payload = {
        "changeType": _text(args, "change_type") or _default_change_type_for_resource(resource),
        "notificationUrl": _text(args, "notification_url"),
        "resource": resource,
        "expirationDateTime": _text(args, "expiration") or utc_timestamp(1),
        "latestSupportedTlsVersion": _text(args, "latest_supported_tls_version") or "v1_2"}
    if client_state := _text(args, "client_state"):
        payload["clientState"] = client_state
    if lifecycle_url := _text(args, "lifecycle_notification_url"):
        payload["lifecycleNotificationUrl"] = lifecycle_url
    result = asyncio.run(build_graph_client().post_json("/subscriptions", json_body=payload))
    sync_graph_subscription_record(store, result, status="active")
    _print_json(result)


def _cmd_renew_subscription(args) -> None:
    subscription_id = _text(args, "subscription_id")
    expiration = _text(args, "expiration")
    store = _open_store(args)
    result = asyncio.run(build_graph_client().patch_json(f"/subscriptions/{subscription_id}", json_body={"expirationDateTime": expiration}))
    merged = {"id": subscription_id, **(result or {}), "expirationDateTime": expiration}
    sync_graph_subscription_record(store, merged, status="active", renewed=True)
    _print_json(merged)


def _cmd_delete_subscription(args) -> None:
    subscription_id = _text(args, "subscription_id")
    store = _open_store(args)
    result = asyncio.run(build_graph_client().delete(f"/subscriptions/{subscription_id}"))
    store.delete_subscription(subscription_id)
    _print_json({"subscription_id": subscription_id, "result": result})


def _cmd_maintain_subscriptions(args) -> None:
    _print_json(asyncio.run(maintain_graph_subscriptions(
        client=build_graph_client(), store=_open_store(args),
        renew_within_hours=_int_arg(args, "renew_within_hours", 24), extend_hours=_int_arg(args, "extend_hours", 24),
        dry_run=bool(getattr(args, "dry_run", False)), client_state=_text(args, "client_state") or None)))


def _cmd_token_health(args) -> None:
    provider = MicrosoftGraphTokenProvider.from_env()
    payload = dict(provider.inspect_token_health())
    if getattr(args, "force_refresh", False):
        try:
            token = asyncio.run(provider.get_access_token(force_refresh=True))
            payload["last_refresh_succeeded"] = True
            payload["access_token_length"] = len(token or "")
        except Exception as exc:
            payload["last_refresh_succeeded"] = False
            payload["refresh_error"] = str(exc)
    _print_json(payload)


def _opt(flag: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    return flag, kwargs


_STORE_PATH = _opt("--store-path", default="")
_EMPTY = {"default": ""}

# Single source of truth for subcommands: (name, aliases, help, arguments, handler).
# Order defines both the --help listing and the usage string.
_SUBCOMMANDS: list[tuple[str, list[str], str, list[tuple[str, dict[str, Any]]], Callable[[Any], None]]] = [
    ("list", ["ls"], "List recent Teams pipeline jobs",
     [_opt("--limit", type=int, default=20), _opt("--status", **_EMPTY), _STORE_PATH], _cmd_list),
    ("show", [], "Show a stored Teams pipeline job", [_opt("job_id"), _STORE_PATH], _cmd_show),
    ("run", ["replay"], "Replay a stored Teams pipeline job", [_opt("job_id"), _STORE_PATH], _cmd_run),
    ("fetch", ["test"], "Dry-run meeting artifact resolution",
     [_opt("--meeting-id", **_EMPTY), _opt("--join-web-url", **_EMPTY),
      _opt("--organizer-user-id", default="", help="Microsoft Entra user ID for organizer-scoped online meeting lookup"),
      _opt("--tenant-id", **_EMPTY), _opt("--call-record-id", **_EMPTY)], _cmd_fetch),
    ("subscriptions", ["subs"], "List Graph subscriptions", [_STORE_PATH], _cmd_subscriptions),
    ("subscribe", [], "Create a Microsoft Graph subscription",
     [_opt("--resource", required=True), _opt("--notification-url", required=True), _opt("--change-type", **_EMPTY),
      _opt("--expiration", **_EMPTY), _opt("--client-state", **_EMPTY), _opt("--lifecycle-notification-url", **_EMPTY),
      _opt("--latest-supported-tls-version", default="v1_2"), _STORE_PATH], _cmd_subscribe),
    ("renew-subscription", [], "Renew a Microsoft Graph subscription",
     [_opt("subscription_id"), _opt("--expiration", required=True), _STORE_PATH], _cmd_renew_subscription),
    ("delete-subscription", [], "Delete a Microsoft Graph subscription", [_opt("subscription_id"), _STORE_PATH], _cmd_delete_subscription),
    ("maintain-subscriptions", [], "Renew near-expiry managed subscriptions",
     [_opt("--renew-within-hours", type=int, default=24), _opt("--extend-hours", type=int, default=24),
      _opt("--dry-run", action="store_true"), _STORE_PATH, _opt("--client-state", **_EMPTY)], _cmd_maintain_subscriptions),
    ("token-health", ["token"], "Inspect Graph token health", [_opt("--force-refresh", action="store_true")], _cmd_token_health),
    ("validate", [], "Validate Teams pipeline configuration snapshot", [_STORE_PATH], _cmd_validate)]

_ACTIONS = {alias: handler for name, aliases, _help, _options, handler in _SUBCOMMANDS for alias in (name, *aliases)}

# Handlers whose positional/required args are validated up front (all must be non-blank), with the message printed.
_REQUIRED_ARGS: dict[Callable[[Any], None], tuple[tuple[str, ...], str]] = {
    _cmd_show: (("job_id",), "job_id is required"),
    _cmd_run: (("job_id",), "job_id is required"),
    _cmd_renew_subscription: (("subscription_id", "expiration"), "subscription_id and --expiration are required"),
    _cmd_delete_subscription: (("subscription_id",), "subscription_id is required")}


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402
from datetime import datetime  # noqa: F401,E402
from datetime import timedelta  # noqa: F401,E402
from datetime import timezone  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'GraphSubscription': ('plugins.teams_pipeline.models', 'GraphSubscription'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
