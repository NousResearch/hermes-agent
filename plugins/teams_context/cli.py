"""CLI commands for Teams chat context capture."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from plugins.teams_context.graph import TeamsContextGraph
from plugins.teams_context.store import TeamsContextStore
from tools.microsoft_graph_auth import GraphCredentials, MicrosoftGraphTokenProvider
from tools.microsoft_graph_client import MicrosoftGraphClient


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="teams_context_action")

    backfill = subs.add_parser("backfill", help="Backfill messages from a Teams chat")
    backfill.add_argument("--chat-id", required=True)
    backfill.add_argument("--since-days", type=int, default=30)
    backfill.add_argument("--limit", type=int, default=0)
    backfill.add_argument("--store-path", default="")

    subscribe = subs.add_parser("subscribe", help="Create a Graph subscription for one chat")
    subscribe.add_argument("--chat-id", required=True)
    subscribe.add_argument("--notification-url", required=True)
    subscribe.add_argument("--client-state", default="")
    subscribe.add_argument("--expiration", default="")
    subscribe.add_argument("--change-type", default="created,updated,deleted")
    subscribe.add_argument("--lifecycle-notification-url", default="")
    subscribe.add_argument("--store-path", default="")

    subs_p = subs.add_parser("subscriptions", aliases=["subs"], help="List stored chat subscriptions")
    subs_p.add_argument("--store-path", default="")

    maintain = subs.add_parser("maintain-subscriptions", help="Renew near-expiry chat subscriptions")
    maintain.add_argument("--renew-within-hours", type=int, default=24)
    maintain.add_argument("--extend-hours", type=int, default=24)
    maintain.add_argument("--dry-run", action="store_true")
    maintain.add_argument("--store-path", default="")

    validate = subs.add_parser("validate", help="Validate Teams context local configuration")
    validate.add_argument("--store-path", default="")

    scrape = subs.add_parser("scrape-ui", help="Scrape visible Teams channel messages from an open Chrome tab")
    scrape.add_argument("--label", required=True)
    scrape.add_argument("--max-scrolls", type=int, default=5)
    scrape.add_argument("--since-days", type=int, default=30)
    scrape.add_argument("--store-path", default="")
    scrape.add_argument("--cdp-url", default="http://127.0.0.1:9222")

    ingest = subs.add_parser("ingest-recording", help="Ingest a local or direct-URL meeting recording")
    ingest.add_argument("path_or_url")
    ingest.add_argument("--meeting-label", required=True)
    ingest.add_argument("--transcript", default="")
    ingest.add_argument("--store-path", default="")
    ingest.add_argument("--artifact-cache", default="")

    download = subs.add_parser("download-meeting", help="Download and ingest one Teams meeting recording")
    download.add_argument("url")
    _add_download_options(download)

    downloads = subs.add_parser("download-meetings", help="Download and ingest Teams meeting recordings from a URL file")
    downloads.add_argument("--url-file", required=True)
    _add_download_options(downloads)

    subparser.set_defaults(func=teams_context_command)


def teams_context_command(args: argparse.Namespace) -> int:
    action = getattr(args, "teams_context_action", None)
    if not action:
        print(
            "Usage: hermes teams-context "
            "{backfill|subscribe|subscriptions|maintain-subscriptions|validate|scrape-ui|ingest-recording|download-meeting|download-meetings}"
        )
        return 2
    if action == "backfill":
        _print(_run_async(_build_runtime(args).backfill_chat(
            getattr(args, "chat_id"),
            since_days=getattr(args, "since_days", 30),
            limit=(getattr(args, "limit", 0) or None),
        )))
    elif action == "subscribe":
        client_state = getattr(args, "client_state", "") or os.getenv("MSGRAPH_WEBHOOK_CLIENT_STATE", "")
        _print(_run_async(_build_runtime(args).create_subscription(
            chat_id=getattr(args, "chat_id"),
            notification_url=getattr(args, "notification_url"),
            client_state=client_state,
            expiration=getattr(args, "expiration", "") or None,
            change_type=getattr(args, "change_type", "") or "created,updated,deleted",
            lifecycle_notification_url=getattr(args, "lifecycle_notification_url", "") or None,
        )))
    elif action in {"subscriptions", "subs"}:
        _print({"subscriptions": TeamsContextStore(getattr(args, "store_path", "") or None).list_subscriptions()})
    elif action == "maintain-subscriptions":
        _print(_run_async(_build_runtime(args).renew_due_subscriptions(
            renew_within_hours=getattr(args, "renew_within_hours", 24),
            extend_hours=getattr(args, "extend_hours", 24),
            dry_run=bool(getattr(args, "dry_run", False)),
        )))
    elif action == "validate":
        store = TeamsContextStore(getattr(args, "store_path", "") or None)
        credentials = GraphCredentials.from_env(required=False)
        _print({
            "store_path": str(store.path),
            "graph_credentials_configured": credentials is not None,
            "subscription_count": len(store.list_subscriptions()),
        })
    elif action == "scrape-ui":
        from plugins.teams_context.ui_scrape import TeamsUIScrapeError, scrape_and_store

        try:
            _print(
                scrape_and_store(
                    label=getattr(args, "label"),
                    max_scrolls=getattr(args, "max_scrolls", 5),
                    since_days=getattr(args, "since_days", 30),
                    store=TeamsContextStore(getattr(args, "store_path", "") or None),
                    cdp_url=getattr(args, "cdp_url", "http://127.0.0.1:9222"),
                )
            )
        except TeamsUIScrapeError as exc:
            print(f"Teams UI scrape failed: {exc}")
            return 1
    elif action == "ingest-recording":
        from plugins.teams_context.recording import RecordingIngestError, ingest_recording

        try:
            _print(
                ingest_recording(
                    getattr(args, "path_or_url"),
                    meeting_label=getattr(args, "meeting_label"),
                    transcript_path=getattr(args, "transcript", "") or None,
                    store=TeamsContextStore(getattr(args, "store_path", "") or None),
                    artifact_cache=getattr(args, "artifact_cache", "") or None,
                )
            )
        except RecordingIngestError as exc:
            print(f"Recording ingest failed: {exc}")
            return 1
    elif action == "download-meeting":
        from plugins.teams_context.meeting_download import download_meetings

        payload = download_meetings(
            [getattr(args, "url")],
            output_dir=getattr(args, "output_dir"),
            cdp_url=getattr(args, "cdp_url", "http://127.0.0.1:9222"),
            store=TeamsContextStore(getattr(args, "store_path", "") or None),
            artifact_cache=getattr(args, "artifact_cache", "") or None,
            force=bool(getattr(args, "force", False)),
            ingest=not bool(getattr(args, "no_ingest", False)),
        )
        _print(payload)
        return 1 if payload.get("failed") else 0
    elif action == "download-meetings":
        from plugins.teams_context.meeting_download import download_meetings, parse_url_file

        payload = download_meetings(
            parse_url_file(getattr(args, "url_file")),
            output_dir=getattr(args, "output_dir"),
            cdp_url=getattr(args, "cdp_url", "http://127.0.0.1:9222"),
            store=TeamsContextStore(getattr(args, "store_path", "") or None),
            artifact_cache=getattr(args, "artifact_cache", "") or None,
            force=bool(getattr(args, "force", False)),
            ingest=not bool(getattr(args, "no_ingest", False)),
        )
        _print(payload)
        return 1 if payload.get("failed") else 0
    else:
        print(f"Unknown teams-context action: {action}")
        return 2
    return 0


def _add_download_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where meeting recordings, transcripts, and the local report are written",
    )
    parser.add_argument("--cdp-url", default="http://127.0.0.1:9222")
    parser.add_argument("--store-path", default="")
    parser.add_argument("--artifact-cache", default="")
    parser.add_argument("--force", action="store_true", help="Overwrite or redownload existing files")
    parser.add_argument("--no-ingest", action="store_true", help="Download only; do not update TeamContext KB")


def _build_runtime(args: argparse.Namespace) -> TeamsContextGraph:
    credentials = GraphCredentials.from_env()
    provider = MicrosoftGraphTokenProvider(credentials)
    return TeamsContextGraph(
        client=MicrosoftGraphClient(provider),
        store=TeamsContextStore(getattr(args, "store_path", "") or None),
        tenant_id=credentials.tenant_id,
    )


def _run_async(coro):
    return asyncio.run(coro)


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
