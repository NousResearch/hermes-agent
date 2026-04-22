from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings
import uvicorn

from .config import BrokerSettings
from .core import IdempotencyStore, ToolExecutionError, run_logged_tool
from .providers.affinity import AffinityProvider
from .providers.google_workspace import GoogleWorkspaceProvider
from .providers.meetings import GrainProvider, GranolaProvider
from .providers.notion import NotionProvider
from .providers.zoom import ZoomProvider


def _tool_security() -> TransportSecuritySettings:
    return TransportSecuritySettings(enable_dns_rebinding_protection=False, allowed_hosts=["*"], allowed_origins=["*"])


def _server(name: str, instructions: str) -> FastMCP:
    return FastMCP(
        name,
        instructions=instructions,
        streamable_http_path="/",
        mount_path="/",
        transport_security=_tool_security(),
        stateless_http=True,
    )


def create_app(settings: BrokerSettings | None = None) -> FastAPI:
    settings = settings or BrokerSettings.load()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
    store = IdempotencyStore(settings.idempotency_store)
    notion = NotionProvider(settings, store)
    gws = GoogleWorkspaceProvider(settings)
    zoom = ZoomProvider(settings)
    affinity = AffinityProvider(settings, store)
    grain = GrainProvider(settings)
    granola = GranolaProvider(settings)

    grain_mcp = _server("grain", "Hermes Spark shared Grain broker for Hermes Spark and Hermes M3.")
    granola_mcp = _server("granola", "Hermes Spark shared Granola broker for Hermes Spark and Hermes M3.")
    notion_mcp = _server("notion_api", "Hermes Spark shared Notion REST broker.")
    gws_mcp = _server("gws_api", "Hermes Spark shared Google Workspace broker.")
    zoom_mcp = _server("zoom_api", "Hermes Spark shared Zoom REST broker.")
    affinity_mcp = _server("affinity_api", "Hermes Spark shared Affinity CRM broker.")
    grain_app = grain_mcp.streamable_http_app()
    granola_app = granola_mcp.streamable_http_app()
    notion_app = notion_mcp.streamable_http_app()
    gws_app = gws_mcp.streamable_http_app()
    zoom_app = zoom_mcp.streamable_http_app()
    affinity_app = affinity_mcp.streamable_http_app()
    mounted_apps = [grain_app, granola_app, notion_app, gws_app, zoom_app, affinity_app]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with AsyncExitStack() as stack:
            for mounted in mounted_apps:
                await stack.enter_async_context(mounted.router.lifespan_context(mounted))
            yield

    app = FastAPI(title="Hermes Shared Tool Broker", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "broker": "hermes-shared-tool-broker",
                "base_url": settings.tailscale_base_url,
                "mcp_paths": {
                    "grain": "/mcp/grain",
                    "granola": "/mcp/granola",
                    "notion_api": "/mcp/notion-api",
                    "gws_api": "/mcp/gws-api",
                    "zoom_api": "/mcp/zoom-api",
                    "affinity_api": "/mcp/affinity-api",
                },
            }
        )

    @grain_mcp.tool(name="grain.meetings.search")
    async def grain_meetings_search(query: str, limit: int = 10, request_id: str | None = None):
        return await run_logged_tool("grain.meetings.search", "grain", lambda: grain.meetings_search(query=query, limit=limit), request_id=request_id)

    @grain_mcp.tool(name="grain.meeting.get")
    async def grain_meeting_get(meeting_id: str, request_id: str | None = None):
        return await run_logged_tool("grain.meeting.get", "grain", lambda: grain.meeting_get(meeting_id=meeting_id), request_id=request_id)

    @grain_mcp.tool(name="grain.transcript.get")
    async def grain_transcript_get(meeting_id: str, request_id: str | None = None):
        return await run_logged_tool("grain.transcript.get", "grain", lambda: grain.transcript_get(meeting_id=meeting_id), request_id=request_id)

    @grain_mcp.tool(name="grain.highlights.list")
    async def grain_highlights_list(meeting_id: str, request_id: str | None = None):
        return await run_logged_tool("grain.highlights.list", "grain", lambda: grain.highlights_list(meeting_id=meeting_id), request_id=request_id)

    @grain_mcp.tool(name="grain.notes.get")
    async def grain_notes_get(meeting_id: str, request_id: str | None = None):
        return await run_logged_tool("grain.notes.get", "grain", lambda: grain.notes_get(meeting_id=meeting_id), request_id=request_id)

    @granola_mcp.tool(name="granola.meetings.search")
    async def granola_meetings_search(query: str | None = None, folder_id: str | None = None, limit: int = 10, request_id: str | None = None):
        return await run_logged_tool("granola.meetings.search", "granola", lambda: granola.meetings_search(query=query, folder_id=folder_id, limit=limit), request_id=request_id)

    @granola_mcp.tool(name="granola.meeting.get")
    async def granola_meeting_get(meeting_id: str, request_id: str | None = None):
        return await run_logged_tool("granola.meeting.get", "granola", lambda: granola.meeting_get(meeting_id=meeting_id), request_id=request_id)

    @granola_mcp.tool(name="granola.transcript.get")
    async def granola_transcript_get(meeting_id: str, request_id: str | None = None):
        return await run_logged_tool("granola.transcript.get", "granola", lambda: granola.transcript_get(meeting_id=meeting_id), request_id=request_id)

    @granola_mcp.tool(name="granola.notes.get")
    async def granola_notes_get(meeting_id: str, request_id: str | None = None):
        return await run_logged_tool("granola.notes.get", "granola", lambda: granola.notes_get(meeting_id=meeting_id), request_id=request_id)

    @granola_mcp.tool(name="granola.folders.list")
    async def granola_folders_list(request_id: str | None = None):
        return await run_logged_tool("granola.folders.list", "granola", lambda: granola.folders_list(), request_id=request_id)

    @notion_mcp.tool(name="notion.search")
    async def notion_search(query: str = "", filter_payload: dict[str, Any] | None = None, page_size: int = 10, debug: bool = False, request_id: str | None = None):
        return await run_logged_tool("notion.search", "notion", lambda: notion.search(query=query, filter_payload=filter_payload, page_size=page_size, debug=debug), request_id=request_id)

    @notion_mcp.tool(name="notion.page.get")
    async def notion_page_get(page_id: str, include_blocks: bool = False, debug: bool = False, request_id: str | None = None):
        return await run_logged_tool("notion.page.get", "notion", lambda: notion.page_get(page_id=page_id, include_blocks=include_blocks, debug=debug), request_id=request_id)

    @notion_mcp.tool(name="notion.page.create")
    async def notion_page_create(parent: dict[str, Any], title: str | None = None, properties: dict[str, Any] | None = None, children: list[dict[str, Any]] | None = None, idempotency_key: str | None = None, dry_run: bool = False, debug: bool = False, request_id: str | None = None):
        return await run_logged_tool("notion.page.create", "notion", lambda: notion.page_create(parent=parent, title=title, properties=properties, children=children, idempotency_key=idempotency_key, dry_run=dry_run, debug=debug), request_id=request_id)

    @notion_mcp.tool(name="notion.page.update_properties")
    async def notion_page_update_properties(page_id: str, properties: dict[str, Any], idempotency_key: str | None = None, dry_run: bool = False, debug: bool = False, request_id: str | None = None):
        return await run_logged_tool("notion.page.update_properties", "notion", lambda: notion.page_update_properties(page_id=page_id, properties=properties, idempotency_key=idempotency_key, dry_run=dry_run, debug=debug), request_id=request_id)

    @notion_mcp.tool(name="notion.blocks.list")
    async def notion_blocks_list(block_id: str, recursive: bool = False, page_size: int = 100, debug: bool = False, request_id: str | None = None):
        return await run_logged_tool("notion.blocks.list", "notion", lambda: notion.blocks_list(block_id=block_id, recursive=recursive, page_size=page_size, debug=debug), request_id=request_id)

    @notion_mcp.tool(name="notion.blocks.append")
    async def notion_blocks_append(block_id: str, children: list[dict[str, Any]], after: str | None = None, idempotency_key: str | None = None, dry_run: bool = False, debug: bool = False, request_id: str | None = None):
        return await run_logged_tool("notion.blocks.append", "notion", lambda: notion.blocks_append(block_id=block_id, children=children, after=after, idempotency_key=idempotency_key, dry_run=dry_run, debug=debug), request_id=request_id)

    @notion_mcp.tool(name="notion.blocks.replace_range")
    async def notion_blocks_replace_range(parent_block_id: str, block_ids: list[str], children: list[dict[str, Any]], idempotency_key: str | None = None, dry_run: bool = False, debug: bool = False, request_id: str | None = None):
        return await run_logged_tool("notion.blocks.replace_range", "notion", lambda: notion.blocks_replace_range(parent_block_id=parent_block_id, block_ids=block_ids, children=children, idempotency_key=idempotency_key, dry_run=dry_run, debug=debug), request_id=request_id)

    @notion_mcp.tool(name="notion.blocks.patch_text")
    async def notion_blocks_patch_text(block_id: str, text: str, idempotency_key: str | None = None, dry_run: bool = False, debug: bool = False, request_id: str | None = None):
        return await run_logged_tool("notion.blocks.patch_text", "notion", lambda: notion.blocks_patch_text(block_id=block_id, text=text, idempotency_key=idempotency_key, dry_run=dry_run, debug=debug), request_id=request_id)

    @notion_mcp.tool(name="notion.database.query")
    async def notion_database_query(database_id: str, query: dict[str, Any] | None = None, debug: bool = False, request_id: str | None = None):
        return await run_logged_tool("notion.database.query", "notion", lambda: notion.database_query(database_id=database_id, query=query, debug=debug), request_id=request_id)

    @notion_mcp.tool(name="notion.database.upsert_page")
    async def notion_database_upsert_page(database_id: str, match_property: str, match_value: str, properties: dict[str, Any], children: list[dict[str, Any]] | None = None, idempotency_key: str | None = None, dry_run: bool = False, debug: bool = False, request_id: str | None = None):
        return await run_logged_tool("notion.database.upsert_page", "notion", lambda: notion.database_upsert_page(database_id=database_id, match_property=match_property, match_value=match_value, properties=properties, children=children, idempotency_key=idempotency_key, dry_run=dry_run, debug=debug), request_id=request_id)

    @gws_mcp.tool(name="gws.gmail.search")
    async def gws_gmail_search(query: str, max_results: int = 10, request_id: str | None = None):
        return await run_logged_tool("gws.gmail.search", "gws", lambda: gws.gmail_search(query=query, max_results=max_results), request_id=request_id)

    @gws_mcp.tool(name="gws.gmail.get")
    async def gws_gmail_get(message_id: str, request_id: str | None = None):
        return await run_logged_tool("gws.gmail.get", "gws", lambda: gws.gmail_get(message_id=message_id), request_id=request_id)

    @gws_mcp.tool(name="gws.gmail.send")
    async def gws_gmail_send(to: str, subject: str, body: str, html: bool = False, request_id: str | None = None):
        return await run_logged_tool("gws.gmail.send", "gws", lambda: gws.gmail_send(to=to, subject=subject, body=body, html=html), request_id=request_id)

    @gws_mcp.tool(name="gws.calendar.events.search")
    async def gws_calendar_events_search(start: str | None = None, end: str | None = None, calendar: str = "primary", request_id: str | None = None):
        return await run_logged_tool("gws.calendar.events.search", "gws", lambda: gws.calendar_events_search(start=start, end=end, calendar=calendar), request_id=request_id)

    @gws_mcp.tool(name="gws.calendar.freebusy")
    async def gws_calendar_freebusy(time_min: str, time_max: str, calendars: list[str], request_id: str | None = None):
        return await run_logged_tool("gws.calendar.freebusy", "gws", lambda: gws.calendar_freebusy(time_min=time_min, time_max=time_max, calendars=calendars), request_id=request_id)

    @gws_mcp.tool(name="gws.calendar.event.create")
    async def gws_calendar_event_create(summary: str, start: str, end: str, location: str | None = None, attendees: list[str] | None = None, request_id: str | None = None):
        return await run_logged_tool("gws.calendar.event.create", "gws", lambda: gws.calendar_event_create(summary=summary, start=start, end=end, location=location, attendees=attendees), request_id=request_id)

    @gws_mcp.tool(name="gws.drive.search")
    async def gws_drive_search(query: str, max_results: int = 10, raw_query: bool = False, request_id: str | None = None):
        return await run_logged_tool("gws.drive.search", "gws", lambda: gws.drive_search(query=query, max_results=max_results, raw_query=raw_query), request_id=request_id)

    @gws_mcp.tool(name="gws.drive.file.get")
    async def gws_drive_file_get(file_id: str, request_id: str | None = None):
        return await run_logged_tool("gws.drive.file.get", "gws", lambda: gws.drive_file_get(file_id=file_id), request_id=request_id)

    @gws_mcp.tool(name="gws.docs.get")
    async def gws_docs_get(doc_id: str, request_id: str | None = None):
        return await run_logged_tool("gws.docs.get", "gws", lambda: gws.docs_get(doc_id=doc_id), request_id=request_id)

    @gws_mcp.tool(name="gws.docs.patch")
    async def gws_docs_patch(doc_id: str, requests_payload: list[dict[str, Any]], request_id: str | None = None):
        return await run_logged_tool("gws.docs.patch", "gws", lambda: gws.docs_patch(doc_id=doc_id, requests_payload=requests_payload), request_id=request_id)

    @gws_mcp.tool(name="gws.sheets.read")
    async def gws_sheets_read(spreadsheet_id: str, cell_range: str, request_id: str | None = None):
        return await run_logged_tool("gws.sheets.read", "gws", lambda: gws.sheets_read(spreadsheet_id=spreadsheet_id, cell_range=cell_range), request_id=request_id)

    @gws_mcp.tool(name="gws.sheets.update")
    async def gws_sheets_update(spreadsheet_id: str, cell_range: str, values: list[list[Any]], request_id: str | None = None):
        return await run_logged_tool("gws.sheets.update", "gws", lambda: gws.sheets_update(spreadsheet_id=spreadsheet_id, cell_range=cell_range, values=values), request_id=request_id)

    @zoom_mcp.tool(name="zoom.meetings.list")
    async def zoom_meetings_list(user_id: str = "me", meeting_type: str = "scheduled", page_size: int = 30, request_id: str | None = None):
        return await run_logged_tool("zoom.meetings.list", "zoom", lambda: zoom.meetings_list(user_id=user_id, meeting_type=meeting_type, page_size=page_size), request_id=request_id)

    @zoom_mcp.tool(name="zoom.meeting.get")
    async def zoom_meeting_get(meeting_id: str, request_id: str | None = None):
        return await run_logged_tool("zoom.meeting.get", "zoom", lambda: zoom.meeting_get(meeting_id=meeting_id), request_id=request_id)

    @zoom_mcp.tool(name="zoom.meeting.create")
    async def zoom_meeting_create(topic: str, user_id: str = "me", start_time: str | None = None, duration_minutes: int | None = None, agenda: str | None = None, dry_run: bool = True, request_id: str | None = None):
        return await run_logged_tool("zoom.meeting.create", "zoom", lambda: zoom.meeting_create(user_id=user_id, topic=topic, start_time=start_time, duration_minutes=duration_minutes, agenda=agenda, dry_run=dry_run), request_id=request_id)

    @zoom_mcp.tool(name="zoom.recordings.list")
    async def zoom_recordings_list(user_id: str = "me", from_date: str = "2026-04-01", to_date: str = "2026-04-21", page_size: int = 30, request_id: str | None = None):
        return await run_logged_tool("zoom.recordings.list", "zoom", lambda: zoom.recordings_list(user_id=user_id, from_date=from_date, to_date=to_date, page_size=page_size), request_id=request_id)

    @zoom_mcp.tool(name="zoom.recording.get")
    async def zoom_recording_get(meeting_id: str, request_id: str | None = None):
        return await run_logged_tool("zoom.recording.get", "zoom", lambda: zoom.recording_get(meeting_id=meeting_id), request_id=request_id)

    @zoom_mcp.tool(name="zoom.recording.transcript_get")
    async def zoom_recording_transcript_get(meeting_id: str, request_id: str | None = None):
        return await run_logged_tool("zoom.recording.transcript_get", "zoom", lambda: zoom.recording_transcript_get(meeting_id=meeting_id), request_id=request_id)

    @zoom_mcp.tool(name="zoom.users.get")
    async def zoom_users_get(user_id: str, request_id: str | None = None):
        return await run_logged_tool("zoom.users.get", "zoom", lambda: zoom.users_get(user_id=user_id), request_id=request_id)

    @affinity_mcp.tool(name="affinity.person.search")
    async def affinity_person_search(query: str, request_id: str | None = None):
        return await run_logged_tool("affinity.person.search", "affinity", lambda: affinity.person_search(query=query), request_id=request_id)

    @affinity_mcp.tool(name="affinity.person.get")
    async def affinity_person_get(person_id: int, request_id: str | None = None):
        return await run_logged_tool("affinity.person.get", "affinity", lambda: affinity.person_get(person_id=person_id), request_id=request_id)

    @affinity_mcp.tool(name="affinity.person.upsert")
    async def affinity_person_upsert(first_name: str, last_name: str, email: str | None = None, organization_id: int | None = None, dry_run: bool = True, idempotency_key: str | None = None, request_id: str | None = None):
        return await run_logged_tool("affinity.person.upsert", "affinity", lambda: affinity.person_upsert(first_name=first_name, last_name=last_name, email=email, organization_id=organization_id, dry_run=dry_run, idempotency_key=idempotency_key), request_id=request_id)

    @affinity_mcp.tool(name="affinity.organization.search")
    async def affinity_organization_search(query: str, request_id: str | None = None):
        return await run_logged_tool("affinity.organization.search", "affinity", lambda: affinity.organization_search(query=query), request_id=request_id)

    @affinity_mcp.tool(name="affinity.organization.get")
    async def affinity_organization_get(organization_id: int, request_id: str | None = None):
        return await run_logged_tool("affinity.organization.get", "affinity", lambda: affinity.organization_get(organization_id=organization_id), request_id=request_id)

    @affinity_mcp.tool(name="affinity.organization.upsert")
    async def affinity_organization_upsert(name: str, domain: str | None = None, dry_run: bool = True, idempotency_key: str | None = None, request_id: str | None = None):
        return await run_logged_tool("affinity.organization.upsert", "affinity", lambda: affinity.organization_upsert(name=name, domain=domain, dry_run=dry_run, idempotency_key=idempotency_key), request_id=request_id)

    @affinity_mcp.tool(name="affinity.opportunity.search")
    async def affinity_opportunity_search(query: str, request_id: str | None = None):
        return await run_logged_tool("affinity.opportunity.search", "affinity", lambda: affinity.opportunity_search(query=query), request_id=request_id)

    @affinity_mcp.tool(name="affinity.opportunity.get")
    async def affinity_opportunity_get(opportunity_id: int, request_id: str | None = None):
        return await run_logged_tool("affinity.opportunity.get", "affinity", lambda: affinity.opportunity_get(opportunity_id=opportunity_id), request_id=request_id)

    @affinity_mcp.tool(name="affinity.opportunity.update_stage")
    async def affinity_opportunity_update_stage(opportunity_id: int, stage_id: int, dry_run: bool = True, idempotency_key: str | None = None, request_id: str | None = None):
        return await run_logged_tool("affinity.opportunity.update_stage", "affinity", lambda: affinity.opportunity_update_stage(opportunity_id=opportunity_id, stage_id=stage_id, dry_run=dry_run, idempotency_key=idempotency_key), request_id=request_id)

    @affinity_mcp.tool(name="affinity.note.create")
    async def affinity_note_create(content: str, person_ids: list[int] | None = None, organization_ids: list[int] | None = None, opportunity_ids: list[int] | None = None, dry_run: bool = True, idempotency_key: str | None = None, request_id: str | None = None):
        return await run_logged_tool("affinity.note.create", "affinity", lambda: affinity.note_create(content=content, person_ids=person_ids, organization_ids=organization_ids, opportunity_ids=opportunity_ids, dry_run=dry_run, idempotency_key=idempotency_key), request_id=request_id)

    app.mount("/mcp/grain", grain_app)
    app.mount("/mcp/granola", granola_app)
    app.mount("/mcp/notion-api", notion_app)
    app.mount("/mcp/gws-api", gws_app)
    app.mount("/mcp/zoom-api", zoom_app)
    app.mount("/mcp/affinity-api", affinity_app)
    return app


def main() -> None:
    settings = BrokerSettings.load()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    try:
        main()
    except ToolExecutionError as exc:
        raise SystemExit(str(exc))
