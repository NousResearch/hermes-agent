"""Authenticated, catalog-bound report and download routes."""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from hermes_cli.report_catalog import ReportCatalogError, load_catalog, public_entry, resolve_artifact
from hermes_cli.web_deps import late

router = APIRouter()
_require_token = late("_require_token")


def _reports():
    catalog = load_catalog()
    entries = []
    for entry in catalog["reports"]:
        item = public_entry(entry)
        for artifact in item["artifacts"]:
            resolve_artifact(entry["report_id"], artifact["filename"])
        entries.append(item)
    return {"schema_version": catalog["schema_version"], "reports": entries}


def _artifact(report_id: str, filename: str, *, inline: bool = False) -> Response:
    try:
        name, content, mime_type, digest = resolve_artifact(report_id, filename)
    except ReportCatalogError as exc:
        status = 409 if "hash drift" in str(exc) else 404
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Report-SHA256": digest,
        "Content-Disposition": f'{"inline" if inline else "attachment"}; filename="{name}"',
    }
    if inline:
        headers["Content-Security-Policy"] = (
            "sandbox allow-scripts; default-src 'none'; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; base-uri 'none'"
        )
    return Response(content=content, media_type=mime_type, headers=headers)


@router.get("/api/reports")
async def list_reports(request: Request):
    _require_token(request)
    try:
        return await asyncio.to_thread(_reports)
    except ReportCatalogError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/api/reports/{report_id}/report.html")
async def get_report_html(request: Request, report_id: str):
    _require_token(request)
    return await asyncio.to_thread(_artifact, report_id, "seo-report.html", inline=True)


@router.get("/reports/{report_id}/f001-details.html")
async def get_f001_detail_html(request: Request, report_id: str):
    # Unlike /api/*, this standalone HTML path does not pass the loopback API gate.
    _require_token(request)
    return await asyncio.to_thread(_artifact, report_id, "f001-details.html", inline=True)


@router.get("/api/reports/{report_id}/{filename}")
async def download_report_artifact(request: Request, report_id: str, filename: str):
    _require_token(request)
    return await asyncio.to_thread(_artifact, report_id, filename)
