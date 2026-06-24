import os
from pathlib import Path
from typing import Any

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException


app = FastAPI(title="Company Knowledge API", version="0.1.0")

DATABASE_URL = os.environ["DATABASE_URL"]
LIGHTRAG_ROOT = Path(os.environ.get("LIGHTRAG_ROOT", "/data/lightrag"))
DEPARTMENT_WORKSPACES = [
    "company_public",
    "company_internal",
    "department_marketing",
    "department_financial",
    "department_hr",
    "department_engineering",
    "department_c_level",
]
LIGHTRAG_API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")
LIGHTRAG_URLS = {
    "company_public": os.environ.get("LIGHTRAG_COMPANY_PUBLIC_URL"),
    "company_internal": os.environ.get("LIGHTRAG_COMPANY_INTERNAL_URL"),
    "department_marketing": os.environ.get("LIGHTRAG_DEPARTMENT_MARKETING_URL"),
    "department_financial": os.environ.get("LIGHTRAG_DEPARTMENT_FINANCIAL_URL"),
    "department_hr": os.environ.get("LIGHTRAG_DEPARTMENT_HR_URL"),
    "department_engineering": os.environ.get("LIGHTRAG_DEPARTMENT_ENGINEERING_URL"),
    "department_c_level": os.environ.get("LIGHTRAG_DEPARTMENT_C_LEVEL_URL"),
}


@app.on_event("startup")
async def startup() -> None:
    app.state.pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    for workspace in DEPARTMENT_WORKSPACES:
        base = LIGHTRAG_ROOT / "workspaces" / workspace
        for child in ("docs", "kv_store", "vector_store", "graph_store"):
            (base / child).mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.pool.close()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/workspaces")
async def workspaces() -> dict[str, Any]:
    rows = await app.state.pool.fetch(
        "SELECT slug, name, visibility_boundary FROM rag_workspaces ORDER BY slug"
    )
    return {
        "workspaces": [dict(row) for row in rows],
        "lightrag_root": str(LIGHTRAG_ROOT),
    }


def route_workspace(payload: dict[str, Any]) -> str:
    visibility = payload.get("visibility", "department")
    department = str(payload.get("department", "")).lower().replace(" ", "_")
    if visibility == "company":
        classification = payload.get("classification", "internal")
        return "company_public" if classification == "public" else "company_internal"
    if visibility == "department":
        return f"department_{department}"
    raise ValueError("only company and department visibility are enabled in this MVP")


def allowed_workspaces(groups: list[str]) -> list[str]:
    allowed = ["company_public", "company_internal"]
    for group in groups:
        if group.startswith("department_") and group in LIGHTRAG_URLS:
            allowed.append(group)
    return sorted(set(allowed))


@app.post("/query")
async def query(payload: dict[str, Any]) -> dict[str, Any]:
    user_groups = payload.get("groups") or []
    allowed = allowed_workspaces(user_groups)
    headers = {"X-API-Key": LIGHTRAG_API_KEY} if LIGHTRAG_API_KEY else {}
    answers = []
    async with httpx.AsyncClient(timeout=180) as client:
        for workspace in allowed:
            base_url = LIGHTRAG_URLS.get(workspace)
            if not base_url:
                continue
            try:
                response = await client.post(
                    f"{base_url}/query",
                    json={
                        "query": payload.get("query"),
                        "mode": payload.get("mode", "mix"),
                        "include_references": payload.get("include_references", True),
                    },
                    headers=headers,
                )
                response.raise_for_status()
                answers.append({"workspace": workspace, "result": response.json()})
            except Exception as exc:
                answers.append({"workspace": workspace, "error": str(exc)})
    return {
        "status": "ok",
        "query": payload.get("query"),
        "allowed_workspaces": allowed,
        "answers": answers,
    }


@app.post("/documents/text")
async def ingest_text(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = route_workspace(payload)
    if workspace not in LIGHTRAG_URLS:
        return {"status": "error", "reason": f"unknown workspace {workspace}"}

    title = payload["title"]
    text = payload["text"]
    classification = payload.get("classification", "internal")
    department = payload.get("department")
    source_text = (
        f"Title: {title}\n"
        f"Department: {department}\n"
        f"Classification: {classification}\n\n"
        f"{text}"
    )

    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO documents (title, department_slug, classification, status)
            VALUES ($1, $2, $3, 'indexed')
            RETURNING id
            """,
            title,
            department,
            classification,
        )
        workspace_row = await conn.fetchrow(
            "SELECT id FROM rag_workspaces WHERE slug = $1",
            workspace,
        )
        if workspace_row:
            await conn.execute(
                """
                INSERT INTO document_workspace_membership (document_id, workspace_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                row["id"],
                workspace_row["id"],
            )

    headers = {"X-API-Key": LIGHTRAG_API_KEY} if LIGHTRAG_API_KEY else {}
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            f"{LIGHTRAG_URLS[workspace]}/documents/text",
            json={"text": source_text, "file_source": title},
            headers=headers,
        )
        if response.is_error:
            raise HTTPException(
                status_code=response.status_code,
                detail={"upstream": "lightrag", "body": response.text[:2000]},
            )
        lightrag_result = response.json()

    return {
        "status": "indexed",
        "document_id": str(row["id"]),
        "workspace": workspace,
        "lightrag": lightrag_result,
    }
