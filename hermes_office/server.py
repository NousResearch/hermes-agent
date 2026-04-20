"""FastAPI server for Hermes Digital Office.

Boots a single ``uvicorn`` worker bound to ``127.0.0.1`` (loopback only) by
default. The static frontend is served from ``hermes_office/frontend/dist/`` if
present (in source checkouts before ``npm run build`` it gracefully serves a
fallback bootstrap page).

See ``.kiro/specs/digital-office-ui/design.md`` §5 for the API contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field, ValidationError
except ImportError as exc:  # pragma: no cover - import-time guard
    raise SystemExit(
        "hermes_office.server requires the [office] extra. "
        "Install with: pip install -e .[office]"
    ) from exc

from . import __version__
from .capacity import compute as compute_capacity
from .capacity import detect_host, model_profile_for
from .eventbus import EventBus
from .models import (
    Activity,
    ActivityEvent,
    AvatarStyle,
    CapacityReport,
    Department,
    Employee,
    ResolvedRole,
    Task,
)
from .presets import PRESETS, Preset
from .runtime import Runtime, TaskResult, make_runtime
from .skill_resolver import SkillResolver
from .store import Store, office_root

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Request / response payload models
# ────────────────────────────────────────────────────────────────────────────


class _DepartmentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    mission: str = Field(default="", max_length=500)
    color: str = "#7c3aed"
    runtime_default: Literal["simulated", "hermes"] = "simulated"


class _DepartmentPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=50)
    mission: str | None = Field(default=None, max_length=500)
    color: str | None = None
    runtime_default: Literal["simulated", "hermes"] | None = None


class _EmployeeCreate(BaseModel):
    department_id: str
    name: str
    role: str = "Helper"
    avatar: AvatarStyle = Field(default_factory=AvatarStyle)
    model: str
    provider: str | None = None
    base_url: str | None = None
    enabled_toolsets: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    runtime: Literal["simulated", "hermes"] = "simulated"


class _EmployeePatch(BaseModel):
    name: str | None = None
    role: str | None = None
    avatar: AvatarStyle | None = None
    model: str | None = None
    provider: str | None = None
    base_url: str | None = None
    enabled_toolsets: list[str] | None = None
    skills: list[str] | None = None
    system_prompt: str | None = None
    runtime: Literal["simulated", "hermes"] | None = None


class _TaskCreate(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)
    department_id: str | None = None
    employee_id: str | None = None


class _ResolveRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)


class _Health(BaseModel):
    ok: bool = True
    version: str
    profile_root: str
    office_root: str
    runtime_default: str
    departments: int
    employees: int


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _problem(status: int, title: str, detail: str = "", **extra) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"https://hermes.local/errors/{title.lower().replace(' ', '-')}",
        "title": title,
        "status": status,
        "detail": detail,
    }
    body.update(extra)
    return JSONResponse(
        status_code=status,
        content=jsonable_encoder(body),
        media_type="application/problem+json",
    )


def _build_cli_command(emp: Employee) -> str:
    parts = ["hermes", "chat", "-Q"]
    if emp.model:
        parts.append(f'--model "{emp.model}"')
    if emp.provider:
        parts.append(f'--provider "{emp.provider}"')
    if emp.base_url:
        parts.append(f'--base-url "{emp.base_url}"')
    if emp.enabled_toolsets:
        parts.append(f'-t "{",".join(emp.enabled_toolsets)}"')
    if emp.skills:
        parts.append(f'--skills "{",".join(emp.skills)}"')
    return " ".join(parts)


def _frontend_dist() -> Path | None:
    """Locate the built frontend; return None in source checkouts before
    ``npm run build`` has been run."""
    try:
        candidate = Path(str(files("hermes_office") / "frontend" / "dist"))
        if (candidate / "index.html").exists():
            return candidate
    except Exception:
        pass
    fallback = Path(__file__).parent / "frontend" / "dist"
    if (fallback / "index.html").exists():
        return fallback
    return None


_FALLBACK_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Hermes Office — frontend not built</title>
<style>
  body{font-family:system-ui;background:#0f172a;color:#e2e8f0;display:flex;
       align-items:center;justify-content:center;height:100vh;margin:0}
  .card{max-width:640px;padding:32px;background:#1e293b;border-radius:16px;
        box-shadow:0 12px 40px rgba(0,0,0,.4)}
  code{background:#0b1220;padding:2px 6px;border-radius:4px}
  a{color:#7dd3fc}
</style></head><body>
<div class="card">
<h1>🏢 Hermes Office</h1>
<p>The backend is up, but the React frontend bundle isn't built yet.</p>
<p>From your repo root run:</p>
<pre><code>cd hermes_office/frontend
npm install
npm run build</code></pre>
<p>Then refresh this page. The API itself is available at
<a href="/api/health">/api/health</a> right now.</p>
</div>
</body></html>"""


# ────────────────────────────────────────────────────────────────────────────
# App factory
# ────────────────────────────────────────────────────────────────────────────


def build_app(
    *,
    store: Store | None = None,
    bus: EventBus | None = None,
    resolver: SkillResolver | None = None,
    runtime_default: Literal["simulated", "hermes"] = "simulated",
) -> FastAPI:
    """Build a fresh FastAPI app. Pass dependencies in for testing."""
    s_store = store or Store()
    s_bus = bus or EventBus()
    s_resolver = resolver or SkillResolver(weights_path=s_store.weights_path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        s_store.boot_from_disk()
        logger.info("Hermes Office booted at %s", office_root())
        yield

    app = FastAPI(
        title="Hermes Digital Office",
        version=__version__,
        lifespan=lifespan,
    )

    # Stash deps on app.state so tests can inspect them.
    app.state.store = s_store
    app.state.bus = s_bus
    app.state.resolver = s_resolver
    app.state.runtime_default = runtime_default
    app.state.runtimes: dict[str, Runtime] = {}

    def get_runtime(kind: str) -> Runtime:
        kind = kind or runtime_default
        rt = app.state.runtimes.get(kind)
        if rt is None:
            rt = make_runtime(kind)
            app.state.runtimes[kind] = rt
        return rt

    # ── Errors ─────────────────────────────────────────────────────────────

    @app.exception_handler(ValidationError)
    async def _on_validation(_req: Request, exc: ValidationError):
        return _problem(
            422,
            "Validation Error",
            "Request body failed validation.",
            errors=exc.errors(),
        )

    @app.exception_handler(KeyError)
    async def _on_key(_req: Request, exc: KeyError):
        return _problem(404, "Not Found", str(exc))

    @app.exception_handler(ValueError)
    async def _on_value(_req: Request, exc: ValueError):
        return _problem(422, "Bad Request", str(exc))

    # ── Health & meta ──────────────────────────────────────────────────────

    @app.get("/api/health", response_model=_Health)
    async def health():
        return _Health(
            version=__version__,
            profile_root=str(office_root().parent),
            office_root=str(s_store.root),
            runtime_default=runtime_default,
            departments=len(s_store.list_departments()),
            employees=len(s_store.list_employees()),
        )

    @app.get("/api/capacity", response_model=CapacityReport)
    async def capacity(model: str | None = None):
        host = detect_host()
        chosen = model or _detect_default_model()
        profile = model_profile_for(chosen)
        return compute_capacity(host, profile, len(s_store.list_employees()))

    @app.get("/api/toolsets")
    async def toolsets():
        try:
            from toolsets import TOOLSETS  # type: ignore

            return [
                {"id": k, "description": v.get("description", "")}
                for k, v in sorted(TOOLSETS.items())
                if not k.startswith("hermes-")        # internal aggregates
            ]
        except Exception as exc:
            logger.debug("toolsets() fallback: %s", exc)
            return [
                {"id": k, "description": ""}
                for k in (
                    "web", "browser", "file", "code_execution", "image_gen",
                    "vision", "tts", "todo", "memory", "delegation",
                    "terminal", "cronjob", "messaging", "session_search",
                )
            ]

    @app.get("/api/skills")
    async def skills():
        # Surface installed SKILL.md docs from $HERMES_HOME/skills/ if any,
        # plus the bundled `skills/` directory. Each entry: {id, title, source}.
        out: dict[str, dict] = {}
        seen_dirs: list[Path] = []
        try:
            from hermes_constants import get_hermes_home
            seen_dirs.append(get_hermes_home() / "skills")
        except Exception:
            pass
        repo_skills = Path(__file__).resolve().parent.parent / "skills"
        if repo_skills.exists():
            seen_dirs.append(repo_skills)

        for base in seen_dirs:
            if not base.exists():
                continue
            for path in base.rglob("SKILL.md"):
                try:
                    rel = path.parent.relative_to(base).as_posix()
                except ValueError:
                    rel = path.parent.name
                if rel in out:
                    continue
                title = ""
                try:
                    head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]
                    for ln in head:
                        if ln.startswith("# "):
                            title = ln.lstrip("# ").strip()
                            break
                except OSError:
                    pass
                out[rel] = {
                    "id": rel,
                    "title": title or rel.split("/")[-1],
                    "source": str(base),
                }
        return sorted(out.values(), key=lambda r: r["id"])

    @app.get("/api/presets")
    async def presets():
        return list(PRESETS.values())

    @app.post("/api/skills/resolve", response_model=ResolvedRole)
    async def resolve(req: _ResolveRequest):
        return s_resolver.resolve(req.text)

    # ── Departments ────────────────────────────────────────────────────────

    @app.get("/api/departments", response_model=list[Department])
    async def list_depts():
        return s_store.list_departments()

    @app.post("/api/departments", response_model=Department, status_code=201)
    async def create_dept(req: _DepartmentCreate):
        dept = Department(
            name=req.name,
            mission=req.mission,
            color=req.color,
            runtime_default=req.runtime_default,
        )
        return s_store.create_department(dept)

    @app.patch("/api/departments/{dept_id}", response_model=Department)
    async def patch_dept(dept_id: str, req: _DepartmentPatch):
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        return s_store.update_department(dept_id, **updates)

    @app.delete("/api/departments/{dept_id}")
    async def delete_dept(dept_id: str):
        removed = s_store.delete_department(dept_id)
        return {"deleted_dept": dept_id, "deleted_employees": removed}

    # ── Employees ──────────────────────────────────────────────────────────

    @app.get("/api/employees", response_model=list[Employee])
    async def list_emps(dept_id: str | None = None):
        return s_store.list_employees(dept_id)

    @app.get("/api/employees/{emp_id}")
    async def get_emp(emp_id: str):
        emp = s_store.get_employee(emp_id)
        if emp is None:
            raise KeyError(emp_id)
        return {
            **emp.model_dump(mode="json"),
            "cli_command": _build_cli_command(emp),
        }

    @app.post("/api/employees", response_model=Employee, status_code=201)
    async def create_emp(req: _EmployeeCreate):
        emp = Employee(**req.model_dump())
        created = s_store.create_employee(emp)

        # Kick off a state_change so any open WebSocket sees the new sprite
        # spawn into the rest zone immediately.
        await s_bus.publish(ActivityEvent(
            employee_id=created.id,
            department_id=created.department_id,
            kind="state_change",
            text=f"{created.name} joined the office",
            meta={"to": Activity.RESTING.value},
        ))
        return created

    @app.patch("/api/employees/{emp_id}", response_model=Employee)
    async def patch_emp(emp_id: str, req: _EmployeePatch):
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        return s_store.update_employee(emp_id, **updates)

    @app.delete("/api/employees/{emp_id}")
    async def delete_emp(emp_id: str):
        s_store.delete_employee(emp_id)
        return {"deleted": emp_id}

    @app.get("/api/employees/{emp_id}/activity")
    async def emp_activity(emp_id: str, limit: int = 50, cursor: int | None = None):
        events, next_cursor = s_store.read_activity(emp_id, limit=limit, cursor=cursor)
        return {"events": events, "next_cursor": next_cursor}

    @app.get("/api/employees/{emp_id}/cli-command")
    async def emp_cli(emp_id: str):
        emp = s_store.get_employee(emp_id)
        if emp is None:
            raise KeyError(emp_id)
        return {"command": _build_cli_command(emp)}

    # ── Tasks ──────────────────────────────────────────────────────────────

    @app.post("/api/tasks", response_model=Task, status_code=201)
    async def create_task(req: _TaskCreate):
        emp_id = req.employee_id
        dept_id = req.department_id
        # Routing: explicit employee > explicit dept (round-robin idle) > error.
        if not emp_id and not dept_id:
            raise ValueError("either employee_id or department_id is required")

        if emp_id:
            emp = s_store.get_employee(emp_id)
            if emp is None:
                raise KeyError(emp_id)
            target_emp = emp
            dept_id = emp.department_id
        else:
            assert dept_id is not None
            dept = s_store.get_department(dept_id)
            if dept is None:
                raise KeyError(dept_id)
            roster = s_store.list_employees(dept_id)
            if not roster:
                raise ValueError(f"department {dept_id!r} has no employees")
            # Idle preferred; if none idle, any employee.
            idle = [e for e in roster if e.activity in (Activity.RESTING, Activity.OFFLINE)]
            target_emp = (idle or roster)[0]

        task = Task(
            department_id=dept_id,
            employee_id=target_emp.id,
            text=req.text,
            status="queued",
        )
        s_store.append_task(task)

        async def _run():
            t = task.model_copy(update={
                "status": "running",
                "started_at": datetime.now(tz=timezone.utc),
            })
            s_store.append_task(t)
            try:
                rt = get_runtime(target_emp.runtime or runtime_default)
                async def _on_event(evt: ActivityEvent) -> None:
                    s_store.append_activity(evt)
                    await s_bus.publish(evt)

                # mark working in store too
                s_store.update_employee(
                    target_emp.id, activity=Activity.WORKING
                )
                result = await rt.run_task(target_emp, t, _on_event)
                final_status: Literal["done", "failed"] = "done" if result.status == "done" else "failed"
                s_store.update_employee(
                    target_emp.id, activity=Activity.RESTING
                )
                done = t.model_copy(update={
                    "status": final_status,
                    "finished_at": datetime.now(tz=timezone.utc),
                    "result_summary": result.summary,
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                })
                s_store.append_task(done)
            except Exception as exc:  # noqa: BLE001
                err_evt = ActivityEvent(
                    employee_id=target_emp.id,
                    department_id=dept_id or "dept_unknown",
                    kind="error",
                    text=f"runtime error: {exc!r}",
                )
                s_store.append_activity(err_evt)
                await s_bus.publish(err_evt)
                s_store.update_employee(
                    target_emp.id, activity=Activity.RESTING
                )
                done = t.model_copy(update={
                    "status": "failed",
                    "finished_at": datetime.now(tz=timezone.utc),
                    "result_summary": repr(exc),
                })
                s_store.append_task(done)

        asyncio.create_task(_run())
        return task

    @app.get("/api/tasks")
    async def list_tasks(limit: int = 50):
        # The task log is append-only — every status transition (queued →
        # running → done/failed) is a fresh row. Collapse to one entry per
        # task id, keeping the latest (last-written wins), and return them
        # ordered by creation time ascending so the slice below picks the
        # most recent ``limit`` tasks.
        rows = s_store.read_recent_tasks(days=2)
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            tid = row.get("id")
            if not tid:
                continue
            latest[tid] = row
        ordered = sorted(
            latest.values(),
            key=lambda r: r.get("created_at") or "",
        )
        return ordered[-limit:]

    # ── Export / import ────────────────────────────────────────────────────

    @app.get("/api/export")
    async def export():
        return s_store.export()

    @app.post("/api/import")
    async def import_(payload: dict):
        return s_store.import_(payload)

    # ── WebSocket ──────────────────────────────────────────────────────────

    @app.websocket("/ws/office")
    async def ws_office(ws: WebSocket):
        # Loopback origin check (lenient: missing Origin allowed for native
        # clients & curl).
        origin = ws.headers.get("origin", "")
        if origin and not origin.startswith(("http://127.0.0.1", "http://localhost")):
            await ws.close(code=1008)
            return
        await ws.accept()
        q = await s_bus.subscribe()
        try:
            await ws.send_text(json.dumps({"kind": "hello", "version": __version__}))
            while True:
                evt = await q.get()
                await ws.send_text(evt.model_dump_json())
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("ws closed: %s", exc)
        finally:
            await s_bus.unsubscribe(q)

    # ── Static frontend / fallback ─────────────────────────────────────────

    dist = _frontend_dist()
    if dist is not None:
        # Mount /assets first then catch-all index.
        if (dist / "assets").exists():
            app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="assets")

        @app.get("/", include_in_schema=False)
        @app.get("/{full_path:path}", include_in_schema=False)
        async def index(full_path: str = ""):
            if full_path.startswith("api/") or full_path.startswith("ws/"):
                raise HTTPException(404)
            f = dist / full_path
            if full_path and f.is_file():
                return FileResponse(f)
            return FileResponse(dist / "index.html")
    else:
        @app.get("/", include_in_schema=False, response_class=HTMLResponse)
        async def fallback_index():
            return _FALLBACK_PAGE

    return app


# ────────────────────────────────────────────────────────────────────────────
# Module-level default app (for `uvicorn hermes_office.server:app`)
# ────────────────────────────────────────────────────────────────────────────


def _detect_default_model() -> str:
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        m = cfg.get("model")
        if isinstance(m, dict):
            name = m.get("default")
            if isinstance(name, str) and name.strip():
                return name.strip()
        if isinstance(m, str) and m.strip():
            return m.strip()
    except Exception:
        pass
    return "gemma4-e2b-hermes"


app = build_app()
