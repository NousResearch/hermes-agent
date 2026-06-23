from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import adapters, handoffs, logs, sessions, settings, tasks

app = FastAPI(title="Hermes Virtual Office API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(sessions.router)
app.include_router(tasks.router)
app.include_router(logs.router)
app.include_router(handoffs.router)
app.include_router(adapters.router)
app.include_router(settings.router)
