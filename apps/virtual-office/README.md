# Hermes Virtual Office

A local operator UI and backend for running Hermes/Codex workflows through a lightweight control layer.

## What it includes

- FastAPI backend in `backend/`
- React/Vite frontend in `frontend/`
- JSON-backed task, handoff, log, and settings stores in `data/`
- Hermes and Codex adapter bridges in `adapters/`

## Operator workflow

1. Open Task Board
2. Create a task
3. Assign room / agent / priority
4. Trigger Hermes -> task -> Codex
5. Review result in Task Detail
6. Inspect handoff and logs
7. Requeue / retry / run again when needed

## Key routes

Frontend:
- `/`
- `/task-board`
- `/trade-room`
- `/handoff-logs`
- `/console-logs`
- `/agents`
- `/settings`

Backend API:
- `GET /api/health`
- `GET/POST/PATCH /api/tasks`
- `POST /api/tasks/{id}/run`
- `POST /api/tasks/{id}/retry`
- `POST /api/tasks/{id}/requeue`
- `GET/POST /api/handoffs`
- `POST /api/handoffs/{id}/run-again`
- `GET /api/logs`
- `GET /api/logs/{id}`
- `GET /api/logs/stream`
- `GET /api/adapters`
- `POST /api/adapters/codex/exec`
- `GET/PUT /api/settings`

## Local development

Backend:

```bash
cd apps/virtual-office
source .venv/bin/activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8647
```

Frontend:

```bash
cd apps/virtual-office/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

## Verification

Backend tests:

```bash
cd apps/virtual-office
source .venv/bin/activate
python -m pytest backend/tests -q
```

Frontend build:

```bash
cd apps/virtual-office/frontend
npm run build
```

Browser E2E harness:

```bash
cd apps/virtual-office
source .venv/bin/activate
python -m pytest backend/tests/test_browser_e2e.py -q
```

Smoke script:

```bash
cd apps/virtual-office
bash scripts/smoke_phase4.sh
```

## Notes

- Codex remains the auto-run happy path.
- Room assignment is operator workflow metadata unless additional execution routing is added later.
- Settings are persisted in `data/config/settings.json`.
- Set `VIRTUAL_OFFICE_DATA_ROOT=/path/to/tmpdir` to isolate runtime JSON stores for tests or parallel runs.
- Set `VIRTUAL_OFFICE_FAKE_CODEX=1` to force deterministic fake Codex results for automated verification.
