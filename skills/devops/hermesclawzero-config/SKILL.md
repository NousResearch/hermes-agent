---
name: hermesclawzero-config
description: Management and troubleshooting for the HermesClawZero memory synchronization pipeline.
---

# HermesClawZero Configuration & Sync

Tools for maintaining and troubleshooting the HermesClawZero-ConfigSidecar memory integration pipeline.

## Overview
This skill manages the local vector memory pipeline used by HermesClawZero, specifically for syncing chat history and configuration context to a local PostgreSQL instance via an API service.

## Core Components
- `main.py`: FastAPI server handling memory capture and vector search.
- `memory_sync.py`: Background watchdog service that monitors the `sync/` directory for new content and posts it to the API.
- `hermes-skill/scripts/memory.py`: CLI utility for manual interaction with the memory API.

## Workflow Improvements
- **"Gbrain" Directory Pattern**: Use the `inbox/` directory for incoming raw data, `knowledge/` for refined/permanent notes, and `archive/` for processed logs. The sync pipeline (`memory_sync.py`) automatically watches both `sync/` and `inbox/`.
- **Ingestion**: Use `ingest.bat` (Windows) for drag-and-drop file ingestion into the `inbox/`.
- **Environment Automation**: Use `setup.bat`/`setup.ps1` or `setup.sh` to initialize environments. These scripts handle Python/Docker checks, dependency installation (`requirements.txt`), and `.env` generation.

## Troubleshooting Pitfalls
- **Sync service not running**: The integration requires a background process to push data from the `sync/` folder. 
  - Check status: `ps aux | grep memory_sync.py`
  - To start: Ensure you are in the project root (`C:\dev\HermesClawZero-ConfigSidecar`) and run `python memory_sync.py` or the configured start command.
- **Dependency Issues**: Ensure `python-dotenv` and `psutil` are in `requirements.txt`. If the API fails with `ModuleNotFoundError`, rebuild the Docker container (`docker compose up -d --build`).
- **API Unauthorized (401)**: Verify that the `OPENCLAW_KEY` in `.env` matches the server configuration.
- **Database Bloat**: If search performance degrades, run `maintenance.bat` to rebuild vector embeddings.
- **Environment Mismatches**: Ensure `.env` uses `OPENCLAW_URL` and `OPENCLAW_KEY` consistently across `main.py`, `memory_sync.py`, and Docker services.

## Big Data Best Practices
- **Inbox Pattern**: The pipeline watches both `sync/` and `inbox/`. Use `inbox/` for raw ingestion.
- **Archive**: All successfully ingested files are automatically moved to the `archive/` folder by the watchdog.
- **Maintenance**: Run `maintenance.bat` to trigger embedding rebuilds after large data imports.
- **Deduplication**: Semantic duplicate detection is built into the backend API.
