"""HERMES_ONE_MODEL_LIBRARY_COMPAT_V1 — model shortcut library for Hermes One.

Compatibility endpoints for remote/SSH model pickers. Upstream Hermes Agent
exposes ``/api/model/options`` and ``/api/model/set``; Hermes One also needs a
small configured-model shortcut library. Shortcuts live in HERMES_HOME
(``models.json``) so they stay on the remote host across desktop restarts
without changing upstream model-assignment semantics.

Mounted by :func:`mount_hermes_one_model_library` from ``web_server.py`` so the
patch footprint in that god-file stays a few lines (merge-friendly). Full
plugin isolation (``/api/plugins/...``) would break Hermes One clients that
hard-code ``/api/model/library`` — tracked as D-004 follow-up.
"""
from __future__ import annotations

import json
import secrets
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from hermes_cli.config import load_config
from hermes_constants import get_hermes_home

# Marker kept for harness patch-guard (do not rename).
HERMES_ONE_MODEL_LIBRARY_COMPAT_V1 = True


def _hermes_one_model_library_path():
    return get_hermes_home() / "models.json"


def _hermes_one_short_model_label(model):
    text = str(model or "").strip()
    return (text.rsplit("/", 1)[-1] if text else "") or text


def _hermes_one_model_key(row):
    return (
        str(row.get("provider", "")).strip().lower(),
        str(row.get("model", "")).strip().lower(),
        str(row.get("baseUrl", row.get("base_url", ""))).strip().rstrip("/").lower(),
    )


def _hermes_one_normalize_model_row(row, index=0):
    if not isinstance(row, dict):
        return None
    provider = str(row.get("provider", "")).strip()
    model = str(row.get("model", "")).strip()
    if not provider or not model:
        return None
    base_url = str(row.get("baseUrl", row.get("base_url", "")) or "").strip()
    return {
        "id": str(row.get("id") or f"remote:library:{provider}:{index}:{model}"),
        "name": str(row.get("name") or _hermes_one_short_model_label(model) or provider),
        "provider": provider,
        "model": model,
        "baseUrl": base_url,
        "createdAt": row.get("createdAt") if isinstance(row.get("createdAt"), (int, float)) else 0,
    }


def _hermes_one_read_model_library():
    path = _hermes_one_model_library_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        raw = []
    rows = []
    seen = set()
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        row = _hermes_one_normalize_model_row(item, index)
        if not row:
            continue
        key = _hermes_one_model_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _hermes_one_write_model_library(rows):
    path = _hermes_one_model_library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp.replace(path)


def _hermes_one_current_model_row():
    try:
        cfg = load_config()
    except Exception:
        return None
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        provider = str(model_cfg.get("provider", "") or "").strip()
        model = str(model_cfg.get("default", model_cfg.get("name", "")) or "").strip()
        base_url = str(model_cfg.get("base_url", "") or "").strip()
    else:
        provider = ""
        model = str(model_cfg or "").strip()
        base_url = ""
    if not provider or not model:
        return None
    return {
        "id": f"remote:active:{provider}:{model}",
        "name": _hermes_one_short_model_label(model) or provider,
        "provider": provider,
        "model": model,
        "baseUrl": base_url,
        "createdAt": 0,
    }


def mount_hermes_one_model_library(app: FastAPI) -> None:
    """Register ``/api/model/library`` routes on *app* (Hermes One compat)."""

    @app.get("/api/model/library")
    def hermes_one_get_model_library():
        rows = _hermes_one_read_model_library()
        current = _hermes_one_current_model_row()
        if current:
            current_key = _hermes_one_model_key(current)
            rows = [current] + [row for row in rows if _hermes_one_model_key(row) != current_key]
        return {"models": rows}

    @app.post("/api/model/library")
    def hermes_one_add_model_library_row(body: Dict[str, Any]):
        provider = str(body.get("provider", "") or "").strip()
        model = str(body.get("model", "") or "").strip()
        if not provider or not model:
            raise HTTPException(status_code=400, detail="provider and model required")
        base_url = str(body.get("baseUrl", body.get("base_url", "")) or "").strip()
        name = str(body.get("name", "") or "").strip() or _hermes_one_short_model_label(model) or provider
        rows = _hermes_one_read_model_library()
        key = (provider.lower(), model.lower(), base_url.rstrip("/").lower())
        for row in rows:
            if _hermes_one_model_key(row) == key:
                return row
        row = {
            "id": f"remote:library:{secrets.token_hex(8)}",
            "name": name,
            "provider": provider,
            "model": model,
            "baseUrl": base_url,
            "createdAt": int(time.time() * 1000),
        }
        rows.append(row)
        _hermes_one_write_model_library(rows)
        return row

    @app.patch("/api/model/library/{model_id:path}")
    def hermes_one_update_model_library_row(model_id: str, body: Dict[str, Any]):
        rows = _hermes_one_read_model_library()
        for index, row in enumerate(rows):
            if row.get("id") != model_id:
                continue
            next_row = dict(row)
            for key in ("name", "provider", "model"):
                if key in body:
                    next_row[key] = str(body.get(key, "") or "").strip()
            if "baseUrl" in body or "base_url" in body:
                next_row["baseUrl"] = str(body.get("baseUrl", body.get("base_url", "")) or "").strip()
            normalized = _hermes_one_normalize_model_row(next_row, index)
            if not normalized:
                raise HTTPException(status_code=400, detail="provider and model required")
            rows[index] = normalized
            _hermes_one_write_model_library(rows)
            return {"ok": True, "model": normalized}
        raise HTTPException(status_code=404, detail="model not found")

    @app.delete("/api/model/library/{model_id:path}")
    def hermes_one_delete_model_library_row(model_id: str):
        rows = _hermes_one_read_model_library()
        filtered = [row for row in rows if row.get("id") != model_id]
        if len(filtered) == len(rows):
            raise HTTPException(status_code=404, detail="model not found")
        _hermes_one_write_model_library(filtered)
        return {"ok": True}
