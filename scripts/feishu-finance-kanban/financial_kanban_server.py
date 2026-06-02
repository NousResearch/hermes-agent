#!/usr/bin/env python3
"""External finance Kanban MCP server for Feishu-oriented workflows."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None


VALID_STATUS = {"queued", "in_progress", "blocked", "done"}
VALID_WORK_TYPES = {
    "research",
    "earnings",
    "valuation",
    "risk",
    "trade-idea",
    "portfolio-review",
}

CARD_STATUS_TO_POOL_STATUS = {
    "queued": "候选",
    "in_progress": "观察中",
    "blocked": "观察中",
    "done": "已入池",
}

NEXUS_PIPELINE = "finance-nexus"
NEXUS_STAGES: list[tuple[str, str, str, list[str]]] = [
    ("T0", "research", "上下文时间线", []),
    ("T1", "research", "宏观/行业/资金", ["T0"]),
    ("T2", "valuation", "技术面/量价", ["T0"]),
    ("T3", "risk", "避雷审查", ["T1", "T2"]),
    ("T4", "risk", "规则校准", ["T3"]),
    ("T5", "trade-idea", "投资决策合成", ["T1", "T2", "T3", "T4"]),
    ("T6", "portfolio-review", "QA门禁", ["T5"]),
]

WORK_TYPE_POOL_FIELD = {
    "earnings": "catalyst",
    "research": "catalyst",
    "valuation": "valuation_summary",
    "risk": "risk_summary",
    "trade-idea": "catalyst",
    "portfolio-review": "risk_summary",
}

NEXUS_STAGE_POOL_FIELD = {
    "T0": "catalyst",
    "T1": "catalyst",
    "T2": "valuation_summary",
    "T3": "risk_summary",
    "T4": "risk_summary",
    "T5": "catalyst",
    "T6": "risk_summary",
}

TERMINAL_CARD_STATUS = {"done", "blocked"}
NEXUS_EXPECTED_STAGES = {stage for stage, *_ in NEXUS_STAGES}


def _default_db_path() -> Path:
    configured = os.environ.get("FINANCIAL_KANBAN_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    hermes_home = Path(
        os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
    ).expanduser()
    return hermes_home / "financial-kanban" / "board.db"


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else _default_db_path()
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS finance_cards (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            symbol TEXT NOT NULL,
            work_type TEXT NOT NULL,
            status TEXT NOT NULL,
            assignee TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            description TEXT,
            chat_id TEXT,
            thread_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS finance_card_deps (
            card_id TEXT NOT NULL,
            parent_id TEXT NOT NULL,
            PRIMARY KEY (card_id, parent_id)
        );
        CREATE TABLE IF NOT EXISTS finance_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            file_path TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_finance_cards_status
            ON finance_cards(status, work_type, symbol);
        CREATE INDEX IF NOT EXISTS idx_finance_artifacts_card
            ON finance_artifacts(card_id, created_at DESC);
        """
    )
    conn.commit()


def _now() -> int:
    return int(time.time())


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_datetime_ms(value: str | int | float | None = None) -> int:
    if value is None or value == "":
        return _now_ms()
    if isinstance(value, (int, float)):
        numeric = int(value)
        return numeric if numeric > 10_000_000_000 else numeric * 1000
    text = str(value).strip()
    if not text:
        return _now_ms()
    if text.isdigit():
        numeric = int(text)
        return numeric if numeric > 10_000_000_000 else numeric * 1000
    raise ValueError(f"datetime field must be unix ms or digits, got: {value!r}")


def _infer_market(symbol: str) -> str:
    code = (symbol or "").strip().upper()
    if re.fullmatch(r"\d{6}", code):
        return "A股"
    if re.fullmatch(r"\d{5}", code):
        return "港股"
    return "美股"


def _auto_sync_mode() -> str:
    return os.environ.get("FINANCE_STOCK_POOL_SYNC_MODE", "artifact").strip().lower()


def _auto_sync_on_complete_enabled() -> bool:
    return _env_flag("FINANCE_STOCK_POOL_AUTO_SYNC_ON_COMPLETE", default=True)


def _should_auto_sync_card(*, event: str) -> bool:
    mode = _auto_sync_mode()
    if mode in {"off", "false", "0", "none"}:
        return False
    if mode in {"symbol", "manual"}:
        return False
    if mode in {"artifact", "on_artifact"}:
        return event == "artifact"
    if mode in {"done", "on_done"}:
        return event == "done"
    if mode in {"all", "always"}:
        return True
    return event == "artifact"


def _card_id() -> str:
    return f"fk_{uuid.uuid4().hex[:12]}"


def _normalize_status(status: str) -> str:
    value = (status or "").strip().lower()
    if value not in VALID_STATUS:
        raise ValueError(f"status must be one of {sorted(VALID_STATUS)}")
    return value


def _normalize_work_type(work_type: str) -> str:
    value = (work_type or "").strip().lower()
    if value not in VALID_WORK_TYPES:
        raise ValueError(f"work_type must be one of {sorted(VALID_WORK_TYPES)}")
    return value


def _json(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _parse_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(value)


def _trim_excerpt(text: str, *, limit: int = 280) -> str:
    compact = " ".join((text or "").strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _slug(text: str) -> str:
    lowered = (text or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "global"


def _default_workflow_id(symbol: str, chat_id: str = "", thread_id: str = "") -> str:
    parts = ["finance-kanban", _slug(symbol)]
    if chat_id:
        parts.append(_slug(chat_id))
    if thread_id:
        parts.append(_slug(thread_id))
    return ":".join(parts)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _base_settings() -> dict[str, str]:
    return {
        "base_token": os.environ.get("FINANCE_STOCK_POOL_BASE_TOKEN", "").strip(),
        "table_id": os.environ.get("FINANCE_STOCK_POOL_TABLE_ID", "").strip(),
        "view_id": os.environ.get("FINANCE_STOCK_POOL_VIEW_ID", "").strip(),
        "identity": os.environ.get("FINANCE_STOCK_POOL_IDENTITY", "user").strip() or "user",
    }


def _stock_pool_sync_enabled() -> bool:
    settings = _base_settings()
    return (
        _env_flag("FINANCE_STOCK_POOL_SYNC", default=False)
        and bool(settings["base_token"])
        and bool(settings["table_id"])
    )


def _run_lark_cli_json(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"lark-cli failed: {message}")
    output = proc.stdout.strip()
    if not output:
        raise RuntimeError("lark-cli returned empty output")
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"lark-cli did not return JSON: {output[:240]}") from exc


def _search_stock_pool_by_symbol(symbol: str) -> dict[str, Any]:
    settings = _base_settings()
    payload = {
        "keyword": symbol.strip().upper(),
        "search_fields": ["股票代码"],
        "select_fields": ["股票代码", "股票名称", "状态", "workflow_id"],
        "limit": 10,
    }
    return _run_lark_cli_json(
        [
            "lark-cli",
            "base",
            "+record-search",
            "--as",
            settings["identity"],
            "--base-token",
            settings["base_token"],
            "--table-id",
            settings["table_id"],
            "--json",
            json.dumps(payload, ensure_ascii=False),
            "--format",
            "json",
        ]
    )


def _upsert_stock_pool_row(
    *,
    symbol: str,
    stock_name: str = "",
    market: str = "",
    pool_status: str = "",
    strategy_style: str = "",
    composite_score: float | None = None,
    catalyst: str = "",
    valuation_summary: str = "",
    risk_summary: str = "",
    next_action: str = "",
    workflow_id: str = "",
    pool_date: str = "",
    last_review_date: str = "",
    record_id: str = "",
) -> dict[str, Any]:
    settings = _base_settings()
    target_record_id = record_id.strip()
    if not target_record_id:
        search = _search_stock_pool_by_symbol(symbol)
        record_ids = search.get("data", {}).get("record_id_list", [])
        target_record_id = str(record_ids[0]).strip() if record_ids else ""

    field_map: dict[str, Any] = {"股票代码": symbol.strip().upper()}
    optional_text = {
        "股票名称": stock_name.strip(),
        "市场": market.strip() or _infer_market(symbol),
        "状态": pool_status.strip(),
        "策略风格": strategy_style.strip(),
        "催化剂": catalyst.strip(),
        "估值摘要": valuation_summary.strip(),
        "风险摘要": risk_summary.strip(),
        "下一步动作": next_action.strip(),
        "workflow_id": workflow_id.strip(),
    }
    for key, value in optional_text.items():
        if value:
            field_map[key] = value
    if composite_score is not None:
        field_map["综合评分"] = float(composite_score)
    field_map["日期"] = _normalize_datetime_ms(pool_date or None)
    field_map["最后复盘日"] = _normalize_datetime_ms(last_review_date or None)
    if pool_date or not target_record_id:
        field_map["入池日期"] = _normalize_datetime_ms(pool_date or None)

    args = [
        "lark-cli",
        "base",
        "+record-upsert",
        "--as",
        settings["identity"],
        "--base-token",
        settings["base_token"],
        "--table-id",
        settings["table_id"],
        "--json",
        json.dumps(field_map, ensure_ascii=False),
    ]
    if target_record_id:
        args.extend(["--record-id", target_record_id])
    result = _run_lark_cli_json(args)
    upsert_record_ids = result.get("data", {}).get("record", {}).get("record_id_list", [])
    return {
        "action": "updated" if target_record_id else "created",
        "record_id": (upsert_record_ids[0] if upsert_record_ids else target_record_id),
        "fields": field_map,
    }


def _delete_stock_pool_row(*, symbol: str = "", record_id: str = "") -> dict[str, Any]:
    settings = _base_settings()
    target_record_id = record_id.strip()
    if not target_record_id:
        if not symbol.strip():
            raise ValueError("symbol or record_id is required")
        search = _search_stock_pool_by_symbol(symbol)
        record_ids = search.get("data", {}).get("record_id_list", [])
        if not record_ids:
            return {"deleted": False, "reason": "record_not_found", "symbol": symbol.strip().upper()}
        target_record_id = str(record_ids[0]).strip()
    result = _run_lark_cli_json(
        [
            "lark-cli",
            "base",
            "+record-delete",
            "--as",
            settings["identity"],
            "--base-token",
            settings["base_token"],
            "--table-id",
            settings["table_id"],
            "--record-id",
            target_record_id,
            "--yes",
        ]
    )
    return {
        "deleted": True,
        "record_id": target_record_id,
        "response": result.get("data", {}),
    }


def _stock_pool_query(*, keyword: str, search_fields: list[str], select_fields: list[str], limit: int = 20) -> dict[str, Any]:
    settings = _base_settings()
    payload = {
        "keyword": keyword,
        "search_fields": search_fields,
        "select_fields": select_fields,
        "limit": max(1, min(int(limit), 50)),
    }
    if settings["view_id"]:
        payload["view_id"] = settings["view_id"]
    return _run_lark_cli_json(
        [
            "lark-cli",
            "base",
            "+record-search",
            "--as",
            settings["identity"],
            "--base-token",
            settings["base_token"],
            "--table-id",
            settings["table_id"],
            "--json",
            json.dumps(payload, ensure_ascii=False),
            "--format",
            "json",
        ]
    )


def _latest_artifact_payload(conn: sqlite3.Connection, card_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT kind, content, file_path
        FROM finance_artifacts
        WHERE card_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (card_id,),
    ).fetchone()
    if row is None:
        return {}
    return {
        "kind": row["kind"],
        "content": row["content"],
        "file_path": row["file_path"] or "",
    }


def _card_pool_field_updates(card: dict[str, Any], artifact_excerpt: str) -> dict[str, str]:
    stage = str(card.get("metadata", {}).get("nexus_stage", "")).strip().upper()
    field_name = NEXUS_STAGE_POOL_FIELD.get(stage) or WORK_TYPE_POOL_FIELD.get(
        card.get("work_type", ""), "catalyst"
    )
    return {field_name: artifact_excerpt}


def sync_card_to_stock_pool(
    conn: sqlite3.Connection,
    *,
    card_id: str,
    stock_name: str = "",
    market: str = "",
    strategy_style: str = "",
    composite_score: float | None = None,
    catalyst: str = "",
    valuation_summary: str = "",
    risk_summary: str = "",
    next_action: str = "",
    pool_date: str = "",
    last_review_date: str = "",
    pool_status: str = "",
) -> dict[str, Any]:
    if not _stock_pool_sync_enabled():
        raise RuntimeError("Feishu stock pool sync is not enabled")
    card = get_card(conn, card_id)
    latest_artifact = _latest_artifact_payload(conn, card_id)
    status_text = pool_status.strip() or CARD_STATUS_TO_POOL_STATUS.get(card["status"], "观察中")
    artifact_excerpt = _trim_excerpt(latest_artifact.get("content", ""), limit=500)
    mapped = _card_pool_field_updates(card, artifact_excerpt)
    return _upsert_stock_pool_row(
        symbol=card["symbol"],
        stock_name=stock_name.strip() or str(card["metadata"].get("stock_name", "")).strip(),
        market=market.strip() or _infer_market(card["symbol"]),
        pool_status=status_text,
        strategy_style=strategy_style.strip() or str(card["metadata"].get("strategy_style", "")).strip(),
        composite_score=composite_score,
        catalyst=catalyst.strip() or mapped.get("catalyst", ""),
        valuation_summary=valuation_summary.strip() or mapped.get("valuation_summary", ""),
        risk_summary=risk_summary.strip() or mapped.get("risk_summary", ""),
        next_action=next_action.strip() or "继续观察",
        workflow_id=str(card["metadata"].get("workflow_id", "")).strip(),
        pool_date=pool_date,
        last_review_date=last_review_date,
    )


def _list_symbol_cards(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    pipeline: str = "",
) -> list[dict[str, Any]]:
    cards = list_cards(conn, symbol=symbol)
    if not pipeline:
        return cards
    return [
        card
        for card in cards
        if str(card.get("metadata", {}).get("pipeline", "")).strip() == pipeline
    ]


def _aggregate_symbol_fields(cards: list[dict[str, Any]], conn: sqlite3.Connection) -> dict[str, Any]:
    catalyst_parts: list[str] = []
    valuation_parts: list[str] = []
    risk_parts: list[str] = []
    workflow_id = ""
    stock_name = ""
    strategy_style = ""
    composite_score: float | None = None
    statuses: list[str] = []
    for card in cards:
        statuses.append(card["status"])
        meta = card.get("metadata", {})
        workflow_id = workflow_id or str(meta.get("workflow_id", "")).strip()
        stock_name = stock_name or str(meta.get("stock_name", "")).strip()
        strategy_style = strategy_style or str(meta.get("strategy_style", "")).strip()
        if composite_score is None and meta.get("composite_score") is not None:
            try:
                composite_score = float(meta["composite_score"])
            except (TypeError, ValueError):
                pass
        latest = _latest_artifact_payload(conn, card["id"])
        excerpt = _trim_excerpt(latest.get("content", ""), limit=400)
        if not excerpt:
            continue
        stage = str(meta.get("nexus_stage", "")).strip().upper()
        bucket = NEXUS_STAGE_POOL_FIELD.get(stage) or WORK_TYPE_POOL_FIELD.get(card["work_type"], "catalyst")
        label = stage or card["work_type"]
        line = f"[{label}] {excerpt}"
        if bucket == "catalyst":
            catalyst_parts.append(line)
        elif bucket == "valuation_summary":
            valuation_parts.append(line)
        else:
            risk_parts.append(line)
    if any(status == "blocked" for status in statuses):
        pool_status = "观察中"
    elif statuses and all(status == "done" for status in statuses):
        pool_status = "已入池"
    elif any(status == "in_progress" for status in statuses):
        pool_status = "观察中"
    else:
        pool_status = "候选"
    symbol = cards[0]["symbol"] if cards else ""
    return {
        "symbol": symbol,
        "stock_name": stock_name,
        "market": _infer_market(symbol),
        "pool_status": pool_status,
        "strategy_style": strategy_style,
        "composite_score": composite_score,
        "catalyst": "\n".join(catalyst_parts)[:1500],
        "valuation_summary": "\n".join(valuation_parts)[:1500],
        "risk_summary": "\n".join(risk_parts)[:1500],
        "next_action": "继续观察" if pool_status != "已入池" else "深度分析",
        "workflow_id": workflow_id,
    }


def sync_symbol_to_stock_pool(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    pipeline: str = NEXUS_PIPELINE,
    workflow_id: str = "",
    stock_name: str = "",
    strategy_style: str = "",
    composite_score: float | None = None,
    next_action: str = "",
    pool_status: str = "",
) -> dict[str, Any]:
    if not _stock_pool_sync_enabled():
        raise RuntimeError("Feishu stock pool sync is not enabled")
    cards = _list_symbol_cards(conn, symbol=symbol, pipeline=pipeline) if pipeline else list_cards(
        conn, symbol=symbol
    )
    if workflow_id.strip():
        cards = [
            card
            for card in cards
            if str(card.get("metadata", {}).get("workflow_id", "")).strip() == workflow_id.strip()
        ]
    if not cards:
        cards = list_cards(conn, symbol=symbol)
        if workflow_id.strip():
            cards = [
                card
                for card in cards
                if str(card.get("metadata", {}).get("workflow_id", "")).strip() == workflow_id.strip()
            ]
    if not cards:
        raise KeyError(f"no finance cards for symbol: {symbol}")
    payload = _aggregate_symbol_fields(cards, conn)
    if stock_name.strip():
        payload["stock_name"] = stock_name.strip()
    if strategy_style.strip():
        payload["strategy_style"] = strategy_style.strip()
    if composite_score is not None:
        payload["composite_score"] = composite_score
    if next_action.strip():
        payload["next_action"] = next_action.strip()
    if pool_status.strip():
        payload["pool_status"] = pool_status.strip()
    return _upsert_stock_pool_row(**payload)


def create_nexus_analysis(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    chat_id: str = "",
    thread_id: str = "",
    assignee: str = "feishu-analyst",
    stock_name: str = "",
    strategy_style: str = "balanced",
    idempotency_key: str = "",
) -> dict[str, Any]:
    symbol = (symbol or "").strip().upper()
    if not re.fullmatch(r"\d{6}", symbol):
        raise ValueError("nexus analysis requires a 6-digit A-share symbol")
    workflow_id = _default_workflow_id(symbol=symbol, chat_id=chat_id, thread_id=thread_id)
    if idempotency_key.strip():
        existing = [
            card
            for card in _list_symbol_cards(conn, symbol=symbol, pipeline=NEXUS_PIPELINE)
            if str(card.get("metadata", {}).get("idempotency_key", "")).strip() == idempotency_key.strip()
        ]
        if existing:
            return {
                "symbol": symbol,
                "workflow_id": workflow_id,
                "idempotent": True,
                "cards": [{"stage": c["metadata"].get("nexus_stage"), "card_id": c["id"]} for c in existing],
            }
    stage_ids: dict[str, str] = {}
    created: list[dict[str, str]] = []
    for stage, work_type, label, parent_stages in NEXUS_STAGES:
        parent_ids = [stage_ids[item] for item in parent_stages if item in stage_ids]
        card = create_card(
            conn,
            title=f"[nexus] {symbol} {label}",
            symbol=symbol,
            work_type=work_type,
            assignee=assignee,
            description=f"Finance Nexus stage {stage} for {symbol}",
            status="queued",
            chat_id=chat_id,
            thread_id=thread_id,
            parent_ids=parent_ids,
            metadata={
                "pipeline": NEXUS_PIPELINE,
                "nexus_stage": stage,
                "workflow_id": workflow_id,
                "stock_name": stock_name,
                "strategy_style": strategy_style,
                "market": _infer_market(symbol),
                "idempotency_key": idempotency_key.strip(),
            },
        )
        stage_ids[stage] = card["id"]
        created.append({"stage": stage, "card_id": card["id"], "work_type": work_type})
    return {
        "symbol": symbol,
        "workflow_id": workflow_id,
        "pipeline": NEXUS_PIPELINE,
        "cards": created,
        "feishu_delivery": "After each add_finance_artifact, call render_feishu_card_update then send_message. When all T0-T6 are done/blocked with artifacts, stock pool Base auto-syncs (FINANCE_STOCK_POOL_AUTO_SYNC_ON_COMPLETE).",
    }


def render_symbol_feishu_summary(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    pipeline: str = NEXUS_PIPELINE,
) -> str:
    cards = _list_symbol_cards(conn, symbol=symbol, pipeline=pipeline) or list_cards(conn, symbol=symbol)
    if not cards:
        raise KeyError(f"no cards for symbol: {symbol}")
    lines = [f"Finance Nexus Summary — {symbol.upper()}", f"Cards: {len(cards)}"]
    workflow_id = str(cards[0].get("metadata", {}).get("workflow_id", "")).strip()
    if workflow_id:
        lines.append(f"Workflow: {workflow_id}")
    for card in sorted(cards, key=lambda item: str(item.get("metadata", {}).get("nexus_stage", ""))):
        stage = card.get("metadata", {}).get("nexus_stage", card["work_type"])
        lines.append(
            f"- {stage} `{card['id']}` [{card['status']}] {_trim_excerpt(card.get('latest_artifact_excerpt', ''), limit=120)}"
        )
    return "\n".join(lines)


def _auto_sync_card(conn: sqlite3.Connection, card_id: str, *, event: str) -> dict[str, Any] | None:
    if not _stock_pool_sync_enabled() or not _should_auto_sync_card(event=event):
        return None
    return sync_card_to_stock_pool(conn, card_id=card_id)


def _patch_card_metadata(conn: sqlite3.Connection, card_id: str, patch: dict[str, Any]) -> None:
    card = get_card(conn, card_id)
    merged = dict(card.get("metadata") or {})
    merged.update(patch)
    conn.execute(
        "UPDATE finance_cards SET metadata_json = ?, updated_at = ? WHERE id = ?",
        (_json(merged), _now(), card_id),
    )
    conn.commit()


def _nexus_batch_cards(conn: sqlite3.Connection, card: dict[str, Any]) -> list[dict[str, Any]]:
    cards = _list_symbol_cards(conn, symbol=card["symbol"], pipeline=NEXUS_PIPELINE)
    workflow_id = str(card.get("metadata", {}).get("workflow_id", "")).strip()
    idempotency_key = str(card.get("metadata", {}).get("idempotency_key", "")).strip()
    if idempotency_key:
        return [
            item
            for item in cards
            if str(item.get("metadata", {}).get("idempotency_key", "")).strip() == idempotency_key
        ]
    if workflow_id:
        return [
            item
            for item in cards
            if str(item.get("metadata", {}).get("workflow_id", "")).strip() == workflow_id
        ]
    return cards


def _is_nexus_batch_complete(cards: list[dict[str, Any]]) -> bool:
    if not cards:
        return False
    stages_present = {
        str(card.get("metadata", {}).get("nexus_stage", "")).strip().upper()
        for card in cards
    }
    if not NEXUS_EXPECTED_STAGES.issubset(stages_present):
        return False
    batch_cards = [
        card
        for card in cards
        if str(card.get("metadata", {}).get("nexus_stage", "")).strip().upper() in NEXUS_EXPECTED_STAGES
    ]
    if len(batch_cards) < len(NEXUS_STAGES):
        return False
    if not all(card["status"] in TERMINAL_CARD_STATUS for card in batch_cards):
        return False
    return all(int(card.get("artifact_count") or 0) > 0 for card in batch_cards)


def _generic_batch_cards(conn: sqlite3.Connection, card: dict[str, Any]) -> list[dict[str, Any]]:
    workflow_id = str(card.get("metadata", {}).get("workflow_id", "")).strip()
    if not workflow_id:
        return []
    cards = list_cards(conn, symbol=card["symbol"])
    return [
        item
        for item in cards
        if str(item.get("metadata", {}).get("workflow_id", "")).strip() == workflow_id
        and str(item.get("metadata", {}).get("pipeline", "")).strip() != NEXUS_PIPELINE
    ]


def _is_generic_batch_complete(cards: list[dict[str, Any]]) -> bool:
    if not cards:
        return False
    if not all(card["status"] in TERMINAL_CARD_STATUS for card in cards):
        return False
    return all(int(card.get("artifact_count") or 0) > 0 for card in cards)


def _batch_sync_token(card: dict[str, Any], *, kind: str) -> str:
    meta = card.get("metadata", {})
    workflow_id = str(meta.get("workflow_id", "")).strip()
    idempotency_key = str(meta.get("idempotency_key", "")).strip()
    if kind == "nexus":
        return f"nexus-final:{workflow_id}:{idempotency_key}"
    return f"finance-final:{workflow_id}"


def _batch_already_synced(cards: list[dict[str, Any]], token: str) -> bool:
    for card in cards:
        if str(card.get("metadata", {}).get("stock_pool_batch_synced", "")).strip() == token:
            return True
    return False


def _mark_batch_synced(conn: sqlite3.Connection, cards: list[dict[str, Any]], token: str) -> None:
    for card in cards:
        _patch_card_metadata(
            conn,
            card["id"],
            {"stock_pool_batch_synced": token, "stock_pool_batch_synced_at": _now()},
        )


def _auto_sync_on_batch_complete(conn: sqlite3.Connection, card_id: str) -> dict[str, Any] | None:
    """When all cards in a Nexus or generic finance batch finish, upsert stock pool once."""
    if not _stock_pool_sync_enabled() or not _auto_sync_on_complete_enabled():
        return None
    card = get_card(conn, card_id)
    pipeline = str(card.get("metadata", {}).get("pipeline", "")).strip()

    if pipeline == NEXUS_PIPELINE:
        batch_cards = _nexus_batch_cards(conn, card)
        if not _is_nexus_batch_complete(batch_cards):
            return None
        token = _batch_sync_token(card, kind="nexus")
        if _batch_already_synced(batch_cards, token):
            return {
                "auto_sync": "skipped",
                "reason": "already_synced",
                "symbol": card["symbol"],
                "batch_token": token,
            }
        pool_sync = sync_symbol_to_stock_pool(
            conn,
            symbol=card["symbol"],
            pipeline=NEXUS_PIPELINE,
            workflow_id=str(card.get("metadata", {}).get("workflow_id", "")).strip(),
        )
        _mark_batch_synced(conn, batch_cards, token)
        return {
            "auto_sync": True,
            "batch_kind": "nexus",
            "symbol": card["symbol"],
            "batch_token": token,
            "cards_total": len(batch_cards),
            "stock_pool_sync": pool_sync,
        }

    batch_cards = _generic_batch_cards(conn, card)
    if not _is_generic_batch_complete(batch_cards):
        return None
    token = _batch_sync_token(card, kind="generic")
    if _batch_already_synced(batch_cards, token):
        return {
            "auto_sync": "skipped",
            "reason": "already_synced",
            "symbol": card["symbol"],
            "batch_token": token,
        }
    workflow_id = str(card.get("metadata", {}).get("workflow_id", "")).strip()
    pool_sync = sync_symbol_to_stock_pool(
        conn,
        symbol=card["symbol"],
        pipeline="",
        workflow_id=workflow_id,
    )
    _mark_batch_synced(conn, batch_cards, token)
    return {
        "auto_sync": True,
        "batch_kind": "generic",
        "symbol": card["symbol"],
        "batch_token": token,
        "cards_total": len(batch_cards),
        "stock_pool_sync": pool_sync,
    }


def _normalize_card_metadata(
    *,
    symbol: str,
    chat_id: str,
    thread_id: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(metadata or {})
    workflow_id = str(
        merged.get("workflow_id")
        or _default_workflow_id(symbol=symbol, chat_id=chat_id, thread_id=thread_id)
    )
    merged["workflow_id"] = workflow_id
    merged.setdefault("memory_session_id", f"{workflow_id}:session")
    merged.setdefault("memory_title", f"{symbol.upper()} finance kanban")
    existing_tags = merged.get("memory_tags")
    if isinstance(existing_tags, list):
        tags = [str(item).strip() for item in existing_tags if str(item).strip()]
    else:
        tags = []
    defaults = ["finance", "kanban", symbol.lower()]
    if chat_id:
        defaults.append("feishu")
    for tag in defaults:
        if tag not in tags:
            tags.append(tag)
    merged["memory_tags"] = tags
    return merged


def create_card(
    conn: sqlite3.Connection,
    *,
    title: str,
    symbol: str,
    work_type: str,
    assignee: str,
    description: str = "",
    status: str = "queued",
    priority: int = 0,
    chat_id: str = "",
    thread_id: str = "",
    parent_ids: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    title = (title or "").strip()
    symbol = (symbol or "").strip().upper()
    assignee = (assignee or "").strip()
    if not title:
        raise ValueError("title is required")
    if not symbol:
        raise ValueError("symbol is required")
    if not assignee:
        raise ValueError("assignee is required")
    work_type = _normalize_work_type(work_type)
    status = _normalize_status(status)
    created_at = _now()
    card_id = _card_id()
    metadata = _normalize_card_metadata(
        symbol=symbol, chat_id=chat_id, thread_id=thread_id, metadata=metadata
    )
    with conn:
        conn.execute(
            """
            INSERT INTO finance_cards (
                id, title, symbol, work_type, status, assignee, priority,
                description, chat_id, thread_id, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card_id,
                title,
                symbol,
                work_type,
                status,
                assignee,
                int(priority),
                description or "",
                chat_id or "",
                thread_id or "",
                _json(metadata),
                created_at,
                created_at,
            ),
        )
        for parent_id in parent_ids or ():
            parent_text = str(parent_id).strip()
            if not parent_text:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO finance_card_deps(card_id, parent_id) VALUES (?, ?)",
                (card_id, parent_text),
            )
    return get_card(conn, card_id)


def get_card(conn: sqlite3.Connection, card_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM finance_cards WHERE id = ?", (card_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown card: {card_id}")
    parents = [
        dep["parent_id"]
        for dep in conn.execute(
            "SELECT parent_id FROM finance_card_deps WHERE card_id = ? ORDER BY parent_id",
            (card_id,),
        ).fetchall()
    ]
    latest_artifact = conn.execute(
        """
        SELECT kind, content, file_path, created_at
        FROM finance_artifacts
        WHERE card_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (card_id,),
    ).fetchone()
    artifact_count = conn.execute(
        "SELECT COUNT(*) AS count FROM finance_artifacts WHERE card_id = ?",
        (card_id,),
    ).fetchone()["count"]
    return {
        "id": row["id"],
        "title": row["title"],
        "symbol": row["symbol"],
        "work_type": row["work_type"],
        "status": row["status"],
        "assignee": row["assignee"],
        "priority": row["priority"],
        "description": row["description"] or "",
        "chat_id": row["chat_id"] or "",
        "thread_id": row["thread_id"] or "",
        "metadata": _parse_json(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "parent_ids": parents,
        "artifact_count": artifact_count,
        "latest_artifact_excerpt": (
            _trim_excerpt(latest_artifact["content"]) if latest_artifact else ""
        ),
    }


def list_cards(
    conn: sqlite3.Connection,
    *,
    status: str = "",
    symbol: str = "",
    work_type: str = "",
    assignee: str = "",
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(_normalize_status(status))
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol.strip().upper())
    if work_type:
        clauses.append("work_type = ?")
        params.append(_normalize_work_type(work_type))
    if assignee:
        clauses.append("assignee = ?")
        params.append(assignee.strip())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT id FROM finance_cards
        {where}
        ORDER BY priority DESC, updated_at DESC, id DESC
        """,
        params,
    ).fetchall()
    return [get_card(conn, row["id"]) for row in rows]


def transition_card(
    conn: sqlite3.Connection,
    *,
    card_id: str,
    status: str,
    note: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = _normalize_status(status)
    updated_at = _now()
    with conn:
        updated = conn.execute(
            "UPDATE finance_cards SET status = ?, updated_at = ? WHERE id = ?",
            (status, updated_at, card_id),
        )
        if updated.rowcount != 1:
            raise KeyError(f"unknown card: {card_id}")
        if note.strip():
            conn.execute(
                """
                INSERT INTO finance_artifacts(card_id, kind, content, file_path, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    card_id,
                    "status-note",
                    note.strip(),
                    "",
                    _json(metadata),
                    updated_at,
                ),
                )
    synced = _auto_sync_card(conn, card_id, event="done")
    final_sync = _auto_sync_on_batch_complete(conn, card_id)
    card = get_card(conn, card_id)
    if synced is not None:
        card["stock_pool_sync"] = synced
    if final_sync is not None:
        card["stock_pool_final_sync"] = final_sync
    return card


def add_artifact(
    conn: sqlite3.Connection,
    *,
    card_id: str,
    kind: str,
    content: str,
    file_path: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not kind.strip():
        raise ValueError("kind is required")
    if not content.strip():
        raise ValueError("content is required")
    created_at = _now()
    with conn:
        exists = conn.execute("SELECT 1 FROM finance_cards WHERE id = ?", (card_id,)).fetchone()
        if exists is None:
            raise KeyError(f"unknown card: {card_id}")
        cur = conn.execute(
            """
            INSERT INTO finance_artifacts(card_id, kind, content, file_path, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                card_id,
                kind.strip(),
                content.strip(),
                file_path.strip(),
                _json(metadata),
                created_at,
            ),
        )
        conn.execute("UPDATE finance_cards SET updated_at = ? WHERE id = ?", (created_at, card_id))
    result = {
        "id": cur.lastrowid,
        "card_id": card_id,
        "kind": kind.strip(),
        "content_excerpt": _trim_excerpt(content),
        "file_path": file_path.strip(),
        "metadata": metadata or {},
        "created_at": created_at,
    }
    synced = _auto_sync_card(conn, card_id, event="artifact")
    if synced is not None:
        result["stock_pool_sync"] = synced
    final_sync = _auto_sync_on_batch_complete(conn, card_id)
    if final_sync is not None:
        result["stock_pool_final_sync"] = final_sync
    return result


def render_feishu_update(
    conn: sqlite3.Connection,
    *,
    card_id: str,
    artifact_limit: int = 3,
) -> str:
    card = get_card(conn, card_id)
    artifacts = conn.execute(
        """
        SELECT kind, content, file_path
        FROM finance_artifacts
        WHERE card_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (card_id, int(artifact_limit)),
    ).fetchall()
    lines = [
        f"Finance Kanban Update: {card['title']}",
        f"Card: {card['id']}",
        f"Symbol: {card['symbol']} | Type: {card['work_type']} | Status: {card['status']}",
        f"Assignee: {card['assignee']}",
    ]
    workflow_id = str(card["metadata"].get("workflow_id", "")).strip()
    if workflow_id:
        lines.append(f"Memory OS Workflow: {workflow_id}")
    if card["chat_id"]:
        scope = f"Chat: {card['chat_id']}"
        if card["thread_id"]:
            scope += f" | Thread: {card['thread_id']}"
        lines.append(scope)
    if card["description"]:
        lines.append(f"Brief: {_trim_excerpt(card['description'], limit=180)}")
    if card["parent_ids"]:
        lines.append("Depends on: " + ", ".join(card["parent_ids"]))
    if artifacts:
        lines.append("Latest work:")
        for item in artifacts:
            excerpt = _trim_excerpt(item["content"], limit=220)
            label = item["kind"]
            if item["file_path"]:
                label = f"{label} [{item['file_path']}]"
            lines.append(f"- {label}: {excerpt}")
    else:
        lines.append("Latest work: no artifacts yet. Add analyst output before sending a final Feishu update.")
    return "\n".join(lines)


def _create_mcp() -> Any:
    if FastMCP is None:
        raise RuntimeError("FastMCP is not installed. Run: pip install fastmcp")
    mcp = FastMCP("financial-kanban")

    @mcp.tool
    def create_finance_card(
        title: str,
        symbol: str,
        work_type: str,
        assignee: str,
        description: str = "",
        status: str = "queued",
        priority: int = 0,
        chat_id: str = "",
        thread_id: str = "",
        parent_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with _connect() as conn:
            return create_card(
                conn,
                title=title,
                symbol=symbol,
                work_type=work_type,
                assignee=assignee,
                description=description,
                status=status,
                priority=priority,
                chat_id=chat_id,
                thread_id=thread_id,
                parent_ids=parent_ids,
                metadata=metadata,
            )

    @mcp.tool
    def list_finance_cards(
        status: str = "",
        symbol: str = "",
        work_type: str = "",
        assignee: str = "",
    ) -> list[dict[str, Any]]:
        with _connect() as conn:
            return list_cards(conn, status=status, symbol=symbol, work_type=work_type, assignee=assignee)

    @mcp.tool
    def get_finance_card(card_id: str) -> dict[str, Any]:
        with _connect() as conn:
            return get_card(conn, card_id)

    @mcp.tool
    def transition_finance_card(
        card_id: str,
        status: str,
        note: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with _connect() as conn:
            return transition_card(conn, card_id=card_id, status=status, note=note, metadata=metadata)

    @mcp.tool
    def add_finance_artifact(
        card_id: str,
        kind: str,
        content: str,
        file_path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with _connect() as conn:
            return add_artifact(
                conn,
                card_id=card_id,
                kind=kind,
                content=content,
                file_path=file_path,
                metadata=metadata,
            )

    @mcp.tool
    def render_feishu_card_update(card_id: str, artifact_limit: int = 3) -> dict[str, Any]:
        with _connect() as conn:
            card = get_card(conn, card_id)
            return {
                "card_id": card_id,
                "chat_id": card["chat_id"],
                "thread_id": card["thread_id"],
                "content": render_feishu_update(conn, card_id=card_id, artifact_limit=artifact_limit),
            }

    @mcp.tool
    def sync_finance_card_to_stock_pool(
        card_id: str,
        stock_name: str = "",
        market: str = "",
        strategy_style: str = "",
        composite_score: float | None = None,
        catalyst: str = "",
        valuation_summary: str = "",
        risk_summary: str = "",
        next_action: str = "",
        pool_date: str = "",
        last_review_date: str = "",
        pool_status: str = "",
    ) -> dict[str, Any]:
        with _connect() as conn:
            return sync_card_to_stock_pool(
                conn,
                card_id=card_id,
                stock_name=stock_name,
                market=market,
                strategy_style=strategy_style,
                composite_score=composite_score,
                catalyst=catalyst,
                valuation_summary=valuation_summary,
                risk_summary=risk_summary,
                next_action=next_action,
                pool_date=pool_date,
                last_review_date=last_review_date,
                pool_status=pool_status,
            )

    @mcp.tool
    def search_stock_pool_records(
        keyword: str,
        search_fields: list[str] | None = None,
        select_fields: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        if not _stock_pool_sync_enabled():
            raise RuntimeError("Feishu stock pool sync is not enabled")
        return _stock_pool_query(
            keyword=keyword.strip(),
            search_fields=search_fields or ["股票代码", "股票名称"],
            select_fields=select_fields
            or ["股票代码", "股票名称", "状态", "综合评分", "下一步动作", "workflow_id"],
            limit=limit,
        )

    @mcp.tool
    def upsert_stock_pool_record(
        symbol: str,
        stock_name: str = "",
        market: str = "",
        pool_status: str = "",
        strategy_style: str = "",
        composite_score: float | None = None,
        catalyst: str = "",
        valuation_summary: str = "",
        risk_summary: str = "",
        next_action: str = "",
        workflow_id: str = "",
        pool_date: str = "",
        last_review_date: str = "",
        record_id: str = "",
    ) -> dict[str, Any]:
        if not _stock_pool_sync_enabled():
            raise RuntimeError("Feishu stock pool sync is not enabled")
        return _upsert_stock_pool_row(
            symbol=symbol,
            stock_name=stock_name,
            market=market,
            pool_status=pool_status,
            strategy_style=strategy_style,
            composite_score=composite_score,
            catalyst=catalyst,
            valuation_summary=valuation_summary,
            risk_summary=risk_summary,
            next_action=next_action,
            workflow_id=workflow_id,
            pool_date=pool_date,
            last_review_date=last_review_date,
            record_id=record_id,
        )

    @mcp.tool
    def delete_stock_pool_record(symbol: str = "", record_id: str = "") -> dict[str, Any]:
        if not _stock_pool_sync_enabled():
            raise RuntimeError("Feishu stock pool sync is not enabled")
        return _delete_stock_pool_row(symbol=symbol, record_id=record_id)

    @mcp.tool
    def create_finance_nexus_analysis(
        symbol: str,
        chat_id: str = "",
        thread_id: str = "",
        assignee: str = "feishu-analyst",
        stock_name: str = "",
        strategy_style: str = "balanced",
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        with _connect() as conn:
            return create_nexus_analysis(
                conn,
                symbol=symbol,
                chat_id=chat_id,
                thread_id=thread_id,
                assignee=assignee,
                stock_name=stock_name,
                strategy_style=strategy_style,
                idempotency_key=idempotency_key,
            )

    @mcp.tool
    def sync_symbol_to_stock_pool(
        symbol: str,
        pipeline: str = NEXUS_PIPELINE,
        workflow_id: str = "",
        stock_name: str = "",
        strategy_style: str = "",
        composite_score: float | None = None,
        next_action: str = "",
        pool_status: str = "",
    ) -> dict[str, Any]:
        with _connect() as conn:
            return sync_symbol_to_stock_pool(
                conn,
                symbol=symbol,
                pipeline=pipeline,
                workflow_id=workflow_id,
                stock_name=stock_name,
                strategy_style=strategy_style,
                composite_score=composite_score,
                next_action=next_action,
                pool_status=pool_status,
            )

    @mcp.tool
    def render_symbol_feishu_summary(
        symbol: str,
        pipeline: str = NEXUS_PIPELINE,
    ) -> dict[str, Any]:
        with _connect() as conn:
            cards = _list_symbol_cards(conn, symbol=symbol, pipeline=pipeline) or list_cards(
                conn, symbol=symbol
            )
            chat_id = cards[0]["chat_id"] if cards else ""
            thread_id = cards[0]["thread_id"] if cards else ""
            return {
                "symbol": symbol.strip().upper(),
                "chat_id": chat_id,
                "thread_id": thread_id,
                "content": render_symbol_feishu_summary(conn, symbol=symbol, pipeline=pipeline),
            }

    return mcp


def _print_json(payload: Any) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db")
    sub.add_parser("serve")

    create = sub.add_parser("create-card")
    create.add_argument("--title", required=True)
    create.add_argument("--symbol", required=True)
    create.add_argument("--work-type", required=True)
    create.add_argument("--assignee", required=True)
    create.add_argument("--description", default="")
    create.add_argument("--status", default="queued")
    create.add_argument("--priority", type=int, default=0)
    create.add_argument("--chat-id", default="")
    create.add_argument("--thread-id", default="")
    create.add_argument("--parent-id", action="append", default=[])

    artifact = sub.add_parser("add-artifact")
    artifact.add_argument("--card-id", required=True)
    artifact.add_argument("--kind", required=True)
    artifact.add_argument("--content", required=True)
    artifact.add_argument("--file-path", default="")

    render = sub.add_parser("render-feishu-update")
    render.add_argument("--card-id", required=True)
    render.add_argument("--artifact-limit", type=int, default=3)

    list_cmd = sub.add_parser("list-cards")
    list_cmd.add_argument("--status", default="")
    list_cmd.add_argument("--symbol", default="")
    list_cmd.add_argument("--work-type", default="")
    list_cmd.add_argument("--assignee", default="")

    transition = sub.add_parser("transition-card")
    transition.add_argument("--card-id", required=True)
    transition.add_argument("--status", required=True)
    transition.add_argument("--note", default="")

    sync_pool = sub.add_parser("sync-stock-pool")
    sync_pool.add_argument("--card-id", required=True)
    sync_pool.add_argument("--stock-name", default="")
    sync_pool.add_argument("--market", default="")
    sync_pool.add_argument("--strategy-style", default="")
    sync_pool.add_argument("--composite-score", type=float)
    sync_pool.add_argument("--catalyst", default="")
    sync_pool.add_argument("--valuation-summary", default="")
    sync_pool.add_argument("--risk-summary", default="")
    sync_pool.add_argument("--next-action", default="")
    sync_pool.add_argument("--pool-date", default="")
    sync_pool.add_argument("--last-review-date", default="")
    sync_pool.add_argument("--pool-status", default="")

    pool_search = sub.add_parser("search-stock-pool")
    pool_search.add_argument("--keyword", required=True)
    pool_search.add_argument("--search-field", action="append", default=[])
    pool_search.add_argument("--select-field", action="append", default=[])
    pool_search.add_argument("--limit", type=int, default=20)

    pool_upsert = sub.add_parser("upsert-stock-pool")
    pool_upsert.add_argument("--symbol", required=True)
    pool_upsert.add_argument("--stock-name", default="")
    pool_upsert.add_argument("--market", default="")
    pool_upsert.add_argument("--pool-status", default="")
    pool_upsert.add_argument("--strategy-style", default="")
    pool_upsert.add_argument("--composite-score", type=float)
    pool_upsert.add_argument("--catalyst", default="")
    pool_upsert.add_argument("--valuation-summary", default="")
    pool_upsert.add_argument("--risk-summary", default="")
    pool_upsert.add_argument("--next-action", default="")
    pool_upsert.add_argument("--workflow-id", default="")
    pool_upsert.add_argument("--pool-date", default="")
    pool_upsert.add_argument("--last-review-date", default="")
    pool_upsert.add_argument("--record-id", default="")

    pool_delete = sub.add_parser("delete-stock-pool")
    pool_delete.add_argument("--symbol", default="")
    pool_delete.add_argument("--record-id", default="")

    nexus = sub.add_parser("create-nexus")
    nexus.add_argument("--symbol", required=True)
    nexus.add_argument("--chat-id", default="")
    nexus.add_argument("--thread-id", default="")
    nexus.add_argument("--assignee", default="feishu-analyst")
    nexus.add_argument("--stock-name", default="")
    nexus.add_argument("--strategy-style", default="balanced")
    nexus.add_argument("--idempotency-key", default="")

    sync_symbol = sub.add_parser("sync-symbol-pool")
    sync_symbol.add_argument("--symbol", required=True)
    sync_symbol.add_argument("--pipeline", default=NEXUS_PIPELINE)
    sync_symbol.add_argument("--stock-name", default="")
    sync_symbol.add_argument("--strategy-style", default="")
    sync_symbol.add_argument("--composite-score", type=float)
    sync_symbol.add_argument("--next-action", default="")
    sync_symbol.add_argument("--pool-status", default="")

    render_symbol = sub.add_parser("render-symbol-summary")
    render_symbol.add_argument("--symbol", required=True)
    render_symbol.add_argument("--pipeline", default=NEXUS_PIPELINE)

    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            if mcp is None:
                raise RuntimeError("FastMCP is not installed. Run: pip install fastmcp")
            mcp.run(transport="stdio", show_banner=False)
            return 0
        with _connect() as conn:
            if args.command == "init-db":
                return _print_json({"ok": True, "db_path": str(_default_db_path())})
            if args.command == "create-card":
                return _print_json(
                    create_card(
                        conn,
                        title=args.title,
                        symbol=args.symbol,
                        work_type=args.work_type,
                        assignee=args.assignee,
                        description=args.description,
                        status=args.status,
                        priority=args.priority,
                        chat_id=args.chat_id,
                        thread_id=args.thread_id,
                        parent_ids=args.parent_id,
                    )
                )
            if args.command == "add-artifact":
                return _print_json(
                    add_artifact(
                        conn,
                        card_id=args.card_id,
                        kind=args.kind,
                        content=args.content,
                        file_path=args.file_path,
                    )
                )
            if args.command == "render-feishu-update":
                print(render_feishu_update(conn, card_id=args.card_id, artifact_limit=args.artifact_limit))
                return 0
            if args.command == "list-cards":
                return _print_json(
                    list_cards(conn, status=args.status, symbol=args.symbol, work_type=args.work_type, assignee=args.assignee)
                )
            if args.command == "transition-card":
                return _print_json(
                    transition_card(conn, card_id=args.card_id, status=args.status, note=args.note)
                )
            if args.command == "sync-stock-pool":
                return _print_json(
                    sync_card_to_stock_pool(
                        conn,
                        card_id=args.card_id,
                        stock_name=args.stock_name,
                        market=args.market,
                        strategy_style=args.strategy_style,
                        composite_score=args.composite_score,
                        catalyst=args.catalyst,
                        valuation_summary=args.valuation_summary,
                        risk_summary=args.risk_summary,
                        next_action=args.next_action,
                        pool_date=args.pool_date,
                        last_review_date=args.last_review_date,
                        pool_status=args.pool_status,
                    )
                )
            if args.command == "search-stock-pool":
                return _print_json(
                    _stock_pool_query(
                        keyword=args.keyword,
                        search_fields=args.search_field or ["股票代码", "股票名称"],
                        select_fields=args.select_field
                        or ["股票代码", "股票名称", "状态", "综合评分", "下一步动作", "workflow_id"],
                        limit=args.limit,
                    )
                )
            if args.command == "upsert-stock-pool":
                return _print_json(
                    _upsert_stock_pool_row(
                        symbol=args.symbol,
                        stock_name=args.stock_name,
                        market=args.market,
                        pool_status=args.pool_status,
                        strategy_style=args.strategy_style,
                        composite_score=args.composite_score,
                        catalyst=args.catalyst,
                        valuation_summary=args.valuation_summary,
                        risk_summary=args.risk_summary,
                        next_action=args.next_action,
                        workflow_id=args.workflow_id,
                        pool_date=args.pool_date,
                        last_review_date=args.last_review_date,
                        record_id=args.record_id,
                    )
                )
            if args.command == "delete-stock-pool":
                return _print_json(_delete_stock_pool_row(symbol=args.symbol, record_id=args.record_id))
            if args.command == "create-nexus":
                return _print_json(
                    create_nexus_analysis(
                        conn,
                        symbol=args.symbol,
                        chat_id=args.chat_id,
                        thread_id=args.thread_id,
                        assignee=args.assignee,
                        stock_name=args.stock_name,
                        strategy_style=args.strategy_style,
                        idempotency_key=args.idempotency_key,
                    )
                )
            if args.command == "sync-symbol-pool":
                return _print_json(
                    sync_symbol_to_stock_pool(
                        conn,
                        symbol=args.symbol,
                        pipeline=args.pipeline,
                        stock_name=args.stock_name,
                        strategy_style=args.strategy_style,
                        composite_score=args.composite_score,
                        next_action=args.next_action,
                        pool_status=args.pool_status,
                    )
                )
            if args.command == "render-symbol-summary":
                print(render_symbol_feishu_summary(conn, symbol=args.symbol, pipeline=args.pipeline))
                return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


mcp = _create_mcp() if FastMCP is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
