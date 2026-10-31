#!/usr/bin/env python3
"""
Ozon API MCP connector for Hermes -- one tool per documented endpoint.

Native connector layer: wraps the Ozon Seller API (https://api-seller.ozon.ru)
and Performance API (https://api-performance.ozon.ru) as MCP tools. Business
logic ("which API/report for X") stays in the ozon-* Hermes skills (unchanged
by this connector) -- this server only does authenticated HTTP I/O.

222 documented endpoints, 222 tools, one function each -- matching the
industry-standard pattern for large API surfaces (e.g. Cloudflare's official
MCP catalog entry ships ~3,300 one-tool-per-OpenAPI-operation): named,
individually typed/discoverable tools beat one wide generic "call" tool,
because the model's tool-call validation happens per-function instead of on
a free-form body dict it has to assemble correctly by hand.

Of these 222:
- 31 have a REAL implementation with typed, named parameters, built from a
  CONFIRMED request schema (ozon-seller-skill's references/*.md + proven
  scripts/*.py, or ozon-ads-performance's working scripts).
- 191 are explicit NOT-IMPLEMENTED stubs: the path/method/description are
  real (from ozon-api-overview/references/api.md), but no confirmed request
  body/params schema exists yet. Calling one returns a `not_implemented`
  error immediately -- no HTTP request is ever sent -- so the agent sees it
  exists, learns it can't be used yet from the result, and stops retrying it
  with guessed fields instead of silently failing against the real API or
  (worse) firing a malformed write against a destructive endpoint.

Upgrading a stub to real: confirm the exact request schema against Ozon's
official docs or a tested script, then replace that one function body --
never guess a field name into an existing stub.

Auth:
- Seller API: Client-Id + Api-Key headers (OZON_API_KEY / OZON_CLIENT_ID).
- Performance API: OAuth2 client_credentials (OZON_PERFORMANCE_CLIENT_ID /
  OZON_PERFORMANCE_CLIENT_SECRET), token cached in-process (~30 min TTL).

WRITE tools (product import, review status, campaign budget) are marked
WRITE in their docstring -- show the current/proposed value and wait for
explicit confirmation before calling one, per the shop's standing rule.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import httpx
from mcp.server.mcpserver import MCPServer

SELLER_BASE_URL = "https://api-seller.ozon.ru"
PERFORMANCE_BASE_URL = "https://api-performance.ozon.ru"


def _env(name: str) -> Optional[str]:
    return os.environ.get(name)


def _seller(method: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    api_key, client_id = _env("OZON_API_KEY"), _env("OZON_CLIENT_ID")
    if not api_key or not client_id:
        return {"error": "OZON_API_KEY / OZON_CLIENT_ID not set (Seller personal account -> Settings -> API keys)"}
    headers = {"Client-Id": client_id, "Api-Key": api_key, "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=60) as client:
            r = client.request(method, f"{SELLER_BASE_URL}{path}", headers=headers, json=body, params=params)
    except httpx.HTTPError as exc:
        return {"error": f"network error: {exc}"}
    if not r.is_success:
        return {"error": f"{r.status_code}: {r.text}"}
    try:
        return r.json()
    except ValueError:
        return {"raw": r.text}


_perf_token: dict[str, Any] = {"value": None, "exp": 0}


def _performance_token() -> Optional[str]:
    if _perf_token["value"] and time.time() < _perf_token["exp"] - 60:
        return _perf_token["value"]
    cid, secret = _env("OZON_PERFORMANCE_CLIENT_ID"), _env("OZON_PERFORMANCE_CLIENT_SECRET")
    if not cid or not secret:
        return None
    r = httpx.post(
        f"{PERFORMANCE_BASE_URL}/api/client/token",
        json={"client_id": cid, "client_secret": secret, "grant_type": "client_credentials"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    _perf_token["value"] = data["access_token"]
    _perf_token["exp"] = time.time() + int(data.get("expires_in", 1800))
    return _perf_token["value"]


def _performance(method: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    token = _performance_token()
    if not token:
        return {"error": "OZON_PERFORMANCE_CLIENT_ID / OZON_PERFORMANCE_CLIENT_SECRET not set (Реклама -> API)"}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=60) as client:
            r = client.request(method, f"{PERFORMANCE_BASE_URL}{path}", headers=headers, json=body, params=params)
    except httpx.HTTPError as exc:
        return {"error": f"network error: {exc}"}
    if not r.is_success:
        return {"error": f"{r.status_code}: {r.text}"}
    ctype = r.headers.get("content-type", "")
    if "application/json" in ctype:
        return r.json()
    return {"raw": r.text}  # CSV endpoints (statistics/daily, statistics/expense)


server = MCPServer(name="ozon", version="0.1.0")


@server.tool()
def ozon_v4_posting_fbs_unfulfilled_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v4/posting/fbs/unfulfilled/list -- Получить список необработанных отправлений

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v4/posting/fbs/unfulfilled/list",
        "description": "Получить список необработанных отправлений",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v3_posting_fbs_unfulfilled_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v3/posting/fbs/unfulfilled/list -- Список необработанных отправлений Deprecated

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v3/posting/fbs/unfulfilled/list",
        "description": "Список необработанных отправлений Deprecated",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v4_posting_fbs_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v4/posting/fbs/list -- Получить список отправлений

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v4/posting/fbs/list",
        "description": "Получить список отправлений",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v3_posting_fbs_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v3/posting/fbs/list -- Список отправлений Deprecated

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v3/posting/fbs/list",
        "description": "Список отправлений Deprecated",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v3_posting_fbs_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v3/posting/fbs/get -- Получить информацию об отправлении по идентификатору

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v3/posting/fbs/get",
        "description": "Получить информацию об отправлении по идентификатору",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_get_by_barcode(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/get-by-barcode -- Получить информацию об отправлении по штрихкоду

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/get-by-barcode",
        "description": "Получить информацию об отправлении по штрихкоду",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_product_country_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/product/country/list -- Список доступных стран-изготовителей

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/product/country/list",
        "description": "Список доступных стран-изготовителей",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_product_country_set(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/product/country/set -- Добавить информацию о стране-изготовителе товара

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/product/country/set",
        "description": "Добавить информацию о стране-изготовителе товара",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbs_restrictions(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbs/restrictions -- Получить ограничения пункта приёма

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbs/restrictions",
        "description": "Получить ограничения пункта приёма",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_package_label(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/package-label -- Напечатать этикетку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/package-label",
        "description": "Напечатать этикетку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_package_label_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/package-label/create -- Создать задание на формирование этикеток

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/package-label/create",
        "description": "Создать задание на формирование этикеток",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbs_package_label_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbs/package-label/create -- Создать задание на выгрузку этикеток

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbs/package-label/create",
        "description": "Создать задание на выгрузку этикеток",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbs_package_label_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbs/package-label/get -- Получить файл с этикетками

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbs/package-label/get",
        "description": "Получить файл с этикетками",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbs_cancel_reason(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbs/cancel-reason -- Причины отмены отправления

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbs/cancel-reason",
        "description": "Причины отмены отправления",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_cancel_reason_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/cancel-reason/list -- Причины отмены отправлений

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/cancel-reason/list",
        "description": "Причины отмены отправлений",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_product_cancel(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/product/cancel -- Отменить отправку некоторых товаров в отправлении

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/product/cancel",
        "description": "Отменить отправку некоторых товаров в отправлении",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_cancel(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/cancel -- Отменить отправление

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/cancel",
        "description": "Отменить отправление",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_arbitration(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/arbitration -- Открыть спор по отправлению

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/arbitration",
        "description": "Открыть спор по отправлению",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_awaiting_delivery(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/awaiting-delivery -- Передать отправление к отгрузке

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/awaiting-delivery",
        "description": "Передать отправление к отгрузке",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbs_pick_up_code_verify(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbs/pick-up-code/verify -- Проверить код курьера

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbs/pick-up-code/verify",
        "description": "Проверить код курьера",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v4_posting_fbs_ship(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v4/posting/fbs/ship -- Собрать заказ (версия 4)

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v4/posting/fbs/ship",
        "description": "Собрать заказ (версия 4)",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v4_posting_fbs_ship_package(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v4/posting/fbs/ship/package -- Частичная сборка отправления (версия 4)

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v4/posting/fbs/ship/package",
        "description": "Частичная сборка отправления (версия 4)",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v6_fbs_posting_product_exemplar_set(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v6/fbs/posting/product/exemplar/set -- Проверить и сохранить данные экземпляров

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v6/fbs/posting/product/exemplar/set",
        "description": "Проверить и сохранить данные экземпляров",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v6_fbs_posting_product_exemplar_create_or_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v6/fbs/posting/product/exemplar/create-or-get -- Получить данные созданных экземпляров

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v6/fbs/posting/product/exemplar/create-or-get",
        "description": "Получить данные созданных экземпляров",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v5_fbs_posting_product_exemplar_status(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v5/fbs/posting/product/exemplar/status -- Получить статус добавления экземпляров

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v5/fbs/posting/product/exemplar/status",
        "description": "Получить статус добавления экземпляров",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v5_fbs_posting_product_exemplar_validate(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v5/fbs/posting/product/exemplar/validate -- Валидация кодов маркировки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v5/fbs/posting/product/exemplar/validate",
        "description": "Валидация кодов маркировки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbs_posting_product_exemplar_update(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbs/posting/product/exemplar/update -- Обновить данные экземпляров

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbs/posting/product/exemplar/update",
        "description": "Обновить данные экземпляров",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_act_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/act/create -- Подтвердить отгрузку и создать документы

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/act/create",
        "description": "Подтвердить отгрузку и создать документы",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbs_split(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbs/split -- Разделить заказ на отправления без сборки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbs/split",
        "description": "Разделить заказ на отправления без сборки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_act_get_postings(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/act/get-postings -- Список отправлений в акте

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/act/get-postings",
        "description": "Список отправлений в акте",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_act_get_container_labels(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/act/get-container-labels -- Этикетки для грузового места

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/act/get-container-labels",
        "description": "Этикетки для грузового места",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_act_get_barcode(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/act/get-barcode -- Штрихкод для отгрузки отправления

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/act/get-barcode",
        "description": "Штрихкод для отгрузки отправления",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_act_get_barcode_text(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/act/get-barcode/text -- Значение штрихкода для отгрузки отправления

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/act/get-barcode/text",
        "description": "Значение штрихкода для отгрузки отправления",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_digital_act_check_status(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/digital/act/check-status -- Статус формирования накладной Deprecated

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/digital/act/check-status",
        "description": "Статус формирования накладной Deprecated",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_act_get_pdf(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/act/get-pdf -- Получить PDF c документами

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/act/get-pdf",
        "description": "Получить PDF c документами",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_act_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/act/list -- Список актов по отгрузкам

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/act/list",
        "description": "Список актов по отгрузкам",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_digital_act_get_pdf(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/digital/act/get-pdf -- Получить лист отгрузки по перевозке Deprecated

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/digital/act/get-pdf",
        "description": "Получить лист отгрузки по перевозке Deprecated",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbs_act_check_status(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbs/act/check-status -- Статус отгрузки и документов

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbs/act/check-status",
        "description": "Статус отгрузки и документов",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbs_traceable_split(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbs/traceable/split -- Разделить отправление с прослеживаемыми товарами

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbs/traceable/split",
        "description": "Разделить отправление с прослеживаемыми товарами",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbs_product_traceable_attribute(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbs/product/traceable/attribute -- Получить список незаполненных атрибутов для прослеживаемых товаров

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbs/product/traceable/attribute",
        "description": "Получить список незаполненных атрибутов для прослеживаемых товаров",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_assembly_carriage_posting_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/assembly/carriage/posting/list -- Получить список отправлений в отгрузке

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/assembly/carriage/posting/list",
        "description": "Получить список отправлений в отгрузке",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_assembly_carriage_product_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/assembly/carriage/product/list -- Получить список товаров в отгрузке

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/assembly/carriage/product/list",
        "description": "Получить список товаров в отгрузке",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_assembly_fbs_posting_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/assembly/fbs/posting/list -- Получить список отправлений

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/assembly/fbs/posting/list",
        "description": "Получить список отправлений",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_assembly_fbs_product_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/assembly/fbs/product/list -- Получить список товаров в отправлениях

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/assembly/fbs/product/list",
        "description": "Получить список товаров в отправлениях",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_fbs_posting_tracking_number_set(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/fbs/posting/tracking-number/set -- Добавить трек-номера

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/fbs/posting/tracking-number/set",
        "description": "Добавить трек-номера",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_fbs_posting_delivering(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/fbs/posting/delivering -- Изменить статус на «Доставляется»

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/fbs/posting/delivering",
        "description": "Изменить статус на «Доставляется»",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_fbs_posting_last_mile(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/fbs/posting/last-mile -- Изменить статус на «Последняя миля»

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/fbs/posting/last-mile",
        "description": "Изменить статус на «Последняя миля»",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_fbs_posting_delivered(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/fbs/posting/delivered -- Изменить статус на «Доставлено»

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/fbs/posting/delivered",
        "description": "Изменить статус на «Доставлено»",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbs_timeslot_change_restrictions(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbs/timeslot/change-restrictions -- Доступные даты для переноса доставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbs/timeslot/change-restrictions",
        "description": "Доступные даты для переноса доставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbs_timeslot_set(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbs/timeslot/set -- Перенести дату доставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbs/timeslot/set",
        "description": "Перенести дату доставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_warehouse_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/warehouse/list -- Получить список партнёрских складов

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/warehouse/list",
        "description": "Получить список партнёрских складов",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/get -- Получить информацию о черновике поставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/get",
        "description": "Получить информацию о черновике поставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/list -- Список черновиков поставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/list",
        "description": "Список черновиков поставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_direct_seller_dlv_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/direct/seller-dlv/create -- Создать черновик с доставкой силами продавца

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/direct/seller-dlv/create",
        "description": "Создать черновик с доставкой силами продавца",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_direct_seller_dlv_edit(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/direct/seller-dlv/edit -- Обновить информацию о доставке силами продавца в черновике

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/direct/seller-dlv/edit",
        "description": "Обновить информацию о доставке силами продавца в черновике",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_direct_timeslot_edit(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/direct/timeslot/edit -- Отредактировать таймслот в черновике

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/direct/timeslot/edit",
        "description": "Отредактировать таймслот в черновике",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_direct_timeslot_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/direct/timeslot/get -- Получить список таймслотов для прямой поставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/direct/timeslot/get",
        "description": "Получить список таймслотов для прямой поставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_direct_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/direct/create -- Создать черновик заявки на поставку без указания способа доставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/direct/create",
        "description": "Создать черновик заявки на поставку без указания способа доставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_direct_delete(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/direct/delete -- Удалить черновик заявки на поставку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/direct/delete",
        "description": "Удалить черновик заявки на поставку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_direct_product_validate(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/direct/product/validate -- Проверить список товаров для склада партнёра

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/direct/product/validate",
        "description": "Проверить список товаров для склада партнёра",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_direct_registrate(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/direct/registrate -- Перевести черновик в действующую поставку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/direct/registrate",
        "description": "Перевести черновик в действующую поставку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_direct_tpl_dlv_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/direct/tpl-dlv/create -- Создать черновик заявки на доставку сторонней транспортной компанией

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/direct/tpl-dlv/create",
        "description": "Создать черновик заявки на доставку сторонней транспортной компанией",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_direct_tpl_dlv_edit(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/direct/tpl-dlv/edit -- Редактировать черновик поставки со способом доставки сторонней транспортной компанией

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/direct/tpl-dlv/edit",
        "description": "Редактировать черновик поставки со способом доставки сторонней транспортной компанией",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_drop_off_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/drop-off/create -- Создать черновик для доставки в drop-off пункт

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/drop-off/create",
        "description": "Создать черновик для доставки в drop-off пункт",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_drop_off_delete(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/drop-off/delete -- Удалить черновик для доставки в drop-off пункт

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/drop-off/delete",
        "description": "Удалить черновик для доставки в drop-off пункт",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_drop_off_dlv_edit(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/drop-off/dlv/edit -- Отредактировать детали доставки для drop-off черновика

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/drop-off/dlv/edit",
        "description": "Отредактировать детали доставки для drop-off черновика",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_drop_off_registrate(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/drop-off/registrate -- Перевести черновик в действующую поставку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/drop-off/registrate",
        "description": "Перевести черновик в действующую поставку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_drop_off_province_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/drop-off/province/list -- Получить список провинций

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/drop-off/province/list",
        "description": "Получить список провинций",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_drop_off_point_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/drop-off/point/list -- Получить список drop-off пунктов в провинции

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/drop-off/point/list",
        "description": "Получить список drop-off пунктов в провинции",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_drop_off_product_validate(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/drop-off/product/validate -- Проверить список товаров, которые склад партнёра может принять

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/drop-off/product/validate",
        "description": "Проверить список товаров, которые склад партнёра может принять",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_pick_up_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/pick-up/create -- Создать черновик заявки на pick-up поставку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/pick-up/create",
        "description": "Создать черновик заявки на pick-up поставку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_pick_up_delete(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/pick-up/delete -- Отменить черновик заявки на pick-up поставку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/pick-up/delete",
        "description": "Отменить черновик заявки на pick-up поставку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_pick_up_dlv_edit(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/pick-up/dlv/edit -- Изменить черновик заявки на pick-up поставку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/pick-up/dlv/edit",
        "description": "Изменить черновик заявки на pick-up поставку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_pick_up_product_validate(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/pick-up/product/validate -- Провалидировать список товаров для pick-up поставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/pick-up/product/validate",
        "description": "Провалидировать список товаров для pick-up поставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_draft_pick_up_registrate(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/draft/pick-up/registrate -- Перевести черновик в действующую поставку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/draft/pick-up/registrate",
        "description": "Перевести черновик в действующую поставку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_order_direct_cancel(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/order/direct/cancel -- Отменить поставку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/order/direct/cancel",
        "description": "Отменить поставку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_order_direct_seller_dlv_edit(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/order/direct/seller-dlv/edit -- Обновить информацию о доставке силами продавца

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/order/direct/seller-dlv/edit",
        "description": "Обновить информацию о доставке силами продавца",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_order_direct_timeslot_edit(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/order/direct/timeslot/edit -- Отредактировать таймслот в заявке на поставку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/order/direct/timeslot/edit",
        "description": "Отредактировать таймслот в заявке на поставку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_order_direct_timeslot_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/order/direct/timeslot/list -- Получить список таймслотов для поставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/order/direct/timeslot/list",
        "description": "Получить список таймслотов для поставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_order_drop_off_cancel(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/order/drop-off/cancel -- Отменить поставку drop-off

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/order/drop-off/cancel",
        "description": "Отменить поставку drop-off",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_order_drop_off_dlv_edit(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/order/drop-off/dlv/edit -- Отредактировать информацию о поставке на drop-off пункт

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/order/drop-off/dlv/edit",
        "description": "Отредактировать информацию о поставке на drop-off пункт",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_order_pick_up_cancel(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/order/pick-up/cancel -- Отменить pick-up поставку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/order/pick-up/cancel",
        "description": "Отменить pick-up поставку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_order_pick_up_dlv_edit(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/order/pick-up/dlv/edit -- Изменить данные о точке забора

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/order/pick-up/dlv/edit",
        "description": "Изменить данные о точке забора",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_act_from_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/act-from/create -- Сгенерировать акт приёмки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/act-from/create",
        "description": "Сгенерировать акт приёмки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_act_from_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/act-from/get -- Получить статус генерации акта приёмки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/act-from/get",
        "description": "Получить статус генерации акта приёмки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_act_to_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/act-to/create -- Сгенерировать транспортную накладную

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/act-to/create",
        "description": "Сгенерировать транспортную накладную",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_act_to_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/act-to/get -- Получить статус генерации транспортной накладной

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/act-to/get",
        "description": "Получить статус генерации транспортной накладной",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_archive_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/archive/get -- Получить информацию о завершённой поставке

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/archive/get",
        "description": "Получить информацию о завершённой поставке",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_archive_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/archive/list -- Получить список завершённых поставок

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/archive/list",
        "description": "Получить список завершённых поставок",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_label_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/label/create -- Cоздать задание на генерацию этикеток

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/label/create",
        "description": "Cоздать задание на генерацию этикеток",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_label_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/label/get -- Получить статус задания на генерацию этикеток

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/label/get",
        "description": "Получить статус задания на генерацию этикеток",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_order_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/order/get -- Получить информацию о конкретной поставке

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/order/get",
        "description": "Получить информацию о конкретной поставке",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_fbp_order_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/fbp/order/list -- Получить список поставок

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/fbp/order/list",
        "description": "Получить список поставок",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbp_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbp/list -- Получить список отправлений

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbp/list",
        "description": "Получить список отправлений",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/create -- Создание отгрузки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/create",
        "description": "Создание отгрузки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_approve(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/approve -- Подтверждение отгрузки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/approve",
        "description": "Подтверждение отгрузки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_set_postings(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/set-postings -- Изменение состава отгрузки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/set-postings",
        "description": "Изменение состава отгрузки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_cancel(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/cancel -- Удаление отгрузки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/cancel",
        "description": "Удаление отгрузки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_carriage_delivery_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/carriage/delivery/list -- Список методов доставки и отгрузок

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/carriage/delivery/list",
        "description": "Список методов доставки и отгрузок",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_delivery_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/delivery/list -- Список методов доставки и отгрузок

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/delivery/list",
        "description": "Список методов доставки и отгрузок",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/get -- Информация о перевозке

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/get",
        "description": "Информация о перевозке",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_act_discrepancy_pdf(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/act-discrepancy/pdf -- Получить акт о расхождениях по отгрузке FBS

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/act-discrepancy/pdf",
        "description": "Получить акт о расхождениях по отгрузке FBS",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_ettn_status(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/ettn/status -- Получить статус проверки электронной ТТН на прослеживаемой перевозке FBS

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/ettn/status",
        "description": "Получить статус проверки электронной ТТН на прослеживаемой перевозке FBS",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_pass_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/pass/create -- Создать пропуск

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/pass/create",
        "description": "Создать пропуск",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_pass_update(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/pass/update -- Обновить пропуск

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/pass/update",
        "description": "Обновить пропуск",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_pass_delete(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/pass/delete -- Удалить пропуск

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/pass/delete",
        "description": "Удалить пропуск",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/create -- Создать грузоместо

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/create",
        "description": "Создать грузоместо",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_fill(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/fill -- Наполнить грузоместо отправлениями

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/fill",
        "description": "Наполнить грузоместо отправлениями",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_approve(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/approve -- Подтвердить состав грузоместа

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/approve",
        "description": "Подтвердить состав грузоместа",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_place_into(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/place-into -- Разместить коробки на палете

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/place-into",
        "description": "Разместить коробки на палете",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_remove_postings(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/remove-postings -- Убрать отправления из грузоместа

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/remove-postings",
        "description": "Убрать отправления из грузоместа",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_remove_from(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/remove-from -- Убрать коробки с палеты

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/remove-from",
        "description": "Убрать коробки с палеты",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_cancel(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/cancel -- Отменить грузоместо

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/cancel",
        "description": "Отменить грузоместо",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/list -- Получить список грузомест

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/list",
        "description": "Получить список грузомест",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/get -- Получить информацию о грузоместах

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/get",
        "description": "Получить информацию о грузоместах",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_status_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/status/get -- Получить статус грузомест FBS

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/status/get",
        "description": "Получить статус грузомест FBS",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_task_info(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/task/info -- Получить статус задачи грузового места

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/task/info",
        "description": "Получить статус задачи грузового места",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_document_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/document/get -- Получить документы по грузоместам — ТрН и лист отгрузки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/document/get",
        "description": "Получить документы по грузоместам — ТрН и лист отгрузки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_carriage_container_label_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/carriage/container/label/get -- Получить этикетку по грузоместам

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/carriage/container/label/get",
        "description": "Получить этикетку по грузоместам",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_return_pass_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/return/pass/create -- Создать пропуск для возврата

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/return/pass/create",
        "description": "Создать пропуск для возврата",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_return_pass_update(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/return/pass/update -- Обновить пропуск для возврата

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/return/pass/update",
        "description": "Обновить пропуск для возврата",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_return_pass_delete(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/return/pass/delete -- Удалить пропуск для возврата

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/return/pass/delete",
        "description": "Удалить пропуск для возврата",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_returns_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/returns/list -- Информация о возвратах FBO и FBS

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/returns/list",
        "description": "Информация о возвратах FBO и FBS",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_returns_settings_utilization_history(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/returns/settings/utilization/history -- Получить историю изменений автоутилизации

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/returns/settings/utilization/history",
        "description": "Получить историю изменений автоутилизации",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_returns_settings_utilization_info(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/returns/settings/utilization/info -- Получить настройки автоутилизации

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/returns/settings/utilization/info",
        "description": "Получить настройки автоутилизации",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_returns_settings_utilization_update(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/returns/settings/utilization/update -- Обновить настройки автоутилизации

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/returns/settings/utilization/update",
        "description": "Обновить настройки автоутилизации",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_returns_rfbs_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/returns/rfbs/list -- Список заявок на возврат

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/returns/rfbs/list",
        "description": "Список заявок на возврат",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_returns_rfbs_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/returns/rfbs/get -- Информация о заявке на возврат

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/returns/rfbs/get",
        "description": "Информация о заявке на возврат",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_returns_rfbs_reject(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/returns/rfbs/reject -- Отклонить заявку на возврат

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/returns/rfbs/reject",
        "description": "Отклонить заявку на возврат",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_returns_rfbs_compensate(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/returns/rfbs/compensate -- Вернуть часть стоимости товара

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/returns/rfbs/compensate",
        "description": "Вернуть часть стоимости товара",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_returns_rfbs_verify(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/returns/rfbs/verify -- Одобрить заявку на возврат

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/returns/rfbs/verify",
        "description": "Одобрить заявку на возврат",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_returns_rfbs_receive_return(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/returns/rfbs/receive-return -- Подтвердить получение товара на проверку

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/returns/rfbs/receive-return",
        "description": "Подтвердить получение товара на проверку",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_returns_rfbs_return_money(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/returns/rfbs/return-money -- Вернуть деньги покупателю

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/returns/rfbs/return-money",
        "description": "Вернуть деньги покупателю",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_returns_rfbs_action_set(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/returns/rfbs/action/set -- Передать доступные действия для rFBS возвратов

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/returns/rfbs/action/set",
        "description": "Передать доступные действия для rFBS возвратов",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_returns_company_fbs_info(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/returns/company/fbs/info -- Количество возвратов FBS

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/returns/company/fbs/info",
        "description": "Количество возвратов FBS",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_return_giveout_is_enabled(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/return/giveout/is-enabled -- Проверить возможность получения возвратных отгрузок по штрихкоду

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/return/giveout/is-enabled",
        "description": "Проверить возможность получения возвратных отгрузок по штрихкоду",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_return_giveout_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/return/giveout/list -- Список возвратных отгрузок

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/return/giveout/list",
        "description": "Список возвратных отгрузок",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_return_giveout_info(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/return/giveout/info -- Информация о возвратной отгрузке

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/return/giveout/info",
        "description": "Информация о возвратной отгрузке",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_return_giveout_barcode(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/return/giveout/barcode -- Значение штрихкода для возвратных отгрузок

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/return/giveout/barcode",
        "description": "Значение штрихкода для возвратных отгрузок",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_return_giveout_get_pdf(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/return/giveout/get-pdf -- Штрихкод для получения возвратной отгрузки в формате PDF

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/return/giveout/get-pdf",
        "description": "Штрихкод для получения возвратной отгрузки в формате PDF",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_return_giveout_get_png(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/return/giveout/get-png -- Штрихкод для получения возвратной отгрузки в формате PNG

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/return/giveout/get-png",
        "description": "Штрихкод для получения возвратной отгрузки в формате PNG",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_return_giveout_barcode_reset(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/return/giveout/barcode-reset -- Сгенерировать новый штрихкод

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/return/giveout/barcode-reset",
        "description": "Сгенерировать новый штрихкод",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_conditional_cancellation_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/conditional-cancellation/list -- Получить список заявок на отмену rFBS

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/conditional-cancellation/list",
        "description": "Получить список заявок на отмену rFBS",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_conditional_cancellation_approve(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/conditional-cancellation/approve -- Подтвердить заявку на отмену rFBS

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/conditional-cancellation/approve",
        "description": "Подтвердить заявку на отмену rFBS",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_conditional_cancellation_reject(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/conditional-cancellation/reject -- Отклонить заявку на отмену rFBS

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/conditional-cancellation/reject",
        "description": "Отклонить заявку на отмену rFBS",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_cancel_reason_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/cancel-reason/list -- Причины отмены отправлений

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/cancel-reason/list",
        "description": "Причины отмены отправлений",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_cancel_reason_list_by_order(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/cancel-reason/list-by-order -- Причины отмены заказа

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/cancel-reason/list-by-order",
        "description": "Причины отмены заказа",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_cancel_reason_list_by_posting(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/cancel-reason/list-by-posting -- Причины отмены отправления

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/cancel-reason/list-by-posting",
        "description": "Причины отмены отправления",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_order_cancel(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/order/cancel -- Отменить заказ

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/order/cancel",
        "description": "Отменить заказ",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_order_cancel_check(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/order/cancel/check -- Проверить возможность отмены заказа

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/order/cancel/check",
        "description": "Проверить возможность отмены заказа",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_order_cancel_status(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/order/cancel/status -- Получить статус отмены заказа

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/order/cancel/status",
        "description": "Получить статус отмены заказа",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_order_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/order/create -- Создать заказ

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/order/create",
        "description": "Создать заказ",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_delivery_method_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/delivery-method/list -- Список методов доставки realFBS-склада

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/delivery-method/list",
        "description": "Список методов доставки realFBS-склада",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_delivery_method_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/delivery-method/list -- Список методов доставки склада

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/delivery-method/list",
        "description": "Список методов доставки склада",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_delivery_method_return_settings_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/delivery-method/return/settings/get -- Получить информацию по возвратным настройкам rFBS и rFBS Express

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/delivery-method/return/settings/get",
        "description": "Получить информацию по возвратным настройкам rFBS и rFBS Express",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_delivery_check(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/delivery/check -- Проверить доступность доставки для покупателя

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/delivery/check",
        "description": "Проверить доступность доставки для покупателя",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_delivery_checkout(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/delivery/checkout -- Получить доступные варианты доставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/delivery/checkout",
        "description": "Получить доступные варианты доставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_delivery_map(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/delivery/map -- Отрисовать точки на карте

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/delivery/map",
        "description": "Отрисовать точки на карте",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_delivery_point_info(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/delivery/point/info -- Получить информацию о точке самовывоза

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/delivery/point/info",
        "description": "Получить информацию о точке самовывоза",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_delivery_point_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/delivery/point/list -- Получить список точек самовывоза

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/delivery/point/list",
        "description": "Получить список точек самовывоза",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v3_posting_multiboxqty_set(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v3/posting/multiboxqty/set -- Указать количество коробок для многокоробочных отправлений

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v3/posting/multiboxqty/set",
        "description": "Указать количество коробок для многокоробочных отправлений",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_global_etgb(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/global/etgb -- Таможенные декларации ETGB

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/global/etgb",
        "description": "Таможенные декларации ETGB",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_unpaid_legal_product_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/unpaid-legal/product/list -- Список неоплаченных товаров, заказанных юридическими лицами

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/unpaid-legal/product/list",
        "description": "Список неоплаченных товаров, заказанных юридическими лицами",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_carriage_available_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/carriage-available/list -- Список доступных перевозок

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/carriage-available/list",
        "description": "Список доступных перевозок",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_cutoff_set(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/cutoff/set -- Уточнить дату отгрузки отправления

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/cutoff/set",
        "description": "Уточнить дату отгрузки отправления",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_cancel(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/cancel -- Отменить отправление из заказа

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/cancel",
        "description": "Отменить отправление из заказа",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_cancel_status(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/cancel/status -- Проверить статус отмены отправления

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/cancel/status",
        "description": "Проверить статус отмены отправления",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_marks(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/marks -- Получить маркировки экземпляров из отправления

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/marks",
        "description": "Получить маркировки экземпляров из отправления",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_polygon_create(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/polygon/create -- Создайте полигон доставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/polygon/create",
        "description": "Создайте полигон доставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_polygon_bind(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/polygon/bind -- Связать метод доставки с полигоном

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/polygon/bind",
        "description": "Связать метод доставки с полигоном",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_polygon_bind(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/polygon/bind -- Свяжите метод доставки с полигоном доставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/polygon/bind",
        "description": "Свяжите метод доставки с полигоном доставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_polygon_delete(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/polygon/delete -- Удалить полигон из области доставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/polygon/delete",
        "description": "Удалить полигон из области доставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_polygon_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/polygon/list -- Получить список установленных полигонов на метод доставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/polygon/list",
        "description": "Получить список установленных полигонов на метод доставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_polygon_time_coordinates_update(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/polygon/time/coordinates/update -- Обновить координаты полигона доставки

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/polygon/time/coordinates/update",
        "description": "Обновить координаты полигона доставки",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_polygon_time_set(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/polygon/time/set -- Установить новое время доставки в полигоне

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/polygon/time/set",
        "description": "Установить новое время доставки в полигоне",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_invoice_create_or_update(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/invoice/create-or-update -- Создать или изменить счёт-фактуру

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/invoice/create-or-update",
        "description": "Создать или изменить счёт-фактуру",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_invoice_file_upload(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/invoice/file/upload -- Загрузка счёта-фактуры

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/invoice/file/upload",
        "description": "Загрузка счёта-фактуры",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_invoice_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/invoice/get -- Получить информацию о счёте-фактуре

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/invoice/get",
        "description": "Получить информацию о счёте-фактуре",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_invoice_delete(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/invoice/delete -- Удалить ссылку на счёт-фактуру

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/invoice/delete",
        "description": "Удалить ссылку на счёт-фактуру",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_receipts_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/receipts/get -- Получить чек в формате PDF

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/receipts/get",
        "description": "Получить чек в формате PDF",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_receipts_seller_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/receipts/seller/list -- Получить список чеков продавца

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/receipts/seller/list",
        "description": "Получить список чеков продавца",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_receipts_upload(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/receipts/upload -- Загрузить чек

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/receipts/upload",
        "description": "Загрузить чек",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v3_posting_fbo_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v3/posting/fbo/list -- Получить список отправлений

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v3/posting/fbo/list",
        "description": "Получить список отправлений",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbo_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbo/list -- Список отправлений Deprecated

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbo/list",
        "description": "Список отправлений Deprecated",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_fbo_get(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/fbo/get -- Информация об отправлении

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/fbo/get",
        "description": "Информация об отправлении",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_fbo_cancel_reason_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/fbo/cancel-reason/list -- Причины отмены отправлений по схеме FBO

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/fbo/cancel-reason/list",
        "description": "Причины отмены отправлений по схеме FBO",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_supplier_available_warehouses(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. GET /v1/supplier/available_warehouses -- Загруженность складов Ozon

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "GET",
        "path": "/v1/supplier/available_warehouses",
        "description": "Загруженность складов Ozon",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_digital_codes_upload(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/digital/codes/upload -- Загрузить коды цифровых товаров для отправления

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/digital/codes/upload",
        "description": "Загрузить коды цифровых товаров для отправления",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v1_posting_digital_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/posting/digital/list -- Получить список отправлений Deprecated

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/posting/digital/list",
        "description": "Получить список отправлений Deprecated",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v2_posting_digital_list(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v2/posting/digital/list -- Получить список отправлений

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v2/posting/digital/list",
        "description": "Получить список отправлений",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v3_product_list(visibility: str = "ALL", offer_id: Optional[list[str]] = None,
    product_id: Optional[list[str]] = None, limit: int = 1000, last_id: str = "") -> dict:
    """POST /v3/product/list -- Список товаров (cursor-пагинация через last_id). Тело: {filter:{offer_id,product_id,visibility}, last_id, limit}.

    Cursor-paginated (last_id). visibility: ALL|VISIBLE|INVISIBLE|EMPTY_STOCK|
    NOT_MODERATED|MODERATED|DISABLED|BANNED|OVERPRICED|IN_SALE|REMOVED_FROM_SALE.
    """
    filt: dict[str, Any] = {"visibility": visibility}
    if offer_id: filt["offer_id"] = offer_id
    if product_id: filt["product_id"] = product_id
    return _seller("POST", "/v3/product/list", {"filter": filt, "last_id": last_id, "limit": limit})


@server.tool()
def ozon_v3_product_info_list(offer_id: Optional[list[str]] = None, sku: Optional[list[str]] = None,
    product_id: Optional[list[str]] = None) -> dict:
    """POST /v3/product/info/list -- Детальная информация по товарам. Один тип id за раз: offer_id|sku|product_id (до 1000). НЕ оборачивать id в filter.

    Pass exactly ONE of offer_id / sku / product_id (up to 1000 each).
    Do NOT wrap the id in a "filter" key -- Ozon rejects that for this endpoint.
    """
    ids = [x for x in (offer_id, sku, product_id) if x]
    if len(ids) != 1:
        return {"error": "pass exactly one of offer_id, sku, product_id"}
    body = {}
    if offer_id: body["offer_id"] = offer_id
    if sku: body["sku"] = sku
    if product_id: body["product_id"] = product_id
    return _seller("POST", "/v3/product/info/list", body)


@server.tool()
def ozon_v1_product_rating_by_sku(skus: list[str]) -> dict:
    """POST /v1/product/rating-by-sku -- Контент-рейтинг карточки (0-100) с разбивкой по группам. Тело: {skus:[...]}."""
    return _seller("POST", "/v1/product/rating-by-sku", {"skus": skus})


@server.tool()
def ozon_v5_product_info_prices(visibility: str = "ALL", offer_id: Optional[list[str]] = None,
    product_id: Optional[list[str]] = None, limit: int = 1000, cursor: str = "") -> dict:
    """POST /v5/product/info/prices -- Цены и прайс-индексы (GREEN/YELLOW/RED). Cursor-пагинация. Тело: {cursor, filter, limit}.

    Cursor-paginated. Returns price, commissions, price_indexes (GREEN/YELLOW/RED), marketing_actions.
    """
    filt: dict[str, Any] = {"visibility": visibility}
    if offer_id: filt["offer_id"] = offer_id
    if product_id: filt["product_id"] = product_id
    return _seller("POST", "/v5/product/info/prices", {"cursor": cursor, "filter": filt, "limit": limit})


@server.tool()
def ozon_v4_product_info_limit() -> dict:
    """POST /v4/product/info/limit -- Лимиты ассортимента (суточные квоты create/update). Тело пустое {}.

    Returns daily_create/daily_update quotas, operation_limits, total assortment limit.
    """
    return _seller("POST", "/v4/product/info/limit", {})


@server.tool()
def ozon_v2_products_stocks(offer_id: list[str]) -> dict:
    """POST /v2/products/stocks -- Остатки товаров по offer_id (массив 1-50). Тело: {offer_id:[...]}.

    offer_id: 1-50 article codes.
    """
    return _seller("POST", "/v2/products/stocks", {"offer_id": offer_id})


@server.tool()
def ozon_v4_product_info_attributes(product_id: Optional[list[str]] = None, offer_id: Optional[list[str]] = None, limit: int = 100) -> dict:
    """POST /v4/product/info/attributes -- Полный список атрибутов карточки (включая description id=4191, хэштеги id=23171). Тело: {filter:{product_id:[...]}, limit}.

    Full raw attribute array for a product (e.g. description id=4191, hashtags
    id=23171, short annotation id=9048, SEO alt-name id=4180, brand id=85 --
    confirmed for M-Style category 17028634; may differ for other categories).
    Pass product_id or offer_id (one or the other).
    """
    filt: dict[str, Any] = {}
    if product_id: filt["product_id"] = product_id
    if offer_id: filt["offer_id"] = offer_id
    return _seller("POST", "/v4/product/info/attributes", {"filter": filt, "limit": limit})


@server.tool()
def ozon_v1_product_info_description(body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """NOT IMPLEMENTED. POST /v1/product/info/description -- Текстовое описание карточки (производное от атрибута 4191).

    This endpoint's request-body/params schema has not been
    verified against real Ozon documentation or a working script yet (only
    the path, method and a one-line description are confirmed from
    ozon-api-overview/references/api.md). Calling this ALWAYS returns
    not_implemented immediately, without making any HTTP request to Ozon --
    do not retry with guessed fields. Needs a human to confirm the real
    request schema (official Ozon API docs or a tested script) before this
    can be implemented for real.
    """
    return {
        "status": "not_implemented",
        "method": "POST",
        "path": "/v1/product/info/description",
        "description": "Текстовое описание карточки (производное от атрибута 4191).",
        "reason": "request schema not verified from documentation",
    }


@server.tool()
def ozon_v3_product_import(offer_id: str, name_: str, description_category_id: int, type_id: int,
    price: str, old_price: str = "", vat: str = "0", currency_code: str = "RUB",
    height: Optional[int] = None, width: Optional[int] = None, depth: Optional[int] = None,
    dimension_unit: str = "mm", weight: Optional[int] = None, weight_unit: str = "g",
    images: Optional[list[str]] = None, primary_image: Optional[str] = None,
    barcode: Optional[str] = None, attributes: Optional[list[dict]] = None) -> dict:
    """WRITE. POST /v3/product/import -- ЗАПИСЬ: создать/обновить карточку товара (описание, хэштеги, цена, атрибуты). type_id обязателен всегда. Перед записью — показать текущее/готовое значение и ждать подтверждения пользователя (см. ozon-api-overview).

    Creates OR updates (by offer_id) a product card -- same endpoint for both.
    type_id is ALWAYS required, even for a text-only update, or Ozon returns
    400 "invalid Request.Items.TypeId".

    SAFE PROCEDURE (do not skip -- a partial payload can wipe fields on a live
    card): first call ozon_v3_product_info_list and ozon_v4_product_info_attributes
    to fetch the current type_id/price/images/attributes, copy the FULL
    attributes array and only replace the values you intend to change (an
    attribute missing from the array entirely -- e.g. no description yet --
    needs a new element appended, not an edit). Show the current and proposed
    values to the user and wait for explicit confirmation before calling this.

    Returns {"result": {"task_id": N}} -- check status with
    ozon_v1_product_import_info({"task_id": N}), wait for "imported" (not
    "pending"), then verify with a fresh read (status alone doesn't prove the
    data actually applied).
    """
    item: dict[str, Any] = {
        "offer_id": offer_id, "name": name_, "description_category_id": description_category_id,
        "type_id": type_id, "currency_code": currency_code, "price": price, "vat": vat,
    }
    if old_price: item["old_price"] = old_price
    if height is not None: item["height"] = height
    if width is not None: item["width"] = width
    if depth is not None: item["depth"] = depth
    if height or width or depth: item["dimension_unit"] = dimension_unit
    if weight is not None:
        item["weight"] = weight
        item["weight_unit"] = weight_unit
    if images: item["images"] = images
    if primary_image: item["primary_image"] = primary_image
    if barcode: item["barcode"] = barcode
    if attributes: item["attributes"] = attributes
    return _seller("POST", "/v3/product/import", {"items": [item]})


@server.tool()
def ozon_v1_product_import_info(task_id: int) -> dict:
    """POST /v1/product/import/info -- Статус задачи импорта карточки. Тело: {task_id}. Ждать status=imported.

    Wait for status == "imported" (not "pending") with errors == [].
    """
    return _seller("POST", "/v1/product/import/info", {"task_id": task_id})


@server.tool()
def ozon_v1_analytics_data(date_from: str, date_to: str, metrics: list[str], dimensions: Optional[list[str]] = None,
    limit: int = 100) -> dict:
    """POST /v1/analytics/data -- Выручка и метрики по дням. dimensions ДОЛЖЕН быть [\"day\"] (не \"date\"). Максимум 3-4 metrics. Rate limit: 429, жди 5+ сек.

    dimensions MUST be ["day"] (NOT "date" -- Ozon validation error). metrics:
    max 3-4 at once or Ozon returns 400. Response revenue is in kopecks (/100
    for rubles). Rate limit: 429 Too Many Requests -- wait 5+ seconds before retrying.
    """
    return _seller("POST", "/v1/analytics/data", {
        "date_from": date_from, "date_to": date_to,
        "dimensions": dimensions or ["day"], "metrics": metrics, "limit": limit,
    })


@server.tool()
def ozon_v2_analytics_stock_on_warehouses(limit: int = 100, offset: int = 0, warehouse_type: str = "ALL") -> dict:
    """POST /v2/analytics/stock_on_warehouses -- Остатки/перемещения по складам. Тело: {limit, offset, warehouse_type: ALL|EXPRESS_DARK_STORE|NOT_EXPRESS_DARK_STORE}.

    warehouse_type: ALL|EXPRESS_DARK_STORE|NOT_EXPRESS_DARK_STORE (no "FBO" value --
    omit/use ALL for FBO warehouses).
    """
    return _seller("POST", "/v2/analytics/stock_on_warehouses", {"limit": limit, "offset": offset, "warehouse_type": warehouse_type})


@server.tool()
def ozon_v1_analytics_turnover_stocks(sku: Optional[list[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> dict:
    """POST /v1/analytics/turnover/stocks -- Оборачиваемость и дни остатка. Rate limit: 1 запрос/мин на Client-Id. Тело: {limit, offset, sku}.

    RATE LIMIT: 1 request per minute per Client-Id -- never loop this over many
    SKUs in a tight loop (429). If sku is given, limit/offset are optional.
    """
    body: dict[str, Any] = {}
    if sku: body["sku"] = sku
    if limit is not None: body["limit"] = limit
    if offset is not None: body["offset"] = offset
    return _seller("POST", "/v1/analytics/turnover/stocks", body)


@server.tool()
def ozon_v1_analytics_stocks(skus: list[str], cluster_ids: Optional[list[str]] = None,
    macrolocal_cluster_ids: Optional[list[str]] = None, warehouse_ids: Optional[list[str]] = None,
    turnover_grades: Optional[list[str]] = None, item_tags: Optional[list[str]] = None) -> dict:
    """POST /v1/analytics/stocks -- Детальная аналитика остатков по кластерам (обновл. 2x/день). sku обязателен (до 100); cluster_ids и macrolocal_cluster_ids взаимоисключающие.

    skus: up to 100. cluster_ids and macrolocal_cluster_ids are MUTUALLY
    EXCLUSIVE -- pass only one. Updated ~07:00 and ~16:00 UTC twice daily.
    """
    if cluster_ids and macrolocal_cluster_ids:
        return {"error": "cluster_ids and macrolocal_cluster_ids are mutually exclusive"}
    body: dict[str, Any] = {"skus": skus}
    if cluster_ids: body["cluster_ids"] = cluster_ids
    if macrolocal_cluster_ids: body["macrolocal_cluster_ids"] = macrolocal_cluster_ids
    if warehouse_ids: body["warehouse_ids"] = warehouse_ids
    if turnover_grades: body["turnover_grades"] = turnover_grades
    if item_tags: body["item_tags"] = item_tags
    return _seller("POST", "/v1/analytics/stocks", body)


@server.tool()
def ozon_v2_finance_realization(month: int, year: int) -> dict:
    """POST /v2/finance/realization -- Отчёт о реализации за месяц (доставленные+возвраты, без отмен). Тело: {month, year}.

    Available by the 5th of the following month. Excludes cancellations/non-pickups.
    """
    return _seller("POST", "/v2/finance/realization", {"month": month, "year": year})


@server.tool()
def ozon_v1_finance_realization_posting(month: int, year: int) -> dict:
    """POST /v1/finance/realization/posting -- Позаказный отчёт о реализации (только до авг. 2023). Тело: {month, year}.

    Per-order detail -- only available through August 2023.
    """
    return _seller("POST", "/v1/finance/realization/posting", {"month": month, "year": year})


@server.tool()
def ozon_v3_finance_transaction_list(date_from: Optional[str] = None, date_to: Optional[str] = None,
    posting_number: Optional[str] = None, transaction_type: str = "all",
    page: int = 1, page_size: int = 1000) -> dict:
    """POST /v3/finance/transaction/list -- Все транзакции (макс. период 1 месяц). Пагинация page/page_size. Тело: {filter:{date|posting_number, transaction_type}, page, page_size}.

    Pass EITHER date_from+date_to (max 1 month span) OR posting_number, not both.
    transaction_type: all|orders|returns|services|compensation|transferDelivery|other.
    """
    if date_from and date_to:
        filt = {
            "date": {"from": f"{date_from}T00:00:00.000Z", "to": f"{date_to}T23:59:59.000Z"},
            "posting_number": "", "operation_type": [], "transaction_type": transaction_type,
        }
    elif posting_number:
        filt = {"posting_number": posting_number, "operation_type": [], "transaction_type": transaction_type}
    else:
        return {"error": "pass either date_from+date_to or posting_number"}
    return _seller("POST", "/v3/finance/transaction/list", {"filter": filt, "page": page, "page_size": page_size})


@server.tool()
def ozon_v3_finance_transaction_totals(date_from: Optional[str] = None, date_to: Optional[str] = None,
    posting_number: Optional[str] = None, transaction_type: str = "all") -> dict:
    """POST /v3/finance/transaction/totals -- Итоговые суммы по транзакциям за период. Тело: {date:{from,to}|posting_number, transaction_type}.

    Invalid posting_number returns all-zero totals rather than an error.
    """
    body: dict[str, Any] = {"transaction_type": transaction_type, "posting_number": posting_number or ""}
    if date_from and date_to:
        body["date"] = {"from": f"{date_from}T00:00:00.000Z", "to": f"{date_to}T23:59:59.000Z"}
    return _seller("POST", "/v3/finance/transaction/totals", body)


@server.tool()
def ozon_v2_review_count() -> dict:
    """POST /v2/review/count -- Количество отзывов по статусам (new/viewed/processed/total). Тело пустое {}. Требует подписку.

    Requires the "Управление отзывами" subscription or Premium Pro -- returns
    403 without it.
    """
    return _seller("POST", "/v2/review/count", {})


@server.tool()
def ozon_v2_review_list(sku: Optional[list[int]] = None, status: Optional[str] = None,
    order_status: Optional[str] = None, date_from: Optional[str] = None,
    date_to: Optional[str] = None, sort_dir: str = "DESC", limit: int = 100,
    last_id: str = "") -> dict:
    """POST /v2/review/list -- Список отзывов, cursor-пагинация last_id, лимит 20-100. Фильтры: sku, status, order_status, даты.

    status: NEW|VIEWED|PROCESSED. order_status: DELIVERED|CANCELLED. limit: 20-100.
    Cursor-paginated via last_id/has_next. Requires review-management subscription.
    """
    filters: dict[str, Any] = {}
    if sku: filters["sku"] = sku
    if status: filters["status"] = status
    if order_status: filters["order_status"] = order_status
    if date_from: filters["published_from"] = f"{date_from}T00:00:00Z"
    if date_to: filters["published_to"] = f"{date_to}T23:59:59Z"
    return _seller("POST", "/v2/review/list", {"filters": filters, "last_id": last_id, "limit": limit, "sort_dir": sort_dir})


@server.tool()
def ozon_v2_review_info(review_id: str) -> dict:
    """POST /v2/review/info -- Детали отзыва с фото/видео. Тело: {review_id}.

    Returns full detail including photos/videos.
    """
    return _seller("POST", "/v2/review/info", {"review_id": review_id})


@server.tool()
def ozon_v2_review_change_status(review_ids: list[str], status: str) -> dict:
    """WRITE. POST /v2/review/change-status -- ЗАПИСЬ: изменить статус отзывов (до 100 за раз). Тело: {review_ids:[...], status: NEW|VIEWED|PROCESSED}.

    status: NEW|VIEWED|PROCESSED. Up to 100 review_ids per call. 200 with no
    body on success. Show the reviews/target status and wait for confirmation
    before calling this.
    """
    return _seller("POST", "/v2/review/change-status", {"review_ids": review_ids, "status": status})


@server.tool()
def ozon_v1_rating_summary() -> dict:
    """POST /v1/rating/summary -- Текущие рейтинги продавца по всем показателям. Тело пустое {}."""
    return _seller("POST", "/v1/rating/summary", {})


@server.tool()
def ozon_v1_rating_history(date_from: str, date_to: str, ratings: Optional[list[str]] = None,
    with_premium_scores: bool = False) -> dict:
    """POST /v1/rating/history -- История рейтингов за период (макс ~30 дней). Тело: {date_from, date_to, ratings:[...], with_premium_scores}.

    Range must be <= ~30 days or Ozon returns 400 "too wide range". ratings
    codes include: rating_on_time, rating_review_avg_score_total, rating_ssl,
    rating_on_time_supply_delivery, rating_order_accuracy,
    rating_on_time_supply_cancellation, rating_reaction_time,
    rating_average_response_time, rating_replied_dialogs_ratio,
    rating_price_green, rating_price_yellow, rating_price_red, rating_price_super.
    Omit ratings to get all of them.
    """
    return _seller("POST", "/v1/rating/history", {
        "date_from": date_from, "date_to": date_to,
        "ratings": ratings or [], "with_premium_scores": with_premium_scores,
    })


@server.tool()
def ozon_v1_seller_info() -> dict:
    """POST /v1/seller/info -- Компания, рейтинги, подписка продавца. Тело пустое {}."""
    return _seller("POST", "/v1/seller/info", {})


@server.tool()
def ozon_v1_seller_ozon_logistics_info() -> dict:
    """POST /v1/seller/ozon-logistics/info -- Статус подключения Ozon Доставки, доступные схемы FBO/FBS. Тело пустое {}."""
    return _seller("POST", "/v1/seller/ozon-logistics/info", {})


@server.tool()
def ozon_api_client_campaign() -> dict:
    """GET /api/client/campaign -- Список рекламных кампаний.

    Note: weeklyBudget often shows a default 2,000,000,000 placeholder for
    parked/zero-real-budget campaigns -- don't treat that field as ground
    truth for actual spend; use statistics/daily for real consumption. state
    (RUNNING/PAUSED) is accurate.
    """
    return _performance("GET", "/api/client/campaign")


@server.tool()
def ozon_api_client_campaign_id_get(campaign_id: str) -> dict:
    """GET /api/client/campaign/{id} -- ВНИМАНИЕ: возвращает 405, не рабочий — используй список выше.

    Documented in this shop's own notes as returning 405 Method Not Allowed in
    practice -- use the GET {campaign} list tool instead for most purposes.
    Kept here because update_campaign_budget's working script calls it as a
    pre-fetch step; if this genuinely 405s for you, that confirms the note.
    """
    return _performance("GET", "/api/client/campaign/" + campaign_id)


@server.tool()
def ozon_api_client_campaign_id_put(campaign_id: str, weekly_budget: int, daily_budget: int = 0,
    title: Optional[str] = None, state: Optional[str] = None) -> dict:
    """WRITE. PUT /api/client/campaign/{id} -- ЗАПИСЬ: изменить еженедельный бюджет кампании.

    weekly_budget/daily_budget in rubles. If title/state are omitted, this
    first GETs the campaign to fill them in from current values (matching the
    proven working script) -- if that GET fails (see the 405 note on the GET
    tool), pass title/state explicitly instead of relying on the pre-fetch.

    Show the current weekly budget and the proposed new one, and wait for
    explicit confirmation before calling this -- changing a live ad budget is
    a standing hard rule for this shop.
    """
    if title is None or state is None:
        current = _performance("GET", "/api/client/campaign/" + campaign_id)
        if isinstance(current, dict) and "error" not in current:
            title = title if title is not None else current.get("title", "")
            state = state if state is not None else current.get("state", "CAMPAIGN_STATE_RUNNING")
        else:
            title = title or ""
            state = state or "CAMPAIGN_STATE_RUNNING"
    body = {
        "id": campaign_id, "weeklyBudget": str(weekly_budget),
        "dailyBudget": str(daily_budget) if daily_budget > 0 else "0",
        "title": title, "state": state,
    }
    return _performance("PUT", "/api/client/campaign/" + campaign_id, body)


@server.tool()
def ozon_api_client_statistics_daily(date_from: str, date_to: str) -> dict:
    """GET /api/client/statistics/daily -- ОСНОВНОЙ: CSV статистика по дням/кампаниям. Параметры dateFrom/dateTo (YYYY-MM-DD).

    THE MAIN endpoint for ad spend. Returns CSV (not JSON): ID;Название;Дата;
    Показы;Клики;Расход, ₽;Заказы, шт.;Заказы, ₽ -- one row per campaign per
    day. Data is final (not revised later). date_from/date_to: YYYY-MM-DD.
    Result comes back as {"raw": "<csv text>"}.
    """
    return _performance("GET", "/api/client/statistics/daily", params={"dateFrom": date_from, "dateTo": date_to})


@server.tool()
def ozon_api_client_statistics_expense(date_from: str, date_to: str) -> dict:
    """GET /api/client/statistics/expense -- CSV, детализация расходов (бонусы/абонентский счёт).

    CSV like statistics/daily but with bonus-account / subscription-account
    expense breakdown. Result comes back as {"raw": "<csv text>"}.
    """
    return _performance("GET", "/api/client/statistics/expense", params={"dateFrom": date_from, "dateTo": date_to})


@server.tool()
def ozon_api_client_statistics(campaign_ids: list[str]) -> dict:
    """POST /api/client/statistics -- JSON статистика по campaignIds (массив). Медленнее, расходует квоты.

    campaignIds must be non-empty (400 if empty). Slower than the GET
    statistics endpoints and consumes extra quota -- prefer statistics/daily
    unless you specifically need this shape.
    """
    return _performance("POST", "/api/client/statistics", {"campaignIds": campaign_ids})


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
