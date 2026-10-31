#!/usr/bin/env python3
"""
Ozon API MCP connector for Hermes -- grouped tools, one per API area.

Native connector layer: wraps the Ozon Seller API (https://api-seller.ozon.ru)
and Performance API (https://api-performance.ozon.ru) as MCP tools. Business
logic ("which API/report for X") stays in the ozon-* Hermes skills (unchanged
by this connector) -- this server only does authenticated HTTP I/O.

10 tools (grouped by action), 222 actions total -- endpoints are grouped by
API area (FBS, FBO, FBP, carriage, returns, delivery, products, finance,
reviews, ads) into one @server.tool() function per area, dispatching on an
`action: str` parameter instead of one dedicated function per endpoint. This
keeps the shop's whole 222-endpoint surface callable while shrinking the
tool list an admin/agent has to scroll through from 222 to 10 -- each tool's
docstring enumerates every `action` it accepts (replacing a JSON-schema enum
description) so an agent never has to leave the tool to learn its options.

Of the 222 actions:
- 31 have a REAL implementation with a CONFIRMED request schema
  (ozon-seller-skill's references/*.md + proven scripts/*.py, or
  ozon-ads-performance's working scripts).
- 191 are explicit NOT-IMPLEMENTED stubs: the path/method/description are
  real (from ozon-api-overview/references/api.md), but no confirmed request
  body/params schema exists yet. Calling one returns a `not_implemented`
  error immediately -- no HTTP request is ever sent -- so the agent sees it
  exists, learns it can't be used yet from the result, and stops retrying it
  with guessed fields instead of silently failing against the real API or
  (worse) firing a malformed write against a destructive endpoint.

Upgrading a stub to real: confirm the exact request schema against Ozon's
official docs or a tested script, then replace that one action's handler --
never guess a field name into an existing stub.

Auth:
- Seller API: Client-Id + Api-Key headers (OZON_API_KEY / OZON_CLIENT_ID).
- Performance API: OAuth2 client_credentials (OZON_PERFORMANCE_CLIENT_ID /
  OZON_PERFORMANCE_CLIENT_SECRET), token cached in-process (~30 min TTL).

WRITE actions (product import, review status, campaign budget) are marked
WRITE in their tool's docstring -- show the current/proposed value and wait
for explicit confirmation before calling one, per the shop's standing rule.
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
def ozon_fbs(action: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Ozon FBS (доставка силами Ozon) — отправления, отгрузка, этикетки, отмены, акты, exemplar'ы, assembly.

    action: один из:
      - 'posting_fbs_unfulfilled_list_v4' (stub) — POST /v4/posting/fbs/unfulfilled/list — Получить список необработанных отправлений
      - 'posting_fbs_unfulfilled_list_v3' (stub) — POST /v3/posting/fbs/unfulfilled/list — Список необработанных отправлений Deprecated
      - 'posting_fbs_list_v4' (stub) — POST /v4/posting/fbs/list — Получить список отправлений
      - 'posting_fbs_list_v3' (stub) — POST /v3/posting/fbs/list — Список отправлений Deprecated
      - 'posting_fbs_get' (stub) — POST /v3/posting/fbs/get — Получить информацию об отправлении по идентификатору
      - 'posting_fbs_get_by_barcode' (stub) — POST /v2/posting/fbs/get-by-barcode — Получить информацию об отправлении по штрихкоду
      - 'posting_fbs_product_country_list' (stub) — POST /v2/posting/fbs/product/country/list — Список доступных стран-изготовителей
      - 'posting_fbs_product_country_set' (stub) — POST /v2/posting/fbs/product/country/set — Добавить информацию о стране-изготовителе товара
      - 'posting_fbs_restrictions' (stub) — POST /v1/posting/fbs/restrictions — Получить ограничения пункта приёма
      - 'posting_fbs_package_label' (stub) — POST /v2/posting/fbs/package-label — Напечатать этикетку
      - 'posting_fbs_package_label_create_v2' (stub) — POST /v2/posting/fbs/package-label/create — Создать задание на формирование этикеток
      - 'posting_fbs_package_label_create_v1' (stub) — POST /v1/posting/fbs/package-label/create — Создать задание на выгрузку этикеток
      - 'posting_fbs_package_label_get' (stub) — POST /v1/posting/fbs/package-label/get — Получить файл с этикетками
      - 'posting_fbs_cancel_reason' (stub) — POST /v1/posting/fbs/cancel-reason — Причины отмены отправления
      - 'posting_fbs_cancel_reason_list' (stub) — POST /v2/posting/fbs/cancel-reason/list — Причины отмены отправлений
      - 'posting_fbs_product_cancel' (stub) — POST /v2/posting/fbs/product/cancel — Отменить отправку некоторых товаров в отправлении
      - 'posting_fbs_cancel' (stub) — POST /v2/posting/fbs/cancel — Отменить отправление
      - 'posting_fbs_arbitration' (stub) — POST /v2/posting/fbs/arbitration — Открыть спор по отправлению
      - 'posting_fbs_awaiting_delivery' (stub) — POST /v2/posting/fbs/awaiting-delivery — Передать отправление к отгрузке
      - 'posting_fbs_pick_up_code_verify' (stub) — POST /v1/posting/fbs/pick-up-code/verify — Проверить код курьера
      - 'posting_fbs_ship' (stub) — POST /v4/posting/fbs/ship — Собрать заказ (версия 4)
      - 'posting_fbs_ship_package' (stub) — POST /v4/posting/fbs/ship/package — Частичная сборка отправления (версия 4)
      - 'fbs_posting_product_exemplar_set' (stub) — POST /v6/fbs/posting/product/exemplar/set — Проверить и сохранить данные экземпляров
      - 'fbs_posting_product_exemplar_create_or_get' (stub) — POST /v6/fbs/posting/product/exemplar/create-or-get — Получить данные созданных экземпляров
      - 'fbs_posting_product_exemplar_status' (stub) — POST /v5/fbs/posting/product/exemplar/status — Получить статус добавления экземпляров
      - 'fbs_posting_product_exemplar_validate' (stub) — POST /v5/fbs/posting/product/exemplar/validate — Валидация кодов маркировки
      - 'fbs_posting_product_exemplar_update' (stub) — POST /v1/fbs/posting/product/exemplar/update — Обновить данные экземпляров
      - 'posting_fbs_act_create' (stub) — POST /v2/posting/fbs/act/create — Подтвердить отгрузку и создать документы
      - 'posting_fbs_split' (stub) — POST /v1/posting/fbs/split — Разделить заказ на отправления без сборки
      - 'posting_fbs_act_get_postings' (stub) — POST /v2/posting/fbs/act/get-postings — Список отправлений в акте
      - 'posting_fbs_act_get_container_labels' (stub) — POST /v2/posting/fbs/act/get-container-labels — Этикетки для грузового места
      - 'posting_fbs_act_get_barcode' (stub) — POST /v2/posting/fbs/act/get-barcode — Штрихкод для отгрузки отправления
      - 'posting_fbs_act_get_barcode_text' (stub) — POST /v2/posting/fbs/act/get-barcode/text — Значение штрихкода для отгрузки отправления
      - 'posting_fbs_digital_act_check_status' (stub) — POST /v2/posting/fbs/digital/act/check-status — Статус формирования накладной Deprecated
      - 'posting_fbs_act_get_pdf' (stub) — POST /v2/posting/fbs/act/get-pdf — Получить PDF c документами
      - 'posting_fbs_act_list' (stub) — POST /v2/posting/fbs/act/list — Список актов по отгрузкам
      - 'posting_fbs_digital_act_get_pdf' (stub) — POST /v2/posting/fbs/digital/act/get-pdf — Получить лист отгрузки по перевозке Deprecated
      - 'posting_fbs_act_check_status' (stub) — POST /v2/posting/fbs/act/check-status — Статус отгрузки и документов
      - 'posting_fbs_traceable_split' (stub) — POST /v1/posting/fbs/traceable/split — Разделить отправление с прослеживаемыми товарами
      - 'posting_fbs_product_traceable_attribute' (stub) — POST /v1/posting/fbs/product/traceable/attribute — Получить список незаполненных атрибутов для прослеживаемых товаров
      - 'assembly_carriage_posting_list' (stub) — POST /v1/assembly/carriage/posting/list — Получить список отправлений в отгрузке
      - 'assembly_carriage_product_list' (stub) — POST /v1/assembly/carriage/product/list — Получить список товаров в отгрузке
      - 'assembly_fbs_posting_list' (stub) — POST /v1/assembly/fbs/posting/list — Получить список отправлений
      - 'assembly_fbs_product_list' (stub) — POST /v1/assembly/fbs/product/list — Получить список товаров в отправлениях
      - 'fbs_posting_tracking_number_set' (stub) — POST /v2/fbs/posting/tracking-number/set — Добавить трек-номера
      - 'fbs_posting_delivering' (stub) — POST /v2/fbs/posting/delivering — Изменить статус на «Доставляется»
      - 'fbs_posting_last_mile' (stub) — POST /v2/fbs/posting/last-mile — Изменить статус на «Последняя миля»
      - 'fbs_posting_delivered' (stub) — POST /v2/fbs/posting/delivered — Изменить статус на «Доставлено»
      - 'posting_fbs_timeslot_change_restrictions' (stub) — POST /v1/posting/fbs/timeslot/change-restrictions — Доступные даты для переноса доставки
      - 'posting_fbs_timeslot_set' (stub) — POST /v1/posting/fbs/timeslot/set — Перенести дату доставки
    """

    handlers = {
        'posting_fbs_unfulfilled_list_v4': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v4/posting/fbs/unfulfilled/list',
            "description": 'Получить список необработанных отправлений',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_unfulfilled_list_v3': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v3/posting/fbs/unfulfilled/list',
            "description": 'Список необработанных отправлений Deprecated',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_list_v4': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v4/posting/fbs/list',
            "description": 'Получить список отправлений',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_list_v3': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v3/posting/fbs/list',
            "description": 'Список отправлений Deprecated',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v3/posting/fbs/get',
            "description": 'Получить информацию об отправлении по идентификатору',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_get_by_barcode': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/get-by-barcode',
            "description": 'Получить информацию об отправлении по штрихкоду',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_product_country_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/product/country/list',
            "description": 'Список доступных стран-изготовителей',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_product_country_set': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/product/country/set',
            "description": 'Добавить информацию о стране-изготовителе товара',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_restrictions': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbs/restrictions',
            "description": 'Получить ограничения пункта приёма',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_package_label': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/package-label',
            "description": 'Напечатать этикетку',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_package_label_create_v2': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/package-label/create',
            "description": 'Создать задание на формирование этикеток',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_package_label_create_v1': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbs/package-label/create',
            "description": 'Создать задание на выгрузку этикеток',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_package_label_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbs/package-label/get',
            "description": 'Получить файл с этикетками',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_cancel_reason': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbs/cancel-reason',
            "description": 'Причины отмены отправления',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_cancel_reason_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/cancel-reason/list',
            "description": 'Причины отмены отправлений',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_product_cancel': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/product/cancel',
            "description": 'Отменить отправку некоторых товаров в отправлении',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_cancel': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/cancel',
            "description": 'Отменить отправление',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_arbitration': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/arbitration',
            "description": 'Открыть спор по отправлению',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_awaiting_delivery': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/awaiting-delivery',
            "description": 'Передать отправление к отгрузке',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_pick_up_code_verify': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbs/pick-up-code/verify',
            "description": 'Проверить код курьера',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_ship': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v4/posting/fbs/ship',
            "description": 'Собрать заказ (версия 4)',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_ship_package': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v4/posting/fbs/ship/package',
            "description": 'Частичная сборка отправления (версия 4)',
            "reason": "request schema not verified from documentation",
        },
        'fbs_posting_product_exemplar_set': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v6/fbs/posting/product/exemplar/set',
            "description": 'Проверить и сохранить данные экземпляров',
            "reason": "request schema not verified from documentation",
        },
        'fbs_posting_product_exemplar_create_or_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v6/fbs/posting/product/exemplar/create-or-get',
            "description": 'Получить данные созданных экземпляров',
            "reason": "request schema not verified from documentation",
        },
        'fbs_posting_product_exemplar_status': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v5/fbs/posting/product/exemplar/status',
            "description": 'Получить статус добавления экземпляров',
            "reason": "request schema not verified from documentation",
        },
        'fbs_posting_product_exemplar_validate': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v5/fbs/posting/product/exemplar/validate',
            "description": 'Валидация кодов маркировки',
            "reason": "request schema not verified from documentation",
        },
        'fbs_posting_product_exemplar_update': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbs/posting/product/exemplar/update',
            "description": 'Обновить данные экземпляров',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_act_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/act/create',
            "description": 'Подтвердить отгрузку и создать документы',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_split': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbs/split',
            "description": 'Разделить заказ на отправления без сборки',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_act_get_postings': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/act/get-postings',
            "description": 'Список отправлений в акте',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_act_get_container_labels': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/act/get-container-labels',
            "description": 'Этикетки для грузового места',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_act_get_barcode': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/act/get-barcode',
            "description": 'Штрихкод для отгрузки отправления',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_act_get_barcode_text': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/act/get-barcode/text',
            "description": 'Значение штрихкода для отгрузки отправления',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_digital_act_check_status': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/digital/act/check-status',
            "description": 'Статус формирования накладной Deprecated',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_act_get_pdf': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/act/get-pdf',
            "description": 'Получить PDF c документами',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_act_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/act/list',
            "description": 'Список актов по отгрузкам',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_digital_act_get_pdf': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/digital/act/get-pdf',
            "description": 'Получить лист отгрузки по перевозке Deprecated',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_act_check_status': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbs/act/check-status',
            "description": 'Статус отгрузки и документов',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_traceable_split': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbs/traceable/split',
            "description": 'Разделить отправление с прослеживаемыми товарами',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_product_traceable_attribute': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbs/product/traceable/attribute',
            "description": 'Получить список незаполненных атрибутов для прослеживаемых товаров',
            "reason": "request schema not verified from documentation",
        },
        'assembly_carriage_posting_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/assembly/carriage/posting/list',
            "description": 'Получить список отправлений в отгрузке',
            "reason": "request schema not verified from documentation",
        },
        'assembly_carriage_product_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/assembly/carriage/product/list',
            "description": 'Получить список товаров в отгрузке',
            "reason": "request schema not verified from documentation",
        },
        'assembly_fbs_posting_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/assembly/fbs/posting/list',
            "description": 'Получить список отправлений',
            "reason": "request schema not verified from documentation",
        },
        'assembly_fbs_product_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/assembly/fbs/product/list',
            "description": 'Получить список товаров в отправлениях',
            "reason": "request schema not verified from documentation",
        },
        'fbs_posting_tracking_number_set': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/fbs/posting/tracking-number/set',
            "description": 'Добавить трек-номера',
            "reason": "request schema not verified from documentation",
        },
        'fbs_posting_delivering': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/fbs/posting/delivering',
            "description": 'Изменить статус на «Доставляется»',
            "reason": "request schema not verified from documentation",
        },
        'fbs_posting_last_mile': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/fbs/posting/last-mile',
            "description": 'Изменить статус на «Последняя миля»',
            "reason": "request schema not verified from documentation",
        },
        'fbs_posting_delivered': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/fbs/posting/delivered',
            "description": 'Изменить статус на «Доставлено»',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_timeslot_change_restrictions': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbs/timeslot/change-restrictions',
            "description": 'Доступные даты для переноса доставки',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbs_timeslot_set': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbs/timeslot/set',
            "description": 'Перенести дату доставки',
            "reason": "request schema not verified from documentation",
        },
    }
    h = handlers.get(action)
    if not h:
        return {"error": f"unknown action '{action}' for ozon_fbs. Valid: {sorted(handlers)}"}
    return h()


@server.tool()
def ozon_fbo(action: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Ozon FBO (доставка со склада Ozon) — поставки, причины отмены, доступные склады поставщика.

    action: один из:
      - 'posting_fbo_list_v3' (stub) — POST /v3/posting/fbo/list — Получить список отправлений
      - 'posting_fbo_list_v2' (stub) — POST /v2/posting/fbo/list — Список отправлений Deprecated
      - 'posting_fbo_get' (stub) — POST /v2/posting/fbo/get — Информация об отправлении
      - 'posting_fbo_cancel_reason_list' (stub) — POST /v1/posting/fbo/cancel-reason/list — Причины отмены отправлений по схеме FBO
      - 'supplier_available_warehouses' (stub) — GET /v1/supplier/available_warehouses — Загруженность складов Ozon
    """

    handlers = {
        'posting_fbo_list_v3': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v3/posting/fbo/list',
            "description": 'Получить список отправлений',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbo_list_v2': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbo/list',
            "description": 'Список отправлений Deprecated',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbo_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/fbo/get',
            "description": 'Информация об отправлении',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbo_cancel_reason_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbo/cancel-reason/list',
            "description": 'Причины отмены отправлений по схеме FBO',
            "reason": "request schema not verified from documentation",
        },
        'supplier_available_warehouses': lambda: {
            "status": "not_implemented",
            "method": 'GET',
            "path": '/v1/supplier/available_warehouses',
            "description": 'Загруженность складов Ozon',
            "reason": "request schema not verified from documentation",
        },
    }
    h = handlers.get(action)
    if not h:
        return {"error": f"unknown action '{action}' for ozon_fbo. Valid: {sorted(handlers)}"}
    return h()


@server.tool()
def ozon_fbp(action: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Ozon FBP (доставка силами продавца через партнёрскую логистику Ozon) — драфты, заказы, акты, этикетки, архив.

    action: один из:
      - 'fbp_warehouse_list' (stub) — POST /v1/fbp/warehouse/list — Получить список партнёрских складов
      - 'fbp_draft_get' (stub) — POST /v1/fbp/draft/get — Получить информацию о черновике поставки
      - 'fbp_draft_list' (stub) — POST /v1/fbp/draft/list — Список черновиков поставки
      - 'fbp_draft_direct_seller_dlv_create' (stub) — POST /v1/fbp/draft/direct/seller-dlv/create — Создать черновик с доставкой силами продавца
      - 'fbp_draft_direct_seller_dlv_edit' (stub) — POST /v1/fbp/draft/direct/seller-dlv/edit — Обновить информацию о доставке силами продавца в черновике
      - 'fbp_draft_direct_timeslot_edit' (stub) — POST /v1/fbp/draft/direct/timeslot/edit — Отредактировать таймслот в черновике
      - 'fbp_draft_direct_timeslot_get' (stub) — POST /v1/fbp/draft/direct/timeslot/get — Получить список таймслотов для прямой поставки
      - 'fbp_draft_direct_create' (stub) — POST /v1/fbp/draft/direct/create — Создать черновик заявки на поставку без указания способа доставки
      - 'fbp_draft_direct_delete' (stub) — POST /v1/fbp/draft/direct/delete — Удалить черновик заявки на поставку
      - 'fbp_draft_direct_product_validate' (stub) — POST /v1/fbp/draft/direct/product/validate — Проверить список товаров для склада партнёра
      - 'fbp_draft_direct_registrate' (stub) — POST /v1/fbp/draft/direct/registrate — Перевести черновик в действующую поставку
      - 'fbp_draft_direct_tpl_dlv_create' (stub) — POST /v1/fbp/draft/direct/tpl-dlv/create — Создать черновик заявки на доставку сторонней транспортной компанией
      - 'fbp_draft_direct_tpl_dlv_edit' (stub) — POST /v1/fbp/draft/direct/tpl-dlv/edit — Редактировать черновик поставки со способом доставки сторонней транспортной компанией
      - 'fbp_draft_drop_off_create' (stub) — POST /v1/fbp/draft/drop-off/create — Создать черновик для доставки в drop-off пункт
      - 'fbp_draft_drop_off_delete' (stub) — POST /v1/fbp/draft/drop-off/delete — Удалить черновик для доставки в drop-off пункт
      - 'fbp_draft_drop_off_dlv_edit' (stub) — POST /v1/fbp/draft/drop-off/dlv/edit — Отредактировать детали доставки для drop-off черновика
      - 'fbp_draft_drop_off_registrate' (stub) — POST /v1/fbp/draft/drop-off/registrate — Перевести черновик в действующую поставку
      - 'fbp_draft_drop_off_province_list' (stub) — POST /v1/fbp/draft/drop-off/province/list — Получить список провинций
      - 'fbp_draft_drop_off_point_list' (stub) — POST /v1/fbp/draft/drop-off/point/list — Получить список drop-off пунктов в провинции
      - 'fbp_draft_drop_off_product_validate' (stub) — POST /v1/fbp/draft/drop-off/product/validate — Проверить список товаров, которые склад партнёра может принять
      - 'fbp_draft_pick_up_create' (stub) — POST /v1/fbp/draft/pick-up/create — Создать черновик заявки на pick-up поставку
      - 'fbp_draft_pick_up_delete' (stub) — POST /v1/fbp/draft/pick-up/delete — Отменить черновик заявки на pick-up поставку
      - 'fbp_draft_pick_up_dlv_edit' (stub) — POST /v1/fbp/draft/pick-up/dlv/edit — Изменить черновик заявки на pick-up поставку
      - 'fbp_draft_pick_up_product_validate' (stub) — POST /v1/fbp/draft/pick-up/product/validate — Провалидировать список товаров для pick-up поставки
      - 'fbp_draft_pick_up_registrate' (stub) — POST /v1/fbp/draft/pick-up/registrate — Перевести черновик в действующую поставку
      - 'fbp_order_direct_cancel' (stub) — POST /v1/fbp/order/direct/cancel — Отменить поставку
      - 'fbp_order_direct_seller_dlv_edit' (stub) — POST /v1/fbp/order/direct/seller-dlv/edit — Обновить информацию о доставке силами продавца
      - 'fbp_order_direct_timeslot_edit' (stub) — POST /v1/fbp/order/direct/timeslot/edit — Отредактировать таймслот в заявке на поставку
      - 'fbp_order_direct_timeslot_list' (stub) — POST /v1/fbp/order/direct/timeslot/list — Получить список таймслотов для поставки
      - 'fbp_order_drop_off_cancel' (stub) — POST /v1/fbp/order/drop-off/cancel — Отменить поставку drop-off
      - 'fbp_order_drop_off_dlv_edit' (stub) — POST /v1/fbp/order/drop-off/dlv/edit — Отредактировать информацию о поставке на drop-off пункт
      - 'fbp_order_pick_up_cancel' (stub) — POST /v1/fbp/order/pick-up/cancel — Отменить pick-up поставку
      - 'fbp_order_pick_up_dlv_edit' (stub) — POST /v1/fbp/order/pick-up/dlv/edit — Изменить данные о точке забора
      - 'fbp_act_from_create' (stub) — POST /v1/fbp/act-from/create — Сгенерировать акт приёмки
      - 'fbp_act_from_get' (stub) — POST /v1/fbp/act-from/get — Получить статус генерации акта приёмки
      - 'fbp_act_to_create' (stub) — POST /v1/fbp/act-to/create — Сгенерировать транспортную накладную
      - 'fbp_act_to_get' (stub) — POST /v1/fbp/act-to/get — Получить статус генерации транспортной накладной
      - 'fbp_archive_get' (stub) — POST /v1/fbp/archive/get — Получить информацию о завершённой поставке
      - 'fbp_archive_list' (stub) — POST /v1/fbp/archive/list — Получить список завершённых поставок
      - 'fbp_label_create' (stub) — POST /v1/fbp/label/create — Cоздать задание на генерацию этикеток
      - 'fbp_label_get' (stub) — POST /v1/fbp/label/get — Получить статус задания на генерацию этикеток
      - 'fbp_order_get' (stub) — POST /v1/fbp/order/get — Получить информацию о конкретной поставке
      - 'fbp_order_list' (stub) — POST /v1/fbp/order/list — Получить список поставок
      - 'posting_fbp_list' (stub) — POST /v1/posting/fbp/list — Получить список отправлений
    """

    handlers = {
        'fbp_warehouse_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/warehouse/list',
            "description": 'Получить список партнёрских складов',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/get',
            "description": 'Получить информацию о черновике поставки',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/list',
            "description": 'Список черновиков поставки',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_direct_seller_dlv_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/direct/seller-dlv/create',
            "description": 'Создать черновик с доставкой силами продавца',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_direct_seller_dlv_edit': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/direct/seller-dlv/edit',
            "description": 'Обновить информацию о доставке силами продавца в черновике',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_direct_timeslot_edit': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/direct/timeslot/edit',
            "description": 'Отредактировать таймслот в черновике',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_direct_timeslot_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/direct/timeslot/get',
            "description": 'Получить список таймслотов для прямой поставки',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_direct_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/direct/create',
            "description": 'Создать черновик заявки на поставку без указания способа доставки',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_direct_delete': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/direct/delete',
            "description": 'Удалить черновик заявки на поставку',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_direct_product_validate': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/direct/product/validate',
            "description": 'Проверить список товаров для склада партнёра',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_direct_registrate': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/direct/registrate',
            "description": 'Перевести черновик в действующую поставку',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_direct_tpl_dlv_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/direct/tpl-dlv/create',
            "description": 'Создать черновик заявки на доставку сторонней транспортной компанией',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_direct_tpl_dlv_edit': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/direct/tpl-dlv/edit',
            "description": 'Редактировать черновик поставки со способом доставки сторонней транспортной компанией',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_drop_off_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/drop-off/create',
            "description": 'Создать черновик для доставки в drop-off пункт',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_drop_off_delete': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/drop-off/delete',
            "description": 'Удалить черновик для доставки в drop-off пункт',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_drop_off_dlv_edit': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/drop-off/dlv/edit',
            "description": 'Отредактировать детали доставки для drop-off черновика',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_drop_off_registrate': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/drop-off/registrate',
            "description": 'Перевести черновик в действующую поставку',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_drop_off_province_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/drop-off/province/list',
            "description": 'Получить список провинций',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_drop_off_point_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/drop-off/point/list',
            "description": 'Получить список drop-off пунктов в провинции',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_drop_off_product_validate': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/drop-off/product/validate',
            "description": 'Проверить список товаров, которые склад партнёра может принять',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_pick_up_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/pick-up/create',
            "description": 'Создать черновик заявки на pick-up поставку',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_pick_up_delete': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/pick-up/delete',
            "description": 'Отменить черновик заявки на pick-up поставку',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_pick_up_dlv_edit': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/pick-up/dlv/edit',
            "description": 'Изменить черновик заявки на pick-up поставку',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_pick_up_product_validate': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/pick-up/product/validate',
            "description": 'Провалидировать список товаров для pick-up поставки',
            "reason": "request schema not verified from documentation",
        },
        'fbp_draft_pick_up_registrate': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/draft/pick-up/registrate',
            "description": 'Перевести черновик в действующую поставку',
            "reason": "request schema not verified from documentation",
        },
        'fbp_order_direct_cancel': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/order/direct/cancel',
            "description": 'Отменить поставку',
            "reason": "request schema not verified from documentation",
        },
        'fbp_order_direct_seller_dlv_edit': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/order/direct/seller-dlv/edit',
            "description": 'Обновить информацию о доставке силами продавца',
            "reason": "request schema not verified from documentation",
        },
        'fbp_order_direct_timeslot_edit': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/order/direct/timeslot/edit',
            "description": 'Отредактировать таймслот в заявке на поставку',
            "reason": "request schema not verified from documentation",
        },
        'fbp_order_direct_timeslot_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/order/direct/timeslot/list',
            "description": 'Получить список таймслотов для поставки',
            "reason": "request schema not verified from documentation",
        },
        'fbp_order_drop_off_cancel': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/order/drop-off/cancel',
            "description": 'Отменить поставку drop-off',
            "reason": "request schema not verified from documentation",
        },
        'fbp_order_drop_off_dlv_edit': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/order/drop-off/dlv/edit',
            "description": 'Отредактировать информацию о поставке на drop-off пункт',
            "reason": "request schema not verified from documentation",
        },
        'fbp_order_pick_up_cancel': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/order/pick-up/cancel',
            "description": 'Отменить pick-up поставку',
            "reason": "request schema not verified from documentation",
        },
        'fbp_order_pick_up_dlv_edit': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/order/pick-up/dlv/edit',
            "description": 'Изменить данные о точке забора',
            "reason": "request schema not verified from documentation",
        },
        'fbp_act_from_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/act-from/create',
            "description": 'Сгенерировать акт приёмки',
            "reason": "request schema not verified from documentation",
        },
        'fbp_act_from_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/act-from/get',
            "description": 'Получить статус генерации акта приёмки',
            "reason": "request schema not verified from documentation",
        },
        'fbp_act_to_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/act-to/create',
            "description": 'Сгенерировать транспортную накладную',
            "reason": "request schema not verified from documentation",
        },
        'fbp_act_to_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/act-to/get',
            "description": 'Получить статус генерации транспортной накладной',
            "reason": "request schema not verified from documentation",
        },
        'fbp_archive_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/archive/get',
            "description": 'Получить информацию о завершённой поставке',
            "reason": "request schema not verified from documentation",
        },
        'fbp_archive_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/archive/list',
            "description": 'Получить список завершённых поставок',
            "reason": "request schema not verified from documentation",
        },
        'fbp_label_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/label/create',
            "description": 'Cоздать задание на генерацию этикеток',
            "reason": "request schema not verified from documentation",
        },
        'fbp_label_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/label/get',
            "description": 'Получить статус задания на генерацию этикеток',
            "reason": "request schema not verified from documentation",
        },
        'fbp_order_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/order/get',
            "description": 'Получить информацию о конкретной поставке',
            "reason": "request schema not verified from documentation",
        },
        'fbp_order_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/fbp/order/list',
            "description": 'Получить список поставок',
            "reason": "request schema not verified from documentation",
        },
        'posting_fbp_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/fbp/list',
            "description": 'Получить список отправлений',
            "reason": "request schema not verified from documentation",
        },
    }
    h = handlers.get(action)
    if not h:
        return {"error": f"unknown action '{action}' for ozon_fbp. Valid: {sorted(handlers)}"}
    return h()


@server.tool()
def ozon_carriage(action: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Ozon перевозки (carriage) — создание/подтверждение перевозок, короба (containers), пропуска, акты расхождений.

    action: один из:
      - 'carriage_create' (stub) — POST /v1/carriage/create — Создание отгрузки
      - 'carriage_approve' (stub) — POST /v1/carriage/approve — Подтверждение отгрузки
      - 'carriage_set_postings' (stub) — POST /v1/carriage/set-postings — Изменение состава отгрузки
      - 'carriage_cancel' (stub) — POST /v1/carriage/cancel — Удаление отгрузки
      - 'carriage_delivery_list_v2' (stub) — POST /v2/carriage/delivery/list — Список методов доставки и отгрузок
      - 'carriage_delivery_list_v1' (stub) — POST /v1/carriage/delivery/list — Список методов доставки и отгрузок
      - 'carriage_get' (stub) — POST /v1/carriage/get — Информация о перевозке
      - 'carriage_act_discrepancy_pdf' (stub) — POST /v1/carriage/act-discrepancy/pdf — Получить акт о расхождениях по отгрузке FBS
      - 'carriage_ettn_status' (stub) — POST /v1/carriage/ettn/status — Получить статус проверки электронной ТТН на прослеживаемой перевозке FBS
      - 'carriage_pass_create' (stub) — POST /v1/carriage/pass/create — Создать пропуск
      - 'carriage_pass_update' (stub) — POST /v1/carriage/pass/update — Обновить пропуск
      - 'carriage_pass_delete' (stub) — POST /v1/carriage/pass/delete — Удалить пропуск
      - 'carriage_container_create' (stub) — POST /v1/carriage/container/create — Создать грузоместо
      - 'carriage_container_fill' (stub) — POST /v1/carriage/container/fill — Наполнить грузоместо отправлениями
      - 'carriage_container_approve' (stub) — POST /v1/carriage/container/approve — Подтвердить состав грузоместа
      - 'carriage_container_place_into' (stub) — POST /v1/carriage/container/place-into — Разместить коробки на палете
      - 'carriage_container_remove_postings' (stub) — POST /v1/carriage/container/remove-postings — Убрать отправления из грузоместа
      - 'carriage_container_remove_from' (stub) — POST /v1/carriage/container/remove-from — Убрать коробки с палеты
      - 'carriage_container_cancel' (stub) — POST /v1/carriage/container/cancel — Отменить грузоместо
      - 'carriage_container_list' (stub) — POST /v1/carriage/container/list — Получить список грузомест
      - 'carriage_container_get' (stub) — POST /v1/carriage/container/get — Получить информацию о грузоместах
      - 'carriage_container_status_get' (stub) — POST /v1/carriage/container/status/get — Получить статус грузомест FBS
      - 'carriage_container_task_info' (stub) — POST /v1/carriage/container/task/info — Получить статус задачи грузового места
      - 'carriage_container_document_get' (stub) — POST /v1/carriage/container/document/get — Получить документы по грузоместам — ТрН и лист отгрузки
      - 'carriage_container_label_get' (stub) — POST /v1/carriage/container/label/get — Получить этикетку по грузоместам
      - 'posting_carriage_available_list' (stub) — POST /v1/posting/carriage-available/list — Список доступных перевозок
    """

    handlers = {
        'carriage_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/create',
            "description": 'Создание отгрузки',
            "reason": "request schema not verified from documentation",
        },
        'carriage_approve': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/approve',
            "description": 'Подтверждение отгрузки',
            "reason": "request schema not verified from documentation",
        },
        'carriage_set_postings': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/set-postings',
            "description": 'Изменение состава отгрузки',
            "reason": "request schema not verified from documentation",
        },
        'carriage_cancel': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/cancel',
            "description": 'Удаление отгрузки',
            "reason": "request schema not verified from documentation",
        },
        'carriage_delivery_list_v2': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/carriage/delivery/list',
            "description": 'Список методов доставки и отгрузок',
            "reason": "request schema not verified from documentation",
        },
        'carriage_delivery_list_v1': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/delivery/list',
            "description": 'Список методов доставки и отгрузок',
            "reason": "request schema not verified from documentation",
        },
        'carriage_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/get',
            "description": 'Информация о перевозке',
            "reason": "request schema not verified from documentation",
        },
        'carriage_act_discrepancy_pdf': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/act-discrepancy/pdf',
            "description": 'Получить акт о расхождениях по отгрузке FBS',
            "reason": "request schema not verified from documentation",
        },
        'carriage_ettn_status': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/ettn/status',
            "description": 'Получить статус проверки электронной ТТН на прослеживаемой перевозке FBS',
            "reason": "request schema not verified from documentation",
        },
        'carriage_pass_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/pass/create',
            "description": 'Создать пропуск',
            "reason": "request schema not verified from documentation",
        },
        'carriage_pass_update': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/pass/update',
            "description": 'Обновить пропуск',
            "reason": "request schema not verified from documentation",
        },
        'carriage_pass_delete': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/pass/delete',
            "description": 'Удалить пропуск',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/create',
            "description": 'Создать грузоместо',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_fill': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/fill',
            "description": 'Наполнить грузоместо отправлениями',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_approve': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/approve',
            "description": 'Подтвердить состав грузоместа',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_place_into': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/place-into',
            "description": 'Разместить коробки на палете',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_remove_postings': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/remove-postings',
            "description": 'Убрать отправления из грузоместа',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_remove_from': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/remove-from',
            "description": 'Убрать коробки с палеты',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_cancel': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/cancel',
            "description": 'Отменить грузоместо',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/list',
            "description": 'Получить список грузомест',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/get',
            "description": 'Получить информацию о грузоместах',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_status_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/status/get',
            "description": 'Получить статус грузомест FBS',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_task_info': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/task/info',
            "description": 'Получить статус задачи грузового места',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_document_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/document/get',
            "description": 'Получить документы по грузоместам — ТрН и лист отгрузки',
            "reason": "request schema not verified from documentation",
        },
        'carriage_container_label_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/carriage/container/label/get',
            "description": 'Получить этикетку по грузоместам',
            "reason": "request schema not verified from documentation",
        },
        'posting_carriage_available_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/carriage-available/list',
            "description": 'Список доступных перевозок',
            "reason": "request schema not verified from documentation",
        },
    }
    h = handlers.get(action)
    if not h:
        return {"error": f"unknown action '{action}' for ozon_carriage. Valid: {sorted(handlers)}"}
    return h()


@server.tool()
def ozon_returns(action: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Ozon возвраты и отмены — returns/rFBS/giveout, условные отмены, причины отмен, отмена заказов/отправлений.

    action: один из:
      - 'return_pass_create' (stub) — POST /v1/return/pass/create — Создать пропуск для возврата
      - 'return_pass_update' (stub) — POST /v1/return/pass/update — Обновить пропуск для возврата
      - 'return_pass_delete' (stub) — POST /v1/return/pass/delete — Удалить пропуск для возврата
      - 'returns_list' (stub) — POST /v1/returns/list — Информация о возвратах FBO и FBS
      - 'returns_settings_utilization_history' (stub) — POST /v1/returns/settings/utilization/history — Получить историю изменений автоутилизации
      - 'returns_settings_utilization_info' (stub) — POST /v1/returns/settings/utilization/info — Получить настройки автоутилизации
      - 'returns_settings_utilization_update' (stub) — POST /v1/returns/settings/utilization/update — Обновить настройки автоутилизации
      - 'returns_rfbs_list' (stub) — POST /v2/returns/rfbs/list — Список заявок на возврат
      - 'returns_rfbs_get' (stub) — POST /v2/returns/rfbs/get — Информация о заявке на возврат
      - 'returns_rfbs_reject' (stub) — POST /v2/returns/rfbs/reject — Отклонить заявку на возврат
      - 'returns_rfbs_compensate' (stub) — POST /v2/returns/rfbs/compensate — Вернуть часть стоимости товара
      - 'returns_rfbs_verify' (stub) — POST /v2/returns/rfbs/verify — Одобрить заявку на возврат
      - 'returns_rfbs_receive_return' (stub) — POST /v2/returns/rfbs/receive-return — Подтвердить получение товара на проверку
      - 'returns_rfbs_return_money' (stub) — POST /v2/returns/rfbs/return-money — Вернуть деньги покупателю
      - 'returns_rfbs_action_set' (stub) — POST /v1/returns/rfbs/action/set — Передать доступные действия для rFBS возвратов
      - 'returns_company_fbs_info' (stub) — POST /v1/returns/company/fbs/info — Количество возвратов FBS
      - 'return_giveout_is_enabled' (stub) — POST /v1/return/giveout/is-enabled — Проверить возможность получения возвратных отгрузок по штрихкоду
      - 'return_giveout_list' (stub) — POST /v1/return/giveout/list — Список возвратных отгрузок
      - 'return_giveout_info' (stub) — POST /v1/return/giveout/info — Информация о возвратной отгрузке
      - 'return_giveout_barcode' (stub) — POST /v1/return/giveout/barcode — Значение штрихкода для возвратных отгрузок
      - 'return_giveout_get_pdf' (stub) — POST /v1/return/giveout/get-pdf — Штрихкод для получения возвратной отгрузки в формате PDF
      - 'return_giveout_get_png' (stub) — POST /v1/return/giveout/get-png — Штрихкод для получения возвратной отгрузки в формате PNG
      - 'return_giveout_barcode_reset' (stub) — POST /v1/return/giveout/barcode-reset — Сгенерировать новый штрихкод
      - 'conditional_cancellation_list' (stub) — POST /v2/conditional-cancellation/list — Получить список заявок на отмену rFBS
      - 'conditional_cancellation_approve' (stub) — POST /v2/conditional-cancellation/approve — Подтвердить заявку на отмену rFBS
      - 'conditional_cancellation_reject' (stub) — POST /v2/conditional-cancellation/reject — Отклонить заявку на отмену rFBS
      - 'cancel_reason_list' (stub) — POST /v1/cancel-reason/list — Причины отмены отправлений
      - 'cancel_reason_list_by_order' (stub) — POST /v1/cancel-reason/list-by-order — Причины отмены заказа
      - 'cancel_reason_list_by_posting' (stub) — POST /v1/cancel-reason/list-by-posting — Причины отмены отправления
      - 'order_cancel' (stub) — POST /v1/order/cancel — Отменить заказ
      - 'order_cancel_check' (stub) — POST /v1/order/cancel/check — Проверить возможность отмены заказа
      - 'order_cancel_status' (stub) — POST /v1/order/cancel/status — Получить статус отмены заказа
      - 'posting_cancel' (stub) — POST /v1/posting/cancel — Отменить отправление из заказа
      - 'posting_cancel_status' (stub) — POST /v1/posting/cancel/status — Проверить статус отмены отправления
    """

    handlers = {
        'return_pass_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/return/pass/create',
            "description": 'Создать пропуск для возврата',
            "reason": "request schema not verified from documentation",
        },
        'return_pass_update': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/return/pass/update',
            "description": 'Обновить пропуск для возврата',
            "reason": "request schema not verified from documentation",
        },
        'return_pass_delete': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/return/pass/delete',
            "description": 'Удалить пропуск для возврата',
            "reason": "request schema not verified from documentation",
        },
        'returns_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/returns/list',
            "description": 'Информация о возвратах FBO и FBS',
            "reason": "request schema not verified from documentation",
        },
        'returns_settings_utilization_history': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/returns/settings/utilization/history',
            "description": 'Получить историю изменений автоутилизации',
            "reason": "request schema not verified from documentation",
        },
        'returns_settings_utilization_info': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/returns/settings/utilization/info',
            "description": 'Получить настройки автоутилизации',
            "reason": "request schema not verified from documentation",
        },
        'returns_settings_utilization_update': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/returns/settings/utilization/update',
            "description": 'Обновить настройки автоутилизации',
            "reason": "request schema not verified from documentation",
        },
        'returns_rfbs_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/returns/rfbs/list',
            "description": 'Список заявок на возврат',
            "reason": "request schema not verified from documentation",
        },
        'returns_rfbs_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/returns/rfbs/get',
            "description": 'Информация о заявке на возврат',
            "reason": "request schema not verified from documentation",
        },
        'returns_rfbs_reject': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/returns/rfbs/reject',
            "description": 'Отклонить заявку на возврат',
            "reason": "request schema not verified from documentation",
        },
        'returns_rfbs_compensate': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/returns/rfbs/compensate',
            "description": 'Вернуть часть стоимости товара',
            "reason": "request schema not verified from documentation",
        },
        'returns_rfbs_verify': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/returns/rfbs/verify',
            "description": 'Одобрить заявку на возврат',
            "reason": "request schema not verified from documentation",
        },
        'returns_rfbs_receive_return': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/returns/rfbs/receive-return',
            "description": 'Подтвердить получение товара на проверку',
            "reason": "request schema not verified from documentation",
        },
        'returns_rfbs_return_money': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/returns/rfbs/return-money',
            "description": 'Вернуть деньги покупателю',
            "reason": "request schema not verified from documentation",
        },
        'returns_rfbs_action_set': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/returns/rfbs/action/set',
            "description": 'Передать доступные действия для rFBS возвратов',
            "reason": "request schema not verified from documentation",
        },
        'returns_company_fbs_info': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/returns/company/fbs/info',
            "description": 'Количество возвратов FBS',
            "reason": "request schema not verified from documentation",
        },
        'return_giveout_is_enabled': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/return/giveout/is-enabled',
            "description": 'Проверить возможность получения возвратных отгрузок по штрихкоду',
            "reason": "request schema not verified from documentation",
        },
        'return_giveout_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/return/giveout/list',
            "description": 'Список возвратных отгрузок',
            "reason": "request schema not verified from documentation",
        },
        'return_giveout_info': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/return/giveout/info',
            "description": 'Информация о возвратной отгрузке',
            "reason": "request schema not verified from documentation",
        },
        'return_giveout_barcode': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/return/giveout/barcode',
            "description": 'Значение штрихкода для возвратных отгрузок',
            "reason": "request schema not verified from documentation",
        },
        'return_giveout_get_pdf': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/return/giveout/get-pdf',
            "description": 'Штрихкод для получения возвратной отгрузки в формате PDF',
            "reason": "request schema not verified from documentation",
        },
        'return_giveout_get_png': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/return/giveout/get-png',
            "description": 'Штрихкод для получения возвратной отгрузки в формате PNG',
            "reason": "request schema not verified from documentation",
        },
        'return_giveout_barcode_reset': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/return/giveout/barcode-reset',
            "description": 'Сгенерировать новый штрихкод',
            "reason": "request schema not verified from documentation",
        },
        'conditional_cancellation_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/conditional-cancellation/list',
            "description": 'Получить список заявок на отмену rFBS',
            "reason": "request schema not verified from documentation",
        },
        'conditional_cancellation_approve': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/conditional-cancellation/approve',
            "description": 'Подтвердить заявку на отмену rFBS',
            "reason": "request schema not verified from documentation",
        },
        'conditional_cancellation_reject': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/conditional-cancellation/reject',
            "description": 'Отклонить заявку на отмену rFBS',
            "reason": "request schema not verified from documentation",
        },
        'cancel_reason_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/cancel-reason/list',
            "description": 'Причины отмены отправлений',
            "reason": "request schema not verified from documentation",
        },
        'cancel_reason_list_by_order': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/cancel-reason/list-by-order',
            "description": 'Причины отмены заказа',
            "reason": "request schema not verified from documentation",
        },
        'cancel_reason_list_by_posting': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/cancel-reason/list-by-posting',
            "description": 'Причины отмены отправления',
            "reason": "request schema not verified from documentation",
        },
        'order_cancel': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/order/cancel',
            "description": 'Отменить заказ',
            "reason": "request schema not verified from documentation",
        },
        'order_cancel_check': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/order/cancel/check',
            "description": 'Проверить возможность отмены заказа',
            "reason": "request schema not verified from documentation",
        },
        'order_cancel_status': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/order/cancel/status',
            "description": 'Получить статус отмены заказа',
            "reason": "request schema not verified from documentation",
        },
        'posting_cancel': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/cancel',
            "description": 'Отменить отправление из заказа',
            "reason": "request schema not verified from documentation",
        },
        'posting_cancel_status': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/cancel/status',
            "description": 'Проверить статус отмены отправления',
            "reason": "request schema not verified from documentation",
        },
    }
    h = handlers.get(action)
    if not h:
        return {"error": f"unknown action '{action}' for ozon_returns. Valid: {sorted(handlers)}"}
    return h()


@server.tool()
def ozon_delivery(action: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Ozon доставка — методы доставки, полигоны, пункты выдачи, счета-фактуры (invoice), чеки (receipts), создание заказа.

    action: один из:
      - 'order_create' (stub) — POST /v2/order/create — Создать заказ
      - 'delivery_method_list_v2' (stub) — POST /v2/delivery-method/list — Список методов доставки realFBS-склада
      - 'delivery_method_list_v1' (stub) — POST /v1/delivery-method/list — Список методов доставки склада
      - 'delivery_method_return_settings_get' (stub) — POST /v1/delivery-method/return/settings/get — Получить информацию по возвратным настройкам rFBS и rFBS Express
      - 'delivery_check' (stub) — POST /v1/delivery/check — Проверить доступность доставки для покупателя
      - 'delivery_checkout' (stub) — POST /v2/delivery/checkout — Получить доступные варианты доставки
      - 'delivery_map' (stub) — POST /v1/delivery/map — Отрисовать точки на карте
      - 'delivery_point_info' (stub) — POST /v1/delivery/point/info — Получить информацию о точке самовывоза
      - 'delivery_point_list' (stub) — POST /v1/delivery/point/list — Получить список точек самовывоза
      - 'posting_multiboxqty_set' (stub) — POST /v3/posting/multiboxqty/set — Указать количество коробок для многокоробочных отправлений
      - 'posting_global_etgb' (stub) — POST /v1/posting/global/etgb — Таможенные декларации ETGB
      - 'posting_unpaid_legal_product_list' (stub) — POST /v1/posting/unpaid-legal/product/list — Список неоплаченных товаров, заказанных юридическими лицами
      - 'posting_cutoff_set' (stub) — POST /v1/posting/cutoff/set — Уточнить дату отгрузки отправления
      - 'posting_marks' (stub) — POST /v1/posting/marks — Получить маркировки экземпляров из отправления
      - 'polygon_create' (stub) — POST /v1/polygon/create — Создайте полигон доставки
      - 'polygon_bind_v2' (stub) — POST /v2/polygon/bind — Связать метод доставки с полигоном
      - 'polygon_bind_v1' (stub) — POST /v1/polygon/bind — Свяжите метод доставки с полигоном доставки
      - 'polygon_delete' (stub) — POST /v1/polygon/delete — Удалить полигон из области доставки
      - 'polygon_list' (stub) — POST /v1/polygon/list — Получить список установленных полигонов на метод доставки
      - 'polygon_time_coordinates_update' (stub) — POST /v1/polygon/time/coordinates/update — Обновить координаты полигона доставки
      - 'polygon_time_set' (stub) — POST /v1/polygon/time/set — Установить новое время доставки в полигоне
      - 'invoice_create_or_update' (stub) — POST /v2/invoice/create-or-update — Создать или изменить счёт-фактуру
      - 'invoice_file_upload' (stub) — POST /v1/invoice/file/upload — Загрузка счёта-фактуры
      - 'invoice_get' (stub) — POST /v2/invoice/get — Получить информацию о счёте-фактуре
      - 'invoice_delete' (stub) — POST /v1/invoice/delete — Удалить ссылку на счёт-фактуру
      - 'receipts_get' (stub) — POST /v1/receipts/get — Получить чек в формате PDF
      - 'receipts_seller_list' (stub) — POST /v1/receipts/seller/list — Получить список чеков продавца
      - 'receipts_upload' (stub) — POST /v1/receipts/upload — Загрузить чек
    """

    handlers = {
        'order_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/order/create',
            "description": 'Создать заказ',
            "reason": "request schema not verified from documentation",
        },
        'delivery_method_list_v2': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/delivery-method/list',
            "description": 'Список методов доставки realFBS-склада',
            "reason": "request schema not verified from documentation",
        },
        'delivery_method_list_v1': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/delivery-method/list',
            "description": 'Список методов доставки склада',
            "reason": "request schema not verified from documentation",
        },
        'delivery_method_return_settings_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/delivery-method/return/settings/get',
            "description": 'Получить информацию по возвратным настройкам rFBS и rFBS Express',
            "reason": "request schema not verified from documentation",
        },
        'delivery_check': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/delivery/check',
            "description": 'Проверить доступность доставки для покупателя',
            "reason": "request schema not verified from documentation",
        },
        'delivery_checkout': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/delivery/checkout',
            "description": 'Получить доступные варианты доставки',
            "reason": "request schema not verified from documentation",
        },
        'delivery_map': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/delivery/map',
            "description": 'Отрисовать точки на карте',
            "reason": "request schema not verified from documentation",
        },
        'delivery_point_info': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/delivery/point/info',
            "description": 'Получить информацию о точке самовывоза',
            "reason": "request schema not verified from documentation",
        },
        'delivery_point_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/delivery/point/list',
            "description": 'Получить список точек самовывоза',
            "reason": "request schema not verified from documentation",
        },
        'posting_multiboxqty_set': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v3/posting/multiboxqty/set',
            "description": 'Указать количество коробок для многокоробочных отправлений',
            "reason": "request schema not verified from documentation",
        },
        'posting_global_etgb': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/global/etgb',
            "description": 'Таможенные декларации ETGB',
            "reason": "request schema not verified from documentation",
        },
        'posting_unpaid_legal_product_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/unpaid-legal/product/list',
            "description": 'Список неоплаченных товаров, заказанных юридическими лицами',
            "reason": "request schema not verified from documentation",
        },
        'posting_cutoff_set': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/cutoff/set',
            "description": 'Уточнить дату отгрузки отправления',
            "reason": "request schema not verified from documentation",
        },
        'posting_marks': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/marks',
            "description": 'Получить маркировки экземпляров из отправления',
            "reason": "request schema not verified from documentation",
        },
        'polygon_create': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/polygon/create',
            "description": 'Создайте полигон доставки',
            "reason": "request schema not verified from documentation",
        },
        'polygon_bind_v2': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/polygon/bind',
            "description": 'Связать метод доставки с полигоном',
            "reason": "request schema not verified from documentation",
        },
        'polygon_bind_v1': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/polygon/bind',
            "description": 'Свяжите метод доставки с полигоном доставки',
            "reason": "request schema not verified from documentation",
        },
        'polygon_delete': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/polygon/delete',
            "description": 'Удалить полигон из области доставки',
            "reason": "request schema not verified from documentation",
        },
        'polygon_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/polygon/list',
            "description": 'Получить список установленных полигонов на метод доставки',
            "reason": "request schema not verified from documentation",
        },
        'polygon_time_coordinates_update': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/polygon/time/coordinates/update',
            "description": 'Обновить координаты полигона доставки',
            "reason": "request schema not verified from documentation",
        },
        'polygon_time_set': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/polygon/time/set',
            "description": 'Установить новое время доставки в полигоне',
            "reason": "request schema not verified from documentation",
        },
        'invoice_create_or_update': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/invoice/create-or-update',
            "description": 'Создать или изменить счёт-фактуру',
            "reason": "request schema not verified from documentation",
        },
        'invoice_file_upload': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/invoice/file/upload',
            "description": 'Загрузка счёта-фактуры',
            "reason": "request schema not verified from documentation",
        },
        'invoice_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/invoice/get',
            "description": 'Получить информацию о счёте-фактуре',
            "reason": "request schema not verified from documentation",
        },
        'invoice_delete': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/invoice/delete',
            "description": 'Удалить ссылку на счёт-фактуру',
            "reason": "request schema not verified from documentation",
        },
        'receipts_get': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/receipts/get',
            "description": 'Получить чек в формате PDF',
            "reason": "request schema not verified from documentation",
        },
        'receipts_seller_list': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/receipts/seller/list',
            "description": 'Получить список чеков продавца',
            "reason": "request schema not verified from documentation",
        },
        'receipts_upload': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/receipts/upload',
            "description": 'Загрузить чек',
            "reason": "request schema not verified from documentation",
        },
    }
    h = handlers.get(action)
    if not h:
        return {"error": f"unknown action '{action}' for ozon_delivery. Valid: {sorted(handlers)}"}
    return h()


@server.tool()
def ozon_products(action: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Ozon товары — карточки, цены, остатки, атрибуты, импорт, аналитика продаж/остатков, цифровые коды.

    action: один из:
      - 'posting_digital_codes_upload' (stub) — POST /v1/posting/digital/codes/upload — Загрузить коды цифровых товаров для отправления
      - 'posting_digital_list_v1' (stub) — POST /v1/posting/digital/list — Получить список отправлений Deprecated
      - 'posting_digital_list_v2' (stub) — POST /v2/posting/digital/list — Получить список отправлений
      - 'product_list' — POST /v3/product/list — Список товаров (cursor-пагинация через last_id). Тело: {filter:{offer_id,product_id,visibility}, last_id, limit}.
      - 'product_info_list' — POST /v3/product/info/list — Детальная информация по товарам. Один тип id за раз: offer_id|sku|product_id (до 1000). НЕ оборачивать id в filter.
      - 'product_rating_by_sku' — POST /v1/product/rating-by-sku — Контент-рейтинг карточки (0-100) с разбивкой по группам. Тело: {skus:[...]}.
      - 'product_info_prices' — POST /v5/product/info/prices — Цены и прайс-индексы (GREEN/YELLOW/RED). Cursor-пагинация. Тело: {cursor, filter, limit}.
      - 'product_info_limit' — POST /v4/product/info/limit — Лимиты ассортимента (суточные квоты create/update). Тело пустое {}.
      - 'products_stocks' — POST /v2/products/stocks — Остатки товаров по offer_id (массив 1-50). Тело: {offer_id:[...]}.
      - 'product_info_attributes' — POST /v4/product/info/attributes — Полный список атрибутов карточки (включая description id=4191, хэштеги id=23171). Тело: {filter:{product_id:[...]}, limit}.
      - 'product_info_description' (stub) — POST /v1/product/info/description — Текстовое описание карточки (производное от атрибута 4191).
      - 'product_import' [WRITE] — POST /v3/product/import — ЗАПИСЬ: создать/обновить карточку товара (описание, хэштеги, цена, атрибуты). type_id обязателен всегда. Перед записью — показать текущее/готовое значение и ждать подтверждения пользователя (см. ozon-api-overview).
      - 'product_import_info' — POST /v1/product/import/info — Статус задачи импорта карточки. Тело: {task_id}. Ждать status=imported.
      - 'analytics_data' — POST /v1/analytics/data — Выручка и метрики по дням. dimensions ДОЛЖЕН быть ["day"] (не "date"). Максимум 3-4 metrics. Rate limit: 429, жди 5+ сек.
      - 'analytics_stock_on_warehouses' — POST /v2/analytics/stock_on_warehouses — Остатки/перемещения по складам. Тело: {limit, offset, warehouse_type: ALL|EXPRESS_DARK_STORE|NOT_EXPRESS_DARK_STORE}.
      - 'analytics_turnover_stocks' — POST /v1/analytics/turnover/stocks — Оборачиваемость и дни остатка. Rate limit: 1 запрос/мин на Client-Id. Тело: {limit, offset, sku}.
      - 'analytics_stocks' — POST /v1/analytics/stocks — Детальная аналитика остатков по кластерам (обновл. 2x/день). sku обязателен (до 100); cluster_ids и macrolocal_cluster_ids взаимоисключающие.
    """

    def _h_product_list(body: dict) -> dict:
        visibility = body.get("visibility", "ALL")
        offer_id = body.get("offer_id")
        product_id = body.get("product_id")
        limit = body.get("limit", 1000)
        last_id = body.get("last_id", "")
        filt: dict[str, Any] = {"visibility": visibility}
        if offer_id: filt["offer_id"] = offer_id
        if product_id: filt["product_id"] = product_id
        return _seller("POST", "/v3/product/list", {"filter": filt, "last_id": last_id, "limit": limit})

    def _h_product_info_list(body: dict) -> dict:
        offer_id = body.get("offer_id")
        sku = body.get("sku")
        product_id = body.get("product_id")
        ids = [x for x in (offer_id, sku, product_id) if x]
        if len(ids) != 1:
            return {"error": "pass exactly one of offer_id, sku, product_id"}
        b: dict[str, Any] = {}
        if offer_id: b["offer_id"] = offer_id
        if sku: b["sku"] = sku
        if product_id: b["product_id"] = product_id
        return _seller("POST", "/v3/product/info/list", b)

    def _h_product_rating_by_sku(body: dict) -> dict:
        return _seller("POST", "/v1/product/rating-by-sku", {"skus": body.get("skus")})

    def _h_product_info_prices(body: dict) -> dict:
        visibility = body.get("visibility", "ALL")
        offer_id = body.get("offer_id")
        product_id = body.get("product_id")
        limit = body.get("limit", 1000)
        cursor = body.get("cursor", "")
        filt: dict[str, Any] = {"visibility": visibility}
        if offer_id: filt["offer_id"] = offer_id
        if product_id: filt["product_id"] = product_id
        return _seller("POST", "/v5/product/info/prices", {"cursor": cursor, "filter": filt, "limit": limit})

    def _h_product_info_limit(body: dict) -> dict:
        return _seller("POST", "/v4/product/info/limit", {})

    def _h_products_stocks(body: dict) -> dict:
        return _seller("POST", "/v2/products/stocks", {"offer_id": body.get("offer_id")})

    def _h_product_info_attributes(body: dict) -> dict:
        product_id = body.get("product_id")
        offer_id = body.get("offer_id")
        limit = body.get("limit", 100)
        filt: dict[str, Any] = {}
        if product_id: filt["product_id"] = product_id
        if offer_id: filt["offer_id"] = offer_id
        return _seller("POST", "/v4/product/info/attributes", {"filter": filt, "limit": limit})

    def _h_product_import(body: dict) -> dict:
        offer_id = body.get("offer_id")
        name_ = body.get("name") or body.get("name_")
        description_category_id = body.get("description_category_id")
        type_id = body.get("type_id")
        price = body.get("price")
        old_price = body.get("old_price", "")
        vat = body.get("vat", "0")
        currency_code = body.get("currency_code", "RUB")
        height = body.get("height")
        width = body.get("width")
        depth = body.get("depth")
        dimension_unit = body.get("dimension_unit", "mm")
        weight = body.get("weight")
        weight_unit = body.get("weight_unit", "g")
        images = body.get("images")
        primary_image = body.get("primary_image")
        barcode = body.get("barcode")
        attributes = body.get("attributes")
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

    def _h_product_import_info(body: dict) -> dict:
        return _seller("POST", "/v1/product/import/info", {"task_id": body.get("task_id")})

    def _h_analytics_data(body: dict) -> dict:
        date_from = body.get("date_from")
        date_to = body.get("date_to")
        metrics = body.get("metrics")
        dimensions = body.get("dimensions")
        limit = body.get("limit", 100)
        return _seller("POST", "/v1/analytics/data", {
            "date_from": date_from, "date_to": date_to,
            "dimensions": dimensions or ["day"], "metrics": metrics, "limit": limit,
        })

    def _h_analytics_stock_on_warehouses(body: dict) -> dict:
        limit = body.get("limit", 100)
        offset = body.get("offset", 0)
        warehouse_type = body.get("warehouse_type", "ALL")
        return _seller("POST", "/v2/analytics/stock_on_warehouses", {"limit": limit, "offset": offset, "warehouse_type": warehouse_type})

    def _h_analytics_turnover_stocks(body: dict) -> dict:
        sku = body.get("sku")
        limit = body.get("limit")
        offset = body.get("offset")
        b: dict[str, Any] = {}
        if sku: b["sku"] = sku
        if limit is not None: b["limit"] = limit
        if offset is not None: b["offset"] = offset
        return _seller("POST", "/v1/analytics/turnover/stocks", b)

    def _h_analytics_stocks(body: dict) -> dict:
        skus = body.get("skus")
        cluster_ids = body.get("cluster_ids")
        macrolocal_cluster_ids = body.get("macrolocal_cluster_ids")
        warehouse_ids = body.get("warehouse_ids")
        turnover_grades = body.get("turnover_grades")
        item_tags = body.get("item_tags")
        if cluster_ids and macrolocal_cluster_ids:
            return {"error": "cluster_ids and macrolocal_cluster_ids are mutually exclusive"}
        b: dict[str, Any] = {"skus": skus}
        if cluster_ids: b["cluster_ids"] = cluster_ids
        if macrolocal_cluster_ids: b["macrolocal_cluster_ids"] = macrolocal_cluster_ids
        if warehouse_ids: b["warehouse_ids"] = warehouse_ids
        if turnover_grades: b["turnover_grades"] = turnover_grades
        if item_tags: b["item_tags"] = item_tags
        return _seller("POST", "/v1/analytics/stocks", b)

    handlers = {
        'posting_digital_codes_upload': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/digital/codes/upload',
            "description": 'Загрузить коды цифровых товаров для отправления',
            "reason": "request schema not verified from documentation",
        },
        'posting_digital_list_v1': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/posting/digital/list',
            "description": 'Получить список отправлений Deprecated',
            "reason": "request schema not verified from documentation",
        },
        'posting_digital_list_v2': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v2/posting/digital/list',
            "description": 'Получить список отправлений',
            "reason": "request schema not verified from documentation",
        },
        'product_list': lambda: _h_product_list(body or {}),
        'product_info_list': lambda: _h_product_info_list(body or {}),
        'product_rating_by_sku': lambda: _h_product_rating_by_sku(body or {}),
        'product_info_prices': lambda: _h_product_info_prices(body or {}),
        'product_info_limit': lambda: _h_product_info_limit(body or {}),
        'products_stocks': lambda: _h_products_stocks(body or {}),
        'product_info_attributes': lambda: _h_product_info_attributes(body or {}),
        'product_info_description': lambda: {
            "status": "not_implemented",
            "method": 'POST',
            "path": '/v1/product/info/description',
            "description": 'Текстовое описание карточки (производное от атрибута 4191).',
            "reason": "request schema not verified from documentation",
        },
        'product_import': lambda: _h_product_import(body or {}),
        'product_import_info': lambda: _h_product_import_info(body or {}),
        'analytics_data': lambda: _h_analytics_data(body or {}),
        'analytics_stock_on_warehouses': lambda: _h_analytics_stock_on_warehouses(body or {}),
        'analytics_turnover_stocks': lambda: _h_analytics_turnover_stocks(body or {}),
        'analytics_stocks': lambda: _h_analytics_stocks(body or {}),
    }
    h = handlers.get(action)
    if not h:
        return {"error": f"unknown action '{action}' for ozon_products. Valid: {sorted(handlers)}"}
    return h()


@server.tool()
def ozon_finance(action: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Ozon финансы — отчёты о реализации, транзакции и их итоги.

    action: один из:
      - 'finance_realization' — POST /v2/finance/realization — Отчёт о реализации за месяц (доставленные+возвраты, без отмен). Тело: {month, year}.
      - 'finance_realization_posting' — POST /v1/finance/realization/posting — Позаказный отчёт о реализации (только до авг. 2023). Тело: {month, year}.
      - 'finance_transaction_list' — POST /v3/finance/transaction/list — Все транзакции (макс. период 1 месяц). Пагинация page/page_size. Тело: {filter:{date|posting_number, transaction_type}, page, page_size}.
      - 'finance_transaction_totals' — POST /v3/finance/transaction/totals — Итоговые суммы по транзакциям за период. Тело: {date:{from,to}|posting_number, transaction_type}.
    """

    def _h_finance_realization(body: dict) -> dict:
        return _seller("POST", "/v2/finance/realization", {"month": body.get("month"), "year": body.get("year")})

    def _h_finance_realization_posting(body: dict) -> dict:
        return _seller("POST", "/v1/finance/realization/posting", {"month": body.get("month"), "year": body.get("year")})

    def _h_finance_transaction_list(body: dict) -> dict:
        date_from = body.get("date_from")
        date_to = body.get("date_to")
        posting_number = body.get("posting_number")
        transaction_type = body.get("transaction_type", "all")
        page = body.get("page", 1)
        page_size = body.get("page_size", 1000)
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

    def _h_finance_transaction_totals(body: dict) -> dict:
        date_from = body.get("date_from")
        date_to = body.get("date_to")
        posting_number = body.get("posting_number")
        transaction_type = body.get("transaction_type", "all")
        b: dict[str, Any] = {"transaction_type": transaction_type, "posting_number": posting_number or ""}
        if date_from and date_to:
            b["date"] = {"from": f"{date_from}T00:00:00.000Z", "to": f"{date_to}T23:59:59.000Z"}
        return _seller("POST", "/v3/finance/transaction/totals", b)

    handlers = {
        'finance_realization': lambda: _h_finance_realization(body or {}),
        'finance_realization_posting': lambda: _h_finance_realization_posting(body or {}),
        'finance_transaction_list': lambda: _h_finance_transaction_list(body or {}),
        'finance_transaction_totals': lambda: _h_finance_transaction_totals(body or {}),
    }
    h = handlers.get(action)
    if not h:
        return {"error": f"unknown action '{action}' for ozon_finance. Valid: {sorted(handlers)}"}
    return h()


@server.tool()
def ozon_reviews(action: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Ozon отзывы и рейтинги — список/статус отзывов, рейтинги продавца, информация о продавце и логистике.

    action: один из:
      - 'review_count' — POST /v2/review/count — Количество отзывов по статусам (new/viewed/processed/total). Тело пустое {}. Требует подписку.
      - 'review_list' — POST /v2/review/list — Список отзывов, cursor-пагинация last_id, лимит 20-100. Фильтры: sku, status, order_status, даты.
      - 'review_info' — POST /v2/review/info — Детали отзыва с фото/видео. Тело: {review_id}.
      - 'review_change_status' [WRITE] — POST /v2/review/change-status — ЗАПИСЬ: изменить статус отзывов (до 100 за раз). Тело: {review_ids:[...], status: NEW|VIEWED|PROCESSED}.
      - 'rating_summary' — POST /v1/rating/summary — Текущие рейтинги продавца по всем показателям. Тело пустое {}.
      - 'rating_history' — POST /v1/rating/history — История рейтингов за период (макс ~30 дней). Тело: {date_from, date_to, ratings:[...], with_premium_scores}.
      - 'seller_info' — POST /v1/seller/info — Компания, рейтинги, подписка продавца. Тело пустое {}.
      - 'seller_ozon_logistics_info' — POST /v1/seller/ozon-logistics/info — Статус подключения Ozon Доставки, доступные схемы FBO/FBS. Тело пустое {}.
    """

    def _h_review_count(body: dict) -> dict:
        return _seller("POST", "/v2/review/count", {})

    def _h_review_list(body: dict) -> dict:
        sku = body.get("sku")
        status = body.get("status")
        order_status = body.get("order_status")
        date_from = body.get("date_from")
        date_to = body.get("date_to")
        sort_dir = body.get("sort_dir", "DESC")
        limit = body.get("limit", 100)
        last_id = body.get("last_id", "")
        filters: dict[str, Any] = {}
        if sku: filters["sku"] = sku
        if status: filters["status"] = status
        if order_status: filters["order_status"] = order_status
        if date_from: filters["published_from"] = f"{date_from}T00:00:00Z"
        if date_to: filters["published_to"] = f"{date_to}T23:59:59Z"
        return _seller("POST", "/v2/review/list", {"filters": filters, "last_id": last_id, "limit": limit, "sort_dir": sort_dir})

    def _h_review_info(body: dict) -> dict:
        return _seller("POST", "/v2/review/info", {"review_id": body.get("review_id")})

    def _h_review_change_status(body: dict) -> dict:
        return _seller("POST", "/v2/review/change-status", {"review_ids": body.get("review_ids"), "status": body.get("status")})

    def _h_rating_summary(body: dict) -> dict:
        return _seller("POST", "/v1/rating/summary", {})

    def _h_rating_history(body: dict) -> dict:
        date_from = body.get("date_from")
        date_to = body.get("date_to")
        ratings = body.get("ratings")
        with_premium_scores = body.get("with_premium_scores", False)
        return _seller("POST", "/v1/rating/history", {
            "date_from": date_from, "date_to": date_to,
            "ratings": ratings or [], "with_premium_scores": with_premium_scores,
        })

    def _h_seller_info(body: dict) -> dict:
        return _seller("POST", "/v1/seller/info", {})

    def _h_seller_ozon_logistics_info(body: dict) -> dict:
        return _seller("POST", "/v1/seller/ozon-logistics/info", {})

    handlers = {
        'review_count': lambda: _h_review_count(body or {}),
        'review_list': lambda: _h_review_list(body or {}),
        'review_info': lambda: _h_review_info(body or {}),
        'review_change_status': lambda: _h_review_change_status(body or {}),
        'rating_summary': lambda: _h_rating_summary(body or {}),
        'rating_history': lambda: _h_rating_history(body or {}),
        'seller_info': lambda: _h_seller_info(body or {}),
        'seller_ozon_logistics_info': lambda: _h_seller_ozon_logistics_info(body or {}),
    }
    h = handlers.get(action)
    if not h:
        return {"error": f"unknown action '{action}' for ozon_reviews. Valid: {sorted(handlers)}"}
    return h()


@server.tool()
def ozon_ads(action: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Ozon Performance API (реклама) — кампании, бюджет кампании, статистика расходов.

    action: один из:
      - 'api_client_campaign' — GET /api/client/campaign — Список рекламных кампаний.
      - 'api_client_campaign_id_get' — GET /api/client/campaign/{id} — ВНИМАНИЕ: возвращает 405, не рабочий — используй список выше.
      - 'api_client_campaign_id_put' [WRITE] — PUT /api/client/campaign/{id} — ЗАПИСЬ: изменить еженедельный бюджет кампании.
      - 'api_client_statistics_daily' — GET /api/client/statistics/daily — ОСНОВНОЙ: CSV статистика по дням/кампаниям. Параметры dateFrom/dateTo (YYYY-MM-DD).
      - 'api_client_statistics_expense' — GET /api/client/statistics/expense — CSV, детализация расходов (бонусы/абонентский счёт).
      - 'api_client_statistics' — POST /api/client/statistics — JSON статистика по campaignIds (массив). Медленнее, расходует квоты.
    """

    def _h_api_client_campaign(body: dict) -> dict:
        return _performance("GET", "/api/client/campaign")

    def _h_api_client_campaign_id_get(body: dict) -> dict:
        campaign_id = body.get("campaign_id")
        return _performance("GET", "/api/client/campaign/" + campaign_id)

    def _h_api_client_campaign_id_put(body: dict) -> dict:
        campaign_id = body.get("campaign_id")
        weekly_budget = body.get("weekly_budget")
        daily_budget = body.get("daily_budget", 0)
        title = body.get("title")
        state = body.get("state")
        if title is None or state is None:
            current = _performance("GET", "/api/client/campaign/" + campaign_id)
            if isinstance(current, dict) and "error" not in current:
                title = title if title is not None else current.get("title", "")
                state = state if state is not None else current.get("state", "CAMPAIGN_STATE_RUNNING")
            else:
                title = title or ""
                state = state or "CAMPAIGN_STATE_RUNNING"
        b = {
            "id": campaign_id, "weeklyBudget": str(weekly_budget),
            "dailyBudget": str(daily_budget) if daily_budget > 0 else "0",
            "title": title, "state": state,
        }
        return _performance("PUT", "/api/client/campaign/" + campaign_id, b)

    def _h_api_client_statistics_daily(body: dict) -> dict:
        return _performance("GET", "/api/client/statistics/daily", params={"dateFrom": body.get("date_from"), "dateTo": body.get("date_to")})

    def _h_api_client_statistics_expense(body: dict) -> dict:
        return _performance("GET", "/api/client/statistics/expense", params={"dateFrom": body.get("date_from"), "dateTo": body.get("date_to")})

    def _h_api_client_statistics(body: dict) -> dict:
        return _performance("POST", "/api/client/statistics", {"campaignIds": body.get("campaign_ids")})

    handlers = {
        'api_client_campaign': lambda: _h_api_client_campaign(body or {}),
        'api_client_campaign_id_get': lambda: _h_api_client_campaign_id_get(body or {}),
        'api_client_campaign_id_put': lambda: _h_api_client_campaign_id_put(body or {}),
        'api_client_statistics_daily': lambda: _h_api_client_statistics_daily(body or {}),
        'api_client_statistics_expense': lambda: _h_api_client_statistics_expense(body or {}),
        'api_client_statistics': lambda: _h_api_client_statistics(body or {}),
    }
    h = handlers.get(action)
    if not h:
        return {"error": f"unknown action '{action}' for ozon_ads. Valid: {sorted(handlers)}"}
    return h()





if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
