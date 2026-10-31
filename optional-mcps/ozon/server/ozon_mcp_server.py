#!/usr/bin/env python3
"""
Ozon API MCP connector for Hermes.

Native connector layer: wraps the Ozon Seller API (https://api-seller.ozon.ru)
and Performance API (https://api-performance.ozon.ru) as MCP tools. Business
logic ("which API/report for X") stays in the ozon-* Hermes skills (unchanged
by this connector) -- this server only does authenticated HTTP I/O, using the
SAME real endpoints already proven working in those skills' scripts/*.py,
extended with every other endpoint documented in
ozon-api-overview/references/api.md and the ozon-seller-skill reference docs.

Same conventions as the Avito connector (optional-mcps/avito/):
- one MCP server, own venv, catalog manifest alongside it
- category tools instead of one tool per endpoint (222 endpoints across 18
  categories -- hand-writing one function per endpoint is unmaintainable).
  Each category tool takes an `endpoint` constrained to a whitelist built
  from the real documentation (never an invented path) plus a `body` dict.
- a meta tool (ozon_list_endpoints) lets the agent discover exact paths and
  descriptions instead of guessing.

Auth:
- Seller API: Client-Id + Api-Key headers (OZON_API_KEY / OZON_CLIENT_ID).
- Performance API: OAuth2 client_credentials (OZON_PERFORMANCE_CLIENT_ID /
  OZON_PERFORMANCE_CLIENT_SECRET), token cached in-process (~30 min TTL),
  same flow as ozon-ads-performance/scripts/init.py.

WRITE endpoints (product import, review status, campaign budget) are marked
`"note": "write"` in the registry and in their tool docstrings -- the hard
rule from ozon-api-overview still applies: show the current/proposed value
and wait for explicit confirmation before calling them.
"""
from __future__ import annotations

import json as _json
import os
import time
from typing import Any, Optional

import httpx
from mcp.server.mcpserver import MCPServer

SELLER_BASE_URL = "https://api-seller.ozon.ru"
PERFORMANCE_BASE_URL = "https://api-performance.ozon.ru"

# ---------------------------------------------------------------------------
# Endpoint registry -- built from ozon-api-overview/references/api.md and the
# ozon-seller-skill reference docs (products/finance/analytics/reviews/
# sellerrating/sellerinfo/practice.md) plus ozon-ads-performance's working
# scripts. Every path here is a real, documented Ozon endpoint -- nothing
# invented. Extend only against a confirmed doc/script, never a guess.
# Stored as a JSON literal (not a Python dict literal) to dodge json/Python
# null-vs-None mismatches; parsed once at import time.
# ---------------------------------------------------------------------------

_REGISTRY_JSON = r'''{"fbs_postings": {"label": "FBS: отправления, статусы, этикетки, упаковка, исключения, таймслоты", "endpoints": [{"method": "POST", "path": "/v4/posting/fbs/unfulfilled/list", "desc": "Получить список необработанных отправлений"}, {"method": "POST", "path": "/v3/posting/fbs/unfulfilled/list", "desc": "Список необработанных отправлений Deprecated"}, {"method": "POST", "path": "/v4/posting/fbs/list", "desc": "Получить список отправлений"}, {"method": "POST", "path": "/v3/posting/fbs/list", "desc": "Список отправлений Deprecated"}, {"method": "POST", "path": "/v3/posting/fbs/get", "desc": "Получить информацию об отправлении по идентификатору"}, {"method": "POST", "path": "/v2/posting/fbs/get-by-barcode", "desc": "Получить информацию об отправлении по штрихкоду"}, {"method": "POST", "path": "/v2/posting/fbs/product/country/list", "desc": "Список доступных стран-изготовителей"}, {"method": "POST", "path": "/v2/posting/fbs/product/country/set", "desc": "Добавить информацию о стране-изготовителе товара"}, {"method": "POST", "path": "/v1/posting/fbs/restrictions", "desc": "Получить ограничения пункта приёма"}, {"method": "POST", "path": "/v2/posting/fbs/package-label", "desc": "Напечатать этикетку"}, {"method": "POST", "path": "/v2/posting/fbs/package-label/create", "desc": "Создать задание на формирование этикеток"}, {"method": "POST", "path": "/v1/posting/fbs/package-label/create", "desc": "Создать задание на выгрузку этикеток"}, {"method": "POST", "path": "/v1/posting/fbs/package-label/get", "desc": "Получить файл с этикетками"}, {"method": "POST", "path": "/v1/posting/fbs/cancel-reason", "desc": "Причины отмены отправления"}, {"method": "POST", "path": "/v2/posting/fbs/cancel-reason/list", "desc": "Причины отмены отправлений"}, {"method": "POST", "path": "/v2/posting/fbs/product/cancel", "desc": "Отменить отправку некоторых товаров в отправлении"}, {"method": "POST", "path": "/v2/posting/fbs/cancel", "desc": "Отменить отправление"}, {"method": "POST", "path": "/v2/posting/fbs/arbitration", "desc": "Открыть спор по отправлению"}, {"method": "POST", "path": "/v2/posting/fbs/awaiting-delivery", "desc": "Передать отправление к отгрузке"}, {"method": "POST", "path": "/v1/posting/fbs/pick-up-code/verify", "desc": "Проверить код курьера"}, {"method": "POST", "path": "/v4/posting/fbs/ship", "desc": "Собрать заказ (версия 4)"}, {"method": "POST", "path": "/v4/posting/fbs/ship/package", "desc": "Частичная сборка отправления (версия 4)"}, {"method": "POST", "path": "/v6/fbs/posting/product/exemplar/set", "desc": "Проверить и сохранить данные экземпляров"}, {"method": "POST", "path": "/v6/fbs/posting/product/exemplar/create-or-get", "desc": "Получить данные созданных экземпляров"}, {"method": "POST", "path": "/v5/fbs/posting/product/exemplar/status", "desc": "Получить статус добавления экземпляров"}, {"method": "POST", "path": "/v5/fbs/posting/product/exemplar/validate", "desc": "Валидация кодов маркировки"}, {"method": "POST", "path": "/v1/fbs/posting/product/exemplar/update", "desc": "Обновить данные экземпляров"}, {"method": "POST", "path": "/v2/posting/fbs/act/create", "desc": "Подтвердить отгрузку и создать документы"}, {"method": "POST", "path": "/v1/posting/fbs/split", "desc": "Разделить заказ на отправления без сборки"}, {"method": "POST", "path": "/v2/posting/fbs/act/get-postings", "desc": "Список отправлений в акте"}, {"method": "POST", "path": "/v2/posting/fbs/act/get-container-labels", "desc": "Этикетки для грузового места"}, {"method": "POST", "path": "/v2/posting/fbs/act/get-barcode", "desc": "Штрихкод для отгрузки отправления"}, {"method": "POST", "path": "/v2/posting/fbs/act/get-barcode/text", "desc": "Значение штрихкода для отгрузки отправления"}, {"method": "POST", "path": "/v2/posting/fbs/digital/act/check-status", "desc": "Статус формирования накладной Deprecated"}, {"method": "POST", "path": "/v2/posting/fbs/act/get-pdf", "desc": "Получить PDF c документами"}, {"method": "POST", "path": "/v2/posting/fbs/act/list", "desc": "Список актов по отгрузкам"}, {"method": "POST", "path": "/v2/posting/fbs/digital/act/get-pdf", "desc": "Получить лист отгрузки по перевозке Deprecated"}, {"method": "POST", "path": "/v2/posting/fbs/act/check-status", "desc": "Статус отгрузки и документов"}, {"method": "POST", "path": "/v1/posting/fbs/traceable/split", "desc": "Разделить отправление с прослеживаемыми товарами"}, {"method": "POST", "path": "/v1/posting/fbs/product/traceable/attribute", "desc": "Получить список незаполненных атрибутов для прослеживаемых товаров"}, {"method": "POST", "path": "/v1/assembly/carriage/posting/list", "desc": "Получить список отправлений в отгрузке"}, {"method": "POST", "path": "/v1/assembly/carriage/product/list", "desc": "Получить список товаров в отгрузке"}, {"method": "POST", "path": "/v1/assembly/fbs/posting/list", "desc": "Получить список отправлений"}, {"method": "POST", "path": "/v1/assembly/fbs/product/list", "desc": "Получить список товаров в отправлениях"}, {"method": "POST", "path": "/v2/fbs/posting/tracking-number/set", "desc": "Добавить трек-номера"}, {"method": "POST", "path": "/v2/fbs/posting/delivering", "desc": "Изменить статус на «Доставляется»"}, {"method": "POST", "path": "/v2/fbs/posting/last-mile", "desc": "Изменить статус на «Последняя миля»"}, {"method": "POST", "path": "/v2/fbs/posting/delivered", "desc": "Изменить статус на «Доставлено»"}, {"method": "POST", "path": "/v1/posting/fbs/timeslot/change-restrictions", "desc": "Доступные даты для переноса доставки"}, {"method": "POST", "path": "/v1/posting/fbs/timeslot/set", "desc": "Перенести дату доставки"}]}, "fbp": {"label": "FBP (Fulfillment by Partner): черновики поставок, заказы, акты, этикетки", "endpoints": [{"method": "POST", "path": "/v1/fbp/warehouse/list", "desc": "Получить список партнёрских складов"}, {"method": "POST", "path": "/v1/fbp/draft/get", "desc": "Получить информацию о черновике поставки"}, {"method": "POST", "path": "/v1/fbp/draft/list", "desc": "Список черновиков поставки"}, {"method": "POST", "path": "/v1/fbp/draft/direct/seller-dlv/create", "desc": "Создать черновик с доставкой силами продавца"}, {"method": "POST", "path": "/v1/fbp/draft/direct/seller-dlv/edit", "desc": "Обновить информацию о доставке силами продавца в черновике"}, {"method": "POST", "path": "/v1/fbp/draft/direct/timeslot/edit", "desc": "Отредактировать таймслот в черновике"}, {"method": "POST", "path": "/v1/fbp/draft/direct/timeslot/get", "desc": "Получить список таймслотов для прямой поставки"}, {"method": "POST", "path": "/v1/fbp/draft/direct/create", "desc": "Создать черновик заявки на поставку без указания способа доставки"}, {"method": "POST", "path": "/v1/fbp/draft/direct/delete", "desc": "Удалить черновик заявки на поставку"}, {"method": "POST", "path": "/v1/fbp/draft/direct/product/validate", "desc": "Проверить список товаров для склада партнёра"}, {"method": "POST", "path": "/v1/fbp/draft/direct/registrate", "desc": "Перевести черновик в действующую поставку"}, {"method": "POST", "path": "/v1/fbp/draft/direct/tpl-dlv/create", "desc": "Создать черновик заявки на доставку сторонней транспортной компанией"}, {"method": "POST", "path": "/v1/fbp/draft/direct/tpl-dlv/edit", "desc": "Редактировать черновик поставки со способом доставки сторонней транспортной компанией"}, {"method": "POST", "path": "/v1/fbp/draft/drop-off/create", "desc": "Создать черновик для доставки в drop-off пункт"}, {"method": "POST", "path": "/v1/fbp/draft/drop-off/delete", "desc": "Удалить черновик для доставки в drop-off пункт"}, {"method": "POST", "path": "/v1/fbp/draft/drop-off/dlv/edit", "desc": "Отредактировать детали доставки для drop-off черновика"}, {"method": "POST", "path": "/v1/fbp/draft/drop-off/registrate", "desc": "Перевести черновик в действующую поставку"}, {"method": "POST", "path": "/v1/fbp/draft/drop-off/province/list", "desc": "Получить список провинций"}, {"method": "POST", "path": "/v1/fbp/draft/drop-off/point/list", "desc": "Получить список drop-off пунктов в провинции"}, {"method": "POST", "path": "/v1/fbp/draft/drop-off/product/validate", "desc": "Проверить список товаров, которые склад партнёра может принять"}, {"method": "POST", "path": "/v1/fbp/draft/pick-up/create", "desc": "Создать черновик заявки на pick-up поставку"}, {"method": "POST", "path": "/v1/fbp/draft/pick-up/delete", "desc": "Отменить черновик заявки на pick-up поставку"}, {"method": "POST", "path": "/v1/fbp/draft/pick-up/dlv/edit", "desc": "Изменить черновик заявки на pick-up поставку"}, {"method": "POST", "path": "/v1/fbp/draft/pick-up/product/validate", "desc": "Провалидировать список товаров для pick-up поставки"}, {"method": "POST", "path": "/v1/fbp/draft/pick-up/registrate", "desc": "Перевести черновик в действующую поставку"}, {"method": "POST", "path": "/v1/fbp/order/direct/cancel", "desc": "Отменить поставку"}, {"method": "POST", "path": "/v1/fbp/order/direct/seller-dlv/edit", "desc": "Обновить информацию о доставке силами продавца"}, {"method": "POST", "path": "/v1/fbp/order/direct/timeslot/edit", "desc": "Отредактировать таймслот в заявке на поставку"}, {"method": "POST", "path": "/v1/fbp/order/direct/timeslot/list", "desc": "Получить список таймслотов для поставки"}, {"method": "POST", "path": "/v1/fbp/order/drop-off/cancel", "desc": "Отменить поставку drop-off"}, {"method": "POST", "path": "/v1/fbp/order/drop-off/dlv/edit", "desc": "Отредактировать информацию о поставке на drop-off пункт"}, {"method": "POST", "path": "/v1/fbp/order/pick-up/cancel", "desc": "Отменить pick-up поставку"}, {"method": "POST", "path": "/v1/fbp/order/pick-up/dlv/edit", "desc": "Изменить данные о точке забора"}, {"method": "POST", "path": "/v1/fbp/act-from/create", "desc": "Сгенерировать акт приёмки"}, {"method": "POST", "path": "/v1/fbp/act-from/get", "desc": "Получить статус генерации акта приёмки"}, {"method": "POST", "path": "/v1/fbp/act-to/create", "desc": "Сгенерировать транспортную накладную"}, {"method": "POST", "path": "/v1/fbp/act-to/get", "desc": "Получить статус генерации транспортной накладной"}, {"method": "POST", "path": "/v1/fbp/archive/get", "desc": "Получить информацию о завершённой поставке"}, {"method": "POST", "path": "/v1/fbp/archive/list", "desc": "Получить список завершённых поставок"}, {"method": "POST", "path": "/v1/fbp/label/create", "desc": "Cоздать задание на генерацию этикеток"}, {"method": "POST", "path": "/v1/fbp/label/get", "desc": "Получить статус задания на генерацию этикеток"}, {"method": "POST", "path": "/v1/fbp/order/get", "desc": "Получить информацию о конкретной поставке"}, {"method": "POST", "path": "/v1/fbp/order/list", "desc": "Получить список поставок"}, {"method": "POST", "path": "/v1/posting/fbp/list", "desc": "Получить список отправлений"}]}, "carriage": {"label": "Перевозки: грузоперевозки, контейнеры, пропуска, акты расхождений", "endpoints": [{"method": "POST", "path": "/v1/carriage/create", "desc": "Создание отгрузки"}, {"method": "POST", "path": "/v1/carriage/approve", "desc": "Подтверждение отгрузки"}, {"method": "POST", "path": "/v1/carriage/set-postings", "desc": "Изменение состава отгрузки"}, {"method": "POST", "path": "/v1/carriage/cancel", "desc": "Удаление отгрузки"}, {"method": "POST", "path": "/v2/carriage/delivery/list", "desc": "Список методов доставки и отгрузок"}, {"method": "POST", "path": "/v1/carriage/delivery/list", "desc": "Список методов доставки и отгрузок"}, {"method": "POST", "path": "/v1/carriage/get", "desc": "Информация о перевозке"}, {"method": "POST", "path": "/v1/carriage/act-discrepancy/pdf", "desc": "Получить акт о расхождениях по отгрузке FBS"}, {"method": "POST", "path": "/v1/carriage/ettn/status", "desc": "Получить статус проверки электронной ТТН на прослеживаемой перевозке FBS"}, {"method": "POST", "path": "/v1/carriage/pass/create", "desc": "Создать пропуск"}, {"method": "POST", "path": "/v1/carriage/pass/update", "desc": "Обновить пропуск"}, {"method": "POST", "path": "/v1/carriage/pass/delete", "desc": "Удалить пропуск"}, {"method": "POST", "path": "/v1/carriage/container/create", "desc": "Создать грузоместо"}, {"method": "POST", "path": "/v1/carriage/container/fill", "desc": "Наполнить грузоместо отправлениями"}, {"method": "POST", "path": "/v1/carriage/container/approve", "desc": "Подтвердить состав грузоместа"}, {"method": "POST", "path": "/v1/carriage/container/place-into", "desc": "Разместить коробки на палете"}, {"method": "POST", "path": "/v1/carriage/container/remove-postings", "desc": "Убрать отправления из грузоместа"}, {"method": "POST", "path": "/v1/carriage/container/remove-from", "desc": "Убрать коробки с палеты"}, {"method": "POST", "path": "/v1/carriage/container/cancel", "desc": "Отменить грузоместо"}, {"method": "POST", "path": "/v1/carriage/container/list", "desc": "Получить список грузомест"}, {"method": "POST", "path": "/v1/carriage/container/get", "desc": "Получить информацию о грузоместах"}, {"method": "POST", "path": "/v1/carriage/container/status/get", "desc": "Получить статус грузомест FBS"}, {"method": "POST", "path": "/v1/carriage/container/task/info", "desc": "Получить статус задачи грузового места"}, {"method": "POST", "path": "/v1/carriage/container/document/get", "desc": "Получить документы по грузоместам — ТрН и лист отгрузки"}, {"method": "POST", "path": "/v1/carriage/container/label/get", "desc": "Получить этикетку по грузоместам"}]}, "returns": {"label": "Возвраты: rFBS, список возвратов, вывоз (giveout), настройки утилизации", "endpoints": [{"method": "POST", "path": "/v1/return/pass/create", "desc": "Создать пропуск для возврата"}, {"method": "POST", "path": "/v1/return/pass/update", "desc": "Обновить пропуск для возврата"}, {"method": "POST", "path": "/v1/return/pass/delete", "desc": "Удалить пропуск для возврата"}, {"method": "POST", "path": "/v1/returns/list", "desc": "Информация о возвратах FBO и FBS"}, {"method": "POST", "path": "/v1/returns/settings/utilization/history", "desc": "Получить историю изменений автоутилизации"}, {"method": "POST", "path": "/v1/returns/settings/utilization/info", "desc": "Получить настройки автоутилизации"}, {"method": "POST", "path": "/v1/returns/settings/utilization/update", "desc": "Обновить настройки автоутилизации"}, {"method": "POST", "path": "/v2/returns/rfbs/list", "desc": "Список заявок на возврат"}, {"method": "POST", "path": "/v2/returns/rfbs/get", "desc": "Информация о заявке на возврат"}, {"method": "POST", "path": "/v2/returns/rfbs/reject", "desc": "Отклонить заявку на возврат"}, {"method": "POST", "path": "/v2/returns/rfbs/compensate", "desc": "Вернуть часть стоимости товара"}, {"method": "POST", "path": "/v2/returns/rfbs/verify", "desc": "Одобрить заявку на возврат"}, {"method": "POST", "path": "/v2/returns/rfbs/receive-return", "desc": "Подтвердить получение товара на проверку"}, {"method": "POST", "path": "/v2/returns/rfbs/return-money", "desc": "Вернуть деньги покупателю"}, {"method": "POST", "path": "/v1/returns/rfbs/action/set", "desc": "Передать доступные действия для rFBS возвратов"}, {"method": "POST", "path": "/v1/returns/company/fbs/info", "desc": "Количество возвратов FBS"}, {"method": "POST", "path": "/v1/return/giveout/is-enabled", "desc": "Проверить возможность получения возвратных отгрузок по штрихкоду"}, {"method": "POST", "path": "/v1/return/giveout/list", "desc": "Список возвратных отгрузок"}, {"method": "POST", "path": "/v1/return/giveout/info", "desc": "Информация о возвратной отгрузке"}, {"method": "POST", "path": "/v1/return/giveout/barcode", "desc": "Значение штрихкода для возвратных отгрузок"}, {"method": "POST", "path": "/v1/return/giveout/get-pdf", "desc": "Штрихкод для получения возвратной отгрузки в формате PDF"}, {"method": "POST", "path": "/v1/return/giveout/get-png", "desc": "Штрихкод для получения возвратной отгрузки в формате PNG"}, {"method": "POST", "path": "/v1/return/giveout/barcode-reset", "desc": "Сгенерировать новый штрихкод"}]}, "orders": {"label": "Заказы: отмена заказов, причины отмены, условные отмены", "endpoints": [{"method": "POST", "path": "/v2/conditional-cancellation/list", "desc": "Получить список заявок на отмену rFBS"}, {"method": "POST", "path": "/v2/conditional-cancellation/approve", "desc": "Подтвердить заявку на отмену rFBS"}, {"method": "POST", "path": "/v2/conditional-cancellation/reject", "desc": "Отклонить заявку на отмену rFBS"}, {"method": "POST", "path": "/v1/cancel-reason/list", "desc": "Причины отмены отправлений"}, {"method": "POST", "path": "/v1/cancel-reason/list-by-order", "desc": "Причины отмены заказа"}, {"method": "POST", "path": "/v1/cancel-reason/list-by-posting", "desc": "Причины отмены отправления"}, {"method": "POST", "path": "/v1/order/cancel", "desc": "Отменить заказ"}, {"method": "POST", "path": "/v1/order/cancel/check", "desc": "Проверить возможность отмены заказа"}, {"method": "POST", "path": "/v1/order/cancel/status", "desc": "Получить статус отмены заказа"}, {"method": "POST", "path": "/v2/order/create", "desc": "Создать заказ"}]}, "delivery": {"label": "Доставка: методы доставки, проверка адреса, точки, карта", "endpoints": [{"method": "POST", "path": "/v2/delivery-method/list", "desc": "Список методов доставки realFBS-склада"}, {"method": "POST", "path": "/v1/delivery-method/list", "desc": "Список методов доставки склада"}, {"method": "POST", "path": "/v1/delivery-method/return/settings/get", "desc": "Получить информацию по возвратным настройкам rFBS и rFBS Express"}, {"method": "POST", "path": "/v1/delivery/check", "desc": "Проверить доступность доставки для покупателя"}, {"method": "POST", "path": "/v2/delivery/checkout", "desc": "Получить доступные варианты доставки"}, {"method": "POST", "path": "/v1/delivery/map", "desc": "Отрисовать точки на карте"}, {"method": "POST", "path": "/v1/delivery/point/info", "desc": "Получить информацию о точке самовывоза"}, {"method": "POST", "path": "/v1/delivery/point/list", "desc": "Получить список точек самовывоза"}]}, "postings_misc": {"label": "Отправления (прочее): маркировка, мультикоробочность, иностранные заказы", "endpoints": [{"method": "POST", "path": "/v3/posting/multiboxqty/set", "desc": "Указать количество коробок для многокоробочных отправлений"}, {"method": "POST", "path": "/v1/posting/global/etgb", "desc": "Таможенные декларации ETGB"}, {"method": "POST", "path": "/v1/posting/unpaid-legal/product/list", "desc": "Список неоплаченных товаров, заказанных юридическими лицами"}, {"method": "POST", "path": "/v1/posting/carriage-available/list", "desc": "Список доступных перевозок"}, {"method": "POST", "path": "/v1/posting/cutoff/set", "desc": "Уточнить дату отгрузки отправления"}, {"method": "POST", "path": "/v1/posting/cancel", "desc": "Отменить отправление из заказа"}, {"method": "POST", "path": "/v1/posting/cancel/status", "desc": "Проверить статус отмены отправления"}, {"method": "POST", "path": "/v1/posting/marks", "desc": "Получить маркировки экземпляров из отправления"}]}, "polygon": {"label": "Полигоны доставки: создание/привязка зон, расписание", "endpoints": [{"method": "POST", "path": "/v1/polygon/create", "desc": "Создайте полигон доставки"}, {"method": "POST", "path": "/v2/polygon/bind", "desc": "Связать метод доставки с полигоном"}, {"method": "POST", "path": "/v1/polygon/bind", "desc": "Свяжите метод доставки с полигоном доставки"}, {"method": "POST", "path": "/v1/polygon/delete", "desc": "Удалить полигон из области доставки"}, {"method": "POST", "path": "/v1/polygon/list", "desc": "Получить список установленных полигонов на метод доставки"}, {"method": "POST", "path": "/v1/polygon/time/coordinates/update", "desc": "Обновить координаты полигона доставки"}, {"method": "POST", "path": "/v1/polygon/time/set", "desc": "Установить новое время доставки в полигоне"}]}, "documents": {"label": "Документы: счета-фактуры (invoice), чеки (receipts)", "endpoints": [{"method": "POST", "path": "/v2/invoice/create-or-update", "desc": "Создать или изменить счёт-фактуру"}, {"method": "POST", "path": "/v1/invoice/file/upload", "desc": "Загрузка счёта-фактуры"}, {"method": "POST", "path": "/v2/invoice/get", "desc": "Получить информацию о счёте-фактуре"}, {"method": "POST", "path": "/v1/invoice/delete", "desc": "Удалить ссылку на счёт-фактуру"}, {"method": "POST", "path": "/v1/receipts/get", "desc": "Получить чек в формате PDF"}, {"method": "POST", "path": "/v1/receipts/seller/list", "desc": "Получить список чеков продавца"}, {"method": "POST", "path": "/v1/receipts/upload", "desc": "Загрузить чек"}]}, "fbo_postings": {"label": "FBO: отправления со склада Ozon, причины отмены, доступные склады", "endpoints": [{"method": "POST", "path": "/v3/posting/fbo/list", "desc": "Получить список отправлений"}, {"method": "POST", "path": "/v2/posting/fbo/list", "desc": "Список отправлений Deprecated"}, {"method": "POST", "path": "/v2/posting/fbo/get", "desc": "Информация об отправлении"}, {"method": "POST", "path": "/v1/posting/fbo/cancel-reason/list", "desc": "Причины отмены отправлений по схеме FBO"}, {"method": "GET", "path": "/v1/supplier/available_warehouses", "desc": "Загруженность складов Ozon"}]}, "digital_postings": {"label": "Цифровые отправления: коды активации", "endpoints": [{"method": "POST", "path": "/v1/posting/digital/codes/upload", "desc": "Загрузить коды цифровых товаров для отправления"}, {"method": "POST", "path": "/v1/posting/digital/list", "desc": "Получить список отправлений Deprecated"}, {"method": "POST", "path": "/v2/posting/digital/list", "desc": "Получить список отправлений"}]}, "products": {"label": "Товары: список, детали, цены, остатки, атрибуты, импорт (запись)", "endpoints": [{"method": "POST", "path": "/v3/product/list", "desc": "Список товаров (cursor-пагинация через last_id). Тело: {filter:{offer_id,product_id,visibility}, last_id, limit}.", "note": null}, {"method": "POST", "path": "/v3/product/info/list", "desc": "Детальная информация по товарам. Один тип id за раз: offer_id|sku|product_id (до 1000). НЕ оборачивать id в filter.", "note": null}, {"method": "POST", "path": "/v1/product/rating-by-sku", "desc": "Контент-рейтинг карточки (0-100) с разбивкой по группам. Тело: {skus:[...]}.", "note": null}, {"method": "POST", "path": "/v5/product/info/prices", "desc": "Цены и прайс-индексы (GREEN/YELLOW/RED). Cursor-пагинация. Тело: {cursor, filter, limit}.", "note": null}, {"method": "POST", "path": "/v4/product/info/limit", "desc": "Лимиты ассортимента (суточные квоты create/update). Тело пустое {}.", "note": null}, {"method": "POST", "path": "/v2/products/stocks", "desc": "Остатки товаров по offer_id (массив 1-50). Тело: {offer_id:[...]}.", "note": null}, {"method": "POST", "path": "/v4/product/info/attributes", "desc": "Полный список атрибутов карточки (включая description id=4191, хэштеги id=23171). Тело: {filter:{product_id:[...]}, limit}.", "note": null}, {"method": "POST", "path": "/v1/product/info/description", "desc": "Текстовое описание карточки (производное от атрибута 4191).", "note": null}, {"method": "POST", "path": "/v3/product/import", "desc": "ЗАПИСЬ: создать/обновить карточку товара (описание, хэштеги, цена, атрибуты). type_id обязателен всегда. Перед записью — показать текущее/готовое значение и ждать подтверждения пользователя (см. ozon-api-overview).", "note": "write"}, {"method": "POST", "path": "/v1/product/import/info", "desc": "Статус задачи импорта карточки. Тело: {task_id}. Ждать status=imported.", "note": null}]}, "analytics": {"label": "Аналитика: выручка/метрики, остатки по складам и кластерам, оборачиваемость", "endpoints": [{"method": "POST", "path": "/v1/analytics/data", "desc": "Выручка и метрики по дням. dimensions ДОЛЖЕН быть [\"day\"] (не \"date\"). Максимум 3-4 metrics. Rate limit: 429, жди 5+ сек.", "note": null}, {"method": "POST", "path": "/v2/analytics/stock_on_warehouses", "desc": "Остатки/перемещения по складам. Тело: {limit, offset, warehouse_type: ALL|EXPRESS_DARK_STORE|NOT_EXPRESS_DARK_STORE}.", "note": null}, {"method": "POST", "path": "/v1/analytics/turnover/stocks", "desc": "Оборачиваемость и дни остатка. Rate limit: 1 запрос/мин на Client-Id. Тело: {limit, offset, sku}.", "note": null}, {"method": "POST", "path": "/v1/analytics/stocks", "desc": "Детальная аналитика остатков по кластерам (обновл. 2x/день). sku обязателен (до 100); cluster_ids и macrolocal_cluster_ids взаимоисключающие.", "note": null}]}, "finance": {"label": "Финансы: реализация, транзакции, итоги", "endpoints": [{"method": "POST", "path": "/v2/finance/realization", "desc": "Отчёт о реализации за месяц (доставленные+возвраты, без отмен). Тело: {month, year}.", "note": null}, {"method": "POST", "path": "/v1/finance/realization/posting", "desc": "Позаказный отчёт о реализации (только до авг. 2023). Тело: {month, year}.", "note": null}, {"method": "POST", "path": "/v3/finance/transaction/list", "desc": "Все транзакции (макс. период 1 месяц). Пагинация page/page_size. Тело: {filter:{date|posting_number, transaction_type}, page, page_size}.", "note": null}, {"method": "POST", "path": "/v3/finance/transaction/totals", "desc": "Итоговые суммы по транзакциям за период. Тело: {date:{from,to}|posting_number, transaction_type}.", "note": null}]}, "reviews": {"label": "Отзывы: список, детали, статус (запись)", "endpoints": [{"method": "POST", "path": "/v2/review/count", "desc": "Количество отзывов по статусам (new/viewed/processed/total). Тело пустое {}. Требует подписку.", "note": null}, {"method": "POST", "path": "/v2/review/list", "desc": "Список отзывов, cursor-пагинация last_id, лимит 20-100. Фильтры: sku, status, order_status, даты.", "note": null}, {"method": "POST", "path": "/v2/review/info", "desc": "Детали отзыва с фото/видео. Тело: {review_id}.", "note": null}, {"method": "POST", "path": "/v2/review/change-status", "desc": "ЗАПИСЬ: изменить статус отзывов (до 100 за раз). Тело: {review_ids:[...], status: NEW|VIEWED|PROCESSED}.", "note": "write"}]}, "rating": {"label": "Рейтинг продавца: сводка, история", "endpoints": [{"method": "POST", "path": "/v1/rating/summary", "desc": "Текущие рейтинги продавца по всем показателям. Тело пустое {}.", "note": null}, {"method": "POST", "path": "/v1/rating/history", "desc": "История рейтингов за период (макс ~30 дней). Тело: {date_from, date_to, ratings:[...], with_premium_scores}.", "note": null}]}, "seller_info": {"label": "Кабинет продавца: инфо о компании, статус логистики", "endpoints": [{"method": "POST", "path": "/v1/seller/info", "desc": "Компания, рейтинги, подписка продавца. Тело пустое {}.", "note": null}, {"method": "POST", "path": "/v1/seller/ozon-logistics/info", "desc": "Статус подключения Ozon Доставки, доступные схемы FBO/FBS. Тело пустое {}.", "note": null}]}, "ads": {"label": "Performance API: рекламные кампании, бюджеты, статистика расходов", "endpoints": [{"method": "GET", "path": "/api/client/campaign", "desc": "Список рекламных кампаний."}, {"method": "GET", "path": "/api/client/campaign/{id}", "desc": "ВНИМАНИЕ: возвращает 405, не рабочий — используй список выше."}, {"method": "PUT", "path": "/api/client/campaign/{id}", "desc": "ЗАПИСЬ: изменить еженедельный бюджет кампании.", "note": "write"}, {"method": "GET", "path": "/api/client/statistics/daily", "desc": "ОСНОВНОЙ: CSV статистика по дням/кампаниям. Параметры dateFrom/dateTo (YYYY-MM-DD)."}, {"method": "GET", "path": "/api/client/statistics/expense", "desc": "CSV, детализация расходов (бонусы/абонентский счёт)."}, {"method": "POST", "path": "/api/client/statistics", "desc": "JSON статистика по campaignIds (массив). Медленнее, расходует квоты."}]}}'''
REGISTRY: dict[str, dict[str, Any]] = _json.loads(_REGISTRY_JSON)

SELLER_CATEGORIES = [c for c in REGISTRY if c != "ads"]
ENDPOINTS_BY_CATEGORY: dict[str, dict[str, dict]] = {
    cat: {e["path"]: e for e in data["endpoints"]} for cat, data in REGISTRY.items()
}


def _env(name: str) -> Optional[str]:
    return os.environ.get(name)


def _seller_headers() -> dict[str, str]:
    api_key, client_id = _env("OZON_API_KEY"), _env("OZON_CLIENT_ID")
    if not api_key or not client_id:
        raise RuntimeError("OZON_API_KEY / OZON_CLIENT_ID not set (Seller personal account -> Settings -> API keys)")
    return {"Client-Id": client_id, "Api-Key": api_key, "Content-Type": "application/json"}


_perf_token: dict[str, Any] = {"value": None, "exp": 0}


def _performance_token() -> str:
    if _perf_token["value"] and time.time() < _perf_token["exp"] - 60:
        return _perf_token["value"]
    cid, secret = _env("OZON_PERFORMANCE_CLIENT_ID"), _env("OZON_PERFORMANCE_CLIENT_SECRET")
    if not cid or not secret:
        raise RuntimeError("OZON_PERFORMANCE_CLIENT_ID / OZON_PERFORMANCE_CLIENT_SECRET not set (Реклама -> API)")
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


def _call_seller(method: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None) -> Any:
    headers = _seller_headers()
    with httpx.Client(timeout=60) as client:
        r = client.request(method, f"{SELLER_BASE_URL}{path}", headers=headers, json=body, params=params)
    if not r.is_success:
        return {"error": f"{r.status_code}: {r.text}"}
    try:
        return r.json()
    except ValueError:
        return {"raw": r.text}


def _call_performance(method: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None) -> Any:
    headers = {"Authorization": f"Bearer {_performance_token()}", "Content-Type": "application/json"}
    with httpx.Client(timeout=60) as client:
        r = client.request(method, f"{PERFORMANCE_BASE_URL}{path}", headers=headers, json=body, params=params)
    if not r.is_success:
        return {"error": f"{r.status_code}: {r.text}"}
    ctype = r.headers.get("content-type", "")
    if "application/json" in ctype:
        return r.json()
    return {"raw": r.text}  # CSV endpoints (statistics/daily, statistics/expense)


server = MCPServer(name="ozon", version="0.1.0")


@server.tool()
def ozon_list_endpoints(category: Optional[str] = None) -> dict:
    """List known Ozon API endpoints by category (discover before calling).

    Args:
        category: one category key to filter, or omit to list all categories
            with their endpoint counts.

    Categories: ads, analytics, carriage, delivery, digital_postings, documents, fbo_postings, fbp, fbs_postings, finance, orders, polygon, postings_misc, products, rating, returns, reviews, seller_info
    (Seller API categories call ozon_seller_call; "ads" calls ozon_performance_call.)
    """
    if category:
        data = REGISTRY.get(category)
        if not data:
            return {"error": f"unknown category {category!r}", "categories": sorted(REGISTRY)}
        return {"category": category, "label": data["label"], "endpoints": data["endpoints"]}
    return {
        "categories": {
            cat: {"label": data["label"], "endpoint_count": len(data["endpoints"])}
            for cat, data in REGISTRY.items()
        }
    }


@server.tool()
def ozon_seller_call(category: str, endpoint: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Call a Seller API endpoint (api-seller.ozon.ru).

    Args:
        category: one of: analytics, carriage, delivery, digital_postings, documents, fbo_postings, fbp, fbs_postings, finance, orders, polygon, postings_misc, products, rating, returns, reviews, seller_info.
            Call ozon_list_endpoints(category) first if unsure which path/body shape to use.
        endpoint: exact path from that category's list, e.g. "/v3/product/list".
            Only whitelisted, documented paths are accepted -- an unknown path
            is rejected rather than guessed.
        body: JSON request body per the endpoint's documented shape (most
            Seller API methods are POST with a JSON body; a documented empty
            body is {}).
        params: query-string params, for the few GET endpoints that use them.

    WRITE endpoints (product import, review status change, etc. -- marked
    "note": "write" in ozon_list_endpoints) change live data: show the
    current/proposed value to the user and wait for explicit confirmation
    before calling one, per the shop's standing rule.
    """
    if category not in ENDPOINTS_BY_CATEGORY or category == "ads":
        return {"error": f"unknown Seller API category {category!r}", "categories": SELLER_CATEGORIES}
    known = ENDPOINTS_BY_CATEGORY[category]
    if endpoint not in known:
        return {"error": f"unknown endpoint {endpoint!r} in category {category!r}",
                "known_endpoints": sorted(known)}
    method = known[endpoint]["method"]
    try:
        return _call_seller(method, endpoint, body=body, params=params)
    except RuntimeError as exc:
        return {"error": str(exc)}


@server.tool()
def ozon_performance_call(endpoint: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Call a Performance API endpoint (api-performance.ozon.ru) -- Ozon ads.

    Args:
        endpoint: exact path from ozon_list_endpoints("ads"), e.g.
            "/api/client/statistics/daily". For the templated path
            "/api/client/campaign/{id}", pass the real id filled in, e.g.
            "/api/client/campaign/12345".
        body: JSON body for POST/PUT endpoints (e.g. {"weeklyBudget": ...}).
        params: query params for GET endpoints (e.g. {"dateFrom": ..., "dateTo": ...}).

    WRITE: PUT /api/client/campaign/{id} changes a campaign's weekly budget --
    show the current budget (from GET /api/client/campaign) and the proposed
    new value, and wait for explicit confirmation before calling it.

    Auth (OAuth2 client_credentials) is handled transparently from
    OZON_PERFORMANCE_CLIENT_ID / OZON_PERFORMANCE_CLIENT_SECRET; the token is
    cached in-process for ~30 minutes.
    """
    known = ENDPOINTS_BY_CATEGORY["ads"]
    if endpoint in known:
        method = known[endpoint]["method"]
    else:
        templated = [k for k in known if "{id}" in k and endpoint.startswith(k.split("{id}")[0])]
        if not templated:
            return {"error": f"unknown endpoint {endpoint!r}", "known_endpoints": sorted(known)}
        method = known[templated[0]]["method"]
    try:
        return _call_performance(method, endpoint, body=body, params=params)
    except RuntimeError as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
