#!/usr/bin/env python3
"""Square API CLI for Hermes Agent.

Wraps the Square Python SDK with a convenient argparse interface.

Usage:
  python square_api.py inventory counts [--location LOC] [--catalog-object-id ID]
  python square_api.py inventory adjust --catalog-object-id ID --location LOC --quantity N --reason TEXT
  python square_api.py inventory changes --location LOC [--start-time DATETIME] [--end-time DATETIME]
  python square_api.py catalog list --types "item,variation"
  python square_api.py catalog search --query QUERY [--types TYPES]
  python square_api.py catalog get OBJECT_ID
  python square_api.py customers list [--max N]
  python square_api.py customers search --query QUERY
  python square_api.py customers create --given-name NAME [--family-name NAME] [--email EMAIL] [--phone PHONE]
  python square_api.py customers update CUSTOMER_ID [--phone PHONE] [--email EMAIL]
  python square_api.py customers get CUSTOMER_ID
  python square_api.py orders list --location LOC [--start-time DATETIME] [--end-time DATETIME]
  python square_api.py orders get ORDER_ID
  python square_api.py locations list
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid

from square_auth import SquareAuthError, get_valid_access_token

API_BASE = "https://connect.squareup.com/v2"
API_VERSION = "2026-01-22"
REQUEST_TIMEOUT_SECONDS = 30


class SquareAPIError(RuntimeError):
    """Raised when Square returns an unsuccessful API response."""


def _get_client():
    """Build a Square SDK client after refreshing credentials if needed."""
    from square.client import Client
    from square.http.auth.o_auth_2 import BearerAuthCredentials

    return Client(
        bearer_auth_credentials=BearerAuthCredentials(get_valid_access_token()),
        square_version=API_VERSION,
    )


def _decode_http_error(error: urllib.error.HTTPError) -> object:
    raw_body = error.read()
    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body.decode("utf-8", errors="replace")


def _api_request(method: str, path: str, body: dict | None = None) -> dict:
    """Make a direct REST API call. Used when SDK coverage is insufficient."""
    url = f"{API_BASE}/{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None

    for attempt in range(2):
        access_token = get_valid_access_token(force_refresh=attempt == 1)
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Square-Version": API_VERSION,
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 401 and attempt == 0:
                continue
            raise SquareAPIError(f"HTTP {exc.code}: {_decode_http_error(exc)}") from exc
        except urllib.error.URLError as exc:
            raise SquareAPIError(f"Request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SquareAPIError("Square API returned invalid JSON") from exc

    raise SquareAPIError("Authentication failed after refreshing the Square token")


def _result_body(result) -> dict:
    """Validate one Square SDK response and return its JSON body."""
    if callable(getattr(result, "is_error", None)) and result.is_error():
        detail = getattr(result, "errors", None) or getattr(result, "body", None)
        status = getattr(result, "status_code", "unknown")
        raise SquareAPIError(f"SDK request failed with status {status}: {detail}")
    body = getattr(result, "body", None)
    if not isinstance(body, dict):
        raise SquareAPIError("Square SDK returned an invalid response body")
    return body


def _paginate_sdk(call, item_key: str, *, max_items: int | None = None, **params) -> dict:
    """Consume cursor-based Square SDK responses into one result object."""
    items = []
    output = {}
    cursor = None

    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        body = _result_body(call(**page_params))
        if not output:
            output = {key: value for key, value in body.items() if key != "cursor"}
        items.extend(body.get(item_key) or [])
        if max_items is not None and len(items) >= max_items:
            items = items[:max_items]
            break
        cursor = body.get("cursor")
        if not cursor:
            break

    output[item_key] = items
    return output


def _body_cursor_call(method, body: dict):
    """Adapt SDK methods that accept a JSON body to cursor pagination."""
    def call(*, cursor=None):
        page_body = dict(body)
        if cursor:
            page_body["cursor"] = cursor
        return method(page_body)

    return call


# -- Inventory --

def cmd_inventory_counts(args):
    client = _get_client()
    params = {}
    if args.catalog_object_id:
        params["catalog_object_ids"] = [args.catalog_object_id]
    if args.location:
        params["location_ids"] = [args.location]
    if args.start_time:
        params["updated_after"] = args.start_time

    result = _paginate_sdk(
        _body_cursor_call(client.inventory.batch_retrieve_inventory_counts, params),
        "counts",
    )
    print(json.dumps(result, indent=2))


def cmd_inventory_adjust(args):
    body = {
        "idempotency_key": args.idempotency_key or str(uuid.uuid4()),
        "changes": [
            {
                "type": "ADJUSTMENT",
                "physical_count": None,
                "adjustment": {
                    "catalog_object_id": args.catalog_object_id,
                    "location_id": args.location,
                    "quantity": str(args.quantity),
                    "reason": args.reason or "UNKNOWN",
                },
            }
        ],
    }
    result = _api_request("POST", "inventory/changes/batch-create", body)
    print(json.dumps(result, indent=2))


def cmd_inventory_changes(args):
    params = {}
    if args.location:
        params["location_ids"] = [args.location]
    if args.start_time:
        params["updated_after"] = args.start_time
    if args.end_time:
        params["updated_before"] = args.end_time

    client = _get_client()
    result = _paginate_sdk(
        _body_cursor_call(client.inventory.batch_retrieve_inventory_changes, params),
        "changes",
    )
    print(json.dumps(result, indent=2))


# -- Catalog --

def cmd_catalog_list(args):
    client = _get_client()
    types = args.types.upper() if args.types else None
    result = _paginate_sdk(client.catalog.list_catalog, "objects", types=types)
    print(json.dumps(result, indent=2))


def cmd_catalog_search(args):
    client = _get_client()
    body = {"query": {"text_query": {"keywords": [args.query]}}}
    if args.types:
        body["object_types"] = [value.upper() for value in args.types.split(",")]
    result = _paginate_sdk(
        _body_cursor_call(client.catalog.search_catalog_objects, body),
        "objects",
    )
    print(json.dumps(result, indent=2))


def cmd_catalog_get(args):
    client = _get_client()
    result = client.catalog.retrieve_catalog_object(args.object_id, include_related_objects=True)
    print(json.dumps(_result_body(result), indent=2))


# -- Customers --

def cmd_customers_list(args):
    client = _get_client()
    maximum = args.max or 50
    result = _paginate_sdk(
        client.customers.list_customers,
        "customers",
        max_items=maximum,
        limit=min(maximum, 100),
    )
    print(json.dumps(result, indent=2))


def cmd_customers_search(args):
    client = _get_client()
    body = {
        "query": {
            "text_query": {
                "attribute_name": "text",
                "text": args.query,
            }
        },
        "limit": args.max or 50,
    }
    result = _paginate_sdk(
        _body_cursor_call(client.customers.search_customers, body),
        "customers",
        max_items=args.max or 50,
    )
    print(json.dumps(result, indent=2))


def cmd_customers_create(args):
    client = _get_client()
    import uuid
    body = {
        "idempotency_key": str(uuid.uuid4()),
        "given_name": args.given_name,
    }
    if args.family_name:
        body["family_name"] = args.family_name
    if args.email:
        body["email_address"] = args.email
    if args.phone:
        body["phone_number"] = args.phone

    result = client.customers.create_customer(body)
    print(json.dumps(_result_body(result), indent=2))


def cmd_customers_update(args):
    client = _get_client()
    body = {}
    if args.email:
        body["email_address"] = args.email
    if args.phone:
        body["phone_number"] = args.phone
    if args.given_name:
        body["given_name"] = args.given_name
    if args.family_name:
        body["family_name"] = args.family_name

    result = client.customers.update_customer(args.customer_id, body)
    print(json.dumps(_result_body(result), indent=2))


def cmd_customers_get(args):
    client = _get_client()
    result = client.customers.retrieve_customer(args.customer_id)
    print(json.dumps(_result_body(result), indent=2))


# -- Orders --

def cmd_orders_list(args):
    client = _get_client()
    from datetime import datetime, timedelta, timezone as tz
    now = datetime.now(tz.utc)
    start_time = args.start_time or (now - timedelta(days=7)).isoformat()
    end_time = args.end_time or now.isoformat()

    body = {
        "location_ids": [args.location],
        "query": {
            "filter": {
                "date_time_filter": {
                    "created_at": {
                        "start_at": start_time,
                        "end_at": end_time,
                    }
                }
            }
        },
    }
    result = _paginate_sdk(
        _body_cursor_call(client.orders.search_orders, body),
        "orders",
    )
    print(json.dumps(result, indent=2))


def cmd_orders_get(args):
    client = _get_client()
    result = client.orders.retrieve_order(args.order_id)
    print(json.dumps(_result_body(result), indent=2))


# -- Locations --

def cmd_locations_list(args):
    client = _get_client()
    result = client.locations.list_locations()
    print(json.dumps(_result_body(result), indent=2))


# -- CLI parser --

def main():
    parser = argparse.ArgumentParser(description="Square API for Hermes Agent")
    sub = parser.add_subparsers(dest="service", required=True)

    # --- Inventory ---
    inv = sub.add_parser("inventory")
    inv_sub = inv.add_subparsers(dest="action", required=True)

    p = inv_sub.add_parser("counts")
    p.add_argument("--location", default="", help="Location ID")
    p.add_argument("--catalog-object-id", default="", help="Catalog object ID")
    p.add_argument("--start-time", default="", help="Only counts updated after this ISO 8601 time")
    p.set_defaults(func=cmd_inventory_counts)

    p = inv_sub.add_parser("adjust")
    p.add_argument("--catalog-object-id", required=True, help="Catalog object ID")
    p.add_argument("--location", required=True, help="Location ID")
    p.add_argument("--quantity", type=int, required=True, help="Quantity to adjust (positive or negative)")
    p.add_argument("--reason", default="", help="Reason for adjustment")
    p.add_argument(
        "--idempotency-key",
        "--retry-key",
        dest="idempotency_key",
        default="",
        help="Reuse only when retrying the exact same adjustment",
    )
    p.set_defaults(func=cmd_inventory_adjust)

    p = inv_sub.add_parser("changes")
    p.add_argument("--location", default="", help="Location ID")
    p.add_argument("--start-time", default="", help="ISO 8601 start time")
    p.add_argument("--end-time", default="", help="ISO 8601 end time")
    p.set_defaults(func=cmd_inventory_changes)

    # --- Catalog ---
    cat = sub.add_parser("catalog")
    cat_sub = cat.add_subparsers(dest="action", required=True)

    p = cat_sub.add_parser("list")
    p.add_argument("--types", default="", help="Comma-separated types (item,variation,category,etc.)")
    p.set_defaults(func=cmd_catalog_list)

    p = cat_sub.add_parser("search")
    p.add_argument("--query", required=True, help="Search query")
    p.add_argument("--types", default="", help="Comma-separated object types")
    p.set_defaults(func=cmd_catalog_search)

    p = cat_sub.add_parser("get")
    p.add_argument("object_id", help="Catalog object ID")
    p.set_defaults(func=cmd_catalog_get)

    # --- Customers ---
    cust = sub.add_parser("customers")
    cust_sub = cust.add_subparsers(dest="action", required=True)

    p = cust_sub.add_parser("list")
    p.add_argument("--max", type=int, default=50)
    p.set_defaults(func=cmd_customers_list)

    p = cust_sub.add_parser("search")
    p.add_argument("--query", required=True, help="Search query")
    p.add_argument("--max", type=int, default=50)
    p.set_defaults(func=cmd_customers_search)

    p = cust_sub.add_parser("create")
    p.add_argument("--given-name", required=True)
    p.add_argument("--family-name", default="")
    p.add_argument("--email", default="")
    p.add_argument("--phone", default="")
    p.set_defaults(func=cmd_customers_create)

    p = cust_sub.add_parser("update")
    p.add_argument("customer_id", help="Customer ID")
    p.add_argument("--given-name", default="")
    p.add_argument("--family-name", default="")
    p.add_argument("--email", default="")
    p.add_argument("--phone", default="")
    p.set_defaults(func=cmd_customers_update)

    p = cust_sub.add_parser("get")
    p.add_argument("customer_id", help="Customer ID")
    p.set_defaults(func=cmd_customers_get)

    # --- Orders ---
    ord_ = sub.add_parser("orders")
    ord_sub = ord_.add_subparsers(dest="action", required=True)

    p = ord_sub.add_parser("list")
    p.add_argument("--location", required=True, help="Location ID")
    p.add_argument("--start-time", default="", help="ISO 8601 start time")
    p.add_argument("--end-time", default="", help="ISO 8601 end time")
    p.set_defaults(func=cmd_orders_list)

    p = ord_sub.add_parser("get")
    p.add_argument("order_id", help="Order ID")
    p.set_defaults(func=cmd_orders_get)

    # --- Locations ---
    loc = sub.add_parser("locations")
    loc_sub = loc.add_subparsers(dest="action", required=True)

    p = loc_sub.add_parser("list")
    p.set_defaults(func=cmd_locations_list)

    args = parser.parse_args()
    try:
        args.func(args)
    except (SquareAuthError, SquareAPIError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
