from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

DEFAULT_BASE_URL = "https://invoicing.co"
DEFAULT_TIMEOUT = 30.0

# Invoice Ninja's API docs expose InvoiceRequest as the editable invoice shape.
INVOICE_REQUEST_FIELDS = {
    "user_id",
    "assigned_user_id",
    "client_id",
    "location_id",
    "number",
    "po_number",
    "terms",
    "public_notes",
    "private_notes",
    "footer",
    "custom_value1",
    "custom_value2",
    "custom_value3",
    "custom_value4",
    "tax_name1",
    "tax_name2",
    "tax_rate1",
    "tax_rate2",
    "tax_name3",
    "tax_rate3",
    "line_items",
    "invitations",
    "discount",
    "partial",
    "is_amount_discount",
    "uses_inclusive_taxes",
    "date",
    "partial_due_date",
    "due_date",
    "custom_surcharge1",
    "custom_surcharge2",
    "custom_surcharge3",
    "custom_surcharge4",
    "custom_surcharge_tax1",
    "custom_surcharge_tax2",
    "custom_surcharge_tax3",
    "custom_surcharge_tax4",
    "project_id",
}

CLIENT_REQUEST_FIELDS = {
    "id",
    "contacts",
    "name",
    "website",
    "private_notes",
    "industry_id",
    "size_id",
    "address1",
    "address2",
    "city",
    "state",
    "postal_code",
    "phone",
    "country_id",
    "custom_value1",
    "custom_value2",
    "custom_value3",
    "custom_value4",
    "vat_number",
    "id_number",
    "number",
    "shipping_address1",
    "shipping_address2",
    "shipping_city",
    "shipping_state",
    "shipping_postal_code",
    "shipping_country_id",
    "is_deleted",
    "group_settings_id",
    "routing_id",
    "is_tax_exempt",
    "has_valid_vat_number",
    "classification",
    "settings",
}

PAYMENT_REQUEST_FIELDS = {
    "client_id",
    "client_contact_id",
    "user_id",
    "type_id",
    "date",
    "transaction_reference",
    "assigned_user_id",
    "private_notes",
    "amount",
    "invoices",
    "credits",
    "number",
}

EXPENSE_REQUEST_FIELDS = {
    "id",
    "user_id",
    "assigned_user_id",
    "project_id",
    "client_id",
    "invoice_id",
    "bank_id",
    "invoice_currency_id",
    "currency_id",
    "invoice_category_id",
    "payment_type_id",
    "recurring_expense_id",
    "private_notes",
    "public_notes",
    "transaction_reference",
    "transcation_id",
    "custom_value1",
    "custom_value2",
    "custom_value3",
    "custom_value4",
    "tax_amount",
    "tax_name1",
    "tax_name2",
    "tax_name3",
    "tax_rate1",
    "tax_rate2",
    "tax_rate3",
    "amount",
    "foreign_amount",
    "exchange_rate",
    "date",
    "payment_date",
    "should_be_invoiced",
    "is_deleted",
    "category_id",
    "number",
    "purchase_order_id",
    "tax_amount1",
    "tax_amount2",
    "tax_amount3",
    "transaction_id",
    "uses_inclusive_taxes",
    "vendor_id",
}


@dataclass(slots=True)
class InvoiceNinjaConfig:
    base_url: str = DEFAULT_BASE_URL
    api_token: str = ""
    timeout: float = DEFAULT_TIMEOUT
    verify_ssl: bool = True


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(item) for item in value]
    return value


def _filter_keys(payload: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key in allowed and value is not None}


def _merge_payload(base: dict[str, Any], updates: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    payload = _filter_keys(base, allowed)
    payload.update(_filter_keys(updates, allowed))
    return _strip_none(payload)


class InvoiceNinjaClient:
    def __init__(self, config: InvoiceNinjaConfig, client: httpx.Client | None = None):
        if not config.api_token:
            raise RuntimeError("INVOICE_NINJA_API_TOKEN is required")
        self.config = config
        self._client = client or httpx.Client(
            timeout=config.timeout,
            verify=config.verify_ssl,
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _build_url(self, path: str) -> str:
        base = self.config.base_url.rstrip("/")
        raw_path = path.strip()
        if not raw_path:
            raise ValueError("path is required")

        if raw_path.startswith("http://") or raw_path.startswith("https://"):
            return raw_path

        normalized = raw_path.lstrip("/")
        if not normalized.startswith("api/v1/"):
            if normalized == "api/v1":
                normalized = "api/v1/"
            else:
                normalized = f"api/v1/{normalized}"
        return f"{base}/{normalized}"

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._build_url(path)
        request_kwargs: dict[str, Any] = {
            "params": _strip_none(params or {}),
            "headers": {
                "X-API-TOKEN": self.config.api_token,
                "Accept": "application/json",
            },
        }
        if body is not None:
            request_kwargs["json"] = _strip_none(body)

        try:
            response = self._client.request(method.upper(), url, **request_kwargs)
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "method": method.upper(),
                "url": url,
                "error": f"Network error: {exc}",
            }

        return self._format_response(response, method.upper(), url)

    def _format_response(self, response: httpx.Response, method: str, url: str) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            data = response.text

        result: dict[str, Any] = {
            "ok": response.is_success,
            "method": method,
            "url": str(response.request.url if response.request is not None else url),
            "status_code": response.status_code,
            "data": data,
        }
        if not response.is_success:
            result["error"] = self._extract_error_message(data, response.text)
        return result

    @staticmethod
    def _extract_error_message(data: Any, fallback: str) -> str:
        if isinstance(data, dict):
            for key in ("message", "error", "errors", "detail"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
                if isinstance(value, list) and value:
                    return "; ".join(str(item) for item in value)
                if isinstance(value, dict) and value:
                    return "; ".join(f"{k}: {v}" for k, v in value.items())
        if fallback.strip():
            return fallback
        return "Unknown Invoice Ninja API error"

    def list_invoices(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", "/invoices", params=params or {})

    def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        return self.request("GET", f"/invoices/{invoice_id}")

    def create_invoice(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/invoices", body=payload)

    def update_invoice(self, invoice_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("PUT", f"/invoices/{invoice_id}", body=payload)

    def append_invoice_line_item(self, invoice_id: str, line_item: dict[str, Any]) -> dict[str, Any]:
        invoice_result = self.get_invoice(invoice_id)
        if not invoice_result.get("ok"):
            return invoice_result

        invoice_data = invoice_result.get("data")
        if not isinstance(invoice_data, dict):
            return {
                "ok": False,
                "error": "Invoice response did not contain an object payload.",
                "invoice_id": invoice_id,
            }

        line_items = list(invoice_data.get("line_items") or [])
        line_items.append(_strip_none(line_item))

        payload = _merge_payload(invoice_data, {"line_items": line_items}, INVOICE_REQUEST_FIELDS)
        return self.update_invoice(invoice_id, payload)

    def list_clients(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", "/clients", params=params or {})

    def create_client(self, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = _filter_keys(payload, CLIENT_REQUEST_FIELDS)
        return self.request("POST", "/clients", body=cleaned)

    def list_payments(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", "/payments", params=params or {})

    def create_payment(self, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = _filter_keys(payload, PAYMENT_REQUEST_FIELDS)
        return self.request("POST", "/payments", body=cleaned)

    def list_expenses(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", "/expenses", params=params or {})

    def get_expense(self, expense_id: str) -> dict[str, Any]:
        return self.request("GET", f"/expenses/{expense_id}")

    def create_expense(self, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = _filter_keys(payload, EXPENSE_REQUEST_FIELDS)
        return self.request("POST", "/expenses", body=cleaned)

    def update_expense(self, expense_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = _filter_keys(payload, EXPENSE_REQUEST_FIELDS)
        return self.request("PUT", f"/expenses/{expense_id}", body=cleaned)

    def delete_expense(self, expense_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/expenses/{expense_id}")

    def create_expense_from_receipt(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "vendor_id": payload.get("vendor_id"),
            "client_id": payload.get("client_id"),
            "project_id": payload.get("project_id"),
            "category_id": payload.get("category_id"),
            "amount": payload.get("amount"),
            "date": payload.get("date"),
            "payment_date": payload.get("payment_date"),
            "currency_id": payload.get("currency_id"),
            "invoice_currency_id": payload.get("invoice_currency_id"),
            "bank_id": payload.get("bank_id"),
            "invoice_id": payload.get("invoice_id"),
            "payment_type_id": payload.get("payment_type_id"),
            "transaction_reference": payload.get("receipt_number") or payload.get("transaction_reference"),
            "private_notes": payload.get("private_notes") or payload.get("notes"),
            "public_notes": payload.get("public_notes"),
            "should_be_invoiced": payload.get("should_be_invoiced"),
            "tax_amount": payload.get("tax_amount"),
            "tax_name1": payload.get("tax_name1"),
            "tax_name2": payload.get("tax_name2"),
            "tax_name3": payload.get("tax_name3"),
            "tax_rate1": payload.get("tax_rate1"),
            "tax_rate2": payload.get("tax_rate2"),
            "tax_rate3": payload.get("tax_rate3"),
            "custom_value1": payload.get("custom_value1"),
            "custom_value2": payload.get("custom_value2"),
            "custom_value3": payload.get("custom_value3"),
            "custom_value4": payload.get("custom_value4"),
            "number": payload.get("number"),
            "purchase_order_id": payload.get("purchase_order_id"),
            "uses_inclusive_taxes": payload.get("uses_inclusive_taxes"),
        }
        return self.create_expense(normalized)

    def attach_expense(self, expense_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_expense(expense_id)
        if not current.get("ok"):
            return current

        data = current.get("data")
        if not isinstance(data, dict):
            return {"ok": False, "error": "Expense response did not contain an object payload.", "expense_id": expense_id}

        merged = _merge_payload(data, payload, EXPENSE_REQUEST_FIELDS)
        return self.update_expense(expense_id, merged)

    def list_unpaid_or_unbilled_expenses(
        self,
        *,
        client_id: str | None = None,
        project_id: str | None = None,
        category_id: str | None = None,
        should_be_invoiced: bool | None = None,
        query: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        result = self.list_expenses()
        if not result.get("ok"):
            return result

        data = result.get("data")
        if isinstance(data, dict):
            items = data.get("data") or data.get("items") or data.get("results") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []

        def truthy(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if v is None:
                return False
            if isinstance(v, (int, float)):
                return bool(v)
            return str(v).strip().lower() not in {"", "0", "false", "no", "off", "null", "none"}

        filtered = []
        q = query.lower().strip() if query else None
        for item in items:
            if not isinstance(item, dict):
                continue
            if client_id and str(item.get("client_id") or "") != client_id:
                continue
            if project_id and str(item.get("project_id") or "") != project_id:
                continue
            if category_id and str(item.get("category_id") or item.get("invoice_category_id") or "") != category_id:
                continue
            if should_be_invoiced is not None and truthy(item.get("should_be_invoiced")) != should_be_invoiced:
                continue
            if q:
                haystack = " ".join(
                    str(item.get(field) or "") for field in ("number", "private_notes", "public_notes", "transaction_reference", "vendor_id")
                ).lower()
                if q not in haystack:
                    continue
            invoice_id = str(item.get("invoice_id") or "")
            payment_date = str(item.get("payment_date") or "")
            if invoice_id and payment_date:
                continue
            filtered.append(item)
            if limit is not None and len(filtered) >= limit:
                break

        return {
            "ok": True,
            "method": "GET",
            "url": f"{self.config.base_url.rstrip('/')}/api/v1/expenses",
            "status_code": 200,
            "data": filtered,
            "count": len(filtered),
            "source_count": len(items),
            "filters": {
                "client_id": client_id,
                "project_id": project_id,
                "category_id": category_id,
                "should_be_invoiced": should_be_invoiced,
                "query": query,
                "limit": limit,
            },
        }

    def list_expense_categories(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", "/expense_categories", params=params or {})

    def create_expense_category(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/expense_categories", body=payload)

    def update_expense_category(self, category_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("PUT", f"/expense_categories/{category_id}", body=payload)

    def delete_expense_category(self, category_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/expense_categories/{category_id}")

    def list_recurring_expenses(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", "/recurring_expenses", params=params or {})

    def create_recurring_expense(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/recurring_expenses", body=payload)

    def update_recurring_expense(self, recurring_expense_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("PUT", f"/recurring_expenses/{recurring_expense_id}", body=payload)

    def delete_recurring_expense(self, recurring_expense_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/recurring_expenses/{recurring_expense_id}")

    def run_expense_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/reports/expenses", body=payload)

    def record_payment(
        self,
        invoice_id: str,
        amount: float,
        payment_type_id: str = "1",
        transaction_reference: str | None = None,
        date: str | None = None,
        private_notes: str | None = None,
        client_id: str | None = None,
        client_contact_id: str | None = None,
        assigned_user_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_client_id = client_id
        if not resolved_client_id:
            invoice_result = self.get_invoice(invoice_id)
            if not invoice_result.get("ok"):
                return invoice_result
            invoice_data = invoice_result.get("data")
            if isinstance(invoice_data, dict):
                resolved_client_id = str(invoice_data.get("client_id") or "") or None
            if not resolved_client_id:
                return {
                    "ok": False,
                    "error": "Could not infer client_id from the invoice and no explicit client_id was provided.",
                    "invoice_id": invoice_id,
                }

        payload: dict[str, Any] = {
            "client_id": resolved_client_id,
            "client_contact_id": client_contact_id,
            "user_id": user_id,
            "type_id": payment_type_id,
            "date": date,
            "transaction_reference": transaction_reference,
            "assigned_user_id": assigned_user_id,
            "private_notes": private_notes,
            "amount": amount,
            "invoices": [{"invoice_id": invoice_id, "amount": str(amount)}],
        }
        return self.create_payment(payload)


def _build_config() -> InvoiceNinjaConfig:
    token = os.getenv("INVOICE_NINJA_API_TOKEN", "").strip()
    base_url = os.getenv("INVOICE_NINJA_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    timeout = float(os.getenv("INVOICE_NINJA_TIMEOUT", str(DEFAULT_TIMEOUT)))
    verify_ssl = _env_bool("INVOICE_NINJA_VERIFY_SSL", True)
    return InvoiceNinjaConfig(base_url=base_url, api_token=token, timeout=timeout, verify_ssl=verify_ssl)


@lru_cache(maxsize=1)
def get_default_client() -> InvoiceNinjaClient:
    return InvoiceNinjaClient(_build_config())


mcp = FastMCP(
    "invoice-ninja",
    instructions=(
        "Use this server to work with Invoice Ninja invoices, clients, and payments. "
        "Authentication uses the X-API-TOKEN header. The base URL defaults to https://invoicing.co, "
        "but you can override it with INVOICE_NINJA_BASE_URL for self-hosted instances."
    ),
)


@mcp.tool()
def invoice_ninja_request(
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make a raw Invoice Ninja API request under /api/v1."""
    return get_default_client().request(method=method, path=path, params=params, body=body)


@mcp.tool()
def invoice_ninja_list_invoices(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """List invoices. Common filters include filter, number, status_id, overdue, payable, sort, date, and date_range."""
    return get_default_client().list_invoices(params=params)


@mcp.tool()
def invoice_ninja_get_invoice(invoice_id: str) -> dict[str, Any]:
    """Fetch a single invoice by its hashed id."""
    return get_default_client().get_invoice(invoice_id)


@mcp.tool()
def invoice_ninja_create_invoice(payload: dict[str, Any]) -> dict[str, Any]:
    """Create an invoice using Invoice Ninja's editable invoice fields."""
    return get_default_client().create_invoice(payload)


@mcp.tool()
def invoice_ninja_update_invoice(invoice_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Update an existing invoice."""
    return get_default_client().update_invoice(invoice_id, payload)


@mcp.tool()
def invoice_ninja_append_invoice_line_item(invoice_id: str, line_item: dict[str, Any]) -> dict[str, Any]:
    """Append one line item to an invoice and save the updated invoice."""
    return get_default_client().append_invoice_line_item(invoice_id, line_item)


@mcp.tool()
def invoice_ninja_list_clients(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """List clients. Common filters include name, email, id_number, number, filter, sort, group, and client_id."""
    return get_default_client().list_clients(params=params)


@mcp.tool()
def invoice_ninja_create_client(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a client. At minimum, Invoice Ninja requires contacts and country_id."""
    return get_default_client().create_client(payload)


@mcp.tool()
def invoice_ninja_list_payments(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """List payments."""
    return get_default_client().list_payments(params=params)


@mcp.tool()
def invoice_ninja_list_expenses(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """List expenses."""
    return get_default_client().list_expenses(params=params)


@mcp.tool()
def invoice_ninja_get_expense(expense_id: str) -> dict[str, Any]:
    """Fetch a single expense by its hashed id."""
    return get_default_client().get_expense(expense_id)


@mcp.tool()
def invoice_ninja_create_expense(payload: dict[str, Any]) -> dict[str, Any]:
    """Create an expense."""
    return get_default_client().create_expense(payload)


@mcp.tool()
def invoice_ninja_create_expense_from_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Create an expense from a receipt-like payload and normalize common fields."""
    return get_default_client().create_expense_from_receipt(payload)


@mcp.tool()
def invoice_ninja_attach_expense(expense_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Attach or update an expense with client/project/category metadata."""
    return get_default_client().attach_expense(expense_id, payload)


@mcp.tool()
def invoice_ninja_list_unpaid_or_unbilled_expenses(
    client_id: str | None = None,
    project_id: str | None = None,
    category_id: str | None = None,
    should_be_invoiced: bool | None = None,
    query: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """List expenses that are unpaid or unbilled, with simple client-side filters."""
    return get_default_client().list_unpaid_or_unbilled_expenses(
        client_id=client_id,
        project_id=project_id,
        category_id=category_id,
        should_be_invoiced=should_be_invoiced,
        query=query,
        limit=limit,
    )


@mcp.tool()
def invoice_ninja_update_expense(expense_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Update an existing expense."""
    return get_default_client().update_expense(expense_id, payload)


@mcp.tool()
def invoice_ninja_delete_expense(expense_id: str) -> dict[str, Any]:
    """Delete an expense."""
    return get_default_client().delete_expense(expense_id)


@mcp.tool()
def invoice_ninja_list_expense_categories(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """List expense categories."""
    return get_default_client().list_expense_categories(params=params)


@mcp.tool()
def invoice_ninja_create_expense_category(payload: dict[str, Any]) -> dict[str, Any]:
    """Create an expense category."""
    return get_default_client().create_expense_category(payload)


@mcp.tool()
def invoice_ninja_update_expense_category(category_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Update an expense category."""
    return get_default_client().update_expense_category(category_id, payload)


@mcp.tool()
def invoice_ninja_delete_expense_category(category_id: str) -> dict[str, Any]:
    """Delete an expense category."""
    return get_default_client().delete_expense_category(category_id)


@mcp.tool()
def invoice_ninja_list_recurring_expenses(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """List recurring expenses."""
    return get_default_client().list_recurring_expenses(params=params)


@mcp.tool()
def invoice_ninja_create_recurring_expense(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a recurring expense."""
    return get_default_client().create_recurring_expense(payload)


@mcp.tool()
def invoice_ninja_update_recurring_expense(recurring_expense_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Update an existing recurring expense."""
    return get_default_client().update_recurring_expense(recurring_expense_id, payload)


@mcp.tool()
def invoice_ninja_delete_recurring_expense(recurring_expense_id: str) -> dict[str, Any]:
    """Delete a recurring expense."""
    return get_default_client().delete_recurring_expense(recurring_expense_id)


@mcp.tool()
def invoice_ninja_run_expense_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the expense report endpoint."""
    return get_default_client().run_expense_report(payload)


@mcp.tool()
def invoice_ninja_record_payment(
    invoice_id: str,
    amount: float,
    payment_type_id: str = "1",
    transaction_reference: str | None = None,
    date: str | None = None,
    private_notes: str | None = None,
    client_id: str | None = None,
    client_contact_id: str | None = None,
    assigned_user_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Record a payment against an invoice."""
    return get_default_client().record_payment(
        invoice_id=invoice_id,
        amount=amount,
        payment_type_id=payment_type_id,
        transaction_reference=transaction_reference,
        date=date,
        private_notes=private_notes,
        client_id=client_id,
        client_contact_id=client_contact_id,
        assigned_user_id=assigned_user_id,
        user_id=user_id,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
