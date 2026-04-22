from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from market_monitor.dashboard import create_dashboard_payload
from market_monitor.db import Database
from market_monitor.queries import get_brand_ranking, get_metric_series
from market_monitor.runners import render_structured_results


class BadRequestError(ValueError):
    pass


def build_api_payloads(db: Database) -> dict:
    dashboard = create_dashboard_payload(db)
    latest_period = dashboard["periods"][0] if dashboard["periods"] else None
    return {
        "periods": {
            "schema_version": dashboard["schema_version"],
            "generated_at": dashboard["generated_at"],
            "periods": dashboard["periods"],
        },
        "results": {period: dashboard["results"][period] for period in dashboard["periods"]},
        "series": get_metric_series(db, metric_name="sales_volume", metric_scope="retail", energy_type="nev_total") if latest_period else [],
        "brand_rankings": get_brand_ranking(db, period_label=latest_period, top_n=10) if latest_period else [],
        "health": {
            "status": "ok",
            "schema_version": dashboard["schema_version"],
            "period_count": len(dashboard["periods"]),
            "generated_at": dashboard["generated_at"],
        },
    }


def start_api_server(*, db: Database, host: str = "127.0.0.1", port: int = 8780) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            try:
                dashboard = create_dashboard_payload(db)
                if path == "/periods":
                    payload = {
                        "schema_version": dashboard["schema_version"],
                        "generated_at": dashboard["generated_at"],
                        "periods": dashboard["periods"],
                    }
                elif path.startswith("/results/"):
                    period_label = path.split("/results/", 1)[1]
                    payload = render_structured_results(db, period_label=period_label)
                elif path == "/series":
                    metric_name = _required_query_value(query, "metric_name")
                    metric_scope = _required_query_value(query, "metric_scope")
                    payload = get_metric_series(
                        db,
                        metric_name=metric_name,
                        metric_scope=metric_scope,
                        energy_type=query.get("energy_type", [None])[0],
                    )
                elif path == "/rankings/brands":
                    period_label = _required_query_value(query, "period_label")
                    top_n = _parse_positive_int(query, "top_n", default=20)
                    payload = get_brand_ranking(db, period_label=period_label, top_n=top_n)
                elif path == "/health":
                    payload = {
                        "status": "ok",
                        "schema_version": dashboard["schema_version"],
                        "period_count": len(dashboard["periods"]),
                        "generated_at": dashboard["generated_at"],
                    }
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self._send_json(200, payload)
            except BadRequestError as exc:
                self._send_json(400, {"status": "error", "error": str(exc)})
            except Exception:
                self._send_json(500, {"status": "error", "error": "internal_error"})

        def _send_json(self, status_code: int, payload: dict | list) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    return ThreadingHTTPServer((host, port), Handler)


def _required_query_value(query: dict[str, list[str]], name: str) -> str:
    value = query.get(name, [""])[0].strip()
    if not value:
        raise BadRequestError(f"missing_{name}")
    return value


def _parse_positive_int(query: dict[str, list[str]], name: str, *, default: int) -> int:
    raw_value = query.get(name, [str(default)])[0]
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise BadRequestError(f"invalid_{name}") from exc
    if value <= 0:
        raise BadRequestError(f"invalid_{name}")
    return value
