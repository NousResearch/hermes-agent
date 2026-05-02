from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Protocol, TypedDict

try:
    from pytdx.hq import TdxHq_API
except Exception:  # pragma: no cover - optional dependency fallback
    TdxHq_API = None


class QuoteProvider(Protocol):
    def get(self, symbol: str) -> dict:
        ...


class RealtimeSource(Protocol):
    def get(self, symbol: str) -> QuoteSnapshot | dict:
        ...


class TechDataProvider(Protocol):
    def get(self, symbol: str) -> dict:
        ...


class QuoteSnapshot(TypedDict, total=False):
    symbol: str
    last_price: float
    source: str
    as_of: str
    tech_data: dict


logger = logging.getLogger(__name__)


def _deepcopy_dict(payload: dict) -> dict:
    return deepcopy(payload or {})


def _default_tech_data_payload(default_tech_data: dict | None = None) -> dict:
    return _deepcopy_dict(default_tech_data or {"summary_signal": "hold", "score": {"total": 50}})


def _normalize_quote_snapshot(symbol: str, payload: dict | None) -> QuoteSnapshot | dict:
    raw = dict(payload or {})
    if not raw:
        return {}

    normalized_symbol = str(raw.get("symbol") or symbol)
    snapshot: QuoteSnapshot = {"symbol": normalized_symbol}
    if raw.get("last_price") is not None:
        snapshot["last_price"] = raw["last_price"]
    if raw.get("source") is not None:
        snapshot["source"] = str(raw["source"])
    if raw.get("as_of") is not None:
        snapshot["as_of"] = str(raw["as_of"])
    if raw.get("tech_data") is not None:
        snapshot["tech_data"] = _deepcopy_dict(raw["tech_data"] or {})
    return snapshot


def _load_json_file(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


_DEFAULT_TDX_SERVERS: list[tuple[str, str, int]] = [
    ("广发深圳", "183.60.224.178", 7709),
    ("招商深圳云", "39.108.28.83", 7709),
    ("招商武汉电信", "119.97.185.5", 7709),
    ("华泰南京", "221.231.141.60", 7709),
    ("银河北京", "218.108.98.244", 7709),
]


def _infer_tdx_market(symbol: str) -> int:
    normalized = str(symbol)
    if normalized.startswith(("5", "6", "9")):
        return 1
    return 0


class FixedTechDataProvider:
    def __init__(self, tech_data: dict):
        self._tech_data = _deepcopy_dict(tech_data)

    def get(self, symbol: str) -> dict:
        return _deepcopy_dict(self._tech_data)


class JsonSymbolTechDataProvider:
    def __init__(self, *, tech_data_by_symbol: dict[str, dict], default_tech_data: dict | None = None):
        self._tech_data_by_symbol = {str(symbol): _deepcopy_dict(payload or {}) for symbol, payload in (tech_data_by_symbol or {}).items()}
        self._default_tech_data = _default_tech_data_payload(default_tech_data)

    def get(self, symbol: str) -> dict:
        return _deepcopy_dict(self._tech_data_by_symbol.get(str(symbol), self._default_tech_data))


class JsonQuoteDataProvider:
    def __init__(self, *, quote_data_by_symbol: dict[str, dict]):
        self._quote_data_by_symbol = {str(symbol): _deepcopy_dict(payload or {}) for symbol, payload in (quote_data_by_symbol or {}).items()}

    def get(self, symbol: str) -> dict:
        return _deepcopy_dict(self._quote_data_by_symbol.get(str(symbol), {}))


class InMemoryQuoteSnapshotProvider:
    def __init__(self, *, snapshots_by_symbol: dict[str, dict]):
        self._snapshots_by_symbol = {
            str(symbol): _normalize_quote_snapshot(str(symbol), payload)
            for symbol, payload in (snapshots_by_symbol or {}).items()
        }

    def get(self, symbol: str) -> QuoteSnapshot | dict:
        snapshot = self._snapshots_by_symbol.get(str(symbol), {})
        return _deepcopy_dict(snapshot)


class FileQuoteSnapshotProvider:
    def __init__(self, *, snapshot_path: str | Path):
        payload = _load_json_file(snapshot_path)
        self._provider = InMemoryQuoteSnapshotProvider(snapshots_by_symbol=payload)

    def get(self, symbol: str) -> QuoteSnapshot | dict:
        return self._provider.get(symbol)


class TdxQuoteSnapshotSource:
    def __init__(
        self,
        *,
        api_cls: type | None = None,
        servers: list[tuple[str, str, int]] | None = None,
        market_by_symbol: dict[str, int] | None = None,
    ):
        self._api_cls = api_cls or TdxHq_API
        self._servers = list(servers or _DEFAULT_TDX_SERVERS)
        self._market_by_symbol = {str(symbol): int(market) for symbol, market in (market_by_symbol or {}).items()}

    def get(self, symbol: str) -> QuoteSnapshot | dict:
        if self._api_cls is None:
            raise RuntimeError("pytdx unavailable")

        normalized_symbol = str(symbol)
        market = self._market_by_symbol.get(normalized_symbol, _infer_tdx_market(normalized_symbol))
        last_error: Exception | None = None

        for _server_name, host, port in self._servers:
            api = self._api_cls(heartbeat=True, auto_retry=True, raise_exception=False)
            try:
                ok = api.connect(host, port, time_out=3)
                if not ok:
                    raise RuntimeError(f"connect failed: {host}:{port}")

                rows = api.get_security_quotes([(market, normalized_symbol)])
                if not rows:
                    raise RuntimeError(f"empty quote: {host}:{port}")

                row = rows[0] or {}
                price = float(row.get("price") or 0)
                if price <= 0:
                    raise RuntimeError(
                        f"invalid quote price={price} from {host}:{port} servertime={row.get('servertime')}"
                    )

                servertime = str(row.get("servertime") or datetime.now().strftime("%H:%M:%S"))
                snapshot: QuoteSnapshot = {
                    "symbol": normalized_symbol,
                    "last_price": price,
                    "source": "tdx_tcp",
                    "as_of": servertime,
                }
                if row.get("tech_data") is not None:
                    snapshot["tech_data"] = _deepcopy_dict(row.get("tech_data") or {})
                return snapshot
            except Exception as exc:
                last_error = exc
            finally:
                try:
                    api.disconnect()
                except Exception:
                    pass

        raise RuntimeError(f"all tdx servers failed or returned invalid quotes: {last_error}")


class EastmoneyQuoteSnapshotSource:
    def get(self, symbol: str) -> QuoteSnapshot | dict:
        raise NotImplementedError("eastmoney quote snapshot source is not implemented")


def build_quote_snapshot_provider(
    *,
    source: str,
    snapshot_path: str | Path | None = None,
    snapshots_by_symbol: dict[str, dict] | None = None,
) -> RealtimeSource:
    normalized_source = str(source).strip().lower()
    if normalized_source == "file":
        if snapshot_path is None:
            raise ValueError("snapshot_path is required for file quote snapshot source")
        return FileQuoteSnapshotProvider(snapshot_path=snapshot_path)
    if normalized_source == "mock":
        return InMemoryQuoteSnapshotProvider(snapshots_by_symbol=snapshots_by_symbol or {})
    if normalized_source == "tdx":
        return TdxQuoteSnapshotSource()
    if normalized_source == "eastmoney":
        return EastmoneyQuoteSnapshotSource()
    raise ValueError(f"Unsupported quote snapshot source: {source}")


class QuoteTechDataAdapter:
    def __init__(self, *, quote_provider: QuoteProvider, default_tech_data: dict | None = None):
        self._quote_provider = quote_provider
        self._default_tech_data = _default_tech_data_payload(default_tech_data)

    def get(self, symbol: str) -> dict:
        payload = _deepcopy_dict(self._quote_provider.get(symbol) or {})
        tech_data = payload.get("tech_data") or self._default_tech_data
        return _deepcopy_dict(tech_data)


class QuoteSnapshotTechDataAdapter:
    def __init__(self, *, quote_source: RealtimeSource, default_tech_data: dict | None = None):
        self._quote_source = quote_source
        self._default_tech_data = _default_tech_data_payload(default_tech_data)

    def get(self, symbol: str) -> dict:
        try:
            payload = _deepcopy_dict(self._quote_source.get(symbol) or {})
        except Exception as exc:
            logger.warning(
                "quote snapshot source failed; falling back to default tech_data",
                extra={"symbol": str(symbol), "error": repr(exc)},
            )
            return _deepcopy_dict(self._default_tech_data)
        tech_data = payload.get("tech_data") or self._default_tech_data
        return _deepcopy_dict(tech_data)


def _build_quote_snapshot_provider_from_config_path(config_path: str | Path) -> RealtimeSource:
    config_file = Path(config_path)
    payload = _load_json_file(config_file)
    if isinstance(payload, dict) and payload.get("source"):
        snapshot_path = payload.get("snapshot_path")
        if snapshot_path is not None:
            snapshot_path = Path(snapshot_path)
            if not snapshot_path.is_absolute():
                snapshot_path = config_file.parent / snapshot_path
        return build_quote_snapshot_provider(
            source=payload["source"],
            snapshot_path=snapshot_path,
            snapshots_by_symbol=payload.get("snapshots_by_symbol"),
        )
    return InMemoryQuoteSnapshotProvider(snapshots_by_symbol=payload)


def build_tech_data_provider(
    *,
    tech_data_config_path: str | Path | None,
    quote_data_config_path: str | Path | None = None,
    quote_snapshot_config_path: str | Path | None = None,
    default_tech_data: dict | None = None,
) -> TechDataProvider:
    if tech_data_config_path:
        payload = _load_json_file(tech_data_config_path)
        return JsonSymbolTechDataProvider(
            tech_data_by_symbol=payload,
            default_tech_data=default_tech_data,
        )
    if quote_data_config_path:
        payload = _load_json_file(quote_data_config_path)
        return QuoteTechDataAdapter(
            quote_provider=JsonQuoteDataProvider(quote_data_by_symbol=payload),
            default_tech_data=default_tech_data,
        )
    if quote_snapshot_config_path:
        return QuoteSnapshotTechDataAdapter(
            quote_source=_build_quote_snapshot_provider_from_config_path(quote_snapshot_config_path),
            default_tech_data=default_tech_data,
        )
    return FixedTechDataProvider(_default_tech_data_payload(default_tech_data))
