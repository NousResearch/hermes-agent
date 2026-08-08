"""Quote/tech-data provider primitives for hermes_t."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

_DEFAULT_TDX_SERVERS: list[tuple[str, int]] = [
    ("119.147.212.81", 7709),
    ("119.147.212.83", 7709),
]


class QuoteProvider(Protocol):
    def get(self, symbol: str) -> dict[str, Any]: ...


class QuoteSnapshotProvider(Protocol):
    def get(self, symbol: str) -> dict[str, Any]: ...


class TechDataProvider(Protocol):
    def get(self, symbol: str) -> dict[str, Any]: ...


class StaticTechDataProvider:
    """Return the same tech_data payload for any symbol."""

    def __init__(self, tech_data: dict[str, Any] | None = None):
        self.tech_data = dict(tech_data or {})

    def get(self, symbol: str) -> dict[str, Any]:
        del symbol
        return dict(self.tech_data)


class JsonQuoteDataProvider:
    """Read per-symbol raw quote payloads from a JSON file."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self._payloads: dict[str, Any] = json.loads(self.config_path.read_text(encoding="utf-8"))

    def get(self, symbol: str) -> dict[str, Any]:
        payload = self._payloads.get(symbol, {})
        return payload if isinstance(payload, dict) else {}


class JsonQuoteSnapshotSource:
    """Read per-symbol quote snapshots from a JSON file or in-memory mapping."""

    def __init__(self, snapshot_path: str | Path):
        self.snapshot_path: Path | None = Path(snapshot_path)
        self._snapshots: dict[str, Any] = json.loads(self.snapshot_path.read_text(encoding="utf-8"))

    @classmethod
    def from_mapping(cls, snapshots_by_symbol: dict[str, Any] | None) -> "JsonQuoteSnapshotSource":
        instance = cls.__new__(cls)
        instance.snapshot_path = None
        instance._snapshots = dict(snapshots_by_symbol or {})
        return instance

    def get(self, symbol: str) -> dict[str, Any]:
        payload = self._snapshots.get(symbol, {})
        return payload if isinstance(payload, dict) else {}


class TdxQuoteSnapshotSource:
    """Fetch standardized quote snapshots from TDX TCP servers."""

    def __init__(
        self,
        *,
        api_cls: type | None = None,
        servers: list[tuple[str, int]] | None = None,
        market_by_symbol: dict[str, int] | None = None,
    ):
        self.api_cls = api_cls or _import_tdx_api_cls()
        self.servers = [tuple(server) for server in (servers or _DEFAULT_TDX_SERVERS)]
        self.market_by_symbol = dict(market_by_symbol or {})

    def get(self, symbol: str) -> dict[str, Any]:
        last_error: Exception | None = None
        market = self.market_by_symbol.get(symbol, _infer_tdx_market(symbol))
        for host, port in self.servers:
            api = self.api_cls()
            try:
                connected = api.connect(host, port)
                if connected is False:
                    raise ConnectionError(f"tdx connect returned false: {host}:{port}")
                quotes = api.get_security_quotes([(market, symbol)])
                return _normalize_tdx_snapshot(symbol, quotes)
            except ValueError:
                raise
            except Exception as exc:  # pragma: no cover - exercised by tests
                last_error = exc
            finally:
                try:
                    api.disconnect()
                except Exception:
                    pass
        raise RuntimeError(f"all tdx servers failed for {symbol}") from last_error


class EastmoneyQuoteSnapshotSource:
    """Fetch standardized quote snapshots from Eastmoney quote payloads."""

    def __init__(
        self,
        *,
        fetcher: Any | None = None,
        market_by_symbol: dict[str, int] | None = None,
    ):
        self.fetcher = fetcher or _import_eastmoney_fetcher()
        self.market_by_symbol = dict(market_by_symbol or {})

    def get(self, symbol: str) -> dict[str, Any]:
        secid = _build_eastmoney_secid(symbol, self.market_by_symbol.get(symbol))
        payload = self.fetcher(secid)
        return _normalize_eastmoney_snapshot(symbol, payload)


class QuoteTechDataAdapter:
    """Adapt raw quote payloads into runtime tech_data payloads."""

    def __init__(self, quote_provider: QuoteProvider, default_tech_data: dict[str, Any] | None = None):
        self.quote_provider = quote_provider
        self.default_tech_data = dict(default_tech_data or {})

    def get(self, symbol: str) -> dict[str, Any]:
        payload = self.quote_provider.get(symbol)
        if not isinstance(payload, dict):
            return dict(self.default_tech_data)
        tech_data = payload.get("tech_data")
        if isinstance(tech_data, dict) and tech_data:
            return tech_data
        return dict(self.default_tech_data)


class QuoteSnapshotTechDataAdapter:
    """Adapt quote snapshot payloads into runtime tech_data payloads."""

    def __init__(self, snapshot_provider: QuoteSnapshotProvider, default_tech_data: dict[str, Any] | None = None):
        self.snapshot_provider = snapshot_provider
        self.default_tech_data = dict(default_tech_data or {})

    def get(self, symbol: str) -> dict[str, Any]:
        try:
            payload = self.snapshot_provider.get(symbol)
        except Exception:
            return dict(self.default_tech_data)
        if not isinstance(payload, dict):
            return dict(self.default_tech_data)
        tech_data = payload.get("tech_data")
        if isinstance(tech_data, dict) and tech_data:
            return tech_data
        return dict(self.default_tech_data)


def _import_tdx_api_cls() -> type:
    try:
        from pytdx.hq import TdxHq_API  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in future tests
        raise RuntimeError("pytdx unavailable") from exc
    return TdxHq_API


def _import_requests_module() -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised in future tests
        raise RuntimeError("requests unavailable") from exc
    return requests


def _import_eastmoney_fetcher() -> Any:
    requests = _import_requests_module()

    def _fetch(secid: str) -> dict[str, Any]:
        response = requests.get(
            "https://push2.eastmoney.com/api/qt/stock/get",
            params={
                "secid": secid,
                "fields": "f43,f57,f58,f86",
                "invt": "2",
                "fltt": "1",
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    return _fetch


def _infer_tdx_market(symbol: str) -> int:
    if symbol.startswith(("5", "6", "9")):
        return 1
    return 0


def _infer_eastmoney_market(symbol: str) -> int:
    if symbol.startswith(("5", "6", "9")):
        return 1
    return 0


def _build_eastmoney_secid(symbol: str, market: int | None = None) -> str:
    return f"{_infer_eastmoney_market(symbol) if market is None else int(market)}.{symbol}"


def _normalize_tdx_snapshot(symbol: str, quotes: Any) -> dict[str, Any]:
    if not isinstance(quotes, list) or not quotes:
        raise ValueError(f"empty tdx quote for {symbol}")
    quote = quotes[0]
    if not isinstance(quote, dict):
        raise ValueError(f"invalid tdx quote payload for {symbol}")
    price = quote.get("price")
    if not isinstance(price, (int, float)) or price <= 0:
        raise ValueError(f"non-positive tdx price for {symbol}")
    snapshot = {
        "symbol": symbol,
        "last_price": float(price),
        "source": "tdx_tcp",
        "as_of": quote.get("servertime") or quote.get("datetime") or "",
    }
    tech_data = quote.get("tech_data")
    if isinstance(tech_data, dict) and tech_data:
        snapshot["tech_data"] = tech_data
    return snapshot


def _normalize_eastmoney_snapshot(symbol: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"invalid eastmoney payload for {symbol}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"missing eastmoney data for {symbol}")
    raw_price = data.get("f43")
    if not isinstance(raw_price, (int, float)) or raw_price <= 0:
        raise ValueError(f"non-positive eastmoney price for {symbol}")
    snapshot = {
        "symbol": symbol,
        "last_price": float(raw_price) / 10000,
        "source": "eastmoney",
        "as_of": data.get("f86") or "",
    }
    tech_data = data.get("tech_data")
    if isinstance(tech_data, dict) and tech_data:
        snapshot["tech_data"] = tech_data
    return snapshot


def _load_json_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return payload, path.parent


def build_quote_snapshot_provider(config: dict[str, Any], *, base_dir: str | Path | None = None) -> QuoteSnapshotProvider:
    source = str(config.get("source") or "").strip()
    if source == "file":
        snapshot_path = config.get("snapshot_path")
        if not isinstance(snapshot_path, str) or not snapshot_path.strip():
            raise ValueError("quote snapshot file source requires snapshot_path")
        snapshot_file = Path(snapshot_path)
        if not snapshot_file.is_absolute() and base_dir is not None:
            snapshot_file = Path(base_dir) / snapshot_file
        return JsonQuoteSnapshotSource(snapshot_file)
    if source == "mock":
        snapshots = config.get("snapshots_by_symbol", {})
        if not isinstance(snapshots, dict):
            raise ValueError("quote snapshot mock source requires snapshots_by_symbol object")
        return JsonQuoteSnapshotSource.from_mapping(snapshots)
    if source == "tdx":
        servers = config.get("servers")
        normalized_servers = None
        if servers is not None:
            normalized_servers = [tuple(server) for server in servers]
        market_by_symbol = config.get("market_by_symbol")
        if market_by_symbol is not None and not isinstance(market_by_symbol, dict):
            raise ValueError("tdx market_by_symbol must be an object")
        return TdxQuoteSnapshotSource(
            api_cls=config.get("api_cls"),
            servers=normalized_servers,
            market_by_symbol=market_by_symbol,
        )
    if source == "eastmoney":
        market_by_symbol = config.get("market_by_symbol")
        if market_by_symbol is not None and not isinstance(market_by_symbol, dict):
            raise ValueError("eastmoney market_by_symbol must be an object")
        return EastmoneyQuoteSnapshotSource(
            fetcher=config.get("fetcher"),
            market_by_symbol=market_by_symbol,
        )
    raise ValueError(f"unsupported quote snapshot source: {source or '<blank>'}")


def build_tech_data_provider(
    *,
    tech_data_config_path: str | Path | None = None,
    quote_data_config_path: str | Path | None = None,
    quote_snapshot_config_path: str | Path | None = None,
    default_tech_data: dict[str, Any] | None = None,
) -> TechDataProvider:
    if tech_data_config_path is not None:
        return JsonQuoteDataProvider(tech_data_config_path)
    if quote_data_config_path is not None:
        return QuoteTechDataAdapter(
            JsonQuoteDataProvider(quote_data_config_path),
            default_tech_data=default_tech_data,
        )
    if quote_snapshot_config_path is not None:
        config, config_dir = _load_json_config(quote_snapshot_config_path)
        return QuoteSnapshotTechDataAdapter(
            build_quote_snapshot_provider(config, base_dir=config_dir),
            default_tech_data=default_tech_data,
        )
    return StaticTechDataProvider(default_tech_data)


__all__ = [
    "EastmoneyQuoteSnapshotSource",
    "JsonQuoteDataProvider",
    "JsonQuoteSnapshotSource",
    "QuoteProvider",
    "QuoteSnapshotProvider",
    "QuoteSnapshotTechDataAdapter",
    "QuoteTechDataAdapter",
    "StaticTechDataProvider",
    "TdxQuoteSnapshotSource",
    "TechDataProvider",
    "build_quote_snapshot_provider",
    "build_tech_data_provider",
]
