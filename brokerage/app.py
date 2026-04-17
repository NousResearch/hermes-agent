"""FastAPI surface for the deterministic brokerage service."""

from __future__ import annotations

import hmac

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

from brokerage.brokers.ibkr_tws import IBKRTwsBrokerAdapter
from brokerage.config import BrokerageSettings
from brokerage.policy import BrokeragePolicy
from brokerage.service import BrokerageService
from brokerage.storage import SQLiteBrokerageStore


class CreateTradeIntentRequest(BaseModel):
    account_mode: str
    broker_account: str | None = None
    symbol: str
    side: str
    quantity: int
    order_type: str
    asset_class: str = "stock"
    limit_price: float | None = None
    stop_price: float | None = None
    raw_request_text: str | None = None
    session_id: str | None = None
    telegram_chat_id: str | None = None
    market_snapshot: dict | None = None


class ConfirmTradeIntentRequest(BaseModel):
    confirmation_text: str


class BrokerageAppDependencies(BaseModel):
    service: BrokerageService
    auth_token: str | None = None

    model_config = {"arbitrary_types_allowed": True}


def build_service(settings: BrokerageSettings | None = None) -> BrokerageService:
    runtime_settings = settings or _load_brokerage_settings_from_config()
    store = SQLiteBrokerageStore()
    policy = BrokeragePolicy(runtime_settings)
    broker = IBKRTwsBrokerAdapter(runtime_settings)
    return BrokerageService(runtime_settings, store, policy, broker)


def _load_brokerage_settings_from_config() -> BrokerageSettings:
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception:
        config = {}

    brokerage_config = config.get("brokerage", {}) if isinstance(config, dict) else {}
    if not isinstance(brokerage_config, dict):
        brokerage_config = {}
    return BrokerageSettings(**brokerage_config)


def create_app(
    *,
    service: BrokerageService | None = None,
    auth_token: str | None = None,
) -> FastAPI:
    deps = BrokerageAppDependencies(
        service=service or build_service(),
        auth_token=auth_token,
    )
    if deps.auth_token is None:
        deps.auth_token = deps.service.settings.service_token

    app = FastAPI(title="Hermes Brokerage Service", version="0.1.0")

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not deps.auth_token:
            return
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authorization header",
            )
        expected = f"Bearer {deps.auth_token}"
        if not hmac.compare_digest(authorization.encode(), expected.encode()):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )

    def _translate_error(exc: ValueError) -> HTTPException:
        detail = str(exc)
        if detail.startswith("Unknown intent:"):
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    @app.get("/healthz")
    def healthz() -> dict:
        """Service health + broker connection status."""
        broker_health = deps.service.broker.health_check()
        return {"ok": True, **broker_health}

    @app.post("/trade-intents", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_auth)])
    def create_trade_intent(payload: CreateTradeIntentRequest) -> dict:
        try:
            return deps.service.create_intent(**payload.model_dump())
        except ValueError as exc:
            raise _translate_error(exc) from exc

    @app.post("/trade-intents/{intent_id}/confirm", dependencies=[Depends(require_auth)])
    def confirm_trade_intent(intent_id: str, payload: ConfirmTradeIntentRequest) -> dict:
        try:
            return deps.service.confirm_intent(intent_id, payload.confirmation_text)
        except ValueError as exc:
            raise _translate_error(exc) from exc

    @app.post("/trade-intents/{intent_id}/cancel", dependencies=[Depends(require_auth)])
    def cancel_trade_intent(intent_id: str) -> dict:
        try:
            return deps.service.cancel_intent(intent_id)
        except ValueError as exc:
            raise _translate_error(exc) from exc

    @app.get("/trade-intents/{intent_id}", dependencies=[Depends(require_auth)])
    def get_trade_intent(intent_id: str) -> dict:
        try:
            return deps.service.get_intent(intent_id)
        except ValueError as exc:
            raise _translate_error(exc) from exc

    @app.get("/positions", dependencies=[Depends(require_auth)])
    def get_positions(account_mode: str | None = None, account: str | None = None) -> dict:
        """Return current broker account positions."""
        positions = deps.service.get_positions(account_mode=account_mode, account=account)
        return {"positions": positions, "count": len(positions)}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("brokerage.app:app", host="127.0.0.1", port=8787, reload=False)


if __name__ == "__main__":
    main()
