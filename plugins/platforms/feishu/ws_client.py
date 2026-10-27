"""Construct Feishu SDK clients without sending unsupported Channel arguments."""
import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def supports_keyword_argument(callable_obj: Any, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or (parameter.name == keyword and parameter.kind != inspect.Parameter.POSITIONAL_ONLY)
        for parameter in parameters
    )


def create_websocket_client(client_type: Any, kwargs: dict[str, Any]) -> Any:
    # Modern SDKs need the Channel tag for group events; legacy constructors reject it.
    if supports_keyword_argument(client_type, "extra_ua_tags"):
        kwargs = {**kwargs, "extra_ua_tags": ["channel"]}
    else:
        logger.warning("[Feishu] SDK lacks extra_ua_tags; connecting without Channel signaling")
    return client_type(**kwargs)
