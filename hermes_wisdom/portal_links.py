"""Stable Portal entry points resolve mutable team slugs after authentication."""

from typing import Literal
from urllib.parse import quote, urlencode


def portal_item_url(
    base: str,
    org_id: str,
    kind: Literal["review", "skills"],
    item_id: str,
    *,
    version: int | None = None,
) -> str:
    query = {"org_id": org_id}
    if version is not None:
        query["version"] = str(version)
    return (
        f"{base.rstrip('/')}/wisdom/{kind}/{quote(item_id, safe='')}"
        f"?{urlencode(query)}"
    )
