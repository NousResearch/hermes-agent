"""SDK clients send their configured ``default_query`` exactly as declared.

A query-bearing endpoint (``https://relay/x?team=a&team=b&blank=``) is handed to the OpenAI and
Anthropic SDKs split: a clean ``base_url`` plus ``default_query`` (``route_identity.split_url_query``),
because the SDKs join the request path onto ``base_url`` as text. Both SDKs then serialize that
mapping with their API-specific array format (Anthropic comma-joins a list into ``team=a%2Cb``,
OpenAI emits ``team[]=a&team[]=b``) and both drop a blank value. The query can select a tenant,
so each of those is another destination than the declared one, and another one than
``route_identity.url_with_query`` reports for identity decisions.

The encoding belongs to the client class (its ``qs`` property), so a client constructed with a
``default_query`` is built from a subclass whose serializer emits the declared pairs verbatim: one
pair per list item, in order, blanks kept. Request-level params keep the SDK's own encoding.
``copy()`` / ``with_options()`` re-instantiate ``self.__class__``, so the subclass survives them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Mapping, Tuple
from urllib.parse import urlencode


class _DeclaredQuerystring:
    """The SDK's ``Querystring`` with the client's ``default_query`` values emitted verbatim."""

    def __init__(self, sdk_qs: Any, declared: Mapping[str, Any]) -> None:
        self._sdk_qs, self._declared = sdk_qs, declared

    def stringify(self, params: Mapping[str, Any], **options: Any) -> str:
        pairs: List[Tuple[str, str]] = []
        for key, value in params.items():
            # Identity, not equality: a request param that overrides the key is the SDK's to encode.
            if key in self._declared and value is self._declared[key]:
                pairs.extend((key, str(item)) for item in (value if isinstance(value, (list, tuple)) else [value]))
            else:
                pairs.extend(self._sdk_qs.stringify_items({key: value}, **options))
        return urlencode(pairs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sdk_qs, name)


@lru_cache(maxsize=None)
def _declared_query_subclass(sdk_cls: type) -> type:
    def qs(self: Any) -> Any:
        return _DeclaredQuerystring(super(subclass, self).qs, self._custom_query)

    subclass = type(sdk_cls.__name__, (sdk_cls,), {"qs": property(qs), "__module__": sdk_cls.__module__})
    return subclass


def declared_query_class(sdk_cls: Any, client_kwargs: Mapping[str, Any]) -> Any:
    """The class to construct an SDK client from: *sdk_cls*, or its declared-query subclass when
    *client_kwargs* carry a ``default_query``. Anything that is not an SDK client class (a test
    double) is returned as is."""
    if (client_kwargs.get("default_query") and isinstance(sdk_cls, type)
            and isinstance(getattr(sdk_cls, "qs", None), property)):
        return _declared_query_subclass(sdk_cls)
    return sdk_cls
