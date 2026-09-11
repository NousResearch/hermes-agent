"""Immutable provider runtimes and complete wire-client bundles.

Provider resolution, client construction, and installation are deliberately
separate operations.  A runtime carries the full transport contract so a
fallback, credential rotation, or model switch cannot silently rebuild a
client from only ``api_key`` and ``base_url``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Optional


class _FrozenList(tuple):
    """Tuple-backed list value that keeps legacy list equality useful in tests."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return NotImplemented


def _freeze(value: Any) -> Any:
    """Freeze JSON-like containers while leaving SDK/client objects opaque."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def mutable_config_copy(value: Any) -> Any:
    """Copy configuration containers, including frozen ones, without copying SDK objects."""
    if isinstance(value, Mapping):
        return {key: mutable_config_copy(item) for key, item in value.items()}
    if isinstance(value, (list, _FrozenList)):
        return [mutable_config_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mutable_config_copy(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return {mutable_config_copy(item) for item in value}
    return value


_CORE_FIELDS = frozenset({
    "provider", "model", "api_mode", "api_key", "base_url",
    "requested_provider", "source", "extra_headers", "ssl_ca_cert", "ssl_verify",
})


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedRuntime(Mapping[str, Any]):
    """Immutable, mapping-compatible result of provider resolution.

    Mapping compatibility lets the current facade migrate incrementally.  All
    values are frozen where they are configuration containers, and unknown
    resolver fields are retained in ``metadata`` rather than discarded.
    """

    provider: str
    model: str = ""
    api_mode: str = "chat_completions"
    api_key: Any = ""
    base_url: str = ""
    requested_provider: str = ""
    source: str = ""
    extra_headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    ssl_ca_cert: Optional[str] = None
    ssl_verify: Optional[bool] = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    _keys: tuple[str, ...] = field(default=(), repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "ResolvedRuntime") -> "ResolvedRuntime":
        if isinstance(value, cls):
            return value
        raw = dict(value)
        headers = raw.get("extra_headers")
        frozen_headers = _freeze(headers) if isinstance(headers, Mapping) else MappingProxyType({})
        metadata = MappingProxyType({
            key: _freeze(item) for key, item in raw.items() if key not in _CORE_FIELDS
        })
        ssl_ca_cert = raw.get("ssl_ca_cert")
        if not isinstance(ssl_ca_cert, str) or not ssl_ca_cert.strip():
            ssl_ca_cert = None
        ssl_verify = raw.get("ssl_verify")
        if not isinstance(ssl_verify, bool):
            ssl_verify = None
        return cls(
            provider=str(raw.get("provider") or ""),
            model=str(raw.get("model") or ""),
            api_mode=str(raw.get("api_mode") or "chat_completions"),
            api_key=raw.get("api_key", ""),
            base_url=str(raw.get("base_url") or ""),
            requested_provider=str(raw.get("requested_provider") or ""),
            source=str(raw.get("source") or ""),
            extra_headers=frozen_headers,
            ssl_ca_cert=ssl_ca_cert,
            ssl_verify=ssl_verify,
            metadata=metadata,
            _keys=tuple(raw.keys()),
        )

    def __getitem__(self, key: str) -> Any:
        if key not in self._keys:
            raise KeyError(key)
        if key in _CORE_FIELDS:
            return getattr(self, key)
        return self.metadata[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable compatibility copy for legacy callers."""
        return mutable_config_copy(self)

    def with_updates(self, **updates: Any) -> "ResolvedRuntime":
        raw = self.as_dict()
        raw.update(updates)
        return type(self).from_mapping(raw)


@dataclass(frozen=True, slots=True)
class ClientBundle:
    """A completely built client pair plus the immutable runtime that built it."""

    runtime: ResolvedRuntime
    client: Any = None
    anthropic_client: Any = None
    client_kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    anthropic_api_key: Any = ""
    anthropic_base_url: str = ""
    is_anthropic_oauth: bool = False

    @property
    def active_client(self) -> Any:
        return self.anthropic_client if self.anthropic_client is not None else self.client


class RuntimeWireNotMigratedError(NotImplementedError):
    """A valid runtime whose special wire remains on its dedicated builder."""


OpenAIBuilder = Callable[[dict[str, Any]], Any]
AnthropicBuilder = Callable[..., Any]


def _default_openai_builder(client_kwargs: dict[str, Any]) -> Any:
    from openai import OpenAI
    from agent.process_bootstrap import build_keepalive_http_client
    from agent.ssl_verify import resolve_httpx_verify
    kwargs = mutable_config_copy(client_kwargs)
    ca = kwargs.pop("ssl_ca_cert", None)
    verify = kwargs.pop("ssl_verify", None)
    kwargs["http_client"] = build_keepalive_http_client(
        kwargs.get("base_url", ""), verify=resolve_httpx_verify(ca_bundle=ca, ssl_verify=verify),
    )
    kwargs.setdefault("max_retries", 0)
    return OpenAI(**kwargs)


def build_client_bundle(
    runtime: Mapping[str, Any] | ResolvedRuntime,
    *,
    openai_builder: Optional[OpenAIBuilder] = None,
    anthropic_builder: Optional[AnthropicBuilder] = None,
    timeout: Optional[float] = None,
    drop_context_1m_beta: bool = False,
) -> ClientBundle:
    """Build a wire client without mutating an agent.

    ``extra_headers`` is merged after declared/default headers because it is
    the provider-entry-specific layer.  TLS, query, command, and argument
    fields stay attached to the same bundle as credentials and endpoint.
    """
    resolved = ResolvedRuntime.from_mapping(runtime)
    provider = resolved.provider.strip().lower()
    api_mode = resolved.api_mode.strip().lower()
    if provider in {"bedrock", "moa"} or api_mode == "bedrock_converse":
        raise RuntimeWireNotMigratedError(
            f"{provider or api_mode} still uses its dedicated runtime builder"
        )

    effective_timeout = timeout
    if effective_timeout is None:
        raw_timeout = resolved.get("timeout")
        if isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool):
            effective_timeout = float(raw_timeout)
    if effective_timeout is not None:
        resolved = resolved.with_updates(timeout=effective_timeout)

    declared_headers = resolved.get("default_headers")
    headers = dict(declared_headers) if isinstance(declared_headers, Mapping) else {}
    headers.update(resolved.extra_headers)

    if api_mode == "anthropic_messages":
        if anthropic_builder is None:
            from agent.anthropic_adapter import build_anthropic_client

            anthropic_builder = build_anthropic_client
        assert anthropic_builder is not None
        transport = {}
        for key in ("ssl_ca_cert", "ssl_verify", "default_query"):
            if resolved.get(key) is not None:
                transport[key] = mutable_config_copy(resolved[key])
        if drop_context_1m_beta:
            transport["drop_context_1m_beta"] = True
        client = anthropic_builder(
            resolved.api_key,
            resolved.base_url or None,
            timeout=effective_timeout,
            default_headers=headers or None,
            **transport,
        )
        is_oauth = False
        if provider == "anthropic" and isinstance(resolved.api_key, str):
            from agent.anthropic_credentials import _is_oauth_token

            is_oauth = _is_oauth_token(resolved.api_key)
        return ClientBundle(
            runtime=resolved,
            anthropic_client=client,
            anthropic_api_key=resolved.api_key,
            anthropic_base_url=resolved.base_url,
            is_anthropic_oauth=is_oauth,
        )

    client_kwargs = openai_client_kwargs(resolved, timeout=effective_timeout)
    resolved = resolved.with_updates(base_url=client_kwargs["base_url"])
    if "default_query" in client_kwargs:
        resolved = resolved.with_updates(default_query=client_kwargs["default_query"])
    client = (openai_builder or _default_openai_builder)(mutable_config_copy(client_kwargs))
    return ClientBundle(
        runtime=resolved,
        client=client,
        client_kwargs=_freeze(client_kwargs),
    )


def openai_client_kwargs(runtime: Mapping[str, Any], *, timeout=None) -> dict[str, Any]:
    """Project transport configuration without carrying another client's owned HTTP handle."""
    resolved = ResolvedRuntime.from_mapping(runtime)
    prior_kwargs = resolved.get("client_kwargs")
    client_kwargs: dict[str, Any] = (
        mutable_config_copy(prior_kwargs) if isinstance(prior_kwargs, Mapping) else {}
    )
    client_kwargs.pop("http_client", None)
    client_kwargs["api_key"] = resolved.api_key
    client_kwargs["base_url"] = resolved.base_url
    headers = dict(client_kwargs.get("default_headers") or {})
    headers.update(resolved.get("default_headers") or {})
    headers.update(resolved.extra_headers)
    if headers:
        client_kwargs["default_headers"] = headers
    effective_timeout = timeout if timeout is not None else resolved.get("timeout")
    if effective_timeout is not None:
        client_kwargs["timeout"] = effective_timeout
    if resolved.ssl_ca_cert:
        client_kwargs["ssl_ca_cert"] = resolved.ssl_ca_cert
    if resolved.ssl_verify is not None:
        client_kwargs["ssl_verify"] = resolved.ssl_verify
    default_query = resolved.get("default_query")
    if isinstance(default_query, Mapping) and default_query:
        client_kwargs["default_query"] = dict(default_query)
    from urllib.parse import parse_qsl, urlsplit, urlunsplit
    parts = urlsplit(resolved.base_url)
    if parts.query:
        client_kwargs["base_url"] = urlunsplit(parts._replace(query=""))
        client_kwargs["default_query"] = {**dict(parse_qsl(parts.query)), **client_kwargs.get("default_query", {})}
    command = resolved.get("command")
    if isinstance(command, str) and command:
        client_kwargs["command"] = command
    args = resolved.get("args")
    if isinstance(args, (list, tuple)) and (command or resolved.base_url.startswith("acp://")):
        client_kwargs["args"] = list(args)

    return client_kwargs


__all__ = [
    "ClientBundle",
    "ResolvedRuntime",
    "RuntimeWireNotMigratedError",
    "build_client_bundle",
    "mutable_config_copy",
    "openai_client_kwargs",
]
