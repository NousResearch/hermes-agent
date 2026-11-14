"""Credentialed provider requests that core makes FOR plugins; the secret never reaches plugin code.

A plugin that needs the user's account data (Codex / ChatGPT, Grok / xAI, or any provider Hermes
already talks to) must not read, refresh or rewrite Hermes's auth store, a provider's API key, or
the vendor CLI's token file (``~/.codex/auth.json``). OAuth refresh tokens are single-use, so a
second refresher races Hermes's locked refresh and can sign the user out. Instead the plugin
declares ``requires_auth: [<provider>]`` in its ``plugin.yaml`` and calls
:func:`credentialed_provider_request`. Hermes resolves the credential through the same path it uses
for inference, attaches ``Authorization`` itself, sends only to that provider's origins, never
follows a redirect, and hands back status, headers and body.

**Adding a provider is not a code change.** The allowlist is the provider's own API host from
``PROVIDER_REGISTRY`` — where Hermes already sends that credential for inference, so an account
read adds no new trust boundary — plus account hosts on another domain, listed in
``_EXTRA_HOSTS``. Every provider Hermes officially supports is therefore usable by a plugin
immediately, OAuth or API key alike. API keys are read-only here: a plugin cannot spend them.
User-defined custom providers are deliberately out of scope.

The caller is identified by file location: the nearest stack frame inside an installed plugin
directory. Like capabilities this is consent and visibility, not a sandbox. In-process plugin code
can still read any file the user can.
"""

from __future__ import annotations

import json as _json
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from hermes_cli.urllib_security import url_origin


class ProviderNotSignedIn(RuntimeError):
    """The user has no usable sign-in for the provider in this Hermes profile."""


@dataclass(frozen=True)
class ProviderResponse:
    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self) -> Any:
        return _json.loads(self.body)


@dataclass(frozen=True)
class _Access:
    """One provider's allowlist and credential resolver, plus whether the credential may write."""

    origins: frozenset
    resolve: Callable[[str], tuple[dict[str, str], str]]
    read_only: bool = False


# Account endpoints on a different host than the provider's API. Nothing in the registry can say
# where a vendor put its billing page, so these stay written down; everything else is derived.
_EXTRA_HOSTS: dict[str, tuple[str, ...]] = {
    "openai-codex": ("api.openai.com",),  # Realtime client-secret endpoint
    "xai-oauth": ("cli-chat-proxy.grok.com", "grok.com"),  # Grok CLI credits and settings
}

# API keys can spend real money, so a plugin gets reads only. OAuth keeps the method it asks for.
_READ_ONLY_METHODS = frozenset({"GET", "HEAD"})


def _provider_registry() -> Mapping[str, Any]:
    """Hermes's provider table (the same data behind ``hermes model``), imported late."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    return PROVIDER_REGISTRY


def _oauth_flows() -> Mapping[str, Any]:
    """Providers whose credential is a sign-in rather than a key, with its locked-refresh resolver."""
    from hermes_cli.auth import OAUTH_PROVIDER_FLOWS

    return OAUTH_PROVIDER_FLOWS


def _origins(provider: str) -> frozenset:
    """HTTPS origins for one provider: its own API host, plus account hosts on other domains.

    The API host is Hermes's static default for the provider, not the user's configured base URL:
    a profile pointing at a custom gateway holds that gateway's key, and the caller refuses it
    later. Local providers are ``http://127.0.0.1`` and therefore get no origins at all.
    """
    hosts = set(_EXTRA_HOSTS.get(provider, ()))
    pconfig = _provider_registry().get(provider)
    base = pconfig.inference_base_url if pconfig is not None else ""
    if base:
        scheme, host, _port = url_origin(base)
        if scheme == "https" and host:
            hosts.add(host)
    return frozenset(("https", host, 443) for host in hosts)


def _not_configured(provider: str) -> str:
    """The message a plugin passes back to the user; it names the exact next step."""
    pconfig = _provider_registry().get(provider)
    if pconfig is not None and pconfig.auth_type == "api_key":
        env = ", ".join(pconfig.api_key_env_vars) or "an API key"
        return (f"No {provider} API key in this Hermes profile. Set {env} or add it under "
                "Providers in the Hermes desktop app.")
    from agent.turn_failure_copy import oauth_relogin_command

    return (f"No usable {provider} sign-in in this Hermes profile. Run "
            f"`{oauth_relogin_command(provider)}` or sign in under Providers in the Hermes "
            "desktop app.")


def _credentials(provider: str) -> Mapping[str, Any]:
    """Resolve the provider's credential the same way inference does, without exposing it.

    An ``AuthError`` that says the grant needs a re-login reads as "not configured"; any other
    auth failure stays an error, because the credential may still be good (a rate limit, for
    example) and the caller should see the real reason.
    """
    from hermes_cli.auth import resolve_api_key_provider_credentials
    from hermes_cli.auth_constants import AuthError

    flow = _oauth_flows().get(provider)
    try:
        if flow is not None:
            return flow.resolve()
        pconfig = _provider_registry().get(provider)
        if pconfig is None or pconfig.auth_type != "api_key":
            return {}
        return resolve_api_key_provider_credentials(provider)
    except AuthError as exc:
        if getattr(exc, "relogin_required", True) is False:
            raise
        return {}


def _codex_account_headers(token: str, host: str) -> Mapping[str, str]:
    """chatgpt.com's Codex backend wants its account headers; api.openai.com takes the bearer alone."""
    if host != "chatgpt.com":
        return {}
    from agent.codex_headers import codex_account_headers

    return codex_account_headers(token)


def _auth(provider: str) -> Callable[[str], tuple[dict[str, str], str]]:
    """Attach the credential at request time and hand back no copy of it."""

    def resolve(url: str) -> tuple[dict[str, str], str]:
        creds = _credentials(provider)
        token = str(creds.get("api_key") or "").strip()
        if not token:
            raise ProviderNotSignedIn(_not_configured(provider))
        headers = dict(_codex_account_headers(token, url_origin(url)[1])) if provider == "openai-codex" else {}
        headers["Authorization"] = f"Bearer {token}"
        return headers, str(creds.get("base_url") or "")

    return resolve


def _access(provider: str) -> Optional[_Access]:
    """How a plugin may use *provider*, or None when Hermes has no credential path for it."""
    origins = _origins(provider)
    if not origins:
        return None
    if provider in _oauth_flows():
        return _Access(origins, _auth(provider))
    pconfig = _provider_registry().get(provider)
    if pconfig is not None and pconfig.auth_type == "api_key":
        return _Access(origins, _auth(provider), read_only=True)
    return None


def credentialed_providers() -> tuple[str, ...]:
    """Providers a plugin can use through :func:`credentialed_provider_request` today."""
    return tuple(sorted(name for name in _provider_registry() if _access(name) is not None))


# Never from the caller: these come from the URL and the body, or from Hermes's own credential.
_RESERVED_HEADERS = frozenset({"authorization", "host", "content-length"})


def _plugin_roots() -> set[Path]:
    """Every plugins directory an installed plugin can live under, as given and resolved."""
    from hermes_cli.plugins import get_bundled_plugins_dir
    from hermes_constants import get_default_hermes_root, get_hermes_home, get_process_hermes_home

    bases = (get_hermes_home() / "plugins", get_process_hermes_home() / "plugins",
             get_default_hermes_root() / "plugins", get_bundled_plugins_dir())
    return {r for base in bases for r in (base.absolute(), base.resolve())}


def _calling_plugin_dir() -> Optional[Path]:
    """Directory of the plugin whose code is nearest on the calling stack, or None."""
    roots = _plugin_roots()
    frame = sys._getframe(2)
    while frame is not None:
        raw = Path(frame.f_code.co_filename)
        for path in (raw.absolute(), raw.resolve()):
            root = next((r for r in roots if path.is_relative_to(r) and path != r), None)
            if root is None:
                continue
            for parent in path.parents:
                if parent == root:
                    break
                if (parent / "plugin.yaml").is_file() or (parent / "plugin.yml").is_file():
                    return parent
            return root / path.relative_to(root).parts[0]
        frame = frame.f_back
    return None


def _detached(exc: BaseException) -> BaseException:
    """*exc* without traceback or chain: those frames hold the token, refresh token or client secret."""
    exc.__cause__ = exc.__context__ = None
    return exc.with_traceback(None)


def _transport_error(exc: BaseException) -> BaseException:
    """Rebuild an httpx error without its request, which carries the attached credential.

    ``HTTPStatusError`` and friends require ``request=``/``response=``, so a plain rebuild can
    raise inside the handler — keeping the original alive through ``__context__``. Fall back to
    a RuntimeError with the same text instead.
    """
    try:
        rebuilt: BaseException = type(exc)(str(exc))
    except Exception:
        rebuilt = RuntimeError(str(exc))
    return _detached(rebuilt)


def _has_userinfo(url: str) -> bool:
    """True when *url* embeds ``user:password@``; ``url_origin`` normalizes that away."""
    parts = urllib.parse.urlsplit(url)
    return parts.username is not None or parts.password is not None


def declared_auth_providers(manifest: Mapping[str, Any]) -> list[str]:
    """``requires_auth`` from a manifest dict: the providers whose sign-in the plugin uses."""
    raw = (manifest or {}).get("requires_auth")
    return [str(p).strip() for p in raw if str(p).strip()] if isinstance(raw, list) else []


def requires_auth_notice(manifest: Mapping[str, Any]) -> str:
    """One line for install/show surfaces; empty when the plugin declares no sign-in use."""
    from hermes_cli.plugin_oauth import oauth_notice

    providers = declared_auth_providers(manifest)
    sign_in = (f"Uses your sign-in for: {', '.join(providers)} (Hermes sends the requests; "
               "the plugin never receives the token)") if providers else ""
    return "; ".join(part for part in (sign_in, oauth_notice(manifest)) if part)


def _declared_providers(plugin_dir: Path) -> list[str]:
    from pm.plugin_declarations import native_manifest_file, read_native_manifest

    manifest = native_manifest_file(plugin_dir)
    return declared_auth_providers(read_native_manifest(manifest)) if manifest else []


def credentialed_provider_request(
    provider: str, method: str, url: str, *, json: Any = None, content: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None, timeout: float = 30.0,
) -> ProviderResponse:
    """Send ``method url`` with the user's *provider* sign-in attached by Hermes.

    *provider* is a Hermes sign-in (listed under ``requires_auth``) or an ``id`` the calling plugin
    declares under ``oauth:`` (:mod:`hermes_cli.plugin_oauth`); the plugin's own entry wins.
    Refuses (``PermissionError``) a caller outside an installed plugin, a plugin whose ``plugin.yaml``
    does not list *provider* under ``requires_auth``, any origin outside the provider's allowlist, and
    a URL with embedded ``user:password@``; raises :class:`ProviderNotSignedIn` when this profile has
    no sign-in. Redirects are returned as-is, never followed. The result carries no credential.

    ``json=`` sends a JSON body; ``content=`` sends raw bytes (a gRPC-web frame, for example) with the
    caller's own ``Content-Type``. Caller headers never replace ``Authorization``, ``Host`` or
    ``Content-Length``.
    """
    from hermes_cli.plugin_oauth import _plugin_oauth_spec

    if json is not None and content is not None:
        raise ValueError("credentialed_provider_request takes json= or content=, not both")
    plugin_dir = _calling_plugin_dir()
    if plugin_dir is None:
        raise PermissionError("credentialed_provider_request must be called from an installed plugin's code")
    spec = _plugin_oauth_spec(plugin_dir, provider)
    if spec is not None:  # the plugin's own OAuth entry, so a core id added later cannot take it over
        access = _Access(frozenset(spec[0]), spec[1])
    else:
        access = _access(provider)
        if access is None:
            raise ValueError(f"No credentialed requests for provider {provider!r} (supported: "
                             f"{', '.join(credentialed_providers())}, or an id the plugin declares "
                             "under oauth: in plugin.yaml)")
        if provider not in _declared_providers(plugin_dir):
            raise PermissionError(
                f"Plugin '{plugin_dir.name}' must declare `requires_auth: [{provider}]` in plugin.yaml")
        if access.read_only and method.upper() not in _READ_ONLY_METHODS:
            raise PermissionError(
                f"'{provider}' is used through its API key; only GET and HEAD are allowed")
    if _has_userinfo(url):
        raise PermissionError(f"Refusing a {provider} request URL with embedded credentials")
    try:
        origin = url_origin(url)
    except ValueError:  # malformed port: fail closed with the same refusal the caller handles
        raise PermissionError(f"Refusing a malformed {provider} request URL") from None
    if origin not in access.origins:
        raise PermissionError(f"Refusing to send the {provider} credential to {origin[1] or url!r}")
    import httpx

    # Every error the plugin receives past this point is rebuilt without traceback, chain or request: those
    # reach the frames and httpx objects that hold the token, refresh token or client secret.
    refusal: Optional[BaseException] = None
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        extra = {name: value for name, value in dict(headers or {}).items()
                 if name.lower() not in _RESERVED_HEADERS}  # no credential yet
        request = client.build_request(method, url, json=json, content=content, headers=extra)
        try:
            auth_headers, credential_base = access.resolve(url)
        except httpx.HTTPError as exc:  # a resolver's own refresh POST failed: its request holds the refresh token
            refusal = _transport_error(exc)
        except (RuntimeError, OSError, ValueError, LookupError, TypeError) as exc:
            refusal = _detached(exc)
        else:
            if url_origin(credential_base) not in access.origins:
                # A pooled gateway key belongs to its own host only (#121486).
                refusal = PermissionError(
                    f"This profile's {provider} credential routes to a custom endpoint; refusing")
            else:
                for name in auth_headers:
                    request.headers.pop(name, None)
                request.headers.update(auth_headers)
                try:
                    response = client.send(request)
                except httpx.HTTPError as exc:
                    refusal = _transport_error(exc)
            auth_headers = request = None
    if refusal is not None:
        raise refusal
    return ProviderResponse(response.status_code, dict(response.headers), response.content)


__all__ = [
    "ProviderNotSignedIn", "ProviderResponse", "credentialed_provider_request", "credentialed_providers",
    "declared_auth_providers", "requires_auth_notice",
]
