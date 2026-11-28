"""Signed-in provider requests that core makes FOR plugins; the token never reaches plugin code.

A plugin that needs the user's subscription sign-in (today: Codex / ChatGPT and xAI Grok) must not
read, refresh or rewrite Hermes's auth store or ``~/.codex/auth.json``. Codex refresh tokens are
single-use, so a second refresher races Hermes's locked refresh and can sign the user out. Instead the
plugin declares ``requires_auth: [openai-codex]`` (or ``xai-oauth``) in its ``plugin.yaml`` and calls
:func:`credentialed_provider_request`.
Hermes resolves the credential through its own locked-refresh path, attaches ``Authorization``
itself, sends only to that provider's origins, never follows a redirect, and hands back
status, headers and body.

The caller is identified by file location: the nearest stack frame inside an installed plugin
directory. Like capabilities this is consent and visibility, not a sandbox. In-process plugin code
can still read any file the user can.
"""

from __future__ import annotations

import json as _json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlsplit

from hermes_cli.urllib_security import url_origin


class ProviderNotSignedIn(RuntimeError):
    """The user has no usable sign-in for the provider in this Hermes profile."""


class ProviderRequestNotSent(RuntimeError):
    """Hermes could not authenticate or connect, so the request never reached the provider."""


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


def _codex_auth(url: str) -> tuple[dict[str, str], str]:
    """``(auth headers, credential base URL)`` from Hermes's own locked-refresh resolver."""
    from agent.codex_headers import codex_account_headers
    from agent.turn_failure_copy import oauth_relogin_command
    from hermes_cli.auth_codex import resolve_codex_runtime_credentials
    from hermes_cli.auth_constants import AuthError

    try:
        creds = resolve_codex_runtime_credentials()
    except AuthError as exc:
        if not exc.relogin_required:
            raise
        creds = {}
    token = str(creds.get("api_key") or "").strip()
    if not token:
        raise ProviderNotSignedIn(
            f"Not signed in to Codex in Hermes. Run `{oauth_relogin_command('openai-codex')}` "
            "or sign in to OpenAI Codex under Providers in the Hermes desktop app.")
    headers = codex_account_headers(token) if url_origin(url)[1] == "chatgpt.com" else {}
    headers["Authorization"] = f"Bearer {token}"
    return headers, str(creds.get("base_url") or "")


def _xai_auth(url: str) -> tuple[dict[str, str], str]:
    """The profile's xAI Grok sign-in, through the same resolver core's own xAI speech-to-text uses."""
    from agent.turn_failure_copy import oauth_relogin_command
    from tools.xai_http import resolve_xai_http_credentials

    creds = resolve_xai_http_credentials()
    token = str(creds.get("api_key") or "").strip()
    # The resolver falls back to XAI_API_KEY. A plugin that asked for the subscription sign-in must not
    # silently bill the user's API key instead.
    if creds.get("provider") != "xai-oauth" or not token:
        raise ProviderNotSignedIn(
            f"Not signed in to xAI Grok in Hermes. Run `{oauth_relogin_command('xai-oauth')}`. "
            "An XAI_API_KEY is never used for plugin requests.")
    return {"Authorization": f"Bearer {token}"}, str(creds.get("base_url") or "")


# provider -> (origins its token may be sent to, auth resolver). chatgpt.com is the Codex backend
# core itself calls; api.openai.com hosts the Realtime client-secret endpoint Codex clients use.
_PROVIDERS: dict[str, tuple[frozenset, Callable[[str], tuple[dict[str, str], str]]]] = {
    "openai-codex": (frozenset({("https", "chatgpt.com", 443), ("https", "api.openai.com", 443)}), _codex_auth),
    "xai-oauth": (frozenset({("https", "api.x.ai", 443)}), _xai_auth),
}

# Sign-ins with the strict contract: a URL with userinfo or a caller-set auth/routing header is refused
# instead of overwritten, and every failure is a fresh exception that says whether the request left
# Hermes. openai-codex keeps the contract its existing plugins were written against.
_STRICT_PROVIDERS = frozenset({"xai-oauth"})
_HOST_OWNED_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie", "host"})
_RESOLVER_ERRORS = (RuntimeError, OSError, ValueError, LookupError, TypeError)


def provider_request_capabilities() -> dict[str, Any]:
    """What :func:`credentialed_provider_request` supports in this Hermes, for feature detection.

    ``providers`` are the Hermes sign-ins it can attach (not whether this profile is signed in);
    ``multipart`` means it accepts ``data=``/``files=``. Reads no auth store, refreshes nothing and
    makes no request, so a settings screen can call it freely.
    """
    return {"providers": sorted(_PROVIDERS), "multipart": True}


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


def _refuse_caller_credentials(provider: str, url: str, headers: Optional[Mapping[str, str]]) -> None:
    """Strict providers: the plugin may not bring its own credential, cookie or routing."""
    parsed = urlsplit(url)
    if parsed.username is not None or parsed.password is not None:
        raise PermissionError(f"Refusing a {provider} URL that carries its own credentials")
    if owned := sorted(name for name in (headers or {}) if str(name).lower() in _HOST_OWNED_HEADERS):
        raise PermissionError(f"Hermes sets {', '.join(owned)} for {provider} requests; remove it from headers")


def _resolve_refusal(provider: str, exc: Exception, strict: bool) -> Optional[BaseException]:
    """What the plugin receives when resolving the sign-in failed; None re-raises *exc* as-is."""
    import httpx

    if isinstance(exc, httpx.HTTPError):  # a resolver's own refresh POST failed: its request holds the refresh token
        if strict:
            return ProviderRequestNotSent(
                f"Refreshing the {provider} sign-in failed ({type(exc).__name__}); the request was not sent")
        return type(exc)(str(exc))
    if isinstance(exc, ProviderNotSignedIn) or (not strict and isinstance(exc, _RESOLVER_ERRORS)):
        return _detached(exc)
    if strict:
        return ProviderRequestNotSent(
            f"Resolving the {provider} sign-in failed ({type(exc).__name__}); the request was not sent")
    return None


def _send_refusal(provider: str, exc: Exception, strict: bool) -> Optional[BaseException]:
    """What the plugin receives when sending failed; None re-raises *exc* as-is."""
    import httpx

    if not strict:
        return type(exc)(str(exc)) if isinstance(exc, httpx.HTTPError) else None
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return ProviderRequestNotSent(f"Could not connect to {provider} ({type(exc).__name__}); the request was not sent")
    # Bytes may have reached the provider, so a non-idempotent request (an upload, a paid job) is not retried.
    return RuntimeError(f"The {provider} request failed ({type(exc).__name__}) after it may have reached the "
                        "provider; Hermes did not retry it")


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
    provider: str, method: str, url: str, *, json: Any = None,
    data: Optional[Mapping[str, str]] = None, files: Any = None,
    headers: Optional[Mapping[str, str]] = None, timeout: float = 30.0,
) -> ProviderResponse:
    """Send ``method url`` with the user's *provider* sign-in attached by Hermes.

    *provider* is a Hermes sign-in (listed under ``requires_auth``) or an ``id`` the calling plugin
    declares under ``oauth:`` (:mod:`hermes_cli.plugin_oauth`); the plugin's own entry wins.
    Refuses (``PermissionError``) a caller outside an installed plugin, a plugin whose ``plugin.yaml``
    does not list *provider* under ``requires_auth``, and any origin outside the provider's
    allowlist; raises :class:`ProviderNotSignedIn` when this profile has no sign-in. Redirects are
    returned as-is, never followed. The result carries no credential.

    The body is ``json=`` or a multipart form (``data=`` fields, then ``files=`` in httpx's format),
    never both. For ``xai-oauth`` a failure before the request leaves Hermes raises
    :class:`ProviderRequestNotSent`; a failure after it may have reached the provider raises
    ``RuntimeError`` and is never retried.
    """
    from hermes_cli.plugin_oauth import _plugin_oauth_spec

    if json is not None and (data is not None or files is not None):
        raise ValueError("Pass json= or a multipart body (data=/files=), not both")
    plugin_dir = _calling_plugin_dir()
    if plugin_dir is None:
        raise PermissionError("credentialed_provider_request must be called from an installed plugin's code")
    # The plugin's own oauth: entry first, so a core provider added later under the same id cannot take it over.
    spec = _plugin_oauth_spec(plugin_dir, provider)
    strict = False
    if spec is None:
        spec = _PROVIDERS.get(provider)
        if spec is None:
            raise ValueError(f"No signed-in requests for provider {provider!r} (supported: {', '.join(_PROVIDERS)}, "
                             "or an id the plugin declares under oauth: in plugin.yaml)")
        if provider not in _declared_providers(plugin_dir):
            raise PermissionError(
                f"Plugin '{plugin_dir.name}' must declare `requires_auth: [{provider}]` in plugin.yaml")
        strict = provider in _STRICT_PROVIDERS
    origins, resolve = spec
    if url_origin(url) not in origins:
        raise PermissionError(f"Refusing to send the {provider} sign-in to {url_origin(url)[1] or url!r}")
    if strict:
        _refuse_caller_credentials(provider, url, headers)
    import httpx

    # Every error the plugin receives past this point is rebuilt without traceback, chain or request: those
    # reach the frames and httpx objects that hold the token, refresh token or client secret.
    refusal: Optional[BaseException] = None
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        request = client.build_request(method, url, json=json, data=data, files=files,
                                       headers=dict(headers or {}))  # no credential yet
        try:
            auth_headers, credential_base = resolve(url)
        except Exception as exc:  # health: allow BLE001 -- each type is mapped in _resolve_refusal; None re-raises
            refusal = _resolve_refusal(provider, exc, strict)
            if refusal is None:
                raise
        else:
            if url_origin(credential_base) not in origins:
                # A pooled gateway key belongs to its own host only (#121486).
                refusal = PermissionError(
                    f"This profile's {provider} credential routes to a custom endpoint; refusing")
            else:
                for name in auth_headers:
                    request.headers.pop(name, None)
                request.headers.update(auth_headers)
                try:
                    response = client.send(request)
                except Exception as exc:  # health: allow BLE001 -- each type is mapped in _send_refusal; None re-raises
                    refusal = _send_refusal(provider, exc, strict)
                    if refusal is None:
                        raise
            auth_headers = request = None
    if refusal is not None:
        raise refusal
    return ProviderResponse(response.status_code, dict(response.headers), response.content)


__all__ = [
    "ProviderNotSignedIn", "ProviderRequestNotSent", "ProviderResponse", "credentialed_provider_request",
    "declared_auth_providers", "provider_request_capabilities", "requires_auth_notice",
]
