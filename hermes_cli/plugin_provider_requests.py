"""Signed-in provider requests that core makes FOR plugins; the token never reaches plugin code.

A plugin that needs the user's subscription sign-in (Codex / ChatGPT, or xAI Grok OAuth) must not
read, refresh or rewrite Hermes's auth store or the vendor CLI's token file (``~/.codex/auth.json``).
Those refresh tokens are single-use, so a second refresher races Hermes's locked refresh and can sign
the user out. Instead the plugin declares ``requires_auth: [openai-codex]`` (or ``[xai-oauth]``) in
its ``plugin.yaml`` and calls :func:`credentialed_provider_request`. Hermes resolves the credential
through its own profile-aware, locked-refresh path, attaches ``Authorization`` itself, sends only to
that provider's origins, never follows a redirect, and hands back status, headers and body.

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


# One message for both failure points: no stored grant, or a grant with no access token.
def _xai_not_signed_in() -> ProviderNotSignedIn:
    from agent.turn_failure_copy import oauth_relogin_command

    return ProviderNotSignedIn(
        "No usable xAI Grok sign-in in this Hermes profile. Run "
        f"`{oauth_relogin_command('xai-oauth')}` or sign in to xAI Grok OAuth under Providers in "
        "the Hermes desktop app.")


def _xai_auth(url: str) -> tuple[dict[str, str], str]:
    """``(auth headers, credential base URL)`` from Hermes's own locked xAI resolver.

    Read-mostly, matching inference: the resolver refreshes only a token that is already inside its
    proactive-skew window, under the auth-store lock and re-reading before rotating, so a usage read
    never forces a rotation and never starts a sign-in. ``AuthError`` is rebuilt without its message
    because a refresh failure can echo the token endpoint's response body.
    """
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.auth_xai import resolve_xai_oauth_runtime_credentials

    del url  # No host-specific account headers, unlike Codex's chatgpt.com backend.
    try:
        creds = resolve_xai_oauth_runtime_credentials(force_refresh=False, refresh_if_expiring=True)
    except AuthError:
        raise _xai_not_signed_in() from None
    token = str(creds.get("api_key") or "").strip()
    if not token:
        raise _xai_not_signed_in()
    return {"Authorization": f"Bearer {token}"}, str(creds.get("base_url") or "")


# provider -> (origins its token may be sent to, auth resolver). chatgpt.com is the Codex backend
# core itself calls; api.openai.com hosts the Realtime client-secret endpoint Codex clients use.
# For xAI, cli-chat-proxy.grok.com and grok.com are the Grok CLI's credits/settings hosts, and
# api.x.ai is the OAuth bearer's own inference host (also the default credential base URL, so the
# custom-endpoint guard below can still run).
_PROVIDERS: dict[str, tuple[frozenset, Callable[[str], tuple[dict[str, str], str]]]] = {
    "openai-codex": (frozenset({("https", "chatgpt.com", 443), ("https", "api.openai.com", 443)}), _codex_auth),
    "xai-oauth": (frozenset({("https", "cli-chat-proxy.grok.com", 443), ("https", "grok.com", 443),
                             ("https", "api.x.ai", 443)}), _xai_auth),
}


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
    caller's own ``Content-Type``. Caller headers never override ``Authorization``, ``Host`` or
    ``Content-Length``.
    """
    from hermes_cli.plugin_oauth import _plugin_oauth_spec

    if json is not None and content is not None:
        raise ValueError("credentialed_provider_request takes json= or content=, not both")
    plugin_dir = _calling_plugin_dir()
    if plugin_dir is None:
        raise PermissionError("credentialed_provider_request must be called from an installed plugin's code")
    # The plugin's own oauth: entry first, so a core provider added later under the same id cannot take it over.
    spec = _plugin_oauth_spec(plugin_dir, provider)
    if spec is None:
        spec = _PROVIDERS.get(provider)
        if spec is None:
            raise ValueError(f"No signed-in requests for provider {provider!r} (supported: {', '.join(_PROVIDERS)}, "
                             "or an id the plugin declares under oauth: in plugin.yaml)")
        if provider not in _declared_providers(plugin_dir):
            raise PermissionError(
                f"Plugin '{plugin_dir.name}' must declare `requires_auth: [{provider}]` in plugin.yaml")
    origins, resolve = spec
    if _has_userinfo(url):
        raise PermissionError(f"Refusing a {provider} request URL with embedded credentials")
    try:
        origin = url_origin(url)
    except ValueError:  # malformed port: fail closed with the same refusal the caller handles
        raise PermissionError(f"Refusing a malformed {provider} request URL") from None
    if origin not in origins:
        raise PermissionError(f"Refusing to send the {provider} sign-in to {origin[1] or url!r}")
    import httpx

    # Every error the plugin receives past this point is rebuilt without traceback, chain or request: those
    # reach the frames and httpx objects that hold the token, refresh token or client secret.
    refusal: Optional[BaseException] = None
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        # Host and Content-Length must derive from the URL and body, never from caller input.
        extra = {name: value for name, value in dict(headers or {}).items()
                 if name.lower() not in {"host", "content-length"}}
        request = client.build_request(method, url, json=json, content=content, headers=extra)  # no credential yet
        try:
            auth_headers, credential_base = resolve(url)
        except httpx.HTTPError as exc:  # a resolver's own refresh POST failed: its request holds the refresh token
            refusal = type(exc)(str(exc))
        except (RuntimeError, OSError, ValueError, LookupError, TypeError) as exc:
            refusal = _detached(exc)
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
                except httpx.HTTPError as exc:
                    refusal = type(exc)(str(exc))
            auth_headers = request = None
    if refusal is not None:
        raise refusal
    return ProviderResponse(response.status_code, dict(response.headers), response.content)


__all__ = [
    "ProviderNotSignedIn", "ProviderResponse", "credentialed_provider_request", "declared_auth_providers",
    "requires_auth_notice",
]
