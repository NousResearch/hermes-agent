"""OAuth clients a plugin declares in ``plugin.yaml``: Hermes signs in, stores and refreshes the grant.

A plugin that calls a third-party API with the user's own OAuth client (a Google Cloud client, for example)
declares it under ``oauth:``. ``hermes auth add <plugin>:<id>`` runs the PKCE login of
:mod:`hermes_cli.auth_oauth_pkce_plugin` in a terminal, the grant lives in the active profile's
``auth.json`` (section ``plugin_oauth``) and Hermes refreshes it. The plugin calls
``credentialed_provider_request(<id>, ...)``: Hermes attaches the bearer for the declared hosts only, so
the plugin never holds a token (#139077; the same contract as the Codex lane, #136117).

Each grant remembers a fingerprint of the ``oauth:`` entry it was made for. A later version of the plugin
that changes the token URL, hosts, scopes or client variables gets ``ProviderNotSignedIn`` until the user
signs in again: the grant is used only for what the user consented to. Like capabilities, this is
consent and visibility, not a sandbox.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse

_SECTION = "plugin_oauth"
_EXPIRY_MARGIN_MS = 120_000  # refresh this long before expiry, as the credential pool does
_TOKEN_TIMEOUT = 10.0  # below AUTH_LOCK_TIMEOUT_SECONDS: a peer waiting for the refresh lock outlives the POST
_CLIENT_ERROR_CODES = frozenset({"invalid_client", "unauthorized_client", "invalid_scope"})
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
# Lowercase DNS name with at least one dot whose last label starts with a letter: no IP literal, port or scheme.
_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z](?:[a-z0-9-]*[a-z0-9])?$")


@dataclass(frozen=True)
class OAuthClient:
    """One ``oauth:`` entry of a plugin manifest. Holds variable names, never their values."""

    id: str
    authorize_url: str
    token_url: str
    scopes: tuple[str, ...]
    client_id_env: str
    hosts: tuple[str, ...]
    client_secret_env: str = ""
    authorize_params: tuple[tuple[str, str], ...] = ()

    @property
    def fingerprint(self) -> str:
        """Hash of every field: a grant is used only with the entry it was made for."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()


# ── Manifest ───────────────────────────────────────────────────────────────────────────────────────

def _parse_url(url: Any):
    """``urlparse`` of *url*; None for a non-string or a URL it cannot read (bad IPv6 literal or port)."""
    if not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url)
        _ = parsed.port  # urlparse checks the port only when it is read
    except ValueError:
        return None
    return parsed


def _url_problems(where: str, url: Any) -> list[str]:
    parsed = _parse_url(url)
    if (parsed is None or not parsed.hostname or parsed.username is not None or parsed.password is not None
            or parsed.fragment):
        return [f"{where}: must be an absolute URL without user info or fragment"]
    if parsed.scheme != "https":  # no loopback exception: the code, refresh token and client secret travel here
        return [f"{where}: must use https"]
    return []


def _hermes_owned_env(name: str) -> bool:
    """A variable Hermes itself reads (a provider key, a platform token): never a plugin's client variable,
    or a manifest could have Hermes POST that secret to the token URL it names."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.config_defaults import OPTIONAL_ENV_VARS

    return name in OPTIONAL_ENV_VARS or any(name in p.api_key_env_vars for p in PROVIDER_REGISTRY.values())


def _entry_problems(entry: dict, where: str) -> list[str]:
    problems = [] if _ID_RE.match(str(entry.get("id") or "")) else [
        f"{where}.id: {entry.get('id')!r} must be lowercase letters, digits and dashes"]
    for field in ("authorize_url", "token_url"):
        problems += _url_problems(f"{where}.{field}", entry.get(field))
    scopes = entry.get("scopes")
    if not (isinstance(scopes, list) and scopes and all(isinstance(s, str) and s.strip() for s in scopes)):
        problems.append(f"{where}.scopes: must be a non-empty list of scope strings")
    for field, required in (("client_id_env", True), ("client_secret_env", False)):
        name = entry.get(field)
        if (required or name) and not (isinstance(name, str) and _ENV_RE.match(name)):
            problems.append(f"{where}.{field}: {name!r} must be an UPPER_SNAKE environment variable name")
        elif name and _hermes_owned_env(name):
            problems.append(f"{where}.{field}: {name} is a variable Hermes uses itself; declare the plugin's own")
    hosts = entry.get("hosts")
    if not (isinstance(hosts, list) and hosts and all(isinstance(h, str) and _HOST_RE.match(h) for h in hosts)):
        problems.append(f"{where}.hosts: must list lowercase DNS host names (no scheme, port, path or IP address)")
    params = entry.get("authorize_params") or {}
    if not (isinstance(params, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in params.items())):
        problems.append(f"{where}.authorize_params: must map strings to strings")
    return problems


def _client(entry: dict) -> OAuthClient:
    return OAuthClient(
        id=entry["id"], authorize_url=entry["authorize_url"], token_url=entry["token_url"],
        scopes=tuple(entry["scopes"]), client_id_env=entry["client_id_env"], hosts=tuple(entry["hosts"]),
        client_secret_env=entry.get("client_secret_env") or "",
        authorize_params=tuple(sorted((entry.get("authorize_params") or {}).items())))


def parse_oauth_clients(manifest: Mapping[str, Any]) -> tuple[list[OAuthClient], list[str]]:
    """``oauth:`` from a manifest dict: the well-formed clients, and one problem string per defect."""
    raw = (manifest or {}).get("oauth")
    if raw is None or raw == []:
        return [], []
    if not isinstance(raw, list):
        return [], ["oauth: must be a list of client entries"]
    clients: list[OAuthClient] = []
    problems: list[str] = []
    for index, entry in enumerate(raw):
        where = f"oauth[{index}]"
        entry_problems = _entry_problems(entry, where) if isinstance(entry, dict) else [f"{where}: must be a mapping"]
        if not entry_problems and any(c.id == entry["id"] for c in clients):
            entry_problems = [f"{where}.id: {entry['id']!r} is declared twice"]
        problems += entry_problems
        if not entry_problems:
            clients.append(_client(entry))
    return clients, problems


def _declared_clients(plugin_dir: Path) -> list[OAuthClient]:
    """The plugin's own ``oauth:`` clients; a malformed block fails loud instead of reading as undeclared."""
    from pm.plugin_declarations import native_manifest_file, read_native_manifest

    manifest_file = native_manifest_file(plugin_dir)
    clients, problems = parse_oauth_clients(read_native_manifest(manifest_file) if manifest_file else {})
    if problems:
        raise ValueError(f"Plugin '{plugin_dir.name}' has an invalid oauth: block in plugin.yaml: " + "; ".join(problems))
    return clients


def plugin_key(plugin_dir: Path) -> str:
    """The plugin's key under the plugins root it lives in (``voice``, ``web/exa``); grants are stored by it."""
    from hermes_cli.plugin_provider_requests import _plugin_roots

    for path in (plugin_dir.absolute(), plugin_dir.resolve()):
        for root in _plugin_roots():
            if path != root and path.is_relative_to(root):
                return path.relative_to(root).as_posix()
    return plugin_dir.name


def _idp_hosts(client: OAuthClient) -> str:
    return ", ".join(dict.fromkeys(urlparse(url).hostname for url in (client.authorize_url, client.token_url)))


def oauth_notice(manifest: Mapping[str, Any]) -> str:
    """Install/show text for the declared clients; empty when the plugin declares none."""
    name = str((manifest or {}).get("name") or "<plugin>")
    return "; ".join(
        f"Signs in to {c.id} at {_idp_hosts(c)} for {', '.join(c.hosts)} with `hermes auth add {name}:{c.id}` "
        "(Hermes stores and refreshes the token; the plugin never receives it)"
        for c in parse_oauth_clients(manifest)[0])


def oauth_surface(manifest: Mapping[str, Any]) -> set[str]:
    """One line per client for the re-pin consent delta: a new entry, or a changed authorize host, token host,
    hosts or scopes, shows up as added. Any other change only alters the fingerprint, so the user signs in again."""
    lines: set[str] = set()
    for c in parse_oauth_clients(manifest)[0]:
        lines.add(f"{c.id} ({_idp_hosts(c)} -> {', '.join(c.hosts)}; {' '.join(c.scopes)})")
    return lines


def _declared_env_names(manifest: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for field in ("requires_env", "optional_env"):
        for entry in manifest.get(field) or []:
            names.add(str(entry.get("name") if isinstance(entry, dict) else entry))
    return names


def oauth_admission_problems(manifest: Mapping[str, Any], plugin_dir: Path) -> list[str]:
    """Catalog admission on top of the shape: declared variables, no clash with core sign-ins or ``custom:``."""
    from hermes_cli.plugin_provider_requests import _PROVIDERS

    clients, problems = parse_oauth_clients(manifest)
    declared = _declared_env_names(manifest)
    for client in clients:
        if client.id in _PROVIDERS:
            problems.append(f"oauth id {client.id!r} is a Hermes sign-in provider; pick another id")
        for name in filter(None, (client.client_id_env, client.client_secret_env)):
            if name not in declared:
                problems.append(f"oauth {client.id!r}: {name} must be declared under requires_env or optional_env")
    if clients and plugin_dir.name == "custom":
        problems.append("a plugin named 'custom' cannot declare oauth: `hermes auth add custom:<name>` is a custom provider")
    return problems


def _parent(host: str) -> str:
    return ".".join(host.split(".")[-2:])


def oauth_admission_warnings(manifest: Mapping[str, Any], plugin_dir: Path) -> list[str]:
    """Reviewer warnings: a host outside the sign-in provider's domains (does the IdP own it?), and plugin
    code that names a declared token URL (Hermes refreshes the grant)."""
    clients = parse_oauth_clients(manifest)[0]
    warnings = [f"oauth {c.id!r}: host {host} shares no domain with {_idp_hosts(c)}; check the provider owns it"
                for c in clients for host in c.hosts
                if _parent(host) not in {_parent(urlparse(u).hostname or "") for u in (c.authorize_url, c.token_url)}]
    urls = sorted({c.token_url for c in clients})
    for path in sorted(plugin_dir.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        warnings += [f"oauth: {path.relative_to(plugin_dir).as_posix()} names the token URL {url}; Hermes "
                     "refreshes the grant, so the plugin should not call it" for url in urls if url in text]
    return warnings


# ── Grant store: section ``plugin_oauth`` of the active profile's auth.json ────────────────────────

def _read_grant(key: str) -> Optional[dict]:
    from hermes_cli.auth import _load_auth_store

    section = _load_auth_store().get(_SECTION)
    row = section.get(key) if isinstance(section, dict) else None
    return dict(row) if isinstance(row, dict) else None


def _write_grant(key: str, row: Optional[dict]) -> None:
    """Replace one grant (``None`` removes it) under the auth-store lock; other sections stay as they are."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store, _store_section

    with _auth_store_lock():
        store = _load_auth_store()
        section = _store_section(store, _SECTION)
        if row is None:
            section.pop(key, None)
        else:
            section[key] = row
        _save_auth_store(store)


# ── Token endpoint ─────────────────────────────────────────────────────────────────────────────────

def _auth_error(key: str, message: str, code: str):
    from hermes_cli.auth_constants import AuthError

    return AuthError(f"{key}: {message}", provider=key, code=code)


def _env(name: str, code: str, key: str) -> str:
    from agent.secret_scope import get_secret

    value = str(get_secret(name) or "").strip()
    if not value:
        raise _auth_error(key, f"{name} is not set. Add it to this profile's .env (the plugin declares it).", code)
    return value


def _json_object(response: Any) -> dict:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _token_error(key: str, status: int, payload: Mapping[str, Any]):
    """Keep the OAuth ``error`` code a caller can act on; never echo the response body."""
    from hermes_cli.auth import _OAUTH_GRANT_DEAD_CODES

    error = str(payload.get("error") or "").strip()
    if error in _OAUTH_GRANT_DEAD_CODES or error in _CLIENT_ERROR_CODES:
        return _auth_error(key, f"the token endpoint refused the request (HTTP {status}, {error}).", error)
    return _auth_error(key, f"the token endpoint failed with HTTP {status}.", "oauth_refresh_failed")


def _require_granted_scopes(client: OAuthClient, payload: Mapping[str, Any], key: str) -> None:
    granted = payload.get("scope")
    if not isinstance(granted, str) or not granted.strip():
        return  # the IdP did not say which scopes it granted
    missing = [scope for scope in client.scopes if scope not in granted.split()]
    if missing:
        raise _auth_error(key, "not every permission was granted, so nothing was saved. Missing: "
                          + ", ".join(missing), "oauth_scope_not_granted")


def _token_request(client: OAuthClient, key: str) -> Callable[[dict[str, str]], Mapping[str, Any]]:
    """``OAuthPKCEConfig.token_request``: the client credentials are read right before each POST, so the
    secret never sits in the frozen config, whose repr would show it."""

    def post(fields: dict[str, str]) -> Mapping[str, Any]:
        from hermes_cli.auth import _default_verify
        from hermes_cli.auth_constants import httpx

        form = {**fields, "client_id": _env(client.client_id_env, "oauth_client_id_missing", key)}
        if client.client_secret_env:
            form["client_secret"] = _env(client.client_secret_env, "oauth_client_secret_missing", key)
        failure = ""
        try:
            response = httpx.post(client.token_url, data=form, headers={"Accept": "application/json"},
                                  timeout=_TOKEN_TIMEOUT, verify=_default_verify(), follow_redirects=False)
        except httpx.HTTPError as exc:
            failure = type(exc).__name__
        if failure:  # raised outside the handler: httpx's error keeps the request, whose body holds the secrets
            raise _auth_error(key, f"the token request failed ({failure}).", "oauth_refresh_failed")
        payload = _json_object(response)
        if not 200 <= response.status_code < 300:  # a redirect is a failure too: it is never followed
            raise _token_error(key, response.status_code, payload)
        if fields.get("grant_type") == "authorization_code":
            _require_granted_scopes(client, payload, key)
        return payload

    return post


def _pkce_config(client: OAuthClient, key: str, *, timeout: Optional[float] = None):
    from hermes_cli.auth_oauth_pkce_plugin import OAuthPKCEConfig

    return OAuthPKCEConfig(
        client_id=_env(client.client_id_env, "oauth_client_id_missing", key), authorize_url=client.authorize_url,
        token_url=client.token_url, scopes=client.scopes, extra_authorize_params=dict(client.authorize_params),
        label=key, token_request=_token_request(client, key), timeout_seconds=timeout or 180.0)


# ── Bearer for credentialed_provider_request ───────────────────────────────────────────────────────

def _not_signed_in(message: str):
    from hermes_cli.plugin_provider_requests import ProviderNotSignedIn

    return ProviderNotSignedIn(message)


def _revoked(client: OAuthClient, key: str, code: str):
    return _not_signed_in(f"The {client.id} sign-in for plugin '{key.rsplit(':', 1)[0]}' was revoked ({code}). "
                          f"Run `hermes auth add {key}` to sign in again.")


def _usable_grant(row: Optional[dict], client: OAuthClient, key: str) -> dict:
    plugin = key.rsplit(":", 1)[0]
    if row is None:
        raise _not_signed_in(f"Not signed in to {client.id} for plugin '{plugin}' in this Hermes profile. "
                             f"Run `hermes auth add {key}`.")
    if row.get("dead"):
        raise _revoked(client, key, str(row["dead"]))
    if row.get("fingerprint") != client.fingerprint:
        raise _not_signed_in(f"Plugin '{plugin}' changed its oauth entry '{client.id}' since you signed in. "
                             f"Run `hermes auth add {key}` to sign in again.")
    return row


def _live_token(row: Mapping[str, Any]) -> str:
    token = str(row.get("access_token") or "")
    expires = row.get("expires_at_ms")
    live = expires is None or int(expires) > int(time.time() * 1000) + _EXPIRY_MARGIN_MS
    return token if token and live else ""


def _refresh(client: OAuthClient, key: str, row: dict) -> str:
    """Rotate the grant; the caller holds the auth-store lock, so a peer process waits instead of spending
    the same refresh token, and a logout cannot be undone by a refresh in flight."""
    from hermes_cli.auth import _OAUTH_GRANT_DEAD_CODES
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.auth_oauth_pkce_plugin import _post_token

    refresh_token = str(row.get("refresh_token") or "")
    if not refresh_token:
        raise _revoked(client, key, "no_refresh_token")
    try:
        fields = _post_token(key, _pkce_config(client, key),
                             {"grant_type": "refresh_token", "refresh_token": refresh_token}, code="oauth_refresh_failed")
    except AuthError as exc:
        if exc.code in _OAUTH_GRANT_DEAD_CODES:
            _write_grant(key, {**row, "dead": exc.code})
            raise _revoked(client, key, str(exc.code)) from None
        raise
    _write_grant(key, {**row, **fields})
    return str(fields["access_token"])


def _bearer(plugin_dir: Path, client: OAuthClient) -> str:
    """A live access token for the plugin's grant. Reading takes no lock (writes are atomic); a refresh
    re-reads under the lock, because a peer may have rotated the grant meanwhile."""
    from hermes_cli.auth import _auth_store_lock

    key = f"{plugin_key(plugin_dir)}:{client.id}"
    token = _live_token(_usable_grant(_read_grant(key), client, key))
    if token:
        return token
    with _auth_store_lock():
        row = _usable_grant(_read_grant(key), client, key)
        return _live_token(row) or _refresh(client, key, row)


def _plugin_oauth_spec(plugin_dir: Path, provider: str):
    """``(origins, resolver)`` for *provider* when the calling plugin declares it under ``oauth:``, else None.

    Core-only, like the ``_PROVIDERS`` resolvers: the resolver returns the bearer, so plugins go through
    ``credentialed_provider_request``, which attributes the caller and checks the origin first."""
    client = next((c for c in _declared_clients(plugin_dir) if c.id == provider), None)
    if client is None:
        return None

    def resolve(url: str) -> tuple[dict[str, str], str]:
        parsed = urlparse(url)
        if parsed.username is not None or parsed.password is not None:
            raise PermissionError(f"Refusing to send the {client.id} sign-in to a URL with user info")
        # The grant is valid for every declared host, so the request URL (origin already checked) is its base.
        return {"Authorization": f"Bearer {_bearer(plugin_dir, client)}"}, url

    return _origins(client), resolve


def _origins(client: OAuthClient) -> frozenset:
    """Origins the bearer may go to: each declared host, HTTPS on the default port."""
    return frozenset(("https", host, 443) for host in client.hosts)


# ── hermes auth add|status|refresh|logout <plugin>:<id> ────────────────────────────────────────────

def _installed_client(plugin: str, oauth_id: str) -> Optional[tuple[Path, OAuthClient]]:
    """Resolve *plugin* the way ``hermes plugins show`` does, then its ``oauth:`` entry *oauth_id*."""
    from hermes_cli.plugins_cmd import _find_plugin_entry

    entry = _find_plugin_entry(plugin)
    if entry is None or not entry[4]:
        return None
    plugin_dir = Path(str(entry[4]))
    client = next((c for c in _declared_clients(plugin_dir) if c.id == oauth_id), None)
    return (plugin_dir, client) if client is not None else None


def _cli_add(args: Any, client: OAuthClient, key: str) -> bool:
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.auth_oauth_pkce_plugin import login

    try:
        fields = login(key, _pkce_config(client, key, timeout=getattr(args, "timeout", None)),
                       open_browser=not getattr(args, "no_browser", False))
    except AuthError as exc:
        raise SystemExit(f"Login failed: {exc}") from None
    _write_grant(key, {**fields, "fingerprint": client.fingerprint})
    print(f"Signed in to {client.id} for plugin '{key.rsplit(':', 1)[0]}'; the grant is stored in this Hermes profile.")
    return True


def _cli_status(args: Any, client: OAuthClient, key: str) -> bool:
    from hermes_cli.plugin_provider_requests import ProviderNotSignedIn

    row = _read_grant(key)
    try:
        _usable_grant(row, client, key)
    except ProviderNotSignedIn as exc:
        state = "logged out" if row is None else "revoked" if row.get("dead") else "sign-in needed"
        print(f"{key}: {state}\n  {exc}")
        return True
    expires = row.get("expires_at_ms") if row else None
    print(f"{key}: logged in\n  hosts: {', '.join(client.hosts)}" + (
        f"\n  access token valid until {datetime.fromtimestamp(int(expires) / 1000, UTC).isoformat(timespec='seconds')}"
        if expires else ""))
    return True


def _cli_refresh(args: Any, client: OAuthClient, key: str) -> bool:
    from hermes_cli.auth import _auth_store_lock
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.plugin_provider_requests import ProviderNotSignedIn

    try:
        with _auth_store_lock():
            _refresh(client, key, _usable_grant(_read_grant(key), client, key))
    except (AuthError, ProviderNotSignedIn) as exc:
        raise SystemExit(f"Refresh failed: {exc}") from None
    print(f"{key}: refreshed")
    return True


def _cli_logout(args: Any, client: OAuthClient, key: str) -> bool:
    _write_grant(key, None)
    print(f"Signed out of {client.id} for plugin '{key.rsplit(':', 1)[0]}'; the grant was removed from this profile.")
    return True


_CLI_ACTIONS = {"add": _cli_add, "status": _cli_status, "refresh": _cli_refresh, "logout": _cli_logout}


def handle_auth_action(action: str, args: Any, provider: str) -> Optional[bool]:
    """``hermes auth add|status|refresh|logout <plugin>:<id>``; None when *provider* is not a plugin's
    ``oauth:`` entry (``custom:<name>`` stays the custom-provider pool)."""
    plugin, sep, oauth_id = str(provider or "").partition(":")
    handler = _CLI_ACTIONS.get(action)
    if handler is None or not (sep and plugin and oauth_id) or plugin == "custom":
        return None
    try:
        found = _installed_client(plugin, oauth_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if found is None:
        return None
    plugin_dir, client = found
    return handler(args, client, f"{plugin_key(plugin_dir)}:{client.id}")


__all__ = [
    "OAuthClient", "handle_auth_action", "oauth_admission_problems", "oauth_admission_warnings",
    "oauth_notice", "oauth_surface", "parse_oauth_clients", "plugin_key",
]
