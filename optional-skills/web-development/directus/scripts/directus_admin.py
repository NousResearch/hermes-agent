#!/usr/bin/env python3
"""Directus 11+ administration and CRUD over the REST API. Stdlib only.

Everything goes through one authenticated HTTP client against a Directus
instance (``--url`` / ``DIRECTUS_URL``) using a static token
(``DIRECTUS_TOKEN``) or an email/password login that is exchanged for a
short-lived access token and never written to disk.

Directus 11 replaced role-based permissions with a policy-based model:

    Role --(directus_access)--> Policy --(directus_permissions)--> Collection/action

A role by itself grants nothing. Permissions hang off a *policy*; the policy
reaches a role (or a user, or the public) through the ``directus_access``
junction. Commands here follow that chain literally, and ``bootstrap-agent``
walks it end to end for one service account.

Subcommands
-----------
  check                      reachable? authenticated as whom? which version?
  collections                list / create / delete a collection
  fields                     list / create a field on a collection
  items                      list / get / create / update / delete rows
  policies                   list / create an access policy
  permissions                list / create a permission rule on a policy
  roles                      list / create a role
  access                     list / grant / revoke a role-to-policy link
  users                      list / create a user
  token                      set or clear a user's static token
  bootstrap-agent            policy + permissions + role + access + user + token

Settings resolve flag > environment > default:

  --url    / DIRECTUS_URL        (required; skill config key ``directus.url``)
  --token  / DIRECTUS_TOKEN      static token, or use --auth-email/--auth-password
           / DIRECTUS_EMAIL, DIRECTUS_PASSWORD
  --timeout / DIRECTUS_TIMEOUT   default 30 seconds

Output is JSON on stdout; progress and warnings go to stderr. Every failure is
one ``error:`` line and exit 2 — no tracebacks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_TIMEOUT = 30.0
MIN_MAJOR_VERSION = 11
PAGE_SIZE = 100
MAX_PAGES = 1000
RETRY_STATUS = (429, 500, 502, 503, 504)
RETRY_BACKOFF = (1.0, 3.0, 6.0)

# System collections a caller should not be able to wipe by typo. Directus
# itself refuses most of these, but the check happens before the request so
# the agent gets an explanation instead of a 403.
SYSTEM_PREFIX = "directus_"

FIELD_TYPES = (
    "string", "text", "integer", "bigInteger", "float", "decimal", "boolean",
    "date", "time", "dateTime", "timestamp", "json", "uuid", "hash", "csv",
)

ACTIONS = ("create", "read", "update", "delete", "share")


class DirectusError(RuntimeError):
    """Any failure the agent should report verbatim to the user."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def resolve_setting(flag: Any, env_name: str, default: Any = None, cast: Callable[[Any], Any] = str) -> Any:
    """Flag wins, then the environment variable, then the default."""
    if flag is not None and flag != "":
        return cast(flag)
    env_value = os.environ.get(env_name, "")
    if env_value.strip():
        return cast(env_value.strip())
    return None if default is None else cast(default)


def normalise_url(url: str) -> str:
    """``cms.example.com/`` -> ``https://cms.example.com``; localhost stays http."""
    url = (url or "").strip().rstrip("/")
    if not url:
        raise DirectusError(
            "No Directus URL. Pass --url, set DIRECTUS_URL, or set the `directus.url` skill setting."
        )
    if not url.startswith(("http://", "https://")):
        host = url.split("/", 1)[0].split(":", 1)[0]
        scheme = "http://" if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") else "https://"
        url = scheme + url
    return url.rstrip("/")


def parse_json_arg(raw: Optional[str], label: str) -> Any:
    """``--data`` / ``--filter`` accept inline JSON, ``@file``, or ``-`` for stdin."""
    if raw is None:
        return None
    text = raw.strip()
    if text == "-":
        text = sys.stdin.read().strip()
    elif text.startswith("@"):
        path = text[1:]
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read().strip()
        except OSError as exc:
            raise DirectusError(f"Cannot read {label} file {path}: {exc}") from exc
    if not text:
        raise DirectusError(f"Empty {label}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise DirectusError(f"{label} is not valid JSON ({exc.msg} at char {exc.pos}): {text[:120]}") from exc


def split_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    parts = [item.strip() for item in raw.split(",")]
    return [item for item in parts if item]


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class DirectusClient:
    """Minimal REST client. ``opener`` and ``sleep`` are injectable for tests."""

    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        opener: Optional[Callable[..., Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = normalise_url(url)
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep
        self._token = token or None
        self._email = email
        self._password = password
        self._version: Optional[str] = None
        if not self._token and not (email and password):
            raise DirectusError(
                "No credentials. Set DIRECTUS_TOKEN (static token from the user's profile), "
                "or pass --auth-email and --auth-password / set DIRECTUS_EMAIL and DIRECTUS_PASSWORD."
            )

    # -- auth ---------------------------------------------------------------

    def _ensure_token(self) -> str:
        if self._token:
            return self._token
        payload = {"email": self._email, "password": self._password}
        body = self._raw("POST", "/auth/login", payload, authenticated=False)
        token = ((body or {}).get("data") or {}).get("access_token")
        if not token:
            raise DirectusError("Login succeeded but returned no access_token; use a static token instead.")
        self._token = str(token)
        return self._token

    # -- transport ----------------------------------------------------------

    def _raw(
        self,
        method: str,
        path: str,
        payload: Any = None,
        params: Optional[Dict[str, Any]] = None,
        authenticated: bool = True,
    ) -> Dict[str, Any]:
        url = self.url + path
        if params:
            query = {k: v for k, v in params.items() if v is not None}
            if query:
                url += "?" + urllib.parse.urlencode(query)
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if authenticated:
            headers["Authorization"] = "Bearer " + self._ensure_token()
        request = urllib.request.Request(url, data=data, headers=headers, method=method)

        last_error: Optional[Exception] = None
        for attempt in range(len(RETRY_BACKOFF) + 1):
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    text = response.read().decode("utf-8")
                return json.loads(text) if text.strip() else {}
            except urllib.error.HTTPError as exc:
                detail = _read_error(exc)
                if exc.code in RETRY_STATUS and attempt < len(RETRY_BACKOFF):
                    self._sleep(RETRY_BACKOFF[attempt])
                    last_error = exc
                    continue
                raise DirectusError(_explain_http(exc.code, detail, method, path)) from exc
            except urllib.error.URLError as exc:
                if attempt < len(RETRY_BACKOFF):
                    self._sleep(RETRY_BACKOFF[attempt])
                    last_error = exc
                    continue
                raise DirectusError(
                    f"Cannot reach Directus at {self.url} ({exc.reason}). Check the URL, that the "
                    f"container is up, and that this machine can route to it."
                ) from exc
            except json.JSONDecodeError as exc:
                raise DirectusError(
                    f"Directus returned non-JSON for {method} {path}; is {self.url} really a Directus API "
                    f"root and not a reverse-proxy error page?"
                ) from exc
        raise DirectusError(f"{method} {path} failed after retries: {last_error}")

    def request(self, method: str, path: str, payload: Any = None, params: Optional[Dict[str, Any]] = None) -> Any:
        body = self._raw(method, path, payload, params)
        if isinstance(body, dict) and "data" in body:
            return body["data"]
        return body

    def request_with_meta(self, path: str, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        body = self._raw("GET", path, None, params)
        data = body.get("data") if isinstance(body, dict) else body
        meta = body.get("meta") or {} if isinstance(body, dict) else {}
        return data, meta

    # -- convenience --------------------------------------------------------

    def server_info(self) -> Dict[str, Any]:
        return self.request("GET", "/server/info") or {}

    def version(self) -> str:
        if self._version is None:
            info = self.server_info()
            self._version = str(info.get("version") or "")
        return self._version

    def require_v11(self, what: str) -> None:
        """Policies/access exist only from Directus 11; fail with the v10 name instead of a 404."""
        major = major_version(self.version())
        if major is not None and major < MIN_MAJOR_VERSION:
            raise DirectusError(
                f"{what} needs Directus {MIN_MAJOR_VERSION}+ (this server reports {self.version()}). "
                f"On Directus 10 permissions attach directly to a role, so use that server's admin app."
            )

    def me(self) -> Dict[str, Any]:
        params = {"fields": "id,email,first_name,last_name,role.id,role.name,role.admin_access"}
        return self.request("GET", "/users/me", params=params) or {}

    def paginate(self, path: str, params: Dict[str, Any], page_size: int = PAGE_SIZE) -> List[Any]:
        """Walk ``limit``/``offset`` until a short page comes back."""
        rows: List[Any] = []
        offset = 0
        for _ in range(MAX_PAGES):
            page_params = dict(params)
            page_params["limit"] = page_size
            page_params["offset"] = offset
            data, _meta = self.request_with_meta(path, page_params)
            batch = data or []
            rows.extend(batch)
            if len(batch) < page_size:
                return rows
            offset += page_size
        raise DirectusError(
            f"Stopped after {MAX_PAGES * page_size} rows from {path}; narrow the query with --filter."
        )


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:  # pragma: no cover - best effort
        return ""
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:300]
    errors = body.get("errors") if isinstance(body, dict) else None
    if isinstance(errors, list) and errors:
        parts = []
        for item in errors[:3]:
            message = str(item.get("message", "")).strip()
            code = str(((item.get("extensions") or {}).get("code") or "")).strip()
            parts.append(f"{message} [{code}]" if code else message)
        return "; ".join(part for part in parts if part)
    return raw[:300]


def _explain_http(code: int, detail: str, method: str, path: str) -> str:
    """Directus' own message plus the one thing that usually causes that status."""
    hints = {
        401: "The token is missing, expired, or belongs to a deleted user. Static tokens live on the "
             "user record (users → Token); regenerate it with `token set`.",
        403: "Authenticated but not permitted. In Directus 11 permissions come from a POLICY reached "
             "through directus_access — a role alone grants nothing. Check `access list --role <id>` "
             "and `permissions list --policy <id>`.",
        404: "No such route or item. Collection names are case-sensitive and a hidden/unregistered "
             "table is invisible to /items until it has a directus_collections row.",
        422: "Directus rejected the payload — usually a field that does not exist on the collection, "
             "or a required field left out.",
    }
    hint = hints.get(code, "")
    message = f"Directus returned HTTP {code} for {method} {path}: {detail or '(no body)'}"
    return f"{message}. {hint}" if hint else message


def major_version(version: str) -> Optional[int]:
    match = re.match(r"\s*v?(\d+)", version or "")
    return int(match.group(1)) if match else None


def log(message: str) -> None:
    print(f"[directus] {message}", file=sys.stderr)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def emit(payload: Any) -> int:
    print(json.dumps(payload, indent=2, sort_keys=False, default=str))
    return 0


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def parse_field_spec(spec: str) -> Dict[str, Any]:
    """``agent_id:string`` / ``payload:json`` / ``score:integer!`` (``!`` = required)."""
    raw = spec.strip()
    required = raw.endswith("!")
    if required:
        raw = raw[:-1]
    name, _, type_name = raw.partition(":")
    name = name.strip()
    type_name = (type_name or "string").strip()
    if not name:
        raise DirectusError(f"Field spec {spec!r} has no name; use name:type, e.g. agent_id:string")
    if type_name not in FIELD_TYPES:
        raise DirectusError(
            f"Unknown field type {type_name!r} in {spec!r}. Known types: {', '.join(FIELD_TYPES)}"
        )
    return {"field": name, "type": type_name, "required": required}


def field_payload(field: Dict[str, Any]) -> Dict[str, Any]:
    name, type_name, required = field["field"], field["type"], field.get("required", False)
    interface = {
        "text": "input-multiline",
        "json": "input-code",
        "boolean": "boolean",
        "date": "datetime",
        "time": "datetime",
        "dateTime": "datetime",
        "timestamp": "datetime",
    }.get(type_name, "input")
    return {
        "field": name,
        "type": type_name,
        "meta": {"interface": interface, "required": required, "hidden": False},
        "schema": {"is_nullable": not required},
    }


def primary_key_field(kind: str) -> Dict[str, Any]:
    """Directus needs the primary key declared when a collection is created."""
    if kind == "auto":
        return {
            "field": "id",
            "type": "integer",
            "meta": {"hidden": True, "interface": "input", "readonly": True},
            "schema": {"is_primary_key": True, "has_auto_increment": True},
        }
    return {
        "field": "id",
        "type": "uuid",
        "meta": {"hidden": True, "interface": "input", "readonly": True, "special": ["uuid"]},
        "schema": {"is_primary_key": True, "length": 36},
    }


def collection_payload(name: str, fields: Sequence[Dict[str, Any]], pk: str, note: Optional[str], singleton: bool) -> Dict[str, Any]:
    return {
        "collection": name,
        "schema": {"name": name},
        "meta": {"note": note, "singleton": singleton, "hidden": False},
        "fields": [primary_key_field(pk)] + [field_payload(f) for f in fields],
    }


def guard_system_collection(name: str, verb: str) -> None:
    if name.startswith(SYSTEM_PREFIX):
        raise DirectusError(
            f"Refusing to {verb} the system collection {name!r}. Directus system tables are managed "
            f"through their own endpoints (/users, /roles, /policies, /permissions)."
        )


# ---------------------------------------------------------------------------
# Permission helpers
# ---------------------------------------------------------------------------


def permission_payload(policy: str, collection: str, action: str, fields: Optional[Sequence[str]], rule: Any, validation: Any) -> Dict[str, Any]:
    if action not in ACTIONS:
        raise DirectusError(f"Unknown action {action!r}. Directus actions: {', '.join(ACTIONS)}")
    return {
        "policy": policy,
        "collection": collection,
        "action": action,
        "fields": list(fields) if fields is not None else ["*"],
        "permissions": rule if rule is not None else {},
        "validation": validation if validation is not None else {},
    }


def check_fields_trap(fields: Optional[Sequence[str]], action: str) -> None:
    """``fields: []`` is not "all fields" — it is the primary key only, and it is silent."""
    if fields is not None and len(list(fields)) == 0:
        warn(
            f"fields is empty for the {action} permission: Directus reads that as the primary key ONLY, "
            f"not as every field. Pass --fields '*' for all fields."
        )


def generate_static_token() -> str:
    return secrets.token_urlsafe(32)


def bootstrap_plan(name: str, collections: Sequence[str], actions: Sequence[str], email: Optional[str], app_access: bool) -> List[Dict[str, Any]]:
    """The exact call chain ``bootstrap-agent`` will make, in order."""
    steps: List[Dict[str, Any]] = [
        {"step": "create_policy", "endpoint": "POST /policies", "name": f"{name} policy", "app_access": app_access},
    ]
    for collection in collections:
        for action in actions:
            steps.append(
                {
                    "step": "create_permission",
                    "endpoint": "POST /permissions",
                    "collection": collection,
                    "action": action,
                    "fields": ["*"],
                }
            )
    steps.append({"step": "create_role", "endpoint": "POST /roles", "name": f"{name} role"})
    steps.append({"step": "link_access", "endpoint": "POST /access", "note": "role -> policy via directus_access"})
    if email:
        steps.append({"step": "create_user", "endpoint": "POST /users", "email": email})
        steps.append({"step": "set_static_token", "endpoint": "PATCH /users/{id}", "note": "static token cannot be set via SQL"})
    return steps


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def build_client(args: argparse.Namespace) -> DirectusClient:
    return DirectusClient(
        url=resolve_setting(getattr(args, "url", None), "DIRECTUS_URL", ""),
        token=resolve_setting(getattr(args, "token", None), "DIRECTUS_TOKEN"),
        email=resolve_setting(getattr(args, "auth_email", None), "DIRECTUS_EMAIL"),
        password=resolve_setting(getattr(args, "auth_password", None), "DIRECTUS_PASSWORD"),
        timeout=float(resolve_setting(getattr(args, "timeout", None), "DIRECTUS_TIMEOUT", DEFAULT_TIMEOUT, float)),
    )


def require_yes(args: argparse.Namespace, what: str) -> None:
    if not getattr(args, "yes", False):
        raise DirectusError(f"Refusing to {what} without --yes. Deleting in Directus is not undoable.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    client = build_client(args)
    info = client.server_info()
    identity = client.me()
    version = str(info.get("version") or "")
    major = major_version(version)
    role = identity.get("role") or {}
    result = {
        "url": client.url,
        "directus": "ok",
        "version": version or "unknown",
        "policy_model": "policy-based (v11+)" if (major or 0) >= MIN_MAJOR_VERSION else "role-based (v10)",
        "project": (info.get("project") or {}).get("project_name"),
        "authenticated_as": identity.get("email") or identity.get("id"),
        "role": role.get("name"),
        "admin_access": bool(role.get("admin_access")),
        "auth": "static token" if getattr(args, "token", None) or os.environ.get("DIRECTUS_TOKEN") else "login",
    }
    emit(result)
    if major is not None and major < MIN_MAJOR_VERSION:
        warn(f"This skill targets Directus {MIN_MAJOR_VERSION}+; policy commands are unavailable on {version}.")
        return 1
    if not result["admin_access"]:
        warn("This account is not an admin: schema and permission commands will return 403.")
    return 0


def cmd_collections_list(args: argparse.Namespace) -> int:
    client = build_client(args)
    rows = client.request("GET", "/collections") or []
    names = [row.get("collection") for row in rows if isinstance(row, dict)]
    if not args.system:
        names = [n for n in names if n and not n.startswith(SYSTEM_PREFIX)]
    return emit({"count": len(names), "collections": sorted(n for n in names if n)})


def cmd_collections_create(args: argparse.Namespace) -> int:
    guard_system_collection(args.name, "create")
    fields = [parse_field_spec(spec) for spec in (args.field or [])]
    payload = collection_payload(args.name, fields, args.pk, args.note, args.singleton)
    if args.dry_run:
        return emit({"dry_run": True, "endpoint": "POST /collections", "payload": payload})
    client = build_client(args)
    created = client.request("POST", "/collections", payload)
    log(f"Created collection {args.name} with {len(fields)} field(s)")
    return emit({"created": args.name, "fields": [f["field"] for f in fields], "meta": created})


def cmd_collections_delete(args: argparse.Namespace) -> int:
    guard_system_collection(args.name, "delete")
    require_yes(args, f"drop collection {args.name!r} and every row in it")
    client = build_client(args)
    client.request("DELETE", f"/collections/{urllib.parse.quote(args.name)}")
    log(f"Deleted collection {args.name}")
    return emit({"deleted": args.name})


def cmd_fields_list(args: argparse.Namespace) -> int:
    client = build_client(args)
    rows = client.request("GET", f"/fields/{urllib.parse.quote(args.collection)}") or []
    fields = [
        {
            "field": row.get("field"),
            "type": row.get("type"),
            "required": bool((row.get("meta") or {}).get("required")),
            "primary_key": bool((row.get("schema") or {}).get("is_primary_key")),
        }
        for row in rows
        if isinstance(row, dict)
    ]
    return emit({"collection": args.collection, "count": len(fields), "fields": fields})


def cmd_fields_create(args: argparse.Namespace) -> int:
    guard_system_collection(args.collection, "alter")
    spec = parse_field_spec(f"{args.field}:{args.type}{'!' if args.required else ''}")
    payload = field_payload(spec)
    if args.dry_run:
        return emit({"dry_run": True, "endpoint": f"POST /fields/{args.collection}", "payload": payload})
    client = build_client(args)
    created = client.request("POST", f"/fields/{urllib.parse.quote(args.collection)}", payload)
    log(f"Added {args.field} ({args.type}) to {args.collection}")
    return emit({"collection": args.collection, "created": args.field, "meta": created})


def _item_query(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    fields = split_list(getattr(args, "fields", None))
    if fields:
        params["fields"] = ",".join(fields)
    rule = parse_json_arg(getattr(args, "filter", None), "--filter")
    if rule is not None:
        params["filter"] = json.dumps(rule)
    if getattr(args, "sort", None):
        params["sort"] = args.sort
    if getattr(args, "search", None):
        params["search"] = args.search
    return params


def cmd_items_list(args: argparse.Namespace) -> int:
    params = _item_query(args)  # parsed before connecting so a bad --filter fails fast
    client = build_client(args)
    path = f"/items/{urllib.parse.quote(args.collection)}"
    if args.all:
        rows = client.paginate(path, params)
        meta: Dict[str, Any] = {"paginated": True}
    else:
        params["limit"] = args.limit
        params["offset"] = args.offset
        params["meta"] = "total_count"
        rows, meta = client.request_with_meta(path, params)
        rows = rows or []
    return emit({"collection": args.collection, "returned": len(rows), "meta": meta, "items": rows})


def cmd_items_get(args: argparse.Namespace) -> int:
    client = build_client(args)
    params = {}
    fields = split_list(args.fields)
    if fields:
        params["fields"] = ",".join(fields)
    item = client.request("GET", f"/items/{urllib.parse.quote(args.collection)}/{urllib.parse.quote(str(args.id))}", params=params)
    return emit({"collection": args.collection, "id": args.id, "item": item})


def cmd_items_create(args: argparse.Namespace) -> int:
    payload = parse_json_arg(args.data, "--data")
    if payload is None:
        raise DirectusError("create needs --data with a JSON object (or a JSON array for a batch).")
    if args.dry_run:
        return emit({"dry_run": True, "endpoint": f"POST /items/{args.collection}", "payload": payload})
    client = build_client(args)
    created = client.request("POST", f"/items/{urllib.parse.quote(args.collection)}", payload)
    count = len(created) if isinstance(created, list) else 1
    log(f"Created {count} item(s) in {args.collection}")
    return emit({"collection": args.collection, "created": count, "items": created})


def cmd_items_update(args: argparse.Namespace) -> int:
    payload = parse_json_arg(args.data, "--data")
    if payload is None:
        raise DirectusError("update needs --data with a JSON object of the fields to change.")
    client = build_client(args)
    path = f"/items/{urllib.parse.quote(args.collection)}/{urllib.parse.quote(str(args.id))}"
    updated = client.request("PATCH", path, payload)
    log(f"Updated {args.collection}/{args.id}")
    return emit({"collection": args.collection, "id": args.id, "item": updated})


def cmd_items_delete(args: argparse.Namespace) -> int:
    require_yes(args, f"delete {args.collection}/{args.id}")
    client = build_client(args)
    client.request("DELETE", f"/items/{urllib.parse.quote(args.collection)}/{urllib.parse.quote(str(args.id))}")
    log(f"Deleted {args.collection}/{args.id}")
    return emit({"collection": args.collection, "deleted": args.id})


def cmd_policies_list(args: argparse.Namespace) -> int:
    client = build_client(args)
    client.require_v11("Access policies")
    rows = client.request("GET", "/policies", params={"fields": "id,name,description,admin_access,app_access", "limit": -1}) or []
    return emit({"count": len(rows), "policies": rows})


def cmd_policies_create(args: argparse.Namespace) -> int:
    payload = {
        "name": args.name,
        "description": args.description,
        "admin_access": bool(args.admin),
        "app_access": bool(args.app),
        "enforce_tfa": False,
    }
    if args.dry_run:
        return emit({"dry_run": True, "endpoint": "POST /policies", "payload": payload})
    client = build_client(args)
    client.require_v11("Access policies")
    created = client.request("POST", "/policies", payload) or {}
    log(f"Created policy {args.name} ({created.get('id')})")
    return emit({"created": "policy", "id": created.get("id"), "policy": created})


def cmd_permissions_list(args: argparse.Namespace) -> int:
    client = build_client(args)
    client.require_v11("Policy permissions")
    params: Dict[str, Any] = {"limit": -1, "fields": "id,policy,collection,action,fields"}
    if args.policy:
        params["filter"] = json.dumps({"policy": {"_eq": args.policy}})
    elif args.collection:
        params["filter"] = json.dumps({"collection": {"_eq": args.collection}})
    rows = client.request("GET", "/permissions", params=params) or []
    return emit({"count": len(rows), "permissions": rows})


def cmd_permissions_create(args: argparse.Namespace) -> int:
    fields = split_list(args.fields)
    if args.fields is not None and fields is None:
        fields = []
    check_fields_trap(fields, args.action)
    payload = permission_payload(
        args.policy,
        args.collection,
        args.action,
        fields,
        parse_json_arg(args.rule, "--rule"),
        parse_json_arg(args.validation, "--validation"),
    )
    if args.dry_run:
        return emit({"dry_run": True, "endpoint": "POST /permissions", "payload": payload})
    client = build_client(args)
    client.require_v11("Policy permissions")
    created = client.request("POST", "/permissions", payload) or {}
    log(f"Granted {args.action} on {args.collection} to policy {args.policy}")
    return emit({"created": "permission", "id": created.get("id"), "permission": created})


def cmd_roles_list(args: argparse.Namespace) -> int:
    client = build_client(args)
    rows = client.request("GET", "/roles", params={"fields": "id,name,description", "limit": -1}) or []
    return emit({"count": len(rows), "roles": rows})


def cmd_roles_create(args: argparse.Namespace) -> int:
    payload = {"name": args.name, "description": args.description}
    if args.dry_run:
        return emit({"dry_run": True, "endpoint": "POST /roles", "payload": payload})
    client = build_client(args)
    created = client.request("POST", "/roles", payload) or {}
    log(f"Created role {args.name} ({created.get('id')})")
    warn("A role grants nothing on its own — link it to a policy with `access grant`.")
    return emit({"created": "role", "id": created.get("id"), "role": created})


def cmd_access_list(args: argparse.Namespace) -> int:
    client = build_client(args)
    client.require_v11("The directus_access junction")
    params: Dict[str, Any] = {"limit": -1, "fields": "id,role,user,policy,sort"}
    if args.role:
        params["filter"] = json.dumps({"role": {"_eq": args.role}})
    elif args.policy:
        params["filter"] = json.dumps({"policy": {"_eq": args.policy}})
    rows = client.request("GET", "/access", params=params) or []
    return emit({"count": len(rows), "access": rows})


def cmd_access_grant(args: argparse.Namespace) -> int:
    if not args.role and not args.user:
        raise DirectusError("access grant needs --role or --user (the thing the policy is attached to).")
    payload = {"policy": args.policy, "role": args.role, "user": args.user}
    if args.dry_run:
        return emit({"dry_run": True, "endpoint": "POST /access", "payload": payload})
    client = build_client(args)
    client.require_v11("The directus_access junction")
    created = client.request("POST", "/access", payload) or {}
    log(f"Linked policy {args.policy} to {'role ' + args.role if args.role else 'user ' + args.user}")
    return emit({"created": "access", "id": created.get("id"), "access": created})


def cmd_access_revoke(args: argparse.Namespace) -> int:
    require_yes(args, f"revoke access link {args.id}")
    client = build_client(args)
    client.require_v11("The directus_access junction")
    client.request("DELETE", f"/access/{urllib.parse.quote(str(args.id))}")
    log(f"Revoked access link {args.id}")
    return emit({"revoked": args.id})


def cmd_users_list(args: argparse.Namespace) -> int:
    client = build_client(args)
    params: Dict[str, Any] = {"limit": -1, "fields": "id,email,status,role.id,role.name"}
    if args.role:
        params["filter"] = json.dumps({"role": {"_eq": args.role}})
    rows = client.request("GET", "/users", params=params) or []
    return emit({"count": len(rows), "users": rows})


def cmd_users_create(args: argparse.Namespace) -> int:
    payload = {"email": args.email, "password": args.password, "role": args.role, "status": "active"}
    if args.dry_run:
        return emit({"dry_run": True, "endpoint": "POST /users", "payload": {**payload, "password": "***"}})
    client = build_client(args)
    created = client.request("POST", "/users", payload) or {}
    log(f"Created user {args.email} ({created.get('id')})")
    return emit({"created": "user", "id": created.get("id"), "email": args.email})


def cmd_token_set(args: argparse.Namespace) -> int:
    token = args.token_value or generate_static_token()
    if args.dry_run:
        return emit({"dry_run": True, "endpoint": f"PATCH /users/{args.user}", "payload": {"token": "***"}})
    client = build_client(args)
    client.request("PATCH", f"/users/{urllib.parse.quote(str(args.user))}", {"token": token})
    log(f"Set static token on user {args.user}")
    warn("Directus stores the static token hashed-at-rest but never shows it again — save it now.")
    return emit({"user": args.user, "static_token": token})


def cmd_token_clear(args: argparse.Namespace) -> int:
    require_yes(args, f"clear the static token of user {args.user}")
    client = build_client(args)
    client.request("PATCH", f"/users/{urllib.parse.quote(str(args.user))}", {"token": None})
    log(f"Cleared static token on user {args.user}")
    return emit({"user": args.user, "static_token": None})


def cmd_bootstrap_agent(args: argparse.Namespace) -> int:
    """policy -> permissions -> role -> access -> user -> static token, in that order."""
    collections = split_list(args.collections) or []
    actions = split_list(args.actions) or ["read"]
    if not collections:
        raise DirectusError("bootstrap-agent needs --collections with at least one collection name.")
    for action in actions:
        if action not in ACTIONS:
            raise DirectusError(f"Unknown action {action!r}. Directus actions: {', '.join(ACTIONS)}")

    plan = bootstrap_plan(args.name, collections, actions, args.email, bool(args.app_access))
    if args.dry_run:
        return emit({"dry_run": True, "agent": args.name, "steps": plan})

    client = build_client(args)
    client.require_v11("bootstrap-agent")
    result: Dict[str, Any] = {"agent": args.name, "collections": collections, "actions": actions}

    policy = client.request("POST", "/policies", {
        "name": f"{args.name} policy",
        "description": f"Managed by the Hermes directus skill for agent {args.name}",
        "admin_access": False,
        "app_access": bool(args.app_access),
        "enforce_tfa": False,
    }) or {}
    result["policy"] = policy.get("id")
    log(f"policy {result['policy']}")

    permissions = []
    for collection in collections:
        for action in actions:
            created = client.request("POST", "/permissions", permission_payload(
                result["policy"], collection, action, ["*"], parse_json_arg(args.rule, "--rule"), None,
            )) or {}
            permissions.append({"collection": collection, "action": action, "id": created.get("id")})
    result["permissions"] = permissions
    log(f"{len(permissions)} permission rule(s)")

    role = client.request("POST", "/roles", {
        "name": f"{args.name} role",
        "description": f"Managed by the Hermes directus skill for agent {args.name}",
    }) or {}
    result["role"] = role.get("id")
    log(f"role {result['role']}")

    access = client.request("POST", "/access", {"role": result["role"], "policy": result["policy"]}) or {}
    result["access"] = access.get("id")
    log(f"access link {result['access']}")

    if args.email:
        user = client.request("POST", "/users", {
            "email": args.email,
            "password": args.password or generate_static_token(),
            "role": result["role"],
            "status": "active",
        }) or {}
        result["user"] = user.get("id")
        token = args.token_value or generate_static_token()
        client.request("PATCH", f"/users/{urllib.parse.quote(str(result['user']))}", {"token": token})
        result["static_token"] = token
        log(f"user {result['user']} with a static token")
        warn("The static token is shown once — store it in the agent's .env as DIRECTUS_TOKEN.")
    return emit(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_connection_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", help="Directus base URL (env DIRECTUS_URL)")
    parser.add_argument("--token", help="static token (env DIRECTUS_TOKEN)")
    parser.add_argument("--auth-email", help="login email (env DIRECTUS_EMAIL)")
    parser.add_argument("--auth-password", help="login password (env DIRECTUS_PASSWORD)")
    parser.add_argument("--timeout", type=float, help=f"seconds (env DIRECTUS_TIMEOUT, default {DEFAULT_TIMEOUT:g})")


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="print the request instead of sending it")


def _sub(top: Any, name: str, help_text: str) -> Any:
    """``top`` is the top-level subparsers action; returns the nested one."""
    group = top.add_parser(name, help=help_text)
    return group.add_subparsers(dest=f"{name}_command", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="directus_admin.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    top = parser.add_subparsers(dest="command", required=True)

    p = top.add_parser("check", help="server reachable, credentials valid, version and policy model")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_check)

    # collections ----------------------------------------------------------
    collections = _sub(top, "collections", "list, create, or delete a collection")
    p = collections.add_parser("list", help="list user collections")
    p.add_argument("--system", action="store_true", help="include directus_* system collections")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_collections_list)

    p = collections.add_parser("create", help="create a collection and its fields")
    p.add_argument("name")
    p.add_argument("--field", action="append", metavar="NAME:TYPE", help="repeatable; trailing ! marks it required")
    p.add_argument("--pk", choices=["uuid", "auto"], default="uuid", help="primary key style (default uuid)")
    p.add_argument("--note", help="collection note shown in the Studio")
    p.add_argument("--singleton", action="store_true")
    _add_connection_flags(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_collections_create)

    p = collections.add_parser("delete", help="drop a collection and every row in it")
    p.add_argument("name")
    p.add_argument("--yes", action="store_true")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_collections_delete)

    # fields ---------------------------------------------------------------
    fields = _sub(top, "fields", "list or create fields on a collection")
    p = fields.add_parser("list", help="list fields with types and primary key")
    p.add_argument("collection")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_fields_list)

    p = fields.add_parser("create", help="add a field to an existing collection")
    p.add_argument("collection")
    p.add_argument("field")
    p.add_argument("--type", default="string", choices=list(FIELD_TYPES))
    p.add_argument("--required", action="store_true")
    _add_connection_flags(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_fields_create)

    # items ----------------------------------------------------------------
    items = _sub(top, "items", "read and write rows")
    p = items.add_parser("list", help="query rows")
    p.add_argument("collection")
    p.add_argument("--filter", help="Directus filter as JSON, @file, or - for stdin")
    p.add_argument("--fields", help="comma-separated field list (default all)")
    p.add_argument("--sort", help="e.g. -date_created")
    p.add_argument("--search", help="full-text search across the collection")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--all", action="store_true", help="page through every matching row")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_items_list)

    p = items.add_parser("get", help="read one row by primary key")
    p.add_argument("collection")
    p.add_argument("id")
    p.add_argument("--fields")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_items_get)

    p = items.add_parser("create", help="insert one row (object) or many (array)")
    p.add_argument("collection")
    p.add_argument("--data", required=True, help="JSON, @file, or - for stdin")
    _add_connection_flags(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_items_create)

    p = items.add_parser("update", help="patch one row by primary key")
    p.add_argument("collection")
    p.add_argument("id")
    p.add_argument("--data", required=True, help="JSON, @file, or - for stdin")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_items_update)

    p = items.add_parser("delete", help="delete one row by primary key")
    p.add_argument("collection")
    p.add_argument("id")
    p.add_argument("--yes", action="store_true")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_items_delete)

    # policies -------------------------------------------------------------
    policies = _sub(top, "policies", "access policies (Directus 11+)")
    p = policies.add_parser("list", help="list policies")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_policies_list)

    p = policies.add_parser("create", help="create a policy")
    p.add_argument("name")
    p.add_argument("--description")
    p.add_argument("--admin", action="store_true", help="admin_access: bypasses every permission")
    p.add_argument("--app", action="store_true", help="app_access: may sign in to the Studio")
    _add_connection_flags(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_policies_create)

    # permissions ----------------------------------------------------------
    permissions = _sub(top, "permissions", "permission rules attached to a policy")
    p = permissions.add_parser("list", help="list permission rules")
    p.add_argument("--policy")
    p.add_argument("--collection")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_permissions_list)

    p = permissions.add_parser("create", help="grant one action on one collection to a policy")
    p.add_argument("--policy", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--action", required=True, choices=list(ACTIONS))
    p.add_argument("--fields", help="comma-separated fields; default '*' (empty means primary key ONLY)")
    p.add_argument("--rule", help="row-level filter as JSON, @file, or -")
    p.add_argument("--validation", help="write-time validation filter as JSON, @file, or -")
    _add_connection_flags(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_permissions_create)

    # roles ----------------------------------------------------------------
    roles = _sub(top, "roles", "roles (a role grants nothing without a policy)")
    p = roles.add_parser("list", help="list roles")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_roles_list)

    p = roles.add_parser("create", help="create a role")
    p.add_argument("name")
    p.add_argument("--description")
    _add_connection_flags(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_roles_create)

    # access ---------------------------------------------------------------
    access = _sub(top, "access", "the directus_access junction between roles/users and policies")
    p = access.add_parser("list", help="list access links")
    p.add_argument("--role")
    p.add_argument("--policy")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_access_list)

    p = access.add_parser("grant", help="attach a policy to a role or a user")
    p.add_argument("--policy", required=True)
    p.add_argument("--role")
    p.add_argument("--user")
    _add_connection_flags(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_access_grant)

    p = access.add_parser("revoke", help="delete an access link by id")
    p.add_argument("id")
    p.add_argument("--yes", action="store_true")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_access_revoke)

    # users ----------------------------------------------------------------
    users = _sub(top, "users", "users")
    p = users.add_parser("list", help="list users")
    p.add_argument("--role")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_users_list)

    p = users.add_parser("create", help="create a user in a role")
    p.add_argument("--email", required=True)
    p.add_argument("--password", dest="password", required=True)
    p.add_argument("--role", required=True)
    _add_connection_flags(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_users_create)

    # token ----------------------------------------------------------------
    token = _sub(top, "token", "static API tokens (API only — a SQL UPDATE does not work)")
    p = token.add_parser("set", help="set (or generate) a user's static token")
    p.add_argument("user", help="user id")
    p.add_argument("--value", dest="token_value", help="use this token instead of a generated one")
    _add_connection_flags(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_token_set)

    p = token.add_parser("clear", help="remove a user's static token")
    p.add_argument("user")
    p.add_argument("--yes", action="store_true")
    _add_connection_flags(p)
    p.set_defaults(func=cmd_token_clear)

    # bootstrap ------------------------------------------------------------
    p = top.add_parser("bootstrap-agent", help="policy + permissions + role + access + user + static token")
    p.add_argument("name", help="agent name; becomes '<name> policy' and '<name> role'")
    p.add_argument("--collections", required=True, help="comma-separated collections the agent may touch")
    p.add_argument("--actions", default="read", help="comma-separated: create,read,update,delete,share")
    p.add_argument("--rule", help="row-level filter applied to every permission, as JSON, @file, or -")
    p.add_argument("--email", help="also create a service user with this email")
    p.add_argument("--password", dest="password", help="password for the service user (generated when omitted)")
    p.add_argument("--value", dest="token_value", help="static token to set instead of a generated one")
    p.add_argument("--app-access", action="store_true", help="let the policy sign in to the Studio")
    _add_connection_flags(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_bootstrap_agent)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DirectusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
