"""1Password helpers for Hermes Universal Browser Operator.

The public helpers return only metadata.  Secret values are only used inside the
browser-fill path and are never included in tool responses.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, AbstractSet
from urllib.parse import urlparse


@dataclass(frozen=True)
class LoginSecrets:
    """Internal-only credential bundle.

    Do not serialize this dataclass into tool responses.  Use
    `extract_login_metadata()` for model-visible metadata.
    """

    item_id: str
    title: str
    vault: str | None
    username: str | None
    password: str | None
    totp: str | None = None


def load_env_file_values(path: str | Path) -> dict[str, str]:
    """Parse a simple dotenv file without evaluating shell expansions."""
    env_path = Path(path).expanduser()
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def op_environment(base: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Return environment for `op` subprocesses, loading ~/.hermes/.env if needed."""
    env = dict(base or os.environ)
    dotenv = load_env_file_values(Path.home() / ".hermes" / ".env")
    for key, value in dotenv.items():
        if key.startswith("OP_") and key not in env:
            env[key] = value
    return env


def _host_from_value(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = (parsed.hostname or value or "").lower().strip().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _item_urls(item: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for entry in item.get("urls") or []:
        if isinstance(entry, dict):
            href = entry.get("href") or entry.get("url")
            if href:
                urls.append(str(href))
        elif isinstance(entry, str):
            urls.append(entry)
    if item.get("url"):
        urls.append(str(item["url"]))
    return urls


def _score_item(item: dict[str, Any], hint: str) -> int:
    hint_l = (hint or "").lower().strip()
    if not hint_l:
        return 0
    hint_host = _host_from_value(hint_l)
    title = str(item.get("title") or "").lower()
    score = 0
    for url in _item_urls(item):
        host = _host_from_value(url)
        if host == hint_host:
            score = max(score, 100)
        elif host.endswith(f".{hint_host}") or hint_host.endswith(f".{host}"):
            score = max(score, 80)
        elif hint_host in host or host in hint_host:
            score = max(score, 55)
    if hint_l and hint_l in title:
        score = max(score, 60)
    for part in re.split(r"[\s._/-]+", hint_host):
        if len(part) >= 3 and part in title:
            score = max(score, 35)
    return score


def choose_best_login_item(items: Iterable[dict[str, Any]], hint: str) -> dict[str, Any] | None:
    """Choose the most relevant 1Password login item for a domain/name hint."""
    best: tuple[int, dict[str, Any]] | None = None
    for item in items:
        score = _score_item(item, hint)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, item)
    return best[1] if best else None


def extract_login_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Return model-visible item metadata without secret field values."""
    fields = item.get("fields") or []
    username_available = False
    password_available = False
    totp_available = False
    field_labels: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        label = str(field.get("label") or field.get("id") or "").strip()
        purpose = str(field.get("purpose") or "").upper()
        ftype = str(field.get("type") or "").upper()
        if label:
            field_labels.append(label)
        if purpose == "USERNAME" or label.lower() in {"username", "email", "login"}:
            username_available = bool(field.get("value") is not None or username_available)
        if purpose == "PASSWORD" or label.lower() in {"password", "passphrase"}:
            password_available = bool(field.get("value") is not None or password_available)
        if ftype == "OTP" or "one-time" in label.lower() or label.lower() in {"otp", "totp"}:
            totp_available = True
    vault_value = item.get("vault")
    vault: dict[str, Any] = vault_value if isinstance(vault_value, dict) else {}
    return {
        "item_id": item.get("id") or item.get("uuid") or "",
        "title": item.get("title") or "",
        "vault": vault.get("name") or item.get("vault_name") or None,
        "urls": _item_urls(item),
        "username_available": username_available,
        "password_available": password_available,
        "totp_available": totp_available,
        "field_labels": field_labels,
        "secret_values_returned": False,
    }


def _run_op(args: list[str], *, timeout: int = 30, env: Optional[dict[str, str]] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["op", *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        env=op_environment(env),
        check=False,
    )


def _run_op_json(args: list[str], *, timeout: int = 30) -> Any:
    proc = _run_op(args, timeout=timeout)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "op command failed").strip()
        raise RuntimeError(err[:500])
    try:
        return json.loads(proc.stdout or "null")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"op returned non-JSON output: {exc}") from exc


def list_login_items(vault: str | None = None) -> list[dict[str, Any]]:
    args = ["item", "list", "--categories", "Login", "--format", "json"]
    if vault:
        args.extend(["--vault", vault])
    data = _run_op_json(args)
    return data if isinstance(data, list) else []


def get_item(item_id_or_title: str, vault: str | None = None) -> dict[str, Any]:
    args = ["item", "get", item_id_or_title, "--format", "json"]
    if vault:
        args.extend(["--vault", vault])
    data = _run_op_json(args)
    if not isinstance(data, dict):
        raise RuntimeError("op item get returned an unexpected payload")
    return data


def _field_value(item: dict[str, Any], *, purposes: AbstractSet[str], labels: AbstractSet[str], types: AbstractSet[str] = frozenset()) -> str | None:
    for field in item.get("fields") or []:
        if not isinstance(field, dict):
            continue
        purpose = str(field.get("purpose") or "").upper()
        label = str(field.get("label") or field.get("id") or "").strip().lower()
        ftype = str(field.get("type") or "").upper()
        if purpose in purposes or label in labels or ftype in types:
            value = field.get("value")
            if value is not None:
                return str(value)
    return None


def _totp_for_item(item_id: str, vault: str | None = None) -> str | None:
    args = ["item", "get", item_id, "--otp"]
    if vault:
        args.extend(["--vault", vault])
    proc = _run_op(args, timeout=30)
    if proc.returncode != 0:
        return None
    code = (proc.stdout or "").strip()
    return code or None


def resolve_login_secrets(hint: str, vault: str | None = None, include_totp: bool = False) -> LoginSecrets:
    """Resolve credentials by domain/name hint for internal browser filling."""
    item_summary = choose_best_login_item(list_login_items(vault=vault), hint)
    if not item_summary:
        raise RuntimeError(f"No matching 1Password Login item found for hint: {hint}")
    summary_vault_value = item_summary.get("vault")
    summary_vault: dict[str, Any] = summary_vault_value if isinstance(summary_vault_value, dict) else {}
    vault_name = vault or summary_vault.get("name")
    item_id = str(item_summary.get("id") or item_summary.get("uuid") or item_summary.get("title") or "")
    item = get_item(item_id, vault=vault_name)
    metadata = extract_login_metadata(item)
    username = _field_value(
        item,
        purposes={"USERNAME"},
        labels={"username", "email", "login", "user"},
    )
    password = _field_value(
        item,
        purposes={"PASSWORD"},
        labels={"password", "passphrase"},
        types={"CONCEALED"},
    )
    totp = _totp_for_item(item_id, vault=vault_name) if include_totp else None
    return LoginSecrets(
        item_id=str(metadata.get("item_id") or item_id),
        title=str(metadata.get("title") or item_summary.get("title") or ""),
        vault=metadata.get("vault") or vault_name,
        username=username,
        password=password,
        totp=totp,
    )


def login_secrets_metadata(secrets: LoginSecrets) -> dict[str, Any]:
    return {
        "item_id": secrets.item_id,
        "title": secrets.title,
        "vault": secrets.vault,
        "username_available": bool(secrets.username),
        "password_available": bool(secrets.password),
        "totp_available": bool(secrets.totp),
        "secret_values_returned": False,
    }
