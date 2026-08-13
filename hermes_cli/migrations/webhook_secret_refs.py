"""Atomic migration of webhook route plaintext secrets to profile references.

The migration deliberately keeps a plaintext source intact until the secure
store has accepted and returned every value.  A caller can use the returned
receipt to audit the operation without receiving any secret value.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from agent.secret_scope import get_secret


DEFAULT_SECRET_PREFIX = "WEBHOOK_ROUTE_"
_SECRET_KEYS = ("secret", "secret_value")


class WebhookSecretMigrationError(RuntimeError):
    """Raised when a webhook secret migration cannot complete safely."""

    def __init__(self, message: str, *, receipt: dict[str, Any] | None = None, source: str = ""):
        super().__init__(message)
        self.receipt = receipt or {}
        self.rollback_receipt = {"source": source, "source_preserved": True}


def _reference(route_name: str, route: Mapping[str, Any]) -> str:
    ref = route.get("secret_ref")
    if isinstance(ref, str) and ref.strip():
        return ref.strip()
    safe = "".join(ch if ch.isalnum() else "_" for ch in route_name.upper())
    return f"{DEFAULT_SECRET_PREFIX}{safe}"


def _route_secret(route: Mapping[str, Any]) -> str | None:
    for key in _SECRET_KEYS:
        value = route.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _default_store(ref: str, value: str) -> None:
    """Persist through the existing profile resolver's .env backend."""
    from hermes_cli.config import save_env_value

    save_env_value(ref, value)


def _default_resolve(ref: str) -> str | None:
    value = get_secret(ref, "")
    if value:
        return str(value)
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv
        return get_env_value_prefer_dotenv(ref)
    except Exception:
        return None


def migrate_webhook_routes(
    source_path: str | Path,
    *,
    store: Callable[[str, str], None] | None = None,
    resolve: Callable[[str], str | None] | None = None,
    backup_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Migrate a route JSON mapping and return a value-free receipt.

    Sequence: write → resolve → verify → atomically switch → scrub.  The
    original route file and backups are untouched until all references resolve.
    A failure before switch leaves the source byte-for-byte unchanged; a
    failure after switch is reported with a rollback receipt and never scrubs.
    """
    path = Path(source_path)
    original = path.read_text(encoding="utf-8")
    try:
        routes = json.loads(original)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebhookSecretMigrationError("Unable to read webhook routes safely") from exc
    if not isinstance(routes, dict):
        raise WebhookSecretMigrationError("Webhook route store must be a JSON object")

    put = store or _default_store
    lookup = resolve or _default_resolve
    staged = copy.deepcopy(routes)
    migrated: list[str] = []
    receipts: list[dict[str, Any]] = []

    # write → resolve → verify: no source mutation has happened yet.
    for name, route in routes.items():
        if not isinstance(route, dict):
            continue
        value = _route_secret(route)
        if not value or route.get("secret_ref"):
            continue
        ref = _reference(str(name), route)
        receipt = {"route": str(name), "reference": ref, "stored": False, "verified": False}
        receipts.append(receipt)
        try:
            put(ref, value)
            receipt["stored"] = True
            resolved = lookup(ref)
            if resolved != value:
                raise WebhookSecretMigrationError("Secret backend verification failed")
            receipt["verified"] = True
        except Exception as exc:
            raise WebhookSecretMigrationError(
                f"Secure persistence failed for route {name!r}; source left untouched",
                receipt=receipt,
                source=str(path),
            ) from exc
        staged[name].pop("secret", None)
        staged[name].pop("secret_value", None)
        staged[name]["secret_ref"] = ref
        migrated.append(str(name))

    # atomically switch only after every staged secret is verified.
    if migrated:
        try:
            _write_json_atomic(path, staged)
        except Exception as exc:
            raise WebhookSecretMigrationError(
                "Atomic route switch failed; source remains available for rollback",
                receipt={"migrated_routes": migrated},
                source=str(path),
            ) from exc

    # scrub legacy secret-bearing backups only after the switched route record is
    # live and verified. Scrubbing is best-effort but never replaces a verified
    # route with an unverifiable state.
    scrubbed: list[str] = []
    for raw_backup in backup_paths:
        backup = Path(raw_backup)
        if not backup.exists():
            continue
        try:
            text = backup.read_text(encoding="utf-8")
            backup_routes = json.loads(text)
            if isinstance(backup_routes, dict):
                changed = False
                for name, route in backup_routes.items():
                    if not isinstance(route, dict):
                        continue
                    ref = staged.get(name, {}).get("secret_ref") if isinstance(staged.get(name), dict) else None
                    if ref and _route_secret(route):
                        route.pop("secret", None)
                        route.pop("secret_value", None)
                        route["secret_ref"] = ref
                        changed = True
                if changed:
                    _write_json_atomic(backup, backup_routes)
                    scrubbed.append(str(backup))
        except Exception as exc:
            raise WebhookSecretMigrationError(
                "Route switched but backup scrub failed; rollback receipt retained",
                receipt={"migrated_routes": migrated, "scrubbed_backups": scrubbed},
                source=str(path),
            ) from exc

    return {
        "migrated_routes": migrated,
        "receipts": receipts,
        "scrubbed_backups": scrubbed,
        "rollback": {"source": str(path), "source_preserved_on_pre_switch_failure": True},
    }


def migrate_webhook_config(
    config_path: str | Path,
    *,
    store: Callable[[str, str], None] | None = None,
    resolve: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Migrate legacy global/inline webhook secrets in config.yaml.

    YAML is parsed and staged in memory; secure persistence and resolution are
    completed before the single atomic YAML replacement.  A failed backend
    operation therefore leaves the original config untouched.
    """
    path = Path(config_path)
    original = path.read_text(encoding="utf-8")
    try:
        config = yaml.safe_load(original) or {}
    except yaml.YAMLError as exc:
        raise WebhookSecretMigrationError("Unable to parse webhook config safely") from exc
    if not isinstance(config, dict):
        raise WebhookSecretMigrationError("Webhook config must be a YAML mapping")
    put = store or _default_store
    lookup = resolve or _default_resolve
    staged = copy.deepcopy(config)
    webhook = staged.get("platforms", {}).get("webhook", {}) if isinstance(staged.get("platforms"), dict) else {}
    if not isinstance(webhook, dict):
        return {"migrated": False, "rollback": {"source": str(path)}}
    extra = webhook.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
        webhook["extra"] = extra
    candidates: list[tuple[str, str, str]] = []
    global_secret = extra.get("secret") or webhook.get("secret")
    if isinstance(global_secret, str) and global_secret:
        candidates.append(("WEBHOOK_SECRET", global_secret, "global"))
    routes = extra.get("routes")
    if isinstance(routes, dict):
        for name, route in routes.items():
            if isinstance(route, dict) and isinstance(route.get("secret"), str) and route["secret"]:
                candidates.append((_reference(str(name), route), route["secret"], str(name)))
    receipts = []
    for ref, value, label in candidates:
        receipt = {"route": label, "reference": ref, "stored": False, "verified": False}
        receipts.append(receipt)
        try:
            put(ref, value)
            receipt["stored"] = True
            if lookup(ref) != value:
                raise WebhookSecretMigrationError("Secret backend verification failed")
            receipt["verified"] = True
        except Exception as exc:
            raise WebhookSecretMigrationError(
                f"Secure persistence failed for webhook secret {label!r}; source left untouched",
                receipt=receipt,
                source=str(path),
            ) from exc
    if candidates:
        extra.pop("secret", None)
        extra["secret_ref"] = "WEBHOOK_SECRET" if global_secret else extra.get("secret_ref")
        webhook.pop("secret", None)
        if isinstance(routes, dict):
            for name, route in routes.items():
                if isinstance(route, dict) and route.get("secret"):
                    route["secret_ref"] = _reference(str(name), route)
                    route.pop("secret", None)
        from utils import atomic_yaml_write
        try:
            atomic_yaml_write(path, staged, sort_keys=False)
        except Exception as exc:
            raise WebhookSecretMigrationError(
                "Atomic config switch failed; source remains available for rollback",
                receipt={"migrated": bool(candidates)},
                source=str(path),
            ) from exc
    return {"migrated": bool(candidates), "receipts": receipts, "rollback": {"source": str(path), "source_preserved_on_pre_switch_failure": True}}


# Compatibility alias for callers that name migrations by operation.
migrate = migrate_webhook_routes
migrate_webhook_secret_refs = migrate_webhook_routes
