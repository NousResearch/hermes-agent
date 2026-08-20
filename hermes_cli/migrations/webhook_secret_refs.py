"""Atomic migration of webhook plaintext secrets to profile references.

The migration keeps plaintext source bytes intact until secure persistence has
accepted and resolved every value. Default webhook writers share one bounded
cross-process lock so CLI updates and runtime migration cannot race each other.
Receipts never contain secret values.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from hermes_cli.webhook_secrets import (
    resolve_webhook_secret,
    store_webhook_secret,
    store_webhook_secret_unlocked,
    webhook_secret_write_lock,
)

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
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _writer_context(store: Callable[[str, str], None] | None):
    # Injected stores are test/operator-owned and must not be serialized by a
    # Hermes-home lock. The production default owns the lock for the complete
    # read → persist → verify → switch transaction.
    return webhook_secret_write_lock() if store is None else nullcontext()


def migrate_webhook_routes(
    source_path: str | Path,
    *,
    store: Callable[[str, str], None] | None = None,
    resolve: Callable[[str], str | None] | None = None,
    backup_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Migrate route JSON using write → resolve → verify → switch → scrub."""
    path = Path(source_path)
    with _writer_context(store):
        try:
            original = path.read_text(encoding="utf-8")
            routes = json.loads(original)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise WebhookSecretMigrationError(
                "Unable to read webhook routes safely", source=str(path)
            ) from None
        if not isinstance(routes, dict):
            raise WebhookSecretMigrationError(
                "Webhook route store must be a JSON object", source=str(path)
            )

        put = store or store_webhook_secret_unlocked
        lookup = resolve or resolve_webhook_secret
        staged = copy.deepcopy(routes)
        migrated: list[str] = []
        receipts: list[dict[str, Any]] = []

        for name, route in routes.items():
            if not isinstance(route, dict):
                continue
            value = _route_secret(route)
            if not value or route.get("secret_ref"):
                continue
            ref = _reference(str(name), route)
            receipt = {
                "route": str(name),
                "reference": ref,
                "stored": False,
                "verified": False,
            }
            receipts.append(receipt)
            try:
                put(ref, value)
                receipt["stored"] = True
                if lookup(ref) != value:
                    raise WebhookSecretMigrationError(
                        "Secret backend verification failed",
                        receipt=receipt,
                        source=str(path),
                    )
                receipt["verified"] = True
            except WebhookSecretMigrationError:
                raise
            except Exception:
                raise WebhookSecretMigrationError(
                    f"Secure persistence failed for route {name!r}; source left untouched",
                    receipt=receipt,
                    source=str(path),
                ) from None
            staged[name].pop("secret", None)
            staged[name].pop("secret_value", None)
            staged[name]["secret_ref"] = ref
            migrated.append(str(name))

        if migrated:
            try:
                _write_json_atomic(path, staged)
            except Exception:
                raise WebhookSecretMigrationError(
                    "Atomic route switch failed; source remains available for rollback",
                    receipt={"migrated_routes": migrated},
                    source=str(path),
                ) from None

        scrubbed: list[str] = []
        for raw_backup in backup_paths:
            backup = Path(raw_backup)
            if not backup.exists():
                continue
            try:
                backup_routes = json.loads(backup.read_text(encoding="utf-8"))
                if not isinstance(backup_routes, dict):
                    continue
                changed = False
                for name, route in backup_routes.items():
                    if not isinstance(route, dict):
                        continue
                    staged_route = staged.get(name)
                    ref = staged_route.get("secret_ref") if isinstance(staged_route, dict) else None
                    if ref and _route_secret(route):
                        route.pop("secret", None)
                        route.pop("secret_value", None)
                        route["secret_ref"] = ref
                        changed = True
                if changed:
                    _write_json_atomic(backup, backup_routes)
                    scrubbed.append(str(backup))
            except Exception:
                raise WebhookSecretMigrationError(
                    "Route switched but backup scrub failed; rollback receipt retained",
                    receipt={"migrated_routes": migrated, "scrubbed_backups": scrubbed},
                    source=str(path),
                ) from None

        return {
            "migrated_routes": migrated,
            "receipts": receipts,
            "scrubbed_backups": scrubbed,
            "rollback": {
                "source": str(path),
                "source_preserved_on_pre_switch_failure": True,
            },
        }


def migrate_webhook_config(
    config_path: str | Path,
    *,
    store: Callable[[str, str], None] | None = None,
    resolve: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Migrate global and static-route webhook secrets in config.yaml."""
    path = Path(config_path)
    with _writer_context(store):
        try:
            original = path.read_text(encoding="utf-8")
            config = yaml.safe_load(original) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            raise WebhookSecretMigrationError(
                "Unable to parse webhook config safely", source=str(path)
            ) from None
        if not isinstance(config, dict):
            raise WebhookSecretMigrationError(
                "Webhook config must be a YAML mapping", source=str(path)
            )

        put = store or store_webhook_secret_unlocked
        lookup = resolve or resolve_webhook_secret
        staged = copy.deepcopy(config)
        platforms = staged.get("platforms")
        webhook = platforms.get("webhook", {}) if isinstance(platforms, dict) else {}
        if not isinstance(webhook, dict):
            return {"migrated": False, "receipts": [], "rollback": {"source": str(path)}}
        extra = webhook.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}
            webhook["extra"] = extra

        candidates: list[tuple[str, str, str]] = []
        global_secret = (
            extra.get("secret")
            or extra.get("secret_value")
            or webhook.get("secret")
            or webhook.get("secret_value")
        )
        if isinstance(global_secret, str) and global_secret:
            candidates.append(("WEBHOOK_SECRET", global_secret, "global"))
        routes = extra.get("routes")
        if isinstance(routes, dict):
            for name, route in routes.items():
                if isinstance(route, dict):
                    value = _route_secret(route)
                    if value:
                        candidates.append((_reference(str(name), route), value, str(name)))

        receipts: list[dict[str, Any]] = []
        for ref, value, label in candidates:
            receipt = {"route": label, "reference": ref, "stored": False, "verified": False}
            receipts.append(receipt)
            try:
                put(ref, value)
                receipt["stored"] = True
                if lookup(ref) != value:
                    raise WebhookSecretMigrationError(
                        "Secret backend verification failed",
                        receipt=receipt,
                        source=str(path),
                    )
                receipt["verified"] = True
            except WebhookSecretMigrationError:
                raise
            except Exception:
                raise WebhookSecretMigrationError(
                    f"Secure persistence failed for webhook secret {label!r}; source left untouched",
                    receipt=receipt,
                    source=str(path),
                ) from None

        if candidates:
            extra.pop("secret", None)
            extra.pop("secret_value", None)
            if global_secret:
                extra["secret_ref"] = "WEBHOOK_SECRET"
            webhook.pop("secret", None)
            webhook.pop("secret_value", None)
            if isinstance(routes, dict):
                for name, route in routes.items():
                    if isinstance(route, dict) and _route_secret(route):
                        route["secret_ref"] = _reference(str(name), route)
                        route.pop("secret", None)
                        route.pop("secret_value", None)
            try:
                from hermes_cli.config import atomic_config_write

                atomic_config_write(path, staged, sort_keys=False)
            except Exception:
                raise WebhookSecretMigrationError(
                    "Atomic config switch failed; source remains available for rollback",
                    receipt={"migrated": True},
                    source=str(path),
                ) from None

        return {
            "migrated": bool(candidates),
            "receipts": receipts,
            "rollback": {
                "source": str(path),
                "source_preserved_on_pre_switch_failure": True,
            },
        }


migrate = migrate_webhook_routes
migrate_webhook_secret_refs = migrate_webhook_routes

__all__ = [
    "WebhookSecretMigrationError",
    "migrate_webhook_config",
    "migrate_webhook_routes",
    "migrate_webhook_secret_refs",
    "resolve_webhook_secret",
    "store_webhook_secret",
]
