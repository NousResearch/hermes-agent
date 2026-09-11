"""Pairing, webhooks, gateway lifecycle, credential pool, memory provider and
operations (doctor/backup/import/hooks/checkpoints) dashboard routes.

Helpers/state that tests monkeypatch on ``web_server`` stay there and are
reached through the late-binding seam (cycle-safe).
"""

import asyncio
import contextlib
import logging
import os
import re
import secrets
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from hermes_cli.config import redact_key
from hermes_cli.web_deps import late
from hermes_cli.web_server_files import _path_is_under
from hermes_cli.web_server_gateway import _restart_gateway_after
from hermes_cli.web_server_memory import _normalize_memory_provider_name, _require_memory_provider_ready
from hermes_cli.web_models import (
    BackupRequest, CredentialPoolAdd, HookCreate, HookDelete, ImportRequest, MemoryProviderSelect,
    MemoryReset, PairingApprove, PairingRevoke, WebhookCreate, WebhookEnabledToggle,
)
from hermes_cli.web_routers._common import _CONFIG_MUTATION_LOCK, http_failure, spawn_profile_action
from hermes_cli.web_routers.files import stream_upload_to_path

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_discover_memory_provider_statuses = late("_discover_memory_provider_statuses", "hermes_cli.web_server_memory")
_gateway_subcommand = late("_gateway_subcommand", "hermes_cli.web_server_gateway")
_resolve_profile_dir = late("_resolve_profile_dir", "hermes_cli.web_server_profiles")
_spawn_hermes_action = late("_spawn_hermes_action", "hermes_cli.web_server_gateway")
_write_platform_enabled = late("_write_platform_enabled", "hermes_cli.web_server_messaging")
get_hermes_home = late("get_hermes_home", "hermes_cli.config")
load_config = late("load_config", "hermes_cli.config")
save_config = late("save_config", "hermes_cli.config")


def _spawn_action(argv: List[str], name: str, *, log_msg: str, prefix: str) -> dict:
    """Spawn a dashboard-profile ``hermes <argv>`` action; spawn failure -> 500."""
    return spawn_profile_action(None, argv, name, log_msg=log_msg, prefix=prefix)


# --- Pairing: how a remote admin onboards messaging users without shell access.


def _pairing_store(profile: Optional[str] = None):
    """Pairing store for ``profile`` — the dashboard's own when unspecified.

    The gateway keeps one store per served profile, so without scoping an
    operator on a named profile would approve into a whitelist their gateway
    never consults. ``PairingStore`` resolves the profile home itself
    (``default`` maps to the global store); only the name is validated here,
    so nothing process-global is swapped across the ``await`` boundary.
    """
    from gateway.pairing import PairingStore

    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        return PairingStore()
    _resolve_profile_dir(requested)  # 400/404 on an unknown profile
    return PairingStore(profile=requested)


@router.get("/api/pairing")
async def list_pairing(profile: Optional[str] = None):
    store = _pairing_store(profile)
    return {"pending": store.list_pending(), "approved": store.list_approved()}


@router.post("/api/pairing/approve")
async def approve_pairing(body: PairingApprove):
    store = _pairing_store(body.profile)
    platform = (body.platform or "").lower().strip()
    # `request_id` is what an admin surface sends after listing pending requests;
    # `code` is the one-time code the user relays. A request-id-shaped value in
    # the older `code` field still routes to the request path.
    target = (body.request_id or body.code or "").strip()
    if not platform or not target:
        raise HTTPException(status_code=400, detail="platform and request_id or code are required")

    by_request_id = bool(body.request_id) or store.looks_like_request_id(target)
    result = store.approve_request(platform, target) if by_request_id else store.approve_code(platform, target.upper())
    if result:
        return {"ok": True, "user": result}
    # Lockout only gates the code path — a stale request id must not surface
    # as a bogus 429 while the platform is locked out for an unrelated reason.
    if not by_request_id and store._is_locked_out(platform):
        raise HTTPException(
            status_code=429, detail=f"Platform '{platform}' is locked out after too many failed approvals.",
        )
    raise HTTPException(
        status_code=404, detail=f"Pairing request or code not found or expired for platform '{platform}'.",
    )


@router.post("/api/pairing/revoke")
async def revoke_pairing(body: PairingRevoke):
    store = _pairing_store(body.profile)
    platform = (body.platform or "").lower().strip()
    if not platform or not body.user_id:
        raise HTTPException(status_code=400, detail="platform and user_id are required")
    if store.revoke(platform, body.user_id):
        return {"ok": True}
    raise HTTPException(status_code=404, detail=f"User {body.user_id} not found in approved list for {platform}.")


@router.post("/api/pairing/clear-pending")
async def clear_pending_pairing(profile: Optional[str] = None):
    return {"ok": True, "cleared": _pairing_store(profile).clear_pending()}


# --- Webhooks: same JSON store as the CLI (hermes_cli.webhook); the adapter
# hot-reloads it. Per-route HMAC secrets are redacted on read, surfaced once on create.


def _webhook_route_summary(name: str, route: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    return {
        "name": name,
        "description": route.get("description", ""),
        "events": list(route.get("events") or []),
        "deliver": route.get("deliver", "log"),
        "deliver_only": bool(route.get("deliver_only")),
        "prompt": route.get("prompt", ""),
        "script": route.get("script", ""),
        "skills": list(route.get("skills") or []),
        "created_at": route.get("created_at"),
        "url": f"{base_url}/webhooks/{name}",
        "secret_set": bool(route.get("secret")),
        # Default-enabled; only an explicit enabled:false turns a route off.
        "enabled": route.get("enabled", True) is not False,
    }


@router.get("/api/webhooks")
async def list_webhooks():
    import hermes_cli.webhook as wh

    base_url = wh._get_webhook_base_url()
    return {
        "enabled": wh._is_webhook_enabled(),
        "base_url": base_url,
        "subscriptions": [
            _webhook_route_summary(name, route, base_url)
            for name, route in wh._load_subscriptions().items()
        ],
    }


@router.post("/api/webhooks/enable")
async def enable_webhooks():
    with http_failure("Failed to enable webhook platform from dashboard", 500, detail="Failed to enable webhook platform."):
        _write_platform_enabled("webhook", True)
    restart_result = _restart_gateway_after(None, what="enabling webhooks", label="Webhook enable")
    return {
        "ok": True,
        "platform": "webhook",
        "enabled": True,
        "needs_restart": not restart_result["restart_started"],
        **restart_result,
    }


@router.post("/api/webhooks")
async def create_webhook(body: WebhookCreate):
    import hermes_cli.webhook as wh

    if not wh._is_webhook_enabled():
        raise HTTPException(
            status_code=400, detail="Webhook platform is not enabled. Enable it from the Webhooks page first.",
        )
    name = (body.name or "").strip().lower().replace(" ", "-")
    if not re.match(r"^[a-z0-9][a-z0-9_-]*$", name):
        raise HTTPException(
            status_code=400, detail="Invalid name. Use lowercase alphanumeric with hyphens/underscores.",
        )
    if body.deliver_only and body.deliver == "log":
        raise HTTPException(
            status_code=400, detail="Direct delivery requires a real target (telegram, discord, …), not 'log'.",
        )

    secret = body.secret or secrets.token_urlsafe(32)
    route: Dict[str, Any] = {
        "description": body.description or f"Dashboard-created subscription: {name}",
        "events": [e.strip() for e in body.events if e.strip()],
        "secret": secret,
        "prompt": body.prompt or "",
        "skills": [s.strip() for s in body.skills if s.strip()],
        "deliver": body.deliver or "log",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if body.script and body.script.strip():
        route["script"] = body.script.strip()
    if body.deliver_only:
        route["deliver_only"] = True
    if body.deliver_chat_id:
        route["deliver_extra"] = {"chat_id": body.deliver_chat_id}

    subs = wh._load_subscriptions()
    subs[name] = route
    wh._save_subscriptions(subs)

    summary = _webhook_route_summary(name, route, wh._get_webhook_base_url())
    summary["secret"] = secret  # surfaced exactly once, on create
    return summary


def _webhook_subs_with(name: str):
    """(module, subscriptions, key) for an existing route; 404 otherwise."""
    import hermes_cli.webhook as wh

    key = (name or "").strip().lower()
    subs = wh._load_subscriptions()
    if key not in subs:
        raise HTTPException(status_code=404, detail=f"No subscription named '{key}'")
    return wh, subs, key


@router.delete("/api/webhooks/{name}")
async def delete_webhook(name: str):
    wh, subs, key = _webhook_subs_with(name)
    del subs[key]
    wh._save_subscriptions(subs)
    return {"ok": True}


@router.put("/api/webhooks/{name}/enabled")
async def set_webhook_enabled(name: str, body: WebhookEnabledToggle):
    """Disabled routes stay on disk (re-enable later) but the gateway rejects
    their events with 403; it hot-reloads the file, so no restart is needed."""
    wh, subs, key = _webhook_subs_with(name)
    subs[key]["enabled"] = bool(body.enabled)
    wh._save_subscriptions(subs)
    return {"ok": True, "name": key, "enabled": bool(body.enabled)}


# --- Gateway lifecycle: spawn the real `hermes gateway <verb>` so behaviour
# matches the CLI exactly (status is surfaced by /api/status).


@router.post("/api/gateway/start")
async def start_gateway(profile: Optional[str] = None):
    with http_failure("Failed to spawn gateway start", 500, "Failed to start gateway"):
        proc = _spawn_hermes_action(_gateway_subcommand(profile, "start"), "gateway-start")
    return {"ok": True, "pid": proc.pid, "name": "gateway-start"}


@router.post("/api/gateway/stop")
async def stop_gateway(profile: Optional[str] = None):
    with http_failure("Failed to spawn gateway stop", 500, "Failed to stop gateway"):
        proc = _spawn_hermes_action(_gateway_subcommand(profile, "stop"), "gateway-stop")
    return {"ok": True, "pid": proc.pid, "name": "gateway-stop"}


# --- Credential pool (auth.json -> credential_pool.<provider>[]): secrets are
# redacted on read; only the agent sees raw values at session start.
#
# load_pool() may hit the network synchronously (Copilot token exchange over raw
# urllib, whose timeout does NOT bound DNS resolution) — on a networkless host it
# once froze the uvicorn loop for 17 minutes. Every pool load below runs off-loop.


def _pool_entry_summary(entry: Any, index: int) -> Dict[str, Any]:
    """Redacted view of one PooledCredential; ``index`` is 1-based to match
    CredentialPool.remove_index()."""
    token = entry.access_token or ""
    return {
        "index": index,
        "id": entry.id,
        "label": entry.label,
        "auth_type": entry.auth_type,
        "source": entry.source,
        "priority": entry.priority,
        "last_status": entry.last_status,
        "request_count": entry.request_count,
        "token_preview": redact_key(token) if token else "",
        "has_refresh": bool(entry.refresh_token),
    }


@router.get("/api/credentials/pool")
async def list_credential_pool():
    from agent.credential_pool import load_pool
    from hermes_cli.auth import read_credential_pool

    def _run():
        providers = []
        # read_credential_pool(None) lists every provider with pooled entries;
        # load_pool() gives the rich PooledCredential objects per provider.
        for provider_id in sorted(read_credential_pool().keys()):
            try:
                pool = load_pool(provider_id)
            except Exception:
                _log.exception("load_pool(%s) failed", provider_id)
                continue
            entries = pool.entries()
            if entries:
                providers.append({
                    "provider": provider_id,
                    "entries": [_pool_entry_summary(e, i) for i, e in enumerate(entries, start=1)],
                })
        return {"providers": providers}

    return await asyncio.to_thread(_run)


@router.post("/api/credentials/pool")
async def add_credential_pool_entry(body: CredentialPoolAdd):
    import uuid
    from agent.credential_pool import (
        AUTH_TYPE_API_KEY,
        CUSTOM_POOL_PREFIX,
        SOURCE_MANUAL,
        PooledCredential,
        load_pool,
    )

    provider = (body.provider or "").strip().lower()
    api_key = (body.api_key or "").strip()
    if not provider or not api_key:
        raise HTTPException(status_code=400, detail="provider and api_key are required")

    def _run():
        try:
            pool = load_pool(provider)
            label = (body.label or "").strip() or f"key #{len(pool.entries()) + 1}"
            pool.add_entry(PooledCredential(
                provider=provider,
                # Add a distinct, self-contained pool entry per account (matching the qwen-oauth /
                # minimax-oauth multi-account patterns, and the xai-oauth path below) instead of routing
                # through the singleton ``_save_codex_tokens`` save path. The singleton round-trip collapsed
                # every added account into the latest login: a second ``hermes auth add openai-codex``
                # overwrote the first account's singleton-mirrored ``device_code`` entry rather than
                # creating an independent one (#39236). ``manual:device_code`` entries refresh from their
                # own token pair, so they need no singleton shadow.
                id=uuid.uuid4().hex[:6],
                label=label,
                auth_type=AUTH_TYPE_API_KEY,
                priority=0,
                source=SOURCE_MANUAL,
                access_token=api_key,
            ))
            # Re-adding is an explicit re-engagement signal: lift every suppression
            # for this provider so a source deleted earlier can seed again
            # (mirrors `hermes auth add`).
            if not provider.startswith(CUSTOM_POOL_PREFIX):
                try:
                    from hermes_cli.auth import _load_auth_store, unsuppress_credential_source

                    suppressed = _load_auth_store().get("suppressed_sources", {})
                    for src in list(suppressed.get(provider, []) or []):
                        unsuppress_credential_source(provider, src)
                except Exception:
                    _log.exception("unsuppress after pool add failed (non-fatal)")
            return {"ok": True, "provider": provider, "count": len(pool.entries())}
        except HTTPException:
            raise
        except Exception as exc:
            _log.exception("POST /api/credentials/pool failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return await asyncio.to_thread(_run)


@router.delete("/api/credentials/pool/{provider}/{index}")
async def remove_credential_pool_entry(provider: str, index: int):
    """Remove a pool entry (``index`` is 1-based, as listed).

    Removal must be sticky: ``load_pool()`` re-seeds entries from their backing
    source (.env var, OAuth file, custom-provider config) on every call, so
    deleting only the row silently reverts on the next refresh. Dispatch through
    the same RemovalStep registry as ``hermes auth remove``: each source cleans
    its external state and suppresses ``(provider, source)`` so seeders skip it.
    Manual entries have no step — nothing external, and they aren't re-seeded.

    See #55217.
    """
    from agent.credential_pool import load_pool
    from agent.credential_sources import find_removal_step
    from hermes_cli.auth import suppress_credential_source

    provider = (provider or "").strip().lower()

    def _run():
        try:
            pool = load_pool(provider)
            removed = pool.remove_index(index)
        except Exception as exc:
            _log.exception("DELETE /api/credentials/pool failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if removed is None:
            raise HTTPException(status_code=404, detail="No pool entry at that index")

        cleaned: List[str] = []
        hints: List[str] = []
        step = find_removal_step(provider, removed.source or "")
        if step is not None:
            try:
                result = step.remove_fn(provider, removed)
                cleaned = list(result.cleaned)
                hints = list(result.hints)
                if result.suppress:
                    suppress_credential_source(provider, removed.source)
            except Exception:
                # Cleanup is best-effort, but suppression is the actual fix —
                # without it the entry resurrects on the next load_pool().
                _log.exception("credential source cleanup failed for %s/%s; suppressing anyway", provider, removed.source)
                try:
                    suppress_credential_source(provider, removed.source)
                except Exception:
                    _log.exception("suppress_credential_source failed")
        return {"ok": True, "provider": provider, "count": len(pool.entries()), "cleaned": cleaned, "hints": hints}

    return await asyncio.to_thread(_run)


# --- Memory provider: setup is dashboard-native only via get_config_schema();
# interactive setup hooks never run here, and activation requires the provider
# to be discoverable, available and fully configured.

_MEMORY_FILES = (("MEMORY.md", "memory"), ("USER.md", "user"))


@router.get("/api/memory")
async def get_memory_status():
    def _run():  # load_config(), stats and discovery are disk reads — off-loop
        cfg = load_config()
        mem = cfg.get("memory")
        active = _normalize_memory_provider_name(mem.get("provider")) if isinstance(mem, dict) else ""
        mem_dir = get_hermes_home() / "memories"
        files = {}  # sizes so the UI can show what a reset would erase
        for fname, key in _MEMORY_FILES:
            path = mem_dir / fname
            files[key] = path.stat().st_size if path.exists() else 0
        return {"active": active, "providers": _discover_memory_provider_statuses(), "builtin_files": files}

    return await asyncio.to_thread(_run)


@router.put("/api/memory/provider")
async def set_memory_provider(body: MemoryProviderSelect):
    provider = _normalize_memory_provider_name(body.provider)

    def _run():
        _require_memory_provider_ready(provider)
        with _CONFIG_MUTATION_LOCK:
            cfg = load_config()
            if not isinstance(cfg.get("memory"), dict):
                cfg["memory"] = {}
            cfg["memory"]["provider"] = provider
            save_config(cfg)
        return {"ok": True, "active": provider}

    return await asyncio.to_thread(_run)


@router.post("/api/memory/reset")
async def reset_memory(body: MemoryReset):
    target = (body.target or "all").strip().lower()
    if target not in {"all", "memory", "user"}:
        raise HTTPException(status_code=400, detail="target must be all, memory, or user")

    mem_dir = get_hermes_home() / "memories"
    deleted = []
    for fname, key in _MEMORY_FILES:
        path = mem_dir / fname
        if target in {"all", key} and path.exists():
            try:
                path.unlink()
                deleted.append(fname)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not delete {fname}: {exc}")
    return {"ok": True, "deleted": deleted}


# --- Operations: long-running text-output commands (doctor, audit, backup,
# import) are spawned as background actions whose logs the dashboard tails via
# /api/actions/{name}/status; cheap structured reads return JSON directly.


@router.post("/api/ops/doctor")
async def run_doctor():
    return _spawn_action(["doctor"], "doctor", log_msg="Failed to spawn doctor", prefix="Failed to run doctor")


@router.post("/api/ops/security-audit")
async def run_security_audit():
    return _spawn_action(
        ["security", "audit"], "security-audit",
        log_msg="Failed to spawn security audit", prefix="Failed to run security audit",
    )


def _dashboard_backup_dir() -> Path:
    return get_hermes_home() / "backups"


@router.post("/api/ops/backup")
async def run_backup(body: BackupRequest):
    archive: Optional[Path] = None
    output = (body.output or "").strip()
    if not output:
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        archive = _dashboard_backup_dir() / f"hermes-backup-{stamp}-{secrets.token_hex(4)}.zip"
        try:
            archive.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not create backup directory: {exc}")
        output = str(archive)
    response = _spawn_action(["backup", "-o", output], "backup", log_msg="Failed to spawn backup", prefix="Failed to run backup")
    if archive is not None:
        response["archive"] = str(archive)
    return response


@router.get("/api/ops/backup/download")
async def download_dashboard_backup(archive: str):
    try:
        backup_dir = _dashboard_backup_dir().expanduser().resolve(strict=False)
        target = Path(archive).expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Backup not found")
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid backup path")

    if not _path_is_under(backup_dir, target):
        raise HTTPException(status_code=403, detail="Backup is outside the dashboard backup directory")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Backup not found")
    return FileResponse(
        path=str(target), media_type="application/zip", filename=target.name, content_disposition_type="attachment",
    )


def _spawn_import(archive: str, force: bool) -> dict:
    args = ["import", archive]
    if force:
        args.append("--force")
    return _spawn_action(args, "import", log_msg="Failed to spawn import", prefix="Failed to run import")


@router.post("/api/ops/import")
async def run_import(body: ImportRequest):
    archive = (body.archive or "").strip()
    if not archive:
        raise HTTPException(status_code=400, detail="archive path is required")
    if not os.path.isfile(archive):
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive}")
    return _spawn_import(archive, body.force)


def _safe_backup_upload_name(filename: str | None) -> str:
    name = Path(filename or "backup.zip").name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-") or "backup.zip"
    if not name.lower().endswith(".zip"):
        name = f"{name}.zip"
    return name


@router.post("/api/ops/import-upload")
async def run_import_upload(
    file: UploadFile = File(...),
    force: bool = Form(False),
):
    staging_dir = _dashboard_backup_dir()
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not create import staging directory: {exc}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = staging_dir / f"dashboard-import-{stamp}-{secrets.token_hex(4)}-{_safe_backup_upload_name(file.filename)}"
    total = await stream_upload_to_path(
        file, target, too_large="Archive is too large",
        not_writable="Import staging directory is not writable", write_failed="Could not write uploaded archive",
    )
    if not zipfile.is_zipfile(target):
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded archive is not a valid zip file")
    return {**_spawn_import(str(target), force), "archive": str(target), "uploaded_bytes": total}


@router.get("/api/ops/hooks")
async def list_hooks():
    """Configured shell hooks with consent (allowlist) status, whether the
    script is currently executable, and the valid hook events for the form."""
    def _run():
        from hermes_cli.config import load_config as _load_config
        from agent import shell_hooks

        valid_events = []
        with contextlib.suppress(Exception):
            from hermes_cli.plugins import VALID_HOOKS
            valid_events = sorted(VALID_HOOKS)

        specs = []
        try:
            specs = shell_hooks.iter_configured_hooks(_load_config())
        except Exception:
            _log.exception("iter_configured_hooks failed")

        out = []
        for spec in specs:
            entry = None
            executable = False
            with contextlib.suppress(Exception):
                entry = shell_hooks.allowlist_entry_for(spec.event, spec.command)
            with contextlib.suppress(Exception):
                executable = shell_hooks.script_is_executable(spec.command)
            out.append({
                "event": spec.event,
                "matcher": spec.matcher,
                "command": spec.command,
                "timeout": spec.timeout,
                "allowed": entry is not None,
                "approved_at": (entry or {}).get("approved_at"),
                "executable": executable,
            })
        return {"hooks": out, "valid_events": valid_events}

    return await asyncio.to_thread(_run)


def _hook_body_fields(body) -> tuple[str, str]:
    event = (body.event or "").strip()
    command = (body.command or "").strip()
    if not event or not command:
        raise HTTPException(status_code=400, detail="event and command are required")
    return event, command


@router.post("/api/ops/hooks")
async def create_hook(body: HookCreate):
    """Add a shell hook to config.yaml and optionally record consent.

    Shell hooks run arbitrary commands, so this is privileged: it writes the
    ``hooks:`` block and, with ``approve``, records the allowlist entry so the
    hook actually fires. Takes effect on the next session / gateway restart.
    """
    from agent import shell_hooks

    event, command = _hook_body_fields(body)
    valid_hooks = None
    with contextlib.suppress(Exception):
        from hermes_cli.plugins import VALID_HOOKS as valid_hooks
    if valid_hooks is not None and event not in valid_hooks:
        raise HTTPException(status_code=400, detail=f"Unknown event '{event}'. Valid: {', '.join(sorted(valid_hooks))}")

    def _run():
        with _CONFIG_MUTATION_LOCK:
            cfg = load_config()
            hooks_cfg = cfg.get("hooks")
            if not isinstance(hooks_cfg, dict):
                hooks_cfg = cfg["hooks"] = {}
            entries = hooks_cfg.get(event)
            if not isinstance(entries, list):
                entries = hooks_cfg[event] = []
            new_entry: Dict[str, Any] = {"command": command}
            if body.matcher:
                new_entry["matcher"] = body.matcher
            if body.timeout is not None:
                new_entry["timeout"] = int(body.timeout)
            entries.append(new_entry)
            save_config(cfg)

        approved = False
        if body.approve:
            try:
                shell_hooks._record_approval(event, command)
                approved = True
            except Exception:
                _log.exception("hook consent record failed")
        return {"ok": True, "event": event, "command": command, "approved": approved}

    return await asyncio.to_thread(_run)


@router.delete("/api/ops/hooks")
async def delete_hook(body: HookDelete):
    """Remove a hook from config.yaml and revoke its consent allowlist entry."""
    from agent import shell_hooks

    event, command = _hook_body_fields(body)

    def _run():
        removed = False
        with _CONFIG_MUTATION_LOCK:
            cfg = load_config()
            hooks_cfg = cfg.get("hooks")
            if isinstance(hooks_cfg, dict) and isinstance(hooks_cfg.get(event), list):
                before = len(hooks_cfg[event])
                hooks_cfg[event] = [
                    e for e in hooks_cfg[event]
                    if not (isinstance(e, dict) and e.get("command") == command)
                ]
                removed = len(hooks_cfg[event]) < before
                if not hooks_cfg[event]:
                    del hooks_cfg[event]
                if not hooks_cfg:
                    cfg.pop("hooks", None)
                save_config(cfg)
        # Revoke consent regardless so a re-add re-prompts.
        with contextlib.suppress(Exception):
            shell_hooks.revoke(command)
        return removed

    if not await asyncio.to_thread(_run):
        raise HTTPException(status_code=404, detail="No matching hook found")
    return {"ok": True}


@router.get("/api/ops/checkpoints")
async def list_checkpoints():
    """/rollback shadow-store checkpoints (read-only): count + size per session
    so the UI can show what a prune reclaims; pruning itself is a spawned CLI
    action so the confirmation logic stays in one place."""
    cp_dir = get_hermes_home() / "checkpoints"
    sessions = []
    total_bytes = 0
    if cp_dir.is_dir():
        with os.scandir(cp_dir) as scan:
            children = sorted((Path(e.path) for e in scan), key=lambda p: p.name)
        for child in children:
            if not child.is_dir():
                continue
            size = count = 0
            for f in child.rglob("*"):
                if f.is_file():
                    try:
                        size += f.stat().st_size
                        count += 1
                    except OSError:
                        pass
            total_bytes += size
            sessions.append({"session": child.name, "files": count, "bytes": size})
    return {"sessions": sessions, "total_bytes": total_bytes}


@router.post("/api/ops/checkpoints/prune")
async def prune_checkpoints():
    return _spawn_action(
        ["checkpoints", "prune"], "checkpoints-prune",
        log_msg="Failed to spawn checkpoints prune", prefix="Failed to prune checkpoints",
    )
