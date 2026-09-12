"""Discord application-command synchronization and rate-limit policy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from .. import adapter as _adapter

logger = _adapter.logger
discord = _adapter.discord
DISCORD_AVAILABLE = _adapter.DISCORD_AVAILABLE
_DISCORD_COMMAND_SYNC_POLICIES = _adapter._DISCORD_COMMAND_SYNC_POLICIES
_scoped_gate_env = lambda *args: _adapter._scoped_gate_env(*args)


def _derive_forum_thread_name(content: str) -> str:
    """Resolve the adapter helper lazily because it is declared below the adapter class."""
    return _adapter._derive_forum_thread_name(content)


def _metadata_marks_nonconversational(metadata):
    return _adapter._metadata_marks_nonconversational(metadata)


def _looks_like_nonconversational_history_message(content: str) -> bool:
    return _adapter._looks_like_nonconversational_history_message(content)


def _is_discord_transport_error(exc: BaseException) -> bool:
    return _adapter._is_discord_transport_error(exc)
_Path = Path
_DISCORD_COMMAND_SYNC_STATE_SUBDIR = _adapter._DISCORD_COMMAND_SYNC_STATE_SUBDIR
_DISCORD_COMMAND_SYNC_STATE_FILENAME = _adapter._DISCORD_COMMAND_SYNC_STATE_FILENAME
_DISCORD_COMMAND_SYNC_MUTATION_INTERVAL_SECONDS = _adapter._DISCORD_COMMAND_SYNC_MUTATION_INTERVAL_SECONDS
_DISCORD_COMMAND_SYNC_MAX_RATE_LIMIT_SLEEP_SECONDS = _adapter._DISCORD_COMMAND_SYNC_MAX_RATE_LIMIT_SLEEP_SECONDS
atomic_json_write = _adapter.atomic_json_write


class CommandSyncMixin:
    """Persisted, rate-limit-aware Discord command synchronization."""
    def _command_sync_state_path(self) -> _Path:
        from hermes_constants import get_hermes_home
        directory = get_hermes_home() / _DISCORD_COMMAND_SYNC_STATE_SUBDIR
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return directory / _DISCORD_COMMAND_SYNC_STATE_FILENAME

    def _read_command_sync_state(self) -> dict:
        try:
            path = self._command_sync_state_path()
            if not path.exists():
                return {}
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_command_sync_state(self, state: dict) -> None:
        atomic_json_write(
            self._command_sync_state_path(), state, indent=None, separators=(",", ":"),
        )

    def _command_sync_state_key(self, app_id: Any) -> str:
        return str(app_id or "unknown")

    def _desired_command_sync_fingerprint(self) -> str:
        tree = self._client.tree if self._client else None
        desired = []
        if tree is not None:
            desired = [
                self._canonicalize_app_command_payload(command.to_dict(tree))
                for command in tree.get_commands()
            ]
        desired.sort(key=lambda item: (item.get("type", 1), item.get("name", "")))
        payload = json.dumps(desired, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _command_sync_skip_reason(self, app_id: Any, fingerprint: str) -> Optional[str]:
        entry = self._read_command_sync_state().get(self._command_sync_state_key(app_id))
        if not isinstance(entry, dict):
            return None
        now = time.time()
        retry_after_until = float(entry.get("retry_after_until") or 0)
        if retry_after_until > now:
            remaining = max(1, int(retry_after_until - now))
            return f"Discord asked us to wait before syncing slash commands; retry in {remaining}s"
        last_success_at = float(entry.get("last_success_at") or 0)
        last_attempt_at = float(entry.get("last_attempt_at") or 0)
        if (
            entry.get("fingerprint") == fingerprint
            and last_success_at
            and last_success_at >= last_attempt_at
        ):
            return "same slash-command fingerprint already synced"
        return None

    def _update_command_sync_entry(self, app_id: Any, fingerprint: str, *, keep_existing: bool, drop=(), fields=None) -> None:
        """Rewrite this app's sync-state entry (optionally merged over the existing one).
        ``fields`` is a callable of ``now`` so timestamps derive from one clock read."""
        state = self._read_command_sync_state()
        key = self._command_sync_state_key(app_id)
        entry = dict(state.get(key)) if keep_existing and isinstance(state.get(key), dict) else {}
        for name in drop:
            entry.pop(name, None)
        now = time.time()
        state[key] = {**entry, "fingerprint": fingerprint, "last_attempt_at": now, **(fields(now) if fields else {})}
        self._write_command_sync_state(state)

    def _record_command_sync_attempt(self, app_id: Any, fingerprint: str) -> None:
        self._update_command_sync_entry(app_id, fingerprint, keep_existing=True, drop=("last_success_at", "summary"))

    def _record_command_sync_rate_limit(self, app_id: Any, fingerprint: str, retry_after: float) -> None:
        retry_after = max(1.0, float(retry_after))
        self._update_command_sync_entry(
            app_id, fingerprint, keep_existing=True,
            fields=lambda now: {"retry_after_until": time.time() + retry_after, "retry_after": retry_after},
        )

    def _record_command_sync_success(self, app_id: Any, fingerprint: str, summary: dict) -> None:
        self._update_command_sync_entry(
            app_id, fingerprint, keep_existing=False,
            fields=lambda now: {"last_success_at": time.time(), "summary": summary},
        )

    @staticmethod
    def _extract_discord_retry_after(exc: BaseException) -> Optional[float]:
        value = getattr(exc, "retry_after", None)
        if value is not None:
            try:
                return max(1.0, float(value))
            except (TypeError, ValueError):
                return None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            for key in ("Retry-After", "X-RateLimit-Reset-After"):
                try:
                    raw = headers.get(key)
                except Exception:
                    raw = None
                if raw is None:
                    continue
                try:
                    return max(1.0, float(raw))
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _is_discord_rate_limit(exc: BaseException) -> bool:
        """True only for Discord 429 rate-limit exceptions (``RateLimited`` or HTTPException
        status 429) — narrower than ``hasattr(exc, 'retry_after')``."""
        # isinstance-of-class guard: a mocked ``discord`` module has MagicMock attrs, not types.
        if DISCORD_AVAILABLE and discord is not None:
            for attr_name in ("RateLimited", "HTTPException"):
                cls = getattr(discord, attr_name, None)
                if not isinstance(cls, type):
                    continue
                if isinstance(exc, cls):
                    if attr_name == "RateLimited":
                        return True
                    status = getattr(exc, "status", None)
                    if status == 429:
                        return True
        # Duck-type fallback: rate-limit-ish name plus numeric retry_after (mocks, exotic transports).
        name = type(exc).__name__.lower()
        if ("ratelimit" in name or "rate_limit" in name) and getattr(exc, "retry_after", None) is not None:
            return True
        response = getattr(exc, "response", None)
        status = getattr(response, "status", None) or getattr(response, "status_code", None)
        return status == 429

    @staticmethod
    def _is_discord_unknown_interaction(exc: BaseException) -> bool:
        """True for Discord's expired interaction token error."""
        code = getattr(exc, "code", None)
        if code is None:
            data = getattr(exc, "data", None)
            if isinstance(data, dict):
                code = data.get("code")
        try:
            code = int(code)
        except (TypeError, ValueError):
            code = None
        status = getattr(exc, "status", None)
        response = getattr(exc, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status", None) or getattr(response, "status_code", None)
        try:
            status = int(status)
        except (TypeError, ValueError):
            status = None
        message = str(exc).lower()
        return code == 10062 or (status == 404 and "unknown interaction" in message)

    def _command_sync_mutation_interval_seconds(self) -> float:
        return _DISCORD_COMMAND_SYNC_MUTATION_INTERVAL_SECONDS

    async def _sleep_between_command_sync_mutations(self) -> None:
        interval = self._command_sync_mutation_interval_seconds()
        if interval > 0:
            await asyncio.sleep(interval)

    async def _run_post_connect_initialization(self) -> None:
        """Finish non-critical startup work after Discord is connected."""
        if not self._client:
            return
        try:
            sync_policy = self._get_discord_command_sync_policy()
            if sync_policy == "off":
                logger.info("[%s] Skipping Discord slash command sync (policy=off)", self.name)
                return
            if sync_policy == "bulk":
                synced = await asyncio.wait_for(self._client.tree.sync(), timeout=30)
                logger.info("[%s] Synced %d slash command(s) via bulk tree sync", self.name, len(synced))
                return
            app_id = getattr(self._client, "application_id", None) or getattr(getattr(self._client, "user", None), "id", None)
            fingerprint = self._desired_command_sync_fingerprint()
            skip_reason = self._command_sync_skip_reason(app_id, fingerprint)
            if skip_reason:
                logger.info("[%s] Skipping Discord slash command sync: %s", self.name, skip_reason)
                return
            self._record_command_sync_attempt(app_id, fingerprint)
            http = getattr(self._client, "http", None)
            has_ratelimit_timeout = http is not None and hasattr(http, "max_ratelimit_timeout")
            previous_ratelimit_timeout = getattr(http, "max_ratelimit_timeout", None) if has_ratelimit_timeout else None
            if has_ratelimit_timeout:
                http.max_ratelimit_timeout = _DISCORD_COMMAND_SYNC_MAX_RATE_LIMIT_SLEEP_SECONDS
            try:
                # The command-management bucket is small and discord.py may sleep long on a 429: bound it.
                summary = await asyncio.wait_for(self._safe_sync_slash_commands(), timeout=600)
            except Exception as e:
                if not self._is_discord_rate_limit(e):
                    raise
                retry_after = self._extract_discord_retry_after(e)
                if retry_after is None:
                    # Rate-limited with no retry-after: back off a conservative default.
                    retry_after = _DISCORD_COMMAND_SYNC_MAX_RATE_LIMIT_SLEEP_SECONDS
                self._record_command_sync_rate_limit(app_id, fingerprint, retry_after)
                logger.warning(
                    "[%s] Discord rate-limited slash command sync; retrying after %.0fs", self.name,
                    retry_after,
                )
                return
            finally:
                if has_ratelimit_timeout:
                    http.max_ratelimit_timeout = previous_ratelimit_timeout
            self._record_command_sync_success(app_id, fingerprint, summary)
            logger.info(
                "[%s] Safely reconciled %d slash command(s): unchanged=%d updated=%d recreated=%d created=%d deleted=%d",
                self.name, summary["total"], summary["unchanged"], summary["updated"],
                summary["recreated"], summary["created"], summary["deleted"],
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] Slash command sync timed out — Discord rate-limit bucket "
                "may be saturated; will retry on next reconnect",
                self.name,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning("[%s] Slash command sync failed: %s", self.name, e, exc_info=True)

    def _get_discord_command_sync_policy(self) -> str:
        raw = _scoped_gate_env("DISCORD_COMMAND_SYNC_POLICY", "safe").lower()
        if raw in _DISCORD_COMMAND_SYNC_POLICIES:
            return raw
        if raw:
            logger.warning(
                "[%s] Invalid DISCORD_COMMAND_SYNC_POLICY=%r; falling back to 'safe'", self.name,
                raw,
            )
        return "safe"

    def _canonicalize_app_command_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce command payloads to the semantic fields Hermes manages."""
        contexts = payload.get("contexts")
        integration_types = payload.get("integration_types")
        return {
            "type": int(payload.get("type", 1) or 1),
            "name": str(payload.get("name", "") or ""),
            "description": str(payload.get("description", "") or ""),
            "default_member_permissions": self._normalize_permissions(
                payload.get("default_member_permissions")
            ),
            "dm_permission": bool(payload.get("dm_permission", True)),
            "nsfw": bool(payload.get("nsfw", False)),
            "contexts": sorted(int(c) for c in contexts) if contexts else None,
            "integration_types": (
                sorted(int(i) for i in integration_types) if integration_types else None
            ),
            "options": [
                self._canonicalize_app_command_option(item)
                for item in payload.get("options", []) or []
                if isinstance(item, dict)
            ],
        }

    @staticmethod
    def _normalize_permissions(value: Any) -> Optional[str]:
        """Normalize default_member_permissions to str-or-None (Discord returns str, discord.py sets int)."""
        if value is None:
            return None
        return str(value)

    def _existing_command_to_payload(self, command: Any) -> Dict[str, Any]:
        """Build a canonical-ready dict from an AppCommand; ``to_dict()`` omits nsfw/dm_permission/
        default_member_permissions, so pull them from attributes or every startup diffs."""
        payload = dict(command.to_dict())
        nsfw = getattr(command, "nsfw", None)
        if nsfw is not None:
            payload["nsfw"] = bool(nsfw)
        guild_only = getattr(command, "guild_only", None)
        if guild_only is not None:
            payload["dm_permission"] = not bool(guild_only)
        default_permissions = getattr(command, "default_member_permissions", None)
        if default_permissions is not None:
            payload["default_member_permissions"] = getattr(
                default_permissions, "value", default_permissions
            )
        return payload

    def _canonicalize_app_command_option(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": int(payload.get("type", 0) or 0),
            "name": str(payload.get("name", "") or ""),
            "description": str(payload.get("description", "") or ""),
            "required": bool(payload.get("required", False)),
            "autocomplete": bool(payload.get("autocomplete", False)),
            "choices": [
                {
                    "name": str(choice.get("name", "") or ""), "value": choice.get("value"),
                }
                for choice in payload.get("choices", []) or []
                if isinstance(choice, dict)
            ],
            "channel_types": list(payload.get("channel_types", []) or []),
            "min_value": payload.get("min_value"),
            "max_value": payload.get("max_value"),
            "min_length": payload.get("min_length"),
            "max_length": payload.get("max_length"),
            "options": [
                self._canonicalize_app_command_option(item)
                for item in payload.get("options", []) or []
                if isinstance(item, dict)
            ],
        }

    def _patchable_app_command_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fields supported by discord.py's edit_global_command route."""
        canonical = self._canonicalize_app_command_payload(payload)
        return {
            "name": canonical["name"], "description": canonical["description"],
            "options": canonical["options"],
        }

    async def _safe_sync_slash_commands(self) -> Dict[str, int]:
        """Diff existing global commands and only mutate the commands that changed."""
        summary = {"total": 0, "unchanged": 0, "updated": 0, "recreated": 0, "created": 0, "deleted": 0}
        if not self._client:
            return summary
        tree = self._client.tree
        app_id = getattr(self._client, "application_id", None) or getattr(getattr(self._client, "user", None), "id", None)
        if not app_id:
            raise RuntimeError("Discord application ID is unavailable for slash command sync")
        desired_payloads = [command.to_dict(tree) for command in tree.get_commands()]
        desired_by_key = {
            (int(payload.get("type", 1) or 1), str(payload.get("name", "") or "").lower()): payload
            for payload in desired_payloads
        }
        existing_commands = await tree.fetch_commands()
        existing_by_key = {
            (
                int(getattr(getattr(command, "type", None), "value", getattr(command, "type", 1)) or 1),
                str(command.name or "").lower(),
            ): command
            for command in existing_commands
        }
        http = self._client.http
        mutation_count = 0

        async def mutate(call, *args):
            nonlocal mutation_count
            if mutation_count:
                await self._sleep_between_command_sync_mutations()
            result = await call(*args)
            mutation_count += 1
            return result
        # Delete obsolete commands FIRST: an upsert pushing the live total over 100 fails with
        # 30032 (breaks ALL slash commands), so an app at the cap must shrink before creating.
        obsolete_keys = set(existing_by_key.keys()) - set(desired_by_key.keys())
        for key in obsolete_keys:
            current = existing_by_key.pop(key)
            await mutate(http.delete_global_command, app_id, current.id)
            summary["deleted"] += 1
        for key, desired in desired_by_key.items():
            current = existing_by_key.pop(key, None)
            if current is None:
                await mutate(http.upsert_global_command, app_id, desired)
                summary["created"] += 1
                continue
            current_existing_payload = self._existing_command_to_payload(current)
            current_payload = self._canonicalize_app_command_payload(current_existing_payload)
            desired_payload = self._canonicalize_app_command_payload(desired)
            if current_payload == desired_payload:
                summary["unchanged"] += 1
                continue
            if self._patchable_app_command_payload(current_existing_payload) == self._patchable_app_command_payload(desired):
                await mutate(http.delete_global_command, app_id, current.id)
                await mutate(http.upsert_global_command, app_id, desired)
                summary["recreated"] += 1
                continue
            await mutate(http.edit_global_command, app_id, current.id, desired)
            summary["updated"] += 1
        summary["total"] = len(desired_payloads)
        return summary
