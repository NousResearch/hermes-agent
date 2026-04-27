#!/usr/bin/env python3
"""
ProviderRouter — intelligent model selection for Hermes Code Mode.

Routes provider/model selection per CodeSession based on task type,
with fallback chains and cost tracking.

Preset task types:
  - fast: cheapest model for quick iterations
  - strong: best model for complex reasoning
  - cheap: most cost-effective option
  - reviewer: model specialized for code review
  - planner: balanced model for planning/architecture
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default model presets — used when no session-specific overrides exist
DEFAULT_PRESETS: Dict[str, Dict[str, str]] = {
    "fast": {"provider": "openrouter", "model": "anthropic/claude-haiku-4.5"},
    "strong": {"provider": "openrouter", "model": "anthropic/claude-opus-4.6"},
    "cheap": {"provider": "openrouter", "model": "anthropic/claude-haiku-4.5"},
    "reviewer": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
    "planner": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
}

VALID_TASK_TYPES = {"fast", "strong", "cheap", "reviewer", "planner"}


class ProviderRouter:
    """Business logic for model selection and cost tracking per CodeSession.

    Presets allow quick switching between models optimized for different tasks.
    Falls back to built-in defaults when no session-specific presets are configured.
    """

    def __init__(self, db_path: Optional[Path] = None, realtime_hub=None):
        self._db_path = db_path
        self._realtime_hub = realtime_hub

    def _router_db(self):
        from hermes_state import ProviderRouterDB

        return ProviderRouterDB(db_path=self._db_path)

    def _session_db(self):
        from hermes_state import CodeSessionDB

        return CodeSessionDB(db_path=self._db_path)

    async def _broadcast(self, event_type: str, payload: dict):
        if self._realtime_hub:
            try:
                await self._realtime_hub.broadcast(event_type, payload)
            except Exception:
                pass

    def _add_timeline_event(
        self,
        code_session_id: str,
        event_type: str,
        message: str,
        payload: dict,
    ):
        db = self._session_db()
        try:
            db.add_event(code_session_id, event_type, message=message, payload=payload)
        except Exception:
            pass
        finally:
            db.close()

    def _validate_session(self, code_session_id: str) -> dict:
        """Validate that the code session exists. Returns session dict."""
        db = self._session_db()
        try:
            session = db.get_session(code_session_id)
            if not session:
                raise ValueError(f"CodeSession not found: {code_session_id}")
            return session
        finally:
            db.close()

    def select_model(
        self,
        code_session_id: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """Select provider/model based on task type for a session.

        Checks session-specific presets first, falls back to built-in defaults.
        Returns the resolved provider/model selection.
        """
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type: {task_type}. "
                f"Must be one of: {', '.join(sorted(VALID_TASK_TYPES))}"
            )

        session = self._validate_session(code_session_id)

        # Check session-specific preset
        preset = None
        rdb = self._router_db()
        try:
            preset = rdb.get_preset_by_name(code_session_id, task_type)
        finally:
            rdb.close()

        if preset:
            result = {
                "task_type": task_type,
                "provider": preset["provider"],
                "model": preset["model"],
                "source": "session_preset",
                "preset_id": preset["id"],
            }
        else:
            # Fall back to default preset
            default = DEFAULT_PRESETS.get(task_type, DEFAULT_PRESETS["fast"])
            result = {
                "task_type": task_type,
                "provider": default["provider"],
                "model": default["model"],
                "source": "default",
                "preset_id": None,
            }

        # Update session provider/model
        sdb = self._session_db()
        try:
            sdb.update_session(
                code_session_id,
                {
                    "provider": result["provider"],
                    "model": result["model"],
                },
            )
        except Exception:
            pass
        finally:
            sdb.close()

        self._add_timeline_event(
            code_session_id,
            "provider.model_selected",
            message=f"Model selected: {result['model']} ({task_type})",
            payload=result,
        )

        return result

    def get_session_model(self, code_session_id: str) -> Dict[str, Any]:
        """Get current provider/model for a session."""
        session = self._validate_session(code_session_id)
        return {
            "code_session_id": code_session_id,
            "provider": session.get("provider"),
            "model": session.get("model"),
        }

    def update_session_model(
        self,
        code_session_id: str,
        provider: str,
        model: str,
    ) -> Dict[str, Any]:
        """Update provider/model for a session directly."""
        session = self._validate_session(code_session_id)

        old_provider = session.get("provider")
        old_model = session.get("model")

        sdb = self._session_db()
        try:
            updated = sdb.update_session(
                code_session_id,
                {
                    "provider": provider,
                    "model": model,
                },
            )
        finally:
            sdb.close()

        self._add_timeline_event(
            code_session_id,
            "provider.model_updated",
            message=f"Model updated: {old_model} -> {model}",
            payload={
                "old_provider": old_provider,
                "old_model": old_model,
                "new_provider": provider,
                "new_model": model,
            },
        )

        return {
            "code_session_id": code_session_id,
            "provider": provider,
            "model": model,
            "old_provider": old_provider,
            "old_model": old_model,
        }

    # ── Presets ──

    def list_presets(self, code_session_id: str) -> List[Dict[str, Any]]:
        """List all presets for a session."""
        self._validate_session(code_session_id)
        rdb = self._router_db()
        try:
            return rdb.list_presets(code_session_id)
        finally:
            rdb.close()

    def create_preset(
        self,
        code_session_id: str,
        name: str,
        provider: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a model preset for a session."""
        self._validate_session(code_session_id)

        if name not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid preset name: {name}. "
                f"Must be one of: {', '.join(sorted(VALID_TASK_TYPES))}"
            )

        rdb = self._router_db()
        try:
            existing = rdb.get_preset_by_name(code_session_id, name)
            if existing:
                return rdb.update_preset(
                    existing["id"],
                    provider=provider,
                    model=model,
                    metadata=metadata,
                )
            preset = rdb.create_preset(
                code_session_id=code_session_id,
                name=name,
                provider=provider,
                model=model,
                metadata=metadata,
            )
        finally:
            rdb.close()

        self._add_timeline_event(
            code_session_id,
            "provider.preset_created",
            message=f"Preset '{name}': {provider}/{model}",
            payload={"name": name, "provider": provider, "model": model},
        )

        return preset

    def delete_preset(self, code_session_id: str, preset_id: str) -> bool:
        """Delete a preset by ID."""
        self._validate_session(code_session_id)
        rdb = self._router_db()
        try:
            return rdb.delete_preset(preset_id)
        finally:
            rdb.close()

    # ── Cost Tracking ──

    def track_cost(
        self,
        code_session_id: str,
        provider: str,
        model: str,
        task_type: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost_usd: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record token usage and cost for a session iteration."""
        self._validate_session(code_session_id)

        rdb = self._router_db()
        try:
            entry = rdb.add_cost_entry(
                code_session_id=code_session_id,
                provider=provider,
                model=model,
                task_type=task_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cost_usd=cost_usd,
                metadata=metadata,
            )
        finally:
            rdb.close()

        return entry

    def get_session_cost_summary(self, code_session_id: str) -> Dict[str, Any]:
        """Get cost summary for a session."""
        self._validate_session(code_session_id)
        rdb = self._router_db()
        try:
            return rdb.get_cost_summary(code_session_id)
        finally:
            rdb.close()

    def list_cost_entries(
        self,
        code_session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List cost entries for a session."""
        self._validate_session(code_session_id)
        rdb = self._router_db()
        try:
            return rdb.list_cost_entries(code_session_id, limit=limit, offset=offset)
        finally:
            rdb.close()

    def get_presets_summary(self, code_session_id: str) -> Dict[str, Any]:
        """Get all presets with defaults filled in for a session."""
        self._validate_session(code_session_id)

        rdb = self._router_db()
        try:
            session_presets = rdb.list_presets(code_session_id)
        finally:
            rdb.close()

        preset_map = {p["name"]: p for p in session_presets}

        result = {}
        for task_type in sorted(VALID_TASK_TYPES):
            if task_type in preset_map:
                p = preset_map[task_type]
                result[task_type] = {
                    "provider": p["provider"],
                    "model": p["model"],
                    "source": "session",
                    "preset_id": p["id"],
                }
            else:
                default = DEFAULT_PRESETS[task_type]
                result[task_type] = {
                    "provider": default["provider"],
                    "model": default["model"],
                    "source": "default",
                    "preset_id": None,
                }

        return result
