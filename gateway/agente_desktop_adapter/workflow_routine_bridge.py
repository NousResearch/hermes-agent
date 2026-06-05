"""
workflow_routine_bridge.py

Bridges for:
- Desktop arg shape normalization (_normalize_desktop_tool_args, _stable_desktop_id etc)
  used by the /api/tools/{tool} dispatch for save_workflow/create_routine.
- Mirroring of workflows/routines YAMLs to $AGENTE_BOUND_FOLDER so companion
  UIs pick them up (patched into the core storage at adapter init time).
- Workflow dispatch URL resolution (AGENTE_DESKTOP_BASE_URL support) patched
  into cron.workflow_dispatch.
- Workflow cron registry AGENTE_DESKTOP_WORKFLOWS_DIR support patched.
- Any future schema parity for create_ticket etc.

This isolates all the "desktop writes YAMLs / uses alternate arg keys / delegates
to companion workflow engine" seams out of core tools/ and cron/ files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _stable_desktop_id(prefix: str, value: Any) -> str:
    """Derive a Hermes-safe ID from Desktop-facing free text. (moved from api_server)"""
    import hashlib
    import re
    import uuid as _uuid

    raw = str(value or "").strip()
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10] if raw else _uuid.uuid4().hex[:10]
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", raw.lower()).strip("-_")
    if not base:
        return f"{prefix}-{digest}"
    return f"{prefix}-{base[:80]}-{digest}"


def _normalize_desktop_action_step(step: Any) -> Dict[str, Any]:
    """Normalize a step in a desktop save_workflow actions list. (moved)"""
    if not isinstance(step, dict):
        return {"kind": str(step)}
    clone = dict(step)
    kind = (
        clone.get("kind")
        or clone.get("action")
        or clone.get("tool")
        or clone.get("toolName")
        or clone.get("name")
    )
    if isinstance(kind, str) and kind.strip():
        clone["kind"] = kind.strip()
    return clone


def _normalize_desktop_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Accept the Desktop HTTP proxy schema while preserving native Hermes schemas. (moved)"""
    if tool_name == "save_workflow" and (
        "name" in args or "description" in args or "steps" in args
    ):
        name = args.get("name_he") or args.get("name") or args.get("id") or "workflow"
        trigger = args.get("trigger")
        trigger_kind = args.get("trigger_kind") or args.get("triggerKind")
        if isinstance(trigger, dict):
            trigger_kind = trigger.get("kind") or trigger_kind
        steps = args.get("actions")
        if steps is None:
            steps = args.get("steps")
        if isinstance(steps, list):
            actions = [_normalize_desktop_action_step(step) for step in steps]
        else:
            actions = []
        return {
            **args,
            "id": args.get("id") or _stable_desktop_id("wf", name),
            "name_he": name,
            "description_he": args.get("description_he") or args.get("description") or "",
            "trigger_kind": trigger_kind or "manual",
            "triage_keywords_he": args.get("triage_keywords_he") or args.get("triageKeywordsHe"),
            "actions": actions,
        }

    if tool_name == "create_routine" and (
        "name" in args or "cron" in args or "schedule" in args or "timezone" in args
    ):
        name = args.get("name_he") or args.get("name") or args.get("workflow_id") or "routine"
        cron_schedule = args.get("cron_schedule") or args.get("cron") or args.get("schedule")
        return {
            **args,
            "id": args.get("id") or _stable_desktop_id("rt", name),
            "name_he": name,
            "cron_schedule": cron_schedule,
            "natural_language_schedule_he": (
                args.get("natural_language_schedule_he")
                or args.get("naturalLanguageScheduleHe")
                or args.get("timezone")
                or ""
            ),
        }

    return args


# --- mirror patching for AGENTE_BOUND_FOLDER (desktop seam, no longer in core storage) ---

def _bound_workflows_dir() -> Optional[Path]:
    bound = os.environ.get("AGENTE_BOUND_FOLDER", "").strip()
    if not bound:
        return None
    path = Path(bound) / "office" / "workflows"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path


def _bound_routines_dir() -> Optional[Path]:
    bound = os.environ.get("AGENTE_BOUND_FOLDER", "").strip()
    if not bound:
        return None
    path = Path(bound) / "office" / "routines"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path


def _wrap_save_with_mirror(orig_save, bound_getter):
    def wrapped(record: Dict[str, Any]) -> Dict[str, Any]:
        saved = orig_save(record)
        bound = bound_getter()
        if bound is not None:
            try:
                import yaml
                from tools.workflow_storage import _workflow_path as _wf_path  # type: ignore[attr-defined]
                # routines use their own
                if "routines" in str(bound):
                    from tools.routine_storage import _routine_path as _rt_path  # type: ignore[attr-defined]
                    _atomic = None
                    try:
                        from tools.routine_storage import _atomic_write_yaml as _aw  # type: ignore
                        _atomic = _aw
                    except Exception:
                        pass
                    if _atomic:
                        _atomic(_rt_path(bound, saved["id"]), saved)
                else:
                    try:
                        from tools.workflow_storage import _atomic_write_yaml as _aw  # type: ignore
                        _aw(_wf_path(bound, saved["id"]), saved)
                    except Exception:
                        pass
            except Exception:
                # best effort mirror, never fail the canonical save
                pass
        return saved
    return wrapped


# --- dispatch URL + workflows dir patching (AGENTE_DESKTOP_* support) ---

def _patch_cron_workflow_dispatch() -> None:
    """Patch resolve_dispatch_url to honor AGENTE_DESKTOP_BASE_URL (in addition to HERMES_)."""
    try:
        import cron.workflow_dispatch as wd
    except Exception:
        return

    _orig_resolve = getattr(wd, "resolve_dispatch_url", None)
    if _orig_resolve is None:
        return

    def _patched_resolve() -> Optional[str]:
        explicit = (os.getenv("HERMES_WORKFLOW_DISPATCH_URL") or "").strip()
        if explicit:
            return explicit
        base = (os.getenv("AGENTE_DESKTOP_BASE_URL") or "").strip().rstrip("/")
        if base:
            return f"{base}/api/workflow-runs"
        return None

    wd.resolve_dispatch_url = _patched_resolve  # type: ignore[attr-defined]
    logger.debug("agente_desktop_adapter: patched cron.workflow_dispatch.resolve_dispatch_url for AGENTE_DESKTOP_BASE_URL")


def _patch_workflow_cron_registry() -> None:
    """Patch _resolve_workflows_dir to honor AGENTE_DESKTOP_WORKFLOWS_DIR."""
    try:
        import hermes_cli.workflow_cron_registry as wcr
    except Exception:
        return

    _orig = getattr(wcr, "_resolve_workflows_dir", None)
    if _orig is None:
        return

    def _patched_resolve() -> Optional[Path]:
        env = os.environ.get("AGENTE_DESKTOP_WORKFLOWS_DIR", "").strip()
        if env:
            p = Path(env).expanduser()
            if p.is_dir():
                return p
            logger.debug(
                "AGENTE_DESKTOP_WORKFLOWS_DIR=%s does not exist; falling back",
                env,
            )
        try:
            from hermes_constants import get_hermes_home
            fallback = get_hermes_home() / "connected" / "workflows"
            return fallback if fallback.is_dir() else None
        except Exception:
            return None

    wcr._resolve_workflows_dir = _patched_resolve  # type: ignore[attr-defined]
    logger.debug("agente_desktop_adapter: patched hermes_cli.workflow_cron_registry for AGENTE_DESKTOP_WORKFLOWS_DIR")


def register(app: Any, adapter: Any = None) -> None:
    """Apply workflow/routine/desktop parity bridges at gateway startup."""
    # Apply storage mirror patches (so AGENTE_BOUND_FOLDER works, but code+strings live only here)
    try:
        import tools.workflow_storage as ws
        if not getattr(ws.save_workflow, "_agente_mirror_wrapped", False):
            ws.save_workflow = _wrap_save_with_mirror(ws.save_workflow, _bound_workflows_dir)
            setattr(ws.save_workflow, "_agente_mirror_wrapped", True)
    except Exception as exc:
        logger.debug("agente_desktop_adapter mirror patch (workflows) skipped: %s", exc)

    try:
        import tools.routine_storage as rs
        if not getattr(rs.save_routine, "_agente_mirror_wrapped", False):
            rs.save_routine = _wrap_save_with_mirror(rs.save_routine, _bound_routines_dir)
            setattr(rs.save_routine, "_agente_mirror_wrapped", True)
    except Exception as exc:
        logger.debug("agente_desktop_adapter mirror patch (routines) skipped: %s", exc)

    _patch_cron_workflow_dispatch()
    _patch_workflow_cron_registry()

    # Future: could also register /api/jobs overrides or ticket related here if needed.
    logger.debug("agente_desktop_adapter.workflow_routine_bridge: bridges applied")
