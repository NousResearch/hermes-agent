"""Build ephemeral project overlays for coordinator chat requests."""

from __future__ import annotations

from typing import Any, Optional


def _normalize_project_context(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    repositories = []
    for item in value.get("repositories") or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        label = str(item.get("label") or "").strip()
        repositories.append({
            "label": label or None,
            "path": path,
        })

    work_items = [
        str(item).strip()
        for item in (value.get("work_items") or [])
        if str(item).strip()
    ]

    context = {
        "project_id": str(value.get("project_id") or "").strip() or None,
        "project_name": str(value.get("project_name") or "").strip() or None,
        "vision": str(value.get("vision") or "").strip() or None,
        "coordinator_profile": str(value.get("coordinator_profile") or "").strip() or None,
        "repositories": repositories[:10],
        "work_items": work_items[:20],
    }
    if not any(context.get(key) for key in ("project_id", "project_name", "vision", "coordinator_profile")) and not repositories and not work_items:
        return None
    return context


def build_chat_project_context_overlay(body: Optional[dict[str, Any]]) -> Optional[str]:
    """Return markdown overlay text for project-scoped chat completions."""
    if not isinstance(body, dict):
        return None

    sections: list[str] = [
        "# Oryn project coordinator context",
        "Use this as live project state for the current coordinator turn.",
    ]

    project_id = str(body.get("project_id") or "").strip()
    if project_id:
        sections.append(f"Scoped project_id: {project_id}")

    coordinator_profile = str(body.get("coordinator_profile") or "").strip()
    if coordinator_profile:
        sections.append(f"Coordinator profile: {coordinator_profile}")

    project_context = _normalize_project_context(body.get("project_context"))
    if project_context:
        sections.append("")
        sections.append("## Structured project context")
        if project_context.get("project_name"):
            sections.append(f"- Project name: {project_context['project_name']}")
        if project_context.get("vision"):
            sections.append(f"- Vision: {project_context['vision']}")
        if project_context.get("coordinator_profile"):
            sections.append(f"- Coordinator profile: {project_context['coordinator_profile']}")
        if project_context.get("repositories"):
            sections.append("- Repositories:")
            for repo in project_context["repositories"]:
                label = repo.get("label") or "Repository"
                sections.append(f"  - {label}: {repo.get('path')}")
        if project_context.get("work_items"):
            sections.append("- Work items:")
            for item in project_context["work_items"]:
                sections.append(f"  - {item}")

    snapshot = str(body.get("project_dashboard_snapshot") or "").strip()
    if snapshot:
        sections.append("")
        sections.append(snapshot)

    if len(sections) <= 2:
        return None
    return "\n".join(sections)
