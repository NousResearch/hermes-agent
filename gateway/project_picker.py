"""Telegram `/new -project` request parsing and project-registry helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_COPY = {
    "usage": "Use /new -project to open the Desktop Project picker. Unknown flags are not accepted.",
    "empty": "No active Desktop Projects with registered folders were found in this profile.",
    "open_failed": "Could not open the project picker. Please try /new -project again.",
    "stale": "This picker is no longer current. Run /new -project again.",
    "cancelled": "Project selection cancelled. Your conversation was not changed.",
    "invalid": "That selection is invalid. Please choose an option from the current picker.",
    "removed": "That project or workspace is no longer available. Your conversation was not changed.",
    "started": "New session started in Desktop Project {project}.",
    "unsupported": "This chat cannot show the project picker.",
    "storage_unavailable": "Session storage is unavailable, so no project session was started. Your conversation was not changed.",
    "create_failed": "Could not start the project session. Your conversation was not changed.",
}


def project_copy(key: str, **values) -> str:
    """User-visible project-picker message; format only trusted server-side values."""
    return _COPY[key].format(**values)


@dataclass(frozen=True)
class NewRequest:
    kind: str
    title: str = ""
    token: str = ""


def parse_new_request(args: str, *, strict_flags: bool) -> NewRequest:
    """Classify `/new` arguments without ever treating an unknown option as a title.

    Whitespace tokens, not shell quoting: a title such as ``it's done`` must stay a title.
    ``strict_flags`` rejects a leading ``-option`` we do not define; the project flag itself
    is rejected anywhere but alone.
    """
    raw = (args or "").strip()
    tokens = raw.split()
    if not tokens:
        return NewRequest("plain")
    is_project = [t.lower() in ("-project", "--project") for t in tokens]
    if is_project[0]:
        return NewRequest("project") if len(tokens) == 1 else NewRequest("invalid", token=tokens[1])
    if any(is_project):
        return NewRequest("invalid", token=tokens[is_project.index(True)])
    if strict_flags and tokens[0].startswith("-") and len(tokens[0]) > 1:
        return NewRequest("invalid", token=tokens[0])
    return NewRequest("plain", title=raw)


def new_request_for_event(event) -> NewRequest:
    """Only Telegram has the picker; every other platform keeps its exact legacy `/new` title rules."""
    args = event.get_command_args()
    if getattr(getattr(event.source, "platform", None), "value", "") != "telegram":
        return NewRequest("plain", title=args.strip())
    return parse_new_request(args, strict_flags=True)


def is_nondestructive_new(event) -> bool:
    """Picker and malformed-option requests do not invoke the reset/interrupt path."""
    if event.get_command() != "new":
        return False
    return new_request_for_event(event).kind != "plain"


def validate_workspace(path: str) -> str:
    """Fail closed unless the still-registered workspace is an accessible, local directory."""
    candidate = Path(os.path.abspath(os.path.expanduser(path)))
    if not candidate.is_absolute() or not candidate.is_dir():
        raise ValueError(
            "That project workspace is no longer available. Please choose another."
        )
    if not os.access(candidate, os.R_OK | os.X_OK):
        raise ValueError(
            "Hermes cannot access that project workspace. Check its permissions."
        )
    return str(candidate)


def list_projects_for_profile(profile_home: Path) -> list[dict]:
    """Profile-scoped registry snapshot; paths remain data, not callback payloads."""
    from hermes_cli import projects_db

    db_path = profile_home / "projects.db"
    if not db_path.is_file():
        return []
    with projects_db.connect_closing(db_path) as conn:
        projects = [
            {
                "id": project.id,
                "name": project.name,
                "slug": project.slug,
                "archived": project.archived,
                "folders": [folder.to_dict() for folder in project.folders],
            }
            for project in projects_db.list_projects(conn, include_archived=True)
        ]
    return sorted(
        projects, key=lambda item: (item["name"].casefold(), item["slug"], item["id"])
    )


_LABEL_MAX = 60  # Telegram button text cap; fit here so the distinguishing suffix is never cut


def eligible_projects(projects: list[dict]) -> list[dict]:
    """The one eligibility rule for listing AND revalidating: active with a registered folder."""
    return [p for p in projects if not p["archived"] and p["folders"]]


def _fit(base: str, suffix: str = "") -> str:
    """Truncate *base* (never *suffix*) so the whole label fits one Telegram button."""
    room = _LABEL_MAX - len(suffix)
    if len(base) > room:
        base = base[: max(1, room - 1)] + "…"
    return base + suffix


def _folder_name(folder: dict) -> str:
    return folder.get("label") or Path(folder["path"]).name or folder["path"]


def _folder_labels(folders: list[dict]) -> list[str]:
    """Distinct labels: duplicates gain the shortest trailing path that tells them apart, then an index."""
    names = [_folder_name(f) for f in folders]
    labels = list(names)
    for depth in (1, 2, 3):
        for name in {n for n in labels if labels.count(n) > 1}:
            for i, folder in enumerate(folders):
                if labels[i] == name:
                    tail = "/".join(Path(folder["path"]).parts[-depth - 1 : -1])
                    labels[i] = f"{names[i]} · {tail}" if tail else names[i]
    for label in {n for n in labels if labels.count(n) > 1}:
        for ordinal, i in enumerate([i for i, n in enumerate(labels) if n == label], start=1):
            labels[i] = f"{label} #{ordinal}"
    return labels


def picker_page(
    state: dict, page: int, *, folder: bool = False
) -> tuple[str, list[dict], list[dict]]:
    """Stable 8-row page; duplicate project names carry their registry slug, never truncated away."""
    items = state.get("folders", []) if folder else state.get("projects", [])
    size = 8
    pages = max(1, (len(items) + size - 1) // size)
    page = max(0, min(page, pages - 1))
    state["page"] = page
    if folder:
        labels = [_fit(label) for label in _folder_labels(items)]
        prefix = "folder_"
    else:
        counts: dict[str, int] = {}
        for item in items:
            counts[item["name"]] = counts.get(item["name"], 0) + 1
        labels = [
            _fit(item["name"], f" · {item['slug']}" if counts[item["name"]] > 1 else "")
            for item in items
        ]
        prefix = "select_"
    start = page * size
    choices = [
        {"label": labels[i], "action": f"{prefix}{i}"}
        for i in range(start, min(start + size, len(items)))
    ]
    controls = []
    if page:
        controls.append({"label": "◀", "action": f"page_{page - 1}"})
    controls.append({"label": "Cancel", "action": "cancel"})
    if page + 1 < pages:
        controls.insert(-1, {"label": "▶", "action": f"page_{page + 1}"})
    if folder:
        controls.insert(0, {"label": "Back", "action": "back"})
    title = "Choose a Hermes Desktop project"
    if folder:
        title = f"Choose a workspace for {state.get('project_name') or 'this project'}"
    if not items:
        title = (
            "No registered workspace folders"
            if folder
            else "No active projects are registered"
        )
    return title, choices, controls
