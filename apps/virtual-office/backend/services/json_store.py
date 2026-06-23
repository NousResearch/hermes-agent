import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("VIRTUAL_OFFICE_DATA_ROOT", str(PROJECT_ROOT / "data")))
TASKS_PATH = DATA_ROOT / "tasks" / "tasks.json"
HANDOFFS_PATH = DATA_ROOT / "handoffs" / "handoffs.json"


def ensure_list_store(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]", encoding="utf-8")


def read_list_store(path: Path) -> list[dict[str, Any]]:
    ensure_list_store(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def write_list_store(path: Path, items: list[dict[str, Any]]) -> None:
    ensure_list_store(path)
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")


def find_by_id(items: list[dict[str, Any]], item_id: str) -> dict[str, Any] | None:
    return next((item for item in items if item.get("id") == item_id), None)
