from __future__ import annotations

from pathlib import Path

from hermes_cli.config import load_config, set_config_value
from tools.auto_learning_store import AutoLearningStore


LABEL = "Karpathy auto-learning"


def _load_store() -> AutoLearningStore:
    config = load_config()
    auto_cfg = config.get("auto_learning", {}) or {}
    store_path = auto_cfg.get("store_path") or None
    max_entries = int(auto_cfg.get("candidate_max_entries", 200) or 200)
    return AutoLearningStore(path=Path(store_path) if store_path else None, max_entries=max_entries)


def auto_learning_command(args) -> None:
    action = getattr(args, "auto_learning_action", None) or "status"
    config = load_config()
    auto_cfg = config.get("auto_learning", {}) or {}

    if action == "status":
        store = _load_store()
        items = store.list_candidates()
        enabled = bool(auto_cfg.get("enabled", False))
        print(f"{LABEL}: {'enabled' if enabled else 'disabled'}")
        print(f"Store: {store.path}")
        print(f"Candidates: {len(items)}")
        if items:
            by_status = {}
            for item in items:
                by_status[item.get("status", "unknown")] = by_status.get(item.get("status", "unknown"), 0) + 1
            for status, count in sorted(by_status.items()):
                print(f"  {status}: {count}")
        return

    if action == "enable":
        set_config_value("auto_learning.enabled", "true")
        print(f"{LABEL} enabled.")
        return

    if action == "disable":
        set_config_value("auto_learning.enabled", "false")
        print(f"{LABEL} disabled.")
        return

    store = _load_store()

    if action == "list":
        items = store.list_candidates(status=getattr(args, "status", None))
        if not items:
            print("No staged auto-learning candidates.")
            return
        for item in items:
            print(f"{item['id']}  [{item.get('status', 'candidate')}]  {item.get('category', 'unknown')}  {item.get('summary', '')}")
        return

    if action == "promote":
        updated = store.mark_status(args.id, "promoted", note="manual CLI promotion")
        print(f"Promoted {updated['id']}: {updated.get('summary', '')}")
        return

    if action == "reject":
        updated = store.mark_status(args.id, "rejected", note="manual CLI rejection")
        print(f"Rejected {updated['id']}: {updated.get('summary', '')}")
        return

    raise SystemExit(f"Unknown auto-learning action: {action}")
