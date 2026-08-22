#!/usr/bin/env python3
"""
Quản lý thứ tự pool openai-codex trong codexpool.

Dùng:
  sudo python3 manage-codexpool.py                        # xem thứ tự hiện tại
  sudo python3 manage-codexpool.py reorder                # sắp lại theo thứ tự mặc định
  sudo python3 manage-codexpool.py reorder nocobase,zeo,leo,neo,llgap
"""
import json
import shutil
import sys
from pathlib import Path

AUTH = Path("/root/.hermes/profiles/codexpool/auth.json")
DEFAULT_ORDER = ["nocobase", "zeo", "leo", "neo", "llgap"]


def load():
    data = json.loads(AUTH.read_text())
    return data, data["credential_pool"]["openai-codex"]


def show():
    _, pool = load()
    print(f"openai-codex ({len(pool)} credentials):")
    for e in pool:
        status = e.get("last_status") or "ok"
        print(f"  #{e['priority']+1}  {e['label']:<12}  {status}")


def reorder(order: list[str]):
    data, pool = load()
    pool_by_label = {e["label"]: e for e in pool}

    missing = [l for l in order if l not in pool_by_label]
    if missing:
        print(f"ERROR: labels không tìm thấy trong pool: {missing}")
        print(f"Pool hiện có: {list(pool_by_label.keys())}")
        sys.exit(1)

    backup = str(AUTH) + ".bak-reorder"
    shutil.copy(AUTH, backup)
    print(f"Backup: {backup}")

    new_pool = []
    for i, label in enumerate(order):
        entry = pool_by_label[label]
        entry["priority"] = i
        new_pool.append(entry)

    data["credential_pool"]["openai-codex"] = new_pool
    AUTH.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    print("Done. Thứ tự mới:")
    for e in new_pool:
        status = e.get("last_status") or "ok"
        print(f"  #{e['priority']+1}  {e['label']:<12}  {status}")


def main():
    args = sys.argv[1:]

    if not args:
        show()
        return

    if args[0] == "reorder":
        order = args[1].split(",") if len(args) > 1 else DEFAULT_ORDER
        order = [x.strip() for x in order]
        reorder(order)
        return

    print(__doc__)
    sys.exit(1)


if __name__ == "__main__":
    main()
