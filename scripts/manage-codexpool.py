#!/usr/bin/env python3
"""
Quản lý thứ tự pool openai-codex trong codexpool.

Dùng:
  sudo python3 manage-codexpool.py                        # xem thứ tự và trạng thái quota thực tế
  sudo python3 manage-codexpool.py --raw                  # chỉ đọc trạng thái last_status trong auth.json
  sudo python3 manage-codexpool.py reorder                # sắp lại theo thứ tự mặc định
  sudo python3 manage-codexpool.py reorder leo,nocobase,zeo,neo,llgap
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/root/.hermes/hermes-agent")
try:
    from agent.account_usage import _fetch_codex_account_usage
except ImportError:
    _fetch_codex_account_usage = None

AUTH = Path("/root/.hermes/profiles/codexpool/auth.json")
DEFAULT_ORDER = ["leo", "nocobase", "zeo", "neo", "llgap"]


def load():
    data = json.loads(AUTH.read_text())
    return data, data["credential_pool"]["openai-codex"]


def get_credential_status(entry: dict, raw: bool = False) -> str:
    runtime_status = entry.get("last_status") or "ok"
    if raw or not _fetch_codex_account_usage:
        return runtime_status

    token = str(entry.get("access_token") or "").strip()
    if not token:
        return "missing token"

    try:
        snapshot = _fetch_codex_account_usage(base_url=entry.get("base_url"), api_key=token)
        if snapshot and getattr(snapshot, "windows", None):
            min_rem = 100
            worst_window = None
            for w in snapshot.windows:
                if w.used_percent is not None:
                    rem = max(0, round(100 - float(w.used_percent)))
                    if rem < min_rem:
                        min_rem = rem
                        worst_window = w
            plan_tag = f"({snapshot.plan})" if snapshot.plan else ""
            if worst_window and min_rem <= 2:
                return f"exhausted  {plan_tag:<10}  0% {worst_window.label.lower()} remaining"
            elif worst_window:
                return f"ok         {plan_tag:<10}  {min_rem}% {worst_window.label.lower()} remaining"
    except Exception:
        pass

    return runtime_status


def show(raw: bool = False):
    _, pool = load()
    print(f"openai-codex ({len(pool)} credentials):")
    for e in pool:
        status = get_credential_status(e, raw=raw)
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
    show()


def main():
    args = sys.argv[1:]

    if not args:
        show()
        return

    if args[0] in ("--raw", "-r", "--offline"):
        show(raw=True)
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
