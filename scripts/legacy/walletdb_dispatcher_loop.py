import json, os, subprocess, sys, time

# Robust, read-only dispatcher loop (for UI/debug only).
# Delivery is owned by walletdb-alerts-deliver.service; this script must never block or write.

DB_PATH = os.environ.get("WALLETDB_DB_PATH") or "data/walletdb.sqlite"
LIMIT = str(int(os.environ.get("WALLETDB_DISPATCH_LIMIT") or "10"))
TIMEOUT_SEC = float(os.environ.get("WALLETDB_DISPATCH_TIMEOUT_SEC") or "6")

VENV_PY = os.environ.get("WALLETDB_VENV_PY") or "/home/yeqiuqiu/projects/walletdb/.venv/bin/python"

CMD = [
    VENV_PY,
    "-m",
    "walletdb.cli",
    "alerts",
    "dispatch",
    "--db-path",
    DB_PATH,
    "--limit",
    LIMIT,
    "--no-mark",  # force peek/read-only
    "--json",
]

ERR_PATH = os.environ.get("WALLETDB_ERROR_PATH") or "/tmp/walletdb_error.txt"

start = time.time()
alerts_out: list[str] = []

while time.time() - start < 55:
    try:
        p = subprocess.run(
            CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        # Don't hard-fail the UI; write a short error note and exit cleanly.
        try:
            with open(ERR_PATH, "a", encoding="utf-8") as f:
                f.write(f"dispatch_timeout after {TIMEOUT_SEC}s\n")
        except Exception:
            pass
        print(json.dumps({"alerts": []}, ensure_ascii=False))
        sys.exit(0)
    except Exception as e:
        try:
            with open(ERR_PATH, "a", encoding="utf-8") as f:
                f.write(f"dispatch_exec_error: {type(e).__name__}: {str(e)[:200]}\n")
        except Exception:
            pass
        print(json.dumps({"alerts": []}, ensure_ascii=False))
        sys.exit(0)

    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        err_last = err.splitlines()[-1] if err else f"exit {p.returncode}"
        # Treat lock errors as non-fatal (UI should still render).
        try:
            with open(ERR_PATH, "a", encoding="utf-8") as f:
                f.write(f"dispatch_failed rc={p.returncode}: {err_last[:200]}\n")
        except Exception:
            pass
        print(json.dumps({"alerts": []}, ensure_ascii=False))
        sys.exit(0)

    out = (p.stdout or "").strip()
    if out:
        try:
            data = json.loads(out)
        except Exception:
            alerts_out.append(out)
        else:
            items = None
            if isinstance(data, dict):
                items = data.get("alerts") or data.get("items")
            if items is None:
                items = data

            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                items = []

            for a in items:
                if isinstance(a, str):
                    alerts_out.append(a)
                elif isinstance(a, dict):
                    alerts_out.append(
                        a.get("text")
                        or a.get("message")
                        or a.get("alert")
                        or json.dumps(a, ensure_ascii=False)
                    )
                else:
                    alerts_out.append(str(a))

    time.sleep(5)

print(json.dumps({"alerts": alerts_out}, ensure_ascii=False))
