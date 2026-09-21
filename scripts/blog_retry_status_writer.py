#!/usr/bin/env python3
"""Write blog-failed-retry status JSON from env vars."""
import json
import os
import re
import datetime
import sys

rc = int(os.environ["BLOG_STATUS_RC"])
status_path = os.environ["BLOG_STATUS_PATH"]
out = os.environ.get("BLOG_STATUS_RAW", "")
status = {
    "rc": rc,
    "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "raw": out[-2000:],
}
m = re.search(r"retry_all_pending_images:\s*\{?(.*?)\}?$", out, re.S)
if m:
    blob = m.group(1)
    for key in ("recovered", "still_failed", "no_draft", "deferred", "idle"):
        km = re.search(r"['\"]?" + re.escape(key) + r"['\"]?\s*:\s*(\[[^\]]*\]|\w+)", blob)
        if km:
            val = km.group(1)
            if val.startswith("["):
                # Try JSON first, fall back to comma-split for unquoted items
                try:
                    status[key] = json.loads(val)
                except json.JSONDecodeError:
                    items = [x.strip().strip("'\"") for x in val.strip("[]").split(",") if x.strip()]
                    status[key] = items
            else:
                status[key] = val
with open(status_path, "w") as f:
    json.dump(status, f, indent=2)