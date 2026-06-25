"""Background loop for VRChat autonomy profile ticks."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    print(f"{stamp} {message}", flush=True)


def main() -> int:
    profile = os.environ.get("HERMES_VRCHAT_AUTONOMY_PROFILE", "").strip()
    interval = float(os.environ.get("HERMES_VRCHAT_AUTONOMY_INTERVAL", "15") or "15")
    interval = max(5.0, min(interval, 300.0))

    from tools.openclaw.vrchat_autonomy import vrchat_autonomy_profile_tick

    _log(f"vrchat-autonomy worker starting interval={interval}s profile={profile or 'default'}")
    while True:
        try:
            result = vrchat_autonomy_profile_tick(
                profile_path=profile or None,
            )
            code = result.get("code") or (result.get("tick") or {}).get("code")
            _log(f"tick code={code} success={result.get('success')}")
            if os.environ.get("HERMES_VRCHAT_AUTONOMY_DEBUG"):
                _log(json.dumps(result, ensure_ascii=False)[:2000])
        except Exception as exc:
            _log(f"tick_error {type(exc).__name__}: {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
