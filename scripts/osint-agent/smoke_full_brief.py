"""Run full osint-agent brief (SitDeck included) and save markdown incrementally."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
OUT_DIR = HERMES_HOME / "osint-agent" / "briefs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M")
out_path = OUT_DIR / f"{stamp}_smoke_full_sitdeck.md"
log_path = OUT_DIR / f"{stamp}_smoke_full_sitdeck.log"


def main() -> int:
    env = os.environ.copy()
    env.setdefault("HERMES_HOME", str(HERMES_HOME))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    py = REPO / ".venv" / "Scripts" / "python.exe"
    if not py.is_file():
        py = Path(sys.executable)

    cmd = [
        str(py),
        "-m",
        "hermes_cli.main",
        "osint-agent",
        "brief",
        "--slot",
        "morning",
        "--source-mode",
        "real",
        "--wm-tier",
        "free",
    ]
    log_path.write_text(f"started {datetime.now().isoformat()}\ncmd: {' '.join(cmd)}\n", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    lines: list[str] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for line in proc.stdout:
            lines.append(line)
            fh.write(line)
            fh.flush()
    code = proc.wait(timeout=1)
    summary = (
        f"\n---\nexit_code={code}\nlines={len(lines)}\n"
        f"out={out_path}\nfinished={datetime.now().isoformat()}\n"
    )
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(summary)
    print(summary, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
