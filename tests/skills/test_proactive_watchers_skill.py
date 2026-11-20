"""proactive-watchers gate templates: each moment fires once, owed replies are never skipped.

Runs the shipped templates as real subprocesses with a private HERMES_CRON_STATE_DIR, the way the
scheduler runs a gate (``tests/skills`` standard: stdlib + pytest, no network).
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[2] / "skills/productivity/proactive-watchers/templates"


def _run(script: Path, state: Path) -> dict:
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "HERMES_CRON_STATE_DIR": str(state)}, check=True, timeout=60)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_moments_gate_wakes_once_per_due_moment_and_stays_silent_otherwise(tmp_path):
    now = datetime.now(UTC)
    moments = [
        {"id": "due", "at": (now - timedelta(minutes=5)).isoformat(), "label": "check-in opens"},
        {"id": "later", "at": (now + timedelta(hours=3)).isoformat(), "label": "leave for the airport"},
        {"id": "past", "at": (now - timedelta(days=1)).isoformat(), "label": "window closed"},
    ]
    src = (TEMPLATES / "moments_gate.py").read_text(encoding="utf-8")
    gate = tmp_path / "gate.py"
    gate.write_text(src.replace("MOMENTS = [\n", f"MOMENTS = {json.dumps(moments)} + [\n", 1), encoding="utf-8")
    first, second = _run(gate, tmp_path / "state"), _run(gate, tmp_path / "state")
    assert first["wakeAgent"] is True and [m["id"] for m in first["context"]["due"]] == ["due"]
    assert second == {"wakeAgent": False}


def test_unanswered_gate_hands_every_message_over_in_batches(tmp_path):
    messages = [{"ts": f"{1700000000 + i}.0", "text": f"msg {i}"} for i in range(13)]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "hermes"  # stands in for `hermes connectors call`: prints the tool's JSON response
    fake.write_text(f"#!{sys.executable}\nimport json\nprint(json.dumps({{'messages': {messages!r}}}))\n",
                    encoding="utf-8")
    fake.chmod(0o755)
    gate = TEMPLATES / "unanswered_gate.py"
    env_path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    seen, state = [], tmp_path / "state"
    for _ in range(3):
        out = subprocess.run([sys.executable, str(gate)], capture_output=True, text=True, encoding="utf-8",
                             env={**os.environ, "PATH": env_path, "HERMES_CRON_STATE_DIR": str(state)},
                             check=True, timeout=60)
        result = json.loads(out.stdout.strip().splitlines()[-1])
        seen.extend(m["text"] for m in (result.get("context") or {}).get("waiting", []))
    assert seen == [m["text"] for m in messages]  # 10, then 3, then silent: none skipped, none repeated
    assert result == {"wakeAgent": False}
