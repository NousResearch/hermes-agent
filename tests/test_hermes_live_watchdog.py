import importlib.util
import json
import sys
import time
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hermes_live_watchdog.py"
SPEC = importlib.util.spec_from_file_location("hermes_live_watchdog", SCRIPT)
watchdog = importlib.util.module_from_spec(SPEC)
assert SPEC is not None
assert SPEC.loader is not None
sys.modules[SPEC.name] = watchdog
SPEC.loader.exec_module(watchdog)


def test_discover_ledgers_includes_profile_ledgers(tmp_path):
    home = tmp_path / ".hermes"
    profile_ledger = home / "profiles" / "film-maker" / "logs" / "turn-watchdog.jsonl"
    profile_ledger.parent.mkdir(parents=True)
    profile_ledger.write_text("", encoding="utf-8")

    assert watchdog.discover_ledgers(home) == [
        home / "logs" / "turn-watchdog.jsonl",
        profile_ledger,
    ]


def test_profile_ledger_stuck_turn_writes_shared_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "process_snapshot", lambda: [])
    home = tmp_path / ".hermes"
    ledger = home / "profiles" / "film-maker" / "logs" / "turn-watchdog.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "monotonic": time.monotonic() - 10,
                "event": "approval.request",
                "session_id": "sid-profile",
                "session_key": "key-profile",
                "cwd": "/tmp/project",
                "running": True,
                "turn_started_at": time.monotonic() - 20,
                "turn_last_progress_at": time.monotonic() - 10,
                "turn_last_progress_event": "approval.request",
                "payload": {},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    args = watchdog.parse_args(
        [
            "--home",
            str(home),
            "--idle-seconds",
            "1",
            "--alert-cooldown",
            "0",
            "--from-start",
            "--once",
        ]
    )

    assert watchdog.run(args) == 0

    alerts = home / "logs" / "live-watchdog-alerts.jsonl"
    rows = [json.loads(line) for line in alerts.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event"] == "turn_stuck"
    assert rows[-1]["session_id"] == "sid-profile"
    assert rows[-1]["last_event"] == "approval.request"
    assert rows[-1]["ledger"] == str(ledger)
