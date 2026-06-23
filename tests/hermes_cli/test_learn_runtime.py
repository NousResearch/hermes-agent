from __future__ import annotations

import json
import threading

from hermes_time import now as hermes_now


def test_background_runtime_collects_without_creating_suggestions(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import runtime, state

    state.start(mode="learn")
    stop_event = threading.Event()

    def sample_once(*, home=None):
        assert home is not None
        learn_dir = home / "learn"
        learn_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": hermes_now().isoformat(),
            "process_name": "code.exe",
            "window_title": "hermes-agent",
            "domain": None,
            "category": "development",
            "idle": False,
            "duration_seconds": 300,
        }
        (learn_dir / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for _ in range(3)), encoding="utf-8")
        state.stop()
        stop_event.set()

    monkeypatch.setattr(runtime.sampler, "sample_once", sample_once)

    runtime._worker(home, stop_event, 0.01)

    assert (home / "learn" / "events.jsonl").exists()
    assert not (home / "cron" / "suggestions.json").exists()
