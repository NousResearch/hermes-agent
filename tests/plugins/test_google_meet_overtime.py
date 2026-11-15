"""End-to-end Playwright Chromium tests for Google Meet overtime extension in _drain_loop."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from pathlib import Path

from playwright.sync_api import sync_playwright
from plugins.google_meet import meet_bot


def _make_cfg(duration_s: float = -1.0) -> SimpleNamespace:
    return SimpleNamespace(
        duration_s=duration_s,
        lobby_timeout=30.0,
        rtc_audio=False,
        guest_name="Emir's Notetaker",
    )


def test_chromium_drain_loop_overtime_extension_and_expiry(tmp_path: Path, capsys) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Positive control: active speech in call across expired deadline -> extends by +900s and drains live caption in Chromium
        stop_flag_pos = {"stop": False}

        def on_drain_called() -> list[dict[str, str]]:
            stop_flag_pos["stop"] = True
            return [{"speaker": "Emir Saffar", "text": "Vi fortsätter mötet i övertid"}]

        page.expose_function("__pyDrainHook", on_drain_called)
        page.set_content(
            """<html><body><script>
            window.__hermesDrainSpeakers = () => [{speaker: "Emir Saffar", ts: Date.now() / 1000}];
            window.__hermesMeetDrain = () => window.__pyDrainHook();
            </script></body></html>"""
        )

        now = time.time()
        pos_dir = tmp_path / "pos"
        state_pos = meet_bot._BotState(pos_dir, "mtg_pos", "https://meet.google.com/swz-vmmb-ygr")
        state_pos.set(in_call=True, joined_at=now - 7805.0, last_caption_at=now - 5.0, mic_state="muted")
        cfg_pos = _make_cfg(duration_s=-1.0)
        rt = {"enabled": False, "session": None}

        meet_bot._drain_loop(page, cfg_pos, state_pos, rt, stop_flag_pos)
        out_pos = capsys.readouterr().out
        status_pos = json.loads((pos_dir / "status.json").read_text(encoding="utf-8"))
        transcript_pos = (pos_dir / "transcript.txt").read_text(encoding="utf-8")

        assert "overtime_extend new_deadline=+900s" in out_pos
        assert getattr(state_pos, "leave_reason", None) is None
        assert status_pos["leaveReason"] is None
        assert status_pos["transcriptLines"] == 1
        assert "Emir Saffar: Vi fortsätter mötet i övertid" in transcript_pos

        # 2. Negative control A: stale speech (>600s) and >4h elapsed -> exits immediately with duration_expired
        stop_flag_neg = {"stop": False}
        neg_dir = tmp_path / "neg"
        state_neg = meet_bot._BotState(neg_dir, "mtg_neg", "https://meet.google.com/swz-vmmb-ygr")
        state_neg.set(in_call=True, joined_at=now - 15000.0, last_caption_at=now - 1200.0, mic_state="muted")
        cfg_neg = _make_cfg(duration_s=-1.0)

        meet_bot._drain_loop(page, cfg_neg, state_neg, rt, stop_flag_neg)
        status_neg = json.loads((neg_dir / "status.json").read_text(encoding="utf-8"))
        assert getattr(state_neg, "leave_reason", None) == "duration_expired"
        assert status_neg["leaveReason"] == "duration_expired"
        assert status_neg["transcriptLines"] == 0

        # 3. Negative control B: 6h hard safety ceiling (>21600s) even with recent speech -> exits with duration_expired
        stop_flag_ceil = {"stop": False}
        ceil_dir = tmp_path / "ceil"
        state_ceil = meet_bot._BotState(ceil_dir, "mtg_ceil", "https://meet.google.com/swz-vmmb-ygr")
        state_ceil.set(in_call=True, joined_at=now - 21700.0, last_caption_at=now - 2.0, mic_state="muted")
        cfg_ceil = _make_cfg(duration_s=-1.0)

        meet_bot._drain_loop(page, cfg_ceil, state_ceil, rt, stop_flag_ceil)
        status_ceil = json.loads((ceil_dir / "status.json").read_text(encoding="utf-8"))
        assert getattr(state_ceil, "leave_reason", None) == "duration_expired"
        assert status_ceil["leaveReason"] == "duration_expired"

        browser.close()
