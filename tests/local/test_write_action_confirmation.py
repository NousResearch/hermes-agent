from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(relative: str, name: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_x_post = _load_module(
    "skills/productivity/x-poster/scripts/x_post.py",
    "local_secretary_x_post",
)
_irodori = _load_module(
    "skills/audio/irodori-tts/scripts/irodori_tts.py",
    "local_secretary_irodori_tts",
)
x_draft_post = _x_post.x_draft_post
x_publish_post = _x_post.x_publish_post
synthesize_speech = _irodori.synthesize_speech

from agent.local_secretary.google_workspace_actions import (
    calendar_create,
    calendar_delete,
    calendar_list,
    calendar_update,
    gmail_search,
    gmail_send,
)
from agent.local_secretary.write_action_gate import WriteActionError, require_write_confirmation


def _loads(payload: str) -> dict:
    return json.loads(payload)


def test_x_publish_blocked_without_confirmation():
    result = _loads(x_publish_post("hello world"))
    assert result["success"] is False
    assert result["confirmation_required"] is True


def test_x_draft_allowed_without_confirmation():
    result = _loads(x_draft_post("draft only"))
    assert result["success"] is True


def test_x_publish_ok_with_confirmation_or_dry_run():
    confirmed = _loads(x_publish_post("hello", confirmed=True, dry_run=True))
    assert confirmed["success"] is True
    assert confirmed["dry_run"] is True


def test_gmail_send_blocked_calendar_list_allowed():
    blocked = _loads(gmail_send("a@example.com", "hi", "body"))
    assert blocked["success"] is False

    allowed = _loads(calendar_list(range_hint="today"))
    assert allowed["success"] is True


def test_calendar_mutations_require_confirmation():
    create_payload = _loads(
        calendar_create(
            summary="Meet",
            start="2026-06-14T10:00:00+09:00",
            end="2026-06-14T11:00:00+09:00",
        )
    )
    assert create_payload["success"] is False

    update_payload = _loads(calendar_update(event_id="evt1", summary="Moved"))
    assert update_payload["success"] is False

    delete_payload = _loads(calendar_delete(event_id="evt1"))
    assert delete_payload["success"] is False


def test_gmail_search_and_tts_without_confirmation(tmp_path, monkeypatch):
    search = _loads(gmail_search("from:me newer_than:7d"))
    assert search["success"] is True

    monkeypatch.setenv("IRODORI_TTS_OUTPUT_DIR", str(tmp_path))
    out = tmp_path / "brief.wav"
    tts = _loads(
        synthesize_speech(
            "Secretary briefing.",
            output_path=out,
            dry_run=True,
        )
    )
    assert tts["success"] is True
    assert out.exists()


def test_require_write_confirmation_raises_for_shell():
    try:
        require_write_confirmation("shell_exec", confirmed=False)
        assert False, "expected WriteActionError"
    except WriteActionError:
        pass


def test_example_config_enforces_write_confirmation_flag():
    repo_root = Path(__file__).resolve().parents[2]
    example = repo_root / "config" / "local-secretary.example.yaml"
    data = yaml.safe_load(example.read_text(encoding="utf-8"))
    assert data["local_secretary"]["write_actions_require_confirmation"] is True
    assert data["model"]["provider"] == "custom"
