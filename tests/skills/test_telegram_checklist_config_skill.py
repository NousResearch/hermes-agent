"""Profile-bound offline CLI contracts; no Telegram or credentials needed."""
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[2] / "optional-skills" /
          "communication" / "telegram-checklist" / "scripts" / "telethon_checklist.py")


def test_explicit_null_config_never_falls_back_to_legacy_grant(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "skills:\n  config:\n    telegram_checklist:\n      chats: null\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TELETHON_CHECKLIST_CHATS=-100999\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "create", "--title", "Tasks", "--task", "A",
         "--chat", "-100999", "--dry-run"],
        env={"PATH": os.defpath, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path)},
        capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert json.loads(result.stdout)["ok"] is False


def test_profile_config_controls_offline_target(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text(
        "skills:\n  config:\n    telegram_checklist:\n"
        "      chats: '-100123:33'\n", encoding="utf-8")
    # Legacy grants in .env must not override the selected profile's config.
    (profile / ".env").write_text("TELETHON_CHECKLIST_CHATS=-100999\n", encoding="utf-8")
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "HERMES_HOME": str(profile)}
    args = [sys.executable, str(SCRIPT), "create", "--title", "Tasks",
            "--task", "Review proposal", "--chat", "-100123", "--thread", "33", "--dry-run"]
    result = subprocess.run(args, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["dry_run"] is True
    assert data["would_send"]["chat"] == -100123
    assert not (profile / "telethon").exists()
    denied = subprocess.run(args[:args.index("-100123")] + ["-100999", "--dry-run"],
                            env=env, capture_output=True, text=True, check=False)
    assert denied.returncode == 1
    assert json.loads(denied.stdout)["ok"] is False
