# Auto-generated/maintained by Hermes. Posts a sanitized public topic only.
# Do not include personal information, environment variables, secrets, paths, or account details in public text.
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT") or r"C:\Users\downl\Documents\New project\hermes-agent")
CORE_PATH = REPO_ROOT / "plugins" / "lm-twitterer" / "core.py"

POST_TEXTS = [
    "Hermes Agent運用メモ: cronで調査・記憶同期・投稿を分離。秘密情報は外へ出さず、公開できる設計原則だけ残す。{stamp} #hermesagent",
    "はくあ運用ログ: 自動化は“便利”より先に“安全”。鍵・Cookie・環境変数を出さず、公開可能な結果だけ届ける。{stamp} #hermesagent",
    "Hermes Agent cron note: scheduled agents are useful when outputs are bounded, auditable, and secret-free. {stamp} #hermesagent",
]
FORBIDDEN = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", ".env", "C:\\Users\\", "/c/Users/")


def _load_core():
    spec = importlib.util.spec_from_file_location("lm_twitterer_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load lm-twitterer core from {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lm_twitterer_core"] = module
    spec.loader.exec_module(module)
    return module


def _validate_public_text(text: str) -> None:
    upper = text.upper()
    for marker in FORBIDDEN:
        if marker.upper() in upper:
            raise SystemExit(f"Refusing unsafe public text marker: {marker}")
    if "#hermesagent" not in text:
        raise SystemExit("Refusing post without #hermesagent")
    if len(text) > 240:
        raise SystemExit(f"Refusing overlong public text: {len(text)} chars")


def main() -> int:
    core = _load_core()
    auth = core.auth_check()
    if not auth.get("ok") or not auth.get("auth_valid"):
        print(json.dumps({"ok": False, "stage": "auth-check", "auth_valid": auth.get("auth_valid"), "error": auth.get("error", "auth check failed")}, ensure_ascii=False))
        return 2
    if os.environ.get("LM_TWITTERER_CRON_PREFLIGHT_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        status = core.status()
        print(json.dumps({"ok": True, "stage": "preflight", "screen_name": auth.get("screen_name"), "identity_name": status.get("identity_name")}, ensure_ascii=False))
        return 0

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M JST")
    text = random.choice(POST_TEXTS).format(stamp=stamp)
    _validate_public_text(text)
    result = core.post("", text=text, dry_run=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") and result.get("posted") else 3


if __name__ == "__main__":
    raise SystemExit(main())
