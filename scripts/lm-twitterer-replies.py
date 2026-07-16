# Auto-generated/maintained by Hermes. Replies only through lm-twitterer with whitelist/follower gates.
# Provider/model は cron 実行時に config.yaml のメイン設定から動的に読み取る。
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = os.environ.get("HERMES_PYTHON") or r"C:\Users\downl\Documents\New project\hermes-agent\.venv\Scripts\python.exe"
REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT") or r"C:\Users\downl\Documents\New project\hermes-agent")
CORE_PATH = REPO_ROOT / "plugins" / "lm-twitterer" / "core.py"
COUNT = int(os.environ.get("LM_TWITTERER_REPLY_COUNT", "20"))

# --- 動的プロバイダ/モデル解決 ---
def _resolve_provider_model() -> tuple[str, str]:
    """config.yaml の model.provider / model.default を読む。
    環境変数 LM_TWITTERER_PROVIDER / LM_TWITTERER_MODEL が設定されていれば優先。"""
    env_provider = os.environ.get("LM_TWITTERER_PROVIDER")
    env_model = os.environ.get("LM_TWITTERER_MODEL")
    if env_provider and env_model:
        return (env_provider, env_model)

    try:
        import yaml
        hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
        config_path = hermes_home / "config.yaml"
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            provider = env_provider or raw.get("model", {}).get("provider", "") or ""
            model = env_model or raw.get("model", {}).get("default", "") or ""
            if provider and model:
                return (provider, model)
    except Exception:
        pass

    # 最終フォールバック: fallback_providers から local llama-server を探す
    try:
        import yaml
        hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
        config_path = hermes_home / "config.yaml"
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            for fb in raw.get("fallback_providers", []):
                if fb.get("api_key") == "local" and fb.get("provider"):
                    return (fb["provider"], fb.get("model", ""))
    except Exception:
        pass
    return ("custom", "huihui-qwythos-9b-mythos-5-q8_0")

PROVIDER, MODEL = _resolve_provider_model()


def _load_core():
    spec = importlib.util.spec_from_file_location("lm_twitterer_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load lm-twitterer core from {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lm_twitterer_core"] = module
    spec.loader.exec_module(module)
    return module


def _run_hermes_reply(*, live: bool) -> subprocess.CompletedProcess[str]:
    prompt = (
        "You are running inside a no-agent cron wrapper. "
        "Use exactly one tool call: lm_twitterer_reply_mentions with "
        f"dry_run={str(not live).lower()}, count={COUNT}, "
        "mark_seen_on_dry_run=false, "
        f"provider='{PROVIDER}', model='{MODEL}'. "
        "Then summarize only safe counts/status; do not expose cookies, tokens, env vars, or raw private data."
    )
    return subprocess.run(
        [
            PYTHON,
            "-m",
            "hermes_cli.main",
            "chat",
            "-Q",
            "-t",
            "lm_twitterer",
            "--provider",
            PROVIDER,
            "-m",
            MODEL,
            "-q",
            prompt,
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )


def main() -> int:
    core = _load_core()
    auth = core.auth_check()
    if not auth.get("ok") or not auth.get("auth_valid"):
        print(json.dumps({"ok": False, "stage": "auth-check", "auth_valid": auth.get("auth_valid"), "error": auth.get("error", "auth check failed")}, ensure_ascii=False))
        return 2
    if os.environ.get("LM_TWITTERER_CRON_PREFLIGHT_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        status = core.status()
        print(json.dumps({"ok": True, "stage": "preflight", "screen_name": auth.get("screen_name"), "whitelist_count": status.get("whitelist_count")}, ensure_ascii=False))
        return 0

    result = _run_hermes_reply(live=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.strip())
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
