# 2026-05-29 upstream sync hermes-agent

## 何をやったか

公式 `NousResearch/hermes-agent` の `upstream/main` を `1b1e30510` まで取り込んだで。こっちの fork は `0dd0f38d2` から始まって、先に独自の Ebbinghaus idle sleep 機能を `fe1a627fd` として退避してから、`py -3 scripts\sync_all.py --merge --target main --allow-preflight-blockers --skip-fetch` でマージした。

## 入った公式更新

- Docker/dashboard の `--insecure` 扱いを安全側へ寄せる修正
- dashboard OAuth / Nous Portal 認証まわり
- Krea 画像生成 plugin
- security-guidance plugin
- MCP の `npx` / `npm` / `node` 解決修正
- provider fallback、billing/entitlement、content-policy block の案内修正
- web/dashboard の stale auth reload 修正
- memory provider の completed-turn `messages` API

## 残した独自機能

- OpenCode Zen `auto-free` と OpenClaw key bridge
- llama.cpp local rollback と Windows autostart
- VRChat Quest2 / Neuro / VOICEVOX 周辺ツール
- gateway Windows 運用補強
- Ebbinghaus idle sleep cycle

## 手動判断

メモリ周辺は公式の completed-turn `messages` API と独自 sleep が同じ面を触ってた。`MemoryManager.sync_all()` は upstream の `messages` 引数対応を採用しつつ、独自の `configure_idle_sleep()` / `maybe_sleep_for_idle()` / `consume_wake_greeting()` を残した。`run_agent.py::_sync_external_memory_for_turn()` は upstream どおり `messages` を渡すので、独自 sleep は公式 API と同じ MemoryManager 上で共存する形や。

## 検証メモ

- `py -3 scripts\sync_all.py --dry-run`
- `py -3 scripts\sync_all.py --merge --target main --allow-preflight-blockers --skip-fetch`
- `git diff --check` は README / `_docs` の CRLF 警告だけで exit 0
- `tests\agent\test_memory_provider.py tests\plugins\test_ebbinghaus_plugin.py tests\run_agent\test_memory_provider_init.py tests\run_agent\test_memory_sync_interrupted.py`: 105 passed
- `tests\hermes_cli\test_config.py tests\scripts\test_sync_all.py`: provider discovery 以外は先に通過
- `tests\providers\test_plugin_discovery.py`: 古い `plugins\model-providers\ai-gateway\__pycache__` を消した後に 4 passed
