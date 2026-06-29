# 2026-06-29 upstream security sync + CVE-class hardening (main)

## 概要

`NousResearch/hermes-agent` の `upstream/main` を正本
`C:\Users\downl\Documents\New project\hermes-agent` の `main` へ追随し、fork
独自機能を維持したまま公式機能、脆弱性更新、バグ修正を取り込んだ。

作業終盤で公式 upstream が追加で進んだため、最終的に
`upstream/main@f8604928422ff3461d4004ae6f71fdfdea9923a2` まで取り込んだ。
SOPログ追加前の同期済み実装HEADは
`b59904c48d522da224370456902e7f29e5d9e79f`。

## 正本

| 項目 | 内容 |
|------|------|
| 正本ルート | `C:\Users\downl\Documents\New project\hermes-agent` |
| fork remote | `origin` = `zapabob/hermes-agent` |
| official remote | `upstream` = `NousResearch/hermes-agent` |
| 作業補助worktree | `C:\Users\downl\hwt\hermes-upstream-security-sync-20260629` |
| 最終同期状態 | `git rev-list --left-right --count HEAD...upstream/main` = `458 0` |

## 実施内容

- `scripts/sync_all.py` と merge policy を使い、公式 upstream を fork main に取り込んだ。
- fork独自の公式API追随分類として、research paper skill を公式優先リストに追加した。
- `uv.lock` / npm lockfile / Desktop Windows起動まわりの公式差分を採用した。
- upstream追随後のCI落ちを修正した。
  - execute-code guard の回帰を修正。
  - Desktop backend readiness probe を遅い初回起動に耐えるよう45秒へ拡張。
  - Windows subprocess flag テストを公式の multiline spawn 変更に追随。
- Dependabot alert `GHSA-8988-4f7v-96qf` 対応として、Photon sidecar の
  OpenTelemetry override と lockfile を更新した。

## CVEになりえる未登録問題 3件

既存issue/PRに同等修正がないことを確認し、ローカルで修正した。

| 領域 | 問題 | 修正 |
|------|------|------|
| `plugins/hermes_gpt/server.py` | HTTP/SSE bridge を noauth で非loopback公開できる | 非loopback noauth は `--i-understand-this-is-unsafe` と `HERMES_GPT_UNSAFE_REMOTE_NOAUTH=1` の二重確認なしでは拒否 |
| `plugins/ai-partner-os/` | AITuber連携WS bridge を認証なしで公開できる | `--confirm-public-host` なしの非loopback公開を拒否 |
| `plugins/aituber-kit/` | linkage WS bridge を認証なしで公開できる | `--confirm-public-host` なしの非loopback公開を拒否 |

## 検証

ローカル:

- `uv run --extra dev python -m pytest ...` の高リスク対象: `170 passed, 46 skipped`
- `uv run --extra dev python -m pytest tests/test_hermes_state.py::TestTitleUniqueness tests/test_hermes_state.py::TestTitleLineage -q`: `19 passed`
- `node --check apps/desktop/electron/main.cjs`: passed
- `node --test apps/desktop/electron/backend-ready.test.cjs apps/desktop/electron/hardening.test.cjs apps/desktop/electron/windows-child-process.test.cjs`: `32 passed, 1 skipped`
- `uv run --extra dev python -m pytest tests/test_windows_subprocess_no_window_flags.py -q`: `15 passed`
- `uv run --extra dev python -m ruff check tests/test_windows_subprocess_no_window_flags.py`: passed
- `uv lock --check`: passed
- `npm --workspace apps/desktop run pack`: passed, build stamp `b59904c48d52 (main)`
- Photon sidecar `npm audit --omit=dev --package-lock-only`: `0 vulnerabilities`

GitHub Actions:

- CI run: `https://github.com/zapabob/hermes-agent/actions/runs/28355622144`
- head SHA: `b59904c48d522da224370456902e7f29e5d9e79f`
- result: `completed / success`
- required gate: `All required checks pass` = success

## Runtime note

このログをmainへ反映した後、ログ込みの最終HEADで Desktop を再パックし、
`HERMES_DASHBOARD_READY` の最新ポートと `/api/status` を再確認する。
ユーザー指定に従い、`llama-server` / `llama.cpp` 系プロセスは起動しない。

## 追記: 追加upstream追随

SOPログ初回コミット後、公式 `upstream/main` がさらに進んだため、以下も追加で
取り込んだ。

- `1289f1281 fix(memory): lazy-install supermemory + mem0 SDKs like honcho/hindsight`
- `0434a9a5e chore: regenerate uv.lock for supermemory + mem0 extras`

衝突は `tests/test_project_metadata.py` と `uv.lock`。前者はlazy-install対象extraの
契約リストを統合し、後者は公式lockfileを土台に現在の `pyproject.toml` から
`uv lock` で再生成した。

追加検証:

- `uv lock --check`: passed
- `uv run --extra dev python -m pytest tests/test_project_metadata.py tests/plugins/memory/test_memory_lazy_install.py tests/plugins/memory/test_supermemory_provider.py -q`: `65 passed, 1 skipped`
- `uv run --extra dev python -m ruff check tests/test_project_metadata.py tests/plugins/memory/test_memory_lazy_install.py tests/plugins/memory/test_supermemory_provider.py`: passed

## 残したもの

- 正本ルートの未追跡ローカル成果物（`.specstory/`, `_tmp/`, 音声/動画一時ファイル等）は既存作業物として触らない。
- 旧worktree側の既存未追跡ファイルは stash
  `codex-preserve-before-canonical-main-sync-20260629` で保護済み。

## 追記: Slack group DM upstream追随

最終確認中に公式 `upstream/main` がさらに3コミット進んだため、正本 `main` に
追加で取り込んだ。

- `29f096827 test(windows): harden pid-scan no-window assertion against captured-call leakage (#54707)`
- `4125cc3b7 fix(slack): subscribe to message.mpim + mpim scopes so group DMs work`
- `34e616e77 feat(slack): nudge stale installs to add mpim scopes; mark message.mpim required`

追加検証:

- `uv lock --check`: passed
- `uv run --extra dev python -m pytest tests/gateway/test_slack_group_dm_scope_warning.py tests/hermes_cli/test_slack_cli.py tests/test_windows_subprocess_no_window_flags.py -q`: `29 passed`
- `uv run --extra dev python -m ruff check tests/gateway/test_slack_group_dm_scope_warning.py tests/hermes_cli/test_slack_cli.py tests/test_windows_subprocess_no_window_flags.py hermes_cli/config.py hermes_cli/slack_cli.py plugins/platforms/slack/adapter.py`: passed
