# Devorch handoff — 2026-05-12

## Scope

This branch packages the current local Hermes Agent changes and the verification pass I completed before upload.

## What changed

### 1. Credential pool least-used routing now persists selection counts
- `agent/credential_pool.py`
- `tests/agent/test_credential_pool.py`

Problem:
- `least_used` only incremented `request_count` in memory.
- After process/profile restart, pools reloaded from `auth.json` with stale counters and repeatedly preferred the first eligible credential.

Change:
- persist the incremented `request_count` immediately after selection.
- added a regression test that verifies the updated count is written back to the pool on disk.

### 2. Single-media Telegram deliveries now use native captions instead of split text + media
- `cron/scheduler.py`
- `gateway/platforms/base.py`
- `gateway/platforms/telegram.py`
- `tools/send_message_tool.py`
- `tests/cron/test_scheduler.py`
- `tests/gateway/test_send_multiple_images.py`
- `tests/tools/test_send_message_tool.py`

Problem:
- when content looked like `text + MEDIA:/path.png`, Telegram-style delivery often produced a detached text message followed by a captionless image/video.
- cron live-adapter delivery and `send_message` had inconsistent behavior.

Change:
- for a single image/video attachment with remaining text, use the text as the native caption instead of sending a separate text message.
- pass captions through cron live-adapter media routing.
- make Telegram `send_multiple_images()` route a single image through the single-image send path so caption semantics stay correct.
- added regression tests for cron caption routing, `send_message` caption routing, and Telegram single-image batching behavior.

### 3. Telegram thread allowlist gate only applies to forum-style supergroups
- `gateway/platforms/telegram.py`
- `tests/gateway/test_telegram_group_gating.py`

Problem:
- the new `allowed_threads` gate could bleed into ordinary group-message tests and behavior if ambient `TELEGRAM_*` env vars were present.
- plain groups do not have forum-topic semantics, so thread allowlisting should not block them.

Change:
- apply `allowed_threads` gating only for `supergroup` chats.
- hardened tests so they do not inherit unrelated `TELEGRAM_*` environment from the current shell/profile.
- added explicit coverage that ordinary groups ignore `allowed_threads` while forum/supergroup threads obey it.

### 4. CLI/provider cleanup included in current local diff
- `hermes_cli/config.py`
- `hermes_cli/models.py`
- `hermes_cli/status.py`
- `plugins/model-providers/xiaomi/__init__.py`
- `tests/hermes_cli/test_status.py`
- `tests/hermes_cli/test_models.py`
- `tests/hermes_cli/test_config.py`
- `tests/hermes_cli/test_xiaomi_provider.py`

Change set present in local diff:
- remove Anthropic key/provider display from CLI config/status surfaces in these codepaths.
- remove Anthropic from the canonical provider picker list in `hermes_cli/models.py`.
- update Xiaomi provider registration to accept `XIAOMI_BASE_URL` and default to `https://token-plan-sgp.xiaomimimo.com/v1`.

## Verification I ran

From repo root with project venv activated:

```bash
pytest -q \
  tests/agent/test_credential_pool.py \
  tests/cron/test_scheduler.py \
  tests/tools/test_send_message_tool.py \
  tests/gateway/test_telegram_caption_merge.py \
  tests/gateway/test_send_multiple_images.py \
  tests/gateway/test_send_image_file.py \
  tests/gateway/test_telegram_group_gating.py \
  tests/hermes_cli/test_status.py \
  tests/hermes_cli/test_models.py \
  tests/hermes_cli/test_config.py \
  tests/hermes_cli/test_xiaomi_provider.py
```

Result:
- `525 passed`
- only existing dependency deprecation warnings in test output

## Known notes / risks

- `graphify update .` was started after code edits per repo rule, but the AST refresh did not complete within the interactive timeout window.
- `graphify-out/` remains untracked and is not part of the commit unless explicitly staged later.
- this handoff reflects the exact local diff present at upload time; no unrelated tracked files are added beyond the current modified set and this handoff note.

## Changed files in this upload

- `agent/credential_pool.py`
- `cron/scheduler.py`
- `gateway/platforms/base.py`
- `gateway/platforms/telegram.py`
- `hermes_cli/config.py`
- `hermes_cli/models.py`
- `hermes_cli/status.py`
- `plugins/model-providers/xiaomi/__init__.py`
- `tests/agent/test_credential_pool.py`
- `tests/cron/test_scheduler.py`
- `tests/gateway/test_send_multiple_images.py`
- `tests/gateway/test_telegram_group_gating.py`
- `tests/tools/test_send_message_tool.py`
- `tools/send_message_tool.py`
- `docs/handoffs/2026-05-12-devorch-handoff.md`
