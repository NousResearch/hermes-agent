---
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
product_contract_source: ce-plan-bootstrap
title: Ship Telegram Media Caption MarkdownV2 Fix
created: 2026-07-02
---

# Ship Telegram Media Caption MarkdownV2 Fix

## Goal Capsule

Open a current-main pull request that fixes Telegram media captions rendering raw Markdown in TTS/audio replies and related media sends. The PR should supersede stale caption-fix PRs that target the old `gateway/platforms/telegram.py` path, link the existing issue, pass CI, and leave a concise Discord developer-channel message for maintainer review.

## Product Contract

- R1: Telegram media captions must render standard assistant Markdown through Telegram MarkdownV2 instead of displaying raw `**bold**`, headings, code markers, or similar syntax.
- R2: The fix must apply to voice/audio captions, since issue `#32029` is triggered by Telegram voice-note auto-TTS replies using a short response as the audio caption.
- R3: Sibling Telegram media caption paths must not be left behind: photos, documents, videos, animations, and media-group items should use the same caption formatting boundary.
- R4: If Telegram rejects MarkdownV2 caption parsing, media delivery must still proceed through a plain-caption retry rather than dropping or demoting the media.
- R5: The PR must cite existing issue `#32029` and explain relationship to stale PRs `#32118` and `#32893`, which target removed path `gateway/platforms/telegram.py`.
- R6: The current working VPS install at `/root/.hermes/hermes-agent` must not be changed; the work lands through the repo/PR path only.

## Scope Boundary

In scope:
- `plugins/platforms/telegram/adapter.py`
- `tests/gateway/test_telegram_documents.py`
- `tests/gateway/test_send_multiple_images.py`
- GitHub issue/PR comments that link the new PR to `#32029`, `#32118`, and `#32893`
- PR creation, CI monitoring, and CI fixes if required

Out of scope:
- Live install patching or Hermes gateway restart on the VPS
- New Telegram rich-message API support for captions
- Large refactors of `format_message()`, message streaming, or Telegram rich text
- Closing existing PRs or issues unless maintainers request it

## Key Technical Decisions

- KTD1: Format captions inside the Telegram adapter, not in `gateway/platforms/base.py`. The adapter owns MarkdownV2 conversion and Telegram API kwargs.
- KTD2: Use the existing `format_message()` path for caption text, then include `parse_mode=ParseMode.MARKDOWN_V2` on Telegram media API calls.
- KTD3: Reuse the existing media send retry boundary so parse-failure fallback preserves DM-topic reply-anchor behavior and media file seek/reset semantics.
- KTD4: Treat stale upstream PRs as prior art. The new PR should be respectful: it ports the fix class to the current plugin adapter path and broadens coverage.

## Implementation Units

### U1: Adapter Caption Formatting Boundary

Files:
- `plugins/platforms/telegram/adapter.py`

Requirements: R1, R2, R3, R4

Work:
- Add a helper that converts non-empty captions with `format_message()` and returns `caption` plus `parse_mode=ParseMode.MARKDOWN_V2`.
- Add a parse-error detector and plain-caption retry helper for Telegram caption parse failures.
- Apply the helper to `send_voice`, `send_image_file`, `send_document`, `send_video`, `send_image`, `send_animation`, and `send_multiple_images`.

Tests:
- `tests/gateway/test_telegram_documents.py` must prove MP3/audio captions use formatted MarkdownV2 kwargs.
- `tests/gateway/test_telegram_documents.py` must prove a caption parse failure retries the media send without parse mode.
- `tests/gateway/test_send_multiple_images.py` must remain compatible with `InputMediaPhoto` receiving `parse_mode`.

### U2: GitHub PR and Cross-Linking

Files:
- No source file changes expected beyond U1 and tests.

Requirements: R5, R6

Work:
- Commit only the relevant Telegram adapter/test changes.
- Push a branch to the user's fork.
- Open a PR against `NousResearch/hermes-agent:main` with `Fixes #32029` and `Related: #32118, #32893`.
- Comment on `#32029`, `#32118`, and `#32893` with the new PR link and the current-path rationale.

Verification:
- `gh pr view` shows an open PR for the branch.
- GitHub comments contain the new PR URL.
- `git status` shows no accidental live-install edits.

### U3: CI Watch and Handoff Message

Files:
- Source/test files only if CI exposes a real failure in this PR.

Requirements: R1, R2, R3, R4, R5

Work:
- Watch PR checks until success or a durable unresolved failure state.
- If a check fails, inspect failed logs, fix the root cause, commit, push, and watch again.
- After checks pass, provide a Discord message for Teknium/developer-channel review.

Verification:
- `gh pr checks --watch` exits successfully, or unresolved CI failures are recorded in the PR body after the allowed fix loop.
- Discord message includes issue, PR, stale PR context, tests, and reason for review.

## Verification Contract

Before opening the PR:
- Run `pytest tests/gateway/test_telegram_documents.py tests/gateway/test_send_image_file.py tests/gateway/test_send_multiple_images.py -q`.
- Run `pytest tests/gateway/test_telegram_thread_fallback.py -q`.
- Run `git diff --check`.

After opening the PR:
- Run `gh pr checks --watch`.
- If CI fails, inspect logs with `gh run view ... --log-failed`, fix the root cause, push, and repeat up to the LFG loop limit.

## Definition of Done

- Current-main PR is open and linked to `#32029`.
- Existing stale PRs `#32118` and `#32893` are commented with a respectful link to the current-path PR.
- CI is green, or any unresolved CI failures are durably recorded on the PR after attempted fixes.
- Final response includes the PR URL and a copy/paste Discord message for Teknium/developer review.
