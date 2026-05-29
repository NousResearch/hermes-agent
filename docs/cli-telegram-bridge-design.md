# CLI ↔ Telegram Bridge Safety Design

Status: phase-3 safe plain-DM continuation implemented. This document describes the target design and the safety invariants enforced by `gateway.bridge.BridgeStateStore`.

## Goal

Make a local Hermes CLI/TUI session and a Telegram DM feel like one controlled workspace, without treating Telegram as trusted local stdin.

The bridge must fail closed. If state is missing, stale, ambiguous, replayed, mismatched, or unsafe, the correct behavior is to reject or pause rather than retry or guess.

## External references checked

- Telegram Bot API: `message_id`, `update_id`, `reply_parameters`, `getUpdates` offset handling.
- `chiendo97/tele-claude`: reply-to-message routing into tmux panes. Useful UX pattern; unsafe if reply text parsing is the only correlation.
- `jsayubi/ccgram`: Telegram approval/session bridge for Claude Code. Useful separation of permission prompts and general text; avoid auto-approval patterns.
- `liuxsh9/TerminalBot`: Telegram terminal control with active connection state, message edits, and confirmation buttons.
- `Pigbibi/TelegramCodexBot`: Telegram topic ↔ tmux/Codex session mapping, message-id tracking, and permission UI.
- Remote shell bots such as `codefeathers/tsh`, `fnzv/trsh-go`, and `Al-Muhandis/ShellRemoteBot`: useful as warning examples; broad remote shell/file access is intentionally out of scope for the Hermes bridge MVP.

## Existing Hermes constraints

- Gateway session identity is derived by `gateway.session.build_session_key()`, not by CLI session id.
- Telegram inbound messages already become `MessageEvent` objects, with `message_id`, `platform_update_id`, `reply_to_message_id`, `reply_to_text`, chat/thread/user source metadata, and batching.
- Gateway has separate handling for `/approve`, `/deny`, `/queue`, `/steer`, `/background`, clarify prompts, slash confirmations, and active-agent interrupt/queue policy.
- CLI sessions use local `session_id` and internal approval/interrupt state. Direct Telegram-to-CLI stdin injection risks bypassing Telegram auth and Hermes approval state.

Therefore the bridge must be a linked-session layer, not a blind merge of CLI and Telegram histories.

## MVP safety model

Allowed initially:

- Private Telegram DM only.
- Explicit allowlisted Telegram `user_id` and `chat_id`.
- One Telegram chat/thread bound to one Hermes session.
- One in-flight turn at a time.
- Plain DM text from an active binding is routed into the linked CLI-originated Hermes session through the normal gateway `MessageEvent` path.
- Slash commands keep normal gateway semantics; bridge routing does not make `/model`, `/tools`, `/reload`, or other control commands target the CLI session.
- Registered reply-to prompts only for future explicit input prompts: Telegram replies are accepted only when the replied-to bot message was explicitly marked as `input_expected`.
- Nonce-bound approval commands/buttons only.

Rejected initially:

- Groups/supergroups unless later enabled with chat, thread, and user allowlists.
- `GATEWAY_ALLOW_ALL_USERS` / `TELEGRAM_ALLOW_ALL_USERS` remote control.
- YOLO remote operation.
- Natural-language approval, e.g. “yes”, “좋아”, “승인”.
- Raw PTY/stdin injection.
- `/background`, `/resume`, `/continue`, `/sessions`, `/model`, `/tools`, `/reload`, quick shell commands, and config-changing commands over the bridge session (plain-text bridge routing skips slash commands).
- Edited/forwarded messages as control signals.
- Crash/restart automatic replay.

## Durable state required

`gateway.bridge.BridgeStateStore` currently implements the foundation:

- `processed_updates`: durable Telegram update dedupe.
- `binding_tokens`: local opt-in, single-use, expiring tokens minted from the local CLI/TUI side before Telegram can bind. Tokens may be pre-bound to expected Telegram chat/user/thread allowlist values.
- `bindings`: explicit bridge id ↔ Hermes session id ↔ Telegram chat/user/thread binding.
- `outbound_messages`: bot message registry for reply-to correlation, TTL, and one-time consumption.
- `approvals`: single-use approval nonce bound to bridge/session/tool call/tool argument hash.
- Pause/resume state per binding.
- Filesystem kill switch support.

## Control-plane rules

1. Telegram text is untrusted user input, never local stdin.
2. Bot output is not input. Outbound message IDs are tracked and only messages marked `input_expected` may be used as reply anchors.
3. Approval is not text. Approval requires a nonce that matches session, user, chat, tool call, and tool argument hash.
4. Any mismatch rejects:
   - wrong user
   - wrong chat/thread
   - wrong session
   - expired reply anchor
   - already consumed anchor/nonce
   - changed tool arguments
   - paused binding
   - active kill switch
5. Agent/tool execution is not retried automatically after ambiguous failure.

## Future integration points

Recommended order:

1. Keep `BridgeStateStore` as the deterministic safety layer.
2. Add a Telegram command such as `/bridge bind` that can only complete after local CLI/TUI opt-in creates a one-time binding token.
3. Add outbound “input prompt” messages to Telegram using `record_outbound_message(... input_expected=True ...)`.
4. On bound Telegram plain DM text, call `validate_telegram_direct_input()` and re-bind that gateway session key to the CLI session id before the normal agent path runs.
5. On Telegram reply to a future explicit input prompt, call `validate_reply_input()` before forwarding to any executor.
6. Add approval UI using `create_approval()` and `consume_approval()` before resolving Hermes tool approvals.
7. Only after these pass should an executor adapter be added:
   - preferred: structured TUI gateway/JSON-RPC or Gateway `MessageEvent` path
   - avoid: raw tmux/PTY keystroke injection

## Test coverage added

`tests/gateway/test_cli_telegram_bridge.py` verifies:

- durable update dedupe across store reopen
- local opt-in binding token is single-use, expiring, and can be limited to an expected Telegram chat/user
- reply input requires registered live outbound prompt, same Telegram user/chat/thread, and unexpired state
- normal mirrored output cannot be used as input
- stale reply anchors are rejected
- approval nonce is single-use and bound to session/user/chat/tool args
- paused binding and filesystem kill switch fail closed

`tests/gateway/test_cli_telegram_bridge_commands.py` verifies command integration:

- `/bridge` registry exposure.
- local token minting from CLI.
- Telegram DM `/bridge_bind`, `/bridge_status`, `/bridge_off`, `/bridge_pause`, and `/bridge_resume`.
- bound Telegram DM plain text switches the gateway session key to the linked CLI session id.
- slash commands are not bridge-routed and paused bindings reject plain text.

## Non-goals for phase 1

- Live CLI injection.
- Telegram group remote control.
- Remote shell/file manager.
- Automatic approval.
- Automatic recovery/replay after crash.
