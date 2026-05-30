# CLI ↔ Telegram Bridge Safety Design

Status: phase-8 bridge reply prompt routing and bridge-routed Telegram update dedupe implemented on top of the safe plain-DM continuation controls. This document describes the target design and the safety invariants enforced by `gateway.bridge.BridgeStateStore`, the higher-level helpers in `gateway.bridge_commands`, and the Gateway approval queue handoff in `tools.approval`.

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

- `processed_updates`: durable Telegram update dedupe. Bridge-routed Telegram input records accepted update IDs before switching into the linked Hermes session, so duplicate platform deliveries fail closed instead of creating a second agent turn.
- `binding_tokens`: local opt-in, single-use, expiring tokens minted from the local CLI/TUI side before Telegram can bind. Tokens may be pre-bound to expected Telegram chat/user/thread allowlist values.
- `bindings`: explicit bridge id ↔ Hermes session id ↔ Telegram chat/user/thread binding.
- `outbound_messages`: bot message registry for reply-to correlation, TTL, and one-time consumption.
- `approvals`: single-use approval nonce bound to bridge/session/tool call/tool argument hash. Telegram `/bridge_approve <nonce>` records the user approval and, when it is running in the same Gateway session as a pending dangerous-command approval, attempts to resume the blocked executor entry. The executor handoff still calls `consume_approval()` with the same turn id, tool call id, and canonical tool argument hash before execution.
- Pause/resume state per binding.
- Filesystem kill switch support.

## Implemented user flow

Local CLI/TUI side:

```text
/bridge bind telegram [--chat CHAT_ID] [--user USER_ID] [--ttl SECONDS]
/bridge status
/bridge disconnect
/bridge off
/bridge on
```

Telegram DM side:

```text
/bridge_bind <token>
/bridge_status
/bridge_pause
/bridge_resume
/bridge_disconnect
/bridge_off
/bridge_approve <nonce>
```

`/bridge_status` and `/bridge status` include the bridge id, Hermes session id, binding state, Telegram chat/user/thread, and last update timestamp. `/bridge_disconnect` removes only the current Telegram DM binding. Local `/bridge disconnect` removes bindings for the current Hermes session. Both delete dependent reply/approval bridge state for the removed binding so stale prompts cannot be reused after disconnect.

## Control-plane rules

1. Telegram text is untrusted user input, never local stdin.
2. Bot output is not input. Outbound message IDs are tracked and only messages marked `input_expected` may be used as reply anchors.
3. Approval is not text. Approval requires a nonce that matches session, user, chat, tool call, and tool argument hash. The Telegram command first records the bound user's approval, then the executor adapter consumes that same nonce only when the pending Gateway approval entry presents the exact turn id, tool call id, Hermes session id, and canonical argument hash. Generic `/approve` approvals do not resolve bridge-protected entries; `/deny` remains available to reject them.
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
2. Add a Telegram command such as `/bridge bind` that can only complete after local CLI/TUI opt-in creates a one-time binding token. (Implemented as local `/bridge bind telegram` + Telegram `/bridge_bind <token>`.)
3. On bound Telegram plain DM text, call `validate_telegram_direct_input()` and re-bind that gateway session key to the CLI session id before the normal agent path runs. (Implemented.)
4. Add bridge UX controls for status, pause/resume, emergency off/on, and disconnect. (Implemented.)
5. Outbound “input prompt” messages use `register_bridge_reply_input_prompt()`, which validates the active Telegram DM binding and records `record_outbound_message(... input_expected=True ...)` for one-shot reply correlation. (Implemented.)
6. Telegram replies to explicit input prompts call `validate_reply_input()` before routing into the linked Hermes session. The bridge state consumes the reply anchor on success, rejects stale/consumed/paused/kill-switched prompt replies without falling back to generic plain-text bridge input, and leaves ordinary unregistered Telegram replies on the direct-input path. (Implemented.)
7. Bridge-routed Telegram input calls `accept_update()` before switching the gateway session key, so duplicate Telegram deliveries for the same `update_id` cannot start a second agent turn. (Implemented.)
8. Approval UI uses `create_bridge_approval_prompt()` and `bridge_tool_args_hash()` to create a `create_approval()` nonce bound to the bridge, session, tool call id, tool name, and canonical tool args hash. Telegram `/bridge_approve <nonce>` records the bound user's approval and, in Gateway runs, resolves the matching pending approval entry only after `consume_approval(..., turn_id=..., tool_call_id=..., tool_args_hash=..., require_user_approval=True)` accepts the exact executor-side metadata. Mismatches fail closed and leave the blocked approval entry waiting. (Implemented.)
9. Future executor integrations should continue to prefer structured TUI gateway/JSON-RPC or Gateway `MessageEvent` paths and avoid raw tmux/PTY keystroke injection.

## Test coverage added

`tests/gateway/test_cli_telegram_bridge.py` verifies:

- durable update dedupe across store reopen
- local opt-in binding token is single-use, expiring, and can be limited to an expected Telegram chat/user
- reply input requires registered live outbound prompt, same Telegram user/chat/thread, and unexpired state
- normal mirrored output cannot be used as input
- stale reply anchors are rejected
- approval nonce is single-use and bound to session/user/chat/turn/tool call/tool args
- paused binding and filesystem kill switch fail closed

`tests/gateway/test_cli_telegram_bridge_commands.py` verifies command integration:

- `/bridge` registry exposure.
- local token minting from CLI.
- Telegram DM `/bridge_bind`, `/bridge_status`, `/bridge_off`, `/bridge_pause`, `/bridge_resume`, and `/bridge_disconnect`.
- detailed local and Telegram bridge status output.
- local and Telegram disconnect behavior scoped to the relevant binding/session.
- bound Telegram DM plain text switches the gateway session key to the linked CLI session id.
- duplicate Telegram update IDs for bridge-routed input are rejected without a second gateway session switch.
- bound Telegram replies to registered input prompts consume the one-shot reply anchor and switch to the linked CLI session; expired registered prompt replies reject without falling back to direct-input routing.
- slash commands are not bridge-routed and paused bindings reject plain text.
- `register_bridge_reply_input_prompt()` records only bound Telegram DM output as one-shot input anchors.
- `create_bridge_approval_prompt()` creates nonce-bound approvals tied to canonical tool argument hashes.
- `/bridge_approve <nonce>` records only the bound Telegram user's approval and resolves a matching pending Gateway executor approval only after exact session/turn/tool-call/argument verification.

`tests/tools/test_approval.py::TestBridgeApprovalContext` verifies that Gateway dangerous-command approval requests include the active Hermes tool-call context (`turn_id`, `tool_call_id`, `tool_name`, and `tool_args`) needed to mint and verify bridge approval nonces.

## Non-goals for phase 1

- Live CLI injection.
- Telegram group remote control.
- Remote shell/file manager.
- Automatic approval.
- Automatic recovery/replay after crash.
