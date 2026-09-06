---
name: realms
description: Use when choosing or managing a private desktop realm. Keep host access explicit and existing approvals intact.
---

# Conversation desktops

Use the `realm` tool to manage this conversation, not another session. Read `status` before reporting its mode or readiness.

- “Use a separate/private desktop”, “work without disturbing my desktop”, “turn the realm on”: call `realm` with `action: on`. This enables lazy start at the next terminal or computer-use action; it does not prove a desktop has started.
- “Use my actual desktop”, “work on the host”, “turn the realm off”: only when explicitly requested, call `realm` with `action: off`. Explain that subsequent terminal/computer actions target the host and all ordinary approvals still apply.
- An unspecified desktop uses the configured default (`realm`, `host`, or `ask`). In `ask` mode ask the user to choose; do not silently choose host.
- “Make the desktop 1280 by 720”: call `size` with `size: 1280x720`.
- “Show/watch the desktop”: call `watch`. Tickets expire; obtain a new URL rather than persist or log it. Viewing starts view-only. Human takeover inhibits agent input until returned.
- “Stop the private desktop”: call `stop`. This kills realm-owned processes and preserves the selected mode. Do not call it after every normal turn; conversation finalization handles cleanup.

Session commands: `/realm on`, `/realm off`, `/realm status`, `/realm size WIDTHxHEIGHT`, `/realm stop`, `/realm watch`, `/realm shot`.

In a realm, use desktop capture (`app: screen`) and the private terminal. Do not attach host applications, host a11y/portal buses, host Cua sockets or input devices. Never fall back to the host after startup, validation or permission errors. Resume only after repairing the realm or explicit user choice to use the host.

Only if the realm Cua `app: screen` capture fails, call `realm` with `action: shot` (or `/realm shot`). This captures the already-running **own-session** compositor through validated `Manager.shot`/grim; it never creates a realm or selects a host display. It returns the actual PNG `path`, `realm_id`, `mime_type`, `width`, `height`, `bytes`, `sha256`, `capture: grim`, and `fallback: true`. The PNG is mode 0600 in a unique mode-0700 directory under the active profile's `realms/`; use the returned path, not an invented filename. No caller-supplied realm ID or output path is accepted. If it fails ownership/startup/validation, stop and report the error—never use host screenshot tools, override permissions, or disable realm routing to obtain an image. This is not a permission-denial workaround.

Cursor visibility is separate from screenshot success. Pinned Cua 0.23.2 can render the `cua.default` layer-shell overlay in a fresh private daemon; enabling the option alone does not prove the current session has a visible overlay. If that overlay is unavailable, WayVNC's server-rendered native cursor is the supported viewer fallback for **actual private pointer movement**, not logical/synthetic-only cursor moves. Grim captures need not include either cursor. Never claim a themed overlay is active from configuration alone or move the host pointer to demonstrate it.

Realm routing is GUI separation, **not a hostile-code sandbox**. Ordinary terminal commands retain local project access. Never claim environment cleanup prevents arbitrary same-user host socket/file access. Do not change permission mode or approval settings to make an action work.

Mode changes are tool/session state only. Never rewrite the system prompt, replay history or alter the tool schema mid-conversation.
