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

## Two kinds of desktop

Always say which kind this conversation is using, in your first reply after turning one on. The user cannot tell from the transcript alone, and the whole point of the VM kind is that they know when work is happening somewhere that cannot reach their machine.

**`realm` (default).** A private labwc desktop in this login session. Starts in about a second, costs almost nothing, works everywhere. It separates GUIs — windows, pointer, clipboard, focus — so parallel agents do not fight over the host's desktop. Ordinary terminal commands still see the real project filesystem, which is usually what you want. Use it for everything that does not specifically need Omarchy.

**`omarchy-vm`.** A disposable Omarchy guest in QEMU, with its own kernel, disk and desktop. Boots in roughly fifteen seconds and holds a few GB of host RAM while it runs. Choose it with `action: on, kind: omarchy-vm` (or `/realm on omarchy`) when the task genuinely needs a real Omarchy:

- building, installing or testing an **omarchy-shell plugin** or bar widget — a labwc realm structurally cannot host `omarchy-shell`, which needs Hyprland and one shell per session;
- **Hyprland, theme or system-level changes** you would not want applied to the user's own desktop;
- anything you would otherwise be tempted to run against the user's live session “just to see if it works”.

Do not reach for the VM kind for ordinary coding, file work or web browsing. It is slower to start, costs real memory, and the project files are not in it.

**The guest does not have your project.** Its filesystem is the guest's own; host paths do not exist in there. Copy work in with `action: push` (`/realm push SOURCE [GUEST_PATH]`) and take results back out with `action: pull` (`/realm pull GUEST_PATH LOCAL_PATH`). Nothing is mounted, and nothing leaves the guest unless you pull it. The guest has no disk encryption and passwordless sudo: never put a credential, token or key in it.

The VM kind has no computer-use driver yet, so `app: screen` capture and clicking are unavailable in it; use `action: shot` for a screenshot of the guest's own desktop, and ordinary terminal commands plus the guest's `omarchy` CLI for everything else. Switching kinds stops the current desktop, so pick one and stay on it for the conversation.

If the VM kind reports setup is required, relay what it says — the user runs `hermes realms vm install` once per profile to build the shared base image (a signed ~5 GB Omarchy ISO, then an unattended install). Never install it silently and never fall back to the host because a VM was unavailable.

## Never escape to the host

A realm strips the host's display, bus and input handles from the environment. Putting them back — `env WAYLAND_DISPLAY=… hyprctl`, `export DISPLAY=:0`, pointing `XDG_RUNTIME_DIR` or `DBUS_SESSION_BUS_ADDRESS` at `/run/user/<uid>` — drives the user's real desktop while you report working privately. That has actually happened. The plugin now refuses those commands; do not work around the refusal by other means. If a task truly needs the host, say so and let the user decide, or ask them to run `/realm off`.

Session commands: `/realm on [omarchy]`, `/realm off`, `/realm status`, `/realm size WIDTHxHEIGHT`, `/realm stop`, `/realm watch`, `/realm shot`, `/realm push SOURCE [GUEST_PATH]`, `/realm pull GUEST_PATH LOCAL_PATH`.

In a realm, use desktop capture (`app: screen`) and the private terminal. Do not attach host applications, host a11y/portal buses, host Cua sockets or input devices. Never fall back to the host after startup, validation or permission errors. Resume only after repairing the realm or explicit user choice to use the host.

Only if the realm Cua `app: screen` capture fails, call `realm` with `action: shot` (or `/realm shot`). This captures the already-running **own-session** compositor through validated `Manager.shot`/grim; it never creates a realm or selects a host display. It returns the actual PNG `path`, `realm_id`, `mime_type`, `width`, `height`, `bytes`, `sha256`, `capture: grim`, and `fallback: true`. The PNG is mode 0600 in a unique mode-0700 directory under the active profile's `realms/`; use the returned path, not an invented filename. No caller-supplied realm ID or output path is accepted. If it fails ownership/startup/validation, stop and report the error—never use host screenshot tools, override permissions, or disable realm routing to obtain an image. This is not a permission-denial workaround.

Cursor visibility is separate from screenshot success. Pinned Cua 0.23.2 can render the `cua.default` layer-shell overlay in a fresh private daemon; enabling the option alone does not prove the current session has a visible overlay. If that overlay is unavailable, WayVNC's server-rendered native cursor is the supported viewer fallback for **actual private pointer movement**, not logical/synthetic-only cursor moves. Grim captures need not include either cursor. Never claim a themed overlay is active from configuration alone or move the host pointer to demonstrate it.

Realm routing is GUI separation, **not a hostile-code sandbox**. Ordinary terminal commands retain local project access. Never claim environment cleanup prevents arbitrary same-user host socket/file access. Do not change permission mode or approval settings to make an action work.

Mode changes are tool/session state only. Never rewrite the system prompt, replay history or alter the tool schema mid-conversation.
