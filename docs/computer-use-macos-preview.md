# macOS Computer-Use Preview for Hermes

This branch is the first public proof that Hermes is crossing the line from “agent with browser tools” into “agent with real local desktop control” on macOS.

It is not finished yet.

But it is already beyond a mock cursor demo.

## What is already real

### 1. Telegram-approvable app access

The `hermes-computer-use` MCP path now supports approval from messaging surfaces instead of only local/manual allowlisting.

Current approval semantics:
- `Allow Once`
- `Session`
- `Always`
- `Deny`

This is especially important for Hermes because desktop control should feel like a governed capability, not a hidden always-on superpower.

### 2. Window-scoped app state

`get_app_state(app_name=...)` now returns a focused app/window view instead of relying only on whole-desktop capture.

Grounded outputs already include:
- app name
- bundle id / bundle path
- process id
- window id
- window bounds
- window title
- screenshot path
- approval-gated accessibility tree

### 3. Safe unlock path from lock screen to desktop

With explicit user permission, Hermes can move from `loginwindow` into the desktop by:
- detecting the lock state
- waking the password UI when only wallpaper is visible
- injecting the password locally
- keeping the password out of tool arguments and transcript text
- verifying success from fresh desktop state

Artifacts:
- [lockscreen-password-ui.png](media/computer-use/lockscreen-password-ui.png)
- [unlocked-terminal-after-input.png](media/computer-use/unlocked-terminal-after-input.png)

### 4. Real local click proof

A grounded local smoke test on TextEdit succeeded.

What was verified:
- the adapter captured the AX close button frame
- the click targeted the real center coordinates of that button
- the local click backend returned `success=true`
- after the click, the target TextEdit window was gone and another TextEdit document became frontmost

Artifacts:
- [textedit-window-state.png](media/computer-use/textedit-window-state.png)
- [textedit-click-overlay.png](media/computer-use/textedit-click-overlay.png)

## What is still in progress

The branch still needs runtime cleanup for the live MCP route.

During testing, we observed a split between:
- the fresh worktree code path, which already proved real local click success
- the live chat/MCP route, which in one run still reported `preview_only`

Grounded inspection showed duplicate `computer_use_mcp_server.py` processes at the same time.

So the remaining work is not “invent clicking from scratch.”
It is “make the live runtime consistently use the newest server/session path and remove stale process confusion.”

## How to describe this branch honestly

Strong but accurate description:
- Hermes now has a real macOS computer-use stack taking shape
- app approval is already wired into Telegram
- window-scoped state is already working
- local unlock is already proven
- local click is already proven
- live runtime cleanup is the remaining short-term polish

In other words:

> Official-computer-use-inspired UX, Hermes-native approval flow, and real local desktop control — already partially working, with final runtime cleanup still underway.

## Suggested GitHub positioning

Good headline directions:
- “macOS computer-use for Hermes”
- “Telegram-approved desktop control for Hermes”
- “Open-source computer-use stack for Hermes on macOS”
- “Codex-style computer-use for Hermes, with real local click receipts”

Good proof-first structure:
1. one bold headline
2. one hero screenshot / animation
3. four receipts
   - approval
   - window state
   - unlock
   - click
4. one short “current gaps” section

That sequence attracts attention without promising magic we have not shipped yet.

## Screenshot gallery

### Lock screen reached password UI

![Lock screen password UI](media/computer-use/lockscreen-password-ui.png)

### Desktop unlocked and Terminal frontmost

![Unlocked desktop Terminal](media/computer-use/unlocked-terminal-after-input.png)

### TextEdit app/window state

![TextEdit window state](media/computer-use/textedit-window-state.png)

### Detached cursor / click overlay preview

![TextEdit click overlay](media/computer-use/textedit-click-overlay.png)
