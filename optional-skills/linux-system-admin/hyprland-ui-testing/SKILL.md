---
name: hyprland-ui-testing
description: Test Hyprland UI without disturbing active workspaces.
version: 1.0.0
author: Kevin Rajan (kvnloo), Hermes Agent
license: MIT
platforms: [linux]
prerequisites:
  commands: [hyprctl]
metadata:
  hermes:
    tags: [hyprland, wayland, gtk, desktop, ui-testing]
    category: linux-system-admin
    related_skills: [computer-use]
    requires_toolsets: [terminal]
---

# Hyprland UI Testing Skill

Test graphical applications without mapping trial windows onto the user's active Hyprland workspace. This skill separates application-rendering evidence from compositor-integration evidence; neither lane substitutes for the other.

## When to Use

- A GTK or Wayland app must be launched or captured without interrupting the user.
- A HUD, overlay, layer-shell surface, window rule, or monitor placement needs verification.
- A UI under development needs a screenshot before it is safe to map normally.
- A previous test window stole focus, appeared on the wrong workspace, or followed the user.

Do not use this skill to automate ordinary desktop tasks. Use `computer-use` for background-first interaction with an existing application.

## Prerequisites

- Linux with an active Hyprland session.
- The Hermes `terminal` tool and `hyprctl`.
- For GTK rendering checks, the Broadway server matching the app's GTK major version.
- For visual Broadway evidence, the browser toolset with `browser_navigate` and `browser_vision`.

Broadway is optional when only compositor behavior is under test. A headless output is optional when only application rendering is under test.

## How to Run

Ask Hermes to use this skill before launching the test application:

```text
Use the hyprland-ui-testing skill to render this GTK app offscreen and show me the result.
```

For compositor behavior:

```text
Use the hyprland-ui-testing skill to verify placement and focus on a temporary Hyprland headless output.
```

Run every shell command through `terminal`. Start long-lived servers and applications with `terminal(background=True)` and retain their returned session IDs so `process(action="kill", session_id=...)` can stop them deterministically.

## Quick Reference

| Goal | Command or tool |
|---|---|
| Record workspace | `hyprctl -j activeworkspace` |
| Record focused window | `hyprctl -j activewindow` |
| List clients | `hyprctl -j clients` |
| List monitors | `hyprctl -j monitors` |
| Create headless output | `hyprctl output create headless HERMES_UI_TEST` |
| Remove headless output | `hyprctl output remove HERMES_UI_TEST` |
| Start GTK 3 Broadway | `broadwayd :5` |
| Start GTK 4 Broadway | `gtk4-broadwayd :5` |
| Open Broadway display 5 | `browser_navigate(url="http://127.0.0.1:8085")` |
| Capture Broadway render | `browser_vision()` |

Broadway display `:N` normally listens on TCP port `8080 + N`; verify the server output rather than assuming the port is free.

## Procedure

### 1. Freeze the user's active state

Use `terminal` to capture:

```bash
hyprctl -j activeworkspace
hyprctl -j activewindow
hyprctl -j monitors
hyprctl -j clients
```

Record the active workspace ID, monitor, focused client address, class, and title. Stop stale test processes and remove stale test outputs before continuing.

Never launch the app on the normal display and move it afterward. Launch-then-move exposes the first mapped frame and can steal focus.

### 2. Choose the evidence lane

Use Broadway for layout, controls, text, colors, dimensions, and screenshots. Use a Hyprland headless output for window rules, monitor/workspace placement, focus behavior, floating state, and workspace transitions.

If both claims matter, run both lanes separately. State which claim each result proves.

### 3. Render GTK offscreen with Broadway

Choose an unused display number. Start the matching Broadway server in the background, then launch only the test app against it:

```bash
broadwayd :5
GDK_BACKEND=broadway BROADWAY_DISPLAY=:5 ./path-to-app
```

Use `gtk4-broadwayd` instead of `broadwayd` for a GTK 4 app when the distribution packages the servers separately.

Open `http://127.0.0.1:8085` with `browser_navigate`, wait for the app to render, and capture it with `browser_vision`. Then run `hyprctl -j clients` and confirm that no matching client exists in Hyprland.

Broadway proves GTK rendering only. It does not exercise layer-shell, Wayland focus, Hyprland rules, monitor scaling, or bottom anchoring.

### 4. Verify compositor behavior on a headless output

Create a uniquely named output:

```bash
hyprctl output create headless HERMES_UI_TEST
hyprctl -j monitors
```

Inspect the loaded Hyprland configuration and parser before installing a runtime rule. Legacy configurations accept `hyprctl keyword`; non-legacy or Lua configurations may require `hyprctl eval` and their configuration API. Do not retry a rejected legacy command with guessed syntax.

Install the exact class/title placement rule before launching the app. The rule must assign the test client to the workspace on `HERMES_UI_TEST` without changing the user's focused monitor. Prove the rule is accepted, then launch the app once.

Keep the test window unpinned. Hyprland pinning intentionally makes a window visible across workspaces, so a pinned window cannot prove isolation. Test pin behavior only when the user explicitly authorizes a visible compositor test.

After map, inspect `hyprctl -j clients`. Record the client address, monitor, workspace, floating state, pinned state, and geometry. Confirm that the active workspace and focused client still match the baseline.

Layer-shell surfaces may not appear in `hyprctl clients`. If the target uses layer-shell, use layer-surface diagnostics supported by the installed Hyprland version and do not build address-based promotion around a client Hyprland does not expose.

### 5. Collect claim-matched evidence

- Rendering claim: fresh Broadway screenshot plus healthy app/server processes.
- Compositor claim: client or layer-surface state plus before/after workspace and focus.
- Isolation claim: no test client on the user's active monitor/workspace.
- Teardown claim: no matching process, client, output, or temporary rule remains.

Never call a Broadway screenshot proof of Hyprland placement. Never call a Hyprland client listing proof of visual correctness.

### 6. Clean up

Stop the app and Broadway server with `process(action="kill", session_id=...)`. Remove the headless output:

```bash
hyprctl output remove HERMES_UI_TEST
```

Remove temporary runtime rules if the active configuration API made them persistent. Re-read clients, monitors, active workspace, and focused window. Compare them with the baseline before reporting success.

## Pitfalls

- `pin` means global workspace visibility, not stronger floating.
- A silent workspace rule does not hide a pinned window.
- Launch-then-move races the first mapped frame.
- Window titles can change after map; GTK class values may differ from the requested application ID.
- Layer-shell surfaces may be absent from `hyprctl clients`.
- `hyprctl keyword` is not valid for every Hyprland configuration parser.
- Broadway and Hyprland compositor evidence answer different questions.
- Auto-restart turns one crash into repeated focus-stealing map attempts; disable it during trials.

## Verification

A run is complete only when all checks pass:

```bash
hyprctl -j clients
hyprctl -j monitors
hyprctl -j activeworkspace
hyprctl -j activewindow
```

Verify that no test client remains, `HERMES_UI_TEST` is absent, and the active workspace and focused client match the recorded baseline. For Broadway runs, also verify that the captured image came from the actual application and that the app never appeared in `hyprctl -j clients`.
