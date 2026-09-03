# HUD mode

The HUD is the chrome-free floating chat: a frameless, always-on-top bar with the live reply hanging under it, so an agent can be driven while you work in another app. This page covers the pointer, capture, switching, voice and pet features layered on it.

## Shortcuts

| Shortcut | Where | What |
| --- | --- | --- |
| `Ctrl/⌘ + Shift + H` | anywhere | open or close the HUD (from another app it opens as a companion, no focus steal) |
| `Ctrl/⌘ + Shift + G` | anywhere, HUD open | move the HUD to the pointer |
| `Ctrl/⌘ + Alt + H` | anywhere | ask about what is under the pointer (configurable in Settings → HUD) |
| `Ctrl/⌘ + Alt + F` | inside Hermes | toggle follow-the-pointer |
| `Ctrl/⌘ + Shift + ]` / `[` | inside the HUD | next / previous agent |
| `Esc` | inside the HUD | close the ask sheet, then the HUD |

## Follow the pointer

On by default. The bar parks just to the lower-right of the pointer, like a context menu, flips to the left near the right edge and above the pointer near the bottom, and holds still while the pointer is on the bar, within 40 px of a resting bar, or while the HUD has keyboard focus. Moving through the transparent band under the bar counts as leaving. Turn it off with the magnet button in the bar, the shortcut, or Settings → HUD. Not available on native Wayland, where windows cannot place themselves.

## Ask about what is under the pointer

The chord captures which app and window is under the pointer (metadata only: app, title, bounds) and a 960×600 crop of the display around the pointer, then opens the HUD beside the pointer with a sheet: **Explain**, **Summarize**, **Do it**, **Ask…**. The first three attach the crop and send through the HUD's composer; **Ask…** attaches and hands you the composer.

Privacy: the full-display capture is deleted immediately after cropping. The crop is saved to the app's `composer-images` folder with the same handling as a pasted screenshot and leaves the machine only inside the prompt you send. Nothing is captured without your gesture.

Optional **Ctrl + right-click anywhere** (⌘ + right-click on macOS) uses the `uiohook-napi` input hook, an optional native dependency. It is off by default. While it is on, the hook observes system-wide mouse buttons and modifier keys to detect that one gesture; it does not log positions or keystrokes, stops the moment the setting is turned off, and a plain right-click is never intercepted. If the module is not installed for your platform, Settings says so and the toggle is inert.

## Agents and rooms

The agent pill in the bar shows the agent you are talking to (avatar or role glyph) and switches between profiles; switching restarts the HUD against that profile's backend, so an in-flight reply is lost. The room pill enters **room mode**: the composer posts into a Bot Mode room and the bar shows the room's recent log as members reply. The room engine runs in the app window; the HUD is a remote for it. Leave the room from the pill or the panel.

## Voice commands

With a voice conversation running, whole-utterance HUD commands are handled by the desktop instead of being sent to the agent: "HUD top left" (any corner, edge or centre), "HUD come here", "HUD follow me" / "HUD stay", "HUD hide". A sentence that merely contains "HUD" still goes to the agent.

## Pets

Each agent has a pet that walks a strip above the bar: the bundled characters Hank and Mina, the agent's Bot Mode avatar, or none (Settings → HUD → Pet per agent). They react to the turn: stop and quote your prompt while the agent thinks, pace while tools run, hop when the reply lands, droop on an error. They stop to look at your pointer when it comes near, ignore the mouse otherwise, and stand still under reduced-motion settings. Artwork license: see `src/components/pet/assets/hud/LICENSE-ART.md`.

## Settings and files

All HUD preferences live in `hud-prefs.json` in the app's user-data folder: `follow`, `askShortcut`, `askOnRightClick`, `pets`, `petByAgent`. The HUD's size and position live in `hud-state.json`.
