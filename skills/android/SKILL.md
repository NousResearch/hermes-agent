---
name: android
description: Control an Android phone remotely — navigate apps, tap, type, swipe, and automate Uber, WhatsApp, Spotify, Maps, Settings, Tinder
version: 1.0.0
metadata:
  hermes:
    tags: [android, phone, automation, accessibility]
    category: android
---

# Android Device Control

You can control an Android phone remotely using the `android_*` tools. The phone runs a companion app called **Hermes Bridge** which exposes an HTTP API. You communicate with it over the network — no USB, no ADB, no physical connection needed.

## How It Works

```
Hermes Agent (this server)  ──HTTP──>  Hermes Bridge app (Android phone)
                                        ├── Reads screen via AccessibilityService
                                        ├── Performs taps, types, swipes
                                        └── Authenticated via pairing code
```

## Setup / Connecting a Phone

When the user wants to connect their phone, ask for their **pairing code** — a 6-character code shown in the Hermes Bridge app (e.g. `K7V3NP`).

Then call:
```
android_setup("<pairing_code>")
```

This does two things:
1. Starts a relay on this server (auto-detects the server's public IP)
2. Returns the exact instructions to tell the user — the server address and pairing code to enter in their phone app

**Relay the `user_instructions` field from the result directly to the user.** It contains the server IP and port they need to type into the phone app.

After the user taps Connect on their phone, the phone connects to this server via WebSocket. Call `android_ping()` to verify the connection is live.

**Do NOT ask about:**
- USB, ADB, or developer options
- The phone's IP address (not needed — the phone connects to the server, not the other way around)
- nginx, firewalls, or port forwarding
- Any networking concepts

**Just ask for the pairing code, call setup, and relay the instructions.**

## Available Tools

You have these 14 tools. Use them by name — they are function calls.

### Connectivity
- `android_ping()` — check if phone is connected and responding
- `android_setup(pairing_code)` — start relay and configure connection

### Reading the Screen
- `android_read_screen(include_bounds=False)` — get the full accessibility tree as JSON. Returns every visible UI element with text, className, nodeId, clickable, etc. **Always call this before interacting.**
- `android_screenshot()` — capture a screenshot as base64 PNG. Use when the accessibility tree doesn't show enough (canvas apps, image-heavy UIs).
- `android_current_app()` — get the package name and activity of the foreground app.

### Opening Apps
- `android_open_app(package)` — launch any app by package name. **This is the primary way to open apps. Do NOT try to find and tap app icons.** Example: `android_open_app("com.instagram.android")`
- `android_get_apps()` — list all installed apps with package names. Use this if you don't know the package name.

### Tapping
- `android_tap(x, y, node_id)` — tap by coordinates or node ID. Prefer node_id from read_screen.
- `android_tap_text(text, exact=False)` — tap the first element matching text. **Most convenient for buttons, menu items, links.**

### Typing
- `android_type(text, clear_first=False)` — type into the currently focused input field. Tap the field first.

### Gestures
- `android_swipe(direction, distance="medium")` — swipe up/down/left/right. Distances: short, medium, long.
- `android_scroll(direction, node_id=None)` — scroll a specific element or the whole screen.

### Keys
- `android_press_key(key)` — press a key. Options: `back`, `home`, `recents`, `power`, `volume_up`, `volume_down`, `enter`, `delete`, `tab`, `escape`, `search`, `notifications`

### Waiting
- `android_wait(text, class_name, timeout_ms=5000)` — poll until an element appears. Use after navigation or loading.

## Rules

### CRITICAL: Do not loop
- **Maximum 5-7 tool calls per user request.** After that, STOP and report what you did and what you see.
- **Do NOT keep taking screenshots in a loop.** Take ONE screenshot, analyze it, act, then report.
- **If an action doesn't work after 2 attempts, STOP and tell the user** what happened.
- **After completing the user's request, STOP and report the result.** Do not keep interacting with the screen.

### Workflow pattern
For any task, follow this pattern and then STOP:
1. `android_open_app(package)` — open the app
2. `android_read_screen()` — see what's on screen
3. 1-3 actions (tap, type, swipe) — do what the user asked
4. `android_read_screen()` or `android_screenshot()` — verify the result
5. **Report to the user and STOP.** Do not take further actions unless the user asks.

### Other rules
1. **ALWAYS open apps with `android_open_app(package)`** — never try to find and tap the icon on the home screen or app drawer.
2. **Prefer `android_read_screen()` over `android_screenshot()`** — read_screen is faster and structured. Only use screenshot when the accessibility tree is insufficient (canvas/image-heavy apps).
3. **Prefer `android_tap_text("Button Text")` over coordinates** — it's more reliable.
4. **If you don't know a package name**, call `android_get_apps()` and search the results.
5. **Confirm destructive actions** (purchases, sends, deletions) with the user before executing.
6. **Handle permission dialogs** — look for "Allow"/"Deny" buttons. Tap "Allow" or "While using the app".
7. **Go back**: `android_press_key("back")`. **Go home**: `android_press_key("home")`.

---

## Common Package Names

| App | Package |
|-----|---------|
| Uber | com.ubercab |
| Bolt | com.bolt.client |
| WhatsApp | com.whatsapp |
| Spotify | com.spotify.music |
| Google Maps | com.google.android.apps.maps |
| Chrome | com.android.chrome |
| Gmail | com.google.android.gm |
| Instagram | com.instagram.android |
| X/Twitter | com.twitter.android |
| Tinder | com.tinder |
| Settings | com.android.settings |

---

## App-Specific Procedures

### Uber — Order a ride

1. `android_open_app("com.ubercab")`
2. `android_wait(text="Where to?", timeout_ms=8000)`
3. `android_tap_text("Where to?")`
4. `android_type("<destination>", clear_first=True)`
5. `android_wait(text="<destination keyword>")` then tap suggestion
6. `android_read_screen()` — read price and car options
7. **STOP** — Report options and price to user, wait for confirmation
8. After confirmation: `android_tap_text("UberX")` then `android_tap_text("Confirm UberX")`
9. `android_wait(text="Finding your driver", timeout_ms=10000)`

**Pitfalls:** Uber may block accessibility taps on some versions — fall back to screenshot + coordinates. Always mention surge pricing to user.

### WhatsApp — Send a message

1. `android_open_app("com.whatsapp")`
2. `android_wait(text="Chats")`
3. Existing chat: `android_tap_text("<contact name>")`
4. New chat: `android_tap_text("New chat")` → type contact name → tap match
5. `android_tap_text("Type a message")`
6. `android_type("<message text>")`
7. **STOP** — Confirm with user before sending
8. `android_tap_text("Send")` or `android_press_key("enter")`

**Pitfalls:** Message input is `android.widget.EditText`. Read screen after typing to verify before sending.

### Spotify — Play music

1. `android_open_app("com.spotify.music")`
2. `android_wait(text="Search", timeout_ms=8000)`
3. `android_tap_text("Search")`
4. `android_wait(class_name="android.widget.EditText")`
5. `android_type("<query>", clear_first=True)`
6. `android_wait(text="Songs", timeout_ms=5000)`
7. `android_read_screen()` then tap desired result

**Playback:** `android_tap_text("Play")`, `android_tap_text("Next")`, `android_tap_text("Pause")`

**Pitfalls:** Spotify uses custom views — screenshot may be more useful than read_screen.

### Google Maps — Get directions

1. `android_open_app("com.google.android.apps.maps")`
2. `android_wait(text="Search here", timeout_ms=8000)`
3. `android_tap_text("Search here")`
4. `android_type("<destination>", clear_first=True)`
5. Tap suggestion → `android_tap_text("Directions")`
6. `android_read_screen()` — report time, distance, route to user
7. Start navigation only if user confirms: `android_tap_text("Start")`

**Pitfalls:** Maps uses heavy canvas rendering — prefer `android_screenshot()`. Exit navigation with `android_press_key("back")`.

### Settings — Change system settings

1. `android_open_app("com.android.settings")`
2. `android_wait(text="Settings", timeout_ms=5000)`
3. Navigate by tapping section names:
   - "Network & internet" → WiFi, mobile data
   - "Connected devices" → Bluetooth, NFC
   - "Display" → Brightness, dark mode
   - "Sound & vibration" → Volume
   - "Apps" → App management
4. `android_read_screen()` to find specific toggles

**Pitfalls:** Settings UI varies across manufacturers (Samsung, Pixel, Xiaomi). Always read_screen to discover actual labels. Use `android_scroll("down")` if setting not visible.

### Tinder — View profiles and interact

1. `android_open_app("com.tinder")`
2. `android_wait(timeout_ms=8000)`
3. `android_read_screen()` + `android_screenshot()` — Tinder is image-heavy
4. Report profile details to user

**IMPORTANT:** Always confirm with user before swiping or messaging.
- Like: `android_swipe("right")`
- Pass: `android_swipe("left")`
- Super Like: `android_swipe("up")`

**Pitfalls:** Tinder uses custom UI — accessibility tree is limited, prefer screenshots. "It's a Match!" popup: tap anywhere to dismiss.
