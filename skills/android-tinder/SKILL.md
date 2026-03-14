---
name: android-tinder
description: Interact with the Tinder dating app on Android
version: 1.0.0
metadata:
  hermes:
    tags: [android, tinder, dating]
    category: android
---

# Tinder App Interaction

## When to Use
User asks to interact with Tinder — view profiles, swipe, send messages.

## Prerequisites
IMPORTANT: Always confirm actions with the user before swiping or sending messages.
This involves real people — do not automate without explicit user approval for each action.

## Procedure — View current profile

1. Open Tinder:
   android_open_app("com.tinder")

2. Wait for main screen:
   android_wait(timeout_ms=8000)

3. Read the current profile:
   android_read_screen()
   android_screenshot()  — Tinder uses heavy image content, screenshot is more informative

4. Report profile details to user:
   Name, age, bio text, distance (if visible in accessibility tree)

## Procedure — Swipe (only with user permission)

1. Read and report current profile to user
2. STOP — Ask user: "Would you like to swipe right (like) or left (pass) on [name]?"
3. Only after user confirms:
   - Like (swipe right): android_swipe("right")
   - Pass (swipe left): android_swipe("left")
   - Super Like (swipe up): android_swipe("up")

## Procedure — Send a message

1. Navigate to matches: android_tap_text("Messages") or android_tap_text("Matches")
2. Tap the match: android_tap_text("<match name>")
3. Tap message input: android_tap_text("Type a message")
4. Type the message: android_type("<message>")
5. STOP — Confirm with user before sending
6. Send: android_tap_text("Send") or android_press_key("enter")

## Pitfalls
- Tinder uses heavy custom UI — accessibility tree may be limited, prefer screenshots
- Profile cards are rendered as images — text extraction from read_screen may be incomplete
- "It's a Match!" popup appears after mutual likes — tap anywhere to dismiss
- Free accounts have limited daily swipes — inform user if limit is reached
- Tinder may detect automation — use natural timing between actions
