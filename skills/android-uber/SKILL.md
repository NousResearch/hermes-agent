---
name: android-uber
description: Order an Uber ride on Android using the Uber app
version: 1.0.0
metadata:
  hermes:
    tags: [android, uber, transport]
    category: android
---

# Ordering an Uber on Android

## When to Use
User asks to order/book/get an Uber, taxi, or ride.

## Prerequisites
IMPORTANT: Before confirming any ride, ask the user for approval.
Show destination, estimated price, and car type.

## Procedure

1. Open the Uber app:
   android_open_app("com.ubercab")

2. Wait for main screen:
   android_wait(text="Where to?", timeout_ms=8000)

3. Tap the "Where to?" field:
   android_tap_text("Where to?")

4. Type the destination:
   android_type("<destination>", clear_first=True)

5. Wait for suggestions and tap the best match:
   android_wait(text="<destination keyword>")
   android_tap_text("<first suggestion>")

6. Read the price and car options from screen:
   android_read_screen()

7. STOP — Report to user:
   "I found the following ride options: [list from screen].
   The recommended option is [UberX] for [price],
   arriving in ~[N] minutes. Reply 'yes' to confirm."

8. Only after user confirms — tap the ride type and confirm:
   android_tap_text("UberX")
   android_tap_text("Confirm UberX")

9. Wait for confirmation screen:
   android_wait(text="Finding your driver", timeout_ms=10000)

10. Report back:
    android_read_screen()  → extract driver details and ETA

## Pitfalls
- Uber may show a "Where to?" placeholder inside a card — tap inside it
- If logged out, the app shows a phone number prompt — report this to user
- Price surges show a multiplier banner — always mention this to user before confirming
- Uber blocks Accessibility on some Android versions — if tap_text fails, try screenshot + coordinates
