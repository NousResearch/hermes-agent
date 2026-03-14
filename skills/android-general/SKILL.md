---
name: android-general
description: General Android navigation patterns
version: 1.0.0
metadata:
  hermes:
    tags: [android, navigation, general]
    category: android
---

# General Android Navigation

## Opening any app
android_open_app("<package_name>")
If unsure of package name: android_get_apps() → search results

## Going back
android_press_key("back")

## Going home
android_press_key("home")

## Opening notifications
android_press_key("notifications")

## Handling permission dialogs
After opening an app, read_screen may show permission dialogs.
Look for "Allow" / "Deny" / "While using the app" buttons.
android_tap_text("Allow")

## Scrolling to find content
android_scroll("down") — repeat until target text is found
After each scroll, android_read_screen() to check current state

## Dealing with app loading
android_wait(text="<expected element>", timeout_ms=8000)
If it times out, android_screenshot() to see the actual screen state

## Typing into a field
1. android_tap_text("<field label or placeholder text>")
2. android_wait(class_name="android.widget.EditText")
3. android_type("<text>", clear_first=True)
