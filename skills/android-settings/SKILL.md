---
name: android-settings
description: Navigate Android system settings
version: 1.0.0
metadata:
  hermes:
    tags: [android, settings, system]
    category: android
---

# Android Settings Navigation

## When to Use
User asks to change a system setting (WiFi, Bluetooth, display, sound, etc).

## Procedure — Open settings

1. Open Settings app:
   android_open_app("com.android.settings")

2. Wait for settings to load:
   android_wait(text="Settings", timeout_ms=5000)

3. Navigate to the relevant section by tapping:
   - "Network & internet" — WiFi, mobile data, airplane mode
   - "Connected devices" — Bluetooth, NFC
   - "Apps" — App management
   - "Notifications" — Notification settings
   - "Battery" — Battery usage and saver
   - "Storage" — Storage management
   - "Sound & vibration" — Volume, ringtone
   - "Display" — Brightness, dark mode, font size
   - "Accessibility" — Accessibility services
   - "Security & privacy" — Lock screen, biometrics

4. Read screen to find the specific setting:
   android_read_screen()

5. Toggle or adjust the setting:
   android_tap_text("<setting name>")

## Common tasks

### Toggle WiFi
1. android_open_app("com.android.settings")
2. android_tap_text("Network & internet")
3. android_tap_text("Internet")
4. android_tap_text("Wi-Fi") to toggle, or tap a network name to connect

### Toggle Bluetooth
1. android_open_app("com.android.settings")
2. android_tap_text("Connected devices")
3. android_tap_text("Connection preferences")
4. android_tap_text("Bluetooth")

### Change brightness
1. android_open_app("com.android.settings")
2. android_tap_text("Display")
3. Look for brightness slider — may need to use android_swipe for slider adjustment

## Pitfalls
- Settings UI varies significantly across Android manufacturers (Samsung, Pixel, Xiaomi, etc.)
- Always use android_read_screen() to discover the actual menu labels on the device
- Some settings require scrolling down — use android_scroll("down") if setting not found
- Toggle switches may show as "android.widget.Switch" in the accessibility tree
