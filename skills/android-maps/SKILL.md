---
name: android-maps
description: Navigate with Google Maps on Android
version: 1.0.0
metadata:
  hermes:
    tags: [android, maps, navigation]
    category: android
---

# Google Maps Navigation

## When to Use
User asks for directions, navigation, nearby places, or to look up a location.

## Procedure — Get directions

1. Open Google Maps:
   android_open_app("com.google.android.apps.maps")

2. Wait for map to load:
   android_wait(text="Search here", timeout_ms=8000)

3. Tap search:
   android_tap_text("Search here")

4. Type destination:
   android_type("<destination>", clear_first=True)

5. Wait for suggestions:
   android_wait(text="<destination keyword>", timeout_ms=5000)

6. Tap the best match:
   android_tap_text("<suggestion>")

7. Tap Directions:
   android_tap_text("Directions")

8. Read route info:
   android_read_screen()
   Report estimated time, distance, and route summary to user.

9. To start navigation (only if user confirms):
   android_tap_text("Start")

## Procedure — Find nearby places

1. Open Maps and search for category:
   android_type("restaurants near me", clear_first=True)
2. Read results: android_read_screen()
3. Report top results with ratings and distance

## Pitfalls
- Maps heavily uses canvas rendering — android_screenshot() is often more useful than read_screen
- Location permission must be granted for "near me" searches
- Navigation mode takes over the screen — use android_press_key("back") to exit
- Transit directions may need tapping the transit icon (bus/train) explicitly
