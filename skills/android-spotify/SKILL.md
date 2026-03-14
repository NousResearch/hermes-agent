---
name: android-spotify
description: Control Spotify playback on Android
version: 1.0.0
metadata:
  hermes:
    tags: [android, spotify, music]
    category: android
---

# Spotify Playback Control

## When to Use
User asks to play music, search for songs/artists/playlists, or control playback on Spotify.

## Procedure — Play a song or artist

1. Open Spotify:
   android_open_app("com.spotify.music")

2. Wait for home screen:
   android_wait(text="Search", timeout_ms=8000)

3. Tap Search:
   android_tap_text("Search")

4. Tap the search input field:
   android_wait(class_name="android.widget.EditText", timeout_ms=3000)
   android_tap_text("What do you want to listen to?")

5. Type the query:
   android_type("<song/artist/playlist name>", clear_first=True)

6. Wait for results:
   android_wait(text="Songs", timeout_ms=5000)

7. Read results:
   android_read_screen()

8. Tap the desired result:
   android_tap_text("<song or artist name>")

## Procedure — Playback controls

- Play/Pause: android_tap_text("Play") or android_tap_text("Pause")
- Next track: android_tap_text("Next")
- Previous track: android_tap_text("Previous")
- Shuffle: android_tap_text("Shuffle")

## Pitfalls
- Spotify uses custom views — some elements may not have text in the accessibility tree
- If search results don't appear, try android_screenshot() to see the actual screen
- The "Now Playing" bar at the bottom can be tapped to expand the player
- Free accounts show ads between songs — these cannot be skipped programmatically
