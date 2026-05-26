# Hermes Agent Fork v0.13.119

This release supersedes v0.13.118 with the Android branch CI resilience guard
aligned to the current SSE stream-event parser, then carries forward the
expanded Android signal intelligence for Gemma-visible diagnostics and top-card
workflows across Wi-Fi, Bluetooth, radio, motion sensors, and MediaTek-aware
device stability checks.

## Android

- Keeps custom endpoint SSE parsing guarded while tracking both delta content
  and finish reasons through the stream-event parser.
- Adds Wi-Fi channel decision packets with route cards, claim-boundary cards,
  and quiet-channel guidance that separates passive scan evidence from
  analyzer recommendations.
- Adds Bluetooth nearby decision packets for scanner readiness, passive evidence,
  route cards, and user-visible top-card summaries.
- Adds radio signal decision packets for AM, FM, SDR, tuner availability,
  permissions, and supported-device claim boundaries.
- Adds accelerometer, gyroscope, IMU, pose, and motion workflow decision packets
  with history freshness, sensor quality, privacy, and MediaTek stack routing.
- Expands quick actions, native chat routing, diagnostic card parsing, and tool
  result compaction so the new packet quartet stays visible to Gemma and users.
- Publishes signed universal APK and AAB artifacts for the Hermes Agent Fork
  Android package `com.mobilefork.hermesagent`.
