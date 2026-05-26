# Hermes Agent Fork v0.13.118

This release expands Android signal intelligence for Gemma-visible diagnostics
and top-card workflows across Wi-Fi, Bluetooth, radio, motion sensors, and
MediaTek-aware device stability checks.

## Android

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
