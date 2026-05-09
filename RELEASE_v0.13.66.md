# Hermes Agent v0.13.66

## Android

- Adds saved and direct audio automation actions for stream volume, ringer mode,
  speakerphone, and microphone mute through Android's normal audio service.
- Imports safe Tasker XML audio actions for alarm, ringer, notification,
  in-call, media, system, DTMF, accessibility, and BT voice volume records,
  plus speakerphone, microphone mute, and explicit string Sound Mode values.
- Keeps imported Tasker audio records disabled by default and reports when
  Android notification-policy access is required for ringer-mode changes.
