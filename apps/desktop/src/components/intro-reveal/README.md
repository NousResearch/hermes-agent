# Intro reveal

The cinematic runs in a transparent Electron window at `?win=intro`.
It covers the primary display and plays over the desktop while the app hides.
The surface owns its animation clock and synthesized sound; it has no gateway.
The main renderer owns the phase and the persistent seen key.

The seven beats are ask, send, working, reply, everywhere, brand and dissolve.
Typing, tool activity, the cube and sound follow the same score.
Normal playback lasts 22 seconds; the exit dissolve lasts 900 ms.
Reduced motion shows the brand briefly. Click, Enter or Escape skips.
Sound defaults on and respects the existing haptics mute preference.

There is one launch gate, enabled by `HERMES_GUEST_ONBOARDING=1` or
`--guest-onboarding`. Preload exposes that decision as `guestOnboardingEnabled`.
First-run eligibility also requires an unseen intro and no first-run skip.
With the launch gate off, neither the store nor native IPC opens the film.

The renderer deadman is 26 seconds. The independent native watchdog is
34 seconds, deliberately longer; both return the main window if playback stalls.
Finishing records seen, clears the phase and requests `{showMain: true}`.
The guided-chat edge and app-shell mount arrive with the later gate/handoff steps.

Rehearse from `apps/desktop` with isolated app state:

```sh
intro_tmp=$(mktemp -d /tmp/hermes-intro.XXXXXX)
env -u NODE_ENV HERMES_GUEST_ONBOARDING=1 HERMES_HOME="$intro_tmp/.hermes" HERMES_DESKTOP_USER_DATA_DIR="$intro_tmp/electron-user-data" npm run dev
```

Collapse comes from the UI package; JetBrains Mono comes from desktop styles.
