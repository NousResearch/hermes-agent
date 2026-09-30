// One window owns each cross-window ambient cue (turn-end sound, spoken reply)
// so N open full windows don't all fire it for the same backend event. The main
// process is the race-free owner (see electron/event-dedupe.ts). Off Electron —
// or when the bridge/claim fails — every window emits, preserving the
// single-window behavior rather than going silent.
export async function ownsAmbientCue(key: string): Promise<boolean> {
  const claim = window.hermesDesktop?.claimAmbientCue

  if (!claim) {
    return true
  }

  try {
    return await claim(key)
  } catch {
    return true
  }
}
