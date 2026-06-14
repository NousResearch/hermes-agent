import { useEffect, useRef } from 'react'

import { stopVoicePlayback } from '@/lib/voice-playback'

/**
 * Invoke `onSwitch` whenever the active session id changes to a *different*
 * session, so session-scoped voice state (an in-flight voice conversation and
 * the global read-aloud / `Preparing audio` / `Speaking` playback status) can
 * be torn down instead of leaking into the newly selected session (#46194).
 *
 * This covers a **warm** switch, where the composer stays mounted and its voice
 * state would otherwise persist across the switch. A **cold** switch (resuming a
 * session with no warm runtime) unmounts the composer instead — the
 * component-local conversation state dies with it, but the module-level
 * read-aloud playback survives, so pair this with
 * {@link useStopVoicePlaybackOnUnmount}.
 *
 * Skips two non-switches, mirroring the composer's placeholder-reset guard:
 *  - the initial mount (no previous session), and
 *  - the `null → id` transition where the brand-new session we're already in
 *    just gets persisted.
 * Leaving a session (`id → null`) and moving between two real sessions both
 * count as a switch.
 */
export function useEndVoiceOnSessionSwitch(sessionId: string | null | undefined, onSwitch: () => void) {
  const lastSessionIdRef = useRef(sessionId)

  useEffect(() => {
    // Treat `undefined` and `null` as the same "no session" so a transient
    // undefined→null at startup isn't mistaken for a switch.
    const previousSessionId = lastSessionIdRef.current ?? null
    const currentSessionId = sessionId ?? null
    lastSessionIdRef.current = sessionId

    if (previousSessionId === currentSessionId || (previousSessionId === null && currentSessionId !== null)) {
      return
    }

    onSwitch()
  }, [onSwitch, sessionId])
}

/**
 * Stop global read-aloud playback when the host component unmounts. On a cold
 * session switch the composer is torn down (see {@link useEndVoiceOnSessionSwitch}),
 * so the module-level `$voicePlayback` singleton would otherwise keep its
 * `Preparing audio` / `Speaking` status and bleed into the resumed session's
 * composer once it remounts (#46194).
 */
export function useStopVoicePlaybackOnUnmount() {
  useEffect(
    () => () => {
      stopVoicePlayback()
    },
    []
  )
}
