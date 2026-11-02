import { useEffect, useRef } from 'react'

import { stopVoicePlayback } from '@/lib/voice-playback'

export function useEndVoiceOnSessionSwitch(sessionId: string | null | undefined, onSwitch: () => void) {
  const lastSessionIdRef = useRef(sessionId)

  // eslint-disable-next-line no-restricted-syntax -- remembers the previous session prop to detect a switch, not an atom mirror
  useEffect(() => {
    const previousSessionId = lastSessionIdRef.current ?? null
    const currentSessionId = sessionId ?? null
    lastSessionIdRef.current = sessionId

    if (previousSessionId === currentSessionId || (previousSessionId === null && currentSessionId !== null)) {
      return
    }

    onSwitch()
  }, [onSwitch, sessionId])
}

export function useStopVoicePlaybackOnUnmount() {
  useEffect(
    () => () => {
      stopVoicePlayback()
    },
    []
  )
}
