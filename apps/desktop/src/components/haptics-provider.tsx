import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect } from 'react'
import { useWebHaptics } from 'web-haptics/react'

import { registerHapticTrigger } from '@/lib/haptics'
import { $hapticsMuted } from '@/store/haptics'

// web-haptics' `debug` option is its desktop audio fallback: on platforms
// where navigator.vibrate exists (Chromium exposes it as a no-op on
// desktop) the library plays synthesized click sounds through an
// AudioContext whenever debug is enabled, briefly seizing the macOS audio
// device and interrupting other playback (Spotify, Apple Music, Bluetooth
// streams) on every prompt submit. Desktop has no vibration motor, so
// keep the provider and the persisted mute control wired, but never
// enable the audio fallback.
export function HapticsProvider({ children }: { children: ReactNode }) {
  const muted = useStore($hapticsMuted)
  const { trigger } = useWebHaptics({ showSwitch: false })

  useEffect(() => {
    registerHapticTrigger(muted ? null : trigger)

    return () => registerHapticTrigger(null)
  }, [muted, trigger])

  return <>{children}</>
}
