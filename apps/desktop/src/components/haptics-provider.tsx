import { type ReactNode, useEffect } from 'react'

import { registerHapticTrigger } from '@/lib/haptics'

// web-haptics (the haptic feedback library used on mobile) falls back to an
// AudioContext-based click polyfill on platforms without navigator.vibrate
// hardware.  Even though Chromium exposes navigator.vibrate on desktop, it is
// a no-op — and the library's debug mode creates an AudioContext that plays
// click sounds through CoreAudio, briefly seizing the audio device and
// interrupting other playback (Spotify, Apple Music, Bluetooth streams) on
// every prompt submit.
//
// Desktop has no vibration motor, so haptic feedback is not useful here.
// Register a permanent no-op trigger so triggerHaptic() is a silent no-op at
// all call sites.  See: https://github.com/Nicepkg/web-haptics
export function HapticsProvider({ children }: { children: ReactNode }) {
  useEffect(() => {
    registerHapticTrigger(null)

    return () => registerHapticTrigger(null)
  }, [])

  return <>{children}</>
}
