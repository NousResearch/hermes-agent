import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect } from 'react'
import { useWebHaptics } from 'web-haptics/react'

import { registerHapticTrigger } from '@/lib/haptics'
import { $hapticsMuted } from '@/store/haptics'

export function HapticsProvider({ children }: { children: ReactNode }) {
  const muted = useStore($hapticsMuted)
  const { trigger } = useWebHaptics({ debug: true, showSwitch: false })

  useEffect(() => {
    registerHapticTrigger(muted ? null : trigger)

    return () => registerHapticTrigger(null)
  }, [muted, trigger])

  return <>{children}</>
}
