import { useEffect } from 'react'

import { refreshDefaultProfile, subscribeDefaultProfile } from '@/store/default-profile'

export function useDefaultProfilePreference(): void {
  useEffect(() => {
    const unsubscribe = subscribeDefaultProfile()
    // A failed read leaves the last known preference intact. It must never
    // prevent gateway boot, and the next boot/menu refresh can retry it.
    void refreshDefaultProfile().catch(() => undefined)

    return unsubscribe
  }, [])
}
