import { useCallback, useEffect, useRef, useState } from 'react'

import { getApiRequestConnection, getApiRequestProfile, type ProfileScope } from '@/api/client'
import { $settingsRequestProfile, $settingsScopeKey } from '@/store/settings-scope'

/** Use inside a page keyed by $settingsScopeKey. A queued operation owns the
 * mounted selection, never a later ambient connection or same-named profile. */
export function useSettingsOwner(profile: ProfileScope = $settingsRequestProfile.get(), key = $settingsScopeKey.get()) {
  const [owner] = useState(() => ({
    profile: profile && typeof profile === 'object' ? Object.freeze({ ...profile }) : profile,
    key,
    generation: $settingsScopeKey.get()
  }))

  const mounted = useRef(true)

  // eslint-disable-next-line no-restricted-syntax -- component lifetime, not a reactive atom mirror
  useEffect(() => {
    mounted.current = true

    return () => {
      mounted.current = false
    }
  }, [])

  const isCurrent = useCallback(
    () => mounted.current && owner.generation === $settingsScopeKey.get(),
    [owner.generation]
  )

  const isActive = useCallback(() => {
    if (!isCurrent()) {
      return false
    }

    const scope = owner.profile
    const activeProfile = getApiRequestProfile() || 'default'

    if (scope && typeof scope === 'object') {
      return scope.connectionId === getApiRequestConnection() && (scope.profile || 'default') === activeProfile
    }

    return scope === undefined || (scope || 'default') === activeProfile
  }, [isCurrent, owner.profile])

  return { profile: owner.profile, scopeKey: owner.key, isCurrent, isActive }
}
