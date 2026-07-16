import { useEffect, useRef } from 'react'

interface ProfileScopeSessionRefreshOptions {
  gatewayState: string
  profileScope: string
  refreshSessions: () => Promise<void>
}

/** Refresh the scoped sidebar list after a profile-scope change, not on mount. */
export function useProfileScopeSessionRefresh({
  gatewayState,
  profileScope,
  refreshSessions
}: ProfileScopeSessionRefreshOptions): void {
  const previousScopeRef = useRef(profileScope)

  useEffect(() => {
    if (profileScope === previousScopeRef.current) {
      return
    }

    previousScopeRef.current = profileScope

    if (gatewayState !== 'open') {
      return
    }

    void refreshSessions().catch(() => undefined)
  }, [gatewayState, profileScope, refreshSessions])
}
