import { useEffect, useRef } from 'react'

/** Initial discovery and reconnect preserve a saved pick; a new owner reseeds it. */
export function useGatewayScopeRefresh(
  connectionId: string | null,
  profile: string,
  refresh: (force: boolean) => void
): void {
  const previous = useRef({ connectionId, profile })

  // eslint-disable-next-line no-restricted-syntax -- track the last resolved gateway identity across reconnects
  useEffect(() => {
    const last = previous.current
    const resolvedId = connectionId ?? last.connectionId

    if (resolvedId === last.connectionId && profile === last.profile) {
      return
    }

    const changedOwner = profile !== last.profile || (last.connectionId !== null && resolvedId !== last.connectionId)
    previous.current = { connectionId: resolvedId, profile }
    // The first descriptor arriving is hydration, not a user changing sources.
    // Keep config/model refreshes non-forced so their saved-selection guards run.
    refresh(changedOwner)
  }, [connectionId, profile, refresh])
}
