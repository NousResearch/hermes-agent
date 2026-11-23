import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useState } from 'react'

import { $gateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { syncBackendLocalePacks } from './backend-packs'
import { I18nProvider } from './context'
import { unregisterAppLocaleSource } from './registry'
import { $requestedLocale } from './runtime'

/** Keep gateway/store imports outside the shared i18n context's import graph. */
export function ProfileI18nProvider({ children }: { children: ReactNode }) {
  const profile = useStore($activeGatewayProfile)
  const connection = useStore($connection)
  const [lastConnection, setLastConnection] = useState(connection)
  const [gatewayOpenSeq, setGatewayOpenSeq] = useState(0)

  // A null $connection is an ordinary disconnect/reconnect blip; keep the last
  // resolved owner so transient drops do not reset scopeKey or clear backend packs.
  useEffect(() => {
    if (connection) {
      setLastConnection(connection)
    }
  }, [connection])

  // Track false -> true transitions of gateway open state so long disconnects
  // that exhaust I18nProvider's startup retry window re-sync once the socket recovers.
  useEffect(() => {
    let prevOpen = $gatewayState.get() === 'open'

    return $gatewayState.listen(state => {
      const isOpen = state === 'open'

      if (isOpen && !prevOpen) {
        setGatewayOpenSeq(seq => seq + 1)
      }

      prevOpen = isOpen
    })
  }, [])

  const resolvedConnection = connection ?? lastConnection

  // The API captures the real read origin (including legacy profile overrides).
  // This is an invalidation key, not an explicit local/registry request pin.
  const ownerKey = JSON.stringify([
    resolvedConnection?.connectionId ?? null,
    resolvedConnection?.registryScoped ?? false,
    resolvedConnection?.baseUrl,
    resolvedConnection?.profile,
    profile
  ])

  const scopeKey = JSON.stringify([
    resolvedConnection?.connectionId ?? null,
    resolvedConnection?.registryScoped ?? false,
    resolvedConnection?.baseUrl,
    resolvedConnection?.profile,
    profile,
    gatewayOpenSeq
  ])

  useBackendLocalePacks(ownerKey)

  return (
    <I18nProvider cacheKey={ownerKey} scopeKey={scopeKey}>
      {children}
    </I18nProvider>
  )
}

/**
 * Mirror the active backend's language packs into the app-locale registry:
 * `i18n.languages` for the switcher, `i18n.catalog {surface:'desktop'}` for the
 * language `display.language` asks for. Re-syncs when the socket comes up or
 * its client is published (boot reports `open` before `$gateway` is set, so
 * the client is a dependency, not a one-off read), when the requested
 * language changes, and on a profile/connection switch (whose first act is
 * dropping the previous backend's layer — a soft re-home must not leak one
 * profile's strings into the next).
 */
function useBackendLocalePacks(scopeKey: string): void {
  const gateway = useStore($gateway)
  const gatewayState = useStore($gatewayState)
  const requested = useStore($requestedLocale)
  const open = gatewayState === 'open'

  useEffect(() => {
    unregisterAppLocaleSource('backend')
  }, [scopeKey])

  useEffect(() => {
    if (!open || !gateway) {
      return
    }

    let current = true

    void syncBackendLocalePacks(
      (method, params) => gateway.request(method, params),
      requested,
      () => current
    )

    return () => {
      current = false
    }
  }, [gateway, open, requested, scopeKey])
}
